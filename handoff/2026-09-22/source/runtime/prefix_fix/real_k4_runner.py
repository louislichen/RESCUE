"""RESCUE 002 / P2: real fixed-K=4 speculative decoding with the locked models.

Target : locked Qwen2-7B (fp16), sharded with the project's existing validated
         balanced device_map pattern; used only via lm_head logits.
Draft  : locked Qwen2-1.5B (bf16, same dtype as Adapter training).
Protocol: frozen greedy, K=4, max_new_tokens inherited from the existing
         generation path; tokenizer identity between the pair is already
         verified (token_id_mapping_equal=true, draft_token_verification_allowed
         =true).  Greedy argmax is taken over the *common* vocabulary
         [0, min(vocab_target, vocab_draft)) for the target-only reference and
         for every verification step, so both sides are compared on one support.

Modes:
  correctness : Target-only greedy vs speculative, token-for-token, in several
                real boundary configurations (normal draft / identical draft /
                adversarial draft).
  eval        : fixed K=4 development evaluation over a row shard, per condition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefix_seqloss import PrefixSequenceBackend, load_backend_module  # noqa: E402

EOS_ID = 151643
MAX_NEW = 16
K = 4


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class CacheModel:
    """Thin wrapper adding the cache protocol the K=4 driver needs."""

    def __init__(self, model, device, vocab_limit):
        self.model = model
        self.device = device
        self.vocab = int(vocab_limit)
        self.forward_calls = 0

    def _ids(self, tokens):
        return torch.tensor([list(tokens)], device=self.device, dtype=torch.long)

    def _mask(self, total):
        return torch.ones((1, total), device=self.device, dtype=torch.long)

    def _greedy(self, logits):
        return int(torch.argmax(logits[: self.vocab]).item())

    def clone_cache(self, cache):
        return cache

    def crop_cache(self, cache, length):
        cache.crop(int(length))
        return cache


class StudentDraft(CacheModel):
    """Draft = Student 1.5B, with the frozen prefill-only Adapter intervention."""

    def __init__(self, backend, vocab_limit, maps=None, gates=None):
        super().__init__(backend.student, backend.device, vocab_limit)
        self.backend = backend
        self.maps = maps or {}
        self.gates = gates or {}
        self.prefill_interventions = 0
        self.decode_interventions = 0
        self.last_prediction = None

    def prefill(self, prompt_ids):
        ids = self._ids(prompt_ids)
        with torch.no_grad():
            if self.maps:
                with self.backend._steering(self.model, self.maps, self.gates, self.backend.eps,
                                            selected_layers=set(self.backend.student_layers),
                                            phase="VALIDATION_PREFILL"):
                    out = self.model(input_ids=ids, attention_mask=self._mask(len(prompt_ids)),
                                     use_cache=True, output_attentions=False)
                    self.prefill_interventions += 1
            else:
                out = self.model(input_ids=ids, attention_mask=self._mask(len(prompt_ids)),
                                 use_cache=True, output_attentions=False)
        self.forward_calls += 1
        self.last_prediction = self._greedy(out.logits[0, -1, :].float())
        return self.last_prediction, out.past_key_values

    def step(self, token, cache):
        total = cache.get_seq_length() + 1
        with torch.no_grad():
            # frozen protocol: STUDENT_CACHED_DECODE installs no intervention
            if getattr(self.model, "_attn_adapter_state", None) is not None:
                raise RuntimeError("Adapter hooks must not be installed during cached decode")
            out = self.model(input_ids=self._ids([token]), attention_mask=self._mask(total),
                             past_key_values=cache, use_cache=True, output_attentions=False)
        self.forward_calls += 1
        self.last_prediction = self._greedy(out.logits[0, -1, :].float())
        return self.last_prediction, out.past_key_values


class TeacherTarget(CacheModel):
    """Target = Teacher 7B, sharded; exact greedy verification.

    ``verify_mode`` selects how the proposal block is evaluated:

    ``batched``     one forward over the whole proposal block (the usual
                    speculative-decoding implementation).  Faster, but the
                    block forward and the single-token reference forward are
                    different bf16 kernels, so an exact top-1/top-2 tie can be
                    broken differently by the two paths.
    ``incremental`` one single-token forward per proposal, i.e. exactly the same
                    computation the Target-only reference performs.  Slower
                    (K target forwards per block instead of one) but makes the
                    accept/reject decisions identical to the reference by
                    construction.
    """

    def __init__(self, model, device, vocab_limit, verify_mode="batched"):
        super().__init__(model, device, vocab_limit)
        self.last_prediction = None
        self.verify_mode = verify_mode
        self.verify_mode_used = {"batched": 0, "incremental": 0}

    def prefill(self, prompt_ids):
        with torch.no_grad():
            out = self.model(input_ids=self._ids(prompt_ids),
                             attention_mask=self._mask(len(prompt_ids)),
                             use_cache=True, output_attentions=False)
        self.forward_calls += 1
        self.last_prediction = self._greedy(out.logits[0, -1, :].float())
        return self.last_prediction, out.past_key_values

    def verify(self, proposal_ids, cache):
        """Return k+1 predictions: one per proposal position plus the bonus token."""
        if self.verify_mode == "incremental":
            self.verify_mode_used["incremental"] += 1
            predictions = [int(self.last_prediction)]
            current = cache
            for token in proposal_ids:
                nxt, current = self.step(int(token), current)
                predictions.append(int(nxt))
            return predictions, current
        self.verify_mode_used["batched"] += 1
        k = len(proposal_ids)
        total = cache.get_seq_length() + k
        with torch.no_grad():
            out = self.model(input_ids=self._ids(proposal_ids),
                             attention_mask=self._mask(total),
                             past_key_values=cache, use_cache=True, output_attentions=False)
        self.forward_calls += 1
        predictions = [int(self.last_prediction)]
        for i in range(k):
            predictions.append(self._greedy(out.logits[0, i, :].float()))
        self.last_prediction = predictions[-1]
        return predictions, out.past_key_values

    def step(self, token, cache):
        total = cache.get_seq_length() + 1
        with torch.no_grad():
            out = self.model(input_ids=self._ids([token]), attention_mask=self._mask(total),
                             past_key_values=cache, use_cache=True, output_attentions=False)
        self.forward_calls += 1
        self.last_prediction = self._greedy(out.logits[0, -1, :].float())
        return self.last_prediction, out.past_key_values


class AdversarialDraft(CacheModel):
    """Always proposes token id 0 -> forces immediate rejection and rollback."""

    def prefill(self, prompt_ids):
        self.forward_calls += 1
        return 0, {"len": len(prompt_ids)}

    def step(self, token, cache):
        self.forward_calls += 1
        return 0, {"len": cache["len"] + 1}

    def clone_cache(self, cache):
        return {"len": cache["len"]}

    def crop_cache(self, cache, length):
        return {"len": int(length)}


class CorruptingDraft:
    """Wrap a real draft and corrupt one proposal out of every K.

    Used to force the partial-acceptance path (accept m < k proposals, then take
    the target correction and roll both caches back) with a real model.  The
    corruption pattern is deterministic in the step counter, so a strict prefix
    is accepted and the remainder rejected.
    """

    def __init__(self, inner, corrupt_index, vocab):
        self.inner = inner
        self.corrupt_index = int(corrupt_index) % 4
        self.vocab = int(vocab)
        self._steps = 0
        self.corruptions = 0
        self.forward_calls = 0

    def prefill(self, prompt_ids):
        next_id, cache = self.inner.prefill(prompt_ids)
        self.forward_calls = self.inner.forward_calls
        return next_id, cache

    def step(self, token, cache):
        next_id, cache = self.inner.step(token, cache)
        self.forward_calls = self.inner.forward_calls
        self._steps += 1
        if (self._steps - 1) % 4 == self.corrupt_index:
            next_id = (int(next_id) + 1) % self.vocab
            self.corruptions += 1
        return next_id, cache

    def clone_cache(self, cache):
        return self.inner.clone_cache(cache)

    def crop_cache(self, cache, length):
        return self.inner.crop_cache(cache, length)


def build_target(args):
    from transformers import AutoModelForCausalLM
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.target_dtype]
    max_memory = {i: args.target_max_memory for i in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, dtype=dtype, local_files_only=True,
        attn_implementation="eager", device_map="balanced",
        max_memory=max_memory, low_cpu_mem_usage=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device = next(model.parameters()).device
    return model, device, dict(getattr(model, "hf_device_map", {}))


def build_draft_condition(args, backend, wrap, row, prior):
    target = backend.prompt_target(row["prompt"], row["answer"])
    with torch.no_grad():
        rows_s = backend._student_rows(target)
        mask = torch.tensor(target.attention_mask, device=backend.device)
        mapped, alpha, _ = wrap.adapter(prior.to(backend.device), rows_s, mask)
        nmapped, nalpha, _ = wrap.adapter(prior.roll(1, dims=-1).to(backend.device), rows_s, mask)
    maps_t = {l: mapped[i] for i, l in enumerate(backend.student_layers)}
    gates_t = {l: alpha[i] for i, l in enumerate(backend.student_layers)}
    maps_n = {l: nmapped[i] for i, l in enumerate(backend.student_layers)}
    gates_n = {l: nalpha[i] for i, l in enumerate(backend.student_layers)}
    return target, maps_t, gates_t, maps_n, gates_n


def run_comparison(backend, wrap, target_model, target_device, vocab, row, prior, args):
    from fixed_k4_protocol import fixed_k4_decode, greedy_reference
    target, maps_t, gates_t, maps_n, gates_n = build_draft_condition(args, backend, wrap, row, prior)
    prompt_ids = target.input_ids
    results = {}
    configs = [
        ("draft_student_base", StudentDraft(backend, vocab)),
        ("draft_student_adapter", StudentDraft(backend, vocab, maps_t, gates_t)),
        ("draft_position_alignment_null", StudentDraft(backend, vocab, maps_n, gates_n)),
        ("draft_teacher_itself", TeacherTarget(target_model, target_device, vocab, args.verify_mode)),
        ("draft_adversarial_token0", AdversarialDraft(None, target_device, vocab)),
    ]
    if args.corrupt_index:
        configs.append((f"draft_student_corrupted_at_{args.corrupt_index}",
                        CorruptingDraft(StudentDraft(backend, vocab), args.corrupt_index, vocab)))
    for name, draft in configs:
        tgt = TeacherTarget(target_model, target_device, vocab, args.verify_mode)
        spec = fixed_k4_decode(prompt_ids, draft, tgt, EOS_ID, k=K, max_new_tokens=args.max_new_tokens)
        ref = greedy_reference(prompt_ids, TeacherTarget(target_model, target_device, vocab,
                                                         args.verify_mode),
                               EOS_ID, args.max_new_tokens)
        mismatch = next((i for i, (a, b) in enumerate(zip(spec.token_ids, ref))
                         if a != b), None)
        if mismatch is None and len(spec.token_ids) != len(ref):
            mismatch = min(len(spec.token_ids), len(ref))
        results[name] = {
            "verify_mode": args.verify_mode,
            "identical_to_target_only": mismatch is None,
            "first_divergence_index": mismatch,
            "speculative_tokens": spec.token_ids,
            "target_only_tokens": ref,
            "counters": vars(spec.counters),
            "target_forward_calls": tgt.forward_calls,
            "draft_forward_calls": draft.forward_calls,
            "draft_prefill_interventions": getattr(draft, "prefill_interventions", 0),
            "draft_decode_interventions": getattr(draft, "decode_interventions", 0),
        }
    return results


def make_backend(args):
    backend_mod, runner_mod = load_backend_module(args.source_dir, args.source_lock)
    backend = backend_mod.QwenAdapterBackend(
        args.teacher_model, args.student_model, args.device,
        source_lock=args.source_lock, load_teacher=False)
    return backend_mod, runner_mod, backend


def correctness(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    backend_mod, runner_mod, backend = make_backend(args)
    wrap = PrefixSequenceBackend(backend)
    if args.checkpoint and Path(args.checkpoint).exists():
        payload = torch.load(args.checkpoint, map_location=backend.device, weights_only=False)
        wrap.adapter.load_state_dict(payload["adapter"])
    target_model, target_device, device_map = build_target(args)
    vocab = min(int(backend.student.config.vocab_size), int(target_model.config.vocab_size))
    rows = runner_mod.load_rows(Path(args.validation_ledger), Path(args.dataset_path))
    priors = torch.load(Path(args.validation_prior_cache), map_location="cpu", weights_only=True)
    chosen = rows[args.shard_lo:args.shard_hi]
    out = {"mode": "correctness", "k": K, "max_new_tokens": args.max_new_tokens,
           "common_vocab_limit": vocab, "eos_id": EOS_ID,
           "target_device_map": device_map,
           "target_dtype": args.target_dtype, "draft_dtype": "bfloat16",
           "checkpoint_identity": sorted(dict(target_model.config.to_dict()).items())[:0],
           "decoding": "greedy argmax over common vocab for target-only reference and verification",
           "frozen_protocol_note": ("Adapter intervention is prefill-only and touches the last "
                                    "query row; cached decode installs no intervention"),
           "records": []}
    for row in chosen:
        target = backend.prompt_target(row["prompt"], row["answer"])
        t0 = time.time()
        res = run_comparison(backend, wrap, target_model, target_device, vocab, row,
                             priors[row["canonical_source_id"]], args)
        out["records"].append({
            "canonical_source_id": row["canonical_source_id"],
            "source_index": row["source_index"],
            "prompt_len": len(target.input_ids),
            "seconds": round(time.time() - t0, 1),
            "configurations": res,
        })
        print(json.dumps({"source_index": row["source_index"],
                          "identical": {k: v["identical_to_target_only"] for k, v in res.items()}},
                         ensure_ascii=False), flush=True)
        (root / "k4_correctness.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out["all_identical"] = all(
        v["identical_to_target_only"]
        for r in out["records"] for v in r["configurations"].values())
    (root / "k4_correctness.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "OK", "all_identical": out["all_identical"],
                      "samples": len(out["records"])}))
    return 0


def evaluate(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    backend_mod, runner_mod, backend = make_backend(args)
    wrap = PrefixSequenceBackend(backend)
    payload = torch.load(args.checkpoint, map_location=backend.device, weights_only=False)
    wrap.adapter.load_state_dict(payload["adapter"])
    target_model, target_device, device_map = build_target(args)
    vocab = min(int(backend.student.config.vocab_size), int(target_model.config.vocab_size))
    from fixed_k4_protocol import fixed_k4_decode, greedy_reference
    rows = runner_mod.load_rows(Path(args.validation_ledger), Path(args.dataset_path))
    priors = torch.load(Path(args.validation_prior_cache), map_location="cpu", weights_only=True)
    conditions = ("StudentBaseline", "StudentAdapter", "PositionAlignmentNull")
    records = []
    for row in rows[args.shard_lo:args.shard_hi]:
        target, maps_t, gates_t, maps_n, gates_n = build_draft_condition(
            args, backend, wrap, row, priors[row["canonical_source_id"]])
        prompt_ids = target.input_ids
        ref = greedy_reference(prompt_ids, TeacherTarget(target_model, target_device, vocab,
                                                         args.verify_mode),
                               EOS_ID, args.max_new_tokens)
        for cond in conditions:
            if cond == "StudentBaseline":
                draft = StudentDraft(backend, vocab)
            elif cond == "StudentAdapter":
                draft = StudentDraft(backend, vocab, maps_t, gates_t)
            else:
                draft = StudentDraft(backend, vocab, maps_n, gates_n)
            tgt = TeacherTarget(target_model, target_device, vocab, args.verify_mode)
            t0 = time.time()
            spec = fixed_k4_decode(prompt_ids, draft, tgt, EOS_ID, k=K,
                                   max_new_tokens=args.max_new_tokens)
            wall = time.time() - t0
            c = vars(spec.counters)
            correct = spec.token_ids == ref
            records.append({
                "canonical_source_id": row["canonical_source_id"],
                "source_index": row["source_index"], "condition": cond,
                "prompt_len": len(prompt_ids),
                "proposed_draft_tokens": c["proposed_draft_tokens"],
                "accepted_draft_tokens": c["accepted_draft_tokens"],
                "correction_tokens": c["correction_tokens"],
                "bonus_tokens": c["bonus_tokens"],
                "blocks": c["blocks"], "full_accept_blocks": c["full_accept_blocks"],
                "reject_blocks": c["reject_blocks"], "short_proposals": c["short_proposals"],
                "eos_terminated": c["eos_terminated"],
                "target_forward_calls": tgt.forward_calls,
                "draft_forward_calls": draft.forward_calls,
                "wall_seconds": round(wall, 3),
                "tokens": spec.token_ids,
                "matches_target_only_greedy": bool(correct),
                "target_only_tokens": ref,
                "correctness_reference": "same-row Target-only greedy with identical decoding config",
                "target_dtype": args.target_dtype,
                "verify_mode": args.verify_mode,
                "verify_blocks": dict(tgt.verify_mode_used),
            })
            print(json.dumps({"src": row["source_index"], "cond": cond,
                              "acc": c["accepted_draft_tokens"], "prop": c["proposed_draft_tokens"],
                              "identical_to_target_only": bool(correct)}, ensure_ascii=False), flush=True)
        (root / f"k4_results_shard_{args.shard_name}.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in records) + "\n", encoding="utf-8")
    (root / f"k4_results_shard_{args.shard_name}.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in records) + "\n", encoding="utf-8")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=("correctness", "eval"))
    p.add_argument("--root", required=True)
    p.add_argument("--source-dir", required=True)
    p.add_argument("--source-lock", required=True)
    p.add_argument("--teacher-model", required=True)
    p.add_argument("--student-model", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--validation-ledger", required=True)
    p.add_argument("--validation-prior-cache", required=True)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--target-max-memory", default="8GiB")
    p.add_argument("--target-dtype", default="bfloat16",
                   choices=("float16", "bfloat16"),
                   help="locked checkpoint is stored bfloat16; float16 produced "
                        "non-finite logits in the current environment (see "
                        "target_numerics_probe.json)")
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW)
    p.add_argument("--shard-name", default="0")
    p.add_argument("--shard-lo", type=int, default=0)
    p.add_argument("--shard-hi", type=int, default=4)
    p.add_argument("--verify-mode", default="batched", choices=("batched", "incremental"),
                   help="batched = one target forward per proposal block (usual spec-decoding "
                        "implementation); incremental = one target forward per proposal, i.e. "
                        "bit-identical in structure to the Target-only reference")
    p.add_argument("--corrupt-index", type=int, default=0,
                   help="0 disables; else corrupt that 0-based proposal position to force "
                        "partial acceptance")
    args = p.parse_args()
    return {"correctness": correctness, "eval": evaluate}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
