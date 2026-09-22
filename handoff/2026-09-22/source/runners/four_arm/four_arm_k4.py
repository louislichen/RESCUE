"""Four-arm fixed-K=4 Draft evaluation — single process, target balanced over the 4 cards.

Arms A/B/C/D only differ in how the draft is steered; Target is never steered
and verification uses the correctness-accepted incremental path.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from four_arm_common import (ARMS, ARM_POLICY, DECODE_PHASE, PREFILL_PHASE,
                            build_backend, cm_for, load_config, load_harness, load_modules,
                            maps_for)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--k4-dir", required=True)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--source-lock", required=True)
    ap.add_argument("--teacher-model", required=True)
    ap.add_argument("--student-model", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--validation-ledger", required=True)
    ap.add_argument("--validation-prior-cache", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--target-max-memory", default="8GiB")
    ap.add_argument("--target-dtype", default="bfloat16")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--shard-lo", type=int, default=0)
    ap.add_argument("--shard-hi", type=int, default=128)
    ap.add_argument("--shard-name", default="k0")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    H = load_harness(args.harness_root)
    backend_mod, runner_mod, steering = load_modules(H, args.source_dir, args.source_lock)
    sys.path.insert(0, str(Path(args.k4_dir).resolve()))
    k4 = __import__("real_k4_runner")
    from fixed_k4_protocol import fixed_k4_decode, greedy_reference

    backend = build_backend(backend_mod, args, steering)
    layers = set(backend.student_layers)
    steering_cm = steering.qwen2_adapter_steering
    model, device, dmap = k4.build_target(args)
    vocab = min(int(backend.student.config.vocab_size), int(model.config.vocab_size))
    rows = runner_mod.load_rows(Path(args.validation_ledger), Path(args.dataset_path))
    priors = torch.load(Path(args.validation_prior_cache), map_location="cpu",
                        weights_only=True)
    shard_rows = rows[args.shard_lo:args.shard_hi]

    class PolicyDraft(k4.CacheModel):
        def __init__(self, maps, alpha, window):
            super().__init__(backend.student, backend.device, vocab)
            self.lm, self.la = maps, alpha
            self.window = window
            self.prefill_mix_calls = 0
            self.decode_mix_calls = 0
            self.decode_steered_steps = 0

        def prefill(self, prompt_ids):
            with torch.no_grad():
                with cm_for(steering_cm, backend, self.lm, self.la, layers,
                            PREFILL_PHASE) as st:
                    o = self.model(input_ids=self._ids(prompt_ids),
                                   attention_mask=self._mask(len(prompt_ids)),
                                   use_cache=True, output_attentions=False)
                    self.prefill_mix_calls += 0 if st is None else len(st.mix_log)
            self.forward_calls += 1
            return self._greedy(o.logits[0, -1, :].float()), o.past_key_values

        def step(self, token, cache):
            total = cache.get_seq_length() + 1
            if self.window == "full" and self.lm:
                with torch.no_grad():
                    with cm_for(steering_cm, backend, self.lm, self.la, layers,
                                DECODE_PHASE) as st:
                        o = self.model(input_ids=self._ids([token]),
                                       attention_mask=self._mask(total),
                                       past_key_values=cache, use_cache=True,
                                       output_attentions=False)
                        self.decode_mix_calls += len(st.mix_log)
                self.decode_steered_steps += 1
            else:
                with torch.no_grad():
                    o = self.model(input_ids=self._ids([token]),
                                   attention_mask=self._mask(total),
                                   past_key_values=cache, use_cache=True,
                                   output_attentions=False)
            self.forward_calls += 1
            return self._greedy(o.logits[0, -1, :].float()), o.past_key_values

    body = out / f"k4_shard_{args.shard_name}.jsonl"
    records = []
    t_shard = time.time()
    for n, row in enumerate(shard_rows):
        sid = row["canonical_source_id"]
        target = backend.prompt_target(row["prompt"], row["answer"])
        prior = priors[sid]
        with torch.no_grad():
            rows_s = backend._student_rows(target)
            mask = torch.tensor(target.attention_mask, device=backend.device)
            mapped, alpha, _ = backend.adapter(prior.to(backend.device), rows_s, mask)
            null_map, null_alpha, _ = backend.adapter(prior.roll(1, dims=-1).to(backend.device),
                                                      rows_s, mask)
        maps_by_arm = {"teacher": maps_for(backend, mapped, alpha),
                       "null": maps_for(backend, null_map, null_alpha)}
        prompt_ids = target.input_ids
        ref = greedy_reference(prompt_ids,
                               k4.TeacherTarget(model, device, vocab, "incremental"),
                               k4.EOS_ID, args.max_new_tokens)
        for arm in ARMS:
            pol = ARM_POLICY[arm]
            maps, al = (({}, None) if pol["prior"] is None else maps_by_arm[pol["prior"]])
            draft = PolicyDraft(maps, al, pol["window"])
            tgt = k4.TeacherTarget(model, device, vocab, "incremental")
            spec = fixed_k4_decode(prompt_ids, draft, tgt, k4.EOS_ID, k=args.k,
                                   max_new_tokens=args.max_new_tokens)
            c = vars(spec.counters)
            rollbacks = c.get("rollback_events", []) or []
            rejected_prefixes = [int(e["accepted_before_reject"]) for e in rollbacks]
            records.append({
                "canonical_source_id": sid,
                "source_index": row["source_index"],
                "condition": arm,
                "arm": arm,
                "prompt_len": len(prompt_ids),
                "proposed_draft_tokens": c["proposed_draft_tokens"],
                "accepted_draft_tokens": c["accepted_draft_tokens"],
                "correction_tokens": c["correction_tokens"],
                "bonus_tokens": c["bonus_tokens"],
                "blocks": c["blocks"],
                "full_accept_blocks": c["full_accept_blocks"],
                "reject_blocks": c["reject_blocks"],
                "short_proposals": c["short_proposals"],
                "zero_accept_blocks": sum(1 for v in rejected_prefixes if v == 0),
                "partial_accept_blocks": sum(1 for v in rejected_prefixes if v > 0),
                "eos_terminated": bool(c["eos_terminated"]),
                "target_forward_calls": tgt.forward_calls,
                "draft_forward_calls": draft.forward_calls,
                "tokens": spec.token_ids,
                "target_only_tokens": ref,
                "matches_target_only_greedy": bool(spec.token_ids == ref),
                "verify_mode": "incremental",
                "k": args.k,
                "window": pol["window"],
                "prior_kind": pol["prior"],
                "target_dtype": args.target_dtype,
                "run_config_hash": cfg["config_hash"],
                "checkpoint_sha256": cfg["checkpoint_sha256"],
                "shard": args.shard_name,
            })
        H.write_result_envelope(
            body, task_id="RESCUE-MINIMAL-FOUR-ARM-VALID-RERUN-001", stage="k4_four_arm",
            records=records, arms=list(ARMS),
            sample_ids=sorted({r["canonical_source_id"] for r in records}),
            config_hash=cfg["config_hash"], checkpoint_sha256=cfg["checkpoint_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
