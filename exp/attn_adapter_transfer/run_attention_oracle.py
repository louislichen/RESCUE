#!/usr/bin/env python3
"""Fit a separate teacher-attention mixture at each first rejected draft state.

This is an offline controllability diagnostic, not an acceptance-rate evaluation.
Both Qwen2 models stay frozen. Only the teacher layer/head mixture and per-student
head gate are optimized; there is no temperature or student-residual adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from common import load_model_and_tokenizer, read_jsonl, sample_to_prompt, tensor_ids
from steering import qwen2_adapter_steering


class NumericallyUnstableState(RuntimeError):
    """An argmax changes when reconstructing the same BF16 prefix."""


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def legacy_cache(cache):
    return cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache


def core_forward(model, ids, cache=None, *, attentions=False, hidden_states=False):
    # Tuple caches create a fresh DynamicCache on each call: the immutable prefix
    # must never accumulate the last token across optimization iterations.
    return model.model(
        input_ids=ids,
        past_key_values=cache,
        use_cache=True,
        output_attentions=attentions,
        output_hidden_states=hidden_states,
        return_dict=True,
    )


def logits_of(model, out):
    # Avoid allocating vocabulary logits for every prompt token during prefill.
    return model.lm_head(out.last_hidden_state[:, -1:, :])[:, -1, :].float()


@torch.no_grad()
def prefix_cache(model, ids, device):
    if len(ids) <= 1:
        return None
    out = core_forward(model, tensor_ids(ids[:-1], device))
    return legacy_cache(out.past_key_values)


@torch.no_grad()
def first_rejection(teacher, student, ids, tokenizer, args):
    """Run normal greedy SD (k=4, full-accept bonus) until its first rejection."""
    teacher_cache = prefix_cache(teacher, ids, args.device)
    student_cache = prefix_cache(student, ids, args.device)
    last_id = ids[-1]
    generated = []
    blocks = 0
    while len(generated) < args.max_new_tokens:
        k = min(args.draft_k, args.max_new_tokens - len(generated))
        draft = []
        next_input = last_id
        for _ in range(k):
            out = core_forward(student, tensor_ids([next_input], args.device), student_cache)
            student_cache = legacy_cache(out.past_key_values)
            next_input = int(logits_of(student, out).argmax(dim=-1).item())
            draft.append(next_input)
        out = core_forward(teacher, tensor_ids([last_id] + draft, args.device), teacher_cache)
        checks = teacher.lm_head(out.last_hidden_state).float().argmax(dim=-1)[0].tolist()
        blocks += 1
        for index, proposed in enumerate(draft):
            target = int(checks[index])
            if target != proposed:
                return {
                    "input_ids": ids + generated + draft[:index],
                    "generated_prefix_ids": generated + draft[:index],
                    "teacher_target_id": target,
                    "baseline_draft_id": int(proposed),
                    "block_index": blocks - 1,
                    "position_in_block": index,
                    "generated_position": len(generated) + index,
                }
            if proposed == tokenizer.eos_token_id:
                return None
        generated.extend(draft)
        if len(generated) >= args.max_new_tokens:
            return None
        bonus = int(checks[-1])
        if bonus == tokenizer.eos_token_id:
            return None
        generated.append(bonus)
        # The verifier cache already contains every accepted draft token. The
        # student's draft loop has not yet consumed its final draft token.
        teacher_cache = legacy_cache(out.past_key_values)
        student_out = core_forward(student, tensor_ids([draft[-1]], args.device), student_cache)
        student_cache = legacy_cache(student_out.past_key_values)
        last_id = bonus
        text = tokenizer.decode(generated, skip_special_tokens=True)
        if "\nQuestion:" in text or "\nQ:" in text:
            return None
    return None


def collect_states(teacher, student, teacher_tokenizer, tokenizer, args, output, target_count=None):
    target_count = args.max_states if target_count is None else target_count
    states_path = output / "states.jsonl"
    states = list(read_jsonl(str(states_path))) if states_path.exists() else []
    if len(states) >= target_count:
        return states[:target_count]
    progress_path = output / "selection.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        "scanned": 0,
        "kept": 0,
        "skipped": {"too_long": 0, "tokenization_mismatch": 0, "no_rejection": 0},
    }
    # Recover safely if interrupted after appending a state but before updating
    # the scan cursor. There is at most one state per input sample.
    if states:
        progress["scanned"] = max(progress["scanned"], states[-1]["sample_cursor"] + 1)
    for cursor, sample in enumerate(read_jsonl(args.samples_path)):
        if cursor < progress["scanned"]:
            continue
        if len(states) >= target_count:
            break
        prompt, gold, question = sample_to_prompt(sample, "all")
        encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded["input_ids"]
        teacher_ids = teacher_tokenizer(prompt, add_special_tokens=False)["input_ids"]
        progress["scanned"] = cursor + 1
        if len(ids) > args.max_prompt_tokens:
            progress["skipped"]["too_long"] += 1
        elif ids != teacher_ids:
            progress["skipped"]["tokenization_mismatch"] += 1
        else:
            state = first_rejection(teacher, student, ids, tokenizer, args)
            if state is None:
                progress["skipped"]["no_rejection"] += 1
            else:
                doc_end = prompt.rfind("\n\nQuestion:")
                if doc_end < 0:
                    raise ValueError("Cannot locate the document/question boundary.")
                # Keep the attention sink, question, and generated suffix fixed
                # in the shuffle control. Only document token positions move.
                document_positions = [
                    index for index, (start, end) in enumerate(encoded["offset_mapping"])
                    if index > 0 and end > start and end <= doc_end
                ]
                state.update({
                    "state_index": len(states),
                    "sample_cursor": cursor,
                    "source_index": sample.get("source_index", cursor),
                    "question": question,
                    "gold": gold,
                    "prompt_tokens": len(ids),
                    "document_positions": document_positions,
                    "teacher_target_text": tokenizer.decode([state["teacher_target_id"]]),
                    "baseline_draft_text": tokenizer.decode([state["baseline_draft_id"]]),
                })
                append_json(states_path, state)
                states.append(state)
                print(f"selected candidate {len(states)}/{target_count}: sample={cursor}, "
                      f"position={state['generated_position']}, "
                      f"target={state['teacher_target_text']!r}, draft={state['baseline_draft_text']!r}", flush=True)
        progress["kept"] = len(states)
        write_json(progress_path, progress)
    return states[:target_count]


class HeadMixture(torch.nn.Module):
    def __init__(self, layers, heads, teacher_heads, *, seed, restart, device):
        super().__init__()
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(layers * heads, teacher_heads, generator=generator, device=device)
        self.mix_logits = torch.nn.Parameter(noise * (0.0 if restart == 0 else 0.05))
        alpha = 0.2 if restart == 0 else 0.8
        self.gate_logits = torch.nn.Parameter(torch.full(
            (layers, heads), math.log(alpha / (1.0 - alpha)), device=device
        ))
        self.layers, self.heads = layers, heads

    def forward(self, bank):
        weights = self.mix_logits.softmax(dim=-1)
        prior = (weights @ bank).reshape(self.layers, self.heads, bank.shape[-1])
        return prior, self.gate_logits.sigmoid()


class LastLayersDecoder:
    """Exact reuse of unaffected lower layers for a last-query-only intervention."""
    def __init__(self, model, ids, start_layer, device):
        self.model, self.start_layer = model, start_layer
        self.cache = prefix_cache(model, ids, device)
        self.last_input = tensor_ids(ids[-1:], device)
        self.position_ids = torch.tensor([[len(ids) - 1]], device=device)
        with torch.no_grad():
            out = core_forward(model, self.last_input, self.cache, attentions=True, hidden_states=True)
            self.hidden = out.hidden_states[start_layer].detach()
            self.baseline_logits = logits_of(model, out).detach()
        self.input_len = len(ids)

    def forward(self):
        # Rebuild the mutable cache wrapper every time; the tuple's tensors are
        # read-only. Causality leaves every earlier token/layer unaffected.
        cache = DynamicCache.from_legacy_cache(self.cache)
        hidden = self.hidden
        for layer in self.model.model.layers[self.start_layer :]:
            hidden = layer(
                hidden,
                attention_mask=None,  # One final query sees all cached keys.
                position_ids=self.position_ids,
                past_key_value=cache,
                output_attentions=False,
                use_cache=True,
            )[0]
        hidden = self.model.model.norm(hidden)
        return self.model.lm_head(hidden)[:, -1, :].float()

    def full_cached_forward(self):
        return logits_of(self.model, core_forward(self.model, self.last_input, self.cache))


@torch.no_grad()
def teacher_bank(teacher, state, layers, device):
    ids = state["input_ids"]
    cache = prefix_cache(teacher, ids, device)
    out = core_forward(teacher, tensor_ids(ids[-1:], device), cache, attentions=True)
    target = int(logits_of(teacher, out).argmax(dim=-1).item())
    if target != state["teacher_target_id"]:
        raise NumericallyUnstableState(f"Verifier/cached teacher argmax differs: {target} vs {state['teacher_target_id']}")
    bank = torch.cat([out.attentions[layer][0, :, -1, :].float() for layer in layers], dim=0)
    return bank.detach() / bank.sum(dim=-1, keepdim=True)


def metric(logits, target):
    row = logits.detach()[0]
    target_logit = row[target]
    competitors = row.clone()
    competitors[target] = -torch.inf
    predicted = int(row.argmax().item())
    return {
        "ce": float(F.cross_entropy(logits.detach(), torch.tensor([target], device=logits.device)).item()),
        "target_prob": float(row.softmax(dim=-1)[target].item()),
        "target_rank": int((row > target_logit).sum().item()) + 1,
        "target_margin": float((target_logit - competitors.max()).item()),
        "argmax_id": predicted,
        "hit": predicted == target,
    }


def update_maps(priors, gates, layers, layer_to_prior, layer_to_alpha):
    layer_to_prior.clear()
    layer_to_alpha.clear()
    for index, layer in enumerate(layers):
        layer_to_prior[layer] = priors[index]
        layer_to_alpha[layer] = gates[index]


def sanity_checks(engine, bank, target, layers, state, args):
    priors = bank.mean(dim=0).reshape(1, 1, -1).expand(len(layers), engine.model.config.num_attention_heads, -1)
    heads = engine.model.config.num_attention_heads
    layer_to_prior, layer_to_alpha = {}, {}
    checks = {}
    with torch.no_grad():
        plain_fast = engine.forward()
    checks["plain_fast_cached_max_logit_error"] = float((plain_fast - engine.baseline_logits).abs().max().item())
    torch.testing.assert_close(plain_fast, engine.baseline_logits, rtol=0, atol=1e-4)
    with qwen2_adapter_steering(engine.model, layer_to_prior, layer_to_alpha, 1e-12):
        zeros = torch.zeros(len(layers), heads, device=args.device)
        update_maps(priors, zeros, layers, layer_to_prior, layer_to_alpha)
        with torch.no_grad():
            zero_logits = engine.forward()
        checks["zero_gate_max_logit_error"] = float((zero_logits - engine.baseline_logits).abs().max().item())
        torch.testing.assert_close(zero_logits, engine.baseline_logits, rtol=0, atol=1e-4)

        mixture = HeadMixture(len(layers), heads, bank.shape[0], seed=args.seed, restart=1, device=args.device)
        prior, gates = mixture(bank)
        update_maps(prior, gates, layers, layer_to_prior, layer_to_alpha)
        fast_logits = engine.forward()
        fast_loss = F.cross_entropy(fast_logits, torch.tensor([target], device=args.device))
        fast_gradients = torch.autograd.grad(fast_loss, tuple(mixture.parameters()), retain_graph=True)
        full_logits = engine.full_cached_forward()
        full_loss = F.cross_entropy(full_logits, torch.tensor([target], device=args.device))
        full_gradients = torch.autograd.grad(full_loss, tuple(mixture.parameters()))
        checks["steered_fast_full_max_logit_error"] = float((fast_logits.detach() - full_logits.detach()).abs().max().item())
        torch.testing.assert_close(fast_logits, full_logits, rtol=0, atol=1e-4)
        checks["mix_gradient_norm"] = float(fast_gradients[0].norm().item())
        checks["gate_gradient_norm"] = float(fast_gradients[1].norm().item())
        checks["fast_full_gradient_max_error"] = max(
            float((fast - full).abs().max().item()) for fast, full in zip(fast_gradients, full_gradients)
        )
        for fast, full in zip(fast_gradients, full_gradients):
            if not torch.isfinite(fast).all() or fast.norm() == 0:
                raise RuntimeError("Missing or non-finite optimization gradient.")
            torch.testing.assert_close(fast, full, rtol=1e-3, atol=1e-5)

    with torch.no_grad():
        full_plain = logits_of(engine.model, core_forward(engine.model, tensor_ids(state["input_ids"], args.device)))
    checks["plain_cached_full_max_logit_error"] = float((full_plain - engine.baseline_logits).abs().max().item())
    checks["plain_cached_full_argmax_match"] = bool(full_plain.argmax().item() == engine.baseline_logits.argmax().item())
    return checks


def fit_state(engine, bank, target, layers, args, state_index, mode, output):
    heads = engine.model.config.num_attention_heads
    best = metric(engine.baseline_logits, target)
    best.update({"restart": -1, "step": 0, "alpha_mean": 0.0})
    best_weights = None
    first_hit = None
    total_updates = 0
    traces = []
    started = time.perf_counter()
    layer_to_prior, layer_to_alpha = {}, {}
    with qwen2_adapter_steering(engine.model, layer_to_prior, layer_to_alpha, 1e-12):
        for restart in range(args.restarts):
            mixture = HeadMixture(
                len(layers), heads, bank.shape[0], seed=args.seed + 1000 * state_index + restart,
                restart=restart, device=args.device,
            )
            optimizer = torch.optim.Adam(mixture.parameters(), lr=args.lr)
            for step in range(args.steps + 1):
                prior, gates = mixture(bank)
                update_maps(prior, gates, layers, layer_to_prior, layer_to_alpha)
                logits = engine.forward()
                loss = F.cross_entropy(logits, torch.tensor([target], device=args.device))
                current = metric(logits, target)
                current.update({"restart": restart, "step": step, "alpha_mean": float(gates.detach().mean().item())})
                if not math.isfinite(current["ce"]):
                    raise RuntimeError("Non-finite loss.")
                if current["ce"] < best["ce"]:
                    best = current
                    best_weights = {name: value.detach().cpu().clone() for name, value in mixture.state_dict().items()}
                if current["hit"] and first_hit is None:
                    first_hit = dict(current)
                    # A hit may have lower target probability than a previous
                    # non-hit. Preserve a successfully matching checkpoint.
                    best = current
                    best_weights = {name: value.detach().cpu().clone() for name, value in mixture.state_dict().items()}
                if step == 0 or step % 10 == 0 or current["hit"] or step == args.steps:
                    traces.append({"state_index": state_index, "mode": mode, **current})
                if current["hit"] or step == args.steps:
                    break
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for parameter in mixture.parameters():
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise RuntimeError("Non-finite or missing mixture/gate gradient.")
                optimizer.step()
                total_updates += 1
            if first_hit is not None:
                break
    if best_weights is not None:
        checkpoint_dir = output / "state_weights"
        checkpoint_dir.mkdir(exist_ok=True)
        torch.save({"state_index": state_index, "mode": mode, "weights": best_weights, "metric": best},
                   checkpoint_dir / f"state_{state_index:03d}_{mode}.pt")
    for trace in traces:
        append_json(output / "optimization_trace.jsonl", trace)
    return {
        "best": best,
        "repaired": first_hit is not None,
        "first_hit": first_hit,
        "optimizer_updates": total_updates,
        "seconds": time.perf_counter() - started,
    }


def summarize(records, args, output, config):
    n = len(records)
    summary = {
        "diagnostic": "per-state optimization on first rejected greedy draft tokens; not held-out SD acceptance",
        "completed_states": n,
        "requested_states": args.max_states,
        "config": config,
        "validation": {},
        "discarded_numerically_unstable_states": sum(1 for _ in read_jsonl(str(output / "discarded_states.jsonl")))
        if (output / "discarded_states.jsonl").exists() else 0,
    }
    if n:
        summary["baseline"] = {
            "hits": sum(record["baseline"]["hit"] for record in records),
            "mean_target_prob": sum(record["baseline"]["target_prob"] for record in records) / n,
            "mean_target_rank": sum(record["baseline"]["target_rank"] for record in records) / n,
        }
        for mode in ("teacher", "shuffled"):
            summary[mode] = {
                "repaired": sum(record[mode]["repaired"] for record in records),
                "repair_fraction": sum(record[mode]["repaired"] for record in records) / n,
                "mean_target_prob": sum(record[mode]["best"]["target_prob"] for record in records) / n,
                "mean_target_rank": sum(record[mode]["best"]["target_rank"] for record in records) / n,
                "mean_target_margin": sum(record[mode]["best"]["target_margin"] for record in records) / n,
                "mean_alpha": sum(record[mode]["best"]["alpha_mean"] for record in records) / n,
                "optimizer_updates": sum(record[mode]["optimizer_updates"] for record in records),
                "total_seconds": sum(record[mode]["seconds"] for record in records),
            }
        summary["paired_repair"] = {
            "both": sum(record["teacher"]["repaired"] and record["shuffled"]["repaired"] for record in records),
            "teacher_only": sum(record["teacher"]["repaired"] and not record["shuffled"]["repaired"] for record in records),
            "shuffled_only": sum(not record["teacher"]["repaired"] and record["shuffled"]["repaired"] for record in records),
            "neither": sum(not record["teacher"]["repaired"] and not record["shuffled"]["repaired"] for record in records),
        }
        checked = [record["validation"] for record in records if record.get("validation")]
        summary["validation"]["checked_states"] = len(checked)
        for key in ("zero_gate_max_logit_error", "steered_fast_full_max_logit_error", "fast_full_gradient_max_error"):
            summary["validation"][key] = max((check[key] for check in checked), default=0)
        summary["validation"]["plain_cached_full_argmax_matches"] = sum(
            check["plain_cached_full_argmax_match"] for check in checked
        )
    write_json(output / "summary.json", summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--samples_path", default="exp/attn_transfer/data/hotpot_stream_300_maxL2048.jsonl")
    parser.add_argument("--teacher_model_path", default="/mnt/8T/cl/models/Qwen2-7B-Instruct")
    parser.add_argument("--student_model_path", default="/mnt/8T/cl/models/Qwen2-1.5B-Instruct")
    parser.add_argument("--output_dir", default="exp/attn_adapter_transfer/out_oracle_last4")
    parser.add_argument("--max_states", type=int, default=8)
    parser.add_argument("--max_prompt_tokens", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--draft_k", type=int, default=4)
    parser.add_argument("--last_layers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate_states", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.steps < 1 or args.restarts < 1 or args.max_states < 1 or args.last_layers < 1 or args.lr <= 0:
        parser.error("steps, restarts, max_states, last_layers, and lr must be positive")
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    # Avoid excessive CPU thread overhead when loading weights and preparing
    # these tiny decode tensors. All model arithmetic runs on the GPU.
    torch.set_num_threads(4)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "samples_path": str(Path(args.samples_path).resolve()),
        "samples_sha256": hashlib.sha256(Path(args.samples_path).read_bytes()).hexdigest(),
        "teacher_model_path": args.teacher_model_path,
        "student_model_path": args.student_model_path,
        "last_layers": args.last_layers,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "draft_k": args.draft_k,
        "steps_per_restart": args.steps,
        "restarts": args.restarts,
        "lr": args.lr,
        "seed": args.seed,
        "dtype": "bfloat16",
        "shuffle": "same permutation for all teacher heads, document positions only; first token/question/generated suffix fixed",
        "stop_rule": "stop mode after first argmax match; otherwise exhaust all restarts",
        "teacher_projection": "identity after checking equal token vocabularies and prompt token IDs",
        "torch": torch.__version__,
    }
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing output uses a different configuration; choose a new output_dir.")
    write_json(config_path, config)
    print("Loading frozen teacher and student", flush=True)
    teacher_tokenizer, teacher = load_model_and_tokenizer(args.teacher_model_path, torch.bfloat16, args.device)
    tokenizer, student = load_model_and_tokenizer(args.student_model_path, torch.bfloat16, args.device)
    for model in (teacher, student):
        model.requires_grad_(False)
        model.eval()
    if teacher_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise ValueError("This first rejection diagnostic requires identical token vocabularies.")
    if args.last_layers > min(teacher.config.num_hidden_layers, student.config.num_hidden_layers):
        raise ValueError("last_layers exceeds model depth")
    teacher_layers = list(range(teacher.config.num_hidden_layers - args.last_layers, teacher.config.num_hidden_layers))
    student_layers = list(range(student.config.num_hidden_layers - args.last_layers, student.config.num_hidden_layers))
    records_path = output / "oracle_results.jsonl"
    records = list(read_jsonl(str(records_path))) if records_path.exists() else []
    done = {record["state_index"] for record in records}
    discarded_path = output / "discarded_states.jsonl"
    discarded = {row["state_index"] for row in read_jsonl(str(discarded_path))} if discarded_path.exists() else set()
    while len(records) < args.max_states:
        target_count = args.max_states + len(discarded)
        states = collect_states(teacher, student, teacher_tokenizer, tokenizer, args, output, target_count)
        for state in states:
            if state["state_index"] in done or state["state_index"] in discarded:
                continue
            if len(records) >= args.max_states:
                break
            started = time.perf_counter()
            bank, engine = None, None
            try:
                bank = teacher_bank(teacher, state, teacher_layers, args.device)
                engine = LastLayersDecoder(student, state["input_ids"], student_layers[0], args.device)
                target = state["teacher_target_id"]
                if target >= student.config.vocab_size:
                    raise ValueError("Teacher target is outside the student output vocabulary.")
                baseline = metric(engine.baseline_logits, target)
                if baseline["argmax_id"] != state["baseline_draft_id"] or baseline["hit"]:
                    raise NumericallyUnstableState(
                        f"Student reconstructed argmax {baseline['argmax_id']} differs from draft {state['baseline_draft_id']}"
                    )
            except NumericallyUnstableState as error:
                append_json(discarded_path, {"state_index": state["state_index"], "source_index": state["source_index"], "reason": str(error)})
                discarded.add(state["state_index"])
                print(f"discard unstable state={state['state_index']}: {error}", flush=True)
                del bank, engine
                continue
            validation = sanity_checks(engine, bank, target, student_layers, state, args) if state["state_index"] < args.validate_states else {}
            positions = torch.tensor(state["document_positions"], device=args.device, dtype=torch.long)
            generator = torch.Generator(device=args.device).manual_seed(args.seed + state["state_index"])
            permuted = positions[torch.randperm(len(positions), generator=generator, device=args.device)]
            shuffled = bank.clone()
            shuffled[:, positions] = bank[:, permuted]
            record = {
                "state_index": state["state_index"],
                "source_index": state["source_index"],
                "question": state["question"],
                "generated_position": state["generated_position"],
                "teacher_target_id": target,
                "teacher_target_text": state["teacher_target_text"],
                "baseline_draft_text": state["baseline_draft_text"],
                "baseline": baseline,
                "validation": validation,
            }
            for mode, current_bank in (("teacher", bank), ("shuffled", shuffled)):
                record[mode] = fit_state(engine, current_bank, target, student_layers, args, state["state_index"], mode, output)
            record["seconds"] = time.perf_counter() - started
            append_json(records_path, record)
            records.append(record)
            done.add(state["state_index"])
            summary = summarize(records, args, output, config)
            print(f"state={state['state_index']:02d} target={state['teacher_target_text']!r} "
                  f"baseline={state['baseline_draft_text']!r} "
                  f"teacher={record['teacher']['repaired']} shuffled={record['shuffled']['repaired']} "
                  f"elapsed={record['seconds']:.1f}s "
                  f"cumulative={summary['teacher']['repaired']}/{len(records)} vs "
                  f"{summary['shuffled']['repaired']}/{len(records)}", flush=True)
            del bank, engine, shuffled
        if len(states) < target_count and len(records) < args.max_states:
            summarize(records, args, output, config)
            raise RuntimeError(f"Dataset exhausted with only {len(records)} numerically stable states.")
    summary = summarize(records, args, output, config)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
