#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Speculative decoding acceptance experiment for attention transfer.

This script compares two draft models for greedy speculative decoding:

  - baseline: Qwen2-1.5B normal draft generation
  - steered:  Qwen2-1.5B draft generation with teacher-attention transfer

The verifier is Qwen2-7B. Token-level acceptance requires teacher and student
tokenizers to align on the prompt and generated draft ids; the script records
and skips samples where that precondition fails.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from run_hotpot_attn_transfer import (
    collect_teacher_priors,
    load_model_and_tokenizer,
    qwen2_attention_steering,
    score_prediction,
    selected_layer_indices,
    str2bool,
    str2dtype,
)


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")


@dataclass
class Timer:
    teacher_prior_s: float = 0.0
    student_draft_s: float = 0.0
    verifier_s: float = 0.0
    teacher_greedy_s: float = 0.0


@dataclass
class SpecStats:
    verifier_calls: int = 0
    blocks: int = 0
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    full_accept_blocks: int = 0
    correction_tokens: int = 0
    bonus_tokens: int = 0
    generated_tokens: int = 0
    first_token_matches: int = 0
    first_token_checks: int = 0
    token_id_mismatch: bool = False
    block_accept_lengths: List[int] = field(default_factory=list)
    timer: Timer = field(default_factory=Timer)


def tensor_ids(ids: Sequence[int], device: str) -> torch.Tensor:
    return torch.tensor([list(ids)], device=device, dtype=torch.long)


@torch.no_grad()
def next_token_plain(model, input_ids: Sequence[int], device: str) -> int:
    out = model(input_ids=tensor_ids(input_ids, device), use_cache=False, output_attentions=False)
    return int(torch.argmax(out.logits[:, -1, :], dim=-1).item())


@torch.no_grad()
def next_token_steered(
    teacher_model,
    teacher_tokenizer,
    student_model,
    student_tokenizer,
    input_ids: Sequence[int],
    text: str,
    layer_indices: Sequence[int],
    lambda_value: float,
    steering_mode: str,
    eps: float,
    teacher_head_mode: str,
    teacher_top_k: int,
    device: str,
    prior_cache: Optional[Dict[int, torch.Tensor]],
    timer: Timer,
) -> int:
    if prior_cache is None:
        t0 = time.perf_counter()
        prior_cache, _, _ = collect_teacher_priors(
            teacher_model,
            teacher_tokenizer,
            student_tokenizer,
            text,
            layer_indices,
            device,
            teacher_head_mode,
            teacher_top_k,
            eps,
            student_model=student_model,
        )
        timer.teacher_prior_s += time.perf_counter() - t0

    with qwen2_attention_steering(student_model, prior_cache, lambda_value, eps, steering_mode):
        out = student_model(
            input_ids=tensor_ids(input_ids, device),
            use_cache=False,
            output_attentions=False,
        )
    return int(torch.argmax(out.logits[:, -1, :], dim=-1).item())


@torch.no_grad()
def draft_tokens(
    mode: str,
    teacher_model,
    teacher_tokenizer,
    student_model,
    student_tokenizer,
    prefix_ids: Sequence[int],
    prompt: str,
    generated_ids: Sequence[int],
    draft_k: int,
    layer_indices: Sequence[int],
    lambda_value: float,
    steering_mode: str,
    eps: float,
    teacher_head_mode: str,
    teacher_top_k: int,
    device: str,
    prior_mode: str,
    prompt_prior_cache: Optional[Dict[int, torch.Tensor]],
    timer: Timer,
) -> List[int]:
    draft: List[int] = []
    for _ in range(draft_k):
        current_ids = list(prefix_ids) + list(generated_ids) + draft
        t0 = time.perf_counter()
        if mode == "baseline":
            next_id = next_token_plain(student_model, current_ids, device)
        elif mode == "steered":
            generated_text = student_tokenizer.decode(list(generated_ids) + draft, skip_special_tokens=True)
            current_text = prompt + generated_text
            prior_cache = prompt_prior_cache if prior_mode == "prompt_once" else None
            next_id = next_token_steered(
                teacher_model,
                teacher_tokenizer,
                student_model,
                student_tokenizer,
                current_ids,
                current_text,
                layer_indices,
                lambda_value,
                steering_mode,
                eps,
                teacher_head_mode,
                teacher_top_k,
                device,
                prior_cache,
                timer,
            )
        else:
            raise ValueError(f"Bad draft mode: {mode}")
        timer.student_draft_s += time.perf_counter() - t0
        draft.append(next_id)
    return draft


@torch.no_grad()
def verify_draft(
    teacher_model,
    prefix_ids: Sequence[int],
    generated_ids: Sequence[int],
    draft_ids: Sequence[int],
    device: str,
    timer: Timer,
) -> Tuple[int, Optional[int], List[int]]:
    """Return accepted length, correction/bonus token, teacher greedy checks."""
    if not draft_ids:
        return 0, None, []

    ids = list(prefix_ids) + list(generated_ids) + list(draft_ids)
    prefix_len = len(prefix_ids) + len(generated_ids)
    t0 = time.perf_counter()
    out = teacher_model(input_ids=tensor_ids(ids, device), use_cache=False, output_attentions=False)
    timer.verifier_s += time.perf_counter() - t0
    logits = out.logits[0]

    checks: List[int] = []
    accepted = 0
    for i, draft_id in enumerate(draft_ids):
        pred = int(torch.argmax(logits[prefix_len - 1 + i], dim=-1).item())
        checks.append(pred)
        if i == 0:
            pass
        if pred == int(draft_id):
            accepted += 1
            continue
        return accepted, pred, checks

    bonus_pos = prefix_len - 1 + len(draft_ids)
    bonus = None
    if bonus_pos < logits.shape[0]:
        bonus = int(torch.argmax(logits[bonus_pos], dim=-1).item())
    return accepted, bonus, checks


@torch.no_grad()
def teacher_greedy_decode(
    teacher_model,
    teacher_tokenizer,
    prefix_ids: Sequence[int],
    max_new_tokens: int,
    device: str,
    stop_on_question: bool,
    timer: Timer,
) -> Tuple[List[int], str]:
    generated: List[int] = []
    eos_id = teacher_tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        t0 = time.perf_counter()
        next_id = next_token_plain(teacher_model, list(prefix_ids) + generated, device)
        timer.teacher_greedy_s += time.perf_counter() - t0
        if eos_id is not None and next_id == eos_id:
            break
        generated.append(next_id)
        text = teacher_tokenizer.decode(generated, skip_special_tokens=True)
        if stop_on_question and ("\nQuestion:" in text or "\nQ:" in text):
            break
    return generated, teacher_tokenizer.decode(generated, skip_special_tokens=True)


@torch.no_grad()
def run_speculative(
    mode: str,
    teacher_model,
    teacher_tokenizer,
    student_model,
    student_tokenizer,
    prompt: str,
    max_new_tokens: int,
    draft_k: int,
    layer_indices: Sequence[int],
    lambda_value: float,
    steering_mode: str,
    eps: float,
    teacher_head_mode: str,
    teacher_top_k: int,
    device: str,
    prior_mode: str,
    append_teacher_on_full_accept: bool,
    stop_on_question: bool,
) -> Tuple[List[int], str, SpecStats]:
    stats = SpecStats()
    teacher_prefix_ids = teacher_tokenizer(prompt, add_special_tokens=False)["input_ids"]
    student_prefix_ids = student_tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if teacher_prefix_ids != student_prefix_ids:
        stats.token_id_mismatch = True
        return [], "", stats

    prompt_prior_cache = None
    if mode == "steered" and prior_mode == "prompt_once":
        t0 = time.perf_counter()
        prompt_prior_cache, _, _ = collect_teacher_priors(
            teacher_model,
            teacher_tokenizer,
            student_tokenizer,
            prompt,
            layer_indices,
            device,
            teacher_head_mode,
            teacher_top_k,
            eps,
            student_model=student_model,
        )
        stats.timer.teacher_prior_s += time.perf_counter() - t0

    generated: List[int] = []
    eos_id = student_tokenizer.eos_token_id

    while len(generated) < max_new_tokens:
        remain = max_new_tokens - len(generated)
        cur_k = min(draft_k, remain)
        draft = draft_tokens(
            mode,
            teacher_model,
            teacher_tokenizer,
            student_model,
            student_tokenizer,
            student_prefix_ids,
            prompt,
            generated,
            cur_k,
            layer_indices,
            lambda_value,
            steering_mode,
            eps,
            teacher_head_mode,
            teacher_top_k,
            device,
            prior_mode,
            prompt_prior_cache,
            stats.timer,
        )

        stats.blocks += 1
        stats.verifier_calls += 1
        stats.proposed_tokens += len(draft)

        accepted, correction_or_bonus, checks = verify_draft(
            teacher_model,
            teacher_prefix_ids,
            generated,
            draft,
            device,
            stats.timer,
        )
        stats.accepted_tokens += accepted
        stats.block_accept_lengths.append(accepted)
        if checks:
            stats.first_token_checks += 1
            if checks[0] == draft[0]:
                stats.first_token_matches += 1

        if accepted > 0:
            generated.extend(draft[:accepted])

        if accepted == len(draft):
            stats.full_accept_blocks += 1
            if append_teacher_on_full_accept and correction_or_bonus is not None and len(generated) < max_new_tokens:
                generated.append(correction_or_bonus)
                stats.bonus_tokens += 1
        elif correction_or_bonus is not None and len(generated) < max_new_tokens:
            generated.append(correction_or_bonus)
            stats.correction_tokens += 1

        generated = generated[:max_new_tokens]
        stats.generated_tokens = len(generated)
        if generated and eos_id is not None and generated[-1] == eos_id:
            break
        text = student_tokenizer.decode(generated, skip_special_tokens=True)
        if stop_on_question and ("\nQuestion:" in text or "\nQ:" in text):
            break

    text = student_tokenizer.decode(generated, skip_special_tokens=True)
    if stop_on_question:
        if "\nQuestion:" in text:
            text = text.split("\nQuestion:", 1)[0].rstrip()
        if "\nQ:" in text:
            text = text.split("\nQ:", 1)[0].rstrip()
    return generated, text, stats


def stats_to_json(stats: SpecStats) -> Dict[str, Any]:
    proposed = max(1, stats.proposed_tokens)
    blocks = max(1, stats.blocks)
    verifier_calls = max(1, stats.verifier_calls)
    return {
        "verifier_calls": stats.verifier_calls,
        "blocks": stats.blocks,
        "proposed_tokens": stats.proposed_tokens,
        "accepted_tokens": stats.accepted_tokens,
        "generated_tokens": stats.generated_tokens,
        "full_accept_blocks": stats.full_accept_blocks,
        "correction_tokens": stats.correction_tokens,
        "bonus_tokens": stats.bonus_tokens,
        "accept_rate": stats.accepted_tokens / proposed,
        "full_accept_rate": stats.full_accept_blocks / blocks,
        "mean_accept_len": sum(stats.block_accept_lengths) / blocks,
        "first_token_match_rate": stats.first_token_matches / max(1, stats.first_token_checks),
        "teacher_call_reduction_proxy": stats.generated_tokens / verifier_calls,
        "token_id_mismatch": stats.token_id_mismatch,
        "block_accept_lengths": stats.block_accept_lengths,
        "time": {
            "teacher_prior_s": stats.timer.teacher_prior_s,
            "student_draft_s": stats.timer.student_draft_s,
            "verifier_s": stats.timer.verifier_s,
            "teacher_greedy_s": stats.timer.teacher_greedy_s,
        },
    }


def aggregate(records: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    vals = [r[key] for r in records if key in r]
    if not vals:
        return {}
    out: Dict[str, Any] = {"n": len(vals)}
    for metric in [
        "accept_rate",
        "full_accept_rate",
        "mean_accept_len",
        "first_token_match_rate",
        "teacher_call_reduction_proxy",
    ]:
        xs = [v["stats"][metric] for v in vals if not v["stats"].get("token_id_mismatch")]
        out[metric] = sum(xs) / len(xs) if xs else None
    for metric in ["clean_em", "clean_f1", "em", "f1"]:
        xs = [v["score"][metric] for v in vals if not v["stats"].get("token_id_mismatch")]
        out[metric] = sum(xs) / len(xs) if xs else None
    time_keys = ["teacher_prior_s", "student_draft_s", "verifier_s", "teacher_greedy_s"]
    out["time"] = {}
    for tk in time_keys:
        xs = [v["stats"]["time"][tk] for v in vals if not v["stats"].get("token_id_mismatch")]
        out["time"][tk] = sum(xs) / len(xs) if xs else None
    return out


def parse_args():
    p = argparse.ArgumentParser("HotpotQA speculative acceptance with attention transfer")
    p.add_argument("--samples_path", required=True)
    p.add_argument("--teacher_model_path", default="/mnt/8T/cl/models/Qwen2-7B-Instruct")
    p.add_argument("--student_model_path", default="/mnt/8T/cl/models/Qwen2-1.5B-Instruct")
    p.add_argument("--max_samples", type=int, default=10)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--draft_k", type=int, default=4)
    p.add_argument("--lambda_value", type=float, default=1.0)
    p.add_argument("--layer_scope", default="last_layer")
    p.add_argument("--steering_mode", default="prob_mix", choices=["prob_mix", "log_prior"])
    p.add_argument("--prior_mode", default="prompt_once", choices=["prompt_once", "stepwise"])
    p.add_argument(
        "--teacher_head_mode",
        default="mean_all",
        choices=["mean_all", "topk_entropy_low", "topk_maxprob", "topk_doc_focus", "head_match_jsd"],
    )
    p.add_argument("--teacher_top_k", type=int, default=0)
    p.add_argument("--append_teacher_on_full_accept", type=str2bool, default=True)
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--stop_on_question", type=str2bool, default=True)
    p.add_argument("--output_dir", default="exp/attn_transfer/out_spec")
    p.add_argument("--output_name", default="spec_acceptance.jsonl")
    p.add_argument("--summary_name", default="summary.json")
    p.add_argument("--overwrite", type=str2bool, default=True)
    p.add_argument("--run_teacher_greedy", type=str2bool, default=True)
    return p.parse_args()


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    dtype = str2dtype(args.dtype)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.output_name)
    summary_path = os.path.join(args.output_dir, args.summary_name)
    if args.overwrite and os.path.exists(out_path):
        os.remove(out_path)
    if args.overwrite and os.path.exists(summary_path):
        os.remove(summary_path)

    print(f">>> loading teacher: {args.teacher_model_path}")
    teacher_tokenizer, teacher_model = load_model_and_tokenizer(args.teacher_model_path, dtype, args.device)
    print(f">>> loading student: {args.student_model_path}")
    student_tokenizer, student_model = load_model_and_tokenizer(args.student_model_path, dtype, args.device)

    n_layers = int(getattr(student_model.config, "num_hidden_layers"))
    layer_indices = selected_layer_indices(n_layers, args.layer_scope)

    records: List[Dict[str, Any]] = []
    seen = 0
    kept = 0
    skipped_tokenizer_mismatch = 0

    for sample in tqdm(read_jsonl(args.samples_path), total=args.max_samples, desc="spec", dynamic_ncols=True):
        if kept >= args.max_samples:
            break
        seen += 1
        prompt = sample["prompt"]
        gold = sample["gold"]
        teacher_ids = teacher_tokenizer(prompt, add_special_tokens=False)["input_ids"]
        student_ids = student_tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if teacher_ids != student_ids:
            skipped_tokenizer_mismatch += 1
            continue

        rec: Dict[str, Any] = {
            "sample_index": sample.get("source_index", seen - 1),
            "question": sample.get("question", ""),
            "gold": gold,
            "teacher_prompt_tokens": len(teacher_ids),
            "student_prompt_tokens": len(student_ids),
            "draft_k": args.draft_k,
            "max_new_tokens": args.max_new_tokens,
            "lambda": args.lambda_value,
            "layer_scope": args.layer_scope,
            "layer_indices": layer_indices,
            "steering_mode": args.steering_mode,
            "prior_mode": args.prior_mode,
            "teacher_head_mode": args.teacher_head_mode,
            "teacher_top_k": args.teacher_top_k,
        }

        if args.run_teacher_greedy:
            teacher_timer = Timer()
            _, teacher_text = teacher_greedy_decode(
                teacher_model,
                teacher_tokenizer,
                teacher_ids,
                args.max_new_tokens,
                args.device,
                args.stop_on_question,
                teacher_timer,
            )
            rec["teacher_greedy"] = {
                "text": teacher_text,
                "score": score_prediction(teacher_text, gold, True),
                "time": {"teacher_greedy_s": teacher_timer.teacher_greedy_s},
            }

        for mode in ["baseline", "steered"]:
            _, text, stats = run_speculative(
                mode,
                teacher_model,
                teacher_tokenizer,
                student_model,
                student_tokenizer,
                prompt,
                args.max_new_tokens,
                args.draft_k,
                layer_indices,
                args.lambda_value,
                args.steering_mode,
                args.eps,
                args.teacher_head_mode,
                args.teacher_top_k,
                args.device,
                args.prior_mode,
                args.append_teacher_on_full_accept,
                args.stop_on_question,
            )
            rec[mode] = {
                "text": text,
                "score": score_prediction(text, gold, True),
                "stats": stats_to_json(stats),
            }

        append_jsonl(out_path, rec)
        records.append(rec)
        kept += 1

    summary = {
        "samples_path": args.samples_path,
        "seen": seen,
        "kept": kept,
        "skipped_tokenizer_mismatch": skipped_tokenizer_mismatch,
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "draft_k": args.draft_k,
            "lambda": args.lambda_value,
            "layer_scope": args.layer_scope,
            "steering_mode": args.steering_mode,
            "prior_mode": args.prior_mode,
            "teacher_head_mode": args.teacher_head_mode,
            "teacher_top_k": args.teacher_top_k,
            "append_teacher_on_full_accept": args.append_teacher_on_full_accept,
        },
        "baseline": aggregate(records, "baseline"),
        "steered": aggregate(records, "steered"),
    }
    if args.run_teacher_greedy:
        teacher_scores = [r["teacher_greedy"]["score"] for r in records if "teacher_greedy" in r]
        summary["teacher_greedy"] = {
            "n": len(teacher_scores),
            "clean_em": sum(s["clean_em"] for s in teacher_scores) / max(1, len(teacher_scores)),
            "clean_f1": sum(s["clean_f1"] for s in teacher_scores) / max(1, len(teacher_scores)),
            "em": sum(s["em"] for s in teacher_scores) / max(1, len(teacher_scores)),
            "f1": sum(s["f1"] for s in teacher_scores) / max(1, len(teacher_scores)),
        }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
