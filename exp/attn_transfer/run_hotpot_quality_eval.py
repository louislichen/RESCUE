#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Greedy answer-quality eval for baseline vs attention-steered student."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple

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


def tensor_ids(ids: Sequence[int], device: str) -> torch.Tensor:
    return torch.tensor([list(ids)], device=device, dtype=torch.long)


@torch.no_grad()
def next_token_plain(model, ids: Sequence[int], device: str) -> int:
    out = model(input_ids=tensor_ids(ids, device), use_cache=False, output_attentions=False)
    return int(torch.argmax(out.logits[:, -1, :], dim=-1).item())


@torch.no_grad()
def greedy_student(
    model,
    tokenizer,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    device: str,
    stop_on_question: bool,
) -> Tuple[List[int], str, float]:
    generated: List[int] = []
    eos_id = tokenizer.eos_token_id
    t0 = time.perf_counter()
    for _ in range(max_new_tokens):
        next_id = next_token_plain(model, list(prompt_ids) + generated, device)
        if eos_id is not None and next_id == eos_id:
            break
        generated.append(next_id)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        if stop_on_question and ("\nQuestion:" in text or "\nQ:" in text):
            break
    elapsed = time.perf_counter() - t0
    text = tokenizer.decode(generated, skip_special_tokens=True)
    if stop_on_question:
        if "\nQuestion:" in text:
            text = text.split("\nQuestion:", 1)[0].rstrip()
        if "\nQ:" in text:
            text = text.split("\nQ:", 1)[0].rstrip()
    return generated, text, elapsed


@torch.no_grad()
def greedy_student_steered(
    teacher_model,
    teacher_tokenizer,
    student_model,
    student_tokenizer,
    prompt: str,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    layer_indices: Sequence[int],
    lambda_value: float,
    steering_mode: str,
    prior_mode: str,
    eps: float,
    teacher_head_mode: str,
    teacher_top_k: int,
    device: str,
    stop_on_question: bool,
) -> Tuple[List[int], str, Dict[str, float]]:
    generated: List[int] = []
    eos_id = student_tokenizer.eos_token_id
    times = {"teacher_prior_s": 0.0, "student_decode_s": 0.0}

    prompt_prior_cache = None
    if prior_mode == "prompt_once":
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
        times["teacher_prior_s"] += time.perf_counter() - t0

    for _ in range(max_new_tokens):
        if prior_mode == "prompt_once":
            prior_cache = prompt_prior_cache
        else:
            current_text = prompt + student_tokenizer.decode(generated, skip_special_tokens=True)
            t0 = time.perf_counter()
            prior_cache, _, _ = collect_teacher_priors(
                teacher_model,
                teacher_tokenizer,
                student_tokenizer,
                current_text,
                layer_indices,
                device,
                teacher_head_mode,
                teacher_top_k,
                eps,
                student_model=student_model,
            )
            times["teacher_prior_s"] += time.perf_counter() - t0

        ids = list(prompt_ids) + generated
        t0 = time.perf_counter()
        with qwen2_attention_steering(student_model, prior_cache, lambda_value, eps, steering_mode):
            out = student_model(input_ids=tensor_ids(ids, device), use_cache=False, output_attentions=False)
        times["student_decode_s"] += time.perf_counter() - t0
        next_id = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())
        if eos_id is not None and next_id == eos_id:
            break
        generated.append(next_id)
        text = student_tokenizer.decode(generated, skip_special_tokens=True)
        if stop_on_question and ("\nQuestion:" in text or "\nQ:" in text):
            break

    text = student_tokenizer.decode(generated, skip_special_tokens=True)
    if stop_on_question:
        if "\nQuestion:" in text:
            text = text.split("\nQuestion:", 1)[0].rstrip()
        if "\nQ:" in text:
            text = text.split("\nQ:", 1)[0].rstrip()
    return generated, text, times


def aggregate(records: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    vals = [r[key] for r in records if key in r]
    out: Dict[str, Any] = {"n": len(vals)}
    for metric in ["clean_em", "clean_f1", "em", "f1"]:
        xs = [v["score"][metric] for v in vals]
        out[metric] = sum(xs) / len(xs) if xs else None
    time_keys = set()
    for v in vals:
        time_keys.update(v.get("time", {}).keys())
    out["time"] = {}
    for tk in sorted(time_keys):
        xs = [v["time"].get(tk, 0.0) for v in vals]
        out["time"][tk] = sum(xs) / len(xs) if xs else None
    return out


def parse_args():
    p = argparse.ArgumentParser("HotpotQA student quality eval")
    p.add_argument("--samples_path", required=True)
    p.add_argument("--teacher_model_path", default="/mnt/8T/cl/models/Qwen2-7B-Instruct")
    p.add_argument("--student_model_path", default="/mnt/8T/cl/models/Qwen2-1.5B-Instruct")
    p.add_argument("--max_samples", type=int, default=300)
    p.add_argument("--max_new_tokens", type=int, default=32)
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
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--stop_on_question", type=str2bool, default=True)
    p.add_argument("--output_dir", default="exp/attn_transfer/out_quality")
    p.add_argument("--output_name", default="quality_eval.jsonl")
    p.add_argument("--summary_name", default="summary.json")
    p.add_argument("--overwrite", type=str2bool, default=True)
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
    layer_indices = selected_layer_indices(int(student_model.config.num_hidden_layers), args.layer_scope)

    records: List[Dict[str, Any]] = []
    seen = 0
    kept = 0
    skipped_tokenizer_mismatch = 0
    for sample in tqdm(read_jsonl(args.samples_path), total=args.max_samples, desc="quality", dynamic_ncols=True):
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

        _, baseline_text, baseline_s = greedy_student(
            student_model,
            student_tokenizer,
            student_ids,
            args.max_new_tokens,
            args.device,
            args.stop_on_question,
        )
        _, steered_text, steered_time = greedy_student_steered(
            teacher_model,
            teacher_tokenizer,
            student_model,
            student_tokenizer,
            prompt,
            student_ids,
            args.max_new_tokens,
            layer_indices,
            args.lambda_value,
            args.steering_mode,
            args.prior_mode,
            args.eps,
            args.teacher_head_mode,
            args.teacher_top_k,
            args.device,
            args.stop_on_question,
        )
        rec = {
            "sample_index": sample.get("source_index", seen - 1),
            "question": sample.get("question", ""),
            "gold": gold,
            "teacher_prompt_tokens": len(teacher_ids),
            "student_prompt_tokens": len(student_ids),
            "lambda": args.lambda_value,
            "layer_scope": args.layer_scope,
            "layer_indices": layer_indices,
            "steering_mode": args.steering_mode,
            "prior_mode": args.prior_mode,
            "teacher_head_mode": args.teacher_head_mode,
            "teacher_top_k": args.teacher_top_k,
            "baseline": {
                "text": baseline_text,
                "score": score_prediction(baseline_text, gold, True),
                "time": {"student_decode_s": baseline_s},
            },
            "steered": {
                "text": steered_text,
                "score": score_prediction(steered_text, gold, True),
                "time": steered_time,
            },
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
            "lambda": args.lambda_value,
            "layer_scope": args.layer_scope,
            "steering_mode": args.steering_mode,
            "prior_mode": args.prior_mode,
            "teacher_head_mode": args.teacher_head_mode,
            "teacher_top_k": args.teacher_top_k,
        },
        "baseline": aggregate(records, "baseline"),
        "steered": aggregate(records, "steered"),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
