#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Greedy QA evaluation for the trained attention adapter."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from tqdm import tqdm

from adapter import load_adapter_checkpoint
from common import (
    adapter_outputs_to_layer_maps,
    append_jsonl,
    collect_projected_teacher_heads,
    collect_student_rows,
    load_model_and_tokenizer,
    read_jsonl,
    sample_to_prompt,
    score_prediction,
    selected_layer_indices,
    str2bool,
    str2dtype,
    tensor_ids,
)
from steering import qwen2_adapter_steering


def parse_args():
    p = argparse.ArgumentParser("Evaluate attention adapter on HotpotQA")
    p.add_argument("--samples_path", default="exp/attn_transfer/data/hotpot_stream_300_maxL2048.jsonl")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--teacher_model_path", default="/mnt/8T/cl/models/Qwen2-7B-Instruct")
    p.add_argument("--student_model_path", default="/mnt/8T/cl/models/Qwen2-1.5B-Instruct")
    p.add_argument("--output_dir", default="exp/attn_adapter_transfer/out_quality")
    p.add_argument("--output_name", default="quality_adapter.jsonl")
    p.add_argument("--summary_name", default="summary.json")
    p.add_argument("--max_samples", type=int, default=20)
    p.add_argument("--skip_samples", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--max_prompt_tokens", type=int, default=1536)
    p.add_argument("--teacher_layer_scope", default="")
    p.add_argument("--student_layer_scope", default="")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--doc_mode", default="all", choices=["all", "gold_only"])
    p.add_argument("--stop_on_question", type=str2bool, default=True)
    p.add_argument("--overwrite", type=str2bool, default=True)
    return p.parse_args()


@torch.no_grad()
def next_token_plain(model, ids: Sequence[int], device: str) -> int:
    out = model(input_ids=tensor_ids(ids, device), use_cache=False, output_attentions=False)
    return int(torch.argmax(out.logits[:, -1, :], dim=-1).item())


@torch.no_grad()
def greedy_plain(model, tokenizer, prompt_ids: Sequence[int], max_new_tokens: int, device: str, stop_on_question: bool):
    generated: List[int] = []
    eos_id = tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        next_id = next_token_plain(model, list(prompt_ids) + generated, device)
        if eos_id is not None and next_id == eos_id:
            break
        generated.append(next_id)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        if stop_on_question and ("\nQuestion:" in text or "\nQ:" in text):
            break
    text = tokenizer.decode(generated, skip_special_tokens=True)
    if stop_on_question:
        text = text.split("\nQuestion:", 1)[0].split("\nQ:", 1)[0].rstrip()
    return text


@torch.no_grad()
def greedy_adapter(
    teacher_model,
    teacher_tokenizer,
    student_model,
    student_tokenizer,
    adapter,
    prompt: str,
    prompt_ids: Sequence[int],
    teacher_layer_indices: Sequence[int],
    student_layer_indices: Sequence[int],
    max_new_tokens: int,
    device: str,
    eps: float,
    stop_on_question: bool,
) -> Tuple[str, Dict[str, float]]:
    generated: List[int] = []
    eos_id = student_tokenizer.eos_token_id
    alpha_values: List[float] = []
    for _ in range(max_new_tokens):
        ids = list(prompt_ids) + generated
        current_text = prompt if not generated else student_tokenizer.decode(ids, skip_special_tokens=False)
        input_ids = tensor_ids(ids, device)
        teacher_projected = collect_projected_teacher_heads(
            teacher_model,
            teacher_tokenizer,
            student_tokenizer,
            current_text,
            teacher_layer_indices,
            device,
            eps,
        ).to(device)
        student_rows = collect_student_rows(student_model, input_ids, student_layer_indices).to(device)
        prior, alpha = adapter(teacher_projected, student_rows)
        alpha_values.append(float(alpha.mean().detach().cpu()))
        layer_to_prior, layer_to_alpha = adapter_outputs_to_layer_maps(prior, alpha, student_layer_indices)
        with qwen2_adapter_steering(student_model, layer_to_prior, layer_to_alpha, eps):
            out = student_model(input_ids=input_ids, use_cache=False, output_attentions=False)
        next_id = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())
        if eos_id is not None and next_id == eos_id:
            break
        generated.append(next_id)
        text = student_tokenizer.decode(generated, skip_special_tokens=True)
        if stop_on_question and ("\nQuestion:" in text or "\nQ:" in text):
            break
    text = student_tokenizer.decode(generated, skip_special_tokens=True)
    if stop_on_question:
        text = text.split("\nQuestion:", 1)[0].split("\nQ:", 1)[0].rstrip()
    return text, {"alpha_mean": sum(alpha_values) / max(1, len(alpha_values))}


def aggregate(records: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    vals = [r[key]["score"] for r in records]
    out = {"n": len(vals)}
    for metric in ["clean_em", "clean_f1", "em", "f1"]:
        out[metric] = sum(v[metric] for v in vals) / max(1, len(vals))
    return out


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name
    summary_path = out_dir / args.summary_name
    if args.overwrite and out_path.exists():
        out_path.unlink()
    if args.overwrite and summary_path.exists():
        summary_path.unlink()

    dtype = str2dtype(args.dtype)
    print(f">>> loading adapter: {args.checkpoint}")
    adapter, extra = load_adapter_checkpoint(args.checkpoint, map_location=args.device)
    adapter.to(args.device).eval()
    print(f">>> loading teacher: {args.teacher_model_path}")
    teacher_tokenizer, teacher_model = load_model_and_tokenizer(args.teacher_model_path, dtype, args.device)
    print(f">>> loading student: {args.student_model_path}")
    student_tokenizer, student_model = load_model_and_tokenizer(args.student_model_path, dtype, args.device)

    train_args = extra.get("args", {})
    teacher_scope = args.teacher_layer_scope or train_args.get("teacher_layer_scope", "last_4_layers")
    student_scope = args.student_layer_scope or train_args.get("student_layer_scope", "last_4_layers")
    teacher_layer_indices = selected_layer_indices(int(teacher_model.config.num_hidden_layers), teacher_scope)
    student_layer_indices = selected_layer_indices(int(student_model.config.num_hidden_layers), student_scope)

    records: List[Dict[str, Any]] = []
    seen = 0
    skipped = {"too_long": 0, "empty": 0}
    for sample in tqdm(read_jsonl(args.samples_path), total=args.skip_samples + args.max_samples, desc="quality", dynamic_ncols=True):
        if seen < args.skip_samples:
            seen += 1
            continue
        if len(records) >= args.max_samples:
            break
        seen += 1
        prompt, gold, question = sample_to_prompt(sample, args.doc_mode)
        if not prompt or not gold:
            skipped["empty"] += 1
            continue
        prompt_ids = student_tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) > args.max_prompt_tokens:
            skipped["too_long"] += 1
            continue
        baseline_text = greedy_plain(student_model, student_tokenizer, prompt_ids, args.max_new_tokens, args.device, args.stop_on_question)
        adapter_text, adapter_stats = greedy_adapter(
            teacher_model,
            teacher_tokenizer,
            student_model,
            student_tokenizer,
            adapter,
            prompt,
            prompt_ids,
            teacher_layer_indices,
            student_layer_indices,
            args.max_new_tokens,
            args.device,
            args.eps,
            args.stop_on_question,
        )
        rec = {
            "sample_index": sample.get("source_index", seen - 1),
            "question": question,
            "gold": gold,
            "teacher_layer_scope": teacher_scope,
            "student_layer_scope": student_scope,
            "baseline": {"text": baseline_text, "score": score_prediction(baseline_text, gold, True)},
            "adapter": {"text": adapter_text, "score": score_prediction(adapter_text, gold, True), "stats": adapter_stats},
        }
        append_jsonl(str(out_path), rec)
        records.append(rec)

    summary = {
        "checkpoint": args.checkpoint,
        "samples_path": args.samples_path,
        "seen": seen,
        "kept": len(records),
        "skipped": skipped,
        "teacher_layer_scope": teacher_scope,
        "student_layer_scope": student_scope,
        "baseline": aggregate(records, "baseline") if records else {},
        "adapter": aggregate(records, "adapter") if records else {},
        "alpha_mean": sum(r["adapter"]["stats"]["alpha_mean"] for r in records) / max(1, len(records)),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
