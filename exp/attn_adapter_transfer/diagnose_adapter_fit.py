#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Diagnose whether an attention adapter fits its token-level training target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from adapter import load_adapter_checkpoint
from common import (
    adapter_outputs_to_layer_maps,
    append_jsonl,
    collect_projected_teacher_heads,
    collect_student_rows,
    gold_target_ids,
    greedy_text,
    load_model_and_tokenizer,
    read_jsonl,
    sample_to_prompt,
    selected_layer_indices,
    str2bool,
    str2dtype,
    tensor_ids,
)
from steering import qwen2_adapter_steering


def parse_args():
    p = argparse.ArgumentParser("Diagnose adapter fitting")
    p.add_argument("--samples_path", default="exp/attn_transfer/data/hotpot_stream_300_maxL2048.jsonl")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--teacher_model_path", default="/mnt/8T/cl/models/Qwen2-7B-Instruct")
    p.add_argument("--student_model_path", default="/mnt/8T/cl/models/Qwen2-1.5B-Instruct")
    p.add_argument("--output_dir", default="exp/attn_adapter_transfer/out_fit_diag")
    p.add_argument("--output_name", default="fit_diag.jsonl")
    p.add_argument("--summary_name", default="summary.json")
    p.add_argument("--max_samples", type=int, default=32)
    p.add_argument("--skip_samples", type=int, default=0)
    p.add_argument("--answer_tokens_per_sample", type=int, default=4)
    p.add_argument("--target_mode", default="", choices=["", "gold", "teacher_greedy"])
    p.add_argument("--teacher_target_tokens", type=int, default=16)
    p.add_argument("--max_prompt_tokens", type=int, default=2048)
    p.add_argument("--teacher_layer_scope", default="")
    p.add_argument("--student_layer_scope", default="")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--doc_mode", default="all", choices=["all", "gold_only"])
    p.add_argument("--overwrite", type=str2bool, default=True)
    return p.parse_args()


def build_target_ids(args, train_args, teacher_model, teacher_tokenizer, student_tokenizer, prompt: str, gold: str) -> List[int]:
    mode = args.target_mode or train_args.get("target_mode", "gold")
    if mode == "gold":
        return gold_target_ids(student_tokenizer, prompt, gold)
    teacher_tokens = args.teacher_target_tokens or int(train_args.get("teacher_target_tokens", 16))
    teacher_answer = greedy_text(teacher_model, teacher_tokenizer, prompt, teacher_tokens, args.device, stop_on_question=True)
    return student_tokenizer(teacher_answer, add_special_tokens=False)["input_ids"]


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
    train_args = extra.get("args", {})

    print(f">>> loading teacher: {args.teacher_model_path}")
    teacher_tokenizer, teacher_model = load_model_and_tokenizer(args.teacher_model_path, dtype, args.device)
    print(f">>> loading student: {args.student_model_path}")
    student_tokenizer, student_model = load_model_and_tokenizer(args.student_model_path, dtype, args.device)

    teacher_scope = args.teacher_layer_scope or train_args.get("teacher_layer_scope", "last_4_layers")
    student_scope = args.student_layer_scope or train_args.get("student_layer_scope", "last_4_layers")
    teacher_layer_indices = selected_layer_indices(int(teacher_model.config.num_hidden_layers), teacher_scope)
    student_layer_indices = selected_layer_indices(int(student_model.config.num_hidden_layers), student_scope)

    totals: Dict[str, float] = {
        "tokens": 0,
        "baseline_ce": 0.0,
        "adapter_ce": 0.0,
        "baseline_argmax_hits": 0,
        "adapter_argmax_hits": 0,
        "adapter_changed_argmax": 0,
        "adapter_target_prob_delta": 0.0,
        "alpha_mean": 0.0,
    }
    kept = 0
    seen = 0
    skipped = {"empty": 0, "too_long": 0, "no_target": 0}

    for sample in tqdm(read_jsonl(args.samples_path), total=args.skip_samples + args.max_samples, desc="fit", dynamic_ncols=True):
        if seen < args.skip_samples:
            seen += 1
            continue
        if kept >= args.max_samples:
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
        target_ids = build_target_ids(args, train_args, teacher_model, teacher_tokenizer, student_tokenizer, prompt, gold)
        if not target_ids:
            skipped["no_target"] += 1
            continue

        running_prefix = list(prompt_ids)
        rec_tokens = []
        for token_step, target_id in enumerate(target_ids[: args.answer_tokens_per_sample]):
            input_ids = tensor_ids(running_prefix, args.device)
            current_text = prompt if token_step == 0 else student_tokenizer.decode(running_prefix, skip_special_tokens=False)

            with torch.no_grad():
                base_out = student_model(input_ids=input_ids, use_cache=False, output_attentions=False)
                base_logits = base_out.logits[:, -1, :].float()
                base_ce = F.cross_entropy(base_logits, torch.tensor([target_id], device=args.device))
                base_argmax = int(torch.argmax(base_logits, dim=-1).item())
                base_prob = float(torch.softmax(base_logits, dim=-1)[0, target_id].detach().cpu())

                teacher_projected = collect_projected_teacher_heads(
                    teacher_model,
                    teacher_tokenizer,
                    student_tokenizer,
                    current_text,
                    teacher_layer_indices,
                    args.device,
                    args.eps,
                ).to(args.device)
                student_rows = collect_student_rows(student_model, input_ids, student_layer_indices).to(args.device)
                prior, alpha = adapter(teacher_projected, student_rows)
                layer_to_prior, layer_to_alpha = adapter_outputs_to_layer_maps(prior, alpha, student_layer_indices)
                with qwen2_adapter_steering(student_model, layer_to_prior, layer_to_alpha, args.eps):
                    ad_out = student_model(input_ids=input_ids, use_cache=False, output_attentions=False)
                ad_logits = ad_out.logits[:, -1, :].float()
                ad_ce = F.cross_entropy(ad_logits, torch.tensor([target_id], device=args.device))
                ad_argmax = int(torch.argmax(ad_logits, dim=-1).item())
                ad_prob = float(torch.softmax(ad_logits, dim=-1)[0, target_id].detach().cpu())
                alpha_mean = float(alpha.mean().detach().cpu())

            totals["tokens"] += 1
            totals["baseline_ce"] += float(base_ce.detach().cpu())
            totals["adapter_ce"] += float(ad_ce.detach().cpu())
            totals["baseline_argmax_hits"] += int(base_argmax == target_id)
            totals["adapter_argmax_hits"] += int(ad_argmax == target_id)
            totals["adapter_changed_argmax"] += int(base_argmax != ad_argmax)
            totals["adapter_target_prob_delta"] += ad_prob - base_prob
            totals["alpha_mean"] += alpha_mean
            rec_tokens.append(
                {
                    "token_step": token_step,
                    "target_id": int(target_id),
                    "baseline_ce": float(base_ce.detach().cpu()),
                    "adapter_ce": float(ad_ce.detach().cpu()),
                    "baseline_argmax": base_argmax,
                    "adapter_argmax": ad_argmax,
                    "baseline_hit": base_argmax == target_id,
                    "adapter_hit": ad_argmax == target_id,
                    "baseline_target_prob": base_prob,
                    "adapter_target_prob": ad_prob,
                    "alpha_mean": alpha_mean,
                }
            )
            running_prefix.append(int(target_id))

        append_jsonl(
            str(out_path),
            {
                "sample_index": sample.get("source_index", seen - 1),
                "question": question,
                "gold": gold,
                "target_mode": args.target_mode or train_args.get("target_mode", "gold"),
                "tokens": rec_tokens,
            },
        )
        kept += 1

    n = max(1.0, totals["tokens"])
    summary = {
        "checkpoint": args.checkpoint,
        "samples_path": args.samples_path,
        "seen": seen,
        "kept": kept,
        "skipped": skipped,
        "tokens": int(totals["tokens"]),
        "target_mode": args.target_mode or train_args.get("target_mode", "gold"),
        "baseline_ce": totals["baseline_ce"] / n,
        "adapter_ce": totals["adapter_ce"] / n,
        "ce_delta_adapter_minus_baseline": (totals["adapter_ce"] - totals["baseline_ce"]) / n,
        "baseline_argmax_acc": totals["baseline_argmax_hits"] / n,
        "adapter_argmax_acc": totals["adapter_argmax_hits"] / n,
        "argmax_acc_delta": (totals["adapter_argmax_hits"] - totals["baseline_argmax_hits"]) / n,
        "adapter_changed_argmax_rate": totals["adapter_changed_argmax"] / n,
        "adapter_target_prob_delta": totals["adapter_target_prob_delta"] / n,
        "alpha_mean": totals["alpha_mean"] / n,
        "teacher_layer_scope": teacher_scope,
        "student_layer_scope": student_scope,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
