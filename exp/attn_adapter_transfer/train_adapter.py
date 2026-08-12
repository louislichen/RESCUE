#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train a small adapter that maps teacher attention to student-compatible priors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from adapter import AttentionAdapterConfig, AttentionPriorAdapter, adapter_regularization, checkpoint_payload
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
    p = argparse.ArgumentParser("Train attention-transfer adapter")
    p.add_argument("--samples_path", default="exp/attn_transfer/data/hotpot_stream_300_maxL2048.jsonl")
    p.add_argument("--teacher_model_path", default="/mnt/8T/cl/models/Qwen2-7B-Instruct")
    p.add_argument("--student_model_path", default="/mnt/8T/cl/models/Qwen2-1.5B-Instruct")
    p.add_argument("--output_dir", default="exp/attn_adapter_transfer/out_train")
    p.add_argument("--checkpoint_name", default="adapter.pt")
    p.add_argument("--log_name", default="train_log.jsonl")
    p.add_argument("--max_train_samples", type=int, default=8)
    p.add_argument("--skip_samples", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--answer_tokens_per_sample", type=int, default=1)
    p.add_argument("--target_mode", default="gold", choices=["gold", "teacher_greedy"])
    p.add_argument("--teacher_target_tokens", type=int, default=16)
    p.add_argument("--max_prompt_tokens", type=int, default=1536)
    p.add_argument("--teacher_layer_scope", default="last_4_layers")
    p.add_argument("--student_layer_scope", default="last_4_layers")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_alpha", type=float, default=0.35)
    p.add_argument("--init_gate_logit", type=float, default=-2.0)
    p.add_argument("--init_residual_logit", type=float, default=-1.5)
    p.add_argument("--beta_kl", type=float, default=0.02)
    p.add_argument("--gamma_alpha", type=float, default=0.02)
    p.add_argument("--delta_sharpness", type=float, default=0.01)
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--doc_mode", default="all", choices=["all", "gold_only"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every_steps", type=int, default=10)
    p.add_argument("--overwrite", type=str2bool, default=True)
    return p.parse_args()


def target_tail_ids(args, teacher_model, teacher_tokenizer, student_tokenizer, prompt: str, gold: str) -> List[int]:
    if args.target_mode == "gold":
        return gold_target_ids(student_tokenizer, prompt, gold)
    teacher_answer = greedy_text(
        teacher_model,
        teacher_tokenizer,
        prompt,
        args.teacher_target_tokens,
        args.device,
        stop_on_question=True,
    )
    return student_tokenizer(teacher_answer, add_special_tokens=False)["input_ids"]


def save_checkpoint(path: str, adapter: AttentionPriorAdapter, args, step: int, losses: List[float]) -> None:
    extra: Dict[str, Any] = {
        "step": step,
        "loss_mean": sum(losses) / max(1, len(losses)),
        "args": vars(args),
    }
    torch.save(checkpoint_payload(adapter, extra), path)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / args.checkpoint_name
    log_path = out_dir / args.log_name
    if args.overwrite and log_path.exists():
        log_path.unlink()
    if args.overwrite and ckpt_path.exists():
        ckpt_path.unlink()

    dtype = str2dtype(args.dtype)
    print(f">>> loading teacher: {args.teacher_model_path}")
    teacher_tokenizer, teacher_model = load_model_and_tokenizer(args.teacher_model_path, dtype, args.device)
    print(f">>> loading student: {args.student_model_path}")
    student_tokenizer, student_model = load_model_and_tokenizer(args.student_model_path, dtype, args.device)

    for param in teacher_model.parameters():
        param.requires_grad_(False)
    for param in student_model.parameters():
        param.requires_grad_(False)
    teacher_model.eval()
    student_model.eval()

    teacher_layer_indices = selected_layer_indices(int(teacher_model.config.num_hidden_layers), args.teacher_layer_scope)
    student_layer_indices = selected_layer_indices(int(student_model.config.num_hidden_layers), args.student_layer_scope)
    cfg = AttentionAdapterConfig(
        num_teacher_layers=len(teacher_layer_indices),
        num_teacher_heads=int(teacher_model.config.num_attention_heads),
        num_student_layers=len(student_layer_indices),
        num_student_heads=int(student_model.config.num_attention_heads),
        max_alpha=args.max_alpha,
        eps=args.eps,
    )
    adapter = AttentionPriorAdapter(cfg).to(args.device)
    adapter.gate_logits.data.fill_(args.init_gate_logit)
    adapter.residual_logits.data.fill_(args.init_residual_logit)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    samples = list(read_jsonl(args.samples_path))
    train_samples = samples[args.skip_samples : args.skip_samples + args.max_train_samples]
    losses: List[float] = []
    step = 0
    skipped = {"empty": 0, "too_long": 0, "no_target": 0}

    for epoch in range(args.epochs):
        pbar = tqdm(train_samples, desc=f"train epoch {epoch + 1}", dynamic_ncols=True)
        for sample_idx, sample in enumerate(pbar):
            prompt, gold, question = sample_to_prompt(sample, args.doc_mode)
            if not prompt or not gold:
                skipped["empty"] += 1
                continue
            prompt_ids = student_tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if len(prompt_ids) > args.max_prompt_tokens:
                skipped["too_long"] += 1
                continue
            tail = target_tail_ids(args, teacher_model, teacher_tokenizer, student_tokenizer, prompt, gold)
            if not tail:
                skipped["no_target"] += 1
                continue

            use_tokens = min(args.answer_tokens_per_sample, len(tail))
            running_prefix = list(prompt_ids)
            for tok_step in range(use_tokens):
                input_ids = tensor_ids(running_prefix, args.device)
                target_id = int(tail[tok_step])
                if tok_step == 0:
                    current_text = prompt
                else:
                    current_text = student_tokenizer.decode(running_prefix, skip_special_tokens=False)

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

                adapter.train()
                prior, alpha = adapter(teacher_projected, student_rows)
                layer_to_prior, layer_to_alpha = adapter_outputs_to_layer_maps(prior, alpha, student_layer_indices)

                with qwen2_adapter_steering(student_model, layer_to_prior, layer_to_alpha, args.eps):
                    out = student_model(input_ids=input_ids, use_cache=False, output_attentions=False)
                ce = F.cross_entropy(out.logits[:, -1, :].float(), torch.tensor([target_id], device=args.device))
                reg, reg_stats = adapter_regularization(
                    prior,
                    alpha,
                    teacher_projected,
                    args.beta_kl,
                    args.gamma_alpha,
                    args.delta_sharpness,
                    args.eps,
                )
                loss = ce + reg
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                step += 1
                losses.append(float(loss.detach().cpu()))
                rec = {
                    "step": step,
                    "epoch": epoch,
                    "sample_index": sample.get("source_index", sample_idx + args.skip_samples),
                    "question": question,
                    "token_step": tok_step,
                    "target_id": target_id,
                    "target_mode": args.target_mode,
                    "loss": float(loss.detach().cpu()),
                    "ce": float(ce.detach().cpu()),
                    **reg_stats,
                }
                append_jsonl(str(log_path), rec)
                pbar.set_postfix(loss=f"{rec['loss']:.4f}", ce=f"{rec['ce']:.4f}", alpha=f"{rec['reg_alpha_mean']:.3f}")
                running_prefix.append(target_id)
                if args.save_every_steps > 0 and step % args.save_every_steps == 0:
                    save_checkpoint(str(ckpt_path), adapter, args, step, losses)

    save_checkpoint(str(ckpt_path), adapter, args, step, losses)
    summary = {
        "checkpoint": str(ckpt_path),
        "log": str(log_path),
        "steps": step,
        "loss_mean": sum(losses) / max(1, len(losses)),
        "skipped": skipped,
        "teacher_layer_indices": teacher_layer_indices,
        "student_layer_indices": student_layer_indices,
        "adapter_config": cfg.to_json(),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
