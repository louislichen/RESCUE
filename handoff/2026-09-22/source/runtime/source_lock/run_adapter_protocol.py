#!/usr/bin/env python3
"""Fail-closed entrypoint for the frozen fresh-Adapter protocol.

This wrapper owns only execution binding and manifest emission. It does not
change the Adapter mechanism or silently choose artifacts, splits, dtypes, or
optimizers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from protocol import build_run_manifest, load_frozen_training_config, sha256_file, validate_role_ledger


SYNTHETIC_SMOKE_PROMPT = (
    "[Doc 1]\nMars is the fourth planet from the Sun.\n\n"
    "[Doc 2]\nVenus is the second planet from the Sun.\n\n"
    "Question: Which planet is fourth from the Sun?\nAnswer:"
)


def construct_exact_synthetic_inputs(tokenizer, target_length: int) -> dict[str, object]:
    """Build deterministic synthetic IDs without decode/re-encode length fitting."""
    if target_length < 1:
        raise ValueError("target_length must be positive")
    prefix_ids = tokenizer("[D] ", add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(" Q:", add_special_tokens=False)["input_ids"]
    filler_ids = tokenizer(" neutral context", add_special_tokens=False)["input_ids"]
    if not prefix_ids or not suffix_ids or not filler_ids:
        raise RuntimeError("synthetic prefix, suffix, or filler tokenization is empty")
    if len(prefix_ids) + len(suffix_ids) > target_length:
        raise ValueError("target_length is shorter than protected prefix/query suffix")
    middle_len = target_length - len(prefix_ids) - len(suffix_ids)
    middle_ids = (filler_ids * ((middle_len + len(filler_ids) - 1) // len(filler_ids)))[:middle_len]
    ids = list(prefix_ids) + list(middle_ids) + list(suffix_ids)
    vocab_size = int(getattr(tokenizer, "vocab_size", 0))
    if len(ids) != target_length or any(int(x) < 0 or int(x) >= vocab_size for x in ids):
        raise RuntimeError("constructed synthetic IDs failed exact length or vocabulary bounds")
    protected = {0, len(ids) - 1}
    return {
        "input_ids": ids,
        "attention_mask": [1] * target_length,
        "protected_indices": sorted(protected),
        "prefix_length": len(prefix_ids),
        "suffix_length": len(suffix_ids),
        "prompt": tokenizer.decode(ids, skip_special_tokens=False),
    }


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser("frozen fresh Adapter protocol")
    p.add_argument("mode", choices=("smoke", "train", "confirm"))
    p.add_argument("--config", required=True)
    p.add_argument("--config-sha256", required=True)
    p.add_argument("--ledger", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--teacher-model", required=True)
    p.add_argument("--student-model", required=True)
    p.add_argument("--dataset-revision", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--student-device", default="cuda:1")
    p.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    p.add_argument("--teacher-layer-scope", default="last_4_layers")
    p.add_argument("--student-layer-scope", default="last_4_layers")
    p.add_argument("--trace-file", default=None)
    p.add_argument("--phase", default="TRAIN_PREFILL", choices=("TRAIN_PREFILL", "VALIDATION_PREFILL", "CONFIRMATION_PREFILL", "STUDENT_CACHED_DECODE"))
    p.add_argument("--protocol-amendment-id", default="ADAPTER-PREFILL-LAST-QUERY-ONLY-V1")
    return p.parse_args()


def _ledger_rows(path: Path) -> list[dict[str, str]]:
    import csv
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    args = _args()
    config_path = Path(args.config)
    ledger_path = Path(args.ledger)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == "smoke" and not args.device.startswith("cuda"):
        raise SystemExit("smoke requires an explicit CUDA device; refusing CPU fallback")
    frozen = load_frozen_training_config(config_path, args.config_sha256)
    ledger = validate_role_ledger(_ledger_rows(ledger_path))
    ledger_sha = sha256_file(ledger_path)
    if args.mode in ("train", "confirm"):
        raise SystemExit("train/confirm require a separately authorized runtime implementation; fail-closed")
    try:
        import torch  # type: ignore
        import transformers  # type: ignore
    except Exception as exc:
        raise SystemExit(f"GPU smoke unavailable: required runtime import failed: {exc}")
    if not torch.cuda.is_available():
        raise SystemExit("GPU smoke unavailable: torch.cuda.is_available() is false")

    if args.dtype != "float16":
        raise SystemExit("GPU smoke requires the authorized float16 execution; refusing a dtype substitution")
    if not args.teacher_model or not args.student_model:
        raise SystemExit("GPU smoke requires explicit teacher and student artifacts")

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    from adapter import AttentionAdapterConfig, AttentionPriorAdapter
    from common import collect_projected_teacher_heads
    from run_hotpot_attn_transfer import selected_layer_indices
    from steering import qwen2_adapter_steering

    torch.manual_seed(20260830)
    torch.cuda.manual_seed_all(20260830)
    dtype = torch.float16
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, use_fast=True, local_files_only=True)
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model, use_fast=True, local_files_only=True)
    for tok in (teacher_tokenizer, student_tokenizer):
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
    synthetic = construct_exact_synthetic_inputs(student_tokenizer, 2048)
    synthetic_prompt = str(synthetic["prompt"])

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=dtype,
        local_files_only=True,
        device_map="balanced_low_0",
        max_memory={0: "10GiB", 1: "1GiB", 2: "10GiB", 3: "10GiB"},
        low_cpu_mem_usage=True,
    )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    teacher_layers = selected_layer_indices(int(teacher.config.num_hidden_layers), args.teacher_layer_scope)
    teacher_num_heads = int(teacher.config.num_attention_heads)
    teacher_device_map = dict(getattr(teacher, "hf_device_map", {}))
    if any(str(device).startswith("cpu") or str(device).startswith("disk") for device in teacher_device_map.values()):
        raise RuntimeError(f"Teacher CPU/disk offload is forbidden: {teacher_device_map}")
    if any(str(device).endswith(":1") or str(device) == "1" for device in teacher_device_map.values()):
        raise RuntimeError(f"Teacher occupies Student target GPU cuda:1: {teacher_device_map}")
    teacher_device = next(teacher.parameters()).device
    with torch.no_grad():
        projected = collect_projected_teacher_heads(
            teacher, teacher_tokenizer, student_tokenizer, synthetic_prompt,
            teacher_layers, str(teacher_device), 1e-8,
        ).to(args.student_device)
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    student = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(args.student_device)
    student.eval()
    for param in student.parameters():
        param.requires_grad_(False)
    student_layers = selected_layer_indices(int(student.config.num_hidden_layers), args.student_layer_scope)
    adapter = AttentionPriorAdapter(AttentionAdapterConfig(
        num_teacher_layers=len(teacher_layers),
        num_teacher_heads=teacher_num_heads,
        num_student_layers=len(student_layers),
        num_student_heads=int(student.config.num_attention_heads),
        max_alpha=0.35,
        eps=1e-8,
    )).to(args.student_device)
    adapter.train()
    trace: list[dict[str, object]] = []
    if args.trace_file:
        trace.append({"stage": "adapter_parameters", "requires_grad": [bool(p.requires_grad) for p in adapter.parameters()]})
        trace.append({
            "stage": "student_runtime",
            "attention_class": type(student.model.layers[0].self_attn).__name__,
            "attention_module": type(student.model.layers[0].self_attn).__module__,
            "attention_forward_module": getattr(type(student.model.layers[0].self_attn).forward, "__module__", None),
            "config_attn_implementation": getattr(student.config, "_attn_implementation", None),
            "model_attn_implementation": getattr(student.model.config, "_attn_implementation", None),
            "gradient_enabled_before_forward": bool(torch.is_grad_enabled()),
            "inference_mode_before_forward": bool(torch.is_inference_mode_enabled()),
        })
    student_input = {
        "input_ids": torch.tensor([synthetic["input_ids"]], dtype=torch.long, device=args.student_device),
        "attention_mask": torch.tensor([synthetic["attention_mask"]], dtype=torch.long, device=args.student_device),
    }
    with torch.no_grad():
        captured_rows: dict[int, torch.Tensor] = {}
        with qwen2_adapter_steering(
            student,
            {},
            {},
            1e-8,
            capture_rows=captured_rows,
            selected_layers=set(student_layers),
            phase=args.phase,
        ):
            student(
                **student_input,
                use_cache=False,
                output_attentions=False,
            )
        if sorted(captured_rows) != sorted(student_layers):
            raise RuntimeError("Hybrid student-row capture did not hit exactly selected layers")
        student_rows = torch.stack([captured_rows[layer] for layer in student_layers], dim=0).to(args.student_device)

    prior, alpha = adapter(projected.detach(), student_rows.detach())
    trace.append({
        "stage": "adapter_outputs",
        "prior_requires_grad": bool(prior.requires_grad),
        "prior_grad_fn": type(prior.grad_fn).__name__ if prior.grad_fn else None,
        "alpha_requires_grad": bool(alpha.requires_grad),
        "alpha_grad_fn": type(alpha.grad_fn).__name__ if alpha.grad_fn else None,
        "prior_device": str(prior.device),
    })
    null_prior = prior.roll(shifts=1, dims=-1)
    payload = {
        "shape": list(prior.shape),
        "dtype": str(prior.dtype),
        "numel": int(prior.numel()),
        "serialization": "float32-row-normalized-prior",
    }
    null_payload = {
        "shape": list(null_prior.shape),
        "dtype": str(null_prior.dtype),
        "numel": int(null_prior.numel()),
        "serialization": "float32-row-normalized-prior",
    }
    if payload != null_payload:
        raise RuntimeError("Treatment/Null payload audit failed")
    layer_to_prior = {layer: prior[i] for i, layer in enumerate(student_layers)}
    layer_to_alpha = {layer: alpha[i] for i, layer in enumerate(student_layers)}
    with torch.inference_mode(False):
        with torch.enable_grad():
            with qwen2_adapter_steering(student, layer_to_prior, layer_to_alpha, 1e-8, trace=trace, phase=args.phase):
                output = student(**student_input, use_cache=False, output_attentions=False)
    trace.append({
        "stage": "student_output",
        "logits_requires_grad": bool(output.logits.requires_grad),
        "logits_grad_fn": type(output.logits.grad_fn).__name__ if output.logits.grad_fn else None,
        "logits_device": str(output.logits.device),
    })
    target = student_tokenizer(" Mars", add_special_tokens=False)["input_ids"][0]
    loss = torch.nn.functional.cross_entropy(
        output.logits[:, -1, :].float(),
        torch.tensor([target], device=args.student_device),
    )
    trace.append({
        "stage": "loss",
        "requires_grad": bool(loss.requires_grad),
        "grad_fn": type(loss.grad_fn).__name__ if loss.grad_fn else None,
        "value": float(loss.detach().cpu()),
    })
    if args.trace_file:
        Path(args.trace_file).write_text("\n".join(json.dumps(x, sort_keys=True) for x in trace) + "\n", encoding="utf-8")
    loss.backward()
    grad_norm = float(torch.sqrt(sum((p.grad.detach().float().pow(2).sum() for p in adapter.parameters() if p.grad is not None))).cpu())
    manifest = {
        "task_id": "RESCUE-NEXT-010-5-7-PREFILL-ONLY-SEMANTIC-LOCK-AND-COMBINED-GATE-001",
        "mode": "smoke",
        "synthetic_input": True,
        "dataset_calls": 0,
        "confirmation_run": False,
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "dataset_revision": args.dataset_revision,
        "config_sha256": frozen["sha256"],
        "ledger_sha256": ledger_sha,
        "checkpoint_cadence": frozen["config"]["checkpoint_save_steps"],
        "total_steps": frozen["config"]["fixed_optimizer_steps"],
        "control": "Position-Alignment-Null",
        "dtype": args.dtype,
        "teacher_device_map": str(teacher_device_map),
        "student_device": args.student_device,
        "gpu_calls": 1,
        "path": ["tokenizer", "teacher_attention_prior", "student_adapter", "treatment_null_audit", "loss", "backward", "manifest"],
        "loss_finite": bool(torch.isfinite(loss).item()),
        "adapter_grad_norm": grad_norm,
        "treatment_null_payload_equal": True,
        "protocol_amendment_id": args.protocol_amendment_id,
        "decode_scope": "PREFILL_LAST_QUERY_ONLY_WITH_STUDENT_CACHED_DECODE",
        "phase": args.phase,
    }
    (out / "smoke_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    main()
