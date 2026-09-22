"""Patched working copy of the frozen teacher-prior worker.

Minimal fixes only (audit findings NUMERICS + RESUME-001 + PROJ-001):
  1. teacher dtype is resolved from the canonical policy instead of hardcoded float16;
     float16 is refused before any model load.
  2. the first real teacher forward is checked for finiteness and fails closed.
  3. the prior identity record now binds the student tokenizer, both weight manifests,
     the projection code and the source-lock manifest (not just the teacher config).
  4. DONE reuse goes through the identity-aware resume check, so a changed prompt,
     tokenizer, weight file or projection code yields CACHE_IDENTITY_MISMATCH instead
     of silently reusing a stale prior.

Everything else (paths, atomic writes, projection call, manifest shape) is unchanged.
The frozen file is NOT modified; this is a separate working copy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import torch

HARNESS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_ROOT))
from rescue_harness import harness as H  # noqa: E402


def sha256(path: Path) -> str:
    return H.sha256_file(path)


def atomic_json(path: Path, value: object) -> None:
    H.atomic_write_text(Path(path), json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_tensor(path: Path, value: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def memory_snapshot() -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for index in range(torch.cuda.device_count()):
        result[str(index)] = {
            "allocated": int(torch.cuda.memory_allocated(index)),
            "reserved": int(torch.cuda.memory_reserved(index)),
            "peak_allocated": int(torch.cuda.max_memory_allocated(index)),
            "peak_reserved": int(torch.cuda.max_memory_reserved(index)),
        }
    return result


def record_identity(rec: dict, prior: torch.Tensor, teacher_config_sha: str,
                    teacher_tokenizer_sha: str, student_tokenizer_sha: str,
                    teacher_weights_sha: str, student_weights_sha: str,
                    projection_sha: str, source_lock_sha: str) -> dict:
    """Extended identity: the audit's RESUME-001 / PROJ-001 closure."""
    prompt = str(rec["prompt"])
    return {
        "canonical_source_id": str(rec["canonical_source_id"]),
        "source_index": rec.get("source_index"),
        "source_sample_id": str(rec["canonical_source_id"]),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "token_ids_sha256": hashlib.sha256(
            json.dumps(rec["teacher_token_ids"], separators=(",", ":")).encode()).hexdigest(),
        "teacher_config_sha256": teacher_config_sha,
        "config_sha256": teacher_config_sha,
        "teacher_tokenizer_sha256": teacher_tokenizer_sha,
        "student_tokenizer_sha256": student_tokenizer_sha,
        "teacher_model_identity": {"path": str(rec.get("teacher_model_path", ""))},
        "student_model_identity": {"path": str(rec.get("student_tokenizer", ""))},
        "teacher_weights_manifest_sha256": teacher_weights_sha,
        "student_weights_manifest_sha256": student_weights_sha,
        "projection_code_sha256": projection_sha,
        "source_lock_manifest_sha256": source_lock_sha,
        "shape": list(prior.shape),
        "dtype": str(prior.dtype),
        "content_sha256": hashlib.sha256(prior.numpy().tobytes()).hexdigest(),
    }


def current_identity(rec: dict, teacher_config_sha: str, teacher_tokenizer_sha: str,
                     student_tokenizer_sha: str, teacher_weights_sha: str,
                     student_weights_sha: str, projection_sha: str,
                     source_lock_sha: str) -> dict:
    encoded = rec["teacher_token_ids"]
    return H.prior_identity_record(
        token_ids=encoded, source_sample_id=str(rec["canonical_source_id"]),
        teacher_tokenizer_sha=teacher_tokenizer_sha,
        student_tokenizer_sha=student_tokenizer_sha,
        teacher_identity={"path": str(rec.get("teacher_model_path", ""))},
        student_identity={"path": str(rec.get("student_tokenizer", ""))},
        teacher_weights_sha=teacher_weights_sha, student_weights_sha=student_weights_sha,
        projection_sha=projection_sha, source_lock_sha=source_lock_sha,
        config_sha=teacher_config_sha)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--student-tokenizer", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-dtype", default=H.CANONICAL_TARGET_DTYPE)
    args = parser.parse_args()
    manifest_path, output_path = Path(args.manifest), Path(args.out)
    state = {"pid": os.getpid(), "started_epoch": time.time(), "status": "STARTED", "records": {}}
    if manifest_path.exists():
        state = json.loads(manifest_path.read_text(encoding="utf-8"))
        state.update({"pid": os.getpid(), "resumed_epoch": time.time(), "status": "RESUMED"})
    atomic_json(manifest_path, state)
    try:
        # ---- NUMERICS: fail closed on the dtype policy before touching a GPU ----
        H.assert_target_dtype_allowed(args.teacher_dtype)
        dtype = H.resolve_dtype(args.teacher_dtype)

        from transformers import AutoTokenizer
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
        if args.source_lock not in sys.path:
            sys.path.insert(0, args.source_lock)
        from common import collect_projected_teacher_heads, selected_layer_indices

        records = json.loads(Path(args.input).read_text(encoding="utf-8"))
        records = records if isinstance(records, list) else [records]
        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, use_fast=True,
                                                          local_files_only=True)
        student_tokenizer = AutoTokenizer.from_pretrained(args.student_tokenizer, use_fast=True,
                                                          local_files_only=True)
        config = Path(args.teacher_model) / "config.json"
        t_tok = Path(args.teacher_model) / "tokenizer.json"
        s_tok = Path(args.student_tokenizer) / "tokenizer.json"
        teacher_config_sha = sha256(config)
        teacher_tokenizer_sha = sha256(t_tok)
        student_tokenizer_sha = sha256(s_tok) if s_tok.exists() else "MISSING_STUDENT_TOKENIZER"
        teacher_weights_sha = H.weights_manifest(args.teacher_model)["manifest_sha256"]
        student_weights_sha = H.weights_manifest(args.student_tokenizer)["manifest_sha256"]
        projection_sha = H.projection_code_sha(Path(args.source_lock))
        source_lock_sha = H.source_lock_manifest(Path(args.source_lock))["manifest_sha256"]

        teacher = Qwen2Model.from_pretrained(
            args.teacher_model, dtype=dtype, local_files_only=True,
            attn_implementation="eager", device_map="balanced",
            max_memory={index: "8GiB" for index in range(torch.cuda.device_count())},
            low_cpu_mem_usage=True,
        )
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

        # ---- NUMERICS: first real forward must be finite, else fail closed ----
        probe = teacher_tokenizer("probe", add_special_tokens=False)["input_ids"][:8]
        with torch.inference_mode():
            warm = teacher(input_ids=torch.tensor([probe], device=args.device),
                           use_cache=False)
        if not torch.isfinite(warm.last_hidden_state.float()).all():
            raise H.NumericsError(
                f"teacher hidden states non-finite on the first forward with dtype={args.teacher_dtype}")

        layers = tuple(selected_layer_indices(int(teacher.config.num_hidden_layers), "last_4_layers"))
        prior_dir = output_path.parent / "prior_records"
        aggregate: dict[str, torch.Tensor] = {}
        state.update({"model_class": "Qwen2Model", "dtype": args.teacher_dtype,
                      "lm_head_loaded": False,
                      "device_map": dict(getattr(teacher, "hf_device_map", {})), "layers": list(layers),
                      "teacher_weights_manifest_sha256": teacher_weights_sha,
                      "student_weights_manifest_sha256": student_weights_sha,
                      "projection_code_sha256": projection_sha,
                      "source_lock_manifest_sha256": source_lock_sha,
                      "teacher_tokenizer_sha256": teacher_tokenizer_sha,
                      "student_tokenizer_sha256": student_tokenizer_sha,
                      "memory_after_load": memory_snapshot(),
                      "record_count_requested": len(records)})
        atomic_json(manifest_path, state)
        for rec in records:
            key = str(rec["canonical_source_id"])
            path = prior_dir / f"{key}.pt"
            rec = dict(rec)
            rec["teacher_model_path"] = args.teacher_model
            rec["student_tokenizer"] = args.student_tokenizer
            rec["teacher_token_ids"] = teacher_tokenizer(str(rec["prompt"]),
                                                         add_special_tokens=False)["input_ids"]
            ident = current_identity(rec, teacher_config_sha, teacher_tokenizer_sha,
                                     student_tokenizer_sha, teacher_weights_sha,
                                     student_weights_sha, projection_sha, source_lock_sha)
            # ---- RESUME-001: identity-aware reuse, never file-existence only ----
            reusable, reason = H.resume_prior_cache(state["records"].get(key), path, ident)
            if reusable:
                state["records"][key]["resume_reason"] = reason
                aggregate[key] = torch.load(path, map_location="cpu", weights_only=True)
                atomic_json(manifest_path, state)
                continue
            before = memory_snapshot()
            with torch.inference_mode():
                prior = collect_projected_teacher_heads(teacher, teacher_tokenizer, student_tokenizer,
                                                        str(rec["prompt"]), layers, args.device, 1e-8)
                prior = prior.detach().float().cpu().contiguous()
            if not torch.isfinite(prior).all():
                raise RuntimeError(f"non-finite prior for {key}")
            identity = record_identity(rec, prior, teacher_config_sha, teacher_tokenizer_sha,
                                       student_tokenizer_sha, teacher_weights_sha,
                                       student_weights_sha, projection_sha, source_lock_sha)
            atomic_tensor(path, prior)
            aggregate[key] = prior
            del prior
            torch.cuda.empty_cache()
            state["records"][key] = {"status": "DONE", **identity, "identity": ident,
                                     "regenerated_because": reason, "file": str(path),
                                     "file_sha256": sha256(path), "memory_before": before,
                                     "memory_after": memory_snapshot()}
            atomic_json(manifest_path, state)
        atomic_tensor(output_path, aggregate)
        state.update({"status": "DONE", "exit_code": 0, "finished_epoch": time.time(),
                      "aggregate_sha256": sha256(output_path), "memory_final": memory_snapshot()})
        atomic_json(manifest_path, state)
        return 0
    except Exception as exc:
        state.update({"status": "FAILED", "exit_code": 1, "finished_epoch": time.time(),
                      "error": repr(exc),
                      "memory_failure": memory_snapshot() if torch.cuda.is_available() else {}})
        atomic_json(manifest_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
