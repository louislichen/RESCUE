"""Teacher-only, resumable, bounded-memory Qwen2 prior materialization."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_tensor(path: Path, value: torch.Tensor) -> None:
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


def record_identity(rec: dict, prior: torch.Tensor, teacher_config_sha: str, tokenizer_sha: str) -> dict:
    prompt = str(rec["prompt"])
    return {
        "canonical_source_id": str(rec["canonical_source_id"]),
        "source_index": rec.get("source_index"),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "token_ids_sha256": hashlib.sha256(json.dumps(rec["teacher_token_ids"], separators=(",", ":")).encode()).hexdigest(),
        "teacher_config_sha256": teacher_config_sha,
        "tokenizer_sha256": tokenizer_sha,
        "shape": list(prior.shape),
        "dtype": str(prior.dtype),
        "content_sha256": hashlib.sha256(prior.numpy().tobytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--student-tokenizer", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    manifest_path, output_path = Path(args.manifest), Path(args.out)
    state = {"pid": os.getpid(), "started_epoch": time.time(), "status": "STARTED", "records": {}}
    if manifest_path.exists():
        state = json.loads(manifest_path.read_text(encoding="utf-8"))
        state.update({"pid": os.getpid(), "resumed_epoch": time.time(), "status": "RESUMED"})
    atomic_json(manifest_path, state)
    try:
        from transformers import AutoTokenizer
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
        if args.source_lock not in os.sys.path:
            os.sys.path.insert(0, args.source_lock)
        from common import collect_projected_teacher_heads, selected_layer_indices

        records = json.loads(Path(args.input).read_text(encoding="utf-8"))
        records = records if isinstance(records, list) else [records]
        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, use_fast=True, local_files_only=True)
        student_tokenizer = AutoTokenizer.from_pretrained(args.student_tokenizer, use_fast=True, local_files_only=True)
        config = Path(args.teacher_model) / "config.json"
        tokenizer = Path(args.teacher_model) / "tokenizer.json"
        teacher_config_sha, tokenizer_sha = sha256(config), sha256(tokenizer)
        # Fixed balanced placement leaves headroom on every 2080 Ti; no lm_head is constructed.
        teacher = Qwen2Model.from_pretrained(
            args.teacher_model, dtype=torch.float16, local_files_only=True,
            attn_implementation="eager", device_map="balanced",
            max_memory={index: "8GiB" for index in range(torch.cuda.device_count())},
            low_cpu_mem_usage=True,
        )
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        layers = tuple(selected_layer_indices(int(teacher.config.num_hidden_layers), "last_4_layers"))
        prior_dir = output_path.parent / "prior_records"
        aggregate: dict[str, torch.Tensor] = {}
        state.update({"model_class": "Qwen2Model", "dtype": "float16", "lm_head_loaded": False,
                      "device_map": dict(getattr(teacher, "hf_device_map", {})), "layers": list(layers),
                      "memory_after_load": memory_snapshot(), "record_count_requested": len(records)})
        atomic_json(manifest_path, state)
        for rec in records:
            key = str(rec["canonical_source_id"])
            path = prior_dir / f"{key}.pt"
            complete = state["records"].get(key)
            if complete and complete.get("status") == "DONE" and path.exists() and sha256(path) == complete.get("file_sha256"):
                aggregate[key] = torch.load(path, map_location="cpu", weights_only=True)
                continue
            encoded = teacher_tokenizer(str(rec["prompt"]), add_special_tokens=False)["input_ids"]
            rec = dict(rec)
            rec["teacher_token_ids"] = encoded
            before = memory_snapshot()
            with torch.inference_mode():
                prior = collect_projected_teacher_heads(teacher, teacher_tokenizer, student_tokenizer,
                                                        str(rec["prompt"]), layers, args.device, 1e-8)
                prior = prior.detach().float().cpu().contiguous()
            if not torch.isfinite(prior).all():
                raise RuntimeError(f"non-finite prior for {key}")
            identity = record_identity(rec, prior, teacher_config_sha, tokenizer_sha)
            atomic_tensor(path, prior)
            aggregate[key] = prior
            del prior
            torch.cuda.empty_cache()
            state["records"][key] = {"status": "DONE", **identity, "file": str(path),
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
                      "error": repr(exc), "memory_failure": memory_snapshot() if torch.cuda.is_available() else {}})
        atomic_json(manifest_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
