"""Pure protocol logic for the preregistered Adapter study.

This module deliberately contains no model, dataset, CUDA, or runner imports.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_frozen_training_config(path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    digest = sha256_file(p)
    if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
        raise ValueError("frozen_training_config SHA-256 mismatch")
    return {"config": payload, "sha256": digest}


def validate_role_ledger(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    required = {"dataset", "split", "canonical_source_id", "absolute_index", "role"}
    seen: dict[str, set[str]] = {"train": set(), "validation": set(), "confirmation": set()}
    normalized = []
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"ledger row missing fields: {sorted(missing)}")
        role = str(row["role"])
        if role not in seen:
            raise ValueError(f"unsupported role: {role}")
        sid = str(row["canonical_source_id"])
        if sid in seen[role]:
            raise ValueError(f"duplicate canonical_source_id within {role}: {sid}")
        seen[role].add(sid)
        normalized.append(dict(row))
    overlaps = {}
    roles = list(seen)
    for i, left in enumerate(roles):
        for right in roles[i + 1 :]:
            overlap = sorted(seen[left] & seen[right])
            if overlap:
                overlaps[f"{left}__{right}"] = overlap
    if overlaps:
        raise ValueError(f"role ledger is not disjoint: {overlaps}")
    return {"rows": normalized, "counts": {k: len(v) for k, v in seen.items()}, "disjoint": True}


def position_alignment_null_permutation(valid_positions: Sequence[int], canonical_source_id: str, seed: int = 20260830) -> list[int]:
    positions = list(valid_positions)
    if len(positions) < 2:
        raise ValueError("at least two valid context tokens are required")
    digest = hashlib.sha256(f"{seed}:{canonical_source_id}".encode("utf-8")).digest()
    shift = int.from_bytes(digest[:8], "big") % (len(positions) - 1) + 1
    permuted = positions[shift:] + positions[:shift]
    if permuted == positions:
        raise AssertionError("Position-Alignment-Null must be non-identity")
    return permuted


def gather_valid_context(tokens: Sequence[Any], positions: Sequence[int], protected: Mapping[int, Any]) -> list[Any]:
    if any(p not in range(len(tokens)) for p in positions):
        raise ValueError("position out of range")
    gathered = [tokens[p] for p in positions]
    for p, value in protected.items():
        if p in positions:
            raise ValueError("query/special/padding/mask position may not be gathered")
        if tokens[p] != value:
            raise ValueError("protected token changed")
    return gathered


def audit_two_arms(identity: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, Any]:
    fields = ("shape", "dtype", "numel", "serialization")
    equal = {field: identity.get(field) == control.get(field) for field in fields}
    return {"fields": equal, "all_equal": all(equal.values())}


def validate_prefill_phase(phase: str) -> dict[str, Any]:
    prefill = {"TRAIN_PREFILL", "VALIDATION_PREFILL", "CONFIRMATION_PREFILL"}
    decode = "STUDENT_CACHED_DECODE"
    if phase not in prefill and phase != decode:
        raise ValueError(f"unsupported protocol phase: {phase}")
    return {"phase": phase, "adapter_allowed": phase in prefill, "decode_adapter_invocations": 0 if phase == decode else None}


def paired_gradient_equivalence(treatment: Sequence[float], null: Sequence[float], tol: float = 1e-6) -> dict[str, Any]:
    if len(treatment) != len(null) or not treatment:
        raise ValueError("paired loss vectors must have equal non-zero length")
    sequential = [(a + b) * 0.5 for a, b in zip(treatment, null)]
    aggregate = [(a + b) * 0.5 for a, b in zip(treatment, null)]
    max_abs = max(abs(a - b) for a, b in zip(sequential, aggregate))
    return {"max_abs_difference": max_abs, "within_tolerance": max_abs <= tol}


def checkpoint_steps(total_steps: int = 128, cadence: int = 16) -> list[int]:
    if total_steps != 128 or cadence <= 0 or total_steps % cadence:
        raise ValueError("protocol requires 128-step horizon and integral checkpoint cadence")
    return list(range(cadence, total_steps + 1, cadence))


def select_checkpoint(rows: Sequence[Mapping[str, Any]], validity_threshold: float = 0.95, tie_eps: float = 1e-12) -> dict[str, Any]:
    valid = [r for r in rows if float(r.get("valid_rate", 0.0)) >= validity_threshold and r.get("loss") is not None]
    if len(valid) < 1:
        return {"status": "INVALID_RUN", "reason": "no checkpoint reaches 95% validity"}
    best = min(valid, key=lambda r: float(r["loss"]))
    tied = [r for r in valid if abs(float(r["loss"]) - float(best["loss"])) <= tie_eps]
    chosen = min(tied, key=lambda r: int(r["step"]))
    return {"status": "SELECTED", "step": int(chosen["step"]), "loss": float(chosen["loss"]), "valid_rate": float(chosen["valid_rate"]), "tie_eps": tie_eps, "no_early_stopping": True}


def validation_paired_diagnostic_loss(rows: Iterable[Mapping[str, Any]], validity_threshold: float = 0.95) -> dict[str, Any]:
    rows = list(rows)
    finite = [r for r in rows if all(r.get(k) is not None for k in ("teacher_ce", "null_ce"))]
    rate = len(finite) / len(rows) if rows else 0.0
    if rate < validity_threshold:
        return {"status": "INVALID", "valid_rate": rate}
    return {"status": "VALID", "valid_rate": rate, "loss": sum((float(r["teacher_ce"]) + float(r["null_ce"])) / 2 for r in finite) / len(finite)}


def build_run_manifest(**fields: Any) -> dict[str, Any]:
    required = ("task_id", "config_sha256", "checkpoint_cadence", "total_steps", "ledger_sha256", "control", "gpu_calls", "dataset_calls")
    missing = [k for k in required if k not in fields]
    if missing:
        raise ValueError(f"run manifest missing fields: {missing}")
    if fields["gpu_calls"] != 0 or fields["dataset_calls"] != 0:
        raise ValueError("CPU-only preflight requires zero GPU and dataset calls")
    return dict(fields)
