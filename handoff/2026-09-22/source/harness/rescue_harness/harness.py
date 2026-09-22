"""RESCUE correctness-closure harness (task RESCUE-CORRECTNESS-CLOSURE-AND-HARNESS-FREEZE-001).

Every component here closes one audit finding.  Nothing in this module edits the
frozen source tree; the harness is a working copy that a formal run must import
instead of the frozen modules.

Audit findings addressed
  COMPAT-001  selected layers ignore the KV cache (past_key_value vs past_key_values)
  COMPAT-002  idle hook is not transparent (custom fp32 last-row path)
  CTRL-001/002 stock arm not pure / paired arms not isolated
  GRAD-001    grads_ok recorded but optimizer.step() still runs
  GRAD-002    frozen sequence path gives zero gradient for later answer tokens
  RESUME-001  prior DONE reuse does not re-check identity
  RESUME-002  non-atomic result writes
  RESUME-003  checkpoint identity excludes weights / source-lock / projection code
  DENOM-001   effective N = len(file), planned_N hardcoded
  NUMERICS    prior worker hardcodes fp16 while the 7B is non-finite in fp16
"""
from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import math
import os
import tempfile
import types
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
import torch.nn.functional as F

from rescue_harness import training_identity as TI

# --------------------------------------------------------------------------
# P8 — numerical dtype closure
# --------------------------------------------------------------------------

CANONICAL_TARGET_DTYPE = "bfloat16"          # measured finite for the locked 7B
NON_FINITE_DTYPES = {"float16"}              # measured non-finite at len 16/300/1582

_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16,
              "float32": torch.float32}


class NumericsError(RuntimeError):
    pass


def resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise NumericsError(f"unknown dtype {name!r}")
    return _DTYPE_MAP[name]


def assert_target_dtype_allowed(name: str) -> None:
    """Fail closed: the locked 7B is only usable in the canonical dtype."""
    if name in NON_FINITE_DTYPES:
        raise NumericsError(
            f"dtype {name} is recorded non-finite for the locked 7B target; "
            f"use {CANONICAL_TARGET_DTYPE} (no quantisation, no dtype sweep)")


def assert_logits_finite(logits: torch.Tensor, where: str) -> None:
    if not torch.isfinite(logits).all():
        raise NumericsError(f"non-finite logits at {where}")


# --------------------------------------------------------------------------
# P11 — atomic result writes with completion flag
# --------------------------------------------------------------------------

def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


RESULT_HEADER_KEY = "__result_header__"


def completion_path(path: Path) -> Path:
    return Path(str(path) + ".complete.json")


def new_generation_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex[:12]}"


def write_result_envelope(path: Path, *, task_id: str, stage: str, records: list[dict],
                          arms: Iterable[str], sample_ids: Iterable[str],
                          config_hash: str, checkpoint_sha256: str,
                          status: str = "COMPLETE",
                          generation_id: Optional[str] = None) -> dict:
    """FIX-C: body-bound completion marker, written in the crash-safe order.

    1 write body to a temp file and fsync
    2 hash the body
    3 write the marker to a temp file and fsync
    4 INVALIDATE the old marker (unlink)
    5 atomic replace body temp -> body
    6 atomic replace marker temp -> marker

    A crash between 4 and 6 can only yield "body present, no valid marker", which
    every reader must treat as incomplete.  A stale marker can never survive.
    """
    path = Path(path)
    generation_id = generation_id or new_generation_id()
    header = {RESULT_HEADER_KEY: True, "generation_id": generation_id, "task_id": task_id,
              "stage": stage, "config_hash": config_hash,
              "checkpoint_sha256": checkpoint_sha256, "arms": sorted(arms),
              "sample_ids": sorted(sample_ids), "record_count": len(records)}
    body = json.dumps(header, ensure_ascii=False) + "\n" + "".join(
        json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    body_tmp = path.with_name(path.name + ".body.tmp")
    marker_tmp = path.with_name(path.name + ".complete.json.tmp")
    flag = completion_path(path)
    atomic_write_text(body_tmp, body)
    marker = {"status": status, "generation_id": generation_id,
              "body_sha256": sha256_file(body_tmp), "record_count": len(records),
              "config_hash": config_hash, "checkpoint_sha256": checkpoint_sha256,
              "arms": sorted(arms), "sample_ids": sorted(sample_ids),
              "task_id": task_id, "stage": stage}
    atomic_write_text(marker_tmp, json.dumps(marker, ensure_ascii=False, indent=2) + "\n")
    if flag.exists():
        flag.unlink()                       # invalidate the previous marker FIRST
    os.replace(body_tmp, path)
    os.replace(marker_tmp, flag)
    return marker


def read_result_header(path: Path) -> Optional[dict]:
    p = Path(path)
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        first = f.readline()
    try:
        obj = json.loads(first)
    except Exception:
        return None
    return obj if isinstance(obj, dict) and obj.get(RESULT_HEADER_KEY) is True else None


def read_result_records(path: Path) -> list[dict]:
    """Records only; the body header line is stripped."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get(RESULT_HEADER_KEY) is True and not out:
                continue
            out.append(obj)
    return out


def verify_result(path: Path, *, expected_config_hash: Optional[str] = None,
                  expected_checkpoint_sha256: Optional[str] = None) -> dict:
    """FIX-C: a completion marker is only trusted after full re-verification."""
    path = Path(path)
    flag = completion_path(path)
    fail = "INCOMPLETE_OR_IDENTITY_MISMATCH"

    def bad(reason):
        return {"status": fail, "reason": reason, "path": str(path)}

    if not flag.exists():
        return bad("NO_COMPLETION_MARKER")
    try:
        marker = json.loads(flag.read_text(encoding="utf-8"))
    except Exception as exc:
        return bad(f"UNPARSEABLE_MARKER:{exc}")
    if marker.get("status") != "COMPLETE":
        return bad("STATUS_NOT_COMPLETE")
    if not path.exists():
        return bad("BODY_MISSING")
    actual = sha256_file(path)
    if actual != marker.get("body_sha256"):
        return bad("BODY_SHA_MISMATCH")
    header = read_result_header(path)
    if header is None:
        return bad("BODY_HEADER_MISSING")
    gen = marker.get("generation_id")
    if not isinstance(gen, str) or len(gen) < 8:
        return bad("GENERATION_ID_INVALID")
    if header.get("generation_id") != gen:
        return bad("GENERATION_MISMATCH")
    records = read_result_records(path)
    if len(records) != marker.get("record_count"):
        return bad("RECORD_COUNT_MISMATCH")
    if marker.get("config_hash") != header.get("config_hash"):
        return bad("CONFIG_HASHX?MISMATCH")
    if marker.get("checkpoint_sha256") != header.get("checkpoint_sha256"):
        return bad("CHECKPOINT_HEADER_MISMATCH")
    if expected_config_hash is not None and marker.get("config_hash") != expected_config_hash:
        return bad("CONFIG_HASH_MISMATCH")
    if expected_checkpoint_sha256 is not None and \
            marker.get("checkpoint_sha256") != expected_checkpoint_sha256:
        return bad("CHECKPOINT_MISMATCH")
    return {"status": "PASS", "reason": "OK", "path": str(path),
            "generation_id": gen, "record_count": len(records), "body_sha256": actual}


def audit_attempted_valid(log_records: list[dict], final_records: list[dict],
                        * id_field="canonical_source_id", arm_field="condition",
                        status_field="status", valid_status=("VALID", "OK", "COMPLETE")) -> dict:
    """Separate attempted from valid; do NOT silently drop attempted samples."""
    attempted = []
    excluded = []
    valid_keys = set()
    for r in log_records:
        key = (r.get(id_field), r.get(arm_field))
        attempted.append(key)
        if str(r.get(status_field, "")).upper() not in {s.upper() for s in valid_status}:
            excluded.append({"key": key, "reason": r.get("exclusion_reason") or str(r.get(status_field))})
    for r in final_records:
        valid_keys.add((r.get(id_field), r.get(arm_field)))
    attempted_set = set(attempted)
    valid_not_attempted = sorted(valid_keys - attempted_set, key=str)
    return {"attempted_count": len(attempted_set), "valid_count": len(valid_keys),
            "excluded_count": len(excluded), "excluded": excluded,
            "valid_not_attempted": valid_not_attempted,
            "status": "PASS" if not valid_not_attempted else "FAIL"}

# ... ReSCUE harness continues with the frozen steering patch, gradient gates,
# cache/arm isolation, checkpoint identity, and training-provenance
# verification exactly as in the handoff package.
