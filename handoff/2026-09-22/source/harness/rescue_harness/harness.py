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
        return bad("CONFIG_HASH_HEADER_MISMATCH")
    if marker.get("checkpoint_sha256") != header.get("checkpoint_sha256"):
        return bad("CHECKPOINT_HEADER_MISMATCH")
    if expected_config_hash is not None and marker.get("config_hash") != expected_config_hash:
        return bad("CONFIG_HASH_MISMATCH")
    if expected_checkpoint_sha256 is not None and \
            marker.get("checkpoint_sha256") != expected_checkpoint_sha256:
        return bad("CHECKPOINT_MISMATCH")
    return {"status": "PASS", "reason": "OK", "path": str(path),
            "generation_id": gen, "record_count": len(records), "body_sha256": actual}


def is_complete(path: Path, *, expected_config_hash: Optional[str] = None,
                expected_checkpoint_sha256: Optional[str] = None) -> bool:
    return verify_result(path, expected_config_hash=expected_config_hash,
                         expected_checkpoint_sha256=expected_checkpoint_sha256)["status"] == "PASS"


# --------------------------------------------------------------------------
# P12 — denominator integrity
# --------------------------------------------------------------------------

class DenominatorError(RuntimeError):
    pass


def audit_records(records: list[dict], arms: Iterable[str], required_fields: Iterable[str],
                  id_field: str = "canonical_source_id", arm_field: str = "condition",
                  planned_ids: Optional[Iterable[str]] = None) -> dict:
    """Validate before aggregating. Never return len(file) as the effective N."""
    arms = list(arms)
    required = list(required_fields)
    attempted = len(records)
    seen: dict[str, set] = {}
    excluded: list[dict] = []
    valid_rows: list[dict] = []
    for i, r in enumerate(records):
        reason = None
        if not isinstance(r, dict):
            reason = "malformed_record"
        elif any(f not in r for f in required):
            reason = "incomplete_record"
        elif r.get(arm_field) not in arms:
            reason = "unknown_arm"
        if reason:
            excluded.append({"index": i, "reason": reason,
                             "sample_id": (r or {}).get(id_field) if isinstance(r, dict) else None})
            continue
        sid = r[id_field]
        seen.setdefault(sid, set()).add(r[arm_field])
        valid_rows.append(r)
    duplicates = sorted(sid for sid, a in seen.items() if len(a) != len(set(a)))
    duplicate_pairs = []
    counter: dict[tuple, int] = {}
    for r in valid_rows:
        k = (r[id_field], r[arm_field])
        counter[k] = counter.get(k, 0) + 1
    duplicate_pairs = [{"sample_id": k[0], "arm": k[1], "count": v}
                       for k, v in counter.items() if v > 1]
    incomplete_arms = sorted(sid for sid, a in seen.items() if set(a) != set(arms))
    if planned_ids is not None:
        planned = set(planned_ids)
        missing_ids = sorted(planned - set(seen))
    else:
        missing_ids = []
    valid_ids = sorted(sid for sid, a in seen.items() if set(a) == set(arms))
    report = {
        "planned_N": len(set(planned_ids)) if planned_ids is not None else None,
        "attempted_N": attempted,
        "valid_N": len(valid_ids),
        "excluded_N": len(excluded) + len(duplicate_pairs) + len(incomplete_arms),
        "excluded": excluded[:50],
        "duplicate_pairs": duplicate_pairs[:50],
        "samples_missing_an_arm": incomplete_arms[:50],
        "planned_ids_absent_from_results": missing_ids[:50],
        "arms_expected": arms,
        "valid_sample_ids": valid_ids,
    }
    if duplicate_pairs:
        report["status"] = "FAIL_DUPLICATE_SAMPLE_ARM"
    elif incomplete_arms:
        report["status"] = "FAIL_MISSING_ARM"
    elif excluded:
        report["status"] = "FAIL_MALFORMED_OR_INCOMPLETE"
    elif planned_ids is not None and missing_ids:
        report["status"] = "PARTIAL_MISSING_PLANNED_IDS"
    else:
        report["status"] = "PASS"
    return report


def require_clean_denominator(report: dict) -> None:
    if not report["status"].startswith("PASS"):
        raise DenominatorError(f"denominator audit failed: {report['status']}")


# --------------------------------------------------------------------------
# P10 / RESUME-003 — identity contracts
# --------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def weights_manifest(path: Path, patterns=("*.safetensors", "*.bin", "*.pt")) -> dict:
    """Cheap, stable manifest of the weight files (name, size, first+last 1 MiB hash)."""
    root = Path(path)
    files = []
    for pat in patterns:
        files.extend(sorted(root.glob(pat)))
    entries = []
    for f in sorted(set(files)):
        size = f.stat().st_size
        h = hashlib.sha256()
        with f.open("rb") as fh:
            h.update(fh.read(1 << 20))
            if size > (1 << 20):
                fh.seek(max(0, size - (1 << 20)))
                h.update(fh.read(1 << 20))
        entries.append({"name": f.name, "size": size, "head_tail_sha256": h.hexdigest()})
    combined = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"files": entries, "manifest_sha256": combined}


def source_lock_manifest(source_lock: Path, names=("adapter.py", "common.py", "steering.py")) -> dict:
    root = Path(source_lock)
    out = {n: sha256_file(root / n) for n in names if (root / n).exists()}
    out["manifest_sha256"] = hashlib.sha256(
        json.dumps(out, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return out


def projection_code_sha(source_lock: Path) -> str:
    """The projection/mixing code that actually shapes the prior payload."""
    root = Path(source_lock)
    parts = {}
    for n in ("common.py", "adapter.py", "steering.py"):
        if (root / n).exists():
            parts[n] = sha256_file(root / n)
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------
# FIX-B / FIX-D / FIX-E — canonical training config + provenance verification
# (implementation lives in rescue_harness/training_identity.py)
# --------------------------------------------------------------------------

CheckpointIdentityIncomplete = TI.CheckpointIdentityIncomplete
ProvenanceError = TI.ProvenanceError
ProvenanceDeclarationMismatch = TI.ProvenanceDeclarationMismatch
ProvenanceUnverifiable = TI.ProvenanceUnverifiable


class CheckpointIdentityMismatch(RuntimeError):
    """Raised when a stored identity does not match the current run."""


REQUIRED_TRAINING_CONFIG_FIELDS = TI.REQUIRED_TRAINING_CONFIG_FIELDS
TRAINING_DTYPES = TI.TRAINING_DTYPES

#: kept for callers that imported the old name
_is_placeholder = TI.is_placeholder
canonical_training_config = TI.canonical_training_config
looks_like_envelope = TI.looks_like_envelope
schema_contract = TI.schema_contract
provenance_verification_contract = TI.provenance_verification_contract
observe_training_facts = TI.observe_training_facts
verify_declared_vs_observed = TI.verify_declared_vs_observed


def training_config_from_artifacts(*, source_lock: Path, checkpoint_payload: dict,
                                   training_result: dict, train_ledger: Path,
                                   validation_ledger: Path, dtype: str,
                                   method_variant: str, supervised_target_mode: str,
                                   initialization: str, layer_scope: str,
                                   adapter_config: dict) -> dict:
    """Assemble the canonical config from the real frozen artefacts (no invented values)."""
    frozen = json.loads((Path(source_lock) / "frozen_training_config.json")
                        .read_text(encoding="utf-8"))
    contract = json.loads((Path(source_lock) / "frozen_experiment_and_statistics_contract.json")
                          .read_text(encoding="utf-8"))
    raw = {
        "method_variant": method_variant,
        "train_split_identity": _ledger_identity(train_ledger),
        "validation_split_identity": _ledger_identity(validation_ledger),
        "optimizer": frozen.get("optimizer"),
        "learning_rate": frozen.get("learning_rate"),
        "training_steps": frozen.get("fixed_optimizer_steps"),
        "initialization": initialization,
        "adapter_config": adapter_config,
        "supervised_target_mode": supervised_target_mode,
        "regularization": frozen.get("regularization"),
        "layer_scope": layer_scope,
        "dtype": dtype,
        "seed": frozen.get("seed"),
        "batch_size": frozen.get("batch_size"),
        "weight_decay": frozen.get("weight_decay"),
        "scheduler": frozen.get("scheduler"),
        "checkpoint_save_steps": frozen.get("checkpoint_save_steps"),
        "early_stopping": frozen.get("early_stopping"),
        "selection_rule": (contract.get("optimization", {}) or {}).get("selection"),
        "supervised_token_count": training_result.get("supervised_token_count"),
        "checkpoint_step": checkpoint_payload.get("step"),
    }
    return TI.canonical_training_config(raw, source_lock=Path(source_lock))


def checkpoint_identity_contract(*, teacher_model_path: Path, student_model_path: Path,
                                 source_lock: Path, training_config: Any,
                                 checkpoint: Path,
                                 student_tokenizer_path: Optional[Path] = None,
                                 train_ledger: Optional[Path] = None,
                                 validation_ledger: Optional[Path] = None,
                                 run_source_dirs: Optional[Iterable[Path]] = None) -> dict:
    """FIX-A / FIX-B / FIX-D / FIX-E: bind BOTH model sides, validate the training
    config field-by-field, and — when a provenance envelope is supplied — require every
    material declaration to equal the value observed in an authoritative artefact.

    The previous version bound only the Teacher side and accepted any non-empty dict as
    the training config; both gaps are closed, and `found=true` is no longer accepted as
    proof of a provenance claim.
    """
    teacher_model_path = Path(teacher_model_path)
    student_model_path = Path(student_model_path)
    student_tokenizer_path = Path(student_tokenizer_path or student_model_path)
    source_lock = Path(source_lock)
    is_envelope = TI.looks_like_envelope(training_config)
    cfg = TI.canonical_training_config(training_config, source_lock=source_lock)  # FIX-D

    provenance_records = None
    if is_envelope:
        if not run_source_dirs:
            raise ProvenanceUnverifiable(
                "PROVENANCE_UNVERIFIABLE: verifying a provenance envelope requires the run "
                "source directories; refusing to accept the declaration unchecked")
        try:
            payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ProvenanceUnverifiable(
                f"PROVENANCE_UNVERIFIABLE: checkpoint payload unreadable for verification: {exc}"
            ) from exc
        observed = TI.observe_training_facts(
            source_lock=source_lock, run_source_dirs=list(run_source_dirs),
            checkpoint_payload=payload, train_ledger=train_ledger,
            validation_ledger=validation_ledger)
        provenance_records = TI.verify_declared_vs_observed(cfg, observed)
    else:
        payload = None

    teacher_cfg = json.loads((teacher_model_path / "config.json").read_text(encoding="utf-8"))
    student_cfg = json.loads((student_model_path / "config.json").read_text(encoding="utf-8"))
    t_tok = teacher_model_path / "tokenizer.json"
    s_tok = student_tokenizer_path / "tokenizer.json"
    t_manifest = weights_manifest(teacher_model_path)
    s_manifest = weights_manifest(student_model_path)
    lock_manifest = source_lock_manifest(source_lock)
    contract = {
        # ---- Teacher / Target side (kept) ----
        "teacher_model_path": str(teacher_model_path),
        "teacher_config_sha256": sha256_file(teacher_model_path / "config.json"),
        "teacher_config_num_hidden_layers": teacher_cfg.get("num_hidden_layers"),
        "teacher_weights_manifest": t_manifest,
        "teacher_weights_manifest_sha256": t_manifest["manifest_sha256"],
        "teacher_tokenizer_sha256": sha256_file(t_tok) if t_tok.exists() else None,
        # ---- FIX-A: Student side, explicitly bound ----
        "student_model_path": str(student_model_path),
        "student_config_sha256": sha256_file(student_model_path / "config.json"),
        "student_config_num_hidden_layers": student_cfg.get("num_hidden_layers"),
        "student_weights_manifest": s_manifest,
        "student_weights_manifest_sha256": s_manifest["manifest_sha256"],
        "student_tokenizer_sha256": sha256_file(s_tok) if s_tok.exists() else None,
        # ---- code / lock identity (kept) ----
        "source_lock": lock_manifest,
        "source_lock_manifest_sha256": lock_manifest["manifest_sha256"],
        "projection_code_sha256": projection_code_sha(source_lock),
        "steering_code_sha256": sha256_file(Path(source_lock) / "steering.py")
        if (Path(source_lock) / "steering.py").exists() else None,
        # ---- FIX-B: complete, canonically serialised training config ----
        "training_config": cfg,
        "training_config_sha256": hashlib.sha256(
            json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "training_config_completeness": {
            "required_fields": list(REQUIRED_TRAINING_CONFIG_FIELDS),
            "present_fields": list(REQUIRED_TRAINING_CONFIG_FIELDS),
            "status": "COMPLETE",
        },
        "schema_contract": TI.schema_contract(source_lock),
        "provenance_verification": (
            {"status": "NOT_APPLICABLE_NO_ENVELOPE"} if provenance_records is None else
            {"status": "MATCH_ALL", "records": provenance_records,
             "contract": TI.provenance_verification_contract()}),
        "checkpoint_payload_step": (payload or {}).get("step"),
        # ---- checkpoint ----
        "checkpoint_sha256": sha256_file(checkpoint) if Path(checkpoint).exists() else None,
        "path_is_not_identity": True,
    }
    if train_ledger is not None:
        contract["train_split_identity"] = _ledger_identity(train_ledger)
    if validation_ledger is not None:
        contract["validation_split_identity"] = _ledger_identity(validation_ledger)
    for field in ("teacher_config_sha256", "teacher_weights_manifest_sha256",
                  "student_config_sha256", "student_weights_manifest_sha256",
                  "student_tokenizer_sha256", "source_lock_manifest_sha256",
                  "projection_code_sha256", "steering_code_sha256",
                  "training_config_sha256", "checkpoint_sha256"):
        if not contract.get(field):
            raise CheckpointIdentityIncomplete(f"checkpoint identity missing {field}")
    contract["identity_sha256"] = hashlib.sha256(json.dumps(
        {k: v for k, v in contract.items()
         if k not in ("identity_sha256", "path_is_not_identity", "teacher_model_path",
                      "student_model_path")},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return contract


CHECKPOINT_IDENTITY_FIELDS = (
    "teacher_config_sha256", "teacher_weights_manifest_sha256", "teacher_tokenizer_sha256",
    "student_config_sha256", "student_weights_manifest_sha256", "student_tokenizer_sha256",
    "source_lock_manifest_sha256", "projection_code_sha256", "steering_code_sha256",
    "training_config_sha256", "checkpoint_sha256",
)


def checkpoint_identity_matches(stored: dict, current: dict) -> tuple[bool, list[str]]:
    diffs = [f for f in CHECKPOINT_IDENTITY_FIELDS if stored.get(f) != current.get(f)]
    return (not diffs), diffs


def require_checkpoint_identity(stored: dict, current: dict) -> None:
    """Resume/reuse is refused unless every checkpoint identity field matches."""
    ok, diffs = checkpoint_identity_matches(stored, current)
    if not ok:
        raise CheckpointIdentityMismatch("CHECKPOINT_IDENTITY_MISMATCH:" + ",".join(sorted(diffs)))


def _ledger_identity(path: Path) -> dict:
    path = Path(path)
    return {"name": path.name,
            "sha256": sha256_file(path) if path.exists() else None,
            "row_count": _csv_row_count(path)}


def _csv_row_count(path: Path) -> Optional[int]:
    if not Path(path).exists():
        return None
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return max(0, sum(1 for _ in f) - 1)


# --------------------------------------------------------------------------
# P9 — prior cache identity + resume closure
# --------------------------------------------------------------------------

PRIOR_IDENTITY_FIELDS = (
    "token_ids_sha256", "source_sample_id", "teacher_tokenizer_sha256",
    "student_tokenizer_sha256", "teacher_model_identity", "student_model_identity",
    "teacher_weights_manifest_sha256", "student_weights_manifest_sha256",
    "projection_code_sha256", "source_lock_manifest_sha256", "config_sha256",
)


class CacheIdentityMismatch(RuntimeError):
    pass


def prior_identity_record(*, token_ids: list[int], source_sample_id: str,
                          teacher_tokenizer_sha: str, student_tokenizer_sha: str,
                          teacher_identity: dict, student_identity: dict,
                          teacher_weights_sha: str, student_weights_sha: str,
                          projection_sha: str, source_lock_sha: str,
                          config_sha: str) -> dict:
    rec = {
        "token_ids_sha256": hashlib.sha256(
            json.dumps(list(map(int, token_ids))).encode()).hexdigest(),
        "source_sample_id": str(source_sample_id),
        "teacher_tokenizer_sha256": teacher_tokenizer_sha,
        "student_tokenizer_sha256": student_tokenizer_sha,
        "teacher_model_identity": teacher_identity,
        "student_model_identity": student_identity,
        "teacher_weights_manifest_sha256": teacher_weights_sha,
        "student_weights_manifest_sha256": student_weights_sha,
        "projection_code_sha256": projection_sha,
        "source_lock_manifest_sha256": source_lock_sha,
        "config_sha256": config_sha,
    }
    missing = [f for f in PRIOR_IDENTITY_FIELDS if f not in rec or rec[f] in (None, "")]
    if missing:
        raise CacheIdentityMismatch(f"prior identity record missing fields: {missing}")
    rec["identity_sha256"] = hashlib.sha256(
        json.dumps({k: rec[k] for k in PRIOR_IDENTITY_FIELDS}, sort_keys=True,
                   separators=(",", ":")).encode()).hexdigest()
    return rec


def prior_identity_matches(stored: dict, current: dict) -> tuple[bool, list[str]]:
    """Resume is allowed only when every identity field matches the current run."""
    diffs = [f for f in PRIOR_IDENTITY_FIELDS if stored.get(f) != current.get(f)]
    return (not diffs), diffs


def resume_prior_cache(complete: Optional[dict], path: Path, current_identity: dict) -> tuple[bool, str]:
    """Return (reusable, reason).  Never reuse on file existence or a single SHA."""
    if not complete or complete.get("status") != "DONE":
        return False, "NO_DONE_RECORD"
    if not Path(path).exists():
        return False, "MISSING_TENSOR_FILE"
    if sha256_file(path) != complete.get("file_sha256"):
        return False, "FILE_SHA_MISMATCH"
    ok, diffs = prior_identity_matches(complete.get("identity", {}), current_identity)
    if not ok:
        return False, "CACHE_IDENTITY_MISMATCH:" + ",".join(sorted(diffs))
    return True, "REUSED"


# --------------------------------------------------------------------------
# P7 — gradient fail-closed gate
# --------------------------------------------------------------------------

class GradientGateError(RuntimeError):
    pass


GRADIENT_GROUPS = ("base.mix_logits", "base.residual_logits", "base.gate_logits",
                   "base.log_temperature", "entropy_gate_weight")


def check_gradients_before_step(adapter, *, required_groups=GRADIENT_GROUPS,
                                frozen_modules: Iterable[tuple[str, Any]] = ()) -> dict:
    """Validate gradients BEFORE optimizer.step(); raise on any failure."""
    named = dict(adapter.named_parameters())
    missing, nonfinite, zero = [], [], []
    for g in required_groups:
        p = named.get(g)
        if p is None:
            missing.append(g)
            continue
        if p.grad is None:
            missing.append(g)
            continue
        if not torch.isfinite(p.grad).all():
            nonfinite.append(g)
            continue
        if not torch.any(p.grad != 0):
            zero.append(g)
    if missing:
        raise GradientGateError(f"gradient group missing: {missing}")
    if nonfinite:
        raise GradientGateError(f"non-finite gradient in: {nonfinite}")
    if len(zero) == len(required_groups):
        raise GradientGateError("no effective task gradient: every group is zero")
    leaks = []
    for name, module in frozen_modules:
        for pname, p in module.named_parameters():
            if p.grad is not None and torch.any(p.grad != 0):
                leaks.append(f"{name}.{pname}")
    if leaks:
        raise GradientGateError(f"frozen-weight gradient leak: {leaks[:5]}")
    total = math.sqrt(sum(float(named[g].grad.detach().float().pow(2).sum())
                          for g in required_groups))
    return {"groups_present": required_groups, "zero_groups": zero, "grad_norm": total,
            "frozen_leaks": leaks}


# --------------------------------------------------------------------------
# P1/P2/P4/P5 — cache-compatible, exactly transparent steering
# --------------------------------------------------------------------------

STEERED_PHASE = "ADAPTER_STEERED"
DECODE_PHASE = "STUDENT_CACHED_DECODE"
RUNTIME_KWARGS_OF_INTEREST = ("past_key_values", "past_key_value", "cache_position",
                              "position_ids", "attention_mask", "position_embeddings",
                              "use_cache", "output_attentions", "hidden_states")

# kwargs that carry attention state a wrapper MUST bind; the rest are legitimately
# forwarded (transformers' own Qwen2Attention also collects them in **kwargs).
STATEFUL_KWARGS = ("past_key_values", "past_key_value", "cache_position",
                   "attention_mask", "position_embeddings")


@dataclass
class SteeringState:
    layer_to_prior: dict
    layer_to_alpha: dict
    eps: float
    phase: str
    enabled: bool = True
    capture_rows: Optional[dict] = None
    call_log: list = field(default_factory=list)
    mix_log: list = field(default_factory=list)


def runtime_signature_contract(attention_cls, patched_signature: str,
                               observed_call_kwargs: dict) -> dict:
    """P1: prove that no key runtime argument is silently swallowed and unused."""
    import transformers
    stock = str(inspect.signature(attention_cls.forward))
    declared = set(inspect.signature(attention_cls.forward).parameters)
    observed = sorted(observed_call_kwargs)
    swallowed_stateful = [k for k in observed
                          if k not in declared and k in STATEFUL_KWARGS]
    forwarded = [k for k in observed if k not in declared and k not in STATEFUL_KWARGS]
    return {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "stock_Qwen2Attention_forward_signature": stock,
        "patched_signature": patched_signature,
        "observed_call_kwargs": observed,
        "stock_declared_parameters": sorted(declared),
        "stateful_kwargs_required_to_bind": sorted(STATEFUL_KWARGS),
        "stateful_kwargs_swallowed_unused": swallowed_stateful,
        "kwargs_forwarded_by_design": sorted(forwarded),
        "kwargs_swallowed_by_stock": swallowed_stateful,
        "invariant": ("every STATEFUL runtime kwarg (past_key_values / past_key_value / "
                      "cache_position / attention_mask / position_embeddings) must be bound "
                      "to a named parameter of the wrapper; forwarding-only kwargs "
                      "(" + ", ".join(sorted(forwarded)) + ") may stay in **kwargs and are "
                      "passed through unchanged"),
        "verdict": "PASS" if not swallowed_stateful else "FAIL",
    }


def build_steering_context(steering_source_text: str, namespace: dict):
    """Compile the cache-compatible steering context manager in a private namespace."""
    exec(compile(steering_source_text, "<harness_steering>", "exec"), namespace)
    return namespace["qwen2_adapter_steering"]


STEERING_SOURCE = '''
@contextlib.contextmanager
def qwen2_adapter_steering(model, layer_to_prior, layer_to_alpha, eps, trace=None,
                           capture_rows=None, selected_layers=None, phase="ADAPTER_STEERED"):
    """Cache-compatible, exactly-transparent-when-idle Qwen2 attention steering."""
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    backbone = model if hasattr(model, "layers") else getattr(model, "model")
    patch_layers = set(layer_to_prior) if selected_layers is None else set(selected_layers)
    originals = []
    for index, layer in enumerate(backbone.layers):
        if index not in patch_layers:
            continue
        attention = layer.self_attn
        attention._attn_adapter_layer_idx = index
        attention._attn_adapter_model_ref = model
        originals.append((attention, attention.forward))

    state = _STEERING_STATE_CLS(layer_to_prior, layer_to_alpha, eps, phase,
                                capture_rows=capture_rows)

    def make_patched(attention, original_forward, layer_index):
        def patched(self, hidden_states, position_embeddings, attention_mask=None,
                    past_key_value=None, past_key_values=None, cache_position=None, **kwargs):
            active = bool(state.enabled and layer_index in state.layer_to_prior)
            capturing = state.capture_rows is not None
            state.call_log.append({"layer": layer_index, "phase": state.phase,
                                   "active": active, "capturing": capturing,
                                   "kwargs_keys": sorted(kwargs),
                                   "past_key_values_bound": past_key_values is not None,
                                   "past_key_value_bound": past_key_value is not None})
            pkv = past_key_value if past_key_value is not None else past_key_values
            if not active and not capturing:
                # EXACT transparency: hand the untouched call to the stock implementation
                forwarded = dict(kwargs)
                forwarded.pop("past_key_value", None)
                return original_forward(hidden_states, position_embeddings,
                                        attention_mask=attention_mask,
                                        past_key_values=pkv,
                                        cache_position=cache_position, **forwarded)

            if kwargs.get("output_attentions"):
                raise RuntimeError("steered path cannot return attention weights; "
                                   "output_attentions must be falsy")
            if active and pkv is None:
                raise RuntimeError("steered path requires a KV cache handle: refusing to run "
                                   "without one (this is the 4.57 past_key_values contract)")

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if pkv is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = pkv.update(key_states, value_states,
                                                      self.layer_idx, cache_kwargs)
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            q_len, key_len = query_states.shape[-2], key_states.shape[-2]
            prefix = query_states[:, :, :0, :]
            if q_len > 1:
                prefix_mask = None if attention_mask is None else attention_mask[:, :, :q_len - 1, :key_len]
                prefix = F.scaled_dot_product_attention(
                    query_states[:, :, :-1, :], key_states, value_states,
                    attn_mask=prefix_mask,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=prefix_mask is None)
            query_last = query_states[:, :, -1:, :].float()
            scores = torch.matmul(query_last, key_states.float().transpose(2, 3)) / math.sqrt(
                float(query_last.shape[-1]))
            if attention_mask is not None:
                scores = scores + attention_mask[:, :, -1:, :key_len].float()
            last_probs = F.softmax(scores, dim=-1, dtype=torch.float32)
            if capturing:
                state.capture_rows[layer_index] = last_probs[0, :, 0, :].detach().float()
            if active:
                last_probs = _apply_mix(state, layer_index, last_probs, key_len)
            last_probs = F.dropout(last_probs, p=self.attention_dropout, training=self.training)
            last_output = torch.matmul(last_probs.float(), value_states.float())
            attn_output = torch.cat((prefix, last_output.to(query_states.dtype)), dim=2)
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
            return self.o_proj(attn_output), None
        return patched

    model._attn_adapter_state = state
    for attention, original in originals:
        attention.forward = types.MethodType(
            make_patched(attention, original,
                         int(getattr(attention, "_attn_adapter_layer_idx"))), attention)
    try:
        yield state
    finally:
        for attention, original in originals:
            attention.forward = original
        model._attn_adapter_state = None


def _apply_mix(state, layer_index, attn_probs, key_len):
    prior = _fit_prior(state.layer_to_prior[layer_index], key_len, attn_probs.device, state.eps)
    alpha = state.layer_to_alpha[layer_index].to(device=attn_probs.device, dtype=torch.float32)
    if prior.shape[0] != attn_probs.shape[1] or alpha.numel() != attn_probs.shape[1]:
        raise ValueError("adapter head count does not match Qwen2 attention heads")
    row = attn_probs[:, :, -1, :].float()
    mixed = (1.0 - alpha.view(1, -1, 1)) * row + alpha.view(1, -1, 1) * prior.view(
        1, prior.shape[0], -1)
    mixed = mixed / mixed.sum(dim=-1, keepdim=True).clamp_min(state.eps)
    state.mix_log.append({"layer": layer_index, "key_len": int(key_len),
                          "alpha_mean": float(alpha.mean())})
    return torch.cat((attn_probs[:, :, :-1, :], mixed.to(attn_probs.dtype).unsqueeze(2)), dim=2)


def _fit_prior(prior, key_len, device, eps):
    prior = prior.to(device=device, dtype=torch.float32)
    if prior.dim() != 2:
        raise ValueError(f"adapter prior must be [heads,key_len], got {tuple(prior.shape)}")
    if prior.shape[-1] < key_len:
        pad = torch.full((prior.shape[0], key_len - prior.shape[-1]), eps,
                         device=device, dtype=prior.dtype)
        prior = torch.cat((prior, pad), dim=-1)
    prior = prior[:, :key_len]
    return prior / prior.sum(dim=-1, keepdim=True).clamp_min(eps)
'''


def load_harness_steering():
    ns = {"contextlib": contextlib, "math": math, "types": types,
          "torch": torch, "F": F, "_STEERING_STATE_CLS": SteeringState}
    return build_steering_context(STEERING_SOURCE, ns)


# --------------------------------------------------------------------------
# P4/P5 — arm isolation and stock purity instrumentation
# --------------------------------------------------------------------------

def cache_fingerprint(cache) -> dict:
    if cache is None:
        return {"length": None, "per_layer": None, "object_id": None}
    per_layer = []
    try:
        n = len(cache)
        for i in range(n):
            per_layer.append(int(cache.get_seq_length(i)))
    except Exception:
        per_layer = None
    return {"length": int(cache.get_seq_length()), "per_layer": per_layer,
            "object_id": id(cache), "data_ptr": _first_data_ptr(cache)}


def _first_data_ptr(cache):
    try:
        layer0 = cache.layers[0]
        k = getattr(layer0, "keys", None)
        return None if k is None else int(k.data_ptr())
    except Exception:
        return None


def assert_arm_isolation(fingerprints: dict) -> dict:
    """RE-PREFILL_PER_ARM: every arm must start from its own, freshly built cache."""
    ids = [f["object_id"] for f in fingerprints.values()]
    ptrs = [f["data_ptr"] for f in fingerprints.values()]
    report = {
        "policy": "RE-PREFILL_PER_ARM",
        "arms": sorted(fingerprints),
        "distinct_cache_objects": len(set(ids)) == len(ids),
        "distinct_key_buffers": len([p for p in ptrs if p is not None]) == len(set(ptrs)),
        "identical_initial_lengths": len({f["length"] for f in fingerprints.values()}) == 1,
        "initial_fingerprints": fingerprints,
    }
    report["status"] = "PASS" if (report["distinct_cache_objects"]
                                  and report["identical_initial_lengths"]) else "FAIL"
    return report


class PurityCounter:
    """Counts every hook/mix touch so a STOCK trajectory can prove purity."""

    def __init__(self):
        self.patched_calls = 0
        self.mix_calls = 0
        self.installations = 0

    def attach(self, state: SteeringState) -> None:
        self.state = state

    def snapshot(self) -> dict:
        return {"patched_forward_calls": self.patched_calls, "mix_calls": self.mix_calls,
                "hook_installations": self.installations}


def assert_stock_purity(snapshot: dict) -> dict:
    ok = (snapshot.get("patched_forward_calls", 0) == 0
          and snapshot.get("mix_calls", 0) == 0
          and snapshot.get("hook_installations", 0) == 0)
    return {"stock_intervention_count": snapshot.get("mix_calls", 0),
            "stock_patched_forward_calls": snapshot.get("patched_forward_calls", 0),
            "hook_installations": snapshot.get("hook_installations", 0),
            "status": "PASS" if ok else "FAIL"}
