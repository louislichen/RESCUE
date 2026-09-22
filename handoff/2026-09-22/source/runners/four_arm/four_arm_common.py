"""Shared machinery for RESCUE-MINIMAL-FOUR-ARM-VALID-RERUN-001.

Frozen prerequisites (never changed here):
  * harness      : HARNESS_FREEZE_V3, manifest 0e569ea8…
  * hook         : the correctness-accepted in-memory steering from
                   ``rescue_harness.harness.STEERING_SOURCE`` (installed over the frozen
                   ``source_lock/steering.py`` context manager, exactly like gpu_smokes)
  * checkpoint   : step80, sha256 6532daad…
  * scorer       : the frozen normalisation / EM / SQuAD-style token F1 used by the
                   earlier window-extension batch (copied verbatim for comparability)
  * decoding     : greedy, max_new_tokens 16, identical for every arm

Four arms, exactly:
  A  BASE                  Student, no Adapter steering at all
  B  PREFILL_ONLY_TEACHER  step80 + Teacher prior, prefill (last query row) only
  C  FULL_WINDOW_TEACHER   step80 + Teacher prior, prefill + every cached-decode query
  D  FULL_WINDOW_NULL      same as C with the frozen Position-Alignment-Null prior
                           (prior.roll(1, dims=-1) through the same Adapter)
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import sys
from pathlib import Path

import torch

ARMS = ("A_base", "B_prefill_only_teacher", "C_full_window_teacher", "D_full_window_null")

#: arm -> (which prior, which window)
ARM_POLICY = {
    "A_base": {"prior": None, "window": "none"},
    "B_prefill_only_teacher": {"prior": "teacher", "window": "prefill"},
    "C_full_window_teacher": {"prior": "teacher", "window": "full"},
    "D_full_window_null": {"prior": "null", "window": "full"},
}

PREFILL_PHASE = "ADAPTER_STEERED_PREFILL"
DECODE_PHASE = "ADAPTER_STEERED_CACHED_DECODE"

#: fields every QA record must carry (frozen integrity contract, §16)
QA_REQUIRED_FIELDS = ("canonical_source_id", "condition", "arm", "tokens", "text", "em",
                      "f1", "n_tokens", "run_config_hash", "checkpoint_sha256", "shard")
K4_REQUIRED_FIELDS = ("canonical_source_id", "condition", "arm", "proposed_draft_tokens",
                      "accepted_draft_tokens", "blocks", "full_accept_blocks",
                      "reject_blocks", "tokens", "target_only_tokens",
                      "matches_target_only_greedy", "run_config_hash", "checkpoint_sha256",
                      "shard")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_harness(harness_root):
    sys.path.insert(0, str(Path(harness_root).resolve()))
    from rescue_harness import harness as H
    return H


def load_modules(H, source_dir, source_lock):
    """Import the frozen run source and install the accepted harness hook in memory."""
    sys.path.insert(0, str(Path(source_dir).resolve()))
    sys.path.insert(0, str(Path(source_lock).resolve()))
    backend_mod = importlib.import_module("qwen_adapter_backend")
    runner_mod = importlib.import_module("sequence_loss_runner")
    steering = importlib.import_module("steering")
    fixed = H.load_harness_steering()
    original = steering.qwen2_adapter_steering
    steering.qwen2_adapter_steering = fixed
    steering._frozen_original = original
    return backend_mod, runner_mod, steering


def build_backend(backend_mod, args, steering):
    backend = backend_mod.QwenAdapterBackend(
        args.teacher_model, args.student_model, args.device,
        source_lock=args.source_lock, load_teacher=False)
    backend._steering = steering.qwen2_adapter_steering
    payload = torch.load(args.checkpoint, map_location=backend.device, weights_only=False)
    backend.adapter.load_state_dict(payload["adapter"])
    return backend


def maps_for(backend, prior, alpha):
    lm = {} if prior is None else {l: prior[i] for i, l in enumerate(backend.student_layers)}
    la = {} if alpha is None else {l: alpha[i] for i, l in enumerate(backend.student_layers)}
    return lm, la


def cm_for(steering_cm, backend, maps, alpha, layers, phase):
    """A fresh context manager (the harness CM is single-use) or a no-op."""
    if not maps:
        return contextlib.nullcontext()
    return steering_cm(backend.student, maps, alpha, backend.eps,
                       selected_layers=layers, phase=phase)


# --------------------------------------------------------------------------- scorer
# frozen scorer, copied verbatim from window_qa.py so the numbers stay comparable
def norm(t):
    return " ".join(str(t).lower().strip().split())


def f1_score(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum(min(p.count(x), g.count(x)) for x in set(p))
    if not common:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def em_score(pred, gold):
    return int(norm(pred) == norm(gold))


# --------------------------------------------------------------------------- data
def dataset_identity(ledger: Path, dataset_path: Path, contract: dict) -> dict:
    """Dataset + split identity and the ORDERED sample id list (§4)."""
    import csv
    with Path(ledger).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    ds_contract = (contract or {}).get("dataset", {})
    return {
        "ledger_path": str(ledger),
        "ledger_sha256": sha256_file(ledger),
        "ledger_rows": len(rows),
        "ledger_columns": sorted(rows[0].keys()) if rows else [],
        "dataset_path": str(dataset_path),
        "dataset_contract": ds_contract,
        "ordered_sample_ids": [
            {"position": i, "source_index": r.get("source_index"),
             "canonical_source_id": r.get("canonical_source_id") or r.get("id")}
            for i, r in enumerate(rows)],
        "row_count": len(rows),
    }


def experiment_config(args, manifest_sha256: str) -> dict:
    """The frozen experiment configuration for this rerun; its sha256 is the config hash."""
    cfg = {
        "task_id": "RESCUE-MINIMAL-FOUR-ARM-VALID-RERUN-001",
        "harness": "HARNESS_FREEZE_V3",
        "harness_manifest_sha256": manifest_sha256,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "teacher_model": str(args.teacher_model),
        "student_model": str(args.student_model),
        "source_lock": str(args.source_lock),
        "source_lock_sha256": {
            n: sha256_file(Path(args.source_lock) / n)
            for n in ("adapter.py", "common.py", "steering.py")},
        "steering_hook": "rescue_harness.harness.STEERING_SOURCE (in-memory install)",
        "arms": {a: ARM_POLICY[a] for a in ARMS},
        "null_control": "prior.roll(1, dims=-1) through the same frozen Adapter",
        "layers": "backend.student_layers (last 4)",
        "dtype": "bfloat16 (target) / student default",
        "decoding": {"mode": "greedy", "max_new_tokens": args.max_new_tokens},
        "k4": {"k": 4, "verify_mode": "incremental", "eos_id": 151643},
        "scorer": {"em": "normalised exact match", "f1": "SQuAD token F1",
                   "normalisation": "lowercase + whitespace collapse"},
        "dataset": {"validation_ledger": str(args.validation_ledger),
                    "dataset_path": str(args.dataset_path),
                    "split_sha256": sha256_file(args.validation_ledger)},
        "prior_cache": {"path": str(args.validation_prior_cache),
                        "sha256": sha256_file(args.validation_prior_cache)},
        "planned_n": args.planned_n,
        "claim_boundary": "development re-measurement; no confirmation, no speedup claim",
    }
    cfg["config_hash"] = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return cfg


def load_config(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
