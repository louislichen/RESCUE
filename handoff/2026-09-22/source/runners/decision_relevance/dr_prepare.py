"""Preparation for RESCUE-TEACHER-PRIOR-DECISION-RELEVANCE-DIAGNOSTIC-001.

Writes, before any measurement row:
  out/dataset_identity.json            frozen split identity + ordered sample ids
  out/dr_experiment_config.json        the frozen diagnostic configuration (config hash)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dr_common import load_harness, sha256_file  # noqa: E402

M_FIXED = 4096


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-root", required=True)
    ap.add_argument("--source-lock", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--teacher-model", required=True)
    ap.add_argument("--student-model", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--validation-ledger", required=True)
    ap.add_argument("--validation-prior-cache", required=True)
    ap.add_argument("--manifest-sha256", required=True)
    ap.add_argument("--parent-summary", required=True)
    ap.add_argument("--planned-n", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    H = load_harness(args.harness_root)
    import csv
    with Path(args.validation_ledger).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    contract_path = Path(args.source_lock) / "frozen_experiment_and_statistics_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    ordered = [{"position": i, "source_index": r.get("source_index"),
                "canonical_source_id": r.get("canonical_source_id") or r.get("id")}
               for i, r in enumerate(rows)]
    dataset = {"ledger_path": str(args.validation_ledger),
               "ledger_sha256": sha256_file(args.validation_ledger),
               "row_count": len(rows), "dataset_path": str(args.dataset_path),
               "dataset_contract": contract.get("dataset", {}),
               "statistics_contract": {"path": str(contract_path),
                                       "sha256": sha256_file(contract_path)},
               "ordered_sample_ids": ordered, "planned_N": args.planned_n}
    parent = json.loads(Path(args.parent_summary).read_text(encoding="utf-8"))

    cfg = {
        "task_id": "RESCUE-TEACHER-PRIOR-DECISION-RELEVANCE-DIAGNOSTIC-001",
        "status_as_received": "PROPOSED_FOR_USER_EXECUTION",
        "mode": "DIAGNOSTIC_ONLY / NO TRAINING / NO METHOD CHANGE",
        "parent_task": "RESCUE-MINIMAL-FOUR-ARM-VALID-RERUN-001",
        "parent_primary_case": "CASE 3",
        "parent_config_hash": parent.get("config_hash"),
        "harness": "HARNESS_FREEZE_V3",
        "harness_manifest_sha256": args.manifest_sha256,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "teacher_model": str(args.teacher_model),
        "student_model": str(args.student_model),
        "source_lock": str(args.source_lock),
        "source_lock_sha256": {n: sha256_file(Path(args.source_lock) / n)
                               for n in ("adapter.py", "common.py", "steering.py")},
        "frozen_method_variables": [
            "Student", "Teacher/Target", "step80 checkpoint", "Teacher prior", "Null prior",
            "alpha", "selected layers/heads", "full-window intervention",
            "tokenizer projection", "dtype", "dataset", "decoding"],
        "null_control": "prior.roll(1, dims=-1) through the same frozen Adapter",
        "intervention": "full window (prefill + every cached-decode query)",
        "decision_positions": "every decision position actually visited by the BASE greedy "
                              "trajectory (prefill + each generated token), max_new_tokens=16",
        "pairing": "all of A/C/D/T evaluated on the identical prefix at the identical "
                   "position; prefix_sha256 recorded per row",
        "measured_vectors": {"A": "stock Student logits", "C": "full-window Teacher logits",
                             "D": "full-window Null logits", "T": "Target Qwen2-7B logits"},
        "metrics": ["dlogp_T (log-prob movement of the Target token)",
                    "dM (Target-token margin movement)",
                    "cos(dz, z_Target - z_Base) full vocab and fixed top-M",
                    "linearized_scale_to_boundary (LINEARIZED_DIAGNOSTIC_ONLY)",
                    "prior-space L1 / JSD vs the Student attention row"],
        "topM_fixed_in_advance": M_FIXED,
        "bootstrap": {"n": 10000, "seed": 20260830, "label": "DEVELOPMENT diagnostic"},
        "decoding": {"mode": "greedy", "max_new_tokens": args.max_new_tokens},
        "dataset": {"validation_ledger": str(args.validation_ledger),
                    "dataset_path": str(args.dataset_path),
                    "split_sha256": sha256_file(args.validation_ledger),
                    "row_count": len(rows)},
        "prior_cache": {"path": str(args.validation_prior_cache),
                        "sha256": sha256_file(args.validation_prior_cache)},
        "planned_n": args.planned_n,
        "claim_boundary": "development decision-level diagnostic; no confirmation, no final "
                          "effect size, no claim that a larger alpha works, no speedup, no "
                          "adaptive-K benefit",
    }
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    cfg["config_hash"] = hashlib.sha256(canonical.encode()).hexdigest()

    H.atomic_write_text(out / "dataset_identity.json",
                        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n")
    H.atomic_write_text(out / "dr_experiment_config.json",
                        json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"planned_N": args.planned_n, "ledger_rows": len(rows),
                      "ledger_sha256": dataset["ledger_sha256"],
                      "config_hash": cfg["config_hash"],
                      "checkpoint_sha256": cfg["checkpoint_sha256"],
                      "parent_config_hash": cfg["parent_config_hash"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
