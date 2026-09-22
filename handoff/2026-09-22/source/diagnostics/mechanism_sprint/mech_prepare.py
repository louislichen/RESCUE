"""Preparation for RESCUE-GUIDANCE-REPRESENTATION-MECHANISM-SPRINT-001.

Writes out/mech_experiment_config.json (the frozen configuration + config hash used by every
result envelope of this sprint) and re-states the dataset identity.  Reuses the parent
diagnostic's frozen objects unchanged; introduces no new method variable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dr_common import load_harness, sha256_file  # noqa: E402


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
    ap.add_argument("--manifest-sha256", default="",
                    help="expected manifest sha; verified against the real file, never trusted")
    ap.add_argument("--manifest-path", default="")
    ap.add_argument("--dr-config", required=True)
    ap.add_argument("--dr-summary", required=True)
    ap.add_argument("--planned-positions", type=int, default=496)
    ap.add_argument("--sne-t-positions", type=int, default=67)
    ap.add_argument("--oracle-k", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    H = load_harness(args.harness_root)
    manifest = Path(args.manifest_path) if args.manifest_path else \
        Path(args.harness_root) / "HARNESS_FREEZE_CANDIDATE_MANIFEST_V3.json"
    manifest_sha = sha256_file(manifest)
    if args.manifest_sha256 and args.manifest_sha256 != manifest_sha:
        raise SystemExit(f"MANIFEST_SHA_MISMATCH: cli={args.manifest_sha256} file={manifest_sha}")
    dr = json.loads(Path(args.dr_config).read_text(encoding="utf-8"))
    dr_sum = json.loads(Path(args.dr_summary).read_text(encoding="utf-8"))
    cfg = {
        "task_id": "RESCUE-GUIDANCE-REPRESENTATION-MECHANISM-SPRINT-001",
        "mode": "MECHANISM DIAGNOSTIC / NO NEW ADAPTER TRAINING / NO HYPERPARAMETER SWEEP",
        "parent_task": "RESCUE-TEACHER-PRIOR-DECISION-RELEVANCE-DIAGNOSTIC-001",
        "parent_result": "CASE INFO-B",
        "harness": "HARNESS_FREEZE_V3",
        "harness_manifest_sha256": manifest_sha,
        "harness_manifest_path": str(manifest),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "teacher_model": str(args.teacher_model),
        "student_model": str(args.student_model),
        "source_lock": str(args.source_lock),
        "source_lock_sha256": {n: sha256_file(Path(args.source_lock) / n)
                               for n in ("adapter.py", "common.py", "steering.py")},
        "parent_config_hash": dr.get("config_hash"),
        "reused_evidence": {
            "decision_positions": str(args.dr_summary),
            "planned_positions": args.planned_positions,
            "S_ne_T_positions": args.sne_t_positions,
            "note": "positions, prefixes, y_base and y_target are reused from the parent "
                    "diagnostic; no experiment is re-run"},
        "frozen_scientific_objects": ["Teacher/Target", "Student", "tokenizer pair",
                                      "step80 checkpoint", "selected Adapter layers",
                                      "Teacher prior definition", "Null definition",
                                      "current alpha", "context", "dtype",
                                      "development split", "greedy decoding"],
        "track_a": {"margin_definition": "z_S[y_T] - z_S[y_S]",
                    "gradient": "d M / d p at the frozen injection rows, Student-only backward",
                    "oracle": "LOCAL_DECISION_ORACLE_DIAGNOSTIC — deterministic greedy mass "
                              "transfer, per-head L1 budget matched to the Teacher direction",
                    "topM_oracle_k": args.oracle_k},
        "track_b": {"representations": ["R0 Teacher attention (current prior)",
                                        "R1 TEACHER_ATTN_VALUE_NORM",
                                        "R2 TEACHER_CONTEXT_CONTRIBUTION"],
                    "projection": "unchanged frozen Teacher->Student projection "
                                  "(project_teacher_heads_to_student + frozen Adapter)"},
        "track_c": {"evidence": "HotpotQA supporting_facts mapped to student token spans by "
                                "character offsets; unmappable rows excluded with reason"},
        "track_d": {"population": "the 496 valid decision positions",
                    "bins": "S=T/S≠T, feature quartiles, natural answer-position bins, "
                            "median context split — fixed before looking at outcomes"},
        "bootstrap": {"n": 10000, "seed": 20260830, "label": "DEVELOPMENT diagnostic"},
        "claim_boundary": "development mechanism diagnostics only; the Oracle is not "
                          "deployable; no confirmation, no final method superiority, no "
                          "speedup, no adaptive-K/Repair benefit, and no claim that any new "
                          "representation works before it is injected and tested",
    }
    cfg["config_hash"] = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    H.atomic_write_text(out / "mech_experiment_config.json",
                        json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    H.atomic_write_text(out / "reused_parent_state.json",
                        json.dumps({"dr_config_hash": dr.get("config_hash"),
                                    "dr_case": dr_sum.get("case", {}).get("case"),
                                    "dr_denominator": dr_sum.get("denominator"),
                                    "dr_groups": {k: v.get("n") for k, v in
                                                  dr_sum.get("groups", {}).items()}},
                                   ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"config_hash": cfg["config_hash"],
                      "checkpoint_sha256": cfg["checkpoint_sha256"],
                      "parent_config_hash": cfg["parent_config_hash"],
                      "planned_positions": args.planned_positions,
                      "S_ne_T_positions": args.sne_t_positions}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
