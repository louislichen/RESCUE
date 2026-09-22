"""TD prepare — frozen-input identity snapshot + the TD experiment config.

Phase C of RESCUE-DECISION-AWARE-TEACHER-SIGNAL-SPRINT-001.  Nothing is run here: this only
freezes, hashes and records the identities the attribution run will be bound to (harness manifest,
checkpoint, source lock, Teacher/Student models, the pre-registered 67-position population and the
pre-registered smoke/pilot set), then writes the config whose `config_hash` every TD result envelope
carries.  Hash method is the one the accepted mechanism sprint used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dr_common import sha256_file  # noqa: E402

FROZEN = {
    "checkpoint_sha256": "6532daad4e3ae54a7e76063e6aef3fef9b59ba6ef709394bf361c30228f19ace",
    "steering.py": "24eaeed620e3a04702ab7b61b1d0b5756d588dbc1b54ae444710a79e2cfb5791",
    "adapter.py": "7253fa9fb1e5f7d513afdbc7d8863cb5113fb98d90f79f77284b3879f4a32722",
    "common.py": "e8f7361c766bd3ee5dd05aa01630eb51c21cb890e795c093f366bea800278581",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-root", required=True)
    ap.add_argument("--manifest-sha256", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--teacher-model", required=True)
    ap.add_argument("--student-model", required=True)
    ap.add_argument("--source-lock", required=True)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--positions-set", required=True)
    ap.add_argument("--parent-config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    H = Path(args.harness_root)
    manifest = H / "HARNESS_FREEZE_CANDIDATE_MANIFEST_V3.json"
    manifest_sha = sha256_file(manifest)
    if manifest_sha != args.manifest_sha256:
        raise SystemExit(f"harness manifest sha mismatch: file {manifest_sha} != cli "
                         f"{args.manifest_sha256}")
    ck_sha = sha256_file(args.checkpoint)
    if ck_sha != FROZEN["checkpoint_sha256"]:
        raise SystemExit(f"checkpoint is not the frozen step80: {ck_sha}")
    lock_sha = {n: sha256_file(Path(args.source_lock) / n)
                for n in ("adapter.py", "common.py", "steering.py")}
    for n, v in lock_sha.items():
        if v != FROZEN[n]:
            raise SystemExit(f"source lock {n} is not the frozen file: {v} != {FROZEN[n]}")

    pop_path = Path(args.population)
    set_path = Path(args.positions_set)
    pop = json.loads(pop_path.read_text(encoding="utf-8"))
    spec = json.loads(set_path.read_text(encoding="utf-8"))
    if not pop.get("identity_gate_pass") or pop.get("population_matched") != 67:
        raise SystemExit("the pre-registered population is not the frozen 67/67 identity-gate set")
    if spec.get("selection_sha256") != "bf055e892bdb9a488463a703d0a1b666746b7124f8fd386fd7aee834231be9b2":
        raise SystemExit("the pre-registered smoke/pilot selection hash changed")
    parent = json.loads(Path(args.parent_config).read_text(encoding="utf-8"))

    cfg = {
        "task_id": "RESCUE-DECISION-AWARE-TEACHER-SIGNAL-SPRINT-001",
        "mode": "PHASE C — TEACHER DECISION ATTRIBUTION / ORACLE DIAGNOSTIC / NO TRAINING / "
                "NO METHOD CHANGE / NO PARAMETER UPDATE",
        "parent_task": "RESCUE-TEACHER-DECISION-ATTRIBUTION-IMPLEMENTATION-REVIEW-001",
        "parent_result": "IMPLEMENTATION_PLAN_ACCEPTABLE",
        "harness": "HARNESS_FREEZE_V3",
        "harness_manifest_sha256": manifest_sha,
        "harness_manifest_path": str(manifest),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": ck_sha,
        "teacher_model": str(args.teacher_model),
        "student_model": str(args.student_model),
        "source_lock": str(args.source_lock),
        "source_lock_sha256": lock_sha,
        "source_dir": str(args.source_dir),
        "parent_config_hash": parent.get("config_hash"),
        "population": {"path": str(pop_path), "sha256": sha256_file(pop_path),
                       "rows": len(pop["rows"]), "identity_gate": pop.get("identity_gate"),
                       "frozen_before_attribution": bool(pop.get("fixed_before_attribution")),
                       "groups": sorted({r["group"] for r in pop["rows"]})},
        "positions_set": {"path": str(set_path), "sha256": sha256_file(set_path),
                          "selection_sha256": spec.get("selection_sha256"),
                          "selected_before_attribution": bool(
                              spec.get("selection_frozen_before_attribution"))},
        "canonical_attribution": {
            "capture_point": "last-query-row attention probability at teacher layers 24-27; "
                             "post-mask, post-softmax (fp32), pre-dropout, post-repeat_kv, pre-AV "
                             "— the tensor the frozen qwen2_teacher_row_observer produces",
            "definition": "M_T = z_T[y_T] - z_T[y_S]; score[l,h,j] = dM_T/da[l,h,j]",
            "sign": "kept; never forced into a distribution",
            "head_aggregation_before_projection": "none; score stays [4,28,K]",
            "gradient_extraction": "retain_grad() + backward(), then slice the last query row",
            "autograd_scope": "layers 0-23 under no_grad (pass A); layers 24-27 re-run with the "
                              "graph (pass B) through the frozen modules",
        },
        "approved_projection_path": {
            "decision": "Supervisor-approved: reuse the frozen operators' weights, drop the "
                        "non-negativity/probability steps, rescale to the same per-head L1 budget",
            "token_space": "frozen project_teacher_prior_to_student weights; clamping and "
                           "renormalisation dropped (frozen file untouched)",
            "head_space": "frozen AttentionPriorAdapter.mix_logits softmax weights; "
                          "normalize_rows, residual mix and final softmax dropped",
            "budget": "per-head L1 budget of the frozen Teacher direction d_T at the Student "
                      "injection site (Track A, persisted)",
            "fallback_if_required": "teacher-token-space-only answers to questions (1) and (2)",
        },
        "oracle_style": "NON-DEPLOYABLE: uses Target decision information; a diagnostic object, "
                        "never a method",
        "bootstrap": {"n": 10000, "seed": 20260830, "label": "DEVELOPMENT diagnostic"},
        "claim_boundary": "development mechanism diagnostics only; TD is oracle-style and not "
                          "deployable; no confirmation, no method superiority, no QA/K4 claim, "
                          "no speedup, no adaptive-K/Repair benefit",
    }
    cfg["config_hash"] = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (out / "td_experiment_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"config_hash": cfg["config_hash"], "checkpoint_sha256": ck_sha,
                      "harness_manifest_sha256": manifest_sha,
                      "population_sha256": cfg["population"]["sha256"],
                      "positions_set_sha256": cfg["positions_set"]["sha256"],
                      "selection_sha256": cfg["positions_set"]["selection_sha256"],
                      "frozen_track_a_config": parent.get("config_hash")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
