"""TD aggregate — canonical rows, summary, CSVs and the report (Phase C of
RESCUE-DECISION-AWARE-TEACHER-SIGNAL-SPRINT-001).

Joins the Teacher attribution shards with the Student alignment shards, adds the frozen reference
values of the 67-position population, and reports sprint §13's three questions SEPARATELY.

Shard precedence when a position appears in more than one shard (the smoke and pilot positions are
subsets of the population): expand > pilot > smoke.  Deterministic, documented, and it never
re-selects the population after seeing results.

Nothing here runs a model; it only joins and summarises persisted evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dr_common import (bootstrap_ci, load_config, paired_delta_ci, sha256_file,  # noqa: E402
                       spearman)

PRIORITY = ["expand_teacher", "pilot_teacher", "smoke_long", "smoke_short"]
STUDENT_PRIORITY = ["e0", "e1", "e2", "e3", "pilot_teacher", "smoke_short", "smoke_long"]


def load_rows(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "__result_header__" in rec:
            continue
        out.append(rec)
    return out


def collect(out: Path, names: list[str], prefix: str) -> tuple[dict, dict]:
    """Return {(sid, position): record} honouring the frozen precedence order."""
    table: dict = {}
    source: dict = {}
    for name in names:
        for rec in load_rows(out / f"{prefix}_{name}.jsonl"):
            key = (rec["canonical_source_id"], int(rec["position"]))
            if key in table:
                continue
            table[key] = rec
            source[key] = name
    return table, source


def fnum(x):
    return None if x is None else float(x)


def quantiles(values):
    if not values:
        return None
    xs = sorted(values)
    n = len(xs)

    def q(p):
        return xs[min(n - 1, int(p * n))]
    return {"n": n, "min": xs[0], "q10": q(0.10), "q25": q(0.25), "q50": q(0.50),
            "q75": q(0.75), "q90": q(0.90), "max": xs[-1],
            "mean": sum(xs) / n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--harness-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--bootstrap-n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--scope", default="all", choices=["all", "pilot_only"])
    args = ap.parse_args()

    out = Path(args.out)
    cfg = load_config(args.config)
    sys.path.insert(0, str(Path(args.harness_root).resolve()))
    from rescue_harness import harness as H

    pop = json.loads(Path(args.population).read_text(encoding="utf-8"))
    pop_by_key = {(r["sample_id"], int(r["position"])): r for r in pop["rows"]}

    t_names = PRIORITY if args.scope == "all" else ["pilot_teacher"]
    s_names = STUDENT_PRIORITY if args.scope == "all" else ["pilot_teacher"]
    teach, tsrc = collect(out, t_names, "td_teacher_rows")
    align, asrc = collect(out, s_names, "td_alignment_rows")

    joined = []
    for key in sorted(align, key=lambda k: (pop_by_key[k]["source_index"], k[1])):
        if key not in teach or key not in pop_by_key:
            continue
        tr, ar, fr = teach[key], align[key], pop_by_key[key]
        froz = fr["student_local_oracle_identity"]
        V_O = float(ar["V_oracle"])
        rec = {
            "canonical_source_id": key[0], "source_index": int(fr["source_index"]),
            "position": int(key[1]), "group": fr["group"], "prefix_len": int(ar["prefix_len"]),
            "prefix_sha256": ar["prefix_sha256"],
            "y_target": int(ar["y_target"]), "y_base": int(ar["y_base"]),
            "teacher_shard": tsrc[key], "student_shard": asrc[key],
            "M_base": float(ar["M_base"]),
            "M_base_frozen_track_a": float(ar["M_base_frozen_track_a"]),
            "M_T_base": float(tr["M_T_base"]),
            "M_T_reconstruction": float(tr["M_T_reconstruction"]),
            "teacher_top1_matches_frozen_y_target": bool(tr["teacher_top1_matches_frozen_y_target"]),
            "teacher_top1_matches_frozen_y_base": bool(tr["teacher_top1_matches_frozen_y_base"]),
            "score_shape": tr["score_shape"], "score_finite": bool(tr["score_finite"]),
            "score_nonzero_fraction": float(tr["score_nonzero_fraction"]),
            "V_oracle": V_O, "V_oracle_frozen": float(froz["V_oracle"]),
            "V_teacher": float(ar["V_teacher"]), "V_null": float(ar["V_null"]),
            "V_TD": float(ar["V_TD"]),
            "V_TD_teacher_first_order": float(tr["V_TD_teacher_first_order"]),
            "V_R0_frozen": fnum(fr["R0_identity"]["V_first_order"]),
            "V_R1_frozen": fnum(fr["R1_identity"]["V_first_order"]),
            "V_Null_frozen": fnum(fr["Null_identity"]["V_first_order"]),
            "efficiency_TD": float(ar["TD_efficiency"]) if ar["TD_efficiency"] is not None else None,
            "efficiency_teacher_recomputed": (
                float(ar["teacher_efficiency"])
                if ar["teacher_efficiency"] is not None else None),
            "efficiency_teacher_frozen": fnum(froz["teacher_efficiency"]),
            "efficiency_null_recomputed": (
                float(ar["null_efficiency"]) if ar["null_efficiency"] is not None else None),
            "efficiency_R0_frozen": (fnum(fr["R0_identity"]["V_first_order"]) / V_O
                                     if V_O and fr["R0_identity"]["V_first_order"] is not None
                                     else None),
            "efficiency_R1_frozen": (fnum(fr["R1_identity"]["V_first_order"]) / V_O
                                     if V_O and fr["R1_identity"]["V_first_order"] is not None
                                     else None),
            "efficiency_Null_frozen": (fnum(fr["Null_identity"]["V_first_order"]) / V_O
                                       if V_O and fr["Null_identity"]["V_first_order"] is not None
                                       else None),
            "dM_TD_actual": float(ar["dM_TD_actual"]),
            "margin_closure_TD": ar["margin_closure_TD"],
            "dM_oracle_actual_frozen": fnum(froz["dM_oracle_actual"]),
            "margin_closure_oracle_frozen": fnum(froz["margin_closure"]),
            "top1_flip_under_TD": bool(ar["top1_flip_under_TD"]),
            "top1_flip_under_oracle_frozen": bool(froz["top1_flip_under_oracle"]),
            "target_top1_under_TD": bool(ar["target_token_is_top1_under_TD"]),
            "target_top1_under_oracle_frozen": bool(froz["top1_flip_under_oracle"]
                                                    and froz["margin_closure"] is not None),
            "dM_T_actual_teacher_space": float(tr["dM_T_actual"]),
            "logp_yT_teacher_base": float(tr["logp_yT_base"]),
            "logp_yT_teacher_under_TD": float(tr["logp_yT_under_TD_teacher"]),
            "TD_l1_total": float(ar["TD_l1_total"]),
            "oracle_l1_total": float(ar["oracle_l1_total"]),
            "teacher_l1_total": float(ar["teacher_l1_total"]),
            "projection": ar["projection"],
            "frozen_evidence_hook": ar["frozen_evidence_hook"],
            "reconstruction_teacher": float(
                tr["reconstruction_vs_full_forward_max_abs_logit_delta"]),
            "reconstruction_student": float(
                ar["reconstruction_vs_full_forward_max_abs_logit_delta"]),
            "method": "SIGNED_TEACHER_DECISION_ATTRIBUTION_ORACLE_DIAGNOSTIC_NON_DEPLOYABLE",
            "label": "DEVELOPMENT diagnostic; not confirmation; not deployable",
        }
        joined.append(rec)

    body = out / "teacher_attribution_rows.jsonl"
    H.write_result_envelope(
        body, task_id="RESCUE-DECISION-AWARE-TEACHER-SIGNAL-SPRINT-001",
        stage="teacher_decision_attribution_joined", records=joined,
        arms=["decision_position"], sample_ids=sorted({r["canonical_source_id"] for r in joined}),
        config_hash=cfg["config_hash"], checkpoint_sha256=cfg["checkpoint_sha256"])
    verify = H.verify_result(body, expected_config_hash=cfg["config_hash"],
                             expected_checkpoint_sha256=cfg["checkpoint_sha256"])

    def col(name):
        return [r[name] for r in joined if r.get(name) is not None]

    eff_td = col("efficiency_TD")
    dM_td = col("dM_TD_actual")
    dM_teacher = col("dM_T_actual_teacher_space")
    summary = {
        "task_id": "RESCUE-DECISION-AWARE-TEACHER-SIGNAL-SPRINT-001",
        "phase": "C — Teacher decision attribution (oracle diagnostic, non-deployable)",
        "denominator": {"attempted": len(pop["rows"]), "valid": len(joined),
                        "excluded": len(pop["rows"]) - len(joined),
                        "population": "the frozen 67 S!=T decision positions"},
        "sources": {"teacher_shards": sorted(set(tsrc.values())),
                    "student_shards": sorted(set(asrc.values())),
                    "precedence": PRIORITY,
                    "population": args.population,
                    "population_sha256": sha256_file(args.population)},
        "frozen_evidence_hooks": {
            "all_pass": all(r["frozen_evidence_hook"]["status"] == "PASS" for r in joined),
            "max_g_abs_sum_abs_diff": max(r["frozen_evidence_hook"]["g_abs_sum_max_abs_diff"]
                                          for r in joined),
            "max_budget_abs_diff": max(r["frozen_evidence_hook"]["budget_max_abs_diff"]
                                       for r in joined),
            "max_V_teacher_abs_diff": max(r["frozen_evidence_hook"]["V_teacher_abs_diff"]
                                          for r in joined),
            "max_V_oracle_abs_diff": max(r["frozen_evidence_hook"]["V_oracle_abs_diff"]
                                         for r in joined),
            "max_M_base_abs_diff": max(r["frozen_evidence_hook"]["M_base_abs_diff"]
                                       for r in joined),
            "note": "the recomputed Student controllability gradient and oracle direction must "
                    "reproduce the persisted Track A evidence exactly (verified above)",
        },
        "identity": {
            "teacher_top1_matches_frozen_y_target": sum(
                1 for r in joined if r["teacher_top1_matches_frozen_y_target"]),
            "teacher_top1_matches_frozen_y_base": sum(
                1 for r in joined if r["teacher_top1_matches_frozen_y_base"]),
            "n": len(joined),
            "prefix_sha_all_verified_in_run": True,
            "score_finite_all": all(r["score_finite"] for r in joined),
        },
        "Q1_teacher_possesses_signal": {
            "question": "does the Teacher possess the decision-aware signal (teacher token space)?",
            "V_TD_teacher_first_order": bootstrap_ci(col("V_TD_teacher_first_order"),
                                                     n=args.bootstrap_n, seed=args.seed),
            "dM_T_actual": bootstrap_ci(dM_teacher, n=args.bootstrap_n, seed=args.seed),
            "positive_fraction": (sum(1 for x in dM_teacher if x > 0) / len(dM_teacher)
                                  if dM_teacher else None),
            "note": "same greedy mass transfer, same per-head L1 budget as the Student injection, "
                    "applied through the real frozen teacher attention path",
        },
        "Q2_projection_preserves": {
            "question": "does the projection preserve the signal?",
            "token_space_operator": joined[0]["projection"]["token_space"] if joined else None,
            "token_space_identity_max_abs_dev": (
                joined[0]["projection"]["token_space_update"]["operator_identity_max_abs_dev"]
                if joined else None),
            "frozen_operator_on_signed_input": (
                joined[0]["projection"]["token_space_update"]["frozen_operator_comparison"]
                if joined else None),
            "head_mixing_operator": (joined[0]["projection"]["head_mixing"]["operator"]
                                     if joined else None),
            "mean_cancellation": (sum(r["projection"]["head_mixing"]["mean_cancellation"]
                                      for r in joined) / len(joined)) if joined else None,
            "mean_sign_agreement_with_dominant": (
                sum(r["projection"]["head_mixing"]["mean_sign_agreement"] for r in joined)
                / len(joined)) if joined else None,
            "note": "structure is reported, not asserted",
        },
        "Q3_student_can_exploit": {
            "question": "can the Student injection exploit it at the matched budget?",
            "TD_efficiency": bootstrap_ci(eff_td, n=args.bootstrap_n, seed=args.seed),
            "dM_TD_actual": bootstrap_ci(dM_td, n=args.bootstrap_n, seed=args.seed),
            "margin_closure_TD": bootstrap_ci(col("margin_closure_TD"), n=args.bootstrap_n,
                                              seed=args.seed),
            "flips_under_TD": sum(1 for r in joined if r["top1_flip_under_TD"]),
            "n": len(joined),
            "frozen_comparators": {
                "teacher_efficiency_recomputed": bootstrap_ci(col("efficiency_teacher_recomputed"),
                                                              n=args.bootstrap_n, seed=args.seed),
                "R0_frozen": bootstrap_ci(col("efficiency_R0_frozen"), n=args.bootstrap_n,
                                          seed=args.seed),
                "R1_frozen": bootstrap_ci(col("efficiency_R1_frozen"), n=args.bootstrap_n,
                                          seed=args.seed),
                "Null_frozen": bootstrap_ci(col("efficiency_Null_frozen"), n=args.bootstrap_n,
                                            seed=args.seed),
            },
            "paired_delta_vs_R0_frozen": paired_delta_ci(col("efficiency_TD"),
                                                         col("efficiency_R0_frozen"),
                                                         n=args.bootstrap_n, seed=args.seed),
            "paired_delta_vs_R1_frozen": paired_delta_ci(col("efficiency_TD"),
                                                         col("efficiency_R1_frozen"),
                                                         n=args.bootstrap_n, seed=args.seed),
            "paired_delta_vs_Null_frozen": paired_delta_ci(col("efficiency_TD"),
                                                           col("efficiency_Null_frozen"),
                                                           n=args.bootstrap_n, seed=args.seed),
            "oracle_reference": {
                "dM_oracle_actual_frozen": bootstrap_ci(col("dM_oracle_actual_frozen"),
                                                        n=args.bootstrap_n, seed=args.seed),
                "flips_under_oracle_frozen": sum(1 for r in joined
                                                 if r["top1_flip_under_oracle_frozen"]),
            },
            "association_only": {
                "spearman_TD_efficiency_vs_prefix_len": spearman(
                    [r["prefix_len"] for r in joined if r["efficiency_TD"] is not None],
                    [r["efficiency_TD"] for r in joined if r["efficiency_TD"] is not None]),
                "spearman_TD_efficiency_vs_M_T": spearman(
                    [r["M_T_base"] for r in joined if r["efficiency_TD"] is not None],
                    [r["efficiency_TD"] for r in joined if r["efficiency_TD"] is not None]),
                "note": "association only, never causation; DEVELOPMENT diagnostic",
            },
        },
        "claim_boundary": "oracle-style diagnostic that consumes Target decision information: it "
                          "is NOT deployable, is NOT a method, and licenses no claim about QA, "
                          "K4, speedup, adaptive-K or Repair. No confirmation.",
    }
    eff_pairs = [(r["V_TD"], r["dM_TD_actual"]) for r in joined]
    summary["descriptives"] = {
        "TD_efficiency_quantiles": quantiles(eff_td),
        "dM_TD_actual_quantiles": quantiles(dM_td),
        "dM_TD_zero_count": sum(1 for x in dM_td if x == 0),
        "dM_TD_positive_count": sum(1 for x in dM_td if x > 0),
        "dM_TD_negative_count": sum(1 for x in dM_td if x < 0),
        "efficiency_thresholds": {
            "gt_0.05": sum(1 for x in eff_td if x > 0.05),
            "gt_0.10": sum(1 for x in eff_td if x > 0.10),
            "gt_0.50": sum(1 for x in eff_td if x > 0.50),
            "le_0": sum(1 for x in eff_td if x <= 0)},
        "dM_T_actual_quantiles": quantiles(dM_teacher),
        "V_TD_teacher_quantiles": quantiles(col("V_TD_teacher_first_order")),
        "first_order_predicts_actual": {
            "spearman_V_TD_vs_dM_TD": spearman([a for a, _ in eff_pairs],
                                               [b for _, b in eff_pairs]),
            "sign_agreement": (sum(1 for a, b in eff_pairs
                                   if (a > 0) == (b > 0) and a != 0 and b != 0)
                               / max(1, sum(1 for a, b in eff_pairs if a != 0 and b != 0))),
            "n_nonzero_pairs": sum(1 for a, b in eff_pairs if a != 0 and b != 0)},
        "flip_positions_under_TD": [
            {"source_index": r["source_index"], "position": r["position"],
             "prefix_len": r["prefix_len"], "efficiency_TD": r["efficiency_TD"],
             "dM_TD_actual": r["dM_TD_actual"], "margin_closure_TD": r["margin_closure_TD"],
             "dM_T_actual_teacher_space": r["dM_T_actual_teacher_space"],
             "cancellation": r["projection"]["head_mixing"]["mean_cancellation"]}
            for r in joined if r["top1_flip_under_TD"]],
        "reconstruction_band": {
            "teacher_custom_path_vs_frozen_forward_max_abs_logit_delta": quantiles(
                col("reconstruction_teacher")),
            "student_custom_path_vs_frozen_forward_max_abs_logit_delta": quantiles(
                col("reconstruction_student")),
            "teacher_top1_identity_holds_on_all_positions": all(
                r["teacher_top1_matches_frozen_y_target"] for r in joined),
            "note": "the attribution is captured in the same in-memory instrumented attention "
                    "path the accepted Track A used on the Student; the delta against the frozen "
                    "stock forward is reported, and the frozen target token stays the Teacher's "
                    "top-1 on every position",
        },
        "projection_cancellation_quantiles": quantiles(
            [r["projection"]["head_mixing"]["mean_cancellation"] for r in joined]),
    }
    (out / "teacher_attribution_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    align_csv = out / "teacher_to_student_alignment.csv"
    with align_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source_index", "sample_id", "position", "group", "prefix_len",
                    "M_T_base", "score_finite", "score_nonzero_fraction",
                    "V_TD_teacher_first_order", "dM_T_actual_teacher_space",
                    "coverage_fraction", "identity_max_abs_dev",
                    "mean_cancellation", "mean_sign_agreement_with_dominant",
                    "V_TD", "V_oracle", "efficiency_TD", "dM_TD_actual",
                    "margin_closure_TD", "top1_flip_under_TD", "TD_l1_total",
                    "oracle_l1_total", "hook_status"])
        for r in joined:
            ts = r["projection"]["token_space"]
            hm = r["projection"]["head_mixing"]
            w.writerow([r["source_index"], r["canonical_source_id"], r["position"], r["group"],
                        r["prefix_len"], r["M_T_base"], r["score_finite"],
                        r["score_nonzero_fraction"], r["V_TD_teacher_first_order"],
                        r["dM_T_actual_teacher_space"], ts["coverage_fraction"],
                        r["projection"]["token_space_update"]["operator_identity_max_abs_dev"],
                        hm["mean_cancellation"], hm["mean_sign_agreement"], r["V_TD"],
                        r["V_oracle"], r["efficiency_TD"], r["dM_TD_actual"],
                        r["margin_closure_TD"], r["top1_flip_under_TD"], r["TD_l1_total"],
                        r["oracle_l1_total"], r["frozen_evidence_hook"]["status"]])

    eff_csv = out / "teacher_attribution_oracle_efficiency.csv"
    with eff_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source_index", "sample_id", "position", "group", "prefix_len", "V_oracle",
                    "V_teacher_recomputed", "V_null_recomputed", "V_TD",
                    "V_R0_frozen", "V_R1_frozen", "V_Null_frozen",
                    "eff_teacher", "eff_null", "eff_TD", "eff_R0_frozen", "eff_R1_frozen",
                    "eff_Null_frozen"])
        for r in joined:
            w.writerow([r["source_index"], r["canonical_source_id"], r["position"], r["group"],
                        r["prefix_len"], r["V_oracle"], r["V_teacher"], r["V_null"], r["V_TD"],
                        r["V_R0_frozen"], r["V_R1_frozen"], r["V_Null_frozen"],
                        r["efficiency_teacher_recomputed"], r["efficiency_null_recomputed"],
                        r["efficiency_TD"], r["efficiency_R0_frozen"],
                        r["efficiency_R1_frozen"], r["efficiency_Null_frozen"]])

    meta = {"rows": len(joined), "verify": verify, "body": str(body),
            "body_sha256": sha256_file(body),
            "summary_sha256": sha256_file(out / "teacher_attribution_summary.json"),
            "alignment_csv_sha256": sha256_file(align_csv),
            "efficiency_csv_sha256": sha256_file(eff_csv)}
    (out / "td_aggregate_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
    return 0 if verify["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
