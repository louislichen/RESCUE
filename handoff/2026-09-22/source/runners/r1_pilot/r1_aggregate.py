"""RESCUE-R1-VALUE-NORM-DECISION-ALIGNMENT-PILOT-001 — aggregation and classification.

Evaluates the falsification criterion preregistered in NEXT_METHOD_CANDIDATE_CARD.md exactly as
written (no threshold is invented, weakened or strengthened here):

    A = teacher_efficiency(R1) 95% CI overlaps teacher_efficiency(R0) 95% CI
    B = cos(dz_R1, z_Target - z_Base) 95% CI contains 0 AND |mean| < 0.05
    C = same-budget oracle efficiency of R1 stays below 10%
    falsified if (A and B) or C
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dr_common import bootstrap_ci, load_config, load_harness, spearman  # noqa: E402

ARMS = ("R0", "R1", "NULL")
OCRIT = 0.10


def stats(vals):
    if not vals:
        return None
    s = bootstrap_ci(vals)
    s["min"] = min(vals)
    s["max"] = max(vals)
    s["stdev"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return s


def paired(fn, rows):
    return stats([fn(r, "R1") - fn(r, o) for r in rows for o in ("R0", "NULL")
                  if o == "R0"]) if False else None


def contrast(rows, key, hi="R1", lo="R0"):
    vals = [key(r, hi) - key(r, lo) for r in rows]
    s = stats(vals)
    s["unchanged_fraction"] = sum(1 for v in vals if v == 0) / len(vals)
    return s


def overlaps(a, b):
    return not (a["ci95_high"] < b["ci95_low"] or b["ci95_high"] < a["ci95_low"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tc-rows", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    cfg = load_config(args.config)
    H = load_harness(args.harness_root)

    integrity, rows = [], []
    for p in sorted(out.glob("r1_shard_*.jsonl")):
        v = H.verify_result(p, expected_config_hash=cfg["config_hash"],
                            expected_checkpoint_sha256=cfg["checkpoint_sha256"])
        integrity.append({"path": str(p), "verify": v})
        if v["status"] == "PASS":
            rows.extend(H.read_result_records(p))
    meta = {}
    for p in sorted(out.glob("r1_shard_*_meta.json")):
        meta.update(json.loads(p.read_text(encoding="utf-8")))
    identity = meta.get("identity", [])
    mismatches = [i for i in identity if not i["identity_match"]]
    planned = meta.get("planned_positions")
    seen = {(r["canonical_source_id"], int(r["position"])) for r in rows}
    planned_keys = {(i["sample_id"], i["position"]) for i in identity}
    denominator = {
        "planned_positions": planned, "attempted_rows": len(rows), "valid_rows": len(rows),
        "missing_positions": len(planned_keys - seen),
        "extra_positions": len(seen - planned_keys),
        "identity_mismatches": len(mismatches),
        "target_top1_recomputed_all_match": all(
            r["target_top1_recomputed_matches_parent"] for r in rows),
        "grad_connected_all": all(all(r["grad_connected"].values()) for r in rows),
        "status": ("PASS" if (planned == len(rows) and not mismatches
                              and not (planned_keys - seen) and not (seen - planned_keys)
                              and all(r["target_top1_recomputed_matches_parent"] for r in rows))
                   else "FAIL_IDENTITY_OR_DENOMINATOR"),
    }

    def dm(r, arm):
        return r["arms"][arm]["M"] - r["M_base"]

    def dlogp(r, arm):
        return r["arms"][arm]["logp_yT"] - r["logp_base_yT"]

    def cosf(r, arm):
        return r["arms"][arm]["cos_to_dtarget"]

    def dznorm(r, arm):
        return r["arms"][arm]["dz_l2"]

    def l1(r, arm):
        return r["arms"][arm]["l1_total"]

    per_arm = {}
    for arm in ("BASE",) + ARMS:
        if arm == "BASE":
            per_arm[arm] = {"n": len(rows), "M": stats([r["M_base"] for r in rows]),
                            "logp_yT": stats([r["logp_base_yT"] for r in rows])}
            continue
        per_arm[arm] = {
            "n": len(rows),
            "dM": stats([dm(r, arm) for r in rows]),
            "dlogp_target": stats([dlogp(r, arm) for r in rows]),
            "cos_to_dtarget": stats([cosf(r, arm) for r in rows]),
            "l1_perturbation": stats([l1(r, arm) for r in rows]),
            "dz_l2": stats([dznorm(r, arm) for r in rows]),
            "flips_to_target": sum(1 for r in rows if r["arms"][arm]["flip_to_target"]),
            "flips_away_from_target": sum(1 for r in rows
                                          if r["arms"][arm]["flip_away_from_target"]),
            "unchanged": sum(1 for r in rows if not r["arms"][arm]["flip"]),
            "diagnostic_efficiency_dM_per_l1": stats(
                [dm(r, arm) / l1(r, arm) for r in rows if l1(r, arm) > 0]),
        }

    v_oracle = [r["track_a"]["V_oracle"] for r in rows]
    eff = {a: [r["V_first_order"][a] / r["track_a"]["V_oracle"] for r in rows]
           for a in ("R0", "R1", "Null")}
    eff_stats = {a: stats(v) for a, v in eff.items()}
    track_a_eff = {"R0": stats([r["track_a"]["teacher_efficiency"] for r in rows]),
                   "Null": stats([r["track_a"]["null_efficiency"] for r in rows])}

    contrasts = {}
    for hi, lo in (("R1", "R0"), ("R1", "NULL")):
        lo_key = "Null" if lo == "NULL" else lo
        contrasts[f"{hi}_minus_{lo}"] = {
            "dM": contrast(rows, dm, hi, lo),
            "dlogp_target": contrast(rows, dlogp, hi, lo),
            "cos_to_dtarget": contrast(rows, cosf, hi, lo),
            "V_first_order": stats([r["V_first_order"][hi] - r["V_first_order"][lo_key]
                                    for r in rows]),
            "l1_perturbation": stats([l1(r, hi) - l1(r, lo) for r in rows]),
            "dz_l2": stats([dznorm(r, hi) - dznorm(r, lo) for r in rows]),
        }

    # ---- preregistered criterion, verbatim ----
    c = cfg["falsification_criterion"]
    A = overlaps(eff_stats["R1"], track_a_eff["R0"])
    cos_r1 = per_arm["R1"]["cos_to_dtarget"]
    B = (cos_r1["ci95_low"] <= 0 <= cos_r1["ci95_high"]) and abs(cos_r1["mean"]) < 0.05
    C = eff_stats["R1"]["mean"] < OCRIT
    falsified = (A and B) or C
    improvement = {
        "dM_R1_minus_R0_mean": contrasts["R1_minus_R0"]["dM"]["mean"],
        "dM_R1_minus_R0_ci": [contrasts["R1_minus_R0"]["dM"]["ci95_low"],
                              contrasts["R1_minus_R0"]["dM"]["ci95_high"]],
        "dM_R1_minus_Null_mean": contrasts["R1_minus_NULL"]["dM"]["mean"],
        "cos_R1_minus_R0_mean": contrasts["R1_minus_R0"]["cos_to_dtarget"]["mean"],
        "oracle_efficiency_R1_mean": eff_stats["R1"]["mean"],
        "oracle_efficiency_R0_mean": track_a_eff["R0"]["mean"],
        "oracle_efficiency_Null_mean": eff_stats["Null"]["mean"],
    }
    shows_positive_improvement = bool(
        improvement["dM_R1_minus_R0_ci"][0] > 0
        or (improvement["oracle_efficiency_R1_mean"] > 0
            and improvement["oracle_efficiency_R1_mean"] > improvement["oracle_efficiency_R0_mean"]
            and eff_stats["R1"]["ci95_low"] > 0))
    if denominator["status"] != "PASS":
        verdict = "R1-AMBIGUOUS"
        reason = "identity/denominator failure — evidence cannot distinguish"
    elif falsified:
        verdict = "R1-FAIL"
        reason = "the preregistered falsification criterion fired"
    elif shows_positive_improvement:
        verdict = "R1-PASS"
        reason = ("preregistered criterion did not fire AND R1 shows a positive "
                  "decision-alignment improvement over R0/Null")
    else:
        verdict = "R1-AMBIGUOUS"
        reason = ("the preregistered rule is a falsification rule, not a superiority rule: it did "
                  "not fire, yet the directional reading shows no positive improvement over R0 — "
                  "returned for Supervisor decision rather than inventing a threshold")
    classification = {"verdict": verdict, "reason": reason,
                      "criterion_source": "NEXT_METHOD_CANDIDATE_CARD.md",
                      "criterion_as_written": c,
                      "A_efficiency_CIs_overlap": bool(A),
                      "B_cos_CI_contains_zero_and_small": bool(B),
                      "C_oracle_efficiency_below_10pct": bool(C),
                      "falsified": bool(falsified), "improvement_reading": improvement}

    summary = {
        "task_id": "RESCUE-R1-VALUE-NORM-DECISION-ALIGNMENT-PILOT-001",
        "status_as_received": cfg["status_as_received"],
        "config_hash": cfg["config_hash"], "checkpoint_sha256": cfg["checkpoint_sha256"],
        "population": "the 67 identity-matched S!=T decision positions",
        "framing": cfg["framing"],
        "denominator": denominator, "integrity": integrity,
        "identity": identity,
        "per_arm": per_arm,
        "paired_contrasts": contrasts,
        "oracle_efficiency": {"per_arm_from_this_run": eff_stats,
                              "R0_and_Null_from_track_a": track_a_eff,
                              "oracle_reference": 1.0,
                              "note": "V_arm / V_Oracle; the Oracle is "
                                      "LOCAL_DECISION_ORACLE_DIAGNOSTIC and not deployable"},
        "classification": classification,
        "claim_boundary": cfg["claim_boundary"],
    }

    H.write_result_envelope(
        out / "r1_decision_alignment_rows.jsonl",
        task_id="RESCUE-R1-VALUE-NORM-DECISION-ALIGNMENT-PILOT-001",
        stage="r1_decision_alignment_merged", records=rows, arms=["decision_position"],
        sample_ids=sorted({r["canonical_source_id"] for r in rows}),
        config_hash=cfg["config_hash"], checkpoint_sha256=cfg["checkpoint_sha256"])

    with (out / "r1_vs_r0_vs_null.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "source_index", "position", "prefix_sha256", "M_base",
                    "dM_R0", "dM_R1", "dM_Null",
                    "dlogp_target_R0", "dlogp_target_R1", "dlogp_target_Null",
                    "cos_R0", "cos_R1", "cos_Null",
                    "l1_R0", "l1_R1", "l1_Null", "dz_l2_R0", "dz_l2_R1", "dz_l2_Null",
                    "flip_to_target_R0", "flip_to_target_R1", "flip_to_target_Null"])
        for r in sorted(rows, key=lambda x: (x["source_index"], x["position"])):
            w.writerow([r["canonical_source_id"], r["source_index"], r["position"],
                        r["prefix_sha256"], round(r["M_base"], 6)] +
                       [round(dm(r, a), 6) for a in ARMS] +
                       [round(dlogp(r, a), 6) for a in ARMS] +
                       [round(cosf(r, a), 6) for a in ARMS] +
                       [round(l1(r, a), 6) for a in ARMS] +
                       [round(dznorm(r, a), 6) for a in ARMS] +
                       [int(r["arms"][a]["flip_to_target"]) for a in ARMS])

    with (out / "r1_oracle_efficiency.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "position", "V_R0", "V_R1", "V_Null", "V_Oracle",
                    "eff_R0", "eff_R1", "eff_Null",
                    "track_a_eff_R0", "track_a_eff_Null",
                    "deltaM_R1_per_l1", "deltaM_R0_per_l1", "deltaM_Null_per_l1"])
        for r in sorted(rows, key=lambda x: (x["source_index"], x["position"])):
            vo = r["track_a"]["V_oracle"]
            w.writerow([r["canonical_source_id"], r["position"],
                        round(r["V_first_order"]["R0"], 6),
                        round(r["V_first_order"]["R1"], 6),
                        round(r["V_first_order"]["Null"], 6), round(vo, 6),
                        round(r["V_first_order"]["R0"] / vo, 6),
                        round(r["V_first_order"]["R1"] / vo, 6),
                        round(r["V_first_order"]["Null"] / vo, 6),
                        "" if r["track_a"]["teacher_efficiency"] is None
                        else round(r["track_a"]["teacher_efficiency"], 6),
                        "" if r["track_a"]["null_efficiency"] is None
                        else round(r["track_a"]["null_efficiency"], 6)] +
                       [round(dm(r, a) / l1(r, a), 6) if l1(r, a) > 0 else ""
                        for a in ("R1", "R0", "NULL")])

    # ---- §10 semantic grounding cross-check ----
    tc = {}
    for line in Path(args.tc_rows).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "__result_header__" in r:
            continue
        rep = r.get("representations", {})
        r1 = rep.get("R1_TEACHER_ATTN_VALUE_NORM_teacher_space")
        if r1:
            tc[r["canonical_source_id"]] = {
                "evidence_mass_R1": r1["evidence_mass"],
                "topk_recall_R1": r1["topk_evidence_recall"],
                "entropy_within_evidence_R1": r1["entropy_within_evidence"]}
    link_rows = []
    for r in rows:
        g = tc.get(r["canonical_source_id"])
        if not g:
            continue
        vo = r["track_a"]["V_oracle"]
        link_rows.append({"sample_id": r["canonical_source_id"], "position": r["position"],
                          **g, "dM_R1": dm(r, "R1"),
                          "V_R1_over_V_oracle": r["V_first_order"]["R1"] / vo,
                          "cos_R1": cosf(r, "R1")})
    with (out / "r1_semantic_decision_link.csv").open("w", encoding="utf-8",
                                                      newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "position", "evidence_mass_R1", "topk_recall_R1",
                    "entropy_within_evidence_R1", "dM_R1", "V_R1_over_V_oracle", "cos_R1"])
        for r in link_rows:
            w.writerow([r["sample_id"], r["position"], round(r["evidence_mass_R1"], 6),
                        round(r["topk_recall_R1"], 6),
                        round(r["entropy_within_evidence_R1"], 6),
                        round(r["dM_R1"], 6), round(r["V_R1_over_V_oracle"], 6),
                        round(r["cos_R1"], 6)])
    semantic = {"positions_linked": len(link_rows),
                "correlation_note": "association only; never causation",
                "spearman_evidence_mass_vs_dM_R1": spearman(
                    [r["evidence_mass_R1"] for r in link_rows],
                    [r["dM_R1"] for r in link_rows]),
                "spearman_evidence_mass_vs_oracle_efficiency_R1": spearman(
                    [r["evidence_mass_R1"] for r in link_rows],
                    [r["V_R1_over_V_oracle"] for r in link_rows]),
                "spearman_topk_recall_vs_dM_R1": spearman(
                    [r["topk_recall_R1"] for r in link_rows],
                    [r["dM_R1"] for r in link_rows]),
                "spearman_evidence_mass_vs_cos_R1": spearman(
                    [r["evidence_mass_R1"] for r in link_rows],
                    [r["cos_R1"] for r in link_rows])}
    summary["semantic_decision_link"] = semantic

    H.atomic_write_text(out / "r1_decision_alignment_summary.json",
                        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "denominator": denominator,
        "dM_R0": {k: round(per_arm["R0"]["dM"][k], 6) for k in ("mean", "ci95_low", "ci95_high")},
        "dM_R1": {k: round(per_arm["R1"]["dM"][k], 6) for k in ("mean", "ci95_low", "ci95_high")},
        "dM_Null": {k: round(per_arm["NULL"]["dM"][k], 6)
                    for k in ("mean", "ci95_low", "ci95_high")},
        "cos_R0": round(per_arm["R0"]["cos_to_dtarget"]["mean"], 6),
        "cos_R1": {k: round(per_arm["R1"]["cos_to_dtarget"][k], 6)
                   for k in ("mean", "ci95_low", "ci95_high")},
        "R1_minus_R0_dM": {k: round(contrasts["R1_minus_R0"]["dM"][k], 6)
                           for k in ("mean", "ci95_low", "ci95_high")},
        "oracle_eff_R0/R1/Null": [round(track_a_eff["R0"]["mean"], 5),
                                  round(eff_stats["R1"]["mean"], 5),
                                  round(eff_stats["Null"]["mean"], 5)],
        "flips_to_target R0/R1/Null": [per_arm[a]["flips_to_target"] for a in ARMS],
        "semantic": {k: (None if v is None else round(v, 4))
                     for k, v in semantic.items() if isinstance(v, (int, float))},
        "criterion": {"A": A, "B": B, "C": C, "falsified": falsified},
        "VERDICT": verdict, "reason": reason,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
