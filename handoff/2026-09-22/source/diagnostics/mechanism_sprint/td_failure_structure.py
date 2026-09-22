"""TRACK D — failure-structure analysis over the existing 496 valid decision positions.

Predetermined features and PREDETERMINED bins only (declared here, before any outcome is
inspected):

  group                 S=T / S≠T                       natural
  student_margin_deficit  -M_A                          quartiles of the feature
  student_entropy         H(z_S)                        quartiles of the feature
  context_length          prefix_len                    median split
  answer_position         decision index                natural bins {0} {1} {2} {3+}
  teacher_prior_distance  L1(teacher prior, Student row) quartiles of the feature

Per stratum this reports the Teacher / Null margin movement, the paired Teacher-minus-Null
contrast and the Teacher direction cosine with paired bootstrap intervals, and flags a
CONDITIONAL_SIGNAL_CANDIDATE when the Teacher signal is positive *and* better than Null with a
95 % interval excluding zero and n >= 10.  Everything is a development diagnostic; the
multiple-comparison caveat is recorded with the candidates.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dr_common import bootstrap_ci  # noqa: E402

MIN_N = 10


def quartile_bins(vals):
    s = sorted(vals)
    n = len(s)
    q1, q2, q3 = s[n // 4], s[n // 2], s[(3 * n) // 4]
    return [q1, q2, q3]


def assign_bin(v, cuts, labels):
    for i, c in enumerate(cuts):
        if v <= c:
            return labels[i]
    return labels[-1]


def stratum(rows, label, key):
    if len(rows) < 2:
        return None
    dM_t = [r["dM_teacher"] for r in rows]
    dM_n = [r["dM_null"] for r in rows]
    cos_t = [r["cos_teacher"] for r in rows]
    st = bootstrap_ci(dM_t)
    sn = bootstrap_ci(dM_n)
    pair = bootstrap_ci([a - b for a, b in zip(dM_t, dM_n)])
    sc = bootstrap_ci(cos_t)
    return {
        "bin_family": key, "bin": label, "n": len(rows),
        "dM_teacher_mean": st["mean"], "dM_teacher_ci": [st["ci95_low"], st["ci95_high"]],
        "dM_teacher_positive_fraction": st["positive_fraction"],
        "dM_null_mean": sn["mean"], "dM_null_ci": [sn["ci95_low"], sn["ci95_high"]],
        "teacher_minus_null_mean": pair["mean"],
        "teacher_minus_null_ci": [pair["ci95_low"], pair["ci95_high"]],
        "cos_teacher_mean": sc["mean"], "cos_teacher_ci": [sc["ci95_low"], sc["ci95_high"]],
        "teacher_dM_positive_CI": st["ci95_low"] > 0,
        "teacher_beats_null_CI": pair["ci95_low"] > 0,
        "conditional_signal_candidate": bool(st["ci95_low"] > 0 and pair["ci95_low"] > 0
                                             and len(rows) >= MIN_N),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dr-rows", required=True)
    args = ap.parse_args()
    out = Path(args.out)

    dr = {}
    for line in Path(args.dr_rows).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if "__result_header__" not in r:
                dr[(r["canonical_source_id"], int(r["position"]))] = r
    feat = {}
    for path in sorted(out.glob("td_shard_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "__result_header__" in r:
                continue
            feat[(r["canonical_source_id"], int(r["position"]))] = r

    rows = []
    missing = 0
    for k, r in dr.items():
        f = feat.get(k)
        if f is None:
            missing += 1
            continue
        pd = r.get("prior_distance") or {}
        rows.append({
            "sample_id": r["canonical_source_id"], "position": int(r["position"]),
            "group": r["group"], "prefix_len": int(r["prefix_len"]),
            "M_A": float(r["M_A"]), "student_margin_deficit": -float(r["M_A"]),
            "student_entropy": float(f["student_entropy"]),
            "student_top1_prob": float(f["student_top1_prob"]),
            "student_top2_gap": float(f["student_top2_gap"]),
            "margin_vs_yS": float(f["margin_vs_yS"]),
            "dM_teacher": float(r["dM_teacher"]), "dM_null": float(r["dM_null"]),
            "dlogp_T_teacher": float(r["dlogp_T_teacher"]),
            "dlogp_T_null": float(r["dlogp_T_null"]),
            "cos_teacher": float(r["cos_teacher"]), "cos_null": float(r["cos_null"]),
            "teacher_prior_distance": pd.get("teacher_l1"),
            "null_prior_distance": pd.get("null_l1"),
            "student_entropy_parent_delta": None,
        })

    fams = {}
    fams["group"] = [("S!=T", [r for r in rows if r["group"] == "S!=T"]),
                     ("S=T", [r for r in rows if r["group"] == "S=T"])]
    cuts = quartile_bins([r["student_margin_deficit"] for r in rows])
    labs = [f"Q1 <= {cuts[0]:.3f}", f"Q2 <= {cuts[1]:.3f}", f"Q3 <= {cuts[2]:.3f}",
            f"Q4 > {cuts[2]:.3f}"]
    fams["student_margin_deficit"] = [
        (labs[i], [r for r in rows
                   if assign_bin(r["student_margin_deficit"], cuts, labs) == labs[i]])
        for i in range(4)]
    cuts = quartile_bins([r["student_entropy"] for r in rows])
    labs = [f"Q1 <= {cuts[0]:.3f}", f"Q2 <= {cuts[1]:.3f}", f"Q3 <= {cuts[2]:.3f}",
            f"Q4 > {cuts[2]:.3f}"]
    fams["student_entropy"] = [
        (labs[i], [r for r in rows
                   if assign_bin(r["student_entropy"], cuts, labs) == labs[i]])
        for i in range(4)]
    med = statistics.median([r["prefix_len"] for r in rows])
    fams["context_length"] = [(f"short <= {int(med)}", [r for r in rows if r["prefix_len"] <= med]),
                              (f"long > {int(med)}", [r for r in rows if r["prefix_len"] > med])]
    fams["answer_position"] = [(str(p), [r for r in rows if (r["position"] == p if p < 3
                                                             else r["position"] >= 3)])
                               for p in (0, 1, 2, 3)]
    fams["answer_position"][-1] = ("3+", [r for r in rows if r["position"] >= 3])
    pd_vals = [r["teacher_prior_distance"] for r in rows if r["teacher_prior_distance"]]
    cuts = quartile_bins(pd_vals)
    labs = [f"Q1 <= {cuts[0]:.3f}", f"Q2 <= {cuts[1]:.3f}", f"Q3 <= {cuts[2]:.3f}",
            f"Q4 > {cuts[2]:.3f}"]
    fams["teacher_prior_distance"] = [
        (labs[i], [r for r in rows if r["teacher_prior_distance"] is not None
                   and assign_bin(r["teacher_prior_distance"], cuts, labs) == labs[i]])
        for i in range(4)]

    table = []
    for fam, bins in fams.items():
        for label, rs in bins:
            s = stratum(rs, label, fam)
            if s:
                table.append(s)
    candidates = [t for t in table if t["conditional_signal_candidate"]]

    with (out / "failure_structure.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_family", "bin", "n", "dM_teacher_mean", "dM_teacher_ci_low",
                    "dM_teacher_ci_high", "dM_teacher_positive_fraction", "dM_null_mean",
                    "teacher_minus_null_mean", "teacher_minus_null_ci_low",
                    "teacher_minus_null_ci_high", "cos_teacher_mean", "cos_teacher_ci_low",
                    "cos_teacher_ci_high", "teacher_dM_positive_CI", "teacher_beats_null_CI",
                    "conditional_signal_candidate"])
        for t in table:
            w.writerow([t["bin_family"], t["bin"], t["n"],
                        round(t["dM_teacher_mean"], 6), round(t["dM_teacher_ci"][0], 6),
                        round(t["dM_teacher_ci"][1], 6),
                        round(t["dM_teacher_positive_fraction"], 4),
                        round(t["dM_null_mean"], 6), round(t["teacher_minus_null_mean"], 6),
                        round(t["teacher_minus_null_ci"][0], 6),
                        round(t["teacher_minus_null_ci"][1], 6),
                        round(t["cos_teacher_mean"], 6), round(t["cos_teacher_ci"][0], 6),
                        round(t["cos_teacher_ci"][1], 6), int(t["teacher_dM_positive_CI"]),
                        int(t["teacher_beats_null_CI"]),
                        int(t["conditional_signal_candidate"])])

    analysis = {
        "task_id": "RESCUE-GUIDANCE-REPRESENTATION-MECHANISM-SPRINT-001",
        "track": "D — failure-structure analysis",
        "population": {"decision_positions_with_features": len(rows),
                       "planned_from_parent": len(dr), "missing_feature_rows": missing},
        "predetermined_features": ["group", "student_margin_deficit", "student_entropy",
                                   "context_length", "answer_position",
                                   "teacher_prior_distance", "dM_teacher", "dM_null",
                                   "dlogp_T_teacher", "cos_teacher"],
        "predetermined_bins": {
            "group": "natural S=T / S!=T",
            "student_margin_deficit": f"quartiles {[round(c, 4) for c in quartile_bins([r['student_margin_deficit'] for r in rows])]}",
            "student_entropy": f"quartiles {[round(c, 4) for c in quartile_bins([r['student_entropy'] for r in rows])]}",
            "context_length": f"median split at {int(med)}",
            "answer_position": "natural {0} {1} {2} {3+}",
            "teacher_prior_distance": f"quartiles {[round(c, 4) for c in quartile_bins(pd_vals)]}",
        },
        "candidate_rule": f"Teacher dM CI>0 AND Teacher-minus-Null CI>0 AND n>={MIN_N}",
        "strata": table,
        "conditional_signal_candidates": candidates,
        "status": ("CONDITIONAL_SIGNAL_CANDIDATE" if candidates else "NO_CONDITIONAL_SIGNAL"),
        "caveats": [
            "exploratory development analysis, not confirmation",
            "bins were fixed before the outcomes were inspected, but multiple strata are "
            "compared; a single positive stratum is not evidence of a regime",
            "do not train a conditional Adapter from this table",
        ],
    }
    out.joinpath("conditional_signal_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # per-position feature table (the scatter source for the required Track-D figure)
    with (out / "failure_structure_positions.csv").open("w", encoding="utf-8",
                                                        newline="") as f:
        w = csv.writer(f)
        w.writerow(list(rows[0].keys()))
        for r in sorted(rows, key=lambda x: (x["sample_id"], x["position"])):
            w.writerow([r[k] for k in rows[0].keys()])
    print(json.dumps({"rows": len(rows), "missing_features": missing,
                      "strata": len(table),
                      "candidates": [(c["bin_family"], c["bin"], round(c["dM_teacher_mean"], 4),
                                      round(c["teacher_minus_null_mean"], 4), c["n"])
                                     for c in candidates],
                      "status": analysis["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
