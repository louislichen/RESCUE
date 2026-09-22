"""Aggregate the decision-relevance diagnostic under the frozen V3 integrity contract.

Outputs (§15):
  decision_relevance_rows.jsonl        merged, verified per-position rows
  decision_relevance_summary.json      all headline numbers + §14 case
  teacher_vs_null_paired.csv           per-position paired teacher/null values
  prior_distance_vs_decision_value.csv prior-space distance vs decision movement
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dr_common import ROW_REQUIRED_FIELDS, bootstrap_ci, load_config, load_harness, spearman  # noqa: E402


def stats(vals):
    if not vals:
        return None
    s = bootstrap_ci(vals)
    s["min"] = min(vals)
    s["max"] = max(vals)
    s["stdev"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return s


def paired(hi, lo):
    return bootstrap_ci([h - l for h, l in zip(hi, lo)])


def group_rows(rows, key):
    return [r for r in rows if r["group"] == key]


def metrics_block(rs, label):
    if not rs:
        return {"label": label, "n": 0}
    return {
        "label": label, "n": len(rs),
        "positions_with_flip_C": sum(1 for r in rs if r["natural_top1_C"] != r["y_base"]),
        "positions_with_flip_D": sum(1 for r in rs if r["natural_top1_D"] != r["y_base"]),
        "dlogp_T_teacher": stats([r["dlogp_T_teacher"] for r in rs]),
        "dlogp_T_null": stats([r["dlogp_T_null"] for r in rs]),
        "dM_teacher": stats([r["dM_teacher"] for r in rs]),
        "dM_null": stats([r["dM_null"] for r in rs]),
        "teacher_minus_null_dM": paired([r["dM_teacher"] for r in rs],
                                        [r["dM_null"] for r in rs]),
        "teacher_minus_null_dlogp": paired([r["dlogp_T_teacher"] for r in rs],
                                           [r["dlogp_T_null"] for r in rs]),
        "cos_teacher": stats([r["cos_teacher"] for r in rs]),
        "cos_null": stats([r["cos_null"] for r in rs]),
        "teacher_minus_null_cos": paired([r["cos_teacher"] for r in rs],
                                         [r["cos_null"] for r in rs]),
        "cos_teacher_topM": stats([r["cos_teacher_topM"] for r in rs]),
        "cos_null_topM": stats([r["cos_null_topM"] for r in rs]),
        "dz_teacher_l2": stats([r["dz_teacher_l2"] for r in rs]),
        "dz_null_l2": stats([r["dz_null_l2"] for r in rs]),
        "M_A_mean": statistics.fmean([r["M_A"] for r in rs]),
    }


def classify(summary):
    """§14 interpretation matrix, evaluated on the S!=T subset (the informative positions)."""
    snt = summary["groups"]["S!=T"]
    if snt["n"] == 0:
        return {"case": "UNDETERMINED", "reason": "no S!=T positions"}
    dM = snt["dM_teacher"]
    paired_dm = snt["teacher_minus_null_dM"]
    cos_t = snt["cos_teacher"]
    flip = snt["positions_with_flip_C"] / snt["n"]
    crit = {
        "teacher_dM_positive_CI": dM["ci95_low"] > 0,
        "teacher_dM_negative_CI": dM["ci95_high"] < 0,
        "teacher_beats_null_CI": paired_dm["ci95_low"] > 0,
        "teacher_vs_null_equivalent": (paired_dm["ci95_low"] <= 0 <= paired_dm["ci95_high"]),
        "teacher_sign_random": abs(dM["positive_fraction"] - 0.5) <= 0.05,
        "cos_teacher_positive_CI": cos_t["ci95_low"] > 0,
        "top1_flip_rate": flip,
        "no_top1_flip": flip < 0.01,
    }
    eq = summary["groups"]["S=T"]
    if crit["teacher_dM_negative_CI"]:
        case = "CASE INFO-C"
    elif crit["teacher_dM_positive_CI"] and crit["teacher_beats_null_CI"] and crit["no_top1_flip"]:
        case = "CASE INFO-A"
    elif crit["teacher_dM_positive_CI"] and (eq["n"] == 0 or eq["dM_teacher"]["ci95_low"] <= 0):
        case = "CASE INFO-D"
    elif crit["teacher_vs_null_equivalent"] or crit["teacher_sign_random"]:
        case = "CASE INFO-B"
    else:
        case = "UNDETERMINED"
    return {"case": case, "criteria": crit, "evaluated_on": "S!=T subset",
            "note": "criteria are stated explicitly so a reviewer can re-derive the case"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parent-qa", default="")
    args = ap.parse_args()
    out = Path(args.out)
    cfg = load_config(args.config)
    H = load_harness(args.harness_root)
    dataset = json.loads((out / "dataset_identity.json").read_text(encoding="utf-8"))
    planned_ids = [x["canonical_source_id"] for x in dataset["ordered_sample_ids"]]

    integrity = {"shards": [], "rejected": []}
    rows: list[dict] = []
    for path in sorted(out.glob("dr_shard_*.jsonl")):
        v = H.verify_result(path, expected_config_hash=cfg["config_hash"],
                            expected_checkpoint_sha256=cfg["checkpoint_sha256"])
        entry = {"path": str(path), "verify": v}
        integrity["shards"].append(entry)
        if v["status"] != "PASS":
            integrity["rejected"].append(entry)
            continue
        recs = H.read_result_records(path)
        entry["records"] = len(recs)
        rows.extend(recs)

    # ---- denominator (§16) ----
    malformed = [i for i, r in enumerate(rows)
                 if not isinstance(r, dict) or any(f not in r for f in ROW_REQUIRED_FIELDS)]
    good = [r for i, r in enumerate(rows) if i not in set(malformed)]
    seen = {}
    dupes = []
    for r in good:
        key = (r["canonical_source_id"], r["position"])
        if key in seen:
            dupes.append(key)
        seen[key] = r
    covered = sorted({r["canonical_source_id"] for r in good})
    missing = sorted(set(planned_ids) - set(covered))
    denominator = {
        "planned_N": len(planned_ids),
        "attempted_rows": len(rows),
        "valid_rows": len(good),
        "excluded_rows": len(malformed) + len(dupes),
        "malformed_rows": len(malformed),
        "duplicate_sample_position_pairs": len(dupes),
        "samples_covered": len(covered),
        "samples_missing": len(missing),
        "missing_sample_ids": missing[:20],
        "positions_per_sample_min": min((sum(1 for r in good
                                             if r["canonical_source_id"] == s)
                                         for s in covered), default=0),
        "positions_per_sample_max": max((sum(1 for r in good
                                             if r["canonical_source_id"] == s)
                                         for s in covered), default=0),
        "status": ("PASS" if (not malformed and not dupes and not missing and good)
                   else "FAIL_DENOMINATOR"),
    }

    # ---- reproducibility cross-check against the parent run's BASE trajectories ----
    repro = {"checked": False}
    if args.parent_qa and Path(args.parent_qa).exists():
        parent_A = {}
        for line in Path(args.parent_qa).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("condition") == "A_base":
                parent_A[r["canonical_source_id"]] = list(map(int, r["tokens"]))
        same = diff = 0
        diffs = []
        for sid, toks in parent_A.items():
            mine = next((r["base_tokens"] for r in good if r["canonical_source_id"] == sid), None)
            if mine is None:
                continue
            if mine == toks:
                same += 1
            else:
                diff += 1
                diffs.append(sid)
        repro = {"checked": True, "parent_arm_A_samples": len(parent_A),
                 "identical": same, "different": diff, "different_ids": diffs[:10],
                 "note": "this run's BASE trajectory must reproduce the parent run's arm A"}
    total = metrics_block(good, "all positions")
    by_group = {g: metrics_block(group_rows(good, g), f"group {g}") for g in ("S=T", "S!=T")}

    lin = [r["linearized_scale_to_boundary_teacher"] for r in good
           if r.get("linearized_scale_to_boundary_teacher") is not None]
    linearized = {"available_positions": len(lin), "label": "LINEARIZED_DIAGNOSTIC_ONLY",
                  "distribution": stats(lin) if lin else None,
                  "caveat": "a linear extrapolation of the current intervention; it is NOT a "
                            "claim that scaling alpha by this factor flips the decision"}

    # ---- prior-space distance vs decision value (§12) ----
    pd_rows = [r for r in good if r.get("prior_distance")]
    prior_block = {"positions_with_prior_distance": len(pd_rows)}
    if pd_rows:
        t_l1 = [r["prior_distance"]["teacher_l1"] for r in pd_rows]
        n_l1 = [r["prior_distance"]["null_l1"] for r in pd_rows]
        t_jsd = [r["prior_distance"]["teacher_jsd"] for r in pd_rows]
        n_jsd = [r["prior_distance"]["null_jsd"] for r in pd_rows]
        dM_t = [r["dM_teacher"] for r in pd_rows]
        dM_n = [r["dM_null"] for r in pd_rows]
        dlogp_t = [r["dlogp_T_teacher"] for r in pd_rows]
        prior_block.update({
            "teacher_l1": stats(t_l1), "null_l1": stats(n_l1),
            "teacher_jsd": stats(t_jsd), "null_jsd": stats(n_jsd),
            "teacher_minus_null_l1": paired(t_l1, n_l1),
            "teacher_minus_null_jsd": paired(t_jsd, n_jsd),
            "spearman_teacher_l1_vs_dM_teacher": spearman(t_l1, dM_t),
            "spearman_teacher_l1_vs_dlogp_teacher": spearman(t_l1, dlogp_t),
            "spearman_null_l1_vs_dM_null": spearman(n_l1, dM_n),
            "spearman_teacher_jsd_vs_dM_teacher": spearman(t_jsd, dM_t),
            "interpretation": "association only; never causation",
        })
        snt = [r for r in pd_rows if r["group"] == "S!=T"]
        if snt:
            prior_block["S!=T_spearman_teacher_l1_vs_dM_teacher"] = spearman(
                [r["prior_distance"]["teacher_l1"] for r in snt],
                [r["dM_teacher"] for r in snt])

    gold_rows = [r for r in good if r.get("gold", {}).get("available")]
    gold_block = {"available_positions": len(gold_rows)}
    if len(gold_rows) >= 20:
        gold_block.update({
            "dlogp_gold_teacher": stats([r["gold"]["dlogp_teacher"] for r in gold_rows]),
            "dlogp_gold_null": stats([r["gold"]["dlogp_null"] for r in gold_rows]),
            "dM_gold_teacher": stats([r["gold"]["dM_teacher"] for r in gold_rows]),
            "dM_gold_null": stats([r["gold"]["dM_null"] for r in gold_rows]),
            "note": "Gold-directed and Target-directed diagnostics are reported separately "
                    "and must not be merged into one metric"})
    else:
        gold_block["status"] = "NOT_AVAILABLE"
        gold_block["note"] = ("strict gold mapping requires the generated prefix to be a "
                              "token-prefix of the gold answer; too few such positions")

    context = {"prompt_len": stats([float(r["prefix_len"]) for r in good])}
    longpos = [r for r in good if r["prefix_len"] > 1500]
    context["long_context_positions"] = len(longpos)
    if longpos:
        context["long_context_dM_teacher"] = stats([r["dM_teacher"] for r in longpos])
        context["long_context_dM_null"] = stats([r["dM_null"] for r in longpos])

    summary = {
        "task_id": "RESCUE-TEACHER-PRIOR-DECISION-RELEVANCE-DIAGNOSTIC-001",
        "status_as_received": cfg["status_as_received"],
        "harness_manifest_sha256": cfg["harness_manifest_sha256"],
        "checkpoint_sha256": cfg["checkpoint_sha256"],
        "config_hash": cfg["config_hash"],
        "parent_task": cfg["parent_task"],
        "parent_primary_case": cfg["parent_primary_case"],
        "denominator": denominator,
        "integrity": integrity,
        "dataset_identity": dataset,
        "base_trajectory_reproducibility": repro,
        "overall": total,
        "groups": by_group,
        "linearized_boundary_distance": linearized,
        "prior_space": prior_block,
        "gold_directed": gold_block,
        "context": context,
        "claim_boundary": cfg["claim_boundary"],
    }
    summary["case"] = classify(summary)

    H.write_result_envelope(
        out / "decision_relevance_rows.jsonl",
        task_id="RESCUE-TEACHER-PRIOR-DECISION-RELEVANCE-DIAGNOSTIC-001",
        stage="decision_relevance_merged", records=good,
        arms=["A_base", "C_full_window_teacher", "D_full_window_null", "T_target"],
        sample_ids=covered, config_hash=cfg["config_hash"],
        checkpoint_sha256=cfg["checkpoint_sha256"])

    with (out / "teacher_vs_null_paired.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "source_index", "position", "group", "prefix_len",
                    "M_A", "dM_teacher", "dM_null", "teacher_minus_null_dM",
                    "dlogp_T_teacher", "dlogp_T_null", "teacher_minus_null_dlogp",
                    "cos_teacher", "cos_null", "dz_teacher_l2", "dz_null_l2",
                    "flip_C", "flip_D", "linearized_scale_teacher"])
        for r in sorted(good, key=lambda x: (x["source_index"], x["position"])):
            w.writerow([r["canonical_source_id"], r["source_index"], r["position"], r["group"],
                        r["prefix_len"], round(r["M_A"], 6), round(r["dM_teacher"], 6),
                        round(r["dM_null"], 6), round(r["dM_teacher"] - r["dM_null"], 6),
                        round(r["dlogp_T_teacher"], 6), round(r["dlogp_T_null"], 6),
                        round(r["dlogp_T_teacher"] - r["dlogp_T_null"], 6),
                        round(r["cos_teacher"], 6), round(r["cos_null"], 6),
                        round(r["dz_teacher_l2"], 6), round(r["dz_null_l2"], 6),
                        int(r["natural_top1_C"] != r["y_base"]),
                        int(r["natural_top1_D"] != r["y_base"]),
                        "" if r["linearized_scale_to_boundary_teacher"] is None
                        else round(r["linearized_scale_to_boundary_teacher"], 4)])

    with (out / "prior_distance_vs_decision_value.csv").open("w", encoding="utf-8",
                                                             newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "position", "group", "prefix_len",
                    "teacher_l1", "teacher_jsd", "null_l1", "null_jsd",
                    "dM_teacher", "dM_null", "dlogp_T_teacher", "dlogp_T_null"])
        for r in sorted(pd_rows, key=lambda x: (x["source_index"], x["position"])):
            pd = r["prior_distance"]
            w.writerow([r["canonical_source_id"], r["position"], r["group"], r["prefix_len"],
                        round(pd["teacher_l1"], 6), round(pd["teacher_jsd"], 6),
                        round(pd["null_l1"], 6), round(pd["null_jsd"], 6),
                        round(r["dM_teacher"], 6), round(r["dM_null"], 6),
                        round(r["dlogp_T_teacher"], 6), round(r["dlogp_T_null"], 6)])

    H.atomic_write_text(out / "decision_relevance_summary.json",
                        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "denominator": {k: denominator[k] for k in
                        ("planned_N", "attempted_rows", "valid_rows", "excluded_rows",
                         "samples_covered", "samples_missing", "status")},
        "reproducibility": {k: repro.get(k) for k in ("identical", "different")},
        "S!=T positions": by_group["S!=T"]["n"], "S=T positions": by_group["S=T"]["n"],
        "S!=T dM_teacher": {k: round(by_group["S!=T"]["dM_teacher"][k], 6)
                            for k in ("mean", "ci95_low", "ci95_high", "positive_fraction")},
        "S!=T dM_null": {k: round(by_group["S!=T"]["dM_null"][k], 6)
                         for k in ("mean", "ci95_low", "ci95_high", "positive_fraction")},
        "S!=T teacher-minus-null dM": {k: round(by_group["S!=T"]["teacher_minus_null_dM"][k], 6)
                                       for k in ("mean", "ci95_low", "ci95_high")},
        "top1 flips (C)": by_group["S!=T"]["positions_with_flip_C"],
        "case": summary["case"]["case"],
    }, ensure_ascii=False, indent=2))
    ok = denominator["status"] == "PASS"
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
