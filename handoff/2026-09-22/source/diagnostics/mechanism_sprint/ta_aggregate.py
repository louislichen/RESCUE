"""TRACK A aggregation — student controllability summary, oracle results, §A9 case."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dr_common import bootstrap_ci, load_config, load_harness, spearman  # noqa: E402


def stats(vals):
    if not vals:
        return None
    s = bootstrap_ci(vals)
    s["min"] = min(vals)
    s["max"] = max(vals)
    s["stdev"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return s


def block(rows, label):
    if not rows:
        return {"label": label, "n": 0}
    eff = [r["teacher_efficiency"] for r in rows if r["teacher_efficiency"] is not None]
    neff = [r["null_efficiency"] for r in rows if r["null_efficiency"] is not None]
    closure = []
    for r in rows:
        deficit = -float(r["M_base"])
        if deficit > 1e-9:
            closure.append(float(r["dM_oracle_actual"]) / deficit)
    return {
        "label": label, "n": len(rows),
        "V_teacher": stats([r["V_teacher"] for r in rows]),
        "V_null": stats([r["V_null"] for r in rows]),
        "V_oracle": stats([r["V_oracle"] for r in rows]),
        "fraction_V_oracle_positive": sum(1 for r in rows if r["V_oracle"] > 0) / len(rows),
        "fraction_V_teacher_positive": sum(1 for r in rows if r["V_teacher"] > 0) / len(rows),
        "fraction_V_null_positive": sum(1 for r in rows if r["V_null"] > 0) / len(rows),
        "teacher_efficiency": stats(eff),
        "null_efficiency": stats(neff),
        "teacher_minus_null_efficiency": (
            bootstrap_ci([a - b for a, b in zip(
                [r["teacher_efficiency"] for r in rows if r["teacher_efficiency"] is not None],
                [r["null_efficiency"] for r in rows if r["null_efficiency"] is not None])])
            if eff and neff and len(eff) == len(neff) else None),
        "M_base": stats([float(r["M_base"]) for r in rows]),
        "dM_oracle_actual": stats([float(r["dM_oracle_actual"]) for r in rows]),
        "margin_closure_fraction": stats(closure),
        "dlogp_target_oracle_actual": stats(
            [float(r["dlogp_target_oracle_actual"]) for r in rows]),
        "oracle_top1_flip_rate": (sum(1 for r in rows if r["top1_flip_under_oracle"])
                                  / len(rows)),
        "oracle_reaches_target_top1_rate": (sum(1 for r in rows
                                                if r["target_token_is_top1_under_oracle"])
                                            / len(rows)),
        "teacher_l1_total": stats([float(r["teacher_l1_total"]) for r in rows]),
        "oracle_l1_total": stats([float(r["oracle_l1_total"]) for r in rows]),
        "budget_ratio_oracle_over_teacher": stats(
            [float(r["oracle_l1_total"]) / float(r["teacher_l1_total"])
             for r in rows if float(r["teacher_l1_total"]) > 0]),
        "reconstruction_max_abs_logit_delta": stats(
            [float(r["forward_framing"][
                "reconstruction_vs_full_forward_max_abs_logit_delta"]) for r in rows]),
        "M_base_alt_minus_parent": stats(
            [float(r["M_base_alt_minus_parent"]) for r in rows
             if r.get("M_base_alt_minus_parent") is not None]),
        "grad_connected_all_layers": all(all(r["grad_connected_per_layer"].values())
                                         for r in rows),
    }


def classify(snt):
    """§A9 Track-A decision."""
    if snt["n"] == 0:
        return {"decision": "UNDETERMINED", "reason": "no S!=T positions"}
    eff = snt["teacher_efficiency"]
    clo = snt["margin_closure_fraction"]
    dM = snt["dM_oracle_actual"]
    crit = {
        "oracle_improves_margin_CI": dM["ci95_low"] > 0,
        "oracle_median_closure": clo["median"] if clo else None,
        "oracle_closure_at_least_half": bool(clo and clo["median"] >= 0.5),
        "oracle_flip_rate": snt["oracle_top1_flip_rate"],
        "teacher_efficiency_median": eff["median"] if eff else None,
        "teacher_efficiency_CI_contains_zero": bool(eff and eff["ci95_low"] <= 0
                                                    <= eff["ci95_high"]),
        "teacher_efficiency_small": bool(eff and eff["median"] <= 0.05),
        "budget_matched": bool(snt["budget_ratio_oracle_over_teacher"]
                               and abs(snt["budget_ratio_oracle_over_teacher"]["mean"] - 1.0)
                               < 0.01),
    }
    if not crit["oracle_improves_margin_CI"] or not crit["oracle_closure_at_least_half"]:
        dec = "CONTROLLABILITY-B"
    elif crit["teacher_efficiency_small"] or crit["teacher_efficiency_CI_contains_zero"]:
        dec = "CONTROLLABILITY-A"
    else:
        dec = "CONTROLLABILITY-C"
    return {"decision": dec, "criteria": crit,
            "note": "criteria are stated explicitly so a reviewer can re-derive the decision"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dr-rows", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    cfg = load_config(args.config)
    H = load_harness(args.harness_root)

    planned = []
    for line in Path(args.dr_rows).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "__result_header__" not in r:
            planned.append((r["canonical_source_id"], r["position"], r["group"]))

    integrity = {"shards": [], "rejected": []}
    rows = []
    for path in sorted(out.glob("ta_shard_*.jsonl")):
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

    malformed = [r for r in rows if "M_base" not in r or "V_oracle" not in r]
    good = [r for r in rows if "M_base" in r and "V_oracle" in r]
    seen, dupes = set(), []
    for r in good:
        k = (r["canonical_source_id"], r["position"])
        if k in seen:
            dupes.append(k)
        seen.add(k)
    planned_keys = {(s, p) for s, p, _ in planned}
    missing = sorted(planned_keys - seen)
    denominator = {
        "planned_positions": len(planned),
        "attempted_rows": len(rows),
        "valid_rows": len(good),
        "excluded_rows": len(malformed) + len(dupes),
        "malformed_rows": len(malformed),
        "duplicate_pairs": len(dupes),
        "missing_planned_positions": len(missing),
        "status": ("PASS" if not malformed and not dupes and not missing and good
                   else "FAIL_DENOMINATOR"),
    }
    snt = [r for r in good if r["group"] == "S!=T"]
    seq = [r for r in good if r["group"] == "S=T"]
    summary = {
        "task_id": "RESCUE-GUIDANCE-REPRESENTATION-MECHANISM-SPRINT-001",
        "track": "A — Student injection-site local controllability",
        "config_hash": cfg["config_hash"],
        "checkpoint_sha256": cfg["checkpoint_sha256"],
        "margin_definition": "M = z_S[y_T] - z_S[y_S]  (§A2)",
        "oracle": "LOCAL_DECISION_ORACLE_DIAGNOSTIC — deterministic greedy mass transfer, "
                  "per-head L1 budget matched to the Teacher direction (§A5/A6)",
        "denominator": denominator,
        "integrity": integrity,
        "primary_population": block(snt, "S!=T (primary)"),
        "secondary_population": block(seq, "S=T (secondary reference)"),
        "degeneracy_note": ("On S=T positions y_T == y_S, so the §A2 margin is identically 0 and "
                            "every gradient / first-order value there is 0 by definition; only "
                            "the direction magnitudes are informative on that subset."),
        "gate": classify(block(snt, "S!=T (primary)")),
        "claim_boundary": cfg["claim_boundary"],
    }
    H.write_result_envelope(
        out / "student_controllability_rows.jsonl",
        task_id="RESCUE-GUIDANCE-REPRESENTATION-MECHANISM-SPRINT-001",
        stage="student_controllability_merged", records=good, arms=["decision_position"],
        sample_ids=sorted({r["canonical_source_id"] for r in good}),
        config_hash=cfg["config_hash"], checkpoint_sha256=cfg["checkpoint_sha256"])
    with (out / "oracle_same_budget_results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "source_index", "position", "group", "prefix_len",
                    "M_base", "M_oracle_actual", "dM_oracle_actual", "margin_closure",
                    "dlogp_target_oracle", "top1_flip", "reaches_target_top1",
                    "V_teacher", "V_null", "V_oracle", "teacher_efficiency",
                    "null_efficiency", "teacher_l1_total", "oracle_l1_total",
                    "budget_ratio", "recon_delta"])
        for r in sorted(good, key=lambda x: (x["source_index"], x["position"])):
            deficit = -float(r["M_base"])
            w.writerow([
                r["canonical_source_id"], r["source_index"], r["position"], r["group"],
                r["prefix_len"], round(float(r["M_base"]), 6),
                round(float(r["M_oracle_actual"]), 6), round(float(r["dM_oracle_actual"]), 6),
                round(float(r["dM_oracle_actual"]) / deficit, 6) if deficit > 1e-9 else "",
                round(float(r["dlogp_target_oracle_actual"]), 6),
                int(r["top1_flip_under_oracle"]), int(r["target_token_is_top1_under_oracle"]),
                round(r["V_teacher"], 6), round(r["V_null"], 6), round(r["V_oracle"], 6),
                "" if r["teacher_efficiency"] is None else round(r["teacher_efficiency"], 6),
                "" if r["null_efficiency"] is None else round(r["null_efficiency"], 6),
                round(float(r["teacher_l1_total"]), 6), round(float(r["oracle_l1_total"]), 6),
                round(float(r["oracle_l1_total"]) / float(r["teacher_l1_total"]), 6)
                if float(r["teacher_l1_total"]) > 0 else "",
                round(float(r["forward_framing"][
                    "reconstruction_vs_full_forward_max_abs_logit_delta"]), 6)])
    H.atomic_write_text(out / "student_controllability_summary.json",
                        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    b = summary["primary_population"]
    print(json.dumps({
        "denominator": denominator,
        "S!=T n": b["n"],
        "M_base": {k: round(b["M_base"][k], 4) for k in ("mean", "median")},
        "dM_oracle_actual": {k: round(b["dM_oracle_actual"][k], 4)
                             for k in ("mean", "median", "ci95_low", "ci95_high")},
        "margin_closure_median": round(b["margin_closure_fraction"]["median"], 4),
        "oracle_top1_flip_rate": b["oracle_top1_flip_rate"],
        "V_oracle": {k: round(b["V_oracle"][k], 5)
                     for k in ("mean", "median", "ci95_low", "ci95_high")},
        "V_teacher": {k: round(b["V_teacher"][k], 5)
                      for k in ("mean", "median", "ci95_low", "ci95_high")},
        "teacher_efficiency": {k: round(b["teacher_efficiency"][k], 5)
                               for k in ("mean", "median", "ci95_low", "ci95_high")},
        "null_efficiency": {k: round(b["null_efficiency"][k], 5)
                            for k in ("mean", "median", "ci95_low", "ci95_high")},
        "budget_ratio": round(b["budget_ratio_oracle_over_teacher"]["mean"], 6),
        "decision": summary["gate"]["decision"],
    }, ensure_ascii=False, indent=2))
    return 0 if denominator["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
