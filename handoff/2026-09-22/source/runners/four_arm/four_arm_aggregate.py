"""Aggregate the four-arm QA + K4 shards under the frozen V3 integrity contract.

Rejects (never silently repairs): missing completion marker, stale/body-sha mismatch,
config/checkpoint mismatch, duplicate sample x arm, missing arm, malformed record.
Reports attempted / valid / excluded per stage (§16) and per arm.

Usage:
  four_arm_aggregate.py --harness-root H --config CFG --out OUTDIR
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from four_arm_common import (ARMS, K4_REQUIRED_FIELDS, QA_REQUIRED_FIELDS,  # noqa: E402
                            load_config, load_harness)

BOOT_N = 10000
BOOT_SEED = 20260830
PRIMARY = ("A_base", "B_prefill_only_teacher", "C_full_window_teacher",
           "D_full_window_null")
PAIRS = (("B−A", "B_prefill_only_teacher", "A_base"),
         ("C−B", "C_full_window_teacher", "B_prefill_only_teacher"),
         ("C−D", "C_full_window_teacher", "D_full_window_null"),
         ("C−A", "C_full_window_teacher", "A_base"),
         ("D−A", "D_full_window_null", "A_base"))


def boot_ci(values: list[float], n: int = BOOT_N, seed: int = BOOT_SEED):
    """Paired bootstrap over samples — a DEVELOPMENT diagnostic, not confirmation."""
    if not values:
        return None
    rng = random.Random(seed)
    k = len(values)
    means = []
    for _ in range(n):
        means.append(sum(values[rng.randrange(k)] for _ in range(k)) / k)
    means.sort()
    return {"mean": statistics.fmean(values),
            "median": statistics.median(values),
            "ci95_low": means[int(0.025 * n)],
            "ci95_high": means[int(0.975 * n) - 1],
            "boot_n": n, "label": "DEVELOPMENT diagnostic (paired bootstrap), not confirmation"}


def by_id(records, arm):
    return {r["canonical_source_id"]: r for r in records if r["condition"] == arm}


def qa_pairwise(records, ids):
    out = {}
    for name, hi, lo in PAIRS:
        h, l = by_id(records, hi), by_id(records, lo)
        common = [i for i in ids if i in h and i in l]
        identical = [i for i in common if h[i]["tokens"] == l[i]["tokens"]]
        changed = [i for i in common if h[i]["tokens"] != l[i]["tokens"]]
        div: dict[str, int] = {}
        for i in changed:
            d = h[i]["first_divergence_from_base"] if lo == "A_base" else None
            if d is None:
                ht, lt = h[i]["tokens"], l[i]["tokens"]
                d = next((j for j, (a, b) in enumerate(zip(ht, lt)) if a != b),
                         min(len(ht), len(lt)))
            div[str(d)] = div.get(str(d), 0) + 1
        better = [i for i in common if h[i]["em"] == 1 and l[i]["em"] == 0]
        worse = [i for i in common if h[i]["em"] == 0 and l[i]["em"] == 1]
        both_ok = [i for i in common if h[i]["em"] == 1 and l[i]["em"] == 1]
        both_bad = [i for i in common if h[i]["em"] == 0 and l[i]["em"] == 0]
        f1_delta = [h[i]["f1"] - l[i]["f1"] for i in common]
        len_delta = [h[i]["n_tokens"] - l[i]["n_tokens"] for i in common]
        out[name] = {
            "arm_high": hi, "arm_low": lo, "paired_n": len(common),
            "identical_token_sequence_count": len(identical),
            "changed_sequence_count": len(changed),
            "first_divergence_position_distribution": dict(
                sorted(div.items(), key=lambda kv: int(kv[0]))),
            "em_transitions": {"better": len(better), "worse": len(worse),
                               "both_correct": len(both_ok), "both_wrong": len(both_bad),
                               "net_better": len(better) - len(worse)},
            "em_rate_high": statistics.fmean([h[i]["em"] for i in common]) if common else None,
            "em_rate_low": statistics.fmean([l[i]["em"] for i in common]) if common else None,
            "f1_delta_per_sample": boot_ci(f1_delta),
            "generated_length_delta_per_sample": boot_ci(len_delta),
            "better_ids": sorted(better)[:20], "worse_ids": sorted(worse)[:20],
        }
    return out


def k4_pairwise(records, ids):
    out = {}
    for name, hi, lo in PAIRS:
        h, l = by_id(records, hi), by_id(records, lo)
        common = [i for i in ids if i in h and i in l]
        acc_delta, rate_delta, full_delta, rej_delta = [], [], [], []
        better = same = worse = 0
        for i in common:
            hh, ll = h[i], l[i]
            acc_delta.append(hh["accepted_draft_tokens"] - ll["accepted_draft_tokens"])
            rh = (hh["accepted_draft_tokens"] / hh["proposed_draft_tokens"]
                  if hh["proposed_draft_tokens"] else 0.0)
            rl = (ll["accepted_draft_tokens"] / ll["proposed_draft_tokens"]
                  if ll["proposed_draft_tokens"] else 0.0)
            rate_delta.append(rh - rl)
            full_delta.append(hh["full_accept_blocks"] - ll["full_accept_blocks"])
            rej_delta.append(hh["reject_blocks"] - ll["reject_blocks"])
            d = hh["accepted_draft_tokens"] - ll["accepted_draft_tokens"]
            better += d > 0
            worse += d < 0
            same += d == 0
        out[name] = {
            "arm_high": hi, "arm_low": lo, "paired_n": len(common),
            "accepted_tokens_delta_per_sample": boot_ci(acc_delta),
            "acceptance_rate_delta_per_sample": boot_ci(rate_delta),
            "full_k_blocks_delta_per_sample": boot_ci(full_delta),
            "reject_blocks_delta_per_sample": boot_ci(rej_delta),
            "per_sample_direction": {"improved": better, "unchanged": same, "worsened": worse},
        }
    return out


def arm_summary_qa(records, arm):
    rs = [r for r in records if r["condition"] == arm]
    if not rs:
        return {"valid_N": 0}
    return {
        "valid_N": len(rs),
        "EM": statistics.fmean([r["em"] for r in rs]),
        "F1": statistics.fmean([r["f1"] for r in rs]),
        "avg_generated_length": statistics.fmean([r["n_tokens"] for r in rs]),
        "zero_length_count": sum(1 for r in rs if r["n_tokens"] == 0),
        "prefill_mix_calls_mean": statistics.fmean([r["interventions"]["prefill_mix_calls"]
                                                    for r in rs]),
        "decode_mix_calls_mean": statistics.fmean([r["interventions"]["decode_mix_calls"]
                                                   for r in rs]),
        "samples_identical_to_base": sum(1 for r in rs if r.get("identical_to_base") is True),
    }


def arm_summary_k4(records, arm):
    rs = [r for r in records if r["condition"] == arm]
    if not rs:
        return {"valid_N": 0}
    proposed = sum(r["proposed_draft_tokens"] for r in rs)
    accepted = sum(r["accepted_draft_tokens"] for r in rs)
    blocks = sum(r["blocks"] for r in rs)
    per_sample_rate = [(r["accepted_draft_tokens"] / r["proposed_draft_tokens"]
                        if r["proposed_draft_tokens"] else 0.0) for r in rs]
    per_sample_mean_accepted = [(r["accepted_draft_tokens"] / r["blocks"]
                                 if r["blocks"] else 0.0) for r in rs]
    return {
        "valid_N": len(rs),
        "pooled": {"proposed": proposed, "accepted": accepted,
                   "acceptance_ratio": (accepted / proposed) if proposed else None,
                   "blocks": blocks},
        "acceptance_rate_per_sample_mean": statistics.fmean(per_sample_rate),
        "acceptance_rate_per_sample_median": statistics.median(per_sample_rate),
        "mean_accepted_tokens_per_round": statistics.fmean(per_sample_mean_accepted),
        "rejection_blocks": sum(r["reject_blocks"] for r in rs),
        "full_k_blocks": sum(r["full_accept_blocks"] for r in rs),
        "zero_accept_blocks": sum(r["zero_accept_blocks"] for r in rs),
        "partial_accept_blocks": sum(r["partial_accept_blocks"] for r in rs),
        "short_proposals": sum(r["short_proposals"] for r in rs),
        "eos_terminated": sum(1 for r in rs if r["eos_terminated"]),
        "correct_rows": sum(1 for r in rs if r["matches_target_only_greedy"]),
        "prefill_mix_calls_mean": statistics.fmean([r["draft_prefill_mix_calls"] for r in rs]),
        "decode_mix_calls_mean": statistics.fmean([r["draft_decode_mix_calls"] for r in rs]),
        "mean_accepted_tokens_per_sample": statistics.fmean(
            [r["accepted_draft_tokens"] for r in rs]),
        "mean_wall_seconds": statistics.fmean([r["wall_seconds"] for r in rs]),
    }


def classify_case(k4_pairs, qa_pairs):
    """§19 interpretation matrix, evaluated on paired per-sample behavior."""
    def reading(pairs, key):
        cb = pairs["C−B"][key]
        cd = pairs["C−D"][key]
        ca = pairs["C−A"][key]
        return {"C_vs_B": {"mean": cb["mean"], "ci": [cb["ci95_low"], cb["ci95_high"]],
                           "positive": cb["ci95_low"] > 0, "null_in_ci":
                           cb["ci95_low"] <= 0 <= cb["ci95_high"]},
                "C_vs_D": {"mean": cd["mean"], "ci": [cd["ci95_low"], cd["ci95_high"]],
                           "positive": cd["ci95_low"] > 0, "null_in_ci":
                           cd["ci95_low"] <= 0 <= cd["ci95_high"]},
                "C_vs_A": {"mean": ca["mean"], "ci": [ca["ci95_low"], ca["ci95_high"]],
                           "null_in_ci": ca["ci95_low"] <= 0 <= ca["ci95_high"]}}
    r = reading(k4_pairs, "accepted_tokens_delta_per_sample")
    q = reading(qa_pairs, "f1_delta_per_sample")
    c_b, c_d, c_a = r["C_vs_B"], r["C_vs_D"], r["C_vs_A"]
    if c_b["positive"] and c_d["positive"]:
        case = "CASE 1"
    elif c_b["positive"] and c_d["null_in_ci"]:
        case = "CASE 2"
    elif c_b["null_in_ci"] and c_d["null_in_ci"] and c_a["null_in_ci"]:
        case = "CASE 3"
    elif not c_b["positive"] and c_d["positive"]:
        case = "CASE 4"
    else:
        case = "UNDETERMINED_BY_CI"
    return {"criterion": "paired per-sample accepted draft tokens, paired bootstrap 95% CI "
                         "(DEVELOPMENT diagnostic); 'C > X' = CI lower bound > 0; "
                         "'C ≈ X' = 0 inside the CI",
            "metric": "k4.accepted_tokens_delta_per_sample", "reading": r,
            "qa_reading": q,
            "case": case}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    cfg = load_config(args.config)
    H = load_harness(args.harness_root)

    dataset = json.loads((out / "dataset_identity.json").read_text(encoding="utf-8"))
    planned_ids = [x["canonical_source_id"] for x in dataset["ordered_sample_ids"]]

    integrity = {"shards": [], "rejected": []}
    qa_records, k4_records = [], []
    for stage, pattern in (("qa", "qa_shard_*.jsonl"), ("k4", "k4_shard_*.jsonl")):
        for path in sorted(out.glob(pattern)):
            v = H.verify_result(path, expected_config_hash=cfg["config_hash"],
                                expected_checkpoint_sha256=cfg["checkpoint_sha256"])
            entry = {"stage": stage, "path": str(path), "verify": v}
            integrity["shards"].append(entry)
            if v["status"] != "PASS":
                integrity["rejected"].append(entry)
                continue
            recs = H.read_result_records(path)
            entry["records"] = len(recs)
            (qa_records if stage == "qa" else k4_records).extend(recs)

    denom = {}
    for stage, recs, fields in (("qa", qa_records, QA_REQUIRED_FIELDS),
                                ("k4", k4_records, K4_REQUIRED_FIELDS)):
        rep = H.audit_records(recs, arms=list(ARMS), required_fields=list(fields),
                              id_field="canonical_source_id", arm_field="condition",
                              planned_ids=planned_ids)
        rep["per_arm"] = {
            a: {"attempted": len({r["canonical_source_id"] for r in recs
                                  if r["condition"] == a}),
                "valid": len([i for i in planned_ids if i in by_id(recs, a)]),
                "excluded": len([i for i in planned_ids if i not in by_id(recs, a)])}
            for a in ARMS}
        rep["planned_N"] = len(planned_ids)
        denom[stage] = rep

    valid_ids = sorted(set(denom["qa"]["valid_sample_ids"]) & set(denom["k4"]["valid_sample_ids"]))
    qa_records = [r for r in qa_records if r["canonical_source_id"] in valid_ids]
    k4_records = [r for r in k4_records if r["canonical_source_id"] in valid_ids]

    qa_arms = {a: arm_summary_qa(qa_records, a) for a in ARMS}
    k4_arms = {a: arm_summary_k4(k4_records, a) for a in ARMS}
    qa_pairs = qa_pairwise(qa_records, valid_ids)
    k4_pairs = k4_pairwise(k4_records, valid_ids)

    correctness = {
        "contract": "speculative token ids == Target-only greedy token ids (incremental verify)",
        "per_arm": {a: {"rows": len(by_id(k4_records, a)),
                        "correct": sum(1 for r in k4_records
                                       if r["condition"] == a
                                       and r["matches_target_only_greedy"]),
                        "failures": [r["canonical_source_id"] for r in k4_records
                                     if r["condition"] == a
                                     and not r["matches_target_only_greedy"]]}
                    for a in ARMS},
        "verify_mode": "incremental", "k": 4,
    }
    correctness["status"] = ("K4_CORRECTNESS_PASS"
                             if all(v["failures"] == [] for v in correctness["per_arm"].values())
                             else "K4_CORRECTNESS_FAILURE")

    # merged, verified canonical ledgers
    H.write_result_envelope(out / "qa_four_arm.jsonl",
                            task_id="RESCUE-MINIMAL-FOUR-ARM-VALID-RERUN-001",
                            stage="qa_four_arm_merged", records=qa_records,
                            arms=list(ARMS), sample_ids=valid_ids,
                            config_hash=cfg["config_hash"],
                            checkpoint_sha256=cfg["checkpoint_sha256"])
    H.write_result_envelope(out / "k4_four_arm.jsonl",
                            task_id="RESCUE-MINIMAL-FOUR-ARM-VALID-RERUN-001",
                            stage="k4_four_arm_merged", records=k4_records,
                            arms=list(ARMS), sample_ids=valid_ids,
                            config_hash=cfg["config_hash"],
                            checkpoint_sha256=cfg["checkpoint_sha256"])

    with (out / "qa_paired_per_sample.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "source_index", "A_em", "B_em", "C_em", "D_em",
                    "A_f1", "B_f1", "C_f1", "D_f1", "A_len", "B_len", "C_len", "D_len",
                    "C_minus_B_f1", "C_minus_D_f1", "C_minus_B_len", "C_minus_D_len"])
        for i in valid_ids:
            row = [i]
            arms = {a: next((r for r in qa_records if r["canonical_source_id"] == i
                             and r["condition"] == a), None) for a in ARMS}
            row.append(arms["A_base"]["source_index"] if arms["A_base"] else "")
            row += [arms[a]["em"] if arms[a] else "" for a in ARMS]
            row += [round(arms[a]["f1"], 6) if arms[a] else "" for a in ARMS]
            row += [arms[a]["n_tokens"] if arms[a] else "" for a in ARMS]
            row += [round(arms["C_full_window_teacher"]["f1"] - arms["B_prefill_only_teacher"]["f1"], 6),
                    round(arms["C_full_window_teacher"]["f1"] - arms["D_full_window_null"]["f1"], 6),
                    arms["C_full_window_teacher"]["n_tokens"] - arms["B_prefill_only_teacher"]["n_tokens"],
                    arms["C_full_window_teacher"]["n_tokens"] - arms["D_full_window_null"]["n_tokens"]]
            w.writerow(row)

    with (out / "k4_paired_per_sample.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "source_index"] +
                   [f"{a}_{k}" for a in ARMS for k in
                    ("accepted", "proposed", "rate", "blocks", "full_k", "reject", "correct")] +
                   ["C_minus_B_accepted", "C_minus_D_accepted",
                    "C_minus_B_rate", "C_minus_D_rate"])
        for i in valid_ids:
            arms = {a: next((r for r in k4_records if r["canonical_source_id"] == i
                             and r["condition"] == a), None) for a in ARMS}
            if not all(arms.values()):
                continue
            row = [i, arms["A_base"]["source_index"]]
            for a in ARMS:
                r = arms[a]
                rate = (r["accepted_draft_tokens"] / r["proposed_draft_tokens"]
                        if r["proposed_draft_tokens"] else 0.0)
                row += [r["accepted_draft_tokens"], r["proposed_draft_tokens"], round(rate, 6),
                        r["blocks"], r["full_accept_blocks"], r["reject_blocks"],
                        int(r["matches_target_only_greedy"])]
            cb = arms["C_full_window_teacher"]
            bb = arms["B_prefill_only_teacher"]
            dd = arms["D_full_window_null"]
            row += [cb["accepted_draft_tokens"] - bb["accepted_draft_tokens"],
                    cb["accepted_draft_tokens"] - dd["accepted_draft_tokens"],
                    round((cb["accepted_draft_tokens"] / cb["proposed_draft_tokens"]
                           if cb["proposed_draft_tokens"] else 0.0)
                          - (bb["accepted_draft_tokens"] / bb["proposed_draft_tokens"]
                             if bb["proposed_draft_tokens"] else 0.0), 6),
                    round((cb["accepted_draft_tokens"] / cb["proposed_draft_tokens"]
                           if cb["proposed_draft_tokens"] else 0.0)
                          - (dd["accepted_draft_tokens"] / dd["proposed_draft_tokens"]
                             if dd["proposed_draft_tokens"] else 0.0), 6)]
            w.writerow(row)

    report = {
        "task_id": "RESCUE-MINIMAL-FOUR-ARM-VALID-RERUN-001",
        "harness_manifest_sha256": cfg["harness_manifest_sha256"],
        "checkpoint_sha256": cfg["checkpoint_sha256"],
        "config_hash": cfg["config_hash"],
        "dataset_identity": dataset,
        "arms": {a: cfg["arms"][a] for a in ARMS},
        "integrity": integrity,
        "denominator": denom,
        "valid_paired_sample_ids": valid_ids,
        "valid_paired_N": len(valid_ids),
        "qa": {"per_arm": qa_arms, "pairwise": qa_pairs},
        "k4": {"per_arm": k4_arms, "pairwise": k4_pairs},
        "correctness": correctness,
        "primary_case": classify_case(k4_pairs, qa_pairs),
        "claim_boundary": cfg["claim_boundary"],
    }
    H.atomic_write_text(out / "qa_four_arm_summary.json",
                        json.dumps({"per_arm": qa_arms, "pairwise": qa_pairs,
                                    "valid_paired_N": len(valid_ids),
                                    "denominator": denom["qa"]},
                                   ensure_ascii=False, indent=2) + "\n")
    H.atomic_write_text(out / "k4_four_arm_summary.json",
                        json.dumps({"per_arm": k4_arms, "pairwise": k4_pairs,
                                    "correctness": correctness,
                                    "valid_paired_N": len(valid_ids),
                                    "denominator": denom["k4"]},
                                   ensure_ascii=False, indent=2) + "\n")
    H.atomic_write_text(out / "four_arm_results.json",
                        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "valid_paired_N": len(valid_ids),
        "denominator": {s: {"attempted": denom[s]["attempted_N"],
                            "valid": denom[s]["valid_N"],
                            "excluded": denom[s]["excluded_N"],
                            "status": denom[s]["status"]} for s in denom},
        "qa_em": {a: qa_arms[a].get("EM") for a in ARMS},
        "qa_f1": {a: qa_arms[a].get("F1") for a in ARMS},
        "k4_ratio": {a: (k4_arms[a].get("pooled") or {}).get("acceptance_ratio")
                     for a in ARMS},
        "correctness": correctness["status"],
        "primary_case": report["primary_case"]["case"],
    }, ensure_ascii=False, indent=2))
    return 0 if (correctness["status"] == "K4_CORRECTNESS_PASS"
                 and all(denom[s]["status"] == "PASS" for s in denom)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
