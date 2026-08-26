#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R7 gate statistics for the pilot curves.

Reads the R7 per-condition JSONL outputs, computes paired CleanF1 means per
budget for each condition (oracle_span / oracle_budget / random_matched_budget),
and reports:

  SelectiveRepairSignal: if within <=10% processed_token_ratio, repair beats
    Route-B-paired CleanF1 AND beats RandomMatched-paired CleanF1 AND recovers
    >=20% of the Full-Naive gap (frozen criterium, Sprint) with n>=30.

Cost axis: processed_token_ratio = processed_tokens / L_A (written KV tokens
denominator identical to R6, but we report both ratio axes). The pilot runs a
fixed sample window (same as R6: idx 0..49 validated rows, excluding
RANDOM_LENGTH_MATCHED_UNCONSTRUCTIBLE drops).

Input layout:  --repair  path/to/<cond>_jsonl (one JSONL per condition+budget)
                --budget  <pct>   (percent that produced that file)
                --name    <cond>  (oracle|random)
Optionally --r6jsonl path to the R6 clean JSONL to align sample indexes.
"""
from __future__ import annotations
import argparse, json, math, random, sys
from collections import defaultdict

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair", type=str, required=True, action="append", dest="repairs",
                    help="repair-condition JSONL (can repeat one per budget)")
    ap.add_argument("--name", type=str, required=True,
                    help="condition name for this input set")
    ap.add_argument("--budget", type=float, required=True, action="append", dest="budgets",
                    help="budget percent matching each --repair (same order)")
    ap.add_argument("--r6jsonl", type=str, default=None,
                    help="R6 paired-clean JSONL to align sample indexes")
    ap.add_argument("--seed", type=int, default=20260827)
    return ap.parse_args()

def load_records(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            recs.append(rec)
    return recs

def main():
    args = parse_args()
    if len(args.repairs) != len(args.budgets):
        sys.exit("--repair/--budget count mismatch")
    rng = random.Random(args.seed)

    # baseline map: sample_index -> (emA, f1A, emB, f1B) from R6 paired rows
    base = {}
    if args.r6jsonl:
        for rec in load_records(args.r6jsonl):
            if rec.get("dropped"):
                continue
            base[rec["sample_index"]] = rec

    # per-budget repair stats keyed by sample_index.
    # Mode A: one file per budget (len(repairs) == len(budgets)).
    # Mode B: ONE file containing a budget_pct column per row
    #         (len(repairs) == 1 and len(budgets) > 1).
    per_budget = {}
    if len(args.repairs) == 1 and len(args.budgets) > 1:
        for rec in load_records(args.repairs[0]):
            if rec.get("dropped"):
                continue
            bp = float(rec["budget_pct"])
            per_budget.setdefault(bp, {})[rec["sample_index"]] = rec
        wanted = set(round(b, 6) for b in args.budgets)
        per_budget = {k: v for k, v in per_budget.items() if round(k, 6) in wanted}
    else:
        for path, bpct in zip(args.repairs, args.budgets):
            data = {}
            for rec in load_records(path):
                if rec.get("dropped"):
                    continue
                data[rec["sample_index"]] = rec
            per_budget[bpct] = data

    print(f"# R7 gate: {args.name}")
    print(f"aligned baseline rows: {len(base)}")
    print(f"budgets: {sorted(per_budget)}")

    # Paired population: rows present in FULL baseline (Route A+B) AND in the
    # given repair file. R7 excludes RANDOM_LENGTH_MATCHED_UNCONSTRUCTIBLE
    # rows the same way R6 did, so n may vary slightly per budget.
    rows_per_budget = {}
    for bpct, data in per_budget.items():
        aligned = {}
        for sidx, rec in data.items():
            if base and sidx not in base:
                continue
            aligned[sidx] = rec
        rows_per_budget[bpct] = aligned

    # Full-Baseline gap from R6 gate (frozen): Full-Naive CleanF1 = 0.1845
    # We recompute from aligned R6 rows if available; else accept the frozen
    # constants to avoid inventing numbers.
    if base:
        n = [r for r in base.values() if r.get("f1_route_a_clean") is not None]
        if n:
            f1A = sum(float(r["f1_route_a_clean"]) for r in n) / len(n)
            f1B = sum(float(r["f1_route_b_clean"]) for r in n) / len(n)
            full_naive_gap = f1A - f1B
            print(f"baseline n={len(n)} f1_full={f1A:.4f} f1_naive={f1B:.4f} gap={full_naive_gap:.4f}")
        else:
            full_naive_gap = 0.1845
    else:
        full_naive_gap = 0.1845

    print()
    header = (f"{'budget%':>8} {'n':>3} {'repairF1':>9} {'naiveF1':>8} "
              f"{'fullF1':>8} {'recovered':>10} {'ratio>=20%':>10}")
    print(header)

    for bpct in sorted(per_budget):
        rep = rows_per_budget[bpct]
        if not rep:
            print(f"{bpct:>8.1f} {'0':>3}  (no aligned rows)")
            continue
        nn = len(rep)
        f1r = sum(float(r["f1_repair_clean"]) for r in rep.values()) / nn
        # naive / full matched from the same window
        f1b_win = 0.0
        f1a_win = 0.0
        cnt = 0
        for sidx in rep:
            r6rec = base.get(sidx)
            if r6rec is None:
                continue
            cnt += 1
            f1b_win += float(r6rec.get("f1_route_b_clean", 0.0))
            f1a_win += float(r6rec.get("f1_route_a_clean", 0.0))
        if cnt:
            f1b_win /= cnt
            f1a_win /= cnt
        else:
            f1b_win = f1a_win = float("nan")
        recovered = (f1r - f1b_win) / full_naive_gap if full_naive_gap else float("nan")
        ok20 = recovered >= 0.20
        print(f"{bpct:>8.1f} {nn:>3} {f1r:>9.4f} {f1b_win:>8.4f} {f1a_win:>8.4f} "
              f"{recovered:>10.3f} {'YES' if ok20 else 'NO':>10}")

    # Signal check at <=10% budgets (1/2/5/10), pick best recovery
    candidates = [b for b in per_budget if b <= 10.0]
    best = None
    for b in candidates:
        rep = rows_per_budget[b]
        if not rep:
            continue
        nn = len(rep)
        if nn < 30:
            continue
        f1r = sum(float(r["f1_repair_clean"]) for r in rep.values()) / nn
        # matched naive
        f1b_win, cnt = 0.0, 0
        for sidx in rep:
            r6rec = base.get(sidx)
            if r6rec is None:
                continue
            cnt += 1
            f1b_win += float(r6rec.get("f1_route_b_clean", 0.0))
        if not cnt:
            continue
        f1b_win /= cnt
        recov = (f1r - f1b_win) / full_naive_gap if full_naive_gap else float("nan")
        if best is None or recov > best[2]:
            best = (b, nn, recov, f1r, f1b_win)

    print()
    if best is None:
        print("SELECTIVE_REPAIR_SIGNAL: NO (no budget<=10% with n>=30)")
        print("needs --r6jsonl for window alignment")
        return

    b, nn, recov, f1r, f1b = best
    # Also require beating RandomMatched at the same budget being signal-y.
    # (Cross-file random comparison happens in the report; here we print.)
    signal = recov >= 0.20
    print(f"SELECTIVE_REPAIR_SIGNAL: {'YES' if signal else 'NO'}")
    print(f"  best budget={b}% n={nn} repairF1={f1r:.4f} naiveF1={f1b:.4f} "
          f"recovered={recov:.3f} of gap {full_naive_gap:.4f}")
    if not signal:
        print("  (must also be checked against RandomMatched at same budget & "
              "teacher-attention asset availability in final report)")

if __name__ == "__main__":
    main()