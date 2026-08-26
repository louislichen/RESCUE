#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R7 gate statistics: multi-condition comparison on the frozen criteria.

Each R7 run row already carries the same-window Route-A/B baselines
(f1_route_a_clean / f1_route_b_clean computed from Route A/B decodes of the
SAME sample and window), so naive/full paired means are read straight from
the repair JSONL itself — no cross-file window drift.

Usage:
  python r7_gate_stats.py \
      --name oracle_span  --budgets 100  --jsonl <oracle_span.jsonl> \
      --name oracle_budget --budgets 1,2,5,10,20 --jsonl <oracle_budget.jsonl> \
      --name random_matched_budget --budgets 1,2,5,10,20 --jsonl <random.jsonl> \
      [--gap 0.1845] [--min-n 30]

Criteria (frozen, Sprint §8):
  SELECTIVE_REPAIR_SIGNAL <= 10% budget where:
    - repair beats same-window naive (repairF1 > naiveF1)
    - repair beats RandomMatched at the same budget
    - recovered >= 20% of the Full-Naive CleanF1 gap
    - paired n >= 30
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict

FROZEN_GAP = 0.1845          # R6 Full-Naive CleanF1 gap (frozen reference)
FROZEN_RECOVERY = 0.20
FROZEN_MAX_BUDGET = 10.0
FROZEN_MIN_N = 30


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", action="append", required=True,
                    help="condition name (repeat per input)")
    ap.add_argument("--budgets", action="append", required=True,
                    help="comma list of budget percents for this condition")
    ap.add_argument("--jsonl", action="append", required=True,
                    help="repair JSONL path for this condition")
    ap.add_argument("--gap", type=float, default=FROZEN_GAP,
                    help="Full-Naive CleanF1 gap (default frozen 0.1845)")
    ap.add_argument("--min-n", type=int, default=FROZEN_MIN_N)
    return ap.parse_args()


def load_records(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def main():
    args = parse_args()
    if not (len(args.name) == len(args.budgets) == len(args.jsonl)):
        sys.exit("--name/--budgets/--jsonl count mismatch")

    conditions = []
    for nm, bspec, path in zip(args.name, args.budgets, args.jsonl):
        budgets = [float(x) for x in bspec.split(",")]
        data = defaultdict(dict)  # budget_pct -> {sample_index: rec}
        for rec in load_records(path):
            if rec.get("dropped"):
                continue
            bp = float(rec["budget_pct"]) if "budget_pct" in rec else budgets[0]
            data[bp][rec["sample_index"]] = rec
        conditions.append({"name": nm, "budgets": budgets,
                           "data": data, "path": path})

    print("# R7 gate: multi-condition comparison")
    print(f"Full-Naive gap (frozen): {args.gap:.4f} | recovery bar: "
          f"{FROZEN_RECOVERY:.0%} | budget bar: <= {FROZEN_MAX_BUDGET:.0f}% | "
          f"n bar: >= {args.min_n}")
    print()

    # ---- stats per condition per budget ----
    stats = {}  # (cond, bp) -> dict
    order = [cond["name"] for cond in conditions]
    all_budgets = sorted({b for c in conditions for b in c["budgets"]})

    for cond in conditions:
        nm = cond["name"]
        print(f"== {nm} ==")
        header = (f"{'budget%':>8} {'n':>3} {'repairF1':>9} {'naiveF1':>8} "
                  f"{'fullF1':>8} {'deltaVnaive':>11} {'recovered':>10} "
                  f"{'ok20%':>6}")
        print(header)
        for bp in sorted(cond["budgets"]):
            rep = cond["data"].get(bp, {})
            nn = len(rep)
            if nn == 0:
                print(f"{bp:>8.1f} {'0':>3}  (no rows)")
                stats[(nm, bp)] = None
                continue
            f1r = sum(float(r.get("f1_repair_clean", 0.0)) for r in rep.values()) / nn
            f1b = sum(float(r.get("f1_route_b_clean", 0.0)) for r in rep.values()) / nn
            f1a = sum(float(r.get("f1_route_a_clean", 0.0)) for r in rep.values()) / nn
            recovered = (f1r - f1b) / args.gap if args.gap else float("nan")
            ok20 = recovered >= FROZEN_RECOVERY
            print(f"{bp:>8.1f} {nn:>3} {f1r:>9.4f} {f1b:>8.4f} {f1a:>8.4f} "
                  f"{f1r - f1b:>+11.4f} {recovered:>10.3f} {'YES' if ok20 else 'NO':>6}")
            stats[(nm, bp)] = {"n": nn, "f1r": f1r, "f1b": f1b, "f1a": f1a,
                               "recovered": recovered}
        print()

    # ---- SELECTIVE_REPAIR_SIGNAL check ----
    print("# SELECTIVE_REPAIR_SIGNAL (<=10% budget, n>=30)")
    # Reference: random condition at same budget beats naive? and oracle
    # conditions must beat BOTH naive and random at <=10%.
    random_cond = next((c for c in conditions
                        if "random" in c["name"]), None)
    oracle_conds = [c for c in conditions if "oracle" in c["name"]]

    def rand_ref(bp):
        if random_cond is None:
            return None
        st = stats.get((random_cond["name"], bp))
        return st["f1r"] if st and st["n"] >= args.min_n else None

    results = {}
    for cond in oracle_conds:
        for bp in sorted(cond["budgets"]):
            if bp > FROZEN_MAX_BUDGET:
                continue
            st = stats.get((cond["name"], bp))
            if st is None or st["n"] < args.min_n:
                continue
            beats_naive = st["f1r"] > st["f1b"]
            recov_ok = st["recovered"] >= FROZEN_RECOVERY
            rref = rand_ref(bp)
            beats_random = rref is not None and st["f1r"] > rref
            ok = beats_naive and recov_ok and beats_random
            results[(cond["name"], bp)] = {
                "f1r": st["f1r"], "f1b": st["f1b"],
                "rand": rref, "recov": st["recovered"], "ok": ok,
                "beats_naive": beats_naive, "beats_random": beats_random,
                "recov_ok": recov_ok}
            print(f"  {cond['name']:>22} budget={bp:>4.1f}% n={st['n']:>3} "
                  f"repair={st['f1r']:.4f} naive={st['f1b']:.4f} "
                  f"rand={rref if rref is None else round(rref, 4)} "
                  f"recovered={st['recovered']:.3f} | "
                  f"beats_naive={'Y' if beats_naive else 'N'} "
                  f"beats_random={'Y' if beats_random else 'N'} "
                  f"recov20={'Y' if recov_ok else 'N'} "
                  f"=> {'SIGNAL' if ok else '-'}")

    pos = [k for k, v in results.items() if v["ok"]]
    print()
    if pos:
        print(f"SELECTIVE_REPAIR_SIGNAL: YES at {len(pos)} budget points: "
              f"{[f'{k[0]}@{k[1]}%' for k in pos]}")
        best = max(pos, key=lambda k: results[k]["recov"])
        v = results[best]
        print(f"  strongest: {best[0]}@{best[1]}% repair={v['f1r']:.4f} "
              f"naive={v['f1b']:.4f} rand={v['rand']:.4f} "
              f"recovered={v['recov']:.3f} of gap {args.gap:.4f}")
    else:
        # allow the pure-oracle signal without random reference if random
        # condition was not supplied or n below bar
        print("SELECTIVE_REPAIR_SIGNAL: NO")
        if random_cond is None:
            print("  (random condition not supplied; cross-condition check skipped)")
        else:
            print("  (no <=10% budget point meets ALL of: beats naive, "
                  "beats random, recovered>=20%, n>=30)")


if __name__ == "__main__":
    main()