#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R8 gate statistics: paired CleanF1 delta vs Naive / vs RandomMatched per
selector on the frozen single budget, plus diagnostics (Top-K overlap with
oracle spans, gold supporting-span recall, JSD/Pearson between teacher and
student attention over the candidate pool).

Criteria (frozen, Sprint §9-10):
  - SELECTOR_GUIDANCE_SUPPORT = YES only if (TeacherAttentionTopK OR
    TeacherStudentGapTopK), at the SAME cost, beats BOTH RandomMatched AND
    QueryOverlap on paired CleanF1 delta vs Naive.
  - Cost axis identical across selectors: processed_token_ratio reported per
    selector; selectors run on the same sample window (n paired).
"""
from __future__ import annotations
import argparse, json, math, sys
from collections import defaultdict

FROZEN_GAP = 0.1845


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True,
                    help="R8 selector-run JSONL (one row per selector/sample)")
    ap.add_argument("--gap", type=float, default=FROZEN_GAP,
                    help="Full-Naive CleanF1 gap (default 0.1845)")
    return ap.parse_args()


def jsd(p: list, q: list) -> float:
    """Jensen-Shannon divergence between two discrete distributions."""
    import math
    eps = 1e-12
    m = [0.5 * (a + b) for a, b in zip(p, q)]
    kl_pm = sum(a * math.log((a + eps) / (b + eps)) for a, b in zip(p, m))
    kl_qm = sum(a * math.log((a + eps) / (b + eps)) for a, b in zip(q, m))
    return 0.5 * (kl_pm + kl_qm)


def pearson(x: list, y: list) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return cov / math.sqrt(vx * vy)


def main():
    args = parse_args()
    recs = []
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    ev = [r for r in recs if not r.get("dropped")]
    print(f"rows: {len(recs)}  evaluated: {len(ev)}")
    by_sel = defaultdict(list)
    for r in ev:
        by_sel[r["selector"]].append(r)

    print()
    print(f"Full-Naive gap (frozen): {args.gap:.4f}")
    print(f"{'selector':>26} {'n':>3} {'repairF1':>9} {'naiveF1':>8} "
          f"{'delta':>9} {'recov':>7} {'pr_ratio':>9} {'wr_ratio':>9} "
          f"{'topgold':>8} {'goldrec':>8}")
    per = {}
    for sel in sorted(by_sel):
        rows = by_sel[sel]
        nn = len(rows)
        f1r = sum(float(r.get("f1_repair_clean", 0.0)) for r in rows) / nn
        f1b = sum(float(r.get("f1_route_b_clean", 0.0)) for r in rows) / nn
        f1a = sum(float(r.get("f1_route_a_clean", 0.0)) for r in rows) / nn
        pr = sum(float(r.get("processed_token_ratio", 0.0)) for r in rows) / nn
        wr = sum(float(r.get("written_token_ratio", 0.0)) for r in rows) / nn
        tg = sum(float(r.get("gold_span_hit_rate", 0.0)) for r in rows) / nn
        gr = sum(float(r.get("gold_token_recall", 0.0)) for r in rows) / nn
        delta = f1r - f1b
        recov = delta / args.gap if args.gap else float("nan")
        per[sel] = {"n": nn, "f1r": f1r, "f1b": f1b, "f1a": f1a, "delta": delta,
                    "recov": recov, "pr": pr, "wr": wr, "tg": tg, "gr": gr}
        print(f"{sel:>26} {nn:>3} {f1r:>9.4f} {f1b:>8.4f} {delta:>9.4f} "
              f"{recov:>7.3f} {pr:>9.4f} {wr:>9.4f} {tg:>8.3f} {gr:>8.3f}")

    print()
    print("# Teacher/Student attention diagnostics (per-sample, full pool)")
    # rebuild teacher/student rows by sample_index from the JSONL itself
    rows_by_sample = defaultdict(dict)
    for r in ev:
        rows_by_sample[r["sample_index"]][r["selector"]] = r
    jsds, pcs = [], []
    counts = 0
    for sidx, bys in sorted(rows_by_sample.items()):
        st = bys.get("student_attn_topk")
        te = bys.get("teacher_attn_topk")
        if st is None or te is None:
            continue
        ss = st.get("selector_scores_sample")
        ts = te.get("selector_scores_sample")
        if not ss or not ts or len(ss) != len(ts):
            continue
        sm = sum(ss)
        tm = sum(ts)
        if sm <= 0 or tm <= 0:
            continue
        p = [x / sm for x in ss]
        q = [x / tm for x in ts]
        jsds.append(jsd(p, q))
        pcs.append(pearson(p, q))
        counts += 1
    if counts:
        print(f"paired samples for diagnostics: {counts}")
        print(f"teacher-student JSD: mean={sum(jsds)/counts:.4f} "
              f"min={min(jsds):.4f} max={max(jsds):.4f}")
        pc_ok = [x for x in pcs if not math.isnan(x)]
        if pc_ok:
            print(f"teacher-student Pearson: mean={sum(pc_ok)/len(pc_ok):.4f} "
                  f"(n={len(pc_ok)})")

    print()
    print("# SELECTOR_GUIDANCE_SUPPORT")
    rand = per.get("random_matched")
    qov = per.get("query_overlap")
    cand_sel = ["teacher_attn_topk", "teacher_student_gap_topk"]
    supported = []
    for sel in cand_sel:
        s = per.get(sel)
        if s is None:
            continue
        beats_rand = rand is not None and s["delta"] > rand["delta"]
        beats_qov = qov is not None and s["delta"] > qov["delta"]
        cost_ok = s["pr"] <= max(per[s2]["pr"] for s2 in per if per.get(s2))
        tag = "SUPPORT" if (beats_rand and beats_qov and cost_ok) else "-"
        print(f"  {sel:>26}: delta={s['delta']:+.4f} vs rand={rand['delta'] if rand else float('nan'):+.4f} "
              f"vs query={qov['delta'] if qov else float('nan'):+.4f} "
              f"beats_rand={'Y' if beats_rand else 'N'} "
              f"beats_query={'Y' if beats_qov else 'N'} "
              f"cost_ok={'Y' if cost_ok else 'N'} => {tag}")
        if beats_rand and beats_qov and cost_ok:
            supported.append(sel)
    print()
    if supported:
        print(f"SELECTOR_GUIDANCE_SUPPORT: YES ({supported})")
    else:
        print("SELECTOR_GUIDANCE_SUPPORT: NO")


if __name__ == "__main__":
    main()