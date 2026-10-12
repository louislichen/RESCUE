#!/usr/bin/env python3
"""Analyze long-context Classic/EAGLE baseline profiler outputs."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else None


def paired_ci(rows_a, rows_b, value_fn, seed=20260923, n_boot=20000):
    a = {r["sample_index"]: value_fn(r) for r in rows_a}
    b = {r["sample_index"]: value_fn(r) for r in rows_b}
    ids = sorted(a.keys() & b.keys())
    delta = [a[i] - b[i] for i in ids]
    rng = random.Random(seed)
    boot = sorted(mean(rng.choices(delta, k=len(delta))) for _ in range(n_boot))
    return mean(delta), boot[int(.025*n_boot)], boot[int(.975*n_boot)]


def fmt(x, signed=False):
    return (f"{x:+.3f}" if signed else f"{x:.3f}") if x is not None else "—"


def ci_text(x):
    return f"{fmt(x[0], True)} [{fmt(x[1], True)}, {fmt(x[2], True)}]"


def stage_per_token(row, stage):
    return row.get("stage_time_ms", {}).get(stage, 0.0) / row["total_generated"]


def working_tokens(row):
    return mean(x["working_cache_tokens"] for x in row.get("rounds", [])) or 0.0


def jaccard(row):
    values = [x["jaccard_with_previous"] for x in row.get("retrieval_events", []) if x["jaccard_with_previous"] is not None]
    return mean(values)


def selected_position(row):
    values = [v for event in row.get("retrieval_events", []) if event["kind"] == "retrieval" for v in event["selected_chunk_centers_normalized"]]
    return mean(values)


def rejection(row):
    return mean(x["rejected_position"] for x in row.get("rounds", [])) or 0.0


def load_rows(directory):
    rows = []
    for path in sorted(Path(directory).glob("*.json")):
        rows.extend(json.loads(path.read_text())["raw_runs"])
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    rows = load_rows(args.directory)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["engine"], row["length_label"], row["policy"])].append(row)

    engines = sorted({r["engine"] for r in rows})
    lengths = [x for x in ("1K", "2K", "4K", "8K", "16K") if any(r["length_label"] == x for r in rows)]
    policies = ("full", "full_hybrid", "recent", "random", "attention")
    out = ["# Reliable long-context speculative-decoding baselines", ""]
    for engine in engines:
        out += [f"## {engine.capitalize()}", ""]
        for length in lengths:
            present = [p for p in policies if groups[(engine, length, p)]]
            if not present:
                continue
            out += [f"### {length}", "",
                    "| Policy | n | Accept | tok/s | Draft ms/token | Verify ms/token | Retrieval ms/token | Working tokens | Jaccard | Root KL | Rejected pos. | Selected pos. |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for policy in present:
                rs = groups[(engine, length, policy)]
                out.append(
                    f"| {policy} | {len(rs)} | {fmt(mean(r['avg_accept_length'] for r in rs))} | "
                    f"{fmt(mean(r['wall_tokens_per_sec'] for r in rs))} | {fmt(mean(stage_per_token(r,'draft') for r in rs))} | "
                    f"{fmt(mean(stage_per_token(r,'verification') for r in rs))} | {fmt(mean(stage_per_token(r,'retrieval') for r in rs))} | "
                    f"{fmt(mean(working_tokens(r) for r in rs))} | {fmt(mean(x for r in rs if (x:=jaccard(r)) is not None))} | "
                    f"{fmt(mean(r.get('mean_root_target_draft_kl') for r in rs))} | {fmt(mean(rejection(r) for r in rs))} | "
                    f"{fmt(mean(x for r in rs if (x:=selected_position(r)) is not None))} |"
                )
            full = groups[(engine, length, "full")]
            out += ["", "Paired differences versus Full (95% bootstrap CI):", ""]
            for policy in ("full_hybrid", "recent", "random", "attention"):
                rs = groups[(engine, length, policy)]
                if full and rs:
                    out.append(f"- {policy}: accept {ci_text(paired_ci(rs, full, lambda r:r['avg_accept_length']))}; "
                               f"tok/s {ci_text(paired_ci(rs, full, lambda r:r['wall_tokens_per_sec']))}")
            attn = groups[(engine, length, "attention")]
            full_hybrid = groups[(engine, length, "full_hybrid")]
            if full_hybrid and full:
                out += ["", "Mechanism decomposition:", ""]
                out.append(f"- efficient target verification (full_hybrid−full): tok/s "
                           f"{ci_text(paired_ci(full_hybrid, full, lambda r:r['wall_tokens_per_sec']))}")
                for policy in ("recent", "random"):
                    rs = groups[(engine, length, policy)]
                    if rs:
                        out.append(f"- cache shortening ({policy}−full_hybrid): accept "
                                   f"{ci_text(paired_ci(rs, full_hybrid, lambda r:r['avg_accept_length']))}; tok/s "
                                   f"{ci_text(paired_ci(rs, full_hybrid, lambda r:r['wall_tokens_per_sec']))}")
            if attn:
                out += ["", "Retrieval-quality controls at the same cache budget:", ""]
                for policy in ("recent", "random"):
                    rs = groups[(engine, length, policy)]
                    if rs:
                        out.append(f"- attention−{policy}: accept {ci_text(paired_ci(attn, rs, lambda r:r['avg_accept_length']))}; "
                                   f"tok/s {ci_text(paired_ci(attn, rs, lambda r:r['wall_tokens_per_sec']))}; "
                                   f"root KL {ci_text(paired_ci(attn, rs, lambda r:r['mean_root_target_draft_kl']))}")
            out.append("")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(out) + "\n")
    print(f"wrote {output} from {len(rows)} runs")


if __name__ == "__main__":
    main()
