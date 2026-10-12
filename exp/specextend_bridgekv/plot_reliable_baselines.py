#!/usr/bin/env python3
"""Render the four reliable speculative-decoding baselines as an SVG."""

from __future__ import annotations

import html
import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results" / "rtx4090_reproduction" / "reliable_baseline"
POLICIES = ["full", "recent", "random", "attention"]
LABELS = {"full": "Full KV", "recent": "Recent-only", "random": "Random chunks", "attention": "Attention Top-K"}
COLORS = {"full": "#3569A8", "recent": "#E47C20", "random": "#777777", "attention": "#2E8B57"}
MARKERS = {"full": "circle", "recent": "square", "random": "diamond", "attention": "triangle"}
LENGTHS = ["1K", "2K", "4K", "8K", "16K"]


def load_rows(engine):
    rows = []
    for path in sorted(DATA.glob(f"{engine}_*.json")):
        rows.extend(json.loads(path.read_text())["raw_runs"])
    return [r for r in rows if r["policy"] in POLICIES]


def avg(values):
    values = list(values)
    return statistics.fmean(values) if values else None


def bootstrap_ci(values, seed, n=10000):
    values = [float(v) for v in values]
    rng = random.Random(seed)
    samples = sorted(avg(rng.choices(values, k=len(values))) for _ in range(n))
    return avg(values), samples[int(.025*n)], samples[int(.975*n)]


def metric(row, name):
    if name == "accept":
        return row["avg_accept_length"]
    if name == "tps":
        return row["wall_tokens_per_sec"]
    if name == "kl":
        return row["mean_root_target_draft_kl"]
    if name == "working":
        return avg(x["working_cache_tokens"] for x in row["rounds"])
    if name == "verify":
        return row["stage_time_ms"]["verification"] / row["total_generated"]
    if name == "jaccard":
        vals = [e["jaccard_with_previous"] for e in row["retrieval_events"] if e["jaccard_with_previous"] is not None]
        return avg(vals)
    raise KeyError(name)


def esc(value):
    return html.escape(str(value))


def text(x, y, value, size=15, anchor="start", weight="400", fill="#222", rotate=None):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}"{transform}>{esc(value)}</text>'


def marker(x, y, kind, color, size=5.5):
    if kind == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{size}" fill="{color}" stroke="#fff" stroke-width="1.4"/>'
    if kind == "square":
        return f'<rect x="{x-size:.1f}" y="{y-size:.1f}" width="{2*size:.1f}" height="{2*size:.1f}" fill="{color}" stroke="#fff" stroke-width="1.4"/>'
    if kind == "diamond":
        return f'<path d="M{x:.1f},{y-size-1:.1f} L{x+size+1:.1f},{y:.1f} L{x:.1f},{y+size+1:.1f} L{x-size-1:.1f},{y:.1f} Z" fill="{color}" stroke="#fff" stroke-width="1.4"/>'
    return f'<path d="M{x:.1f},{y-size-2:.1f} L{x+size+2:.1f},{y+size+1:.1f} L{x-size-2:.1f},{y+size+1:.1f} Z" fill="{color}" stroke="#fff" stroke-width="1.4"/>'


def nice_ticks(lo, hi, count=5):
    span = max(hi-lo, 1e-9)
    raw = span / count
    power = 10 ** math.floor(math.log10(raw))
    step = min((1, 2, 2.5, 5, 10), key=lambda x: abs(x*power-raw)) * power
    start = math.floor(lo/step)*step
    end = math.ceil(hi/step)*step
    ticks = []
    value = start
    while value <= end + step/2:
        ticks.append(value)
        value += step
    return ticks


def panel(parts, x0, y0, width, height, title, ylabel, series, log_y=False, fixed_domain=None, note=None):
    ml, mr, mt, mb = 72, 18, 48, 52
    left, right = x0+ml, x0+width-mr
    top, bottom = y0+mt, y0+height-mb
    parts.append(text(x0+4, y0+24, title, size=18, weight="600"))
    if note:
        parts.append(text(x0+4, y0+43, note, size=12, fill="#666"))

    values = [v for rows in series.values() for _, v, lo, hi in rows if v is not None for v in (lo, hi)]
    if fixed_domain:
        ymin, ymax = fixed_domain
    elif log_y:
        ymin, ymax = min(values)*.85, max(values)*1.15
    else:
        ymin, ymax = min(values), max(values)
        pad = max((ymax-ymin)*.12, .03)
        ymin = max(0, ymin-pad)
        ymax += pad

    def sx(i):
        return left + i*(right-left)/(len(LENGTHS)-1)

    def sy(v):
        if log_y:
            return bottom - (math.log2(v)-math.log2(ymin))/(math.log2(ymax)-math.log2(ymin))*(bottom-top)
        return bottom - (v-ymin)/(ymax-ymin)*(bottom-top)

    if log_y:
        yticks = [1024, 2048, 4096, 8192, 16384]
        ylabels = ["1K", "2K", "4K", "8K", "16K"]
    else:
        yticks = nice_ticks(ymin, ymax)
        ylabels = [f"{v:.0f}" if abs(v) >= 10 else f"{v:.1f}" for v in yticks]

    parts.append(f'<rect x="{left:.1f}" y="{top:.1f}" width="{right-left:.1f}" height="{bottom-top:.1f}" fill="#ffffff" stroke="#b9bec6" stroke-width="1"/>')
    for tick_value, tick_label in zip(yticks, ylabels):
        if tick_value < ymin or tick_value > ymax:
            continue
        yy = sy(tick_value)
        parts.append(f'<line x1="{left:.1f}" x2="{right:.1f}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="#e6e8eb" stroke-width="1"/>')
        parts.append(text(left-9, yy+5, tick_label, size=13, anchor="end", fill="#555"))
    for i, label in enumerate(LENGTHS):
        xx = sx(i)
        parts.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{bottom:.1f}" y2="{bottom+5:.1f}" stroke="#777"/>')
        parts.append(text(xx, bottom+23, label, size=13, anchor="middle", fill="#444"))
    parts.append(text((left+right)/2, y0+height-6, "Context length", size=14, anchor="middle", fill="#333"))
    parts.append(text(x0+17, (top+bottom)/2, ylabel, size=14, anchor="middle", fill="#333", rotate=-90))

    for policy in POLICIES:
        rows = series.get(policy, [])
        if not rows:
            continue
        coords = [(sx(i), sy(v)) for i, (_, v, _, _) in enumerate(rows) if v is not None]
        if len(coords) > 1:
            path = " ".join(("M" if i == 0 else "L") + f"{xx:.1f},{yy:.1f}" for i, (xx, yy) in enumerate(coords))
            parts.append(f'<path d="{path}" fill="none" stroke="{COLORS[policy]}" stroke-width="2.6"/>')
        for i, (_, value, lo, hi) in enumerate(rows):
            if value is None:
                continue
            xx, yy = sx(i), sy(value)
            ylo, yhi = sy(lo), sy(hi)
            parts.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{yhi:.1f}" y2="{ylo:.1f}" stroke="{COLORS[policy]}" stroke-width="1.5"/>')
            parts.append(f'<line x1="{xx-4:.1f}" x2="{xx+4:.1f}" y1="{yhi:.1f}" y2="{yhi:.1f}" stroke="{COLORS[policy]}" stroke-width="1.5"/>')
            parts.append(f'<line x1="{xx-4:.1f}" x2="{xx+4:.1f}" y1="{ylo:.1f}" y2="{ylo:.1f}" stroke="{COLORS[policy]}" stroke-width="1.5"/>')
            parts.append(marker(xx, yy, MARKERS[policy], COLORS[policy]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("classic", "eagle"), default="classic")
    args = parser.parse_args()
    engine = args.engine
    rows = load_rows(engine)
    if not rows:
        raise SystemExit(f"No {engine}_*.json results found in {DATA}")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["length_label"], row["policy"])].append(row)
    metrics = [
        ("accept", "A  Draft quality", "Average acceptance length", False, None, None),
        ("tps", "B  End-to-end performance", "Throughput (token/s)", False, None, None),
        ("kl", "C  Cross-model mismatch", "Root KL(target || draft)", False, None, None),
        ("working", "D  Draft working cache", "Working-cache tokens (log2)", True, (850, 18000), None),
        ("verify", "E  Verification cost", "Target verification (ms/token)", False, None, None),
        ("jaccard", "F  Retrieval stability", "Consecutive-set Jaccard", False, (0, 1.05), "Full KV has no retrieval set"),
    ]
    summaries = {}
    for metric_name, *_ in metrics:
        per_policy = {}
        for pi, policy in enumerate(POLICIES):
            values = []
            for li, length in enumerate(LENGTHS):
                vals = [metric(r, metric_name) for r in grouped[(length, policy)]]
                vals = [v for v in vals if v is not None]
                if vals:
                    values.append((length, *bootstrap_ci(vals, seed=1000+pi*100+li)))
                else:
                    values.append((length, None, None, None))
            per_policy[policy] = values
        summaries[metric_name] = per_policy

    width, height = 1500, 1610
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        text(48, 42, f"{engine.upper() if engine == 'eagle' else 'Classic'} speculative decoding: four long-context baselines", size=25, weight="600"),
        text(48, 68, f"GovReport · n=20 paired documents per length · mean with bootstrap 95% CI · chunk size 32 · cache Top-K {64 if engine == 'eagle' else 32}", size=14, fill="#555"),
    ]
    lx = 48
    for policy in POLICIES:
        parts.append(marker(lx, 94, MARKERS[policy], COLORS[policy], size=5))
        parts.append(text(lx+13, 99, LABELS[policy], size=14, fill="#333"))
        lx += {"full":170, "recent":190, "random":205, "attention":210}[policy]

    margin_x, top_y, gap_x, gap_y = 38, 118, 20, 18
    panel_w = (width - margin_x*2 - gap_x) / 2
    panel_h = 474
    for idx, (metric_name, title, ylabel, log_y, domain, note) in enumerate(metrics):
        col, row = idx % 2, idx // 2
        panel(parts, margin_x+col*(panel_w+gap_x), top_y+row*(panel_h+gap_y), panel_w, panel_h,
              title, ylabel, summaries[metric_name], log_y=log_y, fixed_domain=domain, note=note)

    parts.append(text(48, 1592, "16K is the released nominal bin; actual Vicuna-tokenized inputs are 15.1K–15.4K. Full+Hybrid is intentionally omitted here because this figure shows the requested four baselines.", size=13, fill="#555"))
    parts.append('</svg>')
    out = DATA / f"{engine}_four_baselines.svg"
    out.write_text("\n".join(parts) + "\n")
    print(out)


if __name__ == "__main__":
    main()
