#!/usr/bin/env python3
"""Analyze the full Classic P1 multi-query retrieval matrix and draw SVG figures."""

import html
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results" / "rtx4090_reproduction" / "full_multi_query"
LENGTHS = ["4K", "8K", "16K"]
AGGS = ["mean", "decay", "max"]
QUERIES = [1, 2, 4, 8, 0]
QLABEL = {1: "1", 2: "2", 4: "4", 8: "8", 0: "All"}
COLORS = {"4K": "#3569A8", "8K": "#E47C20", "16K": "#2E8B57"}


def mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.fmean(values) if values else None


def bootstrap(values, seed, n=5000):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return None, None, None
    rng = random.Random(seed)
    samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(n))
    return mean(values), samples[int(0.025 * n)], samples[int(0.975 * n)]


def paired(rows, base, fn, seed, n=7000):
    a = {r["sample_index"]: fn(r) for r in rows}
    b = {r["sample_index"]: fn(r) for r in base}
    ids = sorted(a.keys() & b.keys())
    return bootstrap([a[i] - b[i] for i in ids], seed, n)


def load_rows():
    rows = []
    for path in sorted(DATA.glob("classic_full_mq_*.json")):
        payload = json.loads(path.read_text())
        rows.extend(payload["raw_runs"])
    return rows


def policy(agg, q):
    return f"attention_{agg}_q{'all' if q == 0 else q}"


def jaccard(row):
    return mean(e.get("jaccard_with_previous") for e in row["retrieval_events"])


def selected_position(row):
    return mean(
        value
        for event in row["retrieval_events"]
        if event.get("kind") == "retrieval"
        for value in event.get("selected_chunk_centers_normalized", [])
    )


def round_mean(row, key):
    return mean(item.get(key) for item in row["rounds"])


def metric(row, name):
    if name == "accept":
        return row["avg_accept_length"]
    if name == "tps":
        return row["wall_tokens_per_sec"]
    if name == "kl":
        return row["mean_root_target_draft_kl"]
    if name == "jaccard":
        return jaccard(row)
    if name == "retrieval":
        return row["stage_time_ms"].get("retrieval", 0.0) / row["total_generated"]
    if name == "draft":
        return row["stage_time_ms"].get("draft", 0.0) / row["total_generated"]
    if name == "verify":
        return row["stage_time_ms"].get("verification", 0.0) / row["total_generated"]
    if name == "cache":
        return round_mean(row, "working_cache_tokens")
    if name == "reject":
        return round_mean(row, "rejected_position")
    if name == "position":
        return selected_position(row)
    raise KeyError(name)


def esc(value):
    return html.escape(str(value))


def text(x, y, value, size=14, anchor="start", weight="400", fill="#222", rotate=None):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial,sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" '
        f'fill="{fill}"{transform}>{esc(value)}</text>'
    )


def line(x1, y1, x2, y2, color="#d9dde2", width=1, dash=None):
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}"{dashed}/>'
    )


def grouped(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["length_label"], row["policy"])].append(row)
    return groups


def draw_main(rows, groups):
    metrics = [("accept", "Acceptance length"), ("tps", "End-to-end token/s")]
    width, height = 1560, 970
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        text(42, 40, "P1 full matrix: multi-query attention retrieval", 25, weight="600"),
        text(42, 66, "Classic · GovReport · n=20 paired documents · bootstrap 95% CI · randomized policy order", 14, fill="#555"),
    ]
    for i, length in enumerate(LENGTHS):
        x = 44 + 170 * i
        out += [f'<circle cx="{x}" cy="91" r="5" fill="{COLORS[length]}"/>', text(x + 13, 96, length, 14)]

    for row_index, (metric_name, ylabel) in enumerate(metrics):
        for col_index, agg in enumerate(AGGS):
            x0 = 26 + col_index * 510
            y0 = 116 + row_index * 410
            panel_w, panel_h = 488, 385
            left, right, top, bottom = x0 + 72, x0 + panel_w - 18, y0 + 50, y0 + panel_h - 54
            values = []
            series = {}
            for length_index, length in enumerate(LENGTHS):
                points = []
                for query_index, q in enumerate(QUERIES):
                    chosen_policy = "attention_mean_q1" if q == 1 else policy(agg, q)
                    vals = [metric(item, metric_name) for item in groups[(length, chosen_policy)]]
                    estimate = bootstrap(vals, 1000 + row_index * 300 + col_index * 70 + length_index * 10 + query_index)
                    points.append(estimate)
                    values.extend(v for v in estimate[1:] if v is not None)
                series[length] = points
            ymin = min(values)
            ymax = max(values)
            padding = max((ymax - ymin) * 0.13, 0.02 if metric_name == "accept" else 0.5)
            ymin -= padding
            ymax += padding
            sy = lambda value: bottom - (value - ymin) / (ymax - ymin) * (bottom - top)
            sx = lambda index: left + (right - left) * index / (len(QUERIES) - 1)
            panel_letter = chr(ord("A") + row_index * 3 + col_index)
            out += [
                text(x0 + 5, y0 + 25, f"{panel_letter}  {agg.capitalize()} aggregation", 18, weight="600"),
                f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="#fff" stroke="#b9bec6"/>',
            ]
            for tick in range(5):
                value = ymin + (ymax - ymin) * tick / 4
                y = sy(value)
                out += [line(left, y, right, y), text(left - 8, y + 5, f"{value:.2f}", 12, "end", fill="#555")]
            for qi, q in enumerate(QUERIES):
                out.append(text(sx(qi), bottom + 25, QLABEL[q], 13, "middle"))
            out += [
                text((left + right) / 2, y0 + panel_h - 8, "Number of accepted-path queries", 13, "middle"),
                text(x0 + 17, (top + bottom) / 2, ylabel, 13, "middle", rotate=-90),
            ]
            for length in LENGTHS:
                points = []
                for qi, (value, low, high) in enumerate(series[length]):
                    x = sx(qi)
                    points.append((x, sy(value)))
                    out += [
                        line(x, sy(low), x, sy(high), COLORS[length], 1.4),
                        line(x - 4, sy(low), x + 4, sy(low), COLORS[length], 1.4),
                        line(x - 4, sy(high), x + 4, sy(high), COLORS[length], 1.4),
                    ]
                path = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))
                out.append(f'<path d="{path}" fill="none" stroke="{COLORS[length]}" stroke-width="2.4"/>')
                out.extend(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.8" fill="{COLORS[length]}" stroke="#fff"/>' for x, y in points)
    out += [
        text(42, 950, "Q=1 is the original Last-1 method. “All” uses every token accepted in the preceding verification round.", 13, fill="#555"),
        "</svg>",
    ]
    (DATA / "p1_quality_performance.svg").write_text("\n".join(out) + "\n")


def color(value, limit, positive_good=False):
    if value is None or limit <= 0:
        return "#f4f4f4"
    z = max(-1.0, min(1.0, value / limit))
    if positive_good:
        z = -z
    if z >= 0:
        start, end = (247, 247, 247), (180, 50, 55)
    else:
        start, end = (247, 247, 247), (45, 125, 180)
        z = -z
    return "#" + "".join(f"{round(a + (b-a)*z):02x}" for a, b in zip(start, end))


def draw_diagnostics(rows, groups):
    configs = [(agg, q) for agg in AGGS for q in QUERIES[1:]]
    metrics = [
        ("kl", "A  Root KL delta", False, 3),
        ("jaccard", "B  Jaccard delta", True, 3),
        ("retrieval", "C  Retrieval ms/token delta", False, 4),
        ("draft", "D  Draft ms/token delta", False, 3),
        ("verify", "E  Verification ms/token delta", False, 3),
        ("cache", "F  Working-cache tokens delta", False, 0),
        ("reject", "G  Rejection-position delta", True, 3),
        ("position", "H  Selected-position delta", False, 3),
    ]
    effects = {}
    for metric_index, (name, _, _, _) in enumerate(metrics):
        for length_index, length in enumerate(LENGTHS):
            base = groups[(length, "attention_mean_q1")]
            for config_index, (agg, q) in enumerate(configs):
                effects[(name, length, agg, q)] = paired(
                    groups[(length, policy(agg, q))],
                    base,
                    lambda row, n=name: metric(row, n),
                    5000 + metric_index * 1000 + length_index * 100 + config_index,
                    n=5000,
                )
    width, height = 1540, 1590
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        text(42, 40, "P1 mechanism diagnostics: paired change from Last-1", 25, weight="600"),
        text(42, 66, "Blue is improvement for directional metrics; outlined cells have a 95% CI excluding zero", 14, fill="#555"),
    ]
    for metric_index, (name, title_, positive_good, decimals) in enumerate(metrics):
        col, row_index = metric_index % 2, metric_index // 2
        x0, y0 = 30 + col * 755, 100 + row_index * 365
        cell_w, cell_h = 160, 22
        left, top = x0 + 190, y0 + 53
        vals = [abs(effects[(name, length, agg, q)][0]) for length in LENGTHS for agg, q in configs]
        limit = sorted(vals)[max(0, int(len(vals) * 0.9) - 1)] or max(vals) or 1.0
        out.append(text(x0 + 4, y0 + 23, title_, 18, weight="600"))
        for length_index, length in enumerate(LENGTHS):
            out.append(text(left + length_index * cell_w + cell_w / 2, y0 + 45, length, 13, "middle", weight="600"))
        for config_index, (agg, q) in enumerate(configs):
            y = top + config_index * cell_h
            if config_index % 4 == 0:
                out.append(text(x0 + 5, y + 16, agg.capitalize(), 13, weight="600"))
            out.append(text(x0 + 118, y + 16, f"Q={QLABEL[q]}", 12))
            for length_index, length in enumerate(LENGTHS):
                value, low, high = effects[(name, length, agg, q)]
                x = left + length_index * cell_w
                significant = low > 0 or high < 0
                outline = "#222" if significant else "#ddd"
                out += [
                    f'<rect x="{x}" y="{y}" width="{cell_w-2}" height="{cell_h-1}" fill="{color(value, limit, positive_good)}" stroke="{outline}" stroke-width="{1.5 if significant else 0.6}"/>',
                    text(x + (cell_w - 2) / 2, y + 15.5, f"{value:+.{decimals}f}", 11, "middle"),
                ]
        out.append(text(left, top + len(configs) * cell_h + 22, f"Color clipped at |delta|={limit:.{max(1, decimals)}f}", 11, fill="#666"))
    out += [text(42, 1570, "Each cell is a paired mean difference over the same 20 documents; Q=1 is shared across aggregation rules.", 13, fill="#555"), "</svg>"]
    (DATA / "p1_diagnostics.svg").write_text("\n".join(out) + "\n")
    return effects


def validate(rows, groups):
    expected = {(length, "attention_mean_q1") for length in LENGTHS}
    expected |= {(length, policy(agg, q)) for length in LENGTHS for agg in AGGS for q in QUERIES[1:]}
    actual = set(groups)
    missing = sorted(expected - actual)
    if missing:
        raise SystemExit(f"missing groups: {missing}")
    bad = {key: len(groups[key]) for key in expected if len(groups[key]) != 20}
    if bad:
        raise SystemExit(f"expected 20 samples per group: {bad}")
    keys = [(r["length_label"], r["policy"], r["sample_index"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit("duplicate length/policy/sample rows")


def write_report(rows, groups, effects):
    report = [
        "# P1：多 Query 注意力检索（完整矩阵）",
        "",
        "![质量与性能](p1_quality_performance.png)",
        "",
        "![机制诊断](p1_diagnostics.png)",
        "",
        "## 实验设置",
        "",
        "Classic draft；GovReport；4K/8K/16K；每个条件 20 个配对样本；chunk size 32；top-k 32；每 8 轮检索；候选树 64 nodes；greedy decoding；随机种子 42；策略执行顺序按样本确定性打乱。Decay 使用 lambda=0.7。",
        "",
        "总计 13 个策略（Q=1 只保留一次，因为三种聚合在单 query 时等价），共 780 次运行。",
        "",
        "## 均值",
        "",
        "| Length | Aggregation | Q | Acceptance | token/s | Root KL | Jaccard | Retrieval ms/token |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for length in LENGTHS:
        base = groups[(length, "attention_mean_q1")]
        report.append(
            f"| {length} | — | 1 | {mean(metric(r,'accept') for r in base):.3f} | {mean(metric(r,'tps') for r in base):.3f} | {mean(metric(r,'kl') for r in base):.3f} | {mean(metric(r,'jaccard') for r in base):.3f} | {mean(metric(r,'retrieval') for r in base):.4f} |"
        )
        for agg in AGGS:
            for q in QUERIES[1:]:
                selected = groups[(length, policy(agg, q))]
                report.append(
                    f"| {length} | {agg} | {QLABEL[q]} | {mean(metric(r,'accept') for r in selected):.3f} | {mean(metric(r,'tps') for r in selected):.3f} | {mean(metric(r,'kl') for r in selected):.3f} | {mean(metric(r,'jaccard') for r in selected):.3f} | {mean(metric(r,'retrieval') for r in selected):.4f} |"
                )
    report += ["", "## 相对 Last-1 的配对差值（95% bootstrap CI）", ""]
    for length_index, length in enumerate(LENGTHS):
        report += [f"### {length}", ""]
        base = groups[(length, "attention_mean_q1")]
        for config_index, (agg, q) in enumerate((item for item in [(a, q) for a in AGGS for q in QUERIES[1:]])):
            selected = groups[(length, policy(agg, q))]
            accept = paired(selected, base, lambda r: metric(r, "accept"), 12000 + length_index * 100 + config_index)
            tps = paired(selected, base, lambda r: metric(r, "tps"), 13000 + length_index * 100 + config_index)
            kl = effects[("kl", length, agg, q)]
            fmt = lambda result: f"{result[0]:+.3f} [{result[1]:+.3f}, {result[2]:+.3f}]"
            report.append(f"- `{agg}, Q={QLABEL[q]}`：acceptance {fmt(accept)}；token/s {fmt(tps)}；root KL {fmt(kl)}")
        report.append("")
    report += [
        "## 结论",
        "",
        "**P1 的核心假设没有被支持：`Last-4 + decay` 没有在三个长度上稳定提高接受长度或吞吐。** 相对 Last-1，它在 4K/8K/16K 的接受长度差分别为 -0.052、+0.029、+0.015，三个 95% CI 均跨 0；吞吐差分别为 -1.756、+0.474、+0.264 token/s，其中只有 4K 的下降显著。它不能直接作为后续方法的默认基础。",
        "",
        "多 Query 确实改变了检索机制：mean/decay 在三个长度上几乎都提高了相邻检索集合的 Jaccard，相当于让 chunk 选择更稳定；但稳定性没有转化成草稿质量。4K 下多数多 Query 配置反而降低接受长度并增加 KL，说明简单聚合会抹平最后一个 query 中有用的尖锐注意力信号。8K 与 16K 的质量变化普遍很小且置信区间跨 0。",
        "",
        "16K 最好的趋势来自 max 聚合（Q=4 的接受长度 +0.033，吞吐 +0.312 token/s），但仍不显著。Q=8 与 All 的检索结果几乎等价：三个聚合下，60 个配对样本中只有 1–2 个样本的 chunk 选择不同，因此继续增加 query 数没有实际价值。下一步更值得测试的是保留每个 query 的尖峰后再做 union/rerank，或让系统按上下文长度自适应选择 Last-1 与 max，而不是继续扫更大的 m。",
        "",
        "阶段耗时中的少数红色异常格来自单样本计时尖峰（例如 8K max-All、16K mean-All/max-Q4）；其中位检索耗时仍约 0.12–0.13 ms/token。因此这些格子只作为系统噪声诊断，不解释为算法必然开销。",
        "",
        "## 产物",
        "",
        "- [质量与性能 SVG](p1_quality_performance.svg)",
        "- [机制诊断 SVG](p1_diagnostics.svg)",
        "- [作图与分析代码](../../../plot_full_multi_query.py)",
        "- [4K 原始结果](classic_full_mq_4k.json)",
        "- [8K 原始结果](classic_full_mq_8k.json)",
        "- [16K 原始结果](classic_full_mq_16k.json)",
        "",
    ]
    (DATA / "REPORT.md").write_text("\n".join(report))


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    groups = grouped(rows)
    validate(rows, groups)
    draw_main(rows, groups)
    effects = draw_diagnostics(rows, groups)
    write_report(rows, groups, effects)
    print(f"wrote P1 figures and report from {len(rows)} runs")


if __name__ == "__main__":
    main()
