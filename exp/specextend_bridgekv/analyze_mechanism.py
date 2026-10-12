#!/usr/bin/env python3
"""Summarize the three SpecExtend mechanism experiments with paired bootstrap CIs."""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results" / "rtx4090_reproduction" / "mechanism"


def load(name: str) -> list[dict]:
    return json.loads((DATA / name).read_text())["raw_runs"]


def mean_ci(values, seed=20260923, n_boot=20000):
    a = [float(x) for x in values]
    rng = random.Random(seed)
    boot = sorted(statistics.fmean(rng.choices(a, k=len(a))) for _ in range(n_boot))
    lo = boot[int(0.025 * n_boot)]
    hi = boot[int(0.975 * n_boot)]
    return statistics.fmean(a), lo, hi


def paired(a_rows, b_rows, metric):
    a = {r["sample_index"]: r[metric] for r in a_rows}
    b = {r["sample_index"]: r[metric] for r in b_rows}
    ids = sorted(a.keys() & b.keys())
    return mean_ci([a[i] - b[i] for i in ids])


def group(rows, *keys):
    out = defaultdict(list)
    for row in rows:
        out[tuple(row[k] for k in keys)].append(row)
    return out


def metric_mean(rows, metric):
    return statistics.fmean(r[metric] for r in rows)


def f(x):
    return f"{x:.3f}"


def ci_text(stats):
    m, lo, hi = stats
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def main():
    dense = load("dense.json")
    content = load("content.json")
    policy = load("policy.json") + load("policy_1024.json") + load("policy_1280.json")
    report = [
        "# SpecExtend 机制实验报告",
        "",
        "指标说明：`avg_accept_length` 是当前实现记录的平均接受长度，不等同于论文中可能采用的接受率或 τ；吞吐采用端到端 `wall_tokens_per_sec`。每个单元 20 个配对样本，括号内差值为配对 bootstrap 95% CI。",
        "",
        "## 1. 稠密长度扫描",
        "",
        "| 输入 token | Full 接受长度 | Attention K=32 | Δ接受长度 | Full tok/s | Attention tok/s | Δtok/s |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    dg = group(dense, "input_tokens", "policy")
    dense_summary = []
    for length in sorted({r["input_tokens"] for r in dense}):
        full, attn = dg[(length, "full")], dg[(length, "attention")]
        da = paired(attn, full, "avg_accept_length")
        dt = paired(attn, full, "wall_tokens_per_sec")
        report.append(
            f"| {length} | {f(metric_mean(full, 'avg_accept_length'))} | {f(metric_mean(attn, 'avg_accept_length'))} | {ci_text(da)} | "
            f"{f(metric_mean(full, 'wall_tokens_per_sec'))} | {f(metric_mean(attn, 'wall_tokens_per_sec'))} | {ci_text(dt)} |"
        )
        dense_summary.append({"length": length, "attn_minus_full_accept": da, "attn_minus_full_tps": dt})

    report += ["", "Full 基线相邻长度的配对变化：", ""]
    lengths = sorted({r["input_tokens"] for r in dense})
    for before, after in zip(lengths, lengths[1:]):
        delta = paired(dg[(after, "full")], dg[(before, "full")], "avg_accept_length")
        report.append(f"- {before}→{after}: {ci_text(delta)}")

    report += [
        "",
        "## 2. 内容/冗余控制",
        "",
        "| 内容条件 | Full 接受长度 | Attention K=32 | Δ接受长度 | Full tok/s | Attention tok/s | Δtok/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    cg = group(content, "condition", "policy")
    for cond in ("related", "duplicate", "unrelated"):
        full, attn = cg[(cond, "full")], cg[(cond, "attention")]
        report.append(
            f"| {cond} | {f(metric_mean(full, 'avg_accept_length'))} | {f(metric_mean(attn, 'avg_accept_length'))} | "
            f"{ci_text(paired(attn, full, 'avg_accept_length'))} | {f(metric_mean(full, 'wall_tokens_per_sec'))} | "
            f"{f(metric_mean(attn, 'wall_tokens_per_sec'))} | {ci_text(paired(attn, full, 'wall_tokens_per_sec'))} |"
        )
    report += ["", "相对 related 条件（Full 基线）的配对变化：", ""]
    for cond in ("duplicate", "unrelated"):
        report.append(
            f"- {cond}−related: 接受长度 {ci_text(paired(cg[(cond, 'full')], cg[('related', 'full')], 'avg_accept_length'))}; "
            f"吞吐 {ci_text(paired(cg[(cond, 'full')], cg[('related', 'full')], 'wall_tokens_per_sec'))}"
        )

    report += [
        "",
        "## 3. 检索策略与预算消融",
        "",
        "每个长度的 Full 行是共同基线；其他行相对 Full 给出配对差值。",
        "",
        "| 输入 token | 策略 | K | 接受长度 | Δ vs Full | tok/s | Δ vs Full |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    pg = group(policy, "input_tokens", "policy", "k")
    policy_summary = []
    for length in sorted({r["input_tokens"] for r in policy}):
        full = pg[(length, "full", None)]
        report.append(
            f"| {length} | full | — | {f(metric_mean(full, 'avg_accept_length'))} | — | "
            f"{f(metric_mean(full, 'wall_tokens_per_sec'))} | — |"
        )
        for k in (8, 16, 24, 32):
            for pol in ("random", "recent", "attention"):
                rows = pg[(length, pol, k)]
                da = paired(rows, full, "avg_accept_length")
                dt = paired(rows, full, "wall_tokens_per_sec")
                report.append(
                    f"| {length} | {pol} | {k} | {f(metric_mean(rows, 'avg_accept_length'))} | {ci_text(da)} | "
                    f"{f(metric_mean(rows, 'wall_tokens_per_sec'))} | {ci_text(dt)} |"
                )
                policy_summary.append({"length": length, "policy": pol, "k": k, "accept": metric_mean(rows, "avg_accept_length"), "tps": metric_mean(rows, "wall_tokens_per_sec"), "delta_accept": da, "delta_tps": dt})

    # Direct policy comparisons at the same length and K.
    report += ["", "Attention 对同预算基线的配对差值（接受长度）：", ""]
    for length in sorted({r["input_tokens"] for r in policy}):
        for k in (8, 16, 24, 32):
            attn = pg[(length, "attention", k)]
            dr = paired(attn, pg[(length, "random", k)], "avg_accept_length")
            dc = paired(attn, pg[(length, "recent", k)], "avg_accept_length")
            report.append(f"- L={length}, K={k}: attention−random {ci_text(dr)}; attention−recent {ci_text(dc)}")

    report += [
        "",
        "## 结论",
        "",
        "1. Full-cache 草稿模型的接受长度在约 640 token 达到局部峰值，之后从 640 到 1280 持续下降；这比只比较 512/768/1K 更可靠地定位了退化起点。",
        "2. ‘重复本身有害’不成立：精确重复显著提高接受长度与吞吐；不相关文本则显著降低接受长度。因此更合理的研究变量是内容干扰/注意力稀释，而不是笼统的信息冗余。",
        "3. 注意力选块不是普通的 cache 截断：相同预算下，它在所有长度和绝大多数预算上都优于 random/recent，优势在 1024/1280 token、K=16/24 时最大。",
        "4. 最有价值的工作点是 K=24：在 1280 token 上，attention 比 full 的接受长度高 0.635，吞吐与 full 统计持平；K=32 反而不如 K=24，说明保留更多上下文并非总是更好。",
        "5. 目前还不能宣称端到端加速：除 1280/K=24 与 full 持平外，多数 attention 设置仍更慢。下一阶段应优化检索频率、注意力统计与 KV 搬运开销，并在更长上下文验证收益是否转为真实加速。",
    ]

    (DATA / "REPORT.md").write_text("\n".join(report) + "\n")
    machine = {"dense": dense_summary, "policy": policy_summary}
    (DATA / "summary.json").write_text(json.dumps(machine, indent=2) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
