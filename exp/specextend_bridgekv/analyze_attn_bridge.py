#!/usr/bin/env python3
"""Analyze and visualize an AttnBridgeKV evaluation JSON.

The evaluation runner randomizes policy order within a document, so the
statistical unit here is the document (``sample_index``), not a verification
round.  Confidence intervals for policy differences resample paired documents.
The script is deliberately tolerant of interrupted runs: completed cells are
plotted, while the report and every figure are marked PARTIAL whenever the
expected policy-by-length-by-sample grid is incomplete or ``--partial`` is set.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Optional

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError as error:
    raise SystemExit(
        "AttnBridgeKV plotting requires numpy and a complete matplotlib install "
        f"(missing module: {error.name}). Install with: "
        "python -m pip install numpy matplotlib"
    ) from error


POLICY_ORDER = [
    "recent_native",
    "attention_native",
    "recent_bridge_a25",
    "recent_bridge_a50",
    "attention_bridge_a10",
    "attention_bridge_a25",
    "attention_bridge_value_a25",
    "attention_bridge_shuffle_a25",
    "attention_bridge_last_a50",
    "attention_bridge_a50",
    "attention_bridge_a100",
    "attention_bridge_shuffle_a50",
]

POLICY_LABELS = {
    "recent_native": "Recent\nnative",
    "attention_native": "Attn\nnative",
    "recent_bridge_a25": "Recent B\na=.25",
    "recent_bridge_a50": "Recent B\na=.5",
    "attention_bridge_last_a50": "Attn B last\na=.5",
    "attention_bridge_a10": "Attn B\na=.1",
    "attention_bridge_a25": "Attn B\na=.25",
    "attention_bridge_value_a25": "Attn V-only\na=.25",
    "attention_bridge_shuffle_a25": "Attn B shuffle\na=.25",
    "attention_bridge_a50": "Attn B\na=.5",
    "attention_bridge_a100": "Attn B\na=1.0",
    "attention_bridge_shuffle_a50": "Attn B shuffle\na=.5",
}

POLICY_COLORS = {
    "recent_native": "#8C8C8C",
    "attention_native": "#377EB8",
    "recent_bridge_a25": "#FDB863",
    "recent_bridge_a50": "#F0A202",
    "attention_bridge_last_a50": "#80B918",
    "attention_bridge_a10": "#66C2A5",
    "attention_bridge_a25": "#1B9E77",
    "attention_bridge_value_a25": "#E6AB02",
    "attention_bridge_shuffle_a25": "#E76F51",
    "attention_bridge_a50": "#2A9D8F",
    "attention_bridge_a100": "#7B2CBF",
    "attention_bridge_shuffle_a50": "#D1495B",
}

LENGTH_COLORS = ["#377EB8", "#E67E22", "#2A9D8F", "#7B2CBF", "#D1495B"]
STAGE_COLORS = {
    "draft": "#4C78A8",
    "verification": "#F58518",
    "retrieval": "#54A24B",
    "bridge": "#B279A2",
}

TARGET_EXACT_TOKENS = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create AttnBridgeKV figures, summary JSON, and a Chinese report."
    )
    parser.add_argument("--input", required=True, help="Raw JSON from evaluate mode")
    parser.add_argument(
        "--output-dir",
        help="Artifact directory (default: the raw JSON directory)",
    )
    parser.add_argument(
        "--reference-policy",
        default="attention_native",
        help="Reference used for paired deltas and greedy equality",
    )
    parser.add_argument(
        "--target-reference",
        help=(
            "Optional canonical target-AR JSON. Its raw_runs are matched by "
            "(length_bin_tokens, sample_index) and become the primary source "
            "for full 128-token exactness."
        ),
    )
    parser.add_argument(
        "--target-margin-diagnostic",
        help=(
            "Optional canonical target-AR JSON containing top1_margin_list. "
            "For target-reference failures, report the first mismatch position "
            "and the target top1-minus-top2 logit margin at that position."
        ),
    )
    parser.add_argument(
        "--bootstrap", type=int, default=10_000, help="Bootstrap replicates"
    )
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument(
        "--training-history",
        help=(
            "Optional train_history.json. When runtime KV cosine/NMSE are absent, "
            "the diagnostics figure uses mapper train/validation curves."
        ),
    )
    parser.add_argument(
        "--partial",
        action="store_true",
        help="Force PARTIAL labeling even when the expected grid is complete",
    )
    parser.add_argument(
        "--report-name", default="REPORT.md", help="Markdown report filename"
    )
    return parser.parse_args()


def finite_number(value) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def percentile_interval(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    return (
        float(values.mean()),
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def bootstrap_mean(
    values: Iterable[float], seed: int, replicates: int
) -> Optional[tuple[float, float, float]]:
    array = np.asarray([float(value) for value in values], dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return None
    if len(array) == 1:
        value = float(array[0])
        return value, value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(replicates, len(array)))
    bootstrap_values = array[indices].mean(axis=1)
    mean, low, high = percentile_interval(bootstrap_values)
    # Report the observed mean, not the tiny Monte-Carlo-shifted bootstrap mean.
    return float(array.mean()), low, high


def rows_by_sample(rows: Iterable[dict]) -> dict[int, dict]:
    return {int(row["sample_index"]): row for row in rows}


def paired_difference(
    left_rows: Iterable[dict],
    right_rows: Iterable[dict],
    metric: Callable[[dict], Optional[float]],
    seed: int,
    replicates: int,
) -> tuple[Optional[tuple[float, float, float]], int]:
    left = rows_by_sample(left_rows)
    right = rows_by_sample(right_rows)
    differences = []
    for sample in sorted(left.keys() & right.keys()):
        a = finite_number(metric(left[sample]))
        b = finite_number(metric(right[sample]))
        if a is not None and b is not None:
            differences.append(a - b)
    return bootstrap_mean(differences, seed, replicates), len(differences)


def grouped(rows: Iterable[dict]) -> dict[tuple[int, str], list[dict]]:
    result: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        result[(int(row["length_bin_tokens"]), str(row["policy"]))].append(row)
    return result


def policy_sort_key(policy: str) -> tuple[int, str]:
    try:
        return POLICY_ORDER.index(policy), policy
    except ValueError:
        return len(POLICY_ORDER), policy


def label_for_policy(policy: str) -> str:
    return POLICY_LABELS.get(policy, policy.replace("_", "\n"))


def color_for_policy(policy: str) -> str:
    return POLICY_COLORS.get(policy, "#666666")


def length_label(length: int, rows: list[dict]) -> str:
    labels = [str(row.get("length_label", "")) for row in rows if row.get("length_label")]
    if labels:
        return labels[0]
    return f"{length // 1024}K" if length % 1024 == 0 else str(length)


def acceptance_length(row: dict) -> Optional[float]:
    return finite_number(row.get("avg_accept_length"))


def acceptance_share(row: dict) -> Optional[float]:
    """Accepted draft tokens / all committed tokens on the verified path."""
    values = row.get("accept_length_list")
    if not isinstance(values, list) or not values:
        values = [item.get("accept_length") for item in row.get("rounds", [])]
    values = [finite_number(value) for value in values]
    accepted = [value for value in values if value is not None]
    if not accepted:
        mean_accept = acceptance_length(row)
        return None if mean_accept is None else mean_accept / (mean_accept + 1.0)
    denominator = sum(accepted) + len(accepted)
    return sum(accepted) / denominator if denominator else None


def root_kl(row: dict) -> Optional[float]:
    return finite_number(row.get("mean_root_target_draft_kl"))


def wall_tps(row: dict) -> Optional[float]:
    return finite_number(row.get("wall_tokens_per_sec"))


def wall_ms_per_token(row: dict) -> Optional[float]:
    value = wall_tps(row)
    return None if value is None or value <= 0 else 1000.0 / value


def stage_ms_per_token(stage: str) -> Callable[[dict], Optional[float]]:
    def metric(row: dict) -> Optional[float]:
        generated = finite_number(row.get("total_generated"))
        if generated is None or generated <= 0:
            return None
        if stage == "bridge":
            milliseconds = finite_number(row.get("bridge_map_time_ms"))
        else:
            milliseconds = finite_number((row.get("stage_time_ms") or {}).get(stage))
        return None if milliseconds is None else milliseconds / generated

    return metric


def bridge_metric(field: str, scale: float = 1.0) -> Callable[[dict], Optional[float]]:
    def metric(row: dict) -> Optional[float]:
        value = finite_number(row.get(field))
        return None if value is None else value / scale

    return metric


def output_ids(row: dict) -> Optional[list[int]]:
    value = row.get("output_token_ids")
    return value if isinstance(value, list) else None


def prefix_exact(row: dict, reference: dict) -> Optional[float]:
    candidate = output_ids(row)
    target = output_ids(reference)
    if candidate is None or target is None:
        return None
    count = min(len(candidate), len(target))
    if count == 0:
        return None
    return float(candidate[:count] == target[:count])


def full_output_exact(row: dict, reference: dict) -> Optional[float]:
    candidate = output_ids(row)
    target = output_ids(reference)
    if candidate is None or target is None:
        return None
    return float(candidate == target)


def fixed_token_exact(
    row: dict, reference: dict, token_count: int = TARGET_EXACT_TOKENS
) -> Optional[float]:
    """Return exactness on a complete, fixed-length generated continuation.

    A short continuation is not silently counted as either a success or a
    failure: it is ineligible for the advertised ``token_count``-token check.
    Current evaluation/reference runners cap their scored output at 128, so a
    valid value normally compares the complete stored lists.
    """
    candidate = output_ids(row)
    target = output_ids(reference)
    if candidate is None or target is None:
        return None
    if len(candidate) < token_count or len(target) < token_count:
        return None
    return float(candidate[:token_count] == target[:token_count])


def first_fixed_token_mismatch(
    row: dict, reference: dict, token_count: int = TARGET_EXACT_TOKENS
) -> Optional[int]:
    candidate = output_ids(row)
    target = output_ids(reference)
    if candidate is None or target is None:
        return None
    if len(candidate) < token_count or len(target) < token_count:
        return None
    return next(
        (
            position
            for position, (candidate_id, target_id) in enumerate(
                zip(candidate[:token_count], target[:token_count])
            )
            if candidate_id != target_id
        ),
        None,
    )


def exactness_values(
    rows: list[dict], reference_rows: list[dict], full: bool = False
) -> list[float]:
    references = rows_by_sample(reference_rows)
    function = full_output_exact if full else prefix_exact
    values = []
    for row in rows:
        reference = references.get(int(row["sample_index"]))
        if reference is None:
            continue
        value = function(row, reference)
        if value is not None:
            values.append(value)
    return values


def target_reference_map(rows: Iterable[dict]) -> dict[tuple[int, int], dict]:
    result = {}
    for index, row in enumerate(rows):
        missing = {"length_bin_tokens", "sample_index", "output_token_ids"} - row.keys()
        if missing:
            raise SystemExit(
                f"target reference raw_runs[{index}] is missing {sorted(missing)}"
            )
        key = (int(row["length_bin_tokens"]), int(row["sample_index"]))
        if key in result:
            raise SystemExit(
                "Duplicate (length_bin_tokens, sample_index) rows in target reference"
            )
        result[key] = row
    return result


def target_margin_map(rows: Iterable[dict]) -> dict[tuple[int, int], dict]:
    result = target_reference_map(rows)
    for key, row in result.items():
        margins = row.get("top1_margin_list")
        if not isinstance(margins, list):
            raise SystemExit(
                "target margin diagnostic run "
                f"{key} is missing top1_margin_list"
            )
    return result


def build_target_mismatch_diagnostic(
    rows: list[dict],
    references: dict[tuple[int, int], dict],
    margin_references: Optional[dict[tuple[int, int], dict]],
) -> dict:
    failures = []
    for row in rows:
        length = int(row["length_bin_tokens"])
        sample = int(row["sample_index"])
        reference = references.get((length, sample))
        if reference is None:
            continue
        position = first_fixed_token_mismatch(row, reference)
        if position is None:
            continue
        candidate_ids = output_ids(row) or []
        target_ids = output_ids(reference) or []
        diagnostic_reference = (
            margin_references.get((length, sample))
            if margin_references is not None
            else None
        )
        margin = None
        diagnostic_matches_reference = False
        if diagnostic_reference is not None:
            diagnostic_ids = output_ids(diagnostic_reference)
            diagnostic_matches_reference = bool(
                diagnostic_ids is not None
                and len(diagnostic_ids) >= TARGET_EXACT_TOKENS
                and diagnostic_ids[:TARGET_EXACT_TOKENS]
                == target_ids[:TARGET_EXACT_TOKENS]
            )
            margins = diagnostic_reference.get("top1_margin_list")
            if (
                diagnostic_matches_reference
                and isinstance(margins, list)
                and position < len(margins)
            ):
                margin = finite_number(margins[position])
        failures.append(
            {
                "length_bin_tokens": length,
                "sample_index": sample,
                "policy": str(row["policy"]),
                "first_mismatch_position_zero_based": position,
                "target_token_id": int(target_ids[position]),
                "candidate_token_id": int(candidate_ids[position]),
                "target_top1_top2_margin": margin,
                "margin_run_matches_target_reference": diagnostic_matches_reference,
            }
        )
    covered = sum(item["target_top1_top2_margin"] is not None for item in failures)
    return {
        "token_count": TARGET_EXACT_TOKENS,
        "position_indexing": "zero_based_generated_continuation",
        "failure_count": len(failures),
        "covered_failure_count": covered,
        "all_failures_covered": bool(failures) and covered == len(failures),
        "failures": failures,
    }


def target_exactness_values(
    rows: Iterable[dict],
    length: int,
    references: dict[tuple[int, int], dict],
    token_count: int = TARGET_EXACT_TOKENS,
) -> list[float]:
    values = []
    for row in rows:
        reference = references.get((length, int(row["sample_index"])))
        if reference is None:
            continue
        value = fixed_token_exact(row, reference, token_count)
        if value is not None:
            values.append(value)
    return values


def cell_stats(
    rows: list[dict],
    metric: Callable[[dict], Optional[float]],
    seed: int,
    replicates: int,
) -> Optional[tuple[float, float, float]]:
    values = [metric(row) for row in rows]
    return bootstrap_mean(
        [value for value in values if value is not None], seed, replicates
    )


def errorbar_bars(
    ax,
    groups: dict[tuple[int, str], list[dict]],
    lengths: list[int],
    policies: list[str],
    metric: Callable[[dict], Optional[float]],
    ylabel: str,
    title: str,
    seed: int,
    replicates: int,
    percentage: bool = False,
) -> None:
    x = np.arange(len(policies), dtype=float)
    width = min(0.78 / max(len(lengths), 1), 0.56)
    plotted = False
    for length_index, length in enumerate(lengths):
        offset = (length_index - (len(lengths) - 1) / 2.0) * width
        for policy_index, policy in enumerate(policies):
            stats = cell_stats(
                groups.get((length, policy), []),
                metric,
                seed + length_index * 1009 + policy_index * 37,
                replicates,
            )
            if stats is None:
                continue
            mean, low, high = stats
            scale = 100.0 if percentage else 1.0
            value = mean * scale
            yerr = np.array([[max(0.0, mean - low) * scale], [max(0.0, high - mean) * scale]])
            color = (
                color_for_policy(policy)
                if len(lengths) == 1
                else LENGTH_COLORS[length_index % len(LENGTH_COLORS)]
            )
            ax.bar(
                x[policy_index] + offset,
                value,
                width * 0.9,
                color=color,
                alpha=0.9,
                yerr=yerr,
                capsize=3,
                linewidth=0.5,
                edgecolor="white",
                label=(
                    length_label(length, groups[(length, policy)])
                    if len(lengths) > 1 and policy_index == 0
                    else None
                ),
            )
            plotted = True
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [label_for_policy(policy) for policy in policies],
        fontsize=7,
        rotation=25,
        ha="right",
    )
    ax.grid(axis="y", alpha=0.22)
    if len(lengths) > 1 and plotted:
        ax.legend(title="Nominal context", fontsize=8)
    if not plotted:
        unavailable_panel(ax, title)


def paired_delta_bars(
    ax,
    groups: dict[tuple[int, str], list[dict]],
    lengths: list[int],
    policies: list[str],
    reference: str,
    metric: Callable[[dict], Optional[float]],
    ylabel: str,
    title: str,
    seed: int,
    replicates: int,
) -> None:
    compared = [policy for policy in policies if policy != reference]
    x = np.arange(len(compared), dtype=float)
    width = min(0.78 / max(len(lengths), 1), 0.56)
    plotted = False
    for length_index, length in enumerate(lengths):
        reference_rows = groups.get((length, reference), [])
        offset = (length_index - (len(lengths) - 1) / 2.0) * width
        for policy_index, policy in enumerate(compared):
            stats, count = paired_difference(
                groups.get((length, policy), []),
                reference_rows,
                metric,
                seed + length_index * 1009 + policy_index * 37,
                replicates,
            )
            if stats is None:
                continue
            mean, low, high = stats
            color = (
                color_for_policy(policy)
                if len(lengths) == 1
                else LENGTH_COLORS[length_index % len(LENGTH_COLORS)]
            )
            ax.errorbar(
                x[policy_index] + offset,
                mean,
                yerr=np.array([[max(0.0, mean - low)], [max(0.0, high - mean)]]),
                fmt="o",
                markersize=7,
                capsize=4,
                color=color,
                label=(
                    f"{length_label(length, groups[(length, policy)])} (n={count})"
                    if len(lengths) > 1 and policy_index == 0
                    else None
                ),
            )
            plotted = True
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [label_for_policy(policy) for policy in compared],
        fontsize=7,
        rotation=25,
        ha="right",
    )
    ax.grid(axis="y", alpha=0.22)
    if len(lengths) > 1 and plotted:
        ax.legend(fontsize=8)
    if not plotted:
        unavailable_panel(ax, title)


def exactness_bars(
    ax,
    groups: dict[tuple[int, str], list[dict]],
    lengths: list[int],
    policies: list[str],
    reference: str,
    seed: int,
    replicates: int,
    target_references: Optional[dict[tuple[int, int], dict]] = None,
) -> None:
    x = np.arange(len(policies), dtype=float)
    width = min(0.78 / max(len(lengths), 1), 0.56)
    plotted = False
    for length_index, length in enumerate(lengths):
        reference_rows = groups.get((length, reference), [])
        offset = (length_index - (len(lengths) - 1) / 2.0) * width
        for policy_index, policy in enumerate(policies):
            cell = groups.get((length, policy), [])
            if target_references is not None:
                values = target_exactness_values(cell, length, target_references)
            else:
                values = exactness_values(cell, reference_rows, full=True)
            stats = bootstrap_mean(
                values,
                seed + length_index * 1009 + policy_index * 37,
                replicates,
            )
            if stats is None:
                continue
            mean, low, high = stats
            color = (
                color_for_policy(policy)
                if len(lengths) == 1
                else LENGTH_COLORS[length_index % len(LENGTH_COLORS)]
            )
            ax.bar(
                x[policy_index] + offset,
                100.0 * mean,
                width * 0.9,
                color=color,
                yerr=np.array(
                    [[100.0 * max(0.0, mean - low)], [100.0 * max(0.0, high - mean)]]
                ),
                capsize=3,
                label=(
                    length_label(length, groups[(length, policy)])
                    if len(lengths) > 1 and policy_index == 0
                    else None
                ),
            )
            plotted = True
    ax.set_ylim(0, 105)
    title = (
        f"F  Canonical target-AR {TARGET_EXACT_TOKENS}-token exactness"
        if target_references is not None
        else "F  Greedy full-output exactness vs policy reference"
    )
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel("Exact pairs (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [label_for_policy(policy) for policy in policies],
        fontsize=7,
        rotation=25,
        ha="right",
    )
    ax.grid(axis="y", alpha=0.22)
    if len(lengths) > 1 and plotted:
        ax.legend(title="Nominal context", fontsize=8)
    if not plotted:
        unavailable_panel(ax, title)


def unavailable_panel(ax, title: str) -> None:
    ax.clear()
    ax.set_title(title, loc="left", fontweight="bold")
    ax.text(
        0.5,
        0.5,
        "Diagnostic not recorded",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="#666666",
    )
    ax.set_xticks([])
    ax.set_yticks([])


def status_footer(
    payload: dict,
    groups: dict[tuple[int, str], list[dict]],
    lengths: list[int],
    policies: list[str],
    partial: bool,
) -> str:
    config = payload.get("config") or {}
    counts = [len(groups.get((length, policy), [])) for length in lengths for policy in policies]
    n_text = str(counts[0]) if counts and len(set(counts)) == 1 else f"{min(counts)}-{max(counts)}"
    fixed = (
        f"document n/cell={n_text}; nodes={config.get('tree_nodes', '?')}; "
        f"K={config.get('cache_top_k', '?')}; chunk={config.get('chunk_size', '?')}; "
        f"retrieve every={config.get('retrieve_every', '?')}; "
        "document bootstrap 95% CI (paired for deltas)"
    )
    return ("PARTIAL · " if partial else "") + fixed


def finish_figure(fig, path: Path, footer: str, partial: bool) -> None:
    layout = fig.get_layout_engine()
    if layout is not None and hasattr(layout, "set"):
        # Reserve a real caption band so long rotated policy labels do not
        # collide with the settings/sample-count disclosure.
        layout.set(rect=(0.0, 0.06, 1.0, 0.94))
    fig.text(0.01, 0.008, footer, fontsize=9, color="#555555")
    if partial:
        fig.text(
            0.99,
            0.985,
            "PARTIAL / 部分结果",
            ha="right",
            va="top",
            fontsize=12,
            color="#B00020",
            fontweight="bold",
        )
    for suffix in ("png", "svg"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_quality(
    payload: dict,
    groups: dict[tuple[int, str], list[dict]],
    lengths: list[int],
    policies: list[str],
    reference: str,
    output_dir: Path,
    seed: int,
    replicates: int,
    partial: bool,
    target_references: Optional[dict[tuple[int, int], dict]] = None,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(
        "AttnBridgeKV: proposal quality and greedy exactness",
        fontsize=18,
        fontweight="bold",
    )
    errorbar_bars(
        axes[0, 0], groups, lengths, policies, acceptance_length,
        "Accepted draft tokens / round", "A  Mean acceptance length",
        seed + 10, replicates,
    )
    paired_delta_bars(
        axes[0, 1], groups, lengths, policies, reference, acceptance_length,
        f"Delta vs {reference}", "B  Paired acceptance delta",
        seed + 20, replicates,
    )
    errorbar_bars(
        axes[0, 2], groups, lengths, policies, acceptance_share,
        "Accepted path tokens (%)", "C  Verified-path acceptance share",
        seed + 30, replicates, percentage=True,
    )
    errorbar_bars(
        axes[1, 0], groups, lengths, policies, root_kl,
        "KL(target || draft)", "D  Root distribution KL",
        seed + 40, replicates,
    )
    paired_delta_bars(
        axes[1, 1], groups, lengths, policies, reference, root_kl,
        f"Delta KL vs {reference}", "E  Paired root-KL delta",
        seed + 50, replicates,
    )
    exactness_bars(
        axes[1, 2], groups, lengths, policies, reference,
        seed + 60, replicates, target_references,
    )
    finish_figure(
        fig,
        output_dir / "attn_bridge_quality",
        status_footer(payload, groups, lengths, policies, partial)
        + (
            f"; exactness uses fixed canonical target AR ({TARGET_EXACT_TOKENS} tokens)"
            if target_references is not None
            else f"; exactness reference={reference}"
        ),
        partial,
    )


def plot_stage_latency(
    ax,
    groups: dict[tuple[int, str], list[dict]],
    lengths: list[int],
    policies: list[str],
    seed: int,
    replicates: int,
) -> None:
    categories = [(length, policy) for length in lengths for policy in policies]
    x = np.arange(len(categories), dtype=float)
    stages = ["draft", "verification", "retrieval", "bridge"]
    width = 0.19
    plotted = False
    for stage_index, stage in enumerate(stages):
        means, lows, highs, positions = [], [], [], []
        for category_index, (length, policy) in enumerate(categories):
            stats = cell_stats(
                groups.get((length, policy), []),
                stage_ms_per_token(stage),
                seed + stage_index * 1009 + category_index * 37,
                replicates,
            )
            if stats is None:
                continue
            mean, low, high = stats
            positions.append(x[category_index] + (stage_index - 1.5) * width)
            means.append(mean)
            lows.append(max(0.0, mean - low))
            highs.append(max(0.0, high - mean))
        if means:
            ax.bar(
                positions,
                means,
                width * 0.92,
                color=STAGE_COLORS[stage],
                yerr=np.asarray([lows, highs]),
                capsize=2,
                label=("Bridge map" if stage == "bridge" else stage.title()),
            )
            plotted = True
    labels = []
    for length, policy in categories:
        nominal = length_label(length, groups.get((length, policy), []))
        prefix = f"{nominal} · " if len(lengths) > 1 else ""
        labels.append(prefix + label_for_policy(policy).replace("\n", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=7)
    ax.set_ylabel("CUDA stage ms / generated token")
    ax.set_title("D  Stage latency (overlapping regions)", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.22)
    if plotted:
        ax.legend(ncol=4, fontsize=8)
    else:
        unavailable_panel(ax, "D  Stage latency")


def plot_performance(
    payload: dict,
    groups: dict[tuple[int, str], list[dict]],
    lengths: list[int],
    policies: list[str],
    reference: str,
    output_dir: Path,
    seed: int,
    replicates: int,
    partial: bool,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)
    fig.suptitle(
        "AttnBridgeKV: end-to-end performance and stage latency",
        fontsize=18,
        fontweight="bold",
    )
    errorbar_bars(
        axes[0, 0], groups, lengths, policies, wall_tps,
        "Generated token/s", "A  End-to-end throughput",
        seed + 110, replicates,
    )
    paired_delta_bars(
        axes[0, 1], groups, lengths, policies, reference, wall_tps,
        f"Delta token/s vs {reference}", "B  Paired throughput delta",
        seed + 120, replicates,
    )
    errorbar_bars(
        axes[1, 0], groups, lengths, policies, wall_ms_per_token,
        "Wall-clock ms / generated token", "C  End-to-end latency",
        seed + 130, replicates,
    )
    plot_stage_latency(
        axes[1, 1], groups, lengths, policies,
        seed + 140, replicates,
    )
    finish_figure(
        fig,
        output_dir / "attn_bridge_performance",
        status_footer(payload, groups, lengths, policies, partial)
        + "; retrieval is part of draft, bridge map is part of retrieval",
        partial,
    )


def diagnostic_bars(
    ax,
    groups: dict[tuple[int, str], list[dict]],
    lengths: list[int],
    policies: list[str],
    metrics: list[tuple[str, Callable[[dict], Optional[float]], str]],
    ylabel: str,
    title: str,
    seed: int,
    replicates: int,
) -> None:
    categories = [(length, policy) for length in lengths for policy in policies]
    x = np.arange(len(categories), dtype=float)
    width = min(0.76 / max(len(metrics), 1), 0.55)
    plotted = False
    for metric_index, (metric_label, metric, color) in enumerate(metrics):
        for category_index, (length, policy) in enumerate(categories):
            stats = cell_stats(
                groups.get((length, policy), []),
                metric,
                seed + metric_index * 1009 + category_index * 37,
                replicates,
            )
            if stats is None:
                continue
            mean, low, high = stats
            position = x[category_index] + (metric_index - (len(metrics) - 1) / 2) * width
            ax.bar(
                position,
                mean,
                width * 0.92,
                color=color,
                yerr=np.array([[max(0.0, mean - low)], [max(0.0, high - mean)]]),
                capsize=2,
                label=metric_label if category_index == 0 else None,
            )
            plotted = True
    labels = []
    for length, policy in categories:
        nominal = length_label(length, groups.get((length, policy), []))
        prefix = f"{nominal} · " if len(lengths) > 1 else ""
        labels.append(prefix + label_for_policy(policy).replace("\n", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.22)
    if len(metrics) > 1 and plotted:
        ax.legend(fontsize=8)
    if not plotted:
        unavailable_panel(ax, title)


def rejection_distribution(
    ax,
    groups: dict[tuple[int, str], list[dict]],
    lengths: list[int],
    policies: list[str],
) -> None:
    found = False
    for length_index, length in enumerate(lengths):
        for policy in policies:
            values = []
            for row in groups.get((length, policy), []):
                values.extend(
                    int(item["rejected_position"])
                    for item in row.get("rounds", [])
                    if finite_number(item.get("rejected_position")) is not None
                )
            if not values:
                continue
            found = True
            maximum = max(values)
            support = np.arange(1, maximum + 1)
            cdf = np.asarray([sum(value <= point for value in values) / len(values) for point in support])
            label = label_for_policy(policy).replace("\n", " ")
            if len(lengths) > 1:
                label = f"{length_label(length, groups[(length, policy)])} · {label}"
            ax.step(
                support,
                cdf,
                where="post",
                color=color_for_policy(policy),
                linestyle=["-", "--", ":", "-."][length_index % 4],
                alpha=0.9,
                label=label,
            )
    ax.set_title("E  First-rejection position distribution", loc="left", fontweight="bold")
    ax.set_xlabel("1-indexed first rejected position")
    ax.set_ylabel("Empirical CDF")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.22)
    if found:
        ax.legend(fontsize=7, ncol=2)
    else:
        unavailable_panel(ax, "E  First-rejection position distribution")


def selected_position_distribution(
    ax,
    groups: dict[tuple[int, str], list[dict]],
    lengths: list[int],
    policies: list[str],
) -> None:
    found = False
    bins = np.linspace(0, 1, 21)
    centers = (bins[:-1] + bins[1:]) / 2
    for length_index, length in enumerate(lengths):
        for policy in policies:
            values = []
            for row in groups.get((length, policy), []):
                for event in row.get("retrieval_events", []):
                    values.extend(
                        value
                        for raw in event.get("selected_chunk_centers_normalized", [])
                        if (value := finite_number(raw)) is not None and 0 <= value <= 1
                    )
            if not values:
                continue
            found = True
            histogram, _ = np.histogram(values, bins=bins)
            probability = histogram / max(histogram.sum(), 1)
            label = label_for_policy(policy).replace("\n", " ")
            if len(lengths) > 1:
                label = f"{length_label(length, groups[(length, policy)])} · {label}"
            ax.plot(
                centers,
                probability,
                color=color_for_policy(policy),
                linestyle=["-", "--", ":", "-."][length_index % 4],
                marker="o",
                markersize=2.5,
                alpha=0.85,
                label=label,
            )
    ax.set_title("F  Selected-chunk position distribution", loc="left", fontweight="bold")
    ax.set_xlabel("Normalized chunk-center position in prefix")
    ax.set_ylabel("Selection share per bin")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.22)
    if found:
        ax.legend(fontsize=7, ncol=2)
    else:
        unavailable_panel(ax, "F  Selected-chunk position distribution")


def runtime_metric_available(
    groups: dict[tuple[int, str], list[dict]],
    lengths: list[int],
    policies: list[str],
    field: str,
) -> bool:
    return any(
        finite_number(row.get(field)) is not None
        for length in lengths
        for policy in policies
        for row in groups.get((length, policy), [])
    )


def training_rows(training_history: Optional[dict]) -> list[dict]:
    if not isinstance(training_history, dict):
        return []
    rows = training_history.get("history")
    return rows if isinstance(rows, list) else []


def plot_training_curve(
    ax,
    training_history: Optional[dict],
    metric: str,
) -> None:
    rows = training_rows(training_history)
    epochs = [finite_number(row.get("epoch")) for row in rows]
    valid = [index for index, epoch in enumerate(epochs) if epoch is not None]
    if not valid:
        unavailable_panel(ax, "Mapper training diagnostic")
        return
    x = np.asarray([epochs[index] for index in valid], dtype=float)
    if metric == "cosine":
        train = [
            None
            if finite_number(rows[index].get("train_cosine_loss")) is None
            else 1.0 - float(rows[index]["train_cosine_loss"])
            for index in valid
        ]
        validation = [
            None
            if finite_number(rows[index].get("val_cosine_loss")) is None
            else 1.0 - float(rows[index]["val_cosine_loss"])
            for index in valid
        ]
        title = "A  Mapper cosine from training history"
        ylabel = "Mean cosine(mapped, native)"
    else:
        train = [finite_number(rows[index].get("train_nmse")) for index in valid]
        validation = [finite_number(rows[index].get("val_nmse")) for index in valid]
        title = "B  Mapper NMSE from training history"
        ylabel = "Normalized MSE"
    plotted = False
    for label, values, color in (
        ("Train", train, "#4C78A8"),
        ("Validation", validation, "#E45756"),
    ):
        pairs = [(epoch, value) for epoch, value in zip(x, values) if value is not None]
        if not pairs:
            continue
        ax.plot(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            color=color,
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=label,
        )
        plotted = True
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)
    if plotted:
        ax.legend(fontsize=8)
    else:
        unavailable_panel(ax, title)


def plot_diagnostics(
    payload: dict,
    groups: dict[tuple[int, str], list[dict]],
    lengths: list[int],
    policies: list[str],
    output_dir: Path,
    seed: int,
    replicates: int,
    partial: bool,
    training_history: Optional[dict] = None,
) -> None:
    bridge_policies = [
        policy
        for policy in policies
        if "bridge" in policy
        or any(
            finite_number(row.get("bridge_mapped_tokens")) not in (None, 0.0)
            for length in lengths
            for row in groups.get((length, policy), [])
        )
    ]
    if not bridge_policies:
        bridge_policies = policies
    fig, axes = plt.subplots(2, 3, figsize=(19, 11), constrained_layout=True)
    fig.suptitle(
        "AttnBridgeKV: mechanism diagnostics",
        fontsize=18,
        fontweight="bold",
    )
    if runtime_metric_available(
        groups, lengths, bridge_policies, "bridge_mean_kv_cosine"
    ):
        diagnostic_bars(
            axes[0, 0], groups, lengths, bridge_policies,
            [("Cosine", bridge_metric("bridge_mean_kv_cosine"), "#2A9D8F")],
            "Cosine(mapped, native)", "A  Runtime mapped/native KV cosine",
            seed + 210, replicates,
        )
    elif training_rows(training_history):
        plot_training_curve(axes[0, 0], training_history, "cosine")
    else:
        unavailable_panel(axes[0, 0], "A  Mapped/native KV cosine")
    if runtime_metric_available(
        groups, lengths, bridge_policies, "bridge_mean_kv_nmse"
    ):
        diagnostic_bars(
            axes[0, 1], groups, lengths, bridge_policies,
            [("NMSE", bridge_metric("bridge_mean_kv_nmse"), "#D1495B")],
            "Normalized MSE", "B  Runtime mapped/native KV NMSE",
            seed + 220, replicates,
        )
    elif training_rows(training_history):
        plot_training_curve(axes[0, 1], training_history, "nmse")
    else:
        unavailable_panel(axes[0, 1], "B  Mapped/native KV NMSE")
    diagnostic_bars(
        axes[0, 2], groups, lengths, bridge_policies,
        [("Mapped tokens", bridge_metric("bridge_mapped_tokens"), "#7B2CBF")],
        "Mapped token positions / run", "C  Bridge mapping volume",
        seed + 230, replicates,
    )
    diagnostic_bars(
        axes[1, 0], groups, lengths, bridge_policies,
        [
            ("Target-V read", bridge_metric("bridge_source_bytes", 1024 ** 2), "#4C78A8"),
            ("Draft-KV materialized", bridge_metric("bridge_materialized_bytes", 1024 ** 2), "#F58518"),
        ],
        "MiB / run (repeated refreshes counted)", "D  Bridge data volume",
        seed + 240, replicates,
    )
    rejection_distribution(axes[1, 1], groups, lengths, policies)
    selected_position_distribution(axes[1, 2], groups, lengths, policies)
    finish_figure(
        fig,
        output_dir / "attn_bridge_diagnostics",
        status_footer(payload, groups, lengths, policies, partial)
        + "; distribution panels pool rounds only for mechanism diagnosis",
        partial,
    )


def fmt(value: Optional[float], digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def fmt_ci(stats: Optional[tuple[float, float, float]], digits: int = 3) -> str:
    if stats is None:
        return "—"
    mean, low, high = stats
    return f"{mean:+.{digits}f} [{low:+.{digits}f}, {high:+.{digits}f}]"


def fmt_percent(value: Optional[float], digits: int = 1) -> str:
    return "—" if value is None else f"{100 * value:.{digits}f}%"


def target_mismatch_conclusion(diagnostic: dict) -> Optional[str]:
    failures = diagnostic.get("failures") or []
    if not failures or not diagnostic.get("all_failures_covered"):
        return None
    grouped_failures: dict[tuple[int, int, int, float], list[str]] = defaultdict(list)
    for item in failures:
        margin = finite_number(item.get("target_top1_top2_margin"))
        if margin is None:
            return None
        key = (
            int(item["length_bin_tokens"]),
            int(item["sample_index"]),
            int(item["first_mismatch_position_zero_based"]),
            margin,
        )
        grouped_failures[key].append(str(item["policy"]))
    locations = []
    for (length, sample, position, margin), policies in sorted(grouped_failures.items()):
        nominal = f"{length // 1024}K" if length % 1024 == 0 else str(length)
        locations.append(
            f"({nominal},sample{sample},pos{position})、canonical margin={margin:.6f}"
            f"（策略：{', '.join(f'`{policy}`' for policy in sorted(policies))}）"
        )
    concentration = "均集中在 " if len(locations) == 1 else "分布在 "
    conclusion = (
        f"margin 文件覆盖了全部 {len(failures)} 个失败配对；本次 {len(failures)} 个失败"
        f"{concentration}{'；'.join(locations)}。pos 为生成 continuation 的 0-based 下标。"
    )
    maximum_margin = max(
        abs(float(item["target_top1_top2_margin"])) for item in failures
    )
    if maximum_margin <= 0.02:
        conclusion += (
            "该处 top1/top2 极近，属于树批处理与单步 AR 在 fp16 近并列处走向"
            "不同 token 的数值分支；这是已定位的数值路径差异，不是 Bridge 策略改变"
            "canonical verifier 分布的证据。"
        )
    return conclusion


def metric_mean(rows: list[dict], metric: Callable[[dict], Optional[float]]) -> Optional[float]:
    values = [metric(row) for row in rows]
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def expected_grid_status(
    payload: dict,
    groups: dict[tuple[int, str], list[dict]],
    lengths: list[int],
    policies: list[str],
) -> tuple[bool, int, list[str]]:
    config = payload.get("config") or {}
    expected_samples = int(config.get("samples", 0) or 0)
    problems = []
    if expected_samples <= 0:
        expected_samples = max((len(rows) for rows in groups.values()), default=0)
        problems.append("原始配置未记录预期样本数")
    for length in lengths:
        for policy in policies:
            count = len(groups.get((length, policy), []))
            if count != expected_samples:
                problems.append(
                    f"{length}/{policy}: {count}/{expected_samples} runs"
                )
    return bool(problems), expected_samples, problems


def target_reference_status(
    rows: list[dict], references: dict[tuple[int, int], dict]
) -> list[str]:
    """Describe missing/short canonical or evaluated continuations."""
    problems = []
    evaluation_keys = sorted(
        {
            (int(row["length_bin_tokens"]), int(row["sample_index"]))
            for row in rows
        }
    )
    for length, sample in evaluation_keys:
        reference = references.get((length, sample))
        if reference is None:
            problems.append(
                f"target AR {length}/sample={sample}: missing reference run"
            )
            continue
        ids = output_ids(reference)
        if ids is None or len(ids) < TARGET_EXACT_TOKENS:
            recorded = "missing" if ids is None else str(len(ids))
            problems.append(
                f"target AR {length}/sample={sample}: {recorded}/"
                f"{TARGET_EXACT_TOKENS} output tokens"
            )
    for row in rows:
        ids = output_ids(row)
        if ids is None or len(ids) < TARGET_EXACT_TOKENS:
            recorded = "missing" if ids is None else str(len(ids))
            problems.append(
                f"evaluation {int(row['length_bin_tokens'])}/{row['policy']}/"
                f"sample={int(row['sample_index'])}: {recorded}/"
                f"{TARGET_EXACT_TOKENS} output tokens"
            )
    return problems


def comparison_rows(
    groups: dict[tuple[int, str], list[dict]],
    lengths: list[int],
    policies: list[str],
    seed: int,
    replicates: int,
) -> list[dict]:
    comparisons = [
        ("反事实接受收益", "attention_bridge_a25", "recent_bridge_a25"),
        ("共享状态对齐收益", "attention_bridge_a25", "attention_bridge_shuffle_a25"),
        ("Bridge 相对原生注意力（alpha=0.25）", "attention_bridge_a25", "attention_native"),
        ("低强度 Bridge（alpha=0.1）", "attention_bridge_a10", "attention_native"),
        ("value-only 相对 K+V（alpha=0.25）", "attention_bridge_value_a25", "attention_bridge_a25"),
        ("alpha=0.5 相对 alpha=0.25", "attention_bridge_a50", "attention_bridge_a25"),
        ("全层相对末层注入", "attention_bridge_a50", "attention_bridge_last_a50"),
        ("alpha=1.0 相对 alpha=0.5", "attention_bridge_a100", "attention_bridge_a50"),
        ("注意力检索相对近期检索（原生）", "attention_native", "recent_native"),
    ]
    output = []
    for length_index, length in enumerate(lengths):
        for comparison_index, (name, left, right) in enumerate(comparisons):
            if left not in policies or right not in policies:
                continue
            base_seed = seed + length_index * 10_000 + comparison_index * 100
            accept, n_accept = paired_difference(
                groups.get((length, left), []),
                groups.get((length, right), []),
                acceptance_length,
                base_seed + 1,
                replicates,
            )
            throughput, n_tps = paired_difference(
                groups.get((length, left), []),
                groups.get((length, right), []),
                wall_tps,
                base_seed + 2,
                replicates,
            )
            kl, n_kl = paired_difference(
                groups.get((length, left), []),
                groups.get((length, right), []),
                root_kl,
                base_seed + 3,
                replicates,
            )
            output.append(
                {
                    "length": length,
                    "name": name,
                    "left": left,
                    "right": right,
                    "acceptance_delta": accept,
                    "throughput_delta": throughput,
                    "root_kl_delta": kl,
                    "n": min(n_accept, n_tps),
                    "n_kl": n_kl,
                }
            )
    return output


def actual_context_description(rows: list[dict]) -> str:
    by_nominal: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        by_nominal[int(row["length_bin_tokens"])].append(int(row.get("input_tokens", 0)))
    parts = []
    for nominal, actual in sorted(by_nominal.items()):
        unique = sorted(set(actual))
        if len(unique) == 1:
            text = str(unique[0])
        else:
            text = f"{min(unique)}–{max(unique)}"
        parts.append(f"标称 {nominal}：实际 {text} token")
    return "；".join(parts)


def summarize_training_history(training_history: Optional[dict]) -> Optional[dict]:
    rows = training_rows(training_history)
    usable = [row for row in rows if finite_number(row.get("val_loss")) is not None]
    if not usable:
        return None
    best = min(usable, key=lambda row: float(row["val_loss"]))
    final = usable[-1]

    def compact(row: dict) -> dict:
        cosine_loss = finite_number(row.get("val_cosine_loss"))
        return {
            "epoch": int(row["epoch"]),
            "val_loss": finite_number(row.get("val_loss")),
            "val_cosine": None if cosine_loss is None else 1.0 - cosine_loss,
            "val_nmse": finite_number(row.get("val_nmse")),
        }

    return {
        "best": compact(best),
        "final": compact(final),
        "epochs": len(usable),
        "mapper_config": (training_history or {}).get("mapper_config"),
        "pair_metadata": (training_history or {}).get("pair_metadata"),
    }


def write_report_and_summary(
    payload: dict,
    raw_path: Path,
    output_dir: Path,
    report_name: str,
    groups: dict[tuple[int, str], list[dict]],
    rows: list[dict],
    lengths: list[int],
    policies: list[str],
    reference: str,
    partial: bool,
    expected_samples: int,
    incomplete_cells: list[str],
    seed: int,
    replicates: int,
    training_history_path: Optional[Path] = None,
    training_history: Optional[dict] = None,
    target_reference_path: Optional[Path] = None,
    target_reference_payload: Optional[dict] = None,
    target_references: Optional[dict[tuple[int, int], dict]] = None,
    target_reference_problems: Optional[list[str]] = None,
    target_margin_path: Optional[Path] = None,
    target_mismatch_diagnostic: Optional[dict] = None,
) -> None:
    config = payload.get("config") or {}
    environment = payload.get("environment") or {}
    comparisons = comparison_rows(groups, lengths, policies, seed, replicates)
    mapper_training = summarize_training_history(training_history)
    target_reference_problems = target_reference_problems or []
    target_mismatch_diagnostic = target_mismatch_diagnostic or {
        "failure_count": 0,
        "covered_failure_count": 0,
        "all_failures_covered": False,
        "failures": [],
    }
    has_target_reference = target_references is not None
    title_status = "（部分结果）" if partial else ""
    report = [
        f"# AttnBridgeKV 4090 实验报告{title_status}",
        "",
    ]
    if partial:
        report += [
            "> **状态：PARTIAL / 部分结果。** 当前只汇总已完成运行，不能当作最终实验结论。",
            "",
        ]
    report += [
        "![质量指标](attn_bridge_quality.png)",
        "",
        "![系统性能](attn_bridge_performance.png)",
        "",
        "![机制诊断](attn_bridge_diagnostics.png)",
        "",
        "## 实验设置",
        "",
        f"- GPU：{environment.get('gpu', '未记录')}；PyTorch {environment.get('torch', '未记录')}；CUDA {environment.get('cuda', '未记录')}。",
        f"- 数据：`{Path(str(config.get('source_file', '未记录'))).name}`；sample index 从 {config.get('sample_start', '未记录')} 开始；预期每个单元 {expected_samples} 个配对文档。",
        f"- 上下文：{actual_context_description(rows)}。",
        f"- 固定设置：max generation={config.get('max_gen_len', '未记录')}，tree nodes={config.get('tree_nodes', '未记录')}，chunk={config.get('chunk_size', '未记录')}，K={config.get('cache_top_k', '未记录')}，每 {config.get('retrieve_every', '未记录')} 轮检索，recent-native={config.get('recent_native', '未记录')}。",
        f"- 统计：文档是统计单元；策略差值使用 {replicates:,} 次配对 bootstrap 95% CI；参考策略为 `{reference}`。",
        f"- 性能 run 的 Target attention 配置：`exact_target_attention={config.get('exact_target_attention', '未记录')}`。",
        "- 当前 Bridge 是 delta/blend 模式：仍构建原生 Draft KV，再映射并混合 Target-V→Draft-KV。因此本实验能验证 proposal 质量与映射代价，不能把接受率改善直接写成系统加速。",
        "",
        "## 指标口径",
        "",
        "- 接受长度：每轮被接受的 draft token 数，不含随后提交的 verifier token。",
        "- verified-path acceptance share：`sum(accept_length) / sum(accept_length + 1)`；这是已验证路径中的草稿 token 占比，不是整棵候选树节点的接受率。",
        (
            f"- 主 exactness：每个策略与修复后的 canonical target-only AR 逐位比较完整 {TARGET_EXACT_TOKENS}-token continuation；少于 {TARGET_EXACT_TOKENS} token 的配对记为不可用而不是成功。`{reference}` 仍作为策略间诊断参考，机器汇总保留相对它的共同前缀及完整列表一致率。它验证的是 greedy 路径，不等价于随机采样下完整分布不变证明。"
            if has_target_reference
            else f"- Greedy exactness：与 `{reference}` 的完整输出 token ID 列表逐位相等（当前 runner 已固定截断到 `max_gen_len`）；机器汇总另存共同前缀一致率，便于诊断旧版 runner 的尾部超生成差异。它验证的是 greedy 路径，不等价于随机采样下完整分布不变证明。"
        ),
        "- stage latency 均除以该次运行实际生成 token 数；retrieval 是 draft 的子区间，bridge map 又是 retrieval 的子区间，三者不可相加。",
    ]

    if has_target_reference:
        target_config = (target_reference_payload or {}).get("config") or {}
        report += [
            "",
            "## Canonical Target AR 参考",
            "",
            f"- 来源：修复后的 canonical AR target-only run：`{target_reference_path.name if target_reference_path else 'target reference'}`。",
            f"- 参考设置：max generation={target_config.get('max_gen_len', '未记录')}，sample start={target_config.get('sample_start', '未记录')}，samples={target_config.get('samples', '未记录')}。",
            f"- 主检查要求候选与 target AR 都至少含 {TARGET_EXACT_TOKENS} 个输出 token，并比较恰好前 {TARGET_EXACT_TOKENS} 个 token。",
            f"- 性能 run 使用 `exact_target_attention={config.get('exact_target_attention', '未记录')}`；此配置决定 verifier 的 canonical attention 路径，不能与 target-only AR 参考混为同一计时口径。",
        ]
        if target_margin_path is not None:
            report.append(
                f"- 首处分歧 margin 诊断来自 `{target_margin_path.name}`；位置使用生成 continuation 的 0-based 下标，margin 为 canonical target 的 top1 logit 减 top2 logit。"
            )

    if mapper_training is not None:
        best = mapper_training["best"]
        final = mapper_training["final"]
        report += [
            "",
            "## Mapper 训练诊断",
            "",
            "正式性能计时关闭了运行时 cosine/NMSE 诊断，机制图因此使用独立 `train_history.json`；这些值衡量 mapper 拟合质量，不计入性能 run 延迟。",
            "",
            "| 检查点 | Epoch | Val objective | Val cosine | Val NMSE |",
            "|---|---:|---:|---:|---:|",
            f"| Best val objective | {best['epoch']} | {fmt(best['val_loss'], 5)} | {fmt(best['val_cosine'], 5)} | {fmt(best['val_nmse'], 5)} |",
            f"| Final | {final['epoch']} | {fmt(final['val_loss'], 5)} | {fmt(final['val_cosine'], 5)} | {fmt(final['val_nmse'], 5)} |",
        ]

    report += [
        "",
        "## 汇总均值与相对参考策略的配对差值",
        "",
    ]
    if has_target_reference:
        report += [
            f"| 标称长度 | 策略 | n | 接受长度 | 路径接受占比 | Root KL | token/s | Δ接受长度 | Δtoken/s | Target AR {TARGET_EXACT_TOKENS}-token exact | vs `{reference}` 完整列表 exact（诊断） |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    else:
        report += [
            "| 标称长度 | 策略 | n | 接受长度 | 路径接受占比 | Root KL | token/s | Δ接受长度 | Δtoken/s | Greedy 完整输出 exact |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]

    summary_cells = []
    for length_index, length in enumerate(lengths):
        reference_rows = groups.get((length, reference), [])
        for policy_index, policy in enumerate(policies):
            cell = groups.get((length, policy), [])
            accept_delta, n_accept = paired_difference(
                cell,
                reference_rows,
                acceptance_length,
                seed + 30_000 + length_index * 1000 + policy_index * 10,
                replicates,
            )
            tps_delta, n_tps = paired_difference(
                cell,
                reference_rows,
                wall_tps,
                seed + 40_000 + length_index * 1000 + policy_index * 10,
                replicates,
            )
            prefix_exact_values = exactness_values(cell, reference_rows)
            full_exact_values = exactness_values(cell, reference_rows, full=True)
            target_exact_values = (
                target_exactness_values(cell, length, target_references)
                if target_references is not None
                else []
            )
            prefix_exact_mean = (
                float(np.mean(prefix_exact_values)) if prefix_exact_values else None
            )
            full_exact = float(np.mean(full_exact_values)) if full_exact_values else None
            target_exact = (
                float(np.mean(target_exact_values)) if target_exact_values else None
            )
            accept = metric_mean(cell, acceptance_length)
            share = metric_mean(cell, acceptance_share)
            kl = metric_mean(cell, root_kl)
            tps = metric_mean(cell, wall_tps)
            row_text = (
                f"| {length_label(length, cell)} ({length}) | `{policy}` | {len(cell)} | {fmt(accept)} | "
                f"{fmt_percent(share)} | {fmt(kl)} | {fmt(tps, 2)} | "
                f"{fmt_ci(accept_delta)} | {fmt_ci(tps_delta, 2)} | "
            )
            if has_target_reference:
                row_text += (
                    f"{fmt_percent(target_exact)} ({len(target_exact_values)}/{len(cell)}) | "
                    f"{fmt_percent(full_exact)} |"
                )
            else:
                row_text += f"{fmt_percent(full_exact)} |"
            report.append(row_text)
            summary_cells.append(
                {
                    "length_bin_tokens": length,
                    "policy": policy,
                    "n": len(cell),
                    "paired_n_acceptance": n_accept,
                    "paired_n_throughput": n_tps,
                    "mean_acceptance_length": accept,
                    "mean_verified_path_acceptance_share": share,
                    "mean_root_kl": kl,
                    "mean_wall_tokens_per_sec": tps,
                    "acceptance_delta_vs_reference": accept_delta,
                    "throughput_delta_vs_reference": tps_delta,
                    "target_ar_exact_tokens": (
                        TARGET_EXACT_TOKENS if has_target_reference else None
                    ),
                    "target_ar_exactness_eligible_pairs": (
                        len(target_exact_values) if has_target_reference else None
                    ),
                    "target_ar_full_128_token_exactness": (
                        target_exact if has_target_reference else None
                    ),
                    "greedy_common_prefix_exactness": prefix_exact_mean,
                    "full_output_list_exactness": full_exact,
                    "greedy_common_prefix_exactness_vs_reference_policy": prefix_exact_mean,
                    "full_output_list_exactness_vs_reference_policy": full_exact,
                }
            )

    report += [
        "",
        "## 关键因果对照",
        "",
        "下表左减右；注意力引导创新的主因果检验是 `attention_bridge_a25 − recent_bridge_a25`，二者 mapper、alpha 与缓存预算相同，只改变选块方式；共享状态对齐检验是 `attention_bridge_a25 − attention_bridge_shuffle_a25`。",
        "",
        "| 标称长度 | 对照 | n | Δ接受长度 | Δtoken/s | ΔRoot KL |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        report.append(
            f"| {item['length']} | {item['name']}：`{item['left']}` − `{item['right']}` | "
            f"{item['n']} | {fmt_ci(item['acceptance_delta'])} | "
            f"{fmt_ci(item['throughput_delta'], 2)} | {fmt_ci(item['root_kl_delta'])} |"
        )

    # Generate guarded, data-driven conclusions for the primary comparison.
    primary = [item for item in comparisons if item["name"] == "反事实接受收益"]
    policy_reference_exact_values = [
        value
        for length in lengths
        for policy in policies
        for value in exactness_values(
            groups.get((length, policy), []),
            groups.get((length, reference), []),
            full=True,
        )
    ]
    target_ar_exact_values = (
        [
            value
            for length in lengths
            for policy in policies
            for value in target_exactness_values(
                groups.get((length, policy), []), length, target_references
            )
        ]
        if target_references is not None
        else []
    )
    primary_control_exactness = {}
    if target_references is not None:
        for policy in ("attention_bridge_a25", "recent_bridge_a25"):
            values = [
                value
                for length in lengths
                for value in target_exactness_values(
                    groups.get((length, policy), []), length, target_references
                )
            ]
            primary_control_exactness[policy] = {
                "eligible_pairs": len(values),
                "exact_pairs": sum(value == 1.0 for value in values),
                "exactness": float(np.mean(values)) if values else None,
            }
    report += ["", "## 结论边界", ""]
    conclusions = []
    if has_target_reference and target_ar_exact_values and min(target_ar_exact_values) == 1.0:
        conclusions.append(
            f"相对修复后的 canonical target-only AR，已完成且具备足够长度的 {len(target_ar_exact_values)} 个策略–文档配对全部通过完整 {TARGET_EXACT_TOKENS}-token 逐 token 一致性检查。性能 run 使用 `exact_target_attention={config.get('exact_target_attention', '未记录')}`；结果支持当前 greedy verifier 实现的 exactness，但不是随机采样分布不变的统计证明。"
        )
    elif has_target_reference and target_ar_exact_values:
        failures = sum(value < 1.0 for value in target_ar_exact_values)
        diagnosed = target_mismatch_conclusion(target_mismatch_diagnostic)
        if diagnosed is not None:
            conclusions.append(
                f"相对修复后的 canonical target-only AR，完整 {TARGET_EXACT_TOKENS}-token 一致性检查出现 {failures}/{len(target_ar_exact_values)} 个失败配对。{diagnosed}"
            )
        else:
            conclusions.append(
                f"相对修复后的 canonical target-only AR，完整 {TARGET_EXACT_TOKENS}-token 一致性检查出现 {failures}/{len(target_ar_exact_values)} 个失败配对；margin 诊断未覆盖全部失败，在解释接受率或吞吐前仍需保守排查 verifier exactness。"
            )
    elif has_target_reference:
        conclusions.append(
            f"已提供 canonical target-only AR 参考，但没有候选/参考双方均达到 {TARGET_EXACT_TOKENS} token 的可用配对，暂不能报告完整 {TARGET_EXACT_TOKENS}-token exactness。"
        )
    elif policy_reference_exact_values and min(policy_reference_exact_values) == 1.0:
        conclusions.append(
            f"已完成的 {len(policy_reference_exact_values)} 个策略–文档配对全部通过相对 `{reference}` 的 greedy 完整输出逐 token 一致性检查；这支持“Bridge 只影响 proposal、不改 verifier 输出”的实现检查，但不是随机采样分布不变的统计证明。"
        )
    elif policy_reference_exact_values:
        failures = sum(value < 1.0 for value in policy_reference_exact_values)
        conclusions.append(
            f"相对 `{reference}` 的 greedy 完整输出一致性检查出现 {failures}/{len(policy_reference_exact_values)} 个失败配对；在解释接受率或吞吐前必须先排查 exactness。"
        )
    else:
        conclusions.append("原始结果未包含足够的 output token IDs，暂不能验证 greedy exactness。")
    if has_target_reference and policy_reference_exact_values:
        failures = sum(value < 1.0 for value in policy_reference_exact_values)
        conclusions.append(
            f"辅助诊断：相对策略参考 `{reference}` 的完整列表一致性为 {(len(policy_reference_exact_values) - failures)}/{len(policy_reference_exact_values)}；该数字不替代 canonical target AR exactness。"
        )
    if has_target_reference:
        if primary_control_exactness and all(
            item["eligible_pairs"] > 0 and item["exactness"] == 1.0
            for item in primary_control_exactness.values()
        ):
            attention_count = primary_control_exactness["attention_bridge_a25"][
                "eligible_pairs"
            ]
            recent_count = primary_control_exactness["recent_bridge_a25"][
                "eligible_pairs"
            ]
            conclusions.append(
                f"注意力引导主因果对照的两个策略均保持 target-AR 完整 {TARGET_EXACT_TOKENS}-token exactness 100%：`attention_bridge_a25` 为 {attention_count}/{attention_count}，`recent_bridge_a25` 为 {recent_count}/{recent_count}。因此本次 {target_mismatch_diagnostic.get('failure_count', 0)} 个已定位失败不涉及主对照两端。"
            )
    if primary:
        for item in primary:
            accept = item["acceptance_delta"]
            tps = item["throughput_delta"]
            if accept is not None:
                direction = "提高" if accept[0] > 0 else "降低"
                decisive = "且 CI 不跨 0" if (accept[1] > 0 or accept[2] < 0) else "但 CI 跨 0"
                conclusions.append(
                    f"在标称 {item['length']} token 上，同一 Bridge 下注意力选块相对近期选块使接受长度{direction} {abs(accept[0]):.3f}（95% CI [{accept[1]:+.3f}, {accept[2]:+.3f}]，{decisive}）。"
                )
            if tps is not None:
                speed_claim = (
                    "提供了正向且 CI 不跨 0 的 pilot 证据"
                    if tps[1] > 0
                    else "尚不能证明端到端加速"
                )
                conclusions.append(
                    f"同一对照的吞吐差为 {tps[0]:+.2f} token/s（95% CI [{tps[1]:+.2f}, {tps[2]:+.2f}]），{speed_claim}。"
                )
    else:
        conclusions.append("当前结果缺少 attention-bridge 与 recent-bridge 的同预算配对，不能估计反事实接受收益。")
    conclusions.append(
        "即使 delta 模式出现正吞吐差，也应先复测并进入 replacement 模式（跳过对应 Draft prefill）后再主张共享 KV 带来的系统收益。"
    )
    if partial:
        conclusions.append("本报告为部分结果；所有方向性结论仅描述已完成子集，不能外推到完整样本。")
    report.extend(f"{index}. {text}" for index, text in enumerate(conclusions, start=1))

    mismatch_rows = target_mismatch_diagnostic.get("failures") or []
    if mismatch_rows:
        report += [
            "",
            "## Target AR 首处分歧诊断",
            "",
            "位置为生成 continuation 的 0-based token 下标。只有 margin run 的 canonical 输出与主 target reference 完整对齐时，才采用对应 margin。",
            "",
            "| 标称长度 | sample | 策略 | first mismatch pos | target token | candidate token | canonical top1-top2 margin |",
            "|---:|---:|---|---:|---:|---:|---:|",
        ]
        for item in mismatch_rows:
            length = int(item["length_bin_tokens"])
            nominal = f"{length // 1024}K" if length % 1024 == 0 else str(length)
            report.append(
                f"| {nominal} ({length}) | {item['sample_index']} | `{item['policy']}` | "
                f"{item['first_mismatch_position_zero_based']} | {item['target_token_id']} | "
                f"{item['candidate_token_id']} | {fmt(finite_number(item.get('target_top1_top2_margin')), 6)} |"
            )

    if incomplete_cells:
        report += ["", "## 未完成单元", ""]
        report.extend(f"- {item}" for item in incomplete_cells)
    if target_reference_problems:
        report += ["", "## 128-token exactness 覆盖问题", ""]
        report.extend(f"- {item}" for item in target_reference_problems)

    report += [
        "",
        "## 产物",
        "",
        "- [质量图 PNG](attn_bridge_quality.png) / [SVG](attn_bridge_quality.svg)",
        "- [系统性能图 PNG](attn_bridge_performance.png) / [SVG](attn_bridge_performance.svg)",
        "- [机制诊断图 PNG](attn_bridge_diagnostics.png) / [SVG](attn_bridge_diagnostics.svg)",
        "- [机器可读汇总](summary.json)",
        f"- [原始结果]({Path(os.path.relpath(raw_path, output_dir)).as_posix()})",
        f"- [分析代码]({Path(os.path.relpath(Path(__file__).resolve(), output_dir)).as_posix()})",
    ]
    if training_history_path is not None:
        report.append(
            f"- [Mapper 训练历史]({Path(os.path.relpath(training_history_path, output_dir)).as_posix()})"
        )
    if target_reference_path is not None:
        report.append(
            f"- [Canonical target AR 原始结果]({Path(os.path.relpath(target_reference_path, output_dir)).as_posix()})"
        )
    if target_margin_path is not None:
        report.append(
            f"- [Target AR margin 诊断]({Path(os.path.relpath(target_margin_path, output_dir)).as_posix()})"
        )
    report.append("")
    (output_dir / report_name).write_text("\n".join(report), encoding="utf-8")

    machine_summary = {
        "source": str(raw_path),
        "partial": partial,
        "reference_policy": reference,
        "primary_exactness_reference": (
            "canonical_target_ar" if has_target_reference else reference
        ),
        "target_ar_exact_tokens": (
            TARGET_EXACT_TOKENS if has_target_reference else None
        ),
        "target_reference": (
            {
                "source": str(target_reference_path),
                "description": "fixed canonical target-only autoregressive decoding",
                "config": (target_reference_payload or {}).get("config"),
                "coverage_problems": target_reference_problems,
            }
            if has_target_reference
            else None
        ),
        "performance_exact_target_attention": config.get("exact_target_attention"),
        "primary_attention_control_target_ar_exactness": primary_control_exactness,
        "target_margin_diagnostic": {
            "source": str(target_margin_path) if target_margin_path is not None else None,
            **target_mismatch_diagnostic,
        },
        "bootstrap_replicates": replicates,
        "statistical_unit": "sample_index/document",
        "expected_samples_per_cell": expected_samples,
        "incomplete_cells": incomplete_cells,
        "cells": summary_cells,
        "key_comparisons": comparisons,
        "mapper_training": mapper_training,
        "training_history": (
            str(training_history_path) if training_history_path is not None else None
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(machine_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.bootstrap < 100:
        raise SystemExit("--bootstrap must be at least 100")
    raw_path = Path(args.input).expanduser().resolve()
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    rows = payload.get("raw_runs")
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"No raw_runs found in {raw_path}")

    required = {"length_bin_tokens", "policy", "sample_index"}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise SystemExit(f"raw_runs[{index}] is missing {sorted(missing)}")
    keys = [
        (int(row["length_bin_tokens"]), str(row["policy"]), int(row["sample_index"]))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise SystemExit("Duplicate (length, policy, sample_index) rows found")

    target_reference_path = (
        Path(args.target_reference).expanduser().resolve()
        if args.target_reference
        else None
    )
    target_reference_payload = None
    target_references = None
    target_reference_problems = []
    if target_reference_path is not None:
        target_reference_payload = json.loads(
            target_reference_path.read_text(encoding="utf-8")
        )
        target_rows = target_reference_payload.get("raw_runs")
        if not isinstance(target_rows, list) or not target_rows:
            raise SystemExit(
                f"No raw_runs found in target reference {target_reference_path}"
            )
        target_references = target_reference_map(target_rows)
        target_reference_problems = target_reference_status(rows, target_references)

    target_margin_path = (
        Path(args.target_margin_diagnostic).expanduser().resolve()
        if args.target_margin_diagnostic
        else None
    )
    target_margin_references = None
    if target_margin_path is not None:
        if target_references is None:
            raise SystemExit(
                "--target-margin-diagnostic requires --target-reference"
            )
        target_margin_payload = json.loads(
            target_margin_path.read_text(encoding="utf-8")
        )
        target_margin_rows = target_margin_payload.get("raw_runs")
        if not isinstance(target_margin_rows, list) or not target_margin_rows:
            raise SystemExit(
                f"No raw_runs found in target margin diagnostic {target_margin_path}"
            )
        target_margin_references = target_margin_map(target_margin_rows)
    target_mismatch_diagnostic = (
        build_target_mismatch_diagnostic(
            rows, target_references, target_margin_references
        )
        if target_references is not None
        else None
    )

    groups = grouped(rows)
    lengths = sorted({key[0] for key in groups})
    configured_policies = list((payload.get("policy_specs") or {}).keys())
    observed_policies = {key[1] for key in groups}
    policies = sorted(
        set(configured_policies) | observed_policies | {args.reference_policy},
        key=policy_sort_key,
    )
    auto_partial, expected_samples, incomplete_cells = expected_grid_status(
        payload, groups, lengths, policies
    )
    partial = bool(args.partial or auto_partial or target_reference_problems)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else raw_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    training_history_path = (
        Path(args.training_history).expanduser().resolve()
        if args.training_history
        else None
    )
    training_history = None
    if training_history_path is not None:
        training_history = json.loads(training_history_path.read_text(encoding="utf-8"))
        if not training_rows(training_history):
            raise SystemExit(
                f"No non-empty history list found in {training_history_path}"
            )

    footer_partial = partial
    plot_quality(
        payload, groups, lengths, policies, args.reference_policy,
        output_dir, args.seed, args.bootstrap, footer_partial, target_references,
    )
    plot_performance(
        payload, groups, lengths, policies, args.reference_policy,
        output_dir, args.seed, args.bootstrap, footer_partial,
    )
    plot_diagnostics(
        payload, groups, lengths, policies,
        output_dir, args.seed, args.bootstrap, footer_partial, training_history,
    )
    write_report_and_summary(
        payload,
        raw_path,
        output_dir,
        args.report_name,
        groups,
        rows,
        lengths,
        policies,
        args.reference_policy,
        partial,
        expected_samples,
        incomplete_cells,
        args.seed,
        args.bootstrap,
        training_history_path,
        training_history,
        target_reference_path,
        target_reference_payload,
        target_references,
        target_reference_problems,
        target_margin_path,
        target_mismatch_diagnostic,
    )
    status = "PARTIAL" if partial else "complete"
    print(
        f"Wrote {status} AttnBridgeKV analysis for {len(rows)} runs to {output_dir}"
    )


if __name__ == "__main__":
    main()
