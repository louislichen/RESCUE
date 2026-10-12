#!/usr/bin/env python3
"""Analyze the V-only versus K+V attention-geometry experiment.

The script intentionally keeps quality, real end-to-end performance, and
mechanism diagnostics in separate figures.  Its primary estimand is the
document-paired V-only minus K+V difference.  If several mapper seeds are
present, seed-level differences are averaged within a document before the
document bootstrap, so optimizer repetitions are not treated as independent
documents.

Geometry may be supplied either as one summary per run or as weighted
``bridge_geometry_events``.  The latter are reduced with their explicit
``weight`` (falling back to one), never as an unweighted event average.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np


DEFAULT_POLICIES = {
    "native": "attention_native",
    "value": "attention_bridge_value_a25",
    "key": "attention_bridge_key_a25",
    "kv": "attention_bridge_a25",
    "rescue": "attention_bridge_v25_k25_qk_rescue",
}

GEOMETRY_ALIASES = {
    "geometry_qk_logit_rmse": (
        "geometry_qk_logit_rmse", "bridge_geometry_qk_logit_rmse",
        "qk_logit_rmse", "logit_rmse",
    ),
    "geometry_qk_logit_relative_rmse": (
        "geometry_qk_logit_relative_rmse", "bridge_geometry_qk_logit_relative_rmse",
        "qk_logit_relative_rmse",
        "qk_logit_nrmse", "logit_relative_rmse", "logit_nrmse",
    ),
    "geometry_attention_kl_native_to_bridge": (
        "geometry_attention_kl_native_to_bridge",
        "bridge_geometry_attention_kl",
        "attention_kl_native_to_bridge", "attention_kl", "attn_kl",
    ),
    "geometry_topk_jaccard": (
        "geometry_topk_jaccard", "bridge_geometry_topk_jaccard",
        "topk_jaccard", "attention_topk_jaccard",
    ),
    "geometry_argmax_flip_rate": (
        "geometry_argmax_flip_rate", "bridge_geometry_argmax_flip_rate",
        "argmax_flip_rate",
        "attention_argmax_flip_rate",
    ),
    "geometry_query_projected_key_mae": (
        "geometry_query_projected_key_mae", "bridge_geometry_query_projected_key_mae",
        "query_projected_key_mae",
        "q_projected_key_mae",
    ),
    "bridge_mean_key_cosine": (
        "bridge_mean_key_cosine", "mean_key_cosine", "key_cosine",
    ),
    "bridge_mean_key_nmse": (
        "bridge_mean_key_nmse", "mean_key_nmse", "key_nmse",
    ),
    "bridge_mean_key_norm_ratio": (
        "bridge_mean_key_norm_ratio", "mean_key_norm_ratio", "key_norm_ratio",
    ),
    "bridge_mean_value_cosine": (
        "bridge_mean_value_cosine", "mean_value_cosine", "value_cosine",
    ),
    "bridge_mean_value_nmse": (
        "bridge_mean_value_nmse", "mean_value_nmse", "value_nmse",
    ),
}

QUALITY_METRICS = {
    "acceptance": "avg_accept_length",
    "root_kl": "mean_root_target_draft_kl",
}

PERFORMANCE_METRICS = {
    "throughput": "wall_tokens_per_sec",
    "wall_time": "wall_time",
}

# These are mathematical identities for comparing native attention with
# itself, not measurements synthesized from a diagnostic run.
NATIVE_GEOMETRY_BASELINES = {
    "geometry_qk_logit_rmse": 0.0,
    "geometry_qk_logit_relative_rmse": 0.0,
    "geometry_attention_kl_native_to_bridge": 0.0,
    "geometry_topk_jaccard": 1.0,
    "geometry_argmax_flip_rate": 0.0,
    "geometry_query_projected_key_mae": 0.0,
}

POLICY_COLORS = {
    "native": "#4C78A8",
    "key": "#E45756",
    "value": "#F2CF5B",
    "kv": "#2A9D8F",
    "rescue": "#7B2CBF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired analysis for V-only, K-only, K+V and QK-rescue runs"
    )
    parser.add_argument(
        "--formal-run", action="append", required=True,
        help="Formal evaluation JSON; repeat once per Mapper seed if needed.",
    )
    parser.add_argument(
        "--reference-run", action="append", default=[],
        help=(
            "Canonical target_ar JSON; repeat for disjoint length/sample sets. "
            "Each seed-specific formal row is compared with the same seedless "
            "(length, sample) reference."
        ),
    )
    parser.add_argument(
        "--geometry-run", action="append", default=[],
        help="Geometry JSON; may be repeated. Geometry embedded in formal-run is also read.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--native-policy", default=DEFAULT_POLICIES["native"])
    parser.add_argument("--value-policy", default=DEFAULT_POLICIES["value"])
    parser.add_argument("--key-policy", default=DEFAULT_POLICIES["key"])
    parser.add_argument("--kv-policy", default=DEFAULT_POLICIES["kv"])
    parser.add_argument("--rescue-policy", default=DEFAULT_POLICIES["rescue"])
    parser.add_argument(
        "--expected-samples", type=int,
        help="If set, incomplete policy/length cells are labelled partial.",
    )
    parser.add_argument(
        "--confirmatory-samples",
        help="Fresh document indices for primary inference, e.g. '12-19'.",
    )
    parser.add_argument(
        "--exploratory-samples",
        help="Previously inspected/pilot document indices, e.g. '4-11'.",
    )
    parser.add_argument(
        "--title", default="V-only vs K+V: attention-geometry experiment",
    )
    return parser.parse_args()


def finite(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parse_sample_spec(spec: Optional[str]) -> Optional[set[int]]:
    if spec is None:
        return None
    indices: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" in part:
            start_text, stop_text = part.split(":", 1)
            start, stop = int(start_text), int(stop_text)
            indices.update(range(start, stop))
        elif "-" in part[1:]:
            start_text, stop_text = part.split("-", 1)
            start, stop = int(start_text), int(stop_text)
            if stop < start:
                raise ValueError(f"Descending sample range is invalid: {part}")
            indices.update(range(start, stop + 1))
        else:
            indices.add(int(part))
    if not indices:
        raise ValueError("Sample specification resolved to an empty set")
    return indices


def select_samples(
    rows: list[dict[str, Any]], indices: Optional[set[int]],
) -> list[dict[str, Any]]:
    if indices is None:
        return list(rows)
    selected = []
    for row in rows:
        try:
            sample = int(row.get("sample_index"))
        except (TypeError, ValueError):
            continue
        if sample in indices:
            selected.append(row)
    return selected


def first_finite(mapping: dict[str, Any], names: Iterable[str]) -> Optional[float]:
    for name in names:
        value = finite(mapping.get(name))
        if value is not None:
            return value
    return None


def payload_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("raw_runs", "geometry_runs", "results", "rows", "traces"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def read_payload(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    specs = payload.get("policy_specs", {}) if isinstance(payload, dict) else {}
    return payload_rows(payload), config, specs


def infer_length(row: dict[str, Any]) -> Optional[int]:
    for key in ("length_bin_tokens", "input_tokens", "context_tokens", "length"):
        value = finite(row.get(key))
        if value is not None:
            return int(value)
    label = str(row.get("length_label", ""))
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*[kK]\s*", label)
    return int(float(match.group(1)) * 1024) if match else None


def infer_sample(row: dict[str, Any]) -> Any:
    for key in ("sample_index", "document_index", "doc_id", "sample_id"):
        if key in row:
            return row[key]
    return None


def infer_mapper_seed(row: dict[str, Any]) -> Any:
    for key in ("mapper_seed", "checkpoint_seed", "train_seed", "seed"):
        if key in row:
            return row[key]
    return 0


def event_weight(event: dict[str, Any]) -> float:
    direct = finite(event.get("weight"))
    if direct is not None and direct > 0:
        return direct
    product = 1.0
    found = False
    for key in ("heads", "queries", "kv_tokens"):
        value = finite(event.get(key))
        if value is not None and value > 0:
            product *= value
            found = True
    return product if found else 1.0


def weighted_geometry(events: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"geometry_event_count": len(events)}
    for canonical, aliases in GEOMETRY_ALIASES.items():
        numerator = 0.0
        denominator = 0.0
        for event in events:
            value = first_finite(event, aliases)
            if value is None:
                continue
            weight = event_weight(event)
            numerator += value * weight
            denominator += weight
        if denominator > 0:
            output[canonical] = numerator / denominator
            output[f"{canonical}_weight"] = denominator
    return output


def normalize_row(
    source: dict[str, Any], policy_specs: dict[str, Any]
) -> dict[str, Any]:
    row = dict(source)
    for nested_key in ("geometry", "geometry_summary", "metrics"):
        nested = row.get(nested_key)
        if isinstance(nested, dict):
            for key, value in nested.items():
                row.setdefault(key, value)
    events = row.get("bridge_geometry_events")
    if isinstance(events, list):
        row.update(weighted_geometry([event for event in events if isinstance(event, dict)]))
    elif finite(events) is not None:
        row["geometry_event_count"] = int(float(events))
    post_rescue_events = finite(row.get("bridge_geometry_post_rescue_events"))
    if post_rescue_events is not None and post_rescue_events > 0:
        row["geometry_event_count"] = int(post_rescue_events)

    # Current geometry runs keep a weighted global aggregate plus a compact
    # per-layer list.  Reconstruct the global value from layer event counts
    # only when the flat aggregate is absent.
    by_layer = row.get("bridge_geometry_by_layer")
    if isinstance(by_layer, list):
        layer_rows = [item for item in by_layer if isinstance(item, dict)]
        for canonical, aliases in GEOMETRY_ALIASES.items():
            if first_finite(row, aliases) is not None:
                continue
            numerator = 0.0
            denominator = 0.0
            for item in layer_rows:
                value = first_finite(item, aliases)
                if value is None:
                    continue
                weight = first_finite(item, ("events", "weight")) or 1.0
                numerator += value * weight
                denominator += weight
            if denominator > 0:
                row[canonical] = numerator / denominator
                row[f"{canonical}_weight"] = denominator
    for canonical, aliases in GEOMETRY_ALIASES.items():
        value = first_finite(row, aliases)
        if value is not None:
            row[canonical] = value

    row["policy"] = str(row.get("policy", row.get("policy_name", "unknown")))
    row["length_bin_tokens"] = infer_length(row)
    row["sample_index"] = infer_sample(row)
    row["mapper_seed"] = infer_mapper_seed(row)
    spec = {}
    if isinstance(policy_specs.get(row["policy"]), dict):
        spec.update(policy_specs[row["policy"]])
    if isinstance(row.get("policy_config"), dict):
        spec.update(row["policy_config"])
    row["policy_config"] = spec

    if "round_count" not in row:
        if isinstance(row.get("accept_length_list"), list):
            row["round_count"] = len(row["accept_length_list"])
        elif isinstance(row.get("rounds"), list):
            row["round_count"] = len(row["rounds"])
    if finite(row.get("bridge_map_ms_per_event")) is None:
        total = finite(row.get("bridge_map_time_ms"))
        count = finite(row.get("bridge_map_events"))
        if total is not None and count is not None and count > 0:
            row["bridge_map_ms_per_event"] = total / count
    return row


def merge_rows(
    formal_rows: list[dict[str, Any]], geometry_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row.get("length_bin_tokens"), row.get("sample_index"),
            row.get("mapper_seed"), row.get("policy"),
        )

    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in formal_rows:
        row_key = key(row)
        if row_key in merged:
            raise ValueError(
                "Duplicate formal run key "
                f"{row_key}. Multiple checkpoint files must expose distinct "
                "mapper_seed/checkpoint_seed/train_seed values; evaluation runs "
                "must not be counted as independent seeds merely by file name."
            )
        has_timing = any(
            finite(row.get(field)) is not None
            for field in ("wall_tokens_per_sec", "wall_time", "inference_time")
        )
        formal_diagnostic = row.get("wall_timing_is_diagnostic")
        if isinstance(formal_diagnostic, str):
            formal_diagnostic = formal_diagnostic.strip().lower() in {"1", "true", "yes"}
        merged[row_key] = {
            **row,
            "_has_formal_timing": has_timing,
            "_formal_timing_is_diagnostic": bool(formal_diagnostic),
        }
    grouped_geometry: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in geometry_rows:
        grouped_geometry[key(row)].append(row)
    for row_key, group in grouped_geometry.items():
        target = merged.setdefault(row_key, dict(group[0]))
        for canonical in GEOMETRY_ALIASES:
            values: list[tuple[float, float]] = []
            for row in group:
                value = finite(row.get(canonical))
                if value is None:
                    continue
                weight = finite(row.get(f"{canonical}_weight"))
                if weight is None:
                    weight = finite(row.get("geometry_weight")) or 1.0
                values.append((value, weight))
            if values:
                target[canonical] = sum(v * w for v, w in values) / sum(w for _, w in values)
        for row in group:
            for field, value in row.items():
                if field not in target or target[field] is None:
                    target[field] = value
    output = list(merged.values())
    for row in output:
        diagnostic = row.get("wall_timing_is_diagnostic")
        if isinstance(diagnostic, str):
            diagnostic = diagnostic.strip().lower() in {"1", "true", "yes"}
        exclude_timing = (
            bool(row.get("_formal_timing_is_diagnostic"))
            if row.get("_has_formal_timing")
            else bool(diagnostic)
        )
        if exclude_timing:
            # Geometry instrumentation and QK rescue add synchronization and
            # oracle work.  Their wall clock is not an end-to-end result.
            for field in ("wall_tokens_per_sec", "tokens_per_sec", "wall_time", "inference_time"):
                row.pop(field, None)
            row["_timing_excluded"] = True
    return output


def parse_fraction(token: str) -> float:
    return int(token) / (100.0 if len(token) >= 2 else 10.0)


def infer_alphas(row: dict[str, Any]) -> tuple[float, float]:
    policy = str(row.get("policy", "")).lower()
    spec = row.get("policy_config", {}) if isinstance(row.get("policy_config"), dict) else {}
    key = first_finite(row, ("alpha_key", "key_alpha"))
    value = first_finite(row, ("alpha_value", "value_alpha"))
    if key is None:
        key = first_finite(spec, ("alpha_key", "key_alpha"))
    if value is None:
        value = first_finite(spec, ("alpha_value", "value_alpha"))

    match = re.search(r"v(\d+)_k(\d+)", policy)
    if match:
        value = parse_fraction(match.group(1))
        key = parse_fraction(match.group(2))

    components = str(spec.get("components", "kv")).lower()
    common = first_finite(spec, ("alpha",))
    if common is None:
        match = re.search(r"_a(\d+)(?:_|$)", policy)
        if match:
            common = parse_fraction(match.group(1))
    if "native" in policy and "bridge" not in policy:
        return 0.0, 0.0
    if key is None:
        key = 0.0 if components == "value" else (common or 0.0)
    if value is None:
        value = 0.0 if components == "key" else (common or 0.0)
    return float(key), float(value)


def add_inferred_fields(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        alpha_key, alpha_value = infer_alphas(row)
        row["alpha_key"] = alpha_key
        row["alpha_value"] = alpha_value


def unit_key(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    return row.get("length_bin_tokens"), row.get("sample_index"), row.get("mapper_seed")


def document_key(row: dict[str, Any]) -> tuple[Any, Any]:
    return row.get("length_bin_tokens"), row.get("sample_index")


def rows_for(rows: list[dict[str, Any]], policy: str, length: int) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row.get("policy") == policy and row.get("length_bin_tokens") == length
    ]


def collapse_values_by_document(
    rows: list[dict[str, Any]], metric: str
) -> dict[tuple[Any, Any], float]:
    grouped: dict[tuple[Any, Any], list[float]] = defaultdict(list)
    for row in rows:
        value = finite(row.get(metric))
        if value is not None:
            grouped[document_key(row)].append(value)
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def paired_document_differences(
    rows: list[dict[str, Any]], left: str, right: str, length: int, metric: str,
) -> tuple[np.ndarray, int, int, int]:
    left_units = {
        unit_key(row): finite(row.get(metric)) for row in rows_for(rows, left, length)
        if finite(row.get(metric)) is not None
    }
    right_units = {
        unit_key(row): finite(row.get(metric)) for row in rows_for(rows, right, length)
        if finite(row.get(metric)) is not None
    }
    seed_differences: dict[Any, dict[Any, float]] = defaultdict(dict)
    common = sorted(set(left_units) & set(right_units), key=str)
    for key in common:
        seed_differences[key[1]][key[2]] = float(left_units[key] - right_units[key])
    expected_seeds = {key[2] for key in set(left_units) | set(right_units)}
    candidate_documents = {key[1] for key in set(left_units) | set(right_units)}
    complete = {
        document: values for document, values in seed_differences.items()
        if set(values) == expected_seeds
    }
    document_differences = np.asarray(
        [
            np.mean(list(values.values()))
            for _, values in sorted(complete.items(), key=lambda item: str(item[0]))
        ],
        dtype=float,
    )
    incomplete_documents = len(candidate_documents - set(complete))
    return document_differences, len(common), len(expected_seeds), incomplete_documents


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_mean(
    values: np.ndarray, replicates: int, seed: int,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"mean": None, "low": None, "high": None, "n": 0}
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, values.size, size=(replicates, values.size))
    means = values[samples].mean(axis=1)
    low, high = percentile_ci(means)
    return {
        "mean": float(values.mean()), "low": low, "high": high,
        "n": int(values.size),
    }


def policy_mean(
    rows: list[dict[str, Any]], policy: str, length: int, metric: str,
    replicates: int, seed: int,
) -> dict[str, Any]:
    values = np.asarray(list(collapse_values_by_document(rows_for(rows, policy, length), metric).values()))
    return bootstrap_mean(values, replicates, seed)


def paired_stat(
    rows: list[dict[str, Any]], left: str, right: str, length: int, metric: str,
    replicates: int, seed: int,
) -> dict[str, Any]:
    differences, n_pairs, seed_count, incomplete_documents = paired_document_differences(
        rows, left, right, length, metric
    )
    result = bootstrap_mean(differences, replicates, seed)
    result["n_seed_pairs"] = n_pairs
    result["mapper_seeds"] = seed_count
    result["incomplete_documents_excluded"] = incomplete_documents
    result["left"] = left
    result["right"] = right
    result["metric"] = metric
    return result


def paired_stat_against_definitional_baseline(
    rows: list[dict[str, Any]], left: str, reference: str, length: int,
    metric: str, baseline: float, replicates: int, seed: int,
) -> dict[str, Any]:
    """Pair a measured policy with the native-vs-native mathematical identity.

    A matching native row is still required for every (document, mapper seed),
    but the geometry value is not read from that row because instrumented
    native runs need not materialize a redundant native-vs-native diagnostic.
    """
    left_units = {
        unit_key(row): finite(row.get(metric))
        for row in rows_for(rows, left, length)
        if finite(row.get(metric)) is not None
    }
    reference_units = {
        unit_key(row) for row in rows_for(rows, reference, length)
    }
    expected_seeds = {
        key[2] for key in set(left_units) | reference_units
    }
    candidate_documents = {
        key[1] for key in set(left_units) | reference_units
    }
    by_document: dict[Any, dict[Any, float]] = defaultdict(dict)
    common = sorted(set(left_units) & reference_units, key=str)
    for key in common:
        by_document[key[1]][key[2]] = float(left_units[key] - baseline)
    complete = {
        document: values for document, values in by_document.items()
        if set(values) == expected_seeds
    }
    differences = np.asarray([
        np.mean(list(values.values()))
        for _, values in sorted(complete.items(), key=lambda item: str(item[0]))
    ])
    result = bootstrap_mean(differences, replicates, seed)
    result.update({
        "n_seed_pairs": len(common),
        "mapper_seeds": len(expected_seeds),
        "incomplete_documents_excluded": len(candidate_documents - set(complete)),
        "left": left,
        "right": reference,
        "metric": metric,
        "right_baseline": baseline,
        "right_baseline_kind": "definitional native-vs-native identity; not measured",
    })
    return result


def sign_flip_pvalue(values: np.ndarray, permutations: int, seed: int) -> Optional[float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None
    observed = abs(float(values.mean()))
    if values.size <= 16:
        combinations = np.arange(1 << values.size, dtype=np.uint32)[:, None]
        bits = (combinations >> np.arange(values.size, dtype=np.uint32)) & 1
        signs = bits.astype(float) * 2.0 - 1.0
        null = np.abs((signs * values).mean(axis=1))
        return float((np.count_nonzero(null >= observed - 1e-15) + 1) / (len(null) + 1))
    rng = np.random.default_rng(seed)
    exceed = 0
    done = 0
    batch = 10_000
    while done < permutations:
        count = min(batch, permutations - done)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, values.size))
        null = np.abs((signs * values).mean(axis=1))
        exceed += int(np.count_nonzero(null >= observed))
        done += count
    return float((exceed + 1) / (permutations + 1))


def holm_adjust(pvalues: dict[int, Optional[float]]) -> dict[int, Optional[float]]:
    valid = sorted(
        ((length, value) for length, value in pvalues.items() if value is not None),
        key=lambda item: item[1],
    )
    adjusted: dict[int, Optional[float]] = {length: None for length in pvalues}
    running = 0.0
    total = len(valid)
    for rank, (length, value) in enumerate(valid):
        candidate = min(1.0, value * (total - rank))
        running = max(running, candidate)
        adjusted[length] = running
    return adjusted


def fmt_ci(stat: dict[str, Any], digits: int = 3, signed: bool = False) -> str:
    if stat.get("mean") is None:
        return "n/a"
    pattern = f"{{:{'+' if signed else ''}.{digits}f}}"
    return (
        f"{pattern.format(stat['mean'])} "
        f"[{pattern.format(stat['low'])}, {pattern.format(stat['high'])}]"
    )


def length_label(length: int) -> str:
    return f"{length // 1024}K" if length % 1024 == 0 else str(length)


def settings_caption(config: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    documents = len({
        row.get("sample_index") for row in rows if row.get("sample_index") is not None
    })
    seeds = len({row.get("mapper_seed") for row in rows})
    parts = [f"docs={documents}", f"mapper seeds={seeds}"]
    settings = (
        ("max_gen_len", "gen"), ("tree_nodes", "nodes"),
        ("chunk_size", "chunk"), ("cache_top_k", "topK"),
        ("retrieve_every", "retrieve-every"),
    )
    for field, label in settings:
        value = config.get(field)
        if value is not None:
            parts.append(f"{label}={value}")
    return ", ".join(parts)


def save_figure(fig: Any, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")


def unavailable(ax: Any, title: str) -> None:
    ax.set_title(title, loc="left", fontweight="bold")
    ax.text(0.5, 0.5, "Metric unavailable", transform=ax.transAxes,
            ha="center", va="center", color="#777777")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_policy_metric(
    ax: Any, rows: list[dict[str, Any]], policies: list[tuple[str, str]],
    lengths: list[int], metric: str, ylabel: str, title: str,
    replicates: int, seed: int,
) -> None:
    present = False
    width = 0.8 / max(1, len(lengths))
    x = np.arange(len(policies), dtype=float)
    for length_index, length in enumerate(lengths):
        means, lows, highs = [], [], []
        for policy_index, (role, policy) in enumerate(policies):
            stat = policy_mean(
                rows, policy, length, metric, replicates,
                seed + length_index * 1009 + policy_index * 37,
            )
            means.append(stat["mean"] if stat["mean"] is not None else np.nan)
            lows.append(stat["low"] if stat["low"] is not None else np.nan)
            highs.append(stat["high"] if stat["high"] is not None else np.nan)
            present |= stat["mean"] is not None
        positions = x + (length_index - (len(lengths) - 1) / 2) * width
        yerr = np.asarray([
            np.asarray(means) - np.asarray(lows),
            np.asarray(highs) - np.asarray(means),
        ])
        ax.bar(
            positions, means, width=width * 0.9, alpha=0.85,
            color=[POLICY_COLORS[role] for role, _ in policies],
            edgecolor="#444444", linewidth=0.45, label=length_label(length),
            hatch=("", "//", "xx", "..") [length_index % 4],
            yerr=yerr, capsize=2,
        )
    if not present:
        unavailable(ax, title)
        return
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, [role.replace("value", "V-only").replace("kv", "K+V").replace("key", "K-only").title() for role, _ in policies])
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8)


def plot_paired_delta(
    ax: Any, rows: list[dict[str, Any]], left: str, right: str,
    lengths: list[int], metric: str, ylabel: str, title: str,
    replicates: int, seed: int,
) -> None:
    stats = [
        paired_stat(rows, left, right, length, metric, replicates, seed + index * 101)
        for index, length in enumerate(lengths)
    ]
    if not any(stat["mean"] is not None for stat in stats):
        unavailable(ax, title)
        return
    means = np.asarray([stat["mean"] if stat["mean"] is not None else np.nan for stat in stats])
    low = np.asarray([stat["low"] if stat["low"] is not None else np.nan for stat in stats])
    high = np.asarray([stat["high"] if stat["high"] is not None else np.nan for stat in stats])
    ax.errorbar(
        np.arange(len(lengths)), means, yerr=[means - low, high - means],
        fmt="o", color=POLICY_COLORS["value"], ecolor="#555555", capsize=4,
        markersize=7,
    )
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xticks(np.arange(len(lengths)), [length_label(length) for length in lengths])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)


def make_quality_figure(
    rows: list[dict[str, Any]], policies: dict[str, str], lengths: list[int],
    output_dir: Path, args: argparse.Namespace,
) -> None:
    import matplotlib.pyplot as plt

    roles = [(role, policies[role]) for role in ("native", "key", "value", "kv")]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    plot_policy_metric(
        axes[0, 0], rows, roles, lengths, "avg_accept_length", "Accepted draft tokens / round",
        "A  Proposal quality", args.bootstrap, args.seed,
    )
    plot_paired_delta(
        axes[0, 1], rows, policies["value"], policies["kv"], lengths,
        "avg_accept_length", "V-only - K+V", "B  Primary paired acceptance effect",
        args.bootstrap, args.seed + 100,
    )
    plot_policy_metric(
        axes[1, 0], rows, roles, lengths, "mean_root_target_draft_kl", "Root KL (lower is better)",
        "C  Draft-target root KL", args.bootstrap, args.seed + 200,
    )
    plot_paired_delta(
        axes[1, 1], rows, policies["value"], policies["kv"], lengths,
        "mean_root_target_draft_kl", "V-only - K+V", "D  Paired root-KL effect",
        args.bootstrap, args.seed + 300,
    )
    fig.suptitle(
        f"{args.title}\nQuality; document-paired bootstrap 95% CI; {args.figure_settings}",
        fontsize=14, fontweight="bold",
    )
    save_figure(fig, output_dir / "kv_geometry_quality")
    plt.close(fig)


def make_performance_figure(
    rows: list[dict[str, Any]], policies: dict[str, str], lengths: list[int],
    output_dir: Path, args: argparse.Namespace,
) -> None:
    import matplotlib.pyplot as plt

    # QK rescue is an oracle intervention and is deliberately excluded.
    roles = [(role, policies[role]) for role in ("native", "key", "value", "kv")]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    plot_policy_metric(
        axes[0, 0], rows, roles, lengths, "wall_tokens_per_sec", "End-to-end token/s",
        "A  Actual end-to-end throughput", args.bootstrap, args.seed + 400,
    )
    plot_paired_delta(
        axes[0, 1], rows, policies["value"], policies["kv"], lengths,
        "wall_tokens_per_sec", "V-only - K+V (token/s)", "B  Paired throughput effect",
        args.bootstrap, args.seed + 500,
    )
    plot_policy_metric(
        axes[1, 0], rows, [("value", policies["value"]), ("kv", policies["kv"])],
        lengths, "round_count", "Rounds / fixed output", "C  Verifier rounds (quality-mediated cost)",
        args.bootstrap, args.seed + 600,
    )
    plot_policy_metric(
        axes[1, 1], rows, [("value", policies["value"]), ("kv", policies["kv"])],
        lengths, "bridge_map_ms_per_event", "CUDA ms / bridge event",
        "D  Mapping/blend cost per event", args.bootstrap, args.seed + 700,
    )
    fig.suptitle(
        f"{args.title}\nPerformance; rescue excluded; {args.figure_settings}",
        fontsize=14, fontweight="bold",
    )
    save_figure(fig, output_dir / "kv_geometry_performance")
    plt.close(fig)


def dose_rows(rows: list[dict[str, Any]], length: int) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if row.get("length_bin_tokens") != length:
            continue
        policy = str(row.get("policy", "")).lower()
        if "rescue" in policy or "shuffle" in policy or "recent" in policy:
            continue
        if not math.isclose(float(row.get("alpha_value", -1)), 0.25, abs_tol=1e-8):
            continue
        if float(row.get("alpha_key", -1)) < 0:
            continue
        selected.append(row)
    return selected


def dose_stats(
    rows: list[dict[str, Any]], length: int, metric: str,
    replicates: int, seed: int,
) -> list[dict[str, Any]]:
    selected = dose_rows(rows, length)
    alphas = sorted({round(float(row["alpha_key"]), 8) for row in selected})
    output = []
    for index, alpha in enumerate(alphas):
        subset = [row for row in selected if math.isclose(row["alpha_key"], alpha, abs_tol=1e-8)]
        values = np.asarray(list(collapse_values_by_document(subset, metric).values()))
        stat = bootstrap_mean(values, replicates, seed + index * 31)
        stat["alpha_key"] = alpha
        output.append(stat)
    return output


def plot_dose(
    ax: Any, rows: list[dict[str, Any]], lengths: list[int], metric: str,
    ylabel: str, title: str, args: argparse.Namespace, seed_offset: int,
) -> None:
    present = False
    for index, length in enumerate(lengths):
        stats = dose_stats(rows, length, metric, args.bootstrap, args.seed + seed_offset + index * 101)
        stats = [stat for stat in stats if stat["mean"] is not None]
        if not stats:
            continue
        present = True
        x = np.asarray([stat["alpha_key"] for stat in stats])
        y = np.asarray([stat["mean"] for stat in stats])
        low = np.asarray([stat["low"] for stat in stats])
        high = np.asarray([stat["high"] for stat in stats])
        ax.errorbar(x, y, yerr=[y - low, high - y], marker="o", capsize=3,
                    label=length_label(length))
    if not present:
        unavailable(ax, title)
        return
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(r"Key blend $\alpha_K$ (fixed $\alpha_V=0.25$)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def correlation(x: np.ndarray, y: np.ndarray, spearman: bool = True) -> Optional[float]:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    if spearman:
        x, y = rankdata(x), rankdata(y)
    return float(np.corrcoef(x, y)[0, 1])


def within_document_relation(
    rows: list[dict[str, Any]], x_metric: str, y_metric: str,
    replicates: int, seed: int,
) -> dict[str, Any]:
    usable = [
        row for row in rows
        if finite(row.get(x_metric)) is not None and finite(row.get(y_metric)) is not None
        and "rescue" not in str(row.get("policy", "")).lower()
    ]
    # Remove nuisance baselines within the exact repeated-measures stratum.
    # In particular, mapper seeds can have different overall geometry and
    # acceptance levels; centering only at (length, document) would leave those
    # seed shifts in both variables and could create a spurious correlation.
    strata: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        strata[unit_key(row)].append(row)

    # The independent sampling unit remains the source document.  Each cluster
    # therefore contains every context length, mapper seed, and dose-policy
    # stratum belonging to that document.
    document_clusters: dict[Any, list[list[dict[str, Any]]]] = defaultdict(list)
    for (_, sample_index, _), group in strata.items():
        if len(group) >= 2:
            document_clusters[sample_index].append(group)

    def residual_arrays(
        clusters: list[list[list[dict[str, Any]]]],
    ) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for cluster in clusters:
            for group in cluster:
                x = np.asarray([float(row[x_metric]) for row in group])
                y = np.asarray([float(row[y_metric]) for row in group])
                xs.extend(x - x.mean())
                ys.extend(y - y.mean())
        return np.asarray(xs), np.asarray(ys)

    clusters = list(document_clusters.values())
    x, y = residual_arrays(clusters)
    observed = correlation(x, y)
    if observed is None or not clusters:
        return {
            "spearman": None, "low": None, "high": None,
            "documents": len(clusters), "strata": len(strata), "points": len(x),
        }
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        chosen = rng.integers(0, len(clusters), size=len(clusters))
        bx, by = residual_arrays([clusters[index] for index in chosen])
        value = correlation(bx, by)
        if value is not None:
            samples.append(value)
    low, high = percentile_ci(np.asarray(samples)) if samples else (None, None)
    return {
        "spearman": observed, "low": low, "high": high,
        "documents": len(clusters), "strata": len(strata), "points": len(x),
    }


def scatter_geometry_acceptance(ax: Any, rows: list[dict[str, Any]], relation: dict[str, Any]) -> None:
    metric = "geometry_attention_kl_native_to_bridge"
    points = [
        row for row in rows
        if finite(row.get(metric)) is not None and finite(row.get("avg_accept_length")) is not None
        and "rescue" not in str(row.get("policy", "")).lower()
    ]
    if not points:
        unavailable(ax, "E  Geometry-performance relation")
        return
    for length in sorted({row["length_bin_tokens"] for row in points if row.get("length_bin_tokens") is not None}):
        subset = [row for row in points if row.get("length_bin_tokens") == length]
        ax.scatter(
            [row[metric] for row in subset], [row["avg_accept_length"] for row in subset],
            s=18, alpha=0.55, label=length_label(length),
        )
    rho = relation.get("spearman")
    ci = ""
    if rho is not None:
        ci = f"\nwithin-doc rho={rho:.2f} [{relation['low']:.2f}, {relation['high']:.2f}]"
    ax.set_title("E  Attention divergence vs acceptance" + ci, loc="left", fontweight="bold")
    ax.set_xlabel("Post-RoPE attention KL (native || bridge)")
    ax.set_ylabel("Accepted draft tokens / round")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)


def rescue_panel(
    ax: Any, rows: list[dict[str, Any]], policies: dict[str, str], lengths: list[int],
    args: argparse.Namespace,
) -> None:
    metrics = (
        ("avg_accept_length", "Acceptance", 1.0),
        ("geometry_attention_kl_native_to_bridge", "Attention KL", -1.0),
    )
    x, labels, means, lower, upper, colors = [], [], [], [], [], []
    position = 0
    for length in lengths:
        for metric, label, direction in metrics:
            stat = paired_stat(
                rows, policies["rescue"], policies["kv"], length, metric,
                args.bootstrap, args.seed + 1100 + position * 31,
            )
            if stat["mean"] is None:
                continue
            x.append(position)
            labels.append(f"{length_label(length)}\n{label}")
            means.append(stat["mean"] * direction)
            lower.append(stat["low"] * direction)
            upper.append(stat["high"] * direction)
            if direction < 0:
                lower[-1], upper[-1] = upper[-1], lower[-1]
            colors.append(POLICY_COLORS["rescue"])
            position += 1
    if not x:
        unavailable(ax, "F  Oracle QK-rescue mechanism intervention")
        return
    means_array = np.asarray(means)
    ax.bar(x, means, color=colors, alpha=0.85)
    ax.errorbar(x, means, yerr=[means_array - np.asarray(lower), np.asarray(upper) - means_array],
                fmt="none", ecolor="#333333", capsize=3)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("Improvement over raw K+V\n(KL sign reversed)")
    ax.set_title("F  Oracle QK-rescue mechanism intervention", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)


def make_geometry_figure(
    rows: list[dict[str, Any]], policies: dict[str, str], lengths: list[int],
    output_dir: Path, args: argparse.Namespace, relation: dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.5), constrained_layout=True)
    plot_dose(
        axes[0, 0], rows, lengths, "avg_accept_length", "Accepted draft tokens / round",
        "A  Key-dose response: proposal quality", args, 800,
    )
    plot_dose(
        axes[0, 1], rows, lengths, "geometry_attention_kl_native_to_bridge",
        "Post-RoPE attention KL", "B  Key-dose response: attention geometry", args, 850,
    )
    plot_dose(
        axes[0, 2], rows, lengths, "geometry_topk_jaccard", "Top-k attention Jaccard",
        "C  Key-dose response: routing overlap", args, 900,
    )
    roles = [(role, policies[role]) for role in ("native", "key", "value", "kv", "rescue")]
    plot_policy_metric(
        axes[1, 0], rows, roles, lengths, "geometry_qk_logit_relative_rmse",
        "Relative QK-logit RMSE", "D  Actual post-RoPE QK distortion",
        args.bootstrap, args.seed + 950,
    )
    scatter_geometry_acceptance(axes[1, 1], rows, relation)
    rescue_panel(axes[1, 2], rows, policies, lengths, args)
    fig.suptitle(
        f"{args.title}\nPost-RoPE Q/K; weighted summaries; {args.figure_settings}",
        fontsize=14, fontweight="bold",
    )
    save_figure(fig, output_dir / "kv_geometry_geometry")
    plt.close(fig)


def infer_policies(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, str]:
    policies = {
        "native": args.native_policy,
        "value": args.value_policy,
        "key": args.key_policy,
        "kv": args.kv_policy,
        "rescue": args.rescue_policy,
    }
    available = {str(row.get("policy")) for row in rows}
    # Rescue had one historical spelling during implementation; select it only
    # if the requested canonical name is absent.
    if policies["rescue"] not in available:
        candidates = sorted(policy for policy in available if "rescue" in policy.lower())
        if candidates:
            policies["rescue"] = candidates[0]
    return policies


def cell_counts(
    rows: list[dict[str, Any]], policies: dict[str, str], lengths: list[int]
) -> dict[str, int]:
    result = {}
    for length in lengths:
        for role, policy in policies.items():
            documents = {row.get("sample_index") for row in rows_for(rows, policy, length)}
            documents.discard(None)
            result[f"{length}/{role}"] = len(documents)
    return result


def output_ids(row: dict[str, Any]) -> Optional[list[int]]:
    value = row.get("output_token_ids")
    if not isinstance(value, list):
        return None
    try:
        return [int(token) for token in value]
    except (TypeError, ValueError):
        return None


def first_mismatch(left: list[int], right: list[int], limit: int) -> Optional[int]:
    common = min(len(left), len(right), limit)
    for index in range(common):
        if left[index] != right[index]:
            return index
    if common < limit and len(left) != len(right):
        return common
    return None


def output_consistency(
    rows: list[dict[str, Any]], reference_policy: str, lengths: list[int],
    compare_tokens: int = 128,
) -> dict[str, Any]:
    references = {
        unit_key(row): row for row in rows if row.get("policy") == reference_policy
    }
    policies = sorted({str(row.get("policy")) for row in rows})
    by_length: dict[str, Any] = {}
    all_mismatches: list[dict[str, Any]] = []
    for length in lengths:
        length_block: dict[str, Any] = {}
        for policy in policies:
            candidates = rows_for(rows, policy, length)
            eligible = 0
            exact = 0
            unavailable = 0
            mismatches = []
            positions: dict[str, int] = defaultdict(int)
            for candidate in candidates:
                reference = references.get(unit_key(candidate))
                candidate_ids = output_ids(candidate)
                reference_ids = output_ids(reference) if reference is not None else None
                if (
                    candidate_ids is None or reference_ids is None
                    or len(candidate_ids) < compare_tokens or len(reference_ids) < compare_tokens
                ):
                    unavailable += 1
                    continue
                eligible += 1
                position = first_mismatch(candidate_ids, reference_ids, compare_tokens)
                if position is None:
                    exact += 1
                    continue
                positions[str(position)] += 1
                detail = {
                    "length_bin_tokens": length,
                    "sample_index": candidate.get("sample_index"),
                    "mapper_seed": candidate.get("mapper_seed"),
                    "policy": policy,
                    "first_mismatch_position": position,
                    "reference_token": reference_ids[position],
                    "candidate_token": candidate_ids[position],
                }
                mismatches.append(detail)
                all_mismatches.append(detail)
            length_block[policy] = {
                "eligible": eligible,
                "exact": exact,
                "exact_rate": (exact / eligible) if eligible else None,
                "unavailable": unavailable,
                "mismatch_count": len(mismatches),
                "first_mismatch_position_counts": dict(
                    sorted(positions.items(), key=lambda item: int(item[0]))
                ),
                "mismatches": mismatches,
            }
        by_length[str(length)] = length_block
    return {
        "reference_policy": reference_policy,
        "compare_tokens": compare_tokens,
        "unit": "(mapper_seed, length_bin_tokens, sample_index)",
        "meaning": (
            "Greedy full-output consistency diagnostic only; this is not a proof "
            "that the stochastic output distribution is unchanged."
        ),
        "by_length": by_length,
        "mismatches": all_mismatches,
    }


def canonical_reference_key(row: dict[str, Any]) -> Optional[tuple[int, Any]]:
    length = infer_length(row)
    sample = infer_sample(row)
    if length is None or sample is None:
        return None
    try:
        sample = int(sample)
    except (TypeError, ValueError):
        sample = str(sample)
    return int(length), sample


def canonical_target_reference_map(
    rows: list[dict[str, Any]],
) -> dict[tuple[int, Any], dict[str, Any]]:
    references: dict[tuple[int, Any], dict[str, Any]] = {}
    for row in rows:
        policy = str(row.get("policy", "target_ar"))
        if policy != "target_ar":
            raise ValueError(
                f"Canonical --reference-run contains non-target_ar policy: {policy}"
            )
        key = canonical_reference_key(row)
        if key is None:
            raise ValueError(
                "Canonical target_ar row is missing length_bin_tokens or sample_index"
            )
        if output_ids(row) is None:
            raise ValueError(
                f"Canonical target_ar row {key} is missing valid output_token_ids"
            )
        previous = references.get(key)
        if previous is not None:
            if output_ids(previous) != output_ids(row):
                raise ValueError(
                    f"Conflicting canonical target_ar rows for (length, sample)={key}"
                )
            continue
        references[key] = row
    return references


def target_ar_output_consistency(
    formal_rows: list[dict[str, Any]],
    references: dict[tuple[int, Any], dict[str, Any]],
    lengths: list[int],
    compare_tokens: int = 128,
) -> dict[str, Any]:
    """Compare every mapper-seed policy row with a seedless target-AR run."""
    policies = sorted({str(row.get("policy")) for row in formal_rows})
    by_length: dict[str, Any] = {}
    all_mismatches: list[dict[str, Any]] = []
    totals = {
        "eligible": 0,
        "exact": 0,
        "reference_missing": 0,
        "reference_short": 0,
        "candidate_short_or_missing": 0,
    }
    for length in lengths:
        length_block: dict[str, Any] = {}
        for policy in policies:
            candidates = rows_for(formal_rows, policy, length)
            eligible = 0
            exact = 0
            missing_reference = 0
            short_reference = 0
            short_candidate = 0
            positions: dict[str, int] = defaultdict(int)
            mismatches = []
            for candidate in candidates:
                key = canonical_reference_key(candidate)
                reference = references.get(key) if key is not None else None
                candidate_ids = output_ids(candidate)
                reference_ids = output_ids(reference) if reference is not None else None
                if reference is None:
                    missing_reference += 1
                    continue
                if reference_ids is None or len(reference_ids) < compare_tokens:
                    short_reference += 1
                    continue
                if candidate_ids is None or len(candidate_ids) < compare_tokens:
                    short_candidate += 1
                    continue
                eligible += 1
                position = first_mismatch(candidate_ids, reference_ids, compare_tokens)
                if position is None:
                    exact += 1
                    continue
                positions[str(position)] += 1
                margins = reference.get("top1_margin_list")
                canonical_margin = (
                    finite(margins[position])
                    if isinstance(margins, list) and position < len(margins)
                    else None
                )
                detail = {
                    "length_bin_tokens": length,
                    "sample_index": candidate.get("sample_index"),
                    "mapper_seed": candidate.get("mapper_seed"),
                    "policy": policy,
                    "first_mismatch_position": position,
                    "target_ar_token": reference_ids[position],
                    "candidate_token": candidate_ids[position],
                    "canonical_top1_margin": canonical_margin,
                }
                mismatches.append(detail)
                all_mismatches.append(detail)
            block = {
                "eligible": eligible,
                "exact": exact,
                "exact_rate": (exact / eligible) if eligible else None,
                "reference_missing": missing_reference,
                "reference_short": short_reference,
                "candidate_short_or_missing": short_candidate,
                "mismatch_count": len(mismatches),
                "first_mismatch_position_counts": dict(
                    sorted(positions.items(), key=lambda item: int(item[0]))
                ),
                "mismatches": mismatches,
            }
            length_block[policy] = block
            totals["eligible"] += eligible
            totals["exact"] += exact
            totals["reference_missing"] += missing_reference
            totals["reference_short"] += short_reference
            totals["candidate_short_or_missing"] += short_candidate
        by_length[str(length)] = length_block
    totals["exact_rate"] = (
        totals["exact"] / totals["eligible"] if totals["eligible"] else None
    )
    margin_groups: dict[tuple[int, Any, int, Optional[float]], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for item in all_mismatches:
        margin_groups[
            (
                int(item["length_bin_tokens"]),
                item["sample_index"],
                int(item["first_mismatch_position"]),
                finite(item.get("canonical_top1_margin")),
            )
        ].append(item)
    margin_locations = []
    for (length, sample, position, margin), items in sorted(
        margin_groups.items(), key=lambda entry: str(entry[0])
    ):
        margin_locations.append(
            {
                "length_bin_tokens": length,
                "sample_index": sample,
                "first_mismatch_position": position,
                "canonical_top1_margin": margin,
                "affected_formal_runs": len(items),
                "policies": sorted({str(item["policy"]) for item in items}),
                "mapper_seeds": sorted(
                    {item["mapper_seed"] for item in items}, key=str
                ),
            }
        )
    return {
        "reference_policy": "target_ar",
        "reference_key": "(length_bin_tokens, sample_index); mapper seed intentionally ignored",
        "reference_row_count": len(references),
        "compare_tokens": compare_tokens,
        "formal_unit": "(mapper_seed, length_bin_tokens, sample_index, policy)",
        "meaning": (
            "Numerical greedy-path diagnostic against canonical target-only AR; "
            "this is not a proof that the stochastic output distribution is unchanged."
        ),
        "overall": totals,
        "by_length": by_length,
        "mismatches": all_mismatches,
        "mismatch_margin_diagnostic": {
            "definition": "canonical target-AR top1 logit minus top2 logit at first mismatch",
            "available_failure_rows": sum(
                item["canonical_top1_margin"] is not None for item in all_mismatches
            ),
            "missing_failure_rows": sum(
                item["canonical_top1_margin"] is None for item in all_mismatches
            ),
            "unique_mismatch_locations": margin_locations,
            "interpretation": (
                "Small margins are consistent with a near-tie numerical branch, "
                "but margin alone does not identify its cause and is not a "
                "distributional exactness proof."
            ),
        },
    }


def timing_exclusion_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    excluded = [row for row in rows if row.get("_timing_excluded")]
    diagnostic_policies = sorted({
        str(row.get("policy")) for row in rows
        if row.get("wall_timing_is_diagnostic")
    })
    return {
        "excluded_row_count": len(excluded),
        "excluded_policies": sorted({str(row.get("policy")) for row in excluded}),
        "diagnostic_policies": diagnostic_policies,
        "rescue_policy_always_excluded_from_performance_figure": True,
    }


def primary_statistics(
    rows: list[dict[str, Any]], policies: dict[str, str], lengths: list[int],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[int, Optional[float]], dict[int, Optional[float]]]:
    output: dict[str, Any] = {}
    raw_p = {}
    for index, length in enumerate(lengths):
        key = str(length)
        output[key] = {}
        for metric in (
            "avg_accept_length", "wall_tokens_per_sec", "mean_root_target_draft_kl",
            "wall_time", "round_count", "bridge_map_ms_per_event",
        ):
            output[key][metric] = paired_stat(
                rows, policies["value"], policies["kv"], length, metric,
                args.bootstrap, args.seed + 1200 + index * 101 + len(output[key]),
            )
        differences, _, _, _ = paired_document_differences(
            rows, policies["value"], policies["kv"], length, "avg_accept_length"
        )
        raw_p[length] = sign_flip_pvalue(differences, args.permutations, args.seed + length)
    adjusted = holm_adjust(raw_p)
    for length in lengths:
        output[str(length)]["avg_accept_length"]["sign_flip_p"] = raw_p[length]
        output[str(length)]["avg_accept_length"]["holm_p"] = adjusted[length]
    return output, raw_p, adjusted


def contrast_block(
    rows: list[dict[str, Any]], left: str, right: str, lengths: list[int],
    metrics: Iterable[str], args: argparse.Namespace, seed_offset: int,
) -> dict[str, Any]:
    return {
        str(length): {
            metric: paired_stat(
                rows, left, right, length, metric, args.bootstrap,
                args.seed + seed_offset + length + index * 37,
            )
            for index, metric in enumerate(metrics)
        }
        for length in lengths
    }


def key_only_contrast_block(
    rows: list[dict[str, Any]], policies: dict[str, str], lengths: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    ordinary_metrics = ("avg_accept_length", "mean_root_target_draft_kl")
    geometry_metrics = (
        "geometry_attention_kl_native_to_bridge",
        "geometry_argmax_flip_rate",
    )
    output: dict[str, Any] = {}
    for length_index, length in enumerate(lengths):
        block = {
            metric: paired_stat(
                rows, policies["key"], policies["native"], length, metric,
                args.bootstrap, args.seed + 1600 + length_index * 101 + metric_index * 37,
            )
            for metric_index, metric in enumerate(ordinary_metrics)
        }
        for metric_index, metric in enumerate(geometry_metrics):
            block[metric] = paired_stat_against_definitional_baseline(
                rows, policies["key"], policies["native"], length, metric,
                NATIVE_GEOMETRY_BASELINES[metric], args.bootstrap,
                args.seed + 1700 + length_index * 101 + metric_index * 37,
            )
        output[str(length)] = block
    return output


def seedwise_primary_contrasts(
    rows: list[dict[str, Any]], policies: dict[str, str], lengths: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    seeds = sorted({row.get("mapper_seed") for row in rows}, key=str)
    output: dict[str, Any] = {}
    for seed_index, mapper_seed in enumerate(seeds):
        seed_rows = [row for row in rows if row.get("mapper_seed") == mapper_seed]
        output[str(mapper_seed)] = contrast_block(
            seed_rows, policies["value"], policies["kv"], lengths,
            ("avg_accept_length", "wall_tokens_per_sec", "mean_root_target_draft_kl"),
            args, 2200 + seed_index * 1000,
        )
    return output


def descriptive_primary_contrasts(
    rows: list[dict[str, Any]], policies: dict[str, str], lengths: list[int],
    args: argparse.Namespace, seed_offset: int,
) -> dict[str, Any]:
    return contrast_block(
        rows, policies["value"], policies["kv"], lengths,
        ("avg_accept_length", "wall_tokens_per_sec", "mean_root_target_draft_kl"),
        args, seed_offset,
    )


def report_markdown(
    rows: list[dict[str, Any]], policies: dict[str, str], lengths: list[int],
    configs: list[dict[str, Any]], summary: dict[str, Any], args: argparse.Namespace,
) -> str:
    split = summary["data_splits"]
    confirmatory_text = ", ".join(str(index) for index in split["confirmatory_samples"])
    exploratory_text = ", ".join(str(index) for index in split["exploratory_samples"])
    mapper_seed_count = len(summary["seed_robustness"])
    confirmatory_count = len(split["confirmatory_samples"])
    target_diagnostic_scope = (
        "fresh confirmatory rows"
        if split["pooled_is_descriptive_only"]
        else "all supplied rows（未指定独立 confirmatory split）"
    )
    target_scope_note = (
        "pilot rows 仍保留在下方策略内部 pooled 诊断中，不在此处制造缺 reference 计数。"
        if split["pooled_is_descriptive_only"]
        else ""
    )
    lines = [
        "# V-only 与 K+V：注意力几何实验报告", "",
        "![质量指标](kv_geometry_quality.png)", "",
        "![实际系统性能](kv_geometry_performance.png)", "",
        "![注意力几何](kv_geometry_geometry.png)", "",
        "## 实验与统计口径", "",
        "- 主估计量是同一文档、同一 Mapper seed 下 `V-only − K+V`；多个 Mapper seed 先在文档内平均，再以文档为独立单位做配对 bootstrap。",
        f"- 主确认集文档：{confirmatory_text or '全部输入文档'}。"
        + (f" 已观察的 pilot 文档：{exploratory_text}；它们不进入主检验。" if exploratory_text else ""),
        f"- Mapper seed 是同一文档上的训练随机性重复，不是新的独立样本；因此 {mapper_seed_count} 个 seed 不会把 fresh `n={confirmatory_count}` 扩大成 `n={confirmatory_count * mapper_seed_count}`。逐 seed 结果只作为稳健性诊断。",
        "- 合并 pilot 与 fresh 的 pooled 结果仅作描述性敏感性分析，不能替代 fresh confirmatory 推断。" if split["pooled_is_descriptive_only"] else "- 未指定 confirmatory split；当前全部输入文档被用于主分析。",
        f"- 置信区间使用 {args.bootstrap:,} 次文档级 bootstrap；接受长度另做双侧配对 sign-flip 检验，并对上下文长度做 Holm 校正。",
        "- V-only 与 K+V 使用同一 Mapper；当前 Mapper 在两种策略下都生成 K 和 V，V-only 只是不把映射 K 写入活动缓存。因此质量对照近似计算匹配。",
        "- 吞吐图使用真实端到端 token/s，并分别展示生成轮数与每次 bridge 映射/混合开销；不把二者混成一个质量指标。",
        "- 所有 `wall_timing_is_diagnostic=true` 且没有独立正式计时来源的行都从吞吐统计中剔除；`QK-rescue` 会读取 native QK logit，是 oracle 机制干预，因此无论是否带 wall time 都不进入吞吐图和系统结论。",
        "- 主几何指标基于 Draft 实际 post-RoPE query/key。pre-RoPE 向量 cosine 不能单独决定真实的相对位置 QK 内积。",
        "- K-only 与 attention-native 的几何差使用 native-vs-native 数学定义基线：KL/RMSE/flip/投影误差为 0、top-k Jaccard 为 1。该基线明确标为 derived/definitional，不伪装成 native run 的实测值；非几何指标仍严格读取配对运行。",
        "- 策略间输出一致性以同一 `(mapper seed, context length, document)` 的 `attention_native` 为参考，检查完整前 128 个 greedy token；这是数值路径诊断，不是随机采样分布不变的证明。",
    ]
    lines.extend([
        "", "## \u51e0\u4f55\u673a\u5236\u5047\u8bbe", "",
        "\u5f53\u524d\u5b9e\u73b0\u628a mapped K \u4ee5 pre-RoPE \u5f62\u5f0f\u5199\u5165 Draft cache\uff0c\u7136\u540e\u4e0e native K \u4e00\u6837\u6309\u5b9e\u9645 compact-cache position \u7ecf\u8fc7 Draft RoPE\u3002\u56e0\u6b64\u8981\u68c0\u9a8c\u7684\u4e0d\u662f\u201c\u5fd8\u4e86\u505a RoPE\u201d\uff0c\u800c\u662f mapped K \u662f\u5426\u4fdd\u6301\u4e86 query-conditioned \u5bfb\u5740\u51e0\u4f55\u3002", "",
        "\u5bf9 query \u4f4d\u7f6e `t` \u548c key \u4f4d\u7f6e `p`\uff0c\u771f\u5b9e\u6ce8\u610f\u529b\u5206\u6570\u4e3a `s(t,p) = (R_t q)^T(R_p k)/sqrt(d) = q^T R_{p-t} k/sqrt(d)`\u3002\u82e5 Mapper \u7684 Key \u8bef\u5dee\u4e3a `e = k_hat - k`\uff0c\u5219 `Delta s(t,p) = q^T R_{p-t} e/sqrt(d)`\u3002\u6240\u4ee5\u8f83\u9ad8\u7684 pre-RoPE \u5411\u91cf cosine \u4ecd\u4e0d\u4fdd\u8bc1 QK logits\u3001softmax \u6743\u91cd\u3001argmax \u6216 top-k \u8def\u7531\u4e0d\u53d8\u3002", "",
        "\u5199\u6210 `A = softmax(QK^T/sqrt(d))`\u3001`O = AV` \u540e\uff0cV-only \u5728\u56fa\u5b9a `A` \u4e0b\u7684\u6270\u52a8\u8fd1\u4f3c\u662f\u7ebf\u6027\u7684 `Delta O_V = A Delta V`\uff1b\u800c Key \u8bef\u5dee\u5148\u4ea7\u751f `Delta S`\uff0c\u518d\u901a\u8fc7 `softmax(S + Delta S)` \u975e\u7ebf\u6027\u6539\u53d8\u8def\u7531\u77e9\u9635\u3002\u5f53\u524d Mapper \u7684 cosine+NMSE \u5411\u91cf\u635f\u5931\u6ca1\u6709\u76f4\u63a5\u7ea6\u675f\u8fd9\u4e2a layer/head/query/relative-position-specific \u7684 `Delta s`\uff1b\u540e\u6587\u7684 Key \u5242\u91cf\u66f2\u7ebf\u548c oracle QK-rescue \u5c31\u662f\u5bf9\u8be5\u673a\u5236\u7684\u5e72\u9884\u68c0\u9a8c\u3002",
    ])
    target_consistency = summary.get("target_ar_output_consistency")
    if target_consistency is not None:
        overall = target_consistency["overall"]
        rate = (
            "n/a" if overall["exact_rate"] is None
            else f"{100 * overall['exact_rate']:.1f}%"
        )
        lines.extend([
            "", "## Canonical target-AR 128-token greedy 诊断", "",
            f"该表仅使用 {target_diagnostic_scope}；其中每个 formal policy 的每个 Mapper seed 都与相同 `(context length, sample)` 的 seedless `target_ar` 比较，Mapper seed 不会改变 reference。{target_scope_note}该检查只是 greedy 数值路径诊断，不是目标随机分布保持不变的证明。", "",
            f"总体完整前 128-token exact：{overall['exact']}/{overall['eligible']}（{rate}）。", "",
            "| 上下文 | formal 策略 | 完整前128-token exact | 首分歧位置（0-based，位置×次数） | 缺 target AR | target AR 不足128 | candidate 不足128 |",
            "|---:|---|---:|---|---:|---:|---:|",
        ])
        for length in lengths:
            for policy, block in target_consistency["by_length"].get(str(length), {}).items():
                if not any(
                    block[field]
                    for field in (
                        "eligible", "reference_missing", "reference_short",
                        "candidate_short_or_missing",
                    )
                ):
                    continue
                positions = ", ".join(
                    f"{position}×{count}"
                    for position, count in block["first_mismatch_position_counts"].items()
                ) or "—"
                policy_rate = (
                    "n/a" if block["exact_rate"] is None
                    else f"{100 * block['exact_rate']:.1f}%"
                )
                lines.append(
                    f"| {length_label(length)} | `{policy}` | "
                    f"{block['exact']}/{block['eligible']} ({policy_rate}) | {positions} | "
                    f"{block['reference_missing']} | {block['reference_short']} | "
                    f"{block['candidate_short_or_missing']} |"
                )
        length_rollups = []
        for length in lengths:
            blocks = target_consistency["by_length"].get(str(length), {}).values()
            eligible = sum(int(block["eligible"]) for block in blocks)
            exact = sum(int(block["exact"]) for block in blocks)
            if eligible:
                length_rollups.append(
                    f"{length_label(length)} 为 {exact}/{eligible}"
                )
        if length_rollups:
            lines.append("\n按上下文汇总：" + "；".join(length_rollups) + "。")
        if target_consistency["mismatches"]:
            lines.append(
                f"\n共 {len(target_consistency['mismatches'])} 个 formal run 首次偏离 canonical target AR；逐 run 的 seed、sample、token 位置与 token ID 保存在 `kv_geometry_summary.json`。"
            )
            margin_diagnostic = target_consistency["mismatch_margin_diagnostic"]
            locations = margin_diagnostic["unique_mismatch_locations"]
            available_locations = [
                item for item in locations
                if item["canonical_top1_margin"] is not None
            ]
            if available_locations:
                descriptions = []
                for item in available_locations:
                    descriptions.append(
                        f"({length_label(int(item['length_bin_tokens']))}, "
                        f"sample {item['sample_index']}, pos {item['first_mismatch_position']})="
                        f"{float(item['canonical_top1_margin']):.6g}"
                    )
                lines.append(
                    "\n首分歧处 canonical top1−top2 margin（按唯一上下文/文档/位置去重）："
                    + "；".join(descriptions) + "。"
                )
                if (
                    len(available_locations) == len(locations)
                    and max(
                        abs(float(item["canonical_top1_margin"]))
                        for item in available_locations
                    ) <= 0.05
                ):
                    lines.append(
                        "这些数值均接近 0，与 fp16 近并列处发生数值路径分支相符；"
                        "但 margin 只能提示敏感性，不能单独证明分支的具体来源，更不是"
                        "随机采样分布不变的证明。"
                    )
    lines.extend([
        "", "## 策略间 128-token greedy 输出一致性诊断", "",
        "该辅助诊断以同 seed 的 `attention_native` 为参考，覆盖输入的 pooled evaluation runs；它不参与主假设显著性检验。", "",
    ])
    consistency = summary["output_consistency"]
    exact_rows = []
    for length in lengths:
        for policy, block in consistency["by_length"].get(str(length), {}).items():
            if block["eligible"] == 0 and block["unavailable"] == 0:
                continue
            positions = ", ".join(
                f"{position}×{count}"
                for position, count in block["first_mismatch_position_counts"].items()
            ) or "—"
            rate = "n/a" if block["exact_rate"] is None else f"{100 * block['exact_rate']:.1f}%"
            exact_rows.append(
                f"| {length_label(length)} | `{policy}` | {block['exact']}/{block['eligible']} "
                f"({rate}) | {positions} | {block['unavailable']} |"
            )
    if exact_rows:
        lines.extend([
            "| 上下文 | 策略 | 完整前128-token exact | 首分歧位置（0-based，位置×次数） | 不可用 |",
            "|---:|---|---:|---|---:|",
            *exact_rows,
        ])
    else:
        lines.append("当前输入没有可用的 128-token `output_token_ids` 配对。")
    mismatch_details = consistency.get("mismatches", [])
    if mismatch_details:
        lines.extend([
            "", "首分歧明细：", "",
            "| 上下文 | sample | mapper seed | 策略 | 位置 | reference token | candidate token |",
            "|---:|---:|---:|---|---:|---:|---:|",
        ])
        for item in mismatch_details[:100]:
            lines.append(
                f"| {length_label(int(item['length_bin_tokens']))} | {item['sample_index']} | "
                f"{item['mapper_seed']} | `{item['policy']}` | {item['first_mismatch_position']} | "
                f"{item['reference_token']} | {item['candidate_token']} |"
            )
        if len(mismatch_details) > 100:
            lines.append(
                f"\n报告仅列前 100 条；完整 {len(mismatch_details)} 条位于 `kv_geometry_summary.json`。"
            )
    lines.extend([
        "", "这里的 exact 仅表示 greedy continuation 与 attention-native 路径一致；理论上的目标分布不变仍依赖 speculative acceptance/rejection 校正及 canonical Target KV 不受 Bridge 污染。", "",
        "## 主对照：V-only − K+V（仅 fresh confirmatory）", "",
        "| 上下文 | 文档 n | Mapper seeds | Δ接受长度 | Holm p | Δtoken/s | ΔRoot KL | Δ轮数 | Δmap ms/event |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for length in lengths:
        block = summary["primary"][str(length)]
        acceptance = block["avg_accept_length"]
        pvalue = acceptance.get("holm_p")
        lines.append(
            "| " + " | ".join([
                length_label(length), str(acceptance.get("n", 0)),
                str(acceptance.get("mapper_seeds", 0)), fmt_ci(acceptance, signed=True),
                "n/a" if pvalue is None else f"{pvalue:.4g}",
                fmt_ci(block["wall_tokens_per_sec"], 2, True),
                fmt_ci(block["mean_root_target_draft_kl"], 3, True),
                fmt_ci(block["round_count"], 2, True),
                fmt_ci(block["bridge_map_ms_per_event"], 4, True),
            ]) + " |"
        )

    lines.extend([
        "", "接受长度与 Root KL 是 proposal 质量证据；token/s 是实际系统证据。若 V-only 减少轮数，而每次 map 开销接近，则吞吐差主要来自更好的 proposal，而不是少计算一个映射头。", "",
        "## Pilot 与 pooled 敏感性分析（非主检验）", "",
        "| 数据切分 | 上下文 | 文档 n | Δ接受长度 | Δtoken/s | ΔRoot KL |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    sensitivity_sets = []
    if summary["exploratory_sensitivity"]:
        sensitivity_sets.append(("Pilot / exploratory", summary["exploratory_sensitivity"]))
    sensitivity_sets.append(("Pooled descriptive", summary["pooled_descriptive"]))
    for label, sensitivity in sensitivity_sets:
        for length in lengths:
            block = sensitivity[str(length)]
            acceptance = block["avg_accept_length"]
            lines.append(
                f"| {label} | {length_label(length)} | {acceptance.get('n', 0)} | "
                f"{fmt_ci(acceptance, signed=True)} | "
                f"{fmt_ci(block['wall_tokens_per_sec'], 2, True)} | "
                f"{fmt_ci(block['mean_root_target_draft_kl'], 3, True)} |"
            )

    lines.extend([
        "", "这些区间用于检查 fresh 结果是否与 pilot 方向一致。因为 pilot 已参与方法和超参数选择，pooled 区间即使更窄也不能被称为独立确认。", "",
        "## Mapper seed 稳健性（fresh 文档；非独立样本扩充）", "",
        "| Mapper seed | 上下文 | 文档 n | Δ接受长度 | Δtoken/s | ΔRoot KL |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for mapper_seed, seed_result in summary["seed_robustness"].items():
        for length in lengths:
            block = seed_result[str(length)]
            acceptance = block["avg_accept_length"]
            lines.append(
                f"| {mapper_seed} | {length_label(length)} | {acceptance.get('n', 0)} | "
                f"{fmt_ci(acceptance, signed=True)} | "
                f"{fmt_ci(block['wall_tokens_per_sec'], 2, True)} | "
                f"{fmt_ci(block['mean_root_target_draft_kl'], 3, True)} |"
            )

    seed_count = len(summary["seed_robustness"])
    seed_interpretation = (
        "跨 seed 一致方向可说明结果不依赖某一次初始化；正式 `n` 仍等于 fresh 文档数。"
        if seed_count >= 2 else
        "当前只有一个 Mapper seed，尚不能判断训练初始化稳健性；正式 `n` 仍等于 fresh 文档数。"
    )
    lines.extend([
        "", seed_interpretation, "",
        "## K-only 析因：K-only − Native", "",
        "| 上下文 | Δ接受长度 | ΔRoot KL | ΔAttention KL† | ΔArgmax flip† |",
        "|---:|---:|---:|---:|---:|",
    ])
    for length in lengths:
        block = summary["key_only_vs_native"][str(length)]
        lines.append(
            f"| {length_label(length)} | {fmt_ci(block['avg_accept_length'], signed=True)} | "
            f"{fmt_ci(block['mean_root_target_draft_kl'], signed=True)} | "
            f"{fmt_ci(block['geometry_attention_kl_native_to_bridge'], signed=True)} | "
            f"{fmt_ci(block['geometry_argmax_flip_rate'], signed=True)} |"
        )
    lines.extend([
        "", "† Native 几何项采用定义基线 0，而非补造的 formal measurement；每个差值仍要求存在同 `(document, mapper seed, length)` 的 native 配对行。", "",
    ])

    lines.extend([
        "## QK-rescue 机制干预：Rescue − 原始 K+V", "",
        "| 上下文 | Δ接受长度 | ΔRoot KL | ΔAttention KL | ΔQK relative RMSE |",
        "|---:|---:|---:|---:|---:|",
    ])
    for length in lengths:
        block = summary["rescue_vs_kv"][str(length)]
        lines.append(
            f"| {length_label(length)} | {fmt_ci(block['avg_accept_length'], signed=True)} | "
            f"{fmt_ci(block['mean_root_target_draft_kl'], signed=True)} | "
            f"{fmt_ci(block['geometry_attention_kl_native_to_bridge'], signed=True)} | "
            f"{fmt_ci(block['geometry_qk_logit_relative_rmse'], signed=True)} |"
        )

    relation = summary["geometry_acceptance_relation"]
    lines.extend([
        "", "## 几何—接受率关系", "",
        (
            "在文档内去除基线后，post-RoPE attention KL 与接受长度的 Spearman "
            + ("不可计算。" if relation["spearman"] is None else
               f"ρ={relation['spearman']:.3f}，95% CI [{relation['low']:.3f}, {relation['high']:.3f}]"
               f"（{relation['documents']} 个文档，{relation['points']} 个策略点）。")
        ), "",
        "该相关性与 K 剂量曲线是机制一致性证据；真正更强的因果证据来自 K-only 干预和 oracle QK-rescue 是否同时恢复几何与接受长度。", "",
        "## K 剂量曲线（固定 αV=0.25）", "",
        "| 上下文 | αK | 接受长度 | Attention KL | Top-k Jaccard |",
        "|---:|---:|---:|---:|---:|",
    ])
    for length in lengths:
        dose = summary["dose_response"].get(str(length), {})
        alphas = sorted({float(alpha) for metric in dose.values() for alpha in metric}) if dose else []
        for alpha in alphas:
            key = f"{alpha:g}"
            lines.append(
                f"| {length_label(length)} | {alpha:g} | "
                f"{fmt_ci(dose.get('avg_accept_length', {}).get(key, {}))} | "
                f"{fmt_ci(dose.get('geometry_attention_kl_native_to_bridge', {}).get(key, {}))} | "
                f"{fmt_ci(dose.get('geometry_topk_jaccard', {}).get(key, {}))} |"
            )

    primary_ok = all(
        summary["primary"][str(length)]["avg_accept_length"].get("low") is not None
        and summary["primary"][str(length)]["avg_accept_length"]["low"] > 0
        and summary["primary"][str(length)]["avg_accept_length"].get("holm_p") is not None
        and summary["primary"][str(length)]["avg_accept_length"]["holm_p"] < 0.05
        for length in lengths
    )
    rescue_available = any(
        summary["rescue_vs_kv"][str(length)]["avg_accept_length"].get("mean") is not None
        for length in lengths
    )
    relation_negative = relation.get("high") is not None and relation["high"] < 0
    lines.extend(["", "## 结论", ""])
    if primary_ok:
        lines.append("1. fresh confirmatory 主对照支持 V-only 在所有已测上下文上具有更高的接受长度；具体系统优势须同时以 token/s 的配对区间判断。")
    else:
        lines.append("1. 当前主对照尚不能在所有上下文上确认 V-only 优于 K+V；应按区间而不是单个均值表述。")
    if relation_negative:
        lines.append("2. attention KL 越高、接受长度越低的文档内关系区间不跨 0，支持注意力几何失配解释。")
    else:
        lines.append("2. attention KL 与接受长度的关系尚未形成区间不跨 0 的证据，几何解释目前应写为机制一致性而非定论。")
    if rescue_available:
        lines.append("3. QK-rescue 是 oracle 机制检验：只有当它同时降低几何误差并恢复接受长度时，才能把 K+V 劣势更强地归因于 QK 几何失配；其吞吐无系统意义。")
    else:
        lines.append("3. 尚缺 QK-rescue 数据；没有修复实验时，K 几何指标与接受长度只能构成干预加相关证据。")
    if summary["partial"]:
        lines.append("4. 当前数据网格不完整，所有图表与结论均为 partial，不应作为最终论文数值。")

    lines.extend([
        "", "## 产物", "",
        "- [质量图 PNG](kv_geometry_quality.png) / [SVG](kv_geometry_quality.svg)",
        "- [性能图 PNG](kv_geometry_performance.png) / [SVG](kv_geometry_performance.svg)",
        "- [几何图 PNG](kv_geometry_geometry.png) / [SVG](kv_geometry_geometry.svg)",
        "- [机器可读汇总](kv_geometry_summary.json)",
        "- [分析代码](../../../analyze_kv_geometry.py)",
    ])
    for index, path in enumerate(args.formal_run, start=1):
        lines.append(f"- [正式 run {index} 原始结果](<{Path(path).resolve()}>)")
    for index, path in enumerate(args.reference_run, start=1):
        lines.append(f"- [Canonical target AR {index}](<{Path(path).resolve()}>)")
    for index, path in enumerate(args.geometry_run, start=1):
        lines.append(f"- [Geometry run {index} 原始结果](<{Path(path).resolve()}>)")
    return "\n".join(lines) + "\n"


def make_dose_summary(
    rows: list[dict[str, Any]], lengths: list[int], args: argparse.Namespace,
) -> dict[str, Any]:
    metrics = (
        "avg_accept_length", "mean_root_target_draft_kl",
        "geometry_attention_kl_native_to_bridge", "geometry_qk_logit_relative_rmse",
        "geometry_topk_jaccard", "geometry_argmax_flip_rate",
    )
    result = {}
    for length in lengths:
        result[str(length)] = {}
        for metric_index, metric in enumerate(metrics):
            stats = dose_stats(
                rows, length, metric, args.bootstrap,
                args.seed + 1400 + length + metric_index * 101,
            )
            result[str(length)][metric] = {
                f"{stat['alpha_key']:g}": stat for stat in stats
            }
    return result


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    formal_rows: list[dict[str, Any]] = []
    formal_configs: list[dict[str, Any]] = []
    for path in args.formal_run:
        raw_formal, config, formal_specs = read_payload(path)
        formal_rows.extend(normalize_row(row, formal_specs) for row in raw_formal)
        formal_configs.append(config)
    formal_config = formal_configs[0] if formal_configs else {}
    target_reference_rows: list[dict[str, Any]] = []
    for path in args.reference_run:
        raw_reference, _, reference_specs = read_payload(path)
        target_reference_rows.extend(
            normalize_row(row, reference_specs) for row in raw_reference
        )
    target_references = (
        canonical_target_reference_map(target_reference_rows)
        if args.reference_run
        else None
    )
    if args.reference_run and not target_references:
        raise SystemExit("No canonical target_ar rows found in --reference-run")
    geometry_rows: list[dict[str, Any]] = []
    configs = list(formal_configs)
    for path in args.geometry_run:
        raw, config, specs = read_payload(path)
        geometry_rows.extend(normalize_row(row, specs) for row in raw)
        configs.append(config)
    rows = merge_rows(formal_rows, geometry_rows)
    add_inferred_fields(rows)
    lengths = sorted({
        int(row["length_bin_tokens"]) for row in rows
        if row.get("length_bin_tokens") is not None
    })
    if not rows or not lengths:
        raise SystemExit("No analyzable run rows or context lengths found")
    policies = infer_policies(rows, args)
    confirmatory_indices = parse_sample_spec(args.confirmatory_samples)
    exploratory_indices = parse_sample_spec(args.exploratory_samples)
    if (
        confirmatory_indices is not None and exploratory_indices is not None
        and confirmatory_indices & exploratory_indices
    ):
        overlap = sorted(confirmatory_indices & exploratory_indices)
        raise SystemExit(f"Confirmatory and exploratory sample sets overlap: {overlap}")
    confirmatory_rows = select_samples(rows, confirmatory_indices)
    if confirmatory_indices is not None and not confirmatory_rows:
        raise SystemExit("No rows match --confirmatory-samples")
    confirmatory_formal_rows = select_samples(formal_rows, confirmatory_indices)
    exploratory_rows = select_samples(rows, exploratory_indices) if exploratory_indices else []
    if confirmatory_indices is None:
        pooled_rows = list(rows)
    elif exploratory_indices is None:
        pooled_rows = list(rows)
    else:
        pooled_rows = select_samples(rows, confirmatory_indices | exploratory_indices)

    counts = cell_counts(confirmatory_rows, policies, lengths)
    partial = False
    if args.expected_samples is not None:
        required_roles = ("native", "key", "value", "kv")
        partial = any(
            counts.get(f"{length}/{role}", 0) < args.expected_samples
            for length in lengths for role in required_roles
        )

    primary, _, _ = primary_statistics(confirmatory_rows, policies, lengths, args)
    if any(
        primary[str(length)]["avg_accept_length"].get("incomplete_documents_excluded", 0) > 0
        for length in lengths
    ):
        partial = True
    key_only = key_only_contrast_block(
        confirmatory_rows, policies, lengths, args
    )
    rescue = contrast_block(
        confirmatory_rows, policies["rescue"], policies["kv"], lengths,
        ("avg_accept_length", "mean_root_target_draft_kl",
         "geometry_attention_kl_native_to_bridge", "geometry_qk_logit_relative_rmse"),
        args, 1800,
    )
    relation = within_document_relation(
        dose_rows(confirmatory_rows, lengths[0]) if len(lengths) == 1 else [
            row for length in lengths for row in dose_rows(confirmatory_rows, length)
        ],
        "geometry_attention_kl_native_to_bridge", "avg_accept_length",
        args.bootstrap, args.seed + 2000,
    )
    summary = {
        "input": {
            "formal_runs": [str(Path(path).resolve()) for path in args.formal_run],
            "reference_runs": [str(Path(path).resolve()) for path in args.reference_run],
            "geometry_runs": [str(Path(path).resolve()) for path in args.geometry_run],
        },
        "analysis": {
            "bootstrap_replicates": args.bootstrap,
            "sign_flip_permutations": args.permutations,
            "seed": args.seed,
            "independent_unit": "document; mapper seeds averaged within document",
            "geometry_aggregation": "explicit event weight",
            "qk_space": "post-RoPE",
            "rescue_is_oracle": True,
            "native_geometry_baseline": {
                "kind": "definitional native-vs-native identity; not measured",
                "values": NATIVE_GEOMETRY_BASELINES,
                "scope": "K-only versus attention_native geometry contrasts only",
            },
            "primary_split": (
                "fresh confirmatory documents" if confirmatory_indices is not None
                else "all supplied documents (no confirmatory split specified)"
            ),
        },
        "data_splits": {
            "confirmatory_samples": (
                sorted(confirmatory_indices) if confirmatory_indices is not None
                else sorted({
                    int(row["sample_index"]) for row in confirmatory_rows
                    if row.get("sample_index") is not None
                })
            ),
            "exploratory_samples": (
                sorted(exploratory_indices) if exploratory_indices is not None else []
            ),
            "pooled_is_descriptive_only": confirmatory_indices is not None,
        },
        "policies": policies,
        "lengths": lengths,
        "cell_document_counts": counts,
        "partial": partial,
        "timing_exclusions": timing_exclusion_summary(pooled_rows),
        "output_consistency": output_consistency(
            pooled_rows, policies["native"], lengths, compare_tokens=128
        ),
        "target_ar_output_consistency": (
            {
                "scope": (
                    "fresh confirmatory rows"
                    if confirmatory_indices is not None
                    else "all supplied formal rows"
                ),
                **target_ar_output_consistency(
                    confirmatory_formal_rows,
                    target_references,
                    lengths,
                    compare_tokens=128,
                ),
            }
            if target_references is not None
            else None
        ),
        "primary": primary,
        "exploratory_sensitivity": (
            descriptive_primary_contrasts(
                exploratory_rows, policies, lengths, args, 2400
            ) if exploratory_rows else {}
        ),
        "pooled_descriptive": descriptive_primary_contrasts(
            pooled_rows, policies, lengths, args, 2600
        ),
        "seed_robustness": seedwise_primary_contrasts(
            confirmatory_rows, policies, lengths, args
        ),
        "key_only_vs_native": key_only,
        "rescue_vs_kv": rescue,
        "dose_response": make_dose_summary(confirmatory_rows, lengths, args),
        "geometry_acceptance_relation": relation,
    }

    split_label = "fresh confirmatory" if confirmatory_indices is not None else "all supplied"
    args.figure_settings = f"{split_label}; {settings_caption(formal_config, confirmatory_rows)}"
    make_quality_figure(confirmatory_rows, policies, lengths, output_dir, args)
    make_performance_figure(confirmatory_rows, policies, lengths, output_dir, args)
    make_geometry_figure(confirmatory_rows, policies, lengths, output_dir, args, relation)
    report = report_markdown(confirmatory_rows, policies, lengths, configs, summary, args)
    (output_dir / "KV_GEOMETRY_REPORT.md").write_text(report, encoding="utf-8")
    (output_dir / "kv_geometry_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote V-only/K+V analysis to {output_dir}")


if __name__ == "__main__":
    main()
