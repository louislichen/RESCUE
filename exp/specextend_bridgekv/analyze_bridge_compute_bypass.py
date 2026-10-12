#!/usr/bin/env python3
"""Analyze the BridgeKV prefill-V compute-bypass experiment.

The experiment deliberately contains technical timing repetitions.  Policy
levels use a within-document median.  Contrasts preserve execution-block
pairing: first compute each effect inside a matching timing repeat, then take
the within-document median of those effects.  Documents, rather than repeats,
are the statistical units, and effect intervals are paired document bootstraps.

This analysis is intentionally strict about provenance and compute accounting.
Replacement intervals, K accounting, and layerwise V identities are hard
checks.  Output IDs and accepted lengths are also compared, but the fair
single-full-GEMM shadow and complement-GEMM bypass may differ at FP16 rounding
level because their matrix shapes differ; exact trajectory equality is thus a
reported diagnostic rather than a prerequisite for the full-shape control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np


DEFAULT_POLICIES = {
    "native": "attention_native",
    "recent_native": "recent_native",
    "shadow": "attention_bridge_value_shadow_last_a100",
    "bypass": "attention_bridge_value_bypass_last_a100",
    "recent_bypass": "recent_bridge_value_bypass_last_a100",
    "post_cache": "attention_bridge_value_a100",
}

POLICY_LABELS = {
    "native": "Attention native",
    "recent_native": "Recent native",
    "shadow": "Causal shadow",
    "bypass": "Attention bypass",
    "recent_bypass": "Recent bypass",
    "post_cache": "Post-cache legacy",
}

COMPARISON_SPECS = {
    "bypass_vs_shadow": ("bypass", "shadow", "Causal compute"),
    "bypass_vs_native": ("bypass", "native", "Net method"),
    "attention_bypass_vs_recent_bypass": (
        "bypass", "recent_bypass", "Attention guidance",
    ),
    "recent_bypass_vs_recent_native": (
        "recent_bypass", "recent_native", "Safe bypass",
    ),
    "post_cache_vs_bypass": ("post_cache", "bypass", "Legacy vs bypass"),
}

SCALAR_METRICS = {
    "avg_accept_length": {"label": "Acceptance length", "better": "higher"},
    "mean_root_target_draft_kl": {"label": "Root KL", "better": "lower"},
    "round_count": {"label": "Verification rounds", "better": "lower"},
    "wall_tokens_per_sec": {"label": "End-to-end tokens/s", "better": "higher"},
    "draft_prefill_time_ms": {"label": "Draft prefill latency (ms)", "better": "lower"},
    "bridge_prepare_time_ms": {
        "label": "Full inline Bridge preparation latency (ms)", "better": "lower",
    },
    "bridge_map_time_ms": {"label": "Bridge map latency (ms)", "better": "lower"},
    "draft_prefill_v_projected_tokens": {
        "label": "Prefill V projected token-rows", "better": "lower",
    },
    "draft_prefill_v_skipped_tokens": {
        "label": "Prefill V skipped token-rows", "better": "higher",
    },
    "draft_prefill_k_projected_tokens": {
        "label": "Prefill K projected token-rows", "better": "neutral",
    },
    "draft_prefill_v_projection_macs": {
        "label": "Projected Draft-V MACs", "better": "lower",
    },
    "draft_prefill_v_projection_saved_macs": {
        "label": "Skipped Draft-V MACs", "better": "higher",
    },
    "bridge_prefill_mapper_macs": {
        "label": "Bridge mapper MACs", "better": "lower",
    },
    "draft_prefill_net_projection_macs_saved": {
        "label": "Net projection MACs saved", "better": "higher",
    },
}

COUNT_FIELDS = (
    "draft_k_projected_tokens",
    "draft_v_projected_tokens",
    "draft_v_skipped_tokens",
    "draft_v_shadow_tokens",
    "draft_prefill_k_projected_tokens",
    "draft_prefill_v_projected_tokens",
    "draft_prefill_v_skipped_tokens",
    "draft_prefill_v_shadow_tokens",
)

MAC_FIELDS = (
    "draft_k_projection_macs",
    "draft_v_projection_macs",
    "draft_v_projection_saved_macs",
    "draft_v_shadow_macs",
    "draft_prefill_k_projection_macs",
    "draft_prefill_v_projection_macs",
    "draft_prefill_v_projection_saved_macs",
    "draft_prefill_v_shadow_macs",
    "bridge_prefill_mapper_macs",
    "draft_prefill_net_projection_macs_saved",
)

LAYER_COUNT_FIELDS = (
    "k_projected_tokens",
    "v_projected_tokens",
    "v_skipped_tokens",
    "v_shadow_tokens",
    "prefill_k_projected_tokens",
    "prefill_v_projected_tokens",
    "prefill_v_skipped_tokens",
    "prefill_v_shadow_tokens",
)

FIXED_CONFIG_FIELDS = (
    "base_model_path",
    "draft_model_path",
    "checkpoint",
    "source_file",
    "max_gen_len",
    "tree_nodes",
    "chunk_size",
    "cache_top_k",
    "retrieve_every",
    "recent_native",
    "seed",
    "timing_repeats",
    "warmup_gen_len",
    "bridge_diagnostics",
    "geometry_diagnostics",
    "exact_target_attention",
)


class ValidationError(ValueError):
    """Input failed a prerequisite for causal or statistical interpretation."""


@dataclass(frozen=True)
class Bundle:
    path: Path
    config: dict[str, Any]
    policy_specs: dict[str, Any]
    provenance: dict[str, Any] | None
    metric_definitions: dict[str, Any]
    environment: dict[str, Any]
    rows: list[dict[str, Any]]
    sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict paired analysis for BridgeKV prefill-V compute bypass. "
            "Contrasts are paired by technical repeat before document aggregation."
        )
    )
    parser.add_argument(
        "--input", action="append", nargs="+", required=True,
        help="One or more formal evaluation JSON bundles; may be repeated.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--target-reference", action="append", nargs="+", default=[],
        help=(
            "Optional canonical Target-AR JSON bundle(s), normally docs 12-19 at 4K/8K. "
            "Used only for a separate output-exactness diagnostic."
        ),
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--expected-docs", default="4-19")
    parser.add_argument("--expected-lengths", nargs="+", type=int, default=[4096, 8192])
    parser.add_argument("--native-policy", default=DEFAULT_POLICIES["native"])
    parser.add_argument(
        "--recent-native-policy", default=DEFAULT_POLICIES["recent_native"],
    )
    parser.add_argument("--shadow-policy", default=DEFAULT_POLICIES["shadow"])
    parser.add_argument("--bypass-policy", default=DEFAULT_POLICIES["bypass"])
    parser.add_argument(
        "--recent-bypass-policy", default=DEFAULT_POLICIES["recent_bypass"],
    )
    parser.add_argument("--post-cache-policy", default=DEFAULT_POLICIES["post_cache"])
    parser.add_argument(
        "--shadow-control", choices=("full-shape", "legacy-two-gemm"),
        default="full-shape",
        help=(
            "Expected shadow implementation. full-shape is the fair timing control; "
            "legacy-two-gemm is an explicitly labelled semantic-exact exploratory control."
        ),
    )
    parser.add_argument(
        "--allow-non-4090", action="store_true",
        help="Allow analysis of a non-RTX-4090 bundle (the report records this).",
    )
    return parser.parse_args()


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def as_int(value: Any, name: str) -> int:
    number = finite(value)
    if number is None or int(number) != number:
        raise ValidationError(f"{name} must be a finite integer, got {value!r}")
    return int(number)


def parse_index_spec(spec: str) -> set[int]:
    indices: set[int] = set()
    for raw in spec.split(","):
        part = raw.strip()
        if not part:
            continue
        if "-" in part[1:]:
            start_text, stop_text = part.split("-", 1)
            start, stop = int(start_text), int(stop_text)
            if stop < start:
                raise ValidationError(f"Descending document range: {part}")
            indices.update(range(start, stop + 1))
        elif ":" in part:
            start_text, stop_text = part.split(":", 1)
            indices.update(range(int(start_text), int(stop_text)))
        else:
            indices.add(int(part))
    if not indices:
        raise ValidationError("--expected-docs resolved to an empty set")
    return indices


def stable_seed(seed: int, *parts: Any) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return (seed + int.from_bytes(digest[:4], "little")) % (2**32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_inputs(groups: Sequence[Sequence[str]]) -> list[Path]:
    paths = [Path(item).expanduser().resolve() for group in groups for item in group]
    if len(paths) != len(set(paths)):
        raise ValidationError("The same --input path was supplied more than once")
    return paths


def flatten_optional_inputs(groups: Sequence[Sequence[str]]) -> list[Path]:
    if not groups:
        return []
    return flatten_inputs(groups)


def load_bundle(path: Path) -> Bundle:
    if not path.is_file():
        raise ValidationError(f"Input does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValidationError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValidationError(f"Top-level JSON must be an object: {path}")
    config = payload.get("config")
    policy_specs = payload.get("policy_specs")
    provenance = payload.get("provenance")
    metric_definitions = payload.get("metric_definitions")
    environment = payload.get("environment")
    rows = payload.get("raw_runs")
    if not isinstance(config, dict):
        raise ValidationError(f"Missing object config in {path}")
    if not isinstance(policy_specs, dict):
        raise ValidationError(f"Missing object policy_specs in {path}")
    if provenance is not None and not isinstance(provenance, dict):
        raise ValidationError(f"provenance must be an object when present in {path}")
    if not isinstance(metric_definitions, dict):
        raise ValidationError(f"Missing object metric_definitions in {path}")
    if not isinstance(environment, dict):
        raise ValidationError(f"Missing object environment in {path}")
    if not isinstance(rows, list) or not rows:
        raise ValidationError(f"raw_runs must be a non-empty list in {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValidationError(f"Every raw_runs element must be an object in {path}")
    return Bundle(
        path=path,
        config=config,
        policy_specs=policy_specs,
        provenance=provenance,
        metric_definitions=metric_definitions,
        environment=environment,
        rows=[dict(row) for row in rows],
        sha256=sha256_file(path),
    )


def detect_shadow_control(bundle: Bundle) -> str:
    definition = str(bundle.metric_definitions.get("draft_prefill_v_shadow_tokens", "")).lower()
    if "single full-shape" in definition and "subset" in definition:
        return "full-shape"
    if "additionally submitted" in definition or "discarded v_proj" in definition:
        return "legacy-two-gemm"
    raise ValidationError(
        f"{bundle.path}: cannot identify shadow implementation from the metric definition; "
        "refusing to mix or guess provenance"
    )


def selected_policies(
    args: argparse.Namespace, bundles: Sequence[Bundle],
) -> dict[str, str]:
    policies = {
        "native": args.native_policy,
        "recent_native": args.recent_native_policy,
        "shadow": args.shadow_policy,
        "bypass": args.bypass_policy,
        "recent_bypass": args.recent_bypass_policy,
    }
    availability = [
        args.post_cache_policy in set(bundle.config.get("policies", []))
        for bundle in bundles
    ]
    if any(availability) and not all(availability):
        raise ValidationError(
            "Post-cache policy is present in only some input bundles; refusing a mixed factorial"
        )
    if all(availability):
        policies["post_cache"] = args.post_cache_policy
    if len(set(policies.values())) != len(policies):
        raise ValidationError("Selected policy names must be distinct across roles")
    return policies


def active_comparison_specs(policies: dict[str, str]) -> dict[str, tuple[str, str, str]]:
    return {
        name: spec
        for name, spec in COMPARISON_SPECS.items()
        if spec[0] in policies and spec[1] in policies
    }


def policy_core(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        key: spec.get(key)
        for key in (
            "retrieval_policy", "alpha", "alpha_key", "alpha_value",
            "layers", "source_mode", "components", "compute_mode",
        )
        if key in spec
    }


def validate_policy_specs(
    specs: dict[str, Any], policies: dict[str, str], recent_native: int,
) -> None:
    missing = [name for name in policies.values() if name not in specs]
    if missing:
        raise ValidationError(f"policy_specs is missing selected policies: {missing}")
    selected = {role: specs[name] for role, name in policies.items()}
    if not all(isinstance(spec, dict) for spec in selected.values()):
        raise ValidationError("Every selected policy spec must be an object")

    native = selected["native"]
    if native.get("retrieval_policy") != "attention" or finite(native.get("alpha")) != 0.0:
        raise ValidationError("Native control must be attention retrieval with alpha=0")
    recent_native_control = selected["recent_native"]
    if (
        recent_native_control.get("retrieval_policy") != "recent"
        or finite(recent_native_control.get("alpha")) != 0.0
    ):
        raise ValidationError("Recent-native control must be recent retrieval with alpha=0")

    for role in ("shadow", "bypass", "recent_bypass", "post_cache"):
        if role not in selected:
            continue
        spec = selected[role]
        if str(spec.get("components")) != "value":
            raise ValidationError(f"{role} must use components='value'")
        if finite(spec.get("alpha")) != 1.0:
            raise ValidationError(f"{role} must use hard alpha=1 V replacement")
        if str(spec.get("layers")) not in {"last", "all"}:
            raise ValidationError(f"{role} layers must be 'last' or 'all'")
        if str(spec.get("source_mode")) != "aligned":
            raise ValidationError(f"{role} must use aligned Target-V sources")

    shadow, bypass = selected["shadow"], selected["bypass"]
    ignored = {"compute_mode", "retrieval_policy"}
    shadow_common = {k: v for k, v in policy_core(shadow).items() if k not in ignored}
    bypass_common = {k: v for k, v in policy_core(bypass).items() if k not in ignored}
    if shadow_common != bypass_common:
        raise ValidationError(
            f"Shadow and bypass differ beyond compute_mode: {shadow_common} != {bypass_common}"
        )
    if shadow.get("retrieval_policy") != bypass.get("retrieval_policy"):
        raise ValidationError("Shadow and bypass must use the same retrieval policy")
    if shadow.get("compute_mode") != "prefill_value_shadow":
        raise ValidationError("Shadow policy must use prefill_value_shadow")
    if bypass.get("compute_mode") != "prefill_value_bypass":
        raise ValidationError("Bypass policy must use prefill_value_bypass")

    recent = selected["recent_bypass"]
    if recent.get("retrieval_policy") != "recent":
        raise ValidationError("Recent bypass control must use recent retrieval")
    if recent.get("compute_mode") != "prefill_value_bypass":
        raise ValidationError("Recent bypass control must use prefill_value_bypass")
    for key in ("alpha", "layers", "source_mode", "components"):
        if recent.get(key) != bypass.get(key):
            raise ValidationError(f"Attention/recent bypass mismatch in {key}")

    if "post_cache" in selected:
        post_cache = selected["post_cache"]
        if post_cache.get("retrieval_policy") != "attention":
            raise ValidationError("Post-cache control must use attention retrieval")
        if post_cache.get("compute_mode", "cache_overwrite") != "cache_overwrite":
            raise ValidationError("Post-cache control must use cache_overwrite")
        for key in ("alpha", "layers", "source_mode", "components"):
            if post_cache.get(key) != bypass.get(key):
                raise ValidationError(f"Post-cache/bypass mismatch in {key}")

    if recent_native < 0:
        raise ValidationError("recent_native must be non-negative")


def canonicalize_row(
    raw: dict[str, Any], bundle: Bundle, policies: dict[str, str], row_number: int,
) -> dict[str, Any]:
    row = dict(raw)
    where = f"{bundle.path}:raw_runs[{row_number}]"
    for field in (
        "length_bin_tokens", "input_tokens", "sample_index", "timing_repeat",
        "total_generated",
    ):
        row[field] = as_int(row.get(field), f"{where}.{field}")
    if row["input_tokens"] <= 0:
        raise ValidationError(f"{where}.input_tokens must be positive")
    if row["total_generated"] <= 0:
        raise ValidationError(f"{where}.total_generated must be positive")
    policy = row.get("policy")
    if policy not in policies.values():
        raise ValidationError(f"{where}.policy={policy!r} is not a selected policy")
    row["policy"] = str(policy)
    row["policy_role"] = next(
        role for role, policy_name in policies.items() if policy_name == row["policy"]
    )
    row["_source"] = str(bundle.path)
    row_policy_config = row.get("policy_config")
    if not isinstance(row_policy_config, dict):
        raise ValidationError(f"{where}.policy_config must be an object")
    expected_policy_core = policy_core(bundle.policy_specs[row["policy"]])
    observed_policy_core = policy_core(row_policy_config)
    if observed_policy_core != expected_policy_core:
        raise ValidationError(
            f"{where}.policy_config core {observed_policy_core} != bundle policy spec "
            f"{expected_policy_core}"
        )
    expected_recent = as_int(bundle.config["recent_native"], "config.recent_native")
    if as_int(row_policy_config.get("recent_native"), f"{where}.policy_config.recent_native") != expected_recent:
        raise ValidationError(f"{where}.policy_config.recent_native mismatches config")

    for field in (
        "avg_accept_length", "mean_root_target_draft_kl",
        "wall_tokens_per_sec", "wall_time", "draft_prefill_time_ms",
        "bridge_prepare_time_ms", "bridge_map_time_ms",
    ):
        value = finite(row.get(field))
        if (
            value is None
            and field == "bridge_prepare_time_ms"
            and detect_shadow_control(bundle) == "legacy-two-gemm"
        ):
            # This timing did not exist in the legacy schema.  Keep a real NA;
            # bridge_map_time_ms is a narrower dense-kernel measurement and is
            # not a valid substitute for gather+mapper+materialization.
            row[field] = None
            continue
        if value is None:
            if field == "mean_root_target_draft_kl" and isinstance(row.get("rounds"), list):
                values = [
                    finite(item.get("root_target_draft_kl"))
                    for item in row["rounds"] if isinstance(item, dict)
                ]
                values = [value for value in values if value is not None]
                value = float(np.mean(values)) if values else None
            if field == "avg_accept_length" and isinstance(row.get("accept_length_list"), list):
                values = [finite(value) for value in row["accept_length_list"]]
                if values and all(value is not None for value in values):
                    value = float(np.mean(values))
        if value is None:
            raise ValidationError(f"{where}.{field} must be finite")
        if field in {"wall_tokens_per_sec", "wall_time"} and value <= 0:
            raise ValidationError(f"{where}.{field} must be positive")
        if field.endswith("_ms") and value < 0:
            raise ValidationError(f"{where}.{field} must be non-negative")
        row[field] = float(value)

    output_ids = row.get("output_token_ids")
    accepts = row.get("accept_length_list")
    rounds = row.get("rounds")
    if not isinstance(output_ids, list) or not all(isinstance(x, int) for x in output_ids):
        raise ValidationError(f"{where}.output_token_ids must be a list of integers")
    if not isinstance(accepts, list) or not accepts or not all(isinstance(x, int) for x in accepts):
        raise ValidationError(f"{where}.accept_length_list must be a non-empty integer list")
    if not isinstance(rounds, list):
        raise ValidationError(f"{where}.rounds must be a list")
    if len(output_ids) != row["total_generated"]:
        raise ValidationError(
            f"{where}: len(output_token_ids)={len(output_ids)} != "
            f"total_generated={row['total_generated']}"
        )
    row["round_count"] = len(rounds)
    if len(rounds) != len(accepts):
        raise ValidationError(
            f"{where}: len(rounds)={len(rounds)} != len(accept_length_list)={len(accepts)}"
        )

    # Do not trust duplicated producer summaries blindly.  Acceptance is
    # intentionally stored rounded to three decimals, while KL and TPS are
    # stored at full precision.  Validate each summary against its primitive
    # fields, then use the primitive-derived value for inference.
    exact_acceptance = float(np.mean(accepts))
    if abs(row["avg_accept_length"] - exact_acceptance) > 5.0e-4 + 1.0e-12:
        raise ValidationError(
            f"{where}.avg_accept_length={row['avg_accept_length']} is inconsistent "
            f"with accept_length_list mean={exact_acceptance}"
        )
    row["avg_accept_length"] = exact_acceptance

    round_kls = []
    for round_number, round_payload in enumerate(rounds):
        if not isinstance(round_payload, dict):
            raise ValidationError(f"{where}.rounds[{round_number}] must be an object")
        value = finite(round_payload.get("root_target_draft_kl"))
        if value is None:
            raise ValidationError(
                f"{where}.rounds[{round_number}].root_target_draft_kl must be finite"
            )
        round_kls.append(value)
    exact_root_kl = float(np.mean(round_kls))
    if not math.isclose(
        row["mean_root_target_draft_kl"], exact_root_kl,
        rel_tol=1.0e-12, abs_tol=1.0e-12,
    ):
        raise ValidationError(
            f"{where}.mean_root_target_draft_kl is inconsistent with rounds"
        )
    row["mean_root_target_draft_kl"] = exact_root_kl

    exact_tps = row["total_generated"] / row["wall_time"]
    if not math.isclose(
        row["wall_tokens_per_sec"], exact_tps,
        rel_tol=1.0e-12, abs_tol=1.0e-12,
    ):
        raise ValidationError(
            f"{where}.wall_tokens_per_sec={row['wall_tokens_per_sec']} is inconsistent "
            f"with total_generated/wall_time={exact_tps}"
        )
    row["wall_tokens_per_sec"] = exact_tps

    for field in COUNT_FIELDS + MAC_FIELDS:
        row[field] = as_int(row.get(field), f"{where}.{field}")
        if row[field] < 0 and field != "draft_prefill_net_projection_macs_saved":
            raise ValidationError(f"{where}.{field} must be non-negative")

    expected_net = (
        row["draft_prefill_v_projection_saved_macs"]
        - row["bridge_prefill_mapper_macs"]
    )
    if row["draft_prefill_net_projection_macs_saved"] != expected_net:
        raise ValidationError(
            f"{where}: net MACs {row['draft_prefill_net_projection_macs_saved']} "
            f"!= skipped {row['draft_prefill_v_projection_saved_macs']} "
            f"- mapper {row['bridge_prefill_mapper_macs']}"
        )

    by_layer = row.get("draft_projection_by_layer")
    if not isinstance(by_layer, list) or not by_layer:
        raise ValidationError(f"{where}.draft_projection_by_layer must be non-empty")
    canonical_layers: list[dict[str, int]] = []
    seen_layers: set[int] = set()
    for item_number, item in enumerate(by_layer):
        if not isinstance(item, dict):
            raise ValidationError(f"{where}.draft_projection_by_layer[{item_number}] must be an object")
        layer = as_int(item.get("layer"), f"{where}.layer[{item_number}]")
        if layer in seen_layers:
            raise ValidationError(f"{where}: duplicate projection layer {layer}")
        seen_layers.add(layer)
        parsed = {"layer": layer}
        for field in LAYER_COUNT_FIELDS:
            parsed[field] = as_int(item.get(field), f"{where}.layer[{layer}].{field}")
            if parsed[field] < 0:
                raise ValidationError(f"{where}.layer[{layer}].{field} must be non-negative")
        canonical_layers.append(parsed)
    canonical_layers.sort(key=lambda item: item["layer"])
    if [item["layer"] for item in canonical_layers] != list(range(len(canonical_layers))):
        raise ValidationError(f"{where}: Draft layers must be consecutively indexed from zero")
    row["draft_projection_by_layer"] = canonical_layers

    aggregate_map = {
        "k_projected_tokens": "draft_k_projected_tokens",
        "v_projected_tokens": "draft_v_projected_tokens",
        "v_skipped_tokens": "draft_v_skipped_tokens",
        "v_shadow_tokens": "draft_v_shadow_tokens",
        "prefill_k_projected_tokens": "draft_prefill_k_projected_tokens",
        "prefill_v_projected_tokens": "draft_prefill_v_projected_tokens",
        "prefill_v_skipped_tokens": "draft_prefill_v_skipped_tokens",
        "prefill_v_shadow_tokens": "draft_prefill_v_shadow_tokens",
    }
    for layer_field, aggregate_field in aggregate_map.items():
        layer_sum = sum(item[layer_field] for item in canonical_layers)
        if layer_sum != row[aggregate_field]:
            raise ValidationError(
                f"{where}: sum(layer.{layer_field})={layer_sum} != "
                f"{aggregate_field}={row[aggregate_field]}"
            )

    events = row.get("draft_value_replacement_events")
    if not isinstance(events, list):
        raise ValidationError(f"{where}.draft_value_replacement_events must be a list")
    parsed_events = []
    for event_number, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValidationError(f"{where}.replacement[{event_number}] must be an object")
        parsed = {
            "layer": as_int(event.get("layer"), f"{where}.replacement.layer"),
            "start": as_int(event.get("start"), f"{where}.replacement.start"),
            "stop": as_int(event.get("stop"), f"{where}.replacement.stop"),
            "tokens": as_int(event.get("tokens"), f"{where}.replacement.tokens"),
            "shadow_compute": bool(event.get("shadow_compute", False)),
        }
        if not (0 <= parsed["start"] < parsed["stop"]):
            raise ValidationError(f"{where}: invalid replacement interval {parsed}")
        if parsed["tokens"] != parsed["stop"] - parsed["start"]:
            raise ValidationError(f"{where}: replacement token count does not match interval")
        parsed_events.append(parsed)
    row["draft_value_replacement_events"] = sorted(
        parsed_events, key=lambda item: (item["layer"], item["start"], item["stop"]),
    )

    diagnostic = row.get("wall_timing_is_diagnostic", False)
    if isinstance(diagnostic, str):
        diagnostic = diagnostic.strip().lower() in {"1", "true", "yes"}
    if bool(diagnostic):
        raise ValidationError(f"{where}: diagnostic wall timing cannot support throughput claims")
    validate_embedded_compute_replacement(
        row, bundle.policy_specs[row["policy"]], expected_recent, where,
        detect_shadow_control(bundle),
    )
    return row


def validate_embedded_compute_replacement(
    row: dict[str, Any], spec: dict[str, Any], recent_native: int, where: str,
    shadow_control: str,
) -> None:
    """Cross-check the producer-side invariant certificate against raw fields."""
    mode = str(spec.get("compute_mode", "cache_overwrite"))
    certificate = row.get("compute_replacement_validation")
    inline = mode in {"prefill_value_shadow", "prefill_value_bypass"}
    if not inline:
        if certificate is not None:
            raise ValidationError(f"{where}: non-inline policy has a replacement certificate")
        return
    if certificate is None and shadow_control == "legacy-two-gemm":
        # Producer-side certificates were added after the legacy bundle.  The
        # analyzer still independently checks intervals and layer/aggregate
        # projection accounting below.
        return
    if not isinstance(certificate, dict) or certificate.get("status") != "passed":
        raise ValidationError(f"{where}: inline policy lacks a passed replacement certificate")
    query_tokens = row["input_tokens"] + 1
    mapped_tokens = max(0, row["input_tokens"] - recent_native - 1)
    layer_count = len(row["draft_projection_by_layer"])
    active_count = 1 if spec.get("layers", "all") == "last" else layer_count
    for field, expected in (
        ("query_tokens", query_tokens),
        ("mapped_tokens", mapped_tokens),
        ("active_layers", active_count),
    ):
        if as_int(certificate.get(field), f"{where}.compute_replacement_validation.{field}") != expected:
            raise ValidationError(f"{where}: replacement certificate {field} mismatch")
    v_macs_per_row = as_int(
        certificate.get("v_macs_per_row"),
        f"{where}.compute_replacement_validation.v_macs_per_row",
    )
    if v_macs_per_row <= 0:
        raise ValidationError(f"{where}: replacement certificate has invalid V MACs/row")

    expected_payload = certificate.get("expected")
    if not isinstance(expected_payload, dict):
        raise ValidationError(f"{where}: replacement certificate expected map is missing")
    certified_fields = (
        "bridge_map_events",
        "bridge_prepare_events",
        "draft_prefill_events",
        "bridge_mapped_tokens",
        "bridge_value_bypass_events",
        "bridge_value_shadow_events",
        "draft_prefill_k_projected_tokens",
        "draft_prefill_v_projected_tokens",
        "draft_prefill_v_skipped_tokens",
        "draft_prefill_v_shadow_tokens",
        "draft_prefill_v_projection_macs",
        "draft_prefill_v_projection_saved_macs",
        "draft_prefill_v_shadow_macs",
        "bridge_prefill_mapper_macs",
        "draft_prefill_net_projection_macs_saved",
    )
    for field in certified_fields:
        expected = as_int(expected_payload.get(field), f"{where}.certificate.expected.{field}")
        observed = as_int(row.get(field), f"{where}.{field}")
        if observed != expected:
            raise ValidationError(
                f"{where}: row {field}={observed} != certified expected {expected}"
            )

    mapped_rows = active_count * mapped_tokens
    total_rows = layer_count * query_tokens
    is_bypass = mode == "prefill_value_bypass"
    independent = {
        "bridge_map_events": 1 if mapped_tokens else 0,
        "bridge_prepare_events": 1 if mapped_tokens else 0,
        "draft_prefill_events": 1,
        "bridge_mapped_tokens": mapped_tokens,
        "bridge_value_bypass_events": 1 if is_bypass and mapped_tokens else 0,
        "bridge_value_shadow_events": 1 if (not is_bypass) and mapped_tokens else 0,
        "draft_prefill_k_projected_tokens": total_rows,
        "draft_prefill_v_projected_tokens": total_rows - mapped_rows if is_bypass else total_rows,
        "draft_prefill_v_skipped_tokens": mapped_rows if is_bypass else 0,
        "draft_prefill_v_shadow_tokens": 0 if is_bypass else mapped_rows,
        "draft_prefill_v_projection_macs": (
            total_rows - mapped_rows if is_bypass else total_rows
        ) * v_macs_per_row,
        "draft_prefill_v_projection_saved_macs": (
            mapped_rows * v_macs_per_row if is_bypass else 0
        ),
        "draft_prefill_v_shadow_macs": 0 if is_bypass else mapped_rows * v_macs_per_row,
    }
    for field, expected in independent.items():
        if as_int(row.get(field), f"{where}.{field}") != expected:
            raise ValidationError(f"{where}: independently derived {field} mismatch")


def validate_configs(
    bundles: Sequence[Bundle], policies: dict[str, str], expected_lengths: set[int],
    allow_non_4090: bool, expected_shadow_control: str,
) -> dict[str, Any]:
    baseline = bundles[0]
    observed_shadow_controls = {detect_shadow_control(bundle) for bundle in bundles}
    if observed_shadow_controls != {expected_shadow_control}:
        raise ValidationError(
            f"Shadow provenance mismatch: inputs contain {sorted(observed_shadow_controls)}, "
            f"but --shadow-control={expected_shadow_control}. Old two-GEMM and new full-shape "
            "bundles must never be pooled."
        )
    for bundle in bundles:
        if bundle.config.get("command") != "evaluate":
            raise ValidationError(f"{bundle.path}: config.command must be 'evaluate'")
        missing_config = [field for field in FIXED_CONFIG_FIELDS if field not in bundle.config]
        if missing_config:
            raise ValidationError(f"{bundle.path}: config missing fields {missing_config}")
        if bool(bundle.config.get("geometry_diagnostics")):
            raise ValidationError(f"{bundle.path}: geometry_diagnostics invalidates fair timing")
        if bool(bundle.config.get("bridge_diagnostics")):
            raise ValidationError(f"{bundle.path}: bridge_diagnostics invalidates fair timing")
        repeats = as_int(bundle.config.get("timing_repeats"), "timing_repeats")
        if repeats < 2 and expected_shadow_control == "full-shape":
            raise ValidationError("At least two technical timing repeats are required")
        configured_policies = set(bundle.config.get("policies", []))
        missing = set(policies.values()) - configured_policies
        if missing:
            raise ValidationError(f"{bundle.path}: config.policies missing {sorted(missing)}")
        validate_policy_specs(
            bundle.policy_specs, policies, as_int(bundle.config["recent_native"], "recent_native"),
        )
        lengths = {as_int(value, "config.lengths") for value in bundle.config.get("lengths", [])}
        if not lengths or not lengths <= expected_lengths:
            raise ValidationError(
                f"{bundle.path}: configured lengths {sorted(lengths)} are not a non-empty "
                f"subset of expected {sorted(expected_lengths)}"
            )
        gpu = str(bundle.environment.get("gpu", ""))
        if not gpu:
            raise ValidationError(f"{bundle.path}: environment.gpu is missing")
        if "4090" not in gpu and not allow_non_4090:
            raise ValidationError(
                f"{bundle.path}: expected RTX 4090, observed {gpu!r}; use --allow-non-4090 to override"
            )
        if expected_shadow_control == "full-shape":
            provenance = bundle.provenance
            if not isinstance(provenance, dict):
                raise ValidationError(
                    f"{bundle.path}: full-shape shadow bundle must contain code/data provenance"
                )
            implementation = provenance.get("implementation_sha256")
            if not isinstance(implementation, dict) or not implementation:
                raise ValidationError(
                    f"{bundle.path}: provenance.implementation_sha256 is missing"
                )
            for field in ("checkpoint_sha256", "source_file_sha256"):
                value = provenance.get(field)
                if not isinstance(value, str) or len(value) != 64:
                    raise ValidationError(f"{bundle.path}: invalid provenance.{field}")
            prepare_definition = str(
                bundle.metric_definitions.get("bridge_prepare_time_ms", "")
            ).lower()
            map_definition = str(
                bundle.metric_definitions.get("bridge_map_time_ms", "")
            ).lower()
            if not all(
                token in prepare_definition
                for token in ("gather", "mapper", "materialization")
            ):
                raise ValidationError(
                    f"{bundle.path}: bridge_prepare_time_ms lacks the full inline timing definition"
                )
            if "dense" not in map_definition or "only" not in map_definition:
                raise ValidationError(
                    f"{bundle.path}: bridge_map_time_ms must be labelled dense-mapper-only"
                )

    for bundle in bundles[1:]:
        for field in FIXED_CONFIG_FIELDS:
            if bundle.config.get(field) != baseline.config.get(field):
                raise ValidationError(
                    f"Fixed config mismatch for {field}: {baseline.path} has "
                    f"{baseline.config.get(field)!r}, {bundle.path} has {bundle.config.get(field)!r}"
                )
        for role, policy in policies.items():
            if policy_core(bundle.policy_specs[policy]) != policy_core(baseline.policy_specs[policy]):
                raise ValidationError(f"Policy spec mismatch for {role}/{policy} across bundles")
        if bundle.environment.get("gpu") != baseline.environment.get("gpu"):
            raise ValidationError("GPU model differs across bundles")
        if bundle.provenance != baseline.provenance:
            raise ValidationError(
                "Provenance differs across input bundles; refusing to pool code, checkpoint, "
                "or source-data variants"
            )
    recent_native = as_int(baseline.config["recent_native"], "recent_native")
    return {
        "fixed_config": {field: baseline.config.get(field) for field in FIXED_CONFIG_FIELDS},
        "environment": dict(baseline.environment),
        "shadow_control": expected_shadow_control,
        "shadow_control_interpretation": (
            "single full-shape native v_proj then overwrite; fair timing control with possible "
            "FP16 GEMM-shape trajectory differences"
            if expected_shadow_control == "full-shape"
            else "legacy two-GEMM discarded projection; semantic-exact control, not fair timing"
        ),
        "recent_native_setting": recent_native,
        "recent_native_label": (
            "main conservative setting"
            if recent_native == 1024
            else (
                "aggressive exploratory setting"
                if recent_native == 128
                else "nonstandard exploratory setting"
            )
        ),
        "provenance": baseline.provenance,
    }


def row_key(row: dict[str, Any]) -> tuple[int, int, str, int]:
    return (
        row["length_bin_tokens"], row["sample_index"], row["policy"],
        row["timing_repeat"],
    )


def paired_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return row["length_bin_tokens"], row["sample_index"], row["timing_repeat"]


def normalize_intervals(row: dict[str, Any]) -> list[tuple[int, int, int, int]]:
    return [
        (event["layer"], event["start"], event["stop"], event["tokens"])
        for event in row["draft_value_replacement_events"]
    ]


def layers_by_index(row: dict[str, Any]) -> dict[int, dict[str, int]]:
    return {item["layer"]: item for item in row["draft_projection_by_layer"]}


def validate_replacement_shape(
    row: dict[str, Any], role: str, policy_spec: dict[str, Any], recent_native: int,
) -> None:
    events = row["draft_value_replacement_events"]
    layer_count = len(row["draft_projection_by_layer"])
    if role in {"shadow", "bypass", "recent_bypass"}:
        expected_layers = (
            {layer_count - 1} if policy_spec.get("layers") == "last"
            else set(range(layer_count))
        )
        observed_layers = {event["layer"] for event in events}
        if observed_layers != expected_layers or len(events) != len(expected_layers):
            raise ValidationError(
                f"{role} replacement layers {sorted(observed_layers)} != expected "
                f"{sorted(expected_layers)} at key {row_key(row)}"
            )
        expected_stop = row["input_tokens"] - recent_native
        for event in events:
            if event["start"] != 1 or event["stop"] != expected_stop:
                raise ValidationError(
                    f"{role} replacement interval [{event['start']},{event['stop']}) "
                    f"!= expected [1,{expected_stop}) at key {row_key(row)}"
                )
            expected_shadow = role == "shadow"
            if event["shadow_compute"] != expected_shadow:
                raise ValidationError(
                    f"{role} shadow flag {event['shadow_compute']} != {expected_shadow}"
                )
    elif events:
        raise ValidationError(
            f"{role} must not perform inline prefill replacement, got {events} at {row_key(row)}"
        )


def validate_shadow_bypass_pair(
    shadow: dict[str, Any], bypass: dict[str, Any], counters: dict[str, Any],
) -> None:
    key = paired_key(shadow)
    counters["output_id_pairs_checked"] += 1
    if shadow["output_token_ids"] == bypass["output_token_ids"]:
        counters["output_id_pairs_exact"] += 1
    else:
        counters["output_id_mismatch_keys"].append(list(key))
    counters["acceptance_list_pairs_checked"] += 1
    if shadow["accept_length_list"] == bypass["accept_length_list"]:
        counters["acceptance_list_pairs_exact"] += 1
    else:
        counters["acceptance_list_mismatch_keys"].append(list(key))
    if normalize_intervals(shadow) != normalize_intervals(bypass):
        raise ValidationError(f"Shadow/bypass replacement intervals differ at {key}")
    counters["replacement_interval_pairs"] += 1

    for field in (
        "draft_k_projected_tokens", "draft_k_projection_macs",
        "draft_prefill_k_projected_tokens", "draft_prefill_k_projection_macs",
        "bridge_prefill_mapper_macs",
    ):
        if shadow[field] != bypass[field]:
            raise ValidationError(f"Shadow/bypass {field} differs at {key}")
    counters["aggregate_k_pairs"] += 1

    aggregate_equalities = (
        (
            shadow["draft_v_projected_tokens"],
            bypass["draft_v_projected_tokens"] + bypass["draft_v_skipped_tokens"],
            "total V projected = bypass projected + skipped",
        ),
        (shadow["draft_v_shadow_tokens"], bypass["draft_v_skipped_tokens"], "total shadow = skipped"),
        (
            shadow["draft_prefill_v_projected_tokens"],
            bypass["draft_prefill_v_projected_tokens"] + bypass["draft_prefill_v_skipped_tokens"],
            "prefill V projected = bypass projected + skipped",
        ),
        (
            shadow["draft_prefill_v_shadow_tokens"],
            bypass["draft_prefill_v_skipped_tokens"],
            "prefill shadow = skipped",
        ),
        (
            shadow["draft_v_projection_macs"],
            bypass["draft_v_projection_macs"] + bypass["draft_v_projection_saved_macs"],
            "total V MACs",
        ),
        (shadow["draft_v_shadow_macs"], bypass["draft_v_projection_saved_macs"], "total shadow MACs"),
        (
            shadow["draft_prefill_v_projection_macs"],
            bypass["draft_prefill_v_projection_macs"]
            + bypass["draft_prefill_v_projection_saved_macs"],
            "prefill V MACs",
        ),
        (
            shadow["draft_prefill_v_shadow_macs"],
            bypass["draft_prefill_v_projection_saved_macs"],
            "prefill shadow MACs",
        ),
    )
    for observed, expected, label in aggregate_equalities:
        if observed != expected:
            raise ValidationError(f"Shadow/bypass aggregate invariant '{label}' failed at {key}")
    if shadow["draft_v_skipped_tokens"] != 0 or shadow["draft_prefill_v_skipped_tokens"] != 0:
        raise ValidationError(f"Shadow unexpectedly records skipped V rows at {key}")
    if bypass["draft_v_shadow_tokens"] != 0 or bypass["draft_prefill_v_shadow_tokens"] != 0:
        raise ValidationError(f"Bypass unexpectedly records shadow V rows at {key}")

    shadow_layers, bypass_layers = layers_by_index(shadow), layers_by_index(bypass)
    if shadow_layers.keys() != bypass_layers.keys():
        raise ValidationError(f"Shadow/bypass layer sets differ at {key}")
    for layer in shadow_layers:
        left, right = shadow_layers[layer], bypass_layers[layer]
        for field in ("k_projected_tokens", "prefill_k_projected_tokens"):
            if left[field] != right[field]:
                raise ValidationError(f"Layer {layer} {field} differs at {key}")
        checks = (
            (left["v_projected_tokens"], right["v_projected_tokens"] + right["v_skipped_tokens"]),
            (left["v_shadow_tokens"], right["v_skipped_tokens"]),
            (
                left["prefill_v_projected_tokens"],
                right["prefill_v_projected_tokens"] + right["prefill_v_skipped_tokens"],
            ),
            (left["prefill_v_shadow_tokens"], right["prefill_v_skipped_tokens"]),
        )
        if not all(observed == expected for observed, expected in checks):
            raise ValidationError(f"Layerwise V accounting invariant failed at {key}, layer {layer}")
        if left["v_skipped_tokens"] or left["prefill_v_skipped_tokens"]:
            raise ValidationError(f"Shadow layer {layer} records skipped rows at {key}")
        if right["v_shadow_tokens"] or right["prefill_v_shadow_tokens"]:
            raise ValidationError(f"Bypass layer {layer} records shadow rows at {key}")
        counters["layerwise_v_checks"] += 1


def validate_rows(
    rows: list[dict[str, Any]], bundles: Sequence[Bundle], policies: dict[str, str],
    expected_docs: set[int], expected_lengths: set[int], shadow_control: str,
) -> dict[str, Any]:
    by_key: dict[tuple[int, int, str, int], dict[str, Any]] = {}
    for row in rows:
        key = row_key(row)
        if key in by_key:
            raise ValidationError(f"Duplicate row key across inputs: {key}")
        by_key[key] = row

    observed_docs = {row["sample_index"] for row in rows}
    observed_lengths = {row["length_bin_tokens"] for row in rows}
    if observed_docs != expected_docs:
        raise ValidationError(
            f"Observed documents {sorted(observed_docs)} != expected {sorted(expected_docs)}"
        )
    if observed_lengths != expected_lengths:
        raise ValidationError(
            f"Observed lengths {sorted(observed_lengths)} != expected {sorted(expected_lengths)}"
        )

    repeats = as_int(bundles[0].config["timing_repeats"], "timing_repeats")
    expected_repeats = set(range(repeats))
    expected_keys = {
        (length, document, policy, repeat)
        for length in expected_lengths
        for document in expected_docs
        for policy in policies.values()
        for repeat in expected_repeats
    }
    observed_keys = set(by_key)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys, key=str)[:20]
        extra = sorted(observed_keys - expected_keys, key=str)[:20]
        raise ValidationError(
            f"Incomplete/extra factorial cells: missing(first20)={missing}, extra(first20)={extra}"
        )

    paired_metadata_cells = 0
    for length in sorted(expected_lengths):
        for document in sorted(expected_docs):
            cell_rows = [
                by_key[(length, document, policy, repeat)]
                for policy in policies.values()
                for repeat in sorted(expected_repeats)
            ]
            actual_inputs = {row["input_tokens"] for row in cell_rows}
            generation_budgets = {row["total_generated"] for row in cell_rows}
            if len(actual_inputs) != 1:
                raise ValidationError(
                    f"Actual input tokens differ within nominal paired cell "
                    f"{(length, document)}: {sorted(actual_inputs)}"
                )
            if len(generation_budgets) != 1:
                raise ValidationError(
                    f"Generated-token budgets differ within paired cell "
                    f"{(length, document)}: {sorted(generation_budgets)}"
                )
            paired_metadata_cells += 1

    baseline_specs = bundles[0].policy_specs
    recent_native = as_int(bundles[0].config["recent_native"], "recent_native")
    role_for_policy = {policy: role for role, policy in policies.items()}
    for row in rows:
        role = role_for_policy[row["policy"]]
        validate_replacement_shape(row, role, baseline_specs[row["policy"]], recent_native)

    counters = {
        "shadow_bypass_pairs": 0,
        "output_id_pairs_checked": 0,
        "output_id_pairs_exact": 0,
        "output_id_mismatch_keys": [],
        "acceptance_list_pairs_checked": 0,
        "acceptance_list_pairs_exact": 0,
        "acceptance_list_mismatch_keys": [],
        "replacement_interval_pairs": 0,
        "aggregate_k_pairs": 0,
        "layerwise_v_checks": 0,
        "producer_replacement_certificates_passed": sum(
            row.get("compute_replacement_validation", {}).get("status") == "passed"
            for row in rows
            if isinstance(row.get("compute_replacement_validation"), dict)
        ),
        "paired_actual_context_and_generation_cells": paired_metadata_cells,
    }
    for length in sorted(expected_lengths):
        for document in sorted(expected_docs):
            for repeat in sorted(expected_repeats):
                shadow = by_key[(length, document, policies["shadow"], repeat)]
                bypass = by_key[(length, document, policies["bypass"], repeat)]
                validate_shadow_bypass_pair(shadow, bypass, counters)
                counters["shadow_bypass_pairs"] += 1

    trajectory_by_length: dict[str, Any] = {}
    trajectory_cells_checked = 0
    trajectory_output_exact = 0
    trajectory_accept_exact = 0
    representative_repeat = min(expected_repeats)
    for length in sorted(expected_lengths):
        output_exact = 0
        accept_exact = 0
        for document in sorted(expected_docs):
            shadow = by_key[
                (length, document, policies["shadow"], representative_repeat)
            ]
            bypass = by_key[
                (length, document, policies["bypass"], representative_repeat)
            ]
            output_exact += int(
                shadow["output_token_ids"] == bypass["output_token_ids"]
            )
            accept_exact += int(
                shadow["accept_length_list"] == bypass["accept_length_list"]
            )
        checked = len(expected_docs)
        trajectory_by_length[str(length)] = {
            "checked": checked,
            "output_id_exact": output_exact,
            "acceptance_list_exact": accept_exact,
        }
        trajectory_cells_checked += checked
        trajectory_output_exact += output_exact
        trajectory_accept_exact += accept_exact
    counters.update({
        "unique_trajectory_cells_checked": trajectory_cells_checked,
        "unique_trajectory_output_id_exact": trajectory_output_exact,
        "unique_trajectory_acceptance_list_exact": trajectory_accept_exact,
        "unique_trajectory_by_length": trajectory_by_length,
        "raw_shadow_bypass_repeat_pairs": counters["shadow_bypass_pairs"],
    })

    # Technical repeats should change timings, not the decoded trajectory.
    repeat_exact_groups = 0
    for length in sorted(expected_lengths):
        for document in sorted(expected_docs):
            for policy in policies.values():
                group = [by_key[(length, document, policy, repeat)] for repeat in sorted(expected_repeats)]
                if any(row["output_token_ids"] != group[0]["output_token_ids"] for row in group[1:]):
                    raise ValidationError(
                        f"Output IDs vary across timing repeats: {(length, document, policy)}"
                    )
                if any(row["accept_length_list"] != group[0]["accept_length_list"] for row in group[1:]):
                    raise ValidationError(
                        f"Acceptance list varies across timing repeats: {(length, document, policy)}"
                    )
                repeat_exact_groups += 1
    counters["repeat_exact_groups"] = repeat_exact_groups
    counters["output_id_exact_rate"] = (
        counters["output_id_pairs_exact"] / counters["output_id_pairs_checked"]
    )
    counters["acceptance_list_exact_rate"] = (
        counters["acceptance_list_pairs_exact"]
        / counters["acceptance_list_pairs_checked"]
    )
    if shadow_control == "legacy-two-gemm":
        if counters["output_id_pairs_exact"] != counters["output_id_pairs_checked"]:
            raise ValidationError(
                "Legacy two-GEMM semantic control must have bit-exact shadow/bypass output IDs"
            )
        if (
            counters["acceptance_list_pairs_exact"]
            != counters["acceptance_list_pairs_checked"]
        ):
            raise ValidationError(
                "Legacy two-GEMM semantic control must have exact shadow/bypass acceptance lists"
            )
    counters["trajectory_equality_requirement"] = (
        "hard" if shadow_control == "legacy-two-gemm" else "diagnostic"
    )
    counters["hard_accounting_checks_passed"] = True
    return counters


def aggregate_repeats(
    rows: Sequence[dict[str, Any]], policies: dict[str, str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["length_bin_tokens"], row["sample_index"], row["policy"])].append(row)
    aggregated = []
    for (length, document, policy), group in sorted(grouped.items(), key=lambda item: str(item[0])):
        role = next(role for role, name in policies.items() if name == policy)
        output: dict[str, Any] = {
            "length_bin_tokens": length,
            "sample_index": document,
            "policy": policy,
            "policy_role": role,
            "timing_repeats": len(group),
            "input_tokens_min": min(row["input_tokens"] for row in group),
            "input_tokens_max": max(row["input_tokens"] for row in group),
        }
        for metric in SCALAR_METRICS:
            values = [row.get(metric) for row in group]
            present = [finite(value) for value in values if finite(value) is not None]
            if not present:
                output[metric] = None
            elif len(present) != len(values):
                raise ValidationError(
                    f"Metric {metric} is only partially present across technical repeats "
                    f"for {(length, document, policy)}"
                )
            else:
                output[metric] = float(np.median(present))
        aggregated.append(output)
    return aggregated


def first_sequence_mismatch(left: Sequence[int], right: Sequence[int]) -> int | None:
    for index, (left_value, right_value) in enumerate(zip(left, right)):
        if left_value != right_value:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def target_reference_diagnostics(
    paths: Sequence[Path], formal_rows: Sequence[dict[str, Any]],
    formal_bundles: Sequence[Bundle], policies: dict[str, str],
    expected_lengths: set[int],
) -> dict[str, Any]:
    """Compare selected methods with canonical Target-AR on docs 12-19.

    This is deliberately separate from shadow-vs-bypass equality.  It checks
    the realized greedy output of each speculative method against an external
    canonical Target-AR run and is not used in any quality/performance CI.
    """
    if not paths:
        return {
            "available": False,
            "reason": "no --target-reference supplied",
            "scope": "separate from shadow/bypass trajectory equality",
        }
    baseline_config = formal_bundles[0].config
    reference_map: dict[tuple[int, int], list[int]] = {}
    references = []
    for path in paths:
        if not path.is_file():
            raise ValidationError(f"Target reference does not exist: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValidationError(f"Invalid target-reference JSON {path}: {error}") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
            raise ValidationError(f"{path}: target reference requires an object config")
        config = payload["config"]
        if config.get("command") != "reference":
            raise ValidationError(f"{path}: target reference config.command must be 'reference'")
        # Greedy Target-AR output is deterministic; the evaluation seed only
        # controls formal-policy timing order and need not match the reference.
        for field in ("base_model_path", "source_file", "max_gen_len"):
            if config.get(field) != baseline_config.get(field):
                raise ValidationError(
                    f"{path}: target reference {field}={config.get(field)!r} does not match "
                    f"formal run {baseline_config.get(field)!r}"
                )
        raw_rows = payload.get("raw_runs")
        if not isinstance(raw_rows, list) or not raw_rows:
            raise ValidationError(f"{path}: target reference raw_runs must be non-empty")
        for index, row in enumerate(raw_rows):
            if not isinstance(row, dict) or row.get("policy") != "target_ar":
                raise ValidationError(f"{path}:raw_runs[{index}] is not a target_ar row")
            length = as_int(row.get("length_bin_tokens"), f"{path}:reference.length")
            document = as_int(row.get("sample_index"), f"{path}:reference.sample_index")
            if length not in expected_lengths:
                raise ValidationError(f"{path}: unexpected target-reference length {length}")
            output_ids = row.get("output_token_ids")
            if not isinstance(output_ids, list) or not all(isinstance(value, int) for value in output_ids):
                raise ValidationError(f"{path}: target output_token_ids must be integers")
            key = (length, document)
            if key in reference_map:
                raise ValidationError(f"Duplicate target-reference row {key}")
            reference_map[key] = list(output_ids)
        references.append({
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(raw_rows),
        })

    confirmatory_documents = set(range(12, 20))
    expected_keys = {
        (length, document)
        for length in expected_lengths for document in confirmatory_documents
    }
    if set(reference_map) != expected_keys:
        raise ValidationError(
            "Canonical Target-AR references must cover exactly docs 12-19 for every "
            f"expected length; missing={sorted(expected_keys - set(reference_map))}, "
            f"extra={sorted(set(reference_map) - expected_keys)}"
        )

    # Cross-repeat identity has already been hard checked per policy, so use
    # the first technical repetition for the external reference diagnostic.
    method_map: dict[tuple[int, int, str], list[int]] = {}
    for row in sorted(formal_rows, key=row_key):
        role = next(
            role for role, policy in policies.items() if policy == row["policy"]
        )
        if role not in {"bypass", "native"} or row["sample_index"] not in confirmatory_documents:
            continue
        key = (row["length_bin_tokens"], row["sample_index"], role)
        method_map.setdefault(key, list(row["output_token_ids"]))

    by_length: dict[str, Any] = {}
    for length in sorted(expected_lengths):
        by_role = {}
        for role in ("bypass", "native"):
            exact_documents = []
            mismatches = []
            for document in sorted(confirmatory_documents):
                method_key = (length, document, role)
                if method_key not in method_map:
                    raise ValidationError(f"Missing formal output for Target-AR check: {method_key}")
                observed = method_map[method_key]
                reference = reference_map[(length, document)]
                mismatch = first_sequence_mismatch(observed, reference)
                if mismatch is None:
                    exact_documents.append(document)
                else:
                    mismatches.append({
                        "document": document,
                        "first_mismatch_position_zero_based": mismatch,
                        "method_length": len(observed),
                        "reference_length": len(reference),
                    })
            total = len(confirmatory_documents)
            by_role[role] = {
                "exact_documents": exact_documents,
                "mismatch_details": mismatches,
                "exact_count": len(exact_documents),
                "checked_count": total,
                "exact_rate": len(exact_documents) / total,
            }
        by_length[str(length)] = by_role
    return {
        "available": True,
        "scope": (
            "canonical Target-AR output diagnostic on docs 12-19; separate from "
            "shadow/bypass trajectory equality and not used for paired inference"
        ),
        "documents": sorted(confirmatory_documents),
        "references": references,
        "by_length": by_length,
    }


def bootstrap_mean(values: np.ndarray, replicates: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValidationError("Bootstrap input must be a non-empty finite vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(replicates, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "available": True,
        "mean": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_documents": int(values.size),
        "document_values": values.tolist(),
    }


def unavailable_stat(reason: str, documents: Sequence[int]) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "mean": None,
        "ci_low": None,
        "ci_high": None,
        "n_documents": len(documents),
        "document_values": None,
    }


def metric_vector(
    rows: Sequence[dict[str, Any]], length: int, role: str, metric: str,
) -> tuple[list[int], np.ndarray | None]:
    selected = sorted(
        (row for row in rows if row["length_bin_tokens"] == length and row["policy_role"] == role),
        key=lambda row: row["sample_index"],
    )
    documents = [row["sample_index"] for row in selected]
    values = [row.get(metric) for row in selected]
    available = [finite(value) is not None for value in values]
    if not any(available):
        return documents, None
    if not all(available):
        raise ValidationError(
            f"Metric {metric} is only partially available for role={role}, length={length}"
        )
    return documents, np.asarray(values, dtype=float)


def policy_statistics(
    rows: Sequence[dict[str, Any]], lengths: Sequence[int], bootstrap: int, seed: int,
    policies: dict[str, str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for length in lengths:
        by_role = {}
        for role in policies:
            documents, acceptance_values = metric_vector(
                rows, length, role, "avg_accept_length"
            )
            if acceptance_values is None:
                raise ValidationError(f"Acceptance unexpectedly missing for {role}/{length}")
            metrics = {}
            for metric in SCALAR_METRICS:
                metric_documents, values = metric_vector(rows, length, role, metric)
                if metric_documents != documents:
                    raise ValidationError(f"Document pairing changed within {role}/{length}/{metric}")
                metrics[metric] = (
                    unavailable_stat("not collected in input bundle", documents)
                    if values is None
                    else bootstrap_mean(
                        values, bootstrap,
                        stable_seed(seed, "policy", length, role, metric),
                    )
                )
            by_role[role] = {"documents": documents, "metrics": metrics}
        result[str(length)] = by_role
    return result


def paired_effect(
    rows: Sequence[dict[str, Any]], length: int, left: str, right: str,
    metric: str, bootstrap: int, seed: int,
    transform: Callable[[float, float], float] | None = None,
) -> dict[str, Any]:
    left_rows = {
        (row["sample_index"], row["timing_repeat"]): row.get(metric)
        for row in rows
        if row["length_bin_tokens"] == length and row["policy_role"] == left
    }
    right_rows = {
        (row["sample_index"], row["timing_repeat"]): row.get(metric)
        for row in rows
        if row["length_bin_tokens"] == length and row["policy_role"] == right
    }
    if set(left_rows) != set(right_rows):
        raise ValidationError(
            f"Unpaired repeat blocks for {left} vs {right}, length={length}, metric={metric}"
        )
    documents = sorted({document for document, _ in left_rows})
    availability = [
        finite(left_rows[key]) is not None and finite(right_rows[key]) is not None
        for key in sorted(left_rows)
    ]
    if not any(availability):
        result = unavailable_stat("not collected for one or both paired policies", documents)
        result.update({
            "left": left,
            "right": right,
            "documents": documents,
            "estimand": "left-minus-right" if transform is None else "log(left/right)",
        })
        return result
    if not all(availability):
        raise ValidationError(
            f"Metric {metric} is only partially available in paired repeat blocks for "
            f"{left} vs {right}, length={length}"
        )
    repeat_effects: dict[int, list[float]] = defaultdict(list)
    for (document, repeat), left_value in sorted(left_rows.items()):
        right_value = right_rows[(document, repeat)]
        if transform is None:
            effect = float(left_value) - float(right_value)
        else:
            effect = transform(float(left_value), float(right_value))
        repeat_effects[document].append(float(effect))
    values = np.asarray(
        [float(np.median(repeat_effects[document])) for document in documents],
        dtype=float,
    )
    result = bootstrap_mean(
        values, bootstrap, stable_seed(seed, "paired", length, left, right, metric),
    )
    tolerance = 1.0e-12
    wins = int(np.count_nonzero(values > tolerance))
    losses = int(np.count_nonzero(values < -tolerance))
    ties = int(values.size - wins - losses)
    non_ties = wins + losses
    if non_ties:
        tail = sum(
            math.comb(non_ties, index) for index in range(min(wins, losses) + 1)
        ) / (2.0 ** non_ties)
        sign_p = min(1.0, 2.0 * tail)
    else:
        sign_p = 1.0
    result.update({
        "left": left,
        "right": right,
        "documents": documents,
        "estimand": "left-minus-right" if transform is None else "log(left/right)",
        "pairing": (
            "effect computed within each (document,length,timing_repeat), then median "
            "across repeats within document"
        ),
        "raw_paired_repeats": len(left_rows),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "sign_test_non_ties": non_ties,
        "sign_test_two_sided_p": float(sign_p),
        "sign_test_status": "exploratory, exact, unadjusted",
    })
    return result


def comparison_statistics(
    rows: Sequence[dict[str, Any]], lengths: Sequence[int], bootstrap: int, seed: int,
    policies: dict[str, str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, (left, right, label) in active_comparison_specs(policies).items():
        by_length = {}
        for length in lengths:
            metrics = {
                metric: paired_effect(
                    rows, length, left, right, metric, bootstrap, seed,
                )
                for metric in SCALAR_METRICS
            }
            log_ratio = paired_effect(
                rows, length, left, right, "wall_tokens_per_sec", bootstrap, seed,
                transform=lambda a, b: math.log(a / b),
            )
            log_ratio["geometric_percent"] = 100.0 * (math.exp(log_ratio["mean"]) - 1.0)
            log_ratio["geometric_percent_ci_low"] = 100.0 * (
                math.exp(log_ratio["ci_low"]) - 1.0
            )
            log_ratio["geometric_percent_ci_high"] = 100.0 * (
                math.exp(log_ratio["ci_high"]) - 1.0
            )
            metrics["log_throughput_ratio"] = log_ratio
            by_length[str(length)] = metrics
        result[name] = {
            "label": label,
            "left_role": left,
            "right_role": right,
            "orientation": "left minus right; throughput ratio is geometric left/right",
            "by_length": by_length,
        }
    return result


def context_metadata(
    rows: Sequence[dict[str, Any]], lengths: Sequence[int], expected_docs: set[int],
    repeats: int,
) -> dict[str, Any]:
    output = {}
    for length in lengths:
        selected = [row for row in rows if row["length_bin_tokens"] == length]
        actual = [row["input_tokens"] for row in selected]
        output[str(length)] = {
            "nominal_tokens": length,
            "actual_input_tokens_min": min(actual),
            "actual_input_tokens_max": max(actual),
            "documents": sorted(expected_docs),
            "n_documents": len(expected_docs),
            "technical_repeats_per_cell": repeats,
        }
    return output


def errorbar_policy_panel(
    ax: Any, policy_stats: dict[str, Any], lengths: Sequence[int], metric: str,
    ylabel: str, roles: Sequence[str], scale: float = 1.0,
) -> None:
    colors = {
        "native": "#4C78A8",
        "recent_native": "#76B7B2",
        "shadow": "#9C755F",
        "bypass": "#2A9D8F",
        "recent_bypass": "#F28E2B",
        "post_cache": "#B279A2",
    }
    x = np.arange(len(lengths), dtype=float)
    width = 0.14
    offsets = np.linspace(-2 * width, 2 * width, len(roles))
    plotted = False
    for offset, role in zip(offsets, roles):
        means, lower, upper = [], [], []
        for length in lengths:
            stat = policy_stats[str(length)][role]["metrics"][metric]
            if not stat.get("available", True):
                means.append(np.nan)
                lower.append(0.0)
                upper.append(0.0)
                continue
            plotted = True
            means.append(stat["mean"] / scale)
            lower.append((stat["mean"] - stat["ci_low"]) / scale)
            upper.append((stat["ci_high"] - stat["mean"]) / scale)
        ax.errorbar(
            x + offset, means, yerr=np.vstack([lower, upper]), fmt="o", capsize=3,
            markersize=5, linewidth=1.5, color=colors[role], label=POLICY_LABELS[role],
        )
    ax.set_xticks(x, [f"{length // 1024}K nominal" for length in lengths])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.axhline(0.0, color="#888888", linewidth=0.6, alpha=0.5)
    if not plotted:
        ax.text(
            0.5, 0.5, "NA — not collected in legacy bundle",
            ha="center", va="center", transform=ax.transAxes, color="#666666",
        )


def errorbar_comparison_panel(
    ax: Any, comparison_stats: dict[str, Any], lengths: Sequence[int], metric: str,
    ylabel: str, percent_ratio: bool = False,
) -> None:
    colors = ["#2A9D8F", "#4C78A8", "#F28E2B", "#59A14F", "#B279A2"]
    names = list(comparison_stats)
    x = np.arange(len(lengths), dtype=float)
    width = 0.14
    offsets = np.linspace(
        -0.5 * width * (len(names) - 1),
        0.5 * width * (len(names) - 1),
        len(names),
    )
    plotted = False
    for offset, name, color in zip(offsets, names, colors):
        means, lower, upper = [], [], []
        for length in lengths:
            stat = comparison_stats[name]["by_length"][str(length)][metric]
            if not stat.get("available", True):
                means.append(np.nan)
                lower.append(0.0)
                upper.append(0.0)
                continue
            if percent_ratio:
                mean = stat["geometric_percent"]
                low = stat["geometric_percent_ci_low"]
                high = stat["geometric_percent_ci_high"]
            else:
                mean, low, high = stat["mean"], stat["ci_low"], stat["ci_high"]
            means.append(mean)
            lower.append(mean - low)
            upper.append(high - mean)
            plotted = True
        ax.errorbar(
            x + offset, means, yerr=np.vstack([lower, upper]), fmt="o", capsize=3,
            markersize=5, linewidth=1.5, color=color,
            label=comparison_stats[name]["label"],
        )
    ax.set_xticks(x, [f"{length // 1024}K nominal" for length in lengths])
    ax.set_ylabel(ylabel)
    ax.axhline(0.0, color="#555555", linewidth=0.8)
    ax.grid(axis="y", alpha=0.25)
    if not plotted:
        ax.text(
            0.5, 0.5, "NA — not collected in legacy bundle",
            ha="center", va="center", transform=ax.transAxes, color="#666666",
        )


def save_figure(fig: Any, output_dir: Path, stem: str) -> list[str]:
    paths = []
    for suffix in ("png", "svg"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path.name)
    return paths


def plot_figures(
    policy_stats: dict[str, Any], comparison_stats: dict[str, Any], lengths: Sequence[int],
    contexts: dict[str, Any], output_dir: Path, n_documents: int, bootstrap: int,
    shadow_control: str, fixed_config: dict[str, Any], roles: Sequence[str],
) -> dict[str, list[str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "figure.titlesize": 14,
        "legend.fontsize": 8,
    })
    note = (
        f"Docs 4-19 reused/exploratory; n={n_documents} documents. Absolute policy CIs: "
        f"document bootstrap; contrasts: repeat-paired, then document-median and paired "
        f"bootstrap={bootstrap:,}; "
        f"shadow={shadow_control}; recent_native={fixed_config.get('recent_native')}; "
        f"tree_nodes={fixed_config.get('tree_nodes')}; cache_top_k={fixed_config.get('cache_top_k')}."
    )
    tick_labels = []
    for length in lengths:
        context = contexts[str(length)]
        low, high = context["actual_input_tokens_min"], context["actual_input_tokens_max"]
        actual = str(low) if low == high else f"{low}-{high}"
        tick_labels.append(f"{length // 1024}K nominal\nactual {actual}")

    quality, axes = plt.subplots(2, 2, figsize=(13, 8))
    errorbar_policy_panel(
        axes[0, 0], policy_stats, lengths, "avg_accept_length",
        "Acceptance length (policy mean)", roles,
    )
    errorbar_comparison_panel(
        axes[0, 1], comparison_stats, lengths, "avg_accept_length",
        "Paired acceptance effect (left - right)",
    )
    errorbar_comparison_panel(
        axes[1, 0], comparison_stats, lengths, "mean_root_target_draft_kl",
        "Paired root-KL effect (left - right)",
    )
    errorbar_comparison_panel(
        axes[1, 1], comparison_stats, lengths, "round_count",
        "Paired round-count effect (left - right)",
    )
    axes[0, 0].legend(ncol=2, loc="best")
    axes[0, 1].legend(ncol=2, loc="best")
    for ax in axes.flat:
        ax.set_xticklabels(tick_labels)
    quality.suptitle("BridgeKV compute bypass: quality")
    quality.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=8)
    quality.tight_layout(rect=(0, 0.04, 1, 0.95))
    quality_paths = save_figure(quality, output_dir, "bridge_compute_bypass_quality")
    plt.close(quality)

    performance, axes = plt.subplots(2, 2, figsize=(13, 8))
    errorbar_policy_panel(
        axes[0, 0], policy_stats, lengths, "wall_tokens_per_sec",
        "End-to-end tokens/s (policy mean)", roles,
    )
    errorbar_comparison_panel(
        axes[0, 1], comparison_stats, lengths, "log_throughput_ratio",
        "Geometric throughput change (%)", percent_ratio=True,
    )
    errorbar_comparison_panel(
        axes[1, 0], comparison_stats, lengths, "draft_prefill_time_ms",
        "Paired Draft-prefill latency (ms; left - right)",
    )
    errorbar_comparison_panel(
        axes[1, 1], comparison_stats, lengths, "bridge_prepare_time_ms",
        "Paired full Bridge-prepare latency (ms; left - right)",
    )
    axes[0, 0].legend(ncol=2, loc="best")
    axes[0, 1].legend(ncol=2, loc="best")
    for ax in axes.flat:
        ax.set_xticklabels(tick_labels)
    performance.suptitle("BridgeKV compute bypass: actual system performance")
    performance.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=8)
    performance.tight_layout(rect=(0, 0.04, 1, 0.95))
    performance_paths = save_figure(
        performance, output_dir, "bridge_compute_bypass_performance",
    )
    plt.close(performance)

    mechanism, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, metric, ylabel in (
        (axes[0, 0], "bridge_prepare_time_ms", "Full Bridge prepare (ms)"),
        (axes[0, 1], "draft_prefill_time_ms", "Draft prefill (ms)"),
        (axes[0, 2], "draft_prefill_v_projection_macs", "Projected Draft-V MACs (G)"),
        (axes[1, 0], "draft_prefill_v_projection_saved_macs", "Skipped Draft-V MACs (G)"),
        (axes[1, 1], "bridge_prefill_mapper_macs", "Bridge mapper MACs (G)"),
        (axes[1, 2], "draft_prefill_net_projection_macs_saved", "Net projection MACs saved (G)"),
    ):
        scale = 1.0 if metric.endswith("_ms") else 1.0e9
        errorbar_policy_panel(ax, policy_stats, lengths, metric, ylabel, roles, scale=scale)
    axes[0, 0].legend(ncol=2, loc="best")
    for ax in axes.flat:
        ax.set_xticklabels(tick_labels)
    mechanism.suptitle("BridgeKV compute bypass: mechanism accounting")
    mechanism.text(
        0.5, 0.01,
        note + "  Replacement applies only to initial Draft prefill. MAC accounting excludes "
        "indexing, LayerNorm, SiLU, and memory traffic.",
        ha="center", va="bottom", fontsize=8,
    )
    mechanism.tight_layout(rect=(0, 0.04, 1, 0.95))
    mechanism_paths = save_figure(
        mechanism, output_dir, "bridge_compute_bypass_mechanism",
    )
    plt.close(mechanism)
    return {
        "quality": quality_paths,
        "performance": performance_paths,
        "mechanism": mechanism_paths,
    }


def format_ci(stat: dict[str, Any], digits: int = 3) -> str:
    if not stat.get("available", True):
        return "NA（未采集）"
    return (
        f"{stat['mean']:.{digits}f} "
        f"[{stat['ci_low']:.{digits}f}, {stat['ci_high']:.{digits}f}]"
    )


def comparison_table(
    comparison: dict[str, Any], lengths: Sequence[int], include_quality: bool = True,
) -> str:
    columns = [
        "上下文", "吞吐几何变化", "TPS W/T/L", "TPS sign-p", "Draft prefill Δms",
        "Full prepare Δms", "Dense mapper Δms",
    ]
    if include_quality:
        columns[1:1] = [
            "接受长度 Δ", "Accept W/T/L", "Accept sign-p", "Root KL Δ", "轮数 Δ",
        ]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for length in lengths:
        metrics = comparison["by_length"][str(length)]
        ratio = metrics["log_throughput_ratio"]
        values = [f"{length // 1024}K（nominal）"]
        if include_quality:
            acceptance = metrics["avg_accept_length"]
            values.extend([
                format_ci(acceptance),
                f"{acceptance['wins']}/{acceptance['ties']}/{acceptance['losses']}",
                f"{acceptance['sign_test_two_sided_p']:.4f}",
                format_ci(metrics["mean_root_target_draft_kl"]),
                format_ci(metrics["round_count"]),
            ])
        values.extend([
            (
                f"{ratio['geometric_percent']:.2f}% "
                f"[{ratio['geometric_percent_ci_low']:.2f}%, "
                f"{ratio['geometric_percent_ci_high']:.2f}%]"
            ),
            f"{ratio['wins']}/{ratio['ties']}/{ratio['losses']}",
            f"{ratio['sign_test_two_sided_p']:.4f}",
            format_ci(metrics["draft_prefill_time_ms"]),
            format_ci(metrics["bridge_prepare_time_ms"]),
            format_ci(metrics["bridge_map_time_ms"]),
        ])
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def mechanism_table(
    policy_stats: dict[str, Any], lengths: Sequence[int], roles: Sequence[str],
) -> str:
    lines = [
        "| 上下文 | 策略 | prepare ms | prefill ms | projected V GMAC | skipped V GMAC | mapper GMAC | net saved GMAC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    fields = (
        "draft_prefill_v_projection_macs",
        "draft_prefill_v_projection_saved_macs",
        "bridge_prefill_mapper_macs",
        "draft_prefill_net_projection_macs_saved",
    )
    for length in lengths:
        for role in roles:
            metrics = policy_stats[str(length)][role]["metrics"]
            values = [metrics[field]["mean"] / 1.0e9 for field in fields]
            prepare = metrics["bridge_prepare_time_ms"]
            prepare_text = (
                f"{prepare['mean']:.3f}"
                if prepare.get("available", True) else "NA"
            )
            lines.append(
                f"| {length // 1024}K（nominal） | {POLICY_LABELS[role]} | "
                f"{prepare_text} | "
                f"{metrics['draft_prefill_time_ms']['mean']:.3f} | "
                + " | ".join(f"{value:.3f}" for value in values) + " |"
            )
    return "\n".join(lines)


def write_report(summary: dict[str, Any], output_dir: Path) -> Path:
    lengths = summary["lengths"]
    comparisons = summary["comparisons"]
    contexts = summary["contexts"]
    checks = summary["exactness_and_accounting_checks"]
    target_reference = summary["canonical_target_reference_diagnostics"]
    gpu = summary["validated_configuration"]["environment"].get("gpu", "unknown")
    shadow_control = summary["validated_configuration"]["shadow_control"]
    recent_native = summary["validated_configuration"]["recent_native_setting"]
    recent_label = summary["validated_configuration"]["recent_native_label"]
    exact_ids = checks["unique_trajectory_output_id_exact"]
    checked_ids = checks["unique_trajectory_cells_checked"]
    exact_accepts = checks["unique_trajectory_acceptance_list_exact"]
    checked_accepts = checks["unique_trajectory_cells_checked"]
    trajectory_breakdown = "；".join(
        f"{int(length) // 1024}K output "
        f"{item['output_id_exact']}/{item['checked']}, accept "
        f"{item['acceptance_list_exact']}/{item['checked']}"
        for length, item in sorted(
            checks["unique_trajectory_by_length"].items(), key=lambda pair: int(pair[0])
        )
    )
    if shadow_control == "full-shape":
        shadow_explanation = (
            "本 bundle 使用新的 single full-shape shadow：shadow 对全部 token 做一次原生 `v_proj` 后覆盖"
            "映射区间；bypass 只对原生补集做一次较小 GEMM。两者数学上的有效 V 相同，但不同 GEMM M-shape "
            "可能产生 FP16 舍入差，因此输出/接受完全一致率作为诊断报告，不作为硬通过条件。"
        )
        timing_scope = (
            "该 shadow 是公平的实际计算对照；吞吐和 prefill 时延可以用于评估省略映射区间 V 投影的"
            "运行时效果，但任何发生的数值轨迹差异也必须与计时结果同时披露。"
        )
        certificate_text = (
            f"{checks['producer_replacement_certificates_passed']} 行通过。"
        )
        validation_intro = "以下结构与计算记账为硬校验；输出轨迹相等性为显式诊断："
    else:
        shadow_explanation = (
            "本 bundle 显式标记为旧式 two-GEMM shadow：它先执行与 bypass 相同的补集 GEMM，再额外"
            "执行一个被丢弃的映射区间 GEMM。它适合检查语义/输出精确一致性。"
        )
        timing_scope = (
            "旧式 two-GEMM shadow 包含额外 kernel launch，不能作为 full-shape 原生投影的公平性能对照；"
            "其吞吐差只能作为探索性算子诊断，不能作为正式因果系统收益。"
        )
        certificate_text = "NA（legacy schema 尚未写入该证书；由分析器独立硬校验可用计数）。"
        validation_intro = (
            "以下结构、计算记账以及 shadow↔bypass 输出/接受轨迹均为硬校验："
        )
    context_lines = []
    for length in lengths:
        item = contexts[str(length)]
        actual = (
            str(item["actual_input_tokens_min"])
            if item["actual_input_tokens_min"] == item["actual_input_tokens_max"]
            else f"{item['actual_input_tokens_min']}–{item['actual_input_tokens_max']}"
        )
        context_lines.append(
            f"- nominal {length // 1024}K：实际输入 token 为 {actual}；"
            f"文档数 n={item['n_documents']}；每 cell 技术重复 "
            f"{item['technical_repeats_per_cell']} 次。"
        )
    if target_reference.get("available"):
        target_lines = [
            "| 上下文 | 方法 | 与 canonical Target-AR 完全相同 | 精确率 | 不同文档 |",
            "|---|---|---:|---:|---|",
        ]
        for length in lengths:
            for role in ("bypass", "native"):
                item = target_reference["by_length"][str(length)][role]
                mismatch_docs = [
                    str(row["document"]) for row in item["mismatch_details"]
                ]
                target_lines.append(
                    f"| {length // 1024}K（nominal） | {POLICY_LABELS[role]} | "
                    f"{item['exact_count']}/{item['checked_count']} | "
                    f"{100.0 * item['exact_rate']:.1f}% | "
                    f"{', '.join(mismatch_docs) if mismatch_docs else '无'} |"
                )
        target_section = [
            "## Canonical Target-AR 输出诊断（独立于 shadow↔bypass）",
            "",
            "该表只在 docs 12–19 上分别检查 bypass/native 的完整 greedy 输出是否与外部 canonical "
            "Target-AR reference 完全相同。它不是 shadow↔bypass 相等性，也不参与配对 bootstrap 或"
            "方法优劣推断。",
            "",
            *target_lines,
            "",
            "完整输出不一致本身不能区分算法分布变化与 fp16 近似平票造成的数值路径差异；"
            "该诊断只定位需要复核的文档和首个位置，不把它解释为旁路污染 Target 分布。",
            "",
        ]
        target_reference_artifacts = [
            f"- Target-AR reference `{item['path']}` — `{item['sha256']}`"
            for item in target_reference["references"]
        ]
    else:
        target_section = [
            "## Canonical Target-AR 输出诊断（独立于 shadow↔bypass）",
            "",
            f"未执行：{target_reference.get('reason', 'reference unavailable')}。",
            "",
        ]
        target_reference_artifacts = []

    if "post_cache_vs_bypass" in comparisons:
        legacy_section = [
            "## 5. 旧式 post-cache 注入 vs prefill bypass",
            "",
            comparison_table(comparisons["post_cache_vs_bypass"], lengths),
            "",
            "post-cache 方案在 Draft 已完成原生预填充投影后才改缓存；prefill bypass 在注意力层内用映射 V 替代"
            "稳定前缀并避免对应 `v_proj`。该对比的方向是 legacy post-cache − bypass。",
            "",
        ]
        mechanism_heading = "## 6. 机制记账"
    else:
        legacy_section = []
        mechanism_heading = "## 5. 机制记账"

    sections = [
        "# BridgeKV 真实计算旁路：探索性技术验证",
        "",
        "> **证据边界：本文使用 docs 4–19，这些文档已用于此前的开发、试验或结果检查。"
        "因此本结果只能称为复用样本上的 exploratory technical validation（探索性技术验证），"
        "不能称为 fresh confirmation（新鲜确认性实验）。**",
        "",
        "## 结论口径",
        "",
        "策略绝对值先在每个文档、策略、上下文内对技术重复取中位数。策略对比则保持 execution-block pairing："
        "先在每个 `(document, length, timing_repeat)` 内计算差值或 `log(TPS_left/TPS_right)`，再在文档内对"
        "重复效应取中位数，最后仅以文档为独立统计单位。"
        f"策略绝对值区间为文档 bootstrap；所有对比区间均为固定种子、"
        f"{summary['bootstrap_replicates']:,} 次的文档配对 bootstrap 95% CI。"
        "表中普通效应均为 `left − right`，吞吐为文档级 log ratio 均值指数变换后的几何百分比。",
        "W/T/L 按文档级效应计数；sign-p 是剔除 ties 后的双侧 exact sign test，仅作 exploratory、"
        "未进行多重比较校正。CI 跨零或未检出下降都不等价于证明 equivalence/neutral。",
        "",
        f"运行硬件：`{gpu}`。",
        f"稳定前缀设置：`recent_native={recent_native}`（{recent_label}）。",
        "",
        *context_lines,
        "",
        "## 因果一致性与计算记账检查",
        "",
        shadow_explanation,
        "",
        validation_intro,
        "",
        f"- shadow↔bypass 唯一轨迹 cell：{checked_ids}（每个上下文 "
        f"{len(summary['contexts'][str(lengths[0])]['documents'])} 个文档）；"
        f"底层计时 repeat 配对为 {checks['raw_shadow_bypass_repeat_pairs']}，不作为独立证据。",
        f"- 输出 token IDs 完全一致：{exact_ids}/{checked_ids} 个唯一轨迹 cell "
        f"（{100.0 * exact_ids / checked_ids:.1f}%）。",
        f"- 接受长度序列完全一致：{exact_accepts}/{checked_accepts} 对 "
        f"（{100.0 * exact_accepts / checked_accepts:.1f}%）。",
        f"- 分上下文唯一轨迹：{trajectory_breakdown}。",
        f"- 实际输入长度与生成预算在策略/重复间一致："
        f"{checks['paired_actual_context_and_generation_cells']} 个文档×上下文 cell。",
        f"- 映射替换区间逐项相同：{checks['replacement_interval_pairs']} 对。",
        f"- K 投影 token/MAC 计数相同：{checks['aggregate_k_pairs']} 对。",
        f"- producer-side `compute_replacement_validation`：{certificate_text}",
        f"- 逐层 V 恒等式检查：{checks['layerwise_v_checks']} 项；"
        "`shadow projected = bypass projected + bypass skipped` 且 `shadow extra = bypass skipped`。",
        f"- 每个策略跨技术重复的输出轨迹一致：{checks['repeat_exact_groups']} 组。",
        "",
        *target_section,
        "## 1. 真实旁路的因果计算收益：bypass vs shadow",
        "",
        comparison_table(comparisons["bypass_vs_shadow"], lengths),
        "",
        timing_scope,
        "",
        "## 2. 方法净效果：bypass vs native",
        "",
        comparison_table(comparisons["bypass_vs_native"], lengths),
        "",
        "该对比同时包含映射 V 对 Draft 质量/轮数的影响、mapper 开销和被省略 V 投影的收益，"
        "因此是完整方法效果，而不是纯算子微基准。",
        "",
        "## 3. 注意力引导贡献：attention-bypass vs recent-bypass",
        "",
        comparison_table(comparisons["attention_bypass_vs_recent_bypass"], lengths),
        "",
        "两者使用相同的预填充 V 计算旁路，主要差异是后续 working-cache 的选择策略，"
        "因此该对比用于衡量注意力引导，而不是把它与投影旁路混在一起。",
        "",
        "## 4. 安全旁路：recent-bypass vs recent-native",
        "",
        comparison_table(comparisons["recent_bypass_vs_recent_native"], lengths),
        "",
        "两者都采用 recent working-cache retrieval；差异仅在 initial Draft prefill 是否以内联映射 V 替换"
        "稳定前缀并跳过相应 `v_proj`，因此这是保守检索路径上的安全旁路对照。",
        "",
        *legacy_section,
        mechanism_heading,
        "",
        mechanism_table(summary["policy_summaries"], lengths, list(summary["policies"])),
        "",
        "`net saved MACs = skipped Draft-V MACs − prefill mapper MACs`。该算术只覆盖稠密线性层 MAC，"
        "不包含索引、拼接、LayerNorm、SiLU、kernel launch 或显存流量，因此不能代替端到端 tokens/s。",
        "`bridge_prepare_time_ms` 是 initial prefill 的完整 inline 准备时延（gather + dense mapper + materialize）；"
        "`bridge_map_time_ms` 只计 dense mapper kernel。计算替换仅发生在 initial Draft prefill，后续解码并未"
        "被表中的 skipped MACs 覆盖。",
        "旧式 two-GEMM bundle 若未采集 `bridge_prepare_time_ms`，图表保留为 NA；不会用更窄的 map time 代填。",
        "",
        "## 图与原始数据",
        "",
        "- [质量图](bridge_compute_bypass_quality.png)（[SVG](bridge_compute_bypass_quality.svg)）",
        "- [端到端性能图](bridge_compute_bypass_performance.png)（[SVG](bridge_compute_bypass_performance.svg)）",
        "- [机制记账图](bridge_compute_bypass_mechanism.png)（[SVG](bridge_compute_bypass_mechanism.svg)）",
        "- [机器可读汇总](bridge_compute_bypass_summary.json)",
        "- [分析与绘图源码](analyze_bridge_compute_bypass.py)",
        "",
        "原始 bundle 及 SHA-256：",
        "",
        *[
            f"- `{item['path']}` — shadow=`{item['shadow_control']}` — `{item['sha256']}`"
            for item in summary["inputs"]
        ],
        *target_reference_artifacts,
        "",
        "## 限制",
        "",
        "- docs 4–19 不是独立保留的确认集；所有显著性与区间仅作探索性技术诊断。",
        "- 技术重复只用于稳定计时；策略对比先在同一 repeat block 内配对，再在文档内取中位数，"
        "未被当成额外样本。",
        "- GPU 分层固定为 docs 4–7/8–11/12–15/16–19 分别运行在物理 GPU 0/1/2/3。"
        "所有比较均在同一文档、同一 GPU 内配对，但文档组与 GPU 完全混杂，无法独立估计 GPU 效应。",
        "- `no detected degradation` 不等于 equivalence；本实验未设置等价界值，也未执行等价性检验。",
        "- 若要形成确认性结论，应冻结代码和超参数后，在从未查看的新文档上一次性复验。",
    ]
    path = output_dir / "BRIDGE_COMPUTE_BYPASS_REPORT.md"
    path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return path


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    if args.bootstrap <= 0:
        raise ValidationError("--bootstrap must be positive")
    input_paths = flatten_inputs(args.input)
    target_reference_paths = flatten_optional_inputs(args.target_reference)
    bundles = [load_bundle(path) for path in input_paths]
    policies = selected_policies(args, bundles)
    expected_docs = parse_index_spec(args.expected_docs)
    expected_lengths = set(args.expected_lengths)
    if not expected_lengths:
        raise ValidationError("--expected-lengths cannot be empty")
    config_summary = validate_configs(
        bundles, policies, expected_lengths, args.allow_non_4090, args.shadow_control,
    )

    rows = []
    ignored_policy_rows: dict[str, int] = defaultdict(int)
    for bundle in bundles:
        for index, row in enumerate(bundle.rows):
            if row.get("policy") not in policies.values():
                ignored_policy_rows[str(row.get("policy"))] += 1
                continue
            rows.append(canonicalize_row(row, bundle, policies, index))
    checks = validate_rows(
        rows, bundles, policies, expected_docs, expected_lengths, args.shadow_control,
    )
    target_reference = target_reference_diagnostics(
        target_reference_paths, rows, bundles, policies, expected_lengths,
    )
    aggregated = aggregate_repeats(rows, policies)
    lengths = sorted(expected_lengths)
    policy_stats = policy_statistics(
        aggregated, lengths, args.bootstrap, args.seed, policies,
    )
    comparisons = comparison_statistics(
        rows, lengths, args.bootstrap, args.seed, policies,
    )
    repeats = as_int(bundles[0].config["timing_repeats"], "timing_repeats")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_source = Path(__file__).resolve()
    analysis_copy = output_dir / analysis_source.name
    if analysis_copy.resolve() != analysis_source:
        shutil.copy2(analysis_source, analysis_copy)
    contexts = context_metadata(rows, lengths, expected_docs, repeats)
    figure_paths = plot_figures(
        policy_stats, comparisons, lengths, contexts, output_dir, len(expected_docs),
        args.bootstrap, args.shadow_control, config_summary["fixed_config"],
        list(policies),
    )
    summary = {
        "schema_version": 1,
        "analysis_status": (
            "exploratory technical validation on reused docs 4-19; not fresh confirmation"
        ),
        "statistical_unit": "document",
        "technical_repeat_aggregation": (
            "policy levels: within-document median; contrasts: pair within each "
            "(document,length,timing_repeat), then within-document median of effects"
        ),
        "interval_method": (
            "policy levels: document bootstrap percentile 95% CI; contrasts: paired "
            "document bootstrap percentile 95% CI"
        ),
        "compute_replacement_scope": "initial Draft prefill only",
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "inputs": [
            {
                "path": str(bundle.path),
                "sha256": bundle.sha256,
                "rows": len(bundle.rows),
                "shadow_control": detect_shadow_control(bundle),
                "provenance": bundle.provenance,
            }
            for bundle in bundles
        ],
        "validated_configuration": config_summary,
        "policies": policies,
        "lengths": lengths,
        "contexts": contexts,
        "total_input_row_count": sum(len(bundle.rows) for bundle in bundles),
        "raw_row_count": len(rows),
        "ignored_unselected_policy_rows": dict(sorted(ignored_policy_rows.items())),
        "document_aggregated_row_count": len(aggregated),
        "exactness_and_accounting_checks": checks,
        "canonical_target_reference_diagnostics": target_reference,
        "policy_summaries": policy_stats,
        "comparisons": comparisons,
        "figures": figure_paths,
        "analysis_source": {
            "path": str(analysis_copy),
            "sha256": sha256_file(analysis_copy),
        },
    }
    summary_path = output_dir / "bridge_compute_bypass_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    report_path = write_report(summary, output_dir)
    summary["summary_path"] = str(summary_path)
    summary["report_path"] = str(report_path)
    return summary


def main() -> None:
    args = parse_args()
    try:
        summary = analyze(args)
    except ValidationError as error:
        raise SystemExit(f"VALIDATION ERROR: {error}") from error
    print(json.dumps({
        "status": summary["analysis_status"],
        "raw_rows": summary["raw_row_count"],
        "document_rows": summary["document_aggregated_row_count"],
        "report": summary["report_path"],
        "summary": summary["summary_path"],
        "figures": summary["figures"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
