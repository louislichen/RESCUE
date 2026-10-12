#!/usr/bin/env python3
"""Strict analysis for the Native-V retention dose-response sweep.

The statistical unit is a document.  Technical timing repeats are first
collapsed with a within-document median.  A single document-resampling matrix
is then reused for every context, retention level, and metric; the resulting
95% intervals are pointwise intervals, not simultaneous confidence bands.

This analyzer intentionally refuses incomplete factorials, mixed provenance,
or failed compute-accounting certificates.  Smaller smoke runs are supported
only by explicitly overriding the expected documents, ratios, and repeat
count on the command line; they are labelled partial/structural in the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


BYPASS_POLICY = "attention_bridge_value_bypass_last_a100"
SHADOW_POLICY = "attention_bridge_value_shadow_last_a100"
NATIVE_POLICY = "attention_native"

DEFAULT_LENGTHS = (4096, 8192)
DEFAULT_RATIOS_BP = (0, 1250, 2500, 3750, 5000, 6250, 7500, 8750, 10000)
DEFAULT_DOCUMENTS = tuple(range(4, 20))
DEFAULT_REPEATS = 3

FIXED_CONFIG_FIELDS = (
    "base_model_path",
    "draft_model_path",
    "device_map",
    "checkpoint",
    "source_file",
    "lengths",
    "retention_percentages",
    "max_gen_len",
    "tree_nodes",
    "chunk_size",
    "cache_top_k",
    "retrieve_every",
    "seed",
    "timing_repeats",
    "warmup_gen_len",
    "exact_target_attention",
)

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

LAYER_FIELDS = (
    "k_projected_tokens",
    "v_projected_tokens",
    "v_skipped_tokens",
    "v_shadow_tokens",
    "prefill_k_projected_tokens",
    "prefill_v_projected_tokens",
    "prefill_v_skipped_tokens",
    "prefill_v_shadow_tokens",
)

AGGREGATE_FROM_LAYER = {
    "k_projected_tokens": "draft_k_projected_tokens",
    "v_projected_tokens": "draft_v_projected_tokens",
    "v_skipped_tokens": "draft_v_skipped_tokens",
    "v_shadow_tokens": "draft_v_shadow_tokens",
    "prefill_k_projected_tokens": "draft_prefill_k_projected_tokens",
    "prefill_v_projected_tokens": "draft_prefill_v_projected_tokens",
    "prefill_v_skipped_tokens": "draft_prefill_v_skipped_tokens",
    "prefill_v_shadow_tokens": "draft_prefill_v_shadow_tokens",
}

CURVE_METRICS = (
    "avg_accept_length",
    "draft_prefill_time_ms",
    "bridge_prepare_time_ms",
    "draft_prefill_plus_bridge_prepare_time_ms",
    "wall_tokens_per_sec",
    "bridge_map_time_ms",
    "draft_prefill_v_projected_tokens",
    "draft_prefill_v_skipped_tokens",
    "draft_prefill_net_projection_macs_saved",
)


class ValidationError(ValueError):
    """The input cannot support the requested formal interpretation."""


@dataclass(frozen=True)
class Bundle:
    path: Path
    config: dict[str, Any]
    sweep_spec: dict[str, Any]
    provenance: dict[str, Any]
    metric_definitions: dict[str, Any]
    environment: dict[str, Any]
    rows: list[dict[str, Any]]
    sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the strict Native-V retention BridgeKV sweep."
    )
    parser.add_argument(
        "--input", action="append", nargs="+", required=True,
        help="One or more native_v_retention_sweep JSON bundles; may be repeated.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--expected-docs", default="4-19")
    parser.add_argument(
        "--expected-lengths", nargs="+", type=int, default=list(DEFAULT_LENGTHS)
    )
    parser.add_argument(
        "--expected-retention-percentages", nargs="+", type=float,
        default=[value / 100.0 for value in DEFAULT_RATIOS_BP],
    )
    parser.add_argument("--expected-repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--allow-non-4090", action="store_true")
    args = parser.parse_args()
    if args.bootstrap <= 0:
        raise ValidationError("--bootstrap must be positive")
    if args.expected_repeats <= 0:
        raise ValidationError("--expected-repeats must be positive")
    return args


def finite(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def as_int(value: Any, name: str) -> int:
    number = finite(value)
    if number is None or int(number) != number:
        raise ValidationError(f"{name} must be a finite integer, got {value!r}")
    return int(number)


def ratio_basis_points(value: Any, name: str = "retention percentage") -> int:
    number = finite(value)
    if number is None or not 0.0 <= number <= 100.0:
        raise ValidationError(f"{name} must be finite and in [0,100], got {value!r}")
    basis_points = int(round(number * 100.0))
    if not math.isclose(number, basis_points / 100.0, abs_tol=1.0e-9):
        raise ValidationError(f"{name} supports at most two decimal places")
    return basis_points


def retained_rows(eligible_rows: int, basis_points: int) -> int:
    """Match the producer's integer round-half-up rule."""

    return (eligible_rows * basis_points + 5000) // 10000


def parse_index_spec(spec: str) -> set[int]:
    output: set[int] = set()
    for raw in spec.split(","):
        part = raw.strip()
        if not part:
            continue
        if "-" in part[1:]:
            start_text, stop_text = part.split("-", 1)
            start, stop = int(start_text), int(stop_text)
            if stop < start:
                raise ValidationError(f"Descending document range: {part}")
            output.update(range(start, stop + 1))
        elif ":" in part:
            start_text, stop_text = part.split(":", 1)
            output.update(range(int(start_text), int(stop_text)))
        else:
            output.add(int(part))
    if not output:
        raise ValidationError("--expected-docs resolved to an empty set")
    return output


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


def load_bundle(path: Path) -> Bundle:
    if not path.is_file():
        raise ValidationError(f"Input does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValidationError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValidationError(f"Top-level JSON must be an object: {path}")
    if payload.get("schema_version") != 1:
        raise ValidationError(f"{path}: schema_version must be 1")
    if payload.get("experiment") != "native_v_retention_sweep":
        raise ValidationError(f"{path}: unexpected experiment identifier")
    required_objects = {
        "config": payload.get("config"),
        "sweep_spec": payload.get("sweep_spec"),
        "provenance": payload.get("provenance"),
        "metric_definitions": payload.get("metric_definitions"),
        "environment": payload.get("environment"),
    }
    for name, value in required_objects.items():
        if not isinstance(value, dict):
            raise ValidationError(f"{path}: {name} must be an object")
    rows = payload.get("raw_runs")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValidationError(f"{path}: raw_runs must be a non-empty list of objects")
    return Bundle(
        path=path,
        config=required_objects["config"],
        sweep_spec=required_objects["sweep_spec"],
        provenance=required_objects["provenance"],
        metric_definitions=required_objects["metric_definitions"],
        environment=required_objects["environment"],
        rows=[dict(row) for row in rows],
        sha256=sha256_file(path),
    )


def policy_core(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        key: spec.get(key)
        for key in (
            "retrieval_policy", "alpha", "layers", "source_mode", "components",
            "compute_mode",
        )
        if key in spec
    }


def expected_policy_core(kind: str) -> dict[str, Any]:
    if kind == "measured_bypass":
        return {
            "retrieval_policy": "attention", "alpha": 1.0, "layers": "last",
            "source_mode": "aligned", "components": "value",
            "compute_mode": "prefill_value_bypass",
        }
    if kind == "semantic_shadow":
        return {
            "retrieval_policy": "attention", "alpha": 1.0, "layers": "last",
            "source_mode": "aligned", "components": "value",
            "compute_mode": "prefill_value_shadow",
        }
    if kind == "native_anchor":
        return {
            "retrieval_policy": "attention", "alpha": 0.0, "layers": "all",
            "source_mode": "aligned",
        }
    raise ValidationError(f"Unknown record_kind={kind!r}")


def validate_bundle_metadata(
    bundles: Sequence[Bundle], expected_lengths: set[int], expected_ratios: set[int],
    expected_docs: set[int], expected_repeats: int, allow_non_4090: bool,
) -> dict[str, Any]:
    baseline = bundles[0]
    required_implementation_files = {
        "attn_bridge_experiment.py",
        "classic/attn_bridge.py",
        "classic/model_classic.py",
        "classic/modeling_llama_kv_draft.py",
        "native_v_retention_sweep.py",
    }
    baseline_fixed = {field: baseline.config.get(field) for field in FIXED_CONFIG_FIELDS}
    baseline_provenance = {
        key: baseline.provenance.get(key)
        for key in (
            "implementation_sha256", "checkpoint_sha256", "source_file_sha256",
            "device_layout",
        )
    }
    gpu_document_map = []
    declared_document_sets = []

    for bundle in bundles:
        missing = [field for field in FIXED_CONFIG_FIELDS if field not in bundle.config]
        if missing:
            raise ValidationError(f"{bundle.path}: config missing fields {missing}")
        lengths = {as_int(value, f"{bundle.path}:config.lengths") for value in bundle.config["lengths"]}
        if lengths != expected_lengths:
            raise ValidationError(
                f"{bundle.path}: configured lengths {sorted(lengths)} != expected "
                f"{sorted(expected_lengths)}"
            )
        ratios = {
            ratio_basis_points(value, f"{bundle.path}:config.retention_percentages")
            for value in bundle.config["retention_percentages"]
        }
        if ratios != expected_ratios or len(bundle.config["retention_percentages"]) != len(expected_ratios):
            raise ValidationError(
                f"{bundle.path}: configured ratios {sorted(ratios)} != expected "
                f"{sorted(expected_ratios)}"
            )
        if as_int(bundle.config["timing_repeats"], "timing_repeats") != expected_repeats:
            raise ValidationError(f"{bundle.path}: timing_repeats mismatch")
        if as_int(bundle.config.get("samples"), f"{bundle.path}:samples") <= 0:
            raise ValidationError(f"{bundle.path}: samples must be positive")

        sweep = bundle.sweep_spec
        if sweep.get("base_policy") != BYPASS_POLICY:
            raise ValidationError(f"{bundle.path}: incorrect base sweep policy")
        if policy_core(sweep.get("policy_spec", {})) != expected_policy_core("measured_bypass"):
            raise ValidationError(f"{bundle.path}: incorrect base policy specification")
        if "eligible" not in str(sweep.get("retention_denominator", "")).lower():
            raise ValidationError(f"{bundle.path}: retention denominator is not documented")
        if "recent" not in str(sweep.get("selection_rule", "")).lower():
            raise ValidationError(f"{bundle.path}: selection rule is not recent-position retention")
        if "attention" not in str(sweep.get("working_cache_retrieval", "")).lower():
            raise ValidationError(f"{bundle.path}: working-cache retrieval is not attention-guided")

        gpu = str(bundle.environment.get("gpu", ""))
        if not gpu:
            raise ValidationError(f"{bundle.path}: missing GPU model")
        if "4090" not in gpu and not allow_non_4090:
            raise ValidationError(
                f"{bundle.path}: expected RTX 4090, got {gpu!r}; use --allow-non-4090 only "
                "for structural tests"
            )
        for field in ("torch", "cuda"):
            if not bundle.environment.get(field):
                raise ValidationError(f"{bundle.path}: environment.{field} is missing")

        implementation = bundle.provenance.get("implementation_sha256")
        if not isinstance(implementation, dict):
            raise ValidationError(f"{bundle.path}: implementation hashes are missing")
        if not required_implementation_files <= set(implementation):
            raise ValidationError(
                f"{bundle.path}: implementation hashes missing "
                f"{sorted(required_implementation_files - set(implementation))}"
            )
        for name, digest in implementation.items():
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValidationError(f"{bundle.path}: invalid implementation hash for {name}")
        for field in ("checkpoint_sha256", "source_file_sha256"):
            digest = bundle.provenance.get(field)
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValidationError(f"{bundle.path}: invalid provenance.{field}")
        if not isinstance(bundle.provenance.get("device_layout"), dict):
            raise ValidationError(f"{bundle.path}: provenance.device_layout is missing")

        for field in (
            "avg_accept_length", "draft_prefill_time_ms", "bridge_prepare_time_ms",
            "wall_tokens_per_sec", "timing_repeat",
        ):
            if field not in bundle.metric_definitions:
                raise ValidationError(f"{bundle.path}: metric definition missing {field}")

        if {field: bundle.config.get(field) for field in FIXED_CONFIG_FIELDS} != baseline_fixed:
            raise ValidationError(f"{bundle.path}: fixed configuration differs across bundles")
        comparable_provenance = {
            key: bundle.provenance.get(key) for key in baseline_provenance
        }
        if comparable_provenance != baseline_provenance:
            raise ValidationError(
                f"{bundle.path}: code, checkpoint, source data, or device layout differs"
            )
        if bundle.environment.get("gpu") != baseline.environment.get("gpu"):
            raise ValidationError(f"{bundle.path}: GPU model differs across bundles")
        if bundle.environment.get("torch") != baseline.environment.get("torch"):
            raise ValidationError(f"{bundle.path}: PyTorch version differs across bundles")
        if bundle.environment.get("cuda") != baseline.environment.get("cuda"):
            raise ValidationError(f"{bundle.path}: CUDA version differs across bundles")

        start = as_int(bundle.config.get("sample_start"), f"{bundle.path}:sample_start")
        count = as_int(bundle.config.get("samples"), f"{bundle.path}:samples")
        declared_documents = set(range(start, start + count))
        observed_documents = {
            as_int(row.get("sample_index"), f"{bundle.path}:row.sample_index")
            for row in bundle.rows
        }
        if observed_documents != declared_documents:
            raise ValidationError(
                f"{bundle.path}: observed documents {sorted(observed_documents)} differ from "
                f"declared shard {sorted(declared_documents)}"
            )
        declared_document_sets.append(declared_documents)
        gpu_document_map.append({
            "input": str(bundle.path),
            "documents_declared": sorted(declared_documents),
            "cuda_visible_devices": bundle.environment.get(
                "cuda_visible_devices", bundle.provenance.get("cuda_visible_devices")
            ),
            "gpu_model": gpu,
        })

    declared_union: set[int] = set()
    for documents in declared_document_sets:
        overlap = declared_union & documents
        if overlap:
            raise ValidationError(f"Document shards overlap across bundles: {sorted(overlap)}")
        declared_union.update(documents)
    if declared_union != expected_docs:
        raise ValidationError(
            f"Declared bundle shards {sorted(declared_union)} != expected docs "
            f"{sorted(expected_docs)}"
        )

    return {
        "fixed_config": baseline_fixed,
        "provenance": baseline_provenance,
        "environment": dict(baseline.environment),
        "gpu_document_map": gpu_document_map,
    }


def canonicalize_layer_rows(value: Any, where: str) -> list[dict[str, int]]:
    if not isinstance(value, list) or not value:
        raise ValidationError(f"{where}: draft_projection_by_layer must be non-empty")
    output = []
    seen = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValidationError(f"{where}: layer row {index} must be an object")
        layer = as_int(raw.get("layer"), f"{where}:layer")
        if layer in seen:
            raise ValidationError(f"{where}: duplicate layer {layer}")
        seen.add(layer)
        item = {"layer": layer}
        for field in LAYER_FIELDS:
            item[field] = as_int(raw.get(field), f"{where}:layer[{layer}].{field}")
            if item[field] < 0:
                raise ValidationError(f"{where}: negative layer counter {field}")
        output.append(item)
    output.sort(key=lambda item: item["layer"])
    if [item["layer"] for item in output] != list(range(len(output))):
        raise ValidationError(f"{where}: layers must be consecutive from zero")
    return output


def canonicalize_events(value: Any, where: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValidationError(f"{where}: draft_value_replacement_events must be a list")
    output = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValidationError(f"{where}: replacement event {index} must be an object")
        item = {
            "layer": as_int(raw.get("layer"), f"{where}:event.layer"),
            "start": as_int(raw.get("start"), f"{where}:event.start"),
            "stop": as_int(raw.get("stop"), f"{where}:event.stop"),
            "tokens": as_int(raw.get("tokens"), f"{where}:event.tokens"),
            "shadow_compute": bool(raw.get("shadow_compute", False)),
        }
        if item["start"] < 0 or item["stop"] <= item["start"]:
            raise ValidationError(f"{where}: invalid replacement interval {item}")
        if item["tokens"] != item["stop"] - item["start"]:
            raise ValidationError(f"{where}: replacement token count mismatches interval")
        output.append(item)
    return sorted(output, key=lambda item: (item["layer"], item["start"], item["stop"]))


def validate_certificate(row: dict[str, Any], where: str) -> None:
    kind = row["record_kind"]
    certificate = row.get("compute_replacement_validation")
    if kind == "native_anchor":
        if certificate is not None:
            raise ValidationError(f"{where}: native anchor must not carry a bypass certificate")
        return
    if not isinstance(certificate, dict) or certificate.get("status") != "passed":
        raise ValidationError(f"{where}: inline row lacks a passed compute certificate")
    query_tokens = row["input_tokens"] + 1
    mapped_tokens = row["bridge_v_mapped_rows"]
    for field, expected in (
        ("query_tokens", query_tokens), ("mapped_tokens", mapped_tokens),
        ("active_layers", 1),
    ):
        if as_int(certificate.get(field), f"{where}:certificate.{field}") != expected:
            raise ValidationError(f"{where}: certificate {field} mismatch")
    v_macs_per_row = as_int(certificate.get("v_macs_per_row"), f"{where}:v_macs_per_row")
    if v_macs_per_row <= 0:
        raise ValidationError(f"{where}: v_macs_per_row must be positive")
    expected_payload = certificate.get("expected")
    if not isinstance(expected_payload, dict):
        raise ValidationError(f"{where}: certificate.expected must be an object")
    certified_fields = (
        "bridge_map_events", "bridge_prepare_events", "draft_prefill_events",
        "bridge_mapped_tokens", "bridge_value_bypass_events",
        "bridge_value_shadow_events", "draft_prefill_k_projected_tokens",
        "draft_prefill_v_projected_tokens", "draft_prefill_v_skipped_tokens",
        "draft_prefill_v_shadow_tokens", "draft_prefill_v_projection_macs",
        "draft_prefill_v_projection_saved_macs", "draft_prefill_v_shadow_macs",
        "bridge_prefill_mapper_macs", "draft_prefill_net_projection_macs_saved",
    )
    for field in certified_fields:
        certified = as_int(expected_payload.get(field), f"{where}:certificate.expected.{field}")
        if as_int(row.get(field), f"{where}:{field}") != certified:
            raise ValidationError(f"{where}: row {field} differs from producer certificate")

    layer_count = len(row["draft_projection_by_layer"])
    mapped_rows = mapped_tokens
    total_rows = layer_count * query_tokens
    is_bypass = kind == "measured_bypass"
    independent = {
        "bridge_map_events": int(mapped_tokens > 0),
        "bridge_prepare_events": int(mapped_tokens > 0),
        "draft_prefill_events": 1,
        "bridge_mapped_tokens": mapped_tokens,
        "bridge_value_bypass_events": int(is_bypass and mapped_tokens > 0),
        "bridge_value_shadow_events": int((not is_bypass) and mapped_tokens > 0),
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
        "draft_prefill_v_shadow_macs": (
            0 if is_bypass else mapped_rows * v_macs_per_row
        ),
    }
    for field, expected in independent.items():
        if as_int(row.get(field), f"{where}:{field}") != expected:
            raise ValidationError(f"{where}: independently derived {field} mismatch")


def canonicalize_row(raw: dict[str, Any], bundle: Bundle, index: int) -> dict[str, Any]:
    row = dict(raw)
    where = f"{bundle.path}:raw_runs[{index}]"
    kind = str(row.get("record_kind", ""))
    if kind not in {"measured_bypass", "semantic_shadow", "native_anchor"}:
        raise ValidationError(f"{where}: invalid record_kind={kind!r}")
    row["record_kind"] = kind
    for field in (
        "length_bin_tokens", "input_tokens", "sample_index", "timing_repeat",
        "execution_order", "total_generated", "native_v_retention_basis_points",
        "eligible_native_v_rows", "native_v_retained_rows", "bridge_v_mapped_rows",
        "forced_native_last_layer_rows",
    ):
        row[field] = as_int(row.get(field), f"{where}.{field}")
    if row["input_tokens"] != row["length_bin_tokens"]:
        raise ValidationError(
            f"{where}: actual context {row['input_tokens']} != nominal "
            f"{row['length_bin_tokens']}"
        )
    if row["total_generated"] <= 0:
        raise ValidationError(f"{where}: total_generated must be positive")
    expected_generated = as_int(bundle.config.get("max_gen_len"), f"{where}:max_gen_len")
    if row["total_generated"] != expected_generated:
        raise ValidationError(
            f"{where}: scored generated tokens {row['total_generated']} != fixed budget "
            f"{expected_generated}"
        )
    basis_points = ratio_basis_points(
        row.get("native_v_retention_percent"), f"{where}.native_v_retention_percent"
    )
    if basis_points != row["native_v_retention_basis_points"]:
        raise ValidationError(f"{where}: percentage/basis-point mismatch")
    eligible = row["input_tokens"] - 1
    retained = retained_rows(eligible, basis_points)
    mapped = eligible - retained
    if row["eligible_native_v_rows"] != eligible:
        raise ValidationError(f"{where}: eligible row count mismatch")
    if row["native_v_retained_rows"] != retained:
        raise ValidationError(f"{where}: retained row count mismatch")
    if row["bridge_v_mapped_rows"] != mapped:
        raise ValidationError(f"{where}: mapped row count mismatch")
    if row["forced_native_last_layer_rows"] != 2:
        raise ValidationError(f"{where}: forced-native last-layer rows must equal 2")
    actual_percent = 100.0 * retained / eligible
    observed_actual = finite(row.get("actual_native_v_retention_percent"))
    if observed_actual is None or not math.isclose(
        observed_actual, actual_percent, rel_tol=1.0e-12, abs_tol=1.0e-12
    ):
        raise ValidationError(f"{where}: actual eligible retention percentage mismatch")
    row["actual_native_v_retention_percent"] = actual_percent

    expected_policy = {
        "measured_bypass": f"native_v_retain_bp_{basis_points:05d}",
        "semantic_shadow": f"shadow_retain_bp_{basis_points:05d}",
        "native_anchor": NATIVE_POLICY,
    }[kind]
    # ``base_policy`` is producer metadata whose historical meaning changed
    # across runners.  The actual executed policy and its full configuration
    # are the authoritative fields for analysis.
    if row.get("policy") != expected_policy:
        raise ValidationError(f"{where}: policy name does not match record kind/ratio")
    config = row.get("policy_config")
    if not isinstance(config, dict):
        raise ValidationError(f"{where}: policy_config must be an object")
    if policy_core(config) != expected_policy_core(kind):
        raise ValidationError(f"{where}: policy_config semantics mismatch")
    if as_int(config.get("recent_native"), f"{where}:recent_native") != retained:
        raise ValidationError(f"{where}: policy recent_native differs from retained rows")
    if bool(row.get("formal_timing_eligible")) != (kind == "measured_bypass"):
        raise ValidationError(f"{where}: formal_timing_eligible mismatch")
    if bool(row.get("wall_timing_is_diagnostic", False)):
        raise ValidationError(f"{where}: diagnostic wall timing cannot support TPS analysis")

    scalar_fields = (
        "avg_accept_length", "wall_time", "wall_tokens_per_sec",
        "draft_prefill_time_ms", "bridge_prepare_time_ms", "bridge_map_time_ms",
        "draft_prefill_plus_bridge_prepare_time_ms",
    )
    for field in scalar_fields:
        value = finite(row.get(field))
        if value is None:
            raise ValidationError(f"{where}.{field} must be finite")
        if field in {"wall_time", "wall_tokens_per_sec"} and value <= 0:
            raise ValidationError(f"{where}.{field} must be positive")
        if field.endswith("_ms") and value < 0:
            raise ValidationError(f"{where}.{field} must be non-negative")
        row[field] = float(value)

    output_ids = row.get("output_token_ids")
    accepts = row.get("accept_length_list")
    if not isinstance(output_ids, list) or not all(isinstance(value, int) for value in output_ids):
        raise ValidationError(f"{where}: output_token_ids must be integer IDs")
    if not isinstance(accepts, list) or not accepts or not all(isinstance(value, int) for value in accepts):
        raise ValidationError(f"{where}: accept_length_list must be non-empty integers")
    if len(output_ids) != row["total_generated"]:
        raise ValidationError(f"{where}: generated-token count mismatches output IDs")
    exact_acceptance = float(np.mean(accepts))
    if abs(row["avg_accept_length"] - exact_acceptance) > 5.0e-4 + 1.0e-12:
        raise ValidationError(f"{where}: avg_accept_length mismatches primitive list")
    row["avg_accept_length"] = exact_acceptance
    exact_tps = row["total_generated"] / row["wall_time"]
    if not math.isclose(row["wall_tokens_per_sec"], exact_tps, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValidationError(f"{where}: TPS mismatches generated tokens / wall time")
    row["wall_tokens_per_sec"] = exact_tps
    exact_total = row["draft_prefill_time_ms"] + row["bridge_prepare_time_ms"]
    if not math.isclose(
        row["draft_prefill_plus_bridge_prepare_time_ms"], exact_total,
        rel_tol=1.0e-12, abs_tol=1.0e-12,
    ):
        raise ValidationError(f"{where}: prefill+prepare summary mismatch")
    row["draft_prefill_plus_bridge_prepare_time_ms"] = exact_total

    for field in COUNT_FIELDS + MAC_FIELDS:
        row[field] = as_int(row.get(field), f"{where}.{field}")
        if row[field] < 0 and field != "draft_prefill_net_projection_macs_saved":
            raise ValidationError(f"{where}.{field} must be non-negative")
    if row["draft_prefill_net_projection_macs_saved"] != (
        row["draft_prefill_v_projection_saved_macs"] - row["bridge_prefill_mapper_macs"]
    ):
        raise ValidationError(f"{where}: net MAC identity failed")

    row["draft_projection_by_layer"] = canonicalize_layer_rows(
        row.get("draft_projection_by_layer"), where
    )
    for layer_field, aggregate_field in AGGREGATE_FROM_LAYER.items():
        if sum(item[layer_field] for item in row["draft_projection_by_layer"]) != row[aggregate_field]:
            raise ValidationError(f"{where}: layer sum for {aggregate_field} mismatch")
    row["draft_value_replacement_events"] = canonicalize_events(
        row.get("draft_value_replacement_events"), where
    )

    layer_count = len(row["draft_projection_by_layer"])
    query_tokens = row["input_tokens"] + 1
    last_layer = layer_count - 1
    expected_events = []
    if kind != "native_anchor" and mapped:
        expected_events = [{
            "layer": last_layer, "start": 1, "stop": row["input_tokens"] - retained,
            "tokens": mapped, "shadow_compute": kind == "semantic_shadow",
        }]
    if row["draft_value_replacement_events"] != expected_events:
        raise ValidationError(f"{where}: replacement interval/flag mismatch")

    for layer in row["draft_projection_by_layer"]:
        active = layer["layer"] == last_layer and kind != "native_anchor"
        layer_mapped = mapped if active else 0
        expected = {
            "prefill_k_projected_tokens": query_tokens,
            "prefill_v_projected_tokens": (
                query_tokens - layer_mapped if kind == "measured_bypass" else query_tokens
            ),
            "prefill_v_skipped_tokens": layer_mapped if kind == "measured_bypass" else 0,
            "prefill_v_shadow_tokens": layer_mapped if kind == "semantic_shadow" else 0,
        }
        for field, value in expected.items():
            if layer[field] != value:
                raise ValidationError(f"{where}: layer {layer['layer']} {field} mismatch")

    if kind == "native_anchor":
        zero_fields = (
            "bridge_prefill_mapper_macs", "draft_prefill_v_projection_saved_macs",
            "draft_prefill_v_shadow_macs", "draft_prefill_v_skipped_tokens",
            "draft_prefill_v_shadow_tokens", "draft_prefill_net_projection_macs_saved",
        )
        for field in zero_fields:
            if row[field] != 0:
                raise ValidationError(f"{where}: native anchor {field} must be zero")
    validate_certificate(row, where)
    row["_source"] = str(bundle.path)
    return row


def validate_pair_accounting(
    bypass: dict[str, Any], shadow: dict[str, Any], where: str,
) -> None:
    if bypass["output_token_ids"] != shadow["output_token_ids"]:
        raise ValidationError(f"{where}: shadow/bypass output IDs are not exact")
    if bypass["accept_length_list"] != shadow["accept_length_list"]:
        raise ValidationError(f"{where}: shadow/bypass acceptance trajectory is not exact")
    if bypass["draft_value_replacement_events"] != [
        {**event, "shadow_compute": False}
        for event in shadow["draft_value_replacement_events"]
    ]:
        raise ValidationError(f"{where}: shadow/bypass replacement intervals differ")
    for field in (
        "draft_k_projected_tokens", "draft_k_projection_macs",
        "draft_prefill_k_projected_tokens", "draft_prefill_k_projection_macs",
        "bridge_prefill_mapper_macs",
    ):
        if bypass[field] != shadow[field]:
            raise ValidationError(f"{where}: shadow/bypass {field} differs")
    equalities = (
        (shadow["draft_v_projected_tokens"], bypass["draft_v_projected_tokens"] + bypass["draft_v_skipped_tokens"]),
        (shadow["draft_v_shadow_tokens"], bypass["draft_v_skipped_tokens"]),
        (shadow["draft_prefill_v_projected_tokens"], bypass["draft_prefill_v_projected_tokens"] + bypass["draft_prefill_v_skipped_tokens"]),
        (shadow["draft_prefill_v_shadow_tokens"], bypass["draft_prefill_v_skipped_tokens"]),
        (shadow["draft_v_projection_macs"], bypass["draft_v_projection_macs"] + bypass["draft_v_projection_saved_macs"]),
        (shadow["draft_v_shadow_macs"], bypass["draft_v_projection_saved_macs"]),
        (shadow["draft_prefill_v_projection_macs"], bypass["draft_prefill_v_projection_macs"] + bypass["draft_prefill_v_projection_saved_macs"]),
        (shadow["draft_prefill_v_shadow_macs"], bypass["draft_prefill_v_projection_saved_macs"]),
    )
    if not all(left == right for left, right in equalities):
        raise ValidationError(f"{where}: aggregate shadow/bypass V identity failed")


def validate_factorial(
    rows: Sequence[dict[str, Any]], expected_docs: set[int], expected_lengths: set[int],
    expected_ratios: set[int], expected_repeats: int,
) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    counts = defaultdict(int)
    for row in rows:
        key = (
            row["record_kind"], row["length_bin_tokens"], row["sample_index"],
            row["native_v_retention_basis_points"], row["timing_repeat"],
        )
        if key in grouped:
            raise ValidationError(f"Duplicate factorial row: {key}")
        grouped[key] = row
        counts[row["record_kind"]] += 1

    if {row["sample_index"] for row in rows} != expected_docs:
        raise ValidationError("Observed documents differ from --expected-docs")
    if {row["length_bin_tokens"] for row in rows} != expected_lengths:
        raise ValidationError("Observed lengths differ from --expected-lengths")
    if {row["native_v_retention_basis_points"] for row in rows} != expected_ratios:
        raise ValidationError("Observed retention levels differ from expected levels")

    expected_main = len(expected_docs) * len(expected_lengths) * len(expected_ratios) * expected_repeats
    expected_shadow = len(expected_docs) * len(expected_lengths) * len(expected_ratios)
    expected_anchor = len(expected_docs) * len(expected_lengths)
    expected_counts = {
        "measured_bypass": expected_main,
        "semantic_shadow": expected_shadow,
        "native_anchor": expected_anchor,
    }
    if dict(counts) != expected_counts:
        raise ValidationError(f"Row counts {dict(counts)} != expected {expected_counts}")
    expected_keys = {
        ("measured_bypass", length, document, ratio, repeat)
        for length in expected_lengths
        for document in expected_docs
        for ratio in expected_ratios
        for repeat in range(expected_repeats)
    }
    expected_keys.update({
        ("semantic_shadow", length, document, ratio, 0)
        for length in expected_lengths
        for document in expected_docs
        for ratio in expected_ratios
    })
    expected_keys.update({
        ("native_anchor", length, document, 10000, 0)
        for length in expected_lengths
        for document in expected_docs
    })
    if set(grouped) != expected_keys:
        missing = sorted(expected_keys - set(grouped), key=str)[:20]
        extra = sorted(set(grouped) - expected_keys, key=str)[:20]
        raise ValidationError(
            f"Incomplete/extra factorial cells: missing(first20)={missing}; "
            f"extra(first20)={extra}"
        )

    v_macs_per_row_values = set()
    mapper_macs_per_mapped_row_values = set()
    for row in rows:
        if row["record_kind"] == "native_anchor":
            continue
        certificate = row["compute_replacement_validation"]
        v_macs_per_row_values.add(as_int(certificate["v_macs_per_row"], "v_macs_per_row"))
        mapped = row["bridge_v_mapped_rows"]
        mapper_macs = row["bridge_prefill_mapper_macs"]
        if mapped:
            if mapper_macs % mapped:
                raise ValidationError(
                    "Mapper MACs are not an integer-linear function of mapped token rows"
                )
            mapper_macs_per_mapped_row_values.add(mapper_macs // mapped)
        elif mapper_macs:
            raise ValidationError("Zero mapped rows must have zero mapper MACs")
    if len(v_macs_per_row_values) != 1:
        raise ValidationError(
            f"V-projection MACs/row differ across inputs: {sorted(v_macs_per_row_values)}"
        )
    if len(mapper_macs_per_mapped_row_values) != 1:
        raise ValidationError(
            "Mapper MACs/mapped-row differ across ratios, contexts, or bundles: "
            f"{sorted(mapper_macs_per_mapped_row_values)}"
        )

    exact_repeat_groups = 0
    exact_shadow_pairs = 0
    exact_anchor_pairs = 0
    metadata_cells = 0
    for length in sorted(expected_lengths):
        for document in sorted(expected_docs):
            cell_rows = []
            for ratio in sorted(expected_ratios):
                repeats = [
                    grouped[("measured_bypass", length, document, ratio, repeat)]
                    for repeat in range(expected_repeats)
                ]
                if any(row["output_token_ids"] != repeats[0]["output_token_ids"] for row in repeats[1:]):
                    raise ValidationError(
                        f"Output IDs vary across repeats at {(length, document, ratio)}"
                    )
                if any(row["accept_length_list"] != repeats[0]["accept_length_list"] for row in repeats[1:]):
                    raise ValidationError(
                        f"Acceptance trajectory varies across repeats at {(length, document, ratio)}"
                    )
                exact_repeat_groups += 1
                shadow = grouped[("semantic_shadow", length, document, ratio, 0)]
                validate_pair_accounting(
                    repeats[0], shadow, f"length={length},doc={document},ratio_bp={ratio}"
                )
                exact_shadow_pairs += 1
                cell_rows.extend(repeats + [shadow])
            anchor = grouped[("native_anchor", length, document, 10000, 0)]
            bypass_100 = grouped[("measured_bypass", length, document, 10000, 0)]
            if anchor["output_token_ids"] != bypass_100["output_token_ids"]:
                raise ValidationError(f"100% bypass differs from native output at {(length, document)}")
            if anchor["accept_length_list"] != bypass_100["accept_length_list"]:
                raise ValidationError(f"100% bypass differs from native acceptance at {(length, document)}")
            exact_anchor_pairs += 1
            cell_rows.append(anchor)
            if {row["input_tokens"] for row in cell_rows} != {length}:
                raise ValidationError(f"Actual contexts differ in cell {(length, document)}")
            if len({row["total_generated"] for row in cell_rows}) != 1:
                raise ValidationError(f"Generation budgets differ in cell {(length, document)}")
            metadata_cells += 1

    formal_complete = (
        tuple(sorted(expected_docs)) == DEFAULT_DOCUMENTS
        and tuple(sorted(expected_lengths)) == DEFAULT_LENGTHS
        and tuple(sorted(expected_ratios)) == DEFAULT_RATIOS_BP
        and expected_repeats == DEFAULT_REPEATS
    )
    return {
        "hard_validation_passed": True,
        "formal_complete_design": formal_complete,
        "row_counts": dict(counts),
        "expected_row_counts": expected_counts,
        "total_rows": len(rows),
        "repeat_trajectory_exact_groups": exact_repeat_groups,
        "shadow_bypass_exact_pairs": exact_shadow_pairs,
        "native_anchor_100_percent_exact_pairs": exact_anchor_pairs,
        "paired_context_generation_cells": metadata_cells,
        "producer_certificates_passed": sum(
            isinstance(row.get("compute_replacement_validation"), dict)
            and row["compute_replacement_validation"].get("status") == "passed"
            for row in rows
        ),
        "v_projection_macs_per_row": next(iter(v_macs_per_row_values)),
        "mapper_macs_per_mapped_row": next(iter(mapper_macs_per_mapped_row_values)),
    }


def aggregate_document_medians(
    rows: Sequence[dict[str, Any]], expected_docs: Sequence[int],
    expected_lengths: Sequence[int], expected_ratios: Sequence[int],
) -> dict[tuple[int, int, int], dict[str, float]]:
    groups: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["record_kind"] == "measured_bypass":
            groups[(
                row["length_bin_tokens"], row["native_v_retention_basis_points"],
                row["sample_index"],
            )].append(row)
    output = {}
    for length in expected_lengths:
        for ratio in expected_ratios:
            for document in expected_docs:
                key = (length, ratio, document)
                group = groups.get(key, [])
                if not group:
                    raise ValidationError(f"Missing measured group {key}")
                item = {
                    metric: float(np.median([float(row[metric]) for row in group]))
                    for metric in CURVE_METRICS
                }
                actual_ratios = {row["actual_native_v_retention_percent"] for row in group}
                if len(actual_ratios) != 1:
                    raise ValidationError(f"Actual retention differs within repeats at {key}")
                item["actual_native_v_retention_percent"] = actual_ratios.pop()
                output[key] = item
    return output


def jointly_bootstrap_curves(
    medians: dict[tuple[int, int, int], dict[str, float]], documents: Sequence[int],
    lengths: Sequence[int], ratios: Sequence[int], replicates: int, seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(documents), size=(replicates, len(documents)))
    curves: dict[str, Any] = {}
    for length in lengths:
        points = []
        for ratio in ratios:
            actual = np.asarray([
                medians[(length, ratio, document)]["actual_native_v_retention_percent"]
                for document in documents
            ])
            if not np.allclose(actual, actual[0], rtol=0.0, atol=1.0e-12):
                raise ValidationError(
                    f"Actual retention x-coordinate varies across documents at {length}/{ratio}"
                )
            metrics = {}
            for metric in CURVE_METRICS:
                values = np.asarray([
                    medians[(length, ratio, document)][metric] for document in documents
                ], dtype=float)
                if not np.isfinite(values).all():
                    raise ValidationError(f"Non-finite document values for {length}/{ratio}/{metric}")
                bootstrap_means = values[sample_indices].mean(axis=1)
                low, high = np.percentile(bootstrap_means, [2.5, 97.5])
                metrics[metric] = {
                    "mean": float(values.mean()),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "n_documents": len(documents),
                    "document_medians": values.tolist(),
                }
            points.append({
                "requested_native_v_retention_percent": ratio / 100.0,
                "requested_native_v_retention_basis_points": ratio,
                "actual_eligible_native_v_retention_percent": float(actual[0]),
                "metrics": metrics,
            })
        curves[str(length)] = points
    return curves


def paired_curve_contrasts(
    medians: dict[tuple[int, int, int], dict[str, float]], documents: Sequence[int],
    lengths: Sequence[int], ratios: Sequence[int], replicates: int, seed: int,
) -> dict[str, Any]:
    """Paired document contrasts using the same joint resampling scheme."""

    specifications = {
        "native_cliff_100_vs_87_5": (10000, 8750),
        "mapped_region_87_5_vs_0": (8750, 0),
        "full_range_100_vs_0": (10000, 0),
    }
    available = {
        name: pair for name, pair in specifications.items() if set(pair) <= set(ratios)
    }
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(documents), size=(replicates, len(documents)))
    output: dict[str, Any] = {}
    for name, (left_ratio, right_ratio) in available.items():
        by_length = {}
        for length in lengths:
            metrics = {}
            for metric in CURVE_METRICS:
                values = np.asarray([
                    medians[(length, left_ratio, document)][metric]
                    - medians[(length, right_ratio, document)][metric]
                    for document in documents
                ], dtype=float)
                bootstrap_means = values[sample_indices].mean(axis=1)
                low, high = np.percentile(bootstrap_means, [2.5, 97.5])
                metrics[metric] = {
                    "mean": float(values.mean()),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "n_documents": len(documents),
                    "paired_document_differences": values.tolist(),
                }
            by_length[str(length)] = metrics
        output[name] = {
            "left_retention_basis_points": left_ratio,
            "right_retention_basis_points": right_ratio,
            "estimand": "left minus right after within-document timing-repeat medians",
            "by_length": by_length,
        }
    return output


def save_figure(fig: Any, output_dir: Path, stem: str) -> list[str]:
    paths = []
    for suffix in ("png", "svg"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path.name)
    return paths


def metric_arrays(points: Sequence[dict[str, Any]], metric: str, scale: float = 1.0):
    x = np.asarray([point["actual_eligible_native_v_retention_percent"] for point in points])
    mean = np.asarray([point["metrics"][metric]["mean"] / scale for point in points])
    low = np.asarray([point["metrics"][metric]["ci_low"] / scale for point in points])
    high = np.asarray([point["metrics"][metric]["ci_high"] / scale for point in points])
    return x, mean, low, high


def plot_single_metric(
    curves: dict[str, Any], lengths: Sequence[int], output_dir: Path, metric: str,
    ylabel: str, title: str, stem: str, note: str,
) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {4096: "#4C78A8", 8192: "#E45756"}
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    for index, length in enumerate(lengths):
        x, mean, low, high = metric_arrays(curves[str(length)], metric)
        color = colors.get(length, plt.cm.tab10(index % 10))
        label = f"{length // 1024}K (actual {length} tokens)"
        ax.plot(x, mean, marker="o", linewidth=2.0, color=color, label=label)
        ax.fill_between(x, low, high, color=color, alpha=0.16)
    ax.set_xlabel(
        "Actual final-layer Native-V recent-suffix retention among eligible rows 1..N-1 (%)"
    )
    ax.set_ylabel(ylabel)
    ax.set_xlim(-2, 102)
    ax.set_xticks(np.arange(0, 101, 12.5))
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    ax.set_title(title)
    fig.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    paths = save_figure(fig, output_dir, stem)
    plt.close(fig)
    return paths


def plot_prefill(
    curves: dict[str, Any], lengths: Sequence[int], output_dir: Path, note: str,
) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {4096: "#4C78A8", 8192: "#E45756"}
    fig, axes = plt.subplots(1, len(lengths), figsize=(13.0, 5.8), sharex=True)
    axes = np.atleast_1d(axes)
    for index, (ax, length) in enumerate(zip(axes, lengths)):
        color = colors.get(length, plt.cm.tab10(index % 10))
        x, mean, low, high = metric_arrays(curves[str(length)], "draft_prefill_time_ms")
        ax.plot(x, mean, marker="o", linewidth=2.0, color=color, label="Draft prefill")
        ax.fill_between(x, low, high, color=color, alpha=0.15)
        x2, mean2, low2, high2 = metric_arrays(
            curves[str(length)], "draft_prefill_plus_bridge_prepare_time_ms"
        )
        ax.plot(
            x2, mean2, marker="s", linewidth=1.8, linestyle="--", color=color,
            label="Draft prefill + Bridge prepare",
        )
        ax.fill_between(x2, low2, high2, color=color, alpha=0.07)
        ax.set_xlabel("Actual final-layer eligible recent-suffix Native-V retention (%)")
        ax.set_ylabel("Latency (ms)")
        ax.set_xlim(-2, 102)
        ax.set_xticks(np.arange(0, 101, 25))
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        ax.set_title(f"{length // 1024}K — actual {length} tokens; eligible rows 1..N-1")
    fig.suptitle("Native-V retention sweep: Draft prefill latency", fontsize=14)
    fig.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    paths = save_figure(fig, output_dir, "native_v_retention_prefill_latency")
    plt.close(fig)
    return paths


def plot_mechanism(
    curves: dict[str, Any], lengths: Sequence[int], output_dir: Path, note: str,
) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {4096: "#4C78A8", 8192: "#E45756"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for index, length in enumerate(lengths):
        color = colors.get(length, plt.cm.tab10(index % 10))
        label = f"{length // 1024}K actual {length}"
        for metric, linestyle, marker, suffix in (
            ("bridge_prepare_time_ms", "-", "o", "prepare"),
            ("bridge_map_time_ms", "--", "s", "dense map"),
        ):
            x, mean, low, high = metric_arrays(curves[str(length)], metric)
            axes[0].plot(
                x, mean, color=color, linestyle=linestyle, marker=marker,
                label=f"{label}: {suffix}",
            )
            axes[0].fill_between(x, low, high, color=color, alpha=0.10)
        for metric, linestyle, marker, suffix in (
            ("draft_prefill_v_projected_tokens", "-", "o", "projected"),
            ("draft_prefill_v_skipped_tokens", "--", "s", "skipped"),
        ):
            x, mean, low, high = metric_arrays(curves[str(length)], metric, scale=1.0e3)
            axes[1].plot(
                x, mean, color=color, linestyle=linestyle, marker=marker,
                label=f"{label}: {suffix}",
            )
            axes[1].fill_between(x, low, high, color=color, alpha=0.10)
        x, mean, low, high = metric_arrays(
            curves[str(length)], "draft_prefill_net_projection_macs_saved", scale=1.0e9
        )
        axes[2].plot(x, mean, color=color, marker="o", label=label)
        axes[2].fill_between(x, low, high, color=color, alpha=0.16)
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Bridge preparation / dense map")
    axes[1].set_ylabel("Initial-prefill V token-rows (thousands)")
    axes[1].set_title("Projected and skipped V rows")
    axes[2].set_ylabel("Net projection MACs saved (G)")
    axes[2].set_title("Skipped V-projection MACs − mapper MACs")
    axes[2].axhline(0.0, color="#666666", linewidth=0.8)
    for ax in axes:
        ax.set_xlabel("Final-layer recent-suffix Native-V retention in eligible rows 1..N-1 (%)")
        ax.set_xlim(-2, 102)
        ax.set_xticks(np.arange(0, 101, 25))
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=7)
    fig.suptitle("Native-V retention sweep: mechanism diagnostics", fontsize=14)
    fig.text(0.5, 0.005, note, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    paths = save_figure(fig, output_dir, "native_v_retention_mechanism")
    plt.close(fig)
    return paths


def format_ci(stat: dict[str, Any], digits: int = 3) -> str:
    return (
        f"{stat['mean']:.{digits}f} "
        f"[{stat['ci_low']:.{digits}f}, {stat['ci_high']:.{digits}f}]"
    )


def format_delta_ci(stat: dict[str, Any], digits: int = 3) -> str:
    return (
        f"{stat['mean']:+.{digits}f} "
        f"[{stat['ci_low']:+.{digits}f}, {stat['ci_high']:+.{digits}f}]"
    )


def report_table(points: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| Requested | Actual eligible | Acceptance | Draft prefill ms | Prefill+prepare ms | TPS | prepare ms | map ms | projected V rows | skipped V rows | net saved GMAC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in points:
        metrics = point["metrics"]
        lines.append("| " + " | ".join([
            f"{point['requested_native_v_retention_percent']:.2f}%",
            f"{point['actual_eligible_native_v_retention_percent']:.4f}%",
            format_ci(metrics["avg_accept_length"]),
            format_ci(metrics["draft_prefill_time_ms"]),
            format_ci(metrics["draft_prefill_plus_bridge_prepare_time_ms"]),
            format_ci(metrics["wall_tokens_per_sec"], 2),
            format_ci(metrics["bridge_prepare_time_ms"]),
            format_ci(metrics["bridge_map_time_ms"]),
            f"{metrics['draft_prefill_v_projected_tokens']['mean']:.0f}",
            f"{metrics['draft_prefill_v_skipped_tokens']['mean']:.0f}",
            f"{metrics['draft_prefill_net_projection_macs_saved']['mean'] / 1e9:.3f}",
        ]) + " |")
    return "\n".join(lines)


def write_report(summary: dict[str, Any], output_dir: Path) -> Path:
    validation = summary["validation"]
    design = summary["design"]
    formal = validation["formal_complete_design"]
    status = (
        "完整的 16-document 探索性扫描"
        if formal else "部分/结构自测扫描（不得作为正式结论）"
    )
    lines = [
        "# Native-V 保留比例扫描报告",
        "",
        f"**状态：{status}。** 所有硬性完整性、轨迹和计算量证书检查均已通过。",
        "",
        "## 设计与横轴定义",
        "",
        (
            "横轴是 **最后一层 eligible rows `1..N-1` 中，按 recent suffix 实际保留为 "
            "Native V 的比例**。因此横轴 0% 仍不表示整个 Draft 的 V 都被替换："
            "prompt 位置 0、未验证 root，以及其他 Draft 层仍保持原生计算。替换只发生在首次 "
            "Draft prefill 的最后一层 V；Q/K 始终原生。"
        ),
        "",
        (
            "这里的 attention 只用于 working-cache retrieval；比例扫描的 Native-V 保留集合"
            "是按年龄选择的 recent suffix，**不是 attention-guided replacement mask**。"
        ),
        "",
        (
            f"文档 {design['documents']}；实际上下文严格为 {design['lengths']} tokens；"
            f"每个正式曲线点先在同一文档内对 {design['technical_repeats']} 个 timing repeats "
            "取中位数，再以文档为统计单位。"
        ),
        "",
        "## 硬性验证",
        "",
        f"- 行数：main `{validation['row_counts']['measured_bypass']}`，shadow "
        f"`{validation['row_counts']['semantic_shadow']}`，native anchors "
        f"`{validation['row_counts']['native_anchor']}`，总计 `{validation['total_rows']}`。",
        f"- ratio/document 跨 repeats 轨迹完全一致：`{validation['repeat_trajectory_exact_groups']}` 组。",
        f"- 每个 ratio 的 shadow/bypass output IDs 与 acceptance lists 完全一致："
        f"`{validation['shadow_bypass_exact_pairs']}` 对。",
        f"- 100% bypass 与 attention-native anchor 完全一致："
        f"`{validation['native_anchor_100_percent_exact_pairs']}` 对。",
        f"- producer compute certificates 通过：`{validation['producer_certificates_passed']}` 行；"
        "分析端又独立核验 replacement interval、逐层 projected/skipped/shadow rows 和 MAC 恒等式。",
        "",
        "## 三条主要曲线",
        "",
        "- [接受长度曲线](native_v_retention_acceptance.png)",
        "- [Draft prefill latency 曲线](native_v_retention_prefill_latency.png)（虚线为 prefill + prepare）",
        "- [端到端 TPS 曲线](native_v_retention_throughput.png)",
        "- [机制诊断图](native_v_retention_mechanism.png)",
        "",
    ]
    cliff = summary["paired_contrasts"].get("native_cliff_100_vs_87_5")
    mapped_region = summary["paired_contrasts"].get("mapped_region_87_5_vs_0")
    if cliff is not None:
        lines.extend([
            "## 关键结果：质量恢复集中在 100% Native 边界",
            "",
            (
                "曲线不是平滑的质量—计算折中：从 0% 到 87.5% 时接受长度只小幅变化，"
                "而从 87.5% 到 100%（最后一批 mapped V 完全消失）出现明显跳变。"
                "下面是文档内配对的 `100% − 87.5%` 差值；正的 acceptance/TPS 表示 100% 更高，"
                "负的 latency 表示 100% 更快。"
            ),
            "",
            "| Context | Acceptance Δ | Draft prefill Δms | Prefill+prepare Δms | TPS Δ |",
            "|---|---:|---:|---:|---:|",
        ])
        for length in design["lengths"]:
            metrics = cliff["by_length"][str(length)]
            lines.append("| " + " | ".join([
                f"{length // 1024}K actual {length}",
                format_delta_ci(metrics["avg_accept_length"]),
                format_delta_ci(metrics["draft_prefill_time_ms"]),
                format_delta_ci(metrics["draft_prefill_plus_bridge_prepare_time_ms"]),
                format_delta_ci(metrics["wall_tokens_per_sec"], 2),
            ]) + " |")
        lines.extend([
            "",
            (
                "这说明当前 mapper 的问题不是仅靠逐步提高 recent-suffix Native 比例就能平滑消除："
                "即便只剩 12.5% eligible old-prefix rows 使用 mapped V，仍会显著拖累 proposal quality。"
                "100% 点同时切换到零 mapper/零 materialization 的普通 full-shape native 路径，因此该边界"
                "不能从中间比例作线性外推。"
            ),
            "",
        ])
    if mapped_region is not None:
        lines.extend([
            "在仍存在 mapped V 的区间内，`87.5% − 0%` 的配对结果为：",
            "",
            "| Context | Acceptance Δ | Prefill+prepare Δms | TPS Δ |",
            "|---|---:|---:|---:|",
        ])
        for length in design["lengths"]:
            metrics = mapped_region["by_length"][str(length)]
            lines.append("| " + " | ".join([
                f"{length // 1024}K actual {length}",
                format_delta_ci(metrics["avg_accept_length"]),
                format_delta_ci(metrics["draft_prefill_plus_bridge_prepare_time_ms"]),
                format_delta_ci(metrics["wall_tokens_per_sec"], 2),
            ]) + " |")
        lines.extend([
            "",
            (
                "Bridge prepare、mapped rows 和净节省 MAC 随保留比例按预期下降；但端到端 TPS 还受"
                "接受长度、验证轮数和 GPU kernel shape 影响，不能只由 prefill 曲线推断。"
            ),
            "",
        ])
    for length in design["lengths"]:
        lines.extend([
            f"## {length // 1024}K（实际 {length} tokens）",
            "",
            report_table(summary["curves"][str(length)]),
            "",
        ])
    lines.extend([
        "## 统计口径与限制",
        "",
        (
            f"同一组 document bootstrap 索引被联合用于所有 context、ratio 与 metric，重复 "
            f"`{summary['analysis']['bootstrap_replicates']:,}` 次，以保留曲线点间配对关系。图中的 "
            "95% CI 是逐点（pointwise）区间，**不是 simultaneous confidence band**；扫多个比例后"
            "不能把单个点的区间解释为全曲线错误率控制。"
        ),
        "",
        "- docs 4–19 已被前序开发/验证查看，本扫描是 reused-data exploratory evidence，不是 fresh confirmation。",
        "- 每个文档分片固定运行在一张物理 GPU 上，document 与 GPU/运行批次混杂；文档 bootstrap 不会消除该系统混杂。",
        "- timing repeats 是技术重复，先在文档内取中位数，不能把它们当作独立样本扩大 n。",
        "- 曲线或 CI 覆盖零/重叠只能写成“未检测到差异”，本分析没有预注册等价界值，也没有做等价性检验。",
        "- TPS 包含端到端系统噪声；prefill、prepare 和 mapper CUDA 计时用于定位机制，不能直接相加推断全部 wall-time 变化。",
        "- Producer provenance 覆盖 sweep runner、Bridge/model 核心代码、checkpoint 与数据文件，但未包含 `baseline_profiler.py` 和 shell launcher；因此“代码一致”只对已哈希的核心实现成立。",
        "- 每个 8K shape/ratio 都做过 warmup，但全部 8K 正式测量排在 4K 正式块之后；无法完全排除跨块温度或时钟漂移。",
        "",
        "## GPU / document 分片",
        "",
        "| Documents | CUDA_VISIBLE_DEVICES | GPU model | Input bundle |",
        "|---|---:|---|---|",
    ])
    for shard in summary["metadata"]["gpu_document_map"]:
        lines.append(
            "| "
            + f"{shard['documents_declared']} | {shard['cuda_visible_devices']} | "
            + f"{shard['gpu_model']} | `{shard['input']}` |"
        )
    lines.extend([
        "",
        "## 可复现产物",
        "",
        "- [机器可读 summary](native_v_retention_summary.json)",
        "- [分析脚本](analyze_native_v_retention_sweep.py)",
        "",
        "输入 bundles：",
        "",
    ])
    for item in summary["inputs"]:
        lines.append(f"- `{item['path']}` — SHA-256 `{item['sha256']}`，rows={item['rows']}")
    path = output_dir / "NATIVE_V_RETENTION_SWEEP_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    expected_docs = parse_index_spec(args.expected_docs)
    expected_lengths = {as_int(value, "expected length") for value in args.expected_lengths}
    expected_ratios = {
        ratio_basis_points(value, "expected retention percentage")
        for value in args.expected_retention_percentages
    }
    if len(expected_ratios) != len(args.expected_retention_percentages):
        raise ValidationError("Expected retention percentages must be unique")
    paths = flatten_inputs(args.input)
    bundles = [load_bundle(path) for path in paths]
    metadata = validate_bundle_metadata(
        bundles, expected_lengths, expected_ratios, expected_docs,
        args.expected_repeats,
        args.allow_non_4090,
    )
    rows = [
        canonicalize_row(raw, bundle, index)
        for bundle in bundles for index, raw in enumerate(bundle.rows)
    ]
    validation = validate_factorial(
        rows, expected_docs, expected_lengths, expected_ratios, args.expected_repeats
    )
    if validation["formal_complete_design"] and len(bundles) != 8:
        raise ValidationError(
            "The formal docs 4-19 design expects eight disjoint GPU/document bundles; "
            f"received {len(bundles)}"
        )
    validation["bundle_count"] = len(bundles)
    documents = sorted(expected_docs)
    lengths = sorted(expected_lengths)
    ratios = sorted(expected_ratios)
    medians = aggregate_document_medians(rows, documents, lengths, ratios)
    curves = jointly_bootstrap_curves(
        medians, documents, lengths, ratios, args.bootstrap, args.seed
    )
    contrasts = paired_curve_contrasts(
        medians, documents, lengths, ratios, args.bootstrap, args.seed
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_label = (
        "Reused-data exploratory"
        if validation["formal_complete_design"] else "PARTIAL structural test"
    )
    fixed = metadata["fixed_config"]
    note = (
        f"{evidence_label}; n={len(documents)} documents; document median of "
        f"{args.expected_repeats} technical repeats; joint document bootstrap="
        f"{args.bootstrap:,}; 95% pointwise (not simultaneous) CIs.\n"
        f"Actual contexts={lengths} tokens; tree_nodes={fixed['tree_nodes']}; "
        f"cache_top_k={fixed['cache_top_k']}; max_new_tokens={fixed['max_gen_len']}. "
        "X=final-layer recent suffix retained as Native V / eligible rows 1..N-1."
    )
    figures = {
        "acceptance": plot_single_metric(
            curves, lengths, output_dir, "avg_accept_length", "Acceptance length",
            "Native-V retention sweep: acceptance length",
            "native_v_retention_acceptance", note,
        ),
        "prefill_latency": plot_prefill(curves, lengths, output_dir, note),
        "throughput": plot_single_metric(
            curves, lengths, output_dir, "wall_tokens_per_sec",
            "End-to-end generated tokens/s",
            "Native-V retention sweep: end-to-end throughput",
            "native_v_retention_throughput", note,
        ),
        "mechanism": plot_mechanism(curves, lengths, output_dir, note),
    }

    analysis_script = Path(__file__).resolve()
    copied_script = output_dir / analysis_script.name
    copied_script.write_bytes(analysis_script.read_bytes())
    summary = {
        "schema_version": 1,
        "experiment": "native_v_retention_sweep_analysis",
        "analysis": {
            "statistical_unit": "document",
            "technical_repeat_reduction": "within-document median before inference",
            "bootstrap_replicates": args.bootstrap,
            "bootstrap_seed": args.seed,
            "bootstrap_pairing": (
                "one shared document-resampling index matrix for all contexts, ratios, and metrics"
            ),
            "confidence_intervals": "95% pointwise percentile; not simultaneous",
            "equivalence_test": "not performed; no equivalence margin specified",
            "analysis_script_sha256": sha256_file(analysis_script),
        },
        "design": {
            "documents": documents,
            "n_documents": len(documents),
            "lengths": lengths,
            "actual_context_tokens": lengths,
            "requested_retention_basis_points": ratios,
            "technical_repeats": args.expected_repeats,
            "x_axis": (
                "actual final-layer recent-suffix Native-V retained rows / eligible prompt "
                "rows (positions 1..N-1)"
            ),
            "forced_native_scope": (
                "position 0, unverified root, and all non-final Draft layers remain native"
            ),
        },
        "validation": validation,
        "metadata": metadata,
        "curves": curves,
        "paired_contrasts": contrasts,
        "figures": figures,
        "inputs": [
            {"path": str(bundle.path), "sha256": bundle.sha256, "rows": len(bundle.rows)}
            for bundle in bundles
        ],
        "limitations": [
            "docs 4-19 are reused exploratory data, not fresh confirmation",
            "document is confounded with physical GPU/run shard",
            "confidence intervals are pointwise rather than simultaneous",
            "no equivalence margin or equivalence test was specified",
            (
                "producer provenance omits baseline_profiler.py and the shell launcher; "
                "code identity is proven only for the hashed core implementation"
            ),
            (
                "8K shapes were warmed, but formal 8K measurements followed the entire "
                "formal 4K block, leaving possible cross-block thermal/clock drift"
            ),
        ],
    }
    summary_path = output_dir / "native_v_retention_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path = write_report(summary, output_dir)
    print(f"Validated {len(rows)} rows from {len(bundles)} bundle(s).")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
