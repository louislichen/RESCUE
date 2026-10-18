"""Measure the Native-V retention/Bridge-V replacement dose-response curve.

The target verifier and Draft Q/K projections remain native.  Only eligible
positions in the final Draft layer are varied: a recent suffix is retained as
Native V, while the older prefix is produced by the frozen Target-V mapper and
skips the corresponding native ``v_proj`` rows.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

from attn_bridge_experiment import (
    POLICY_SPECS,
    _sha256_file,
    build_eval_provenance,
    load_model,
    load_texts,
    make_input,
    seed_everything,
    validate_compute_replacement,
)


BASE_POLICY_NAME = "attention_bridge_value_bypass_last_a100"
BASE_POLICY = POLICY_SPECS[BASE_POLICY_NAME]
SHADOW_POLICY_NAME = "attention_bridge_value_shadow_last_a100"
SHADOW_POLICY = POLICY_SPECS[SHADOW_POLICY_NAME]
NATIVE_POLICY_NAME = "attention_native"
NATIVE_POLICY = POLICY_SPECS[NATIVE_POLICY_NAME]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--draft_model_path", required=True)
    parser.add_argument("--device_map", default="auto", choices=["auto", "balanced"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source_file", required=True)
    parser.add_argument("--sample_start", type=int, default=4)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--lengths", nargs="+", type=int, default=[4096, 8192])
    parser.add_argument(
        "--retention_percentages",
        nargs="+",
        type=float,
        default=[0.0, 12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5, 100.0],
    )
    parser.add_argument("--max_gen_len", type=int, default=128)
    parser.add_argument("--tree_nodes", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--cache_top_k", type=int, default=32)
    parser.add_argument("--retrieve_every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=542)
    parser.add_argument("--timing_repeats", type=int, default=3)
    parser.add_argument("--warmup_gen_len", type=int, default=128)
    parser.add_argument("--exact_target_attention", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.samples <= 0 or args.timing_repeats <= 0 or args.warmup_gen_len <= 0:
        raise ValueError("samples, timing_repeats, and warmup_gen_len must be positive")
    if not args.lengths or any(length <= 1 for length in args.lengths):
        raise ValueError("Every context length must exceed one token")
    if not args.retention_percentages:
        raise ValueError("At least one Native-V retention percentage is required")
    if any(percent < 0 or percent > 100 for percent in args.retention_percentages):
        raise ValueError("Native-V retention percentages must be in [0, 100]")
    basis_points = [retention_basis_points(percent) for percent in args.retention_percentages]
    if len(set(basis_points)) != len(basis_points):
        raise ValueError("Native-V retention percentages must be unique")
    args.retention_percentages = sorted(args.retention_percentages)
    return args


def retention_basis_points(retention_percent: float) -> int:
    basis_points = int(round(float(retention_percent) * 100.0))
    if abs(float(retention_percent) - basis_points / 100.0) > 1.0e-9:
        raise ValueError("Retention percentages support at most two decimal places")
    return basis_points


def retained_native_rows(input_tokens: int, retention_percent: float) -> int:
    """Round half-up over the mapper-eligible positions 1..N-1."""

    eligible = int(input_tokens) - 1
    basis_points = retention_basis_points(retention_percent)
    return (eligible * basis_points + 5000) // 10000


def configure_bridge(
    model,
    args,
    input_tokens: int,
    retention_percent: float,
    policy_spec=None,
) -> int:
    recent_native = retained_native_rows(input_tokens, retention_percent)
    policy_spec = BASE_POLICY if policy_spec is None else policy_spec
    model.attn_bridge_config = {
        **policy_spec,
        "recent_native": recent_native,
        "collect_diagnostics": False,
        "collect_geometry": False,
    }
    return recent_native


def save_atomic(path, args, torch, records, provenance):
    payload = {
        "schema_version": 1,
        "experiment": "native_v_retention_sweep",
        "config": vars(args),
        "sweep_spec": {
            "base_policy": BASE_POLICY_NAME,
            "policy_spec": BASE_POLICY,
            "retention_denominator": (
                "mapper-eligible final-layer prompt positions 1..N-1"
            ),
            "forced_native_positions": (
                "prompt position 0 and the unverified root remain native"
            ),
            "selection_rule": "retain the most recent eligible positions",
            "working_cache_retrieval": "target-attention-guided",
            "semantic_controls": (
                "one full-shape shadow per ratio/document/context and one "
                "attention-native anchor at 100%"
            ),
        },
        "metric_definitions": {
            "avg_accept_length": "mean accepted proposal length per verification round",
            "draft_prefill_time_ms": (
                "CUDA time for initial Draft transformer prefill; Bridge prepare is separate"
            ),
            "bridge_prepare_time_ms": (
                "CUDA-stream time for Target-V gather, mapper, and mapped-V materialization"
            ),
            "wall_tokens_per_sec": "scored generated tokens divided by end-to-end wall time",
            "timing_repeat": "technical repeat; not an independent statistical sample",
        },
        "provenance": provenance,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "raw_runs": records,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def run(args):
    import torch
    import classic.model_classic as engine_module
    from baseline_profiler import RuntimeProfiler
    from classic.attn_bridge import AttnBridgeMapper, summarize_bridge_runtime

    seed_everything(args.seed, torch)
    model = load_model(args, torch)
    draft_device = model.draft_model.model.embed_tokens.weight.device
    model.attn_bridge_mapper = AttnBridgeMapper.from_checkpoint(
        args.checkpoint, device=draft_device, dtype=torch.float16
    )
    texts = load_texts(args.source_file, args.sample_start, args.samples)
    tokenizer = model.tokenizer
    device = next(model.parameters()).device
    profiler = RuntimeProfiler(torch, engine_module, model, "classic")
    provenance = build_eval_provenance(args, model)
    provenance["implementation_sha256"]["native_v_retention_sweep.py"] = (
        _sha256_file(Path(__file__).resolve())
    )
    provenance["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")

    if os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as handle:
            saved = json.load(handle)
        expected = {key: value for key, value in vars(args).items() if key != "output"}
        observed = {
            key: value
            for key, value in saved.get("config", {}).items()
            if key != "output"
        }
        if observed != expected:
            raise RuntimeError("Refusing to resume a sweep with a different configuration")
        if saved.get("sweep_spec", {}).get("policy_spec") != BASE_POLICY:
            raise RuntimeError("Refusing to resume with a different policy specification")
        if saved.get("provenance") != provenance:
            raise RuntimeError("Refusing to resume with different code, checkpoint, data, or GPU")
        records = saved.get("raw_runs", [])
    else:
        records = []

    completed = {
        (
            str(row.get("record_kind", "measured_bypass")),
            int(row["length_bin_tokens"]),
            int(row["native_v_retention_basis_points"]),
            int(row["sample_index"]),
            int(row["timing_repeat"]),
        )
        for row in records
    }

    # Every retention point changes the complement GEMM M-shape, so warm every
    # point at each context length before collecting latency or throughput.
    for length in args.lengths:
        warmup = make_input(tokenizer, texts[0], length, torch, device)
        for percent in args.retention_percentages:
            configure_bridge(model, args, int(warmup.shape[1]), percent)
            profiler.reset()
            model.spgenerate(
                warmup,
                temperature=0,
                max_new_tokens=args.warmup_gen_len,
                nodes=args.tree_nodes,
                output_result_line=False,
                verbose=False,
                return_output_ids=False,
                use_specextend=True,
                retrieval_chunk_size=args.chunk_size,
                retrieve_top_k=args.cache_top_k,
                retrieve_every_n_steps=args.retrieve_every,
                retrieval_policy=BASE_POLICY["retrieval_policy"],
                force_flash_prefill=True,
                exact_target_attention=args.exact_target_attention,
                retrieval_verbose=False,
            )
            torch.cuda.synchronize()
            profiler.finish()

    measured_total = (
        len(args.lengths)
        * args.samples
        * args.timing_repeats
        * len(args.retention_percentages)
    )
    shadow_total = (
        len(args.lengths) * args.samples * len(args.retention_percentages)
    )
    anchor_total = len(args.lengths) * args.samples
    total = measured_total + shadow_total + anchor_total
    job = 0

    def execute_record(
        *,
        input_ids,
        length,
        sample_index,
        percent,
        timing_repeat,
        execution_order,
        record_kind,
        policy_name,
        policy_spec,
    ):
        nonlocal job
        job += 1
        input_tokens = int(input_ids.shape[1])
        eligible_rows = input_tokens - 1
        percent_basis_points = retention_basis_points(percent)
        key = (
            record_kind,
            int(length),
            percent_basis_points,
            int(sample_index),
            int(timing_repeat),
        )
        if key in completed:
            return
        recent_native = configure_bridge(
            model,
            args,
            input_tokens,
            percent,
            policy_spec=policy_spec,
        )
        profiler.reset()
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        result = model.spgenerate(
            input_ids,
            temperature=0,
            max_new_tokens=args.max_gen_len,
            nodes=args.tree_nodes,
            output_result_line=False,
            verbose=False,
            return_output_ids=True,
            use_specextend=True,
            retrieval_chunk_size=args.chunk_size,
            retrieve_top_k=args.cache_top_k,
            retrieve_every_n_steps=args.retrieve_every,
            retrieval_policy=policy_spec["retrieval_policy"],
            force_flash_prefill=True,
            exact_target_attention=args.exact_target_attention,
            retrieval_verbose=False,
        )
        torch.cuda.synchronize()
        wall_time = time.perf_counter() - wall_start
        diagnostics = profiler.finish()
        bridge = summarize_bridge_runtime(model)
        validation = validate_compute_replacement(
            model,
            policy_spec,
            SimpleNamespace(recent_native=recent_native),
            input_tokens,
            bridge,
        )
        actual_percent = 100.0 * recent_native / eligible_rows
        row = {
            "engine": "classic",
            "record_kind": record_kind,
            "formal_timing_eligible": record_kind == "measured_bypass",
            "policy": policy_name,
            "base_policy": policy_name,
            "policy_config": {
                **policy_spec,
                "recent_native": recent_native,
            },
            "input_tokens": input_tokens,
            "length_bin_tokens": int(length),
            "length_label": (
                f"{length // 1024}K" if length % 1024 == 0 else str(length)
            ),
            "sample_index": sample_index,
            "timing_repeat": timing_repeat,
            "execution_order": execution_order,
            "native_v_retention_percent": float(percent),
            "native_v_retention_basis_points": percent_basis_points,
            "actual_native_v_retention_percent": actual_percent,
            "eligible_native_v_rows": eligible_rows,
            "native_v_retained_rows": recent_native,
            "bridge_v_mapped_rows": eligible_rows - recent_native,
            "forced_native_last_layer_rows": 2,
            "actual_last_layer_native_percent": (
                100.0 * (recent_native + 2) / (input_tokens + 1)
            ),
            "actual_global_native_v_percent": (
                100.0
                * (
                    len(model.draft_model.model.layers) * (input_tokens + 1)
                    - (eligible_rows - recent_native)
                )
                / (len(model.draft_model.model.layers) * (input_tokens + 1))
            ),
            "cache_top_k": args.cache_top_k,
            "chunk_size": args.chunk_size,
            "seed": args.seed,
            "nodes": args.tree_nodes,
            "wall_timing_is_diagnostic": False,
            "compute_replacement_validation": validation,
            **result,
            "wall_time": wall_time,
            "wall_tokens_per_sec": result["total_generated"] / wall_time,
            "draft_prefill_plus_bridge_prepare_time_ms": (
                bridge["draft_prefill_time_ms"]
                + bridge["bridge_prepare_time_ms"]
            ),
            **diagnostics,
            **bridge,
        }
        records.append(row)
        completed.add(key)
        save_atomic(args.output, args, torch, records, provenance)
        print(
            f"[{job}/{total}] {record_kind} {row['length_label']} "
            f"retain={percent:g}% actual={actual_percent:.3f}% "
            f"sample={sample_index} repeat={timing_repeat} "
            f"accept={result['avg_accept_length']:.3f} "
            f"prefill={bridge['draft_prefill_time_ms']:.3f}ms "
            f"tok/s={row['wall_tokens_per_sec']:.2f}",
            flush=True,
        )

    for length in args.lengths:
        for offset, text in enumerate(texts):
            sample_index = args.sample_start + offset
            input_ids = make_input(tokenizer, text, length, torch, device)
            for timing_repeat in range(args.timing_repeats):
                order = list(args.retention_percentages)
                random.Random(
                    args.seed
                    + length * 1000
                    + sample_index
                    + timing_repeat * 1_000_003
                ).shuffle(order)
                for execution_order, percent in enumerate(order):
                    percent_basis_points = retention_basis_points(percent)
                    execute_record(
                        input_ids=input_ids,
                        length=length,
                        sample_index=sample_index,
                        percent=percent,
                        timing_repeat=timing_repeat,
                        execution_order=execution_order,
                        record_kind="measured_bypass",
                        policy_name=(
                            f"native_v_retain_bp_{percent_basis_points:05d}"
                        ),
                        policy_spec=BASE_POLICY,
                    )

            # One same-effective-V full-shape shadow per ratio separates mapped
            # V quality from complement-GEMM numerical path effects.
            shadow_order = list(args.retention_percentages)
            random.Random(args.seed + length * 1000 + sample_index + 9_000_001).shuffle(
                shadow_order
            )
            for execution_order, percent in enumerate(shadow_order):
                percent_basis_points = retention_basis_points(percent)
                execute_record(
                    input_ids=input_ids,
                    length=length,
                    sample_index=sample_index,
                    percent=percent,
                    timing_repeat=0,
                    execution_order=execution_order,
                    record_kind="semantic_shadow",
                    policy_name=f"shadow_retain_bp_{percent_basis_points:05d}",
                    policy_spec=SHADOW_POLICY,
                )

            # At 100%, the bypass path maps zero rows and must reduce exactly to
            # the ordinary attention-native path.
            execute_record(
                input_ids=input_ids,
                length=length,
                sample_index=sample_index,
                percent=100.0,
                timing_repeat=0,
                execution_order=0,
                record_kind="native_anchor",
                policy_name=NATIVE_POLICY_NAME,
                policy_spec=NATIVE_POLICY,
            )


if __name__ == "__main__":
    run(parse_args())
