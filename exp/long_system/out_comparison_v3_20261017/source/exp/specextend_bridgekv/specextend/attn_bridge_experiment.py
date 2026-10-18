"""Collect, train, and evaluate the AttnBridgeKV pilot on one GPU.

The pilot freezes both language models.  It learns a low-rank mapper from the
target model's committed-prefix last-layer V cache to draft-native K/V, then
tests whether verifier-attention-selected bridge blocks improve draft quality
under the same working-cache budget.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path


POLICY_SPECS = {
    "recent_native": {
        "retrieval_policy": "recent", "alpha": 0.0, "layers": "all", "source_mode": "aligned",
    },
    "attention_native": {
        "retrieval_policy": "attention", "alpha": 0.0, "layers": "all", "source_mode": "aligned",
    },
    "recent_bridge_a50": {
        "retrieval_policy": "recent", "alpha": 0.5, "layers": "all", "source_mode": "aligned",
    },
    "recent_bridge_a25": {
        "retrieval_policy": "recent", "alpha": 0.25, "layers": "all", "source_mode": "aligned",
    },
    "attention_bridge_last_a50": {
        "retrieval_policy": "attention", "alpha": 0.5, "layers": "last", "source_mode": "aligned",
    },
    "attention_bridge_a50": {
        "retrieval_policy": "attention", "alpha": 0.5, "layers": "all", "source_mode": "aligned",
    },
    "attention_bridge_a10": {
        "retrieval_policy": "attention", "alpha": 0.1, "layers": "all", "source_mode": "aligned",
    },
    "attention_bridge_a25": {
        "retrieval_policy": "attention", "alpha": 0.25, "layers": "all", "source_mode": "aligned",
    },
    "attention_bridge_value_a25": {
        "retrieval_policy": "attention", "alpha": 0.25, "layers": "all",
        "source_mode": "aligned", "components": "value",
    },
    "attention_bridge_value_a100": {
        "retrieval_policy": "attention", "alpha": 1.0, "layers": "last",
        "source_mode": "aligned", "components": "value",
        "compute_mode": "cache_overwrite",
    },
    "attention_bridge_value_shadow_last_a100": {
        "retrieval_policy": "attention", "alpha": 1.0, "layers": "last",
        "source_mode": "aligned", "components": "value",
        "compute_mode": "prefill_value_shadow",
    },
    "attention_bridge_value_bypass_last_a100": {
        "retrieval_policy": "attention", "alpha": 1.0, "layers": "last",
        "source_mode": "aligned", "components": "value",
        "compute_mode": "prefill_value_bypass",
    },
    "recent_bridge_value_bypass_last_a100": {
        "retrieval_policy": "recent", "alpha": 1.0, "layers": "last",
        "source_mode": "aligned", "components": "value",
        "compute_mode": "prefill_value_bypass",
    },
    "attention_bridge_value_shadow_all_a100": {
        "retrieval_policy": "attention", "alpha": 1.0, "layers": "all",
        "source_mode": "aligned", "components": "value",
        "compute_mode": "prefill_value_shadow",
    },
    "attention_bridge_value_bypass_all_a100": {
        "retrieval_policy": "attention", "alpha": 1.0, "layers": "all",
        "source_mode": "aligned", "components": "value",
        "compute_mode": "prefill_value_bypass",
    },
    "attention_bridge_key_a25": {
        "retrieval_policy": "attention", "alpha": 0.25, "layers": "all",
        "source_mode": "aligned", "components": "key",
    },
    "attention_bridge_v25_k05": {
        "retrieval_policy": "attention", "alpha": 0.25, "alpha_key": 0.05,
        "alpha_value": 0.25, "layers": "all", "source_mode": "aligned",
    },
    "attention_bridge_v25_k10": {
        "retrieval_policy": "attention", "alpha": 0.25, "alpha_key": 0.10,
        "alpha_value": 0.25, "layers": "all", "source_mode": "aligned",
    },
    "attention_bridge_v25_k50": {
        "retrieval_policy": "attention", "alpha": 0.25, "alpha_key": 0.50,
        "alpha_value": 0.25, "layers": "all", "source_mode": "aligned",
    },
    "attention_bridge_v25_k25_qk_rescue": {
        "retrieval_policy": "attention", "alpha": 0.25, "alpha_key": 0.25,
        "alpha_value": 0.25, "layers": "all", "source_mode": "aligned",
        "geometry_rescue": True,
    },
    "attention_bridge_a100": {
        "retrieval_policy": "attention", "alpha": 1.0, "layers": "all", "source_mode": "aligned",
    },
    "attention_bridge_shuffle_a50": {
        "retrieval_policy": "attention", "alpha": 0.5, "layers": "all", "source_mode": "shuffle",
    },
    "attention_bridge_shuffle_a25": {
        "retrieval_policy": "attention", "alpha": 0.25, "layers": "all", "source_mode": "shuffle",
    },
}


def add_model_args(parser):
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--draft_model_path", required=True)
    parser.add_argument("--device_map", default="auto", choices=["auto", "balanced"])


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect")
    add_model_args(collect)
    collect.add_argument("--source_file", required=True)
    collect.add_argument("--sample_start", type=int, default=0)
    collect.add_argument("--samples", type=int, default=4)
    collect.add_argument("--input_tokens", type=int, default=4096)
    collect.add_argument("--max_gen_len", type=int, default=8)
    collect.add_argument("--tree_nodes", type=int, default=64)
    collect.add_argument("--chunk_size", type=int, default=32)
    collect.add_argument("--cache_top_k", type=int, default=32)
    collect.add_argument("--retrieve_every", type=int, default=4)
    collect.add_argument("--max_pairs", type=int, default=12000)
    collect.add_argument("--pairs_per_event", type=int, default=1024)
    collect.add_argument("--seed", type=int, default=42)
    collect.add_argument("--output", required=True)

    train = sub.add_parser("train")
    train.add_argument("--pairs", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--history", required=True)
    train.add_argument("--rank", type=int, default=64)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch_size", type=int, default=256)
    train.add_argument("--learning_rate", type=float, default=1e-3)
    train.add_argument("--weight_decay", type=float, default=1e-4)
    train.add_argument("--nmse_weight", type=float, default=0.25)
    train.add_argument("--val_fraction", type=float, default=0.25)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cuda")

    evaluate = sub.add_parser("evaluate")
    add_model_args(evaluate)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--source_file", required=True)
    evaluate.add_argument("--sample_start", type=int, default=0)
    evaluate.add_argument("--samples", type=int, default=5)
    evaluate.add_argument("--lengths", nargs="+", type=int, default=[4096])
    evaluate.add_argument("--max_gen_len", type=int, default=128)
    evaluate.add_argument("--tree_nodes", type=int, default=64)
    evaluate.add_argument("--chunk_size", type=int, default=32)
    evaluate.add_argument("--cache_top_k", type=int, default=32)
    evaluate.add_argument("--retrieve_every", type=int, default=4)
    evaluate.add_argument("--recent_native", type=int, default=128)
    evaluate.add_argument("--policies", nargs="+", choices=sorted(POLICY_SPECS),
                          default=[
                              "recent_native", "attention_native", "recent_bridge_a50",
                              "attention_bridge_last_a50", "attention_bridge_a50",
                              "attention_bridge_a100", "attention_bridge_shuffle_a50",
                          ])
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument(
        "--timing_repeats", type=int, default=1,
        help="Repeat paired policy timings; repeats are not independent samples",
    )
    evaluate.add_argument(
        "--warmup_gen_len", type=int, default=16,
        help="Generated tokens per policy warm-up before measured runs",
    )
    evaluate.add_argument("--bridge_diagnostics", action="store_true")
    evaluate.add_argument(
        "--geometry_diagnostics",
        action="store_true",
        help="Replay native and bridged Draft QK attention for mechanism diagnostics; disables fair timing",
    )
    evaluate.add_argument("--geometry_top_k", type=int, default=8)
    evaluate.add_argument("--geometry_max_queries", type=int, default=16)
    evaluate.add_argument("--geometry_max_events_per_layer", type=int, default=64)
    evaluate.add_argument(
        "--exact_target_attention",
        action="store_true",
        help="Use the exact target attention path instead of Flash/Hybrid kernels",
    )
    evaluate.add_argument("--output", required=True)

    reference = sub.add_parser("reference")
    add_model_args(reference)
    reference.add_argument("--source_file", required=True)
    reference.add_argument("--sample_start", type=int, default=4)
    reference.add_argument("--samples", type=int, default=8)
    reference.add_argument("--lengths", nargs="+", type=int, default=[4096, 8192])
    reference.add_argument("--max_gen_len", type=int, default=128)
    reference.add_argument("--seed", type=int, default=42)
    reference.add_argument("--output", required=True)
    return parser.parse_args()


def seed_everything(seed, torch):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_texts(path, start, count):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if isinstance(value.get("text"), str):
                rows.append(value["text"])
    end = start + count
    if len(rows) < end:
        raise RuntimeError(f"Need rows [{start}, {end}), found {len(rows)} in {path}")
    return rows[start:end]


def make_input(tokenizer, text, target_tokens, torch, device):
    if "### Text:" in text and "### Summary:" in text:
        from baseline_profiler import exact_input
        return exact_input(tokenizer, text, target_tokens, torch, device)
    ids = tokenizer.encode(text, add_special_tokens=True)
    if len(ids) < target_tokens:
        raise ValueError({"target_tokens": target_tokens, "available_tokens": len(ids)})
    return torch.tensor([ids[:target_tokens]], dtype=torch.long, device=device)


def load_model(args, torch):
    from classic.model_classic import SPModel
    return SPModel.from_pretrained(
        base_model_path=args.base_model_path,
        draft_model_path=args.draft_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=args.device_map,
    ).eval()


def collect_pairs(args):
    import torch
    from classic.attn_bridge import BridgePairCollector

    seed_everything(args.seed, torch)
    model = load_model(args, torch)
    device = next(model.parameters()).device
    texts = load_texts(args.source_file, args.sample_start, args.samples)
    collector = BridgePairCollector(
        args.max_pairs,
        args.pairs_per_event,
        max_pairs_per_group=max(1, args.max_pairs // args.samples),
    )
    model.bridge_collector = collector

    for offset, text in enumerate(texts):
        sample_index = args.sample_start + offset
        collector.set_group(sample_index)
        input_ids = make_input(model.tokenizer, text, args.input_tokens, torch, device)
        result = model.spgenerate(
            input_ids,
            temperature=0,
            max_new_tokens=args.max_gen_len,
            nodes=args.tree_nodes,
            output_result_line=False,
            verbose=False,
            use_specextend=True,
            retrieval_chunk_size=args.chunk_size,
            retrieve_top_k=args.cache_top_k,
            retrieve_every_n_steps=args.retrieve_every,
            retrieval_policy="attention",
            force_flash_prefill=True,
            retrieval_verbose=False,
        )
        collector.save_atomic(args.output, {
            "source_file": args.source_file,
            "input_tokens": args.input_tokens,
            "sample_start": args.sample_start,
            "samples": args.samples,
            "seed": args.seed,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        })
        print(
            f"collect sample={sample_index} pairs={collector.pair_count} "
            f"accept={result['avg_accept_length']:.3f}",
            flush=True,
        )


def bridge_loss(mapper, source, target, nmse_weight):
    import torch
    import torch.nn.functional as F

    prediction = torch.stack(mapper(source), dim=1)
    cosine = 1.0 - F.cosine_similarity(prediction, target, dim=-1).mean()
    nmse = (
        (prediction - target).pow(2).mean(dim=-1)
        / target.pow(2).mean(dim=-1).clamp_min(1e-12)
    ).mean()
    return cosine + nmse_weight * nmse, cosine, nmse


def evaluate_mapper(mapper, source, target, indices, batch_size, nmse_weight, device):
    import torch

    totals = torch.zeros(3, device=device)
    count = 0
    mapper.eval()
    with torch.no_grad():
        for start in range(0, indices.numel(), batch_size):
            batch = indices[start:start + batch_size]
            x = source.index_select(0, batch).to(device=device, dtype=torch.float32)
            y = target.index_select(0, batch).to(device=device, dtype=torch.float32)
            losses = bridge_loss(mapper, x, y, nmse_weight)
            size = batch.numel()
            totals += torch.stack([value.detach() for value in losses]) * size
            count += size
    return [float(value.item() / max(count, 1)) for value in totals]


def train_mapper(args):
    import torch
    from classic.attn_bridge import AttnBridgeMapper

    seed_everything(args.seed, torch)
    torch.set_float32_matmul_precision("high")
    payload = torch.load(args.pairs, map_location="cpu")
    source = payload["source"]
    target = payload["target"]
    groups = payload["group_ids"]
    metadata = payload["metadata"]
    unique_groups = sorted(set(int(value) for value in groups.tolist()))
    if len(unique_groups) > 1:
        val_count = max(1, round(len(unique_groups) * args.val_fraction))
        val_groups = set(unique_groups[-val_count:])
        val_mask = torch.tensor([int(value) in val_groups for value in groups.tolist()])
    else:
        generator = torch.Generator().manual_seed(args.seed)
        permutation = torch.randperm(source.shape[0], generator=generator)
        val_size = max(1, round(source.shape[0] * args.val_fraction))
        val_mask = torch.zeros(source.shape[0], dtype=torch.bool)
        val_mask[permutation[:val_size]] = True
        val_groups = {unique_groups[0]} if unique_groups else set()
    train_indices = torch.nonzero(~val_mask, as_tuple=False).flatten()
    val_indices = torch.nonzero(val_mask, as_tuple=False).flatten()

    device = torch.device(args.device)
    mapper = AttnBridgeMapper(
        source_dim=int(metadata["source_dim"]),
        draft_dim=int(metadata["draft_dim"]),
        num_draft_layers=int(metadata["num_draft_layers"]),
        rank=args.rank,
    ).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        mapper.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = []
    best_val = float("inf")
    best_state = None
    generator = torch.Generator().manual_seed(args.seed)

    for epoch in range(1, args.epochs + 1):
        mapper.train()
        shuffled = train_indices[torch.randperm(train_indices.numel(), generator=generator)]
        totals = torch.zeros(3, device=device)
        count = 0
        for start in range(0, shuffled.numel(), args.batch_size):
            batch = shuffled[start:start + args.batch_size]
            x = source.index_select(0, batch).to(device=device, dtype=torch.float32)
            y = target.index_select(0, batch).to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss, cosine, nmse = bridge_loss(mapper, x, y, args.nmse_weight)
            loss.backward()
            optimizer.step()
            size = batch.numel()
            totals += torch.stack([loss.detach(), cosine.detach(), nmse.detach()]) * size
            count += size
        train_values = [float(value.item() / max(count, 1)) for value in totals]
        val_values = evaluate_mapper(
            mapper, source, target, val_indices, args.batch_size,
            args.nmse_weight, device,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_values[0],
            "train_cosine_loss": train_values[1],
            "train_nmse": train_values[2],
            "val_loss": val_values[0],
            "val_cosine_loss": val_values[1],
            "val_nmse": val_values[2],
        }
        history.append(row)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in mapper.state_dict().items()}
        print(
            f"epoch={epoch:02d} train={row['train_loss']:.5f} "
            f"val={row['val_loss']:.5f} val_nmse={row['val_nmse']:.5f}",
            flush=True,
        )

    mapper.load_state_dict(best_state)
    checkpoint = {
        "state_dict": {key: value.cpu() for key, value in mapper.state_dict().items()},
        "mapper_config": mapper.config_dict(),
        "training": {
            "pairs": args.pairs,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "nmse_weight": args.nmse_weight,
            "train_pairs": int(train_indices.numel()),
            "val_pairs": int(val_indices.numel()),
            "val_groups": sorted(val_groups),
            "best_val_loss": best_val,
        },
        "pair_metadata": metadata,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    temporary = args.output + ".tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, args.output)
    history_payload = {
        "config": vars(args),
        "mapper_config": mapper.config_dict(),
        "pair_metadata": metadata,
        "history": history,
        "best_val_loss": best_val,
    }
    Path(args.history).parent.mkdir(parents=True, exist_ok=True)
    temporary = args.history + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(history_payload, handle, indent=2)
    os.replace(temporary, args.history)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_eval_provenance(args, model):
    source_root = Path(__file__).resolve().parent
    implementation_files = {
        "attn_bridge_experiment.py": Path(__file__).resolve(),
        "classic/attn_bridge.py": source_root / "classic" / "attn_bridge.py",
        "classic/model_classic.py": source_root / "classic" / "model_classic.py",
        "classic/modeling_llama_kv_draft.py": (
            source_root / "classic" / "modeling_llama_kv_draft.py"
        ),
    }
    return {
        "implementation_sha256": {
            name: _sha256_file(path) for name, path in implementation_files.items()
        },
        "checkpoint_sha256": _sha256_file(args.checkpoint),
        "source_file_sha256": _sha256_file(args.source_file),
        "device_layout": {
            "target_last_layer": str(
                next(
                    model.base_model.model.layers[-1].self_attn.parameters()
                ).device
            ),
            "mapper": str(next(model.attn_bridge_mapper.parameters()).device),
            "draft_embedding": str(
                model.draft_model.model.embed_tokens.weight.device
            ),
        },
    }


def validate_compute_replacement(model, spec, args, input_tokens, bridge):
    mode = str(spec.get("compute_mode", "cache_overwrite"))
    if mode not in {"prefill_value_shadow", "prefill_value_bypass"}:
        return None

    draft_layers = list(model.draft_model.model.layers)
    layer_count = len(draft_layers)
    active_count = 1 if spec.get("layers", "all") == "last" else layer_count
    query_tokens = int(input_tokens) + 1
    recent_native = max(0, int(args.recent_native))
    mapped_tokens = max(0, int(input_tokens) - recent_native - 1)
    v_proj = draft_layers[0].self_attn.v_proj
    v_macs_per_row = int(v_proj.in_features * v_proj.out_features)
    mapper = model.attn_bridge_mapper
    mapper_macs = mapped_tokens * (
        int(mapper.source_dim * mapper.rank)
        + active_count * int(mapper.rank * mapper.draft_dim)
    )
    total_v_rows = layer_count * query_tokens
    mapped_rows = active_count * mapped_tokens
    is_bypass = mode == "prefill_value_bypass"
    expected = {
        "bridge_map_events": 1 if mapped_tokens else 0,
        "bridge_prepare_events": 1 if mapped_tokens else 0,
        "draft_prefill_events": 1,
        "bridge_mapped_tokens": mapped_tokens,
        "bridge_value_bypass_events": 1 if is_bypass and mapped_tokens else 0,
        "bridge_value_shadow_events": 1 if not is_bypass and mapped_tokens else 0,
        "draft_prefill_k_projected_tokens": layer_count * query_tokens,
        "draft_prefill_v_projected_tokens": (
            total_v_rows - mapped_rows if is_bypass else total_v_rows
        ),
        "draft_prefill_v_skipped_tokens": mapped_rows if is_bypass else 0,
        "draft_prefill_v_shadow_tokens": 0 if is_bypass else mapped_rows,
        "draft_prefill_v_projection_macs": (
            total_v_rows - mapped_rows if is_bypass else total_v_rows
        ) * v_macs_per_row,
        "draft_prefill_v_projection_saved_macs": (
            mapped_rows * v_macs_per_row if is_bypass else 0
        ),
        "draft_prefill_v_shadow_macs": (
            0 if is_bypass else mapped_rows * v_macs_per_row
        ),
        "bridge_prefill_mapper_macs": mapper_macs,
        "draft_prefill_net_projection_macs_saved": (
            mapped_rows * v_macs_per_row if is_bypass else 0
        ) - mapper_macs,
    }
    mismatches = {
        key: {"expected": value, "observed": bridge.get(key)}
        for key, value in expected.items()
        if bridge.get(key) != value
    }

    by_layer = {
        int(row["layer"]): row for row in bridge["draft_projection_by_layer"]
    }
    active_layers = (
        {layer_count - 1}
        if spec.get("layers", "all") == "last"
        else set(range(layer_count))
    )
    for layer_index in range(layer_count):
        row = by_layer.get(layer_index, {})
        active = layer_index in active_layers
        layer_mapped = mapped_tokens if active else 0
        layer_expected = {
            "prefill_k_projected_tokens": query_tokens,
            "prefill_v_projected_tokens": (
                query_tokens - layer_mapped if is_bypass else query_tokens
            ),
            "prefill_v_skipped_tokens": layer_mapped if is_bypass else 0,
            "prefill_v_shadow_tokens": (
                0 if is_bypass else layer_mapped
            ),
        }
        for field, value in layer_expected.items():
            if row.get(field) != value:
                mismatches[f"layer_{layer_index}.{field}"] = {
                    "expected": value,
                    "observed": row.get(field),
                }
    if mismatches:
        raise RuntimeError(
            "Bridge compute-replacement invariant failure: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        "status": "passed",
        "query_tokens": query_tokens,
        "mapped_tokens": mapped_tokens,
        "active_layers": active_count,
        "v_macs_per_row": v_macs_per_row,
        "expected": expected,
    }


def save_eval_atomic(path, args, torch, records, provenance):
    payload = {
        "config": vars(args),
        "policy_specs": {name: POLICY_SPECS[name] for name in args.policies},
        "provenance": provenance,
        "metric_definitions": {
            "bridge_map_time_ms": "CUDA time for the dense Bridge mapper forward only",
            "bridge_prepare_time_ms": "CUDA-stream time for full inline Bridge preparation: Target-V gather, mapper forward, and mapped-V materialization",
            "bridge_source_bytes": "Target-V bytes read by the bridge mapper; counts repeated refreshes",
            "bridge_materialized_bytes": "Mapped Draft K/V bytes materialized before blending; counts repeated refreshes",
            "draft_prefill_time_ms": "CUDA time for the initial Draft transformer prefill; mapper preparation is reported separately",
            "draft_prefill_v_projected_tokens": "Actual token rows submitted to Draft v_proj during initial prefill, summed over layers",
            "draft_prefill_v_skipped_tokens": "Initial-prefill Draft V rows replaced inline and therefore not submitted to v_proj",
            "draft_prefill_v_shadow_tokens": "Mapped rows included in the shadow control's single full-shape native v_proj before overwrite; these rows are a subset of projected_tokens, not an additional projection count",
            "draft_prefill_v_projection_saved_macs": "Dense Draft v_proj MACs skipped from actual input shapes",
            "bridge_prefill_mapper_macs": "Dense encoder plus active value-decoder MACs used to build prefill mapped V",
            "draft_prefill_net_projection_macs_saved": "Skipped Draft V MACs minus prefill mapper MACs; excludes indexing, LayerNorm, and SiLU",
            "draft_projection_by_layer": "Actual projected, skipped, and shadow token-row counts for each Draft layer",
            "timing_repeat": "Technical timing repetition within one document; not an independent statistical sample",
            "bridge_mean_kv_cosine": "Cosine similarity of mapped and native Draft K/V on mapped positions",
            "bridge_mean_kv_nmse": "Normalized MSE of mapped versus native Draft K/V on mapped positions",
            "bridge_geometry_qk_logit_relative_rmse": "Relative RMSE between native-K and effective QK logits under the same post-RoPE Draft Q; effective means raw blended-K normally and post-rescue logits for oracle rows",
            "bridge_geometry_attention_kl": "Mean KL(native Draft attention || effective attention) under teacher-forced identical Q; oracle rows are measured after rescue",
            "bridge_geometry_topk_jaccard": "Mean Jaccard overlap between native-K and effective attention top-k sets",
            "bridge_geometry_argmax_flip_rate": "Fraction of Draft query/head rows whose effective attention argmax differs from native-K attention",
            "bridge_geometry_query_projected_key_mae": "Mean absolute native-versus-effective QK-logit error over valid positions",
            "bridge_geometry_by_layer": "The same teacher-forced geometry metrics aggregated separately for each Draft layer",
            "bridge_geometry_post_rescue_events": "Number of sampled geometry events measured from effective post-rescue logits",
            "bridge_geometry_rescue_events": "Attention forwards where the oracle restored native-K logits on the stable prefix while retaining mapped V",
            "wall_timing_is_diagnostic": "True when bridge cosine/NMSE diagnostics, geometry replay, or oracle rescue adds timed work; exclude this row from formal throughput claims",
            "output_token_ids": "Greedy output IDs used for bit-exact equality checks across policies",
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "raw_runs": records,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def evaluate_bridge(args):
    import torch
    import classic.model_classic as engine_module
    from baseline_profiler import RuntimeProfiler
    from classic.attn_bridge import AttnBridgeMapper, summarize_bridge_runtime

    seed_everything(args.seed, torch)
    if args.timing_repeats <= 0 or args.warmup_gen_len <= 0:
        raise ValueError("timing_repeats and warmup_gen_len must be positive")
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
    expected_policy_specs = {
        name: POLICY_SPECS[name] for name in args.policies
    }

    if os.path.exists(args.output):
        saved = json.load(open(args.output, encoding="utf-8"))
        expected = {key: value for key, value in vars(args).items() if key != "output"}
        observed = {key: value for key, value in saved.get("config", {}).items() if key != "output"}
        if observed != expected:
            raise RuntimeError("Refusing to mix evaluation runs with different configurations")
        if saved.get("policy_specs") != expected_policy_specs:
            raise RuntimeError("Refusing to resume with different policy specifications")
        if saved.get("provenance") != provenance:
            raise RuntimeError("Refusing to resume with different code, checkpoint, or data")
        records = saved.get("raw_runs", [])
    else:
        records = []
    completed = {
        (
            row["length_bin_tokens"],
            row["policy"],
            row["sample_index"],
            int(row.get("timing_repeat", 0)),
        )
        for row in records
    }

    # Warm every measured context shape and policy. The first longer generation
    # otherwise pays one-time kernel/autotuning work and can be an extreme
    # throughput outlier even after a short warm-up at another context length.
    for warm_length in args.lengths:
        warmup = make_input(tokenizer, texts[0], warm_length, torch, device)
        for policy in args.policies:
            spec = POLICY_SPECS[policy]
            model.attn_bridge_config = {
                **spec,
                "recent_native": args.recent_native,
                "collect_diagnostics": args.bridge_diagnostics,
                "collect_geometry": args.geometry_diagnostics,
                "geometry_top_k": args.geometry_top_k,
                "geometry_max_queries": args.geometry_max_queries,
                "geometry_max_events_per_layer": args.geometry_max_events_per_layer,
            }
            profiler.reset()
            model.spgenerate(
                warmup, temperature=0, max_new_tokens=args.warmup_gen_len,
                nodes=args.tree_nodes,
                output_result_line=False, verbose=False, return_output_ids=False,
                use_specextend=True, retrieval_chunk_size=args.chunk_size,
                retrieve_top_k=args.cache_top_k, retrieve_every_n_steps=args.retrieve_every,
                retrieval_policy=spec["retrieval_policy"], force_flash_prefill=True,
                exact_target_attention=args.exact_target_attention,
                retrieval_verbose=False,
            )
            torch.cuda.synchronize()
            profiler.finish()

    total = (
        len(args.lengths) * len(args.policies) * args.samples * args.timing_repeats
    )
    job = 0
    for length in args.lengths:
        for offset, text in enumerate(texts):
            sample_index = args.sample_start + offset
            input_ids = make_input(tokenizer, text, length, torch, device)
            for timing_repeat in range(args.timing_repeats):
                order = list(args.policies)
                random.Random(
                    args.seed + length * 1000 + sample_index
                    + timing_repeat * 1_000_003
                ).shuffle(order)
                for execution_order, policy in enumerate(order):
                    job += 1
                    key = (length, policy, sample_index, timing_repeat)
                    if key in completed:
                        continue
                    spec = POLICY_SPECS[policy]
                    model.attn_bridge_config = {
                        **spec,
                        "recent_native": args.recent_native,
                        "collect_diagnostics": args.bridge_diagnostics,
                        "collect_geometry": args.geometry_diagnostics,
                        "geometry_top_k": args.geometry_top_k,
                        "geometry_max_queries": args.geometry_max_queries,
                        "geometry_max_events_per_layer": args.geometry_max_events_per_layer,
                    }
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
                        retrieval_policy=spec["retrieval_policy"],
                        force_flash_prefill=True,
                        exact_target_attention=args.exact_target_attention,
                        retrieval_verbose=False,
                    )
                    torch.cuda.synchronize()
                    wall_time = time.perf_counter() - wall_start
                    diagnostics = profiler.finish()
                    bridge = summarize_bridge_runtime(model)
                    compute_validation = validate_compute_replacement(
                        model,
                        spec,
                        args,
                        int(input_ids.shape[1]),
                        bridge,
                    )
                    row = {
                        "engine": "classic",
                        "input_tokens": int(input_ids.shape[1]),
                        "length_bin_tokens": length,
                        "length_label": (
                            f"{length // 1024}K" if length % 1024 == 0 else str(length)
                        ),
                        "policy": policy,
                        "policy_config": {**spec, "recent_native": args.recent_native},
                        "sample_index": sample_index,
                        "timing_repeat": timing_repeat,
                        "cache_top_k": args.cache_top_k,
                        "chunk_size": args.chunk_size,
                        "seed": args.seed,
                        "nodes": args.tree_nodes,
                        "execution_order": execution_order,
                        "wall_timing_is_diagnostic": bool(
                            args.bridge_diagnostics
                            or args.geometry_diagnostics
                            or spec.get("geometry_rescue", False)
                        ),
                        "compute_replacement_validation": compute_validation,
                        **result,
                        "wall_time": wall_time,
                        "wall_tokens_per_sec": result["total_generated"] / wall_time,
                        **diagnostics,
                        **bridge,
                    }
                    records.append(row)
                    completed.add(key)
                    save_eval_atomic(
                        args.output, args, torch, records, provenance
                    )
                    print(
                        f"[{job}/{total}] {row['length_label']} {policy} "
                        f"sample={sample_index} repeat={timing_repeat} "
                        f"accept={result['avg_accept_length']:.3f} "
                        f"tok/s={row['wall_tokens_per_sec']:.2f} "
                        f"bridge_ms={bridge['bridge_map_time_ms']:.2f}",
                        flush=True,
                    )


def evaluate_target_reference(args):
    """Run canonical target-only greedy decoding for exactness diagnosis."""
    import torch

    seed_everything(args.seed, torch)
    model = load_model(args, torch)
    texts = load_texts(args.source_file, args.sample_start, args.samples)
    device = next(model.parameters()).device
    records = []
    total = len(args.lengths) * args.samples
    job = 0
    for length in args.lengths:
        for offset, text in enumerate(texts):
            job += 1
            sample_index = args.sample_start + offset
            input_ids = make_input(model.tokenizer, text, length, torch, device)
            result = model.ar_generate(
                input_ids,
                max_new_tokens=args.max_gen_len,
                verbose=False,
                return_metrics=True,
            )
            records.append({
                "policy": "target_ar",
                "length_bin_tokens": length,
                "length_label": f"{length // 1024}K" if length % 1024 == 0 else str(length),
                "input_tokens": int(input_ids.shape[1]),
                "sample_index": sample_index,
                **result,
            })
            print(
                f"[{job}/{total}] target_ar length={length} sample={sample_index} "
                f"tok/s={result['tokens_per_sec']:.2f}",
                flush=True,
            )
    payload = {
        "config": vars(args),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "raw_runs": records,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    temporary = args.output + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, args.output)


def main():
    args = parse_args()
    if args.command == "collect":
        collect_pairs(args)
    elif args.command == "train":
        train_mapper(args)
    elif args.command == "evaluate":
        evaluate_bridge(args)
    elif args.command == "reference":
        evaluate_target_reference(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
