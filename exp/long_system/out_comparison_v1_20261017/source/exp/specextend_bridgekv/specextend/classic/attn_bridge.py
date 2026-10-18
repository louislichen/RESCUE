"""Attention-guided Target-V to Draft-KV bridge utilities.

This module deliberately keeps the target verifier cache read-only.  It uses
the last target layer's value cache from the *committed* prefix as a bridge
source, and either collects aligned training pairs or maps selected positions
into draft-native key/value tensors.  The target model never consumes the
mapped tensors, so enabling the bridge can only change proposal quality.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnBridgeMapper(nn.Module):
    """Low-rank Target-V -> per-layer Draft-(K,V) transducer."""

    def __init__(self, source_dim: int, draft_dim: int, num_draft_layers: int, rank: int = 64):
        super().__init__()
        self.source_dim = int(source_dim)
        self.draft_dim = int(draft_dim)
        self.num_draft_layers = int(num_draft_layers)
        self.rank = int(rank)
        self.source_norm = nn.LayerNorm(self.source_dim)
        self.encoder = nn.Linear(self.source_dim, self.rank, bias=False)
        self.decoders = nn.ModuleList(
            nn.Linear(self.rank, 2 * self.draft_dim, bias=False)
            for _ in range(self.num_draft_layers)
        )

    def _encode(self, source: torch.Tensor) -> torch.Tensor:
        model_dtype = self.source_norm.weight.dtype
        source = source.to(dtype=model_dtype)
        return F.silu(self.encoder(self.source_norm(source)))

    def _active_layers(
        self,
        layer_indices: Optional[Iterable[int]],
    ) -> set[int]:
        return (
            set(range(self.num_draft_layers))
            if layer_indices is None else set(int(index) for index in layer_indices)
        )

    def forward(
        self,
        source: torch.Tensor,
        layer_indices: Optional[Iterable[int]] = None,
    ) -> List[Optional[torch.Tensor]]:
        latent = self._encode(source)
        active = self._active_layers(layer_indices)
        return [
            decoder(latent) if index in active else None
            for index, decoder in enumerate(self.decoders)
        ]

    def forward_component(
        self,
        source: torch.Tensor,
        component: str,
        layer_indices: Optional[Iterable[int]] = None,
    ) -> List[Optional[torch.Tensor]]:
        """Decode only K or V, avoiding the unused half of each output GEMM."""

        if component not in {"key", "value"}:
            raise ValueError(f"Unknown bridge component: {component}")
        latent = self._encode(source)
        active = self._active_layers(layer_indices)
        offset = 0 if component == "key" else self.draft_dim
        stop = offset + self.draft_dim
        outputs: List[Optional[torch.Tensor]] = []
        for index, decoder in enumerate(self.decoders):
            if index not in active:
                outputs.append(None)
                continue
            bias = decoder.bias[offset:stop] if decoder.bias is not None else None
            outputs.append(F.linear(latent, decoder.weight[offset:stop], bias))
        return outputs

    def forward_values(
        self,
        source: torch.Tensor,
        layer_indices: Optional[Iterable[int]] = None,
    ) -> List[Optional[torch.Tensor]]:
        return self.forward_component(source, "value", layer_indices)

    def forward_keys(
        self,
        source: torch.Tensor,
        layer_indices: Optional[Iterable[int]] = None,
    ) -> List[Optional[torch.Tensor]]:
        return self.forward_component(source, "key", layer_indices)

    def config_dict(self) -> Dict[str, int]:
        return {
            "source_dim": self.source_dim,
            "draft_dim": self.draft_dim,
            "num_draft_layers": self.num_draft_layers,
            "rank": self.rank,
        }

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> "AttnBridgeMapper":
        payload = torch.load(path, map_location="cpu")
        mapper = cls(**payload["mapper_config"])
        mapper.load_state_dict(payload["state_dict"])
        return mapper.to(device=device, dtype=dtype).eval()


@dataclass
class BridgeCollectionStats:
    events: int = 0
    pairs: int = 0
    skipped_no_alignment: int = 0


class BridgePairCollector:
    """Collect a bounded, de-duplicated set of aligned cache pairs on CPU."""

    def __init__(
        self,
        max_pairs: int = 12000,
        pairs_per_event: int = 1024,
        max_pairs_per_group: Optional[int] = None,
    ):
        self.max_pairs = int(max_pairs)
        self.pairs_per_event = int(pairs_per_event)
        self.max_pairs_per_group = (
            int(max_pairs_per_group) if max_pairs_per_group is not None else self.max_pairs
        )
        self.current_group = -1
        self.sources: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.groups: List[torch.Tensor] = []
        self._seen = set()
        self._group_counts: Dict[int, int] = {}
        self.stats = BridgeCollectionStats()
        self.metadata: Dict[str, int] = {}

    def set_group(self, group_id: int) -> None:
        self.current_group = int(group_id)

    @property
    def pair_count(self) -> int:
        return self.stats.pairs

    def _choose_indices(self, aligned_length: int) -> List[int]:
        remaining = self.max_pairs - self.pair_count
        group_remaining = self.max_pairs_per_group - self._group_counts.get(
            self.current_group, 0
        )
        # In the released classic engine, calibration shows draft-cache
        # position 0 remains an all-zero reserved slot even though Target V[0]
        # is populated.  Exclude that unaligned position from supervision.
        first_valid_position = 1
        valid_count = aligned_length - first_valid_position
        if remaining <= 0 or group_remaining <= 0 or valid_count <= 0:
            return []
        budget = min(self.pairs_per_event, remaining, group_remaining, valid_count)
        if budget == valid_count:
            candidates = list(range(first_valid_position, aligned_length))
        else:
            candidates = torch.linspace(
                first_valid_position, aligned_length - 1,
                steps=budget, dtype=torch.float64,
            ).round().to(torch.long).tolist()
        chosen = []
        for position in candidates:
            key = (self.current_group, int(position))
            if key not in self._seen:
                self._seen.add(key)
                chosen.append(int(position))
        return chosen

    @torch.no_grad()
    def add_from_model(self, model) -> int:
        if self.pair_count >= self.max_pairs:
            return 0
        if not getattr(model, "past_key_values", None) or not getattr(model, "full_draft_kv", None):
            self.stats.skipped_no_alignment += 1
            return 0

        target_value_cache = model.past_key_values[-1][1]
        target_length = int(target_value_cache.shape[2])
        draft_length = int(getattr(model, "total_seq_len", 0))
        aligned_length = min(target_length, draft_length)
        positions = self._choose_indices(aligned_length)
        if not positions:
            self.stats.skipped_no_alignment += 1
            return 0

        source_device = target_value_cache.data.device
        source_indices = torch.tensor(positions, dtype=torch.long, device=source_device)
        source = target_value_cache.data.index_select(2, source_indices)
        source = source.permute(0, 2, 1, 3).reshape(len(positions), -1)

        layer_targets = []
        for full_key, full_value in model.full_draft_kv:
            draft_indices = source_indices.to(full_key.device)
            key = full_key.index_select(2, draft_indices)
            value = full_value.index_select(2, draft_indices)
            key = key.permute(0, 2, 1, 3).reshape(len(positions), -1)
            value = value.permute(0, 2, 1, 3).reshape(len(positions), -1)
            layer_targets.append(torch.cat([key, value], dim=-1))
        target = torch.stack(layer_targets, dim=1)

        self.sources.append(source.detach().to(device="cpu", dtype=torch.float16))
        self.targets.append(target.detach().to(device="cpu", dtype=torch.float16))
        self.groups.append(torch.full((len(positions),), self.current_group, dtype=torch.long))
        self.stats.events += 1
        self.stats.pairs += len(positions)
        self._group_counts[self.current_group] = (
            self._group_counts.get(self.current_group, 0) + len(positions)
        )
        if not self.metadata:
            self.metadata = {
                "source_dim": int(source.shape[-1]),
                "draft_dim": int(target.shape[-1] // 2),
                "num_draft_layers": int(target.shape[1]),
                "target_value_heads": int(target_value_cache.data.shape[1]),
                "target_head_dim": int(target_value_cache.data.shape[3]),
                "draft_heads": int(model.full_draft_kv[0][0].shape[1]),
                "draft_head_dim": int(model.full_draft_kv[0][0].shape[3]),
            }
        return len(positions)

    def payload(self, extra_metadata: Optional[Dict] = None) -> Dict:
        if not self.sources:
            raise RuntimeError("No bridge pairs were collected")
        metadata = dict(self.metadata)
        metadata.update({
            "max_pairs": self.max_pairs,
            "pairs_per_event": self.pairs_per_event,
            "max_pairs_per_group": self.max_pairs_per_group,
            "group_pair_counts": dict(sorted(self._group_counts.items())),
            "events": self.stats.events,
            "pairs": self.stats.pairs,
            "skipped_no_alignment": self.stats.skipped_no_alignment,
        })
        if extra_metadata:
            metadata.update(extra_metadata)
        return {
            "source": torch.cat(self.sources, dim=0),
            "target": torch.cat(self.targets, dim=0),
            "group_ids": torch.cat(self.groups, dim=0),
            "metadata": metadata,
        }

    def save_atomic(self, path: str, extra_metadata: Optional[Dict] = None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        temporary = path + ".tmp"
        torch.save(self.payload(extra_metadata), temporary)
        os.replace(temporary, path)


def reset_bridge_runtime(model) -> None:
    model._attn_bridge_skip_initial_refresh = False
    model.bridge_runtime = {
        "map_events": [],
        "prepare_events": [],
        "draft_prefill_events": [],
        "mapped_tokens": 0,
        "source_bytes": 0,
        "materialized_bridge_bytes": 0,
        "mapper_macs": 0,
        "prefill_mapper_macs": 0,
        "draft_k_projected_tokens": 0,
        "draft_v_projected_tokens": 0,
        "draft_v_skipped_tokens": 0,
        "draft_k_projection_macs": 0,
        "draft_v_projection_macs": 0,
        "draft_v_projection_saved_macs": 0,
        "draft_prefill_k_projected_tokens": 0,
        "draft_prefill_v_projected_tokens": 0,
        "draft_prefill_v_skipped_tokens": 0,
        "draft_prefill_k_projection_macs": 0,
        "draft_prefill_v_projection_macs": 0,
        "draft_prefill_v_projection_saved_macs": 0,
        "value_bypass_events": 0,
        "value_shadow_events": 0,
        "draft_v_shadow_tokens": 0,
        "draft_v_shadow_macs": 0,
        "draft_prefill_v_shadow_tokens": 0,
        "draft_prefill_v_shadow_macs": 0,
        "projection_by_layer": {},
        "value_replacement_events": [],
        "cosine_tensors": [],
        "nmse_tensors": [],
        "diagnostic_weights": [],
        "geometry_events": [],
        "geometry_layer_event_counts": {},
        "geometry_rescue_events": 0,
    }
    # Geometry payloads live on Draft attention modules so they can observe the
    # exact post-RoPE query from the real forward pass.  Projection accounting
    # uses the same request-local object, including native controls.
    draft_core = getattr(getattr(model, "draft_model", None), "model", None)
    for layer_index, layer in enumerate(getattr(draft_core, "layers", ())):
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is not None:
            self_attn._attn_bridge_geometry_payload = None
            self_attn._attn_bridge_runtime = model.bridge_runtime
            self_attn._attn_bridge_layer_index = layer_index


def maybe_collect_bridge_pairs(model) -> int:
    collector = getattr(model, "bridge_collector", None)
    if collector is None:
        return 0
    return collector.add_from_model(model)


def _flatten_target_values(target_value_cache, indices: torch.Tensor) -> torch.Tensor:
    source_indices = indices.to(target_value_cache.data.device)
    source = target_value_cache.data.index_select(2, source_indices)
    return source.permute(0, 2, 1, 3).reshape(indices.numel(), -1)


def _active_bridge_layers(layer_mode: str, layer_count: int) -> set[int]:
    if layer_mode == "last":
        return {layer_count - 1}
    if layer_mode == "all":
        return set(range(layer_count))
    raise ValueError(f"Unknown bridge layer mode: {layer_mode}")


def _record_mapper_macs(
    runtime: Dict[str, object],
    mapper: AttnBridgeMapper,
    token_count: int,
    active_layer_count: int,
    decoded_components: int,
) -> None:
    # MAC accounting covers dense mapper projections. LayerNorm/SiLU and
    # indexing are intentionally reported separately from projection MACs.
    per_token = mapper.source_dim * mapper.rank
    per_token += active_layer_count * mapper.rank * mapper.draft_dim * decoded_components
    runtime["mapper_macs"] += int(token_count * per_token)


@torch.no_grad()
def prepare_prefill_value_bridge(
    model,
    draft_query_length: int,
) -> Optional[List[Optional[Dict[str, object]]]]:
    """Build hard V replacements before Draft prefill and skip their v_proj.

    The target cache stays read-only. Native Draft K is still projected for
    every token, preserving the Draft QK coordinate system. Position 0 and a
    configurable recent suffix stay native; only the aligned middle prefix is
    decoded by the Target-V mapper. Hard replacement (alpha_v=1) is required:
    a blend would still need the native V and therefore would not save compute.
    """

    mapper = getattr(model, "attn_bridge_mapper", None)
    config = getattr(model, "attn_bridge_config", None) or {}
    compute_mode = str(config.get("compute_mode", "cache_overwrite"))
    if compute_mode not in {"prefill_value_bypass", "prefill_value_shadow"}:
        return None
    if mapper is None:
        raise RuntimeError("Inline prefill V replacement requires an AttnBridge mapper")
    if getattr(model, "bridge_collector", None) is not None:
        raise RuntimeError("Bridge pair collection and inline V replacement are mutually exclusive")

    components = str(config.get("components", "value"))
    alpha = float(config.get("alpha", 0.0))
    alpha_key = float(config.get("alpha_key", 0.0 if components == "value" else alpha))
    alpha_value = float(config.get("alpha_value", alpha))
    if components != "value" or alpha_key != 0.0 or abs(alpha_value - 1.0) > 1.0e-8:
        raise ValueError(
            "prefill_value_bypass requires components='value', alpha_key=0, alpha_value=1"
        )

    target_cache = getattr(model, "past_key_values", None)
    if not target_cache:
        raise RuntimeError("Target cache must be populated before Draft V bypass")
    target_value_cache = target_cache[-1][1]
    mapper_device = next(mapper.parameters()).device
    if target_value_cache.data.device != mapper_device:
        raise RuntimeError(
            "Inline Bridge V replacement currently requires Target cache and "
            f"mapper on one CUDA device, got {target_value_cache.data.device} "
            f"and {mapper_device}"
        )
    target_length = int(target_value_cache.shape[2])
    query_length = int(draft_query_length)
    if query_length != target_length + 1:
        raise ValueError(
            "Inline Draft V replacement expects exactly one unverified root "
            f"after the committed Target cache, got q_len={query_length} and "
            f"target_len={target_length}"
        )

    recent_native = max(0, int(config.get("recent_native", 128)))
    # Position 0 stays native because existing calibration excluded it from
    # mapper supervision. The target-selected root token beyond target_length
    # also remains native automatically.
    map_start = 1
    map_stop = max(map_start, target_length - recent_native)
    if map_stop <= map_start:
        return None

    prepare_start = prepare_end = None
    mapper_stream = None
    if mapper_device.type == "cuda":
        prepare_start = torch.cuda.Event(enable_timing=True)
        prepare_end = torch.cuda.Event(enable_timing=True)
        mapper_stream = torch.cuda.current_stream(mapper_device)
        prepare_start.record(mapper_stream)

    source_indices = torch.arange(
        map_start,
        map_stop,
        dtype=torch.long,
        device=target_value_cache.data.device,
    )
    source = _flatten_target_values(target_value_cache, source_indices)
    source_mode = str(config.get("source_mode", "aligned"))
    if source_mode == "shuffle":
        source = torch.flip(source, dims=(0,))
    elif source_mode == "zero":
        source = torch.zeros_like(source)
    elif source_mode != "aligned":
        raise ValueError(f"Unknown bridge source_mode: {source_mode}")

    draft_core = getattr(getattr(model, "draft_model", None), "model", None)
    layer_count = len(getattr(draft_core, "layers", ()))
    if layer_count <= 0:
        raise RuntimeError("Draft model has no decoder layers")
    active_layers = _active_bridge_layers(str(config.get("layers", "all")), layer_count)
    source = source.to(mapper_device, non_blocking=True)
    start = end = None
    if source.is_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(mapper_stream)
    mapped_layers = mapper.forward_values(source, layer_indices=active_layers)
    if end is not None:
        end.record(mapper_stream)
        model.bridge_runtime["map_events"].append((start, end))

    overrides: List[Optional[Dict[str, object]]] = []
    materialized_bytes = 0
    for layer_index, mapped_value in enumerate(mapped_layers):
        if mapped_value is None:
            overrides.append(None)
            continue
        mapped_value = mapped_value.unsqueeze(0).contiguous()
        materialized_bytes += mapped_value.numel() * mapped_value.element_size()
        overrides.append({
            "start": map_start,
            "stop": map_stop,
            "value": mapped_value,
            "shadow_compute": compute_mode == "prefill_value_shadow",
        })

    if prepare_end is not None:
        prepare_end.record(mapper_stream)
        model.bridge_runtime["prepare_events"].append(
            (prepare_start, prepare_end)
        )

    token_count = int(source_indices.numel())
    runtime = model.bridge_runtime
    runtime["mapped_tokens"] += token_count
    runtime["source_bytes"] += int(source.numel() * source.element_size())
    runtime["materialized_bridge_bytes"] += int(materialized_bytes)
    if compute_mode == "prefill_value_bypass":
        runtime["value_bypass_events"] += 1
    else:
        runtime["value_shadow_events"] += 1
    mapper_macs_before = int(runtime["mapper_macs"])
    _record_mapper_macs(runtime, mapper, token_count, len(active_layers), 1)
    runtime["prefill_mapper_macs"] += int(runtime["mapper_macs"]) - mapper_macs_before
    # The first working-cache refresh selects from the cache produced by this
    # very forward, so its V entries are already bridged. Compute-mode refreshes
    # subsequently select from this persisted full cache without remapping.
    model._attn_bridge_skip_initial_refresh = True
    return overrides


@torch.no_grad()
def record_attn_bridge_geometry(
    attention_module,
    query_states: torch.Tensor,
    native_key_states: torch.Tensor,
    bridged_key_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    bridged_logits_override: Optional[torch.Tensor] = None,
) -> None:
    """Measure mapped-K damage under the exact same Draft query.

    Inputs already have Draft RoPE applied and KV heads repeated. This function
    is observational and never replaces tensors used by the real forward pass.
    """

    payload = getattr(attention_module, "_attn_bridge_geometry_payload", None)
    if not payload or not bool(payload.get("collect_geometry", False)):
        return
    runtime = payload["runtime"]
    layer_index = int(payload["layer_index"])
    counts = runtime["geometry_layer_event_counts"]
    max_events = int(payload.get("max_events_per_layer", 8))
    if int(counts.get(layer_index, 0)) >= max_events:
        return
    if native_key_states.shape != bridged_key_states.shape:
        return

    max_queries = max(1, int(payload.get("max_queries", 16)))
    query_count = int(query_states.shape[2])
    if query_count > max_queries:
        query_indices = torch.linspace(
            0, query_count - 1, steps=max_queries,
            device=query_states.device, dtype=torch.float64,
        ).round().to(torch.long)
        query_indices = torch.unique(query_indices, sorted=True)
        query = query_states.index_select(2, query_indices)
    else:
        query_indices = None
        query = query_states

    query = query.float()
    native_key = native_key_states.float()
    bridged_key = bridged_key_states.float()
    scale = float(query_states.shape[-1]) ** -0.5
    native_logits = torch.matmul(query, native_key.transpose(2, 3)) * scale
    if bridged_logits_override is None:
        bridged_logits = torch.matmul(query, bridged_key.transpose(2, 3)) * scale
        post_rescue = False
    else:
        if bridged_logits_override.shape != (
            query_states.shape[0],
            query_states.shape[1],
            query_states.shape[2],
            bridged_key_states.shape[2],
        ):
            return
        bridged_logits = bridged_logits_override
        if query_indices is not None:
            bridged_logits = bridged_logits.index_select(
                2, query_indices.to(bridged_logits.device)
            )
        bridged_logits = bridged_logits.to(
            device=native_logits.device,
            dtype=torch.float32,
        )
        post_rescue = True

    if attention_mask is None:
        selected_mask = None
        valid = torch.ones(
            (native_logits.shape[0], 1, native_logits.shape[2], native_logits.shape[3]),
            dtype=torch.bool, device=native_logits.device,
        )
    else:
        selected_mask = attention_mask
        if query_indices is not None:
            selected_mask = selected_mask.index_select(2, query_indices.to(selected_mask.device))
        selected_mask = selected_mask.to(device=native_logits.device, dtype=torch.float32)
        # Disallowed locations are finfo.min in this engine (and may be -inf
        # in other attention paths), while allowed locations are zero.
        valid = selected_mask > -1.0e4
    valid = valid.expand_as(native_logits)

    delta_logits = bridged_logits - native_logits
    delta_valid = delta_logits.masked_select(valid)
    native_valid = native_logits.masked_select(valid)
    projected_mae = delta_valid.abs().mean()
    projected_rmse = delta_valid.square().mean().sqrt()
    native_rms = native_valid.square().mean().sqrt().clamp_min(1.0e-12)
    relative_rmse = projected_rmse / native_rms

    if selected_mask is not None:
        native_for_softmax = native_logits + selected_mask
        bridged_for_softmax = bridged_logits + selected_mask
    else:
        native_for_softmax = native_logits
        bridged_for_softmax = bridged_logits
    native_prob = torch.softmax(native_for_softmax, dim=-1, dtype=torch.float32)
    bridged_prob = torch.softmax(bridged_for_softmax, dim=-1, dtype=torch.float32)
    epsilon = 1.0e-12
    attention_kl = (
        native_prob
        * (native_prob.clamp_min(epsilon).log() - bridged_prob.clamp_min(epsilon).log())
    ).sum(dim=-1).mean().clamp_min(0.0)

    min_valid_keys = int(valid.sum(dim=-1).min().item())
    top_k = min(
        max(1, int(payload.get("top_k", 8))),
        int(native_logits.shape[-1]),
        max(1, min_valid_keys),
    )
    native_top = torch.topk(native_for_softmax, k=top_k, dim=-1).indices
    bridged_top = torch.topk(bridged_for_softmax, k=top_k, dim=-1).indices
    matches = native_top.unsqueeze(-1).eq(bridged_top.unsqueeze(-2))
    intersection = matches.any(dim=-1).sum(dim=-1).float()
    topk_jaccard = (intersection / (2.0 * top_k - intersection).clamp_min(1.0)).mean()
    argmax_flip = native_for_softmax.argmax(dim=-1).ne(
        bridged_for_softmax.argmax(dim=-1)
    ).float().mean()

    batch_size, head_count, measured_queries, key_count = native_logits.shape
    query_row_count = int(batch_size * head_count * measured_queries)
    runtime["geometry_events"].append({
        "layer": layer_index,
        "batch_size": int(batch_size),
        "heads": int(head_count),
        "queries": int(measured_queries),
        "kv_tokens": int(key_count),
        "top_k": int(top_k),
        "post_rescue": post_rescue,
        "weight": query_row_count,
        "query_row_count": query_row_count,
        "valid_logit_count": int(valid.sum().item()),
        "qk_logit_sq_error_sum": float(delta_valid.square().sum().item()),
        "qk_logit_abs_error_sum": float(delta_valid.abs().sum().item()),
        "native_logit_sq_sum": float(native_valid.square().sum().item()),
        "qk_logit_rmse": float(projected_rmse.item()),
        "qk_logit_relative_rmse": float(relative_rmse.item()),
        "attention_kl_native_to_bridge": float(attention_kl.item()),
        "topk_jaccard": float(topk_jaccard.item()),
        "argmax_flip_rate": float(argmax_flip.item()),
        "query_projected_key_mae": float(projected_mae.item()),
    })
    counts[layer_index] = int(counts.get(layer_index, 0)) + 1


@torch.no_grad()
def maybe_rescue_attn_bridge_logits(
    attention_module,
    query_states: torch.Tensor,
    native_key_states: Optional[torch.Tensor],
    bridged_logits: torch.Tensor,
) -> torch.Tensor:
    """Oracle ablation that restores native-QK routing for the stable prefix.

    Values, tree/current-tail keys, and every other part of the K+V run remain
    unchanged. Only logits against the snapshotted stable-cache prefix are
    overwritten by logits from native Draft K. This is intentionally an oracle
    diagnostic, not a deployable inference mechanism.
    """

    payload = getattr(attention_module, "_attn_bridge_geometry_payload", None)
    if not payload or not bool(payload.get("geometry_rescue", False)):
        return bridged_logits
    if native_key_states is None or native_key_states.shape != (
        bridged_logits.shape[0],
        bridged_logits.shape[1],
        bridged_logits.shape[3],
        query_states.shape[-1],
    ):
        return bridged_logits
    base_length = min(
        int(payload["native_base_key"].shape[2]),
        int(bridged_logits.shape[-1]),
    )
    if base_length <= 0:
        return bridged_logits
    native_prefix_logits = torch.matmul(
        query_states,
        native_key_states[:, :, :base_length, :].transpose(2, 3),
    ) / (float(query_states.shape[-1]) ** 0.5)
    rescued = bridged_logits.clone()
    rescued[:, :, :, :base_length] = native_prefix_logits.to(rescued.dtype)
    payload["runtime"]["geometry_rescue_events"] += 1
    return rescued


@torch.no_grad()
def maybe_apply_attn_bridge(
    model,
    retrieved_indices: torch.Tensor,
    working_kv: List[Tuple[torch.Tensor, torch.Tensor]],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    mapper = getattr(model, "attn_bridge_mapper", None)
    config = getattr(model, "attn_bridge_config", None) or {}
    alpha = float(config.get("alpha", 0.0))
    components = str(config.get("components", "kv"))
    if components not in {"kv", "key", "value"}:
        raise ValueError(f"Unknown bridge component mode: {components}")
    alpha_key = float(config.get("alpha_key", alpha))
    alpha_value = float(config.get("alpha_value", alpha))
    if components == "key":
        alpha_value = 0.0
    elif components == "value":
        alpha_key = 0.0

    # Every refresh invalidates the previous native cache snapshot, including
    # refreshes from a subsequent value-only/native ablation.
    draft_core = getattr(getattr(model, "draft_model", None), "model", None)
    for layer in getattr(draft_core, "layers", ()):
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is not None:
            self_attn._attn_bridge_geometry_payload = None

    if mapper is None or max(alpha_key, alpha_value) <= 0.0 or not working_kv:
        return working_kv
    if str(config.get("compute_mode", "cache_overwrite")) in {
        "prefill_value_bypass",
        "prefill_value_shadow",
    }:
        # Inline replacement already persisted its mapped V into full_draft_kv.
        # Re-mapping here would add cache-overwrite work that is unrelated to
        # the projection bypass and would blur the shadow-vs-bypass control.
        model._attn_bridge_skip_initial_refresh = False
        return working_kv
    if bool(getattr(model, "_attn_bridge_skip_initial_refresh", False)):
        model._attn_bridge_skip_initial_refresh = False
        return working_kv

    target_value_cache = model.past_key_values[-1][1]
    target_length = int(target_value_cache.shape[2])
    recent_native = max(0, int(config.get("recent_native", 128)))
    cutoff = max(0, target_length - recent_native)
    # Position 0 stays native because it was excluded from mapper supervision.
    map_mask = (retrieved_indices > 0) & (retrieved_indices < cutoff)
    if not bool(map_mask.any()):
        return working_kv

    working_positions = torch.nonzero(map_mask, as_tuple=False).flatten()
    source_indices = retrieved_indices.index_select(0, working_positions)
    source = _flatten_target_values(target_value_cache, source_indices)
    source_mode = str(config.get("source_mode", "aligned"))
    if source_mode == "shuffle":
        # Reverse the selected sequence so the counterfactual is not merely an
        # adjacent-token shift (adjacent Target-V vectors are highly correlated).
        source = torch.flip(source, dims=(0,))
    elif source_mode == "zero":
        source = torch.zeros_like(source)
    elif source_mode != "aligned":
        raise ValueError(f"Unknown bridge source_mode: {source_mode}")

    mapper_device = next(mapper.parameters()).device
    source = source.to(mapper_device, non_blocking=True)
    start = end = None
    if source.is_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        mapper_stream = torch.cuda.current_stream(mapper_device)
        start.record(mapper_stream)
    active_layers = _active_bridge_layers(
        str(config.get("layers", "all")), len(working_kv)
    )
    if alpha_key > 0.0 and alpha_value > 0.0:
        mapped_layers = mapper(source, layer_indices=active_layers)
        decoded_components = 2
        decoded_kind = "kv"
    elif alpha_key > 0.0:
        mapped_layers = mapper.forward_keys(source, layer_indices=active_layers)
        decoded_components = 1
        decoded_kind = "key"
    else:
        mapped_layers = mapper.forward_values(source, layer_indices=active_layers)
        decoded_components = 1
        decoded_kind = "value"

    collect_diagnostics = bool(config.get("collect_diagnostics", False))
    collect_geometry = bool(config.get(
        "collect_geometry_diagnostics",
        config.get("collect_geometry", False),
    ))
    geometry_rescue = bool(config.get("geometry_rescue", False))

    cosine_values = []
    nmse_values = []
    materialized_bytes = 0
    for layer_index, (native_key, native_value) in enumerate(working_kv):
        if layer_index not in active_layers:
            continue
        native_key_for_geometry = (
            native_key.detach().clone()
            if alpha_key > 0.0 and (collect_geometry or geometry_rescue)
            else None
        )
        head_count = native_key.shape[1]
        head_dim = native_key.shape[3]
        draft_dim = head_count * head_dim
        decoded = mapped_layers[layer_index]
        assert decoded is not None
        decoded = decoded.to(native_key.device, dtype=native_key.dtype)
        mapped_key = mapped_value = None
        if decoded_kind == "kv":
            mapped_key, mapped_value = decoded.split(draft_dim, dim=-1)
        elif decoded_kind == "key":
            mapped_key = decoded
        else:
            mapped_value = decoded
        if mapped_key is not None:
            mapped_key = mapped_key.reshape(
                1, -1, head_count, head_dim
            ).permute(0, 2, 1, 3).contiguous()
        if mapped_value is not None:
            mapped_value = mapped_value.reshape(
                1, -1, head_count, head_dim
            ).permute(0, 2, 1, 3).contiguous()
        local_positions = working_positions.to(native_key.device)
        native_key_slice = (
            native_key.index_select(2, local_positions) if alpha_key > 0.0 else None
        )
        native_value_slice = (
            native_value.index_select(2, local_positions) if alpha_value > 0.0 else None
        )
        if alpha_key > 0.0:
            assert mapped_key is not None and native_key_slice is not None
            native_key.index_copy_(
                2, local_positions, torch.lerp(native_key_slice, mapped_key, alpha_key)
            )
        if alpha_value > 0.0:
            assert mapped_value is not None and native_value_slice is not None
            native_value.index_copy_(
                2, local_positions, torch.lerp(native_value_slice, mapped_value, alpha_value)
            )

        if native_key_for_geometry is not None:
            attention_layer = getattr(draft_core.layers[layer_index], "self_attn", None)
            if attention_layer is not None:
                attention_layer._attn_bridge_geometry_payload = {
                    "runtime": model.bridge_runtime,
                    "layer_index": layer_index,
                    "native_base_key": native_key_for_geometry,
                    "collect_geometry": collect_geometry,
                    "geometry_rescue": geometry_rescue,
                    "top_k": int(config.get("geometry_top_k", 8)),
                    "max_queries": int(config.get("geometry_max_queries", 16)),
                    "max_events_per_layer": int(
                        config.get("geometry_max_events_per_layer", 8)
                    ),
                }

        if collect_diagnostics:
            prediction_parts = []
            teacher_parts = []
            if alpha_key > 0.0:
                assert mapped_key is not None and native_key_slice is not None
                prediction_parts.append(mapped_key.flatten())
                teacher_parts.append(native_key_slice.flatten())
            if alpha_value > 0.0:
                assert mapped_value is not None and native_value_slice is not None
                prediction_parts.append(mapped_value.flatten())
                teacher_parts.append(native_value_slice.flatten())
            prediction = torch.cat(prediction_parts)
            teacher = torch.cat(teacher_parts)
            cosine_values.append(
                F.cosine_similarity(prediction.float(), teacher.float(), dim=0)
            )
            nmse_values.append(
                (prediction.float() - teacher.float()).pow(2).mean()
                / teacher.float().pow(2).mean().clamp_min(1e-12)
            )
        if mapped_key is not None:
            materialized_bytes += mapped_key.numel() * mapped_key.element_size()
        if mapped_value is not None:
            materialized_bytes += mapped_value.numel() * mapped_value.element_size()

    if end is not None:
        end.record(mapper_stream)
        model.bridge_runtime["map_events"].append((start, end))
    model.bridge_runtime["mapped_tokens"] += int(source_indices.numel())
    model.bridge_runtime["source_bytes"] += int(source.numel() * source.element_size())
    model.bridge_runtime["materialized_bridge_bytes"] += int(materialized_bytes)
    _record_mapper_macs(
        model.bridge_runtime,
        mapper,
        int(source_indices.numel()),
        len(active_layers),
        decoded_components,
    )
    if cosine_values:
        model.bridge_runtime["cosine_tensors"].append(torch.stack(cosine_values).mean())
        model.bridge_runtime["nmse_tensors"].append(torch.stack(nmse_values).mean())
        model.bridge_runtime["diagnostic_weights"].append(
            int(source_indices.numel()) * len(active_layers)
        )
    return working_kv


def summarize_bridge_runtime(model) -> Dict[str, object]:
    runtime = getattr(model, "bridge_runtime", None) or {}
    events = runtime.get("map_events", [])
    for _, end in events:
        end.synchronize()
    map_time_ms = sum(start.elapsed_time(end) for start, end in events)
    prepare_events = runtime.get("prepare_events", [])
    for _, end in prepare_events:
        end.synchronize()
    prepare_time_ms = sum(
        start.elapsed_time(end) for start, end in prepare_events
    )
    prefill_events = runtime.get("draft_prefill_events", [])
    for _, end in prefill_events:
        end.synchronize()
    draft_prefill_time_ms = sum(start.elapsed_time(end) for start, end in prefill_events)
    prefill_v_projected = int(runtime.get("draft_prefill_v_projected_tokens", 0))
    prefill_v_skipped = int(runtime.get("draft_prefill_v_skipped_tokens", 0))
    prefill_v_total = prefill_v_projected + prefill_v_skipped
    prefill_saved_macs = int(runtime.get("draft_prefill_v_projection_saved_macs", 0))
    prefill_mapper_macs = int(runtime.get("prefill_mapper_macs", 0))
    cosine = runtime.get("cosine_tensors", [])
    nmse = runtime.get("nmse_tensors", [])
    weights = runtime.get("diagnostic_weights", [])
    if cosine and weights:
        denominator = float(sum(weights))
        mean_cosine = sum(
            float(value.item()) * weight for value, weight in zip(cosine, weights)
        ) / denominator
        mean_nmse = sum(
            float(value.item()) * weight for value, weight in zip(nmse, weights)
        ) / denominator
    else:
        mean_cosine = None
        mean_nmse = None
    geometry_events = runtime.get("geometry_events", [])
    geometry_metric_names = (
        "qk_logit_rmse",
        "qk_logit_relative_rmse",
        "attention_kl_native_to_bridge",
        "topk_jaccard",
        "argmax_flip_rate",
        "query_projected_key_mae",
    )

    def summarize_geometry(events_to_summarize):
        if not events_to_summarize:
            return {name: None for name in geometry_metric_names}
        total_rows = float(sum(
            max(1, int(event["query_row_count"])) for event in events_to_summarize
        ))
        valid_logits = float(sum(
            max(1, int(event["valid_logit_count"])) for event in events_to_summarize
        ))
        squared_error = sum(
            float(event["qk_logit_sq_error_sum"]) for event in events_to_summarize
        )
        absolute_error = sum(
            float(event["qk_logit_abs_error_sum"]) for event in events_to_summarize
        )
        native_squared = sum(
            float(event["native_logit_sq_sum"]) for event in events_to_summarize
        )
        qk_rmse = (squared_error / valid_logits) ** 0.5
        native_rms = (native_squared / valid_logits) ** 0.5
        row_weighted_names = (
            "attention_kl_native_to_bridge",
            "topk_jaccard",
            "argmax_flip_rate",
        )
        summary = {
            name: sum(
                float(event[name]) * max(1, int(event["query_row_count"]))
                for event in events_to_summarize
            ) / total_rows
            for name in row_weighted_names
        }
        summary.update({
            "qk_logit_rmse": qk_rmse,
            "qk_logit_relative_rmse": qk_rmse / max(native_rms, 1.0e-12),
            "query_projected_key_mae": absolute_error / valid_logits,
        })
        return summary

    geometry_summary = summarize_geometry(geometry_events)
    geometry_by_layer = []
    for layer_index in sorted({int(event["layer"]) for event in geometry_events}):
        layer_events = [
            event for event in geometry_events
            if int(event["layer"]) == layer_index
        ]
        geometry_by_layer.append({
            "layer": layer_index,
            "events": len(layer_events),
            **summarize_geometry(layer_events),
        })
    return {
        "bridge_map_time_ms": float(map_time_ms),
        "bridge_map_events": int(len(events)),
        "bridge_prepare_time_ms": float(prepare_time_ms),
        "bridge_prepare_events": int(len(prepare_events)),
        "draft_prefill_time_ms": float(draft_prefill_time_ms),
        "draft_prefill_events": int(len(prefill_events)),
        "bridge_mapped_tokens": int(runtime.get("mapped_tokens", 0)),
        "bridge_source_bytes": int(runtime.get("source_bytes", 0)),
        "bridge_materialized_bytes": int(runtime.get("materialized_bridge_bytes", 0)),
        "bridge_mapper_macs": int(runtime.get("mapper_macs", 0)),
        "bridge_prefill_mapper_macs": prefill_mapper_macs,
        "bridge_value_bypass_events": int(runtime.get("value_bypass_events", 0)),
        "bridge_value_shadow_events": int(runtime.get("value_shadow_events", 0)),
        "draft_k_projected_tokens": int(runtime.get("draft_k_projected_tokens", 0)),
        "draft_v_projected_tokens": int(runtime.get("draft_v_projected_tokens", 0)),
        "draft_v_skipped_tokens": int(runtime.get("draft_v_skipped_tokens", 0)),
        "draft_k_projection_macs": int(runtime.get("draft_k_projection_macs", 0)),
        "draft_v_projection_macs": int(runtime.get("draft_v_projection_macs", 0)),
        "draft_v_projection_saved_macs": int(
            runtime.get("draft_v_projection_saved_macs", 0)
        ),
        "draft_v_shadow_tokens": int(runtime.get("draft_v_shadow_tokens", 0)),
        "draft_v_shadow_macs": int(runtime.get("draft_v_shadow_macs", 0)),
        "draft_prefill_k_projected_tokens": int(
            runtime.get("draft_prefill_k_projected_tokens", 0)
        ),
        "draft_prefill_v_projected_tokens": prefill_v_projected,
        "draft_prefill_v_skipped_tokens": prefill_v_skipped,
        "draft_prefill_v_skipped_fraction": (
            prefill_v_skipped / prefill_v_total if prefill_v_total else 0.0
        ),
        "draft_prefill_k_projection_macs": int(
            runtime.get("draft_prefill_k_projection_macs", 0)
        ),
        "draft_prefill_v_projection_macs": int(
            runtime.get("draft_prefill_v_projection_macs", 0)
        ),
        "draft_prefill_v_projection_saved_macs": prefill_saved_macs,
        "draft_prefill_v_shadow_tokens": int(
            runtime.get("draft_prefill_v_shadow_tokens", 0)
        ),
        "draft_prefill_v_shadow_macs": int(
            runtime.get("draft_prefill_v_shadow_macs", 0)
        ),
        "draft_prefill_net_projection_macs_saved": prefill_saved_macs - prefill_mapper_macs,
        "draft_projection_by_layer": [
            {"layer": int(layer), **values}
            for layer, values in sorted(
                runtime.get("projection_by_layer", {}).items(),
                key=lambda item: int(item[0]),
            )
        ],
        "draft_value_replacement_events": list(
            runtime.get("value_replacement_events", [])
        ),
        "bridge_mean_kv_cosine": mean_cosine,
        "bridge_mean_kv_nmse": mean_nmse,
        "bridge_geometry_events": int(len(geometry_events)),
        "bridge_geometry_post_rescue_events": int(sum(
            bool(event.get("post_rescue", False)) for event in geometry_events
        )),
        "bridge_geometry_rescue_events": int(runtime.get("geometry_rescue_events", 0)),
        "bridge_geometry_qk_logit_rmse": geometry_summary["qk_logit_rmse"],
        "bridge_geometry_qk_logit_relative_rmse": geometry_summary[
            "qk_logit_relative_rmse"
        ],
        "bridge_geometry_attention_kl": geometry_summary[
            "attention_kl_native_to_bridge"
        ],
        "bridge_geometry_topk_jaccard": geometry_summary["topk_jaccard"],
        "bridge_geometry_argmax_flip_rate": geometry_summary["argmax_flip_rate"],
        "bridge_geometry_query_projected_key_mae": geometry_summary[
            "query_projected_key_mae"
        ],
        "bridge_geometry_by_layer": geometry_by_layer,
    }
