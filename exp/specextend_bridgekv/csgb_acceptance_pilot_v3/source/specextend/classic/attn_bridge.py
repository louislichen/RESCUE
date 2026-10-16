"""Attention-guided Target-V to Draft-KV bridge utilities.

This module deliberately keeps the target verifier cache read-only.  It uses
the last target layer's value cache from the *committed* prefix as a bridge
source, and either collects aligned training pairs or maps selected positions
into draft-native key/value tensors.  The target model never consumes the
mapped tensors, so enabling the bridge can only change proposal quality.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnBridgeMapper(nn.Module):
    """Low-rank Target-V -> per-layer Draft cache-component transducer.

    ``components="value"`` is a physically V-only architecture: every decoder
    has ``draft_dim`` outputs and the checkpoint contains no unused K rows.
    ``components="kv"`` retains the original layout so checkpoints produced by
    the first BridgeKV experiments remain loadable.
    """

    _COMPONENT_ORDERS = {
        "kv": ("key", "value"),
        "key": ("key",),
        "value": ("value",),
    }
    _ARCHITECTURES = {
        "kv": "low_rank_shared_encoder_per_layer_kv_v1",
        "key": "low_rank_shared_encoder_per_layer_key_v1",
        "value": "low_rank_shared_encoder_per_layer_value_v1",
    }

    def __init__(
        self,
        source_dim: int,
        draft_dim: int,
        num_draft_layers: int,
        rank: int = 64,
        components: str = "kv",
        architecture: Optional[str] = None,
    ):
        super().__init__()
        self.source_dim = int(source_dim)
        self.draft_dim = int(draft_dim)
        self.num_draft_layers = int(num_draft_layers)
        self.rank = int(rank)
        self.components = str(components)
        # Checkpoint loading replaces this default from immutable training
        # metadata.  Keeping it on the module lets the bridge itself reject a
        # residual/replacement policy mismatch instead of relying only on an
        # outer experiment driver.
        self.output_semantics = "replacement"
        if self.components not in self._COMPONENT_ORDERS:
            raise ValueError(f"Unknown mapper component mode: {self.components}")
        expected_architecture = self._ARCHITECTURES[self.components]
        self.architecture = architecture or expected_architecture
        if self.architecture != expected_architecture:
            raise ValueError(
                f"Architecture {self.architecture!r} is incompatible with "
                f"components={self.components!r}; expected {expected_architecture!r}"
            )
        if min(self.source_dim, self.draft_dim, self.num_draft_layers, self.rank) <= 0:
            raise ValueError("All mapper dimensions and rank must be positive")
        self.component_order = self._COMPONENT_ORDERS[self.components]
        self.source_norm = nn.LayerNorm(self.source_dim)
        self.encoder = nn.Linear(self.source_dim, self.rank, bias=False)
        self.decoders = nn.ModuleList(
            nn.Linear(
                self.rank,
                len(self.component_order) * self.draft_dim,
                bias=False,
            )
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
        if component not in self.component_order:
            raise RuntimeError(
                f"This mapper stores {self.components!r}, so it cannot decode "
                f"the requested {component!r} component"
            )
        latent = self._encode(source)
        active = self._active_layers(layer_indices)
        offset = self.component_order.index(component) * self.draft_dim
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

    def supports_component(self, component: str) -> bool:
        return component in self.component_order

    def parameter_count(self, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        if trainable_only:
            return sum(
                parameter.numel()
                for parameter in parameters
                if parameter.requires_grad
            )
        return sum(parameter.numel() for parameter in parameters)

    def dense_projection_macs_per_token(
        self,
        active_layer_count: Optional[int] = None,
        decoded_components: Optional[int] = None,
    ) -> int:
        """Return exact Linear-layer MACs; LayerNorm and SiLU are excluded."""

        layer_count = (
            self.num_draft_layers
            if active_layer_count is None else int(active_layer_count)
        )
        component_count = (
            len(self.component_order)
            if decoded_components is None else int(decoded_components)
        )
        if not 0 <= layer_count <= self.num_draft_layers:
            raise ValueError(f"Invalid active layer count: {layer_count}")
        if not 1 <= component_count <= len(self.component_order):
            raise ValueError(
                f"Cannot decode {component_count} component(s) from "
                f"components={self.components!r}"
            )
        return int(
            self.source_dim * self.rank
            + layer_count * self.rank * self.draft_dim * component_count
        )

    def architecture_metadata(self) -> Dict[str, object]:
        return {
            "architecture": self.architecture,
            "components": self.components,
            "stored_components": list(self.component_order),
            "decoder_output_dim_per_layer": len(self.component_order) * self.draft_dim,
            "parameter_count": self.parameter_count(),
            "trainable_parameter_count": self.parameter_count(trainable_only=True),
            "dense_projection_macs_per_token_all_layers": (
                self.dense_projection_macs_per_token()
            ),
            "mac_definition": "Linear-layer multiply-accumulates; excludes LayerNorm and SiLU",
        }

    def config_dict(self) -> Dict[str, object]:
        return {
            "source_dim": self.source_dim,
            "draft_dim": self.draft_dim,
            "num_draft_layers": self.num_draft_layers,
            "rank": self.rank,
            "components": self.components,
            "architecture": self.architecture,
        }

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> "AttnBridgeMapper":
        # These checkpoints contain only tensors and experiment metadata. Pass
        # the policy explicitly because PyTorch 2.6 changed the default and
        # legacy metadata may include a TorchVersion object.
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch versions before the weights_only argument.
            payload = torch.load(path, map_location="cpu")
        config = dict(payload["mapper_config"])
        # Legacy checkpoints predate component/architecture metadata and always
        # used a concatenated [K,V] decoder. Infer from the stored decoder width
        # as an additional guard against hand-edited metadata.
        if "components" not in config:
            decoder_weight = payload["state_dict"].get("decoders.0.weight")
            if decoder_weight is None:
                raise KeyError("Checkpoint has no decoders.0.weight")
            output_dim = int(decoder_weight.shape[0])
            draft_dim = int(config["draft_dim"])
            if output_dim == 2 * draft_dim:
                config["components"] = "kv"
            elif output_dim == draft_dim:
                training_components = payload.get("training", {}).get("components")
                if training_components not in {"key", "value"}:
                    raise ValueError(
                        "Single-component checkpoint is missing unambiguous "
                        "component metadata"
                    )
                config["components"] = training_components
            else:
                raise ValueError(
                    f"Decoder output width {output_dim} is incompatible with "
                    f"draft_dim={draft_dim}"
                )
        config.setdefault("architecture", cls._ARCHITECTURES[config["components"]])
        mapper = cls(**config)
        mapper.load_state_dict(payload["state_dict"])
        output_semantics = str(
            payload.get("training", {}).get("output_semantics", "replacement")
        )
        if output_semantics not in {"replacement", "native_residual"}:
            raise ValueError(
                f"Unknown checkpoint output semantics: {output_semantics!r}"
            )
        mapper.output_semantics = output_semantics
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
        # Position 0 is deliberately retained as a native sink/anchor by the
        # deployment policy.  Exclude it from mapper supervision so training
        # and inference cover exactly the same replaceable positions.
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
    # Sparse cache-residual state is deliberately request-local.  Absolute
    # token positions can be reused by the next spgenerate call, but their
    # Target-V inputs and learned residuals cannot.
    model._attn_bridge_cache_value_residual_seen_refresh = False
    model._attn_bridge_cache_value_residual_delta = None
    model._attn_bridge_cache_value_residual_valid = None
    model._attn_bridge_cache_value_residual_capacity = None
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
        "value_residual_events": 0,
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
        "cache_value_residual_refresh_events": [],
        "cache_value_residual_mapper_events": [],
        "cache_value_residual_refresh_time_ms_cpu": 0.0,
        "cache_value_residual_mapper_time_ms_cpu": 0.0,
        "cache_value_residual_refreshes": 0,
        "cache_value_residual_initial_native_refreshes": 0,
        "cache_value_residual_mapper_forwards": 0,
        "cache_value_residual_mapper_unique_rows": 0,
        "cache_value_residual_cache_hits": 0,
        "cache_value_residual_cache_misses": 0,
        "cache_value_residual_applied_rows": 0,
        "cache_value_residual_mapper_macs": 0,
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
    per_token = mapper.dense_projection_macs_per_token(
        active_layer_count=active_layer_count,
        decoded_components=decoded_components,
    )
    runtime["mapper_macs"] += int(token_count * per_token)


def _require_cache_value_residual_contract(
    model,
    mapper: AttnBridgeMapper,
    config: Dict[str, object],
    retrieved_indices: torch.Tensor,
    working_kv: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[Dict[str, object], object, torch.Tensor, int, int, float]:
    """Validate the deliberately narrow sparse residual runtime contract.

    This mode is intentionally not a general cache-overwrite variant.  It is
    bound to a V-only, last-Draft-layer, aligned native-residual checkpoint so
    a replacement checkpoint or a broader policy cannot silently enter the
    additive path.
    """

    runtime = getattr(model, "bridge_runtime", None)
    if not isinstance(runtime, dict) or "cache_value_residual_refreshes" not in runtime:
        raise RuntimeError(
            "cache_value_residual requires reset_bridge_runtime before refresh"
        )
    if mapper is None:
        raise RuntimeError("cache_value_residual requires an AttnBridge mapper")
    if getattr(model, "bridge_collector", None) is not None:
        raise RuntimeError(
            "Bridge pair collection and cache_value_residual are mutually exclusive"
        )
    if str(config.get("compute_mode", "")) != "cache_value_residual":
        raise RuntimeError("Internal cache_value_residual dispatch mismatch")

    required_policy = {
        "components": "value",
        "layers": "last",
        "source_mode": "aligned",
        "output_semantics": "native_residual",
    }
    for field, expected in required_policy.items():
        observed = config.get(field)
        if observed != expected:
            raise ValueError(
                "cache_value_residual requires policy "
                f"{field}={expected!r}, got {observed!r}"
            )
    if str(getattr(mapper, "components", "")) != "value":
        raise RuntimeError(
            "cache_value_residual requires a physically V-only mapper checkpoint"
        )
    if str(getattr(mapper, "output_semantics", "")) != "native_residual":
        raise RuntimeError(
            "cache_value_residual requires checkpoint "
            "output_semantics='native_residual'"
        )
    if not mapper.supports_component("value"):
        raise RuntimeError("cache_value_residual mapper cannot decode values")

    alpha = float(config.get("alpha", 0.0))
    alpha_key = float(config.get("alpha_key", 0.0))
    alpha_value = float(config.get("alpha_value", alpha))
    if (
        not math.isfinite(alpha)
        or not math.isfinite(alpha_key)
        or not math.isfinite(alpha_value)
        or alpha_key != 0.0
        or alpha_value < 0.0
    ):
        raise ValueError(
            "cache_value_residual requires finite nonnegative alpha/alpha_value "
            "and alpha_key=0"
        )

    if not isinstance(retrieved_indices, torch.Tensor):
        raise TypeError("retrieved_indices must be a Tensor")
    if retrieved_indices.dtype != torch.long or retrieved_indices.ndim != 1:
        raise ValueError("retrieved_indices must be a rank-1 torch.long Tensor")
    if retrieved_indices.numel() <= 0:
        raise ValueError("cache_value_residual requires at least one selected row")
    if bool((retrieved_indices < 0).any()):
        raise IndexError("retrieved_indices contains a negative absolute position")
    if retrieved_indices.numel() > 1 and bool(
        (retrieved_indices[1:] <= retrieved_indices[:-1]).any()
    ):
        raise ValueError(
            "retrieved_indices must be strictly increasing and unique absolute positions"
        )

    if not isinstance(working_kv, (list, tuple)) or not working_kv:
        raise ValueError("cache_value_residual requires nonempty working_kv")
    layer_count = len(working_kv)
    if int(getattr(mapper, "num_draft_layers", -1)) != layer_count:
        raise RuntimeError(
            "Mapper/Draft layer-count mismatch: "
            f"mapper={getattr(mapper, 'num_draft_layers', None)}, "
            f"working_kv={layer_count}"
        )

    row_count = int(retrieved_indices.numel())
    draft_device = None
    draft_dtype = None
    last_native_value = None
    for layer_index, pair in enumerate(working_kv):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"working_kv layer {layer_index} is not a K/V pair")
        native_key, native_value = pair
        if not isinstance(native_key, torch.Tensor) or not isinstance(
            native_value, torch.Tensor
        ):
            raise TypeError(f"working_kv layer {layer_index} must contain Tensors")
        if (
            native_key.ndim != 4
            or native_value.ndim != 4
            or native_key.shape != native_value.shape
            or int(native_key.shape[0]) != 1
            or int(native_key.shape[2]) != row_count
        ):
            raise ValueError(
                f"Invalid working_kv shape at layer {layer_index}: "
                f"K={tuple(native_key.shape)}, V={tuple(native_value.shape)}, "
                f"selected_rows={row_count}"
            )
        if (
            native_key.device != native_value.device
            or native_key.dtype != native_value.dtype
            or not native_key.is_floating_point()
        ):
            raise RuntimeError(
                f"K/V device or dtype mismatch at Draft layer {layer_index}"
            )
        flat_dim = int(native_key.shape[1] * native_key.shape[3])
        if flat_dim != int(getattr(mapper, "draft_dim", -1)):
            raise RuntimeError(
                f"Mapper/Draft width mismatch at layer {layer_index}: "
                f"mapper={getattr(mapper, 'draft_dim', None)}, Draft={flat_dim}"
            )
        if draft_device is None:
            draft_device = native_value.device
            draft_dtype = native_value.dtype
        elif native_value.device != draft_device or native_value.dtype != draft_dtype:
            raise RuntimeError(
                "cache_value_residual requires every Draft working-cache layer "
                "on one device and dtype"
            )
        last_native_value = native_value

    assert draft_device is not None and last_native_value is not None
    if retrieved_indices.device != draft_device:
        raise RuntimeError(
            "retrieved_indices and Draft working cache must share a device, got "
            f"{retrieved_indices.device} and {draft_device}"
        )
    mapper_parameters = list(mapper.parameters())
    if not mapper_parameters:
        raise RuntimeError("cache_value_residual mapper has no parameters")
    mapper_devices = {parameter.device for parameter in mapper_parameters}
    if mapper_devices != {draft_device}:
        raise RuntimeError(
            "cache_value_residual requires mapper and Draft working cache on one "
            f"device, got mapper={sorted(map(str, mapper_devices))}, Draft={draft_device}"
        )

    target_cache = getattr(model, "past_key_values", None)
    if not isinstance(target_cache, (list, tuple)) or not target_cache:
        raise RuntimeError("Target cache must be populated before residual refresh")
    if not isinstance(target_cache[-1], (list, tuple)) or len(target_cache[-1]) < 2:
        raise RuntimeError("Last Target cache entry is not a K/V pair")
    target_value_cache = target_cache[-1][1]
    target_data = getattr(target_value_cache, "data", None)
    if not isinstance(target_data, torch.Tensor) or target_data.ndim != 4:
        raise RuntimeError("Last Target value cache must expose rank-4 Tensor data")
    if int(target_data.shape[0]) != 1 or not target_data.is_floating_point():
        raise RuntimeError("Last Target value cache must be floating point with batch=1")
    if target_data.device != draft_device:
        raise RuntimeError(
            "cache_value_residual requires Target-V, mapper, and Draft cache on "
            f"one device, got Target-V={target_data.device}, Draft={draft_device}"
        )
    source_dim = int(target_data.shape[1] * target_data.shape[3])
    if source_dim != int(getattr(mapper, "source_dim", -1)):
        raise RuntimeError(
            "Mapper/Target-V width mismatch: "
            f"mapper={getattr(mapper, 'source_dim', None)}, Target-V={source_dim}"
        )

    try:
        target_shape = tuple(int(dimension) for dimension in target_value_cache.shape)
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("Target value cache has no valid logical shape") from error
    if len(target_shape) != 4 or target_shape[:2] != tuple(target_data.shape[:2]) or (
        target_shape[3] != int(target_data.shape[3])
    ):
        raise RuntimeError(
            f"Target value cache logical/data shape mismatch: {target_shape} vs "
            f"{tuple(target_data.shape)}"
        )
    target_length = int(target_shape[2])
    capacity = int(getattr(model, "full_cache_budget", -1))
    total_seq_len = int(getattr(model, "total_seq_len", -1))
    if capacity <= 0:
        raise RuntimeError("full_cache_budget must be positive")
    if int(target_data.shape[2]) != capacity:
        raise RuntimeError(
            "Target cache capacity disagrees with full_cache_budget: "
            f"Target={target_data.shape[2]}, budget={capacity}"
        )
    if not 0 <= target_length <= capacity:
        raise RuntimeError(
            f"Target logical length {target_length} exceeds cache capacity {capacity}"
        )
    if not 0 <= total_seq_len <= capacity:
        raise RuntimeError(
            f"Draft total_seq_len {total_seq_len} exceeds cache capacity {capacity}"
        )
    maximum_position = int(retrieved_indices[-1].item())
    if maximum_position >= total_seq_len or maximum_position >= capacity:
        raise IndexError(
            "Selected absolute position exceeds populated Draft cache/capacity: "
            f"max={maximum_position}, total_seq_len={total_seq_len}, capacity={capacity}"
        )

    delta_cache = getattr(model, "_attn_bridge_cache_value_residual_delta", None)
    valid_cache = getattr(model, "_attn_bridge_cache_value_residual_valid", None)
    cached_capacity = getattr(
        model, "_attn_bridge_cache_value_residual_capacity", None
    )
    if (delta_cache is None) != (valid_cache is None):
        raise RuntimeError("Residual delta-cache tensors are only partially initialized")
    if delta_cache is not None:
        expected_shape = (capacity, int(mapper.draft_dim))
        if (
            tuple(delta_cache.shape) != expected_shape
            or delta_cache.device != draft_device
            or delta_cache.dtype != draft_dtype
            or tuple(valid_cache.shape) != (capacity,)
            or valid_cache.device != draft_device
            or valid_cache.dtype != torch.bool
            or cached_capacity != capacity
        ):
            raise RuntimeError(
                "Residual delta-cache shape/device/dtype/capacity changed within request"
            )

    return (
        runtime,
        target_value_cache,
        last_native_value,
        target_length,
        capacity,
        alpha_value,
    )


@torch.no_grad()
def _apply_cache_value_residual(
    model,
    mapper: AttnBridgeMapper,
    config: Dict[str, object],
    retrieved_indices: torch.Tensor,
    working_kv: List[Tuple[torch.Tensor, torch.Tensor]],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Add cached Target-V-derived deltas to selected last-layer Draft V rows."""

    (
        runtime,
        target_value_cache,
        last_native_value,
        target_length,
        capacity,
        alpha_value,
    ) = _require_cache_value_residual_contract(
        model, mapper, config, retrieved_indices, working_kv
    )
    device = last_native_value.device
    refresh_cpu_start = time.perf_counter()
    refresh_start = refresh_end = refresh_stream = None
    if device.type == "cuda":
        refresh_start = torch.cuda.Event(enable_timing=True)
        refresh_end = torch.cuda.Event(enable_timing=True)
        refresh_stream = torch.cuda.current_stream(device)
        refresh_start.record(refresh_stream)

    def finish_refresh() -> None:
        if refresh_end is not None:
            refresh_end.record(refresh_stream)
            runtime["cache_value_residual_refresh_events"].append(
                (refresh_start, refresh_end)
            )
        else:
            runtime["cache_value_residual_refresh_time_ms_cpu"] += (
                time.perf_counter() - refresh_cpu_start
            ) * 1000.0

    runtime["cache_value_residual_refreshes"] += 1
    if not bool(
        getattr(model, "_attn_bridge_cache_value_residual_seen_refresh", False)
    ):
        # No verifier-attention selection exists before the first working-cache
        # refresh.  Preserve the native cache exactly and do not even allocate
        # or invoke the mapper on this refresh.
        model._attn_bridge_cache_value_residual_seen_refresh = True
        runtime["cache_value_residual_initial_native_refreshes"] += 1
        finish_refresh()
        return working_kv

    recent_native = max(0, int(config.get("recent_native", 128)))
    cutoff = max(0, target_length - recent_native)
    eligible_mask = (retrieved_indices > 0) & (retrieved_indices < cutoff)
    if not bool(eligible_mask.any()):
        finish_refresh()
        return working_kv

    working_positions = torch.nonzero(eligible_mask, as_tuple=False).flatten()
    absolute_positions = retrieved_indices.index_select(0, working_positions)
    delta_cache = getattr(model, "_attn_bridge_cache_value_residual_delta", None)
    valid_cache = getattr(model, "_attn_bridge_cache_value_residual_valid", None)
    if delta_cache is None:
        delta_cache = torch.empty(
            (capacity, int(mapper.draft_dim)),
            device=device,
            dtype=last_native_value.dtype,
        )
        valid_cache = torch.zeros((capacity,), device=device, dtype=torch.bool)
        model._attn_bridge_cache_value_residual_delta = delta_cache
        model._attn_bridge_cache_value_residual_valid = valid_cache
        model._attn_bridge_cache_value_residual_capacity = capacity

    cache_hit_mask = valid_cache.index_select(0, absolute_positions)
    miss_positions = absolute_positions[~cache_hit_mask]
    hit_count = int(absolute_positions.numel() - miss_positions.numel())
    miss_count = int(miss_positions.numel())
    runtime["cache_value_residual_cache_hits"] += hit_count
    runtime["cache_value_residual_cache_misses"] += miss_count

    if miss_count:
        source = _flatten_target_values(target_value_cache, miss_positions)
        if source.device != device or int(source.shape[1]) != int(mapper.source_dim):
            raise RuntimeError(
                "Target-V miss gather produced an incompatible shape/device: "
                f"shape={tuple(source.shape)}, device={source.device}"
            )
        mapper_cpu_start = time.perf_counter()
        mapper_start = mapper_end = None
        if device.type == "cuda":
            mapper_start = torch.cuda.Event(enable_timing=True)
            mapper_end = torch.cuda.Event(enable_timing=True)
            mapper_start.record(refresh_stream)
        mapped_layers = mapper.forward_values(
            source, layer_indices={len(working_kv) - 1}
        )
        if mapper_end is not None:
            mapper_end.record(refresh_stream)
            event_pair = (mapper_start, mapper_end)
            runtime["map_events"].append(event_pair)
            runtime["cache_value_residual_mapper_events"].append(event_pair)
        else:
            runtime["cache_value_residual_mapper_time_ms_cpu"] += (
                time.perf_counter() - mapper_cpu_start
            ) * 1000.0
        if not isinstance(mapped_layers, (list, tuple)) or len(mapped_layers) != len(
            working_kv
        ):
            raise RuntimeError(
                "V-only mapper returned the wrong number of Draft-layer outputs"
            )
        if any(
            value is not None
            for value in mapped_layers[: len(working_kv) - 1]
        ):
            raise RuntimeError("V-only mapper materialized a non-last Draft layer")
        mapped_delta = mapped_layers[-1]
        expected_shape = (miss_count, int(mapper.draft_dim))
        if not isinstance(mapped_delta, torch.Tensor) or tuple(
            mapped_delta.shape
        ) != expected_shape:
            raise RuntimeError(
                "V-only mapper returned an invalid residual shape: "
                f"expected={expected_shape}, observed="
                f"{getattr(mapped_delta, 'shape', None)}"
            )
        if mapped_delta.device != device or not mapped_delta.is_floating_point():
            raise RuntimeError(
                "V-only mapper residual must be floating point on the Draft device"
            )
        mapped_delta = mapped_delta.to(dtype=last_native_value.dtype)
        delta_cache.index_copy_(0, miss_positions, mapped_delta)
        valid_cache.index_fill_(0, miss_positions, True)

        runtime["cache_value_residual_mapper_forwards"] += 1
        runtime["cache_value_residual_mapper_unique_rows"] += miss_count
        mapper_macs_before = int(runtime["mapper_macs"])
        _record_mapper_macs(runtime, mapper, miss_count, 1, 1)
        mapper_macs = int(runtime["mapper_macs"]) - mapper_macs_before
        runtime["cache_value_residual_mapper_macs"] += mapper_macs
        runtime["mapped_tokens"] += miss_count
        runtime["source_bytes"] += int(source.numel() * source.element_size())
        runtime["materialized_bridge_bytes"] += int(
            mapped_delta.numel() * mapped_delta.element_size()
        )

    base_delta = delta_cache.index_select(0, absolute_positions)
    head_count = int(last_native_value.shape[1])
    head_dim = int(last_native_value.shape[3])
    residual = base_delta.reshape(-1, head_count, head_dim).permute(
        1, 0, 2
    ).unsqueeze(0).contiguous()
    native_slice = last_native_value.index_select(2, working_positions)
    effective_value = torch.add(native_slice, residual, alpha=alpha_value)
    last_native_value.index_copy_(
        2,
        working_positions,
        effective_value,
    )
    # V diagnostics are deliberately opt-in because the FP32 reductions and
    # retained scalars perturb wall timing.  The evaluator marks such rows as
    # diagnostic and the hard-evaluation analyzer never uses their TPS.  For
    # the residual policy the meaningful comparison is effective V
    # (native + learned delta) versus the untouched native V at exactly the
    # rows that received the residual.
    if bool(config.get("collect_diagnostics", False)):
        prediction = effective_value.float().reshape(-1)
        teacher = native_slice.float().reshape(-1)
        runtime["cosine_tensors"].append(
            F.cosine_similarity(prediction, teacher, dim=0)
        )
        runtime["nmse_tensors"].append(
            (prediction - teacher).pow(2).mean()
            / teacher.pow(2).mean().clamp_min(1.0e-12)
        )
        runtime["diagnostic_weights"].append(int(absolute_positions.numel()))
    runtime["cache_value_residual_applied_rows"] += int(
        absolute_positions.numel()
    )
    finish_refresh()
    return working_kv


@torch.no_grad()
def prepare_prefill_value_bridge(
    model,
    draft_query_length: int,
) -> Optional[List[Optional[Dict[str, object]]]]:
    """Build V replacement or native-anchored residuals before Draft prefill.

    The target cache stays read-only. Native Draft K is still projected for
    every token, preserving the Draft QK coordinate system. Position 0 and a
    configurable recent suffix stay native; only the aligned middle prefix is
    decoded by the Target-V mapper. Replacement mode can bypass mapped rows of
    native v_proj. Residual mode deliberately retains the full native v_proj
    and adds a learned delta, so it makes no projection-saving claim.
    """

    mapper = getattr(model, "attn_bridge_mapper", None)
    config = getattr(model, "attn_bridge_config", None) or {}
    compute_mode = str(config.get("compute_mode", "cache_overwrite"))
    if compute_mode not in {
        "prefill_value_bypass",
        "prefill_value_shadow",
        "prefill_value_residual",
    }:
        return None
    if mapper is None:
        raise RuntimeError("Inline prefill V replacement requires an AttnBridge mapper")
    if getattr(model, "bridge_collector", None) is not None:
        raise RuntimeError("Bridge pair collection and inline V replacement are mutually exclusive")

    components = str(config.get("components", "value"))
    alpha = float(config.get("alpha", 0.0))
    alpha_key = float(config.get("alpha_key", 0.0 if components == "value" else alpha))
    alpha_value = float(config.get("alpha_value", alpha))
    residual_mode = compute_mode == "prefill_value_residual"
    expected_semantics = "native_residual" if residual_mode else "replacement"
    mapper_semantics = str(getattr(mapper, "output_semantics", "replacement"))
    if mapper_semantics != expected_semantics:
        raise RuntimeError(
            "Bridge compute mode and Mapper checkpoint semantics disagree: "
            f"mode={compute_mode!r} requires {expected_semantics!r}, "
            f"Mapper declares {mapper_semantics!r}"
        )
    if (
        components != "value"
        or not math.isfinite(alpha_key)
        or not math.isfinite(alpha_value)
        or alpha_key != 0.0
        or alpha_value < 0.0
        or (not residual_mode and abs(alpha_value - 1.0) > 1.0e-8)
    ):
        raise ValueError(
            "inline prefill V requires components='value', alpha_key=0; "
            "replacement modes additionally require alpha_value=1"
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
    # Position 0 stays native as the fixed sink/anchor used by calibration.
    # The target-selected root token beyond target_length also remains native
    # automatically.
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
            "shadow_compute": compute_mode in {
                "prefill_value_shadow", "prefill_value_residual"
            },
            "combine": "native_residual" if residual_mode else "replace",
            "residual_scale": alpha_value if residual_mode else 1.0,
            # Diagnostics are deliberately evaluated inside Draft attention,
            # where the native hidden states needed for a V teacher exist.
            # The extra teacher projection is diagnostic-only and is excluded
            # from the clean-path projection/MAC accounting below.
            "collect_diagnostics": bool(config.get("collect_diagnostics", False)),
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
        if residual_mode:
            runtime["value_residual_events"] += 1
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
    compute_mode = str(config.get("compute_mode", "cache_overwrite"))
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

    if compute_mode == "cache_value_residual":
        return _apply_cache_value_residual(
            model, mapper, config, retrieved_indices, working_kv
        )
    if mapper is None or max(alpha_key, alpha_value) <= 0.0 or not working_kv:
        return working_kv
    if compute_mode in {
        "prefill_value_bypass",
        "prefill_value_shadow",
        "prefill_value_residual",
    }:
        # Inline replacement/residual already persisted its effective V into
        # full_draft_kv. Re-mapping here would apply it twice and blur the
        # shadow-vs-bypass/residual control.
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
    if alpha_key > 0.0 and not mapper.supports_component("key"):
        raise RuntimeError(
            f"Bridge policy requests K, but checkpoint contains "
            f"components={mapper.components!r}"
        )
    if alpha_value > 0.0 and not mapper.supports_component("value"):
        raise RuntimeError(
            f"Bridge policy requests V, but checkpoint contains "
            f"components={mapper.components!r}"
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
    cache_residual_refresh_events = runtime.get(
        "cache_value_residual_refresh_events", []
    )
    for _, end in cache_residual_refresh_events:
        end.synchronize()
    cache_residual_refresh_time_ms = (
        sum(
            start.elapsed_time(end)
            for start, end in cache_residual_refresh_events
        )
        if cache_residual_refresh_events
        else float(runtime.get("cache_value_residual_refresh_time_ms_cpu", 0.0))
    )
    cache_residual_mapper_events = runtime.get(
        "cache_value_residual_mapper_events", []
    )
    for _, end in cache_residual_mapper_events:
        end.synchronize()
    cache_residual_mapper_time_ms = (
        sum(
            start.elapsed_time(end)
            for start, end in cache_residual_mapper_events
        )
        if cache_residual_mapper_events
        else float(runtime.get("cache_value_residual_mapper_time_ms_cpu", 0.0))
    )
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
        "bridge_cache_value_residual_refreshes": int(
            runtime.get("cache_value_residual_refreshes", 0)
        ),
        "bridge_cache_value_residual_initial_native_refreshes": int(
            runtime.get("cache_value_residual_initial_native_refreshes", 0)
        ),
        "bridge_cache_value_residual_mapper_forwards": int(
            runtime.get("cache_value_residual_mapper_forwards", 0)
        ),
        "bridge_cache_value_residual_mapper_unique_rows": int(
            runtime.get("cache_value_residual_mapper_unique_rows", 0)
        ),
        "bridge_cache_value_residual_cache_hits": int(
            runtime.get("cache_value_residual_cache_hits", 0)
        ),
        "bridge_cache_value_residual_cache_misses": int(
            runtime.get("cache_value_residual_cache_misses", 0)
        ),
        "bridge_cache_value_residual_applied_rows": int(
            runtime.get("cache_value_residual_applied_rows", 0)
        ),
        "bridge_cache_value_residual_mapper_macs": int(
            runtime.get("cache_value_residual_mapper_macs", 0)
        ),
        "bridge_cache_value_residual_mapper_time_ms": float(
            cache_residual_mapper_time_ms
        ),
        "bridge_cache_value_residual_refresh_time_ms": float(
            cache_residual_refresh_time_ms
        ),
        "bridge_cache_value_residual_cached_rows": int(
            runtime.get("cache_value_residual_mapper_unique_rows", 0)
        ),
        "bridge_cache_value_residual_capacity_rows": int(
            getattr(model, "_attn_bridge_cache_value_residual_capacity", 0) or 0
        ),
        "bridge_value_bypass_events": int(runtime.get("value_bypass_events", 0)),
        "bridge_value_shadow_events": int(runtime.get("value_shadow_events", 0)),
        "bridge_value_residual_events": int(runtime.get("value_residual_events", 0)),
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
