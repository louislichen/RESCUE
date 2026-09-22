# -*- coding: utf-8 -*-
"""Qwen2 attention hooks for the frozen Adapter prefill-only protocol."""
from __future__ import annotations

import contextlib
import math
import types
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


_PREFILL_PHASES = {"TRAIN_PREFILL", "VALIDATION_PREFILL", "CONFIRMATION_PREFILL"}
_DECODE_PHASE = "STUDENT_CACHED_DECODE"


@dataclass
class AdapterSteeringState:
    layer_to_prior: Dict[int, torch.Tensor]
    layer_to_alpha: Dict[int, torch.Tensor]
    eps: float
    enabled: bool = True
    trace: Optional[list[dict[str, Any]]] = None
    capture_rows: Optional[dict[int, torch.Tensor]] = None
    phase: str = "TRAIN_PREFILL"
    invocation_count: int = 0


def _get_state(module) -> Optional[AdapterSteeringState]:
    model_ref = getattr(module, "_attn_adapter_model_ref", None)
    return None if model_ref is None else getattr(model_ref, "_attn_adapter_state", None)


def _layer_idx(module) -> int:
    idx = getattr(module, "layer_idx", None)
    if idx is None:
        idx = getattr(module, "_attn_adapter_layer_idx", None)
    if idx is None:
        raise RuntimeError("Qwen2 attention layer index is unavailable")
    return int(idx)


def _fit_prior(prior: torch.Tensor, key_len: int, device, eps: float) -> torch.Tensor:
    prior = prior.to(device=device, dtype=torch.float32)
    if prior.dim() != 2:
        raise ValueError(f"adapter prior must be [heads,key_len], got {tuple(prior.shape)}")
    if prior.shape[-1] < key_len:
        pad = torch.full(
            (prior.shape[0], key_len - prior.shape[-1]),
            eps,
            device=device,
            dtype=prior.dtype,
        )
        prior = torch.cat((prior, pad), dim=-1)
    prior = prior[:, :key_len]
    return prior / prior.sum(dim=-1, keepdim=True).clamp_min(eps)


def _apply_adapter_prob_mix(module, attn_probs: torch.Tensor) -> torch.Tensor:
    state = _get_state(module)
    if state is None or not state.enabled:
        return attn_probs
    if state.phase not in _PREFILL_PHASES:
        raise RuntimeError(f"Adapter intervention is forbidden in phase {state.phase}")
    idx = _layer_idx(module)
    if idx not in state.layer_to_prior:
        return attn_probs
    state.invocation_count += 1
    prior = _fit_prior(
        state.layer_to_prior[idx], attn_probs.shape[-1], attn_probs.device, state.eps
    )
    alpha = state.layer_to_alpha[idx].to(
        device=attn_probs.device, dtype=torch.float32
    )
    if prior.shape[0] != attn_probs.shape[1] or alpha.numel() != attn_probs.shape[1]:
        raise ValueError("adapter head count does not match Qwen2 attention heads")
    row = attn_probs[:, :, -1, :].float()
    mixed = (
        (1.0 - alpha.view(1, -1, 1)) * row
        + alpha.view(1, -1, 1) * prior.view(1, prior.shape[0], -1)
    )
    denominator = mixed.sum(dim=-1, keepdim=True)
    mixed = mixed / denominator.clamp_min(state.eps)
    out = torch.cat(
        (attn_probs[:, :, :-1, :], mixed.to(attn_probs.dtype).unsqueeze(2)),
        dim=2,
    )
    if state.trace is not None:
        state.trace.append(
            {
                "stage": "mixed_probs",
                "layer": idx,
                "phase": state.phase,
                "mixed_requires_grad": bool(mixed.requires_grad),
                "output_requires_grad": bool(out.requires_grad),
                "denominator_finite": bool(torch.isfinite(denominator).all().item()),
                "adapter_invocation_count": state.invocation_count,
            }
        )
    return out


@contextlib.contextmanager
def qwen2_adapter_steering(
    model,
    layer_to_prior: Dict[int, torch.Tensor],
    layer_to_alpha: Dict[int, torch.Tensor],
    eps: float,
    trace: Optional[list[dict[str, Any]]] = None,
    capture_rows: Optional[dict[int, torch.Tensor]] = None,
    selected_layers: Optional[set[int]] = None,
    phase: str = "TRAIN_PREFILL",
):
    """Install instance-only Qwen2 hooks for prefill, or explicit stock decode."""
    try:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
    except Exception as exc:
        raise RuntimeError("This experiment currently supports Qwen2 models only") from exc
    if phase not in _PREFILL_PHASES | {_DECODE_PHASE}:
        raise ValueError(f"unsupported explicit protocol phase: {phase}")
    if phase == _DECODE_PHASE:
        if trace is not None:
            trace.append(
                {
                    "stage": "decode_adapter_disabled",
                    "phase": phase,
                    "adapter_invocation_count": 0,
                    "teacher_calls": 0,
                }
            )
        yield
        return

    def patched_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        q_len, key_len = query_states.shape[-2], key_states.shape[-2]
        prefix = query_states[:, :, :0, :]
        if q_len > 1:
            prefix_mask = (
                None
                if attention_mask is None
                else attention_mask[:, :, : q_len - 1, :key_len]
            )
            prefix = F.scaled_dot_product_attention(
                query_states[:, :, :-1, :],
                key_states,
                value_states,
                attn_mask=prefix_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=prefix_mask is None,
            )
        query_last = query_states[:, :, -1:, :].float()
        scores = torch.matmul(
            query_last, key_states.float().transpose(2, 3)
        ) / math.sqrt(float(query_last.shape[-1]))
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, -1:, :key_len].float()
        last_probs = F.softmax(scores, dim=-1, dtype=torch.float32)
        state = _get_state(self)
        if state is not None and state.capture_rows is not None:
            state.capture_rows[_layer_idx(self)] = (
                last_probs[0, :, 0, :].detach().float()
            )
        last_probs = _apply_adapter_prob_mix(self, last_probs)
        last_probs = F.dropout(
            last_probs,
            p=self.attention_dropout,
            training=self.training,
        )
        last_output = torch.matmul(
            last_probs.float(), value_states.float()
        )
        attn_output = torch.cat(
            (prefix, last_output.to(query_states.dtype)), dim=2
        )
        if state is not None and state.trace is not None:
            state.trace.append(
                {
                    "stage": "value_aggregation",
                    "layer": _layer_idx(self),
                    "query_length": int(q_len),
                    "last_row_only_fp32": True,
                    "last_probs_finite": bool(
                        torch.isfinite(last_probs).all().item()
                    ),
                    "last_output_finite": bool(
                        torch.isfinite(last_output).all().item()
                    ),
                    "output_requires_grad": bool(attn_output.requires_grad),
                }
            )
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .reshape(*input_shape, -1)
        )
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    patch_layers = set(layer_to_prior) if selected_layers is None else set(selected_layers)
    backbone = model if hasattr(model, "layers") else getattr(model, "model")
    originals = []
    for index, layer in enumerate(backbone.layers):
        if index not in patch_layers:
            continue
        attention = layer.self_attn
        attention._attn_adapter_layer_idx = index
        attention._attn_adapter_model_ref = model
        originals.append((attention, attention.forward))
    state = AdapterSteeringState(
        layer_to_prior,
        layer_to_alpha,
        eps,
        trace=trace,
        capture_rows=capture_rows,
        phase=phase,
    )
    model._attn_adapter_state = state
    for attention, _ in originals:
        attention.forward = types.MethodType(patched_forward, attention)
    if trace is not None:
        trace.append(
            {
                "stage": "patch_installed",
                "phase": phase,
                "selected_layers": sorted(patch_layers),
                "global_class_patch": False,
            }
        )
    try:
        yield
    finally:
        for attention, original in originals:
            attention.forward = original
        model._attn_adapter_state = None


@contextlib.contextmanager
def qwen2_teacher_row_observer(
    model,
    observed_layers,
    captured_rows,
    trace=None,
):
    """Observe selected final-query rows without global output_attentions."""
    try:
        from transformers.models.qwen2.modeling_qwen2 import (
            apply_rotary_pos_emb,
            repeat_kv,
        )
    except Exception as exc:
        raise RuntimeError("This experiment currently supports Qwen2 models only") from exc

    def observed_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=attention_mask is None,
        )
        scale = 1.0 / math.sqrt(float(query_states.shape[-1]))
        scores = (
            torch.matmul(
                query_states[:, :, -1:, :].float(),
                key_states.float().transpose(2, 3),
            )
            * scale
        )
        if attention_mask is not None:
            scores = scores + attention_mask[
                :, :, -1:, : key_states.shape[-2]
            ].float()
        row = F.softmax(scores, dim=-1, dtype=torch.float32)[
            0, :, 0, :
        ].detach().cpu()
        layer = _layer_idx(self)
        captured_rows[layer] = row
        if trace is not None:
            trace.append(
                {
                    "stage": "teacher_streaming_row",
                    "layer": layer,
                    "row_shape": list(row.shape),
                    "row_finite": bool(torch.isfinite(row).all().item()),
                }
            )
        output = output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        return self.o_proj(output), None

    wanted = set(observed_layers)
    backbone = model if hasattr(model, "layers") else getattr(model, "model")
    originals = []
    for index, layer in enumerate(backbone.layers):
        if index not in wanted:
            continue
        attention = layer.self_attn
        attention._attn_adapter_layer_idx = index
        originals.append((attention, attention.forward))
        attention.forward = types.MethodType(observed_forward, attention)
    try:
        yield
    finally:
        for attention, original in originals:
            attention.forward = original
