#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Qwen2 attention patch for trainable adapter priors."""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class AdapterSteeringState:
    layer_to_prior: Dict[int, torch.Tensor]
    layer_to_alpha: Dict[int, torch.Tensor]
    eps: float
    enabled: bool = True


def _get_state(module) -> Optional[AdapterSteeringState]:
    model_ref = getattr(module, "_attn_adapter_model_ref", None)
    if model_ref is None:
        return None
    return getattr(model_ref, "_attn_adapter_state", None)


def _layer_idx(module) -> Optional[int]:
    idx = getattr(module, "layer_idx", None)
    if idx is None:
        idx = getattr(module, "_attn_adapter_layer_idx", None)
    return idx


def _fit_prior(prior: torch.Tensor, key_len: int, device, eps: float) -> torch.Tensor:
    prior = prior.to(device=device, dtype=torch.float32)
    if prior.dim() != 2:
        raise ValueError(f"adapter prior must be [heads,key_len], got {tuple(prior.shape)}")
    if prior.shape[-1] < key_len:
        pad = torch.full((prior.shape[0], key_len - prior.shape[-1]), eps, device=device, dtype=prior.dtype)
        prior = torch.cat([prior, pad], dim=-1)
    prior = prior[:, :key_len]
    return prior / prior.sum(dim=-1, keepdim=True).clamp_min(eps)


def _apply_adapter_prob_mix(module, attn_probs: torch.Tensor) -> torch.Tensor:
    state = _get_state(module)
    if state is None or not state.enabled:
        return attn_probs
    idx = _layer_idx(module)
    if idx not in state.layer_to_prior:
        return attn_probs

    prior = _fit_prior(state.layer_to_prior[idx], attn_probs.shape[-1], attn_probs.device, state.eps)
    alpha = state.layer_to_alpha[idx].to(device=attn_probs.device, dtype=torch.float32)
    if prior.shape[0] != attn_probs.shape[1]:
        raise ValueError(f"prior has {prior.shape[0]} heads, attention has {attn_probs.shape[1]}")
    if alpha.numel() != attn_probs.shape[1]:
        raise ValueError(f"alpha has {alpha.numel()} heads, attention has {attn_probs.shape[1]}")

    row = attn_probs[:, :, -1, :].float()
    mixed = (1.0 - alpha.view(1, -1, 1)) * row + alpha.view(1, -1, 1) * prior.view(1, prior.shape[0], -1)
    mixed = mixed / mixed.sum(dim=-1, keepdim=True).clamp_min(state.eps)
    # A BF16 softmax row need not sum to exactly one in FP32. Renormalizing
    # even at alpha=0 can change its rounded values; zero intervention must
    # preserve the original row exactly.
    mixed = torch.where(alpha.view(1, -1, 1) == 0, row, mixed)
    out = attn_probs.clone()
    out[:, :, -1, :] = mixed.to(attn_probs.dtype)
    return out


@contextlib.contextmanager
def qwen2_adapter_steering(
    model,
    layer_to_prior: Dict[int, torch.Tensor],
    layer_to_alpha: Dict[int, torch.Tensor],
    eps: float,
):
    """Patch Qwen2Attention.forward to mix adapter priors into the last query row."""
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb, repeat_kv
    except Exception as exc:
        raise RuntimeError("This experiment currently supports Qwen2 models only.") from exc

    original_forward = Qwen2Attention.forward

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if kwargs.get("padding_mask") is not None:
            attention_mask = kwargs["padding_mask"]

        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError("Qwen2Attention layer_idx is required when using cache.")
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be {(bsz, self.num_heads, q_len, kv_seq_len)}, "
                f"but got {tuple(attn_weights.size())}"
            )
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = _apply_adapter_prob_mix(self, attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be {(bsz, self.num_heads, q_len, self.head_dim)}, "
                f"but got {tuple(attn_output.size())}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value

    for idx, layer in enumerate(model.model.layers):
        layer.self_attn._attn_adapter_layer_idx = idx
        layer.self_attn._attn_adapter_model_ref = model

    model._attn_adapter_state = AdapterSteeringState(layer_to_prior, layer_to_alpha, eps)
    Qwen2Attention.forward = patched_forward
    try:
        yield
    finally:
        Qwen2Attention.forward = original_forward
        model._attn_adapter_state = None
