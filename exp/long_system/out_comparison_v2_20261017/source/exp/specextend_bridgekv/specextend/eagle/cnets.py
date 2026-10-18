# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import copy
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN

from shared.opt_tree import Tree
from termcolor import colored

from flash_attn import flash_attn_func

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # Assume cos, sin shape: (1, 1, seq_len, head_dim)
    # and x is shape: (batch, num_heads, seq_len, head_dim)
    # Flatten the batch dimension of position_ids: shape (seq_len,)
    idx = position_ids.squeeze(0)
    # Index along the sequence length dimension (dim=2)
    cos_indexed = torch.index_select(cos, dim=2, index=idx)  # shape: (1, 1, seq_len, head_dim)
    sin_indexed = torch.index_select(sin, dim=2, index=idx)
    
    # Optionally expand to match x's number of heads if needed.
    # For example, if x has shape (B, H, seq_len, head_dim) and cos_indexed is (1, 1, seq_len, head_dim),
    # then:
    cos_indexed = cos_indexed.expand(x.size(0), x.size(1), -1, -1)
    sin_indexed = sin_indexed.expand(x.size(0), x.size(1), -1, -1)
    
    x_embed = (x * cos_indexed) + (rotate_half(x) * sin_indexed)
    return x_embed


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        past_key_position_ids: bool = False,
        init: bool = False,
        draft_use_flash_prefill = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Prefill
        if init:
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            
            kv_seq_len = key_states.shape[-2]
            
            if past_key_value is not None:
                kv_seq_len += past_key_value[0].shape[-2]
                
            # Concatenate and return BEFORE rotating
            if past_key_value is not None:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

            past_key_value = (key_states, value_states) if use_cache else None

            query_position_ids = torch.arange(
                        kv_seq_len - q_len,  # past_length
                        kv_seq_len,           # past_length + q_len
                        device=key_states.device
                    ).unsqueeze(0)
            key_position_ids = torch.arange(0, kv_seq_len, device=key_states.device).unsqueeze(0)

            past_key_position_ids = key_position_ids
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

            query_states = apply_rotary_pos_emb_single(query_states, cos, sin, query_position_ids)
            key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)

            # repeat k/v heads if n_kv_heads < n_heads
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            if draft_use_flash_prefill:
                query_states = query_states.transpose(1,2)
                key_states = key_states.transpose(1,2)
                value_states = value_states.transpose(1,2)

                attn_output = flash_attn_func(query_states, key_states, value_states, 
                                              window_size=(512,-1),
                                              causal=True)
                
                attn_output = attn_output.contiguous()
                attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
                attn_output = self.o_proj(attn_output)
                
            else:
                if query_states.device.type == "cuda" and attention_mask is not None:
                    query_states = query_states.contiguous()
                    key_states = key_states.contiguous()
                    value_states = value_states.contiguous()

                is_causal = True if attention_mask is None and q_len > 1 else False

                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=attention_mask.to(dtype=query_states.dtype),
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=is_causal,
                )
                
                if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                    raise ValueError(
                        f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                        f" {attn_output.size()}"
                    )

                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
                attn_output = self.o_proj(attn_output)

        # Init forward / tree attention
        else:
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            
            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                cache_len = past_key_value[0].shape[-2]
                kv_seq_len += cache_len
                
            # Concatenate and return BEFORE rotating
            if past_key_value is not None:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

            past_key_value = (key_states, value_states) if use_cache else None

            # for prefill / init forward
            if position_ids is None:
                query_position_ids = torch.arange(
                            kv_seq_len - q_len,  # past_length
                            kv_seq_len,           # past_length + q_len
                            device=key_states.device
                        ).unsqueeze(0)
                key_position_ids = torch.arange(0, kv_seq_len, device=key_states.device).unsqueeze(0)

            # for tree drafting 
            else:
                query_position_ids = position_ids.unsqueeze(0)
                if past_key_position_ids is not None:
                    key_position_ids = torch.cat([past_key_position_ids, query_position_ids], dim=1)
                else:
                    key_position_ids = torch.arange(0, kv_seq_len, device='cuda').unsqueeze(0)

            past_key_position_ids = key_position_ids
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

            query_states = apply_rotary_pos_emb_single(query_states, cos, sin, query_position_ids)
            key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)

            # repeat k/v heads if n_kv_heads < n_heads
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, past_key_position_ids


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config,index):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.index=index
        if self.index!=0:
            self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_position_ids=None,
        init: bool = False,
        draft_use_flash_prefill = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        if self.index != 0:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value, past_key_position_ids = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            past_key_position_ids=past_key_position_ids,
            init=init,
            draft_use_flash_prefill=draft_use_flash_prefill
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs, past_key_position_ids

class I(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.ones(1, dtype=torch.float32))
    def forward(self,x):
        return x + self.dummy - self.dummy #(also tried x+self.dummy)

def len_list(x,n):
    return [i for i in x if len(i)<=n]

class Model(nn.Module):
    def __init__(self,config,load_emb=False,path=None,bias=True):
        super().__init__()

        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        if load_emb:
            from safetensors import safe_open
            import json
            try:
                with open(os.path.join(path,"model.safetensors.index.json"),"r") as f:
                    index_json=json.loads(f.read())
                    emb_path=index_json["weight_map"]["model.embed_tokens.weight"]
                with safe_open(os.path.join(path,emb_path),
                               framework="pt",
                               device="cpu") as f:
                    tensor_slice = f.get_slice("model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()
            except:
                with open(os.path.join(path, "pytorch_model.bin.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                weights=torch.load(os.path.join(path,emb_path))
                tensor=weights["model.embed_tokens.weight"].float()
            self.embed_tokens.weight.data = tensor

        self.layers = nn.ModuleList([LlamaDecoderLayer(config,index) for index in range(config.num_hidden_layers)])
        self.fc=nn.Linear(2*config.hidden_size,config.hidden_size,bias=bias)
        self.act=ACT2FN[config.hidden_act]
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

        self.past_key_position_ids = None

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                #inputs_embeds.dtype,
                torch.float32, # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, torch.float32, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        # [MODIFIED] add tree mask
        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            tree_len = tree_mask.size(-1)
            combined_attention_mask[:, :, -tree_len:, -tree_len:][
                tree_mask == 0
                ] = torch.finfo(torch.float32).min

        return combined_attention_mask

    def forward(
        self,
        hidden_states,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        tree_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        init: bool = False,
        draft_use_flash_prefill=False
    ):
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
            
        if tree_attention_mask is None:
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
                )
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
            )
        else:
            attention_mask=tree_attention_mask

        inputs_embeds=inputs_embeds.to(hidden_states.dtype)
        hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))

        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                )
            else:
                layer_outputs, past_key_position_ids = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    init=init,
                    past_key_position_ids=self.past_key_position_ids,
                    draft_use_flash_prefill=draft_use_flash_prefill
                )

            hidden_states = layer_outputs[0]

            self.past_key_position_ids = past_key_position_ids
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if use_cache:
            return hidden_states,next_decoder_cache

        return hidden_states

    @torch.no_grad()
    def generate(self,hidden_states,input_ids,head,max_length=4,use_cache=False):
        return_input_ids=copy.deepcopy(input_ids[0].tolist())
        input_ids=input_ids[:,1:]

        #input_ids=input_ids.to(hidden_states.device)
        if use_cache:
            past_key_values=None
            for i in range(max_length):
                if past_key_values!=None:
                    out_hidden,past_key_values = self(out_hidden[:, -1:], input_ids=torch.tensor([[token]]).to(input_ids.device),past_key_values=past_key_values,use_cache=True)
                else:
                    out_hidden, past_key_values = self(hidden_states, input_ids=input_ids,use_cache=True)
                last_hidden = out_hidden[:, -1]
                last_headout = head(last_hidden)
                token = torch.argmax(last_headout)
                #input_ids = torch.cat((input_ids, torch.tensor([[token]]).to(input_ids.device)), dim=1)
                return_input_ids.append(token.item())
                if token == 2:
                    break
                #hidden_states = torch.cat((hidden_states, out_hidden[:, -1:]), dim=1)
        else:
            for i in range(max_length):
                out_hidden=self(hidden_states,input_ids=input_ids)
                last_hidden = out_hidden[:, -1]
                last_headout = head(last_hidden)
                token = torch.argmax(last_headout)
                return_input_ids.append(token.item())
                input_ids = torch.cat((input_ids, torch.tensor([[token]]).to(input_ids.device)), dim=1)
                if token==2:
                    break
                hidden_states = torch.cat((hidden_states, out_hidden[:, -1:]), dim=1)

        return return_input_ids

    @torch.no_grad()
    def repeat_kv(self,kv,numr):
        newkv=[]
        for i in kv:
            newkv.append((i[0].repeat(numr,1,1,1),i[1].repeat(numr,1,1,1)))
        return tuple(newkv)

    @torch.no_grad()
    def reduce_kv(self,kv,numr):
        newkv=[]
        for i in kv:
            newkv.append((i[0][:numr],i[1][:numr]))
        return tuple(newkv)


    def reset_kv(self):
        self.draft_stable_kv=None


    def process_tree_mask(self,tree_attention_mask,init_len):
        attention_mask=torch.full((tree_attention_mask.size(0), init_len), 0, device=tree_attention_mask.device)
        tree_mask = torch.where(tree_attention_mask == 0, torch.finfo(torch.float32).min, 0)
        attention_mask=torch.cat([attention_mask,tree_mask],dim=-1)
        attention_mask = attention_mask[None, None, :, :]
        return attention_mask

    @torch.no_grad()
    def topK_genrate(self, hidden_states, input_ids, head, nodes, threshold=0.5, max_depth=10,print_time=False):
        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)
        len_posi = input_ids.shape[1]
        
        # Initial forward with draft model
        if hasattr(self, "draft_stable_kv") and self.draft_stable_kv is not None:
            if self.use_retrieval_cache:
                full_kv_len = self.total_seq_len
                out_hidden, past_key_values = self(
                    hidden_states, 
                    input_ids=input_ids[:, full_kv_len:],
                    past_key_values=self.draft_stable_kv, 
                    use_cache=True,
                    draft_use_flash_prefill = self.draft_use_flash_prefill
                    )
            else:
                kv_len = self.draft_stable_kv[0][0].shape[2]
                out_hidden, past_key_values = self(hidden_states, 
                                                input_ids=input_ids[:, kv_len:],
                                                past_key_values=self.draft_stable_kv, 
                                                use_cache=True,
                                                draft_use_flash_prefill=self.draft_use_flash_prefill
                                                )
        # Prefill draft model
        else:
            out_hidden, past_key_values = self(hidden_states, 
                                            input_ids=input_ids, 
                                            use_cache=True,
                                            init=True,
                                            draft_use_flash_prefill=self.draft_use_flash_prefill)
        if self.use_retrieval_cache:
            newly_appended_len = input_ids.shape[-1] - self.total_seq_len
            self.update_full_draft_cache(past_key_values, tokens_appended=newly_appended_len)
            self.draft_stable_kv = self.update_working_cache_retrieval_main(top_k_chunks=self.retrieve_top_k)
        else:
            self.draft_stable_kv = past_key_values
        
        past_key_values=self.draft_stable_kv
        
        # new total length of kv cache after initial forward
        init_len=past_key_values[0][0].size(2)

        if self.use_retrieval_cache:
            target_model_pos_diff = len_posi - (init_len - 1) 

        last_hidden = out_hidden[:, -1]
        if not self.diff_device:
            last_headout = head(last_hidden)
        else:
            if hasattr(self, "layer_device"):
                last_headout = head(last_hidden)
                last_headout = last_headout.to(self.layer_device)
            else:
                last_headout = F.linear(last_hidden, self.headweight)

        # Root distribution used to grow this draft tree.  The profiling
        # runner compares it with the target model's root distribution.
        self.last_draft_root_logits = last_headout.detach()

        tree = Tree(nodes, hidden_states.device,threshold,max_depth)
        logits = last_headout.unsqueeze(0)

        step = 0
        while True:
            tree_output = tree.update(
                torch.softmax(logits.to(hidden_states.device), dim=-1, dtype=torch.float32))

            input_ids = tree_output["input_ids"].unsqueeze(0)

            if self.use_retrieval_cache:
                position_ids = tree_output["position_ids"] + init_len-1
            else:
                position_ids = tree_output["position_ids"] + len_posi

            if tree_output["is_final"]:
                break
            tree_attention_mask_with_kv=self.process_tree_mask(tree_output["attention_mask"],init_len)

            if step==0:
                hidden_states=last_hidden.repeat(1,nodes,1)
            else:
                hidden_states=out_hidden[:,tree_output["parent_last"],:]

            # tree attention with draft model (pass last hidden states)
            out_hidden, past_key_values = self(hidden_states, 
                                               input_ids=input_ids,
                                               tree_attention_mask=tree_attention_mask_with_kv,
                                               past_key_values=past_key_values,
                                               position_ids=position_ids-1, 
                                               use_cache=True,
                                               draft_use_flash_prefill = self.draft_use_flash_prefill
                                               )

            if not self.diff_device:
                last_headout = head(out_hidden[0])
            else:
                if hasattr(self, "layer_device"):
                    last_headout = head(out_hidden[0])
                    last_headout = last_headout.to(self.layer_device)
                else:
                    last_headout = F.linear(out_hidden[0], self.headweight)

            logits = last_headout.unsqueeze(0)
            step += 1

        if self.use_retrieval_cache:
            position_ids += target_model_pos_diff

        return input_ids, position_ids, tree_output["attention_mask"], tree_output["parent_last"]

    @torch.no_grad()
    def test_autoregressive_generation(self, out_hidden, next_token, head, past_key_values, num_tokens=10):
        # We'll assume that self.draft_stable_kv has been updated already.
        # device = next_token.device
        device = out_hidden.device
        
        # Set the current past KV to the updated working cache.
        current_past = past_key_values
        
        # We'll collect the generated tokens here.
        generated_tokens = [next_token.item()]

        # Loop for num_tokens steps.
        for _ in range(num_tokens):
            # Run a forward pass with the current input.
            out_hidden, past_key_values = self(
                out_hidden[:,-1:],
                input_ids=next_token,
                past_key_values=current_past,
                use_cache=True
            )
            # outputs[0] is the logits, outputs[1] is the updated past key/values.
            last_hidden = out_hidden[:,-1]
            if not self.diff_device:
                last_headout = head(last_hidden)
            else:
                if hasattr(self, "layer_device"):
                    last_headout = head(last_hidden)
                    last_headout = last_headout.to(self.layer_device)
                else:
                    last_headout = F.linear(last_hidden, self.headweight)
            logits = last_headout.unsqueeze(0)

            current_past = past_key_values  # update the past KV for the next step

            probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
            next_token = torch.multinomial(probabilities, num_samples=1).view(1, -1)
            # Greedy decoding: select the token with highest probability.
            # logits[:, -1, :] corresponds to the logits for the last time step.
            generated_tokens.append(next_token.item())

        # Decode the generated tokens (using your tokenizer).
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(colored(f"{generated_text}", 'red'))

    def update_full_draft_cache(self, new_kv: List[Tuple[torch.Tensor, torch.Tensor]], tokens_appended: int):
        """
        Update the full draft KV cache with the new tokens.
        new_kv is the returned KV from the forward pass (a working-cache view).
        tokens_appended is the number of new tokens processed in this forward pass.
        
        The full cache is preallocated with size self.full_cache_budget, and
        self.total_seq_len tracks the current number of tokens stored.
        This function copies the last tokens_appended tokens from new_kv (from the working view)
        into the full cache.
        """
        # Check that we don't exceed the allocated budget.
        if self.total_seq_len + tokens_appended > self.full_cache_budget:
            raise RuntimeError(
                f"Full cache budget exceeded: total_seq_len {self.total_seq_len} + new {tokens_appended} > {self.full_cache_budget}"
            )

        # Precompute destination slice indices.
        dest_start = self.total_seq_len
        dest_end = dest_start + tokens_appended
        device = self.device

        # For each layer in the new KV, copy the last tokens_appended tokens into the full cache.
        for i, (new_K, new_V) in enumerate(new_kv):
            full_K, full_V = self.full_draft_kv[i]
            # Ensure new_K and new_V are on the correct device.
            new_K = new_K.to(device, non_blocking=True)
            new_V = new_V.to(device, non_blocking=True)
            # Copy the last tokens_appended tokens from new_K/new_V into the full cache.
            full_K[:, :, dest_start:dest_end, :].copy_(new_K[:, :, -tokens_appended:, :])
            full_V[:, :, dest_start:dest_end, :].copy_(new_V[:, :, -tokens_appended:, :])
        
        # Update the total sequence length.
        self.total_seq_len = dest_end

    def update_full_draft_cache(self, new_kv: List[Tuple[torch.Tensor, torch.Tensor]], tokens_appended: int):
        """
        Update the full draft KV cache with the new tokens.
        new_kv is the returned KV from the forward pass (a working-cache view).
        tokens_appended is the number of new tokens processed in this forward pass.
        
        The full cache is preallocated with size self.full_cache_budget, and
        self.total_seq_len tracks the current number of tokens stored.
        This function copies the last tokens_appended tokens from new_kv (from the working view)
        into the full cache.
        """
        # Check that we don't exceed the allocated budget.
        if self.total_seq_len + tokens_appended > self.full_cache_budget:
            raise RuntimeError(
                f"Full cache budget exceeded: total_seq_len {self.total_seq_len} + new {tokens_appended} > {self.full_cache_budget}"
            )

        # Precompute destination slice indices.
        dest_start = self.total_seq_len
        dest_end = dest_start + tokens_appended
        device = self.device

        # For each layer in the new KV, copy the last tokens_appended tokens into the full cache.
        for i, (new_K, new_V) in enumerate(new_kv):
            full_K, full_V = self.full_draft_kv[i]
            # Ensure new_K and new_V are on the correct device.
            new_K = new_K.to(device, non_blocking=True)
            new_V = new_V.to(device, non_blocking=True)
            # Copy the last tokens_appended tokens from new_K/new_V into the full cache.
            full_K[:, :, dest_start:dest_end, :].copy_(new_K[:, :, -tokens_appended:, :])
            full_V[:, :, dest_start:dest_end, :].copy_(new_V[:, :, -tokens_appended:, :])
        
        # Update the total sequence length.
        self.total_seq_len = dest_end

    def update_working_cache_from_full(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Build the working cache (self.draft_stable_kv) by indexing into the full cache.
        The working cache is defined to be the concatenation of:
        - the first sink_size tokens (the "sink" region), and 
        - the last recent_size tokens (the "recent" region)
        If total_seq_len is less than sink_size+recent_size, simply use all tokens.
        Also updates self.evicted to be the number of tokens that are outside the working window.
        """
        working_kv = []
        for (full_K, full_V) in self.full_draft_kv:
            if self.total_seq_len <= self.sink_size + self.recent_size:
                working_K_layer = full_K[:, :, :self.total_seq_len, :].clone()
                working_V_layer = full_V[:, :, :self.total_seq_len, :].clone()
                # print(colored(f'Using cache regions: {0}~{self.total_seq_len-1}','magenta'))
            else:
                sink_part_K = full_K[:, :, :self.sink_size, :].clone()
                sink_part_V = full_V[:, :, :self.sink_size, :].clone()
                recent_part_K = full_K[:, :, self.total_seq_len - self.recent_size:self.total_seq_len, :].clone()
                recent_part_V = full_V[:, :, self.total_seq_len - self.recent_size:self.total_seq_len, :].clone()
                working_K_layer = torch.cat([sink_part_K, recent_part_K], dim=2)
                working_V_layer = torch.cat([sink_part_V, recent_part_V], dim=2)
                # print(colored(f'Using cache regions: {0}~{self.sink_size-1}, {self.total_seq_len - self.recent_size}~{self.total_seq_len-1}','magenta'))
            working_kv.append((working_K_layer, working_V_layer))
        self.evicted = max(self.total_seq_len - (self.sink_size + self.recent_size), 0)
        working_kv_len = working_kv[0][0].shape[2]
        # print(colored(f'Working KV length: {working_kv_len}','red'))
        
        # truncate Draft model's past_key_position_ids:
         # this is when the prefill chunk size is smaller than the working cache size (only right after prefill)
        self.past_key_position_ids = (
            torch.cat([self.past_key_position_ids,
                    torch.arange(self.past_key_position_ids.shape[1], working_kv_len, device=self.past_key_position_ids.device).unsqueeze(0)],
                    dim=1)
            if self.past_key_position_ids.shape[1] < working_kv_len
            else self.past_key_position_ids
        )[:, :working_kv_len]

        self.recent_start = max(0, self.total_seq_len - self.recent_size)
        self.recent_end = self.total_seq_len-1
        return working_kv

    def update_working_cache_retrieval_main(self, top_k_chunks: int = 15):
        """
        Convenience function that first updates the chunk metadata (if new tokens were appended)
        and then updates the working cache (self.draft_stable_kv) based on retrieval.
        
        It assumes that self.attn_scores_final is already set (e.g., computed from the last accepted query)
        and that self.total_seq_len has been updated by update_full_draft_cache.
        """
        is_updated_chunks = self.update_chunks()
        working_kv = self.update_working_cache_retrieval(top_k_chunks=top_k_chunks,
                                                         do_retrieval=self.retrieval_condition,
                                                         is_updated_chunks=is_updated_chunks
                                                         ) 
        return working_kv

    def update_chunks(self):
        """
        Called after new tokens have been appended to the full KV cache.
        self.total_seq_len has been updated externally (by update_full_draft_cache).
        This function updates self.chunks to reflect the new total length.
        
        It does so by:
        1) Filling the last chunk (if not already full) with some of the new tokens.
        2) Creating new chunk(s) (each of size self.retrieval_chunk_size, except possibly the last one)
            for any remaining new tokens.
        """
        # Calculate how many new tokens were appended.
        new_tokens = self.total_seq_len - self.seq_len_total_old
        if new_tokens <= 0:
            return  # No new tokens; nothing to do.

        # If there are no chunks yet, create them from scratch.
        if not hasattr(self, "chunks") or self.chunks is None or len(self.chunks) == 0:
            self.prepare_chunks()
            return True

        # Get the last chunk's info.
        last_chunk_idx, last_start, last_end = self.chunks[-1]
        last_chunk_size = last_end - last_start
        remaining_new_tokens = new_tokens

        # 1) If the last chunk is not full, fill it up as much as possible.
        capacity = self.retrieval_chunk_size - last_chunk_size
        if capacity > 0:
            tokens_to_add = min(capacity, remaining_new_tokens)
            # Update the last chunk's end index.
            self.chunks[-1] = (last_chunk_idx, last_start, last_end + tokens_to_add)
            remaining_new_tokens -= tokens_to_add
    
        # 2) For any remaining tokens, create new chunks.
        current_start = self.total_seq_len - remaining_new_tokens
        while remaining_new_tokens > 0:
            tokens_in_chunk = min(self.retrieval_chunk_size, remaining_new_tokens)
            new_chunk = (self.chunks[-1][0] + 1, current_start, current_start + tokens_in_chunk)
            self.chunks.append(new_chunk)
            current_start += tokens_in_chunk
            remaining_new_tokens -= tokens_in_chunk

        # Update the stored old full length.
        self.seq_len_total_old = self.total_seq_len
        self.num_chunks = len(self.chunks)


        if self.num_chunks > self.num_chunks_old:
            self.num_chunks_old = self.num_chunks
            return True # new chunk was added => update working cache
        return False

    def prepare_chunks(self):
        """
        Called once (right after prefill) to split the full cache (of length self.total_seq_len)
        into consecutive chunks of fixed size (self.retrieval_chunk_size). Each chunk is represented as a tuple:
        (chunk_idx, start, end) where end - start <= self.retrieval_chunk_size.
        """
        self.chunks = []
        current_start = 0
        chunk_idx = 0
        while current_start < self.total_seq_len:
            end_pos = min(current_start + self.retrieval_chunk_size, self.total_seq_len)
            self.chunks.append((chunk_idx, current_start, end_pos))
            chunk_idx += 1
            current_start = end_pos
        # Save the current full length so that later we know how many new tokens were appended.
        self.seq_len_total_old = self.total_seq_len
        self.num_chunks = len(self.chunks)
        self.num_chunks_old = self.num_chunks

    def update_working_cache_retrieval(self, top_k_chunks: int = 15,
                                       do_retrieval=False,
                                       is_updated_chunks=False) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        # initial cache: use recent chunks (we don't have attn scores yet)
        if not hasattr(self, "selected_chunks") or self.selected_chunks is None:
            num_init = min(self.retrieve_top_k, len(self.chunks))
            if self.retrieval_policy == "random":
                generator = torch.Generator(device="cpu")
                generator.manual_seed(self.retrieval_seed)
                indices = torch.randperm(len(self.chunks), generator=generator)[:num_init].tolist()
                self.selected_chunks = sorted(
                    [self.chunks[i] for i in indices], key=lambda x: x[0]
                )
            else:
                self.selected_chunks = self.chunks[-num_init:]
            
        # Only retrieve top-k upon retrieval condition
        if do_retrieval:
            n = len(self.chunks)
            if n == 0:
                raise ValueError("No chunks available for retrieval.")
            k = min(top_k_chunks, n)

            if self.retrieval_policy == "recent":
                selected_chunks = self.chunks[-k:]
            elif self.retrieval_policy == "random":
                generator = torch.Generator(device="cpu")
                generator.manual_seed(self.retrieval_seed + self.timestep)
                indices = torch.randperm(n, generator=generator)[:k].tolist()
                selected_chunks = [self.chunks[i] for i in indices]
            else:
                attn = self.attn_scores_final
                chunks_tensor = torch.tensor(
                    [[start, end] for (_, start, end) in self.chunks],
                    dtype=torch.long,
                    device=attn.device
                )
                starts = chunks_tensor[:, 0]
                ends = chunks_tensor[:, 1]
                cum_attn = torch.cumsum(attn, dim=0)
                lower = torch.where(starts > 0, cum_attn[starts - 1], torch.zeros_like(starts, dtype=attn.dtype))
                ends_minus_one = torch.clamp(ends - 1, max=cum_attn.size(0) - 1)
                chunk_sums = cum_attn[ends_minus_one] - lower
                lengths = (ends - starts).float()
                chunk_means = chunk_sums / lengths
                topk = torch.topk(chunk_means, k=k)
                selected_chunks = [self.chunks[i] for i in topk.indices.tolist()]
            
            selected_chunks.sort(key=lambda x: x[0])
            self.selected_chunks = selected_chunks
            
            # reset retrieval condition and attn scores
            self.retrieval_condition = False
            self.attn_scores = None
            self.attn_scores_final = None

        # if new chunk is added, automatically update
        if is_updated_chunks:
            # grab the newly created chunk
            new_chunk = self.chunks[-1]  # (chunk_id, start, end)
            new_chunk_id = new_chunk[0]
            existing_ids = {cid for cid, _, _ in self.selected_chunks}
            # only append it if it’s not already in the selected set
            if new_chunk_id not in existing_ids:
                self.selected_chunks.append(new_chunk)

        # update last selected chunk
        if self.selected_chunks[-1][0] == self.chunks[-1][0]:
            chunk_id, start, _ = self.selected_chunks[-1]
            new_end = self.chunks[-1][2]
            self.selected_chunks[-1] = (chunk_id, start, new_end)

        all_indices = []
        for (_, start, end) in self.selected_chunks:
            all_indices.extend(range(start, end))
        if len(all_indices) == 0:
            raise ValueError("No tokens retrieved from the full cache. Check your chunk settings and attn_scores_final.")
        
        retrieved_indices = torch.tensor(all_indices, dtype=torch.long)
        retrieved_indices = torch.unique(retrieved_indices, sorted=True).to(self.device)

        if retrieved_indices.numel() == 0:
            raise ValueError("No tokens retrieved from the full cache. Check your chunk settings and attn_scores_final.")

        # Build working cache by indexing into full cache for each layer
        working_kv = []
        for (full_K, full_V) in self.full_draft_kv:
            working_K_layer = full_K.index_select(dim=2, index=retrieved_indices)
            working_V_layer = full_V.index_select(dim=2, index=retrieved_indices)
            working_kv.append((working_K_layer, working_V_layer))
        self.draft_stable_kv = working_kv

        # Update evicted count
        self.evicted = self.total_seq_len - retrieved_indices.numel()

        # Efficient update of past_key_position_ids:
        past_ids = self.past_key_position_ids  # shape [1, current_length]
        current_length = past_ids.shape[1]
        target_length = retrieved_indices.numel()
        if current_length < target_length:
            extra_ids = torch.arange(current_length, target_length, device=past_ids.device).unsqueeze(0)
            new_past_ids = torch.cat([past_ids, extra_ids], dim=1)
        else:
            new_past_ids = past_ids[:, :target_length]
        self.past_key_position_ids = new_past_ids

        if self.retrieval_verbose:
            if do_retrieval:
                self.print_retrieved_chunks(order="id")
        return working_kv
    
    def print_retrieved_chunks(self, order="id"):
        if order == "score":
            chunks_list = self.selected_chunks
            msg = "\nRetrieved chunk IDs (descending attn score): "
        elif order == "id":
            chunks_list = sorted(self.selected_chunks, key=lambda x: x[0])
            msg = "\nRetrieved chunk IDs: "
        else:
            print(colored(f"Unknown 'order' option: {order}. Choose 'score' or 'id'.", 'red'))
            return

        chunk_ids_str = ", ".join(str(chunk[0]) for chunk in chunks_list)

        print(colored(msg + chunk_ids_str + '\n', 'yellow'))

class Vhead(nn.Module):
    def __init__(self,ins=6566,outs=32000):
        super().__init__()
        self.fc = nn.Linear(ins,outs,bias=False)
    def forward(self,x):
        return self.fc(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


