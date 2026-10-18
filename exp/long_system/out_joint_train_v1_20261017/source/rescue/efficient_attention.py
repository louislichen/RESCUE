"""Qwen2 SDPA adapter with compact, explicitly requested Q snapshots.

The engine uses an unpadded single sequence. Cached multi-query calls require
bottom-right causal alignment; SDPA's is_causal=True alone aligns top-left.
"""
import math
from types import MethodType

import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb


def attention_rows(query, keys):
    """Return H x Q x N diagnostic rows without expanding GQA K heads."""
    batch, heads, queries, dim = query.shape
    if batch != 1:
        raise ValueError('Compact observations support one unpadded sequence')
    groups = heads//keys.shape[1]
    grouped = query.reshape(1, keys.shape[1], groups*queries, dim)
    scores = torch.matmul(grouped.float(), keys.float().transpose(-1, -2))
    return (scores.reshape(heads, queries, -1)/math.sqrt(dim)).softmax(-1).detach()


def install_efficient_attention(model):
    """Install once; preserve the eager implementation for explicit API fallback."""
    if model.config.model_type != 'qwen2':
        raise ValueError('Efficient adapter requires Qwen2')
    model.model._attn_implementation = 'sdpa'
    model.config._attn_implementation = 'sdpa'
    for layer in model.model.layers:
        module = layer.self_attn
        if getattr(module, '_rescue_efficient', False):
            continue
        original = module.forward
        module._rescue_efficient = True
        module._rescue_capture = None
        module._rescue_query = None

        def forward(module, hidden_states, attention_mask=None, position_ids=None,
                    past_key_value=None, output_attentions=False, use_cache=False,
                    _original=original, **kwargs):
            if output_attentions:
                return _original(hidden_states, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_value=past_key_value,
                    output_attentions=True, use_cache=use_cache, **kwargs)
            batch, queries, _ = hidden_states.shape
            if batch != 1:
                raise ValueError('Efficient engine supports one unpadded sequence')
            q = module.q_proj(hidden_states).view(batch, queries, module.num_heads, module.head_dim).transpose(1, 2)
            k = module.k_proj(hidden_states).view(batch, queries, module.num_key_value_heads, module.head_dim).transpose(1, 2)
            v = module.v_proj(hidden_states).view(batch, queries, module.num_key_value_heads, module.head_dim).transpose(1, 2)
            prefix = past_key_value.get_seq_length(module.layer_idx) if past_key_value is not None else 0
            cos, sin = module.rotary_emb(v, seq_len=prefix+queries)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
            if past_key_value is not None:
                k, v = past_key_value.update(k, v, module.layer_idx, {'sin': sin, 'cos': cos})
            if module._rescue_capture:
                observed = q[:, :, -1:] if module._rescue_capture == 'last' else q
                module._rescue_query = observed.detach().clone()
            else:
                module._rescue_query = None
            causal = queries > 1 and prefix == 0 and attention_mask is None
            if queries > 1 and prefix and attention_mask is None:
                columns = torch.arange(k.shape[-2], device=q.device)
                rows = prefix+torch.arange(queries, device=q.device)
                attention_mask = (columns[None, :] <= rows[:, None])[None, None]
            if q.dtype == torch.float32 and q.is_cuda:
                # Fold KV heads into batch. Broadcast views preserve GQA
                # storage without enabling SDPA's math/Flash-only GQA path;
                # the memory-efficient CUDA kernel supports FP32 this way.
                heads, groups = module.num_key_value_heads, module.num_key_value_groups
                grouped_q = q.reshape(heads, groups, queries, module.head_dim)
                grouped_k = k[0, :, None].expand(-1, groups, -1, -1)
                grouped_v = v[0, :, None].expand(-1, groups, -1, -1)
                out = F.scaled_dot_product_attention(grouped_q, grouped_k, grouped_v,
                    attn_mask=attention_mask, is_causal=causal, dropout_p=0.)
                out = out.reshape(1, module.num_heads, queries, module.head_dim)
            else:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask,
                    is_causal=causal, dropout_p=0., enable_gqa=True)
            out = out.transpose(1, 2).contiguous().reshape(batch, queries, module.hidden_size)
            return module.o_proj(out), None, past_key_value

        module.forward = MethodType(forward, module)
    return model
