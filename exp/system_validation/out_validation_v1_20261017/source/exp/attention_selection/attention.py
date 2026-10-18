"""Frozen native-KV attention selection, with original RoPE positions preserved."""
from __future__ import annotations

import math
from types import MethodType

import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv


class AttentionController:
    def __init__(self, model):
        self.model = model
        self.enabled = False
        self.capture = False
        self.selection = None
        self.bias_layer = None
        self.bias_prior = None
        self.strength = 0.0
        # Optional read-only working-cache transformation. The backing cache
        # above is updated with native tensors before this callback runs.
        self.kv_transform = None
        # Optional diagnostic of the actual current Q and native full cache.
        # The consumer must account for any extra full-history reads it makes.
        self.reference_observer = None
        self.rows = {}
        self.visible = {}
        self.query = None
        self.profile = False
        self.gather_events = []
        self.originals = []

    def install(self):
        for index, layer in enumerate(self.model.model.layers):
            module = layer.self_attn
            original = module.forward
            self.originals.append((module, original))

            def forward(module, hidden_states, attention_mask=None, position_ids=None,
                        past_key_value=None, output_attentions=False, use_cache=False,
                        _index=index, _original=original, **kwargs):
                if _index == 0 and self.capture:
                    # Frozen current-query features, independent of cache selection.
                    self.query = hidden_states[0, -1].float().detach()
                if not self.enabled:
                    out, rows, cache = _original(
                        hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions or self.capture,
                        use_cache=use_cache, **kwargs)
                    if self.capture:
                        self.rows[_index] = rows[0, :, -1].detach().float()
                        self.visible[_index] = torch.ones(rows.shape[-1], device=rows.device, dtype=torch.bool)
                    return out, rows if output_attentions else None, cache
                return self.forward(module, _index, hidden_states, attention_mask,
                                    position_ids, past_key_value, output_attentions)

            module.forward = MethodType(forward, module)
        return self

    def configure(self, selection=None, *, bias_layer=None, prior=None, strength=0.0):
        self.enabled = True
        self.selection = selection
        self.bias_layer, self.bias_prior, self.strength = bias_layer, prior, strength

    def snapshot(self):
        rows = torch.stack([self.rows[i].mean(0) for i in range(len(self.model.model.layers))])
        visible = torch.stack([self.visible[i] for i in range(len(self.model.model.layers))])
        return rows, visible, self.query

    def gather_seconds(self):
        torch.cuda.synchronize()
        result = sum(start.elapsed_time(end) for start, end in self.gather_events) / 1000
        self.gather_events.clear()
        return result

    def forward(self, module, index, hidden, mask, positions, cache, output_attentions):
        batch, queries, _ = hidden.shape
        q, k, v = module.q_proj(hidden), module.k_proj(hidden), module.v_proj(hidden)
        q = q.view(batch, queries, module.num_heads, module.head_dim).transpose(1, 2)
        k = k.view(batch, queries, module.num_key_value_heads, module.head_dim).transpose(1, 2)
        v = v.view(batch, queries, module.num_key_value_heads, module.head_dim).transpose(1, 2)
        length = k.shape[-2]
        if cache is not None:
            length += cache.get_usable_length(length, module.layer_idx)
        # Rotary embeddings use full absolute positions, never the gathered length.
        cos, sin = module.rotary_emb(v, seq_len=length)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, positions)
        if cache is not None:
            k, v = cache.update(k, v, module.layer_idx, {'sin': sin, 'cos': cos})
        if self.reference_observer is not None:
            self.reference_observer(index, module, q, k, v, mask, positions)
        if self.selection is None:
            indices = torch.arange(length, device=hidden.device)
        else:
            base = self.selection[index] if isinstance(self.selection, list) else self.selection
            # New queries are appended between retrievals; historical membership stays fixed.
            frontier = base[-1] + 1
            appended = torch.arange(length, device=hidden.device)
            appended = appended[appended >= frontier]
            indices = torch.unique(torch.cat((base[base < length], appended, positions[0])), sorted=True)
        event_pair = None
        if self.profile:
            event_pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            event_pair[0].record()
        if self.selection is not None:
            k = k.index_select(-2, indices)
            v = v.index_select(-2, indices)
        if event_pair is not None:
            event_pair[1].record()
            self.gather_events.append(event_pair)
        if self.kv_transform is not None:
            k, v = self.kv_transform(index, k, v, indices, hidden, positions)
        k, v = repeat_kv(k, module.num_key_value_groups), repeat_kv(v, module.num_key_value_groups)
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(module.head_dim)
        if mask is not None:
            scores = scores + mask.index_select(-1, indices)
        else:
            future = indices[None, :] > positions[0, :, None]
            scores = scores.masked_fill(future[None, None], torch.finfo(scores.dtype).min)
        if index == self.bias_layer and self.strength:
            base_probs = F.softmax(scores, dim=-1, dtype=torch.float32)
            covered = indices < self.bias_prior.shape[-1]
            prior = self.bias_prior[indices[covered]].float().clamp_min(1e-12)
            prior = prior / prior.sum().clamp_min(1e-12)
            student = base_probs[..., covered]
            student = student / student.sum(-1, keepdim=True).clamp_min(1e-12)
            bias = prior.log()[None, None, None] - student.clamp_min(1e-12).log()
            # Current-query offline prior covers all positions. For an online
            # committed-prefix prior, new draft keys receive no teacher bias.
            if not bool(covered.all()):
                bias = bias - bias.mean(-1, keepdim=True)
            scores[..., covered] += (self.strength * bias).to(scores.dtype)
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        probs = F.dropout(probs, p=module.attention_dropout, training=module.training)
        if self.capture:
            expanded = torch.zeros(module.num_heads, length, device=hidden.device, dtype=torch.float32)
            expanded[:, indices] = probs[0, :, -1].float()
            self.rows[index] = expanded.detach()
            visible = torch.zeros(length, device=hidden.device, dtype=torch.bool)
            visible[indices] = True
            self.visible[index] = visible
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().reshape(batch, queries, module.hidden_size)
        out = module.o_proj(out)
        if output_attentions:
            expanded = torch.zeros(batch, module.num_heads, queries, length, device=hidden.device, dtype=probs.dtype)
            expanded[..., indices] = probs
            probs = expanded
        return out, probs if output_attentions else None, cache
