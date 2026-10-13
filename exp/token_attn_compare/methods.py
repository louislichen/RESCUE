"""Decode-only, token-level interventions for eager Qwen2 attention.

The prompt prefix is always evaluated by the frozen base model. Interventions
start at its final query and continue through the generated suffix. This makes
prefix reuse exact, including when all entries of a Q projection are trained.
"""
from __future__ import annotations

import math
from types import MethodType

import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv


class TokenAdapter(torch.nn.Module):
    def __init__(self, layers, heads, teacher_heads, hidden_size, max_alpha=0.5):
        super().__init__()
        self.mix = torch.nn.Parameter(torch.zeros(layers, heads, teacher_heads))
        self.gate_bias = torch.nn.Parameter(torch.full((layers, heads), -3.0))
        self.gate_in = torch.nn.Linear(hidden_size, 32)
        self.gate_out = torch.nn.Linear(32, heads)
        torch.nn.init.zeros_(self.gate_out.weight)
        torch.nn.init.zeros_(self.gate_out.bias)
        self.max_alpha = max_alpha

    def prior(self, bank):
        return torch.einsum("lht,tk->lhk", self.mix.softmax(-1), bank)

    def gate(self, hidden, layer):
        hidden = F.layer_norm(hidden.float(), (hidden.shape[-1],))
        logits = self.gate_out(F.silu(self.gate_in(hidden))) + self.gate_bias[layer]
        return (self.max_alpha * logits.sigmoid()).transpose(1, 2).unsqueeze(-1)


class FullQUpdate(torch.nn.Module):
    """Full matrix updates, without low-rank factorization or hidden adapter."""
    def __init__(self, layers, hidden_size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(layers, hidden_size, hidden_size))


class Intervention:
    def __init__(self, model, layers, teacher_heads):
        self.model, self.layers = model, list(layers)
        self.teacher_heads = teacher_heads
        self.kind = "baseline"
        self.network = None
        self.strength = 0.0
        self.enabled = False
        self.force_zero = False
        self.bank = self.prior = self.bias = None
        self.originals = []
        self.merged_q = None
        self.merged_version = None

    def install(self):
        for rel, index in enumerate(self.layers):
            module = self.model.model.layers[index].self_attn
            original = module.forward
            self.originals.append((module, original))

            def forward(module, hidden_states, attention_mask=None, position_ids=None,
                        past_key_value=None, output_attentions=False, use_cache=False,
                        _relative=rel, _original=original, **kwargs):
                if not self.enabled or self.kind == "baseline":
                    return _original(hidden_states, attention_mask=attention_mask,
                                     position_ids=position_ids, past_key_value=past_key_value,
                                     output_attentions=output_attentions, use_cache=use_cache, **kwargs)
                return self.forward_attention(module, _relative, hidden_states,
                                              attention_mask, position_ids,
                                              past_key_value, output_attentions)

            module.forward = MethodType(forward, module)

    def restore(self):
        for module, original in self.originals:
            module.forward = original
        self.originals.clear()

    def configure(self, kind, network=None, strength=0.0):
        self.kind, self.network, self.strength = kind, network, strength
        self.merged_q = self.merged_version = None

    def set_context(self, bank, doc_positions):
        self.bank = bank
        if self.kind.startswith("adapter"):
            if self.kind == "adapter_uniform":
                bank = torch.full_like(bank, 1.0 / bank.shape[-1])
            self.prior = self.network.prior(bank)
        elif self.kind == "bias":
            # Center and bound the document scores. The sink, question and
            # generated positions have exactly zero score bias.
            rows = bank.reshape(len(self.layers), self.teacher_heads, -1).mean(1)
            self.bias = torch.zeros_like(rows)
            positions = torch.tensor(doc_positions, device=rows.device, dtype=torch.long)
            if positions.numel():
                logp = rows[:, positions].clamp_min(1e-12).log()
                self.bias[:, positions] = (logp - logp.mean(-1, keepdim=True)).clamp(-2, 2)

    def forward_attention(self, module, rel, hidden, mask, positions, cache, output_attentions):
        batch, queries, _ = hidden.shape
        if self.kind.startswith("q_native") and not self.force_zero:
            if torch.is_grad_enabled():
                weight = (module.q_proj.weight.float() + self.network.weight[rel]).to(hidden.dtype)
            else:
                # Cache the folded weights for inference; optimizer changes
                # invalidate them. The frozen prompt still uses base weights.
                version = self.network.weight._version
                if self.merged_version != version:
                    self.merged_q = [(self.model.model.layers[index].self_attn.q_proj.weight.float()
                                      + self.network.weight[i]).to(hidden.dtype)
                                     for i, index in enumerate(self.layers)]
                    self.merged_version = version
                weight = self.merged_q[rel]
            q = F.linear(hidden, weight, module.q_proj.bias)
        else:
            q = module.q_proj(hidden)
        if self.kind in ("q_attention", "q_output_only") and not self.force_zero:
            q = q + F.linear(hidden.float(), self.network.weight[rel]).to(q.dtype)
        k, v = module.k_proj(hidden), module.v_proj(hidden)
        q = q.view(batch, queries, module.num_heads, module.head_dim).transpose(1, 2)
        k = k.view(batch, queries, module.num_key_value_heads, module.head_dim).transpose(1, 2)
        v = v.view(batch, queries, module.num_key_value_heads, module.head_dim).transpose(1, 2)
        length = k.shape[-2]
        if cache is not None:
            length += cache.get_usable_length(length, module.layer_idx)
        cos, sin = module.rotary_emb(v, seq_len=length)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, positions)
        if cache is not None:
            k, v = cache.update(k, v, module.layer_idx, {"sin": sin, "cos": cos})
        k, v = repeat_kv(k, module.num_key_value_groups), repeat_kv(v, module.num_key_value_groups)
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(module.head_dim)
        if mask is not None:
            scores = scores + mask
        if self.kind == "bias" and self.strength and not self.force_zero:
            bias = F.pad(self.bias[rel], (0, length - self.bias.shape[-1]))
            scores = scores + (self.strength * bias).to(scores.dtype)[None, None, None, :]
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        if self.kind.startswith("adapter") and not self.force_zero:
            prior = F.pad(self.prior[rel], (0, length - self.prior.shape[-1]))[None, :, None, :]
            alpha = self.network.gate(hidden, rel)
            mixed = (1 - alpha) * probs.float() + alpha * prior
            mixed = mixed / mixed.sum(-1, keepdim=True).clamp_min(1e-12)
            probs = mixed.to(probs.dtype)
        probs = F.dropout(probs, p=module.attention_dropout, training=module.training)
        output = torch.matmul(probs, v).transpose(1, 2).contiguous().reshape(batch, queries, module.hidden_size)
        return module.o_proj(output), probs if output_attentions else None, cache
