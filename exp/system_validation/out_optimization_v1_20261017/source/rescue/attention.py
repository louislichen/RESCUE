"""Qwen2 hooks: last-layer Target observation and reusable Draft working KV."""
import math
from types import MethodType

import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv


class TargetObservation:
    def __init__(self, model):
        self.model, self.originals = model, []
        self.mode, self.rows = None, None

    def install(self):
        if self.originals:
            return self
        module = self.model.model.layers[-1].self_attn
        original = module.forward
        self.originals.append((module, original))

        def forward(module, hidden_states, attention_mask=None, position_ids=None,
                    past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
            out, weights, cache = original(hidden_states, attention_mask=attention_mask,
                                          position_ids=position_ids, past_key_value=past_key_value,
                                          output_attentions=output_attentions or self.mode is not None,
                                          use_cache=use_cache, **kwargs)
            if self.mode is not None:
                # A clone releases a prompt-sized attention matrix immediately.
                self.rows = (weights[0, :, -1].clone() if self.mode == 'last' else weights[0]).detach().float()
            return out, weights if output_attentions else None, cache

        module.forward = MethodType(forward, module)
        return self

    def restore(self):
        for module, original in self.originals:
            module.forward = original
        self.originals.clear()
        self.rows = None


class DraftAttention:
    def __init__(self, model, working, injector, backend='sdpa', observe=True, optimized=False):
        self.model, self.working, self.injector = model, working, injector
        self.backend = backend
        self.observe = observe
        self.optimized = optimized
        self.originals, self.shadow = [], None
        self.prefill = True
        self.steps = 0

    def install(self):
        for index, layer in enumerate(self.model.model.layers):
            module = layer.self_attn
            original = module.forward
            self.originals.append((module, original))

            def forward(module, hidden_states, attention_mask=None, position_ids=None,
                        past_key_value=None, output_attentions=False, use_cache=False,
                        _index=index, _original=original, **kwargs):
                if self.prefill:
                    out, rows, cache = _original(hidden_states, attention_mask=attention_mask,
                        position_ids=position_ids, past_key_value=past_key_value,
                        output_attentions=self.observe and _index == len(self.model.model.layers)-1,
                        use_cache=use_cache, **kwargs)
                    if rows is not None:
                        self.shadow = rows[0, :, -1].detach().float().clone()
                    return out, rows if output_attentions else None, cache
                return self.decode(module, _index, hidden_states, position_ids, past_key_value)

            module.forward = MethodType(forward, module)
        return self

    def decode(self, module, layer, hidden, positions, native):
        if hidden.shape[:2] != (1, 1):
            raise ValueError('Draft decode accepts one unpadded query')
        q = module.q_proj(hidden).view(1, 1, module.num_heads, module.head_dim).transpose(1, 2)
        k = module.k_proj(hidden).view(1, 1, module.num_key_value_heads, module.head_dim).transpose(1, 2)
        v = module.v_proj(hidden).view(1, 1, module.num_key_value_heads, module.head_dim).transpose(1, 2)
        length = native.get_seq_length(layer)+1
        cos, sin = module.rotary_emb(v, seq_len=length)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, positions)
        full_k, full_v = native.update(k, v, layer)
        if self.observe and layer == len(self.model.model.layers)-1:
            if self.optimized:
                grouped = q.reshape(1, module.num_key_value_heads, module.num_key_value_groups, module.head_dim)
                scores = torch.matmul(grouped, full_k.transpose(-1, -2)).reshape(module.num_heads, -1)/math.sqrt(module.head_dim)
                self.shadow = scores.float().softmax(-1).detach()
            else:
                scores = torch.matmul(q, repeat_kv(full_k, module.num_key_value_groups).transpose(-1, -2))/math.sqrt(module.head_dim)
                self.shadow = scores.float().softmax(-1)[0, :, 0].detach()
        if self.working is None:
            wk, wv = full_k, full_v
        else:
            wk, wv = self.working.append(layer, k, v)
        if self.injector is not None and (not self.optimized or layer in self.injector.active):
            wv = self.injector.apply(layer, wv, hidden)
        if self.backend == 'sdpa':
            if self.optimized:
                out = F.scaled_dot_product_attention(q, wk, wv, dropout_p=0., is_causal=False, enable_gqa=True)
            else:
                wk, wv = repeat_kv(wk, module.num_key_value_groups), repeat_kv(wv, module.num_key_value_groups)
                out = F.scaled_dot_product_attention(q, wk, wv, dropout_p=0., is_causal=False)
        else:
            wk, wv = repeat_kv(wk, module.num_key_value_groups), repeat_kv(wv, module.num_key_value_groups)
            scores = torch.matmul(q, wk.transpose(-1, -2))/math.sqrt(module.head_dim)
            out = torch.matmul(scores.float().softmax(-1).to(q.dtype), wv)
        out = module.o_proj(out.transpose(1, 2).contiguous().view(1, 1, module.hidden_size))
        return out, None, native

    def step(self, token, native, logits=True):
        model = self.model.model
        ids = torch.as_tensor([[token]], dtype=torch.long, device=model.embed_tokens.weight.device)
        positions = torch.tensor([[native.get_seq_length()]], device=ids.device)
        hidden = model.embed_tokens(ids)
        for layer in model.layers:
            hidden = layer(hidden, attention_mask=None, position_ids=positions,
                           past_key_value=native, output_attentions=False, use_cache=True)[0]
        self.steps += 1
        if not logits:
            return None
        return self.model.lm_head(model.norm(hidden))[:, -1].float()

    def restore(self):
        for module, original in self.originals:
            module.forward = original
        self.originals.clear()
