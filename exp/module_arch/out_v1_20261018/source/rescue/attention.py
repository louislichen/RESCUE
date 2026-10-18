"""Qwen2 hooks: last-layer Target observation and reusable Draft working KV."""
import math
from types import MethodType

import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

from .efficient_attention import attention_rows


class TargetObservation:
    def __init__(self, model):
        self.model, self.originals = model, []
        self.mode, self.rows = None, None
        self.queries, self.cache = None, None

    def install(self):
        if self.originals:
            return self
        module = self.model.model.layers[-1].self_attn
        original = module.forward
        self.originals.append((module, original))

        def forward(module, hidden_states, attention_mask=None, position_ids=None,
                    past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
            if getattr(module, '_rescue_efficient', False) and not output_attentions:
                module._rescue_capture = self.mode
                try:
                    out, weights, cache = original(hidden_states, attention_mask=attention_mask,
                        position_ids=position_ids, past_key_value=past_key_value,
                        output_attentions=False, use_cache=use_cache, **kwargs)
                finally:
                    module._rescue_capture = None
                self.queries, self.cache = module._rescue_query, cache
                self.rows = self.row(-1, cache.get_seq_length()) if self.mode == 'last' else None
                return out, None, cache
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

    def row(self, index, committed):
        if self.queries is not None:
            return attention_rows(self.queries[:, :, index:index+1] if index >= 0 else self.queries[:, :, -1:],
                                  self.cache[-1][0][:, :, :committed])[:, 0]
        return self.rows[:, index, :committed].clone()

    def restore(self):
        for module, original in self.originals:
            module.forward = original
        self.originals.clear()
        self.rows = None
        self.queries = self.cache = None


class DraftAttention:
    def __init__(self, model, working, injector, backend='sdpa', observe=True, optimized=False, deferred=False,
                 observer=None):
        self.model, self.working, self.injector = model, working, injector
        self.backend = backend
        self.observe = observe
        self.optimized = optimized
        self.deferred = deferred and observe
        self.last_query = None
        self.probe_calls = 0
        self.originals, self.shadow = [], None
        self.prefill = True
        self.steps = 0
        self.observer, self.capture_step = observer, False
        self.conditioning, self.condition_features = False, None
        self.condition_callback = None

    def install(self):
        for index, layer in enumerate(self.model.model.layers):
            module = layer.self_attn
            original = module.forward
            self.originals.append((module, original))

            def forward(module, hidden_states, attention_mask=None, position_ids=None,
                        past_key_value=None, output_attentions=False, use_cache=False,
                        _index=index, _original=original, **kwargs):
                if self.prefill:
                    if getattr(module, '_rescue_efficient', False):
                        observe = self.observe and _index == len(self.model.model.layers)-1
                        module._rescue_capture = 'last' if observe else None
                        try:
                            out, _, cache = _original(hidden_states, attention_mask=attention_mask,
                                position_ids=position_ids, past_key_value=past_key_value,
                                output_attentions=False, use_cache=use_cache, **kwargs)
                        finally:
                            module._rescue_capture = None
                        if observe:
                            self.shadow = attention_rows(module._rescue_query, cache[-1][0])[:, 0]
                        return out, None, cache
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
            self.last_query = q.detach()
            self.shadow = None if self.deferred else self.probe(self.last_query, native)
        if self.working is None:
            wk, wv = full_k, full_v
        else:
            wk, wv = self.working.append(layer, k, v)
        if self.observer is not None and self.capture_step and layer == len(self.model.model.layers)-1:
            self.observer(dict(query=q, keys=wk, values=wv, normalized_hidden=hidden,
                               residual=self.current_residual, injector=self.injector))
        if self.injector is not None and (not self.optimized or layer in self.injector.active):
            wv = self.injector.apply(layer, wv, hidden,query=q,keys=wk)
        if self.conditioning and layer == len(self.model.model.layers)-1:
            injector = self.injector
            moments = torch.stack(injector.summaries).mean(0) if injector.summaries else q.new_zeros(2).float()
            active = injector.active.get(layer)
            mass = (attention_rows(q, wk)[:, 0, active[1]].sum(-1).mean()
                    if active is not None else moments.new_zeros(()))
            count = len(active[1]) if active is not None else 0
            self.condition_features = torch.cat((moments, torch.stack((mass,
                mass.new_tensor(len(injector.active)/len(self.model.model.layers)),
                mass.new_tensor(count/max(1, len(self.working.positions)))))))
            if self.condition_callback is not None:
                self.condition_callback(self.condition_features)
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

    def probe(self, query, native):
        module = self.model.model.layers[-1].self_attn
        full_k = native[-1][0]
        if self.optimized:
            grouped = query.reshape(1, module.num_key_value_heads, module.num_key_value_groups, module.head_dim)
            scores = torch.matmul(grouped, full_k.transpose(-1, -2)).reshape(module.num_heads, -1)/math.sqrt(module.head_dim)
            row = scores.float().softmax(-1)
        else:
            scores = torch.matmul(query, repeat_kv(full_k, module.num_key_value_groups).transpose(-1, -2))/math.sqrt(module.head_dim)
            row = scores.float().softmax(-1)[0, :, 0]
        self.probe_calls += 1
        return row.detach()

    def step(self, token, native, logits=True):
        model = self.model.model
        ids = torch.as_tensor([[token]], dtype=torch.long, device=model.embed_tokens.weight.device)
        positions = torch.tensor([[native.get_seq_length()]], device=ids.device)
        hidden = model.embed_tokens(ids)
        self.capture_step = logits
        if self.injector is not None:
            self.injector.capture_summary = self.conditioning or getattr(self, 'record_injection_summary', False)
            self.injector.summaries = []
        for index, layer in enumerate(model.layers):
            if self.observer is not None and index == len(model.layers)-1:
                self.current_residual = hidden.detach()
            hidden = layer(hidden, attention_mask=None, position_ids=positions,
                           past_key_value=native, output_attentions=False, use_cache=True)[0]
        self.steps += 1
        if self.injector is not None:
            self.injector.capture_summary = False
        if not logits:
            return None
        scores = self.model.lm_head(model.norm(hidden))[:, -1].float()
        if self.observer is not None and hasattr(self.observer, 'on_logits'):
            self.observer.on_logits(scores)
        return scores

    def restore(self):
        for module, original in self.originals:
            module.forward = original
        self.originals.clear()
