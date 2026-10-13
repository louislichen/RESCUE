"""Full-prefix Q/K/V updates and fresh teacher attention interventions."""
import math

import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

from methods import Intervention


def move_cache(cache, device):
    return tuple(tuple(t.to(device) for t in pair) for pair in cache)


class ProjectionUpdates(torch.nn.Module):
    def __init__(self, model, layers, projections):
        super().__init__()
        self.weights = torch.nn.ParameterDict({
            name: torch.nn.Parameter(torch.zeros(len(layers),
                *getattr(model.model.layers[layers[0]].self_attn, name+"_proj").weight.shape,
                device=model.device, dtype=torch.float32)) for name in projections})


class AttentionController(Intervention):
    def __init__(self, model, layers, teacher_heads):
        super().__init__(model, layers, teacher_heads)
        self.capture = False
        self.query_start = 0
        self.attention_rows, self.value_rows = {}, {}
        self.prior_rows = None
        self.prior_docs = None
        self.scope = "document"
        self.folded = {}
        self.folded_versions = None

    def projection(self, network):
        self.network = network
        self.kind = "projection" if network is not None else "baseline"
        self.enabled = network is not None
        self.folded, self.folded_versions = {}, None
        self.capture = False

    def configure_prior(self, strength, docs, scope="document"):
        self.network = None
        self.kind = "prior" if strength else "baseline"
        self.enabled = bool(strength)
        self.strength, self.scope = strength, scope
        self.prior_docs = torch.as_tensor(docs, device=self.model.device)
        self.capture = False

    def linear(self, module, name, hidden, relative):
        projection = getattr(module, name+"_proj")
        if self.network is None or name not in self.network.weights:
            return projection(hidden)
        if torch.is_grad_enabled():
            weight = (projection.weight.float()+self.network.weights[name][relative]).to(hidden.dtype)
        else:
            versions = tuple(p._version for p in self.network.parameters())
            if versions != self.folded_versions:
                self.folded = {key:[(getattr(self.model.model.layers[layer].self_attn, key+"_proj").weight.float()
                                    + delta[i]).to(hidden.dtype) for i,layer in enumerate(self.layers)]
                               for key,delta in self.network.weights.items()}
                self.folded_versions = versions
            weight = self.folded[name][relative]
        return F.linear(hidden, weight, projection.bias)

    def forward_attention(self, module, rel, hidden, mask, positions, cache, output_attentions):
        batch, queries, _ = hidden.shape
        q, k, v = (self.linear(module, name, hidden, rel) for name in ("q", "k", "v"))
        q = q.view(batch, queries, module.num_heads, module.head_dim).transpose(1, 2)
        k = k.view(batch, queries, module.num_key_value_heads, module.head_dim).transpose(1, 2)
        v = v.view(batch, queries, module.num_key_value_heads, module.head_dim).transpose(1, 2)
        length = k.shape[-2]+(cache.get_usable_length(queries, module.layer_idx) if cache is not None else 0)
        cos, sin = module.rotary_emb(v, seq_len=length)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, positions)
        if cache is not None:
            k, v = cache.update(k, v, module.layer_idx, {"sin":sin, "cos":cos})
        repeated_k, repeated_v = repeat_kv(k, module.num_key_value_groups), repeat_kv(v, module.num_key_value_groups)
        scores = torch.matmul(q, repeated_k.transpose(2, 3))/math.sqrt(module.head_dim)
        if mask is not None:
            scores = scores+mask
        if self.kind == "prior":
            if queries != 1 or self.prior_rows is None:
                raise ValueError("Teacher prior applies only to cached single-query decoding")
            row = scores[0, :, 0, :].float()
            delta = torch.zeros_like(row)
            if self.scope == "document":
                docs = self.prior_docs
                target = self.prior_rows[rel, docs].float().clamp_min(1e-12)
                target /= target.sum()
                delta[:, docs] = target.log()-row[:, docs].log_softmax(-1)
            else:
                target = self.prior_rows[rel].float().clamp_min(1e-12)
                target /= target.sum()
                if target.numel() != length:
                    raise ValueError("Fresh full prior must cover every visible token")
                delta = target.log()-row.log_softmax(-1)
            scores = scores.float()+self.strength*delta[None, :, None, :]
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        if self.capture:
            self.attention_rows[rel] = probs[0, :, self.query_start:, :].float().mean(0)
            query_values = v[:, :, length-queries+self.query_start:, :].float()
            value_scores = query_values@v.float().transpose(2, 3)/math.sqrt(module.head_dim)
            if mask is not None:
                value_scores = value_scores+mask[:, :, self.query_start:, :].float()
            self.value_rows[rel] = value_scores.softmax(-1)[0].mean(0)
        probs = F.dropout(probs, p=module.attention_dropout, training=module.training)
        hidden = (probs@repeated_v).transpose(1, 2).contiguous().reshape(batch, queries, module.hidden_size)
        return module.o_proj(hidden), probs if output_attentions else None, cache


def upper_forward(student, frame, controller, capture=False):
    """Recompute all last-four-layer prompt and suffix states with gradients."""
    hidden = frame["lower_hidden"].to(student.device)
    length = hidden.shape[1]
    positions = torch.arange(length, device=student.device)
    forbidden = positions[None, :] > positions[:, None]
    mask = torch.zeros(length, length, device=student.device, dtype=hidden.dtype)
    mask.masked_fill_(forbidden, torch.finfo(hidden.dtype).min)
    controller.capture = capture
    controller.query_start = len(frame["ids"])-1
    controller.attention_rows, controller.value_rows = {}, {}
    for layer in student.model.layers[controller.layers[0]:]:
        hidden = layer(hidden, attention_mask=mask[None, None], position_ids=positions[None],
                       use_cache=False, output_attentions=False)[0]
    suffix = student.model.norm(hidden[:, controller.query_start:, :])
    logits = student.lm_head(suffix)[0].float()
    return logits, (torch.stack(list(controller.attention_rows.values())) if capture else None), \
        (torch.stack(list(controller.value_rows.values())) if capture else None)


def distribution_kl(logits, target_logits, vocab_size):
    teacher_logp = target_logits.to(logits.device).float()[:, :vocab_size].log_softmax(-1)
    student_logp = logits[:, :vocab_size].log_softmax(-1)
    return (teacher_logp.exp()*(teacher_logp-student_logp)).sum(-1).mean()


def relation_kl(rows, target):
    target = target.to(rows.device).float()
    target = target/target.sum(-1, keepdim=True).clamp_min(1e-12)
    rows = rows/rows.sum(-1, keepdim=True).clamp_min(1e-12)
    return (target*(target.clamp_min(1e-12).log()-rows.clamp_min(1e-12).log())).sum(-1).mean()
