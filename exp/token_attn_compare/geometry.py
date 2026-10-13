"""Query-space decomposition of teacher-guided token score corrections.

At fixed keys, score changes achievable by a query lie in col(K). Softmax
also permits an arbitrary constant. Centering K and the desired score change
removes that constant, and least squares gives the achievable part and its
orthogonal residual. Prefix sufficient statistics are reused; suffix keys are
recomputed by the actual intervened model. No frozen generated KV is reused.
"""
import math

import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

from methods import Intervention


class GeometryContext:
    def __init__(self, model, layers, frame, seed=42):
        self.prefix_len = len(frame["ids"])-1
        self.docs = torch.tensor(frame["docs"], device=frame["bank"].device, dtype=torch.long)
        if self.docs.numel() == 0 or int(self.docs.max()) >= self.prefix_len:
            raise ValueError("Document positions must belong to the frozen prompt prefix")
        teacher_heads = frame["bank"].shape[0]//len(layers)
        rows = frame["bank"].reshape(len(layers), teacher_heads, -1).double().mean(1)
        doc_prior = rows[:, self.docs].clamp_min(1e-12)
        doc_prior /= doc_prior.sum(-1, keepdim=True)
        self.log_prior = doc_prior.log()
        generator = torch.Generator(device="cpu").manual_seed(seed+frame["cursor"])
        permutation = torch.randperm(self.docs.numel(), generator=generator).to(self.docs.device)
        self.shuffled_log_prior = self.log_prior[:, permutation]
        self.prefix, self.sums, self.grams, self.full_rank = [], [], [], []
        for index in layers:
            # Cache contains unrepeated, rotated grouped-query keys [KVheads,N,D].
            keys = frame["cache"][index][0][0].double()
            sums = keys.sum(1)
            gram = keys.transpose(1, 2) @ keys
            centered_gram = gram-sums[:, :, None]*sums[:, None, :]/self.prefix_len
            _, info = torch.linalg.cholesky_ex(centered_gram)
            self.prefix.append(keys)
            self.sums.append(sums)
            self.grams.append(gram)
            self.full_rank.append(bool((info == 0).all()))

    def decompose(self, relative, keys, desired_delta):
        """Return centered desired scores, their projection and residual.

        keys: [KVheads,N,D], desired_delta: [query_heads,N]. Computation uses
        FP64 to avoid cancellation in the centered Gram matrix. Q/K are already
        position-rotated, so RoPE is included in the attainable space.
        """
        groups, length, dim = keys.shape
        heads = desired_delta.shape[0]
        repeats = heads//groups
        prefix = self.prefix[relative]
        suffix = keys[:, self.prefix_len:, :].double()
        sums = self.sums[relative]+suffix.sum(1)
        gram = self.grams[relative]+suffix.transpose(1, 2)@suffix
        centered_gram = gram-sums[:, :, None]*sums[:, None, :]/length
        centered_gram = (centered_gram+centered_gram.transpose(1, 2))/2
        delta = desired_delta.double().reshape(groups, repeats, length)
        delta_sum = delta.sum(-1)
        rhs = prefix.transpose(1, 2)@delta[:, :, :self.prefix_len].transpose(1, 2)
        rhs += suffix.transpose(1, 2)@delta[:, :, self.prefix_len:].transpose(1, 2)
        rhs -= sums[:, :, None]*delta_sum[:, None, :]/length
        if self.full_rank[relative]:
            factor, _ = torch.linalg.cholesky_ex(centered_gram, check_errors=False)
            coefficients = torch.cholesky_solve(rhs, factor)
        else:
            values, vectors = torch.linalg.eigh(centered_gram)
            tolerance = values.amax(-1, keepdim=True)*1e-12
            inv = torch.where(values > tolerance, values.clamp_min(1e-300).reciprocal(), 0)
            coefficients = vectors@(inv[:, :, None]*(vectors.transpose(1, 2)@rhs))
        prefix_projection = (prefix@coefficients).transpose(1, 2)
        suffix_projection = (suffix@coefficients).transpose(1, 2)
        projection = torch.cat((prefix_projection, suffix_projection), dim=-1)
        projection -= (sums[:, None, :]@coefficients).transpose(1, 2)/length
        centered = delta-delta.mean(-1, keepdim=True)
        residual = centered-projection
        return (centered.reshape(heads, length), projection.reshape(heads, length),
                residual.reshape(heads, length), coefficients)


class GeometryIntervention(Intervention):
    def __init__(self, model, layers, teacher_heads):
        super().__init__(model, layers, teacher_heads)
        self.context = None
        self.projected_strength = self.residual_strength = 0.0
        self.use_native = self.shuffle = False
        self.native_weights = None
        self.capture = False
        self.captured = {}

    def load_native(self, delta):
        self.native_weights = [(self.model.model.layers[index].self_attn.q_proj.weight.float()
                                + delta[relative]).to(self.model.dtype)
                               for relative, index in enumerate(self.layers)]

    def route(self, projected, residual, *, native=False, shuffle=False):
        self.kind = "geometry"
        self.projected_strength, self.residual_strength = float(projected), float(residual)
        self.use_native, self.shuffle = native, shuffle

    def bind(self, frame):
        self.context = frame["geometry"]

    def set_context(self, bank, doc_positions):
        if self.context is None:
            return
        self.doc_log_prior = (self.context.shuffled_log_prior if self.shuffle else self.context.log_prior)

    def forward_attention(self, module, rel, hidden, mask, positions, cache, output_attentions):
        batch, queries, _ = hidden.shape
        if batch != 1 or queries != 1:
            raise ValueError("Geometry prototype intervenes in single-query cached decoding")
        if self.use_native and self.native_weights is not None:
            q = F.linear(hidden, self.native_weights[rel], module.q_proj.bias)
        else:
            q = module.q_proj(hidden)
        k, v = module.k_proj(hidden), module.v_proj(hidden)
        q = q.view(batch, queries, module.num_heads, module.head_dim).transpose(1, 2)
        k = k.view(batch, queries, module.num_key_value_heads, module.head_dim).transpose(1, 2)
        v = v.view(batch, queries, module.num_key_value_heads, module.head_dim).transpose(1, 2)
        length = k.shape[-2]+(cache.get_usable_length(k.shape[-2], module.layer_idx) if cache is not None else 0)
        cos, sin = module.rotary_emb(v, seq_len=length)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, positions)
        if cache is not None:
            k, v = cache.update(k, v, module.layer_idx, {"sin": sin, "cos": cos})
        grouped_keys = k[0]
        repeated_keys = repeat_kv(k, module.num_key_value_groups)
        repeated_values = repeat_kv(v, module.num_key_value_groups)
        scores = torch.matmul(q, repeated_keys.transpose(2, 3))/math.sqrt(module.head_dim)
        if mask is not None:
            scores = scores+mask
        row = scores[0, :, -1, :]
        if self.capture:
            self.captured[rel] = dict(keys=grouped_keys.detach(), scores=row.detach(),
                                      query=q[0, :, -1, :].detach())
        a, b = self.projected_strength, self.residual_strength
        if a or b:
            # Change only the document conditional distribution. At strength
            # one its total mass and all non-document probabilities stay fixed.
            docs = self.context.docs
            desired = torch.zeros_like(row, dtype=torch.float64)
            desired[:, docs] = self.doc_log_prior[rel]-row[:, docs].double().log_softmax(-1)
            if a == b:
                correction = a*desired  # Exact nondecomposed control; constant is immaterial.
            else:
                _, projected, residual, _ = self.context.decompose(rel, grouped_keys, desired)
                correction = a*projected+b*residual
            scores = scores.float()+correction.float()[None, :, None, :]
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        probs = F.dropout(probs, p=module.attention_dropout, training=module.training)
        output = torch.matmul(probs, repeated_values).transpose(1, 2).contiguous().reshape(batch, queries, module.hidden_size)
        return module.o_proj(output), probs if output_attentions else None, cache


def distribution_metrics(scores, target, context, relative, keys, query):
    """Fixed-state geometry diagnostics, separate from acceptance evaluation."""
    scores = scores.double()
    target = target.double().clamp_min(1e-12)
    target = target/target.sum(-1, keepdim=True)
    if target.ndim == 1:
        target = target.expand(scores.shape[0], -1)
    desired = target.log()-scores
    centered, projected, residual, coefficients = context.decompose(relative, keys, desired)
    energy = centered.square().sum(-1)
    ratio = projected.square().sum(-1)/energy.clamp_min(1e-24)
    log_target = target.log()
    base_kl = (target*(log_target-scores.log_softmax(-1))).sum(-1)
    projected_kl = (target*(log_target-(scores+projected).log_softmax(-1))).sum(-1)
    full = (scores+projected+residual).softmax(-1)
    full_kl = (target*(log_target-full.clamp_min(1e-300).log())).sum(-1)
    groups, length, dim = keys.shape
    repeated_keys = keys.double().repeat_interleave(scores.shape[0]//groups, dim=0)
    centered_keys = repeated_keys-repeated_keys.mean(1, keepdim=True)
    orthogonality = torch.einsum("hnd,hn->hd", centered_keys, residual).norm(dim=-1)
    orthogonality /= centered_keys.flatten(1).norm(dim=-1)*residual.norm(dim=-1)+1e-24
    query_delta = math.sqrt(dim)*coefficients.transpose(1, 2).reshape(query.shape)
    query_ratio = query_delta.norm(dim=-1)/query.double().norm(dim=-1).clamp_min(1e-12)
    return dict(reachable_score_energy_fraction=ratio.mean().item(),
                residual_score_energy_fraction=(residual.square().sum(-1)/energy.clamp_min(1e-24)).mean().item(),
                baseline_attention_kl=base_kl.mean().item(), projected_attention_kl=projected_kl.mean().item(),
                complemented_attention_kl=full_kl.mean().item(),
                complemented_probability_max_error=(full-target).abs().max().item(),
                score_reconstruction_max_error=(centered-projected-residual).abs().max().item(),
                residual_orthogonality_error=orthogonality.max().item(),
                projected_query_delta_relative_norm=query_ratio.mean().item())
