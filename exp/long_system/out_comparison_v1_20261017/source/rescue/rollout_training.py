"""Recorded joint-rollout supervision and differentiable last-layer replay.

Only the final Draft layer is adapted here. No claim is made that this replay
trains earlier layers through the whole historical cache or arbitrary routes.
"""
import torch
import torch.nn.functional as F

from . import features
from .efficient_attention import attention_rows


def injection_features(network, relative, source, native, normalized_hidden, token_features,
                       query, keys, indices, strength, layer_fraction, budget):
    corrected, gates, normalized_delta = network.correction(relative, source, native,
        normalized_hidden[0, -1], token_features, alpha=strength)
    mass = attention_rows(query, keys)[:, 0, indices].sum(-1).mean()
    extra = torch.stack((gates.mean(), (gates*normalized_delta).square().mean().sqrt(),
                         mass, mass.new_tensor(layer_fraction), mass.new_tensor(len(indices)/budget)))
    return corrected, extra


class RolloutCollector:
    def __init__(self, student, max_records=24):
        self.student, self.max_records = student, max_records
        self.engine = None
        self.pending, self.records, self.frames = [], [], []

    @torch.no_grad()
    def __call__(self, packet):
        engine, model = self.engine, self.student
        layer = len(model.model.layers)-1
        injector = packet['injector']
        active = injector.active.get(layer)
        record = dict(extra=torch.zeros(5))
        if active is not None:
            relative, indices, token_features, native, *_ = active
            absolute = engine.working.positions[indices]
            teacher_layer = injector.network.teacher_layers[relative]
            source = engine.native_t[teacher_layer][1][0, :, absolute].transpose(0, 1).flatten(1).to(torch.bfloat16).float()
            _, extra = injection_features(injector.network, relative, source, native,
                packet['normalized_hidden'], token_features, packet['query'], packet['keys'], indices,
                injector.strength, len(injector.routes)/len(model.model.layers), len(engine.working.positions))
            record['extra'] = extra.cpu()
            if len(self.records)+len(self.pending) < self.max_records:
                original = packet['values'].clone()
                original[0, :, indices] = native.reshape(len(indices), original.shape[1], -1).transpose(0, 1).to(original.dtype)
                record.update(relative=relative, indices=indices.cpu(), source=source.cpu(), native=native.cpu(),
                    token_features=token_features.cpu(), query=packet['query'].cpu().clone(), keys=packet['keys'].cpu().clone(),
                    values=original.cpu(), residual=packet['residual'].cpu(),
                    normalized_hidden=packet['normalized_hidden'].cpu(), strength=injector.strength)
        self.pending.append(record)

    def on_logits(self, logits):
        if 'query' in self.pending[-1]:
            self.pending[-1]['student_logits'] = logits.detach().cpu()

    @torch.no_grad()
    def round(self, packet):
        assert len(self.pending) == len(packet['draft'])
        prepared = features.prepare(packet['teacher_state'], packet['student_state'],
            packet['progress'], packet['remaining'], self.engine.config.recent)
        vector = features.finish(prepared, packet['teacher_state'], packet['student_state'], packet['selected'])
        last = packet['selector'].last_refresh
        vector[3] = (packet['round_index']-last)/self.engine.config.refresh_max if last is not None else 0
        first = torch.cat((vector.cpu(), self.pending[0]['extra']))
        labels = []
        for index, token in enumerate(packet['draft']):
            correct = int(packet['target'][index].argmax())
            label = 0 if token == correct else 1
            if token == correct and token == getattr(self, 'eos_token_id', None):
                label = 2
            labels.append(label)
            if label != 0:
                break
        self.frames.append(dict(features=first, labels=labels, prefix=packet['prefix'],
            accepted=packet['accepted'], proposed=len(packet['draft'])))
        for index, record in enumerate(self.pending):
            if 'query' in record and index <= packet['accepted']:
                record.update(target=packet['target'][index].cpu(), query_index=index,
                              prefix=packet['prefix'], risk_features=first)
                self.records.append(record)
        self.pending.clear()


def replay(student, mapper, record, strength=None):
    device = student.model.embed_tokens.weight.device
    r = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in record.items()}
    alpha = r['strength'] if strength is None else strength
    corrected, gates, delta = mapper.correction(r['relative'], r['source'], r['native'],
        r['normalized_hidden'][0, -1], r['token_features'], alpha=alpha)
    values = r['values'].index_copy(2, r['indices'], corrected.reshape(len(r['indices']),
        r['values'].shape[1], -1).transpose(0, 1)[None].to(r['values'].dtype))
    out = F.scaled_dot_product_attention(r['query'], r['keys'], values, dropout_p=0., enable_gqa=True)
    layer = student.model.layers[-1]
    hidden = r['residual']+layer.self_attn.o_proj(out.transpose(1, 2).reshape(1, 1, -1))
    hidden = hidden+layer.mlp(layer.post_attention_layernorm(hidden))
    logits = student.lm_head(student.model.norm(hidden))[0, -1].float()
    if 'target' in r:
        logits = logits[:r['target'].numel()]
    regularizer = (gates.square()*delta.square().mean(-1, keepdim=True)).mean()
    return logits, regularizer


def output_loss(logits, target, regularizer):
    target = target.to(logits.device).float()
    lp, lq = target.log_softmax(-1), logits.log_softmax(-1)
    return (lp.exp()*(lp-lq)).sum()+.25*F.cross_entropy(logits[None], target.argmax()[None])+.01*regularizer
