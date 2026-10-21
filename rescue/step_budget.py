"""Causal two-to-four draft decision from attention and observed confidence.

Classes 0..4 encode a first rejection after 0..3 accepts, or four accepts.
Classes 5..8 encode an accepted EOS at draft positions 1..4. Short accepted
rollouts provide partial class sets, not invented rejection labels.
"""
import math
import random
from contextvars import ContextVar

import torch

from .attention_summary import mean_attention


_legacy_cap_calls = ContextVar('step_budget_legacy_cap_calls', default=())


def compact_attention(t, s, selected, age, progress, remaining, last_acceptance=.75):
    ta, sa = mean_attention(t), mean_attention(s)
    return torch.stack((ta.new_tensor(math.log(t.n)/10), ta[selected].sum(),
        sa[selected].sum(), (ta[selected]-sa[selected]).abs().sum(),
        ta.new_tensor(age/4), ta.new_tensor(progress/128),
        ta.new_tensor(remaining/128), ta.new_tensor(last_acceptance)))


def step_features(base, margins, moments):
    return torch.cat((base, torch.stack(margins[:2]).clamp(0, 30)/10,
                      moments.to(base.device).float()))


def compatible_classes(labels, proposed):
    if not 1<=proposed<=4 or not labels or len(labels)>proposed or any(y not in (0, 1, 2) for y in labels):
        raise ValueError('Invalid observed prefix labels')
    if any(y != 0 for y in labels[:-1]):
        raise ValueError('Labels continue after rejection/EOS')
    mask = torch.zeros(9, dtype=torch.bool)
    if labels[-1] == 1:
        mask[len(labels)-1] = True
    elif labels[-1] == 2:
        mask[4+len(labels)] = True
    elif proposed >= 4:
        mask[4] = True
    else:
        # No failure/EOS was observed: deeper continuations remain unknown.
        mask[proposed:5] = True
        mask[4+proposed+1:] = True
    return mask


def _outcome_geometry(remaining, device):
    """Exact class geometry, independent of prefix and mutable cost estimates."""
    accepts = torch.tensor([0, 1, 2, 3, 4, 1, 2, 3, 4], device=device)
    eos = torch.tensor([False]*5+[True]*4, device=device)
    caps = torch.tensor([min(2, remaining), min(4, remaining)], device=device)
    accepted = torch.minimum(accepts[:, None], caps[None])
    ends_eos = eos[:, None] & (accepts[:, None] <= caps[None])
    committed = torch.minimum(accepted+(~ends_eos).long(), caps.new_tensor(remaining)).float()
    # When the next Teacher bonus is EOS, the engine also omits catchup.
    terminal_bonus = eos[:, None] & (accepts[:, None] == caps[None]+1)
    catchup = (accepted == caps[None]) & (~ends_eos) & (~terminal_bonus) & (committed < remaining)
    return caps, committed, catchup


def _outcome_seconds(prefix, cost, caps, catchup):
    # Preserve the historical floating-point evaluation order. In particular,
    # do not fold cost coefficients into a differently associated expression.
    nf = prefix/2048
    b = cost['verify_beta']
    seconds = (cost.get('fixed_round', 0.)+b[0]+b[1]*nf+
        (b[2]+b[3]*nf+cost['student_step'])*caps.float()[None]+
        catchup.float()*cost['student_step']+cost.get('cache_update', 0.))
    return seconds


def outcome_tables(prefix, remaining, cost, device='cpu'):
    """Uncached public tables used by training and diagnostic callers."""
    caps, committed, catchup = _outcome_geometry(remaining, device)
    return committed, _outcome_seconds(prefix, cost, caps, catchup)


class StepBudget(torch.nn.Module):
    stepwise = True
    requires_features = False
    group = 'attention_step'
    horizon = 4

    def __init__(self, hidden=16, prior_weight=.25, margin=.02, exploration=0., seed=412):
        super().__init__()
        self.hidden, self.prior_weight, self.margin = hidden, prior_weight, margin
        self.exploration, self.seed = exploration, seed
        self.register_buffer('mean', torch.zeros(12))
        self.register_buffer('scale', torch.ones(12))
        self.register_buffer('prior', torch.ones(9)/9)
        self.net = torch.nn.Sequential(torch.nn.Linear(12, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, 9))
        # Geometry saturates at remaining >= 6, not >= 4: with five tokens left,
        # a four-accept path's bonus exhausts the output budget and omits catchup.
        # Nonpersistent buffers follow device moves without changing old
        # checkpoint/state_dict keys. Costs are deliberately never cached here.
        geometry = [_outcome_geometry(n, self.mean.device) for n in range(1, 7)]
        for name, index in (('_outcome_caps', 0), ('_outcome_committed', 1), ('_outcome_catchup', 2)):
            self.register_buffer(name, torch.stack([g[index] for g in geometry]), persistent=False)
        self.reset()

    def reset(self):
        self.last_acceptance = .75
        self.rng = random.Random(self.seed)

    def observe(self, labels):
        if labels:
            compatible_classes(labels, len(labels))
            self.last_acceptance = sum(y != 1 for y in labels)/len(labels)

    def forward(self, x):
        return self.net((x-self.mean)/self.scale.clamp_min(.05))

    def probabilities(self, x):
        return (1-self.prior_weight)*self(x).softmax(-1)+self.prior_weight*self.prior

    def _cached_outcome_tables(self, prefix, remaining, cost):
        index = min(remaining, 6)-1
        caps, catchup = self._outcome_caps[index], self._outcome_catchup[index]
        return self._outcome_committed[index].float(), _outcome_seconds(prefix, cost, caps, catchup)

    def _decide(self, x, cost, prefix, remaining, with_score):
        if remaining <= 2 or self.rng.random()<self.exploration:
            cap = min(4, remaining)
            return (cap, 0.) if with_score else cap
        probability = self.probabilities(x)
        committed, seconds = self._cached_outcome_tables(prefix, remaining, cost)
        rate = (probability@committed)/(probability@seconds).clamp_min(1e-6)
        cap = 2 if bool(rate[0] > rate[1]*(1+self.margin)) else min(4, remaining)
        return (cap, float((rate[0]/rate[1]-1).item())) if with_score else cap

    @torch.no_grad()
    def decide(self, x, cost, prefix, remaining):
        """Historical diagnostic API, including the relative-rate score."""
        return self._decide(x, cost, prefix, remaining, with_score=True)

    @torch.no_grad()
    def decide_cap(self, x, cost, prefix, remaining):
        """Online action only; avoid materializing an unused host score."""
        # An older external policy may override only decide(). Do not silently
        # bypass its action rule just because it inherits this new fast API.
        if type(self).decide is not StepBudget.decide:
            active = _legacy_cap_calls.get()
            if id(self) not in active:
                token = _legacy_cap_calls.set((*active, id(self)))
                try:
                    return self.decide(x, cost, prefix, remaining)[0]
                finally:
                    _legacy_cap_calls.reset(token)
            # A newer override can itself delegate to super().decide_cap().
            # Resolve that inner call directly, rather than recursing forever.
        return self._decide(x, cost, prefix, remaining, with_score=False)

    def payload(self):
        return dict(kind='step_budget', hidden=self.hidden, prior_weight=self.prior_weight,
                    margin=self.margin, state_dict=self.state_dict(), seed=self.seed)

    @classmethod
    def from_payload(cls, payload, device):
        model = cls(hidden=payload['hidden'], prior_weight=payload['prior_weight'],
                    margin=payload['margin'], seed=payload.get('seed', 412)).to(device)
        model.load_state_dict(payload['state_dict'])
        return model.eval().requires_grad_(False)
