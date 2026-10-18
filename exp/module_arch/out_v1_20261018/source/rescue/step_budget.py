"""Causal two-to-four draft decision from attention and observed confidence.

Classes 0..4 encode a first rejection after 0..3 accepts, or four accepts.
Classes 5..8 encode an accepted EOS at draft positions 1..4. Short accepted
rollouts provide partial class sets, not invented rejection labels.
"""
import math
import random

import torch


def compact_attention(t, s, selected, age, progress, remaining, last_acceptance=.75):
    ta, sa = t.a.mean(0), s.a.mean(0)
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


def outcome_tables(prefix, remaining, cost, device='cpu'):
    accepts = torch.tensor([0, 1, 2, 3, 4, 1, 2, 3, 4], device=device)
    eos = torch.tensor([False]*5+[True]*4, device=device)
    caps = torch.tensor([min(2, remaining), min(4, remaining)], device=device)
    accepted = torch.minimum(accepts[:, None], caps[None])
    ends_eos = eos[:, None] & (accepts[:, None] <= caps[None])
    committed = torch.minimum(accepted+(~ends_eos).long(), caps.new_tensor(remaining)).float()
    # When the next Teacher bonus is EOS, the engine also omits catchup.
    terminal_bonus = eos[:, None] & (accepts[:, None] == caps[None]+1)
    catchup = (accepted == caps[None]) & (~ends_eos) & (~terminal_bonus) & (committed < remaining)
    nf = prefix/2048
    b = cost['verify_beta']
    seconds = (cost.get('fixed_round', 0.)+b[0]+b[1]*nf+
        (b[2]+b[3]*nf+cost['student_step'])*caps.float()[None]+
        catchup.float()*cost['student_step']+cost.get('cache_update', 0.))
    return committed, seconds


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

    @torch.no_grad()
    def decide(self, x, cost, prefix, remaining):
        if remaining <= 2 or self.rng.random()<self.exploration:
            return min(4, remaining), 0.
        probability = self.probabilities(x)
        committed, seconds = outcome_tables(prefix, remaining, cost, x.device)
        rate = (probability@committed)/(probability@seconds).clamp_min(1e-6)
        cap = 2 if bool(rate[0] > rate[1]*(1+self.margin)) else min(4, remaining)
        return cap, float((rate[0]/rate[1]-1).item())

    def payload(self):
        return dict(kind='step_budget', hidden=self.hidden, prior_weight=self.prior_weight,
                    margin=self.margin, state_dict=self.state_dict(), seed=self.seed)

    @classmethod
    def from_payload(cls, payload, device):
        model = cls(hidden=payload['hidden'], prior_weight=payload['prior_weight'],
                    margin=payload['margin'], seed=payload.get('seed', 412)).to(device)
        model.load_state_dict(payload['state_dict'])
        return model.eval().requires_grad_(False)
