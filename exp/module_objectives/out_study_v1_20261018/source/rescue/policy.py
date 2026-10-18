"""Small calibrated horizon scorer; no dependency on experiment import paths."""
import json
import torch


class RiskModel(torch.nn.Module):
    def __init__(self, group='attention_read', horizon=12, hidden=32):
        super().__init__()
        groups = dict(attention_read=list(range(28)), attention=[i for i in range(28) if i not in (14, 15)],
                      attention_injected=list(range(36)))
        if group not in groups:
            raise ValueError('System risk model uses attention or attention_read features')
        self.group, self.horizon = group, horizon
        indices = groups[group]
        self.register_buffer('indices', torch.tensor(indices))
        self.register_buffer('mean', torch.zeros(len(indices)))
        self.register_buffer('scale', torch.ones(len(indices)))
        self.net = torch.nn.Sequential(torch.nn.Linear(len(indices), hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, horizon*3))
        self.temperature = 1.

    def forward(self, x):
        z = (x[..., self.indices]-self.mean)/self.scale.clamp_min(1e-4)
        return self.net(z).reshape(*x.shape[:-1], self.horizon, 3)/self.temperature

    def continuation(self, x):
        return self(x).softmax(-1)[..., 0].cumprod(-1)

    def calibrated(self, scale, accept_bias):
        """Fold a training-only two-parameter hazard calibration into weights."""
        import copy
        import math
        if not math.isfinite(scale) or scale <= 0 or not math.isfinite(accept_bias):
            raise ValueError('Invalid hazard calibration')
        model = copy.deepcopy(self)
        with torch.no_grad():
            model.net[-1].weight.mul_(scale)
            model.net[-1].bias.mul_(scale)
            model.net[-1].bias.reshape(self.horizon, 3)[:, 0].add_(accept_bias*self.temperature)
        return model.eval().requires_grad_(False)

    @classmethod
    def load(cls, path, device):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        model = cls(payload['group'], payload['horizon']).to(device)
        model.load_state_dict(payload['state_dict'])
        model.temperature = payload['temperature']
        model = model.eval().requires_grad_(False)
        if 'prior_hazards' in payload:
            model = HazardMixture(model, payload['prior_hazards'], payload['prior_weight'])
        return model


class HazardMixture(torch.nn.Module):
    """Training-only depth-prior shrinkage, with an explicit featureless control."""
    def __init__(self, base, prior, weight):
        super().__init__()
        if not 0 <= weight <= 1:
            raise ValueError('Prior weight must be in [0,1]')
        self.base, self.weight = base, float(weight)
        self.group, self.horizon = base.group, base.horizon
        self.requires_features = weight < 1
        self.register_buffer('prior', torch.as_tensor(prior, dtype=base.mean.dtype, device=base.mean.device))
        if self.prior.shape != (base.horizon,3) or not torch.isfinite(self.prior).all() or (self.prior<=0).any():
            raise ValueError('Invalid conditional depth prior')
        if not torch.allclose(self.prior.sum(-1), torch.ones_like(self.prior[:,0]), atol=1e-5):
            raise ValueError('Conditional hazards must sum to one')

    def forward(self, x):
        if not self.requires_features:
            return self.prior.log().expand(*x.shape[:-1],self.horizon,3)
        return ((1-self.weight)*self.base(x).softmax(-1)+self.weight*self.prior).clamp_min(1e-12).log()

    def continuation(self, x):
        return self(x).softmax(-1)[...,0].cumprod(-1)


@torch.no_grad()
def choose(model, vector, cost, n, remaining, max_draft=12):
    survival = model.continuation(vector).clamp(0, 1)
    caps = sorted({min(k, remaining) for k in (1, 2, 4, 8, 12) if k <= max_draft})
    k = torch.tensor(caps, device=vector.device, dtype=torch.long)
    cumulative = torch.cat((survival.new_zeros(1), survival.cumsum(0)))
    expected = 1+cumulative[torch.minimum(k, k.new_tensor(remaining-1))]
    nf = n/2048
    design = torch.stack((torch.ones_like(k), k.new_full(k.shape, 0).float()+nf,
                          k.float(), k.float()*nf), -1).float()
    verification = design@vector.new_tensor(cost['verify_beta'])
    seconds = (cost.get('fixed_round', 0.)+k*cost['student_step']+verification+cost.get('cache_update', 0.)+
               torch.where(k < remaining, survival[k-1]*cost['student_step'], 0.))
    scores = expected/seconds.clamp_min(1e-5)
    best = int(scores.argmax())
    return caps[best], float(scores[best])


def load_cost(path):
    cost = json.loads(open(path).read())
    if len(cost.get('verify_beta', [])) != 4 or cost.get('student_step', 0) <= 0:
        raise ValueError('Invalid cost calibration')
    return cost
