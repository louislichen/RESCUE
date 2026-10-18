"""Small calibrated horizon scorer; no dependency on experiment import paths."""
import json
import torch


class RiskModel(torch.nn.Module):
    def __init__(self, group='attention_read', horizon=12, hidden=32):
        super().__init__()
        groups = dict(attention_read=list(range(28)), attention=[i for i in range(28) if i not in (14, 15)])
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

    @classmethod
    def load(cls, path, device):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        model = cls(payload['group'], payload['horizon']).to(device)
        model.load_state_dict(payload['state_dict'])
        model.temperature = payload['temperature']
        return model.eval().requires_grad_(False)


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
