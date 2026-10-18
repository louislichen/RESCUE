"""A cached content proposal plus a query-conditioned vector correction.

The original mapper changes a scalar gate with the query. This variant can
also rotate its V correction direction. Zero output initialization preserves
the original mapper before training. Keys and backing values stay native.
"""
import torch
import torch.nn.functional as F
from exp.selective_kv.bridge import ValueMapper


class QueryValueMapper(ValueMapper):
    has_query_residual = True

    def __init__(self, residual_rank=32, **kwargs):
        super().__init__(**kwargs)
        rank=self.config['rank'];dim=self.config['value_dim']
        self.config.update(kind='query_residual',residual_rank=residual_rank)
        self.residuals=torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Linear(3*rank+2,residual_rank),torch.nn.SiLU(),
            torch.nn.Linear(residual_rank,dim)) for _ in self.layers])
        for block in self.residuals:
            torch.nn.init.zeros_(block[-1].weight);torch.nn.init.zeros_(block[-1].bias)

    def correction(self, relative_layer, source, native, query, features,
                   mode='gated', alpha=1., shuffle=False):
        native=native.float()
        mapped,latent,scale=self.proposal(relative_layer,source,native)
        q=self.query[relative_layer](query.float()).expand(len(native),-1)
        sv=self.native[relative_layer](native)
        cosine=F.cosine_similarity(mapped,native,dim=-1)[:,None]
        gate=(self.gates[relative_layer](torch.cat((latent,sv,q,features,cosine),-1)).sigmoid()*alpha
              if mode=='gated' else torch.full_like(scale,alpha))
        direction=self.residuals[relative_layer](torch.cat((latent,sv,q,features),-1))
        if shuffle:
            permutation=torch.randperm(len(native),generator=torch.Generator(device=native.device).manual_seed(self.config['seed']),device=native.device)
            mapped=mapped[permutation]/scale[permutation]*scale
            direction=direction[permutation]
        delta=2*scale*torch.tanh(((mapped-native)/scale+direction)/2)
        return native+gate*delta,gate,delta/scale


def mapper_from_payload(payload,device):
    config=dict(payload['config']);kind=config.pop('kind',None)
    cls=QueryValueMapper if kind=='query_residual' else ValueMapper
    if kind not in (None,'query_residual'):
        raise ValueError('Unknown mapper checkpoint kind')
    model=cls(**config).to(device)
    model.load_state_dict(payload['state_dict'])
    return model
