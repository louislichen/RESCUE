"""A cached content proposal plus a query-conditioned vector correction.

The original mapper changes a scalar gate with the query. This variant can
also rotate its V correction direction. Zero output initialization preserves
the original mapper before training. Keys and backing values stay native.
"""
import torch
import torch.nn.functional as F
from exp.selective_kv.bridge import ValueMapper
from .efficient_attention import attention_rows


def mapper_features(mapper,features,query,keys,indices):
    heads=getattr(mapper,'attention_features',0)
    if not heads or features.shape[-1]==2+heads:
        return features
    rows=attention_rows(query,keys)[:,0,indices]
    weights=rows.reshape(heads,-1,len(indices)).mean(1).T
    return torch.cat((features,weights.clamp_min(1e-12).log()/10),-1)


class QueryValueMapper(ValueMapper):
    has_query_residual = True

    def __init__(self, residual_rank=32, attention_features=0, **kwargs):
        super().__init__(**kwargs)
        rank=self.config['rank'];dim=self.config['value_dim']
        self.attention_features=attention_features
        self.config.update(kind='query_attention_residual' if attention_features else 'query_residual',
            residual_rank=residual_rank,attention_features=attention_features)
        self.residuals=torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Linear(3*rank+2+attention_features,residual_rank),torch.nn.SiLU(),
            torch.nn.Linear(residual_rank,dim)) for _ in self.layers])
        self.attention_gates=torch.nn.ModuleList([torch.nn.Linear(attention_features,1) for _ in self.layers] if attention_features else [])
        for gate in self.attention_gates:
            torch.nn.init.zeros_(gate.weight);torch.nn.init.zeros_(gate.bias)
        for block in self.residuals:
            torch.nn.init.zeros_(block[-1].weight);torch.nn.init.zeros_(block[-1].bias)

    def correction(self, relative_layer, source, native, query, features,
                   mode='gated', alpha=1., shuffle=False):
        native=native.float()
        mapped,latent,scale=self.proposal(relative_layer,source,native)
        q=self.query[relative_layer](query.float()).expand(len(native),-1)
        sv=self.native[relative_layer](native)
        cosine=F.cosine_similarity(mapped,native,dim=-1)[:,None]
        if features.shape[-1]!=2+self.attention_features:
            raise ValueError('Current-query attention features are required for this mapper')
        gate_logit=self.gates[relative_layer](torch.cat((latent,sv,q,features[:,:2],cosine),-1))
        if self.attention_features:
            gate_logit=gate_logit+self.attention_gates[relative_layer](features[:,2:])
        gate=(gate_logit.sigmoid()*alpha
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
    cls=QueryValueMapper if kind in ('query_residual','query_attention_residual') else ValueMapper
    if kind=='native_residual_query':
        from .residual_mapper import NativeResidualMapper
        cls=NativeResidualMapper
    if kind not in (None,'query_residual','query_attention_residual','native_residual_query'):
        raise ValueError('Unknown mapper checkpoint kind')
    model=cls(**config).to(device)
    model.load_state_dict(payload['state_dict'])
    return model
