"""Compact Teacher/native residual mapper with query-conditioned directions."""
import torch

from .query_mapper import QueryValueMapper


class NativeResidualMapper(QueryValueMapper):
    def __init__(self, gate_hidden=16, **kwargs):
        super().__init__(**kwargs)
        if self.attention_features:
            raise ValueError('Compact residual mapper uses the original two token features')
        self.config.update(kind='native_residual_query', gate_hidden=gate_hidden)
        rank = self.config['rank']
        self.gates = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Linear(3*rank+3, gate_hidden), torch.nn.SiLU(),
            torch.nn.Linear(gate_hidden, 1)) for _ in self.layers])
        for gate in self.gates:
            torch.nn.init.normal_(gate[-1].weight, std=.01)
            torch.nn.init.constant_(gate[-1].bias, -2.2)
        for decoder in self.decoders:
            torch.nn.init.zeros_(decoder.weight)
            torch.nn.init.zeros_(decoder.bias)

    def proposal(self, relative_layer, source, native):
        native = native.float()
        latent = self.encoders[relative_layer](source.float())+self.native[relative_layer](native)
        scale = native.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-4)
        return native+self.decoders[relative_layer](latent)*scale, latent, scale
