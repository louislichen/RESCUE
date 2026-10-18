"""Optional immutable projection packing, shared by RESCUE and controls."""
import torch
import torch.nn.functional as F


class PackedMLP(torch.nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.gate_up = torch.nn.Parameter(torch.cat((mlp.gate_proj.weight, mlp.up_proj.weight)), requires_grad=False)
        self.down_proj = mlp.down_proj

    def forward(self, hidden):
        gate, up = F.linear(hidden, self.gate_up).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate)*up)


class ProjectionGroup:
    def __init__(self, modules):
        self.sizes = [m.out_features for m in modules]
        self.weight = torch.cat([m.weight for m in modules]).detach()
        self.bias = torch.cat([m.bias for m in modules]).detach() if modules[0].bias is not None else None
        self.outputs, self.input = None, None

    def get(self, index, hidden):
        if index == 0:
            self.outputs = F.linear(hidden, self.weight, self.bias).split(self.sizes, dim=-1)
            self.input = hidden
        if self.outputs is None or self.input is not hidden:
            raise RuntimeError('Packed QKV requires consecutive Q/K/V calls with the same input')
        result = self.outputs[index]
        if index == 2:
            self.outputs = self.input = None
        return result


class PackedProjection(torch.nn.Module):
    def __init__(self, group, index, original):
        super().__init__()
        self.group, self.index = group, index
        self.in_features, self.out_features = original.in_features, original.out_features
        start = sum(group.sizes[:index])
        self.weight = torch.nn.Parameter(group.weight[start:start+self.out_features], requires_grad=False)
        if group.bias is not None:
            self.bias = torch.nn.Parameter(group.bias[start:start+self.out_features], requires_grad=False)
        else:
            self.register_parameter('bias', None)

    def forward(self, hidden):
        return self.group.get(self.index, hidden)


def pack_projections(model):
    for layer in model.model.layers:
        attention = layer.self_attn
        if isinstance(attention.q_proj, PackedProjection):
            continue
        originals = [attention.q_proj, attention.k_proj, attention.v_proj]
        group = ProjectionGroup(originals)
        for index, name in enumerate(('q_proj', 'k_proj', 'v_proj')):
            setattr(attention, name, PackedProjection(group, index, originals[index]))
        layer.mlp = PackedMLP(layer.mlp)
    return model


def share_packed_projections(source, architecture_port):
    """Share packed ops with a wrapper constructed from the identical weights.

    These engines execute sequentially. A QKV group owns temporary per-forward
    views and is not a concurrent serving object.
    """
    if len(source.model.layers) != len(architecture_port.model.layers):
        raise ValueError('Packed sharing requires matching layers')
    pack_projections(source)
    for left, right in zip(source.model.layers, architecture_port.model.layers):
        for name in ('q_proj', 'k_proj', 'v_proj'):
            current, target = getattr(left.self_attn, name), getattr(right.self_attn, name)
            if current.weight.shape != target.weight.shape:
                raise ValueError('Packed sharing requires identical projection shapes')
            setattr(right.self_attn, name, current)
        right.mlp = left.mlp
