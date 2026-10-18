"""Token- and query-dependent V correction; native backing K/V stay untouched."""
import torch
import torch.nn.functional as F


class ValueMapper(torch.nn.Module):
    def __init__(self, layers, teacher_layers, source_dim=512, value_dim=256,
                 query_dim=1536, rank=64, seed=42):
        super().__init__()
        self.config = dict(layers=layers, teacher_layers=teacher_layers, source_dim=source_dim,
                           value_dim=value_dim, query_dim=query_dim, rank=rank, seed=seed)
        self.layers = list(layers)
        self.teacher_layers = list(teacher_layers)
        self.encoders = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.LayerNorm(source_dim), torch.nn.Linear(source_dim, rank),
                                torch.nn.SiLU()) for _ in layers])
        self.decoders = torch.nn.ModuleList([torch.nn.Linear(rank, value_dim) for _ in layers])
        self.query = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.LayerNorm(query_dim), torch.nn.Linear(query_dim, rank),
                                torch.nn.SiLU()) for _ in layers])
        self.native = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.LayerNorm(value_dim), torch.nn.Linear(value_dim, rank),
                                torch.nn.SiLU()) for _ in layers])
        self.gates = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(rank * 3 + 3, 32), torch.nn.SiLU(),
                                torch.nn.Linear(32, 1)) for _ in layers])
        for gate in self.gates:
            torch.nn.init.normal_(gate[-1].weight, std=0.01)
            torch.nn.init.constant_(gate[-1].bias, -2.2)

    def proposal(self, relative_layer, source, native):
        latent = self.encoders[relative_layer](source.float())
        scale = native.float().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-4)
        mapped = self.decoders[relative_layer](latent) * scale
        return mapped, latent, scale

    def correction(self, relative_layer, source, native, query, features,
                   mode='gated', alpha=1.0, shuffle=False):
        native = native.float()
        mapped, latent, scale = self.proposal(relative_layer, source, native)
        if mode == 'gated':
            q = self.query[relative_layer](query.float()).expand(source.shape[0], -1)
            sv = self.native[relative_layer](native)
            cosine = F.cosine_similarity(mapped, native, dim=-1)[:, None]
            gate_features = torch.cat((latent, sv, q, features, cosine), -1)
            gate = self.gates[relative_layer](gate_features).sigmoid() * alpha
        else:
            gate = torch.full_like(scale, alpha)
        if shuffle:
            # Keep the correct-source gate and location scores unchanged;
            # shuffle only the mapped content direction (same per-token scale).
            generator = torch.Generator(device=source.device).manual_seed(self.config['seed'])
            permutation = torch.randperm(source.shape[0], generator=generator, device=source.device)
            direction = self.decoders[relative_layer](latent)
            mapped = direction[permutation] * scale
        delta = 2 * scale * torch.tanh((mapped-native) / (2 * scale))
        return native + gate * delta, gate, delta / scale


class Injection:
    def __init__(self, controller):
        self.controller = controller
        self.network = None
        self.positions = None
        self.source = None
        self.prior = None
        self.mode = 'gated'
        self.alpha = 1.0
        self.shuffle = False
        self.regularizers = []
        self.gate_stats = []

    def bind(self, network, positions, source, prior, mode, alpha, shuffle=False):
        self.network, self.positions, self.source, self.prior = network, positions, source, prior
        self.mode, self.alpha, self.shuffle = mode, alpha, shuffle
        self.regularizers, self.gate_stats = [], []
        self.controller.kv_transform = self

    def clear(self):
        self.controller.kv_transform = None
        self.network = self.source = self.positions = self.prior = None
        self.regularizers, self.gate_stats = [], []

    def __call__(self, layer, k, v, visible_positions, hidden, query_positions):
        if layer not in self.network.layers or self.positions.numel() == 0 or self.alpha == 0:
            return k, v
        assert hidden.shape[1] == 1, 'Draft queries must be sequential for query-dependent gates'
        relative = self.network.layers.index(layer)
        indices = torch.searchsorted(visible_positions, self.positions)
        assert torch.equal(visible_positions[indices], self.positions)
        native = v[0, :, indices, :].transpose(0, 1).reshape(len(indices), -1)
        n = self.prior.shape[-1]
        attention = self.prior[self.positions].clamp_min(1e-12).log() / 10
        position = self.positions.float() / max(1, n-1)
        features = torch.stack((attention, position), -1)
        corrected, gate, delta = self.network.correction(
            relative, self.source[relative], native, hidden[0, -1], features,
            self.mode, self.alpha, self.shuffle)
        result = v.clone()
        result[0, :, indices, :] = corrected.reshape(len(indices), v.shape[1], v.shape[-1]).transpose(0, 1).to(v.dtype)
        self.regularizers.append((gate.square() * delta.square().mean(-1, keepdim=True)).mean())
        self.gate_stats.append(gate.detach().flatten())
        # K is returned by identity; native cache was updated before this hook.
        return k, result

    def penalty(self):
        return torch.stack(self.regularizers).mean()

    def stats(self):
        if not self.gate_stats:
            return dict(mean=0.0, std=0.0, above_01=0.0)
        values = torch.cat(self.gate_stats).float()
        return dict(mean=float(values.mean()), std=float(values.std(unbiased=False)),
                    above_01=float((values > 0.1).float().mean()))
