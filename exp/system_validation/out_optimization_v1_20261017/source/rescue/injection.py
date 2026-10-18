"""Request-local projected-content memo and in-place working-V correction."""
import torch
import torch.nn.functional as F

from .selection import mandatory


class ContentInjector:
    def __init__(self, network, capacity, device, tokens=128, strength=.5, memoize=True, optimized=False):
        self.network = network
        self.tokens, self.strength, self.memoize = tokens, strength, memoize
        self.optimized = optimized
        self.gate_static = {}
        self.valid = torch.zeros(capacity, device=device, dtype=torch.bool)
        self.memo = {}
        self.active = {}
        self.projected_positions = 0
        if network is not None:
            network.eval().requires_grad_(False)
            c = network.config
            for layer in network.layers:
                self.memo[layer] = [torch.empty(capacity, d, device=device) for d in
                                    (c['value_dim'], c['rank'], c['rank'], 1, c['value_dim'])]

    @torch.no_grad()
    def bind(self, teacher, student, positions, prior, recent):
        self.active = {}
        if self.network is None or not self.tokens or not self.strength:
            return
        remote = positions[~torch.isin(positions, mandatory(len(prior), recent, positions.device))]
        inject = remote[torch.argsort(prior[remote], descending=True, stable=True)[:self.tokens]].sort().values
        if not inject.numel():
            return
        missing = inject[~self.valid[inject]] if self.memoize else inject
        if missing.numel():
            sources = {}
            for relative, (layer, source_layer) in enumerate(zip(self.network.layers, self.network.teacher_layers)):
                if source_layer not in sources:
                    sources[source_layer] = teacher[source_layer][1][0, :, missing].transpose(0, 1).flatten(1).to(torch.bfloat16).float()
                native = student[layer][1][0, :, missing].transpose(0, 1).flatten(1).float()
                mapped, latent, scale = self.network.proposal(relative, sources[source_layer], native)
                sv = self.network.native[relative](native)
                cosine = F.cosine_similarity(mapped, native, dim=-1)[:, None]
                delta = 2*scale*torch.tanh((mapped-native)/(2*scale))
                for target, value in zip(self.memo[layer], (native, latent, sv, cosine, delta)):
                    target[missing] = value
            self.valid[missing] = True
            self.projected_positions += len(missing)
        indices = torch.searchsorted(positions, inject)
        features = torch.stack((prior[inject].clamp_min(1e-12).log()/10,
                                inject.float()/max(1, len(prior)-1)), -1)
        for relative, layer in enumerate(self.network.layers):
            self.active[layer] = (relative, indices, features, *(value[inject] for value in self.memo[layer]))
            if self.optimized:
                _, _, _, _, latent, sv, cosine, _ = self.active[layer]
                linear = self.network.gates[relative][0]
                rank = self.network.config['rank']
                # W[latent, native, query, features, cosine]+b. Only query
                # changes within the round; fold the other terms once.
                static = torch.cat((latent, sv, features, cosine), -1)
                weight = torch.cat((linear.weight[:, :2*rank], linear.weight[:, 3*rank:]), -1)
                self.gate_static[layer] = F.linear(static, weight, linear.bias)

    @torch.no_grad()
    def apply(self, layer, working_v, hidden):
        if layer not in self.active:
            return working_v
        relative, indices, features, native, latent, sv, cosine, delta = self.active[layer]
        q = self.network.query[relative](hidden[0, -1].float()).expand(len(indices), -1)
        if self.optimized:
            first, _, last = self.network.gates[relative]
            rank = self.network.config['rank']
            dynamic = F.linear(q[:1], first.weight[:, 2*rank:3*rank])
            gate = last(F.silu(self.gate_static[layer]+dynamic)).sigmoid()*self.strength
        else:
            gate = self.network.gates[relative](torch.cat((latent, sv, q, features, cosine), -1)).sigmoid()*self.strength
        corrected = native+gate*delta
        # The buffer belongs exclusively to WorkingCache. Native KV and the
        # memoized native values never receive the correction.
        working_v[0, :, indices] = corrected.reshape(len(indices), working_v.shape[1], -1).transpose(0, 1).to(working_v.dtype)
        return working_v

    def clear(self):
        self.active = {}
        self.gate_static = {}
