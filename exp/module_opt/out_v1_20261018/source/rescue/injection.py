"""Request-local projected-content memo and in-place working-V correction."""
import torch
import torch.nn.functional as F

from .selection import mandatory


class ContentInjector:
    def __init__(self, network, capacity, device, tokens=128, strength=.5, memoize=True, optimized=False,
                 memo_affine=False, layers=None):
        self.network = network
        self.tokens, self.strength, self.memoize = tokens, strength, memoize
        self.optimized = optimized
        self.memo_affine = memo_affine and optimized
        self.gate_static = {}
        self.affine_memo, self.gate_weights = {}, {}
        self.valid = torch.zeros(capacity, device=device, dtype=torch.bool)
        self.memo = {}
        self.active = {}
        self.projected_positions = 0
        self.affine_positions = 0
        self.routes = []
        self.capture_summary, self.summaries = False, []
        self.query_residual = bool(getattr(network,'has_query_residual',False))
        self.residual_static = {}
        if network is not None:
            network.eval().requires_grad_(False)
            c = network.config
            if layers is not None and not set(layers) <= set(network.layers):
                raise ValueError('Injection route contains a layer absent from the mapper')
            self.routes = [(r, l, t) for r, (l, t) in enumerate(zip(network.layers, network.teacher_layers))
                           if layers is None or l in layers]
            self.layer_indices = torch.tensor([l for _, l, _ in self.routes], device=device, dtype=torch.long)
            for relative, layer, _ in self.routes if tokens and strength else []:
                self.memo[layer] = [torch.empty(capacity, d, device=device) for d in
                                    (c['value_dim'], c['rank'], c['rank'], 1, c['value_dim'])]
                if self.memo_affine:
                    first = network.gates[relative][0]
                    rank = c['rank']
                    self.gate_weights[layer] = (
                        torch.cat((first.weight[:, :2*rank], first.weight[:, 3*rank+2:]), -1).detach(),
                        first.weight[:, 3*rank:3*rank+2].detach().contiguous(),
                        first.weight[:, 2*rank:3*rank].detach().contiguous())
                    self.affine_memo[layer] = torch.empty(capacity, first.out_features, device=device)

    @torch.no_grad()
    def bind(self, teacher, student, positions, prior, recent, working=None, priority=None):
        self.active = {}
        if self.network is None or not self.tokens or not self.strength or not self.routes:
            return
        remote = positions[~torch.isin(positions, mandatory(len(prior), recent, positions.device))]
        ranking = prior if priority is None else priority
        inject = remote[torch.argsort(ranking[remote], descending=True, stable=True)[:self.tokens]].sort().values
        if not inject.numel():
            return
        missing = inject[~self.valid[inject]] if self.memoize else inject
        if missing.numel():
            sources = {}
            for relative, layer, source_layer in self.routes:
                if source_layer not in sources:
                    sources[source_layer] = teacher[source_layer][1][0, :, missing].transpose(0, 1).flatten(1).to(torch.bfloat16).float()
                native = student[layer][1][0, :, missing].transpose(0, 1).flatten(1).float()
                mapped, latent, scale = self.network.proposal(relative, sources[source_layer], native)
                sv = self.network.native[relative](native)
                cosine = F.cosine_similarity(mapped, native, dim=-1)[:, None]
                delta = 2*scale*torch.tanh((mapped-native)/(2*scale))
                if self.query_residual:
                    delta = (mapped-native)/scale
                for target, value in zip(self.memo[layer], (native, latent, sv, cosine, delta)):
                    target[missing] = value
                if self.memo_affine:
                    content = torch.cat((latent, sv, cosine), -1)
                    self.affine_memo[layer][missing] = F.linear(content, self.gate_weights[layer][0],
                                                              self.network.gates[relative][0].bias)
                    self.affine_positions += len(missing)
            self.valid[missing] = True
            self.projected_positions += len(missing)
        indices = torch.searchsorted(positions, inject)
        if working is not None:
            indices = working.physical_indices(indices)
            working.mark_dirty(self.layer_indices, indices)
        features = torch.stack((prior[inject].clamp_min(1e-12).log()/10,
                                inject.float()/max(1, len(prior)-1)), -1)
        for relative, layer, _ in self.routes:
            if self.query_residual:
                first=self.network.residuals[relative][0]
                rank=self.network.config['rank']
                latent,sv=self.memo[layer][1][inject],self.memo[layer][2][inject]
                static=torch.cat((latent,sv,features),-1)
                weight=torch.cat((first.weight[:,:2*rank],first.weight[:,3*rank:]),-1)
                self.residual_static[layer]=F.linear(static,weight,first.bias)
            if self.memo_affine:
                native, _, _, _, delta = self.memo[layer]
                self.active[layer] = (relative, indices, features, native[inject], None, None, None, delta[inject])
                self.gate_static[layer] = self.affine_memo[layer][inject]+F.linear(features, self.gate_weights[layer][1])
                continue
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
                self.affine_positions += len(inject)

    @torch.no_grad()
    def apply(self, layer, working_v, hidden):
        if layer not in self.active:
            return working_v
        relative, indices, features, native, latent, sv, cosine, delta = self.active[layer]
        q = self.network.query[relative](hidden[0, -1].float()).expand(len(indices), -1)
        if self.optimized:
            first, _, last = self.network.gates[relative]
            rank = self.network.config['rank']
            weight = self.gate_weights[layer][2] if self.memo_affine else first.weight[:, 2*rank:3*rank]
            dynamic = F.linear(q[:1], weight)
            gate = last(F.silu(self.gate_static[layer]+dynamic)).sigmoid()*self.strength
        else:
            gate = self.network.gates[relative](torch.cat((latent, sv, q, features, cosine), -1)).sigmoid()*self.strength
        if self.query_residual:
            block=self.network.residuals[relative]
            rank=self.network.config['rank']
            direction=block[-1](F.silu(self.residual_static[layer]+F.linear(q[:1],block[0].weight[:,2*rank:3*rank])))
            scale=native.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-4)
            delta=2*scale*torch.tanh((delta+direction)/2)
        corrected = native+gate*delta
        if self.capture_summary:
            scale = native.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-4)
            self.summaries.append(torch.stack((gate.mean(), (gate*delta/scale).square().mean().sqrt())))
        # The buffer belongs exclusively to WorkingCache. Native KV and the
        # memoized native values never receive the correction.
        working_v[0, :, indices] = corrected.reshape(len(indices), working_v.shape[1], -1).transpose(0, 1).to(working_v.dtype)
        return working_v

    def clear(self):
        self.active = {}
        self.gate_static = {}
        self.residual_static = {}
        self.capture_summary, self.summaries = False, []
