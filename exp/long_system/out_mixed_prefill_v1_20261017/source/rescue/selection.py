"""Shared per-model attention/read states and budgeted position selectors."""
import torch
import torch.nn.functional as F

from exp.attention_horizon.selection import ReadState


class LazyReadState(ReadState):
    """Same read equations; defer V conversion/full A·V until actually used."""
    def __init__(self, rows, values):
        self.a = rows.float()
        self.a = self.a/self.a.sum(-1, keepdim=True).clamp_min(1e-12)
        self._values, self._v, self._target, self._scale = values, None, None, None
        self.heads, self.n = self.a.shape
        self.groups = self.heads//values.shape[0]
        self.vheads = torch.arange(self.heads, device=rows.device)//self.groups

    @property
    def v(self):
        if self._v is None:
            self._v = self._values.float()
        return self._v

    @property
    def target(self):
        if self._target is None:
            self._target = torch.einsum('hgn,hnd->hgd',
                self.a.reshape(self.v.shape[0], self.groups, self.n), self.v).flatten(0, 1)
        return self._target

    @property
    def scale(self):
        if self._scale is None:
            self._scale = self.target.square().sum(-1).clamp_min(1e-6)
        return self._scale


def mandatory(n, recent, device):
    return torch.cat((torch.zeros(1, dtype=torch.long, device=device),
                      torch.arange(max(1, n-recent), n, device=device)))


class ContextSelector:
    def __init__(self, config):
        self.config = config
        self.previous = None
        self.version = 0
        self.refreshes = 0
        self.priority = None
        self._content, self._content_counts = [None, None], [0, 0]
        self.anchor, self.last_refresh = None, None
        self.last_drift, self.last_coverage = 0., 1.
        self.similarity_elements = 0

    def _normalized_content(self, state, index):
        values = state._values if hasattr(state, '_values') else state.v
        n, dim = state.n, values.shape[0]*values.shape[-1]
        old = self._content_counts[index]
        bank = self._content[index]
        if bank is None or len(bank) < n:
            grown = values.new_empty(1 << (n-1).bit_length(), dim, dtype=torch.float32)
            if bank is not None:
                grown[:old] = bank[:old]
            bank = self._content[index] = grown
        if n < old:
            old = 0
        if n > old:
            bank[old:n] = F.normalize(values[:, old:n].transpose(0, 1).flatten(1).float(), dim=-1)
        self._content_counts[index] = n
        return bank[:n]

    def _hierarchical(self, t, s, fixed, free, slots, joint):
        c, n = self.config, t.n
        weights = joint.clone()
        weights[fixed] = 0
        chunks = (n+c.coarse_chunk_size-1)//c.coarse_chunk_size
        scores, counts = weights.new_zeros(chunks), weights.new_zeros(chunks)
        chunk_ids = free//c.coarse_chunk_size
        scores.scatter_add_(0, chunk_ids, weights[free])
        counts.scatter_add_(0, chunk_ids, torch.ones_like(weights[free]))
        scores.masked_fill_(counts == 0, -torch.inf)
        take_chunks = min(int((counts > 0).sum()),
                          max((c.coverage_candidates+c.coarse_chunk_size-1)//c.coarse_chunk_size,
                              (slots+c.coarse_chunk_size-1)//c.coarse_chunk_size+2))
        coarse = scores.topk(take_chunks).indices
        pool = free[torch.isin(chunk_ids, coarse)]
        if len(pool) < slots:
            raise RuntimeError('Coarse pool cannot satisfy the actual token budget')
        support = free[weights[free].topk(min(c.coverage_rows, len(free))).indices]
        similarities, coverage, objective = [], [], []
        for index, state in enumerate((t, s)):
            content = self._normalized_content(state, index)
            sim = (content[support]@content[pool].T).clamp_min_(0)
            similarities.append(sim)
            self.similarity_elements += sim.numel()
            coverage.append((content[support]@content[fixed].T).clamp_min_(0).amax(-1))
            weight = state.a[:, support].mean(0)
            objective.append(weight/weight.sum().clamp_min(1e-12))
        chosen, used = [], torch.zeros(len(pool), device=pool.device, dtype=torch.bool)
        for offset in range(0, slots, c.coverage_batch):
            gains = sum(scale*((sim-cover[:, None]).clamp_min(0)*weight[:, None]).sum(0)
                        for scale, sim, cover, weight in zip((1., c.student_weight), similarities, coverage, objective))
            gains.masked_fill_(used, -torch.inf)
            take = gains.topk(min(c.coverage_batch, slots-offset)).indices
            used[take] = True
            for index, sim in enumerate(similarities):
                coverage[index] = torch.maximum(coverage[index], sim[:, take].amax(-1))
            chosen.append(pool[take])
        return torch.cat((fixed, *chosen)).sort().values

    @torch.no_grad()
    def select(self, t, s, round_index):
        c = self.config
        n, device = t.n, t.a.device
        if c.selector == 'full' or n <= c.budget:
            self.version += 1
            return torch.arange(n, device=device)
        fixed = mandatory(n, c.recent, device)
        if c.optimized_execution:
            free = torch.arange(1, max(1, n-c.recent), device=device)
        else:
            available = torch.arange(n, device=device)
            free = available[~torch.isin(available, fixed)]
        slots = c.budget-len(fixed)
        joint = t.a.mean(0)+c.student_weight*s.a.mean(0)
        self.priority = joint
        refresh = self.previous is None or round_index % c.refresh_every == 0
        remote = None
        if c.refresh_policy == 'adaptive':
            remote = joint.clone()
            remote[fixed] = 0
            remote /= remote.sum().clamp_min(1e-12)
            refresh = self.previous is None
            if not refresh:
                age = round_index-self.last_refresh
                anchor = F.pad(self.anchor, (0, n-len(self.anchor)))
                drift = .5*(remote-anchor).abs().sum()
                covered = remote[self.previous].sum()
                self.last_drift, self.last_coverage = float(drift), float(covered)
                refresh = age >= c.refresh_max or (age >= c.refresh_min and
                    (self.last_drift > c.attention_drift or self.last_coverage < c.coverage_floor))
        if refresh:
            self.last_refresh = round_index
            if remote is not None:
                self.anchor = remote.clone()
        if not refresh:
            # Strict budget and mandatory recent positions are maintained even
            # between expensive refreshes; history is not silently enlarged.
            old = (self.previous[(self.previous > 0) & (self.previous < n-c.recent)]
                   if c.optimized_execution else self.previous[~torch.isin(self.previous, fixed)])
            order = old[torch.argsort(joint[old], descending=True, stable=True)]
            fill = free[:0]
            if not c.optimized_execution or len(order) < slots:
                fill = free[~torch.isin(free, order)]
                fill = fill[torch.argsort(joint[fill], descending=True, stable=True)]
            selected = torch.cat((fixed, torch.cat((order, fill))[:slots])).sort().values
        elif c.selector == 'read_joint':
            self.refreshes += 1
            from exp.attention_horizon.selection import select
            selected = select('read_joint1', t, s, c.budget, c.recent, self.previous)
        elif c.selector == 'hierarchical':
            self.refreshes += 1
            selected = self._hierarchical(t, s, fixed, free, slots, joint)
        else:
            self.refreshes += 1
            scores = t.a.mean(0) if c.selector == 'teacher' else joint
            if c.chunk_size > 1:
                # Rank chunks, then expand positions; the last chunk may be
                # partial to enforce the same actual token budget.
                ids = torch.arange(n, device=device)//c.chunk_size
                sums = scores.new_zeros(int((n+c.chunk_size-1)//c.chunk_size))
                counts = sums.clone()
                sums.scatter_add_(0, ids, scores)
                counts.scatter_add_(0, ids, torch.ones_like(scores))
                scores = (sums/counts.clamp_min(1))[ids]
            if c.selector != 'coverage':
                selected = torch.cat((fixed, free[torch.argsort(scores[free], descending=True, stable=True)[:slots]])).sort().values
            else:
                pool = free[torch.argsort(scores[free], descending=True, stable=True)[:max(slots, c.coverage_candidates)]]
                if c.chunk_size > 1:
                    pool = free[torch.isin(free//c.chunk_size, (pool//c.chunk_size).unique())]
                similarities, coverage, weights = [], [], []
                for state in (t, s):
                    content = F.normalize(state.v.transpose(0, 1).flatten(1), dim=-1)
                    similarities.append((content@content[pool].T).clamp_min_(0))
                    coverage.append((content@content[fixed].T).clamp_min_(0).amax(-1))
                    weight = state.a.mean(0).clone()
                    weight[fixed] = 0
                    weights.append(weight/weight.sum().clamp_min(1e-12))
                if c.chunk_size == 1:
                    chosen = []
                    used = torch.zeros(len(pool), device=device, dtype=torch.bool)
                    for offset in range(0, slots, c.coverage_batch):
                        gains = sum(scale*((sim-cover[:, None]).clamp_min(0)*weight[:, None]).sum(0)
                                    for scale, sim, cover, weight in zip((1., c.student_weight), similarities, coverage, weights))
                        gains.masked_fill_(used, -torch.inf)
                        take = gains.topk(min(c.coverage_batch, slots-offset)).indices
                        used[take] = True
                        for i, sim in enumerate(similarities):
                            coverage[i] = torch.maximum(coverage[i], sim[:, take].amax(-1))
                        chosen.append(pool[take])
                else:
                    # A chunk's representation covers the maximum similarity
                    # over its member tokens, not a mean-value surrogate. Rank
                    # marginal gain per retained token; only the final chunk
                    # may be partial to meet the exact token budget.
                    groups, inverse = torch.unique(pool//c.chunk_size, return_inverse=True)
                    members = [[] for _ in range(len(groups))]
                    for i, group in enumerate(inverse.tolist()):
                        members[group].append(i)
                    counts = torch.tensor([len(v) for v in members], device=device)
                    aggregated = []
                    for sim in similarities:
                        grouped = sim.new_zeros(n, len(groups))
                        grouped.scatter_reduce_(1, inverse[None].expand(n, -1), sim, reduce='amax')
                        aggregated.append(grouped)
                    used = torch.zeros(len(groups), device=device, dtype=torch.bool)
                    chosen, left = [], slots
                    while left:
                        gains = sum(scale*((sim-cover[:, None]).clamp_min(0)*weight[:, None]).sum(0)
                                    for scale, sim, cover, weight in zip((1., c.student_weight), aggregated, coverage, weights))
                        gains = gains/counts
                        gains.masked_fill_(used, -torch.inf)
                        group = int(gains.argmax())
                        take = torch.tensor(members[group][:left], device=device)
                        used[group] = True
                        for i, sim in enumerate(similarities):
                            coverage[i] = torch.maximum(coverage[i], sim[:, take].amax(-1))
                        chosen.append(pool[take])
                        left -= len(take)
                selected = torch.cat((fixed, *chosen)).sort().values
        self.previous = selected
        self.version += 1
        return selected
