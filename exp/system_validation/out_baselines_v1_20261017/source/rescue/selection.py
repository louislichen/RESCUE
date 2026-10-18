"""Shared per-model attention/read states and budgeted position selectors."""
import torch
import torch.nn.functional as F

from exp.attention_horizon.selection import ReadState


def mandatory(n, recent, device):
    return torch.cat((torch.zeros(1, dtype=torch.long, device=device),
                      torch.arange(max(1, n-recent), n, device=device)))


class ContextSelector:
    def __init__(self, config):
        self.config = config
        self.previous = None
        self.version = 0
        self.refreshes = 0

    @torch.no_grad()
    def select(self, t, s, round_index):
        c = self.config
        n, device = t.n, t.a.device
        if c.selector == 'full' or n <= c.budget:
            self.version += 1
            return torch.arange(n, device=device)
        fixed = mandatory(n, c.recent, device)
        available = torch.arange(n, device=device)
        free = available[~torch.isin(available, fixed)]
        slots = c.budget-len(fixed)
        joint = t.a.mean(0)+c.student_weight*s.a.mean(0)
        refresh = self.previous is None or round_index % c.refresh_every == 0
        if not refresh:
            # Strict budget and mandatory recent positions are maintained even
            # between expensive refreshes; history is not silently enlarged.
            old = self.previous[~torch.isin(self.previous, fixed)]
            order = old[torch.argsort(joint[old], descending=True, stable=True)]
            fill = free[~torch.isin(free, order)]
            fill = fill[torch.argsort(joint[fill], descending=True, stable=True)]
            selected = torch.cat((fixed, torch.cat((order, fill))[:slots])).sort().values
        elif c.selector == 'read_joint':
            self.refreshes += 1
            from exp.attention_horizon.selection import select
            selected = select('read_joint1', t, s, c.budget, c.recent, self.previous)
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
