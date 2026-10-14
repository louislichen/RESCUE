"""Equal-budget retrieval rules and a small token-utility predictor."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def normalize(x):
    return x / x.sum(-1, keepdim=True).clamp_min(1e-12)


def js_matrix(student, teacher):
    s, t = normalize(student).double(), normalize(teacher).double()
    m = (s[:, None] + t[None]) / 2
    return (0.5 * (s[:, None] * (s[:, None].clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(-1)
            + 0.5 * (t[None] * (t[None].clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(-1)).float()


def mandatory(n, recent, device):
    return torch.unique(torch.cat((torch.zeros(1, device=device, dtype=torch.long),
                                   torch.arange(max(0, n - recent), n, device=device))), sorted=True)


def rank(scores, candidates):
    return candidates[torch.argsort(scores[candidates], descending=True, stable=True)]


def retrieve(method, teacher, student, budget, recent=64, chunk=32, ratio=0.5,
             seed=42, visible=None, network=None, query=None):
    n = teacher.shape[-1]
    fixed = mandatory(n, recent, teacher.device)
    b = min(budget, n)
    if fixed.numel() > b:
        raise ValueError('Budget smaller than mandatory sink/recent positions')
    free = torch.ones(n, device=teacher.device, dtype=torch.bool)
    free[fixed] = False
    candidates = torch.arange(n, device=teacher.device)[free]
    slots = b - fixed.numel()
    if slots == 0:
        return fixed
    t, s = normalize(teacher[-1]), normalize(student[-1])
    observed = torch.ones(n, device=t.device, dtype=torch.bool) if visible is None else visible[-1]
    tr = rank(t, candidates)
    sr = rank(s, candidates[observed[candidates]])
    generator = torch.Generator(device=t.device).manual_seed(seed)
    if method == 'recent':
        chosen = candidates.flip(0)[:slots]
    elif method == 'random':
        chosen = candidates[torch.randperm(len(candidates), generator=generator, device=t.device)[:slots]]
    elif method.endswith('_chunk'):
        x = t if method.startswith('teacher') else s
        chunks = torch.div(candidates, chunk, rounding_mode='floor')
        scores = torch.zeros((n + chunk - 1) // chunk, device=t.device)
        counts = torch.zeros_like(scores)
        scores.scatter_add_(0, chunks, x[candidates])
        counts.scatter_add_(0, chunks, torch.ones_like(x[candidates]))
        scores /= counts.clamp_min(1)
        # Keep whole ranked chunks when possible, then clip one boundary chunk
        # by original position to enforce exactly the same token budget.
        chosen = candidates[torch.argsort(scores[chunks], descending=True, stable=True)][:slots]
    elif method == 'teacher_token':
        chosen = tr[:slots]
    elif method == 'student_token':
        chosen = torch.cat((sr[:slots], tr))
    elif method == 'union':
        nt = int(round(slots * ratio))
        peaks = torch.cat((tr[:nt], sr[:slots-nt]))
        chosen = torch.cat((peaks, tr))
    elif method in ('difference', 'difference_core'):
        if not bool(observed.all()):
            raise ValueError('Full-history attention difference needs common visibility')
        difference = t * (t - s).clamp_min(0)
        dr = rank(difference, candidates)
        if method == 'difference_core':
            core = int(round(slots * (1-ratio)))
            chosen = torch.cat((sr[:core], dr, tr))
        else:
            chosen = dr[:slots]
    elif method == 'learned':
        chosen = rank(network(token_features(teacher, student, query, visible)).detach(), candidates)[:slots]
    else:
        raise ValueError(method)
    # Preserve quota order while deduplicating; teacher ranking fills any overlap.
    if chosen.numel() > slots:
        used = set()
        unique = []
        for i in chosen.tolist():
            if i not in used:
                unique.append(i)
                used.add(i)
            if len(unique) == slots:
                break
        chosen = torch.tensor(unique, device=t.device, dtype=torch.long)
    result = torch.unique(torch.cat((fixed, chosen)), sorted=True)
    assert result.numel() == b, (method, result.numel(), b)
    return result


def token_features(teacher, student, query, visible=None):
    """Teacher-only deployable features plus frozen current-query summaries.

    Student rows are deliberately excluded from this predictor: online student
    attention is unavailable outside its working set. Full common-history
    teacher/student differences are evaluated separately as rule diagnostics.
    """
    t = normalize(teacher)
    n = t.shape[-1]
    grouped = torch.stack([t[a:b].mean(0) for a, b in [(0, 9), (9, 18), (18, t.shape[0])]])
    logs = t.clamp_min(1e-12).log()
    pos = torch.arange(n, device=t.device).float() / max(1, n-1)
    last = t[-1]
    chunkmean = F.avg_pool1d(F.pad(last[None, None], (0, (-n) % 32)), 32, stride=32).flatten()
    local = chunkmean[torch.arange(n, device=t.device) // 32]
    entropy = -(last * last.clamp_min(1e-12).log()).sum() / max(1, torch.log(torch.tensor(float(n), device=t.device)))
    q = query if query is not None else torch.zeros(1536, device=t.device)
    qsummary = torch.stack((q.float().norm() / q.numel()**0.5, q.float().mean(), q.float().std()))
    columns = [last.clamp_min(1e-12).log(), logs.mean(0), logs.std(0),
               *grouped.clamp_min(1e-12).log().unbind(), local.clamp_min(1e-12).log(),
               pos, pos.square(), torch.full_like(pos, float(entropy)),
               torch.full_like(pos, qsummary[0]), torch.full_like(pos, qsummary[1]), torch.full_like(pos, qsummary[2])]
    attention_features = torch.stack(columns, -1).float()
    # A small frozen query-content feature, broadcast across candidate tokens.
    return torch.cat((attention_features, q[:16].float()[None].expand(n, -1)), -1)


class UtilitySelector(torch.nn.Module):
    def __init__(self, dimensions=13):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dimensions))
        self.register_buffer('scale', torch.ones(dimensions))
        self.net = torch.nn.Sequential(torch.nn.Linear(dimensions, 32), torch.nn.SiLU(), torch.nn.Linear(32, 1))

    def forward(self, features):
        return self.net((features-self.mean)/self.scale.clamp_min(1e-4)).squeeze(-1)
