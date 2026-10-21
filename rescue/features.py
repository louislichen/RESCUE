"""Split frozen 31-feature contract into R-independent and R-dependent work."""
import math
import torch

from .attention_summary import mean_attention


def entropy(p):
    return -(p*p.clamp_min(1e-12).log()).sum(-1)


def prepare(t, s, progress, remaining, recent):
    n = t.n
    ta, sa = mean_attention(t), mean_attention(s)
    end = max(2, n-recent)
    tr, sr = ta[1:end], sa[1:end]
    tr, sr = tr/tr.sum().clamp_min(1e-12), sr/sr.sum().clamp_min(1e-12)
    mix = (tr+sr)/2
    js = .5*((tr*(tr.clamp_min(1e-12)/mix.clamp_min(1e-12)).log()).sum()+
              (sr*(sr.clamp_min(1e-12)/mix.clamp_min(1e-12)).log()).sum())
    et, es = entropy(t.a), entropy(s.a)
    vector = ta.new_zeros(31)
    vector[0], vector[1], vector[4] = math.log(n)/10, progress/64, remaining/64
    vector[7:14] = torch.stack((js, entropy(tr)/max(1., math.log(end-1)),
        entropy(sr)/max(1., math.log(end-1)), F_cos(tr, sr), ta[end:].sum(), sa[end:].sum(), sa.sum()))
    vector[22:28] = torch.stack((et.mean()/max(1., math.log(n)), es.mean()/max(1., math.log(n)),
                               et.std(unbiased=False), es.std(unbiased=False),
                               tr.topk(min(16, len(tr))).values.sum(), sr.topk(min(16, len(sr))).values.sum()))
    return vector, tr, sr, end


def F_cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0)


def finish(prepared, t, s, selected, include_read=True):
    vector, tr, sr, end = prepared
    vector = vector.clone()
    remote = selected[(selected > 0) & (selected < end)]
    tc, sc = t.a[:, selected].sum(-1), s.a[:, selected].sum(-1)
    vector[2] = len(selected)/t.n
    vector[5], vector[6] = tr[remote-1].sum(), sr[remote-1].sum()
    if include_read:
        vector[14],vector[15]=torch.log1p(t.error(selected)),torch.log1p(s.error(selected))
    vector[16:22] = torch.stack(((selected.float()/t.n).mean(), (selected.float()/t.n).std(unbiased=False),
        tc.mean(), tc.std(unbiased=False), sc.mean(), sc.std(unbiased=False)))
    return vector
