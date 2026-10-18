"""Per-head native read preservation with bounded vectorized local swaps."""
import torch


def mandatory(n, recent=64, device='cuda:0'):
    return torch.unique(torch.cat((torch.zeros(1, dtype=torch.long, device=device),
        torch.arange(max(0, n-recent), n, device=device))), sorted=True)


class ReadState:
    def __init__(self, rows, values):
        # rows: query heads x positions; values: KV heads x positions x dimension.
        self.a = rows.float()
        self.a = self.a / self.a.sum(-1, keepdim=True).clamp_min(1e-12)
        self.v = values.float()
        self.heads, self.n = self.a.shape
        self.groups = self.heads // self.v.shape[0]
        self.vheads = torch.arange(self.heads, device=rows.device) // self.groups
        self.target = torch.einsum('hgn,hnd->hgd', self.a.reshape(self.v.shape[0], self.groups, self.n), self.v).flatten(0, 1)
        self.scale = self.target.square().sum(-1).clamp_min(1e-6)

    def contribution(self, positions):
        vv = self.v[self.vheads[:, None], positions[None]].permute(1, 0, 2)
        return self.a[:, positions].T[:, :, None] * vv

    def sums(self, positions):
        w = self.a[:, positions].sum(-1)
        num = torch.einsum('hgn,hnd->hgd', self.a[:, positions].reshape(self.v.shape[0], self.groups, -1),
                           self.v[:, positions]).flatten(0, 1)
        return w, num

    def error(self, positions):
        w, num = self.sums(positions)
        return (((num/w[:, None].clamp_min(1e-12)-self.target).square().sum(-1))/self.scale).mean()

    def swap_errors(self, selected, remove, add):
        w, num = self.sums(selected)
        numerators = num[None, None] - self.contribution(remove)[:, None] + self.contribution(add)[None]
        weights = w[None, None] - self.a[:, remove].T[:, None] + self.a[:, add].T[None]
        delta = numerators/weights[..., None].clamp_min(1e-12)-self.target
        return (delta.square().sum(-1)/self.scale).mean(-1)


def select(rule, teacher, student, budget=512, recent=64, previous=None, swaps=4):
    n, device = teacher.n, teacher.a.device
    fixed = mandatory(n, recent, device)
    budget = min(n, budget)
    slots = budget-len(fixed)
    available = torch.arange(n, device=device)
    free = available[~torch.isin(available, fixed)]
    t, s = teacher.a.mean(0), student.a.mean(0)
    tr = free[torch.argsort(t[free], descending=True, stable=True)]
    sr = free[torch.argsort(s[free], descending=True, stable=True)]
    if rule == 'student':
        chosen = sr[:slots]
    elif rule == 'union':
        nt = round(slots*.75)
        order = torch.cat((tr[:nt], sr[:slots-nt], tr)).tolist()
        chosen = torch.tensor(list(dict.fromkeys(order))[:slots], device=device)
    else:
        chosen = tr[:slots]
    selected = torch.unique(torch.cat((fixed, chosen)), sorted=True)
    assert len(selected) == budget
    if not rule.startswith('read_'):
        return selected
    beta = {'read_teacher': 0.0, 'read_joint025': .25, 'read_joint1': 1.0}[rule]
    if previous is not None:
        retained = previous[(previous < n) & ~torch.isin(previous, fixed)]
        retained = retained[torch.argsort(t[retained], descending=True, stable=True)]
        order = torch.cat((retained[:slots], tr)).tolist()
        chosen = torch.tensor(list(dict.fromkeys(order))[:slots], device=device)
        selected = torch.unique(torch.cat((fixed, chosen)), sorted=True)
    for _ in range(swaps):
        movable = selected[~torch.isin(selected, fixed)]
        remove = movable[torch.argsort(t[movable]+beta*s[movable], stable=True)[:16]]
        outside = available[~torch.isin(available, selected)]
        if not len(outside) or not len(remove):
            break
        ti = outside[torch.argsort(t[outside], descending=True, stable=True)[:32]]
        si = outside[torch.argsort(s[outside], descending=True, stable=True)[:32]]
        add = torch.unique(torch.cat((ti, si)), sorted=True)
        errors = teacher.swap_errors(selected, remove, add)
        if beta:
            errors = errors + beta*student.swap_errors(selected, remove, add)
        base = teacher.error(selected) + beta*student.error(selected)
        index = errors.argmin()
        if float(errors.flatten()[index]) >= float(base)-1e-8:
            break
        ri = int(index)//len(add)
        ai = int(index)%len(add)
        selected = torch.cat((selected[selected != remove[ri]], add[ai:ai+1])).sort().values
    assert len(selected) == budget and bool(torch.isin(fixed, selected).all())
    return selected


def entropy(p):
    return -(p*p.clamp_min(1e-12).log()).sum(-1)


def feature_vector(teacher, student, selection, progress, age, confidence, remaining, recent=64):
    # Teacher reference covers only old keys. The student row may include new
    # draft keys; compare only the actual shared prefix, tracking new-key mass.
    n = teacher.n
    ta, sa_all = teacher.a.mean(0), student.a.mean(0)
    sa = sa_all[:n]
    end = max(2, n-recent)
    tr, sr = ta[1:end], sa[1:end]
    tr, sr = tr/tr.sum().clamp_min(1e-12), sr/sr.sum().clamp_min(1e-12)
    mixture = (tr+sr)/2
    js = .5*((tr*(tr.clamp_min(1e-12)/mixture.clamp_min(1e-12)).log()).sum()+
             (sr*(sr.clamp_min(1e-12)/mixture.clamp_min(1e-12)).log()).sum())
    remote = selection[(selection > 0) & (selection < end)]
    selected_s = torch.cat((selection, torch.arange(n, student.n, device=selection.device)))
    layer_entropy_t = entropy(teacher.a)
    layer_entropy_s = entropy(student.a)
    old_mass = sa.sum()
    tcov = teacher.a[:, selection].sum(-1)
    scov = student.a[:, selected_s].sum(-1)
    columns = [torch.log(torch.tensor(float(n), device=ta.device))/10,
               ta.new_tensor(progress/64), ta.new_tensor(len(selection)/n),
               ta.new_tensor(age/12), ta.new_tensor(remaining/64),
               tr[remote-1].sum(), sr[remote-1].sum(), js,
               entropy(tr)/max(1., __import__('math').log(end-1)),
               entropy(sr)/max(1., __import__('math').log(end-1)),
               torch.nn.functional.cosine_similarity(tr, sr, dim=0),
               ta[end:].sum(), sa[end:].sum(), old_mass,
               torch.log1p(teacher.error(selection)), torch.log1p(student.error(selected_s)),
               (selection.float()/n).mean(), (selection.float()/n).std(unbiased=False),
               tcov.mean(), tcov.std(unbiased=False), scov.mean(), scov.std(unbiased=False),
               layer_entropy_t.mean()/max(1., __import__('math').log(n)),
               layer_entropy_s.mean()/max(1., __import__('math').log(student.n)),
               layer_entropy_t.std(unbiased=False), layer_entropy_s.std(unbiased=False),
               tr.topk(min(16, len(tr))).values.sum(), sr.topk(min(16, len(sr))).values.sum()]
    return torch.cat((torch.stack(columns).float(), confidence.float()))


FEATURE_GROUPS = dict(structural=list(range(5)), confidence=list(range(5))+[28,29,30],
                      attention=[i for i in range(28) if i not in (14,15)],
                      attention_read=list(range(28)), combined=list(range(31)))
