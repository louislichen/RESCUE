"""Coverage/shift features and small disjoint linear UCB budget models.

Independent of the decoding engine. Times are seconds; reward is tokens.
"""
import math
import numpy as np

BUDGETS = (32, 64, 96)


def features(attention, readable_positions, anchor, chunk_size=32):
    """Use token mass for coverage, absolute chunk mass for anchored JS.

    Return None until a real selection anchor exists. Padding adds no mass.
    """
    p = np.asarray(attention, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Invalid attention mass")
    if anchor is None or p.sum() <= 0:
        return None
    p = p / p.sum()
    ids = np.unique(np.asarray(readable_positions, dtype=np.int64))
    ids = ids[(ids >= 0) & (ids < len(p))]
    coverage = float(p[ids].sum())
    def chunks(a):
        a = np.asarray(a, dtype=np.float64)
        if a.ndim != 1 or not np.isfinite(a).all() or (a < 0).any() or a.sum() <= 0:
            raise ValueError("Invalid anchor")
        a = a / a.sum()
        return np.pad(a, (0, (-len(a)) % chunk_size)).reshape(-1, chunk_size).sum(1)
    q, r = chunks(p), chunks(anchor)
    n = max(len(q), len(r))
    q, r = np.pad(q, (0, n-len(q))), np.pad(r, (0, n-len(r)))
    m = (q+r)/2
    def kl(a):
        valid = a > 0
        return float(np.sum(a[valid]*np.log(a[valid]/m[valid])))
    shift = min(1., max(0., (kl(q)+kl(r))/(2*math.log(2))))
    return np.array([1., coverage, shift, coverage*shift])


class LinUCB:
    def __init__(self, width=4, alpha=0.5, ridge=1., min_exploration=1,
                 epsilon=0., seed=42, budgets=BUDGETS, fallback=64):
        if min_exploration < 1 or not 0 <= epsilon <= 1:
            raise ValueError('Invalid exploration configuration')
        self.alpha = alpha
        self.min_exploration = min_exploration
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.budgets = tuple(budgets)
        if not self.budgets or len(set(self.budgets)) != len(self.budgets) or any(type(b) is not int or b < 1 for b in self.budgets):
            raise ValueError('Budgets must be unique positive integers')
        if fallback not in self.budgets:
            raise ValueError('Fallback must be in budgets')
        self.fallback = fallback
        self.a = np.repeat((np.eye(width)*ridge)[None], len(self.budgets), axis=0)
        self.y = np.zeros((len(self.budgets), width))
        self.counts = np.zeros(len(self.budgets), dtype=np.int64)

    def choose(self, x, explore=False):
        if x is None:
            return self.fallback
        if explore and (self.counts < self.min_exploration).any():
            tied = np.flatnonzero(self.counts == self.counts.min())
            return self.budgets[int(self.rng.choice(tied))]
        if explore and self.rng.random() < self.epsilon:
            return self.budgets[int(self.rng.integers(len(self.budgets)))]
        values = []
        for a, y in zip(self.a, self.y):
            invx = np.linalg.solve(a, x)
            values.append(float(invx @ y) + (self.alpha*math.sqrt(max(0., x @ invx)) if explore else 0.))
        return self.budgets[int(np.argmax(values))]

    def update(self, x, budget, committed, seconds, rho):
        if x is None:
            return
        if seconds <= 0 or not np.isfinite(seconds):
            raise ValueError("Round time must be positive seconds")
        j = self.budgets.index(budget)
        self.a[j] += np.outer(x, x)
        self.y[j] += x*(committed-rho*seconds)
        self.counts[j] += 1
