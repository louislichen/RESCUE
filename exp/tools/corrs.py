import math
from typing import List, Optional


def pearson_corr(a: List[float], b: List[float]) -> Optional[float]:
    n = min(len(a), len(b))
    if n <= 1:
        return None
    a = a[:n]
    b = b[:n]
    ma = sum(a) / n
    mb = sum(b) / n
    cov = 0.0
    va = 0.0
    vb = 0.0
    for x, y in zip(a, b):
        dx = x - ma
        dy = y - mb
        cov += dx * dy
        va += dx * dx
        vb += dy * dy
    denom = math.sqrt(va * vb)
    if denom == 0.0:
        return None
    return cov / denom


def rankdata_average_ties(x: List[float]) -> List[float]:
    n = len(x)
    order = sorted(range(n), key=lambda i: x[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(a: List[float], b: List[float]) -> Optional[float]:
    n = min(len(a), len(b))
    if n <= 1:
        return None
    a = a[:n]
    b = b[:n]
    ra = rankdata_average_ties(a)
    rb = rankdata_average_ties(b)
    return pearson_corr(ra, rb)


def _normalize_prob(x: List[float], eps: float = 1e-12) -> Optional[List[float]]:
    if not x:
        return None
    y = [0.0 if (v is None or not math.isfinite(float(v)) or float(v) < 0.0) else float(v) for v in x]
    s = sum(y)
    if s <= 0.0:
        return None
    y = [v / s for v in y]
    y = [max(eps, v) for v in y]
    s2 = sum(y)
    if s2 <= 0.0:
        return None
    return [v / s2 for v in y]


def js_divergence(p: List[float], q: List[float], eps: float = 1e-12, log_base: float = 2.0) -> Optional[float]:
    n = min(len(p), len(q))
    if n <= 1:
        return None
    p = _normalize_prob(p[:n], eps=eps)
    q = _normalize_prob(q[:n], eps=eps)
    if p is None or q is None:
        return None

    m = [(pi + qi) * 0.5 for pi, qi in zip(p, q)]

    def kl(a, b):
        out = 0.0
        for ai, bi in zip(a, b):
            out += ai * (math.log(ai / bi) / math.log(log_base))
        return out

    jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    if not math.isfinite(jsd):
        return None
    return max(0.0, float(jsd))