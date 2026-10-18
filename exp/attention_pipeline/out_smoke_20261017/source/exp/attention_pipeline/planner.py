"""Versioned length messages and a single bounded asynchronous planning lane."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import time


@dataclass(frozen=True)
class PlanKey:
    request: str
    round_index: int
    committed_prefix: int
    workset_version: int


@dataclass(frozen=True)
class LengthDecision:
    key: PlanKey
    cap: int
    score: float
    wall_seconds: float


class LengthTicket:
    def __init__(self, key, remaining, future):
        self.key, self.remaining, self.future = key, remaining, future

    def result(self, expected_key):
        if expected_key != self.key:
            raise ValueError('Length message belongs to another request/round/workset')
        decision = self.future.result()
        if (decision.key != expected_key or type(decision.cap) is not int
                or not 1 <= decision.cap <= self.remaining
                or not math.isfinite(decision.score)
                or not math.isfinite(decision.wall_seconds) or decision.wall_seconds < 0):
            raise ValueError('Invalid or stale length decision')
        return decision


class LengthPlanner:
    """No speculative second token is allowed before resolving its ticket.

    A future can later be backed by an RPC; this executor is local only. The
    first Draft token is safe to compute because every supported cap is >=1.
    """
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rescue-length')
        self.pending = None

    def submit(self, key, remaining, compute):
        if type(remaining) is not int or remaining < 1:
            raise ValueError('A length plan requires a positive output budget')
        if self.pending is not None and not self.pending.future.done():
            raise RuntimeError('Previous round has not finished planning')

        def run():
            start = time.perf_counter()
            cap, score = compute()
            return LengthDecision(key, cap, score, time.perf_counter()-start)

        self.pending = LengthTicket(key, remaining, self.executor.submit(run))
        return self.pending

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
