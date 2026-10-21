"""Opt-in replay of the unchanged greedy coverage inner loop.

Coarse selection, support, content normalization and actual pool construction
remain outside this primitive. No padding or score perturbation is introduced.
An attention bonus, when present, must already equal the *complete weighted*
term from the original expression, preserving its multiplication/division order.
"""
from dataclasses import dataclass
import math
import time

import torch


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _metadata(tensor):
    return (tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype), str(tensor.device))


def _validate(similarities, coverage, objective, pool, slots, batch, student_weight, attention_bonus):
    _require(len(similarities) == len(coverage) == len(objective) == 2,
             'Coverage primitive requires exactly Target and Draft inputs')
    _require(isinstance(pool, torch.Tensor) and pool.ndim == 1 and pool.dtype == torch.long,
             'Pool must be a one-dimensional integer position tensor')
    _require(type(slots) is int and 0 < slots <= pool.numel(), 'Invalid actual pool/slot count')
    _require(type(batch) is int and batch > 0, 'Coverage batch must be a positive integer')
    _require(type(student_weight) in (int, float) and math.isfinite(student_weight),
             'Student weight must be a finite Python scalar')
    tensors = [pool]
    for sim, initial, weight in zip(similarities, coverage, objective):
        _require(all(isinstance(t, torch.Tensor) for t in (sim, initial, weight)),
                 'Coverage inputs must be tensors')
        _require(sim.ndim == 2 and sim.shape[0] > 0 and sim.shape[1] == pool.numel(),
                 'Similarity shape must match the actual nonempty support and pool')
        _require(initial.ndim == weight.ndim == 1 and initial.shape[0] == weight.shape[0] == sim.shape[0],
                 'Initial coverage/objective shapes must match their support')
        _require(all(t.is_floating_point() for t in (sim, initial, weight)),
                 'Similarity, coverage and objective inputs must be floating point')
        tensors.extend((sim, initial, weight))
    if attention_bonus is not None:
        _require(isinstance(attention_bonus, torch.Tensor) and attention_bonus.is_floating_point()
                 and attention_bonus.shape == pool.shape, 'Weighted attention bonus must match the actual pool')
        tensors.append(attention_bonus)
    _require(all(t.device == pool.device for t in tensors), 'Coverage tensors must share one device')
    # The live selector produces contiguous matmul/reduction/gather results.
    # Refuse other layouts rather than silently change reduction traversal.
    _require(all(t.is_contiguous() for t in tensors), 'Coverage primitive requires contiguous actual-pool inputs')
    return tensors


def _greedy(similarities, coverage, objective, pool, slots, batch, student_weight, attention_bonus,
            graph_safe=False):
    """Literal operator order of ContextSelector._hierarchical's inner loop."""
    current = list(coverage)
    chosen, used = [], torch.zeros(len(pool), device=pool.device, dtype=torch.bool)
    for offset in range(0, slots, batch):
        gains = sum(scale*((sim-cover[:, None]).clamp_min(0)*weight[:, None]).sum(0)
                    for scale, sim, cover, weight in
                    zip((1., student_weight), similarities, current, objective))
        if attention_bonus is not None:
            gains = gains+attention_bonus
        gains.masked_fill_(used, -torch.inf)
        take = gains.topk(min(batch, slots-offset)).indices
        if graph_safe:
            # Tensor.__setitem__(take, True) constructs/transfers a CPU scalar
            # on CUDA. index_fill_ passes the identical Boolean as a kernel
            # scalar and does not introduce a forbidden capture-time H2D copy.
            used.index_fill_(0, take, True)
        else:
            used[take] = True
        for index, sim in enumerate(similarities):
            current[index] = torch.maximum(current[index], sim[:, take].amax(-1))
        chosen.append(pool[take])
    return torch.cat(chosen)


@torch.no_grad()
def greedy_coverage(similarities, coverage, objective, pool, slots, batch,
                    student_weight=1., attention_bonus=None):
    """Eager reference: return chosen absolute positions in batch-pick order.

    Inputs are read-only. ``attention_bonus`` is the already weighted additive
    term, not a normalized score that should be multiplied a second time.
    The caller retains the original fixed-position concatenation and sorting.
    """
    _validate(similarities, coverage, objective, pool, slots, batch, student_weight, attention_bonus)
    return _greedy(similarities, coverage, objective, pool, slots, batch,
                   student_weight, attention_bonus)


@dataclass
class _Entry:
    similarities: tuple
    coverage: tuple
    objective: tuple
    pool: object
    bonus: object
    graph: object = None
    output: object = None


class CoverageGraphRuntime:
    """Private, persistent shape-keyed coverage replay; no silent fallback."""
    MAX_ENTRIES = 64
    WARMUP_ITERATIONS = 3

    def __init__(self, mode='cuda_graph', max_entries=64):
        _require(mode in ('eager', 'cuda_graph'), 'Unknown coverage execution mode')
        _require(type(max_entries) is int and 0 < max_entries <= self.MAX_ENTRIES,
                 'Coverage graph limit must be between one and 64 entries')
        self.mode, self.max_entries = mode, max_entries
        self.device, self._stream, self._entries = None, None, {}
        self.closed = False
        self._counts = dict(captures=0, replays=0, capture_host_seconds=0.,
                            eager_calls=0, input_copies=0)

    def _entry(self, similarities, coverage, objective, pool, slots, batch, student_weight, bonus):
        key = (tuple(_metadata(t) for t in similarities), tuple(_metadata(t) for t in coverage),
               tuple(_metadata(t) for t in objective), _metadata(pool), slots, batch,
               student_weight, None if bonus is None else _metadata(bonus))
        if key not in self._entries:
            _require(len(self._entries) < self.max_entries,
                     'Coverage graph entry limit reached; no eviction or silent fallback')
            self._entries[key] = _Entry(tuple(torch.empty_like(t) for t in similarities),
                tuple(torch.empty_like(t) for t in coverage), tuple(torch.empty_like(t) for t in objective),
                torch.empty_like(pool), None if bonus is None else torch.empty_like(bonus))
        entry = self._entries[key]
        for destination, source in zip((*entry.similarities, *entry.coverage, *entry.objective, entry.pool),
                                       (*similarities, *coverage, *objective, pool)):
            destination.copy_(source)
        if bonus is not None:
            entry.bonus.copy_(bonus)
        self._counts['input_copies'] += 1
        return entry

    def _body(self, entry, slots, batch, student_weight):
        return _greedy(entry.similarities, entry.coverage, entry.objective,
                       entry.pool, slots, batch, student_weight, entry.bonus,
                       graph_safe=self.mode == 'cuda_graph')

    def _capture(self, entry, slots, batch, student_weight):
        start = time.perf_counter()
        current, stream = torch.cuda.current_stream(self.device), self._stream
        stream.wait_stream(current)
        try:
            with torch.cuda.stream(stream):
                for _ in range(self.WARMUP_ITERATIONS):
                    self._body(entry, slots, batch, student_weight)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            # No shared pool across entries/owners or branch-order assumption.
            with torch.cuda.graph(graph, stream=stream):
                output = self._body(entry, slots, batch, student_weight)
            entry.graph, entry.output = graph, output
        finally:
            current.wait_stream(stream)
            stream.synchronize()
            self._counts['capture_host_seconds'] += time.perf_counter()-start
        self._counts['captures'] += 1

    @torch.no_grad()
    def select(self, similarities, coverage, objective, pool, slots, batch,
               student_weight=1., attention_bonus=None):
        _require(not self.closed, 'Coverage runtime has been closed')
        _validate(similarities, coverage, objective, pool, slots, batch, student_weight, attention_bonus)
        if self.mode == 'cuda_graph':
            _require(pool.device.type == 'cuda', 'CUDA graph coverage requires CUDA inputs')
        if self.device is None:
            self.device = pool.device
            if self.mode == 'cuda_graph':
                self._stream = torch.cuda.Stream(device=self.device)
        _require(pool.device == self.device, 'A coverage owner cannot change device')
        entry = self._entry(similarities, coverage, objective, pool, slots, batch,
                            student_weight, attention_bonus)
        if self.mode == 'cuda_graph':
            if entry.graph is None:
                self._capture(entry, slots, batch, student_weight)
            entry.graph.replay()
            self._counts['replays'] += 1
        else:
            entry.output = self._body(entry, slots, batch, student_weight)
            self._counts['eager_calls'] += 1
        # The caller may retain previous selections across subsequent refreshes.
        return entry.output.detach().clone()

    def stats(self):
        return dict(mode=self.mode, entries=len(self._entries), **self._counts)

    def close(self):
        if not self.closed:
            try:
                if self.device is not None and self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
            finally:
                self._entries.clear()
                self._stream = None
                self.closed = True
