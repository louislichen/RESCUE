"""Lean native SD and a clearly scoped Qwen2 SpecExtend retrieval control.

The latter ports chunk retrieval, not the upstream tree/Flash/hybrid engine.
Both share RESCUE's preallocated native storage and Draft attention backend.
Native SD computes no attention probes, read states, selectors or mapper.
"""
import threading
import time
import uuid

import torch

from .attention import DraftAttention, TargetObservation
from .cache import NativeCache, WorkingCache
from .engine import RescueEngine, exclusive


class ChunkRetrieval:
    """Upstream-style recent initialization, chunk means, and newest anchor.

    A scheduled verification's last committed query supplies the next round's
    retrieval. Between retrievals only chunk endpoints and the frontier grow.
    Strict chunk-budget eviction follows the local archived implementation.
    """
    def __init__(self, top_chunks, chunk_size=32):
        if top_chunks < 1 or chunk_size < 1:
            raise ValueError('Positive chunk count and size required')
        self.top_chunks, self.chunk_size = top_chunks, chunk_size
        self.selected = None
        self.refreshes = 0

    def select(self, n, device, attention=None):
        chunks = (n+self.chunk_size-1)//self.chunk_size
        count = min(chunks, self.top_chunks)
        if self.selected is None:
            self.selected = list(range(chunks-count, chunks))
        elif attention is not None:
            scores = attention.float().mean(0)
            if scores.shape != (n,):
                raise ValueError('Retrieval attention must match the committed prefix')
            starts = torch.arange(chunks, device=device)*self.chunk_size
            ends = (starts+self.chunk_size).clamp_max(n)
            cumulative = torch.cat((scores.new_zeros(1), scores.cumsum(0)))
            means = (cumulative[ends]-cumulative[starts])/(ends-starts)
            self.selected = sorted(means[:-1].topk(count-1).indices.tolist()+[chunks-1])
            self.refreshes += 1
        # Grow into a newly committed frontier; evict oldest retained chunk.
        if self.selected[-1] != chunks-1:
            self.selected.append(chunks-1)
            self.selected = self.selected[-count:]
        return torch.cat([torch.arange(i*self.chunk_size, min((i+1)*self.chunk_size, n),
                                      device=device) for i in self.selected])


class BaselineEngine(RescueEngine):
    def __init__(self, teacher, student, config, vocab_size=None, top_chunks=None):
        if config.length_policy != 'fixed':
            raise ValueError('Baseline uses a fixed draft length')
        if teacher.config.model_type != 'qwen2' or student.config.model_type != 'qwen2':
            raise ValueError('Qwen2 adapter required')
        self.teacher, self.student, self.config = teacher, student, config
        self.device = teacher.model.embed_tokens.weight.device
        if student.model.embed_tokens.weight.device != self.device:
            raise ValueError('Models must use the same device')
        self.vocab = vocab_size or min(teacher.config.vocab_size, student.config.vocab_size)
        self.top_chunks = top_chunks
        if top_chunks is not None and (type(top_chunks) is not int or top_chunks < 1):
            raise ValueError('top_chunks must be positive or None')
        self.observation = TargetObservation(teacher)
        if top_chunks is not None:
            self.observation.install()
        teacher.eval().requires_grad_(False)
        student.eval().requires_grad_(False)
        self.native_t = self.native_s = self.working = None
        self.closed = False
        self._request_lock = threading.Lock()

    def _buffers(self, capacity):
        capacity = 1 << (capacity-1).bit_length()
        def allocate(model):
            c = model.config
            return NativeCache(c.num_hidden_layers, c.num_key_value_heads,
                               c.hidden_size//c.num_attention_heads, capacity,
                               model.model.embed_tokens.weight.dtype, self.device)
        self.native_t, self.native_s = allocate(self.teacher), allocate(self.student)
        if self.top_chunks is not None:
            self.working = WorkingCache(self.native_s, self.top_chunks*32, self.config.max_draft)

    @torch.no_grad()
    @exclusive
    def generate(self, input_ids, eos_token_id=None):
        if self.closed:
            raise RuntimeError('Engine has been closed')
        if len(input_ids) < 3:
            raise ValueError('At least three prompt tokens required')
        c = self.config
        self.synchronize()
        start = time.perf_counter()
        self._buffers(len(input_ids)+c.max_new_tokens+c.max_draft+1)
        adapter = DraftAttention(self.student, self.working, None, c.attention_backend,
                                 observe=False).install()
        selector = ChunkRetrieval(self.top_chunks) if self.top_chunks is not None else None
        generated, trace = [], []
        attention = None
        proposed = accepted = 0
        try:
            self.observation.mode = None
            self._prefill(self.teacher, input_ids[:-1], self.native_t)
            self._prefill(self.student, input_ids[:-1], self.native_s)
            adapter.prefill = False
            self.synchronize()
            prefill_end = time.perf_counter()
            current = input_ids[-1]
            while len(generated) < c.max_new_tokens:
                n, progress = self.native_t.get_seq_length(), len(generated)
                if self.native_s.get_seq_length() != n:
                    raise RuntimeError('Committed native prefixes diverged')
                workset_tokens = n
                if selector is not None:
                    positions = selector.select(n, self.device, attention)
                    self.working.bind(self.native_s, positions)
                    workset_tokens = len(positions)
                    attention = None
                cap = min(c.fixed_k, c.max_new_tokens-progress)
                draft, token = [], current
                for _ in range(cap):
                    token = int(adapter.step(token, self.native_s)[0, :self.vocab].argmax())
                    draft.append(token)
                    if token == eos_token_id:
                        break
                capture = selector is not None and len(trace) % c.refresh_every == 0
                self.observation.mode = 'all' if capture else None
                out = self._forward(self.teacher, [current]+draft, self.native_t)
                checks = self.teacher.lm_head(out.last_hidden_state)[0, :, :self.vocab].float().argmax(-1).tolist()
                count = 0
                for index, token in enumerate(draft):
                    if token != checks[index]:
                        break
                    count += 1
                    if token == eos_token_id:
                        break
                proposed += len(draft)
                accepted += count
                generated.extend(draft[:count])
                stop = len(generated) >= c.max_new_tokens or (bool(generated) and generated[-1] == eos_token_id)
                if not stop:
                    current = checks[count]
                    generated.append(current)
                    stop = len(generated) >= c.max_new_tokens or current == eos_token_id
                if not stop:
                    committed = n+count+1
                    if count == len(draft):
                        adapter.step(draft[-1], self.native_s)
                    if capture:
                        attention = self.observation.rows[:, count, :committed].clone()
                    self.native_t.crop(committed)
                    self.native_s.crop(committed)
                trace.append(dict(round_index=len(trace), prefix=n, cap=cap, proposed=len(draft),
                                  accepted=count, terminal=stop, workset_tokens=workset_tokens))
                self.observation.rows = None
                if self.working is not None:
                    self.working.invalidate()
                if stop:
                    break
            self.synchronize()
            decode_end = time.perf_counter()
        finally:
            adapter.restore()
            self.observation.mode = None
            self.observation.rows = None
            if self.working is not None:
                self.working.invalidate()
        end = time.perf_counter()
        return dict(request_id=uuid.uuid4().hex, generated=generated, tokens=len(generated),
                    proposed=proposed, accepted=accepted, acceptance_rate=accepted/max(1, proposed),
                    blocks=len(trace), trace=trace, prefill_seconds=prefill_end-start,
                    decode_seconds=decode_end-prefill_end, cleanup_seconds=end-decode_end,
                    total_seconds=end-start, decode_tps=len(generated)/(decode_end-prefill_end),
                    total_tps=len(generated)/(end-start), draft_forwards=adapter.steps,
                    working_gather_layers=self.working.gathered_layers if self.working is not None else 0,
                    selector_refreshes=selector.refreshes if selector is not None else 0,
                    target_probe_rounds=sum(i % c.refresh_every == 0 for i in range(len(trace))) if selector else 0,
                    draft_shadow_probes=0, projected_positions=0, rpc_calls=0)

    @exclusive
    def close(self):
        if not self.closed:
            self.observation.restore()
            self.closed = True
