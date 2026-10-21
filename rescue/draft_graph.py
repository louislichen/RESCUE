"""Optional, single-token resident Qwen execution with lazy CUDA graphs.

Only tensor computation is graphed. Native-cache writes, Python cache lengths,
and every token/EOS/length decision remain outside the graph. A runtime owns
one stable pair of Native/Working buffers; replacing them requires a new owner.
"""
import copy
from dataclasses import dataclass
import time

import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _signature(value):
    if isinstance(value, torch.Tensor):
        return ('tensor', tuple(value.shape), tuple(value.stride()), str(value.dtype), str(value.device))
    if isinstance(value, (tuple, list)):
        return tuple(_signature(v) for v in value)
    if value is None or type(value) in (int, bool, str, float):
        return ('constant', type(value).__name__, value)
    raise ValueError('Unsupported graph binding value')


def _clone_binding(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_binding(v) for v in value)
    if isinstance(value, list):
        return [_clone_binding(v) for v in value]
    return value


def _copy_binding(destination, source):
    _require(_signature(destination) == _signature(source), 'Graph binding layout changed')
    if isinstance(source, torch.Tensor):
        destination.copy_(source)
    elif isinstance(source, (tuple, list)):
        for left, right in zip(destination, source):
            _copy_binding(left, right)


@dataclass
class _Entry:
    key: tuple
    injector: object
    graph: object = None
    outputs: object = None


class ResidentDraftRuntime:
    """Replay exactly one Draft forward, never a proposal/decision loop.

    ``scores`` is a borrowed output valid until that graph is replayed again.
    Returned query and summary tensors have their own storage, because rollback
    may need them after several subsequent Draft forwards.
    """
    MAX_ENTRIES = 32
    WARMUP_ITERATIONS = 3

    @torch.no_grad()
    def __init__(self, model, native, working, injector, mode='resident', *,
                 max_entries=None, retain_query=True):
        _require(mode in ('resident', 'cuda_graph'), 'Unknown Draft runtime mode')
        limit = self.MAX_ENTRIES if max_entries is None else max_entries
        _require(type(limit) is int and limit > 0, 'Positive explicit graph entry limit required')
        _require(type(retain_query) is bool, 'retain_query must be boolean')
        _require(model.config.model_type == 'qwen2', 'Resident adapter requires Qwen2')
        _require(working is not None and working.mode == 'packed', 'Resident adapter requires packed Working KV')
        _require(injector is not None, 'Resident adapter requires an explicit injector')
        _require(not getattr(injector.network, 'attention_features', 0), 'Q/K-conditioned mapper is not graph-supported')
        self.model, self.native, self.working, self.injector = model, native, working, injector
        self.mode = mode
        self.max_entries, self.retain_query = limit, retain_query
        self.device = model.model.embed_tokens.weight.device
        _require(mode != 'cuda_graph' or self.device.type == 'cuda', 'CUDA graph mode requires a CUDA model')
        _require(native.key_storage.device == self.device and working.key_storage.device == self.device,
                 'Model and both caches must share a device')
        _require(len(native) == len(model.model.layers) == len(working.keys), 'Layer count mismatch')
        self._owners = (native.key_storage.data_ptr(), native.value_storage.data_ptr(),
                        working.key_storage.data_ptr(), working.value_storage.data_ptr())
        self._token = torch.empty((1, 1), device=self.device, dtype=torch.long)
        self._position = torch.empty((1, 1), device=self.device, dtype=torch.long)
        self._rope = []
        dtype = model.model.embed_tokens.weight.dtype
        for layer in model.model.layers:
            attention = layer.self_attn
            rotary = attention.rotary_emb
            _require(native.capacity <= getattr(rotary, 'max_seq_len_cached', 0),
                     'Preallocated RoPE cache is too short; do not silently change RoPE')
            dummy = torch.empty((1, attention.num_key_value_heads, 1, attention.head_dim),
                                device=self.device, dtype=dtype)
            # Existing cached values and original dtype conversion, not a new
            # frequency formula. Keep these tensor addresses alive for graphs.
            self._rope.append(rotary(dummy, seq_len=native.capacity))
        self._entries = {}
        self._capture_stream = torch.cuda.Stream(device=self.device) if mode == 'cuda_graph' else None
        self._counts = dict(captures=0, replays=0, resident_steps=0,
                            capture_host_seconds=0., shadow_updates=0)
        self.closed = False

    def _bindings(self):
        return (tuple(sorted((layer, _signature(value)) for layer, value in self.injector.active.items())),
                tuple(sorted((layer, _signature(value)) for layer, value in self.injector.gate_static.items())),
                tuple(sorted((layer, _signature(value)) for layer, value in self.injector.residual_static.items())))

    def _entry(self, key):
        if key not in self._entries:
            _require(len(self._entries) < self.max_entries, 'Draft graph entry limit reached; no silent fallback')
            shadow = copy.copy(self.injector)
            shadow.active = {layer: _clone_binding(value) for layer, value in self.injector.active.items()}
            shadow.gate_static = {layer: _clone_binding(value) for layer, value in self.injector.gate_static.items()}
            shadow.residual_static = {layer: _clone_binding(value) for layer, value in self.injector.residual_static.items()}
            shadow.summaries = []
            self._entries[key] = _Entry(key, shadow)
        entry = self._entries[key]
        _require(entry.injector.network is self.injector.network, 'Mapper owner changed')
        for name in ('active', 'gate_static', 'residual_static'):
            destination, source = getattr(entry.injector, name), getattr(self.injector, name)
            _require(destination.keys() == source.keys(), 'Mapper binding keys changed')
            for layer in source:
                _copy_binding(destination[layer], source[layer])
        self._counts['shadow_updates'] += 1
        return entry

    def _body(self, entry, work_after, logits, summary):
        """Graph-safe tensor body; no Native writes or Python cache mutation."""
        network = self.model.model
        hidden = network.embed_tokens(self._token)
        shadow = entry.injector
        shadow.summaries = []
        shadow.capture_summary = summary
        new_keys, new_values, last_query = [], [], None
        slot = work_after-1
        for index, layer in enumerate(network.layers):
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            attention = layer.self_attn
            q = attention.q_proj(normalized).view(1, 1, attention.num_heads, attention.head_dim).transpose(1, 2)
            k = attention.k_proj(normalized).view(1, 1, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)
            v = attention.v_proj(normalized).view(1, 1, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)
            cos, sin = self._rope[index]
            q, k = apply_rotary_pos_emb(q, k, cos, sin, self._position)
            new_keys.append(k); new_values.append(v)
            self.working.keys[index][:, :, slot:slot+1].copy_(k)
            self.working.values[index][:, :, slot:slot+1].copy_(v)
            wk = self.working.keys[index][:, :, :work_after]
            wv = self.working.values[index][:, :, :work_after]
            if index in shadow.active:
                wv = shadow.apply(index, wv, normalized, query=q, keys=wk)
            out = F.scaled_dot_product_attention(q, wk, wv, dropout_p=0., is_causal=False, enable_gqa=True)
            out = attention.o_proj(out.transpose(1, 2).contiguous().view(1, 1, attention.hidden_size))
            hidden = residual+out
            residual = hidden
            hidden = residual+layer.mlp(layer.post_attention_layernorm(hidden))
            last_query = q
        shadow.capture_summary = False
        scores = self.model.lm_head(network.norm(hidden))[:, -1].float() if logits else None
        return scores, last_query, tuple(shadow.summaries), torch.stack(new_keys), torch.stack(new_values)

    def _capture(self, entry, work_after, logits, summary):
        start = time.perf_counter()
        # Warmup/capture mutate append slots and injected historical V. Restore
        # every potentially touched slot before the first real replay.
        saved_k = self.working.key_storage[:, :, :, :work_after].clone()
        saved_v = self.working.value_storage[:, :, :, :work_after].clone()
        current = torch.cuda.current_stream(self.device)
        stream = self._capture_stream
        try:
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(self.WARMUP_ITERATIONS):
                    self._body(entry, work_after, logits, summary)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            # Independent graph pools avoid assuming a fixed 2/4 branch order
            # or aliasing live probe outputs across separate captures.
            with torch.cuda.graph(graph, stream=stream):
                outputs = self._body(entry, work_after, logits, summary)
            entry.graph, entry.outputs = graph, outputs
        finally:
            current.wait_stream(stream)
            self.working.key_storage[:, :, :, :work_after].copy_(saved_k)
            self.working.value_storage[:, :, :, :work_after].copy_(saved_v)
            current.synchronize()
            self._counts['capture_host_seconds'] += time.perf_counter()-start
        self._counts['captures'] += 1

    @torch.no_grad()
    def step(self, token, logits=True, summary=False):
        _require(not self.closed, 'Draft runtime has been closed')
        _require(type(logits) is bool and type(summary) is bool, 'Graph flags must be boolean')
        _require(not getattr(self.injector.network, 'attention_features', 0), 'Unsupported mapper attention features')
        owners = (self.native.key_storage.data_ptr(), self.native.value_storage.data_ptr(),
                  self.working.key_storage.data_ptr(), self.working.value_storage.data_ptr())
        _require(owners == self._owners, 'Owned cache buffers were replaced')
        _require(self.working.active, 'Working cache must be bound before a Draft step')
        n, work_before = self.native.get_seq_length(), self.working.lengths[0]
        _require(all(v == n for v in self.native.lengths), 'Native layers have inconsistent lengths')
        _require(all(v == work_before for v in self.working.lengths), 'Working layers have inconsistent lengths')
        _require(n < self.native.capacity and work_before < self.working.capacity, 'Cache capacity exceeded')
        if isinstance(token, torch.Tensor):
            _require(token.dtype == torch.long and token.numel() == 1, 'Expected a single integer token tensor')
            self._token.copy_(token.reshape(1, 1))
        else:
            _require(type(token) is int and 0 <= token < self.model.config.vocab_size, 'Invalid token')
            self._token.fill_(token)
        self._position.fill_(n)
        work_after = work_before+1
        key = (work_after, logits, summary, self._bindings())
        entry = self._entry(key)
        if self.mode == 'cuda_graph':
            if entry.graph is None:
                self._capture(entry, work_after, logits, summary)
            entry.graph.replay()
            outputs = entry.outputs
            self._counts['replays'] += 1
        else:
            outputs = self._body(entry, work_after, logits, summary)
            self._counts['resident_steps'] += 1
        scores, query, summaries, keys, values = outputs
        # The native destination uses the *actual* host length at every call,
        # outside capture. All layers retain full unmodified backing history.
        self.native.key_storage[:, :, :, n:n+1].copy_(keys)
        self.native.value_storage[:, :, :, n:n+1].copy_(values)
        self.native.lengths[:] = [n+1]*len(self.native.lengths)
        self.working.lengths[:] = [work_after]*len(self.working.lengths)
        return (scores, query.detach().clone() if self.retain_query else None,
                [value.detach().clone() for value in summaries])

    def stats(self):
        return dict(mode=self.mode, entries=len(self._entries), **self._counts)

    def close(self):
        if not self.closed:
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            self._entries.clear()
            self.closed = True
