"""Native append/crop storage and separately owned working reads."""
import torch
from transformers.cache_utils import Cache


class NativeCache(Cache):
    def __init__(self, layers, kv_heads, head_dim, capacity, dtype, device):
        self.capacity = capacity
        self.keys = [torch.empty(1, kv_heads, capacity, head_dim, dtype=dtype, device=device) for _ in range(layers)]
        self.values = [torch.empty_like(k) for k in self.keys]
        self.lengths = [0]*layers

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        start = self.lengths[layer_idx]
        end = start+key_states.shape[-2]
        if end > self.capacity:
            raise ValueError('Native cache capacity exceeded; no silent eviction')
        self.keys[layer_idx][:, :, start:end].copy_(key_states)
        self.values[layer_idx][:, :, start:end].copy_(value_states)
        self.lengths[layer_idx] = end
        return self[layer_idx]

    def get_seq_length(self, layer_idx=0):
        return self.lengths[layer_idx]

    def get_max_length(self):
        return self.capacity

    def get_usable_length(self, new_seq_length, layer_idx=0):
        if self.lengths[layer_idx]+new_seq_length > self.capacity:
            raise ValueError('Native cache capacity exceeded')
        return self.lengths[layer_idx]

    def __getitem__(self, layer):
        n = self.lengths[layer]
        return self.keys[layer][:, :, :n], self.values[layer][:, :, :n]

    def __len__(self):
        return len(self.keys)

    def crop(self, n):
        if type(n) is not int or n < 0 or any(n > old for old in self.lengths):
            raise ValueError('Cannot extend native cache during rollback')
        self.lengths[:] = [n]*len(self.lengths)

    def reset(self):
        self.lengths[:] = [0]*len(self.lengths)

    def to_legacy_cache(self):
        return tuple(self[i] for i in range(len(self)))


class WorkingCache:
    def __init__(self, native, budget, max_draft, direct_gather=False):
        self.capacity = min(native.capacity, budget)+max_draft+1
        self.keys = [k.new_empty(1, k.shape[1], self.capacity, k.shape[-1]) for k in native.keys]
        self.values = [torch.empty_like(k) for k in self.keys]
        self.lengths = [0]*len(self.keys)
        self.bound_prefix = None
        self.positions = None
        self.active = False
        self.gathered_layers = 0
        self.direct_gather = direct_gather

    def bind(self, native, positions):
        n = native.get_seq_length()
        if positions.numel() > self.capacity or not positions.numel():
            raise ValueError('Invalid working set capacity')
        if self.direct_gather:
            valid = bool((positions[0] >= 0) & (positions[-1] < n) & (positions[1:] > positions[:-1]).all())
        else:
            valid = not (int(positions[0]) < 0 or int(positions[-1]) >= n or not bool((positions[1:] > positions[:-1]).all()))
        if not valid:
            raise ValueError('Workset must be sorted, unique and inside committed native KV')
        self.positions, self.bound_prefix = positions, n
        for layer in range(len(native)):
            k, v = native[layer]
            # One gather per layer/round, never per Draft token.
            if self.direct_gather:
                torch.index_select(k, 2, positions, out=self.keys[layer][:, :, :len(positions)])
                torch.index_select(v, 2, positions, out=self.values[layer][:, :, :len(positions)])
            else:
                self.keys[layer][:, :, :len(positions)].copy_(k.index_select(2, positions))
                self.values[layer][:, :, :len(positions)].copy_(v.index_select(2, positions))
            self.lengths[layer] = len(positions)
        self.gathered_layers += len(native)
        self.active = True

    def append(self, layer, key, value):
        if not self.active or key.shape[-2] != 1:
            raise ValueError('Working reads require one sequential query')
        start = self.lengths[layer]
        if start >= self.capacity:
            raise ValueError('Working cache capacity exceeded')
        self.keys[layer][:, :, start:start+1].copy_(key)
        self.values[layer][:, :, start:start+1].copy_(value)
        self.lengths[layer] = start+1
        return self.keys[layer][:, :, :start+1], self.values[layer][:, :, :start+1]

    def invalidate(self):
        self.active = False
        self.positions = self.bound_prefix = None
