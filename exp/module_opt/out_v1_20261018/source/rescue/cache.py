"""Native append/crop storage and separately owned working reads."""
import torch
from transformers.cache_utils import Cache


class NativeCache(Cache):
    def __init__(self, layers, kv_heads, head_dim, capacity, dtype, device):
        self.capacity = capacity
        self.key_storage = torch.empty(layers, 1, kv_heads, capacity, head_dim, dtype=dtype, device=device)
        self.value_storage = torch.empty_like(self.key_storage)
        self.keys = list(self.key_storage.unbind(0))
        self.values = list(self.value_storage.unbind(0))
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
    def __init__(self, native, budget, max_draft, direct_gather=False, mode='layerwise'):
        self.capacity = min(native.capacity, budget)+max_draft+1
        self.key_storage = native.key_storage.new_empty(len(native), 1, native.keys[0].shape[1],
                                                        self.capacity, native.keys[0].shape[-1])
        self.value_storage = torch.empty_like(self.key_storage)
        self.keys = list(self.key_storage.unbind(0))
        self.values = list(self.value_storage.unbind(0))
        self.lengths = [0]*len(self.keys)
        self.bound_prefix = None
        self.positions = None
        self.active = False
        self.gathered_layers = 0
        self.direct_gather = direct_gather
        if mode not in ('layerwise', 'packed', 'incremental'):
            raise ValueError('Unknown working-cache mode')
        self.mode = mode
        self.reset()

    def bind(self, native, positions):
        n = native.get_seq_length()
        if positions.numel() > self.capacity or not positions.numel():
            raise ValueError('Invalid working set capacity')
        ids = positions.tolist() if self.mode == 'incremental' else None
        if ids is not None:
            valid = ids[0] >= 0 and ids[-1] < n and all(a < b for a, b in zip(ids, ids[1:]))
        elif self.direct_gather:
            valid = bool((positions[0] >= 0) & (positions[-1] < n) & (positions[1:] > positions[:-1]).all())
        else:
            valid = not (int(positions[0]) < 0 or int(positions[-1]) >= n or not bool((positions[1:] > positions[:-1]).all()))
        if not valid:
            raise ValueError('Workset must be sorted, unique and inside committed native KV')
        self.positions, self.bound_prefix = positions, n
        missing = [p for p in ids if p not in self._slot_by_id] if ids is not None else []
        incremental = (ids is not None and len(ids) == len(self._resident_ids) and
                       self._resident_prefix is not None and n >= self._resident_prefix and
                       len(missing) <= len(ids)//4)
        if incremental:
            # Restore previous query corrections before reusing any physical V
            # slots, including tokens that are no longer selected for injection.
            if self._dirty is not None:
                layers, slots = self._dirty
                source = self._resident_positions[slots]
                self.value_storage[layers[:, None], 0, :, slots[None, :], :] = (
                    native.value_storage[layers[:, None], 0, :, source[None, :], :])
                self.restored_value_positions += len(layers)*len(slots)
            if missing:
                retained = set(ids)
                free = [i for i, p in enumerate(self._resident_ids) if p not in retained]
                assert len(free) == len(missing)
                slots, source = positions.new_tensor(free), positions.new_tensor(missing)
                self.key_storage.index_copy_(3, slots, native.key_storage.index_select(3, source))
                self.value_storage.index_copy_(3, slots, native.value_storage.index_select(3, source))
                for slot, token in zip(free, missing):
                    del self._slot_by_id[self._resident_ids[slot]]
                    self._resident_ids[slot] = token
                    self._slot_by_id[token] = slot
                self._resident_positions.index_copy_(0, slots, source)
            self._physical_for_sorted = positions.new_tensor([self._slot_by_id[p] for p in ids])
            self.incremental_binds += 1
            gathered = len(missing)
        else:
            if self.mode == 'layerwise':
                for layer in range(len(native)):
                    k, v = native[layer]
                    if self.direct_gather:
                        torch.index_select(k, 2, positions, out=self.keys[layer][:, :, :len(positions)])
                        torch.index_select(v, 2, positions, out=self.values[layer][:, :, :len(positions)])
                    else:
                        self.keys[layer][:, :, :len(positions)].copy_(k.index_select(2, positions))
                        self.values[layer][:, :, :len(positions)].copy_(v.index_select(2, positions))
            else:
                # Two gathers cover all layers; no per-layer launch loop.
                torch.index_select(native.key_storage, 3, positions, out=self.key_storage[:, :, :, :len(positions)])
                torch.index_select(native.value_storage, 3, positions, out=self.value_storage[:, :, :, :len(positions)])
            self.full_binds += 1
            self.full_gather_layers += len(native)
            gathered = len(positions)
            if ids is not None:
                self._resident_ids = list(ids)
                self._slot_by_id = {p: i for i, p in enumerate(ids)}
                self._resident_positions = positions.clone()
            self._physical_for_sorted = None
        self._dirty = None
        self._resident_prefix = n
        self.lengths[:] = [len(positions)]*len(native)
        self.gathered_layers += len(native)
        self.gathered_position_layers += gathered*len(native)
        self.active = True

    def physical_indices(self, logical_indices):
        if self._physical_for_sorted is None:
            return logical_indices
        return self._physical_for_sorted[logical_indices]

    def mark_dirty(self, layers, slots):
        if self.mode == 'incremental':
            self._dirty = (layers, slots)

    def reset(self):
        self.invalidate()
        self._resident_ids, self._slot_by_id = [], {}
        self._resident_positions = self._resident_prefix = self._physical_for_sorted = None
        self._dirty = None
        self.gathered_layers = self.full_gather_layers = self.gathered_position_layers = 0
        self.full_binds = self.incremental_binds = self.restored_value_positions = 0

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
