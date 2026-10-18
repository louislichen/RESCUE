"""Probe deferral, immutable-content affine memo, and physical KV slot reuse."""
import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from exp.selective_kv.bridge import ValueMapper
from rescue.cache import NativeCache, WorkingCache
from rescue.config import SystemConfig
from rescue.engine import RescueEngine
from rescue.injection import ContentInjector
from rescue.policy import RiskModel
from rescue.selection import ContextSelector
from test_rescue_system import models, COST


class ReuseTests(unittest.TestCase):
    def test_packed_and_layerwise_gather_have_identical_native_views(self):
        native = NativeCache(3, 2, 4, 32, torch.float32, 'cpu')
        for i in range(3):
            native.update(torch.randn(1, 2, 17, 4), torch.randn(1, 2, 17, 4), i)
        p = torch.tensor([0, 2, 4, 9, 15, 16])
        old, new = (WorkingCache(native, 8, 4, direct_gather=True, mode=m) for m in ('layerwise', 'packed'))
        old.bind(native, p)
        new.bind(native, p)
        self.assertTrue(torch.equal(old.key_storage[:, :, :, :len(p)], new.key_storage[:, :, :, :len(p)]))
        self.assertTrue(torch.equal(old.value_storage[:, :, :, :len(p)], new.value_storage[:, :, :, :len(p)]))

    def test_incremental_restores_old_injection_and_discards_rejected_tail(self):
        native = NativeCache(3, 2, 4, 32, torch.float32, 'cpu')
        for i in range(3):
            native.update(torch.randn(1, 2, 16, 4), torch.randn(1, 2, 16, 4), i)
        working = WorkingCache(native, 8, 4, mode='incremental')
        original = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14])
        working.bind(native, original)
        layers, dirty = torch.tensor([0, 2]), torch.tensor([1, 5])
        working.mark_dirty(layers, dirty)
        working.value_storage[layers[:, None], 0, :, dirty[None, :], :] += 100
        # Append a speculative suffix then commit only one state.
        for layer in range(3):
            for _ in range(3):
                k, v = torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4)
                native.update(k, v, layer)
                working.append(layer, k, v)
        native.crop(17)
        working.invalidate()
        selected = torch.tensor([0, 2, 6, 8, 10, 12, 14, 16])
        working.bind(native, selected)
        physical = working.physical_indices(torch.arange(8))
        for layer in range(3):
            torch.testing.assert_close(working.keys[layer][:, :, physical], native[layer][0][:, :, selected])
            torch.testing.assert_close(working.values[layer][:, :, physical], native[layer][1][:, :, selected])
        self.assertEqual(working.incremental_binds, 1)
        self.assertEqual(working.gathered_position_layers, 3*(8+1))
        self.assertEqual(working.restored_value_positions, 4)
        # New request invalidates resident-token identity even at same length.
        working.reset()
        for layer in range(3):
            native.values[layer].zero_()
        working.bind(native, selected)
        self.assertEqual(working.full_binds, 1)
        self.assertTrue(torch.equal(working.value_storage[:, :, :, :8], torch.zeros_like(working.value_storage[:, :, :, :8])))

    def test_affine_memo_remains_correct_when_prior_and_positions_change(self):
        torch.manual_seed(7)
        network = ValueMapper([0], [0], source_dim=8, value_dim=4, query_dim=8, rank=4).eval()
        t, s = (NativeCache(1, heads, 4, 24, torch.float32, 'cpu') for heads in (2, 1))
        t.update(torch.randn(1, 2, 12, 4), torch.randn(1, 2, 12, 4), 0)
        s.update(torch.randn(1, 1, 12, 4), torch.randn(1, 1, 12, 4), 0)
        normal = ContentInjector(network, 24, 'cpu', tokens=2, optimized=True)
        fast = ContentInjector(network, 24, 'cpu', tokens=2, optimized=True, memo_affine=True)
        selected = torch.tensor([0, 2, 4, 6, 10, 11])
        for prior in (torch.arange(1., 13.)/78, torch.arange(12., 0., -1)/78):
            for injector in (normal, fast):
                injector.bind(t, s, selected, prior, 2)
            for _ in range(3):
                hidden = torch.randn(1, 1, 8)
                native = s[0][1][:, :, selected]
                torch.testing.assert_close(fast.apply(0, native.clone(), hidden),
                                           normal.apply(0, native.clone(), hidden), rtol=1e-6, atol=1e-7)
        first = fast.affine_positions
        fast.bind(t, s, selected, prior, 2)
        self.assertEqual(first, fast.affine_positions)
        self.assertLess(fast.affine_positions, normal.affine_positions+2)

    def test_deferral_matches_every_boundary_attention_and_target_output(self):
        t, s, mapper = models()
        risk = RiskModel().eval()
        base = SystemConfig(budget=8, recent=3, max_new_tokens=15, attention_backend='eager', working_mode='packed')
        ids, observations, trajectories = list(range(3, 35)), [], []
        original = ContextSelector.select
        for deferred in (False, True):
            seen = []
            def capture(selector, teacher, student, index):
                seen.append(student.a.clone())
                return original(selector, teacher, student, index)
            with RescueEngine(t, s, replace(base, deferred_probes=deferred), mapper, risk, COST) as engine:
                with patch.object(ContextSelector, 'select', capture):
                    result = engine.generate(ids)
                self.assertEqual(result['generated'], engine.reference(ids)['generated'])
                self.assertEqual(result['draft_shadow_probes'], result['blocks']-1 if deferred else result['draft_forwards'])
            observations.append(seen)
            trajectories.append([(r['cap'], r['accepted'], r['proposed']) for r in result['trace']])
        self.assertEqual(trajectories[0], trajectories[1])
        for a, b in zip(*observations):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_incremental_engine_all_accepted_eos_and_request_reset(self):
        t, _, _ = models()
        s = copy.deepcopy(t)
        cfg = SystemConfig(selector='full', budget=8, recent=3, max_new_tokens=9,
                           length_policy='fixed', fixed_k=3, working_mode='incremental')
        ids = list(range(3, 36))
        with RescueEngine(t, s, cfg) as engine:
            expected = engine.reference(ids)['generated']
            for _ in range(2):
                result = engine.generate(ids)
                self.assertEqual(result['generated'], expected)
                self.assertTrue(all(r['accepted'] == r['proposed'] for r in result['trace']))
                self.assertEqual(result['draft_shadow_probes'], result['blocks']-1)
            eos = expected[0]
            terminal = engine.generate(ids, eos)
            self.assertEqual(terminal['generated'], [eos])
            self.assertEqual(terminal['draft_shadow_probes'], 0)

    def test_incremental_engine_with_mapper_and_hard_layer_route(self):
        t, s, mapper = models()
        cfg = SystemConfig(budget=8, recent=3, max_new_tokens=12, length_policy='fixed',
                           working_mode='incremental', injection_layers=(1,), injection_rule='joint')
        ids = list(range(3, 38))
        with RescueEngine(t, s, cfg, mapper) as engine:
            reference = engine.reference(ids)['generated']
            for _ in range(2):
                result = engine.generate(ids)
                self.assertEqual(result['generated'], reference)
                self.assertGreater(result['working_incremental_binds'], 0)
                self.assertGreater(result['working_restored_value_positions'], 0)
            self.assertEqual(set(engine.injector.memo), {1})
        with RescueEngine(t, s, replace(cfg, injection_layers=()), mapper) as engine:
            result = engine.generate(ids)
            self.assertEqual(result['generated'], reference)
            self.assertEqual(result['projected_positions'], 0)
            self.assertFalse(engine.injector.memo)


if __name__ == '__main__':
    unittest.main()
