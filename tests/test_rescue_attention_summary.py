"""CPU correctness checks for round-local sharing, not throughput measurements."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from exp.attention_horizon.selection import ReadState as LegacyReadState
from exp.attention_horizon.selection import feature_vector
from rescue import features
from rescue.attention_summary import mean_attention
from rescue.config import SystemConfig
from rescue.selection import ContextSelector, LazyReadState, ReadState
from rescue.step_budget import compact_attention


def inputs(n=31):
    generator = torch.Generator().manual_seed(891)
    # Different row sums make averaging before normalization observably wrong.
    t = torch.rand(6, n, generator=generator)*torch.arange(1., 7.)[:, None]
    s = torch.rand(4, n, generator=generator)*torch.arange(2., 6.)[:, None]
    tv = torch.randn(2, n, 4, generator=generator)
    sv = torch.randn(1, n, 3, generator=generator)
    return t, s, tv, sv


def states(kind=LazyReadState, n=31):
    t, s, tv, sv = inputs(n)
    return kind(t, tv), kind(s, sv)


class AttentionSummaryTests(unittest.TestCase):
    def test_eager_and_lazy_cache_the_normalized_fp32_head_mean(self):
        rows = torch.tensor([[0., 0., 0., 0.], [1., 3., 2., 4.],
                             [20., 0., 4., 16.], [0., 1., 0., 0.]])
        values = torch.arange(24.).reshape(2, 4, 3)
        for kind in (ReadState, LazyReadState):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(kind=kind.__name__, dtype=dtype):
                    state = kind(rows.to(dtype), values.to(dtype))
                    expected_rows = rows.to(dtype).float()
                    expected_rows = expected_rows/expected_rows.sum(-1, keepdim=True).clamp_min(1e-12)
                    self.assertTrue(torch.equal(state.a, expected_rows))
                    self.assertEqual(state.mean_attention.dtype, torch.float32)
                    self.assertTrue(torch.equal(state.mean_attention, expected_rows.mean(0)))
                    self.assertTrue(torch.isfinite(state.mean_attention).all())
                    self.assertIs(mean_attention(state), state.mean_attention)
                    self.assertIs(mean_attention(state), mean_attention(state))
                    self.assertFalse(torch.equal(state.mean_attention, rows.to(dtype).float().mean(0)))

    def test_each_state_constructs_exactly_one_full_head_mean(self):
        rows, _, values, _ = inputs()
        original = torch.Tensor.mean
        for kind in (ReadState, LazyReadState):
            with self.subTest(kind=kind.__name__):
                observed = []

                def tracked(tensor, *args, **kwargs):
                    if tensor.ndim == 2 and (args == (0,) or kwargs.get('dim') == 0):
                        observed.append(tensor)
                    return original(tensor, *args, **kwargs)

                with patch.object(torch.Tensor, 'mean', new=tracked):
                    state = kind(rows, values)
                    for _ in range(4):
                        self.assertIs(mean_attention(state), state.mean_attention)
                self.assertEqual(len(observed), 1)
                self.assertIs(observed[0], state.a)

    def test_lazy_mean_does_not_materialize_native_values_or_readouts(self):
        state, _ = states()
        self.assertIsInstance(state, ReadState)
        self.assertIsInstance(state, LegacyReadState)
        self.assertIsNone(state._v)
        self.assertIsNone(state._target)
        self.assertIsNone(state._scale)
        mean_attention(state)
        self.assertIsNone(state._v)
        self.assertIsNone(state._target)
        self.assertIsNone(state._scale)
        # Inherited read equations still work when deliberately requested.
        legacy = LegacyReadState(state.a, state._values)
        selected = torch.tensor([0, 3, 7, 18, 29, 30])
        torch.testing.assert_close(state.error(selected), legacy.error(selected), rtol=1e-6, atol=1e-7)

    def test_legacy_and_minimal_state_fallback_does_not_install_a_cache(self):
        t, _, tv, _ = inputs()
        for state in (LegacyReadState(t, tv), SimpleNamespace(a=t)):
            with self.subTest(kind=type(state).__name__):
                before = set(vars(state))
                expected = state.a.mean(0)
                original = torch.Tensor.mean
                count = []

                def tracked(tensor, *args, **kwargs):
                    if tensor is state.a:
                        count.append(1)
                    return original(tensor, *args, **kwargs)

                with patch.object(torch.Tensor, 'mean', new=tracked):
                    first, second = mean_attention(state), mean_attention(state)
                self.assertTrue(torch.equal(first, expected))
                self.assertTrue(torch.equal(second, expected))
                self.assertEqual(len(count), 2)
                self.assertEqual(set(vars(state)), before)
                self.assertFalse(hasattr(state, 'mean_attention'))

    def test_cached_branch_never_evaluates_fallback_mean(self):
        state = SimpleNamespace(a=object(), mean_attention=torch.arange(5.))
        # The object deliberately has no .mean method. An eagerly evaluated
        # getattr default, or an unnecessary recomputation, fails this test.
        self.assertIs(mean_attention(state), state.mean_attention)

    def test_split_features_match_frozen_legacy_contract_and_leave_means_intact(self):
        selected = torch.tensor([0, 2, 8, 17, 28, 29, 30])
        for kind in (ReadState, LazyReadState):
            with self.subTest(kind=kind.__name__):
                t, s = states(kind)
                ta, sa = t.mean_attention.clone(), s.mean_attention.clone()
                prepared = features.prepare(t, s, progress=8, remaining=24, recent=3)
                actual = features.finish(prepared, t, s, selected)
                expected = feature_vector(t, s, selected, 8, 0, torch.zeros(3), 24, 3)
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
                self.assertTrue(torch.equal(t.mean_attention, ta))
                self.assertTrue(torch.equal(s.mean_attention, sa))

    def test_compact_features_equal_prior_formula_for_cached_and_minimal_states(self):
        t, s = states()
        selected = torch.tensor([0, 2, 8, 17, 28, 29, 30])
        ta, sa = t.a.mean(0), s.a.mean(0)
        expected = torch.stack((ta.new_tensor(math.log(t.n)/10), ta[selected].sum(),
            sa[selected].sum(), (ta[selected]-sa[selected]).abs().sum(),
            ta.new_tensor(3/4), ta.new_tensor(11/128), ta.new_tensor(29/128), ta.new_tensor(.4)))
        for teacher, student in ((t, s),
                                 (SimpleNamespace(a=t.a, n=t.n), SimpleNamespace(a=s.a, n=s.n))):
            actual = compact_attention(teacher, student, selected, 3, 11, 29, .4)
            self.assertTrue(torch.equal(actual, expected))

    def test_selector_modes_match_legacy_states_across_refresh_and_carry(self):
        base = SystemConfig(budget=10, recent=3, coverage_candidates=16,
            coverage_batch=2, coverage_rows=8, coarse_chunk_size=4, refresh_every=4)
        settings = [replace(base, selector='teacher'), replace(base, selector='joint'),
                    replace(base, selector='coverage'), replace(base, selector='coverage', chunk_size=4),
                    replace(base, selector='hierarchical'),
                    replace(base, selector='hierarchical', refresh_policy='adaptive',
                            refresh_min=1, refresh_max=4, attention_drift=.2)]
        for cfg in settings:
            with self.subTest(selector=cfg.selector, chunk=cfg.chunk_size, refresh=cfg.refresh_policy):
                cached_selector, legacy_selector = ContextSelector(cfg), ContextSelector(cfg)
                for round_index, n in ((0, 31), (1, 33), (4, 36)):
                    rows_t, rows_s, tv, sv = inputs(n)
                    t, s = LazyReadState(rows_t, tv), LazyReadState(rows_s, sv)
                    lt, ls = LegacyReadState(rows_t, tv), LegacyReadState(rows_s, sv)
                    ta, sa = t.mean_attention.clone(), s.mean_attention.clone()
                    actual = cached_selector.select(t, s, round_index)
                    expected = legacy_selector.select(lt, ls, round_index)
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertTrue(torch.equal(t.mean_attention, ta))
                    self.assertTrue(torch.equal(s.mean_attention, sa))

    def test_coverage_mask_operates_on_clone_not_shared_prior(self):
        cfg = SystemConfig(budget=10, recent=3, selector='coverage', coverage_candidates=16,
                           coverage_batch=2)
        t, s = states()
        ta, sa = t.mean_attention.clone(), s.mean_attention.clone()
        ContextSelector(cfg).select(t, s, 0)
        self.assertGreater(float(ta[0]), 0.)
        self.assertTrue(torch.equal(mean_attention(t), ta))
        self.assertTrue(torch.equal(mean_attention(s), sa))

    def test_consumers_reuse_eager_mean_without_reducing_full_attention_again(self):
        t, s = states()
        selected = torch.tensor([0, 2, 8, 17, 28, 29, 30])
        ta, sa = t.mean_attention.clone(), s.mean_attention.clone()
        cfg = SystemConfig(budget=10, recent=3, selector='coverage', coverage_candidates=16,
                           coverage_batch=2)
        original = torch.Tensor.mean

        def no_full_mean(tensor, *args, **kwargs):
            if tensor is t.a or tensor is s.a:
                raise AssertionError('A consumer recomputed the cached full head mean')
            return original(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, 'mean', new=no_full_mean):
            # CPU concurrency checks read-only use, not CUDA event ordering.
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(features.prepare, t, s, 8, 24, 3)
                ContextSelector(cfg).select(t, s, 0)
                prepared = future.result()
            features.finish(prepared, t, s, selected)
            compact_attention(t, s, selected, 3, 11, 29, .4)
        self.assertTrue(torch.equal(t.mean_attention, ta))
        self.assertTrue(torch.equal(s.mean_attention, sa))


if __name__ == '__main__':
    unittest.main()
