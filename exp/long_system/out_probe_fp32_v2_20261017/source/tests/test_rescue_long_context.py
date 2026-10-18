import copy
from dataclasses import replace
import unittest

import torch

from rescue.attention import TargetObservation
from rescue.cache import NativeCache
from rescue.config import SystemConfig
from rescue.efficient_attention import install_efficient_attention
from rescue.engine import RescueEngine
from rescue.selection import ContextSelector, LazyReadState, mandatory
from test_rescue_system import models


def cache_for(model, capacity=128):
    c = model.config
    return NativeCache(c.num_hidden_layers, c.num_key_value_heads,
                       c.hidden_size//c.num_attention_heads, capacity, torch.float32, 'cpu')


class EfficientAttentionTests(unittest.TestCase):
    @torch.no_grad()
    def test_chunk_prefill_and_cached_queries_match_dense_causal_reference(self):
        dense, _, _ = models()
        fused = install_efficient_attention(copy.deepcopy(dense))
        ids = torch.tensor([[3, 7, 2, 8, 4, 9, 1, 12, 6, 3, 4]])
        native = cache_for(fused)
        expected = dense(ids, use_cache=True).logits
        outputs = []
        for chunk in (ids[:, :4], ids[:, 4:8], ids[:, 8:]):
            outputs.append(fused(chunk, past_key_values=native, use_cache=True).logits)
        torch.testing.assert_close(torch.cat(outputs, 1), expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(native.get_seq_length(), ids.shape[1])

    @torch.no_grad()
    def test_compact_observation_selects_only_causally_visible_keys(self):
        dense, _, _ = models()
        fused = install_efficient_attention(copy.deepcopy(dense))
        a, b = cache_for(dense), cache_for(fused)
        prefix = torch.tensor([[4, 2, 7, 10, 6, 3]])
        query = torch.tensor([[12, 3, 8]])
        observation = TargetObservation(fused).install()
        try:
            observation.mode = 'last'
            expected = dense(prefix, past_key_values=a, use_cache=True, output_attentions=True)
            fused(prefix, past_key_values=b, use_cache=True)
            torch.testing.assert_close(observation.rows, expected.attentions[-1][0, :, -1], rtol=1e-5, atol=1e-6)
            self.assertEqual(observation.queries.shape[-2], 1)
            observation.mode = 'all'
            expected = dense(query, past_key_values=a, use_cache=True, output_attentions=True)
            fused(query, past_key_values=b, use_cache=True)
            self.assertIsNone(observation.rows)
            for index in range(3):
                n = prefix.shape[1]+index+1
                torch.testing.assert_close(observation.row(index, n),
                    expected.attentions[-1][0, :, index, :n], rtol=1e-5, atol=1e-6)
        finally:
            observation.restore()

    @torch.no_grad()
    def test_complete_fused_engine_preserves_target_output_across_requests(self):
        teacher, student, mapper = models()
        install_efficient_attention(teacher)
        install_efficient_attention(student)
        config = SystemConfig(budget=8, recent=2, selector='coverage', coverage_candidates=12,
                              length_policy='fixed', fixed_k=4, max_new_tokens=17,
                              prefill_chunk_size=5)
        with RescueEngine(teacher, student, config, mapper) as engine:
            for ids in ([1, 4, 3, 7, 2, 11, 6, 10, 3, 2, 9], [3, 8, 4, 9, 1, 5, 12]):
                reference = engine.reference(ids)['generated']
                result = engine.generate(ids)
                self.assertEqual(result['generated'], reference)
                self.assertEqual(result['draft_shadow_probes'], result['blocks']-1)


class HierarchicalSelectionTests(unittest.TestCase):
    def test_budget_anchors_bounded_similarity_and_append_only_normalization(self):
        torch.manual_seed(3)
        cfg = SystemConfig(budget=16, recent=4, selector='hierarchical', coverage_candidates=32,
            coverage_rows=20, coarse_chunk_size=8, coverage_batch=4, refresh_policy='adaptive',
            refresh_max=3, attention_drift=1., coverage_floor=0.)
        selector = ContextSelector(cfg)
        tv, sv = torch.randn(2, 138, 8), torch.randn(1, 138, 8)
        saved = None
        for round_index in range(5):
            n = 126+round_index*3
            t, s = LazyReadState(torch.rand(4, n), tv[:, :n]), LazyReadState(torch.rand(3, n), sv[:, :n])
            chosen = selector.select(t, s, round_index)
            self.assertEqual(len(chosen), cfg.budget)
            self.assertEqual(len(chosen.unique()), cfg.budget)
            self.assertTrue(torch.isin(mandatory(n, cfg.recent, 'cpu'), chosen).all())
            if saved is None:
                saved = selector._content[0][:126].clone()
            self.assertTrue(torch.equal(saved, selector._content[0][:126]))
        self.assertEqual(selector.refreshes, 2)
        self.assertLessEqual(selector.similarity_elements, 2*2*cfg.coverage_rows*48)

    def test_attention_shift_refreshes_before_maximum_age(self):
        cfg = SystemConfig(budget=12, recent=4, selector='hierarchical', coverage_rows=16,
            coverage_candidates=24, coarse_chunk_size=8, refresh_policy='adaptive',
            refresh_max=16, attention_drift=.2, coverage_floor=0.)
        selector = ContextSelector(cfg)
        values = torch.randn(1, 64, 8)
        for index, peak in enumerate((4, 4, 40)):
            rows = torch.full((2, 64), 1e-6)
            rows[:, peak] = 1
            state = LazyReadState(rows, values)
            selector.select(state, state, index)
            self.assertEqual(selector.refreshes, 1 if index < 2 else 2)


if __name__ == '__main__':
    unittest.main()
