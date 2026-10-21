"""CPU primitive checks; these do not establish CUDA-graph equivalence/speed."""
import copy
import unittest

import torch

from rescue.attention import DraftAttention
from rescue.cache import NativeCache, WorkingCache
from rescue.draft_graph import ResidentDraftRuntime, _clone_binding, _copy_binding
from rescue.injection import ContentInjector
from rescue.packing import pack_projections
from rescue.query_mapper import QueryValueMapper
from test_rescue_system import models


def native_for(model, capacity=32, prefix=6):
    c = model.config
    cache = NativeCache(c.num_hidden_layers, c.num_key_value_heads,
                        c.hidden_size//c.num_attention_heads, capacity, torch.float32, 'cpu')
    for layer in range(c.num_hidden_layers):
        shape = (1, c.num_key_value_heads, prefix, c.hidden_size//c.num_attention_heads)
        cache.update(torch.randn(shape), torch.randn(shape), layer)
    return cache


class DraftGraphPrimitiveTests(unittest.TestCase):
    def setUp(self):
        teacher, self.student, _ = models()
        self.student.requires_grad_(False)
        pack_projections(self.student)
        self.teacher_cache = native_for(teacher)
        self.left = native_for(self.student)
        self.right = copy.deepcopy(self.left)
        mapper = QueryValueMapper(layers=[1], teacher_layers=[2], source_dim=16,
            value_dim=8, query_dim=24, rank=4, residual_rank=3).eval().requires_grad_(False)
        with torch.no_grad():
            mapper.residuals[0][-1].weight.normal_(0, .1)
        self.injectors = [ContentInjector(mapper, 32, 'cpu', tokens=2, strength=.5,
                         optimized=True, memo_affine=True, layers=(1,)) for _ in range(2)]
        self.works = [WorkingCache(cache, 4, 5, direct_gather=True, mode='packed')
                      for cache in (self.left, self.right)]
        self.bind()

    def bind(self):
        for native, work, injector in zip((self.left, self.right), self.works, self.injectors):
            selected = torch.tensor([0, 2, 4, 5])
            work.bind(native, selected)
            prior = torch.arange(native.get_seq_length(), dtype=torch.float32)+1
            injector.bind(self.teacher_cache, native, selected, prior, recent=1, working=work)

    def eager_step(self, token, logits, summary):
        observer = DraftAttention(self.student, self.works[0], self.injectors[0],
                                  optimized=True, deferred=True).install()
        observer.prefill = False
        observer.record_injection_summary = summary
        try:
            scores = observer.step(token, self.left, logits=logits)
            return scores, observer.last_query.clone(), [v.clone() for v in self.injectors[0].summaries]
        finally:
            observer.restore()

    def compare_cache(self):
        self.assertEqual(self.left.lengths, self.right.lengths)
        self.assertEqual(self.works[0].lengths, self.works[1].lengths)
        for index in range(len(self.left)):
            for a, b in zip(self.left[index], self.right[index]):
                self.assertTrue(torch.equal(a, b))
            end = self.works[0].lengths[index]
            self.assertTrue(torch.equal(self.works[0].keys[index][:, :, :end],
                                        self.works[1].keys[index][:, :, :end]))
            self.assertTrue(torch.equal(self.works[0].values[index][:, :, :end],
                                        self.works[1].values[index][:, :, :end]))

    def test_resident_matches_eager_tokens_native_working_queries_and_summaries(self):
        runtime = ResidentDraftRuntime(self.student, self.right, self.works[1], self.injectors[1])
        queries, summaries = [], []
        for index, token in enumerate((7, 12, 3, 18)):
            expected = self.eager_step(token, logits=index != 3, summary=index < 2)
            actual = runtime.step(torch.tensor([[token]]), logits=index != 3, summary=index < 2)
            if expected[0] is None:
                self.assertIsNone(actual[0])
            else:
                self.assertTrue(torch.equal(expected[0], actual[0]))
            self.assertTrue(torch.equal(expected[1], actual[1]))
            self.assertEqual(len(expected[2]), len(actual[2]))
            for a, b in zip(expected[2], actual[2]):
                self.assertTrue(torch.equal(a, b))
            queries.append((actual[1], actual[1].clone()))
            summaries.extend((v, v.clone()) for v in actual[2])
            self.compare_cache()
        for value, snapshot in queries+summaries:
            self.assertTrue(torch.equal(value, snapshot))
        self.assertTrue(any(float(snapshot[-1]) > 0 for _, snapshot in summaries))
        self.assertEqual(runtime.stats()['resident_steps'], 4)
        self.assertEqual(runtime.stats()['shadow_updates'], 4)
        self.assertEqual(runtime.stats()['captures'], 0)
        runtime.close()
        with self.assertRaises(ValueError):
            runtime.step(3)

    def test_rebinding_updates_same_shape_shadow_values_and_real_native_position(self):
        runtime = ResidentDraftRuntime(self.student, self.right, self.works[1], self.injectors[1])
        for token in (7, 9):
            self.bind()
            for injector in self.injectors:
                injector.gate_static[1] = injector.gate_static[1]+.125
            expected, actual = self.eager_step(token, True, True), runtime.step(token, True, True)
            self.assertTrue(torch.equal(expected[0], actual[0]))
            self.compare_cache()
        self.assertEqual(runtime.stats()['entries'], 1)
        self.assertEqual(self.right.get_seq_length(), 8)

    def test_binding_copy_has_owned_storage_and_rejects_changed_shapes(self):
        value = (1, torch.arange(4), None, [torch.ones(2)])
        shadow = _clone_binding(value)
        self.assertNotEqual(value[1].data_ptr(), shadow[1].data_ptr())
        value[1].add_(3); _copy_binding(shadow, value)
        self.assertTrue(torch.equal(shadow[1], value[1]))
        with self.assertRaises(ValueError):
            _copy_binding(shadow, (1, torch.ones(5), None, [torch.ones(2)]))

    def test_unsupported_mode_device_rope_and_cache_ownership_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'CUDA model'):
            ResidentDraftRuntime(self.student, self.right, self.works[1], self.injectors[1], 'cuda_graph')
        runtime = ResidentDraftRuntime(self.student, self.right, self.works[1], self.injectors[1])
        self.right.key_storage = self.right.key_storage.clone()
        with self.assertRaisesRegex(ValueError, 'buffers were replaced'):
            runtime.step(1)
        self.right.capacity = 512
        with self.assertRaisesRegex(ValueError, 'RoPE cache'):
            ResidentDraftRuntime(self.student, self.right, self.works[1], self.injectors[1])


if __name__ == '__main__':
    unittest.main()
