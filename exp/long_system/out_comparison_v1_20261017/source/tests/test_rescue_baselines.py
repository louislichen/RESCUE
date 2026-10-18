"""Correctness and absence of intervention-only work in comparison baselines."""
import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from rescue.attention import DraftAttention
from rescue.baselines import BaselineEngine, ChunkRetrieval
from rescue.config import SystemConfig
from test_rescue_system import models


class BaselineTests(unittest.TestCase):
    def test_chunk_means_recent_initialization_frontier_and_budget(self):
        selector = ChunkRetrieval(2, 4)
        self.assertEqual(selector.select(14, 'cpu').tolist(), list(range(8, 14)))
        a = torch.tensor([[.1]*4+[.02]*4+[.01]*4+[.001]*2])
        self.assertEqual(selector.select(14, 'cpu', a).tolist(), list(range(4))+[12, 13])
        # At a new frontier, oldest selected chunk is evicted by local protocol.
        self.assertEqual(selector.select(17, 'cpu').tolist(), list(range(12, 17)))
        self.assertEqual(selector.refreshes, 1)

    def test_fixed_lengths_native_and_retrieval_match_full_target(self):
        teacher, student, _ = models()
        ids = list(range(3, 44))*2
        config = SystemConfig(max_new_tokens=9, length_policy='fixed', max_draft=12)
        original_teacher = teacher.model.layers[-1].self_attn.forward
        original_student = student.model.layers[-1].self_attn.forward
        for chunks in (None, 1):
            for k in (1, 2, 4, 8, 12):
                with BaselineEngine(teacher, student, replace(config, fixed_k=k), top_chunks=chunks) as engine:
                    expected = engine.reference(ids)['generated']
                    with patch('rescue.engine.ReadState', side_effect=AssertionError('unnecessary read state')):
                        actual = engine.generate(ids)
                    self.assertEqual(actual['generated'], expected)
                    self.assertEqual(actual['draft_shadow_probes'], 0)
                    self.assertEqual(actual['projected_positions'], 0)
                    self.assertEqual(actual['draft_forwards'], actual['proposed']+sum(
                        not r['terminal'] and r['accepted'] == r['proposed'] for r in actual['trace']))
                    if chunks is None:
                        self.assertEqual(actual['working_gather_layers'], 0)
                        self.assertEqual(actual['target_probe_rounds'], 0)
                        self.assertEqual(engine.observation.originals, [])
                    else:
                        self.assertTrue(all(r['workset_tokens'] <= chunks*32 for r in actual['trace']))
                    eos = expected[0]
                    self.assertEqual(engine.generate(ids, eos)['generated'], [eos])
                self.assertEqual(teacher.model.layers[-1].self_attn.forward, original_teacher)
                self.assertEqual(student.model.layers[-1].self_attn.forward, original_student)

    def test_all_accepted_catchup_and_exception_restore(self):
        teacher, _, _ = models()
        student = copy.deepcopy(teacher)
        config = SystemConfig(max_new_tokens=8, length_policy='fixed', fixed_k=3)
        ids = list(range(3, 39))
        with BaselineEngine(teacher, student, config) as engine:
            result = engine.generate(ids)
            self.assertTrue(all(r['proposed'] == r['accepted'] for r in result['trace']))
            self.assertEqual(result['draft_forwards'], result['proposed']+len(result['trace'])-1)
            original = student.model.layers[-1].self_attn.forward
            with patch.object(DraftAttention, 'step', side_effect=RuntimeError('injected')):
                with self.assertRaises(RuntimeError):
                    engine.generate(ids)
            self.assertEqual(student.model.layers[-1].self_attn.forward, original)
            self.assertEqual(engine.generate(ids)['generated'], result['generated'])


if __name__ == '__main__':
    unittest.main()
