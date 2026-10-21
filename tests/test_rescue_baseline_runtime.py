"""Toy equivalence/ownership checks, not measured pretrained-model speed."""
import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from rescue.attention import DraftAttention
from rescue.baselines import BASELINE_GRAPH_ENTRIES, BaselineEngine
from rescue.config import SystemConfig
from rescue.draft_graph_benchmark import HookOwners, owner_signature
from rescue.draft_graph_smoke import native_digest, record_queries
from rescue.efficient_attention import install_efficient_attention
from rescue.packing import pack_projections
from rescue.round_overhead_run import CorrectnessObserver
from test_rescue_system import models


def fixture(device='cpu', identical=False):
    teacher, student, _ = models()
    if identical:
        student = copy.deepcopy(teacher)
        # Force genuine all-accepted/catchup boundaries even when CUDA uses
        # FP32 Teacher versus BF16 Draft. Attention/KV computations stay real.
        with torch.no_grad():
            teacher.lm_head.weight.zero_(); student.lm_head.weight.zero_()
    teacher, student = teacher.to(device), student.to(device)
    if device == 'cuda':
        student.to(torch.bfloat16)
    for model in (teacher, student):
        model.requires_grad_(False)
        install_efficient_attention(model); pack_projections(model)
    return teacher, student


def cfg(mode='eager', limit=17, k=4):
    return SystemConfig(budget=8, recent=2, length_policy='fixed', fixed_k=k,
        max_draft=12, max_new_tokens=limit, attention_backend='sdpa', schedule='serial',
        working_mode='packed', optimized_execution=True, deferred_probes=True,
        draft_execution=mode)


def prompt(length):
    return [3+i % 40 for i in range(length)]


def trajectory(row):
    return [{k: t[k] for k in ('round_index', 'prefix', 'cap', 'proposed', 'accepted',
                              'terminal', 'workset_tokens')} for t in row['trace']]


class BaselineRuntimeMixin:
    device = 'cpu'
    mode = 'resident'

    @torch.no_grad()
    def paired(self, chunks=1, limit=17, k=4, identical=False, eos=None, lengths=(63, 65)):
        teacher, student = fixture(self.device, identical)
        results = []
        for mode in ('eager', self.mode):
            observer = CorrectnessObserver()
            with BaselineEngine(teacher, student, cfg(mode, limit, k), top_chunks=chunks,
                                round_observer=observer) as engine:
                runs, signature = [], None
                for length in lengths:
                    ids = prompt(length); offset = len(observer.rounds)
                    canonical = engine.reference(ids, eos)['generated']
                    with record_queries(DraftAttention) as queries:
                        row = engine.generate(ids, eos)
                    self.assertEqual(row['generated'], canonical)
                    self.assertEqual(row['draft_forwards'], len(queries))
                    self.assertEqual(row['projected_positions'], 0)
                    self.assertEqual(row['draft_shadow_probes'], 0)
                    current = owner_signature(engine)
                    if signature is not None:
                        self.assertEqual(current, signature)
                    signature = current
                    runs.append(dict(row=row, rounds=copy.deepcopy(observer.rounds[offset:]),
                        queries=copy.deepcopy(queries), native_t=native_digest(engine.native_t),
                        native_s=native_digest(engine.native_s)))
                    if mode != 'eager':
                        self.assertIsNone(engine.injector.network)
                        self.assertEqual(engine.injector.active, {})
                        self.assertEqual(engine.injector.projected_positions, 0)
                        self.assertEqual(engine.draft_runtime.max_entries, BASELINE_GRAPH_ENTRIES)
                results.append(runs)
        for left, right in zip(*results):
            for field in ('generated', 'tokens', 'accepted', 'proposed', 'blocks', 'draft_forwards',
                          'selector_refreshes', 'target_probe_rounds', 'working_gathered_position_layers'):
                self.assertEqual(left['row'][field], right['row'][field], field)
            self.assertEqual(trajectory(left['row']), trajectory(right['row']))
            for field in ('rounds', 'queries', 'native_t', 'native_s'):
                self.assertEqual(left[field], right[field], field)
        return results

    @torch.no_grad()
    def shared_owners(self):
        teacher, student = fixture(self.device); group = HookOwners(teacher, student)
        engines, seen = [], {}
        try:
            for mode in ('eager', 'resident', self.mode):
                engine = BaselineEngine(teacher, student, cfg(mode, limit=9), top_chunks=1)
                group.add(engine); engines.append(engine)
            for repeat, order in enumerate(((1, 0, 2), (2, 1, 0))):
                expected, expected_native = None, None
                for index in order:
                    engine = engines[index]
                    row = group.generate(engine, prompt(63+repeat), None)
                    native = dict(target=native_digest(engine.native_t), student=native_digest(engine.native_s))
                    if expected is None:
                        expected, expected_native = row, native
                    self.assertEqual(row['generated'], expected['generated'])
                    self.assertEqual(trajectory(row), trajectory(expected))
                    self.assertEqual(native, expected_native)
                    state = owner_signature(engine)
                    if index in seen:
                        self.assertEqual(state, seen[index])
                    seen[index] = state
                    group.idle()
            self.assertEqual(len({s['native_s']['key_ptr'] for s in seen.values()}), 3)
        finally:
            for engine in list(group.engines):
                group.close(engine)
            group.idle()


class ResidentBaselineTests(BaselineRuntimeMixin, unittest.TestCase):
    def test_variable_actual_retrieval_lengths_queries_and_native_kv_are_exact(self):
        for chunks in (1, 2):
            result = self.paired(chunks=chunks)
            first = result[1][0]
            self.assertGreater(len({len(r['selected']) for r in first['rounds']}), 1)
            for r in first['rounds']:
                self.assertEqual(len(r['selected']), next(t['workset_tokens'] for t in first['row']['trace']
                                 if t['round_index'] == r['round_index']))
                self.assertLessEqual(len(r['selected']), chunks*32)

    def test_eos_and_fixed_lengths_keep_original_accept_and_catchup_boundaries(self):
        for k in (1, 2, 4, 8, 12):
            result = self.paired(limit=7, k=k, lengths=(63,))
            eos = result[0][0]['row']['generated'][0]
            self.paired(limit=7, k=k, eos=eos, lengths=(63,))
        result = self.paired(limit=8, k=3, identical=True, lengths=(63,))
        self.assertTrue(all(t['proposed'] == t['accepted'] for t in result[0][0]['row']['trace']))
        self.assertGreater(result[1][0]['row']['draft_forwards'], result[1][0]['row']['proposed'])
        self.assertTrue(any(not q['logits'] for q in result[1][0]['queries']))

    def test_timing_retains_no_unused_query_and_builds_no_mapper_or_probes(self):
        teacher, student = fixture()
        query_states = []
        original = DraftAttention.step

        def step(owner, *args, **kwargs):
            output = original(owner, *args, **kwargs)
            query_states.append(owner.last_query)
            return output

        with BaselineEngine(teacher, student, cfg('resident', limit=7), top_chunks=1) as engine:
            with patch.object(DraftAttention, 'step', new=step):
                row = engine.generate(prompt(63))
            self.assertTrue(query_states and all(q is None for q in query_states))
            self.assertFalse(engine.draft_runtime.retain_query)
            self.assertFalse(row['draft_runtime']['retain_query'])
            self.assertEqual(engine.injector.summaries, [])
            self.assertEqual(row['draft_shadow_probes'], 0)
            self.assertEqual(row['projected_positions'], 0)

    def test_native_eager_buffers_persist_and_unsupported_backends_do_not_fallback(self):
        teacher, student = fixture()
        with BaselineEngine(teacher, student, cfg(limit=7)) as engine:
            first = engine.generate(prompt(63)); key_ptr = engine.native_s.key_storage.data_ptr()
            second = engine.generate(prompt(65))
            self.assertEqual(key_ptr, engine.native_s.key_storage.data_ptr())
            self.assertIsNone(engine.working); self.assertIsNone(engine.injector)
            self.assertIsNone(engine.draft_runtime); self.assertFalse(engine.observation.originals)
            self.assertEqual(first['working_gather_layers'], 0)
            self.assertEqual(second['target_probe_rounds'], 0)
        original = teacher.model.layers[-1].self_attn.forward
        for mode in ('resident', 'cuda_graph'):
            with self.assertRaisesRegex(ValueError, 'eager execution only'):
                BaselineEngine(teacher, student, cfg(mode))
            self.assertIs(teacher.model.layers[-1].self_attn.forward, original)
        with self.assertRaisesRegex(ValueError, 'CUDA device'):
            BaselineEngine(teacher, student, cfg('cuda_graph'), top_chunks=1)

    def test_capacity_growth_rebuilds_owner_and_close_failure_restores_hooks(self):
        teacher, student = fixture()
        engine = BaselineEngine(teacher, student, cfg('resident', limit=7), top_chunks=1)
        original = engine.observation.originals[0][1]
        engine.generate(prompt(20)); first = engine.draft_runtime
        engine.generate(prompt(70))
        self.assertTrue(first.closed)
        self.assertIsNot(engine.draft_runtime, first)
        with patch.object(engine.draft_runtime, 'close', side_effect=RuntimeError('synthetic close failure')):
            with self.assertRaisesRegex(RuntimeError, 'synthetic close failure'):
                engine.close()
        self.assertTrue(engine.closed)
        self.assertFalse(engine.observation.originals)
        self.assertIs(teacher.model.layers[-1].self_attn.forward, original)

    def test_three_persistent_retrieval_owners_share_models_only_through_handoff(self):
        self.shared_owners()


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable; CPU tests make no Graph-speed claim')
class GraphBaselineTests(BaselineRuntimeMixin, unittest.TestCase):
    device = 'cuda'
    mode = 'cuda_graph'

    def test_graph_variable_length_geometry_full_kv_and_queries_match_eager(self):
        result = self.paired(limit=65, k=4, lengths=(63, 65))
        graph = result[1]
        self.assertGreater(graph[0]['row']['draft_runtime']['request']['captures'], 32)
        self.assertGreater(graph[1]['row']['draft_runtime']['request']['replays'], 0)
        self.assertEqual(graph[0]['row']['draft_runtime']['max_graph_entries'], 128)

    def test_graph_fixed_k12_eos_and_all_accept_catchup(self):
        result = self.paired(limit=17, k=12, identical=True, lengths=(63,))
        self.assertTrue(any(not q['logits'] for q in result[1][0]['queries']))
        self.paired(limit=9, k=12, eos=result[0][0]['row']['generated'][0], lengths=(63,))

    def test_graph_retrieval_shared_model_owner_crossover(self):
        self.shared_owners()


if __name__ == '__main__':
    unittest.main()
