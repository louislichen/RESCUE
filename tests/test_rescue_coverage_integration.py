"""Coverage execution integration; CPU reference is not GPU speed evidence."""
from dataclasses import replace
import unittest

import torch

from rescue.config import SystemConfig
from rescue.coverage_graph import CoverageGraphRuntime
from rescue.engine import RescueEngine
from rescue.selection import ContextSelector, LazyReadState
from rescue.step_budget import compact_attention
from test_rescue_system import models


class CoverageIntegrationCPU(unittest.TestCase):
    def test_opt_in_validation_and_no_silent_cpu_engine_fallback(self):
        base = SystemConfig(selector='hierarchical')
        for updates in ({'coverage_execution': 'resident'},
                        {'coverage_execution': 'cuda_graph', 'selector': 'teacher'},
                        {'coverage_execution': 'cuda_graph', 'schedule': 'async'},
                        {'coverage_execution': 'cuda_graph', 'optimized_execution': False}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                replace(base, **updates)
        graph = replace(base, coverage_execution='cuda_graph', length_policy='fixed')
        with self.assertRaisesRegex(ValueError, 'persistent runtime'):
            ContextSelector(graph)
        teacher, student, mapper = models()
        original = teacher.model.layers[-1].self_attn.forward
        with self.assertRaisesRegex(ValueError, 'requires a CUDA'):
            RescueEngine(teacher, student, graph, mapper)
        self.assertEqual(original, teacher.model.layers[-1].self_attn.forward)

    @torch.no_grad()
    def paired(self, device='cpu', mode='eager', sizes=(126, 129, 132, 135, 138)):
        torch.manual_seed(512)
        for refresh in ('periodic', 'adaptive'):
            for anchor, bonus in ((0., 0.), (.25, .13)):
                cfg = SystemConfig(selector='hierarchical', budget=16, recent=4,
                    coverage_candidates=32, coverage_rows=20, coarse_chunk_size=8,
                    coverage_batch=4, refresh_policy=refresh, refresh_max=3,
                    attention_drift=1., coverage_floor=0., coverage_anchor_fraction=anchor,
                    coverage_attention_weight=bonus)
                base = ContextSelector(cfg)
                runtime = CoverageGraphRuntime(mode=mode)
                candidate = ContextSelector(replace(cfg, coverage_execution='cuda_graph'), runtime=runtime)
                tv, sv = torch.randn(2, max(sizes), 8, device=device), torch.randn(1, max(sizes), 8, device=device)
                try:
                    for i, n in enumerate(sizes):
                        t = LazyReadState(torch.rand(4, n, device=device), tv[:, :n])
                        s = LazyReadState(torch.rand(3, n, device=device), sv[:, :n])
                        a, b = base.select(t, s, i), candidate.select(t, s, i)
                        self.assertTrue(torch.equal(a, b))
                        self.assertTrue(torch.equal(base.previous, candidate.previous))
                        self.assertTrue(torch.equal(base.priority, candidate.priority))
                        self.assertEqual(base.similarity_elements, candidate.similarity_elements)
                        fa = compact_attention(t, s, a, i-base.last_refresh, 3*i, 128-3*i)
                        fb = compact_attention(t, s, b, i-candidate.last_refresh, 3*i, 128-3*i)
                        self.assertTrue(torch.equal(fa, fb))
                    if mode == 'cuda_graph':
                        self.assertEqual(runtime.stats()['replays'], base.refreshes)
                        self.assertGreater(runtime.stats()['captures'], 0)
                finally:
                    runtime.close()
                self.assertTrue(runtime.closed)

    def test_literal_selector_positions_and_controller_features_remain_exact(self):
        self.paired()


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class CoverageIntegrationCUDA(CoverageIntegrationCPU):
    def test_real_graph_selector_refreshes_remain_exact(self):
        self.paired(device='cuda', mode='cuda_graph')

    @torch.no_grad()
    def test_nonzero_mapper_engine_all_kv_queries_and_trajectory_exact(self):
        from rescue.attention import DraftAttention
        from rescue.draft_graph_smoke import native_digest, record_queries
        from rescue.efficient_attention import install_efficient_attention
        from rescue.packing import pack_projections
        from rescue.round_overhead_run import CorrectnessObserver, check_pair
        teacher, student, mapper = models()
        teacher.to('cuda'); student.to(device='cuda', dtype=torch.bfloat16); mapper.to('cuda')
        for model in (teacher, student):
            model.requires_grad_(False)
            install_efficient_attention(model); pack_projections(model)
        base = SystemConfig(selector='hierarchical', budget=8, recent=2,
            coverage_candidates=12, coverage_rows=8, coarse_chunk_size=4, coverage_batch=2,
            length_policy='fixed', fixed_k=4, max_new_tokens=17,
            draft_execution='cuda_graph', inject_tokens=2, injection_strength=.5)
        results = []
        for execution in ('eager', 'cuda_graph'):
            observer = CorrectnessObserver()
            with RescueEngine(teacher, student, replace(base, coverage_execution=execution),
                              mapper, round_observer=observer) as engine:
                rows = []
                for length in (63, 65, 136):
                    ids = [3+i % 40 for i in range(length)]
                    offset = len(observer.rounds)
                    reference = engine.reference(ids)['generated']
                    with record_queries(DraftAttention) as queries:
                        row = engine.generate(ids)
                    self.assertEqual(row['generated'], reference)
                    rows.append(dict(row=row, queries=queries, rounds=observer.rounds[offset:],
                        target=native_digest(engine.native_t), draft=native_digest(engine.native_s)))
                    if execution == 'cuda_graph':
                        self.assertEqual(row['coverage_runtime']['request']['replays'], row['selector_refreshes'])
                runtime = engine.coverage_runtime
            if runtime is not None:
                self.assertTrue(runtime.closed)
            results.append(rows)
        for a, b in zip(*results):
            check_pair(a['row'], b['row'])
            for field in ('queries', 'rounds', 'target', 'draft'):
                self.assertEqual(a[field], b[field], field)


if __name__ == '__main__':
    unittest.main()
