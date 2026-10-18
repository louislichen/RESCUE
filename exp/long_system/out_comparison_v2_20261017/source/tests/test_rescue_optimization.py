from dataclasses import replace
from types import SimpleNamespace
import unittest

import torch

from exp.selective_kv.bridge import Injection, ValueMapper
from rescue.cache import NativeCache, WorkingCache
from rescue.config import SystemConfig
from rescue.engine import RescueEngine
from rescue.injection import ContentInjector
from rescue.policy import RiskModel
from rescue.selection import LazyReadState, ReadState
from test_rescue_system import models, COST


class OptimizationTests(unittest.TestCase):
    def test_lazy_state_and_direct_output_gather_match(self):
        a, v = torch.rand(6, 19), torch.randn(2, 19, 4)
        eager, lazy = ReadState(a, v), LazyReadState(a, v)
        self.assertIsNone(lazy._target)
        positions = torch.tensor([0, 2, 6, 12, 18])
        torch.testing.assert_close(lazy.error(positions), eager.error(positions))
        cache = NativeCache(1, 2, 4, 32, torch.float32, 'cpu')
        cache.update(v[None], (v+1)[None], 0)
        working = WorkingCache(cache, 8, 4, direct_gather=True)
        working.bind(cache, positions)
        torch.testing.assert_close(working.keys[0][:, :, :5], cache[0][0][:, :, positions])
        torch.testing.assert_close(working.values[0][:, :, :5], cache[0][1][:, :, positions])

    def test_static_gate_factorization_matches_dynamic_original(self):
        torch.manual_seed(5)
        network = ValueMapper([0], [0], source_dim=8, value_dim=4, query_dim=8, rank=4).eval()
        teacher = NativeCache(1, 2, 4, 24, torch.float32, 'cpu')
        student = NativeCache(1, 1, 4, 24, torch.float32, 'cpu')
        teacher.update(torch.randn(1, 2, 12, 4), torch.randn(1, 2, 12, 4), 0)
        student.update(torch.randn(1, 1, 12, 4), torch.randn(1, 1, 12, 4), 0)
        positions = torch.tensor([0, 2, 4, 6, 10, 11])
        prior = torch.arange(1., 13.)/78
        normal = ContentInjector(network, 24, 'cpu', tokens=2)
        fast = ContentInjector(network, 24, 'cpu', tokens=2, optimized=True)
        for injector in (normal, fast):
            injector.bind(teacher, student, positions, prior, 2)
        for _ in range(4):
            hidden = torch.randn(1, 1, 8)
            native = student[0][1][:, :, positions]
            torch.testing.assert_close(fast.apply(0, native.clone(), hidden),
                                       normal.apply(0, native.clone(), hidden), rtol=1e-6, atol=1e-7)
        fast.clear()
        self.assertFalse(fast.active or fast.gate_static)

    def test_folded_hazard_calibration_matches_logits(self):
        risk = RiskModel().eval()
        risk.temperature = 1.5
        x = torch.randn(4, 31)
        expected = risk(x)*.7
        expected[:, :, 0] += 1.2
        actual = risk.calibrated(.7, 1.2)
        torch.testing.assert_close(actual(x), expected, rtol=1e-5, atol=1e-6)

    def test_optimized_and_legacy_target_outputs_and_eager_trajectory(self):
        teacher, student, mapper = models()
        risk = RiskModel().eval()
        config = SystemConfig(budget=8, recent=3, max_new_tokens=11, attention_backend='eager')
        ids = list(range(3, 35))
        trajectories = []
        for optimized in (False, True):
            with RescueEngine(teacher, student, replace(config, optimized_execution=optimized),
                              mapper, risk, COST) as engine:
                result = engine.generate(ids)
                self.assertEqual(result['generated'], engine.reference(ids)['generated'])
                trajectories.append([(r['cap'], r['accepted'], r['proposed']) for r in result['trace']])
        self.assertEqual(trajectories[0], trajectories[1])


if __name__ == '__main__':
    unittest.main()
