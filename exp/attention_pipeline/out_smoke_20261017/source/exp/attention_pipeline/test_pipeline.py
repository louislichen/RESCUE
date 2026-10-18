"""Message ordering and cached-gate equivalence, without loading base models."""
from concurrent.futures import Future
from threading import Event
from types import SimpleNamespace
import unittest

import torch

from exp.attention_pipeline.cached_injection import CachedInjection
from exp.attention_pipeline.planner import LengthDecision, LengthPlanner, LengthTicket, PlanKey
from exp.selective_kv.bridge import Injection, ValueMapper


class PlanningTests(unittest.TestCase):
    def test_first_token_can_run_while_plan_is_unfinished(self):
        entered, release = Event(), Event()
        order = []
        def compute():
            entered.set()
            if not release.wait(5):
                raise TimeoutError('Test did not release its simulated network')
            order.append('length_ready')
            return 1, .2
        key = PlanKey('request-a', 0, 100, 0)
        with LengthPlanner() as planner:
            ticket = planner.submit(key, 8, compute)
            self.assertTrue(entered.wait(5))
            try:
                order.extend(('mapping', 'first_draft_token'))
                self.assertFalse(ticket.future.done())
                with self.assertRaises(RuntimeError):
                    planner.submit(key, 8, compute)
            finally:
                release.set()
            self.assertEqual(ticket.result(key).cap, 1)
        self.assertEqual(order, ['mapping', 'first_draft_token', 'length_ready'])

    def test_reject_stale_request_round_prefix_or_workset(self):
        key = PlanKey('a', 2, 111, 3)
        future = Future()
        future.set_result(LengthDecision(key, 2, .5, .001))
        ticket = LengthTicket(key, 8, future)
        for wrong in (PlanKey('b', 2, 111, 3), PlanKey('a', 3, 111, 3),
                      PlanKey('a', 2, 112, 3), PlanKey('a', 2, 111, 4)):
            with self.assertRaises(ValueError):
                ticket.result(wrong)
        self.assertEqual(ticket.result(key).cap, 2)

    def test_invalid_caps_and_worker_error_propagate(self):
        key = PlanKey('a', 0, 100, 0)
        with LengthPlanner() as planner:
            for cap in (0, 9, True):
                with self.assertRaises(ValueError):
                    planner.submit(key, 8, lambda cap=cap: (cap, .1)).result(key)
            def failed():
                raise RuntimeError('remote worker failed')
            with self.assertRaisesRegex(RuntimeError, 'remote worker failed'):
                planner.submit(key, 8, failed).result(key)


class CachedGateTests(unittest.TestCase):
    @torch.no_grad()
    def test_matches_original_across_queries_and_new_keys_without_touching_backing(self):
        torch.manual_seed(42)
        network = ValueMapper([0, 1], [0, 1], source_dim=12, value_dim=8, query_dim=16, rank=4).eval()
        positions = torch.tensor([2, 6])
        selected = torch.tensor([0, 2, 5, 6, 8])
        source = torch.randn(2, 2, 12)
        prior = torch.rand(9)
        cache = tuple((torch.randn(1, 2, 9, 4).bfloat16(),
                       torch.randn(1, 2, 9, 4).bfloat16()) for _ in range(2))
        saved = tuple((k.clone(), v.clone()) for k, v in cache)
        original = Injection(SimpleNamespace(kv_transform=None))
        cached = CachedInjection(SimpleNamespace(kv_transform=None))
        original.bind(network, positions, source, prior, 'gated', .5)
        cached.prepare(cache, selected)
        cached.bind(network, positions, source, prior, 'gated', .5)
        for step in range(3):
            visible = torch.cat((selected, torch.arange(9, 10+step)))
            hidden = torch.randn(1, 1, 16)
            for layer in range(2):
                k, v = cache[layer]
                wk = torch.cat((k[:, :, selected], torch.randn(1, 2, step+1, 4).bfloat16()), 2)
                wv = torch.cat((v[:, :, selected], torch.randn(1, 2, step+1, 4).bfloat16()), 2)
                k1, v1 = original(layer, wk, wv, visible, hidden, torch.tensor([[9+step]]))
                k2, v2 = cached(layer, wk, wv, visible, hidden, torch.tensor([[9+step]]))
                self.assertIs(k1, wk)
                self.assertIs(k2, wk)
                torch.testing.assert_close(v1, v2, rtol=0, atol=0)
                for old, new in zip(saved[layer], cache[layer]):
                    self.assertTrue(torch.equal(old, new))
        self.assertEqual(cached.regularizers, [])
        self.assertEqual(cached.gate_stats, [])
        cached.clear()
        self.assertEqual(cached.static, {})
        self.assertIsNone(cached.controller.kv_transform)

    def test_rebind_replaces_stale_state_and_zero_gate_is_identity(self):
        network = ValueMapper([0], [0], source_dim=4, value_dim=4, query_dim=4, rank=2).eval()
        hook = CachedInjection(SimpleNamespace(kv_transform=None))
        cache = ((torch.randn(1, 1, 3, 4), torch.randn(1, 1, 3, 4)),)
        hook.prepare(cache, torch.arange(3))
        hook.bind(network, torch.tensor([1]), torch.randn(1, 1, 4), torch.rand(3), 'gated', .5)
        self.assertTrue(hook.static)
        hook.prepare(cache, torch.arange(3))
        hook.bind(network, torch.tensor([1]), torch.randn(1, 1, 4), torch.rand(3), 'gated', 0.)
        self.assertFalse(hook.static)
        k, v = cache[0]
        out = hook(0, k, v, torch.arange(3), torch.randn(1, 1, 4), torch.tensor([[3]]))
        self.assertIs(out[0], k)
        self.assertIs(out[1], v)


if __name__ == '__main__':
    unittest.main()
