import json
from dataclasses import replace
from pathlib import Path
import socket
import struct
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from exp.attention_horizon.selection import ReadState, feature_vector
from exp.selective_kv.bridge import Injection, ValueMapper
from rescue.cache import NativeCache, WorkingCache
from rescue.config import SystemConfig
from rescue.engine import RescueEngine
from rescue import features
from rescue.injection import ContentInjector
from rescue.messages import LengthPacket, ProtocolError, RoundKey
from rescue.policy import RiskModel, choose
from rescue.scheduler import Scheduler
from rescue.selection import ContextSelector, mandatory
from rescue.transport import LengthClient, LengthServer, MAX_MESSAGE, receive

torch.set_num_threads(1)


def models():
    torch.manual_seed(42)
    t = Qwen2ForCausalLM(Qwen2Config(vocab_size=48, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)).eval()
    s = Qwen2ForCausalLM(Qwen2Config(vocab_size=48, hidden_size=24, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=3, num_key_value_heads=1, max_position_embeddings=256)).eval()
    mapper = ValueMapper([0, 1], [1, 2], source_dim=16, value_dim=8, query_dim=24, rank=4).eval()
    return t, s, mapper


COST = dict(verify_beta=[.004, .001, .0001, .00001], student_step=.001, cache_update=.0001)


class CacheTests(unittest.TestCase):
    def test_native_append_rollback_overwrite_and_capacity(self):
        cache = NativeCache(2, 1, 4, 8, torch.float32, 'cpu')
        first = torch.randn(1, 1, 3, 4)
        for layer in range(2):
            cache.update(first, first+1, layer)
        saved = cache[0][0].clone()
        for layer in range(2):
            cache.update(torch.zeros(1, 1, 2, 4), torch.zeros(1, 1, 2, 4), layer)
        cache.crop(3)
        for layer in range(2):
            cache.update(torch.ones(1, 1, 1, 4), torch.ones(1, 1, 1, 4), layer)
        self.assertTrue(torch.equal(cache[0][0][:, :, :3], saved))
        self.assertTrue(torch.equal(cache[0][0][:, :, 3], torch.ones(1, 1, 4)))
        with self.assertRaises(ValueError):
            cache.get_usable_length(5)
        with self.assertRaises(ValueError):
            cache.crop(5)

    def test_working_append_is_private_and_gathers_only_on_bind(self):
        native = NativeCache(1, 1, 4, 16, torch.float32, 'cpu')
        native.update(torch.randn(1, 1, 7, 4), torch.randn(1, 1, 7, 4), 0)
        old = native[0][1].clone()
        working = WorkingCache(native, 4, 3)
        working.bind(native, torch.tensor([0, 2, 5, 6]))
        for _ in range(3):
            _, v = working.append(0, torch.zeros(1, 1, 1, 4), torch.ones(1, 1, 1, 4))
            v[:, :, 1] = 100
        self.assertEqual(working.gathered_layers, 1)
        self.assertTrue(torch.equal(native[0][1], old))
        working.invalidate()
        with self.assertRaises(ValueError):
            working.append(0, torch.zeros(1, 1, 1, 4), torch.ones(1, 1, 1, 4))


class FeatureSelectionTests(unittest.TestCase):
    def test_split_matches_frozen_feature_contract_with_gqa(self):
        torch.manual_seed(4)
        t = ReadState(torch.rand(6, 51), torch.randn(2, 51, 4))
        s = ReadState(torch.rand(4, 51), torch.randn(1, 51, 3))
        selected = torch.tensor([0, 2, 8, 23, 48, 49, 50])
        actual = features.finish(features.prepare(t, s, 8, 24, 3), t, s, selected)
        expected = feature_vector(t, s, selected, 8, 0, torch.zeros(3), 24, 3)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_coverage_avoids_redundant_high_attention_tokens(self):
        values = torch.tensor([[[0., 0.], [1., 0.], [1., 0.], [0., 1.], [0., .5], [0., 0.]]])
        a = torch.tensor([[.01, .3, .29, .2, .19, .01]])
        state = ReadState(a, values)
        c = SystemConfig(budget=4, recent=1, coverage_batch=1, refresh_every=1)
        selected = ContextSelector(c).select(state, state, 0).tolist()
        self.assertIn(0, selected)
        self.assertIn(5, selected)
        self.assertEqual(sum(i in selected for i in (1, 2)), 1)
        self.assertEqual(sum(i in selected for i in (3, 4)), 1)

    def test_carry_and_chunk_modes_keep_strict_token_budget(self):
        c = SystemConfig(budget=8, recent=3, selector='joint', chunk_size=4, refresh_every=4)
        selector = ContextSelector(c)
        for round_index, n in enumerate((20, 22, 25, 27, 29)):
            t = ReadState(torch.rand(4, n), torch.randn(2, n, 4))
            s = ReadState(torch.rand(2, n), torch.randn(1, n, 4))
            positions = selector.select(t, s, round_index)
            self.assertEqual(len(positions), 8)
            self.assertTrue(bool(torch.isin(mandatory(n, 3, 'cpu'), positions).all()))
            self.assertEqual(len(positions.unique()), 8)
        self.assertEqual(selector.refreshes, 2)

    def test_chunk_coverage_preserves_budget_and_anchors(self):
        cfg = SystemConfig(budget=10, recent=3, selector='coverage', chunk_size=4, refresh_every=1)
        state = ReadState(torch.rand(4, 31), torch.randn(2, 31, 4))
        positions = ContextSelector(cfg).select(state, state, 0)
        self.assertEqual(len(positions), 10)
        self.assertTrue(bool(torch.isin(mandatory(31, 3, 'cpu'), positions).all()))

    def test_vectorized_cost_choice_matches_scalar_expected_tokens(self):
        model = RiskModel().eval()
        x = torch.randn(31)
        for remaining in (1, 2, 5, 12, 20):
            cap, score = choose(model, x, COST, 1300, remaining)
            survival = model.continuation(x).clamp(0, 1)
            scores = {}
            for k in sorted({min(v, remaining) for v in (1, 2, 4, 8, 12)}):
                expected = 1+survival[:min(k, remaining-1)].sum()
                cost = (COST['student_step']*k+COST['cache_update']+
                    x.new_tensor(COST['verify_beta'])@x.new_tensor([1., 1300/2048, k, k*1300/2048])+
                    (survival[k-1]*COST['student_step'] if k < remaining else 0.))
                scores[k] = float(expected/cost)
            self.assertEqual(cap, max(scores, key=scores.get))
            self.assertAlmostEqual(score, scores[cap], places=4)


class InjectionTests(unittest.TestCase):
    @torch.no_grad()
    def test_inplace_matches_original_and_memo_reuses_committed_content(self):
        torch.manual_seed(8)
        mapper = ValueMapper([0], [0], source_dim=8, value_dim=4, query_dim=8, rank=4).eval()
        tc = NativeCache(1, 2, 4, 32, torch.float32, 'cpu')
        sc = NativeCache(1, 1, 4, 32, torch.float32, 'cpu')
        tc.update(torch.randn(1, 2, 12, 4), torch.randn(1, 2, 12, 4), 0)
        sc.update(torch.randn(1, 1, 12, 4), torch.randn(1, 1, 12, 4), 0)
        saved = sc[0][1].clone()
        selected = torch.tensor([0, 2, 4, 6, 10, 11])
        prior = torch.arange(1., 13.)/78
        injector = ContentInjector(mapper, 32, 'cpu', tokens=2)
        injector.bind(tc, sc, selected, prior, recent=2)
        positions = torch.tensor([4, 6])
        source = tc[0][1][0, :, positions].transpose(0, 1).flatten(1).to(torch.bfloat16).float().unsqueeze(0)
        original = Injection(SimpleNamespace(kv_transform=None))
        original.bind(mapper, positions, source, prior, 'gated', .5)
        for _ in range(3):
            hidden = torch.randn(1, 1, 8)
            wk, wv = sc[0][0][:, :, selected], sc[0][1][:, :, selected]
            expected = original(0, wk, wv.clone(), selected, hidden, torch.tensor([[12]]))[1]
            actual = injector.apply(0, wv.clone(), hidden)
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        injector.bind(tc, sc, selected, prior, recent=2)
        self.assertEqual(injector.projected_positions, 2)
        self.assertTrue(torch.equal(saved, sc[0][1]))


class TransportTests(unittest.TestCase):
    def test_stale_and_nonfinite_lengths_are_rejected(self):
        key = RoundKey('a', 1, 10, 1)
        for packet in (LengthPacket(RoundKey('b', 1, 10, 1), 1, .1),
                       LengthPacket(key, 0, .1), LengthPacket(key, 2, float('nan')),
                       LengthPacket(key, 2, 'not a score')):
            with self.assertRaises(ProtocolError):
                packet.validate(key, 8)

    def test_real_tcp_policy_roundtrip(self):
        model = RiskModel().eval()
        server = LengthServer(('127.0.0.1', 0), model)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = LengthClient(server.server_address)
        try:
            for round_index in range(2):
                key = RoundKey('tcp-test', round_index, 1200+round_index, round_index)
                x = torch.randn(31)
                packet = client.choose(key, x, COST, 16, 12)
                self.assertEqual(packet.cap, choose(model, x, COST, key.prefix, 16)[0])
            self.assertEqual(client.calls, 2)
            self.assertGreater(client.bytes_sent, 0)
        finally:
            client.close()
            server.shutdown()
            server.server_close()
            thread.join(5)

    def test_oversize_frame_is_rejected_before_allocation(self):
        a, b = socket.socketpair()
        try:
            a.sendall(struct.pack('!I', MAX_MESSAGE+1))
            with self.assertRaises(ProtocolError):
                receive(b)
        finally:
            a.close()
            b.close()

    def test_rpc_failure_uses_explicit_safe_length(self):
        cfg = SystemConfig(length_endpoint=('127.0.0.1', 12345), fallback_k=1)
        scheduler = Scheduler(cfg, None, COST, 'cpu')
        key = RoundKey('a', 0, 100, 1)
        try:
            with patch.object(scheduler.client, 'choose', side_effect=TimeoutError('late')):
                packet = scheduler._decision(key, torch.zeros(31), 8)
            self.assertEqual(packet.cap, 1)
            self.assertEqual(packet.fallback, 'TimeoutError')
        finally:
            scheduler.close()


class EngineTests(unittest.TestCase):
    def test_chunked_prefill_matches_full_teacher_output(self):
        t, s, mapper = models()
        cfg = SystemConfig(budget=8, recent=3, selector='joint', length_policy='fixed', max_new_tokens=7)
        ids = list(range(3, 30))
        with RescueEngine(t, s, cfg, mapper) as engine:
            reference = engine.reference(ids)
        with RescueEngine(t, s, replace(cfg, prefill_chunk_size=5), mapper) as engine:
            self.assertEqual(engine.generate(ids)['generated'], reference['generated'])

    def test_concurrent_use_is_rejected_without_corrupting_request(self):
        t, s, mapper = models()
        cfg = SystemConfig(budget=8, recent=3, selector='joint', length_policy='fixed', max_new_tokens=1)
        with RescueEngine(t, s, cfg, mapper) as engine:
            engine._request_lock.acquire()
            try:
                with self.assertRaisesRegex(RuntimeError, 'active request'):
                    engine.generate(list(range(3, 22)))
                with self.assertRaisesRegex(RuntimeError, 'active request'):
                    engine.close()
            finally:
                engine._request_lock.release()
            self.assertEqual(engine.generate(list(range(3, 22)))['tokens'], 1)

    def test_full_workflow_and_serial_async_trajectory(self):
        t, s, mapper = models()
        risk = RiskModel().eval()
        ids = list(range(3, 22))
        baseline = None
        for schedule in ('serial', 'async'):
            cfg = SystemConfig(budget=8, recent=3, selector='joint', attention_backend='eager',
                               max_new_tokens=9, schedule=schedule)
            with RescueEngine(t, s, cfg, mapper, risk, COST) as engine:
                reference = engine.reference(ids)
                result = engine.generate(ids)
                self.assertEqual(result['generated'], reference['generated'])
                self.assertEqual(result['working_gather_layers'], result['blocks']*2)
                trajectory = [(b['cap'], b['accepted'], b['proposed']) for b in result['trace']]
                if baseline is None:
                    baseline = trajectory
                else:
                    self.assertEqual(trajectory, baseline)
                again = engine.generate(ids)
                self.assertNotEqual(again['request_id'], result['request_id'])
                self.assertEqual(again['generated'], result['generated'])

    def test_real_model_coverage_and_sdpa_path(self):
        t, s, mapper = models()
        cfg = SystemConfig(budget=8, recent=3, selector='coverage', coverage_batch=2,
                           max_new_tokens=7, length_policy='fixed', fixed_k=4)
        with RescueEngine(t, s, cfg, mapper) as engine:
            ids = list(range(3, 22))
            self.assertEqual(engine.generate(ids)['generated'], engine.reference(ids)['generated'])

    def test_eos_and_output_budget_omit_terminal_catchup(self):
        t, s, mapper = models()
        t.lm_head.weight.data.zero_()
        s.lm_head.weight.data.zero_()
        for eos, budget in ((0, 8), (None, 1)):
            cfg = SystemConfig(budget=8, recent=3, selector='full', max_new_tokens=budget,
                               length_policy='fixed', fixed_k=4)
            with RescueEngine(t, s, cfg, mapper) as engine:
                result = engine.generate(list(range(3, 22)), eos)
                self.assertEqual(result['generated'], [0])
                self.assertEqual(result['draft_forwards'], 1)

    def test_failure_unblocks_pending_planner_and_restores_model(self):
        t, s, mapper = models()
        cfg = SystemConfig(budget=8, recent=3, selector='joint', schedule='async', max_new_tokens=2)
        with RescueEngine(t, s, cfg, mapper, RiskModel().eval(), COST) as engine:
            with patch.object(ContextSelector, 'select', side_effect=ValueError('selection failed')):
                with self.assertRaisesRegex(ValueError, 'selection failed'):
                    engine.generate(list(range(3, 22)))
            self.assertEqual(engine.generate(list(range(3, 22)))['tokens'], 2)


if __name__ == '__main__':
    unittest.main()
