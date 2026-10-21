"""CPU trajectory regressions against the original uncached execution rules."""
import copy
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import torch

from exp.attention_horizon.selection import ReadState as LegacyReadState
from rescue.config import SystemConfig
from rescue.attention_summary import mean_attention
from rescue.engine import RescueEngine
from rescue import features
from rescue.policy import RiskModel
from rescue.query_mapper import QueryValueMapper
from rescue.selection import LazyReadState
from rescue.step_budget import StepBudget, compact_attention, outcome_tables
from test_rescue_system import COST, models


class UncachedLazyReadState(LazyReadState):
    def __init__(self, rows, values):
        super().__init__(rows, values)
        del self.mean_attention


class LegacyStepBudget(StepBudget):
    """Original online arithmetic, including its unused diagnostic host read."""
    decide_cap = None

    @torch.no_grad()
    def decide(self, x, cost, prefix, remaining):
        if remaining <= 2 or self.rng.random() < self.exploration:
            return min(4, remaining), 0.
        probability = self.probabilities(x)
        committed, seconds = outcome_tables(prefix, remaining, cost, x.device)
        rate = (probability@committed)/(probability@seconds).clamp_min(1e-6)
        cap = 2 if bool(rate[0] > rate[1]*(1+self.margin)) else min(4, remaining)
        return cap, float((rate[0]/rate[1]-1).item())


class RoundOverheadTests(unittest.TestCase):
    def run_pair(self, cfg, policy, legacy_policy):
        teacher, student, _ = models()
        mapper = QueryValueMapper(layers=[1], teacher_layers=[2], source_dim=16,
            value_dim=8, query_dim=24, rank=4, residual_rank=3).eval()
        with torch.no_grad():
            mapper.residuals[0][-1].weight.normal_(0, .1)
        frozen = {name: value.clone() for name, value in mapper.state_dict().items()}
        ids = list(range(3, 22))
        histories = []
        for cached, deployed in ((False, legacy_policy), (True, policy)):
            rounds = []

            def observe(row):
                rounds.append(dict(selected=row['selected'].clone(),
                    features=None if row['budget_features'] is None else row['budget_features'].clone(),
                    draft=list(row['draft']), checks=list(row['checks'])))

            with ExitStack() as stack:
                if not cached:
                    stack.enter_context(patch('rescue.engine.ReadState', LegacyReadState))
                    stack.enter_context(patch('rescue.engine.LazyReadState', UncachedLazyReadState))
                elif getattr(deployed, 'stepwise', False):
                    stack.enter_context(patch.object(deployed, 'decide',
                        side_effect=AssertionError('Online engine requested the unused score')))
                engine = stack.enter_context(RescueEngine(teacher, student, cfg, mapper,
                    deployed, COST, round_observer=observe))
                reference = engine.reference(ids)['generated']
                runs = []
                for prompt in (ids, ids[:-2]):
                    offset = len(rounds)
                    result = engine.generate(prompt)
                    self.assertEqual(result['generated'], engine.reference(prompt)['generated'])
                    self.assertGreater(result['projected_positions'], 0)
                    runs.append((result, rounds[offset:]))
                self.assertEqual(runs[0][0]['generated'], reference)
            histories.append(runs)

        for old_run, new_run in zip(*histories):
            old, old_rounds = old_run
            new, new_rounds = new_run
            for key in ('generated', 'accepted', 'proposed', 'blocks', 'draft_forwards',
                        'selector_refreshes', 'projected_positions', 'static_affine_positions'):
                self.assertEqual(old[key], new[key], key)
            trace_keys = ('prefix', 'cap', 'proposed', 'accepted', 'terminal', 'workset_tokens')
            self.assertEqual([tuple(r[k] for k in trace_keys) for r in old['trace']],
                             [tuple(r[k] for k in trace_keys) for r in new['trace']])
            self.assertEqual(len(old_rounds), len(new_rounds))
            for before, after in zip(old_rounds, new_rounds):
                self.assertTrue(torch.equal(before['selected'], after['selected']))
                self.assertEqual(before['draft'], after['draft'])
                self.assertEqual(before['checks'], after['checks'])
                if before['features'] is not None:
                    self.assertTrue(torch.equal(before['features'], after['features']))
                    # Actual nonzero correction summaries feed the controller.
                    self.assertGreater(float(after['features'][-1]), 0.)
        for name, value in mapper.state_dict().items():
            self.assertTrue(torch.equal(value, frozen[name]), name)

    def test_stepwise_nonzero_mapper_matches_legacy_caps_and_trajectories(self):
        for optimized in (False, True):
            for selector in ('joint', 'coverage', 'hierarchical'):
                for outcome in (0, 4):
                    with self.subTest(optimized=optimized, selector=selector, outcome=outcome):
                        policy = StepBudget(prior_weight=0., margin=.05)
                        with torch.no_grad():
                            for parameter in policy.parameters():
                                parameter.zero_()
                            policy.net[-1].bias[outcome] = 20.
                        legacy = LegacyStepBudget(prior_weight=0., margin=.05)
                        legacy.load_state_dict(policy.state_dict())
                        cfg = SystemConfig(budget=8, recent=2, selector=selector,
                            coverage_batch=2, coverage_rows=4, coarse_chunk_size=4,
                            inject_tokens=2, injection_layers=(1,), max_new_tokens=17,
                            attention_backend='eager', optimized_execution=optimized)
                        self.run_pair(cfg, policy, legacy)

    def test_serial_and_async_feature_policies_keep_nonzero_mapper_trajectories(self):
        policy = RiskModel().eval()
        for schedule in ('serial', 'async'):
            with self.subTest(schedule=schedule):
                cfg = SystemConfig(budget=8, recent=2, selector='hierarchical',
                    coverage_rows=4, coarse_chunk_size=4, inject_tokens=2,
                    injection_layers=(1,), max_new_tokens=11, schedule=schedule,
                    attention_backend='eager')
                self.run_pair(cfg, copy.deepcopy(policy), copy.deepcopy(policy))

    def test_fixed_length_keeps_nonzero_mapper_trajectories(self):
        cfg = SystemConfig(budget=8, recent=2, selector='teacher', inject_tokens=2,
            injection_layers=(1,), max_new_tokens=7, length_policy='fixed')
        self.run_pair(cfg, None, None)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA not available; no GPU validation claimed')
class CudaRoundOverheadTests(unittest.TestCase):
    def test_cached_outcome_tables_follow_policy_to_device(self):
        policy = StepBudget().to('cuda').eval()
        for remaining in (*range(1, 10), 128):
            expected = outcome_tables(8192, remaining, COST, policy.mean.device)
            actual = policy._cached_outcome_tables(8192, remaining, COST)
            for before, after in zip(expected, actual):
                self.assertTrue(torch.equal(before, after))
            self.assertEqual(policy.decide(torch.zeros(12, device='cuda'), COST, 8192, remaining)[0],
                             policy.decide_cap(torch.zeros(12, device='cuda'), COST, 8192, remaining))

    def test_eager_mean_is_ready_for_planner_stream(self):
        # Mirror Scheduler.begin's existing event ordering, without new probes
        # or GPU model execution: construct on the producer before recording it.
        producer, planner = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(producer):
            t = LazyReadState(torch.rand(4, 53, device='cuda'), torch.randn(2, 53, 4, device='cuda'))
            s = LazyReadState(torch.rand(3, 53, device='cuda'), torch.randn(1, 53, 8, device='cuda'))
            selected = torch.tensor([0, 3, 8, 30, 51, 52], device='cuda')
            expected_t, expected_s = t.a.mean(0), s.a.mean(0)
            expected = features.prepare(t, s, 3, 17, 2)[0]
            ready = torch.cuda.Event()
            ready.record(producer)
        with torch.cuda.stream(planner):
            planner.wait_event(ready)
            actual_t, actual_s = mean_attention(t).clone(), mean_attention(s).clone()
            actual = features.prepare(t, s, 3, 17, 2)[0]
            compact = compact_attention(t, s, selected, 0, 3, 17)
        planner.synchronize()
        self.assertTrue(torch.equal(actual_t, expected_t))
        self.assertTrue(torch.equal(actual_s, expected_s))
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.isfinite(compact).all())


if __name__ == '__main__':
    unittest.main()
