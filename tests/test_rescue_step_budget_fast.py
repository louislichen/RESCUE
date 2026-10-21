"""CPU equivalence checks for cached geometry and cap-only inference.

These checks do not benchmark model throughput or open experiment outputs.
"""
import copy
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from rescue.step_budget import StepBudget, compact_attention, outcome_tables


COST = dict(verify_beta=[.02, .003, .0003, .0001], student_step=.01,
            fixed_round=.004, cache_update=.002)


def legacy_tables(prefix, remaining, cost, device='cpu'):
    """Independent copy of the pre-change arithmetic and class convention."""
    accepts = torch.tensor([0, 1, 2, 3, 4, 1, 2, 3, 4], device=device)
    eos = torch.tensor([False]*5+[True]*4, device=device)
    caps = torch.tensor([min(2, remaining), min(4, remaining)], device=device)
    accepted = torch.minimum(accepts[:, None], caps[None])
    ends_eos = eos[:, None] & (accepts[:, None] <= caps[None])
    committed = torch.minimum(accepted+(~ends_eos).long(), caps.new_tensor(remaining)).float()
    terminal_bonus = eos[:, None] & (accepts[:, None] == caps[None]+1)
    catchup = (accepted == caps[None]) & (~ends_eos) & (~terminal_bonus) & (committed < remaining)
    nf = prefix/2048
    b = cost['verify_beta']
    seconds = (cost.get('fixed_round', 0.)+b[0]+b[1]*nf+
        (b[2]+b[3]*nf+cost['student_step'])*caps.float()[None]+
        catchup.float()*cost['student_step']+cost.get('cache_update', 0.))
    return committed, seconds


@torch.no_grad()
def legacy_decide(model, x, cost, prefix, remaining):
    if remaining <= 2 or model.rng.random()<model.exploration:
        return min(4, remaining), 0.
    probability = model.probabilities(x)
    committed, seconds = legacy_tables(prefix, remaining, cost, x.device)
    rate = (probability@committed)/(probability@seconds).clamp_min(1e-6)
    cap = 2 if bool(rate[0] > rate[1]*(1+model.margin)) else min(4, remaining)
    return cap, float((rate[0]/rate[1]-1).item())


class StepBudgetFastTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(215)
        self.policy = StepBudget().eval()

    def test_cached_and_public_tables_exactly_match_legacy_all_boundaries(self):
        costs = [COST, dict(verify_beta=[0., 0., 0., 0.], student_step=0.),
                 dict(verify_beta=[.3, -.001, .02, .0002], student_step=.007,
                      fixed_round=.03, cache_update=.005)]
        for remaining in (*range(1, 10), 128, 4096):
            for prefix in (0, 1, 1000, 8192, 16384):
                for cost in costs:
                    with self.subTest(remaining=remaining, prefix=prefix, cost=cost):
                        expected = legacy_tables(prefix, remaining, cost)
                        for actual in (outcome_tables(prefix, remaining, cost),
                                       self.policy._cached_outcome_tables(prefix, remaining, cost)):
                            for a, b in zip(actual, expected):
                                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_remaining_five_and_six_keep_distinct_catchup_geometry(self):
        cost = dict(verify_beta=[0., 0., 0., 0.], student_step=1.)
        c5, t5 = self.policy._cached_outcome_tables(1000, 5, cost)
        c6, t6 = self.policy._cached_outcome_tables(1000, 6, cost)
        self.assertEqual(float(c5[4, 1]), 5.)
        self.assertEqual(float(c6[4, 1]), 5.)
        self.assertEqual(float(t5[4, 1]), 4.)
        self.assertEqual(float(t6[4, 1]), 5.)
        # Accepted EOS3 is the cap2 bonus; neither case runs catchup.
        self.assertEqual(float(t5[7, 0]), 2.)
        self.assertEqual(float(t6[7, 0]), 2.)

    def test_caps_scores_and_rng_match_old_rule_including_exploration(self):
        for exploration in (0., .37, 1.):
            policy = StepBudget(exploration=exploration, seed=37).eval()
            diagnostic, reference = copy.deepcopy(policy), copy.deepcopy(policy)
            for index, remaining in enumerate((1, 2, 3, 4, 5, 6, 7, 128)*3):
                x = torch.arange(12).float()/10+index/100
                prefix = 1000+index
                expected = legacy_decide(reference, x, COST, prefix, remaining)
                self.assertEqual(policy.decide_cap(x, COST, prefix, remaining), expected[0])
                self.assertEqual(diagnostic.decide(x, COST, prefix, remaining), expected)
                self.assertEqual(policy.rng.getstate(), reference.rng.getstate())
                self.assertEqual(diagnostic.rng.getstate(), reference.rng.getstate())

    def test_short_tail_does_not_run_network_or_consume_rng(self):
        state = self.policy.rng.getstate()
        with mock.patch.object(self.policy, 'probabilities', side_effect=AssertionError('unexpected network')):
            for remaining in (1, 2):
                self.assertEqual(self.policy.decide(torch.zeros(12), COST, 1000, remaining), (remaining, 0.))
                self.assertEqual(self.policy.decide_cap(torch.zeros(12), COST, 1000, remaining), remaining)
        self.assertEqual(self.policy.rng.getstate(), state)

    def test_cap_only_omits_python_score_item_and_runtime_table_construction(self):
        x = torch.zeros(12)
        with mock.patch.object(torch.Tensor, 'item', side_effect=AssertionError('unused score read')):
            self.assertIn(self.policy.decide_cap(x, COST, 1000, 128), (2, 4))
            with self.assertRaisesRegex(AssertionError, 'unused score read'):
                self.policy.decide(x, COST, 1000, 128)
        with mock.patch('torch.tensor', side_effect=AssertionError('runtime class construction')), \
             mock.patch.object(torch.Tensor, 'new_tensor', side_effect=AssertionError('runtime boundary construction')):
            self.assertIn(self.policy.decide_cap(x, COST, 1000, 128), (2, 4))

    def test_strict_threshold_tie_and_margin_are_not_redefined(self):
        probability = torch.zeros(9)
        probability[4] = 1.
        cost = dict(verify_beta=[0., 0., 0., 0.], student_step=1.)
        with mock.patch.object(self.policy, 'probabilities', return_value=probability):
            self.policy.margin = 0.
            # Both rates are exactly one; the historical strict > keeps cap4.
            self.assertEqual(self.policy.decide(torch.zeros(12), cost, 1000, 6), (4, 0.))
            self.assertEqual(self.policy.decide_cap(torch.zeros(12), cost, 1000, 6), 4)
            self.policy.margin = -.001
            self.assertEqual(self.policy.decide_cap(torch.zeros(12), cost, 1000, 6), 2)
            self.policy.margin = .001
            self.assertEqual(self.policy.decide_cap(torch.zeros(12), cost, 1000, 6), 4)

    def test_cost_is_not_frozen_in_geometry_cache(self):
        cost = copy.deepcopy(COST)
        for student_step in (.01, .4, 0.):
            cost['student_step'] = student_step
            for a, b in zip(self.policy._cached_outcome_tables(8192, 128, cost),
                            legacy_tables(8192, 128, cost)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_cap_only_preserves_subclass_legacy_decide_override(self):
        class LegacyPolicy(StepBudget):
            def decide(self, x, cost, prefix, remaining):
                return min(3, remaining), 17.

        policy = LegacyPolicy()
        for remaining in (1, 2, 3, 6, 128):
            self.assertEqual(policy.decide_cap(torch.zeros(12), COST, 1000, remaining),
                             policy.decide(torch.zeros(12), COST, 1000, remaining)[0])

    def test_subclass_delegation_does_not_recurse_or_draw_rng_twice(self):
        class SuperDiagnostic(StepBudget):
            def decide(self, x, cost, prefix, remaining):
                return super().decide(x, cost, prefix, remaining)

        class SuperCap(StepBudget):
            def decide(self, x, cost, prefix, remaining):
                return super().decide_cap(x, cost, prefix, remaining), 17.

        for kind in (SuperDiagnostic, SuperCap):
            with self.subTest(kind=kind.__name__):
                policy = kind(exploration=.37, seed=37)
                reference = copy.deepcopy(policy)
                for remaining in (1, 2, 3, 4, 5, 6, 128):
                    expected = legacy_decide(reference, torch.zeros(12), COST, 1000, remaining)[0]
                    self.assertEqual(policy.decide_cap(torch.zeros(12), COST, 1000, remaining), expected)
                    self.assertEqual(policy.rng.getstate(), reference.rng.getstate())

    def test_legacy_state_dict_and_payload_keys_remain_strictly_loadable(self):
        expected = {'mean', 'scale', 'prior', 'net.0.weight', 'net.0.bias',
                    'net.2.weight', 'net.2.bias'}
        self.assertEqual(set(self.policy.state_dict()), expected)
        payload = self.policy.payload()
        self.assertEqual(set(payload), {'kind', 'hidden', 'prior_weight', 'margin', 'state_dict', 'seed'})
        loaded = StepBudget.from_payload(payload, 'cpu')
        loaded.load_state_dict(copy.deepcopy(payload['state_dict']), strict=True)
        for remaining in (3, 4, 5, 6, 128):
            self.assertEqual(self.policy.decide(torch.zeros(12), COST, 1000, remaining),
                             loaded.decide(torch.zeros(12), COST, 1000, remaining))
        buffers = dict(loaded.named_buffers())
        self.assertTrue(all(buffers[n].device == loaded.mean.device for n in
                            ('_outcome_caps', '_outcome_committed', '_outcome_catchup')))
        # Device/dtype moves retain exact small integers; geometry calculations
        # return the same float32 tables as before.
        loaded.to(dtype=torch.float64)
        for a, b in zip(loaded._cached_outcome_tables(8192, 128, COST), legacy_tables(8192, 128, COST)):
            torch.testing.assert_close(a, b, rtol=0, atol=0, check_dtype=True)

    def test_compact_attention_uses_shared_means_and_legacy_state_fallback(self):
        t = SimpleNamespace(a=torch.rand(3, 13), n=13)
        s = SimpleNamespace(a=torch.rand(3, 13), n=13)
        selected = torch.tensor([0, 3, 8, 12])
        expected = compact_attention(t, s, selected, 2, 7, 121, .6)
        t.mean_attention, s.mean_attention = t.a.mean(0), s.a.mean(0)
        # Shared results, rather than another read of a, provide the means.
        t.a, s.a = None, None
        actual = compact_attention(t, s, selected, 2, 7, 121, .6)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
