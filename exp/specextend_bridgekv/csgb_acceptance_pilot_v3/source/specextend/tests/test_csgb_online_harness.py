"""No torch/numpy/GPU required: execute the new Harness against a fake engine."""

import copy
import math
import unittest
from types import SimpleNamespace

from csgb_online_harness import OnlineHarness


class Clock:
    def __init__(self):
        self.value = 0.
        self.sync_calls = 0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds

    def sync(self):
        self.sync_calls += 1
        self.advance(.02)


class IDs:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (1, len(self.values))

    def __getitem__(self, key):
        return SimpleNamespace(tolist=lambda: self.values[key[1]])


class Learner:
    def __init__(self, clock):
        self.clock = clock
        self.choices = []
        self.updates = []

    def choose(self, x, explore=False):
        self.choices.append((copy.deepcopy(x), explore, self.clock()))
        self.clock.advance(.1)
        return 16

    def transform(self, x):
        self.clock.advance(.01)
        return [x[0]] + [v * 2 for v in x[1:]]

    def update(self, x, budget, useful_n, seconds, rho):
        self.updates.append((copy.deepcopy(x), budget, useful_n, seconds, rho, self.clock()))
        self.clock.advance(1.)


class Model:
    def __init__(self, clock):
        self.clock = clock
        self.selected_chunks = None
        self.total_seq_len = 0
        self.attn_scores_final = None
        self.csgb_attention_observed_timestep = None
        self.retrieval_condition = False
        self.drafted = []
        self.actual_node_offset = 0

    def draft(self, input_ids, nodes, threshold, max_depth):
        self.drafted.append(nodes)
        self.clock.advance(.2)
        self.total_seq_len = input_ids.shape[-1]
        self.update_working_cache_retrieval(do_retrieval=self.retrieval_condition)
        return (SimpleNamespace(shape=(1, nodes + self.actual_node_offset)), None, None, None)

    def update_working_cache_retrieval(self, top_k_chunks=32, do_retrieval=False):
        if self.selected_chunks is None:
            self.selected_chunks = [(0, 0, 2)]
        if do_retrieval:
            self.selected_chunks = [(1, 2, 4)]
            self.attn_scores_final = None
            self.retrieval_condition = False
        return "working-cache"


class HarnessTests(unittest.TestCase):
    def make(self, policy="csgb", *, clean=False, explore=True, limit=128, eos=None,
             feature_fn=None, calibration=False):
        self.clock = Clock()
        self.model = Model(self.clock)
        self.copies, self.feature_calls = [], []

        def to_host(a):
            self.copies.append(id(a))
            self.clock.advance(.03)
            return list(a)

        def feature(a, readable, anchor):
            self.feature_calls.append((list(a), list(readable), copy.deepcopy(anchor)))
            self.clock.advance(.04)
            if anchor is None:
                return None
            c, d = sum(a[i] for i in readable)/sum(a), .25
            return [1., c, d, c*d]

        self.h = OnlineHarness(self.model, clock=self.clock, synchronize=self.clock.sync,
                               attention_to_host=to_host, feature_fn=feature_fn or feature)
        self.learner = Learner(self.clock)
        fixed = policy.startswith("fixed")
        self.h.reset(policy, None if fixed else self.learner,
                     explore=explore and not fixed and not clean, clean=clean,
                     rho=2., context=4, max_new_tokens=limit, eos_token_id=eos,
                     calibration=calibration, provenance={"document": 0})
        return self.h

    def proposal(self, length):
        return self.model.draft(IDs(range(length)), 999, .7, 10)

    def observed(self, committed_length, timestep, *, refresh=False):
        self.model.attn_scores_final = [1.] * committed_length
        self.model.csgb_attention_observed_timestep = timestep
        self.model.retrieval_condition = refresh
        self.clock.advance(.3)  # verification/commit before the next draft hook

    def ready(self):
        self.proposal(5)
        self.observed(6, 0, refresh=True)
        self.proposal(7)
        self.observed(8, 1)
        self.proposal(9)

    def test_initial_and_real_anchor_root_geometry_and_copy_reuse(self):
        self.make()
        self.ready()
        self.assertEqual(self.model.drafted, [8, 8, 16])
        self.assertEqual(len(self.copies), 2)  # no second D2H for the refresh anchor
        self.assertIsNone(self.h.rows[0]["raw_x"])
        self.assertIsNone(self.h.rows[1]["raw_x"])
        self.assertFalse(self.h.rows[1]["has_real_anchor"])
        self.assertTrue(self.h.pending["has_real_anchor"])
        self.assertEqual(self.h.pending["anchor_timestep"], 0)
        self.assertEqual(self.h.pending["attention_observed_timestep"], 1)
        self.assertEqual(self.feature_calls[-1][1], [2, 3, 7])
        self.assertEqual(self.h.pending["raw_x"], [1., 3/8, .25, 3/32])
        self.assertEqual(self.h.pending["learner_input"], [1., 3/4, .5, 3/16])
        self.assertTrue(self.model.csgb_attention_every_round)
        self.assertTrue(self.model.csgb_skip_terminal_draft)
        self.assertTrue(self.model.csgb_root_coverage)
        self.assertEqual(self.learner.updates, [])

    def test_round_start_excludes_prior_update_and_executed_arm_only(self):
        self.make()
        self.ready()
        round_start = self.h.pending["start"]
        self.observed(10, 2)
        self.proposal(11)
        previous = self.h.rows[2]
        self.assertEqual(previous["raw_committed"], 2)
        self.assertEqual(previous["useful_committed"], 2)
        self.assertAlmostEqual(previous["seconds"], .03+.04+.1+.01+.2+.3+.02)
        self.assertAlmostEqual(previous["decision_cpu_seconds"], .18)
        self.assertAlmostEqual(previous["update_cpu_seconds"], 1.)
        self.assertAlmostEqual(previous["reward"], 2-2*previous["seconds"])
        self.assertEqual(len(self.learner.updates), 1)
        self.assertEqual(self.learner.updates[0][:3], (previous["raw_x"], 16, 2))
        self.assertAlmostEqual(self.h.pending["start"], round_start+previous["seconds"]+1.)
        self.assertTrue(previous["updated"])
        self.assertFalse(self.h.rows[0]["updated"])

    def test_constant_gets_full_features_and_same_fresh_path(self):
        self.make("constant")
        self.ready()
        self.assertEqual(self.learner.choices[0][0], [1., 3/8, .25, 3/32])
        self.assertEqual(self.h.pending["raw_features"], self.h.pending["raw_x"])

    def test_new_refresh_anchor_only_applies_to_the_following_root(self):
        self.make()
        self.ready()
        self.observed(10, 2, refresh=True)
        self.proposal(11)
        self.assertEqual(len(self.feature_calls[-1][2]), 6)  # old Root cache's anchor
        self.assertEqual(self.h.pending["anchor_timestep"], 0)
        self.assertEqual(self.h.anchor_timestep, 2)  # applied inside actual Draft
        self.observed(12, 3)
        self.proposal(13)
        self.assertEqual(len(self.feature_calls[-1][2]), 10)
        self.assertEqual(self.h.pending["anchor_timestep"], 2)

    def test_actual_dependency_free_policy_api_and_scaled_logging(self):
        from csgb_online_policy import FeatureScale, OnlineCSGBPolicy
        for policy in ("csgb", "constant", "history_native"):
            with self.subTest(policy=policy):
                self.make(policy)
                scale = FeatureScale(4096, (.2, .1, .02), (.1, .05, .01), 24)
                self.learner = OnlineCSGBPolicy(policy, context=4096, seed=5, rho=2.,
                                              scale=None if policy == "history_native" else scale)
                # Harness context is actual prompt length, not policy context label.
                self.h.reset(policy, self.learner, explore=True, rho=2., context=4)
                self.ready()
                pending = copy.deepcopy(self.h.pending)
                self.assertEqual(pending["learner_input"], list(self.learner.transform(pending["raw_x"])))
                self.observed(10, 2)
                self.proposal(11)
                self.assertEqual(sum(self.learner.counts), 1)
                self.assertEqual(self.learner.counts[(8, 16, 24, 32).index(pending["budget"])], 1)

    def test_fixed8_fresh_records_calibration_features_without_learning(self):
        self.make("fixed8_fresh", calibration=True)
        self.ready()
        self.assertEqual(self.model.drafted, [8, 8, 8])
        self.assertEqual(self.h.pending["raw_x"], [1., 3/8, .25, 3/32])
        self.assertTrue(self.h.pending["eligible_for_calibration"])
        self.assertTrue(self.h.pending["calibration"])
        self.assertEqual(len(self.copies), 2)

    def test_all_native_fixed_budgets_skip_attention_features(self):
        for budget in (8, 16, 24, 32):
            with self.subTest(budget=budget):
                self.make(f"fixed{budget}_native")
                self.ready()
                self.assertEqual(self.model.drafted, [budget]*3)
                self.assertEqual(self.copies, [])
                self.assertEqual(self.feature_calls, [])
                self.assertIsNone(self.h.pending["raw_x"])
                self.assertTrue(self.h.pending["eligible_for_calibration"])
                self.assertFalse(self.model.csgb_attention_every_round)

    def test_history_uses_only_prior_useful_acceptance_without_feature_d2h(self):
        self.make("history_native")
        self.ready()
        self.assertEqual(self.model.drafted, [8, 8, 16])
        self.assertEqual(self.copies, [])
        self.assertEqual(self.feature_calls, [])
        self.assertEqual(self.h.pending["raw_x"], [1., .1, .04375])
        self.assertIsNone(self.h.pending["raw_features"])
        self.assertFalse(self.model.csgb_attention_every_round)

    def test_clean_eval_never_syncs_labels_or_updates(self):
        self.make(clean=True)
        self.ready()
        self.observed(10, 2)
        self.proposal(11)
        self.assertEqual(self.clock.sync_calls, 0)
        self.assertEqual(self.learner.updates, [])
        for row in self.h.rows:
            self.assertIsNone(row["seconds"])
            self.assertIsNone(row["reward"])
            self.assertFalse(row["updated"])

    def test_terminal_overshoot_caps_useful_reward_updates_once(self):
        self.make(limit=5)
        self.ready()  # already committed four, fifth token is the maximum
        self.clock.advance(.3)
        self.model.csgb_terminal_commit_callback(IDs(range(11)))  # raw seven generated
        row = self.h.rows[-1]
        self.assertEqual((row["raw_committed"], row["useful_committed"]), (3, 1))
        self.assertEqual((row["useful_generated_before"], row["useful_generated_after"]), (4, 5))
        self.assertEqual(self.learner.updates[-1][2], 1)
        self.assertAlmostEqual(row["reward"], 1-2*row["seconds"])
        self.assertTrue(row["terminal_commit"])
        self.assertIsNone(self.h.pending)
        self.assertEqual(len(self.model.drafted), 3)
        with self.assertRaises(RuntimeError):
            self.h.finalize_terminal(IDs(range(11)))
        with self.assertRaises(RuntimeError):
            self.proposal(12)
        self.assertEqual(len(self.learner.updates), 1)

    def test_terminal_exact_limit_and_max_one(self):
        for limit, committed in ((1, 5), (8, 12)):
            with self.subTest(limit=limit):
                self.make(limit=limit)
                self.proposal(5)
                self.clock.advance(.3)
                self.h.finalize_terminal(IDs(range(committed)))
                self.assertEqual(self.h.rows[-1]["useful_committed"], limit)
                self.assertTrue(self.h.finished)
                self.assertEqual(self.model.drafted, [8])
                self.assertIsNone(self.h.pending)

    def test_eos_is_inclusive_prompt_eos_ignored_and_after_eos_rejected(self):
        self.make(limit=128, eos=99)
        self.proposal(5)
        self.clock.advance(.3)
        self.h.finalize_terminal(IDs([99, 0, 0, 0, 7, 99]))
        self.assertEqual(self.h.rows[-1]["useful_committed"], 2)
        self.make(eos=99)
        self.proposal(5)
        with self.assertRaisesRegex(RuntimeError, "after first EOS"):
            self.h.finalize_terminal(IDs([0]*4+[99, 7]))

    def test_missing_terminal_fix_fails_before_constructing_unused_proposal(self):
        self.make(limit=1)
        self.proposal(5)
        with self.assertRaisesRegex(RuntimeError, "after the output limit"):
            self.proposal(6)
        self.assertEqual(self.model.drafted, [8])

    def test_unjustified_terminal_callback_and_inconsistent_tokens_rejected(self):
        self.make(eos=99)
        self.proposal(5)
        with self.assertRaisesRegex(RuntimeError, "without EOS or output limit"):
            self.h.finalize_terminal(IDs(range(6)))
        self.h.generated_tokens_fn = lambda ids, context: []
        with self.assertRaisesRegex(RuntimeError, "inconsistent length"):
            self.h.finalize_terminal(IDs(range(6)))

    def test_stale_or_incomplete_fresh_attention_fails_closed(self):
        for length, timestamp in ((6, -1), (5, 0)):
            with self.subTest(length=length, timestamp=timestamp):
                self.make()
                self.proposal(5)
                self.observed(length, timestamp, refresh=True)
                with self.assertRaises(RuntimeError):
                    self.proposal(7)
                self.assertEqual(self.model.drafted, [8])

    def test_nonfinite_or_out_of_range_features_fail_closed(self):
        for features in ([1, math.nan, .2, 0], [1, 1.1, .2, .22], [1, .5, .2, .8]):
            with self.subTest(features=features):
                self.make(feature_fn=lambda a, p, anchor: None if anchor is None else features)
                self.proposal(5)
                self.observed(6, 0, refresh=True)
                self.proposal(7)
                self.observed(8, 1)
                with self.assertRaises((RuntimeError, ValueError)):
                    self.proposal(9)

    def test_wrong_actual_tree_size_fails(self):
        self.make()
        self.model.actual_node_offset = -1
        with self.assertRaisesRegex(RuntimeError, "tree size"):
            self.proposal(5)

    def test_reset_preserves_provenance_and_rejects_unfinished_request(self):
        self.make(limit=1)
        self.proposal(5)
        with self.assertRaises(RuntimeError):
            self.h.reset("fixed8_native", context=4)
        self.clock.advance(.3)
        self.h.finalize_terminal(IDs(range(5)))
        old = self.h.rows[0]
        self.h.reset("fixed8_native", context=4, max_new_tokens=1,
                     provenance={"document": 1})
        self.assertEqual(old["provenance"], {"document": 0})
        self.assertEqual(self.h.request_index, 1)
        self.assertEqual(self.h.rows, [])
        self.assertFalse(self.h.has_real_anchor)

    def test_invalid_configuration_and_double_attachment_rejected(self):
        self.make()
        for options in ({"clean": True, "explore": True}, {"max_new_tokens": 0},
                        {"context": True}, {"rho": float("inf")}, {"eos_token_id": -1}):
            with self.subTest(options=options):
                kw = {"policy": "csgb", "learner": self.learner, "context": 4}
                kw.update(options)
                with self.assertRaises(ValueError):
                    self.h.reset(**kw)
        with self.assertRaises(ValueError):
            OnlineHarness(self.model)


if __name__ == "__main__":
    unittest.main()
