"""CPU-only tests for the original, locked online CSGB policy."""

import copy
import json
import math
import random
import statistics
import unittest

from specextend.csgb_online_policy import (
    BUDGETS, CONTEXTS, FeatureScale, OnlineCSGBPolicy, fit_csgb_scales,
)


def scale_rows():
    result = []
    for context in CONTEXTS:
        for doc in range(24):
            c, d = 0.2 + doc / 40, 0.02 + (doc % 5) / 10
            result.append({"document_id": doc, "context": context, "raw_x": [1, c, d, c * d],
                           "policy": "fixed8", "valid_round": True, "split": "train"})
    return result


class OnlinePolicyTests(unittest.TestCase):
    def setUp(self):
        self.scales = fit_csgb_scales(scale_rows())
        self.raw = [1, 0.75, 0.2, 0.15]

    def policy(self, mode="csgb", seed=123, context=4096):
        return OnlineCSGBPolicy(mode, context=context, seed=seed, rho=30,
                                scale=self.scales[context] if mode == "csgb" else None)

    def test_population_scale_raw_product(self):
        rows = [r for r in scale_rows() if r["context"] == 4096]
        fitted = self.scales[4096]
        expected_means = tuple(statistics.fmean(r["raw_x"][i] for r in rows) for i in (1, 2, 3))
        self.assertEqual(fitted.mean, expected_means)
        self.assertEqual(fitted.scale, tuple(statistics.pstdev(r["raw_x"][i] for r in rows) for i in (1, 2, 3)))
        transformed = self.policy().transform(self.raw)
        self.assertAlmostEqual(transformed[3], (self.raw[3] - fitted.mean[2]) / fitted.scale[2])
        self.assertNotAlmostEqual(transformed[3], transformed[1] * transformed[2])

    def test_zero_and_near_zero_population_variance(self):
        for delta, expected_one in ((0, True), (1e-13, True), (1e-9, False)):
            rows = scale_rows()
            for r in rows:
                c = 0.5 + delta * (r["document_id"] % 2)
                r["raw_x"] = [1, c, 0.2, c * 0.2]
            scale = fit_csgb_scales(rows)[4096]
            self.assertEqual(scale.scale[0] == 1, expected_one)
            self.assertEqual(scale.scale[1], 1)

    def test_fit_rejects_leakage_missing_context_document_and_extra_schema(self):
        mutations = (("document_id", 24), ("document_id", True), ("context", 1024),
                     ("split", "dev"), ("policy", "csgb"), ("valid_round", False),
                     ("valid_round", 1), ("raw_x", [1, .4, .5, .3]), ("extra", 1))
        for key, value in mutations:
            rows = scale_rows()
            rows[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                fit_csgb_scales(rows)
        for rows in (scale_rows()[:-1], scale_rows()[:24], []):
            with self.assertRaises(ValueError):
                fit_csgb_scales(rows)

    def test_feature_scale_roundtrip_and_context_validation(self):
        scale = self.scales[4096]
        self.assertEqual(FeatureScale.from_state(json.loads(json.dumps(scale.to_state()))), scale)
        with self.assertRaises(ValueError):
            OnlineCSGBPolicy("csgb", context=8192, seed=1, rho=1, scale=scale)
        with self.assertRaises(ValueError):
            OnlineCSGBPolicy("csgb", context=4096, seed=1, rho=1)

    def test_constant_consumes_full_raw_path_and_history_identity(self):
        constant = self.policy("constant")
        self.assertEqual(constant.transform(self.raw), (1,))
        with self.assertRaises(ValueError):
            constant.transform([1])
        with self.assertRaises(ValueError):
            constant.transform([1, .4, .5, .3])
        history = self.policy("history_native")
        self.assertEqual(history.transform([1, .8, .4]), (1, .8, .4))
        with self.assertRaises(ValueError):
            history.transform([1, 1.1, .2])

    def test_invalid_constructor_feature_reward_values(self):
        for kwargs in ({"mode": "other"}, {"context": True}, {"context": 1024},
                       {"seed": -1}, {"seed": True}, {"seed": 2**63}, {"rho": 0},
                       {"rho": math.nan}, {"rho": math.inf}, {"rho": True}):
            base = dict(mode="constant", context=4096, seed=1, rho=1)
            base.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OnlineCSGBPolicy(**base)
        p = self.policy()
        for raw in ([0, .5, .2, .1], [1, -.1, .2, -.02], [1, .5, math.nan, .1],
                    [1, True, .2, .2], [1, .5, .2, .1001], [1, .5, .2]):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                p.choose(raw)
        for budget, n, seconds, rho in ((9, 1, .1, None), (True, 1, .1, None),
                                        (8, -1, .1, None), (8, 1.0, .1, None),
                                        (8, True, .1, None), (8, 1, 0, None),
                                        (8, 10**400, .1, None),
                                        (8, 1, math.inf, None), (8, 1, .1, 31)):
            before = p.to_state()
            with self.subTest(values=(budget, n, seconds, rho)), self.assertRaises(ValueError):
                p.update(self.raw, budget, n, seconds, rho)
            self.assertEqual(before, p.to_state())

    def test_only_executed_arm_updated_with_useful_reward(self):
        p = self.policy()
        before = p.to_state()
        reward = p.update(self.raw, 24, 5, .1, 30)
        self.assertEqual(reward, 2)
        after = p.to_state()
        self.assertEqual(after["counts"], [0, 0, 1, 0])
        self.assertEqual(after["rewards"], [0, 0, 2, 0])
        x = p.transform(self.raw)
        for arm in range(4):
            if arm != 2:
                self.assertEqual(after["a"][arm], before["a"][arm])
                self.assertEqual(after["b"][arm], before["b"][arm])
            else:
                for i in range(4):
                    self.assertAlmostEqual(after["b"][arm][i], 2 * x[i])
                    for j in range(4):
                        self.assertAlmostEqual(after["a"][arm][i][j], float(i == j) + x[i] * x[j])
        self.assertEqual(before["rng"], after["rng"])

    def test_startup_fallback_no_update_or_rng_mutation(self):
        p = self.policy()
        before = p.to_state()
        self.assertIsNone(p.transform(None))
        self.assertEqual(p.choose(None), 8)
        self.assertEqual(p.choose(None, explore=False), 8)
        self.assertIsNone(p.update(None, 8, 1, .1, 30))
        self.assertEqual(p.to_state(), before)

    def test_warmup_32_per_arm_and_rng_independence(self):
        p, same, other = self.policy(seed=7), self.policy(seed=7), self.policy(seed=8)
        global_before = random.getstate()
        actions, other_actions = [], []
        for _ in range(128):
            action = p.choose(self.raw)
            self.assertEqual(action, same.choose(self.raw))
            actions.append(action)
            oa = other.choose(self.raw)
            other_actions.append(oa)
            for policy, chosen in ((p, action), (same, action), (other, oa)):
                policy.update(self.raw, chosen, 2, .1)
            self.assertLessEqual(max(p.counts) - min(p.counts), 1)
        self.assertEqual(p.counts, [32] * 4)
        self.assertEqual(p.to_state(), same.to_state())
        self.assertNotEqual(actions, other_actions)
        self.assertEqual(global_before, random.getstate())
        independent_context = self.policy(seed=7, context=8192)
        self.assertEqual(independent_context.counts, [0] * 4)

    def test_frozen_mean_tie_and_no_mutation_after_training(self):
        p = self.policy("constant")
        self.assertEqual(p.choose(self.raw, explore=False), 8)
        for arm, n in zip(BUDGETS, (1, 2, 9, 3)):
            p.update(self.raw, arm, n, .1)
        before = p.to_state()
        for _ in range(10):
            self.assertEqual(p.choose(self.raw, explore=False), 24)
        self.assertEqual(before, p.to_state())
        with self.assertRaises(ValueError):
            p.choose(self.raw, explore=1)

    def test_roundtrip_models_rng_and_continued_training(self):
        for mode, raw in (("csgb", self.raw), ("constant", self.raw), ("history_native", [1, .8, .2])):
            p = self.policy(mode)
            for i in range(140):
                action = p.choose(raw)
                p.update(raw, action, 1 + i % 5, .05 + i % 3 / 100)
            restored = OnlineCSGBPolicy.from_state(json.loads(json.dumps(p.to_state())))
            self.assertEqual(p.to_state(), restored.to_state())
            for i in range(30):
                action = p.choose(raw)
                self.assertEqual(action, restored.choose(raw))
                self.assertEqual(p.update(raw, action, 3, .1), restored.update(raw, action, 3, .1))
            self.assertEqual(p.to_state(), restored.to_state())

    def test_load_rejects_protocol_schema_rng_and_covariance_tampering(self):
        p = self.policy()
        base = p.to_state()
        mutations = (("schema", 2), ("schema", True), ("schema", 1.0), ("source_schema", "other"),
                     ("budgets", [8, 16, 32, 24]), ("budgets", [8., 16, 24, 32]), ("fallback", 16), ("alpha", 2),
                     ("ridge", 2), ("min_exploration", 31), ("epsilon", .1),
                     ("exploration_unit", "total"), ("extra", 1), ("counts", [False, 0, 0, 0]),
                     ("rewards", [math.inf, 0, 0, 0]))
        for key, value in mutations:
            bad = copy.deepcopy(base)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                OnlineCSGBPolicy.from_state(bad)
        for mutate in (
            lambda s: s["rng"].update(version=2),
            lambda s: s["rng"].update(state=[1]),
            lambda s: s["rng"]["state"].__setitem__(0, True),
            lambda s: s["rng"].update(gauss_next=.2),
            lambda s: s["scale"].update(mean=None),
            lambda s: s["scale"].update(documents=None),
            lambda s: s["a"][0][0].__setitem__(0, 0),
            lambda s: s["a"][0][0].__setitem__(1, .2),
            lambda s: s["a"][0][1].__setitem__(1, 2),
            lambda s: s["b"][0].__setitem__(1, .2),
        ):
            bad = copy.deepcopy(base)
            mutate(bad)
            with self.assertRaises(ValueError):
                OnlineCSGBPolicy.from_state(bad)

    def test_optional_numpy_independent_predictions(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("Optional independent NumPy check requires NumPy")
        p = self.policy()
        for i in range(144):
            c, d = .2 + (i % 20) / 30, .03 + (i % 7) / 10
            raw = [1, c, d, c * d]
            p.update(raw, BUDGETS[i % 4], i % 8, .03 + (i % 3) / 100)
        state, x = p.to_state(), np.asarray(p.transform(self.raw))
        means = [float(x @ np.linalg.solve(np.asarray(a), np.asarray(b))) for a, b in zip(state["a"], state["b"])]
        self.assertEqual(p.choose(self.raw, explore=False), BUDGETS[int(np.argmax(means))])
        # Fresh restored RNG is advanced exactly as choose does after warm-up.
        rng = random.Random()
        rng.setstate(p._rng.getstate())
        epsilon_draw = rng.random()
        if epsilon_draw < .05:
            expected = rng.choice(BUDGETS)
        else:
            scores = [m + math.sqrt(float(x @ np.linalg.solve(np.asarray(a), x))) for m, a in zip(means, state["a"])]
            expected = BUDGETS[int(np.argmax(scores))]
        self.assertEqual(p.choose(self.raw), expected)


if __name__ == "__main__":
    unittest.main()
