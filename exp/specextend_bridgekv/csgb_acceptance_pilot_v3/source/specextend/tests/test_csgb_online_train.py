"""Pure fake-engine end-to-end accounting/replay tests for training-only CSGB.

No torch/numpy, remote service, frozen dataset, or GPU is used.  Fixtures are
produced by the real NEW Harness and real private-RNG learner, not handwritten
learner outcomes.  Synthetic contexts retain the locked 4096/8192 lengths.
"""

import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import csgb_online_train

from csgb_online_harness import OnlineHarness
from csgb_online_policy import FeatureScale, OnlineCSGBPolicy
from csgb_online_train import (
    MODES, SEEDS, calibrate_rho, exclusive_json, private_seed, validate_rounds,
    validate_training_source,
)


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds

    def sync(self):
        self.advance(.01)


class IDs:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.shape = (1, len(self.tokens))

    def __getitem__(self, key):
        return SimpleNamespace(tolist=lambda: self.tokens[key[1]])


class Engine:
    def __init__(self, clock):
        self.clock = clock
        self.selected_chunks = None
        self.total_seq_len = 0
        self.attn_scores_final = None
        self.csgb_attention_observed_timestep = None
        self.retrieval_condition = False

    def draft(self, ids, nodes, threshold, max_depth):
        self.clock.advance(.1)
        self.total_seq_len = ids.shape[-1]
        self.update_working_cache_retrieval(do_retrieval=self.retrieval_condition)
        return SimpleNamespace(shape=(1, nodes)), None, None, None

    def update_working_cache_retrieval(self, top_k_chunks=32, do_retrieval=False):
        if self.selected_chunks is None:
            self.selected_chunks = [(0, 0, 32)]
        if do_retrieval:
            self.selected_chunks = [(1, 32, 64)]
            self.attn_scores_final = None
            self.retrieval_condition = False
        return None


def request_fixture(mode="csgb", *, context=4096, document=0, overshoot=False):
    """Produce a real three-round early-EOS or fifteen-round capped trace."""
    clock = Clock()
    model = Engine(clock)

    def copy_attention(attention):
        clock.advance(.02)
        return list(attention)

    def features(attention, readable, anchor):
        clock.advance(.02)
        if anchor is None:
            return None
        c, d = sum(attention[i] for i in readable)/sum(attention), .25
        return [1., c, d, c*d]

    h = OnlineHarness(model, clock=clock, synchronize=clock.sync,
                      attention_to_host=copy_attention, feature_fn=features)
    learner = None if mode == "fixed8_fresh" else OnlineCSGBPolicy(
        mode, context=context, seed=private_seed(101, context, mode), rho=2.,
        scale=FeatureScale(context, (.2, .1, .02), (.1, .05, .01), 24)
        if mode == "csgb" else None)
    before = None if learner is None else learner.to_state()
    phase = "calibration" if learner is None else "train"
    h.reset(mode, learner, explore=learner is not None, rho=0. if learner is None else 2.,
            context=context, eos_token_id=99, max_new_tokens=128,
            calibration=learner is None, provenance={"document": document, "context": context,
                                                       "phase": phase, "seed": 101})
    committed = [0] * context
    raw_counts = [9]*14+[5] if overshoot else [2, 2, 2]
    start = clock()
    model.draft(IDs(committed+[7]), 8, .7, 10)
    for j, count in enumerate(raw_counts):
        committed.extend([7]*count)
        terminal = j == len(raw_counts)-1
        if terminal and not overshoot:
            committed[-1] = 99
        model.retrieval_condition = j % 4 == 0
        if model.csgb_attention_every_round or model.retrieval_condition:
            model.attn_scores_final = [1.] * len(committed)
            model.csgb_attention_observed_timestep = j
        else:
            model.attn_scores_final = None
        clock.advance(.05)  # completed verification/commit/stop handling
        if terminal:
            h.finalize_terminal(IDs(committed))
        else:
            model.draft(IDs(committed+[7]), 8, .7, 10)
    generated = committed[context:]
    output = generated[:128]
    scored_counts = list(raw_counts)
    scored_counts[-1] -= len(generated)-len(output)
    result = {"output_token_ids": output, "total_generated": len(output),
              "raw_total_generated": len(generated), "generation_overshoot": len(generated)-len(output),
              "raw_accept_length_list": [n-1 for n in raw_counts],
              "accept_length_list": [n-1 for n in scored_counts]}
    return {"result": result, "reference": {"output_token_ids": output}, "rows": copy.deepcopy(h.rows),
            "pending": h.pending, "context": context, "seconds": clock()-start,
            "mode": mode, "learner_before": before,
            "learner_after": None if learner is None else learner.to_state(),
            "rho": 0. if learner is None else 2.}


def validate(fixture):
    return validate_rounds(fixture["result"], fixture["reference"], fixture["rows"],
                           fixture["pending"], fixture["context"], fixture["seconds"],
                           mode=fixture["mode"], learner_before=fixture["learner_before"],
                           learner_after=fixture["learner_after"], rho=fixture["rho"])


def calibration_fixture():
    rows = []
    for doc in range(24):
        for context in (4096, 8192):
            f = request_fixture("fixed8_fresh", context=context, document=doc)
            rows.append({"document": doc, "context": context, "phase": "calibration",
                         "policy": "fixed8_fresh", "rounds": f["rows"]})
    return rows


class TrainingValidationTests(unittest.TestCase):
    def test_real_harness_trace_replays_all_modes_and_contexts(self):
        for mode in (*MODES, "fixed8_fresh"):
            for context in (4096, 8192):
                with self.subTest(mode=mode, context=context):
                    f = request_fixture(mode, context=context)
                    before, global_rng = copy.deepcopy(f), random.getstate()
                    self.assertIsNone(validate(f))
                    self.assertEqual(f, before)
                    self.assertEqual(random.getstate(), global_rng)
                    self.assertEqual([r["raw_x"] for r in f["rows"][:2]], [None, None])
                    self.assertEqual([r["updated"] for r in f["rows"][:2]], [False, False])
                    if mode != "fixed8_fresh":
                        self.assertEqual(sum(f["learner_after"]["counts"]), 1)
                        self.assertNotEqual(f["learner_before"]["rng"], f["learner_after"]["rng"])

    def test_early_eos_counts_actual_tokens_not_maximum_128(self):
        f = request_fixture()
        validate(f)
        self.assertEqual(f["result"]["total_generated"], 6)
        self.assertEqual(f["result"]["output_token_ids"][-1], 99)
        self.assertEqual(sum(r["useful_committed"] for r in f["rows"]), 6)
        self.assertTrue(f["rows"][-1]["terminal_commit"])

    def test_overshoot_reward_and_final_scored_acceptance_are_capped(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                f = request_fixture(mode, overshoot=True)
                validate(f)
                self.assertEqual(f["result"]["raw_total_generated"], 131)
                self.assertEqual(f["result"]["total_generated"], 128)
                self.assertEqual(f["result"]["generation_overshoot"], 3)
                self.assertEqual(f["rows"][-1]["raw_committed"], 5)
                self.assertEqual(f["rows"][-1]["useful_committed"], 2)
                self.assertEqual(f["rows"][-1]["reward"], 2-2*f["rows"][-1]["seconds"])

    def test_constant_uses_full_raw4_history_only_prior_useful_acceptance(self):
        constant = request_fixture("constant")
        validate(constant)
        self.assertEqual(len(constant["rows"][2]["x"]), 4)
        self.assertEqual(constant["rows"][2]["learner_input"], [1.])
        history = request_fixture("history_native")
        validate(history)
        self.assertEqual(history["rows"][2]["x"], [1., .1, .04375])
        self.assertTrue(all(r["raw_features"] is None for r in history["rows"]))

    def test_mutated_reward_after_state_rng_action_and_timestamp_rejected(self):
        def action(f):
            r = f["rows"][2]
            r["budget"] = 8 if r["budget"] != 8 else 16
            r["actual_draft_nodes"] = r["budget"]

        mutations = (
            lambda f: f["rows"][2].update(reward=1000.),
            lambda f: f["learner_after"]["rng"]["state"].__setitem__(0, f["learner_after"]["rng"]["state"][0] ^ 1),
            lambda f: f["learner_before"]["rng"]["state"].__setitem__(-1, 623),
            action,
            lambda f: f["rows"][2].update(attention_observed_timestep=2),
            lambda f: f["rows"][0].update(updated=True),
        )
        for i, mutate in enumerate(mutations):
            with self.subTest(mutation=i):
                f = request_fixture()
                mutate(f)
                with self.assertRaises(ValueError):
                    validate(f)

    def test_mutated_commitment_pending_terminal_and_canonical_output_rejected(self):
        mutations = (
            lambda f: f.update(pending={"unverified": True}),
            lambda f: f["rows"][0].update(prefix=4098),
            lambda f: f["rows"][0].update(committed=3),
            lambda f: f["rows"][2].update(useful_committed=3),
            lambda f: f["rows"][2].update(terminal_commit=False),
            lambda f: f["rows"][0].update(terminal_commit=True),
            lambda f: f["rows"][2].update(actual_draft_nodes=999),
            lambda f: f["result"].update(raw_total_generated=9),
            lambda f: f["result"].update(generation_overshoot=1),
            lambda f: f["reference"].update(output_token_ids=[1, 2]),
            lambda f: f.update(seconds=0.),
        )
        for i, mutate in enumerate(mutations):
            with self.subTest(mutation=i):
                f = request_fixture()
                mutate(f)
                with self.assertRaises(ValueError):
                    validate(f)

    def test_raw_feature_and_history_leakage_mutations_rejected(self):
        for mode, mutate in (
            ("constant", lambda r: r.update(raw_features=[1.])),
            ("constant", lambda r: r.update(x=[1.], raw_features=[1.])),
            ("history_native", lambda r: r.update(raw_features=[1., .5, .25, .125])),
            ("history_native", lambda r: r.update(x=[1., .2, .04375])),
        ):
            with self.subTest(mode=mode):
                f = request_fixture(mode)
                mutate(f["rows"][2])
                with self.assertRaises(ValueError):
                    validate(f)

    def test_mislabeled_model_mode_and_context_rejected(self):
        f = request_fixture("csgb")
        f["mode"] = "constant"
        with self.assertRaises(ValueError):
            validate(f)
        f = request_fixture("constant")
        for name in ("learner_before", "learner_after"):
            f[name]["context"] = 8192
        with self.assertRaises(ValueError):
            validate(f)

    def test_audit_aliases_and_real_anchor_cannot_be_forged(self):
        for key, value in (("raw_x", [1., .5, .25, .125]), ("raw_committed", 999),
                           ("has_real_anchor", False), ("eligible_for_calibration", False),
                           ("anchor_timestep", 2)):
            with self.subTest(key=key):
                f = request_fixture()
                f["rows"][2][key] = value
                with self.assertRaises(ValueError):
                    validate(f)

    def test_round_and_update_time_fit_inside_complete_request(self):
        f = request_fixture()
        f["rows"][2]["decision_cpu_seconds"] = f["rows"][2]["seconds"]+1.
        with self.assertRaises(ValueError):
            validate(f)
        f = request_fixture()
        f["rows"][2]["update_cpu_seconds"] = f["seconds"]+1.
        with self.assertRaises(ValueError):
            validate(f)

    def test_root_anchor_must_be_latest_selection_not_just_causally_older(self):
        f = request_fixture(overshoot=True)
        self.assertEqual(f["rows"][6]["anchor_timestep"], 4)
        f["rows"][6]["anchor_timestep"] = 0  # old and divisible by four, but not its Root anchor
        with self.assertRaises(ValueError):
            validate(f)

    def test_fixed_fresh_raw4_is_validated_even_without_learner_replay(self):
        for x in ([1., .5, .25, .9], [1.], [1., float("inf"), .25, .1],
                  [1., -.5, .25, -.125]):
            with self.subTest(raw_x=x):
                f = request_fixture("fixed8_fresh")
                f["rows"][2].update(x=x, raw_x=x, raw_features=x)
                with self.assertRaises(ValueError):
                    validate(f)


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = calibration_fixture()

    def test_all24_docs_both_contexts_only_real_anchored_rounds(self):
        rho = calibrate_rho(self.rows)
        for context in (4096, 8192):
            group = [r for r in self.rows if r["context"] == context]
            eligible = [row for request in group for row in request["rounds"] if row["x"] is not None]
            expected_n = sum(r["useful_committed"] for r in eligible)
            expected_t = sum(r["seconds"] for r in eligible)
            self.assertEqual(rho[str(context)]["documents"], list(range(24)))
            self.assertEqual(rho[str(context)]["eligible_rounds"], 24)
            self.assertEqual(rho[str(context)]["useful_tokens"], expected_n)
            self.assertEqual(rho[str(context)]["round_seconds"], expected_t)
            self.assertEqual(rho[str(context)]["rho"], expected_n/expected_t)

    def test_prefill_and_missing_anchor_round_times_do_not_enter_rho(self):
        bad_startups = copy.deepcopy(self.rows)
        for request in bad_startups:
            for row in request["rounds"]:
                if row["x"] is None:
                    row["seconds"] *= 10000
        self.assertEqual(calibrate_rho(self.rows), calibrate_rho(bad_startups))

    def test_missing_duplicate_dev_wrong_phase_and_policy_rejected(self):
        for mutate in (
            lambda rows: rows.pop(),
            lambda rows: rows.append(copy.deepcopy(rows[0])),
            lambda rows: rows[0].update(document=24),
            lambda rows: rows[0].update(document=True),
            lambda rows: rows[0].update(context=1024),
            lambda rows: rows[0].update(phase="train"),
            lambda rows: rows[0].update(policy="history_native"),
        ):
            rows = copy.deepcopy(self.rows)
            mutate(rows)
            with self.assertRaises(ValueError):
                calibrate_rho(rows)

    def test_future_attention_invalid_useful_and_time_rejected(self):
        for key, value in (("attention_observed_timestep", 2), ("useful_committed", True),
                           ("useful_committed", 0), ("seconds", 0.), ("seconds", float("nan"))):
            with self.subTest(key=key, value=value):
                rows = copy.deepcopy(self.rows)
                rows[0]["rounds"][2][key] = value
                with self.assertRaises(ValueError):
                    calibrate_rho(rows)

    def test_present_features_still_require_actual_anchor_readiness(self):
        for key, value in (("has_real_anchor", False), ("eligible_for_calibration", False),
                           ("anchor_timestep", 2)):
            with self.subTest(key=key):
                rows = copy.deepcopy(self.rows)
                rows[0]["rounds"][2][key] = value
                with self.assertRaises(ValueError):
                    calibrate_rho(rows)

    def test_calibration_raw4_cannot_be_malformed_without_a_learner(self):
        rows = copy.deepcopy(self.rows)
        malformed = [1., .5, .25, .9]
        rows[0]["rounds"][2].update(x=malformed, raw_x=malformed, raw_features=malformed)
        with self.assertRaises(ValueError):
            calibrate_rho(rows)


class InputsAndSeedsTests(unittest.TestCase):
    def test_exactly_training24_texts_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"train24.jsonl"
            good = [{"text": f"synthetic training document {i}"} for i in range(24)]
            path.write_text("\n".join(json.dumps(r) for r in good))
            self.assertEqual(validate_training_source(path), [r["text"] for r in good])
            for bad in (good[:-1], good+[{"text": "held-out must not be staged"}],
                        good[:-1]+[{"text": ""}], good[:-1]+[{"text": 7}]):
                path.write_text("\n".join(json.dumps(r) for r in bad))
                with self.assertRaises(ValueError):
                    validate_training_source(path)

    def test_private_seeds_unique_locked_and_do_not_touch_global_rng(self):
        state = random.getstate()
        values = [private_seed(s, c, m) for s in SEEDS for c in (4096, 8192) for m in MODES]
        self.assertEqual(len(set(values)), 18)
        self.assertEqual(private_seed(101, 4096, "constant"), 101*100000+4096+10000)
        for args in ((999, 4096, "csgb"), (True, 4096, "csgb"),
                     (101, 1024, "csgb"), (101, 4096, "unknown")):
            with self.assertRaises(ValueError):
                private_seed(*args)
        self.assertEqual(state, random.getstate())


class ExclusiveJSONTests(unittest.TestCase):
    def test_complete_json_fsynced_before_atomic_exclusive_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/"result.json"
            pending = Path(tmp)/"result.json.pending"
            value = {"passed": True, "tokens": [7, 99]}
            events = []
            real_link, real_fsync = csgb_online_train.os.link, csgb_online_train.os.fsync

            def fsync(fd):
                events.append("fsync")
                return real_fsync(fd)

            def link(source, destination):
                self.assertEqual(Path(source), pending)
                self.assertEqual(Path(destination), target)
                self.assertFalse(target.exists())
                self.assertEqual(json.loads(pending.read_text()), value)
                self.assertTrue(pending.read_text().endswith("\n"))
                self.assertEqual(events, ["fsync"])
                events.append("publish")
                return real_link(source, destination)

            with mock.patch.object(csgb_online_train.os, "fsync", side_effect=fsync), \
                    mock.patch.object(csgb_online_train.os, "link", side_effect=link):
                exclusive_json(target, value)
            self.assertEqual(events, ["fsync", "publish"])
            self.assertEqual(json.loads(target.read_text()), value)
            self.assertFalse(pending.exists())

    def test_existing_target_and_dangling_symlink_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/"result.json"
            original = b"immutable prior evidence\n"
            target.write_bytes(original)
            with self.assertRaises(FileExistsError):
                exclusive_json(target, {"replacement": True})
            self.assertEqual(target.read_bytes(), original)
            self.assertFalse(Path(str(target)+".pending").exists())
            link = Path(tmp)/"dangling.json"
            link.symlink_to(Path(tmp)/"does-not-exist.json")
            with self.assertRaises(FileExistsError):
                exclusive_json(link, {"replacement": True})
            self.assertTrue(link.is_symlink())
            self.assertFalse(link.exists())

    def test_preexisting_pending_is_preserved_and_never_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/"result.json"
            pending = Path(str(target)+".pending")
            original = b"partial experiment data: inspect, do not reuse"
            pending.write_bytes(original)
            with self.assertRaises(FileExistsError):
                exclusive_json(target, {"passed": True})
            self.assertFalse(target.exists())
            self.assertEqual(pending.read_bytes(), original)

    def test_nan_write_failure_leaves_unpublished_pending_and_blocks_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/"result.json"
            pending = Path(str(target)+".pending")
            with self.assertRaises(ValueError):
                exclusive_json(target, {"completed": [1, 2], "invalid": float("nan")})
            self.assertFalse(target.exists())
            self.assertTrue(pending.exists())
            original = pending.read_bytes()
            self.assertTrue(original)
            with self.assertRaises(FileExistsError):
                exclusive_json(target, {"passed": True})
            self.assertEqual(pending.read_bytes(), original)
            self.assertFalse(target.exists())

    def test_racing_target_is_not_overwritten_and_complete_pending_survives(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/"result.json"
            pending = Path(str(target)+".pending")
            real_link = csgb_online_train.os.link

            def race(source, destination):
                Path(destination).write_bytes(b"winner evidence")
                return real_link(source, destination)

            with mock.patch.object(csgb_online_train.os, "link", side_effect=race):
                with self.assertRaises(FileExistsError):
                    exclusive_json(target, {"passed": True})
            self.assertEqual(target.read_bytes(), b"winner evidence")
            self.assertEqual(json.loads(pending.read_text()), {"passed": True})


if __name__ == "__main__":
    unittest.main()
