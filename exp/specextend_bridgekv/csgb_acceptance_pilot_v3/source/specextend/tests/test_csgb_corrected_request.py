"""Standard-library regression tests for the frozen corrected request runner."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from csgb_corrected_request import (
    FEATURE_COLUMNS, POLICIES, assert_gpu_uncontended, current_gpu_uuid, validate_documents,
    validate_model_evidence, validate_models, validate_request,
)


def frozen_state():
    models = {}
    for context in ("4096", "8192"):
        models[context] = {}
        for name, cols in FEATURE_COLUMNS.items():
            theta = [[0., 1., 0., 0.]] + [[0., 0., 0., 0.] for _ in cols[1:]]
            models[context][name] = dict(
                cols=cols.copy(), budgets=[8, 16, 24, 32],
                mean=[0.] * (len(cols) - 1), scale=[1.] * (len(cols) - 1), theta=theta,
            )
    return dict(schema=1, train_documents=list(range(24)), corrected_root_coverage=True,
                attention_observation_every_round=True, contexts=[4096, 8192],
                budgets=[8, 16, 24, 32], intervention_rounds=[5, 6, 7, 8],
                repeats=2, ridge=1., models=models)


class StdlibFrozenLearner:
    """The same scaling/dot/first-argmax rule, with no NumPy dependency."""
    def __init__(self, state):
        self.state = copy.deepcopy(state)
        self.calls = []

    def choose(self, x, explore=False):
        self.calls.append((copy.deepcopy(x), explore))
        if explore:
            raise ValueError("Frozen evaluation cannot explore")
        if x is None:
            return 8
        if len(x) != 4:
            raise ValueError("Expected four CSGB features")
        cols = self.state["cols"]
        z = [x[i] for i in cols]
        if not all(math.isfinite(v) for v in z):
            return 8
        z[1:] = [(v - mean) / scale for v, mean, scale in
                 zip(z[1:], self.state["mean"], self.state["scale"])]
        values = [sum(z[i] * self.state["theta"][i][b] for i in range(len(z)))
                  for b in range(4)]
        return self.state["budgets"][max(range(4), key=values.__getitem__)]


def request_fixture(policy="csgb", overshoot=False):
    context = 4096
    adaptive = policy in ("csgb", "constant_gain")
    b = 8 if adaptive else int(policy[5:-7])
    rows = [dict(proposal=0, prefix=context + 1, budget=b, actual_draft_nodes=b,
                 committed=2, x=None, coverage=None, shift=None,
                 attention_observed_timestep=None, seconds=None, reward=None,
                 decision_cpu_seconds=.001)]
    b = 16 if adaptive else b
    x = [1., .8, .1, .08] if adaptive else None
    rows.append(dict(proposal=1, prefix=context + 3, budget=b, actual_draft_nodes=b,
                     committed=3, x=x, coverage=.8 if adaptive else None,
                     shift=.1 if adaptive else None, attention_observed_timestep=0,
                     seconds=None, reward=None, decision_cpu_seconds=.002,
                     terminal_commit=True))
    scored = [1, 0] if overshoot else [1, 2]
    output = [10, 11, 12] if overshoot else [10, 11, 12, 13, 14]
    result = dict(output_token_ids=output, total_generated=len(output),
                  raw_total_generated=5, generation_overshoot=5 - len(output),
                  raw_accept_length_list=[1, 2], accept_length_list=scored,
                  avg_accept_length=round(sum(scored) / len(scored), 3))
    learner = StdlibFrozenLearner(frozen_state()["models"][str(context)][
        "constant" if policy == "constant_gain" else "csgb"]) if adaptive else None
    return dict(result=result, reference=dict(output_token_ids=output.copy()), rows=rows,
                pending=None, policy=policy, context=context, seconds=.5, learner=learner)


class CorrectedModelValidationTests(unittest.TestCase):
    def test_complete_frozen_24_document_protocol(self):
        state = frozen_state()
        before = copy.deepcopy(state)
        validate_models(state)
        self.assertEqual(state, before)

    def test_old_four_document_and_wrong_feature_protocols_rejected(self):
        for key, value in (("train_documents", [0, 1, 2, 3]),
                           ("corrected_root_coverage", False),
                           ("attention_observation_every_round", False),
                           ("contexts", [4096]), ("budgets", [32, 64, 96]),
                           ("intervention_rounds", [5, 8, 12]),
                           ("ridge", 2.), ("repeats", 3)):
            with self.subTest(key=key):
                state = frozen_state(); state[key] = value
                with self.assertRaises(ValueError):
                    validate_models(state)

    def test_missing_context_or_ablation_rejected(self):
        state = frozen_state(); del state["models"]["8192"]
        with self.assertRaises(ValueError): validate_models(state)
        state = frozen_state(); del state["models"]["4096"]["constant"]
        with self.assertRaises(ValueError): validate_models(state)

    def test_shapes_columns_and_invalid_frozen_numbers_rejected(self):
        for field, value in (("cols", [0, 2, 1, 3]), ("budgets", [8, 16, 24]),
                             ("mean", [0.]), ("scale", [1., 1., 0.]),
                             ("scale", [1., 1., float("nan")]),
                             ("theta", [[0., 1.]]),
                             ("theta", [[0., 1., 0., 0.]] * 3 + [[0., True, 0., 0.]])):
            with self.subTest(field=field, value=value):
                state = frozen_state(); state["models"]["4096"]["csgb"][field] = value
                with self.assertRaises(ValueError): validate_models(state)


class CorrectedDocumentValidationTests(unittest.TestCase):
    def test_development_and_training_only_smoke(self):
        validate_documents(list(range(24, 32)))
        validate_documents([24, 31])
        validate_documents([0], smoke=True)

    def test_nondev_confirmation_empty_duplicate_and_wrong_smoke_rejected(self):
        for docs, smoke in (([], False), ([0], False), ([23], False), ([32], False),
                            ([24, 32], False), ([24, 24], False), ([24], True),
                            ([0, 1], True), ([0, 0], True)):
            with self.subTest(docs=docs, smoke=smoke), self.assertRaises(ValueError):
                validate_documents(docs, smoke)


class CorrectedEvidenceValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.inputs = {}

        def save(name, content):
            path = cls.root / name
            path.write_text(content)
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            cls.inputs[str(path)] = sha
            return str(path), sha

        # Real tiny files exercise every evidence read/hash, not a patched
        # digest that could hide a missing or altered physical input.
        for i in range(1536):
            save(f"case_{i}.json", json.dumps({"case": i}))
        cls.sidecar, cls.sidecar_sha = save(
            "corrected_features_sidecar.json",
            json.dumps({"passed": True, "cases": 1536, "states": 192}),
        )
        cls.fitter, cls.fitter_sha = save("fit_csgb_train24.py", "# isolated fitter fixture\n")
        cls.helper, cls.helper_sha = save("fit_helper.py", "# isolated helper fixture\n")
        cls.state = dict(
            cases=1536, states=192, feature_meaning="corrected_root_coverage",
            cache_refresh_every_rounds=4, target_dtype="float32", draft_dtype="float16",
            no_hyperparameter_search=True, development_read=False, confirmation_read=False,
            fitter_sha256=cls.fitter_sha, input_sha256=cls.inputs,
            provenance=dict(sidecar_path=cls.sidecar, sidecar_sha256=cls.sidecar_sha,
                            fitter_source_sha256={cls.fitter: cls.fitter_sha, cls.helper: cls.helper_sha}),
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_complete_physical_training_evidence(self):
        validate_model_evidence(copy.deepcopy(self.state))

    def test_changed_fitter_helper_or_sidecar_bytes_rejected(self):
        for name in (self.fitter, self.helper, self.sidecar):
            path = Path(name); original = path.read_bytes()
            try:
                path.write_bytes(original + b"\nchanged\n")
                with self.subTest(path=path.name), self.assertRaises(ValueError):
                    validate_model_evidence(copy.deepcopy(self.state))
            finally:
                path.write_bytes(original)

    def test_wrong_complete_case_metadata_and_leakage_flags_rejected(self):
        for key, value in (("cases", 1535), ("states", 191),
                           ("feature_meaning", "old_chunk_only_coverage"),
                           ("cache_refresh_every_rounds", 8), ("target_dtype", "float16"),
                           ("draft_dtype", "float32"), ("no_hyperparameter_search", False),
                           ("development_read", True), ("confirmation_read", True)):
            state = copy.deepcopy(self.state); state[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_model_evidence(state)

    def test_missing_or_misaligned_immutable_manifests_rejected(self):
        variants = []
        state = copy.deepcopy(self.state); state["input_sha256"] = {}
        variants.append(state)
        state = copy.deepcopy(self.state); state["provenance"]["fitter_source_sha256"] = {}
        variants.append(state)
        state = copy.deepcopy(self.state); state["fitter_sha256"] = "0" * 64
        variants.append(state)
        state = copy.deepcopy(self.state); state["provenance"]["sidecar_sha256"] = "0" * 64
        variants.append(state)
        state = copy.deepcopy(self.state); del state["input_sha256"][self.helper]
        variants.append(state)
        for i, state in enumerate(variants):
            with self.subTest(variant=i), self.assertRaises(ValueError):
                validate_model_evidence(state)


class CorrectedGPUIdentityTests(unittest.TestCase):
    def query(self, stdout):
        with patch("csgb_corrected_request.os.getpid", return_value=4242), \
             patch("csgb_corrected_request.subprocess.run", return_value=SimpleNamespace(stdout=stdout)) as run:
            value = current_gpu_uuid()
        run.assert_called_once_with(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return value

    def test_current_pid_single_uuid_and_ignore_other_processes(self):
        self.assertEqual(self.query("GPU-ours, 4242\nGPU-other, 99\n"), "GPU-ours")
        self.assertEqual(self.query("GPU-ours,4242\nGPU-ours, 4242\nmalformed\n"), "GPU-ours")

    def test_zero_or_two_current_process_uuids_fail_closed(self):
        for stdout in ("", "GPU-other, 99\n", "GPU-a, 4242\nGPU-b, 4242\n"):
            with self.subTest(stdout=stdout), self.assertRaises(RuntimeError):
                self.query(stdout)


class CorrectedGPUContentionTests(unittest.TestCase):
    def check(self, stdout):
        with patch("csgb_corrected_request.os.getpid", return_value=4242), \
             patch("csgb_corrected_request.subprocess.run", return_value=SimpleNamespace(stdout=stdout)) as run:
            assert_gpu_uncontended("GPU-ours")
        run.assert_called_once_with(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )

    def test_only_current_process_on_physical_gpu_passes(self):
        self.check("GPU-ours, 4242\n")

    def test_another_process_on_same_gpu_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.check("GPU-ours, 4242\nGPU-ours, 999\n")

    def test_no_current_process_or_no_gpu_records_is_rejected(self):
        for stdout in ("", "GPU-other, 4242\n", "GPU-ours, 999\n"):
            with self.subTest(stdout=stdout), self.assertRaises(RuntimeError):
                self.check(stdout)

    def test_unrelated_gpu_processes_do_not_block_worker(self):
        self.check("GPU-ours, 4242\nGPU-other, 999\nGPU-another, 123\n")


class CorrectedRequestValidationTests(unittest.TestCase):
    def test_all_six_policy_positive_cases(self):
        for policy in POLICIES:
            with self.subTest(policy=policy):
                data = request_fixture(policy)
                validate_request(**data)
                if data["learner"] is not None:
                    self.assertEqual(len(data["learner"].calls), 2)
                    self.assertTrue(all(explore is False for _, explore in data["learner"].calls))

    def test_overshoot_uses_scored_denominator_and_preserves_raw_commitment(self):
        for policy in POLICIES:
            with self.subTest(policy=policy): validate_request(**request_fixture(policy, True))

    def test_frozen_action_replay_rejects_unselected_budget(self):
        for policy in ("csgb", "constant_gain"):
            data = request_fixture(policy)
            data["rows"][1].update(budget=24, actual_draft_nodes=24)
            with self.subTest(policy=policy), self.assertRaises(ValueError): validate_request(**data)

    def test_terminal_pending_prefix_commit_nodes_and_timing_errors(self):
        changes = (("terminal_commit", False), ("prefix", 4100), ("proposal", 2),
                   ("committed", 4), ("actual_draft_nodes", 24), ("seconds", .1),
                   ("reward", 0.), ("decision_cpu_seconds", -1.),
                   ("decision_cpu_seconds", float("nan")))
        for key, value in changes:
            with self.subTest(key=key, value=value):
                data = request_fixture(); data["rows"][1][key] = value
                with self.assertRaises(ValueError): validate_request(**data)
        data = request_fixture(); data["pending"] = {}
        with self.assertRaises(ValueError): validate_request(**data)
        data = request_fixture(); data["rows"][0]["terminal_commit"] = True
        with self.assertRaises(ValueError): validate_request(**data)

    def test_terminal_callback_flags_must_be_actual_booleans(self):
        for value in ("true", "false", 1, 0, None):
            with self.subTest(value=value):
                data = request_fixture(); data["rows"][-1]["terminal_commit"] = value
                with self.assertRaises(ValueError): validate_request(**data)

    def test_fresh_timestamp_and_coverage_shift_values(self):
        for key, value in (("attention_observed_timestep", -1),
                           ("attention_observed_timestep", 1),
                           ("coverage", .7), ("shift", .2)):
            with self.subTest(key=key, value=value):
                data = request_fixture(); data["rows"][1][key] = value
                with self.assertRaises(ValueError): validate_request(**data)
        for x in ([1., .8, .1], [0., .8, .1, .08], [1., 1.1, .1, .11],
                  [1., .8, 1.1, .88], [1., .8, .1, .09],
                  [1., float("nan"), .1, float("nan")]):
            with self.subTest(x=x):
                data = request_fixture(); data["rows"][1]["x"] = x
                with self.assertRaises(ValueError): validate_request(**data)

    def test_fixed_native_cannot_use_controller_features_or_switch_budget(self):
        data = request_fixture("fixed8_native")
        data["rows"][0]["x"] = [1., .8, .1, .08]
        with self.assertRaises(ValueError): validate_request(**data)
        data = request_fixture("fixed8_native")
        data["rows"][1].update(budget=16, actual_draft_nodes=16)
        with self.assertRaises(ValueError): validate_request(**data)

    def test_missing_features_require_budget_eight_and_a_learner(self):
        data = request_fixture(); data["rows"][1]["x"] = None
        with self.assertRaises(ValueError): validate_request(**data)
        data = request_fixture(); data["learner"] = None
        with self.assertRaises(ValueError): validate_request(**data)

    def test_bad_denominators_scored_acceptance_and_canonical_output(self):
        mutations = (("total_generated", 6), ("raw_total_generated", 6),
                     ("generation_overshoot", 1), ("avg_accept_length", 1.),
                     ("accept_length_list", [0, 3]), ("raw_accept_length_list", [1, -1]),
                     ("raw_accept_length_list", [True, 2]),
                     ("output_token_ids", [10, 11, 12, 13, 99]))
        for key, value in mutations:
            with self.subTest(key=key):
                data = request_fixture(); data["result"][key] = value
                with self.assertRaises(ValueError): validate_request(**data)
        data = request_fixture(overshoot=True)
        data["result"]["accept_length_list"] = [0, 1]
        data["result"]["avg_accept_length"] = .5
        with self.assertRaises(ValueError): validate_request(**data)

    def test_nonpositive_nonfinite_elapsed_or_missing_verified_round_rejected(self):
        for seconds in (0., -1., float("nan"), float("inf")):
            with self.subTest(seconds=seconds):
                data = request_fixture(); data["seconds"] = seconds
                with self.assertRaises(ValueError): validate_request(**data)
        data = request_fixture(); data["rows"].pop()
        with self.assertRaises(ValueError): validate_request(**data)


if __name__ == "__main__":
    unittest.main()
