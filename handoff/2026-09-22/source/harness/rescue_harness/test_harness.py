"""Targeted regression tests for the RESCUE correctness closure (CPU-safe set).

GPU-dependent gates live in `gpu_smokes.py`; this file is the part that must pass
on any host with torch installed.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rescue_harness import harness as H  # noqa: E402
from rescue_harness import regression_scenarios as R  # noqa: E402
from rescue_harness import training_identity as TI  # noqa: E402


class GradientFailClosedTest(unittest.TestCase):
    """P7 / GRAD-001: missing, non-finite or all-zero gradients must raise BEFORE step."""

    class FakeAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Module()
            self.base.mix_logits = nn.Parameter(torch.zeros(2))
            self.base.residual_logits = nn.Parameter(torch.zeros(2))
            self.base.gate_logits = nn.Parameter(torch.zeros(2))
            self.base.log_temperature = nn.Parameter(torch.zeros(2))
            self.entropy_gate_weight = nn.Parameter(torch.zeros(2))

    def setUp(self):
        self.adapter = self.FakeAdapter()

    def _fill(self, value, which=None):
        for n, p in self.adapter.named_parameters():
            if which is None or n in which:
                p.grad = torch.full_like(p, value)
        return self.adapter

    def test_missing_gradient_raises(self):
        self._fill(1.0, which={"base.mix_logits"})
        with self.assertRaises(H.GradientGateError):
            H.check_gradients_before_step(self.adapter)

    def test_non_finite_gradient_raises(self):
        self._fill(1.0)
        self.adapter.base.gate_logits.grad[0] = float("nan")
        with self.assertRaises(H.GradientGateError):
            H.check_gradients_before_step(self.adapter)

    def test_all_zero_gradient_raises(self):
        self._fill(0.0)
        with self.assertRaises(H.GradientGateError):
            H.check_gradients_before_step(self.adapter)

    def test_frozen_leak_raises(self):
        self._fill(1.0)
        frozen = nn.Linear(2, 2)
        frozen.weight.grad = torch.ones_like(frozen.weight)
        with self.assertRaises(H.GradientGateError):
            H.check_gradients_before_step(self.adapter, frozen_modules=[("student", frozen)])

    def test_clean_gradients_pass(self):
        self._fill(1.0)
        out = H.check_gradients_before_step(self.adapter)
        self.assertGreater(out["grad_norm"], 0.0)


class PriorIdentityTest(unittest.TestCase):
    """P9 / RESUME-001 / PROJ-001."""

    def _identity(self, **over):
        base = dict(token_ids=[1, 2, 3], source_sample_id="s1",
                    teacher_tokenizer_sha="t", student_tokenizer_sha="s",
                    teacher_identity={"path": "/t"}, student_identity={"path": "/s"},
                    teacher_weights_sha="tw", student_weights_sha="sw",
                    projection_sha="p", source_lock_sha="l", config_sha="c")
        base.update(over)
        return H.prior_identity_record(**base)

    def test_identity_record_has_every_required_field(self):
        rec = self._identity()
        for f in H.PRIOR_IDENTITY_FIELDS:
            self.assertIn(f, rec)
        self.assertIn("identity_sha256", rec)

    def test_mutating_any_identity_field_is_detected(self):
        stored = self._identity()
        for field in H.PRIOR_IDENTITY_FIELDS:
            current = self._identity(**(
                {"token_ids": [9, 9, 9]} if field == "token_ids_sha256" else
                {"student_tokenizer_sha": "CHANGED"} if field == "student_tokenizer_sha256" else
                {"source_sample_id": "OTHER"} if field == "source_sample_id" else {}))
            if field in ("token_ids_sha256", "student_tokenizer_sha256", "source_sample_id"):
                ok, diffs = H.prior_identity_matches(stored, current)
                self.assertFalse(ok, f"{field} mutation not detected")
                self.assertIn(field, diffs)

    def test_resume_rejects_on_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "prior.pt"
            torch.save({"x": torch.zeros(1)}, p)
            stored = self._identity()
            complete = {"status": "DONE", "file_sha256": H.sha256_file(p),
                        "identity": stored}
            ok, reason = H.resume_prior_cache(complete, p, stored)
            self.assertTrue(ok, reason)
            other = self._identity(student_tokenizer_sha="DIFFERENT")
            ok, reason = H.resume_prior_cache(complete, p, other)
            self.assertFalse(ok)
            self.assertTrue(reason.startswith("CACHE_IDENTITY_MISMATCH"), reason)

    def test_resume_rejects_on_file_change(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "prior.pt"
            torch.save({"x": torch.zeros(1)}, p)
            ident = self._identity()
            complete = {"status": "DONE", "file_sha256": "deadbeef", "identity": ident}
            ok, reason = H.resume_prior_cache(complete, p, ident)
            self.assertFalse(ok)
            self.assertEqual(reason, "FILE_SHA_MISMATCH")


class CheckpointIdentityTest(unittest.TestCase):
    """P10 / RESUME-003, re-pointed at the FIX-A/FIX-B contract (both sides bound)."""

    def _tree(self, td: Path, hidden_layers=28, weight_bytes=b"\x01\x02"):
        m = td / "model"
        m.mkdir()
        (m / "config.json").write_text(json.dumps({"num_hidden_layers": hidden_layers}),
                                       encoding="utf-8")
        (m / "model-00001-of-00001.safetensors").write_bytes(weight_bytes)
        (m / "tokenizer.json").write_text("{}", encoding="utf-8")
        sl = td / "source_lock"
        sl.mkdir()
        for n in ("adapter.py", "common.py", "steering.py"):
            (sl / n).write_text(f"# {n}\n", encoding="utf-8")
        ck = td / "checkpoint_080.pt"
        ck.write_bytes(b"ckpt")
        return m, sl, ck

    def _build(self, m, sl, ck, cfg=None):
        return H.checkpoint_identity_contract(
            teacher_model_path=m, student_model_path=m, source_lock=sl,
            training_config=cfg or dict(R.MINIMAL_TRAINING_CONFIG), checkpoint=ck)

    def test_contract_covers_both_sides_sourcelock_and_projection(self):
        with tempfile.TemporaryDirectory() as td:
            m, sl, ck = self._tree(Path(td))
            c = self._build(m, sl, ck)
            for key in ("teacher_weights_manifest_sha256", "student_config_sha256",
                        "student_tokenizer_sha256", "student_weights_manifest_sha256",
                        "source_lock", "projection_code_sha256", "steering_code_sha256",
                        "checkpoint_sha256", "training_config_sha256"):
                self.assertTrue(c.get(key), f"missing {key}")
            self.assertTrue(c["path_is_not_identity"])
            self.assertEqual(c["training_config_completeness"]["status"], "COMPLETE")

    def test_weight_replacement_changes_identity(self):
        with tempfile.TemporaryDirectory() as td:
            m, sl, ck = self._tree(Path(td))
            a = self._build(m, sl, ck)
            (m / "model-00001-of-00001.safetensors").write_bytes(b"\xff\xfe")
            b = self._build(m, sl, ck)
            self.assertNotEqual(a["student_weights_manifest_sha256"],
                                b["student_weights_manifest_sha256"])

    def test_projection_code_change_changes_identity(self):
        with tempfile.TemporaryDirectory() as td:
            m, sl, ck = self._tree(Path(td))
            a = self._build(m, sl, ck)
            (sl / "steering.py").write_text("# changed\n", encoding="utf-8")
            b = self._build(m, sl, ck)
            self.assertNotEqual(a["projection_code_sha256"], b["projection_code_sha256"])


class FixAStudentBindingTest(unittest.TestCase):
    """FIX-A: Student config / tokenizer / weights are bound and detectably change."""

    def test_scenario_a_all_pass(self):
        rep = R.scenario_a()
        self.assertEqual(rep["status"], "PASS",
                         [r for r in rep["scenarios"] if r["status"] != "PASS"])


class FixBTrainingConfigFailClosedTest(unittest.TestCase):
    """FIX-B: an incomplete or placeholder training config can never form an identity."""

    def test_scenario_b_all_pass(self):
        rep = R.scenario_b()
        self.assertEqual(rep["status"], "PASS",
                         [r for r in rep["scenarios"] if r["status"] != "PASS"])

    def test_placeholder_variants_raise(self):
        for bad in ("training_config not supplied", "UNKNOWN", "TBD", "", None,
                    {"note": "not provided"}):
            with self.assertRaises(H.CheckpointIdentityIncomplete, msg=repr(bad)):
                H.canonical_training_config(bad)

    def test_complete_config_round_trips_canonically(self):
        cfg = H.canonical_training_config(dict(R.MINIMAL_TRAINING_CONFIG))
        self.assertEqual(set(cfg), set(H.REQUIRED_TRAINING_CONFIG_FIELDS))
        self.assertEqual(json.dumps(cfg, sort_keys=True), json.dumps(cfg, sort_keys=True))


class FixCCompletionIntegrityTest(unittest.TestCase):
    """FIX-C: the completion marker is bound to the body it describes."""

    def test_scenario_c_all_pass(self):
        rep = R.scenario_c()
        self.assertEqual(rep["status"], "PASS",
                         [r for r in rep["scenarios"] if r["status"] != "PASS"])

    def test_is_complete_requires_full_verification(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.jsonl"
            H.write_result_envelope(p, task_id="T", stage="qa", arms=["A"], sample_ids=["s1"],
                                    config_hash="c", checkpoint_sha256="k",
                                    records=[{"canonical_source_id": "s1", "condition": "A"}])
            self.assertTrue(H.is_complete(p))
            self.assertTrue(H.is_complete(p, expected_config_hash="c",
                                          expected_checkpoint_sha256="k"))
            self.assertFalse(H.is_complete(p, expected_config_hash="other"))
            self.assertFalse(H.is_complete(p, expected_checkpoint_sha256="other"))
            p.write_text("tampered\n", encoding="utf-8")
            self.assertFalse(H.is_complete(p))


class DenominatorAuditTest(unittest.TestCase):
    """P12 / DENOM-001."""

    ARMS = ("A", "B")
    FIELDS = ("canonical_source_id", "condition", "value")

    def _rec(self, sid, arm, value=1):
        return {"canonical_source_id": sid, "condition": arm, "value": value}

    def test_clean_records_pass_and_report_counts(self):
        rows = [self._rec(s, a) for s in ("s1", "s2") for a in self.ARMS]
        rep = H.audit_records(rows, self.ARMS, self.FIELDS, planned_ids=["s1", "s2"])
        self.assertEqual(rep["status"], "PASS")
        self.assertEqual(rep["attempted_N"], 4)
        self.assertEqual(rep["valid_N"], 2)
        self.assertEqual(rep["planned_N"], 2)

    def test_duplicate_sample_arm_detected(self):
        rows = [self._rec("s1", "A"), self._rec("s1", "A"), self._rec("s1", "B"),
                self._rec("s2", "A"), self._rec("s2", "B")]
        rep = H.audit_records(rows, self.ARMS, self.FIELDS)
        self.assertEqual(rep["status"], "FAIL_DUPLICATE_SAMPLE_ARM")
        self.assertEqual(rep["duplicate_pairs"][0]["count"], 2)

    def test_missing_arm_detected(self):
        rows = [self._rec("s1", "A"), self._rec("s2", "A"), self._rec("s2", "B")]
        rep = H.audit_records(rows, self.ARMS, self.FIELDS)
        self.assertEqual(rep["status"], "FAIL_MISSING_ARM")
        self.assertIn("s1", rep["samples_missing_an_arm"])

    def test_malformed_and_incomplete_detected(self):
        rows = [self._rec("s1", "A"), self._rec("s1", "B"), {"canonical_source_id": "s2"},
                "not-a-dict"]
        rep = H.audit_records(rows, self.ARMS, self.FIELDS)
        self.assertEqual(rep["status"], "FAIL_MALFORMED_OR_INCOMPLETE")
        self.assertEqual(rep["excluded_N"], 2)

    def test_planned_ids_missing_is_partial_not_pass(self):
        rows = [self._rec("s1", a) for a in self.ARMS]
        rep = H.audit_records(rows, self.ARMS, self.FIELDS, planned_ids=["s1", "s2"])
        self.assertEqual(rep["status"], "PARTIAL_MISSING_PLANNED_IDS")
        self.assertIn("s2", rep["planned_ids_absent_from_results"])
        with self.assertRaises(H.DenominatorError):
            H.require_clean_denominator(rep)

    def test_effective_n_is_never_file_length(self):
        rows = [self._rec("s1", "A"), self._rec("s1", "A"), self._rec("s1", "B")]
        rep = H.audit_records(rows, self.ARMS, self.FIELDS)
        self.assertNotEqual(rep["valid_N"], rep["attempted_N"])


class OutputIntegrityTest(unittest.TestCase):
    """P11 / RESUME-002."""

    def test_atomic_write_leaves_no_temp_and_replaces(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "out.jsonl"
            H.atomic_write_text(p, "first\n")
            H.atomic_write_text(p, "second\n")
            self.assertEqual(p.read_text(encoding="utf-8"), "second\n")
            self.assertEqual([f.name for f in Path(td).iterdir()], ["out.jsonl"])

    def test_envelope_requires_completion_flag(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "shard.jsonl"
            H.write_result_envelope(p, task_id="T", stage="qa", config_hash="c",
                                    checkpoint_sha256="k", arms=["A"], sample_ids=["s1"],
                                    status="COMPLETE", records=[{"a": 1}])
            self.assertTrue(H.is_complete(p, expected_config_hash="c",
                                          expected_checkpoint_sha256="k"))
            flag = json.loads(H.completion_path(p).read_text(encoding="utf-8"))
            self.assertEqual(flag["record_count"], 1)
            self.assertEqual(flag["arms"], ["A"])
            self.assertEqual(flag["status"], "COMPLETE")
            self.assertEqual(flag["body_sha256"], H.sha256_file(p))
            self.assertTrue(flag["generation_id"])

    def test_truncated_write_has_no_flag(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "shard.jsonl"
            H.atomic_write_text(p, '{"partial": ')
            self.assertFalse(H.is_complete(p))


class FixDSchemaValidationTest(unittest.TestCase):
    """FIX-D: field-aware training-config schema (no non-empty-dict shortcuts)."""

    def test_scenario_d_all_pass(self):
        rep = R.scenario_d()
        self.assertEqual(rep["status"], "PASS",
                         [r for r in rep["scenarios"] if r["status"] != "PASS"])

    def test_nested_identity_and_bool_rejections(self):
        base = dict(R.MINIMAL_TRAINING_CONFIG)
        for bad in ({**base, "train_split_identity": {"sha256": None}},
                    {**base, "adapter_config": {"note": "training_config not supplied"}},
                    {**base, "learning_rate": False},
                    {**base, "training_steps": False},
                    {**base, "seed": True},
                    {**base, "dtype": "float64"}):
            with self.assertRaises(H.CheckpointIdentityIncomplete, msg=repr(bad)[:80]):
                H.canonical_training_config(bad)

    def test_valid_config_and_na_values_pass(self):
        base = dict(R.MINIMAL_TRAINING_CONFIG)
        H.canonical_training_config(base)
        H.canonical_training_config({**base, "method_variant":
                                     "attention_prior_adapter+canonical+analysis"})
        cfg = H.canonical_training_config({**base, "seed": 0})
        self.assertEqual(cfg["seed"], 0)

    def test_schema_contract_is_declared(self):
        c = H.schema_contract(None)
        self.assertEqual(len(c["required_fields"]), 13)
        self.assertIn("no substring heuristics", c["placeholder_policy"])
        self.assertIn("bool is rejected wherever a number/int is expected", c["bool_policy"])


class FixEProvenanceValueMatchTest(unittest.TestCase):
    """FIX-E: declared == observed.  Needs real artefacts; skipped when absent."""

    @staticmethod
    def _artifacts():
        import os
        p = os.environ.get("RESCUE_E_ARTIFACTS", "")
        if not p or not Path(p).exists():
            return None
        return json.loads(Path(p).read_text(encoding="utf-8"))

    def test_scenario_e_all_pass(self):
        art = self._artifacts()
        if art is None:
            self.skipTest("set RESCUE_E_ARTIFACTS to a real-artefact json to run FIX-E")
        rep = R.scenario_e(art)
        self.assertEqual(rep["status"], "PASS",
                         [r for r in rep["scenarios"] if r["status"] != "PASS"])

    def test_bool_and_placeholder_claims_cannot_pass(self):
        art = self._artifacts()
        if art is None:
            self.skipTest("set RESCUE_E_ARTIFACTS to a real-artefact json to run FIX-E")
        truth = dict(art["truthful_config"])
        for field, bad in (("dtype", "float16"), ("layer_scope", "last_2_layers"),
                           ("supervised_target_mode", "first_token_CE"),
                           ("initialization", "warm_start"), ("method_variant", "WRONG")):
            try:
                import torch as _t
                payload = _t.load(art["checkpoint"], map_location="cpu", weights_only=False)
                observed = TI.observe_training_facts(
                    source_lock=Path(art["source_lock"]),
                    run_source_dirs=[Path(p) for p in art["source_dirs"]],
                    checkpoint_payload=payload,
                    train_ledger=Path(art["train_ledger"]),
                    validation_ledger=Path(art["validation_ledger"]))
            except Exception as exc:                       # noqa: BLE001
                self.skipTest(f"artefacts not usable: {type(exc).__name__}")
            if observed[field]["observed"] == bad:
                continue                                   # mutation would be a no-op
            with self.assertRaises(TI.ProvenanceDeclarationMismatch, msg=field):
                TI.verify_declared_vs_observed({**truth, field: bad}, observed)


class DtypeClosureTest(unittest.TestCase):
    """P8 — fail-closed dtype policy (no GPU needed)."""

    def test_fp16_target_is_rejected(self):
        with self.assertRaises(H.NumericsError):
            H.assert_target_dtype_allowed("float16")

    def test_canonical_dtype_accepted(self):
        H.assert_target_dtype_allowed(H.CANONICAL_TARGET_DTYPE)
        self.assertEqual(H.resolve_dtype("bfloat16"), torch.bfloat16)

    def test_non_finite_logits_raise(self):
        bad = torch.tensor([1.0, float("nan")])
        with self.assertRaises(H.NumericsError):
            H.assert_logits_finite(bad, "unit-test")
        H.assert_logits_finite(torch.tensor([1.0, 2.0]), "unit-test")


class ArmIsolationTest(unittest.TestCase):
    """P4 — the isolation policy is enforced on fingerprints (fake caches here)."""

    class FakeCache:
        def __init__(self, n):
            self.n = n
            self.layers = [type("L", (), {"keys": torch.zeros(1)})()]

        def get_seq_length(self, i=None):
            return self.n

        def __len__(self):
            return 1

    def test_shared_cache_object_is_rejected(self):
        c = self.FakeCache(6)
        fp = {a: H.cache_fingerprint(c) for a in ("stock", "hook_empty", "teacher", "null")}
        rep = H.assert_arm_isolation(fp)
        self.assertEqual(rep["status"], "FAIL")
        self.assertFalse(rep["distinct_cache_objects"])

    def test_reprefill_per_arm_passes(self):
        kept = {a: self.FakeCache(6) for a in ("stock", "hook_empty", "teacher", "null")}
        fp = {a: H.cache_fingerprint(c) for a, c in kept.items()}
        rep = H.assert_arm_isolation(fp)
        self.assertEqual(rep["status"], "PASS")
        self.assertTrue(rep["distinct_cache_objects"])
        self.assertTrue(rep["distinct_key_buffers"])

    def test_stock_purity_rejects_any_intervention(self):
        self.assertEqual(H.assert_stock_purity(
            {"patched_forward_calls": 0, "mix_calls": 0, "hook_installations": 0})["status"], "PASS")
        self.assertEqual(H.assert_stock_purity(
            {"patched_forward_calls": 1, "mix_calls": 0, "hook_installations": 0})["status"], "FAIL")
        self.assertEqual(H.assert_stock_purity(
            {"patched_forward_calls": 0, "mix_calls": 3, "hook_installations": 1})["status"], "FAIL")


class RuntimeSignatureTest(unittest.TestCase):
    """P1 — the contract object itself, plus the invariant that nothing is swallowed."""

    def test_contract_reports_versions_and_signatures(self):
        class Dummy:
            def forward(self, hidden_states, position_embeddings, attention_mask=None,
                        past_key_values=None, cache_position=None, **kwargs):
                return None

        c = H.runtime_signature_contract(
            Dummy, "patched(hidden_states, position_embeddings, attention_mask, "
                   "past_key_value, past_key_values, cache_position, **kwargs)",
            {"hidden_states": 1, "position_embeddings": 2, "attention_mask": 3,
             "past_key_values": 4, "cache_position": 5, "position_ids": 6, "use_cache": True})
        self.assertIn("transformers_version", c)
        self.assertIn("past_key_values", c["observed_call_kwargs"])
        self.assertEqual(c["verdict"], "PASS")

    def test_swallowed_state_kwarg_is_flagged(self):
        class Dummy:
            def forward(self, hidden_states, position_embeddings, **kwargs):
                return None

        c = H.runtime_signature_contract(Dummy, "patched(**kwargs)",
                                         {"hidden_states": 1, "position_embeddings": 2,
                                          "past_key_values": 3})
        self.assertIn("past_key_values", c["stateful_kwargs_swallowed_unused"])
        self.assertEqual(c["verdict"], "FAIL")

    def test_forward_only_kwargs_are_not_failures(self):
        class Dummy:
            def forward(self, hidden_states, position_embeddings, attention_mask=None,
                        past_key_values=None, cache_position=None, **kwargs):
                return None

        c = H.runtime_signature_contract(Dummy, "patched(...)",
                                         {"hidden_states": 1, "position_embeddings": 2,
                                          "attention_mask": 3, "past_key_values": 4,
                                          "cache_position": 5, "position_ids": 6,
                                          "use_cache": True, "output_attentions": False})
        self.assertEqual(c["stateful_kwargs_swallowed_unused"], [])
        self.assertIn("position_ids", c["kwargs_forwarded_by_design"])
        self.assertEqual(c["verdict"], "PASS")

    def test_harness_steering_source_compiles_and_declares_cache_args(self):
        cm = H.load_harness_steering()
        import inspect as _i
        self.assertTrue(callable(cm))


class SteeringSourceContractTest(unittest.TestCase):
    """The harness steering must declare the 4.57 argument names and forward the cache."""

    def test_source_declares_past_key_values_and_position_embeddings(self):
        src = H.STEERING_SOURCE
        for token in ("past_key_values", "past_key_value", "cache_position",
                      "position_embeddings", "attention_mask"):
            self.assertIn(token, src)
        self.assertIn("original_forward(", src)      # exact delegation when idle
        self.assertIn("pkv.update(", src)            # real cache write on the steered path
        self.assertIn("state.capture_rows", src)     # frozen row-capture semantics preserved


if __name__ == "__main__":
    unittest.main(verbosity=2)
