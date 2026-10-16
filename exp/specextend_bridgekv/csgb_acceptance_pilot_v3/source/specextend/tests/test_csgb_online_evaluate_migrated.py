"""Synthetic CPU-only v4 closure and real full582-request replay tests.

No real development text/outcome, server, model or GPU is accessed. The only
external derivation replaced is the already-audited2522-file training scaler.
"""

import ast
import builtins
import contextlib
import copy
import json
import io
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import csgb_online_evaluate as previous
import csgb_online_evaluate_migrated as evaluate
import csgb_online_policy as policy_module
import csgb_online_train as training
from csgb_online_harness import OnlineHarness
from csgb_online_policy import OnlineCSGBPolicy
from test_csgb_online_evaluate import FrozenFixture, clean_fixture, write_json
from test_csgb_online_train import Clock, Engine, IDs

REAL_HARDWARE_VERIFY = evaluate.migration_provenance.verify_hardware_migration


def training_request(mode, context, document, seed, gpu, phase, *, learner=None, rho=0.):
    """Eight actual fake-engine verified rounds, with six real-anchor updates."""
    clock, model = Clock(), None
    model = Engine(clock)

    def to_host(attention):
        clock.advance(.02)
        return list(attention)

    def features(attention, readable, anchor):
        clock.advance(.02)
        if anchor is None:
            return None
        c, d = sum(attention[i] for i in readable)/sum(attention), .25
        return [1., c, d, c*d]

    h = OnlineHarness(model, clock=clock, synchronize=clock.sync,
                      attention_to_host=to_host, feature_fn=features)
    before = None if learner is None else learner.to_state()
    h.reset(mode, learner, explore=learner is not None, clean=False, rho=rho,
            context=context, max_new_tokens=128, eos_token_id=99,
            calibration=mode == "fixed8_fresh",
            provenance={"document": document, "context": context, "seed": seed, "phase": phase})
    committed = [0]*context
    model.draft(IDs(committed+[7]), 8, .7, 10)
    for j in range(8):
        committed.append(99 if j == 7 else 7)
        model.retrieval_condition = j % 4 == 0
        if model.csgb_attention_every_round or model.retrieval_condition:
            model.attn_scores_final = [1.]*len(committed)
            model.csgb_attention_observed_timestep = j
        else:
            model.attn_scores_final = None
        clock.advance(.05)
        if j == 7:
            h.finalize_terminal(IDs(committed))
        else:
            model.draft(IDs(committed+[7]), 8, .7, 10)
    output = committed[context:]
    return {"document": document, "context": context, "seed": seed, "gpu_uuid": gpu,
            "phase": phase, "policy": mode, "seconds": clock(), "rho": rho,
            "pending_is_none": h.pending is None, "exact": True, "performance_claim": False,
            "instrumented_training_request_tps": len(output)/clock(),
            "result": {"output_token_ids": output, "total_generated": 8, "raw_total_generated": 8,
                       "generation_overshoot": 0, "raw_accept_length_list": [0]*8, "accept_length_list": [0]*8},
            "rounds": copy.deepcopy(h.rows), "learner_before": before,
            "learner_after": None if learner is None else learner.to_state()}


class MigratedFixture(FrozenFixture):
    """Real source, raw, artifact hashes; all582 traces use actual CPU learners."""
    def __init__(self, tmp, *, publish=True):
        super().__init__(tmp)
        self.migrated = Path(tmp).resolve()/"migrated"
        self.prior = Path(tmp).resolve()/"failed_retry"
        self.migrated.mkdir(); self.prior.mkdir()
        self.engine.rename(self.migrated/"analysis_tools")
        self.engine = self.migrated/"analysis_tools"
        source_stage = (self.root/"stage.sha256").read_bytes()
        entries = {}
        for line in source_stage.decode().splitlines():
            wanted, name = line.split(maxsplit=1)
            entries[name] = wanted
            target = self.migrated/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((self.root/name).read_bytes())
        runner = evaluate.migration_provenance.RUNNER_RELATIVE
        changed = evaluate.migration_provenance.expected_migrated_runner((self.root/runner).read_bytes())
        (self.migrated/runner).write_bytes(changed)
        self.new_entries = {**entries, runner: evaluate.sha(self.migrated/runner)}
        (self.migrated/"stage.sha256").write_bytes(source_stage.replace(entries[runner].encode(), self.new_entries[runner].encode()))
        self.new_stage_sha = evaluate.sha(self.migrated/"stage.sha256")
        for dest in (self.migrated/"csgb_migration_provenance.py", self.engine/"csgb_migration_provenance.py"):
            dest.write_bytes(Path(evaluate.migration_provenance.__file__).read_bytes())
        self.auditor = self.engine/"analyze_csgb_online_training_migrated.py"
        self.auditor.write_bytes((Path(evaluate.__file__).resolve().parents[1]/"analyze_csgb_online_training_migrated.py").read_bytes())
        self.receipt = self.migrated/"checkpoint_observation.json"
        write_json(self.receipt, {"synthetic_only": True})
        self.receipt_sha = evaluate.sha(self.receipt)
        old = self.root/"seed_202"
        old.rename(self.migrated/"seed_202")
        old.mkdir()
        for name in ("protocol.json", "reference_4096_0.json", "warmup_4096_0_fixed8_fresh.json"):
            write_json(old/name, {"synthetic_failed_attempt": True})
        for name in ("seed_202.log", "controller_pipeline.log"):
            (self.root/name).write_text("synthetic terminal failure; not a performance verdict\n")
        failed = self.prior/"failed_seed202_snapshot.json"
        old_evidence = {str(p): evaluate.sha(p) for p in old.glob("*.json")}
        old_evidence.update({str(self.root/n): evaluate.sha(self.root/n) for n in ("seed_202.log", "controller_pipeline.log")})
        write_json(failed, {"schema": 1, "source_root": str(self.root), "seed": 202,
                   "status": "FAILED_GPU_CONTENTION_NO_PERFORMANCE_VERDICT", "output_reuse": False,
                   "old_done_present": False, "old_frozen_present": False, "observation_utc": "synthetic-only",
                   "file_sha256": old_evidence})
        write_json(self.prior/"recovery_protocol.json", {"schema": 1, "source_root": str(self.root),
                   "failed_attempt_receipt": str(failed), "failed_attempt_receipt_sha256": evaluate.sha(failed)})
        (self.prior/"stage.sha256").write_text("synthetic failed retry stage receipt\n")
        (self.prior/"recovery_pipeline.log").write_text("synthetic failure before output creation\n")
        self.seed_roots = {str(s): str((self.migrated if s == 202 else self.root)/f"seed_{s}") for s in training.SEEDS}
        self.source_roots = {str(s): str(self.migrated if s == 202 else self.root) for s in training.SEEDS}
        self.stages = {str(s): self.new_stage_sha if s == 202 else self.stage_sha for s in training.SEEDS}
        prior_evidence = {**old_evidence, **{str(p): evaluate.sha(p) for p in self.prior.iterdir()}}
        self.plan_path = self.migrated/"migration_protocol.json"
        self.plan = {"schema": 1, "source_root": str(self.root), "source_stage_sha256": self.stage_sha,
                     "migrated_root": str(self.migrated), "migrated_stage_sha256": self.new_stage_sha,
                     "original_runner_sha256": entries[runner], "migrated_runner_sha256": self.new_entries[runner],
                     "seed": 202, "physical_gpu_index": 3,
                     "expected_gpu_uuid": evaluate.migration_provenance.GPU_MAP["202"]["expected_gpu_uuid"],
                     "algorithm_or_seed_change": False, "training_seed_roots": self.seed_roots,
                     "training_analysis_path": str(self.migrated/"training_analysis"),
                     "reuse_partial_output": False, "prior_retry_root": str(self.prior),
                     "prior_retry_output_created": False, "no_best_attempt_or_seed_selection": True,
                     "primary_checkpoint_seed": 101, "performance_claim": False,
                     "development_opened": False, "confirmation_opened": False,
                     "no_automatic_development_or_confirmation_launch": True,
                     "checkpoint_receipt": str(self.receipt), "checkpoint_receipt_sha256": self.receipt_sha,
                     "prior_attempt_sha256": prior_evidence}
        write_json(self.plan_path, self.plan)
        self.plan_sha = evaluate.sha(self.plan_path)
        launch_files = [p for p in self.migrated.rglob("*") if p.is_file() and "seed_202" not in p.parts]
        # The copied complete seed202 is not part of the source-launch receipt.
        for i in range(98-len(launch_files)):
            p = self.migrated/f"launch_metadata_{i}.txt"; p.write_text(f"synthetic launch metadata {i}\n")
            launch_files.append(p)
        assert len(launch_files) == 98
        (self.migrated/"launch.sha256").write_text("".join(f"{evaluate.sha(p)}  {p.relative_to(self.migrated)}\n"
                                                           for p in sorted(launch_files)))
        self.launch_sha = evaluate.sha(self.migrated/"launch.sha256")
        self._generate_requests()
        self.analysis = self.migrated/"training_analysis"
        (self.root/"training_analysis").rename(self.analysis)
        with self.patches():
            proof = evaluate.migration_proof(self.root, expected_stage_sha=self.stage_sha)
            audited = evaluate._audit_completed_training(self.root, proof, self.engine)
        self.audited = audited
        summary = {"seeds": list(training.SEEDS), "smoke": False, "held_out_evaluated": False,
                   "request_counts": {"train": 432, "calibration": 144, "warmup": 6},
                   "migration": audited["migration"]}
        write_json(self.analysis/"summary.json", summary)
        write_json(self.analysis/"request_records.json", audited["records"])
        (self.analysis/"analysis_source.py").write_bytes(self.auditor.read_bytes())
        artifacts = {p.name: evaluate.sha(p) for p in self.analysis.iterdir() if p.name != "manifest.json"}
        write_json(self.analysis/"manifest.json", {"phase_artifacts_complete": True, "study_complete": False,
                   "held_out_evaluated": False, "deployment_tps_measured": False,
                   "raw_roots": [self.seed_roots[str(s)] for s in training.SEEDS],
                   "input_and_source_sha256": audited["input_and_source_sha256"],
                   "analysis_source_sha256": audited["analysis_source_sha256"],
                   "migration": audited["migration"], "artifacts_sha256": artifacts})
        self.assembly_path = self.migrated/"training_assembly.json"
        self.assembly = {"schema": 1, "kind": evaluate.ASSEMBLY_KIND,
                         "source_root": str(self.root), "migrated_root": str(self.migrated),
                         "source_stage_sha256": self.stage_sha, "migrated_stage_sha256": self.new_stage_sha,
                         "source_roots_by_seed": self.source_roots, "stage_sha256_by_seed": self.stages,
                         "seed_roots": self.seed_roots, "analysis_manifest": str(self.analysis/"manifest.json"),
                         "analysis_manifest_sha256": evaluate.sha(self.analysis/"manifest.json"),
                         "migration_plan_path": str(self.plan_path), "migration_plan_sha256": self.plan_sha,
                         "prior_attempt_sha256": prior_evidence,
                         "migration_proof_source_sha256": evaluate.MIGRATION_PROOF_SHA256,
                         "training_auditor_sha256": evaluate.TRAINING_AUDITOR_SHA256,
                         "launch_manifest_sha256": self.launch_sha,
                         "terminal_sha256": {str(Path(p)/n): evaluate.sha(Path(p)/n) for p in self.seed_roots.values()
                                             for n in ("DONE.json", "frozen_models.json")}}
        if publish:
            write_json(self.assembly_path, self.assembly)

    def _generate_requests(self):
        for seed in training.SEEDS:
            child = Path(self.seed_roots[str(seed)])
            gpu_state = evaluate.migration_provenance.GPU_MAP[str(seed)]
            gpu = gpu_state["expected_gpu_uuid"]
            protocol = evaluate.load(child/"protocol.json")
            code = self.migrated if seed == 202 else self.root
            staged = self.new_entries if seed == 202 else {str(Path(n).relative_to(self.root)): h for n, h in self.source_manifest.items()}
            sources = {str(code/n): h for n, h in staged.items() if n.startswith("specextend/") and n.endswith(".py")}
            protocol.update(gpu_uuid=gpu, gpu_name="NVIDIA GeForce RTX 4090", gpu_memory_bytes=48*1024**3,
                            physical_gpu_index=gpu_state["physical_gpu_index"], expected_gpu_uuid=gpu,
                            visible_devices=gpu, source_manifest=sources,
                            runner_sha256=evaluate.sha(code/"specextend/csgb_online_train.py"),
                            private_seeds={str(c): {m: training.private_seed(seed,c,m) for m in training.MODES}
                                           for c in training.CONTEXTS})
            write_json(child/"protocol.json", protocol)
            calibration = []
            for context in training.CONTEXTS:
                write_json(child/f"warmup_{context}_0_fixed8_fresh.json",
                           training_request("fixed8_fresh", context, 0, seed, gpu, "warmup"))
                for doc in range(24):
                    reference = {"output_token_ids": [7]*7+[99]}
                    write_json(child/f"reference_{context}_{doc}.json", reference)
                    row = training_request("fixed8_fresh", context, doc, seed, gpu, "calibration")
                    calibration.append(row)
                    write_json(child/f"calibration_{context}_{doc}_fixed8_fresh.json", row)
            rho = training.calibrate_rho(calibration)
            write_json(child/"rho.json", {"schema": 1, "gpu_uuid": gpu, "by_context": rho,
                       "calibration_sha256": {str(child/f"calibration_{c}_{d}_fixed8_fresh.json"):
                           evaluate.sha(child/f"calibration_{c}_{d}_fixed8_fresh.json") for c in training.CONTEXTS for d in range(24)}})
            docs = list(range(24)); random.Random(seed).shuffle(docs)
            write_json(child/"training_order.json", {"documents": docs, "contexts": list(training.CONTEXTS), "passes": 1})
            models = {}
            for context in training.CONTEXTS:
                models[str(context)] = {}
                for mode in training.MODES:
                    learner = OnlineCSGBPolicy(mode, context=context, seed=training.private_seed(seed,context,mode),
                        rho=rho[str(context)]["rho"], scale=self.scalers[context] if mode == "csgb" else None)
                    for doc in docs:
                        row = training_request(mode, context, doc, seed, gpu, "train", learner=learner, rho=learner.rho)
                        write_json(child/f"train_{context}_{doc}_{mode}.json", row)
                    models[str(context)][mode] = learner.to_state()
            frozen = evaluate.load(child/"frozen_models.json")
            frozen.update(gpu_uuid=gpu, source_manifest=sources, models=models,
                          rho_sha256=evaluate.sha(child/"rho.json"),
                          training_evidence={str(p): evaluate.sha(p) for p in child.glob("*.json")
                                             if p.name not in ("DONE.json", "frozen_models.json")})
            write_json(child/"frozen_models.json", frozen)
            done = evaluate.load(child/"DONE.json")
            done.update(performance_claim=False, frozen_models_sha256=evaluate.sha(child/"frozen_models.json"))
            write_json(child/"DONE.json", done)

    @contextlib.contextmanager
    def patches(self):
        def verify(source, migrated, plan):
            return REAL_HARDWARE_VERIFY(source, migrated, plan, expected_source_root=self.root,
                        expected_migrated_root=self.migrated, expected_source_stage_sha=self.stage_sha)
        with contextlib.ExitStack() as stack:
            for name, value in (("SOURCE_ROOT", self.root), ("MIGRATED_ROOT", self.migrated),
                                ("PRIOR_RETRY_ROOT", self.prior), ("STAGE_SHA256", self.stage_sha),
                                ("MIGRATED_STAGE_SHA256", self.new_stage_sha), ("MIGRATION_PLAN_SHA256", self.plan_sha),
                                ("LAUNCH_SHA256", self.launch_sha), ("CHECKPOINT_OBSERVATION_SHA256", self.receipt_sha)):
                stack.enter_context(mock.patch.object(evaluate, name, value))
            stack.enter_context(mock.patch.object(evaluate.migration_provenance, "verify_hardware_migration", side_effect=verify))
            result = (self.scale_state, self.scalers, self.scale_evidence)
            stack.enter_context(mock.patch.object(evaluate, "validate_scaler", return_value=result))
            stack.enter_context(mock.patch.object(training, "validate_scaler", return_value=result))
            stack.enter_context(mock.patch.object(training, "__file__", str(self.engine/"specextend/csgb_online_train.py")))
            stack.enter_context(mock.patch.object(policy_module, "__file__", str(self.engine/"specextend/csgb_online_policy.py")))
            yield self

    def load(self):
        with self.patches():
            return evaluate.load_frozen_training(self.root, training_assembly=self.assembly_path,
                engine_root=self.engine, expected_stage_sha=self.stage_sha)

    @contextlib.contextmanager
    def preserve(self, *paths):
        before = {Path(p): Path(p).read_bytes() if Path(p).exists() else None for p in paths}
        try:
            yield
        finally:
            for path, data in before.items():
                if data is None:
                    if path.exists() or path.is_symlink():
                        path.unlink()
                else:
                    path.write_bytes(data)

    def resign_raw_attempt(self, seed):
        child = Path(self.seed_roots[str(seed)])
        frozen = evaluate.load(child/"frozen_models.json")
        frozen["training_evidence"] = {str(p): evaluate.sha(p) for p in child.glob("*.json")
                                       if p.name not in ("DONE.json", "frozen_models.json")}
        write_json(child/"frozen_models.json", frozen)
        done = evaluate.load(child/"DONE.json"); done["frozen_models_sha256"] = evaluate.sha(child/"frozen_models.json")
        write_json(child/"DONE.json", done)
        manifest = evaluate.load(self.analysis/"manifest.json")
        manifest["input_and_source_sha256"].update({str(p): evaluate.sha(p) for p in child.glob("*.json")})
        write_json(self.analysis/"manifest.json", manifest)
        assembly = evaluate.load(self.assembly_path)
        assembly["terminal_sha256"].update({str(child/n): evaluate.sha(child/n) for n in ("DONE.json", "frozen_models.json")})
        assembly["analysis_manifest_sha256"] = evaluate.sha(self.analysis/"manifest.json")
        write_json(self.assembly_path, assembly)

    def evaluation_protocol(self):
        method = evaluate.load(Path(evaluate.__file__).resolve().parents[1]/"design/csgb_online_frozen_evaluation_v4.json")
        method.update(training_root=str(self.root), source_root=str(self.root),
                      training_assembly_path=str(self.assembly_path), training_analysis_path=str(self.analysis),
                      training_seed_roots=self.seed_roots, training_source_roots_by_seed=self.source_roots,
                      training_stage_sha256_by_seed=self.stages,
                      migration_protocol_path=str(self.plan_path), migration_protocol_sha256=self.plan_sha,
                      training_launch_manifest_path=str(self.migrated/"launch.sha256"), training_launch_manifest_sha256=self.launch_sha,
                      checkpoint_observation_sha256=self.receipt_sha)
        return method


class MigratedEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.f = MigratedFixture(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_actual_full582_request_cpu_replay_and_genuine_source_mapping(self):
        loaded = self.f.load()
        self.assertEqual(len(loaded["models"]), 18)
        self.assertEqual(loaded["source_roots_by_seed"], self.f.source_roots)
        self.assertEqual(loaded["stage_sha256_by_seed"], self.f.stages)
        self.assertEqual(loaded["seed_roots"], self.f.seed_roots)
        self.assertEqual(len(self.f.audited["records"]), 582)
        for state in loaded["models"].values():
            self.assertGreaterEqual(min(state["counts"]), 32)
        for seed in training.SEEDS:
            child = Path(self.f.seed_roots[str(seed)])
            self.assertEqual(len(list(child.glob("train_*.json"))), 144)
            self.assertEqual(len(list(child.glob("calibration_*.json"))), 48)
            self.assertEqual(len(list(child.glob("warmup_*.json"))), 2)
            self.assertEqual(len(list(child.glob("reference_*.json"))), 48)

    def test_clean_validator_order_and_policy_scope_are_ast_identical_to_v3(self):
        def functions(path):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(Path(path).read_text()).body
                    if isinstance(n, ast.FunctionDef)}
        old, new = functions(previous.__file__), functions(evaluate.__file__)
        for name in ("validate_clean_rounds", "policy_order", "policy_spec", "validate_documents",
                     "checkpoint_runtime_files", "validate_checkpoint_receipt", "validate_physical_gpu"):
            self.assertEqual(old[name], new[name], name)
        self.assertEqual(evaluate.POLICIES, previous.POLICIES)
        for policy in evaluate.POLICIES:
            for context in training.CONTEXTS:
                fixture, _ = clean_fixture(policy, context)
                self.assertTrue(evaluate.validate_clean_rounds(**fixture)["exact"])

    def test_raw_changed_after_freeze_is_rejected(self):
        path = Path(self.f.seed_roots["202"])/"train_4096_0_csgb.json"
        original = path.read_bytes()
        try:
            row = evaluate.load(path); row["rounds"][2]["reward"] += 1
            write_json(path, row)
            with self.assertRaises(ValueError):
                self.f.load()
        finally:
            path.write_bytes(original)

    def test_undertrained_frozen_arm_cannot_be_resigned_into_acceptance(self):
        child = Path(self.f.seed_roots["202"])
        with self.f.preserve(child/"frozen_models.json", child/"DONE.json",
                             self.f.analysis/"manifest.json", self.f.assembly_path):
            frozen = evaluate.load(child/"frozen_models.json")
            frozen["models"]["4096"]["csgb"]["counts"][0] = 0
            write_json(child/"frozen_models.json", frozen)
            self.f.resign_raw_attempt(202)
            with self.assertRaises(ValueError):
                self.f.load()

    def test_self_consistent_resigned_reward_still_fails_real_cpu_replay(self):
        child = Path(self.f.seed_roots["202"])
        request = child/"train_4096_0_csgb.json"
        with self.f.preserve(request, child/"frozen_models.json", child/"DONE.json",
                             self.f.analysis/"manifest.json", self.f.assembly_path):
            row = evaluate.load(request); row["rounds"][2]["reward"] += 1
            write_json(request, row)
            self.f.resign_raw_attempt(202)
            with self.assertRaisesRegex(ValueError, "Reward not based"):
                self.f.load()

    def test_v4_assembly_mapping_terminal_and_kind_substitutions_rejected(self):
        original = copy.deepcopy(self.f.assembly)
        for mutate in (lambda a: a.update(kind="legacy_v3"),
                       lambda a: a["seed_roots"].update({"202": str(self.f.root/"seed_202")}),
                       lambda a: a["source_roots_by_seed"].update({"202": str(self.f.root)}),
                       lambda a: a["stage_sha256_by_seed"].update({"202": self.f.stage_sha}),
                       lambda a: a["terminal_sha256"].update({next(iter(a["terminal_sha256"])): "0"*64}),
                       lambda a: a.update(migration_plan_sha256="0"*64),
                       lambda a: a.update(analysis_manifest_sha256="0"*64)):
            with self.subTest(mutate=mutate), self.f.preserve(self.f.assembly_path):
                changed = copy.deepcopy(original); mutate(changed); write_json(self.f.assembly_path, changed)
                with self.assertRaises(ValueError):
                    self.f.load()

    def test_source_engine_and_all98_launch_dependencies_are_actual_checked_bytes(self):
        paths = (self.f.migrated/"specextend/csgb_online_train.py",
                 self.f.migrated/"csgb_migration_provenance.py",
                 self.f.auditor, self.f.receipt)
        for path in paths:
            with self.subTest(path=path.name), self.f.preserve(path):
                path.write_bytes(path.read_bytes()+b"\nchanged\n")
                with self.assertRaises(ValueError):
                    self.f.load()
        extra = self.f.migrated/"specextend/unrecorded_engine.py"
        with self.f.preserve(extra):
            extra.write_text("# unrecorded runtime source\n")
            with self.assertRaisesRegex(ValueError, "Unstaged"):
                self.f.load()

    def test_both_failed_attempts_and_old_terminal_or_prior_output_are_locked(self):
        for path in (self.f.root/"seed_202.log", self.f.prior/"recovery_pipeline.log",
                     self.f.prior/"failed_seed202_snapshot.json"):
            with self.subTest(path=path.name), self.f.preserve(path):
                path.write_bytes(path.read_bytes()+b"changed")
                with self.assertRaises(ValueError):
                    self.f.load()
        terminal = self.f.root/"seed_202/DONE.json"
        with self.f.preserve(terminal):
            write_json(terminal, {"passed": True})
            with self.assertRaisesRegex(ValueError, "failed seed202"):
                self.f.load()
        new_output = self.f.prior/"seed_202"
        new_output.mkdir()
        try:
            with self.assertRaisesRegex(ValueError, "failed seed202"):
                self.f.load()
        finally:
            new_output.rmdir()

    def test_pending_assembly_and_missing_completed_request_fail_closed(self):
        pending = self.f.assembly_path.with_name("training_assembly.json.pending")
        with self.f.preserve(pending):
            pending.write_text("preserve partial manifest\n")
            with self.assertRaises(ValueError):
                self.f.load()
        request = Path(self.f.seed_roots["202"])/"calibration_8192_23_fixed8_fresh.json"
        with self.f.preserve(request):
            request.unlink()
            with self.assertRaises((ValueError, FileNotFoundError)):
                self.f.load()

    def test_protocol_and_migration_binding_are_crossbound_to_actual_evidence(self):
        method = self.f.evaluation_protocol()
        path = self.f.migrated/"future_evaluation_protocol.json"
        with self.f.preserve(path), self.f.patches():
            write_json(path, method)
            self.assertEqual(evaluate.validate_evaluation_protocol(path, self.f.root), method)
            loaded = self.f.load()
            evaluate.validate_migration_binding(loaded, method)
            for key in ("migration_protocol_sha256", "training_launch_manifest_sha256",
                        "checkpoint_observation_sha256", "development_corpus_sha256"):
                changed = copy.deepcopy(method); changed[key] = "0"*64; write_json(path, changed)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    evaluate.validate_evaluation_protocol(path, self.f.root)
            bad = copy.deepcopy(loaded)
            bad["evidence"].pop(str(self.f.migrated/"launch.sha256"))
            with self.assertRaises(ValueError):
                evaluate.validate_migration_binding(bad, method)

    def test_validate_only_never_imports_cuda_or_creates_output(self):
        source, method_path = self.f.migrated/"synthetic_corpus.jsonl", self.f.migrated/"future_method.json"
        output = self.f.migrated/"must_not_be_created"
        with self.f.preserve(source, method_path), self.f.patches():
            source.write_text("\n".join(json.dumps({"text": t}) for t in self.f.texts+["synthetic dev"]*8))
            method = self.f.evaluation_protocol(); method["development_corpus_sha256"] = evaluate.sha(source)
            write_json(method_path, method)
            argv = ["csgb_online_evaluate_migrated.py", "--training-root", str(self.f.root),
                    "--training-assembly", str(self.f.assembly_path), "--source", str(source),
                    "--base", "synthetic/vicuna-7b-v1.5-16k", "--draft", "synthetic/vicuna-68m", "--out", str(output),
                    "--documents", "24", "25", "--expected-gpu-uuid", evaluate.migration_provenance.GPU_MAP["101"]["expected_gpu_uuid"],
                    "--physical-gpu-index", "0", "--checkpoint-receipt", str(self.f.receipt),
                    "--evaluation-protocol", str(method_path), "--validate-only"]
            real_import = builtins.__import__
            def guarded(name, *args, **kwargs):
                if name.split(".")[0] in {"torch", "numpy", "transformers"}:
                    raise AssertionError("Heavy import in CPU-only preflight: "+name)
                return real_import(name, *args, **kwargs)
            captured = io.StringIO()
            real_load = evaluate.load_frozen_training
            with mock.patch.object(evaluate, "DEVELOPMENT_CORPUS_SHA256", evaluate.sha(source)), \
                    mock.patch.object(evaluate.sys, "argv", argv), \
                    mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": argv[argv.index("--expected-gpu-uuid")+1]}), \
                    mock.patch.object(evaluate, "validate_physical_gpu", return_value=argv[argv.index("--expected-gpu-uuid")+1]), \
                    mock.patch.object(evaluate, "load_frozen_training", side_effect=lambda root, **kw:
                        real_load(root, engine_root=self.f.engine, expected_stage_sha=self.f.stage_sha, **kw)), \
                    mock.patch.object(evaluate, "validate_checkpoint_receipt", return_value=({"scope": "synthetic only"}, {str(i): "a"*64 for i in range(14)})), \
                    mock.patch.object(builtins, "__import__", side_effect=guarded), contextlib.redirect_stdout(captured):
                evaluate.main()
            self.assertFalse(output.exists())
            self.assertTrue(json.loads(captured.getvalue())["read_only_preflight_passed"])
            self.assertFalse(json.loads(captured.getvalue())["cuda_imported"])


if __name__ == "__main__":
    unittest.main()
