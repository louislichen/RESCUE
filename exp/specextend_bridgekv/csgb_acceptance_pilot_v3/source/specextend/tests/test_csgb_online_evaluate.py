"""Synthetic stdlib-only checks; never reads real held-out text or outcomes."""

import builtins
import copy
import contextlib
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import csgb_online_evaluate as evaluate
from csgb_online_harness import OnlineHarness
from csgb_online_policy import FeatureScale, OnlineCSGBPolicy
from csgb_online_train import MODES, SEEDS, private_seed
from test_csgb_online_train import Clock, IDs, Engine


def trained(mode, context, seed):
    scale = FeatureScale(context, (.2, .1, .02), (.1, .05, .01), 24)
    p = OnlineCSGBPolicy(mode, context=context, seed=private_seed(seed, context, mode), rho=2.,
                        scale=scale if mode == "csgb" else None)
    raw = [1., .2, .1] if mode == "history_native" else [1., .5, .25, .125]
    for budget in (8, 16, 24, 32):
        for _ in range(32):
            p.update(raw, budget, 2, .1)
    return p


def clean_fixture(policy="csgb_s101", context=4096, *, overshoot=False, max_one=False):
    mode, seed = evaluate.policy_spec(policy)
    clock, copies = Clock(), []
    model = Engine(clock)

    def to_host(a):
        copies.append(1); clock.advance(.02)
        return list(a)

    def features(a, readable, anchor):
        clock.advance(.02)
        if anchor is None:
            return None
        c, d = sum(a[i] for i in readable)/sum(a), .25
        return [1., c, d, c*d]

    h = OnlineHarness(model, clock=clock, synchronize=lambda: (_ for _ in ()).throw(
        AssertionError("Clean evaluation must not explicitly synchronize a round")),
        attention_to_host=to_host, feature_fn=features)
    learner = None if seed is None else trained(mode, context, seed)
    before = None if learner is None else learner.to_state()
    h.reset(mode, learner, explore=False, clean=True, context=context,
            max_new_tokens=128, eos_token_id=99)
    committed = [0]*context
    counts = [1] if max_one else ([9]*14+[5] if overshoot else [2, 2, 2])
    model.draft(IDs(committed+[7]), 8, .7, 10)
    for j, n in enumerate(counts):
        committed += [7]*n
        terminal = j == len(counts)-1
        if terminal and not overshoot:
            committed[-1] = 99
        model.retrieval_condition = j % 4 == 0
        if model.csgb_attention_every_round or model.retrieval_condition:
            model.attn_scores_final = [1.]*len(committed)
            model.csgb_attention_observed_timestep = j
        else:
            model.attn_scores_final = None
        clock.advance(.05)
        if terminal:
            h.finalize_terminal(IDs(committed))
        else:
            model.draft(IDs(committed+[7]), 8, .7, 10)
    tokens = committed[context:]
    scored = list(counts); scored[-1] -= max(0, len(tokens)-128)
    result = {"output_token_ids": tokens[:128], "total_generated": min(len(tokens), 128),
              "raw_total_generated": len(tokens), "generation_overshoot": max(0, len(tokens)-128),
              "raw_accept_length_list": [n-1 for n in counts], "accept_length_list": [n-1 for n in scored]}
    return {"result": result, "reference": {"output_token_ids": tokens[:128]},
            "rows": h.rows, "pending": h.pending, "context": context, "seconds": clock(),
            "policy": policy, "learner_before": before,
            "learner_after": None if learner is None else learner.to_state(), "eos_token_id": 99}, copies


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def protocol_fixture(source, retry):
    path = Path(evaluate.__file__).resolve().parents[1]/"design"/"csgb_online_frozen_evaluation_v2.json"
    p = evaluate.load(path)
    p.update(runtime_version=3, training_root=str(source), source_root=str(source),
             training_assembly_path=str(retry/"training_assembly.json"),
             training_seed_roots={str(s): str((retry if s == 202 else source)/f"seed_{s}") for s in SEEDS},
             training_analysis_path=str(retry/"training_analysis"),
             recovery_protocol_path=str(retry/"recovery_protocol.json"), recovery_protocol_sha256="a"*64)
    return p


class FrozenFixture:
    """Real tiny files/hash graphs; only external full-scaler audit is mocked.

    Fake training request contents are inert hash evidence here. Their real
    chronological replay is covered by test_csgb_online_train, and a formal
    loader additionally demands the immutable completed training analysis.
    """
    def __init__(self, tmp):
        self.root, self.engine = Path(tmp).resolve()/"training", Path(tmp).resolve()/"evaluation"
        self.root.mkdir(); self.engine.mkdir()
        local = Path(evaluate.__file__).resolve().parent
        sources = {}
        for relative in evaluate.ENGINE_SOURCES:
            old, new = self.root/"specextend"/relative, self.engine/"specextend"/relative
            old.parent.mkdir(parents=True, exist_ok=True); new.parent.mkdir(parents=True, exist_ok=True)
            data = (local/relative).read_bytes()
            old.write_bytes(data); new.write_bytes(data)
            sources[str(old)] = evaluate.sha(old)
        for i in range(33-len(evaluate.ENGINE_SOURCES)):
            relative = f"synthetic_stage_source_{i}.py"
            old, new = self.root/"specextend"/relative, self.engine/"specextend"/relative
            old.write_text(f"# synthetic staged closure source {i}\n"); new.write_bytes(old.read_bytes())
            sources[str(old)] = evaluate.sha(old)
        self.source_manifest = sources
        analyzer = self.root/"analyze_csgb_online_training.py"
        analyzer.write_text("# synthetic completed training audit source\n")
        self.texts = [f"synthetic train document {i}" for i in range(24)]
        text = self.root/"train24.jsonl"
        text.write_text("\n".join(json.dumps({"text": t}) for t in self.texts))
        method = self.root/"online_protocol.json"
        write_json(method, {"schema": 1, "training_documents": list(range(24)), "contexts": [4096, 8192],
                            "maximum_output_tokens": 128, "method": {"seeds": list(SEEDS), "budgets": [8,16,24,32]},
                            "no_development_or_confirmation_generation_in_this_launcher": True})
        stage = [Path(n) for n in sources]+[analyzer, text, method]
        for i in range(42-len(stage)):
            p = self.root/f"staged_fixture_{i}.txt"; p.write_text(f"inert staged fixture {i}\n"); stage.append(p)
        stage_file = self.root/"stage.sha256"
        stage_file.write_text("".join(f"{evaluate.sha(p)}  {p.relative_to(self.root)}\n" for p in stage))
        self.stage_sha = evaluate.sha(stage_file)
        self.scalers = {c: FeatureScale(c, (.2, .1, .02), (.1, .05, .01), 24) for c in (4096, 8192)}
        scale_path = self.root/"scaler.json"
        self.scale_state = {"purpose": "train-only fixed8 full-round scaling",
                            "collection_source_sha256": {str(text): evaluate.sha(text)}}
        write_json(scale_path, self.scale_state)
        self.scale_evidence = {str(text): evaluate.sha(text)}
        inputs = {str(p): evaluate.sha(p) for p in (text, method, scale_path)}
        analysis_inputs = {**sources, **inputs}
        for seed in SEEDS:
            child = self.root/f"seed_{seed}"; child.mkdir()
            gpu = f"GPU-synthetic-{seed}"
            protocol = {**evaluate.SETTINGS, "schema": 1, "phase": "online_train_only", "seed": seed,
                        "documents": list(range(24)), "contexts": [4096,8192], "policies": list(MODES),
                        "budgets": [8,16,24,32], "training_passes": 1, "performance_claim": False,
                        "development_opened": False, "confirmation_opened": False, "gpu_uuid": gpu,
                        "physical_gpu_index": SEEDS.index(seed), "expected_gpu_uuid": gpu, "visible_devices": gpu,
                        "source_manifest": sources, "input_manifest": inputs, "scaler_evidence": self.scale_evidence}
            write_json(child/"protocol.json", protocol)
            for c in (4096,8192):
                for d in range(24):
                    write_json(child/f"reference_{c}_{d}.json", {"fixture": True})
                    write_json(child/f"calibration_{c}_{d}_fixed8_fresh.json", {"fixture": True})
                    for m in MODES:
                        write_json(child/f"train_{c}_{d}_{m}.json", {"fixture": True})
                write_json(child/f"warmup_{c}_0_fixed8_fresh.json", {"fixture": True})
            write_json(child/"training_order.json", {"fixture": True})
            write_json(child/"rho.json", {"gpu_uuid": gpu, "by_context": {str(c): {"rho": 2.} for c in (4096,8192)}})
            evidence = {str(p): evaluate.sha(p) for p in child.glob("*.json")}
            frozen = {"schema": 1, "seed": seed, "gpu_uuid": gpu, "train_documents": list(range(24)),
                      "contexts": [4096,8192], "input_manifest": inputs, "source_manifest": sources,
                      "training_evidence": evidence, "scaler_sha256": evaluate.sha(scale_path),
                      "rho_sha256": evaluate.sha(child/"rho.json"), "development_opened": False,
                      "confirmation_opened": False,
                      "models": {str(c): {m: trained(m,c,seed).to_state() for m in MODES} for c in (4096,8192)}}
            write_json(child/"frozen_models.json", frozen)
            write_json(child/"DONE.json", {"passed": True, "smoke": False, "seed": seed,
                       "calibration_requests": 48, "training_requests": 144, "warmup_requests": 2,
                       "goal_achieved": False, "development_opened": False, "confirmation_opened": False,
                       "frozen_models_sha256": evaluate.sha(child/"frozen_models.json")})
            analysis_inputs.update({str(p): evaluate.sha(p) for p in child.glob("*.json")})
        analysis = self.root/"training_analysis"; analysis.mkdir()
        summary = {"seeds": list(SEEDS), "smoke": False, "held_out_evaluated": False,
                   "request_counts": {"train": 432, "calibration": 144, "warmup": 6}}
        write_json(analysis/"summary.json", summary)
        for stem in ("training_acceptance", "training_request_tps", "training_observed_budget", "training_latency_arm_counts"):
            for ext in ("png", "svg"):
                (analysis/f"{stem}.{ext}").write_bytes(b"synthetic immutable artifact fixture")
        (analysis/"REPORT.md").write_text("Synthetic training-only audit fixture; no result.\n")
        write_json(analysis/"request_records.json", [])
        (analysis/"analysis_source.py").write_bytes(analyzer.read_bytes())
        artifacts = {p.name: evaluate.sha(p) for p in analysis.iterdir()}
        write_json(analysis/"manifest.json", {"phase_artifacts_complete": True, "study_complete": False,
                   "held_out_evaluated": False, "deployment_tps_measured": False,
                   "raw_roots": [str(self.root/f"seed_{s}") for s in SEEDS],
                   "input_and_source_sha256": analysis_inputs,
                   "analysis_source_sha256": {str(analyzer): evaluate.sha(analyzer)}, "artifacts_sha256": artifacts})

    def load(self):
        # Isolate only the external2522-file scaler derivation; all manifests,
        # real fixture files, byte binding, model states and42 stage entries run.
        with mock.patch.object(evaluate, "validate_scaler", return_value=(self.scale_state, self.scalers, self.scale_evidence)):
            return evaluate.load_frozen_training(self.root, engine_root=self.engine, expected_stage_sha=self.stage_sha)


class MixedFixture(FrozenFixture):
    """Move only a synthetic complete202; preserve a separate failed original."""
    def __init__(self, tmp, publish=True):
        super().__init__(tmp)
        self.retry = Path(tmp).resolve()/"retry202"; self.retry.mkdir()
        old, new = self.root/"seed_202", self.retry/"seed_202"
        old.rename(new)
        frozen = evaluate.load(new/"frozen_models.json")
        frozen["training_evidence"] = {str(p): evaluate.sha(p) for p in new.glob("*.json")
                                        if p.name not in {"DONE.json", "frozen_models.json"}}
        write_json(new/"frozen_models.json", frozen)
        done = evaluate.load(new/"DONE.json"); done["frozen_models_sha256"] = evaluate.sha(new/"frozen_models.json")
        write_json(new/"DONE.json", done)
        old.mkdir()
        write_json(old/"protocol.json", evaluate.load(new/"protocol.json"))
        write_json(old/"warmup_4096_0_fixed8_fresh.json", {"failed_attempt_fixture": True})
        for name in ("seed_202.log", "controller_pipeline.log"):
            (self.root/name).write_text("synthetic terminal GPU-contention failure; no performance verdict\n")
        self.failed = self.retry/"failed_seed202_snapshot.json"
        paths = list(old.glob("*.json"))+[self.root/"seed_202.log", self.root/"controller_pipeline.log"]
        write_json(self.failed, {"schema": 1, "source_root": str(self.root), "seed": 202,
                   "status": "FAILED_GPU_CONTENTION_NO_PERFORMANCE_VERDICT", "output_reuse": False,
                   "old_done_present": False, "old_frozen_present": False, "observation_utc": "synthetic-only",
                   "file_sha256": {str(p): evaluate.sha(p) for p in paths}})
        self.recovery = self.retry/"recovery_protocol.json"
        write_json(self.recovery, {"schema": 1, "source_root": str(self.root),
                   "failed_attempt_receipt": str(self.failed), "failed_attempt_receipt_sha256": evaluate.sha(self.failed)})
        analysis = self.retry/"training_analysis"
        (self.root/"training_analysis").rename(analysis)
        am = evaluate.load(analysis/"manifest.json")
        am["raw_roots"] = [str((self.retry if s == 202 else self.root)/f"seed_{s}") for s in SEEDS]
        original = am["input_and_source_sha256"]
        am["input_and_source_sha256"] = {p: h for p, h in original.items() if not p.startswith(str(old)+"/")}
        am["input_and_source_sha256"].update({str(p): evaluate.sha(p) for p in new.glob("*.json")})
        write_json(analysis/"manifest.json", am)
        self.assembly_path = self.retry/"training_assembly.json"
        self.assembly = {"schema": 1, "source_root": str(self.root), "source_stage_sha256": self.stage_sha,
                         "seed_roots": {str(s): str((self.retry if s == 202 else self.root)/f"seed_{s}") for s in SEEDS},
                         "analysis_manifest": str(analysis/"manifest.json"),
                         "analysis_manifest_sha256": evaluate.sha(analysis/"manifest.json"),
                         "terminal_sha256": {str(Path(am["raw_roots"][i])/n): evaluate.sha(Path(am["raw_roots"][i])/n)
                                             for i in range(3) for n in ("DONE.json", "frozen_models.json")},
                         "recovery_receipt": str(self.recovery), "recovery_receipt_sha256": evaluate.sha(self.recovery),
                         "failed_attempt_receipt": str(self.failed), "failed_attempt_receipt_sha256": evaluate.sha(self.failed)}
        if publish:
            write_json(self.assembly_path, self.assembly)

    def load(self):
        with mock.patch.object(evaluate, "validate_scaler", return_value=(self.scale_state, self.scalers, self.scale_evidence)):
            return evaluate.load_frozen_training(self.root, training_assembly=self.assembly_path,
                                                 engine_root=self.engine, expected_stage_sha=self.stage_sha)


class CleanRoundTests(unittest.TestCase):
    def test_all13_policies_and_both_contexts_exact_frozen_clean(self):
        global_rng = random.getstate()
        for policy in evaluate.POLICIES:
            for context in (4096,8192):
                with self.subTest(policy=policy, context=context):
                    f, copies = clean_fixture(policy, context)
                    original = copy.deepcopy(f)
                    out = evaluate.validate_clean_rounds(**f)
                    self.assertTrue(out["exact"])
                    self.assertEqual(out["scored_acceptance"], 1.)
                    self.assertEqual(out["tps"], 6/f["seconds"])
                    self.assertEqual(f, original)
                    self.assertEqual(len(copies), 2 if policy.startswith(("csgb", "constant")) else 0)
        self.assertEqual(random.getstate(), global_rng)

    def test_overshoot_and_eos_one_token_have_no_unused_proposal(self):
        for options in ({"overshoot": True}, {"max_one": True}):
            f, _ = clean_fixture(**options)
            evaluate.validate_clean_rounds(**f)
            self.assertIsNone(f["pending"])
            self.assertEqual(f["rows"][-1]["useful_committed"], 2 if options.get("overshoot") else 1)

    def test_accounting_timing_action_anchor_and_reward_mutations_rejected(self):
        mutations = (
            lambda f: f.update(pending={"unverified": True}),
            lambda f: f["rows"][2].update(reward=1.),
            lambda f: f["rows"][2].update(seconds=.1),
            lambda f: f["rows"][2].update(updated=True),
            lambda f: f["rows"][2].update(update_cpu_seconds=.1),
            lambda f: f["rows"][2].update(actual_draft_nodes=999),
            lambda f: f["rows"][2].update(anchor_timestep=4),
            lambda f: f["rows"][2].update(attention_observed_timestep=2),
            lambda f: f["rows"][2].update(useful_committed=3),
            lambda f: f["rows"][2].update(terminal_commit=False),
            lambda f: f["rows"][2].update(raw_features=[1.]),
            lambda f: f["reference"].update(output_token_ids=[1]),
            lambda f: f.update(seconds=.01),
            lambda f: f["learner_after"]["rng"]["state"].__setitem__(0, 1),
        )
        for i, mutate in enumerate(mutations):
            with self.subTest(mutation=i):
                f, _ = clean_fixture(); mutate(f)
                with self.assertRaises(ValueError):
                    evaluate.validate_clean_rounds(**f)

    def test_history_cannot_use_future_or_copied_attention(self):
        for change in ({"x": [1., .2, .04375]}, {"raw_features": [1., .5, .25, .125]}):
            f, _ = clean_fixture("history_native_s101")
            f["rows"][2].update(change)
            with self.assertRaises(ValueError):
                evaluate.validate_clean_rounds(**f)

    def test_wrong_frozen_action_rejected_without_training_update(self):
        f, _ = clean_fixture()
        r = f["rows"][2]
        r["budget"] = r["actual_draft_nodes"] = 8 if r["budget"] != 8 else 16
        with self.assertRaises(ValueError):
            evaluate.validate_clean_rounds(**f)


class FrozenArtifactTests(unittest.TestCase):
    def test_complete_synthetic_hash_graph_loads_and_does_not_mutate_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = FrozenFixture(tmp)
            before = {str(p): evaluate.sha(p) for p in Path(tmp).rglob("*") if p.is_file()}
            loaded = fixture.load()
            self.assertEqual(len(loaded["models"]), 18)
            self.assertEqual(loaded["training_texts"], fixture.texts)
            self.assertEqual(before, {str(p): evaluate.sha(p) for p in Path(tmp).rglob("*") if p.is_file()})

    def test_original_source_new_copy_raw_or_artifact_tampering_rejected(self):
        for relative in ("training/specextend/csgb_online_harness.py", "evaluation/specextend/classic/model_classic.py",
                         "training/seed_101/train_4096_0_csgb.json", "training/training_analysis/training_request_tps.png"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as tmp:
                fixture = FrozenFixture(tmp)
                with (Path(tmp)/relative).open("ab") as handle:
                    handle.write(b"changed")
                with self.assertRaises(ValueError):
                    fixture.load()

    def test_partial_analysis_or_missing_seed_cannot_open_development(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = FrozenFixture(tmp)
            (fixture.root/"seed_303"/"DONE.json").unlink()
            with self.assertRaises((ValueError, FileNotFoundError)):
                fixture.load()
        with tempfile.TemporaryDirectory() as tmp:
            fixture = FrozenFixture(tmp)
            path = fixture.root/"training_analysis"/"manifest.json"
            value = evaluate.load(path); value["phase_artifacts_complete"] = False; write_json(path, value)
            with self.assertRaises(ValueError):
                fixture.load()

    def test_complete_staged33_source_closure_requires_all_new_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = FrozenFixture(tmp)
            self.assertEqual(len(f.source_manifest), 33)
            evaluate.validate_engine_sources(f.root, f.source_manifest, f.source_manifest)
            for relative in ("run_csgb_controller.py", "classic/attn_bridge.py",
                             "classic/attention_observation.py", "shared/triton_tree_attn.py"):
                with self.subTest(missing=relative):
                    changed = dict(f.source_manifest)
                    changed.pop(str(f.root/"specextend"/relative))
                    with self.assertRaises(ValueError):
                        evaluate.validate_engine_sources(f.root, changed, f.source_manifest)
            changed = {**f.source_manifest, str(f.root/"specextend"/"unrecorded.py"): "0"*64}
            with self.assertRaises(ValueError):
                evaluate.validate_engine_sources(f.root, changed, f.source_manifest)
            changed = dict(f.source_manifest)
            changed[next(iter(changed))] = "0"*64
            with self.assertRaises(ValueError):
                evaluate.validate_engine_sources(f.root, changed, f.source_manifest)


class AssemblyTests(unittest.TestCase):
    def test_mixed_three_seed_assembly_keeps_source_and_old101303_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = MixedFixture(tmp)
            before = {str(p): evaluate.sha(p) for p in f.root.rglob("*") if p.is_file()}
            loaded = f.load()
            self.assertEqual(len(loaded["models"]), 18)
            self.assertEqual(loaded["seed_roots"], f.assembly["seed_roots"])
            self.assertEqual(loaded["assembly_path"], str(f.assembly_path))
            self.assertEqual(loaded["assembly_sha256"], evaluate.sha(f.assembly_path))
            self.assertIn(str(f.failed), loaded["evidence"])
            self.assertNotIn(str(f.root/"seed_202"/"DONE.json"), loaded["evidence"])
            self.assertEqual(before, {str(p): evaluate.sha(p) for p in f.root.rglob("*") if p.is_file()})

    def test_root_terminal_hash_analysis_or_recovery_substitution_rejected(self):
        for change in ("root", "terminal", "analysis", "recovery", "failed-receipt"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                f = MixedFixture(tmp)
                state = copy.deepcopy(f.assembly)
                if change == "root":
                    state["seed_roots"]["202"] = str(f.root/"seed_202")
                elif change == "terminal":
                    state["terminal_sha256"][str(f.retry/"seed_202"/"DONE.json")] = "0"*64
                elif change == "analysis":
                    state["analysis_manifest"] = str(f.root/"training_analysis"/"manifest.json")
                elif change == "recovery":
                    state["recovery_receipt_sha256"] = "0"*64
                else:
                    state["failed_attempt_receipt_sha256"] = "0"*64
                write_json(f.assembly_path, state)
                with self.assertRaises(ValueError):
                    f.load()

    def test_changed_done_failed_log_or_new_old202_json_fail_closed(self):
        for relative in ("retry202/seed_202/DONE.json", "training/seed_202.log", "training/seed_202/unrecorded.json"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as tmp:
                f = MixedFixture(tmp)
                path = Path(tmp)/relative
                path.write_text("changed or unexpected terminal evidence")
                with self.assertRaises(ValueError):
                    f.load()

    def test_assembly_pending_or_fake_reusable_old_completion_rejected(self):
        for change in ("pending", "old-done", "reuse"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                f = MixedFixture(tmp)
                if change == "pending":
                    Path(str(f.assembly_path)+".pending").write_text("incomplete")
                elif change == "old-done":
                    write_json(f.root/"seed_202"/"DONE.json", {"passed": True})
                else:
                    snapshot = evaluate.load(f.failed); snapshot["output_reuse"] = True; write_json(f.failed, snapshot)
                    recovery = evaluate.load(f.recovery); recovery["failed_attempt_receipt_sha256"] = evaluate.sha(f.failed)
                    write_json(f.recovery, recovery)
                    state = evaluate.load(f.assembly_path)
                    state.update(failed_attempt_receipt_sha256=evaluate.sha(f.failed), recovery_receipt_sha256=evaluate.sha(f.recovery))
                    write_json(f.assembly_path, state)
                with self.assertRaises(ValueError):
                    f.load()


class SelectionSourceTests(unittest.TestCase):
    def test_locked_development_or_separate_smoke_only(self):
        self.assertEqual(evaluate.validate_documents([24,25,31]), [24,25,31])
        self.assertEqual(evaluate.validate_documents([0], True), [0])
        for docs, smoke in (([0], False), ([32], False), ([24,24], False), ([True], False),
                            ([24], True), ([0,24], True), ([], False)):
            with self.assertRaises(ValueError):
                evaluate.validate_documents(docs, smoke)

    def test_source_parsing_stops_before_confirmation_and_prefix_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"synthetic48.jsonl"
            texts = [f"synthetic text {i}" for i in range(32)]
            path.write_text("\n".join(json.dumps({"text": t}) for t in texts)+"\nINVALID MUST NOT BE PARSED\n")
            self.assertEqual(evaluate.load_development_source(path, texts[:24]), texts)
            with self.assertRaises(ValueError):
                evaluate.load_development_source(path, ["changed"]+texts[1:24])

    def test_all_seeds_policies_and_order_preserved_without_seed_selection(self):
        self.assertEqual(len(evaluate.POLICIES), 13)
        state = random.getstate()
        order = evaluate.policy_order(24,4096,0)
        self.assertEqual(set(order), set(evaluate.POLICIES))
        self.assertEqual(order, evaluate.policy_order(24,4096,0))
        self.assertNotEqual(order, evaluate.policy_order(24,4096,1))
        self.assertEqual(state, random.getstate())
        for name in ("csgb_s999", "fixed64_native", "best_seed"):
            with self.assertRaises(ValueError):
                evaluate.policy_spec(name)


class ProtocolAndCheckpointTests(unittest.TestCase):
    def test_registered_evaluation_protocol_and_primary_seed_locked(self):
        # This is a method document, never an actual held-out source/result.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()/"training"
            retry = Path(tmp).resolve()/"retry202"
            path = Path(tmp)/"evaluation_protocol.json"
            protocol = protocol_fixture(root, retry)
            write_json(path, protocol)
            self.assertEqual(evaluate.validate_evaluation_protocol(path, root, expected_retry_root=retry), protocol)
            for mutate in (
                lambda p: p.update(primary_checkpoint_seed=202),
                lambda p: p.update(primary_checkpoint_seed=True),
                lambda p: p["development"].update(documents=list(range(32,48))),
                lambda p: p["development"].update(timing_repeats=1),
                lambda p: p["development"].update(no_automatic_confirmation_launch=False),
                lambda p: p["policies"].update(native_fixed=[8,16,24]),
                lambda p: p["development"].update(physical_gpu_indices_for_shards=[0,1,2,3]),
                lambda p: p["training_seed_roots"].update({"202": str(root/"seed_202")}),
                lambda p: p.update(recovery_protocol_path=str(root/"recovery_protocol.json")),
                lambda p: p.update(recovery_protocol_sha256="not-a-valid-sha"),
                lambda p: p.update(recovery_protocol_sha256=True),
                lambda p: p.update(recovery_protocol_sha256="A"*64),
            ):
                changed = copy.deepcopy(protocol); mutate(changed); write_json(path, changed)
                with self.assertRaises(ValueError):
                    evaluate.validate_evaluation_protocol(path, root, expected_retry_root=retry)

            for index, docs in ((0,[24,25]), (3,[26,27]), (4,[28,29]), (7,[30,31])):
                evaluate.validate_worker_assignment(protocol, docs, index)
            with self.assertRaises(ValueError):
                evaluate.validate_worker_assignment(protocol, [24,25], 1)
            with self.assertRaises(ValueError):
                evaluate.validate_worker_assignment(protocol, [28,29], 0)

    def test_self_consistent_assembly_must_match_preregistered_recovery_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = MixedFixture(tmp)
            loaded = f.load()
            protocol = protocol_fixture(f.root, f.retry)
            protocol["recovery_protocol_sha256"] = evaluate.sha(f.recovery)
            evaluate.validate_recovery_binding(loaded, protocol)
            changed = copy.deepcopy(protocol); changed["recovery_protocol_sha256"] = "0"*64
            with self.assertRaises(ValueError):
                evaluate.validate_recovery_binding(loaded, changed)
            changed = copy.deepcopy(loaded)
            changed["assembly"]["recovery_receipt"] = str(f.root/"recovery_protocol.json")
            with self.assertRaises(ValueError):
                evaluate.validate_recovery_binding(changed, protocol)
            changed = copy.deepcopy(loaded)
            changed["evidence"].pop(str(f.recovery))
            with self.assertRaises(ValueError):
                evaluate.validate_recovery_binding(changed, protocol)
            changed = copy.deepcopy(loaded)
            changed["assembly"]["recovery_receipt_sha256"] = "0"*64
            changed["evidence"][str(f.recovery)] = "0"*64  # internally consistent, wrong preregistered receipt
            with self.assertRaises(ValueError):
                evaluate.validate_recovery_binding(changed, protocol)

    def checkpoints(self, tmp):
        root = Path(tmp).resolve()
        base, draft = root/"vicuna-7b-v1.5-16k", root/"vicuna-68m"
        files = {}
        for folder in (base, draft):
            folder.mkdir()
            for name in evaluate.CHECKPOINT_NAMES[folder.name]:
                path = folder/name
                path.write_bytes((folder.name+":"+name+":synthetic").encode())
                files[str(path)] = evaluate.sha(path)
        receipt = {"schema": 1, "training_root": str(root/"training"),
                   "training_stage_sha256": evaluate.STAGE_SHA256,
                   "scope": "Startup observation; NOT a retroactive prelaunch proof",
                   "versions": evaluate.VERSIONS, "file_sha256": files,
                   "no_model_files_modified": True}
        path = root/"checkpoint_receipt.json"; write_json(path, receipt)
        return path, base, draft, root/"training", receipt

    def test_complete14_runtime_files_and_versions_match_startup_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, base, draft, root, receipt = self.checkpoints(tmp)
            for name in ("README.md", "trainer_state.json", "training_args.bin"):
                (base/name).write_text("unused trainer artifact")
            observed, files = evaluate.validate_checkpoint_receipt(path, base, draft, root,
                                                                   current_versions=dict(evaluate.VERSIONS))
            self.assertEqual(observed, receipt)
            self.assertEqual(files, receipt["file_sha256"])
            self.assertEqual(len(files), 14)
            self.assertIn("NOT a retroactive", observed["scope"])

    def test_known_git_lfs_and_hf_download_metadata_are_not_traversed_or_hashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, base, draft, root, receipt = self.checkpoints(tmp)
            metadata = (base/".git"/"config", base/".git"/"lfs"/"objects"/"aa"/"object",
                        base/".gitattributes", draft/".gitattributes",
                        draft/".cache"/"huggingface"/".gitignore",
                        draft/".cache"/"huggingface"/"download"/"config.json.metadata",
                        draft/".cache"/"huggingface"/"download"/"model.safetensors.lock")
            for p in metadata:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("synthetic metadata; its content must never be read")
            real_sha = evaluate.sha

            def guarded_sha(p):
                self.assertNotIn(Path(p), metadata)
                self.assertNotIn(".git", Path(p).parts)
                return real_sha(p)

            real_walk = os.walk
            visited = []

            def guarded_walk(*args, **kwargs):
                for current, dirs, names in real_walk(*args, **kwargs):
                    visited.append(Path(current))
                    self.assertNotIn(".git", Path(current).parts)
                    yield current, dirs, names

            with mock.patch.object(evaluate, "sha", side_effect=guarded_sha), \
                    mock.patch.object(evaluate.os, "walk", side_effect=guarded_walk):
                _, files = evaluate.validate_checkpoint_receipt(path, base, draft, root,
                                                                 current_versions=evaluate.VERSIONS)
            self.assertEqual(files, receipt["file_sha256"])
            self.assertTrue(visited)

    def test_unrecorded_adapter_config_weight_or_hidden_file_still_rejected(self):
        for extra in ("adapter_model.safetensors", "adapter_config.json", ".unrecognized",
                      ".cache/unknown/config.json", ".cache/huggingface/download/adapter_model.safetensors"):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as tmp:
                path, base, draft, root, _ = self.checkpoints(tmp)
                p = draft/extra; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("not recorded")
                with self.assertRaises(ValueError):
                    evaluate.validate_checkpoint_receipt(path, base, draft, root, current_versions=evaluate.VERSIONS)

    def test_changed_missing_or_new_runtime_checkpoint_file_rejected(self):
        for change in ("changed", "missing", "extra"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                path, base, draft, root, _ = self.checkpoints(tmp)
                if change == "changed":
                    (base/"config.json").write_text("changed")
                elif change == "missing":
                    (draft/"tokenizer.model").unlink()
                else:
                    (base/"tokenizer.json").write_text("additional runtime input")
                with self.assertRaises(ValueError):
                    evaluate.validate_checkpoint_receipt(path, base, draft, root,
                                                          current_versions=evaluate.VERSIONS)

    def test_extra_tokenizer_symlink_cannot_deduplicate_into_recorded_runtime_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, base, draft, root, _ = self.checkpoints(tmp)
            (base/"tokenizer.json").symlink_to(base/"tokenizer.model")
            with self.assertRaisesRegex(ValueError, "Runtime checkpoint file symlinks"):
                evaluate.validate_checkpoint_receipt(path, base, draft, root,
                                                      current_versions=evaluate.VERSIONS)

    def test_cross_model_runtime_symlink_is_rejected_before_path_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, base, draft, root, _ = self.checkpoints(tmp)
            (draft/"tokenizer.model").unlink()
            (draft/"tokenizer.model").symlink_to(base/"tokenizer.model")
            with self.assertRaisesRegex(ValueError, "Runtime checkpoint file symlinks"):
                evaluate.validate_checkpoint_receipt(path, base, draft, root,
                                                      current_versions=evaluate.VERSIONS)

    def test_wrong_stage_version_roles_or_receipt_coverage_rejected(self):
        for change in ("stage", "version", "roles", "coverage", "runtime"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                path, base, draft, root, receipt = self.checkpoints(tmp)
                versions = dict(evaluate.VERSIONS)
                if change == "stage":
                    receipt["training_stage_sha256"] = "0"*64
                elif change == "version":
                    receipt["versions"] = {**receipt["versions"], "torch": "wrong"}
                elif change == "roles":
                    base, draft = draft, base
                elif change == "coverage":
                    receipt["file_sha256"].pop(next(iter(receipt["file_sha256"])))
                else:
                    versions["torch"] = "wrong"
                write_json(path, receipt)
                with self.assertRaises(ValueError):
                    evaluate.validate_checkpoint_receipt(path, base, draft, root, current_versions=versions)


class PhysicalGPUAndPreflightTests(unittest.TestCase):
    UUID = "GPU-0dab56d0-2236-4c0e-fc96-fe28081dba9b"

    def test_physical_indices_zero_three_four_seven_map_before_cuda(self):
        for index in (0,3,4,7):
            with self.subTest(index=index):
                query = mock.Mock(return_value=SimpleNamespace(stdout=self.UUID+"\n", stderr=""))
                self.assertEqual(evaluate.validate_physical_gpu(index, self.UUID, query=query), self.UUID)
                query.assert_called_once_with(
                    ["nvidia-smi", "-i", str(index), "--query-gpu=uuid", "--format=csv,noheader"],
                    capture_output=True, text=True, check=True, timeout=15)

    def test_unsupported_empty_ambiguous_and_failed_observations_fail_closed(self):
        for text in ("", "N/A\n", "GPU-ffffffff-ffff-ffff-ffff-ffffffffffff\n", self.UUID+"\n"+self.UUID):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                evaluate.validate_physical_gpu(7, self.UUID,
                    query=mock.Mock(return_value=SimpleNamespace(stdout=text, stderr="")))
        for error in (subprocess.TimeoutExpired("nvidia-smi", 15),
                      subprocess.CalledProcessError(1, "nvidia-smi"), FileNotFoundError("missing query tool")):
            with self.subTest(error=type(error).__name__), self.assertRaises(type(error)):
                evaluate.validate_physical_gpu(4, self.UUID, query=mock.Mock(side_effect=error))
        for index in (-1,8,True):
            with self.assertRaises(ValueError):
                evaluate.validate_physical_gpu(index, self.UUID, query=mock.Mock())

    def test_valid_stdout_with_any_stderr_warning_is_rejected(self):
        for warning in ("warning: observation may be incomplete\n", "\n", " "):
            with self.subTest(warning=warning), self.assertRaises(RuntimeError):
                evaluate.validate_physical_gpu(4, self.UUID, query=mock.Mock(
                    return_value=SimpleNamespace(stdout=self.UUID+"\n", stderr=warning)))
        with self.assertRaises(RuntimeError):
            evaluate.validate_physical_gpu(4, self.UUID, query=mock.Mock(
                return_value=SimpleNamespace(stdout=self.UUID+"\n")))

    def test_validate_only_completes_cpu_checks_without_heavy_import_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            helper = ProtocolAndCheckpointTests()
            receipt, base, draft, training_root, _ = helper.checkpoints(tmp)
            source = Path(tmp)/"synthetic48.jsonl"
            texts = [f"synthetic corpus {i}" for i in range(32)]
            source.write_text("\n".join(json.dumps({"text": text}) for text in texts))
            retry = Path(tmp).resolve()/"retry202"
            p = protocol_fixture(training_root, retry)
            assembly = retry/"training_assembly.json"; write_json(assembly, {"synthetic_loader_isolated": True})
            recovery = retry/"recovery_protocol.json"; write_json(recovery, {"synthetic_loader_isolated": True})
            p["recovery_protocol_sha256"] = evaluate.sha(recovery)
            protocol = Path(tmp)/"eval_protocol.json"; write_json(protocol, p)
            output = Path(tmp)/"must_not_be_created"
            argv = ["csgb_online_evaluate.py", "--training-root", str(training_root), "--source", str(source),
                    "--base", str(base), "--draft", str(draft), "--out", str(output),
                    "--expected-gpu-uuid", self.UUID, "--physical-gpu-index", "4", "--documents", "28", "29",
                    "--checkpoint-receipt", str(receipt), "--evaluation-protocol", str(protocol),
                    "--training-assembly", str(assembly), "--validate-only"]
            frozen = {"models": {i: {} for i in range(18)}, "training_texts": texts[:24],
                      "evidence": {str(recovery): evaluate.sha(recovery)},
                      "assembly": {"recovery_receipt": str(recovery), "recovery_receipt_sha256": evaluate.sha(recovery)},
                      "seed_roots": p["training_seed_roots"], "analysis_root": p["training_analysis_path"]}
            real_import = builtins.__import__

            def no_heavy_import(name, *args, **kwargs):
                if name.split(".")[0] in {"numpy", "torch"}:
                    raise AssertionError("Validate-only must stop before numerical/CUDA imports")
                return real_import(name, *args, **kwargs)

            stdout = io.StringIO()
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(evaluate, "RETRY_ROOT", retry), \
                    mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": self.UUID}), \
                    mock.patch.object(evaluate, "validate_physical_gpu", return_value=self.UUID) as gpu_check, \
                    mock.patch.object(evaluate, "load_frozen_training", return_value=frozen) as frozen_check, \
                    mock.patch.object(evaluate.importlib.metadata, "version", side_effect=lambda n: evaluate.VERSIONS[n]), \
                    mock.patch("builtins.__import__", side_effect=no_heavy_import), contextlib.redirect_stdout(stdout):
                evaluate.main()
            checked = json.loads(stdout.getvalue())
            self.assertTrue(checked["read_only_preflight_passed"])
            self.assertFalse(checked["experiment_launched"])
            self.assertFalse(checked["output_created"])
            self.assertFalse(checked["cuda_imported"])
            self.assertEqual(checked["checkpoint_runtime_files"], 14)
            self.assertFalse(output.exists())
            gpu_check.assert_called_once_with(4, self.UUID)
            frozen_check.assert_called_once_with(str(training_root), training_assembly=str(assembly))


if __name__ == "__main__":
    unittest.main()
