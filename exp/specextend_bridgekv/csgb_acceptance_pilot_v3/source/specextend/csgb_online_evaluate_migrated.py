"""Future DEVELOPMENT-only clean evaluation of all three frozen online seeds.

Preparation, not an experiment result or a launcher authorization.  Heavy
model dependencies are imported only by main().  No confirmation option exists.
Training roots are immutable inputs; a new exclusive evaluation output is used.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

TOP = Path(__file__).resolve().parents[1]
if str(TOP) not in sys.path:
    sys.path.insert(0, str(TOP))
import csgb_migration_provenance as migration_provenance

from csgb_online_policy import BUDGETS, CONTEXTS, OnlineCSGBPolicy
from csgb_online_train import (MODES, SEEDS, exclusive_json, private_seed,
                               validate_raw_x, validate_scaler, validate_training_source)


STAGE_SHA256 = "a288d1a934d9c3540bca94aca5e02a451f90c4d73f1947598144f17663b19fff"
SOURCE_ROOT = migration_provenance.SOURCE_ROOT
MIGRATED_ROOT = migration_provenance.MIGRATED_ROOT
PRIOR_RETRY_ROOT = Path("/storage/dzj/spec_repro/CSGB_online_retry202_20261016_v1")
EVALUATION_ROOT = Path("/storage/dzj/spec_repro/CSGB_online_migrated_development_20261016_v1")
MIGRATION_PLAN_SHA256 = "6beef2478dec8e6e8cb27a211afc90420cb8564a3844f5c7468517510bb8fec2"
MIGRATED_STAGE_SHA256 = "904ef9e448f5f230b5cf4cc69ebbb9a929ffb9ff34e038ef0951df871d7c2e8e"
MIGRATION_PROOF_SHA256 = "53a21a86c8d50e9dfba19d7b32b88deb27104e024d7f8886e55409a78b5955e2"
TRAINING_AUDITOR_SHA256 = "a47492f5e6739848dc3e40d5a255d5755633f95527fbeb730234e080d9513dff"
LAUNCH_SHA256 = "429e6ecdf10e17b8e0b9bd2daa072db038a5269483301622a19271a14aa14718"
CHECKPOINT_OBSERVATION_SHA256 = "62c02665688daea7647ca193cf4c9dcf5029e454caaa624ff267c866cd8b7893"
DEVELOPMENT_CORPUS_SHA256 = "cbb9628e63104a41009824711909c9c80993b3e060a460dc068e3a4e79a67f6c"
ASSEMBLY_KIND = "csgb_online_hardware_migrated_training_v1"
POLICIES = tuple(f"fixed{b}_native" for b in BUDGETS) + tuple(
    f"{mode}_s{seed}" for mode in MODES for seed in SEEDS)
SETTINGS = {"model_seed": 42, "target_dtype": "float32", "draft_dtype": "float16",
            "max_new_tokens": 128, "max_depth": 10, "threshold": .7,
            "cache_refresh_every": 4, "chunk_size": 32, "retrieve_top_k": 32,
            "corrected_root_coverage": True, "terminal_draft_elision": True, "tf32": False}
ENGINE_SOURCES = ("csgb_online_train.py", "csgb_online_policy.py", "csgb_online_harness.py",
                  "csgb_cache_geometry.py", "csgb.py", "baseline_profiler.py",
                  "csgb_corrected_request.py", "csgb_same_state.py",
                  "classic/model_classic.py", "classic/utils_classic.py",
                  "classic/modeling_llama_kv_draft.py", "classic/terminal_commit.py",
                  "shared/modeling_llama_kv_target.py", "shared/kv_cache.py", "shared/opt_tree.py",
                  "run_csgb_controller.py", "classic/attn_bridge.py",
                  "classic/attention_observation.py", "shared/triton_tree_attn.py")
VERSIONS = {"flash-attn": "2.6.3", "numpy": "1.24.4", "torch": "2.4.0+cu121",
            "transformers": "4.41.0", "triton": "3.0.0"}
CHECKPOINT_NAMES = {
    "vicuna-68m": {"config.json", "generation_config.json", "model.safetensors", "special_tokens_map.json",
                   "tokenizer.model", "tokenizer_config.json"},
    "vicuna-7b-v1.5-16k": {"config.json", "generation_config.json", "pytorch_model-00001-of-00002.bin",
        "pytorch_model-00002-of-00002.bin", "pytorch_model.bin.index.json", "special_tokens_map.json",
        "tokenizer.model", "tokenizer_config.json"}}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def state_sha(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def _finite(value, label, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError("Invalid "+label)
    return value


def validate_physical_gpu(physical_index, expected_uuid, *, query=None):
    """Read only the physical index→UUID mapping, before any CUDA allocation."""
    if (type(physical_index) is not int or physical_index not in range(8)
            or not isinstance(expected_uuid, str)
            or re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", expected_uuid) is None):
        raise ValueError("Invalid checked physical GPU identity")
    query = subprocess.run if query is None else query
    response = query(["nvidia-smi", "-i", str(physical_index), "--query-gpu=uuid", "--format=csv,noheader"],
                     capture_output=True, text=True, check=True, timeout=15)
    if getattr(response, "stderr", None) != "":
        raise RuntimeError("GPU identity query emitted stderr; observation is not clean")
    if response.stdout.strip().splitlines() != [expected_uuid]:
        raise RuntimeError("Physical GPU index does not match the expected UUID")
    return expected_uuid


def _manifest(mapping, evidence, cache):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Missing immutable evidence manifest")
    for name, expected in mapping.items():
        if (not isinstance(name, str) or not Path(name).is_absolute()
                or not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None):
            raise ValueError("Invalid manifest path/hash")
        if name in evidence and evidence[name] != expected:
            raise ValueError("Contradictory immutable evidence")
        if name not in cache:
            cache[name] = sha(name)
        if cache[name] != expected:
            raise ValueError("Changed immutable evidence: "+name)
        evidence[name] = expected


def validate_documents(documents, smoke=False):
    if (type(smoke) is not bool or not isinstance(documents, (list, tuple)) or not documents
            or any(type(d) is not int for d in documents) or len(set(documents)) != len(documents)
            or (list(documents) != [0] if smoke else any(not 24 <= d < 32 for d in documents))):
        raise ValueError("Use only unique development documents24..31, or doc0 in separate smoke")
    return list(documents)


def policy_spec(name):
    if name not in POLICIES:
        raise ValueError("Unregistered evaluation policy")
    if name.startswith("fixed"):
        return name, None
    mode, seed = name.rsplit("_s", 1)
    return mode, int(seed)


def policy_order(document, context, repeat):
    if type(document) is not int or context not in CONTEXTS or type(repeat) is not int or repeat not in (-1, 0, 1, 2):
        raise ValueError("Invalid order identity")
    result = list(POLICIES)
    random.Random(918+document*100000+context*10+repeat).shuffle(result)
    return result


def load_development_source(path, training_texts):
    """Hash may cover the full48 source; parsing stops before confirmation32."""
    with Path(path).open() as handle:
        rows = [json.loads(line) for line in itertools.islice(handle, 32)]
    if (len(rows) != 32 or any(not isinstance(r, dict) or not isinstance(r.get("text"), str)
                              or not r["text"] for r in rows)):
        raise ValueError("Need the first32 nonempty text records")
    texts = [r["text"] for r in rows]
    if len(training_texts) != 24 or texts[:24] != training_texts:
        raise ValueError("Source prefix24 does not match immutable training texts")
    return texts


def validate_evaluation_protocol(path, training_root, *, expected_migrated_root=None):
    p = load(path)
    retry = Path(MIGRATED_ROOT if expected_migrated_root is None else expected_migrated_root).resolve()
    source = Path(training_root).resolve()
    roots = {str(s): str((retry if s == 202 else source)/f"seed_{s}") for s in SEEDS}
    expected = {"schema": 1, "runtime_version": 4, "primary_checkpoint_seed": 101, "robustness_seeds": [202,303],
                "no_best_seed_selection": True, "not_an_inference_ensemble": True,
                "migration_protocol_path": str(retry/"migration_protocol.json"),
                "migration_protocol_sha256": MIGRATION_PLAN_SHA256,
                "migration_proof_source_sha256": MIGRATION_PROOF_SHA256,
                "training_auditor_sha256": TRAINING_AUDITOR_SHA256,
                "training_launch_manifest_path": str(retry/"launch.sha256"),
                "training_launch_manifest_sha256": LAUNCH_SHA256,
                "checkpoint_observation_sha256": CHECKPOINT_OBSERVATION_SHA256,
                "development_corpus_sha256": DEVELOPMENT_CORPUS_SHA256,
                "training_source_roots_by_seed": {str(s): str(retry if s == 202 else source) for s in SEEDS},
                "training_stage_sha256_by_seed": {str(s): MIGRATED_STAGE_SHA256 if s == 202 else STAGE_SHA256 for s in SEEDS}}
    settings = {"contexts": list(CONTEXTS), "maximum_output_tokens": 128,
                "target_dtype": "float32", "draft_dtype": "float16", "tf32": False,
                "max_depth": 10, "threshold": .7, "retrieval_chunk_size": 32,
                "retrieve_top_k": 32, "cache_refresh_every_rounds": 4}
    if (any(p.get(k) != v or type(p.get(k)) is not type(v) for k, v in expected.items())
            or source != Path(SOURCE_ROOT).resolve()
            or p.get("evaluation_root") != str(EVALUATION_ROOT)
            or p.get("training_root") != str(Path(training_root).resolve())
            or p.get("source_root") != str(source)
            or p.get("training_assembly_path") != str(retry/"training_assembly.json")
            or p.get("training_seed_roots") != roots
            or p.get("training_analysis_path") != str(retry/"training_analysis")
            or any(p.get("settings", {}).get(k) != v for k, v in settings.items())
            or p.get("policies", {}).get("native_fixed") != list(BUDGETS)
            or p.get("policies", {}).get("frozen_learned") != list(POLICIES[4:])
            or p.get("development", {}).get("documents") != list(range(24,32))
            or p.get("development", {}).get("worker_document_shards") != [[24,25],[26,27],[28,29],[30,31]]
            or p.get("development", {}).get("physical_gpu_indices_for_shards") != [0,3,4,7]
            or p.get("development", {}).get("timing_repeats") != 3
            or p.get("development", {}).get("warmups_per_policy_document_context") != 1
            or p.get("development", {}).get("no_automatic_confirmation_launch") is not True):
        raise ValueError("Evaluation protocol changed settings, primary seed, policies or development scope")
    return p


def validate_migration_binding(frozen, protocol):
    """An internally valid assembly must still match the preregistered receipt."""
    assembly = frozen.get("assembly")
    proof = frozen.get("migration", {})
    path, wanted = protocol["migration_protocol_path"], protocol["migration_protocol_sha256"]
    checks = (("source_roots_by_seed", "training_source_roots_by_seed"),
              ("stage_sha256_by_seed", "training_stage_sha256_by_seed"),
              ("seed_roots", "training_seed_roots"), ("analysis_root", "training_analysis_path"))
    if (not isinstance(assembly, dict) or assembly.get("migration_plan_path") != path
            or assembly.get("migration_plan_sha256") != wanted
            or proof.get("plan_path") != path or proof.get("plan_sha256") != wanted
            or any(frozen.get(k) != protocol.get(v) for k, v in checks)
            or proof.get("source_roots_by_seed") != frozen.get("source_roots_by_seed")
            or proof.get("expected_stage_by_seed") != frozen.get("stage_sha256_by_seed")
            or frozen.get("evidence", {}).get(path) != wanted):
        raise ValueError("Assembly hardware/source provenance differs from frozen v4 protocol")
    for path, wanted in ((MIGRATED_ROOT/"launch.sha256", protocol["training_launch_manifest_sha256"]),
                         (MIGRATED_ROOT/"checkpoint_observation.json", protocol["checkpoint_observation_sha256"]),
                         (Path(migration_provenance.__file__).resolve(), protocol["migration_proof_source_sha256"]),
                         (MIGRATED_ROOT/"analysis_tools/analyze_csgb_online_training_migrated.py", protocol["training_auditor_sha256"])):
        if frozen.get("evidence", {}).get(str(path)) != wanted:
            raise ValueError("Frozen v4 module/launch/checkpoint binding is missing")


def _canonical_path(value):
    if not isinstance(value, str) or not Path(value).is_absolute() or str(Path(value).resolve()) != value:
        raise ValueError("Explicit canonical absolute provenance path required")
    return Path(value)


def _launch_evidence(root):
    path = root/"launch.sha256"
    if path.is_symlink() or sha(path) != LAUNCH_SHA256:
        raise ValueError("Migration launch receipt differs from its registered98-file manifest")
    mapping = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)", line)
        if match is None:
            raise ValueError("Malformed migration launch entry")
        wanted, name = match.groups()
        p = Path(name)
        if (p.is_absolute() or str(p) != name or ".." in p.parts or ".git" in p.parts
                or "\\" in name):
            raise ValueError("Unsafe/duplicate migration launch input")
        actual = _canonical_path(str(root/p))
        if actual.is_symlink() or not actual.is_file() or str(actual) in mapping:
            raise ValueError("Missing/aliased migration launch input")
        mapping[str(actual)] = wanted
    if len(mapping) != 98:
        raise ValueError("Require the complete registered98-file migration launch closure")
    evidence = {}; _manifest(mapping, evidence, {})
    _manifest({str(path): LAUNCH_SHA256}, evidence, {})
    return evidence


def migration_proof(source_root, *, expected_stage_sha=STAGE_SHA256):
    """Read-only hardware proof, full launch closure, and both failed attempts."""
    source = _canonical_path(str(source_root))
    migrated = _canonical_path(str(MIGRATED_ROOT))
    prior = _canonical_path(str(PRIOR_RETRY_ROOT))
    if source != Path(SOURCE_ROOT).resolve():
        raise ValueError("Unregistered original training source root")
    plan_path = migrated/"migration_protocol.json"
    if sha(plan_path) != MIGRATION_PLAN_SHA256:
        raise ValueError("Migration plan differs from the frozen registration")
    proof = migration_provenance.verify_hardware_migration(source, migrated, plan_path)
    if (proof.get("verified") is not True or proof.get("source_stage_sha256") != expected_stage_sha
            or proof.get("migrated_stage_sha256") != MIGRATED_STAGE_SHA256):
        raise ValueError("Hardware/source proof differs from fixed original/migrated stages")
    evidence = {}; cache = {}
    _manifest(proof["evidence"], evidence, cache)
    _manifest(_launch_evidence(migrated), evidence, cache)
    module_sources = {str(Path(migration_provenance.__file__).resolve()): MIGRATION_PROOF_SHA256,
                      str(migrated/"csgb_migration_provenance.py"): MIGRATION_PROOF_SHA256,
                      str(migrated/"analysis_tools/csgb_migration_provenance.py"): MIGRATION_PROOF_SHA256,
                      str(migrated/"analysis_tools/analyze_csgb_online_training_migrated.py"): TRAINING_AUDITOR_SHA256,
                      str(migrated/"checkpoint_observation.json"): CHECKPOINT_OBSERVATION_SHA256}
    _manifest(module_sources, evidence, cache)
    plan = proof["plan"]
    flags = {"reuse_partial_output": False, "prior_retry_root": str(prior),
             "prior_retry_output_created": False, "no_best_attempt_or_seed_selection": True,
             "primary_checkpoint_seed": 101, "performance_claim": False,
             "development_opened": False, "confirmation_opened": False,
             "no_automatic_development_or_confirmation_launch": True,
             "checkpoint_receipt": str(migrated/"checkpoint_observation.json"),
             "checkpoint_receipt_sha256": CHECKPOINT_OBSERVATION_SHA256}
    if any(plan.get(k) != v or type(plan.get(k)) is not type(v) for k, v in flags.items()):
        raise ValueError("Migration plan permits selection/reuse or substitutes prior attempts")
    old = source/"seed_202"
    original_names = {"protocol.json", "reference_4096_0.json", "warmup_4096_0_fixed8_fresh.json"}
    if ({p.name for p in old.glob("*.json")} != original_names
            or any(old.glob("*.pending"))
            or any((old/n).exists() or (old/n).is_symlink() for n in ("DONE.json", "frozen_models.json"))
            or (prior/"seed_202").exists() or (prior/"seed_202").is_symlink()):
        raise ValueError("A failed seed202 attempt changed or unexpectedly created reusable output")
    failed = prior/"failed_seed202_snapshot.json"
    recovery = prior/"recovery_protocol.json"
    old_mapping = {str(old/n) for n in original_names} | {str(source/"seed_202.log"), str(source/"controller_pipeline.log")}
    expected_prior = old_mapping | {str(failed), str(recovery), str(prior/"stage.sha256"), str(prior/"recovery_pipeline.log")}
    prior_mapping = plan.get("prior_attempt_sha256")
    if not isinstance(prior_mapping, dict) or set(prior_mapping) != expected_prior:
        raise ValueError("Both failed attempts need their exact nine immutable evidence paths")
    for name in prior_mapping:
        _canonical_path(name)
    _manifest(prior_mapping, evidence, cache)
    snapshot, recovery_state = load(failed), load(recovery)
    if (type(snapshot.get("schema")) is not int or snapshot["schema"] != 1
            or snapshot.get("source_root") != str(source) or type(snapshot.get("seed")) is not int
            or snapshot["seed"] != 202 or snapshot.get("output_reuse") is not False
            or snapshot.get("status") != "FAILED_GPU_CONTENTION_NO_PERFORMANCE_VERDICT"
            or snapshot.get("old_done_present") is not False or snapshot.get("old_frozen_present") is not False
            or snapshot.get("file_sha256") != {p: prior_mapping[p] for p in old_mapping}
            or type(recovery_state.get("schema")) is not int or recovery_state["schema"] != 1
            or recovery_state.get("source_root") != str(source)
            or recovery_state.get("failed_attempt_receipt") != str(failed)
            or recovery_state.get("failed_attempt_receipt_sha256") != prior_mapping[str(failed)]):
        raise ValueError("Failed-attempt snapshot/recovery chain is not identical and non-reusable")
    proof = {**proof, "evidence": evidence}
    return proof


def validate_training_assembly(path, source_root, *, expected_stage_sha=STAGE_SHA256, _state=None):
    """Reference actual old101/new202/old303 paths; never merge/select attempts."""
    path = _canonical_path(str(Path(path).absolute()))
    pending = path.with_name(path.name+".pending")
    if path != MIGRATED_ROOT/"training_assembly.json" or pending.exists() or pending.is_symlink():
        raise ValueError("Only the fixed, fully published v4 assembly path is allowed")
    if _state is not None and (path.exists() or path.is_symlink()):
        raise FileExistsError("Never replace a published assembly")
    a = load(path) if _state is None else _state
    migration = migration_proof(source_root, expected_stage_sha=expected_stage_sha)
    roots = migration["seed_roots"]
    manifest = MIGRATED_ROOT/"training_analysis/manifest.json"
    terminals = {str(Path(roots[str(s)])/n) for s in SEEDS for n in ("DONE.json", "frozen_models.json")}
    expected = {"schema": 1, "kind": ASSEMBLY_KIND, "source_root": migration["source_root"],
                "migrated_root": migration["migrated_root"], "source_stage_sha256": expected_stage_sha,
                "migrated_stage_sha256": migration["migrated_stage_sha256"],
                "source_roots_by_seed": migration["source_roots_by_seed"],
                "stage_sha256_by_seed": migration["expected_stage_by_seed"], "seed_roots": roots,
                "analysis_manifest": str(manifest), "migration_plan_path": migration["plan_path"],
                "migration_plan_sha256": migration["plan_sha256"],
                "prior_attempt_sha256": migration["plan"]["prior_attempt_sha256"],
                "migration_proof_source_sha256": MIGRATION_PROOF_SHA256,
                "training_auditor_sha256": TRAINING_AUDITOR_SHA256,
                "launch_manifest_sha256": LAUNCH_SHA256}
    if (not isinstance(a, dict) or any(a.get(k) != v or type(a.get(k)) is not type(v) for k, v in expected.items())
            or not isinstance(a.get("terminal_sha256"), dict) or set(a["terminal_sha256"]) != terminals):
        raise ValueError("V4 assembly substituted hardware/code/raw/analysis/failed-attempt provenance")
    for raw_root in roots.values():
        _canonical_path(raw_root)
    evidence = dict(migration["evidence"]); cache = {}
    _manifest(a["terminal_sha256"], evidence, cache)
    _manifest({str(manifest): a.get("analysis_manifest_sha256")}, evidence, cache)
    if _state is None:
        _manifest({str(path): sha(path)}, evidence, cache)
    return {"path": str(path), "state": a, "seed_roots": roots, "analysis": str(manifest.parent),
            "evidence": evidence, "migration": migration}


def validate_worker_assignment(protocol, documents, physical_index, smoke=False):
    if smoke:
        return
    development = protocol["development"]
    assignments = dict(zip(development["physical_gpu_indices_for_shards"], development["worker_document_shards"]))
    if assignments.get(physical_index) != list(documents):
        raise ValueError("Development worker documents must match its frozen physical GPU shard")


def validate_checkpoint_receipt(path, base, draft, training_root, *, current_versions=None):
    """Compare the exact14 runtime files to the observed STARTUP digest receipt.

    This preserves its explicit non-retroactive limitation. Exact known root
    trainer/VCS and HF-download metadata are not runtime inputs; no .git content
    is traversed/read. Other unrecorded files, including hidden files, fail closed.
    Version discovery reads package metadata only; it does not create CUDA.
    """
    receipt = load(path)
    if (type(receipt.get("schema")) is not int or receipt["schema"] != 1
            or receipt.get("training_root") != str(Path(training_root).resolve())
            or receipt.get("training_stage_sha256") != STAGE_SHA256
            or receipt.get("versions") != VERSIONS or receipt.get("no_model_files_modified") is not True
            or "NOT a retroactive" not in str(receipt.get("scope"))):
        raise ValueError("Wrong startup checkpoint observation receipt")
    if current_versions is None:
        current_versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    if current_versions != VERSIONS:
        raise ValueError("Runtime versions differ from observed training startup")
    if Path(base).name != "vicuna-7b-v1.5-16k" or Path(draft).name != "vicuna-68m":
        raise ValueError("Target/Draft checkpoint roles were swapped or substituted")
    selected = set()
    for folder in (Path(base).resolve(), Path(draft).resolve()):
        files = checkpoint_runtime_files(folder)
        expected = {str((folder/name).resolve()) for name in CHECKPOINT_NAMES[folder.name]}
        if files != expected:
            raise ValueError("Missing/additional runtime checkpoint file; never silently drop it")
        selected |= files
    if set(receipt.get("file_sha256", {})) != selected or len(selected) != 14:
        raise ValueError("Checkpoint receipt does not cover the exact14 runtime inputs")
    evidence = {}; _manifest(receipt["file_sha256"], evidence, {})
    return receipt, evidence


def checkpoint_runtime_files(folder):
    """Inventory runtime paths without reading known VCS/download metadata.

    Only root .git is pruned wholesale (including LFS objects). HF's exact
    download subtree admits only *.metadata/*.lock and its parent .gitignore.
    An unrecorded adapter/weight/config anywhere else is still an extra input.
    Unknown symlink directories and runtime-file aliases fail closed. Reject
    runtime symlinks before resolving paths, so set deduplication cannot hide
    an additional runtime input behind an already-recorded file.
    """
    folder = Path(folder)
    files = set()
    root_metadata = {"README.md", "trainer_state.json", "training_args.bin", ".gitattributes", ".git"}

    def walk_error(error):
        raise error

    for current, directories, names in os.walk(folder, topdown=True, followlinks=False, onerror=walk_error):
        current = Path(current)
        for name in list(directories):
            child = current/name
            relative = child.relative_to(folder)
            if relative.parts == (".git",):
                directories.remove(name)  # Never open/traverse .git/config or LFS objects.
            elif child.is_symlink():
                raise ValueError("Unrecognized symlink directory in checkpoint inputs")
        for name in names:
            path = current/name
            relative = path.relative_to(folder)
            if len(relative.parts) == 1 and name in root_metadata:
                continue
            if relative.parts == (".cache", "huggingface", ".gitignore"):
                continue
            if (relative.parts[:3] == (".cache", "huggingface", "download")
                    and path.suffix in {".metadata", ".lock"}):
                continue
            if path.is_symlink():
                raise ValueError("Runtime checkpoint file symlinks are forbidden")
            files.add(str(path.resolve()))
    return files


def _stage(root, expected, evidence, cache):
    path = root/"stage.sha256"
    if sha(path) != expected:
        raise ValueError("Training stage differs from the registered42-file receipt")
    entries = {}
    for line in path.read_text().splitlines():
        wanted, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        p = Path(name)
        if p.is_absolute() or ".." in p.parts or name in entries:
            raise ValueError("Unsafe/duplicate staged source path")
        entries[name] = wanted
    if len(entries) != 42:
        raise ValueError("Incomplete original42-file training stage")
    _manifest({str((root/name).resolve()): value for name, value in entries.items()}, evidence, cache)
    _manifest({str(path.resolve()): expected}, evidence, cache)
    engine = {str((root/name).resolve()): value for name, value in entries.items()
              if Path(name).parts[0] == "specextend" and Path(name).suffix == ".py"}
    if len(engine) != 33:
        raise ValueError("Original training stage must bind all33 engine Python sources")
    return engine


def validate_engine_sources(root, sources, staged_sources):
    required = {str((Path(root)/"specextend"/n).resolve()) for n in ENGINE_SOURCES}
    if not isinstance(sources, dict) or not required <= set(sources) or sources != staged_sources:
        raise ValueError("Engine source manifest must equal the complete original staged Python closure")


def _audit_completed_training(root, migration, engine_root):
    """Execute the registered pure CPU auditor using the original33 replay engine."""
    auditor_path = Path(engine_root)/"analyze_csgb_online_training_migrated.py"
    if auditor_path.is_symlink() or sha(auditor_path) != TRAINING_AUDITOR_SHA256:
        raise ValueError("Missing byte-identical registered training auditor for CPU replay")
    spec = importlib.util.spec_from_file_location("_csgb_v4_registered_training_auditor", auditor_path)
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    audited = auditor.audit_training([migration["seed_roots"][str(s)] for s in SEEDS], False,
        source_root=root, migrated_root=MIGRATED_ROOT, migration_plan=migration["plan_path"])
    if (audited.get("smoke") is not False or audited.get("seeds") != list(SEEDS)
            or len(audited.get("records", [])) != 582):
        raise ValueError("Independent full three-seed training replay was incomplete")
    return audited


def load_frozen_training(training_root, *, training_assembly=None, engine_root=None, expected_stage_sha=STAGE_SHA256,
                         _assembly_state=None):
    """Full old-root SHA audit; copied engine paths may differ, bytes may not.

    All three complete seed roots AND the finished training figures/report audit
    are prerequisites.  No training analysis or GPU work is run by this loader.
    The override arguments permit synthetic filesystem tests; CLI has neither.
    """
    root = _canonical_path(str(Path(training_root).absolute()))
    engine_root = Path(engine_root or Path(__file__).resolve().parents[1]).resolve()
    evidence, cache = {}, {}
    staged_sources = _stage(root, expected_stage_sha, evidence, cache)
    if _assembly_state is not None and training_assembly is None:
        raise ValueError("Unpublished assembly validation needs its explicit future path")
    if training_assembly is None:
        raise ValueError("V4 always requires the explicit migrated training assembly")
    proof = validate_training_assembly(training_assembly, root, expected_stage_sha=expected_stage_sha, _state=_assembly_state)
    migration = proof["migration"]
    _manifest(proof["evidence"], evidence, cache)
    analysis = Path(proof["analysis"])
    am = load(analysis/"manifest.json")
    seed_roots = [Path(proof["seed_roots"][str(s)]) for s in SEEDS]
    if (am.get("phase_artifacts_complete") is not True or am.get("study_complete") is not False
            or am.get("held_out_evaluated") is not False or am.get("deployment_tps_measured") is not False
            or am.get("raw_roots") != [str(p) for p in seed_roots]):
        raise ValueError("Require completed training-only audit and all three frozen seeds")
    _manifest(am.get("input_and_source_sha256"), evidence, cache)
    _manifest(am.get("analysis_source_sha256"), evidence, cache)
    artifacts = am.get("artifacts_sha256", {})
    required = {"REPORT.md", "summary.json", "request_records.json", "analysis_source.py"}
    required |= {f"{name}.{suffix}" for name in ("training_acceptance", "training_request_tps",
                 "training_observed_budget", "training_latency_arm_counts") for suffix in ("png", "svg")}
    if (not isinstance(artifacts, dict) or not required <= set(artifacts)
            or any(Path(n).name != n for n in artifacts)):
        raise ValueError("Training figures/report/raw audit are incomplete")
    _manifest({str((analysis/n).resolve()): h for n, h in artifacts.items()}, evidence, cache)
    _manifest({str((analysis/"manifest.json").resolve()): sha(analysis/"manifest.json")}, evidence, cache)
    summary = load(analysis/"summary.json")
    if (summary.get("seeds") != list(SEEDS) or summary.get("smoke") is not False
            or summary.get("held_out_evaluated") is not False
            or summary.get("request_counts") != {"train": 432, "calibration": 144, "warmup": 6}):
        raise ValueError("Training summary is partial or not the complete three-seed phase")
    # The local replay/inference engine must remain original, including the
    # trainer: the migrated202 hardware-only trainer is never used for replay.
    for name, wanted in staged_sources.items():
        relative = Path(name).relative_to(root)
        _manifest({str((engine_root/relative).resolve()): wanted}, evidence, cache)
    import csgb_online_train as original_training
    import csgb_online_policy as original_policy
    for module, relative in ((original_training, "specextend/csgb_online_train.py"),
                             (original_policy, "specextend/csgb_online_policy.py")):
        if Path(module.__file__).resolve() != (engine_root/relative).resolve():
            raise ValueError("Replay imported a different original-engine path")
    models, training_texts, common_scaler, seen_gpus = {}, None, None, set()
    scale_cache = {}
    for seed, child in zip(SEEDS, seed_roots):
        protocol, done, frozen = (load(child/name) for name in ("protocol.json", "DONE.json", "frozen_models.json"))
        expected = {**SETTINGS, "schema": 1, "phase": "online_train_only", "seed": seed,
                    "documents": list(range(24)), "contexts": list(CONTEXTS), "policies": list(MODES),
                    "budgets": list(BUDGETS), "training_passes": 1,
                    "performance_claim": False, "development_opened": False, "confirmation_opened": False}
        if any(protocol.get(k) != v or type(protocol.get(k)) is not type(v) for k, v in expected.items()):
            raise ValueError("Training configuration differs from frozen evaluation configuration")
        gpu = protocol.get("gpu_uuid")
        gpu_expected = migration["gpu_map"][str(seed)]
        if (not isinstance(gpu, str) or not gpu.startswith("GPU-") or gpu in seen_gpus
                or type(protocol.get("physical_gpu_index")) is not int
                or protocol["physical_gpu_index"] != gpu_expected["physical_gpu_index"]
                or gpu != gpu_expected["expected_gpu_uuid"]
                or protocol.get("expected_gpu_uuid") != gpu or protocol.get("visible_devices") != gpu):
            raise ValueError("Missing/duplicate physical training worker identity")
        seen_gpus.add(gpu)
        if (done.get("passed") is not True or done.get("smoke") is not False or done.get("seed") != seed
                or done.get("calibration_requests") != 48 or done.get("training_requests") != 144
                or done.get("warmup_requests") != 2 or done.get("goal_achieved") is not False
                or done.get("development_opened") is not False or done.get("confirmation_opened") is not False
                or done.get("frozen_models_sha256") != sha(child/"frozen_models.json")):
            raise ValueError("Seed training is not completely frozen")
        request_names = {f"{phase}_{c}_{d}_{m}.json" for c in CONTEXTS for phase, docs, modes in
                         (("train", range(24), MODES), ("calibration", range(24), ("fixed8_fresh",)),
                          ("warmup", (0,), ("fixed8_fresh",))) for d in docs for m in modes}
        names = request_names | {f"reference_{c}_{d}.json" for c in CONTEXTS for d in range(24)}
        names |= {"protocol.json", "rho.json", "training_order.json"}
        actual = {p.name for p in child.glob("*.json")}
        if actual != names | {"DONE.json", "frozen_models.json"}:
            raise ValueError("Incomplete or unexpected full training evidence matrix")
        raw = {str((child/n).resolve()): sha(child/n) for n in names}
        if (frozen.get("schema") != 1 or frozen.get("seed") != seed or frozen.get("gpu_uuid") != gpu
                or frozen.get("train_documents") != list(range(24)) or frozen.get("contexts") != list(CONTEXTS)
                or frozen.get("training_evidence") != raw
                or frozen.get("input_manifest") != protocol.get("input_manifest")
                or frozen.get("source_manifest") != protocol.get("source_manifest")
                or frozen.get("development_opened") is not False or frozen.get("confirmation_opened") is not False):
            raise ValueError("Frozen models have incomplete/substituted provenance")
        _manifest(raw, evidence, cache)
        for name in ("input_manifest", "source_manifest", "scaler_evidence"):
            _manifest(protocol.get(name), evidence, cache)
        code_root = Path(migration["source_roots_by_seed"][str(seed)])
        entries = migration["migrated_stage_entries"] if seed == 202 else migration["source_stage_entries"]
        collected = {str(code_root/n): h for n, h in entries.items() if n.startswith("specextend/") and n.endswith(".py")}
        validate_engine_sources(code_root, protocol["source_manifest"], collected)
        if protocol.get("runner_sha256") != migration["runner_sha_by_seed"][str(seed)]:
            raise ValueError("Training runner is not its registered genuine collection source")
        inputs = protocol["input_manifest"]
        text_paths = [Path(p) for p in inputs if Path(p).suffix == ".jsonl"]
        if len(inputs) != 3 or len(text_paths) != 1:
            raise ValueError("Wrong isolated training inputs")
        texts = validate_training_source(text_paths[0])
        if training_texts is not None and texts != training_texts:
            raise ValueError("Training texts differ between seeds")
        training_texts = texts
        scales = [p for p in inputs if p != str(text_paths[0]) and load(p).get("purpose") == "train-only fixed8 full-round scaling"]
        if len(scales) != 1:
            raise ValueError("Missing unique frozen scaler")
        scale_path = scales[0]
        method = load(next(p for p in inputs if p not in (str(text_paths[0]), scale_path)))
        if (method.get("schema") != 1 or method.get("training_documents") != list(range(24))
                or method.get("contexts") != list(CONTEXTS) or method.get("maximum_output_tokens") != 128
                or method.get("method", {}).get("seeds") != list(SEEDS)
                or method.get("method", {}).get("budgets") != list(BUDGETS)
                or method.get("no_development_or_confirmation_generation_in_this_launcher") is not True):
            raise ValueError("Wrong locked training method input")
        if scale_path not in scale_cache:
            scale_cache[scale_path] = validate_scaler(scale_path)
        scale_state, scalers, scale_evidence = scale_cache[scale_path]
        if (protocol["scaler_evidence"] != scale_evidence or frozen.get("scaler_sha256") != sha(scale_path)
                or (common_scaler is not None and common_scaler != sha(scale_path))
                or [h for p, h in scale_state["collection_source_sha256"].items()
                    if Path(p).name == "train24.jsonl"] != [sha(text_paths[0])]):
            raise ValueError("Scalers/training text evidence differ between seeds")
        common_scaler = sha(scale_path)
        rho = load(child/"rho.json")
        if rho.get("gpu_uuid") != gpu or frozen.get("rho_sha256") != sha(child/"rho.json"):
            raise ValueError("Calibration is not bound to the training worker")
        if set(frozen.get("models", {})) != {str(c) for c in CONTEXTS}:
            raise ValueError("Missing frozen context")
        for c in CONTEXTS:
            if set(frozen["models"][str(c)]) != set(MODES):
                raise ValueError("Missing frozen policy")
            for mode in MODES:
                state = frozen["models"][str(c)][mode]
                p = OnlineCSGBPolicy.from_state(state)
                if ((p.mode, p.context, p.seed, p.rho) != (mode, c, private_seed(seed, c, mode), rho["by_context"][str(c)]["rho"])
                        or min(p.counts) < 32 or state["scale"] != (scalers[c].to_state() if mode == "csgb" else None)):
                    raise ValueError("Unfrozen/undertrained/wrong-context policy or calibration")
                models[c, mode, seed] = state
        terminal_files = {str((child/n).resolve()): sha(child/n) for n in ("DONE.json", "frozen_models.json")}
        if proof is not None and any(proof["state"]["terminal_sha256"].get(p) != h for p, h in terminal_files.items()):
            raise ValueError("Assembly terminal hashes do not identify the selected complete attempt")
        _manifest(terminal_files, evidence, cache)
        for mapping in (raw, terminal_files, protocol["input_manifest"],
                        protocol["source_manifest"], protocol["scaler_evidence"]):
            if any(am["input_and_source_sha256"].get(p) != h for p, h in mapping.items()):
                raise ValueError("Completed training analysis did not audit all frozen raw/source inputs")
    # Bind the new provenance-aware auditor, while its pure validators and
    # statistical/plotting functions remain byte/AST proven to the old engine.
    if sha(analysis/"analysis_source.py") != TRAINING_AUDITOR_SHA256:
        raise ValueError("Completed training audit used substituted analyzer bytes")
    audited = _audit_completed_training(root, migration, engine_root)
    for mapping in (audited["input_and_source_sha256"], audited["analysis_source_sha256"]):
        _manifest(mapping, evidence, cache)
    # The immutable report was rendered against the genuine collection/analysis
    # roots, not an alias of its publisher's future replay-copy paths.
    if am.get("migration") != audited.get("migration") or summary.get("migration") != audited.get("migration"):
        raise ValueError("Completed training analysis omits the registered migration provenance")
    _manifest({str(engine_root/"analyze_csgb_online_training_migrated.py"): TRAINING_AUDITOR_SHA256}, evidence, cache)
    for name, wanted in evidence.items():
        if sha(name) != wanted:
            raise ValueError("Evidence changed during frozen-input audit")
    return {"models": models, "training_texts": training_texts, "evidence": evidence,
            "stage_sha256": expected_stage_sha, "scaler_sha256": common_scaler,
            "assembly": None if proof is None else proof["state"],
            "assembly_path": None if proof is None else proof["path"],
            "assembly_sha256": None if proof is None or _assembly_state is not None else sha(proof["path"]),
            "seed_roots": {str(s): str(p) for s, p in zip(SEEDS, seed_roots)}, "analysis_root": str(analysis),
            "source_roots_by_seed": migration["source_roots_by_seed"],
            "stage_sha256_by_seed": migration["expected_stage_by_seed"], "migration": migration}


def validate_clean_rounds(result, reference, rows, pending, context, seconds, *, policy,
                          learner_before=None, learner_after=None, eos_token_id=None):
    mode, seed = policy_spec(policy)
    _finite(seconds, "complete request seconds", True)
    output = result.get("output_token_ids")
    if (not isinstance(output, list) or not 0 < len(output) <= 128
            or any(type(v) is not int or v < 0 for v in output)
            or output != reference.get("output_token_ids") or result.get("total_generated") != len(output)
            or any(type(result.get(k)) is not int for k in ("total_generated", "raw_total_generated", "generation_overshoot"))):
        raise ValueError("Canonical output/useful numerator differs")
    if eos_token_id is not None and eos_token_id in output and output.index(eos_token_id) != len(output)-1:
        raise ValueError("Tokens after first committed EOS")
    raw, scored = result.get("raw_accept_length_list"), result.get("accept_length_list")
    if (not isinstance(raw, list) or not isinstance(scored, list) or not rows or pending is not None
            or len(raw) != len(rows) or len(scored) != len(rows)
            or any(type(n) is not int or n < 0 for n in raw+scored)
            or raw[:-1] != scored[:-1] or scored[-1] > raw[-1]
            or sum(n+1 for n in raw) != result["raw_total_generated"]
            or sum(n+1 for n in scored) != len(output)
            or result["generation_overshoot"] != result["raw_total_generated"]-len(output)):
        raise ValueError("Incomplete raw/scored verified-round accounting")
    replay = None if learner_before is None else OnlineCSGBPolicy.from_state(learner_before)
    if seed is None:
        if learner_before is not None or learner_after is not None:
            raise ValueError("Native fixed unexpectedly has a learner")
    elif (replay is None or replay.mode != mode or replay.context != context
          or replay.seed != private_seed(seed, context, mode) or min(replay.counts) < 32
          or learner_before != learner_after or state_sha(learner_before) != state_sha(learner_after)):
        raise ValueError("Frozen policy identity/parameters/private RNG changed")
    before_hash = None if replay is None else state_sha(replay.to_state())
    prefix, generated, previous_accept, ema = context+1, 0, 0., 0.
    for j, (r, a, s) in enumerate(zip(rows, raw, scored)):
        x = r.get("x")
        useful = min(generated+a+1, 128)-min(generated, 128)
        anchor = None if j < 2 else 4*((j-2)//4)
        integer_fields = ("proposal", "prefix", "budget", "actual_draft_nodes", "committed",
                          "raw_committed", "useful_committed", "committed_prefix_before", "committed_prefix_after")
        if (any(type(r.get(k)) is not int for k in integer_fields)
                or r["proposal"] != j or r["prefix"] != prefix or r["policy"] != mode or r["context"] != context
                or r["budget"] not in BUDGETS or r["actual_draft_nodes"] != r["budget"]
                or a > min(10, r["budget"]) or r["committed"] != a+1 or r["raw_committed"] != a+1
                or r["useful_committed"] != useful or useful != s+1
                or r["committed_prefix_before"] != prefix-1 or r["committed_prefix_after"] != prefix+a
                or r.get("useful_generated_before") != min(generated, 128)
                or r.get("useful_generated_after") != min(generated+a+1, 128)
                or r.get("raw_x") != x or r.get("anchor_timestep") != anchor
                or r.get("has_real_anchor") is not (j >= 2)
                or r.get("eligible_for_calibration") is not (j >= 2)
                or r.get("terminal_commit") is not (j == len(rows)-1)
                or r.get("seconds") is not None or r.get("reward") is not None
                or r.get("updated") is not False or _finite(r.get("update_cpu_seconds"), "update time") != 0):
            raise ValueError("Clean round/action/readiness/terminal/accounting mismatch")
        if _finite(r.get("decision_cpu_seconds"), "decision wall") < 0:
            raise ValueError("Negative decision wall time")
        if mode in ("csgb", "constant"):
            validate_raw_x(x, mode)
            if ((x is not None) != (j >= 2) or r.get("raw_features") != x
                    or r.get("coverage") != (None if x is None else x[1])
                    or r.get("shift") != (None if x is None else x[2])
                    or r.get("attention_observed_timestep") != (None if j == 0 else j-1)):
                raise ValueError("Stale/fabricated fresh feature")
        else:
            if any(r.get(k) is not None for k in ("raw_features", "coverage", "shift", "attention_observed_timestep")):
                raise ValueError("Native policy unexpectedly extracted fresh CSGB features")
            expected_x = [1., previous_accept/10., ema/10.] if mode == "history_native" and j >= 2 else None
            if x != expected_x:
                raise ValueError("History/startup contains future or incorrect useful acceptance")
        if replay is not None:
            expected_budget = replay.choose(x, explore=False)
            if (r["budget"] != expected_budget or r.get("learner_input") !=
                    (None if x is None else list(replay.transform(x)))):
                raise ValueError("Action/scaling differs from frozen mean-only replay")
        elif r["budget"] != int(mode[5:].split("_")[0]) or r.get("learner_input") is not None:
            raise ValueError("Native fixed arm differs")
        previous_accept = float(max(0, useful-1)); ema = .75*ema+.25*previous_accept
        generated += a+1; prefix += a+1
    if (sum(r["useful_committed"] for r in rows) != len(output)
            or sum(r["decision_cpu_seconds"] for r in rows) > seconds+1e-9
            or (replay is not None and state_sha(replay.to_state()) != before_hash)):
        raise ValueError("Full request/frozen-replay invariants failed")
    return {"exact": True, "scored_acceptance": sum(scored)/len(scored),
            "tps": len(output)/seconds, "frozen_state_sha256": before_hash}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("training-root", "source", "base", "draft", "out", "expected-gpu-uuid",
                "checkpoint-receipt", "evaluation-protocol", "training-assembly"):
        ap.add_argument("--"+key, required=True)
    ap.add_argument("--documents", type=int, nargs="+", required=True)
    ap.add_argument("--physical-gpu-index", type=int, choices=range(8), required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--validate-only", action="store_true", help="Read-only complete CPU preflight; no CUDA import, model, or output creation")
    args = ap.parse_args()
    documents = validate_documents(args.documents, args.smoke)
    if (re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", args.expected_gpu_uuid) is None
            or os.environ.get("CUDA_VISIBLE_DEVICES") != args.expected_gpu_uuid):
        raise ValueError("Bind CUDA to the exact physical UUID checked idle by the launcher")
    validate_physical_gpu(args.physical_gpu_index, args.expected_gpu_uuid)
    out = Path(args.out)
    if out.exists() or out.is_symlink():
        raise FileExistsError("Never overwrite/reuse a partial evaluation root")
    evaluation_protocol = validate_evaluation_protocol(args.evaluation_protocol, args.training_root)
    validate_worker_assignment(evaluation_protocol, documents, args.physical_gpu_index, args.smoke)
    if str(Path(args.training_assembly).resolve()) != evaluation_protocol["training_assembly_path"]:
        raise ValueError("Training assembly path is not the frozen v4 protocol input")
    frozen = load_frozen_training(args.training_root, training_assembly=args.training_assembly)
    validate_migration_binding(frozen, evaluation_protocol)
    if (frozen["seed_roots"] != evaluation_protocol["training_seed_roots"]
            or frozen["analysis_root"] != evaluation_protocol["training_analysis_path"]):
        raise ValueError("Assembly seed/analysis mapping differs from the frozen evaluation protocol")
    if sha(args.source) != evaluation_protocol["development_corpus_sha256"]:
        raise ValueError("Development corpus is not the registered fixed source")
    if (str(Path(args.checkpoint_receipt).resolve()) != str(MIGRATED_ROOT/"checkpoint_observation.json")
            or sha(args.checkpoint_receipt) != evaluation_protocol["checkpoint_observation_sha256"]):
        raise ValueError("Startup checkpoint observation is not the registered receipt")
    texts = load_development_source(args.source, frozen["training_texts"])
    sources = {str(p.resolve()): sha(p) for p in Path(__file__).parent.rglob("*.py")}
    inputs = {str(Path(p).resolve()): sha(p) for p in
              (args.source, args.checkpoint_receipt, args.evaluation_protocol, args.training_assembly)}
    inputs.update(frozen["evidence"])
    checkpoint_receipt, weights = validate_checkpoint_receipt(args.checkpoint_receipt, args.base, args.draft, args.training_root)
    if args.validate_only:
        print(json.dumps({"read_only_preflight_passed": True, "experiment_launched": False,
                          "documents": documents, "contexts": list(CONTEXTS), "policies": len(POLICIES),
                          "primary_checkpoint_seed": evaluation_protocol["primary_checkpoint_seed"],
                          "physical_gpu_index": args.physical_gpu_index, "gpu_uuid": args.expected_gpu_uuid,
                          "frozen_models": len(frozen["models"]), "checkpoint_runtime_files": len(weights),
                          "output_created": False, "cuda_imported": False}))
        return
    import numpy as np
    import torch
    from baseline_profiler import exact_input
    from classic.model_classic import SPModel
    from csgb_corrected_request import current_gpu_uuid, assert_gpu_uncontended
    from csgb_online_harness import OnlineHarness
    from csgb_same_state import release_request_caches

    if torch.cuda.device_count() != 1:
        raise RuntimeError("One visible GPU per complete document-paired worker")
    probe = torch.empty(1, device="cuda:0")
    gpu = current_gpu_uuid()
    if gpu != args.expected_gpu_uuid:
        raise RuntimeError("Actual CUDA device is not the checked physical UUID")
    assert_gpu_uncontended(gpu); del probe
    torch.manual_seed(42); np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = SPModel.from_pretrained(base_model_path=args.base, draft_model_path=args.draft,
        torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map={"": 0}).eval()
    model.base_model.float()
    if {str(p.device) for p in model.parameters()} != {"cuda:0"} or current_gpu_uuid() != gpu:
        raise RuntimeError("All components must remain on the proved worker GPU")
    assert_gpu_uncontended(gpu)
    h = OnlineHarness(model)
    learners = {key: OnlineCSGBPolicy.from_state(state) for key, state in frozen["models"].items()}
    frozen_state_hashes = {key: state_sha(state) for key, state in frozen["models"].items()}
    repeats = 1 if args.smoke else 3
    out.mkdir(exist_ok=False)
    exclusive_json(out/"protocol.json", {"schema": 1, "runtime_version": 4, "phase": "smoke" if args.smoke else "development",
        "documents": documents, "contexts": list(CONTEXTS), "policies": list(POLICIES),
        "seeds": list(SEEDS), "repeats": repeats, "warmups_per_policy_document_context": 1,
        "settings": SETTINGS, "timing": "full request; request-boundary explicit profiler sync only; engine/feature implicit sync remains; no learner updates",
        "order_rule": "Random(918+document*100000+context*10+repeat).shuffle; warmup repeat=-1",
        "gpu_uuid": gpu, "physical_gpu_index": args.physical_gpu_index,
        "expected_gpu_uuid": args.expected_gpu_uuid, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0), "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "source_manifest": sources, "input_manifest": inputs, "checkpoint_manifest": weights,
        "checkpoint_hashes_captured_at_evaluation_not_retroactive_training_weight_proof": True,
        "checkpoint_observation_scope": checkpoint_receipt["scope"], "runtime_versions": VERSIONS,
        "evaluation_protocol": evaluation_protocol, "gpu_contention_boundary_checks": True,
        "training_stage_sha256": frozen["stage_sha256"], "training_scaler_sha256": frozen["scaler_sha256"],
        "training_root": str(Path(args.training_root).resolve()),
        "training_assembly_path": str(Path(args.training_assembly).resolve()),
        "training_assembly": frozen["assembly"], "training_seed_roots": frozen["seed_roots"],
        "training_source_roots_by_seed": frozen["source_roots_by_seed"],
        "training_stage_sha256_by_seed": frozen["stage_sha256_by_seed"],
        "training_analysis_path": frozen["analysis_root"],
        "frozen_model_sha256": {f"{c}_{m}_{s}": value for (c, m, s), value in frozen_state_hashes.items()},
        "confirmation_opened": False, "production_speed_claim": False, "winner_or_seed_selected": False})
    count = 0
    for doc in documents:
        for context in CONTEXTS:
            ids = exact_input(model.tokenizer, texts[doc], context, torch, torch.device("cuda:0"))
            if int(ids.shape[-1]) != context:
                raise RuntimeError("Actual prompt length differs from the context label")
            input_hash = hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
            release_request_caches(model); assert_gpu_uncontended(gpu)
            with torch.inference_mode():
                reference = model.ar_generate(ids, max_new_tokens=128, verbose=False, return_metrics=True)
            assert_gpu_uncontended(gpu)
            exclusive_json(out/f"reference_{context}_{doc}.json", reference)
            for repeat in range(-1, repeats):
                for policy in policy_order(doc, context, repeat):
                    mode, seed = policy_spec(policy)
                    learner = None if seed is None else learners[context, mode, seed]
                    before = None if learner is None else copy.deepcopy(learner.to_state())
                    if learner is not None and state_sha(before) != frozen_state_hashes[context, mode, seed]:
                        raise RuntimeError("Frozen learner changed between clean requests")
                    release_request_caches(model)
                    h.reset(mode, learner=learner, explore=False, clean=True, context=context,
                            max_new_tokens=128, eos_token_id=model.tokenizer.eos_token_id,
                            provenance={"document": doc, "context": context, "policy": policy, "repeat": repeat})
                    assert_gpu_uncontended(gpu)
                    torch.cuda.synchronize(); start = time.perf_counter()
                    with torch.inference_mode():
                        result = model.spgenerate(ids, nodes=8, max_depth=10, threshold=.7,
                            max_new_tokens=128, temperature=0, use_specextend=True,
                            retrieval_chunk_size=32, retrieve_top_k=32, retrieve_every_n_steps=4,
                            retrieval_num_queries=1, exact_target_attention=True, return_output_ids=True,
                            verbose=False, output_result_line=False)
                    torch.cuda.synchronize(); seconds = time.perf_counter()-start
                    assert_gpu_uncontended(gpu)
                    after = None if learner is None else copy.deepcopy(learner.to_state())
                    checked = validate_clean_rounds(result, reference, h.rows, h.pending, context, seconds,
                        policy=policy, learner_before=before, learner_after=after, eos_token_id=model.tokenizer.eos_token_id)
                    phase = "warmup" if repeat < 0 else ("smoke" if args.smoke else "development")
                    exclusive_json(out/f"{phase}_{context}_{doc}_{policy}_{repeat}.json",
                        {"document": doc, "context": context, "policy": policy, "repeat": repeat,
                         "phase": phase,
                         "gpu_uuid": gpu, "uncontended_at_request_boundaries": True,
                         "input_sha256": input_hash, "seconds": seconds, **checked,
                         "result": result, "rounds": h.rows, "pending_is_none": h.pending is None,
                         "learner_before": before, "learner_after": after})
                    count += 1
                    print(json.dumps({"document": doc, "context": context, "policy": policy,
                                      "repeat": repeat, "exact": True, "phase": phase}), flush=True)
    if any(state_sha(learner.to_state()) != frozen_state_hashes[key] for key, learner in learners.items()):
        raise RuntimeError("Frozen parameters/private RNG changed across the full evaluation")
    for name, wanted in {**sources, **inputs, **weights}.items():
        if sha(name) != wanted:
            raise RuntimeError("Immutable evaluation input changed: "+name)
    raw = {str(p.resolve()): sha(p) for p in out.glob("*.json")}
    exclusive_json(out/"DONE.json", {"raw_data_complete": True, "phase": "smoke" if args.smoke else "development",
        "documents": documents, "requests_including_warmups": count, "raw_sha256": raw,
        "confirmation_opened": False, "winner_or_seed_selected": False,
        "study_complete": False, "figures_and_report_still_required": True, "production_speed_claim": False})


if __name__ == "__main__":
    main()
