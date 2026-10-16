"""CPU-only proof of the narrowly authorized seed202 hardware relocation.

No file is created, altered, imported as experiment code, or executed.  The
original42-file stage is pinned; its migrated copy permits exactly two full-line
byte replacements in the trainer, and exactly that trainer SHA in the stage
manifest. Wrapper/launch metadata outside those42 entries is a separate scope.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re


SOURCE_ROOT = Path("/storage/dzj/spec_repro/CSGB_online_allround_20261016_v1")
MIGRATED_ROOT = Path("/storage/dzj/spec_repro/CSGB_online_migrated202_20261016_v1")
PLAN_PATH = MIGRATED_ROOT/"migration_protocol.json"
SOURCE_STAGE_SHA256 = "a288d1a934d9c3540bca94aca5e02a451f90c4d73f1947598144f17663b19fff"
RUNNER_RELATIVE = "specextend/csgb_online_train.py"
RUNNER_REPLACEMENTS = (
    (b"    ap.add_argument('--physical-gpu-index', type=int, choices=(0, 1, 2), required=True)\n",
     b"    ap.add_argument('--physical-gpu-index', type=int, choices=(0, 1, 2, 3), required=True)\n"),
    (b"            or args.physical_gpu_index != SEEDS.index(args.seed)\n",
     b"            or args.physical_gpu_index != {101: 0, 202: 3, 303: 2}[args.seed]\n"),
)
GPU_MAP = {
    "101": {"physical_gpu_index": 0, "expected_gpu_uuid": "GPU-0dab56d0-2236-4c0e-fc96-fe28081dba9b"},
    "202": {"physical_gpu_index": 3, "expected_gpu_uuid": "GPU-5a72b284-e371-652c-c392-fba68b39e44d"},
    "303": {"physical_gpu_index": 2, "expected_gpu_uuid": "GPU-9b4741ce-4366-7033-50a1-8b80d7c2bff9"},
}
SEEDS = ("101", "202", "303")


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(value):
    p = Path(value)
    if not p.is_absolute() or p != p.resolve():
        raise ValueError("Canonical absolute migration paths required; aliases are forbidden")
    return p


def _regular(path):
    if path.is_symlink() or path != path.resolve() or not path.is_file():
        raise ValueError("Missing, partial, nonregular or aliased migration input")
    return path


def expected_migrated_runner(original):
    """Apply only the two agreed full-line byte changes; no formatting pass."""
    if not isinstance(original, bytes):
        raise ValueError("Runner proof requires exact bytes")
    result = original
    for old, new in RUNNER_REPLACEMENTS:
        if original.count(old) != 1 or original.count(new) != 0:
            raise ValueError("Original runner does not have the unique agreed hardware guard")
        result = result.replace(old, new, 1)
    return result


def _stage(root):
    path = _regular(root/"stage.sha256")
    pending = path.with_name(path.name+".pending")
    if pending.exists() or pending.is_symlink():
        raise ValueError("Ambiguous partial stage publication")
    data = path.read_bytes()
    entries, line_names = {}, []
    lines = data.splitlines(keepends=True)
    for line in lines:
        try:
            text = line.decode("utf-8").removesuffix("\n")
        except UnicodeDecodeError as exc:
            raise ValueError("Invalid stage encoding") from exc
        match = re.fullmatch(r"([0-9a-f]{64}) [ *]([^\r\n]+)", text)
        if match is None:
            raise ValueError("Malformed stage entry")
        wanted, name = match.groups()
        relative = PurePosixPath(name)
        normalized = str(relative)
        if (relative.is_absolute() or normalized == "." or ".." in relative.parts
                or ".git" in relative.parts or "\\" in name or "\0" in name or normalized in entries):
            raise ValueError("Unknown, duplicate or unsafe stage entry")
        entries[normalized] = wanted
        line_names.append(normalized)
    if len(entries) != 42 or RUNNER_RELATIVE not in entries:
        raise ValueError("Require the complete original42-entry stage including its trainer")
    engine = {name for name in entries if PurePosixPath(name).parts[0] == "specextend" and name.endswith(".py")}
    if len(engine) != 33:
        raise ValueError("Require all33 original staged engine sources")
    return data, entries, lines, line_names


def _engine_inventory(root, entries):
    """Additional Python engine code cannot escape the staged33-source proof."""
    expected = {name for name in entries if name.startswith("specextend/") and name.endswith(".py")}
    observed = set()
    engine = root/"specextend"
    for directory, subdirs, filenames in os.walk(engine, followlinks=False):
        directory = Path(directory)
        if any((directory/name).is_symlink() or name == ".git" for name in subdirs):
            raise ValueError("Aliased or VCS engine directory is outside the frozen source closure")
        for name in filenames:
            if name.endswith(".py"):
                path = _regular(directory/name)
                observed.add(path.relative_to(root).as_posix())
    if observed != expected:
        raise ValueError("Unstaged or missing Python engine source")


def verify_hardware_migration(source_root, migrated_root, plan_path, *, expected_source_root=None,
                              expected_migrated_root=None, expected_source_stage_sha=None):
    """Verify a fixed original101/migrated202/original303 provenance mapping.

    Overrides exist ONLY for explicit synthetic filesystem tests. Formal
    callers use the three registered constants; there is no network/GPU work.
    Evidence includes both42-file stages, their manifests, and the exact plan.
    It does not assert that any training run completed or permit resume/reuse.
    """
    source, migrated, plan = _path(source_root), _path(migrated_root), _path(plan_path)
    allowed_source = _path(SOURCE_ROOT if expected_source_root is None else expected_source_root)
    allowed_migrated = _path(MIGRATED_ROOT if expected_migrated_root is None else expected_migrated_root)
    source_stage_sha = SOURCE_STAGE_SHA256 if expected_source_stage_sha is None else expected_source_stage_sha
    if (source != allowed_source or migrated != allowed_migrated or source == migrated
            or plan != migrated/"migration_protocol.json"
            or not isinstance(source_stage_sha, str) or re.fullmatch(r"[0-9a-f]{64}", source_stage_sha) is None):
        raise ValueError("Unregistered migration roots, plan or original stage pin")
    _regular(plan)
    if plan.with_name(plan.name+".pending").exists() or plan.with_name(plan.name+".pending").is_symlink():
        raise ValueError("Migration plan is only partially published")
    source_data, source_entries, source_lines, source_names = _stage(source)
    if sha_bytes(source_data) != source_stage_sha:
        raise ValueError("Original42 stage is not the immutable registered receipt")
    migrated_data, migrated_entries, _, _ = _stage(migrated)
    if set(migrated_entries) != set(source_entries):
        raise ValueError("Migrated stage has unknown/missing entries")
    _engine_inventory(source, source_entries)
    _engine_inventory(migrated, migrated_entries)
    runner_before = source_entries[RUNNER_RELATIVE]
    runner_after = migrated_entries[RUNNER_RELATIVE]
    evidence = {str(source/"stage.sha256"): source_stage_sha,
                str(migrated/"stage.sha256"): sha_bytes(migrated_data)}
    for relative, wanted in source_entries.items():
        left, right = _regular(source/relative), _regular(migrated/relative)
        original, copied = left.read_bytes(), right.read_bytes()
        if sha_bytes(original) != wanted or sha_bytes(copied) != migrated_entries[relative]:
            raise ValueError("Staged source bytes differ from their declared digest")
        if relative == RUNNER_RELATIVE:
            if copied != expected_migrated_runner(original):
                raise ValueError("Trainer differs beyond the two precise hardware-only replacements")
        elif original != copied or migrated_entries[relative] != wanted:
            raise ValueError("One of the other41 immutable files changed")
        evidence[str(left)] = wanted
        evidence[str(right)] = migrated_entries[relative]
    expected_manifest = b"".join(runner_after.encode()+line[64:] if name == RUNNER_RELATIVE else line
                                  for line, name in zip(source_lines, source_names))
    if migrated_data != expected_manifest:
        raise ValueError("Migrated42 manifest changed more than the single trainer digest")
    plan_bytes = plan.read_bytes()
    state = json.loads(plan_bytes)
    roots = {s: str((migrated if s == "202" else source)/f"seed_{s}") for s in SEEDS}
    expected = {"schema": 1, "source_root": str(source), "source_stage_sha256": source_stage_sha,
                "migrated_root": str(migrated), "migrated_stage_sha256": sha_bytes(migrated_data),
                "original_runner_sha256": runner_before, "migrated_runner_sha256": runner_after,
                "seed": 202, "physical_gpu_index": 3, "expected_gpu_uuid": GPU_MAP["202"]["expected_gpu_uuid"],
                "algorithm_or_seed_change": False, "training_seed_roots": roots,
                "training_analysis_path": str(migrated/"training_analysis")}
    if not isinstance(state, dict) or any(state.get(k) != v or type(state.get(k)) is not type(v)
                                          for k, v in expected.items()):
        raise ValueError("Migration plan substituted algorithm/seed/hardware/source provenance")
    evidence[str(plan)] = sha_bytes(plan_bytes)
    for name, wanted in evidence.items():
        if sha_file(_regular(Path(name))) != wanted:
            raise ValueError("Migration evidence changed during verification")
    return {"verified": True, "scope": "hardware/source delta only; not training completion or efficacy",
            "source_root": str(source), "migrated_root": str(migrated), "plan_path": str(plan),
            "plan_sha256": evidence[str(plan)], "plan": state, "evidence": evidence,
            "source_stage_sha256": source_stage_sha, "migrated_stage_sha256": sha_bytes(migrated_data),
            "original_runner_sha256": runner_before, "migrated_runner_sha256": runner_after,
            "source_stage_entries": source_entries, "migrated_stage_entries": migrated_entries,
            "source_roots_by_seed": {s: str(migrated if s == "202" else source) for s in SEEDS},
            "expected_stage_by_seed": {s: sha_bytes(migrated_data) if s == "202" else source_stage_sha for s in SEEDS},
            "runner_sha_by_seed": {s: runner_after if s == "202" else runner_before for s in SEEDS},
            "gpu_map": {s: dict(GPU_MAP[s]) for s in SEEDS}, "seed_roots": roots,
            "training_analysis_path": str(migrated/"training_analysis"), "algorithm_or_seed_change": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--migrated-root", required=True)
    parser.add_argument("--migration-plan", required=True)
    args = parser.parse_args(argv)
    try:
        result = verify_hardware_migration(args.source_root, args.migrated_root, args.migration_plan)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
