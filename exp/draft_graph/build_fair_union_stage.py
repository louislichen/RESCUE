"""Build a disabled UNION capsule locally, only from two verified studies.

This is not a launcher and never opens candidate corpus bytes or model assets.
The complete immutable scientific runtime comes from stage2's saved source,
not the working checkout. Only explicitly named new code/templates are added.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile


REMOTE_ROOT = '/mnt/8T/cl/rescue_experiment_cache/fair_union_confirmation_20261020_v1'
STATUS = 'STAGED_DISABLED_NOT_LAUNCHED'
MANIFEST = 'LAUNCH_MANIFEST.json'
PIPELINE = 'exp/draft_graph/run_fair_union_confirmation_20261020_v1.sh'
DISABLED_LAUNCHER = (b'#!/usr/bin/env bash\n'
    b'# Unexecuted protocol template: deliberately has no state-changing branch.\n'
    b'set -Eeuo pipefail\n'
    b"printf '%s\\n' 'DISABLED: both baseline studies, retained union, fresh-data rule and final sources must be independently verified and frozen before any confirmation launch.' >&2\n"
    b'exit 64\n')
EXTRA_SOURCE = (
    'AGENTS.md', 'docs/PAPER_HANDOFF.md',
    'exp/draft_graph/build_fair_union_stage.py',
    'exp/draft_graph/receive_fair_union_stage.py',
    'tests/test_rescue_fair_union_stage.py',
    'tests/test_rescue_fair_union_confirmation_disabled.py',
    # The synthetic UNION tests import this fixture and its retired template.
    # Including source is not permission to run the retired disabled protocol.
    'tests/test_rescue_fair_confirmation_data.py',
    'tests/test_rescue_draft_runtime.py', 'tests/test_rescue_system.py',
    'exp/draft_graph/fair_confirmation_20261020_preregistration.json')
CORE_SOURCE = ('rescue/engine.py', 'rescue/config.py', 'rescue/attention.py',
    'rescue/draft_graph.py', 'rescue/baselines.py', 'rescue/query_mapper.py',
    'rescue/step_budget.py', 'rescue/fair_confirmation_data.py',
    'rescue/baseline_horizon_data.py',
    'exp/attention_horizon/selection.py', 'exp/selective_kv/bridge.py')
EVIDENCE_SUFFIXES = {'.py', '.json', '.jsonl', '.md', '.png', '.svg', '.sh', '.log', '.txt'}
EVIDENCE_SPECIAL_NAMES = {'evidence_snapshot.tar.gz', 'terminal.tar.gz', 'collection.stderr'}
FORBIDDEN_PARTS = {'.git', '.ssh', '__pycache__', '.env'}
FORBIDDEN_SUFFIXES = {'.safetensors', '.bin', '.pt', '.pth', '.ckpt', '.gguf', '.pem', '.key', '.p12', '.pfx'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(name):
    require(isinstance(name, str) and '\\' not in name and '\x00' not in name,
            'Non-POSIX capsule path')
    path = PurePosixPath(name)
    require(path.parts and path.as_posix() == name and not path.is_absolute() and
            '..' not in path.parts and not FORBIDDEN_PARTS.intersection(path.parts),
            'Unsafe/non-canonical capsule path: '+name)
    require(path.suffix.lower() not in FORBIDDEN_SUFFIXES, 'Weights/credentials are not capsule artifacts')
    return Path(*path.parts)


def safe_file(root, name):
    part = relative(name)
    path = root/part
    require(path.is_file() and not any(p.is_symlink() for p in [path, *path.parents] if p != root.parent) and
            path.resolve().is_relative_to(root), 'Missing/unsafe regular file: '+name)
    return path


def source_name(name):
    part = relative(name)
    return (part.suffix == '.py' and (
        (part.parent == Path('rescue')) or
        (part.parent == Path('tests') and part.name.startswith('test_rescue')) or
        name.startswith('exp/specextend_bridgekv/specextend/')) or
        name in ('exp/attention_horizon/selection.py', 'exp/selective_kv/bridge.py') or
        (part.parent == Path('exp/draft_graph') and part.suffix in ('.py', '.json', '.sh')))


def load_data(repo):
    """Use the exact local stdlib gate, not a foreign installed namespace."""
    sys.path.insert(0, str(repo))
    module = importlib.import_module('rescue.fair_union_confirmation_data')
    require(Path(module.__file__).resolve() == repo/'rescue/fair_union_confirmation_data.py',
            'Foreign UNION gate namespace')
    return module


def matrix(proof):
    m, r = proof.get('M'), proof.get('R')
    require(proof.get('status') == 'UNION_VERIFIED' and type(m) is int and 8 <= m <= 16 and
            type(r) is int and 0 <= r <= 8, 'Complete retained UNION multiplicity required')
    counts = dict(reference_requests=32, own_quality_requests=64, baseline_quality_requests=m,
        matching_quality_requests=r, prime_requests=m+2, timing_requests=32*(m+2), owners=m+2)
    require(proof.get('expected_counts') == counts and proof.get('scientific_requests') == 162+34*m+r,
            'Frozen UNION request counts changed')
    return counts


def arguments(values):
    names = {stage+'_'+field for stage in ('stage1', 'stage2')
             for field in ('results', 'analysis', 'verified', 'manifest')}
    require(set(values) == names and all(isinstance(p, (str, Path)) and Path(p).is_absolute() for p in values.values()),
            'Exactly eight explicit absolute study-proof paths required')
    return {key: Path(value).resolve() for key, value in values.items()}


def evidence_plan(values, stage):
    manifest, verified = values[stage+'_manifest'], values[stage+'_verified']
    root = manifest.parent
    require(manifest.name == 'FINAL_ARTIFACTS_MANIFEST.json' and verified == root/'VERIFIED_COMPLETE.json',
            'Only the standard complete sealed study roots may be staged')
    inventory = json.loads(manifest.read_text()).get('file_sha256', {})
    require(isinstance(inventory, dict) and inventory, 'Empty final evidence inventory')
    files = dict(inventory)
    require(manifest.name not in files and verified.name not in files, 'Evidence seal contains a circular self pin')
    files[manifest.name], files[verified.name] = sha(manifest), sha(verified)
    plan = {}
    for name, digest in files.items():
        part = relative(name)
        require(part.suffix.lower() in EVIDENCE_SUFFIXES or part.name in EVIDENCE_SPECIAL_NAMES,
                'Unregistered binary/evidence type: '+name)
        source = safe_file(root, name)
        require(isinstance(digest, str) and len(digest) == 64 and sha(source) == digest,
                'Sealed evidence changed before staging: '+name)
        plan[stage+'_evidence/'+name] = (source, digest)
    portable = {}
    for field in ('results', 'analysis', 'verified', 'manifest'):
        path = values[stage+'_'+field]
        require(path.is_relative_to(root), 'Proof path escapes its sealed root')
        portable[stage+'_'+field] = stage+'_evidence/'+path.relative_to(root).as_posix()
    return plan, portable


def build(repo, output, **study_arguments):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    values = arguments(study_arguments)
    protected = (repo, *(values[s+'_manifest'].parent for s in ('stage1', 'stage2')))
    require(not output.exists() and not any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected),
            'Fresh local capsule must be disjoint from repository and sealed studies')
    data = load_data(repo)
    # The real offline dual proof is mandatory before the FIRST output mkdir.
    # No corpus, model, old mutable remote root, SSH or GPU path is accessed.
    proof = data.validate_union_freeze(**values)
    counts = matrix(proof)
    p2 = json.loads((values['stage2_results']/'protocol.json').read_text())
    old_sources = p2.get('source_sha256', {})
    require(isinstance(old_sources, dict) and set(CORE_SOURCE) <= set(old_sources),
            'Complete saved stage2 scientific runtime required')
    tree = {n: s for n, s in old_sources.items() if n.startswith(data.old.TREE_PREFIX)}
    require(len(tree) == 27, 'Exactly27 frozen archived tree source files required')
    source_plan = {}
    for name, digest in old_sources.items():
        require(source_name(name), 'Stage2 inventory contains a non-source artifact: '+name)
        source = safe_file(values['stage2_results']/'source', name)
        require(sha(source) == digest, 'Stage2 saved source changed: '+name)
        source_plan[name] = (source, digest)
    new_names = set(data.REQUIRED_NEW_SOURCE) | set(EXTRA_SOURCE)
    for name in sorted(new_names):
        source = safe_file(repo, name)
        digest = sha(source)
        if name in source_plan:
            require(digest == source_plan[name][1], 'Current checkout changed a preserved stage2 source: '+name)
        else:
            source_plan[name] = (source, digest)
    require(safe_file(repo, PIPELINE).read_bytes() == DISABLED_LAUNCHER,
            'Only the exact hard-disabled UNION launcher can be packaged by this preparation tool')
    template = data.read_json(safe_file(repo, data.REQUIRED_NEW_SOURCE[-2]))
    require(template == data.protocol_template(), 'Unexecuted UNION template differs from the frozen rule')
    evidence, portable = {}, {}
    for stage in ('stage1', 'stage2'):
        plan, paths = evidence_plan(values, stage)
        evidence.update(plan); portable.update(paths)
    # Every validation above precedes even a partial local capsule directory.
    output.mkdir()
    payload = output/'payload'
    for name, (original, digest) in {**{'after/'+n: v for n, v in source_plan.items()}, **evidence}.items():
        destination = payload/relative(name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as handle:
            handle.write(original.read_bytes())
        require(sha(original) == digest == sha(destination), 'Artifact changed during capsule copy: '+name)
    portable_proof = data.validate_union_freeze(**{k: payload/v for k, v in portable.items()})
    require(portable_proof == proof, 'Portable dual-study proof differs')
    scientific_sources = data._sources(payload/'after', old_sources)
    source_hashes = {name: digest for name, (_, digest) in sorted(source_plan.items())}
    inventory = {p.relative_to(payload).as_posix(): sha(p) for p in sorted(payload.rglob('*')) if p.is_file()}
    manifest = dict(schema=1, status=STATUS, created_at_utc=datetime.now(timezone.utc).isoformat(),
        remote_root=REMOTE_ROOT, launcher_disabled=True, launch_authorized=False,
        union_freeze=proof, M=proof['M'], R=proof['R'], expected_counts=counts,
        expected_requests=proof['scientific_requests'], proof_arguments=portable,
        source_sha256=source_hashes, scientific_source_sha256=scientific_sources,
        file_sha256=inventory, no_new_confirmation_opened=True,
        no_model_weights_or_credentials_transferred=True, no_shared_runtime_change=True, no_training=True,
        scope='Disabled source/evidence capsule only; no final data gate, preparation, deployment or experiment launch')
    with (payload/MANIFEST).open('x') as handle:
        # The retained group/stage order determines the matching-reference
        # list. Do not alphabetize nested arm dictionaries while transporting.
        json.dump(manifest, handle, indent=2); handle.write('\n')
    for path in payload.rglob('*'):
        if path.is_file(): path.chmod(0o444)
    transport = output/'transport.tar.gz'
    with tarfile.open(transport, 'w:gz', format=tarfile.USTAR_FORMAT) as archive:
        for path in sorted(payload.rglob('*')):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(payload).as_posix(), recursive=False)
    receipt = dict(status=STATUS, remote_root=REMOTE_ROOT, transport=str(transport), transport_sha256=sha(transport),
        launch_manifest_sha256=sha(payload/MANIFEST), files=len(inventory)+1, source_files=len(source_hashes),
        M=proof['M'], R=proof['R'], expected_counts=counts, expected_requests=proof['scientific_requests'],
        launcher_disabled=True, launch_authorized=False)
    with (output/'TRANSPORT_RECEIPT.json').open('x') as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True); handle.write('\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'output', *(s+'-'+f for s in ('stage1', 'stage2') for f in ('results', 'analysis', 'verified', 'manifest'))):
        parser.add_argument('--'+name, type=Path, required=True)
    args = vars(parser.parse_args())
    print(json.dumps(build(**args), indent=2))
