"""Receive a disabled, whole-inventory SHA-pinned capsule; never launch it."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile


ROOT = Path('/mnt/8T/cl/rescue_experiment_cache/fair_union_confirmation_20261020_v1')
STATUS = 'STAGED_DISABLED_NOT_LAUNCHED'
MANIFEST = 'LAUNCH_MANIFEST.json'
PIPELINE = 'exp/draft_graph/run_fair_union_confirmation_20261020_v1.sh'
DISABLED_LAUNCHER = (b'#!/usr/bin/env bash\n'
    b'# Unexecuted protocol template: deliberately has no state-changing branch.\n'
    b'set -Eeuo pipefail\n'
    b"printf '%s\\n' 'DISABLED: both baseline studies, retained union, fresh-data rule and final sources must be independently verified and frozen before any confirmation launch.' >&2\n"
    b'exit 64\n')
FORBIDDEN_PARTS = {'.git', '.ssh', '__pycache__', '.env'}
FORBIDDEN_SUFFIXES = {'.safetensors', '.bin', '.pt', '.pth', '.ckpt', '.gguf', '.pem', '.key', '.p12', '.pfx'}
EVIDENCE_SUFFIXES = {'.py', '.json', '.jsonl', '.md', '.png', '.svg', '.sh', '.log', '.txt'}
EVIDENCE_SPECIAL_NAMES = {'evidence_snapshot.tar.gz', 'terminal.tar.gz', 'collection.stderr'}
NEW_SOURCE = {
    'rescue/fair_union_confirmation_data.py', 'rescue/fair_union_confirmation.py',
    'rescue/fair_union_confirmation_report.py', 'tests/test_rescue_fair_union_confirmation_data.py',
    'tests/test_rescue_fair_union_confirmation.py', 'tests/test_rescue_fair_union_confirmation_report.py',
    'exp/draft_graph/fair_union_confirmation_20261020_preregistration.json', PIPELINE,
    'AGENTS.md', 'docs/PAPER_HANDOFF.md', 'exp/draft_graph/build_fair_union_stage.py',
    'exp/draft_graph/receive_fair_union_stage.py', 'tests/test_rescue_fair_union_stage.py',
    'tests/test_rescue_fair_union_confirmation_disabled.py', 'tests/test_rescue_fair_confirmation_data.py',
    'tests/test_rescue_draft_runtime.py', 'tests/test_rescue_system.py',
    'exp/draft_graph/fair_confirmation_20261020_preregistration.json'}


def require(ok, message):
    if not ok: raise RuntimeError(message)


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def relative(name):
    require(isinstance(name, str) and '\\' not in name and '\x00' not in name, 'Non-POSIX capsule path')
    path = PurePosixPath(name)
    require(path.parts and path.as_posix() == name and not path.is_absolute() and '..' not in path.parts and
            not FORBIDDEN_PARTS.intersection(path.parts) and path.suffix.lower() not in FORBIDDEN_SUFFIXES,
            'Unsafe/non-canonical capsule path')
    require(path.parts[0] in ('after', 'stage1_evidence', 'stage2_evidence') or name == MANIFEST,
            'Unregistered capsule top-level member')
    return path


def validate_manifest(manifest, contents):
    require(manifest.get('schema') == 1 and manifest.get('status') == STATUS and manifest.get('remote_root') == str(ROOT) and
            manifest.get('launcher_disabled') is True and manifest.get('launch_authorized') is False and all(manifest.get(k) is True for k in
            ('no_new_confirmation_opened', 'no_model_weights_or_credentials_transferred', 'no_shared_runtime_change', 'no_training')),
            'Only the exact disabled preparation capsule is accepted')
    proof = manifest.get('union_freeze', {}); m, r = manifest.get('M'), manifest.get('R')
    require(proof.get('status') == 'UNION_VERIFIED' and type(m) is int and 8 <= m <= 16 and type(r) is int and 0 <= r <= 8 and
            (proof.get('M'), proof.get('R')) == (m, r), 'Invalid frozen UNION multiplicity')
    choices = proof.get('choices', {})
    require(isinstance(choices, dict) and set(choices) == {'4096', '8192'} and
            all(isinstance(v, dict) and 4 <= len(v) <= 8 for v in choices.values()) and
            sum(len(v) for v in choices.values()) == m, 'M does not count the retained context/mode contrasts')
    extra = []
    for length in (4096, 8192):
        arms = list(choices[str(length)].values())
        require(all(isinstance(a, dict) and mode == a.get('mode') for mode, a in choices[str(length)].items()),
                'Retained mode/arm mismatch')
        seen = {(a['top_chunks'], a['fixed_k']) for a in arms if a.get('family') == 'retrieval' and a.get('backend') == 'eager'}
        for arm in arms:
            if arm.get('family') != 'retrieval' or arm.get('backend') == 'eager': continue
            geometry = arm['top_chunks'], arm['fixed_k']
            if geometry not in seen:
                extra.append(dict(length=length, top_chunks=geometry[0], fixed_k=geometry[1])); seen.add(geometry)
    require(len(extra) == r and proof.get('extra_eager_reference_count') == r and proof.get('extra_eager_reference_geometries') == extra,
            'R does not count the frozen missing matching-eager geometries')
    expected = dict(reference_requests=32, own_quality_requests=64, baseline_quality_requests=m,
        matching_quality_requests=r, prime_requests=m+2, timing_requests=32*(m+2), owners=m+2)
    require(manifest.get('expected_counts') == proof.get('expected_counts') == expected and
            manifest.get('expected_requests') == proof.get('scientific_requests') == 162+34*m+r,
            'Incorrect variable-M/R UNION count')
    inventory = manifest.get('file_sha256', {})
    require(isinstance(inventory, dict) and inventory and set(contents) == set(inventory) | {MANIFEST},
            'Unknown/missing archive members')
    for name, expected_sha in inventory.items():
        relative(name)
        require(name != MANIFEST and digest(expected_sha) and hashlib.sha256(contents[name]).hexdigest() == expected_sha,
                'Whole-inventory artifact SHA mismatch')
    sources = manifest.get('source_sha256', {})
    require(isinstance(sources, dict) and sources and {n for n in contents if n.startswith('after/')} == {'after/'+n for n in sources},
            'Unlisted/missing staged source')
    for name, expected_sha in sources.items():
        relative('after/'+name)
        require(digest(expected_sha) and inventory.get('after/'+name) == expected_sha, 'Source inventory SHA mismatch')
    scientific = manifest.get('scientific_source_sha256', {})
    require(isinstance(scientific, dict) and scientific and all(sources.get(n) == s for n, s in scientific.items()) and
            len([n for n in scientific if n.startswith('exp/specextend_bridgekv/specextend/') and n.endswith('.py')]) == 27,
            'Complete scientific source/27-tree map missing')
    require(contents.get('after/'+PIPELINE) == DISABLED_LAUNCHER, 'Capsule launcher is not hard-disabled')
    arguments = manifest.get('proof_arguments', {})
    require(isinstance(arguments, dict) and set(arguments) == {s+'_'+f for s in ('stage1', 'stage2') for f in ('results', 'analysis', 'verified', 'manifest')},
            'Exactly eight portable dual-proof paths required')
    for stage in ('stage1', 'stage2'):
        prefix = stage+'_evidence/'
        for field in ('results', 'analysis', 'verified', 'manifest'):
            name = arguments[stage+'_'+field]; relative(name)
            require(name.startswith(prefix), 'Portable proof escapes its exact stage root')
            require(name in contents if field in ('verified', 'manifest') else any(n.startswith(name+'/') for n in contents),
                    'Missing portable study proof path')
        require(arguments[stage+'_manifest'] == prefix+'FINAL_ARTIFACTS_MANIFEST.json' and
                arguments[stage+'_verified'] == prefix+'VERIFIED_COMPLETE.json', 'Nonstandard portable seal filenames')
        seal = json.loads(contents[arguments[stage+'_manifest']]); sealed = seal.get('file_sha256', {})
        require(isinstance(sealed, dict) and sealed, 'Empty portable final artifact manifest')
        allowed = {prefix+n for n in sealed} | {arguments[stage+'_manifest'], arguments[stage+'_verified']}
        require({n for n in contents if n.startswith(prefix)} == allowed, 'Unlisted/missing portable evidence file')
        for name, expected_sha in sealed.items():
            part = relative(prefix+name)
            require(part.suffix.lower() in EVIDENCE_SUFFIXES or part.name in EVIDENCE_SPECIAL_NAMES,
                    'Unregistered portable evidence type')
            require(digest(expected_sha) and inventory.get(prefix+name) == expected_sha, 'Portable final evidence SHA mismatch')
        pinned = proof.get(stage, {})
        require(inventory[arguments[stage+'_manifest']] == pinned.get('final_manifest_sha256') and
                inventory[arguments[stage+'_verified']] == pinned.get('verified_complete_sha256'), 'Portable seal differs from verified UNION')
    p2_name = arguments['stage2_results']+'/protocol.json'
    require(p2_name in contents, 'Missing sealed stage2 source protocol')
    old_sources = json.loads(contents[p2_name]).get('source_sha256', {})
    require(isinstance(old_sources, dict) and {'rescue/engine.py', 'rescue/draft_graph.py', 'rescue/fair_confirmation_data.py'} <= set(old_sources) and
            set(sources) == set(old_sources) | NEW_SOURCE and all(sources.get(n) == s for n, s in old_sources.items()),
            'Missing/changed stage2 source or unregistered new source')


def receive(expected, stream=None):
    require(digest(expected), 'Invalid transport SHA')
    payload = (sys.stdin.buffer if stream is None else stream).read()
    require(hashlib.sha256(payload).hexdigest() == expected, 'Transport SHA mismatch; no root was created')
    require(not ROOT.exists(), 'Never overwrite or clean a partial/existing isolated root')
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
        # Reject pax/global metadata rather than interpreting renamed members.
        require(not archive.pax_headers, 'PAX metadata is forbidden')
        members = archive.getmembers(); names, contents, cursor = set(), {}, 0
        require(not archive.pax_headers, 'Global PAX metadata is forbidden')
        for entry in members:
            path = relative(entry.name)
            require(entry.offset == cursor and entry.offset_data == entry.offset+512,
                    'Non-contiguous/hidden archive headers are forbidden')
            require(entry.isfile() and not entry.pax_headers and entry.name not in names and
                    (ROOT/entry.name).resolve().is_relative_to(ROOT.resolve()),
                    'Link/special/PAX/duplicate archive member; no root was created')
            archive.fileobj.seek(entry.offset+156)
            require(archive.fileobj.read(1) in (b'0', b'\x00'),
                    'Extended/sparse/renaming archive headers are forbidden')
            names.add(entry.name)
            with archive.extractfile(entry) as handle: contents[entry.name] = handle.read()
            require(len(contents[entry.name]) == entry.size, 'Truncated archive member')
            archive.fileobj.seek(entry.offset_data+entry.size)
            padding = archive.fileobj.read((-entry.size) % 512)
            require(not any(padding), 'Nonzero bytes outside the artifact inventory')
            cursor = entry.offset_data+((entry.size+511)//512)*512
        if members:
            archive.fileobj.seek(cursor)
            trailing = archive.fileobj.read()
            require(len(trailing) >= 1024 and not any(trailing),
                    'Hidden/trailing archive data outside the artifact inventory')
        for name in names:
            path = PurePosixPath(name)
            require(not any(PurePosixPath(*path.parts[:i]).as_posix() in names for i in range(1, len(path.parts))),
                    'Archive file-ancestor collision; no root was created')
        require(MANIFEST in contents, 'Missing capsule manifest')
        validate_manifest(json.loads(contents[MANIFEST]), contents)
    # ALL archive members and every source/evidence byte were checked first.
    ROOT.mkdir()  # exclusive; any race/partial output is preserved, not cleaned
    for name, raw in contents.items():
        destination = ROOT/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as handle: handle.write(raw)
        destination.chmod(0o444)
    for name, raw in contents.items():
        require((ROOT/name).read_bytes() == raw, 'Staged artifact changed; preserve partial root')
    receipt = dict(status=STATUS, root=str(ROOT), transport_sha256=expected, files=len(contents),
        launch_manifest_sha256=hashlib.sha256(contents[MANIFEST]).hexdigest(), launcher_disabled=True, launch_authorized=False)
    print(json.dumps(receipt))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--sha256', required=True)
    receive(parser.parse_args().sha256)
