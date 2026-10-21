"""Build an immutable, fresh stage capsule without any remote operations."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def build(repo, stage1, output):
    repo, stage1, output = (Path(p).resolve() for p in (repo, stage1, output))
    require(not output.exists(), 'Fresh capsule directory only')
    require(not any(output == p or output.is_relative_to(p) or p.is_relative_to(output)
                    for p in (repo, stage1)), 'Capsule must not overlap frozen evidence/repository')
    sys.path.insert(0, str(repo))
    from rescue.fair_confirmation_data import validate_baseline_freeze
    from rescue.baseline_horizon import source_inventory, validate_preregistration
    proof = validate_baseline_freeze(stage1/'results', stage1/'results/analysis_final_v1',
        stage1/'VERIFIED_COMPLETE.json', stage1/'FINAL_ARTIFACTS_MANIFEST.json')
    old = json.loads((stage1/'results/protocol.json').read_text())
    prereg = json.loads((repo/'exp/draft_graph/baseline_horizon_20261020_preregistration.json').read_text())
    validate_preregistration(prereg)
    output.mkdir()
    payload = output/'payload'
    after = payload/'after'
    copies = [(p, p.relative_to(repo)) for p in (repo/'rescue').glob('*.py')]
    copies += [(p, p.relative_to(repo)) for p in (repo/'tests').glob('test_rescue*.py')]
    for name in ('AGENTS.md', 'docs/PAPER_HANDOFF.md', 'exp/attention_horizon/selection.py',
        'exp/selective_kv/bridge.py'):
        copies.append((repo/name, Path(name)))
    copies += [(p, p.relative_to(repo)) for p in (repo/'exp/draft_graph').iterdir()
               if p.is_file() and p.suffix in ('.py', '.json', '.sh')]
    prefix = 'exp/specextend_bridgekv/specextend/'
    tree = {name: digest for name, digest in old['source_sha256'].items() if name.startswith(prefix)}
    require(len(tree) == 27, 'Exactly the frozen27 tree files required')
    for name, digest in tree.items():
        original = stage1/'after'/name
        require(original.is_file() and sha(original) == digest, 'Frozen archived tree changed')
        copies.append((original, Path(name)))
    for original, relative in copies:
        require(original.is_file() and not original.is_symlink(), 'Missing/unsafe stage source')
        destination = after/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        require(not destination.exists(), 'Duplicate stage target: '+str(relative))
        shutil.copyfile(original, destination)
    scientific_sources = source_inventory(after, old)
    verified_manifest = json.loads((stage1/'FINAL_ARTIFACTS_MANIFEST.json').read_text())
    evidence_files = dict(verified_manifest['file_sha256'])
    evidence_files['FINAL_ARTIFACTS_MANIFEST.json'] = proof['final_manifest_sha256']
    evidence_files['VERIFIED_COMPLETE.json'] = proof['verified_complete_sha256']
    for name, digest in evidence_files.items():
        relative = Path(name)
        original = stage1/relative
        require(not relative.is_absolute() and '..' not in relative.parts and
                original.resolve().is_relative_to(stage1) and not original.is_symlink() and
                original.is_file() and sha(original) == digest, 'Unsafe/unverified stage1 evidence')
        target = payload/'stage1_evidence'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        require(sha(target) == digest, 'Stage1 whitelist copy differs')
    portable = payload/'stage1_evidence'
    require(validate_baseline_freeze(portable/'results', portable/'results/analysis_final_v1',
        portable/'VERIFIED_COMPLETE.json', portable/'FINAL_ARTIFACTS_MANIFEST.json') == proof,
        'Portable verified stage1 evidence changed')
    source_map = {str(p.relative_to(after)): sha(p) for p in sorted(after.rglob('*')) if p.is_file()}
    manifest = dict(schema=1, status='STAGED_NOT_LAUNCHED',
        created_at_utc=datetime.now(timezone.utc).isoformat(), remote_root=prereg['remote_root'],
        source_sha256=source_map, scientific_source_sha256=scientific_sources,
        stage1_verified=proof, expected_requests=2540, no_new_confirmation_opened=True,
        no_shared_runtime_change=True, no_training=True)
    (payload/'LAUNCH_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for path in payload.rglob('*'):
        if path.is_file():
            path.chmod(0o444)
    transport = output/'transport.tar.gz'
    with tarfile.open(transport, 'w:gz') as archive:
        for path in sorted(payload.rglob('*')):
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(payload)), recursive=False)
    receipt = dict(status='STAGED_NOT_LAUNCHED', transport=str(transport),
        transport_sha256=sha(transport), launch_manifest_sha256=sha(payload/'LAUNCH_MANIFEST.json'),
        stage_files=len(source_map), scientific_sources=len(scientific_sources),
        remote_root=prereg['remote_root'], expected_requests=2540)
    (output/'TRANSPORT_RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'stage1', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.repo, args.stage1, args.output), indent=2))
