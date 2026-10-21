"""Seal a locally audited PARTIAL snapshot; never writes a final-result receipt."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(root, rerun):
    root, rerun = Path(root).resolve(), Path(rerun).resolve()
    destination = root / 'LOCAL_PARTIAL_ARTIFACTS_MANIFEST.json'
    if destination.exists():
        raise RuntimeError('Do not overwrite a previously sealed snapshot')
    snapshot = json.loads((root / 'LIVE_SNAPSHOT.json').read_text())
    assert snapshot['status'] == 'PARTIAL' and snapshot['read_only'] is True
    assert snapshot['atomic_across_files'] is False
    for name, receipt in snapshot['files'].items():
        path = root / name
        assert path.resolve().is_relative_to(root) and path.is_file()
        assert digest(path) == receipt['sha256'], name
    analysis = root / 'results/analysis_partial_v1'
    summary = json.loads((analysis / 'summary.json').read_text())
    assert summary['status'] == 'PARTIAL'
    assert (analysis / 'summary.json').read_bytes() == (rerun / 'summary.json').read_bytes()
    assert not (analysis / 'VERIFIED_WINNERS.json').exists()
    assert not (root / 'VERIFIED_COMPLETE.json').exists()
    audit = json.loads((root / 'INDEPENDENT_PARTIAL_AUDIT.json').read_text())
    assert audit['status'] == 'PARTIAL_AUDIT_PASSED'
    render = json.loads((analysis / 'RENDER_INSPECTION.json').read_text())
    assert len(render['rendered_pngs_inspected']) == 7
    assert (root / 'PROGRESS.md').is_file()
    paths = sorted(p for p in root.rglob('*') if p.is_file())
    manifest = dict(status='PARTIAL', scientific_experiment_complete=False,
        sealed_at_utc=datetime.now(timezone.utc).isoformat(),
        standalone_summary_exact_rerun=True, independent_snapshot_audit_passed=True,
        files_sha256={str(p.relative_to(root)): digest(p) for p in paths})
    with destination.open('x') as handle:
        handle.write(json.dumps(manifest, indent=2) + '\n')
    for path in paths + [destination]:
        path.chmod(0o444)
    return dict(status='PARTIAL', files=len(paths), manifest_sha256=digest(destination))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--standalone-rerun', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(seal(args.snapshot, args.standalone_rerun), indent=2))
