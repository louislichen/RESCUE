"""Seal audited local stage1 evidence; never edits the original server root."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


COUNTS = dict(reference_requests=6, prime_requests=76, timing_requests=456, owners=76)
FIGURES = ('quality', 'tps', 'captures', 'capture_time', 'cold', 'service', 'selected_repeat1')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def seal(root, rerun):
    root, rerun = Path(root).resolve(), Path(rerun).resolve()
    manifest_path, verified_path = root / 'FINAL_ARTIFACTS_MANIFEST.json', root / 'VERIFIED_COMPLETE.json'
    require(not manifest_path.exists() and not verified_path.exists(), 'Never overwrite a scientific seal')
    collection = read(root / 'TERMINAL_COLLECTION.json')
    require(collection.get('status') == 'TERMINAL_COLLECTED_NOT_YET_INDEPENDENTLY_AUDITED' and
            collection.get('pipeline_exit_code') == 0 and collection.get('process_and_tmux_absent') is True and
            collection.get('complete_file_set_rechecked') is True and
            collection.get('all_file_bytes_rechecked') is True and collection.get('counts') == COUNTS,
            'Terminal process/collection proof incomplete')
    for name, receipt in collection['files'].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root) and path.is_file() and sha(path) == receipt['sha256'],
                'Collected source/raw bytes changed: ' + name)
    results = root / 'results'
    analysis = results / 'analysis_final_v1'
    summary = read(analysis / 'summary.json')
    require(summary.get('status') == 'COMPLETE' and summary.get('problems') == [] and
            summary.get('winner_status') == 'VERIFIED_FROZEN' and summary.get('observed_counts') == COUNTS,
            'Strict saved-data analysis not complete')
    require((analysis / 'summary.json').read_bytes() == (rerun / 'summary.json').read_bytes(),
            'Saved standalone analysis did not reproduce exact summary')
    require(read(results / 'COMPLETE.json').get('goal_complete') is False,
            'Baseline selection cannot complete the overall goal')
    audit_path = root / 'INDEPENDENT_RAW_AUDIT.json'
    audit = read(audit_path)
    require(audit.get('status') == 'INDEPENDENT_RAW_AUDIT_PASSED' and audit.get('counts') == COUNTS and
            audit.get('all538_requests_verified') is True and audit.get('all76_owners_verified') is True and
            audit.get('eight_frozen_choices_verified') is True,
            'Independent author must audit full raw/owner/frozen-choice evidence')
    require(audit.get('baseline_protocol_sha256') == sha(results / 'protocol.json') and
            audit.get('frozen_selection_sha256') == sha(results / 'FROZEN_SELECTION.json') and
            audit.get('analysis_summary_sha256') == sha(analysis / 'summary.json'),
            'Independent audit is not bound to the analyzed bytes')
    render_path = analysis / 'RENDER_INSPECTION.json'
    render = read(render_path)
    require(render.get('actually_inspected_by_root') is True and
            render.get('rendered_pngs_inspected') == list(FIGURES),
            'All seven primary/diagnostic figures need actual root inspection')
    for name in FIGURES:
        for suffix in ('.png', '.svg'):
            path = analysis / (name + suffix)
            require(path.is_file() and render.get('figure_sha256', {}).get(path.name) == sha(path),
                    'Inspected figure receipt differs: ' + path.name)
    require((root / 'PROGRESS.md').is_file(), 'Missing written interpretation')
    paths = sorted(p for p in root.rglob('*') if p.is_file())
    require(not any(p.is_symlink() for p in paths), 'Do not seal symlink targets')
    manifest = dict(status='VERIFIED_COMPLETE_BASELINE_DEVELOPMENT_ONLY',
        sealed_at_utc=datetime.now(timezone.utc).isoformat(), goal_complete=False,
        file_sha256={str(p.relative_to(root)): sha(p) for p in paths})
    with manifest_path.open('x') as handle:
        handle.write(json.dumps(manifest, indent=2) + '\n')
    verified = dict(status='VERIFIED_COMPLETE', passed=True,
        raw_independent_audit_passed=True, rendered_figures_inspected=True,
        strict_analysis_complete=True, all538_requests_verified=True,
        all76_owners_verified=True, eight_frozen_choices_verified=True,
        standalone_summary_exact_rerun=True, goal_complete=False,
        baseline_complete_sha256=sha(results / 'COMPLETE.json'),
        frozen_selection_sha256=sha(results / 'FROZEN_SELECTION.json'),
        analysis_summary_sha256=sha(analysis / 'summary.json'),
        verified_winners_sha256=sha(analysis / 'VERIFIED_WINNERS.json'),
        final_artifacts_manifest_sha256=sha(manifest_path),
        independent_raw_audit_sha256=sha(audit_path), render_inspection_sha256=sha(render_path))
    with verified_path.open('x') as handle:
        handle.write(json.dumps(verified, indent=2) + '\n')
    for path in paths + [manifest_path, verified_path]:
        path.chmod(0o444)
    return verified


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--standalone-rerun', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(seal(args.root, args.standalone_rerun), indent=2))
