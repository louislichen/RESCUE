"""Bind completed stages and compare saved trajectories without running models."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT/'exp/module_opt/out_v1_20261018')
    parser.add_argument('--key', type=Path, default=ROOT/'exp/module_opt/out_key_v1_20261018')
    parser.add_argument('--output', type=Path, default=ROOT/'exp/module_opt/final_audit.json')
    args = parser.parse_args()
    stages = {}
    for name, directory in (('primary', args.base), ('read_conditioned', args.key)):
        protocol = json.loads((directory/'protocol.json').read_text())
        assert json.loads((directory/'audit.json').read_text())['passed']
        changed = {}
        for file, frozen in protocol['source_sha256'].items():
            assert digest(directory/'source'/file) == frozen
            current = digest(ROOT/file)
            if current != frozen:
                changed[file] = dict(executed_sha256=frozen, current_sha256=current)
        stages[name] = dict(protocol_sha256=digest(directory/'protocol.json'),
            audit_sha256=digest(directory/'audit.json'), changes_since_frozen_execution=changed)
    rows, extra = read(args.base/'runtime.jsonl'), read(args.key/'runtime.jsonl')
    # Original controls and the direction-only candidate must keep their exact
    # deterministic cap/proposed/accepted sequence across the staged code changes.
    def trajectories(values, mode):
        return {(r['source_index'], r['length'], r['repeat']):
            [(t['cap'], t['proposed'], t['accepted']) for t in r['trace']]
            for r in values if r['mode'] == mode}
    comparisons = {}
    for before, after in (('m1_new', 'native'), ('m2_old', 'old'),
                          ('m2_query', 'direction'), ('m3_old', 'old_risk'),
                          ('m3_selected', 'feedback_surplus')):
        a, b = trajectories(rows, before), trajectories(extra, after)
        assert set(a) == set(b) and len(a) == 12
        assert a == b
        comparisons[f'{before}_to_{after}'] = dict(matched=12, total=12)
    primary = json.loads((args.base/'protocol.json').read_text())
    supplementary = json.loads((args.key/'protocol.json').read_text())
    assert primary['data_manifest']['data_sha256'] == supplementary['data_sha256']
    # Supplementary replay consumes the same immutable collected tensors.
    for row in read(args.base/'bundles.jsonl'):
        assert digest(Path(row['path'])) == row['sha256']
    quality = read(args.key/'quality.jsonl')
    expected = {(i, n, q) for i in (303, 306, 327)
                for n in (8192, 16384) for q in range(24)}
    for mode in ('full_native', 'hier512', 'm2_old', 'hier512_goal'):
        values = [r for r in quality if r['mode'] == mode]
        assert len(values) == 144
        assert {(r['source_index'], r['length'], r['query_index']) for r in values} == expected
    paused = ROOT/'exp/long_system/out_comparison_v4_20261017'
    count = len(read(paused/'runtime.jsonl')) if (paused/'runtime.jsonl').exists() else None
    # Its runtime includes completed development requests; the paused test is
    # counted by the explicit split, never by total file length.
    test_rows = [r for r in read(paused/'runtime.jsonl') if r.get('split') == 'test']
    assert len(test_rows) == 71 and (paused/'PAUSED.json').exists()
    checks = ROOT/'exp/module_opt/cpu_checks.txt'
    assert 'Ran 52 tests' in checks.read_text() and checks.read_text().strip().endswith('OK')
    result = dict(passed=True, scope='Saved-record/source checks, not a new GPU experiment',
        stages=stages, unchanged_cross_stage_trajectories=comparisons,
        exact_confirmation_requests=len(rows)+len(extra),
        exact_saved_sd_requests=194+len(extra),
        confirmation_documents=3, cohorts_are_independent=False,
        source_version_note='Each stage used its own frozen source. Current CLI/RPC validation and analysis/export edits were made after GPU execution; they are not timed speed evidence.',
        cpu_checks=dict(passed=52, log_sha256=digest(checks)),
        paused_test=dict(test_requests=71, runtime_rows=count, pause_sha256=digest(paused/'PAUSED.json')),
        analysis_sha256=digest(Path(__file__)))
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, confirmation_requests=len(rows)+len(extra),
        saved_sd_requests=194+len(extra), matched_cross_stage_pairs=len(comparisons), cpu_checks=52)))


if __name__ == '__main__':
    main()
