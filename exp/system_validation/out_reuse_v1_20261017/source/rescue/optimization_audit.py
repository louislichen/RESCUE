"""Audit optimization records and frozen training-only choices without GPU."""
import argparse
import hashlib
import json
from pathlib import Path

from .compare_baselines import write
from .optimization import compare, fit_hazards


def audit(directory):
    protocol = json.loads((directory/'protocol.json').read_text())
    references = json.loads((directory/'references.json').read_text())
    selected = json.loads((directory/'selected_lengths.json').read_text())
    args = protocol['args']
    for path, digest in protocol['source_sha256'].items():
        assert hashlib.sha256((directory/'source'/path).read_bytes()).hexdigest() == digest, path
    sources = [set(protocol[k]) for k in ('fit_indices', 'tune_indices', 'validation_indices')]
    assert all(not sources[i] & sources[j] for i in range(3) for j in range(i))
    def read(name):
        return [json.loads(line) for line in (directory/name).read_text().splitlines()]
    validation, tuning, fitting = read('runtime.jsonl'), read('tuning.jsonl'), read('fit_requests.jsonl')
    modes = ('native', 'specextend16', 'specextend32', 'legacy', 'optimized_oldrisk', 'optimized_fixed', 'optimized_calibrated', 'target_ar')
    expected = {(d, repeat, mode) for d in protocol['validation_indices'] for repeat in range(args['repeats']) for mode in modes}
    assert len(validation) == len(expected)
    assert {(r['source_index'], r['repeat'], r['mode']) for r in validation} == expected
    candidates = [(m, k) for m in selected for k in protocol['fixed_menu']]+[(m, 4) for m in ('legacy', 'optimized_oldrisk', 'optimized_calibrated')]
    expected_tuning = {(d, 0, m, k) for d in protocol['tune_indices'] for m, k in candidates}
    assert len(tuning) == len(expected_tuning)
    assert {(r['source_index'], r['repeat'], r['mode'], r['k']) for r in tuning} == expected_tuning
    uuids = set()
    for row in validation+tuning+fitting:
        assert row['output_agrees'] and row['generated'] == references[str(row['source_index'])]
        assert row['tokens'] == len(row['generated']) <= args['max_new_tokens']
        assert all(row[x] > 0 for x in ('total_seconds', 'decode_seconds', 'prefill_seconds'))
        if row.get('mode') == 'target_ar':
            continue
        assert row['request_id'] not in uuids
        uuids.add(row['request_id'])
        trace = row['trace']
        assert row['blocks'] == len(trace)
        assert row['proposed'] == sum(t['proposed'] for t in trace)
        assert row['accepted'] == sum(t['accepted'] for t in trace)
        assert row['draft_forwards'] == row['proposed']+sum(not t['terminal'] and t['accepted'] == t['proposed'] for t in trace)
        count = 0
        for i, t in enumerate(trace):
            assert t['round_index'] == i and t['terminal'] == (i == len(trace)-1)
            assert 0 <= t['accepted'] <= t['proposed'] <= t['cap'] <= min(12, args['max_new_tokens']-count)
            if i:
                assert t['prefix'] == trace[i-1]['prefix']+trace[i-1]['accepted']+1
            if row.get('mode') == 'native':
                assert t['workset_tokens'] == t['prefix']
            elif row.get('mode', '').startswith('specextend'):
                assert t['workset_tokens'] <= 32*int(row['mode'].removeprefix('specextend'))
            else:
                assert t['workset_tokens'] <= protocol['config']['budget']
            count += t['accepted']+(not t['terminal'] or row['tokens'] > count+t['accepted'])
        assert count == row['tokens']
        if row.get('mode') == 'native':
            assert row['working_gather_layers'] == row['target_probe_rounds'] == row['draft_shadow_probes'] == 0
    for mode, choice in selected.items():
        scores = {k: sum(r['tokens'] for r in tuning if r['mode'] == mode and r['k'] == k)/
                     sum(r['total_seconds'] for r in tuning if r['mode'] == mode and r['k'] == k) for k in protocol['fixed_menu']}
        assert choice['k'] == max(scores, key=scores.get)
        assert all(r['k'] == choice['k'] for r in validation if r['mode'] == mode)
    assert compare(validation) == json.loads((directory/'summary.json').read_text())
    frames = json.loads((directory/'hazard_frames.json').read_text())
    assert {r['source_index'] for r in frames} == set(protocol['fit_indices'])
    fitted = fit_hazards(frames)
    saved = json.loads((directory/'risk_calibration.json').read_text())
    assert abs(fitted['scale']-saved['scale']) < 1e-5
    assert abs(fitted['accept_bias']-saved['accept_bias']) < 1e-5
    return dict(passed=True, frozen_sources=len(protocol['source_sha256']), validation_requests=len(validation),
        tuning_requests=len(tuning), fitting_requests=len(fitting), exact_outputs=len(validation)+len(tuning)+len(fitting),
        training_only_calibration_recomputed=True, unique_speculative_request_ids=len(uuids),
        scope='Saved records/source/hash/statistical audit; no GPU rerun or full SpecExtend reproduction')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    write(args.directory/'audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
