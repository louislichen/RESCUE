"""Audit frozen sources, matched request matrix, outputs and baseline contracts."""
import argparse
import hashlib
import json
from pathlib import Path

from .compare_baselines import summary, write


def audit(directory):
    protocol = json.loads((directory/'protocol.json').read_text())
    references = json.loads((directory/'references.json').read_text())
    selected = json.loads((directory/'selected_lengths.json').read_text())
    environment = json.loads((directory/'environment.json').read_text())
    args = protocol['args']
    for path, digest in protocol['source_sha256'].items():
        assert hashlib.sha256((directory/'source'/path).read_bytes()).hexdigest() == digest, path
    rows = [json.loads(line) for line in (directory/'runtime.jsonl').read_text().splitlines()]
    training = [json.loads(line) for line in (directory/'training.jsonl').read_text().splitlines()]
    modes = ('native', 'native_k4', 'specextend16', 'specextend32', 'rescue', 'target_ar')
    expected = {(d, repeat, mode) for d in protocol['validation_indices']
                for repeat in range(args['repeats']) for mode in modes}
    assert len(rows) == len(expected)
    assert {(r['source_index'], r['repeat'], r['mode']) for r in rows} == expected
    expected_training = {(d, repeat, mode, k) for d in protocol['train_indices']
                         for repeat in range(args['training_repeats'])
                         for mode in ('native', 'specextend16', 'specextend32') for k in protocol['menu']}
    assert len(training) == len(expected_training)
    assert {(r['source_index'], r['repeat'], r['mode'], r['k']) for r in training} == expected_training
    assert not set(protocol['train_indices']) & set(protocol['validation_indices'])
    uuids = set()
    for row in rows+training:
        assert row['output_agrees'] and row['generated'] == references[str(row['source_index'])]
        assert row['tokens'] == len(row['generated']) <= args['max_new_tokens']
        for name in ('decode_seconds', 'total_seconds', 'prefill_seconds'):
            assert row[name] > 0
        if row['mode'] == 'target_ar':
            continue
        assert row['request_id'] not in uuids
        uuids.add(row['request_id'])
        trace = row['trace']
        assert row['blocks'] == len(trace) and trace[-1]['terminal']
        assert row['proposed'] == sum(r['proposed'] for r in trace)
        assert row['accepted'] == sum(r['accepted'] for r in trace)
        assert row['draft_forwards'] == row['proposed']+sum(
            not r['terminal'] and r['accepted'] == r['proposed'] for r in trace)
        generated = 0
        for i, r in enumerate(trace):
            assert r['round_index'] == i
            assert 0 <= r['accepted'] <= r['proposed'] <= r['cap'] <= min(12, args['max_new_tokens']-generated)
            assert r['terminal'] == (i == len(trace)-1)
            if i:
                assert r['prefix'] == trace[i-1]['prefix']+trace[i-1]['accepted']+1
            if row['mode'] != 'rescue':
                assert r['cap'] == min(row['k'], args['max_new_tokens']-generated)
            if row['mode'].startswith('native'):
                assert r['workset_tokens'] == r['prefix']
            elif row['mode'].startswith('specextend'):
                assert r['workset_tokens'] <= 32*int(row['mode'].removeprefix('specextend'))
            else:
                assert r['workset_tokens'] <= protocol['config']['budget']
            generated += r['accepted']+(not r['terminal'] or row['tokens'] > generated+r['accepted'])
        assert generated == row['tokens']
        if row['mode'].startswith('native'):
            assert row['working_gather_layers'] == row['target_probe_rounds'] == row['draft_shadow_probes'] == row['projected_positions'] == 0
        elif row['mode'].startswith('specextend'):
            assert row['working_gather_layers'] == len(trace)*environment['draft_layers']
            assert row['draft_shadow_probes'] == row['projected_positions'] == 0
    for mode, choice in selected.items():
        scores = {k: sum(r['tokens'] for r in training if r['mode'] == mode and r['k'] == k)/
                     sum(r['total_seconds'] for r in training if r['mode'] == mode and r['k'] == k)
                  for k in protocol['menu']}
        assert choice['k'] == max(protocol['menu'], key=lambda k: scores[k])
        for row in rows:
            if row['mode'] == mode:
                assert row['k'] == choice['k']
    calculated = summary(rows)
    assert calculated == json.loads((directory/'summary.json').read_text())
    return dict(passed=True, frozen_sources=len(protocol['source_sha256']), validation_requests=len(rows),
                training_requests=len(training), exact_outputs=len(rows)+len(training),
                unique_speculative_request_ids=len(uuids), frozen_training_selected_lengths={m: v['k'] for m, v in selected.items()},
                scope='Saved records and frozen source hashes; no GPU rerun or full SpecExtend reproduction')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    write(args.directory/'audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
