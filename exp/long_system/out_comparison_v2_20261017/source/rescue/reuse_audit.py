"""CPU audit of reuse counters, matrices, exact outputs and train-only choices."""
import argparse
import hashlib
import json
from pathlib import Path

from .compare_baselines import write
from .reuse_benchmark import summarize


def audit(directory):
    protocol = json.loads((directory/'protocol.json').read_text())
    args = protocol['args']
    references = json.loads((directory/'references.json').read_text())
    configs = json.loads((directory/'mode_configs.json').read_text())
    environment = json.loads((directory/'environment.json').read_text())
    selected = json.loads((directory/'execution_selection.json').read_text())
    route = json.loads((directory/'route_selection.json').read_text())
    routes = json.loads((directory/'route_candidates.json').read_text())
    for name, digest in protocol['source_sha256'].items():
        assert hashlib.sha256((directory/'source'/name).read_bytes()).hexdigest() == digest, name
    def read(name):
        return [json.loads(x) for x in (directory/name).read_text().splitlines()]
    rows, training, routing = read('runtime.jsonl'), read('execution_training.jsonl'), read('route_training.jsonl')
    modes = (*configs, 'native', 'specextend16', 'specextend32', 'target_ar')
    expected = {(d, r, m) for d in protocol['validation_indices'] for r in range(args['repeats']) for m in modes}
    assert len(rows) == len(expected)
    assert {(r['source_index'], r['repeat'], r['mode']) for r in rows} == expected
    expected_training = {(d, r, m) for d in protocol['train_indices'][:2] for r in range(2)
                         for m in protocol['execution_candidates']}
    assert len(training) == len(expected_training)
    assert {(r['source_index'], r['repeat'], r['mode']) for r in training} == expected_training
    expected_routing = {(d, m) for d in protocol['train_indices'] for m in routes}
    assert len(routing) == len(expected_routing)
    assert {(r['source_index'], r['mode']) for r in routing} == expected_routing
    assert not set(protocol['train_indices']) & set(protocol['validation_indices'])
    uuid = set()
    for row in rows+training+routing:
        assert row['output_agrees'] and row['generated'] == references[str(row['source_index'])]
        assert row['tokens'] == len(row['generated']) <= args['max_new_tokens']
        assert all(row[k] > 0 for k in ('total_seconds', 'decode_seconds', 'prefill_seconds'))
        if row['mode'] == 'target_ar':
            continue
        assert row['request_id'] not in uuid
        uuid.add(row['request_id'])
        trace = row['trace']
        assert len(trace) == row['blocks']
        assert sum(r['accepted'] for r in trace) == row['accepted']
        assert sum(r['proposed'] for r in trace) == row['proposed']
        assert row['draft_forwards'] == row['proposed']+sum(not r['terminal'] and r['accepted'] == r['proposed'] for r in trace)
        count = 0
        for i, t in enumerate(trace):
            assert t['round_index'] == i and t['terminal'] == (i == len(trace)-1)
            assert 0 <= t['accepted'] <= t['proposed'] <= t['cap'] <= min(12, args['max_new_tokens']-count)
            if i:
                assert t['prefix'] == trace[i-1]['prefix']+trace[i-1]['accepted']+1
            count += t['accepted']+(not t['terminal'] or row['tokens'] > count+t['accepted'])
        assert count == row['tokens']
        if row in training:
            cfg = protocol['execution_candidates'][row['mode']]
        elif row in routing:
            cfg = routes[row['mode']]
        elif row['mode'] in configs:
            cfg = configs[row['mode']]
        else:
            cfg = None
        if cfg is not None:
            assert all(t['workset_tokens'] <= cfg['budget'] for t in trace)
            assert row['draft_shadow_probes'] == (row['blocks']-1 if cfg['deferred_probes'] else row['draft_forwards'])
            assert row['working_gather_layers'] == row['blocks']*environment['student_layers']
            assert row['working_full_gather_layers'] <= row['working_gather_layers']
            expected_affine = (row['projected_positions'] if cfg['memoized_gate_affine']
                               else sum(min(cfg['inject_tokens'], t['workset_tokens']-cfg['recent']-1) for t in trace))
            layer_count = 4 if cfg['injection_layers'] is None else len(cfg['injection_layers'])
            if cfg['inject_tokens']:
                assert row['static_affine_positions'] == expected_affine*layer_count
            else:
                assert row['projected_positions'] == row['static_affine_positions'] == 0
        elif row['mode'] == 'native':
            assert row['working_gather_layers'] == row['draft_shadow_probes'] == 0
    def scores(records):
        return {m: sum(r['tokens'] for r in records if r['mode'] == m)/
                   sum(r['total_seconds'] for r in records if r['mode'] == m)
                for m in sorted({r['mode'] for r in records})}
    execution_scores, route_scores = scores(training), scores(routing)
    assert selected['chosen'] == max(('affine', 'packed', 'incremental'), key=execution_scores.get)
    assert route['positive'] == max((m for m in routes if m != 'none'), key=route_scores.get)
    assert route['guarded'] == max(routes, key=route_scores.get)
    assert summarize(rows) == json.loads((directory/'summary.json').read_text())
    indexed = {(r['mode'], r['source_index'], r['repeat']): r for r in rows}
    def trajectory(r):
        return [(x['cap'], x['proposed'], x['accepted']) for x in r['trace']]
    matches = {label: sum(trajectory(r) == trajectory(indexed[(b, r['source_index'], r['repeat'])])
                         for r in rows if r['mode'] == a)
               for label, a, b in [('risk', 'previous_risk', 'new_risk'), ('fixed', 'previous_fixed', 'new_fixed')]}
    return dict(passed=True, frozen_sources=len(protocol['source_sha256']), validation_requests=len(rows),
        execution_training_requests=len(training), route_training_requests=len(routing),
        exact_saved_outputs=len(rows)+len(training)+len(routing), unique_speculative_requests=len(uuid),
        paired_trajectory_matches=matches, paired_trajectory_total=args['documents']*args['repeats'],
        scope='Saved outputs, counters, frozen source hashes and selection/statistical audit, not a GPU rerun')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    write(args.directory/'audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
