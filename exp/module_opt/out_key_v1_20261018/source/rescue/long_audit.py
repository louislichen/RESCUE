"""CPU reconstruction of the registered long-context comparison records."""
import argparse
import hashlib
import json
from pathlib import Path

from .compare_baselines import write
from .long_benchmark import summary
from .loading import ROOT


def audit(directory):
    protocol = json.loads((directory/'protocol.json').read_text())
    args = protocol['args']
    for name, digest in protocol['source_sha256'].items():
        assert hashlib.sha256((directory/'source'/name).read_bytes()).hexdigest() == digest, name
    data = {}
    for split, digest in protocol['data_sha256'].items():
        path = ROOT/'exp/long_system/data'/(split+'.jsonl')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        data[split] = [json.loads(l) for l in path.read_text().splitlines()]
    hashes = [{r['document_sha256'] for r in data[s]} for s in ('train', 'development', 'test')]
    assert not hashes[0] & hashes[1] and not hashes[0] & hashes[2] and not hashes[1] & hashes[2]
    for name, digest in protocol['checkpoint_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    def read(name):
        return [json.loads(l) for l in (directory/name).read_text().splitlines()]
    tuning, rows = read('development.jsonl'), read('runtime.jsonl')
    selection = json.loads((directory/'selection.json').read_text())
    matrix = json.loads((directory/'test_matrix.json').read_text())
    references = json.loads((directory/'references.json').read_text())
    expected = {(doc, n, repeat, mode) for doc in matrix['source_indices']
                for n in matrix['lengths'] for repeat in range(matrix['repeats']) for mode in matrix['modes']}
    assert len(rows) == len(expected)
    assert {(r['source_index'], r['length'], r['repeat'], r['mode']) for r in rows} == expected
    candidate_names = {*protocol['configs'], *protocol['retrieval'], *protocol['full_tree']}
    expected_tuning = {(doc, n, candidate) for doc in protocol['development_indices']
                       for n in protocol['tuning_lengths'] for candidate in candidate_names}
    assert len(tuning) == len(expected_tuning)
    assert {(r['source_index'], r['length'], r['candidate']) for r in tuning} == expected_tuning
    for r in rows+tuning:
        assert r['split'] == ('test' if r in rows else 'development')
        key = r['split']+':'+str(r['source_index'])+':'+str(r['length'])
        assert r['output_agrees'] and r['generated'] == references[key]
        assert r['tokens'] == len(r['generated']) <= args['max_new_tokens']
        assert r['total_seconds'] > 0
        if 'trace' in r:
            trace = r['trace']
            assert len(trace) == r['blocks']
            assert sum(t['accepted'] for t in trace) == r['accepted']
            assert sum(t['proposed'] for t in trace) == r['proposed']
            assert r['draft_forwards'] == r['proposed']+sum(not t['terminal'] and t['accepted'] == t['proposed'] for t in trace)
            assert all(0 <= t['accepted'] <= t['proposed'] <= t['cap'] <= 12 for t in trace)
            if r['candidate'] in protocol['configs']:
                cfg = protocol['configs'][r['candidate']]
                assert all(t['workset_tokens'] <= cfg['budget'] for t in trace)
                assert r['draft_shadow_probes'] == r['blocks']-1
                if cfg['length_policy'] == 'risk':
                    assert r['injection_condition_steps'] == r['blocks']
            for previous, following in zip(trace, trace[1:]):
                assert following['prefix'] == previous['prefix']+previous['accepted']+1
    scores = {name: sum(r['tokens'] for r in tuning if r['candidate'] == name)/
                    sum(r['total_seconds'] for r in tuning if r['candidate'] == name) for name in candidate_names}
    expected_chosen = dict(rescue_positive=max((n for n,c in protocol['configs'].items() if c['inject_tokens']), key=scores.get),
        rescue_guarded=max(protocol['configs'], key=scores.get), rescue_dynamic='dynamic',
        retrieval=max(protocol['retrieval'], key=scores.get), full_specextend=max(protocol['full_tree'], key=scores.get))
    assert selection['chosen'] == expected_chosen == matrix['modes']
    recalculated = summary(rows)
    assert recalculated == json.loads((directory/'summary.json').read_text())
    gate = {}
    for candidate in ('rescue_positive', 'rescue_dynamic'):
        gate[candidate] = all(
            recalculated[str(n)]['comparisons'][candidate+'_vs_'+b]['mean_percent'] >= 10 and
            recalculated[str(n)]['comparisons'][candidate+'_vs_'+b]['ci95'][0] > 0
            for n in (8192, 16384) for b in ('retrieval', 'full_specextend'))
    return dict(passed=True, frozen_sources=len(protocol['source_sha256']), test_requests=len(rows),
        development_requests=len(tuning), exact_saved_outputs=len(rows)+len(tuning),
        disjoint_raw_document_hashes=True, source_of_choices='Development only, independent test configuration frozen',
        registered_success_gate=gate, scope='Saved output/source/matrix/selection/statistical audit, not a GPU rerun')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    a = p.parse_args()
    result = audit(a.directory)
    write(a.directory/'audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
