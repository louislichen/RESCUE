"""Audit development refinement and its frozen independent test matrix."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from .compare_baselines import write
from .loading import ROOT
from .long_benchmark import summary


def audit(directory):
    protocol=json.loads((directory/'protocol.json').read_text())
    initial=ROOT/protocol['args']['initial']
    assert hashlib.sha256((initial/'protocol.json').read_bytes()).hexdigest()==protocol['initial_protocol_sha256']
    assert hashlib.sha256((initial/'development.jsonl').read_bytes()).hexdigest()==protocol['initial_development_sha256']
    assert json.loads((initial/'DEVELOPMENT_STAGE_COMPLETE.json').read_text())['test_timing_rows']==0
    for folder in (initial,directory):
        metadata=json.loads((folder/'protocol.json').read_text())
        for name,digest in metadata['source_sha256'].items():
            assert hashlib.sha256((folder/'source'/name).read_bytes()).hexdigest()==digest,name
    datasets={}
    for split,digest in protocol['data_sha256'].items():
        path=ROOT/'exp/long_system/data'/(split+'.jsonl')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
        datasets[split]=[json.loads(l) for l in path.read_text().splitlines()]
    document_sets=[{r['document_sha256'] for r in datasets[s]} for s in ('train','development','test')]
    assert not document_sets[0]&document_sets[1] and not document_sets[0]&document_sets[2] and not document_sets[1]&document_sets[2]
    for name,digest in protocol['checkpoint_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    def read(folder,name):
        return [json.loads(l) for l in (folder/name).read_text().splitlines()]
    original=read(initial,'development.jsonl')
    development=read(directory,'development.jsonl')
    measured=read(directory,'runtime.jsonl')
    previous=json.loads((initial/'protocol.json').read_text())
    expected_old={(d,n,c) for d in previous['development_indices'] for n in previous['tuning_lengths']
                  for c in (*previous['configs'],*previous['retrieval'],*previous['full_tree'])}
    expected_new={(d,n,c) for d in protocol['development_indices'] for n in protocol['tuning_lengths']
                  for c in protocol['refinement_candidates']}
    assert len(original)==len(expected_old) and {(r['source_index'],r['length'],r['candidate']) for r in original}==expected_old
    assert len(development)==len(expected_new) and {(r['source_index'],r['length'],r['candidate']) for r in development}==expected_new
    matrix=json.loads((directory/'test_matrix.json').read_text())
    expected={(d,n,r,m) for d in matrix['source_indices'] for n in matrix['lengths']
              for r in range(matrix['repeats']) for m in matrix['modes']}
    assert len(measured)==len(expected) and {(r['source_index'],r['length'],r['repeat'],r['mode']) for r in measured}==expected
    references=json.loads((directory/'references.json').read_text())
    uuids=set()
    for group,split in ((original,'development'),(development,'development'),(measured,'test')):
        for row in group:
            key=f"{split}:{row['source_index']}:{row['length']}"
            assert row['split']==split and row['output_agrees'] and row['generated']==references[key]
            assert row['tokens']==len(row['generated'])<=protocol['max_new_tokens']
            assert row['total_seconds']>0
            if 'request_id' not in row:
                assert row['generated']==row['upstream']['output_token_ids'][:row['tokens']]
                continue
            assert row['request_id'] not in uuids
            uuids.add(row['request_id'])
            trace=row['trace']
            assert len(trace)==row['blocks']
            assert sum(t['proposed'] for t in trace)==row['proposed']
            assert sum(t['accepted'] for t in trace)==row['accepted']
            assert row['draft_forwards']==row['proposed']+sum(not t['terminal'] and t['accepted']==t['proposed'] for t in trace)
            assert all(0<=t['accepted']<=t['proposed']<=t['cap']<=12 for t in trace)
            for a,b in zip(trace,trace[1:]):
                assert b['prefix']==a['prefix']+a['accepted']+1
            if row['candidate'] in protocol['configs']:
                cfg=protocol['configs'][row['candidate']]
                assert row['draft_shadow_probes']==row['blocks']-1
                assert all(t['workset_tokens']<=cfg['budget'] for t in trace)
                assert row['injection_condition_steps']==(row['blocks'] if cfg['length_policy']=='risk' else 0)
    def scores(rows):
        control={(r['source_index'],r['length']):r['total_tps'] for r in rows if r['candidate']==protocol['reference_control']}
        return {name:math.exp(sum(math.log(r['total_tps']/control[(r['source_index'],r['length'])])
            for r in rows if r['candidate']==name)/sum(r['candidate']==name for r in rows))
            for name in {r['candidate'] for r in rows}}
    normalized=scores(original);normalized.update(scores(development))
    choices=dict(rescue_positive=max((n for n,c in protocol['configs'].items() if c['inject_tokens']),key=normalized.get),
        rescue_guarded=max(protocol['configs'],key=normalized.get),rescue_dynamic='dynamic',
        retrieval=max(protocol['retrieval'],key=normalized.get),full_specextend=max(protocol['full_tree'],key=normalized.get))
    selection=json.loads((directory/'selection.json').read_text())
    assert normalized==selection['normalized_development_scores']
    assert choices==selection['chosen']==matrix['modes']
    result=summary(measured)
    assert result==json.loads((directory/'summary.json').read_text())
    gates={c:all(result[str(n)]['comparisons'][c+'_vs_'+b]['mean_percent']>=10 and
        result[str(n)]['comparisons'][c+'_vs_'+b]['ci95'][0]>0
        for n in (8192,16384) for b in ('retrieval','full_specextend'))
        for c in ('rescue_positive','rescue_dynamic')}
    return dict(passed=True,initial_development_requests=len(original),refinement_development_requests=len(development),
        test_requests=len(measured),exact_saved_outputs=len(original)+len(development)+len(measured),
        current_frozen_sources=len(protocol['source_sha256']),original_frozen_sources=len(previous['source_sha256']),
        unique_linear_speculative_uuids=len(uuids),independent_test_choices_frozen=True,
        registered_success_gate=gates,scope='Saved source/output/configuration/selection/statistical audit; not a GPU rerun')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    args=p.parse_args()
    result=audit(args.directory)
    write(args.directory/'audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
