"""Audit M2 full-span supplement and separate M3 speed endpoints."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from rescue.module_opt_report import aggregate,read
from rescue.module_report import paired
from rescue.objectives import accounting
from rescue.query_mapper import mapper_from_payload
from rescue.module_arch_warm import sample_span
import torch


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);args=p.parse_args()
    d=args.directory;protocol=json.loads((d/'protocol.json').read_text());locked=json.loads((d/'locked.json').read_text())
    assert json.loads((d/'COMPLETE.json').read_text())['passed']
    for name,h in protocol['source_sha256'].items():assert sha(d/'source'/name)==h
    assert sha(d/'PLAN.json')==protocol['plan_sha256']
    assert sha(ROOT/'exp/module_arch/length_speed_plan.json')==protocol['length_plan_sha256']
    assert sha(d/'warm_final.pt')==locked['mapper_sha256']
    base=Path(protocol['base']);refs=json.loads((base/'references.json').read_text());uuids=set();exact=0
    for file in ('collection.jsonl','runtime.jsonl'):
        for r in read(d/file):
            assert r['request_id'] not in uuids;uuids.add(r['request_id']);exact+=1
            assert r['generated']==refs[f"{r['split']}:{r['source_index']}:{r['length']}"][:r['tokens']] and r['output_agrees']
            assert r['accounting']==accounting(r)
            assert r['draft_forwards']==r['proposed']+sum(not t['terminal'] and t['accepted']==t['proposed'] for t in r['trace'])
            assert all(0<=t['accepted']<=t['proposed']<=t['cap']<=4 for t in r['trace'])
    collection=[]
    for b in read(d/'bundles.jsonl'):
        file=Path(b['path']);assert sha(file)==b['sha256']
        payload=torch.load(file,map_location='cpu',weights_only=False)
        expected=sample_span(payload['records'],monitor=payload['source_index']==236)
        assert expected==b['selected_indices']
        collection.append(dict(phase=b['phase'],source_index=payload['source_index'],length=payload['length'],
            raw_records=b['raw_records'],selected=len(expected),selected_wrong=b['chosen_wrong'],
            generated_prefix_span=b['max_prefix']-b['min_prefix']))
    rows=read(d/'runtime.jsonl')
    expected={(i,n,r,m) for i in locked['confirmation_ids'] for n in protocol['lengths'] for r in range(2) for m in locked['modes']}
    assert len(rows)==84 and {(r['source_index'],r['length'],r['repeat'],r['mode']) for r in rows}==expected
    qq=read(d/'quality'/'quality.jsonl');maps={}
    expectedq={(i,n,q) for i in locked['confirmation_ids'] for n in protocol['lengths'] for q in range(24)}
    for mode in ('hier512','m2_old','hier512_goal'):
        rr=[r for r in qq if r['mode']==mode];maps[mode]={(r['source_index'],r['length'],r['query_index']):r for r in rr}
        assert len(rr)==144 and set(maps[mode])==expectedq
    qstats={name:dict(queries=144,matches=sum(r['top1_matches_target'] for r in rr.values()),
        mean_kl=statistics.mean(r['target_kl'] for r in rr.values())) for name,rr in maps.items()}
    differences={}
    for name in ('hier512','m2_old'):
        a,b=maps['hier512_goal'],maps[name]
        differences[name]=dict(corrected=sum(a[k]['top1_matches_target'] and not b[k]['top1_matches_target'] for k in a),
            harmed=sum(not a[k]['top1_matches_target'] and b[k]['top1_matches_target'] for k in a))
    def trajectories(rr,name):
        return {(r['source_index'],r['length'],r['repeat']):[(t['cap'],t['proposed'],t['accepted']) for t in r['trace']] for r in rr if r['mode']==name}
    base_rows=read(base/'runtime.jsonl');same={}
    for a,b in (('native','native'),('query','query'),('full','full'),('compact','compact_fixed'),('step','length_constrained')):
        assert trajectories(base_rows,a)==trajectories(rows,b)
        same[a+'_to_'+b]=12
    pairs=[('warm','native'),('warm','query'),('warm','full'),('length_fast','compact_fixed'),
           ('length_constrained','compact_fixed'),('length_fast','length_constrained')]
    comparisons={a+'_vs_'+b:dict(paired=paired(rows,a,b),by_length={str(n):paired(rows,a,b,n) for n in protocol['lengths']}) for a,b in pairs}
    payload=torch.load(d/'warm_final.pt',map_location='cpu',weights_only=False);model=mapper_from_payload(payload,'cpu')
    summary=dict(scope='M2 training package and separate M3 speed endpoint; same three documents, no combined warm mapper/length test',
        quality=qstats,quality_contrasts=differences,collection=collection,
        parameters=sum(p.numel() for p in model.parameters()),
        nonzero_residual=any(torch.count_nonzero(p)>0 for block in (model.decoders[0],model.residuals[0][-1]) for p in block.parameters()),
        modes={name:dict(**aggregate([r for r in rows if r['mode']==name]),
            caps=dict(Counter(t['cap'] for r in rows if r['mode']==name for t in r['trace']))) for name in locked['modes']},
        by_length={str(n):{name:aggregate([r for r in rows if r['mode']==name and r['length']==n]) for name in locked['modes']} for n in protocol['lengths']},
        comparisons=comparisons,unchanged_cross_stage_trajectories=same,
        training={name:json.loads((d/f'{name}_training.json').read_text()) for name in ('warm_stage1','warm_final')})
    (d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    audit=dict(passed=True,exact_saved_requests=exact,confirmation_requests=84,quality_rows=432,
        frozen_sources=len(protocol['source_sha256']),unchanged_cross_stage_trajectories=same,
        analysis_sha256=sha(Path(__file__)),scope='Saved-record/source verification, not extra GPU experiment')
    (d/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(audit=audit,quality=qstats,quality_contrasts=differences,comparisons=comparisons),indent=2))


if __name__=='__main__':main()
