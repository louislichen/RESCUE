"""Audit frozen evidence and recompute paired module-architecture outcomes."""
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
import torch


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);args=p.parse_args()
    d=args.directory;protocol=json.loads((d/'protocol.json').read_text());locked=json.loads((d/'locked.json').read_text())
    assert json.loads((d/'COMPLETE.json').read_text())['passed']
    for file,digest in protocol['source_sha256'].items():assert sha(d/'source'/file)==digest
    assert sha(d/'PLAN.md')==protocol['plan_sha256']
    for file,digest in locked['checkpoints_sha256'].items():assert sha(Path(file))==digest
    for file,digest in json.loads((d/'input_checkpoint_sha256.json').read_text()).items():assert sha(Path(file))==digest
    groups={}
    for file,digest in protocol['data_manifest']['data_sha256'].items():
        path=Path(protocol['args']['data'])/f'{file}.jsonl';assert sha(path)==digest;groups[file]=read(path)
    hashes={name:{r['document_sha256'] for r in rr} for name,rr in groups.items()}
    assert all(not hashes[a]&hashes[b] for a in hashes for b in hashes if a!=b)
    refs=json.loads((d/'references.json').read_text());uuids=set();exact=0
    for filename in ('collection.jsonl','mapper_tune.jsonl','length_tune.jsonl','runtime.jsonl'):
        for r in read(d/filename):
            assert r['generated']==refs[f"{r['split']}:{r['source_index']}:{r['length']}"][:r['tokens']] and r['output_agrees']
            assert r['request_id'] not in uuids;uuids.add(r['request_id']);exact+=1
            assert r['accounting']==accounting(r)
            assert r['draft_forwards']==r['proposed']+sum(not t['terminal'] and t['accepted']==t['proposed'] for t in r['trace'])
            assert all(0<=t['accepted']<=t['proposed']<=t['cap']<=4 for t in r['trace'])
    for b in read(d/'bundles.jsonl'):assert sha(Path(b['path']))==b['sha256']
    rows=read(d/'runtime.jsonl')
    expected={(i,n,r,m) for i in locked['confirmation_ids'] for n in protocol['lengths'] for r in range(2) for m in locked['modes']}
    assert len(rows)==72 and {(r['source_index'],r['length'],r['repeat'],r['mode']) for r in rows}==expected
    selected=locked['selected_mapper']
    contrasts=[('query','native'),('full','query'),('compact','query'),('compact','full'),
               ('full','native'),('compact','native'),('step',selected),('prior',selected),('step','prior')]
    comparisons={a+'_vs_'+b:dict(paired=paired(rows,a,b),by_length={str(n):paired(rows,a,b,n) for n in protocol['lengths']}) for a,b in contrasts}
    quality={};oldkeys=None
    expectedq={(i,n,q) for i in locked['confirmation_ids'] for n in protocol['lengths'] for q in range(24)}
    for candidate in ('full','compact'):
        qq=read(d/f'quality_{candidate}'/'quality.jsonl');maps={}
        for mode in ('hier512','m2_old','hier512_goal'):
            rr=[r for r in qq if r['mode']==mode]
            maps[mode]={(r['source_index'],r['length'],r['query_index']):r for r in rr}
            assert len(rr)==144 and set(maps[mode])==expectedq
        old={k:(r['top1_matches_target'],r['target_kl']) for k,r in maps['m2_old'].items()}
        assert oldkeys is None or old==oldkeys;oldkeys=old
        qstats={name:dict(queries=len(rr),matches=sum(r['top1_matches_target'] for r in rr.values()),
            mean_kl=statistics.mean(r['target_kl'] for r in rr.values())) for name,rr in maps.items()}
        differences={}
        for base in ('hier512','m2_old'):
            a,b=maps['hier512_goal'],maps[base]
            differences[base]=dict(corrected=sum(a[k]['top1_matches_target'] and not b[k]['top1_matches_target'] for k in a),
                harmed=sum(not a[k]['top1_matches_target'] and b[k]['top1_matches_target'] for k in a))
        quality[candidate]=dict(modes=qstats,contrasts=differences)
    parameters={}
    for name,file in (('query',ROOT/'exp/module_opt/out_v1_20261018/mapper_query.pt'),('full',d/'full_final.pt'),('compact',d/'compact_final.pt')):
        payload=torch.load(file,map_location='cpu',weights_only=False);model=mapper_from_payload(payload,'cpu')
        rel=model.layers.index(27)
        active=sum(p.numel() for block in ('encoders','decoders','native','query','gates','residuals') for p in getattr(model,block)[rel].parameters())
        parameters[name]=dict(total=sum(p.numel() for p in model.parameters()),active_last_layer=active)
        if name=='compact':
            parameters[name]['zero_direction']=all(torch.count_nonzero(p)==0 for block in (model.decoders[rel],model.residuals[rel][-1]) for p in block.parameters())
    summary=dict(scope='Fresh three-document development confirmation of architecture/training packages, one model and seed',
        selected_mapper=selected,selected_length=locked['selected_length'],quality=quality,parameters=parameters,
        modes={name:dict(**aggregate([r for r in rows if r['mode']==name]),
            caps=dict(Counter(t['cap'] for r in rows if r['mode']==name for t in r['trace']))) for name in locked['modes']},
        by_length={str(n):{name:aggregate([r for r in rows if r['mode']==name and r['length']==n]) for name in locked['modes']} for n in protocol['lengths']},
        comparisons=comparisons,mapper_choice=json.loads((d/'mapper_choice.json').read_text()),length_choice=json.loads((d/'length_choice.json').read_text()))
    (d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    paused=ROOT/'exp/long_system/out_comparison_v4_20261017'
    assert len([r for r in read(paused/'runtime.jsonl') if r.get('split')=='test'])==71
    changes={name:dict(executed=digest,current=sha(ROOT/name)) for name,digest in protocol['source_sha256'].items() if sha(ROOT/name)!=digest}
    audit=dict(passed=True,exact_saved_requests=exact,confirmation_requests=72,quality_rows=864,
        frozen_sources=len(protocol['source_sha256']),current_source_changes=changes,paused_test_requests=71,
        analysis_sha256=sha(Path(__file__)),source='Saved-record verification, not extra GPU runs')
    (d/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(audit=audit,quality=quality,parameters=parameters,comparisons=comparisons),indent=2))


if __name__=='__main__':main()
