"""Audit the predefined read-conditioned M2 confirmation and paired effects."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from .module_opt_report import aggregate,read
from .module_report import paired
from .objectives import accounting


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);args=p.parse_args()
    d=args.directory;protocol=json.loads((d/'protocol.json').read_text());locked=json.loads((d/'locked.json').read_text())
    assert json.loads((d/'COMPLETE.json').read_text())['passed']
    base=Path(protocol['base']);refs=json.loads((base/'references.json').read_text())
    for name,digest in protocol['source_sha256'].items():assert hashlib.sha256((d/'source'/name).read_bytes()).hexdigest()==digest
    assert hashlib.sha256(Path('exp/module_opt/key_attention_plan.json').read_bytes()).hexdigest()==protocol['plan_sha256']
    assert hashlib.sha256(Path('exp/module_opt/length_pareto_plan.json').read_bytes()).hexdigest()==protocol['length_pareto_plan_sha256']
    assert hashlib.sha256((d/'mapper_query.pt').read_bytes()).hexdigest()==locked['mapper_sha256']
    rows=read(d/'runtime.jsonl');quality=read(d/'quality.jsonl');uuids=set()
    expected={(i,n,r,m) for i in locked['confirmation_ids'] for n in protocol['lengths'] for r in range(2) for m in locked['modes']}
    assert len(rows)==len(expected) and {(r['source_index'],r['length'],r['repeat'],r['mode']) for r in rows}==expected
    for r in rows:
        assert r['generated']==refs[f"confirmation:{r['source_index']}:{r['length']}"] and r['output_agrees']
        assert r['request_id'] not in uuids;uuids.add(r['request_id'])
        assert r['accounting']==accounting(r)
    q={}
    for name in sorted({r['mode'] for r in quality}):
        values=[r for r in quality if r['mode']==name]
        q[name]=dict(queries=len(values),matches=sum(r['top1_matches_target'] for r in values),mean_kl=statistics.mean(r['target_kl'] for r in values))
    keyed={name:{(r['source_index'],r['length'],r['query_index']):r for r in quality if r['mode']==name} for name in q}
    contrasts={}
    for a,b in (('hier512_goal','m2_old'),('hier512_goal','hier512')):
        assert set(keyed[a])==set(keyed[b])
        contrasts[a+'_vs_'+b]=dict(corrected=sum(keyed[a][k]['top1_matches_target'] and not keyed[b][k]['top1_matches_target'] for k in keyed[a]),
            harmed=sum(not keyed[a][k]['top1_matches_target'] and keyed[b][k]['top1_matches_target'] for k in keyed[a]))
    comparisons={a+'_vs_'+b:dict(paired=paired(rows,a,b),by_length={str(n):paired(rows,a,b,n) for n in protocol['lengths']})
        for a,b in (('keyed','old'),('keyed','native'),('keyed','direction'),('feedback_ratio','old_risk'),
                    ('feedback_surplus','old_risk'),('feedback_ratio','feedback_surplus'),('feedback_ratio','old'),('feedback_surplus','old'))}
    result=dict(scope='M2 and separate matched M3 endpoints on the same confirmation cohort; not independent replication of the base run',
        quality=q,quality_contrasts=contrasts,modes={name:aggregate([r for r in rows if r['mode']==name]) for name in locked['modes']},
        by_length={str(n):{name:aggregate([r for r in rows if r['mode']==name and r['length']==n]) for name in locked['modes']} for n in protocol['lengths']},
        comparisons=comparisons,training=json.loads((d/'mapper_training.json').read_text()))
    (d/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    audit=dict(passed=True,exact_system_outputs=len(rows),quality_rows=len(quality),frozen_sources=len(protocol['source_sha256']),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Saved-record verification, not an extra GPU run')
    (d/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(audit=audit,quality=q,quality_contrasts=contrasts,comparisons=comparisons),indent=2))


if __name__=='__main__':main()
