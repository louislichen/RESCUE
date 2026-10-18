"""Saved-record audit, matched module statistics, and deployable policy export."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

import torch

from .module_report import paired
from .objectives import accounting
from .policy import RiskModel


def read(path): return [json.loads(l) for l in path.read_text().splitlines()]


def aggregate(rows):
    tokens=sum(r['tokens'] for r in rows);proposed=sum(r['proposed'] for r in rows);accepted=sum(r['accepted'] for r in rows)
    rounds=sum(r['blocks'] for r in rows)
    return dict(requests=len(rows),tokens=tokens,accepted=accepted,proposed=proposed,rounds=rounds,
        accepted_per_round=accepted/rounds,acceptance_rate=accepted/proposed,
        wasted_proposals=proposed-accepted,waste_per_committed=(proposed-accepted)/tokens,
        draft_forwards=sum(r['draft_forwards'] for r in rows),
        mean_reference_tokens=statistics.mean(r['accounting']['mean_reference_tokens'] for r in rows),
        verify_rounds_per_100_committed=100*rounds/tokens,
        total_tps=tokens/sum(r['total_seconds'] for r in rows),
        decode_tps=tokens/sum(r['decode_seconds'] for r in rows))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);args=p.parse_args()
    d=args.directory;protocol=json.loads((d/'protocol.json').read_text());locked=json.loads((d/'locked.json').read_text())
    assert json.loads((d/'COMPLETE.json').read_text())['passed']
    for name,digest in protocol['source_sha256'].items():
        assert hashlib.sha256((d/'source'/name).read_bytes()).hexdigest()==digest
    for name,digest in locked['checkpoint_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
    for name,digest in json.loads((d/'input_checkpoint_sha256.json').read_text()).items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
    datasets={}
    for name,digest in protocol['data_manifest']['data_sha256'].items():
        path=Path(protocol['args']['data'])/f'{name}.jsonl'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
        datasets[name]=read(path)
    sets=[{r['document_sha256'] for r in datasets[n]} for n in ('fit','tune','confirmation','warmup')]
    assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
    old={r['document_sha256'] for split in ('train','development','test')
         for r in read(Path(f'exp/long_system/data/{split}.jsonl'))}
    assert not old&sets[2]
    refs=json.loads((d/'references.json').read_text());uuids=set();exact=0
    groups={name:read(d/name) for name in ('selector_tune.jsonl','collection.jsonl','length_tune.jsonl','rate_calibration.jsonl','runtime.jsonl')}
    for name,rows in groups.items():
        for row in rows:
            reference=refs[f"{row['split']}:{row['source_index']}:{row['length']}"]
            assert row['generated']==reference[:row['tokens']] and row['output_agrees']
            assert row['request_id'] not in uuids;uuids.add(row['request_id']);exact+=1
            assert row['accounting']==accounting(row)
            assert row['proposed']==sum(t['proposed'] for t in row['trace'])
            assert row['accepted']==sum(t['accepted'] for t in row['trace'])
            assert row['draft_forwards']==row['proposed']+sum(not t['terminal'] and t['accepted']==t['proposed'] for t in row['trace'])
            assert all(0<=t['accepted']<=t['proposed']<=t['cap']<=12 for t in row['trace'])
    expected={(i,n,r,m) for i in locked['confirmation_ids'] for n in protocol['lengths']
        for r in range(protocol['confirmation_repeats']) for m in locked['modes']}
    rows=groups['runtime.jsonl']
    assert len(rows)==len(expected) and {(r['source_index'],r['length'],r['repeat'],r['mode']) for r in rows}==expected
    modes={name:aggregate([r for r in rows if r['mode']==name]) for name in locked['modes']}
    by_length={str(n):{name:aggregate([r for r in rows if r['mode']==name and r['length']==n])
        for name in locked['modes']} for n in protocol['lengths']}
    comparisons={}
    for a,b in (('m1_new','m1_old'),('m2_old','m1_new'),('m2_query','m2_old'),('m2_query','m1_new'),
                ('m3_selected','m3_old'),('m3_selected','fixed_best'),('m3_history','fixed_best'),('m3_selected','m3_history')):
        comparisons[a+'_vs_'+b]=dict(paired=paired(rows,a,b),
            by_length={str(n):paired(rows,a,b,n) for n in protocol['lengths']})
    quality=read(d/'quality.jsonl');keys={}
    for mode in sorted({r['mode'] for r in quality}):
        keys[mode]={(r['source_index'],r['length'],r['query_index']):r for r in quality if r['mode']==mode}
    assert all(set(v)==set(keys['hier512']) for v in keys.values())
    qstats={}
    for mode,values in keys.items():
        rr=list(values.values());qstats[mode]=dict(queries=len(rr),matches=sum(r['top1_matches_target'] for r in rr),
            mean_kl=statistics.mean(r['target_kl'] for r in rr),mean_reference_tokens=statistics.mean(r['reference_tokens'] for r in rr))
    qcontrasts={}
    for a,b in (('m1_new','hier512'),('m1_new_old','m1_new'),('hier512_goal','m1_new'),('hier512_goal','m1_new_old')):
        qcontrasts[a+'_vs_'+b]=dict(corrected=sum(keys[a][k]['top1_matches_target'] and not keys[b][k]['top1_matches_target'] for k in keys[a]),
            harmed=sum(not keys[a][k]['top1_matches_target'] and keys[b][k]['top1_matches_target'] for k in keys[a]))
    # Export precisely the frozen policy. No tuning with confirmation outcomes.
    risk=json.loads((d/'risk_training.json').read_text());choice=json.loads((d/'length_choice.json').read_text())
    payload=torch.load(d/'risk_light.pt',map_location='cpu',weights_only=False)
    for label,name in (('risk_selected',choice['chosen_attention']),('risk_history','history_surplus')):
        saved=dict(payload)
        if name!='light':
            saved['feedback_policy']=dict(prior=risk['prior'],strength=8.,decay=.95,
                use_attention=name!='history_surplus',rule='ratio' if name=='feedback_ratio' else 'surplus',rates=choice['rates'])
        torch.save(saved,d/f'{label}.pt')
        model=RiskModel.load(d/f'{label}.pt','cpu')
        assert torch.isfinite(model.continuation(torch.zeros(36))).all()
    result=dict(scope='Three previously unused natural-document development confirmation, single model/seed; not generalization or SOTA proof',
        modes=modes,by_length=by_length,comparisons=comparisons,quality=qstats,quality_contrasts=qcontrasts,
        selector_choice=json.loads((d/'selector_choice.json').read_text()),
        mapper_training=json.loads((d/'mapper_training.json').read_text()),length_choice=choice,
        attribution='Learned direction benefit does not establish Teacher-content or learned-gate independent causality; M3 history-only control is separate')
    (d/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    audit=dict(passed=True,exact_saved_requests=exact,confirmation_requests=len(rows),unique_requests=len(uuids),
        request_groups={name:len(v) for name,v in groups.items()},quality_rows=len(quality),frozen_sources=len(protocol['source_sha256']),
        confirmation_document_hashes_disjoint=True,choice_frozen_before_confirmation=True,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Saved source/data/output/choice/statistics verification, not a new GPU run')
    (d/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(audit=audit,comparisons=comparisons,quality=qstats,quality_contrasts=qcontrasts),indent=2))


if __name__=='__main__': main()
