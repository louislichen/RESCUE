"""Verify and summarize the completed goal-directed development pilot."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics

from .objectives import accounting


def read(path): return [json.loads(l) for l in path.read_text().splitlines()]


def paired(rows,first,second,length=None):
    selected=[r for r in rows if length is None or r['length']==length]
    ratios=defaultdict(list)
    for doc,n in sorted({(r['source_index'],r['length']) for r in selected}):
        a=statistics.median(r['total_tps'] for r in selected if r['source_index']==doc and r['length']==n and r['mode']==first)
        b=statistics.median(r['total_tps'] for r in selected if r['source_index']==doc and r['length']==n and r['mode']==second)
        ratios[doc].append(100*(a/b-1))
    values=[statistics.mean(v) for v in ratios.values()]
    rng=random.Random(58)
    bootstrap=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(5000))
    return dict(documents=len(values),mean_percent=statistics.mean(values),ci95=[bootstrap[124],bootstrap[4874]],
        by_document={str(k):statistics.mean(v) for k,v in ratios.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);args=p.parse_args()
    d=args.directory;protocol=json.loads((d/'protocol.json').read_text())
    assert json.loads((d/'COMPLETE.json').read_text())['passed']
    for name,digest in protocol['source_sha256'].items():
        assert hashlib.sha256((d/'source'/name).read_bytes()).hexdigest()==digest
    for name,digest in protocol['input_checkpoint_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
    for split,digest in protocol['data_sha256'].items():
        assert hashlib.sha256(Path(f'exp/long_system/data/{split}.jsonl').read_bytes()).hexdigest()==digest
    rows=read(d/'runtime.jsonl');q=read(d/'quality.jsonl');profiles=read(d/'profiles.jsonl')
    expected={(doc,n,r,m) for doc in protocol['fresh_development_indices'] for n in protocol['lengths']
        for r in range(protocol['repeats']) for m in protocol['configs']}
    assert len(rows)==len(expected) and {(r['source_index'],r['length'],r['repeat'],r['mode']) for r in rows}==expected
    refs=json.loads((d/'references.json').read_text());seen=set()
    for r in rows:
        assert r['output_agrees'] and r['generated']==refs[f"{r['source_index']}:{r['length']}"]
        assert r['request_id'] not in seen;seen.add(r['request_id'])
        assert r['accounting']==accounting(r)
        assert r['tokens']<=protocol['max_new_tokens']
    quality={}
    for split in ('train','development'):
        group=[r for r in q if r['split']==split]
        modes=sorted({r['mode'] for r in group})
        keyed={m:{(r['source_index'],r['length'],r['query_index']):r for r in group if r['mode']==m} for m in modes}
        assert all(set(keyed[m])==set(keyed[modes[0]]) for m in modes)
        stats={}
        for m,g in keyed.items():
            rr=list(g.values())
            stats[m]=dict(queries=len(rr),greedy_agreement=sum(r['top1_matches_target'] for r in rr)/len(rr),
                mean_kl=statistics.mean(r['target_kl'] for r in rr),
                mean_reference_tokens=statistics.mean(r['reference_tokens'] for r in rr),
                mean_teacher_mass=statistics.mean(r['teacher_attention_mass'] for r in rr),
                mean_teacher_read_error=statistics.mean(r['teacher_read_error'] for r in rr))
        contrasts={}
        for a,b in (('teacher512','full_native'),('joint512','teacher512'),('hier512','joint512'),
                    ('hier1024','hier512'),('hier512_old','hier512'),('hier512_goal','hier512'),('hier512_goal','hier512_old')):
            corrected=sum(keyed[a][k]['top1_matches_target'] and not keyed[b][k]['top1_matches_target'] for k in keyed[a])
            harmed=sum(not keyed[a][k]['top1_matches_target'] and keyed[b][k]['top1_matches_target'] for k in keyed[a])
            doc_deltas=defaultdict(list)
            for k in keyed[a]:
                doc_deltas[k[0]].append(100*(int(keyed[a][k]['top1_matches_target'])-int(keyed[b][k]['top1_matches_target'])))
            values=[statistics.mean(v) for v in doc_deltas.values()]
            rng=random.Random(58)
            draws=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(5000))
            contrasts[a+'_vs_'+b]=dict(corrected=corrected,harmed=harmed,
                net_percentage_points=100*(corrected-harmed)/len(keyed[a]),
                doc_bootstrap_ci95_pp=[draws[124],draws[4874]],documents=len(values),
                by_document_pp={str(k):statistics.mean(v) for k,v in doc_deltas.items()})
        quality[split]=dict(modes=stats,contrasts=contrasts)
    runtime={}
    for n in protocol['lengths']:
        modes={}
        for m in protocol['configs']:
            group=[r for r in rows if r['mode']==m and r['length']==n]
            commits=sum(r['tokens'] for r in group)
            modes[m]=dict(total_tps=commits/sum(r['total_seconds'] for r in group),
                acceptance_rate=sum(r['accepted'] for r in group)/sum(r['proposed'] for r in group),
                mean_rounds=statistics.mean(r['blocks'] for r in group),
                wasted_per_committed=sum(r['accounting']['wasted_proposals'] for r in group)/commits,
                avoidable_tail_per_committed=sum(r['accounting']['avoidable_rejected_tail'] for r in group)/commits,
                draft_forwards_per_committed=sum(r['draft_forwards'] for r in group)/commits,
                verify_rounds_per_100_committed=100*sum(r['blocks'] for r in group)/commits,
                mean_reference_tokens=statistics.mean(r['accounting']['mean_reference_tokens'] for r in group),
                length_wait_seconds=sum(r['accounting']['length_wait_seconds'] for r in group))
        runtime[str(n)]=modes
    contrasts={}
    for a,b in (('teacher512','full_native'),('joint512','teacher512'),('hier512','teacher512'),
                ('hier1024','hier512'),('hier512_old','hier512'),('hier512_goal','hier512'),
                ('hier512_goal','hier512_old'),('risk_serial','hier512_old'),('risk_utility','risk_serial'),
                ('risk_utility','hier512_old'),('risk_async_late','risk_serial'),
                ('risk_async_early','risk_async_late'),('risk_async_early','risk_serial')):
        contrasts[a+'_vs_'+b]=dict(all_lengths=paired(rows,a,b),
            by_length={str(n):paired(rows,a,b,n) for n in protocol['lengths']})
    trajectories={}
    for a,b in (('risk_async_late','risk_serial'),('risk_async_early','risk_async_late')):
        matches=0;pairs=0
        for doc,n,r in sorted({(x['source_index'],x['length'],x['repeat']) for x in rows}):
            aa=next(x for x in rows if (x['source_index'],x['length'],x['repeat'],x['mode'])==(doc,n,r,a))
            bb=next(x for x in rows if (x['source_index'],x['length'],x['repeat'],x['mode'])==(doc,n,r,b))
            def signature(row): return [(t['cap'],t['proposed'],t['accepted']) for t in row['trace']]
            matches+=signature(aa)==signature(bb);pairs+=1
        trajectories[a+'_vs_'+b]=dict(matches=matches,pairs=pairs)
    result=dict(scope='Three-document fresh-development pilot, paired by document; not independent paper confirmation or SOTA evidence',
        quality=quality,runtime=runtime,contrasts=contrasts,schedule_trajectories=trajectories,
        profile_scope='Host/stream elapsed intervals include overlap/idle; profiling requests excluded from throughput',
        profiles=[dict(mode=r['mode'],length=r['length'],timeline=r.get('timeline'),
            deadline_ready=sum(t.get('length_ready_at_deadline') is True for t in r['trace']),
            rounds=r['blocks'],wait_seconds=sum(t.get('length_wait_seconds',0.) for t in r['trace'])) for r in profiles])
    (d/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    audit=dict(passed=True,system_requests=len(rows),exact_system_outputs=len(rows),paired_quality_rows=len(q),
        quality_modes=len({r['mode'] for r in q}),frozen_sources=len(protocol['source_sha256']),
        uuid_unique=len(seen),schedule_trajectories=trajectories,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Saved records/configuration/hash/statistical verification, not a second GPU run')
    (d/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(audit=audit,development_quality=quality['development'],contrasts=contrasts),indent=2))


if __name__=='__main__': main()
