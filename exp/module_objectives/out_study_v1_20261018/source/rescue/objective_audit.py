"""CPU-only matched module accounting and training-only horizon utility fit."""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch

from .objectives import accounting, horizon_outcome, horizon_seconds
from .policy import RiskModel


def read(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def totals(rows):
    a=[accounting(r) for r in rows]
    return dict(requests=len(rows),committed=sum(r['tokens'] for r in rows),
        accepted=sum(r['accepted'] for r in rows),proposed=sum(r['proposed'] for r in rows),
        rounds=sum(r['blocks'] for r in rows),draft_forwards=sum(r['draft_forwards'] for r in rows),
        wasted_proposals=sum(v['wasted_proposals'] for v in a),
        avoidable_tail=sum(v['avoidable_rejected_tail'] for v in a),
        mean_reference_tokens=statistics.mean(v['mean_reference_tokens'] for v in a),
        total_tps=sum(r['tokens'] for r in rows)/sum(r['total_seconds'] for r in rows))


def fit_horizon(training, cost, output):
    frames=json.loads((training/'risk_frames.json').read_text())
    frames=[r for r in frames if r['proposed']==12 and round(r['features'][4]*64)>12]
    fit=[r for r in frames if not r['monitor']]
    monitor=[r for r in frames if r['monitor']]
    risk=RiskModel.load(training/'risk_injected.pt','cpu')
    counts=torch.ones(12,3)*.5
    for f in fit:
        for i,y in enumerate(f['labels']): counts[i,y]+=1
    prior=counts/counts.sum(-1,keepdim=True)
    menu=[1,2,4,8,12]
    def arrays(group):
        x=torch.tensor([r['features'] for r in group])
        outcomes=[[horizon_outcome(r,k) for k in menu] for r in group]
        seconds=torch.tensor([[horizon_seconds(cost,r['prefix'],k,o) for k,o in zip(menu,oo)] for r,oo in zip(group,outcomes)])
        committed=torch.tensor([[o['committed'] for o in oo] for oo in outcomes])
        waste=torch.tensor([[o['waste'] for o in oo] for oo in outcomes])
        tail=torch.tensor([[o['avoidable_tail'] for o in oo] for oo in outcomes])
        # Predicted cost accounts for expected all-accepted cache catchup.
        nf=torch.tensor([r['prefix']/2048 for r in group])[:,None]
        k=torch.tensor(menu)[None]
        b=cost['verify_beta']
        predictable=cost.get('fixed_round',0.)+b[0]+b[1]*nf+b[2]*k+b[3]*k*nf+k*cost['student_step']+cost.get('cache_update',0.)
        return dict(x=x,logits=risk(x).detach(),outcomes=outcomes,seconds=seconds,committed=committed,
            waste=waste,tail=tail,utility=committed/seconds,predictable=predictable)
    fs,ms=arrays(fit),arrays(monitor)
    def choice(a,weight,bias):
        logits=a['logits'].clone();logits[:,:,0]+=bias
        survival=((1-weight)*logits.softmax(-1)+weight*prior)[...,0].cumprod(-1)
        ks=torch.tensor(menu)-1
        expected=1+survival.cumsum(-1)[:,ks]
        seconds=a['predictable']+survival[:,ks]*cost['student_step']
        return (expected/seconds).argmax(-1)
    def assess(a,group,indices):
        ii=torch.arange(len(group))
        actual=a['utility'][ii,indices]
        regret=1-actual/a['utility'].max(-1).values
        bydoc=defaultdict(list)
        doc_totals=defaultdict(lambda:[0.,0.])
        for r,v in zip(group,regret.tolist()): bydoc[r['source_index']].append(v)
        for i,r in enumerate(group):
            doc_totals[r['source_index']][0]+=float(a['committed'][i,indices[i]])
            doc_totals[r['source_index']][1]+=float(a['seconds'][i,indices[i]])
        committed=a['committed'][ii,indices].sum().item()
        return dict(frames=len(group),document_mean_myopic_regret=statistics.mean(statistics.mean(v) for v in bydoc.values()),
            mean_log_document_modeled_tps=statistics.mean(math.log(v[0]/v[1]) for v in doc_totals.values()),
            selected_horizons={str(k):int((indices==i).sum()) for i,k in enumerate(menu)},
            modeled_committed_per_second=float(committed/a['seconds'][ii,indices].sum()),
            wasted_proposals_per_committed=float(a['waste'][ii,indices].sum()/committed),
            avoidable_tail_per_committed=float(a['tail'][ii,indices].sum()/committed),
            verify_rounds_per_100_committed=100*len(group)/committed)
    candidates=[]
    for weight in (0.,.25,.5,.75,1.):
        for bias in (0.,.5,1.,1.5):
            fi,mi=choice(fs,weight,bias),choice(ms,weight,bias)
            candidates.append(dict(prior_weight=weight,accept_bias=bias,fit=assess(fs,fit,fi),monitor=assess(ms,monitor,mi)))
    # Select the rate objective on the existing training monitor; local oracle
    # regret is diagnostic and is not interchangeable with pooled throughput.
    # Fresh development documents supply the subsequent real rollout check.
    best=max(candidates,key=lambda v:v['monitor']['mean_log_document_modeled_tps'])
    payload=torch.load(training/'risk_injected.pt',map_location='cpu',weights_only=False)
    calibrated=risk.calibrated(1.,best['accept_bias'])
    payload.update(state_dict=calibrated.state_dict(),prior_hazards=prior.tolist(),prior_weight=best['prior_weight'])
    torch.save(payload,output/'risk_utility.pt')
    result=dict(scope='Single-start model-cost counterfactuals on old-mapper training states, not full-trajectory TPS or a post-update rollout',
        source_sha256=hashlib.sha256((training/'risk_frames.json').read_bytes()).hexdigest(),menu=menu,
        limitations='Proposed=12 and remaining>12 only; EOS bonus termination is not fully recorded; costs are fitted and must be checked online.',
        fit_documents=sorted({r['source_index'] for r in fit}),monitor_documents=sorted({r['source_index'] for r in monitor}),
        selection='Maximize geometric document-modeled throughput on the training monitor; no fresh development/test outcomes used',
        selected=best,candidates=candidates,
        original_monitor=assess(ms,monitor,choice(ms,0.,0.)),
        fixed4_monitor=assess(ms,monitor,torch.full((len(monitor),),2)),
        oracle_monitor=assess(ms,monitor,ms['utility'].argmax(-1)))
    (output/'horizon_utility.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--training',type=Path,default=Path('exp/long_system/out_joint_train_v2_20261017'))
    args=p.parse_args()
    if args.output.exists(): p.error('Use a fresh output')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    initial=Path('exp/long_system/out_comparison_v2_20261017')
    refinement=Path('exp/long_system/out_comparison_v4_20261017')
    old,new=read(initial/'development.jsonl'),read(refinement/'development.jsonl')
    pairs={}
    for label,rows,names in (
        ('matched_injection',new,('budget1024','refine_matched_noinj')),
        ('matched_length',old,('h64_k4','dynamic')),
        ('reference_budget',old,('h64_k4','budget1024'))):
        pairs[label]={name:totals([r for r in rows if r['candidate']==name]) for name in names}
    (args.output/'existing_accounting.json').write_text(json.dumps(pairs,indent=2)+'\n')
    result=fit_horizon(args.training,json.loads((initial/'cost.json').read_text()),args.output)
    print(json.dumps(dict(accounting=pairs,horizon_selected=result['selected'],original_monitor=result['original_monitor'],fixed4_monitor=result['fixed4_monitor']),indent=2))


if __name__=='__main__': main()
