"""Last-layer prefix-aware replay and censored two-to-four budget fitting."""
import copy
import random

import torch
import torch.nn.functional as F

from .compare_baselines import write
from .objectives import distribution_quality
from .rollout_training import RolloutCollector, replay
from .step_budget import StepBudget, compatible_classes, outcome_tables


class ArchitectureCollector(RolloutCollector):
    step_budget = True

    def __init__(self, student, max_records=48):
        super().__init__(student,max_records)
        self.budget_frames=[]

    def round(self, packet):
        x=packet.get('budget_features')
        if x is not None:
            labels=[]
            for j,token in enumerate(packet['draft']):
                y=0 if token==packet['checks'][j] else 1
                if y==0 and token==getattr(self,'eos_token_id',None):y=2
                labels.append(y)
                if y:break
            self.budget_frames.append(dict(x=x.cpu().tolist(),labels=labels,
                proposed=len(packet['draft']),remaining=packet['remaining'],prefix=packet['prefix']))
        super().round(packet)


def mapper_records(payloads,mapper):
    last=mapper.layers[-1]
    return [dict(r,source_index=p['source_index'],length=p['length'],trajectory=p.get('phase',str(i)),
                 relative=mapper.layers.index(last)) for i,p in enumerate(payloads) for r in p['records']]


def fit_mapper(student,model,payloads,output,label,epochs=4):
    records=mapper_records(payloads,model)
    held=max(p['source_index'] for p in payloads)
    fit=[r for r in records if r['source_index']!=held]
    monitor=[r for r in records if r['source_index']==held]
    model.requires_grad_(False)
    rel=model.layers.index(len(student.model.layers)-1)
    for name in ('encoders','decoders','native','query','gates','residuals'):
        getattr(model,name)[rel].requires_grad_(True)
    student.requires_grad_(False)
    native={}
    with torch.no_grad():
        for r in records:native[id(r)]=replay(student,model,r,strength=0.)[0].detach()
    def evaluate():
        rows=[]
        with torch.no_grad():
            for r in monitor:
                logits,_=replay(student,model,r)
                q=distribution_quality(logits,r['target'].to(logits.device))
                good=int(native[id(r)].argmax())==int(r['target'].argmax())
                rows.append(dict(**q,native_correct=good))
        return dict(queries=len(rows),matches=sum(r['top1_matches_target'] for r in rows),
            corrected=sum(r['top1_matches_target'] and not r['native_correct'] for r in rows),
            harmed=sum(not r['top1_matches_target'] and r['native_correct'] for r in rows),
            kl=sum(r['target_kl'] for r in rows)/len(rows))
    baseline=evaluate();best=baseline;state=copy.deepcopy(model.state_dict());epochbest=0
    parameters=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(parameters,lr=2e-4,weight_decay=.01)
    groups={}
    for r in fit:groups.setdefault((r['trajectory'],r['source_index'],r['length'],r['prefix']),[]).append(r)
    groups=[sorted(v,key=lambda r:r['query_index']) for v in groups.values()]
    rng=random.Random(612);history=[]
    for epoch in range(epochs):
        rng.shuffle(groups);losses=[]
        for group in groups:
            opt.zero_grad(set_to_none=True);margins=[];terms=[]
            for r in group:
                logits,penalty=replay(student,model,r);target=r['target'].to(logits.device).float()
                y=target.argmax();wrong=logits.clone();wrong[y]=-torch.inf
                margin=logits[y]-wrong.max();margins.append(margin)
                n=native[id(r)];nw=n.clone();nw[y]=-torch.inf
                native_margin=(n[y]-nw.max()).detach()
                lp,lq=target.log_softmax(-1),logits.log_softmax(-1)
                kl=(lp.exp()*(lp-lq)).sum()
                # Keep already-correct predictions; focus error gradients on
                # the logits that decide actual greedy acceptance.
                preserve=(native_margin>0).float()*F.relu(native_margin-margin)
                terms.append(.10*kl+F.softplus(.2-margin)+preserve+.01*penalty)
            probabilities=torch.stack(margins).sigmoid()
            # This is a fixed-state prefix surrogate. Recollection supplies
            # changed history/refresh states, not gradient through old history.
            loss=torch.stack(terms).mean()-.5*probabilities.cumprod(0).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.);opt.step()
            losses.append(float(loss.detach()))
        value=evaluate();history.append(dict(epoch=epoch+1,loss=sum(losses)/len(losses),monitor=value))
        if (value['matches'],-value['harmed'],-value['kl'])>(best['matches'],-best['harmed'],-best['kl']):
            best,state,epochbest=value,copy.deepcopy(model.state_dict()),epoch+1
        print('architecture mapper',label,history[-1],flush=True)
    model.load_state_dict(state);model.eval().requires_grad_(False)
    torch.save(dict(config=model.config,state_dict=model.state_dict()),output/f'{label}.pt')
    result=dict(fit_queries=len(fit),monitor_queries=len(monitor),baseline=baseline,selected=best,
        epoch=epochbest,history=history,trained='All final-layer mapper blocks; frozen base, no historical backprop',
        objective='Greedy margin, native-success preservation, KL and differentiable prefix survival; no per-query V oracle',
        selection='Monitor matches, then fewer harms, then KL; initial weights eligible')
    write(output/f'{label}_training.json',result)
    return model,result


def fit_budget(payloads,cost,output,label,epochs=250):
    held=max(p['source_index'] for p in payloads)
    fit=[r for p in payloads if p['source_index']!=held for r in p['budget_frames'] if r['remaining']>4]
    monitor=[r for p in payloads if p['source_index']==held for r in p['budget_frames'] if r['remaining']>4]
    model=StepBudget().cpu()
    def arrays(rows):
        x=torch.tensor([r['x'] for r in rows]);mask=torch.stack([compatible_classes(r['labels'],r['proposed']) for r in rows])
        tables=[outcome_tables(r['prefix'],r['remaining'],cost) for r in rows]
        return x,mask,torch.stack([v[0] for v in tables]),torch.stack([v[1] for v in tables])
    x,y,c,t=arrays(fit);xm,ym,cm,tm=arrays(monitor)
    with torch.no_grad():
        model.mean.copy_(x.mean(0));model.scale.copy_(x.std(0,unbiased=False).clamp_min(.05))
        counts=torch.ones(9)*.5
        for m in y:
            if int(m.sum())==1:counts+=m.float()
        model.prior.copy_(counts/counts.sum())
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.02)
    best=float('inf');state=copy.deepcopy(model.state_dict());bestepoch=0;history=[]
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True);lp=model(x).log_softmax(-1)
        # Right-censored short trajectories contribute log probability of the
        # whole compatible set. No unverified tail is marked as rejection.
        loss=-lp.masked_fill(~y,-torch.inf).logsumexp(-1).mean()
        # Resolve only fully observed classes for the decision-sensitive term.
        p=lp.exp();rate=(p[:,:,None]*c).sum(1)/(p[:,:,None]*t).sum(1).clamp_min(1e-5)
        actual=(y.float()[:,:,None]*c).sum(1)/(y.float()[:,:,None]*t).sum(1).clamp_min(1e-5)
        observed=y.sum(-1)==1
        sign=(actual[:,0]-actual[:,1]).sign()
        if observed.any():loss=loss+.2*F.softplus(-sign[observed]*(rate[observed,0]-rate[observed,1])/10).mean()
        loss.backward();opt.step()
        with torch.no_grad():
            value=float(-model(xm).log_softmax(-1).masked_fill(~ym,-torch.inf).logsumexp(-1).mean())
        if value<best:best,state,bestepoch=value,copy.deepcopy(model.state_dict()),epoch+1
        if (epoch+1)%50==0:history.append(dict(epoch=epoch+1,fit=float(loss.detach()),monitor_nll=value))
    model.load_state_dict(state);model.eval().requires_grad_(False)
    torch.save(model.payload(),output/f'{label}.pt')
    result=dict(fit_frames=len(fit),monitor_frames=len(monitor),fit_censored=int((y.sum(-1)>1).sum()),
        monitor_censored=int((ym.sum(-1)>1).sum()),selected_epoch=bestepoch,monitor_nll=best,history=history,
        contract='12 causal features at draft depth2; exact-class/partial-class likelihood, bounded actions2/4',
        training='Mixed fixed4 and policy/exploration states; no assumed labels beyond verified prefix')
    write(output/f'{label}_training.json',result)
    return model,result
