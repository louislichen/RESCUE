"""Query-direction distillation and lightweight attention-risk training."""
import copy
import json
import random
from pathlib import Path
import torch
import torch.nn.functional as F

from .compare_baselines import write
from .mapper_capacity import OracleValues
from .objectives import distribution_quality,margin_guard_loss
from .query_mapper import QueryValueMapper
from .rollout_training import replay
from .policy import RiskModel


def train_mapper(student,old,payloads,output,epochs=12):
    student.eval().requires_grad_(False)
    held=max(p['source_index'] for p in payloads)
    fit=[r for p in payloads if p['source_index']!=held for r in p['records']]
    monitor=[r for p in payloads if p['source_index']==held for r in p['records']]
    network=QueryValueMapper(**old.config).to(student.device)
    missing=network.load_state_dict(old.state_dict(),strict=False)
    assert not missing.unexpected_keys and all(n.startswith('residuals.') for n in missing.missing_keys)
    network.requires_grad_(False)
    relative=network.layers.index(len(student.model.layers)-1)
    network.residuals[relative].requires_grad_(True);network.gates[relative].requires_grad_(True)
    native={};targets={};oracle_rows=[]
    for i,r in enumerate(fit+monitor):
        with torch.no_grad():
            native[id(r)]=replay(student,old,r,strength=0.)[0].detach()
    # Per-query privileged optimization is restricted to FIT documents.
    for i,r in enumerate(fit):
        y=r['target'].to(student.device).argmax()
        logits=native[id(r)];correct=int(logits.argmax())==int(y)
        best=torch.zeros_like(r['native'],device=student.device)
        margin0=float(logits[y]-torch.cat((logits[:y],logits[y+1:])).max())
        best_margin=margin0;steps=0
        if not correct:
            oracle=OracleValues(r['native'].to(student.device));opt=torch.optim.Adam(oracle.parameters(),lr=.1)
            for step in range(24):
                opt.zero_grad(set_to_none=True)
                scores,penalty=replay(student,oracle,r)
                wrong=scores.clone();wrong[y]=-torch.inf
                margin=float((scores[y]-wrong.max()).detach())
                if margin>best_margin:
                    best_margin=margin;best=oracle.raw.detach().tanh().clone()
                if int(scores.argmax())==int(y): break
                loss=F.cross_entropy(scores[None],y[None])+.01*penalty
                loss.backward();opt.step();steps=step+1
            with torch.no_grad():
                scores,_=replay(student,oracle,r);wrong=scores.clone();wrong[y]=-torch.inf
                margin=float(scores[y]-wrong.max())
                if margin>best_margin: best_margin=margin;best=oracle.raw.detach().tanh().clone()
        targets[id(r)]=best
        oracle_rows.append(dict(fit_query=i,native_correct=correct,initial_margin=margin0,
            oracle_margin=best_margin,oracle_correct=best_margin>0,steps=steps))
    write(output/'oracle_fit.json',dict(scope='Teacher-aware auxiliary FIT targets only; monitor/confirmation never generate oracle targets',queries=oracle_rows))
    def evaluate(model):
        out=[]
        with torch.no_grad():
            for r in monitor:
                logits,_=replay(student,model,r);target=r['target'].to(student.device)
                q=distribution_quality(logits,target)
                before=int(native[id(r)].argmax())==int(target.argmax())
                out.append(dict(**q,native_correct=before))
        return dict(queries=len(out),matches=sum(r['top1_matches_target'] for r in out),
            corrected=sum(r['top1_matches_target'] and not r['native_correct'] for r in out),
            harmed=sum(not r['top1_matches_target'] and r['native_correct'] for r in out),
            kl=sum(r['target_kl'] for r in out)/len(out))
    baseline=evaluate(old);initial=evaluate(network);best=initial;best_state=copy.deepcopy(network.state_dict());best_epoch=0
    hard=[r for r in fit if int(native[id(r)].argmax())!=int(r['target'].argmax())]
    optimizer=torch.optim.AdamW([p for p in network.parameters() if p.requires_grad],lr=.001,weight_decay=.01)
    rng=random.Random(81);history=[]
    for epoch in range(epochs):
        # Train on every fit query, repeat difficult queries once to prevent
        # already-correct positions dominating the value-direction objective.
        order=list(fit)+list(hard);rng.shuffle(order);losses=[]
        for r in order:
            optimizer.zero_grad(set_to_none=True)
            logits,penalty=replay(student,network,r)
            device=student.device
            source,nv,q,features=(r['source'].to(device),r['native'].to(device),
                r['normalized_hidden'][0,-1].to(device),r['token_features'].to(device))
            corrected,_,_=network.correction(r['relative'],source,nv,q,features,alpha=r['strength'])
            scale=nv.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-4)
            auxiliary=((corrected-nv)/scale-targets[id(r)]).square().mean()
            loss=margin_guard_loss(logits,r['target'].to(device),penalty,native[id(r)])+2*auxiliary
            loss.backward();torch.nn.utils.clip_grad_norm_([p for p in network.parameters() if p.requires_grad],1.)
            optimizer.step();losses.append(float(loss.detach()))
        value=evaluate(network);history.append(dict(epoch=epoch+1,fit_loss=sum(losses)/len(losses),monitor=value))
        if (value['matches'],-value['kl'])>(best['matches'],-best['kl']):
            best,best_state,best_epoch=value,copy.deepcopy(network.state_dict()),epoch+1
        print('direction epoch',history[-1],flush=True)
    network.load_state_dict(best_state);network.eval().requires_grad_(False)
    torch.save(dict(config=network.config,state_dict=network.state_dict()),output/'mapper_query.pt')
    promoted=best['matches']>baseline['matches']
    result=dict(fit_queries=len(fit),monitor_queries=len(monitor),monitor_document=held,
        baseline=baseline,initial_query_mapper=initial,selected=best,selected_epoch=best_epoch,promoted_for_length=promoted,
        selection='Monitor greedy matches first; KL breaks ties. Downstream replacement requires strictly more matches.',
        oracle_corrected=sum(not r['native_correct'] and r['oracle_correct'] for r in oracle_rows),
        objective='Teacher greedy margin/native-success preservation plus normalized V oracle distillation; train residual vector head and last gate only',history=history)
    write(output/'mapper_training.json',result)
    return network,promoted


def train_risk(payloads,output):
    held=max(p['source_index'] for p in payloads)
    frames=[dict(source_index=p['source_index'],length=p['length'],monitor=p['source_index']==held,
        **{k:v.tolist() if isinstance(v,torch.Tensor) else v for k,v in r.items()}) for p in payloads for r in p['frames']]
    write(output/'risk_frames.json',frames)
    fit=[f for f in frames if not f['monitor']];monitor=[f for f in frames if f['monitor']]
    model=RiskModel(group='attention_injected_light').cpu()
    def arrays(group):
        x=torch.tensor([r['features'] for r in group]);y=torch.full((len(group),12),-100,dtype=torch.long)
        for i,r in enumerate(group): y[i,:len(r['labels'])]=torch.tensor(r['labels'])
        return x,y
    x,y=arrays(fit);xm,ym=arrays(monitor)
    with torch.no_grad():
        model.mean.copy_(x[:,model.indices].mean(0));model.scale.copy_(x[:,model.indices].std(0,unbiased=False).clamp_min(.01))
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    best=float('inf');state=None;epochbest=0
    for epoch in range(250):
        opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x).flatten(0,1),y.flatten())
        loss.backward();opt.step()
        with torch.no_grad(): value=float(F.cross_entropy(model(xm).flatten(0,1),ym.flatten()))
        if value<best: best,state,epochbest=value,copy.deepcopy(model.state_dict()),epoch+1
    model.load_state_dict(state)
    counts=torch.full((12,3),.5)
    for r in fit:
        for j,label in enumerate(r['labels']): counts[j,label]+=1
    prior=(counts/counts.sum(-1,keepdim=True)).tolist()
    torch.save(dict(group=model.group,horizon=12,state_dict=state,temperature=1.),output/'risk_light.pt')
    write(output/'risk_training.json',dict(fit_frames=len(fit),monitor_frames=len(monitor),monitor_document=held,
        selected_epoch=epochbest,monitor_nll=best,prior=prior,
        scope='New matched selector/deployed mapper rollouts; read-error columns excluded, no monitor gradient updates'))
    return model.eval().requires_grad_(False),prior
