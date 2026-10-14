"""Causal workset/horizon planning and exact greedy speculative decoding."""
import json
import math
import random
import time

import numpy as np
import torch

from experiment import (append, read_rows, write, sync, core_forward, tensor_ids,
                        legacy_cache, trim, confidence)
from selection import ReadState, feature_vector
from risk import load, outcome_labels

MENU = (1, 2, 4, 8, 12)


class HistoricalRisk:
    """Cheap baseline updated only after completed target verification."""
    def __init__(self,e,rate):
        if not hasattr(e,'history_prior'):
            rows=read_rows(e.out/'branches_train.jsonl')
            count={d:sum(r['source_index']==d for r in rows) for d in {r['source_index'] for r in rows}}
            numer=np.zeros(12);denom=np.zeros(12)
            for r in rows:
                w=1/count[r['source_index']]
                for j,(label,valid) in enumerate(zip(r['labels'],r['mask'])):
                    if valid:
                        numer[j]+=w*(label==0);denom[j]+=w
            prior=(numer/denom.clip(min=1e-9)).clip(.01,.99)
            write(e.out/'historical_prior.json',dict(continue_probability=prior.tolist(),source='training at-risk hazards, document weighted'))
            e.history_prior=torch.tensor(prior,dtype=torch.float32,device=e.a.device)
        self.p=e.history_prior.clone();self.rate=rate

    def continuation(self,vector):
        return self.p.cumprod(0)

    def update(self,draft,accepted,eos):
        labels,mask=outcome_labels(draft,accepted,eos)
        indices=torch.tensor(mask,device=self.p.device)
        target=torch.tensor([float(v==0) for v in labels],device=self.p.device)
        self.p[indices]=(1-self.rate)*self.p[indices]+self.rate*target[indices]
        self.p.clamp_(.01,.99)


def features(e, t, s, selected, progress, age, conf):
    return feature_vector(t, s, selected, progress, age, conf,
                          e.a.max_new_tokens-progress, e.a.recent)


def cheap_features(e,n,budget,progress,conf):
    vector=conf.new_zeros(31)
    vector[:5]=conf.new_tensor([math.log(n)/10,progress/64,budget/n,0.,(e.a.max_new_tokens-progress)/64])
    vector[28:]=conf
    return vector


def cost_fit(e):
    """Training measurements only; conservative positive linear cost fit."""
    rows = read_rows(e.out/'branches_train.jsonl')
    samples = [(r['prefix_length'], int(k), v) for r in rows
               for k, v in r.get('verify_profile', {}).items()]
    x = np.array([[1., n/2048, k, k*n/2048] for n,k,_ in samples])
    y = np.array([v for _,_,v in samples])
    # NNLS without a scipy dependency: projected coordinate descent.
    beta = np.zeros(4)
    for _ in range(400):
        for j in range(4):
            beta[j] = max(0., np.dot(x[:,j], y-x@beta+x[:,j]*beta[j]) / max(1e-9,np.dot(x[:,j],x[:,j])))
    result = dict(verify_beta=beta.tolist(),
                  student_step=float(np.median([v for r in rows for v in r['step_seconds']])),
                  cache_update=float(np.median([r['cache_seconds'] for r in rows
                      if r['accepted']<r['proposed'] or r['terminal']])),
                  training_verify_mae=float(np.abs(x@beta-y).mean()), samples=len(y),
                  formula='verify=b0+b1*N/2048+b2*k+b3*k*N/2048; expected tokens=1+sum P(first j accepted non-EOS)')
    write(e.out/'cost_model.json',result)
    e.cost=result
    return result


def risk_model(e, group):
    if group not in e.risks:
        chosen=json.loads((e.out/'risk_selection.json').read_text())
        e.risks[group]=load(e.out/chosen[group]['path'],e.a.device)[0]
    return e.risks[group]


def choose_length(e, model, vector, n, remaining, spent, age=0, online=False):
    survival=model.continuation(vector).clamp(0,1)
    # k is an absolute cap in this block. A current frame predicts outcomes at
    # absolute positions, with known draft age as an input (no verification yet).
    caps=sorted(set([age] if age else []) | {min(k,remaining) for k in MENU if k>=age})
    caps=[k for k in caps if k>0 and k>=age]
    beta=vector.new_tensor(e.cost['verify_beta'])
    scores=[]
    for k in caps:
        # E[min(remaining, committed)] truncates the survival SUM, not its
        # expectation. At k==remaining there is no space for a bonus token.
        expected=1+survival[:min(k,remaining-1)].sum()
        verification=beta@vector.new_tensor([1.,n/2048,k,k*n/2048])
        cost=(spent+(k-age)*e.cost['student_step']+verification+
              e.cost['cache_update']+(survival[k-1]*e.cost['student_step'] if k<remaining else 0.))
        scores.append(expected/cost.clamp_min(1e-5))
    best=int(torch.stack(scores).argmax())
    return caps[best],float(scores[best])


@torch.no_grad()
def speculative(e,item,method,config,collect=False,profile=False):
    if config.get('inject'):
        e.load_mapper()
    if e.injection is not None:
        e.injection.clear()
    tc,sc,tr,sr,conf,prefill=e.prefill(item)
    probing=config.get('probe',True)
    need_confidence=probing or config.get('risk')=='confidence'
    observer,capture=e.sc.reference_observer,e.sc.capture
    if not probing:
        assert config.get('risk') in (None,'structural','confidence') and not config.get('inject') and config['rule'] in ('teacher','native_full')
        e.sc.reference_observer=None
        e.sc.capture=False
    historical=HistoricalRisk(e,config['history']) if 'history' in config else None
    generated=[]; blocks=[]; previous={}
    current=item['ids'][-1]
    stages=dict(selection=0.,decision=0.,source_pack=0.,draft=0.,verify=0.,cache_update=0.)
    proposed=accepted=0
    sync(); started=time.perf_counter()
    while len(generated)<e.a.max_new_tokens:
        block_start=time.perf_counter()
        block_input=current
        n=tc[-1][0].shape[-2]
        progress=len(generated); remaining=e.a.max_new_tokens-progress
        t,s=e.banks(tc,sc,tr,sr)
        rule=config['rule']
        tick=time.perf_counter()
        rules=list(dict.fromkeys(['teacher','read_teacher',rule])) if config.get('joint') else [rule]
        candidates=[]
        for candidate in rules:
            chosen=(torch.arange(n,device=e.a.device) if candidate=='native_full'
                    else e.make_selection(candidate,t,s,previous.get(candidate)))
            previous[candidate]=chosen
            candidates.append((candidate,chosen))
        sync(); stages['selection']+=time.perf_counter()-tick
        tick=time.perf_counter()
        group=config.get('risk')
        root_vector=None
        if group:
            model=risk_model(e,group)
            scored=[]
            vectors=[features(e,t,s,chosen,progress,0,conf) if probing else
                     cheap_features(e,n,len(chosen),progress,conf) for _,chosen in candidates]
            sync(); spent=time.perf_counter()-block_start
            for (candidate,chosen),vector in zip(candidates,vectors):
                cap,score=choose_length(e,model,vector,n,remaining,spent)
                scored.append((score,cap,candidate,chosen,vector))
            _,cap,rule,selection,root_vector=max(scored,key=lambda r:r[0])
        elif historical is not None:
            rule,selection=candidates[0]
            sync()
            cap,_=choose_length(e,historical,tr.new_zeros(31),n,remaining,time.perf_counter()-block_start)
        else:
            rule,selection=candidates[0]
            cap=min(config.get('k',4),remaining)
        sync(); stages['decision']+=time.perf_counter()-tick
        e.sc.configure(None if rule=='native_full' else selection)
        tick=time.perf_counter()
        if config.get('inject'):
            e.bind_injection(tc,selection,t)
        elif e.injection is not None:
            e.injection.clear()
        sync(); stages['source_pack']+=time.perf_counter()-tick
        frames=[]
        if collect:
            if root_vector is None:
                root_vector=features(e,t,s,selection,progress,0,conf)
            frames.append(dict(offset=0,features=root_vector.cpu().tolist()))
        draft=[]; probes=[]; steps=[]; next_input=current
        while len(draft)<cap:
            tick=time.perf_counter()
            sout=e.forward(e.student,next_input,sc)
            sc=legacy_cache(sout.past_key_values)
            logits=e.logit(e.student,sout)
            conf_next=confidence(logits) if need_confidence else conf
            next_input=int(logits.argmax())
            draft.append(next_input)
            probes.append((e.shadow if probing else None,conf_next))
            sync(); elapsed=time.perf_counter()-tick
            stages['draft']+=elapsed; steps.append(elapsed)
            if next_input==e.st.eos_token_id:
                break
            if collect or (group and config.get('online')):
                tick=time.perf_counter()
                sv=ReadState(e.shadow,sc[-1][1][0])
                vector=features(e,t,sv,selection,progress,len(draft),conf_next)
                if collect:
                    frames.append(dict(offset=len(draft),features=vector.cpu().tolist()))
                if group and config.get('online'):
                    sync()
                    cap,_=choose_length(e,model,vector,n,remaining,time.perf_counter()-block_start,len(draft),True)
                sync(); stages['decision']+=time.perf_counter()-tick
        oldtc=tc
        tick=time.perf_counter()
        tout=core_forward(e.teacher,tensor_ids([current]+draft,e.a.device),oldtc,attentions=True)
        checks=e.teacher.lm_head(tout.last_hidden_state)[0,:,:e.vocab].float().argmax(-1).tolist()
        sync(); verify_seconds=time.perf_counter()-tick
        stages['verify']+=verify_seconds
        count=0
        for j,token in enumerate(draft):
            if token!=checks[j]:break
            count+=1
            if token==e.st.eos_token_id:break
        if historical is not None:
            historical.update(draft,count,e.st.eos_token_id)
        proposed+=len(draft); accepted+=count
        context=len(item['ids'])+progress
        generated.extend(draft[:count])
        stop=bool(generated and generated[-1]==e.st.eos_token_id) or len(generated)>=e.a.max_new_tokens
        if not stop:
            current=int(checks[count]); generated.append(current)
            stop=current==e.st.eos_token_id or len(generated)>=e.a.max_new_tokens
        committed=context+count
        tick=time.perf_counter()
        if not stop:
            if count==len(draft):
                sout=e.forward(e.student,draft[-1],sc)
                sc=legacy_cache(sout.past_key_values)
                next_probe=(e.shadow if probing else None,confidence(e.logit(e.student,sout)) if need_confidence else conf)
            else:
                next_probe=probes[count]
            tr=tout.attentions[27][0,:,count,:committed].float().detach()
            sr,conf=next_probe
            sr=sr[:,:committed] if probing else tr.new_full((e.student.config.num_attention_heads,committed),1/committed)
            tc,sc=trim(tout.past_key_values,committed),trim(sc,committed)
            assert tc[-1][0].shape[-2]==sc[-1][0].shape[-2]==tr.shape[-1]==sr.shape[-1]
        sync(); cache_seconds=time.perf_counter()-tick
        stages['cache_update']+=cache_seconds
        trace=dict(block_index=len(blocks),rule=rule,proposed=len(draft),accepted=count,
                   progress=progress,prefix_length=n,cap=cap,terminal=stop)
        blocks.append(trace)
        if collect:
            labels,mask=outcome_labels(draft,count,e.st.eos_token_id)
            timings={str(len(draft)):verify_seconds}
            if profile and len(blocks)%3==1:
                for k in MENU:
                    if k>=len(draft):continue
                    sync(); tick=time.perf_counter()
                    extra=core_forward(e.teacher,tensor_ids([block_input]+draft[:k],e.a.device),oldtc)
                    _=e.teacher.lm_head(extra.last_hidden_state)[0,:,:e.vocab].float().argmax(-1)
                    sync(); timings[str(k)]=time.perf_counter()-tick
                    del extra
            append(e.out/f'branches_{collect}.jsonl',dict(source_index=item['source_index'],method=method,
                **trace,frames=frames,labels=labels,mask=mask,verify_profile=timings,
                step_seconds=steps,cache_seconds=cache_seconds))
        if stop:break
    sync(); decode=time.perf_counter()-started
    if e.injection is not None:e.injection.clear()
    e.sc.enabled=False
    e.sc.reference_observer,e.sc.capture=observer,capture
    return dict(source_index=item['source_index'],method=method,config=config,generated=generated,
                tokens=len(generated),proposed=proposed,accepted=accepted,blocks=len(blocks),
                acceptance_rate=accepted/max(1,proposed),mean_accepted=accepted/max(1,len(blocks)),
                mean_draft=proposed/max(1,len(blocks)),prefill_seconds=prefill,decode_seconds=decode,
                total_seconds=prefill+decode,decode_tps=len(generated)/decode,
                total_tps=len(generated)/(prefill+decode),stage_seconds=stages,trace=blocks,
                extra_teacher_query_forwards=0,full_backing_cache_retained=True)


@torch.no_grad()
def run_methods(e,split,methods,filename=None,collect=False,profile=False):
    path=e.out/(filename or f'runtime_{split}.jsonl')
    old=read_rows(path)
    done={(r['source_index'],r['method']) for r in old}
    refs={r['source_index']:r for r in old if r['method']=='teacher_greedy'}
    for config in methods.values():
        if config.get('risk'):risk_model(e,config['risk'])
        if 'history' in config:HistoricalRisk(e,config['history'])
    if not old:
        # Warmup excludes evaluation documents and includes all custom kernels.
        speculative(e,e.data['train'][0],'warmup',dict(rule='read_joint025',k=4))
    for i,item in enumerate(e.data[split]):
        source=item['source_index']
        if source not in refs:
            refs[source]=e.reference(item)
            append(path,refs[source])
        order=list(methods.items())
        random.Random(e.a.seed+source).shuffle(order)
        for method,config in order:
            if (source,method) in done:continue
            if collect:
                # Atomic document/method completion: remove an interrupted partial
                # trajectory before re-running it, retaining all completed ones.
                bp=e.out/f'branches_{split}.jsonl'
                rows=read_rows(bp)
                rows=[r for r in rows if not (r['source_index']==source and r['method']==method)]
                bp.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result=speculative(e,item,method,config,split if collect else False,profile)
            result['output_agrees']=result['generated']==refs[source]['generated']
            append(path,result); done.add((source,method))
            assert result['output_agrees'],f'Greedy mismatch {split} {source} {method}'
        print(f'{path.stem} {i+1}/{len(e.data[split])} methods={len(methods)} all_outputs_match',flush=True)
