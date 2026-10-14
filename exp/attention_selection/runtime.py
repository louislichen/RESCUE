"""Greedy SD with committed-prefix retrieval; all acquisition costs are timed."""
from __future__ import annotations

import json
import random
import time

import torch

from selection import retrieve


def sync():
    torch.cuda.synchronize()


@torch.no_grad()
def teacher_reference(e,item):
    from run import core_forward,legacy_cache,tensor_ids
    e.tc.capture=False
    sync()
    started=time.perf_counter()
    out=core_forward(e.teacher,tensor_ids(item['ids'][:-1],e.a.device))
    cache=legacy_cache(out.past_key_values)
    sync()
    prefill=time.perf_counter()-started
    generated=[]
    current=item['ids'][-1]
    started=time.perf_counter()
    for _ in range(e.a.max_new_tokens):
        out=e.forward(e.teacher,current,cache)
        current=int(e.logit(e.teacher,out).argmax())
        generated.append(current)
        cache=legacy_cache(out.past_key_values)
        if current==e.st.eos_token_id:
            break
    sync()
    decode=time.perf_counter()-started
    return dict(source_index=item['source_index'],method='teacher_greedy',budget=0,generated=generated,
                tokens=len(generated),prefill_seconds=prefill,decode_seconds=decode,total_seconds=prefill+decode,
                decode_tps=len(generated)/decode,total_tps=len(generated)/(prefill+decode),output_agrees=True)


@torch.no_grad()
def speculative(e,item,method,budget,chosen,pairs,*,profile=False):
    from run import core_forward,legacy_cache,tensor_ids,trim
    e.sc.enabled=False
    e.sc.capture=False
    e.tc.capture=False
    e.sc.profile=False
    sync()
    start=time.perf_counter()
    ids=item['ids']
    tout=core_forward(e.teacher,tensor_ids(ids[:-1],e.a.device))
    tc=legacy_cache(tout.past_key_values)
    sout=core_forward(e.student,tensor_ids(ids[:-1],e.a.device))
    sc=legacy_cache(sout.past_key_values)
    # Query-row collection is part of prefill and charged to every SD route.
    e.tc.capture=e.sc.capture=True
    tout=e.forward(e.teacher,ids[-1],tc)
    sout=e.forward(e.student,ids[-1],sc)
    tbank,_,_=e.tc.snapshot()
    sbank,visible,q=e.sc.snapshot()
    sync()
    prefill=time.perf_counter()-start
    timings=dict(selection=0.0,cache_update=0.0,draft=0.0,verify=0.0)
    proposed=accepted=0
    generated=[]
    blocks=[]
    current=ids[-1]
    last_retrieval=-1
    frontier=len(ids)
    state_selection=None
    bias=None
    sync()
    decode_start=time.perf_counter()
    while len(generated)<e.a.max_new_tokens:
        sync()
        tick=time.perf_counter()
        # Every bank row was produced from the same committed query, with
        # verification tails removed before retrieval. No future row is used.
        if method in ('native_full','bias_full'):
            state_selection=None
        elif method=='learned_global':
            state_selection=e.learned_retrieval(tbank,sbank,q,budget)
        elif method.startswith('layer_'):
            route=method.removeprefix('layer_')
            state_selection=[retrieve('teacher_token',tbank[pairs[route][l]:pairs[route][l]+1],
                              sbank[l:l+1],budget,e.a.recent) for l in range(sbank.shape[0])]
        else:
            rule='teacher_token' if method=='bias_teacher' else method.removesuffix('_available')
            ratio=chosen['ratios'].get(f'{rule}_{budget}',0.5)
            state_selection=retrieve(rule,tbank,sbank,budget,e.a.recent,e.a.chunk,ratio,
                                     e.a.seed+item['source_index']+len(blocks),visible=visible)
        if method.startswith('bias_'):
            layer,tl,strength,_=chosen['bias']['pairing']
            bias=(layer,tbank[tl],strength)
            e.sc.configure(state_selection,bias_layer=layer,prior=tbank[tl],strength=strength)
        else:
            e.sc.configure(state_selection)
        sync()
        timings['selection']+=time.perf_counter()-tick
        e.sc.profile=profile
        e.sc.capture=True
        k=min(e.a.draft_k,e.a.max_new_tokens-len(generated))
        draft=[]
        student_rows=[]
        next_input=current
        sync()
        tick=time.perf_counter()
        for _ in range(k):
            sout=e.forward(e.student,next_input,sc)
            sc=legacy_cache(sout.past_key_values)
            next_input=int(e.logit(e.student,sout).argmax())
            draft.append(next_input)
            student_rows.append(e.sc.snapshot())
            if next_input==e.st.eos_token_id:
                break
        sync()
        timings['draft']+=time.perf_counter()-tick
        tick=time.perf_counter()
        e.tc.capture=False
        tout=core_forward(e.teacher,tensor_ids([current]+draft,e.a.device),tc,attentions=True)
        checks=e.teacher.lm_head(tout.last_hidden_state)[0,:,:e.vocab].float().argmax(-1).tolist()
        sync()
        timings['verify']+=time.perf_counter()-tick
        count=0
        for index,token in enumerate(draft):
            if token!=checks[index]:
                break
            count+=1
            if token==e.st.eos_token_id:
                break
        proposed+=len(draft)
        accepted+=count
        context=len(ids)+len(generated)
        generated.extend(draft[:count])
        stop=bool(generated and generated[-1]==e.st.eos_token_id) or len(generated)>=e.a.max_new_tokens
        if not stop:
            current=int(checks[count])
            generated.append(current)
            stop=current==e.st.eos_token_id or len(generated)>=e.a.max_new_tokens
        blocks.append(dict(proposed=len(draft),accepted=count,committed_before=context,
                           verification_query_index=count,committed_key_length=context+count))
        sync()
        tick=time.perf_counter()
        if not stop:
            if count==len(draft):
                # Bring the student's full backing cache up to the final
                # accepted draft token before the teacher bonus is consumed.
                sout=e.forward(e.student,draft[-1],sc)
                sc=legacy_cache(sout.past_key_values)
                sbank,visible,q=e.sc.snapshot()
            else:
                sbank,visible,q=student_rows[count]
            committed=context+count
            tbank=torch.stack([row[0,:,count,:committed].float().mean(0) for row in tout.attentions])
            sbank=sbank[:,:committed]
            visible=visible[:,:committed]
            assert tbank.shape==sbank.shape==visible.shape
            # No rejected speculative tail can survive into either cache.
            tc=trim(tout.past_key_values,committed)
            sc=trim(sc,committed)
            assert all(k.shape[-2]==committed for k,v in tc)
            assert all(k.shape[-2]==committed for k,v in sc)
        sync()
        timings['cache_update']+=time.perf_counter()-tick
        if stop:
            break
    sync()
    decode=time.perf_counter()-decode_start
    gather=e.sc.gather_seconds() if profile else None
    e.sc.profile=False
    e.tc.capture=e.sc.capture=False
    e.sc.enabled=False
    result=dict(source_index=item['source_index'],method=method,budget=budget,tokens=len(generated),
                generated=generated,proposed=proposed,accepted=accepted,blocks=len(blocks),
                mean_accepted=accepted/len(blocks),acceptance_rate=accepted/max(1,proposed),
                tokens_per_block=len(generated)/len(blocks),prefill_seconds=prefill,decode_seconds=decode,
                total_seconds=prefill+decode,decode_tps=len(generated)/decode,total_tps=len(generated)/(prefill+decode),
                stage_seconds=timings,profile=profile,gather_gpu_seconds=gather,
                visibility='teacher full committed prefix; student observed working positions only',
                refresh='every verification block',extra_teacher_query_forwards=0,
                teacher_prefill_query_rows=1,attention_collection_included=True,
                working_budget='B at retrieval, including sink and recent; grows by at most draft_k between retrievals',
                full_backing_kv_retained=True,trace=blocks)
    return result


def run_runtime(e):
    from run import append,rows,write
    e.load_selector()
    chosen=json.loads(e.selected_path.read_text())
    pairs=json.loads((e.out/'layer_pairs.json').read_text())
    # The forced nonzero bias controls remain in E5. Deployment also considers
    # zero intervention, selected solely by validation, before touching test.
    validation_native=[]
    for item in e.data['validation']:
        bundle=torch.load(e.doc_path(item),weights_only=False)
        validation_native.extend(s['native']['kl'] for s in bundle['states'])
    layer,tl,strength,shuffled=chosen['bias']['pairing']
    calibration=json.loads((e.out/'calibration.json').read_text())
    matched=[r['kl'] for r in calibration if r['kind']=='bias' and r['route']=='pairing'
             and r['layer']==layer and r['strength']==strength]
    native_kl=sum(validation_native)/len(validation_native)
    active_kl=sum(matched)/len(matched)
    if active_kl>=native_kl:
        chosen['bias']['pairing'][2]=0.0
    bias_validation=dict(native_kl=native_kl,active_kl=active_kl,
                         forced_strength=strength,selected_strength=chosen['bias']['pairing'][2])
    write(e.out/'runtime_bias_selection.json',bias_validation)
    path=e.out/'runtime_test.jsonl'
    done={(r['source_index'],r['budget'],r['method']) for r in rows(path)}
    profile_path=e.out/'runtime_profile.jsonl'
    profile_done={(r['source_index'],r['budget'],r['method']) for r in rows(profile_path)}
    methods=['recent','random','student_chunk','teacher_chunk','student_token_available','teacher_token',
             'union_available','layer_depth','layer_pairing','layer_random','learned_global','bias_teacher']
    plan=[('native_full',0),('bias_full',0)]+[(m,b) for b in e.a.budgets for m in methods]
    write(e.out/'runtime_protocol.json',dict(methods=plan,test_docs=len(e.data['test']),
        acceptance='accepted draft tokens / all proposed tokens; mean accepted draft tokens per verification block',
        prior='completed prefill/verification query; fresh current-query teacher is offline-only',
        unavailable_student='masked availability; no zero imputation in disagreement rules',
        timing='CUDA synchronization around stages; selection, full prefill, attention extraction and cache updates included',
        profiles='Separate instrumented reruns; gather CUDA events not included in primary throughput runs',
        numerical='BF16 eager block verification; output agreement checked against sequential teacher reference',
        bias_validation=bias_validation))
    for index,item in enumerate(e.data['test']):
        teacher_key=(item['source_index'],0,'teacher_greedy')
        if teacher_key not in done:
            reference=teacher_reference(e,item)
            bundle=torch.load(e.doc_path(item),weights_only=False)
            assert reference['generated']==bundle['reference'], 'Reference trajectory changed across phases'
            append(path,reference)
            done.add(teacher_key)
        else:
            reference=next(r for r in rows(path) if (r['source_index'],r['budget'],r['method'])==teacher_key)
        randomized=list(plan)
        random.Random(e.a.seed+item['source_index']).shuffle(randomized)
        for method,b in randomized:
            if (item['source_index'],b,method) in done:
                continue
            result=speculative(e,item,method,b,chosen,pairs)
            result['output_agrees']=result['generated']==reference['generated']
            result['teacher_decode_tps']=reference['decode_tps']
            result['teacher_total_tps']=reference['total_tps']
            append(path,result)
            done.add((item['source_index'],b,method))
        print(f'runtime {index+1}/{len(e.data["test"])} source={item["source_index"]} methods={len(plan)}',flush=True)
    first=e.data['test'][0]
    for method,b in [('native_full',0),('teacher_chunk',e.a.budgets[0]),('teacher_token',e.a.budgets[0]),
                     ('union_available',e.a.budgets[0]),('layer_pairing',e.a.budgets[0]),('learned_global',e.a.budgets[0])]:
        if (first['source_index'],b,method) not in profile_done:
            result=speculative(e,first,method,b,chosen,pairs,profile=True)
            append(profile_path,result)
    print('E6 runtime complete',flush=True)
