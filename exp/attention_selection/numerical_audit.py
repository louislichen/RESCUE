#!/usr/bin/env python3
"""Separate batch-shape rounding from causal/KV bugs; FP32 teacher recheck."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from argparse import Namespace

import torch

from run import Experiment, append, core_forward, legacy_cache, rows, tensor_ids, write
from runtime import speculative, teacher_reference


@torch.no_grad()
def compare_paths(e,item,prefix,position):
    ids=item['ids']
    e.tc.capture=False
    out=core_forward(e.teacher,tensor_ids(ids[:-1],e.a.device))
    cache=legacy_cache(out.past_key_values)
    for token in ([ids[-1]]+prefix[:position-1] if position else []):
        out=e.forward(e.teacher,token,cache)
        cache=legacy_cache(out.past_key_values)
    query=ids[-1] if position==0 else prefix[position-1]
    one=e.forward(e.teacher,query,cache)
    sequential=e.logit(e.teacher,one)
    tail=prefix[position:position+4]
    if not tail:
        tail=[17]
    block=core_forward(e.teacher,tensor_ids([query]+tail,e.a.device),cache)
    batched=e.teacher.lm_head(block.last_hidden_state)[0,0,:e.vocab].float()
    alternative=core_forward(e.teacher,tensor_ids([query]+[17]*len(tail),e.a.device),cache)
    causal=e.teacher.lm_head(alternative.last_hidden_state)[0,0,:e.vocab].float()
    top=sequential.topk(2)
    return dict(source_index=item['source_index'],position=position,
       dtype=str(next(e.teacher.parameters()).dtype),sequential_argmax=int(sequential.argmax()),
       block_argmax=int(batched.argmax()),block_vs_sequential_max_error=float((batched-sequential).abs().max()),
       sequential_top_margin=float(top.values[0]-top.values[1]),
       same_shape_future_change_max_error=float((causal-batched).abs().max()),
       same_shape_future_change_argmax_equal=bool(causal.argmax()==batched.argmax()),
       canonical_prefix_tokens=len(ids)+position-1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fp32-runtime',action='store_true')
    args=p.parse_args()
    config=json.loads(open('exp/attention_selection/out_stage2/config.json').read())
    allowed=['data','output','cache','teacher','student','device','budgets','query_steps','recent','chunk',
             'max_new_tokens','draft_k','utility_candidates','utility_joint','seed']
    a=Namespace(**{k:config[k] for k in allowed})
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    e=Experiment(a)
    runtime=rows(e.out/'runtime_test.jsonl')
    references={r['source_index']:r for r in runtime if r['method']=='teacher_greedy'}
    native={r['source_index']:r for r in runtime if r['method']=='native_full'}
    failure=[]
    for item in e.data['test']:
        r=native[item['source_index']]
        if not r['output_agrees']:
            reference=references[item['source_index']]['generated']
            divergent=next((i for i,(x,y) in enumerate(zip(r['generated'],reference)) if x!=y),min(len(r['generated']),len(reference)))
            if divergent<len(reference):
                failure.append((item,reference,divergent))
    sample=failure[:5]
    if not sample:
        item=e.data['test'][0]
        sample=[(item,references[item['source_index']]['generated'],0)]
    bf16=[compare_paths(e,*case) for case in sample]
    e.teacher.to(dtype=torch.float32)
    fp32=[compare_paths(e,*case) for case in sample]
    write(e.out/'numerical_audit.json',dict(
        bf16=bf16,fp32=fp32,native_divergent_docs=len(failure),
        method='Same canonical prefix and current query; compare one-query and multi-query shapes. Change only future masked input to test causality. FP32 uses forced same prefix, not a new algorithm.',
        source_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest()))
    print('Numerical audit:',json.dumps(dict(bf16=bf16,fp32=fp32)),flush=True)
    if not args.fp32_runtime:
        return
    e.load_selector()
    # Warm both precisions before timing, on training-prefix inputs only.
    e.sc.enabled=False
    e.sc.capture=e.tc.capture=False
    warm_ids=e.data['train'][0]['ids'][:512]
    for model in (e.teacher,e.student):
        out=core_forward(model,tensor_ids(warm_ids,e.a.device))
        cache=legacy_cache(out.past_key_values)
        for _ in range(4):
            out=e.forward(model,17,cache)
            cache=legacy_cache(out.past_key_values)
    torch.cuda.synchronize()
    chosen=json.loads(e.selected_path.read_text())
    runtime_bias=json.loads((e.out/'runtime_bias_selection.json').read_text())
    chosen['bias']['pairing'][2]=runtime_bias['selected_strength']
    pairs=json.loads((e.out/'layer_pairs.json').read_text())
    path=e.out/'runtime_fp32_test.jsonl'
    done={(r['source_index'],r['budget'],r['method']) for r in rows(path)}
    plan=[('native_full',0),('bias_full',0)]+[(m,b) for b in a.budgets for m in
         ('teacher_chunk','teacher_token','union_available','layer_pairing','learned_global','bias_teacher')]
    write(e.out/'runtime_fp32_protocol.json',dict(teacher_dtype='float32',student_dtype='bfloat16',
        model_weights='same checkpoints, cast frozen teacher to FP32',matmul_allow_tf32=False,
        methods=plan,test_docs=len(e.data['test']),
        hyperparameters='Frozen original BF16 validation choices; no tuning on test or FP32 recheck',
        timing='Same full acquisition/prefill/decode timing as original; teacher-greedy reference separately regenerated and timed in FP32',
        purpose='Numerically consistent E6 recheck; keep separate from original BF16 offline and runtime results'))
    for index,item in enumerate(e.data['test']):
        teacher_key=(item['source_index'],0,'teacher_greedy')
        if teacher_key not in done:
            reference=teacher_reference(e,item)
            reference['teacher_dtype']='float32'
            append(path,reference)
            done.add(teacher_key)
        else:
            reference=next(r for r in rows(path) if (r['source_index'],r['budget'],r['method'])==teacher_key)
        order=list(plan)
        random.Random(a.seed+item['source_index']).shuffle(order)
        for method,b in order:
            if (item['source_index'],b,method) in done:
                continue
            result=speculative(e,item,method,b,chosen,pairs)
            result['output_agrees']=result['generated']==reference['generated']
            result['teacher_dtype']='float32'
            result['teacher_decode_tps']=reference['decode_tps']
            result['teacher_total_tps']=reference['total_tps']
            append(path,result)
            done.add((item['source_index'],b,method))
        print(f'FP32 runtime {index+1}/{len(e.data["test"])}',flush=True)


if __name__=='__main__':
    main()
