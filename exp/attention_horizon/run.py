"""Execute the three mechanism groups without test-set decisions."""
import json
import time

import numpy as np
import torch

from experiment import Experiment, parser, read_rows, write, core_forward, tensor_ids, legacy_cache
from selection import ReadState, select, mandatory
from runtime import run_methods, speculative, cost_fit, MENU
from risk import fit


def score(rows,key='decode_tps'):
    if key=='decode_tps':return sum(r['tokens'] for r in rows)/sum(r['decode_seconds'] for r in rows)
    if key=='acceptance_rate':return sum(r['accepted'] for r in rows)/sum(r['proposed'] for r in rows)
    return np.mean([r[key] for r in rows])


@torch.no_grad()
def verify(e):
    torch.manual_seed(42)
    a=torch.rand(6,113,device=e.a.device).softmax(-1)
    v=torch.randn(2,113,9,device=e.a.device)
    t=ReadState(a,v)
    manual=torch.stack([a[h]@v[h//3] for h in range(6)])
    assert torch.allclose(t.target,manual,atol=1e-6)
    allpos=torch.arange(113,device=e.a.device)
    assert float(t.error(allpos))<1e-10
    selected=select('teacher',t,t,45,8)
    improved=select('read_teacher',t,t,45,8)
    assert float(t.error(improved))<=float(t.error(selected))+1e-7
    errors=t.swap_errors(selected,selected[1:3],allpos[~torch.isin(allpos,selected)][:4])
    changed=torch.cat((selected[selected!=selected[1]],allpos[~torch.isin(allpos,selected)][:1]))
    assert torch.allclose(errors[0,0],t.error(changed),atol=1e-6)
    item=e.data['train'][0]
    tc,sc,tr,sr,conf,_=e.prefill(item)
    current=item['ids'][-1]
    e.sc.enabled=False
    native=e.forward(e.student,current,sc)
    native_logits=e.logit(e.student,native)
    oldvalue=sc[-1][1].clone()
    e.sc.configure(None)
    observed=e.forward(e.student,current,sc)
    assert torch.equal(native_logits,e.logit(e.student,observed))
    assert torch.equal(oldvalue,sc[-1][1])
    assert torch.allclose(e.shadow,e.sc.rows[27],atol=.002,rtol=.02)
    reference=e.reference(item)
    results=[]
    for method,config in [('full',dict(rule='native_full',k=4)),('sparse',dict(rule='read_joint025',k=12))]:
        result=speculative(e,item,method,config)
        assert result['generated']==reference['generated']
        results.append(dict(method=method,tokens=result['tokens'],output_agrees=True))
    write(e.out/'verification.json',dict(gqa_matches_manual=True,swap_objective_matches_recompute=True,
        read_objective_nonincreasing=True,native_probe_logits_identical=True,backing_unchanged=True,
        greedy_matches=results))
    print('verification passed',flush=True)


def collection(e):
    rules=['teacher','read_teacher','read_joint025','read_joint1']
    methods={f'collect_{rule}':dict(rule=rule,k=12) for rule in rules}
    for split in ('train','validation'):
        run_methods(e,split,methods,f'collection_{split}.jsonl',collect=True,profile=split=='train')
    cost_fit(e)


def validate(e):
    e.cost=json.loads((e.out/'cost_model.json').read_text())
    a={rule:dict(rule=rule,k=4) for rule in
       ('teacher','student','union','read_teacher','read_joint025','read_joint1','native_full')}
    run_methods(e,'validation',a,'selection_validation.jsonl')
    rows=read_rows(e.out/'selection_validation.jsonl')
    joint=max(('read_teacher','read_joint025','read_joint1'),key=lambda k:
              score([r for r in rows if r['method']==k],'acceptance_rate'))
    # Selection uses validation acceptance at a fixed horizon. A proxy cannot
    # select a workset that worsens the actual drafting objective silently.
    static={f'{rule}_k{k}':dict(rule=rule,k=k) for rule in dict.fromkeys(['teacher',joint]) for k in MENU}
    run_methods(e,'validation',static,'static_validation.jsonl')
    static_rows=read_rows(e.out/'static_validation.jsonl')
    best_static={rule:max(MENU,key=lambda k:score([r for r in static_rows if r['method']==f'{rule}_k{k}']))
                 for rule in dict.fromkeys(['teacher',joint])}
    adaptive={}
    for rule in dict.fromkeys(['teacher',joint]):
        for group in ('structural','confidence','attention','attention_read','combined'):
            adaptive[f'{rule}_{group}']=dict(rule=rule,risk=group)
        adaptive[f'{rule}_attention_online']=dict(rule=rule,risk='attention',online=True)
    run_methods(e,'validation',adaptive,'adaptive_validation.jsonl')
    adaptive_rows=read_rows(e.out/'adaptive_validation.jsonl')
    # Pick attention-only or attention+read (not confidence) for the paper's
    # central A/B/C integration, using measured validation throughput.
    choices=['attention','attention_read','attention_online']
    chosen=max(choices,key=lambda c:score([r for r in adaptive_rows if r['method']==f'{joint}_{c}']))
    config=adaptive[f'{joint}_{chosen}']
    joint_config={**config,'joint':True}
    run_methods(e,'validation',dict(joint_attention=joint_config),'joint_validation.jsonl')
    write(e.out/'selected.json',dict(joint_rule=joint,best_static=best_static,
        attention_policy=chosen,sequential_config=config,joint_config=joint_config,
        selection_criterion='Fixed-k=4 validation acceptance; static/adaptive validation pooled decode TPS',
        adaptive_best_validation={rule:max(
            [k for k in adaptive if adaptive[k]['rule']==rule],
            key=lambda k:score([r for r in adaptive_rows if r['method']==k])) for rule in best_static}))


def testing(e):
    selected=json.loads((e.out/'selected.json').read_text())
    e.cost=json.loads((e.out/'cost_model.json').read_text())
    joint=selected['joint_rule']
    methods={f'A_{rule}':dict(rule=rule,k=4) for rule in
        ('teacher','student','union','read_teacher','read_joint025','read_joint1','native_full')}
    methods['B_best_static']=dict(rule='teacher',k=selected['best_static']['teacher'])
    for group in ('structural','confidence','attention','attention_read','combined'):
        methods[f'B_{group}']=dict(rule='teacher',risk=group)
    methods['B_attention_online']=dict(rule='teacher',risk='attention',online=True)
    methods['C_jointworkset_static']=dict(rule=joint,k=selected['best_static'][joint])
    seq=selected['sequential_config']; both=selected['joint_config']
    methods['C_sequential']=seq
    methods['C_joint']=both
    methods['C_sequential_V']={**seq,'inject':True}
    methods['C_joint_V']={**both,'inject':True}
    write(e.out/'test_methods.json',methods)
    run_methods(e,'test',methods)
    # Labels for held-out calibration are obtained separately; these extra
    # verifications are diagnostic data collection and never timed as deployment.
    collect={f'collect_{rule}':dict(rule=rule,k=12) for rule in
             dict.fromkeys(['teacher','read_teacher',joint])}
    run_methods(e,'test',collect,'collection_test.jsonl',collect=True)


def deployment(e):
    # Additional practical controls need neither student shadow attention nor
    # per-layer attention capture. Their outputs must reproduce the controls
    # with probes exactly; compare real cost against this cheaper baseline too.
    e.cost=json.loads((e.out/'cost_model.json').read_text())
    methods={f'lean_k{k}':dict(rule='teacher',k=k,probe=False) for k in MENU}
    methods.update({f'history_{r}':dict(rule='teacher',history=r,probe=False) for r in (0.,.1,.3,.6)})
    speculative(e,e.data['train'][0],'warmup',dict(rule='teacher',k=4,probe=False))
    run_methods(e,'validation',methods,'lean_validation.jsonl')
    rows=read_rows(e.out/'lean_validation.jsonl')
    best=max(MENU,key=lambda k:score([r for r in rows if r['method']==f'lean_k{k}']))
    rate=max((0.,.1,.3,.6),key=lambda r:score([v for v in rows if v['method']==f'history_{r}']))
    methods=dict(O_teacher_k4=dict(rule='teacher',k=4,probe=False),
        O_teacher_static=dict(rule='teacher',k=best,probe=False),
        O_native_full=dict(rule='native_full',k=4,probe=False),
        O_history=dict(rule='teacher',history=rate,probe=False),
        O_confidence=dict(rule='teacher',risk='confidence',probe=False))
    saved=json.loads((e.out/'test_methods.json').read_text())
    saved.update(methods);write(e.out/'test_methods.json',saved)
    write(e.out/'lean_selected.json',dict(best_static=best,history_rate=rate,criterion='validation pooled decode TPS'))
    run_methods(e,'test',methods)


def main():
    args=parser().parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.phase=='fit':
        from pathlib import Path
        fit(Path(args.output));return
    e=Experiment(args)
    if args.phase in ('verify','all'):verify(e)
    if args.phase in ('offline','all'):
        for split in ('train','validation','test'):e.offline(split)
    if args.phase in ('collect','all'):collection(e)
    if args.phase=='all':fit(e.out)
    if args.phase in ('validate','all'):validate(e)
    if args.phase in ('test','all'):testing(e)
    if args.phase in ('deployment','all'):deployment(e)
    print('phase complete',args.phase,flush=True)


if __name__=='__main__':main()
