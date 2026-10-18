"""Goal-directed diagnostics, mapper fitting and matched module/schedule trials.

Uses training documents plus three previously unused development documents.
The paused long-context test matrix is never read or resumed.
"""
import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random

import torch

from .attention import DraftAttention
from .baselines import BaselineEngine
from .cache import WorkingCache
from .compare_baselines import append, write
from .config import SystemConfig
from .engine import RescueEngine
from .injection import ContentInjector
from .loading import ROOT, load_models, load_modules
from .long_data import context
from .objectives import accounting, distribution_quality, margin_guard_loss
from .packing import pack_projections
from .policy import RiskModel
from .rollout_training import replay
from .selection import ContextSelector, LazyReadState


def mapper_fit(student, mapper, training, output):
    student.eval().requires_grad_(False)
    for parameter in student.parameters():
        parameter.grad = None
    bundles=[json.loads(l) for l in (training/'bundles.jsonl').read_text().splitlines()]
    records=[]
    for b in bundles:
        path=Path(b['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==b['sha256']
        payload=torch.load(path,weights_only=False)
        for r in payload['records']:
            r=dict(r,source_index=payload['source_index'])
            records.append(r)
    held=max(r['source_index'] for r in records)
    fit=[r for r in records if r['source_index']!=held]
    monitor=[r for r in records if r['source_index']==held]
    native={}
    with torch.no_grad():
        for r in records:
            native[id(r)]=replay(student,mapper,r,strength=0.)[0].detach()
    network=copy.deepcopy(mapper).train().requires_grad_(True)
    opt=torch.optim.AdamW(network.parameters(),lr=2e-4,weight_decay=.01)
    def evaluate(model,group):
        result=[]
        with torch.no_grad():
            for r in group:
                logits,_=replay(student,model,r)
                q=distribution_quality(logits,r['target'].to(logits.device))
                z=distribution_quality(native[id(r)],r['target'].to(logits.device))
                result.append(dict(**q,native_correct=z['top1_matches_target'],
                    corrected=(not z['top1_matches_target'] and q['top1_matches_target']),
                    harmed=(z['top1_matches_target'] and not q['top1_matches_target'])))
        return dict(queries=len(result),matches=sum(r['top1_matches_target'] for r in result),
            native_matches=sum(r['native_correct'] for r in result),
            corrected=sum(r['corrected'] for r in result),harmed=sum(r['harmed'] for r in result),
            mean_kl=sum(r['target_kl'] for r in result)/len(result))
    original=evaluate(mapper,monitor)
    best=original;best_state=copy.deepcopy(network.state_dict());best_epoch=0
    history=[];rng=random.Random(58)
    for epoch in range(6):
        order=list(fit);rng.shuffle(order);losses=[]
        for r in order:
            opt.zero_grad(set_to_none=True)
            logits,penalty=replay(student,network,r)
            loss=margin_guard_loss(logits,r['target'].to(logits.device),penalty,native[id(r)])
            loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),1.);opt.step()
            losses.append(float(loss.detach()))
        score=evaluate(network,monitor)
        history.append(dict(epoch=epoch+1,fit_loss=sum(losses)/len(losses),monitor=score))
        # Explicitly prefer actual greedy agreement; KL only breaks ties.
        if (score['matches'],-score['mean_kl'])>(best['matches'],-best['mean_kl']):
            best,best_state,best_epoch=score,copy.deepcopy(network.state_dict()),epoch+1
        print('goal mapper',history[-1],flush=True)
    network.load_state_dict(best_state);network.eval().requires_grad_(False)
    torch.save(dict(config=network.config,state_dict=network.state_dict()),output/'mapper_goal.pt')
    write(output/'mapper_goal.json',dict(scope='Same frozen last-layer old-rollout queries; not historical backprop or actual accepted-prefix improvement',
        base_parameters_optimized=False,base_gradients_disabled=True,
        fit_queries=len(fit),monitor_queries=len(monitor),monitor_document=held,
        objective='0.25 KL + softplus teacher top1 margin + 0.5 native-success margin preservation + 0.01 residual',
        selection='Training monitor greedy matches, then KL; original weights are eligible',
        selected_epoch=best_epoch,original=original,selected=best,history=history))
    return network


@torch.no_grad()
def probe(teacher,student,tokenizer,old,new,items,configs,output,queries):
    for item in items:
        for length in (8192,16384):
            base=replace(configs['hier512'],selector='full',inject_tokens=0)
            with RescueEngine(teacher,student,base,old,vocab_size=len(tokenizer)) as engine:
                ids=context(item,length)
                engine._buffers(length+queries+16)
                engine.observation.mode='last'
                engine._prefill(teacher,ids[:-1],engine.native_t)
                tr=engine.observation.rows
                def native_step(tokens):
                    draft=DraftAttention(student,None,None,optimized=True,deferred=True).install()
                    try:
                        engine._prefill(student,tokens,engine.native_s)
                        return draft.shadow
                    finally: draft.restore()
                sr=native_step(ids[:-1])
                routes={}
                for name,c in configs.items():
                    cfg=replace(c,refresh_every=1)
                    network=new if name=='hier512_goal' else old
                    working=WorkingCache(engine.native_s,engine.native_s.capacity if cfg.selector=='full' else cfg.budget,
                        cfg.max_draft,direct_gather=True,mode='packed')
                    injector=ContentInjector(network,engine.native_s.capacity,teacher.device,cfg.inject_tokens,
                        cfg.injection_strength,True,optimized=True,memo_affine=True,layers=cfg.injection_layers)
                    routes[name]=(cfg,ContextSelector(cfg),working,injector)
                current=ids[-1]
                for qi in range(queries):
                    n=engine.native_s.get_seq_length()
                    t=LazyReadState(tr,engine.native_t[-1][1][0])
                    s=LazyReadState(sr,engine.native_s[-1][1][0])
                    teacher_out=engine._forward(teacher,[current],engine.native_t)
                    target=teacher.lm_head(teacher_out.last_hidden_state)[0,-1,:len(tokenizer)].float()
                    next_token=int(target.argmax());next_tr=engine.observation.rows
                    for name,(cfg,selector,working,injector) in routes.items():
                        selected=selector.select(t,s,qi)
                        working.bind(engine.native_s,selected)
                        injector.bind(engine.native_t,engine.native_s,selected,t.a.mean(0),cfg.recent,working=working)
                        draft=DraftAttention(student,working,injector,optimized=True,deferred=True,observe=False).install()
                        draft.prefill=False
                        try: logits=draft.step(current,engine.native_s)[0,:len(tokenizer)]
                        finally:
                            draft.restore();engine.native_s.crop(n);working.invalidate();injector.clear()
                        record=dict(source_index=item['source_index'],split=item['split'],length=length,
                            query_index=qi,prefix=n,mode=name,reference_tokens=len(selected),
                            teacher_attention_mass=float(t.a[:,selected].sum(-1).mean()),
                            student_attention_mass=float(s.a[:,selected].sum(-1).mean()),
                            teacher_read_error=float(t.error(selected)),student_read_error=float(s.error(selected)),
                            **distribution_quality(logits,target))
                        append(output/'quality.jsonl',record)
                    sr=native_step([current]);tr=next_tr;current=next_token
                    if current==tokenizer.eos_token_id: break
                print('paired quality complete',item['split'],item['source_index'],length,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--training',type=Path,default=Path('exp/long_system/out_joint_train_v2_20261017'))
    p.add_argument('--utility',type=Path,default=Path('exp/module_objectives/out_audit_v2_20261018'))
    p.add_argument('--queries',type=int,default=24)
    p.add_argument('--max-new-tokens',type=int,default=96)
    args=p.parse_args()
    if args.output.exists(): p.error('Use a fresh output')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4);torch.manual_seed(58)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    teacher,student,tokenizer=load_models(efficient=True)
    old,risk,_=load_modules(teacher.device,mapper=args.training/'mapper_joint.pt',risk=args.training/'risk_injected.pt')
    cost=json.loads(Path('exp/long_system/out_comparison_v2_20261017/cost.json').read_text())
    utility=RiskModel.load(args.utility/'risk_utility.pt',teacher.device)
    train=[json.loads(l) for l in (ROOT/'exp/long_system/data/train.jsonl').read_text().splitlines()][6:8]
    fresh=[json.loads(l) for l in (ROOT/'exp/long_system/data/development.jsonl').read_text().splitlines()][3:]
    base=SystemConfig(selector='hierarchical',coverage_rows=64,length_policy='fixed',fixed_k=4,
        injection_layers=(27,),inject_tokens=0,prefill_precision='bfloat16',max_new_tokens=args.max_new_tokens)
    configs=dict(full_native=replace(base,selector='full'),teacher512=replace(base,selector='teacher'),
        joint512=replace(base,selector='joint',student_weight=.5),hier512=base,
        hier1024=replace(base,budget=1024,coverage_candidates=1536),
        hier512_old=replace(base,inject_tokens=64),hier512_goal=replace(base,inject_tokens=64))
    study=dict(configs, risk_serial=replace(base,inject_tokens=64,length_policy='risk'),
        risk_utility=replace(base,inject_tokens=64,length_policy='risk'),
        risk_async_late=replace(base,inject_tokens=64,length_policy='risk',schedule='async'),
        risk_async_early=replace(base,inject_tokens=64,length_policy='risk',schedule='async',early_length_publish=True))
    sources={}
    for path in [*sorted((ROOT/'rescue').glob('*.py')),*sorted((ROOT/'tests').glob('*.py'))]:
        name=str(path.relative_to(ROOT));sources[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        dest=args.output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    write(args.output/'protocol.json',dict(seed=58,source_sha256=sources,
        configs={n:asdict(c) for n,c in study.items()},fresh_development_indices=[r['source_index'] for r in fresh],
        train_diagnostic_indices=[r['source_index'] for r in train],lengths=[8192,16384],repeats=2,
        max_new_tokens=args.max_new_tokens,queries=args.queries,
        data_sha256={s:hashlib.sha256((ROOT/f'exp/long_system/data/{s}.jsonl').read_bytes()).hexdigest() for s in ('train','development')},
        input_checkpoint_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            args.training/'mapper_joint.pt',args.training/'risk_injected.pt',args.utility/'risk_utility.pt')},
        fairness='Identical prefixes/precision/native history for query diagnostics; matched budget/refresh/mapper for one-module SD comparisons',
        scope='Fresh development pilot, not held-out test or SOTA comparison; paused test remains untouched',
        quality_scope='Teacher-forced common native history, previous-query selection signals, refresh each query; does not replace rolling SD',
        timing='Unprofiled fresh-request system times; separately profiled training requests do not supply throughput claims'))
    new=mapper_fit(student,old,args.training,args.output)
    # Replay fitting needs ordinary modules; install immutable packing afterward.
    pack_projections(teacher);pack_projections(student)
    probe(teacher,student,tokenizer,old,new,train+fresh,configs,args.output,args.queries)
    def run(item,length,mode,limit=None,profile=False):
        cfg=replace(study[mode],profile=profile,max_new_tokens=limit or args.max_new_tokens)
        network=new if mode=='hier512_goal' else old
        model=utility if mode=='risk_utility' else risk
        if mode=='full_native':
            with BaselineEngine(teacher,student,cfg,vocab_size=len(tokenizer)) as engine:
                result=engine.generate(context(item,length),tokenizer.eos_token_id)
        else:
            with RescueEngine(teacher,student,cfg,network,model,cost,vocab_size=len(tokenizer)) as engine:
                result=engine.generate(context(item,length),tokenizer.eos_token_id)
        result.update(source_index=item['source_index'],split=item['split'],length=length,mode=mode)
        return result
    refs={}
    for length in (8192,16384):
        for mode in study: run(train[0],length,mode,limit=8)
        for mode in ('hier512_old','risk_serial','risk_async_late','risk_async_early'):
            result=run(train[0],length,mode,limit=32,profile=True)
            append(args.output/'profiles.jsonl',result)
        print('warmup and profiles complete',length,flush=True)
    for item in fresh:
        for length in (8192,16384):
            with BaselineEngine(teacher,student,base,vocab_size=len(tokenizer)) as engine:
                reference=engine.reference(context(item,length),tokenizer.eos_token_id)['generated']
            refs[f"{item['source_index']}:{length}"]=reference;write(args.output/'references.json',refs)
            for repeat in range(2):
                order=list(study);random.Random(58+item['source_index']*1000+length+repeat).shuffle(order)
                for mode in order:
                    row=run(item,length,mode)
                    row.update(repeat=repeat,output_agrees=row['generated']==reference,accounting=accounting(row))
                    append(args.output/'runtime.jsonl',row)
                    if not row['output_agrees']: raise AssertionError('Target output mismatch')
                print('matched SD complete',item['source_index'],length,repeat,flush=True)
    write(args.output/'COMPLETE.json',dict(passed=True,requests=len(study)*len(fresh)*2*2))


if __name__=='__main__': main()
