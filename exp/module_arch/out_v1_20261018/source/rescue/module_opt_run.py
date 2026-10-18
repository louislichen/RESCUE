"""Optimize module goals on train streams, then freeze a new confirmation pilot."""
import argparse
import copy
from dataclasses import asdict,replace
import hashlib
import json
import math
from pathlib import Path
import random

import torch

from .baselines import BaselineEngine
from .calibration import calibrate
from .compare_baselines import append,write
from .config import SystemConfig
from .engine import RescueEngine
from .loading import ROOT,load_models,load_modules
from .long_data import context
from .module_opt_training import train_mapper,train_risk
from .module_study import probe
from .objectives import accounting
from .packing import pack_projections
from .policy import RiskModel,FeedbackRisk
from .rollout_training import RolloutCollector


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('exp/module_opt/data_v1_20261018'))
    p.add_argument('--cache',type=Path,default=Path('/mnt/8T/cl/rescue_experiment_cache/module_opt_v1'))
    p.add_argument('--max-new-tokens',type=int,default=128)
    args=p.parse_args()
    if args.output.exists() or args.cache.exists(): p.error('Use fresh output and tensor cache')
    args.output.mkdir(parents=True);args.cache.mkdir(parents=True)
    sources={}
    for path in [*sorted((ROOT/'rescue').glob('*.py')),*sorted((ROOT/'tests').glob('*.py')),
                 ROOT/'exp/selective_kv/bridge.py',ROOT/'exp/attention_horizon/selection.py']:
        name=str(path.relative_to(ROOT));sources[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        dest=args.output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    manifest=json.loads((args.data/'manifest.json').read_text())
    splits={s:[json.loads(l) for l in (args.data/f'{s}.jsonl').read_text().splitlines()] for s in manifest['data_sha256']}
    write(args.output/'protocol.json',dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        source_sha256=sources,data_manifest=manifest,seed=81,
        lengths=[8192,16384],confirmation_repeats=2,
        selector_selection='At most baseline total fixed4 verification rounds on tune documents, then max same-cell geometric TPS ratio',
        mapper_selection='Monitor greedy matches, then KL; downstream replacement needs strictly more matches',
        length_selection='Best fixed from2/4/8; positive attention policy: require no more waste and <=5% extra rounds vs best fixed when possible, then geometric TPS',
        scope='New development confirmation pilot, train-only parameter choice; original paused test is not resumed',
        precision='FP32 Target decode/weights, BF16 prefix and Draft, TF32 off; common packed/efficient execution'))
    torch.set_num_threads(4);torch.manual_seed(81)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    teacher,student,tokenizer=load_models(efficient=True)
    old,oldrisk,cost=load_modules(teacher.device,mapper=ROOT/'exp/long_system/out_joint_train_v2_20261017/mapper_joint.pt',
        risk=ROOT/'exp/long_system/out_joint_train_v2_20261017/risk_injected.pt',cost=ROOT/'exp/long_system/out_comparison_v2_20261017/cost.json')
    pack_projections(teacher);pack_projections(student)
    references={};vocab=len(tokenizer);eos=tokenizer.eos_token_id
    original=SystemConfig(selector='hierarchical',coverage_rows=64,length_policy='fixed',fixed_k=4,
        injection_layers=(27,),inject_tokens=0,prefill_precision='bfloat16',max_new_tokens=args.max_new_tokens)
    def reference(item,n):
        key=f"{item['split']}:{item['source_index']}:{n}"
        if key not in references:
            with BaselineEngine(teacher,student,original,vocab_size=vocab) as engine:
                references[key]=engine.reference(context(item,n),eos)['generated']
            write(args.output/'references.json',references)
        return references[key]
    def run(item,n,cfg,mapper=old,model=None,observer=None,limit=None):
        cfg=replace(cfg,max_new_tokens=limit or args.max_new_tokens)
        with RescueEngine(teacher,student,cfg,mapper,model or oldrisk,cost,vocab_size=vocab,
            draft_observer=observer,round_observer=observer.round if observer else None) as engine:
            if observer: observer.engine=engine
            result=engine.generate(context(item,n),eos)
        if observer: observer.engine=None
        result.update(source_index=item['source_index'],split=item['split'],length=n,
            output_agrees=result['generated']==reference(item,n)[:cfg.max_new_tokens],accounting=accounting(result))
        if not result['output_agrees']:
            append(args.output/'mismatch.jsonl',result);raise AssertionError('Output differs from matched Target')
        return result
    choices=dict(original=original,
        anchor50=replace(original,coverage_anchor_fraction=.5,coverage_attention_weight=.25),
        anchor75=replace(original,coverage_anchor_fraction=.75,coverage_attention_weight=.25),
        small_anchor75=replace(original,budget=256,coverage_anchor_fraction=.75,coverage_attention_weight=.25))
    tune=splits['tune'];warm=splits['warmup'][0]
    for n in (8192,16384):
        for cfg in choices.values(): run(warm,n,cfg,limit=8)
    selector_rows=[]
    for item in tune:
        for n in (8192,16384):
            names=list(choices);random.Random(81+item['source_index']+n).shuffle(names)
            for name in names:
                row=run(item,n,choices[name]);row['mode']=name;selector_rows.append(row);append(args.output/'selector_tune.jsonl',row)
            print('selector tune complete',item['source_index'],n,flush=True)
    def geometric(rows,name,control):
        values=[]
        for r in rows:
            if r['mode']!=name: continue
            b=next(x for x in rows if x['mode']==control and x['source_index']==r['source_index'] and x['length']==r['length'])
            values.append(math.log(r['total_tps']/b['total_tps']))
        return math.exp(sum(values)/len(values))
    baseline_rounds=sum(r['blocks'] for r in selector_rows if r['mode']=='original')
    eligible=[name for name in choices if sum(r['blocks'] for r in selector_rows if r['mode']==name)<=baseline_rounds]
    chosen=max(eligible,key=lambda name:geometric(selector_rows,name,'original'))
    selected=choices[chosen]
    write(args.output/'selector_choice.json',dict(chosen=chosen,config=asdict(selected),baseline_rounds=baseline_rounds,
        eligible=eligible,scores={name:geometric(selector_rows,name,'original') for name in choices}))
    print('selector locked',chosen,flush=True)
    def collect(label,mapper,records):
        payloads=[];cfg=replace(selected,inject_tokens=64,fixed_k=12)
        for item in splits['fit']:
            for n in (8192,16384):
                collector=RolloutCollector(student,max_records=records);collector.eos_token_id=eos
                result=run(item,n,cfg,mapper,observer=collector)
                result.update(mode=label,records=len(collector.records),frames=len(collector.frames))
                append(args.output/'collection.jsonl',result)
                payload=dict(source_index=item['source_index'],length=n,document_sha256=item['document_sha256'],
                    records=collector.records,frames=collector.frames)
                path=args.cache/f'{label}_{item["source_index"]}_{n}.pt';torch.save(payload,path)
                append(args.output/'bundles.jsonl',dict(phase=label,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                payloads.append(payload)
                print('collected',label,item['source_index'],n,len(collector.records),len(collector.frames),flush=True)
        return payloads
    fit=collect('mapper',old,48)
    query,promoted=train_mapper(student,old,fit,args.output)
    del fit
    deployed=query if promoted else old
    print('mapper locked',promoted,flush=True)
    payloads=collect('risk',deployed,0)
    light,prior=train_risk(payloads,args.output);del payloads
    light=light.to(teacher.device)
    # Fit costs separately on training inputs, not confirmation timing.
    calibration_config=replace(selected,inject_tokens=64,max_new_tokens=32)
    calibration_docs=[dict(ids=context(warm,n),source_index=f'{warm["source_index"]}:{n}') for n in (8192,16384)]
    cost,calibration=calibrate(teacher,student,deployed,light,calibration_config,calibration_docs,eos,vocab)
    write(args.output/'cost.json',cost);write(args.output/'calibration.json',calibration)
    rates={}
    for n in (8192,16384):
        result=run(warm,n,replace(selected,inject_tokens=64),deployed,limit=64)
        rates[n]=result['decode_tps']
        append(args.output/'rate_calibration.jsonl',result)
    policies=dict(old=oldrisk,light=light,
        feedback_ratio=FeedbackRisk(light,prior,strength=8,rule='ratio'),
        feedback_surplus=FeedbackRisk(light,prior,strength=8,rule='surplus',rates=rates),
        history_surplus=FeedbackRisk(light,prior,strength=8,use_attention=False,rule='surplus',rates=rates))
    length_names=['fixed2','fixed4','fixed8',*policies]
    for n in (8192,16384):
        for name in length_names:
            cfg=replace(selected,inject_tokens=64,length_policy='fixed' if name.startswith('fixed') else 'risk',
                fixed_k=int(name[5:]) if name.startswith('fixed') else 4)
            run(warm,n,cfg,deployed,policies.get(name),limit=8)
    length_rows=[]
    for item in tune:
        for n in (8192,16384):
            order=list(length_names);random.Random(181+item['source_index']+n).shuffle(order)
            for name in order:
                cfg=replace(selected,inject_tokens=64,length_policy='fixed' if name.startswith('fixed') else 'risk',
                    fixed_k=int(name[5:]) if name.startswith('fixed') else 4)
                row=run(item,n,cfg,deployed,policies.get(name));row['mode']=name
                length_rows.append(row);append(args.output/'length_tune.jsonl',row)
            print('length tune complete',item['source_index'],n,flush=True)
    fixed=max(('fixed2','fixed4','fixed8'),key=lambda name:geometric(length_rows,name,'fixed4'))
    fixed_rows=[r for r in length_rows if r['mode']==fixed]
    fixed_waste=sum(r['proposed']-r['accepted'] for r in fixed_rows);fixed_rounds=sum(r['blocks'] for r in fixed_rows)
    attention=('light','feedback_ratio','feedback_surplus')
    eligible=[name for name in attention if sum(r['proposed']-r['accepted'] for r in length_rows if r['mode']==name)<=fixed_waste
        and sum(r['blocks'] for r in length_rows if r['mode']==name)<=fixed_rounds*1.05]
    positive=max(eligible or attention,key=lambda name:geometric(length_rows,name,fixed))
    write(args.output/'length_choice.json',dict(best_fixed=fixed,chosen_attention=positive,eligible=eligible,
        rates=rates,scores={name:geometric(length_rows,name,fixed) for name in length_names},scope='Tune documents only; positive policy remains separate if fixed wins'))
    modes=dict(m1_old=(original,old,None),m1_new=(selected,old,None),
        m2_old=(replace(selected,inject_tokens=64),old,None),m2_query=(replace(selected,inject_tokens=64),query,None),
        fixed_best=(replace(selected,inject_tokens=64,fixed_k=int(fixed[5:])),deployed,None),
        m3_old=(replace(selected,inject_tokens=64,length_policy='risk'),deployed,oldrisk),
        m3_selected=(replace(selected,inject_tokens=64,length_policy='risk'),deployed,policies[positive]),
        m3_history=(replace(selected,inject_tokens=64,length_policy='risk'),deployed,policies['history_surplus']))
    locked=dict(modes={name:asdict(cfg) for name,(cfg,_,_) in modes.items()},mapper_promoted=promoted,
        best_fixed=fixed,positive_length=positive,checkpoint_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            args.output/'mapper_query.pt',args.output/'risk_light.pt')},confirmation_ids=[r['source_index'] for r in splits['confirmation']])
    write(args.output/'locked.json',locked)
    print('all choices locked',chosen,promoted,fixed,positive,flush=True)
    # Counterfactual quality is deliberately separate from rolling timing.
    quality_configs=dict(full_native=replace(original,selector='full'),hier512=original,
        m1_new=selected,m1_new_old=replace(selected,inject_tokens=64),hier512_goal=replace(selected,inject_tokens=64))
    probe(teacher,student,tokenizer,old,query,splits['confirmation'],quality_configs,args.output,24)
    for n in (8192,16384):
        for cfg,mapper,model in modes.values(): run(warm,n,cfg,mapper,model,limit=8)
    for item in splits['confirmation']:
        for n in (8192,16384):
            for repeat in range(2):
                names=list(modes);random.Random(8181+item['source_index']+n+repeat).shuffle(names)
                for name in names:
                    cfg,mapper,model=modes[name];row=run(item,n,cfg,mapper,model)
                    row.update(mode=name,repeat=repeat);append(args.output/'runtime.jsonl',row)
                print('confirmation complete',item['source_index'],n,repeat,flush=True)
    write(args.output/'COMPLETE.json',dict(passed=True,confirmation_requests=len(modes)*3*2*2))


if __name__=='__main__': main()
