"""Train internal M2/M3 changes, lock choices, run fresh matched confirmation."""
import argparse
import copy
from dataclasses import asdict,replace
import hashlib
import json
import math
from pathlib import Path
import random

import torch

from .architecture_training import ArchitectureCollector,fit_mapper,fit_budget
from .baselines import BaselineEngine
from .calibration import calibrate
from .compare_baselines import write,append
from .config import SystemConfig
from .engine import RescueEngine
from .loading import ROOT,load_models,load_modules
from .long_data import context
from .module_study import probe
from .objectives import accounting
from .packing import pack_projections
from .query_mapper import mapper_from_payload
from .residual_mapper import NativeResidualMapper


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('exp/module_arch/data_v1_20261018'))
    p.add_argument('--cache',type=Path,default=Path('/mnt/8T/cl/rescue_experiment_cache/module_arch_v1'))
    args=p.parse_args()
    if args.output.exists() or args.cache.exists():p.error('Use fresh output and cache directories')
    args.output.mkdir(parents=True);args.cache.mkdir(parents=True)
    sources={}
    for path in [*sorted((ROOT/'rescue').glob('*.py')),*sorted((ROOT/'tests').glob('*.py')),
                 ROOT/'exp/selective_kv/bridge.py',ROOT/'exp/attention_horizon/selection.py']:
        name=str(path.relative_to(ROOT));sources[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        dest=args.output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    manifest=json.loads((args.data/'manifest.json').read_text())
    splits={s:[json.loads(l) for l in (args.data/f'{s}.jsonl').read_text().splitlines()] for s in manifest['data_sha256']}
    write(args.output/'protocol.json',dict(args={k:str(v) for k,v in vars(args).items()},source_sha256=sources,
        data_manifest=manifest,plan_sha256=hashlib.sha256(Path('exp/module_arch/README.md').read_bytes()).hexdigest(),
        seed=612,lengths=[8192,16384],max_new_tokens=128,repeats=2,
        mapper_epochs=[4,2],budget_epochs=250,exploration=.25,
        scope='Internal module architecture packages, fresh three-document development confirmation; original test paused'))
    torch.set_num_threads(4);torch.manual_seed(612)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    teacher,student,tokenizer=load_models(efficient=True)
    _,oldrisk,initial_cost=load_modules(teacher.device)
    path=ROOT/'exp/module_opt/out_v1_20261018/mapper_query.pt'
    query=mapper_from_payload(torch.load(path,map_location='cpu',weights_only=False),teacher.device).eval().requires_grad_(False)
    write(args.output/'input_checkpoint_sha256.json',{str(path):hashlib.sha256(path.read_bytes()).hexdigest()})
    pack_projections(teacher);pack_projections(student)
    cfg=SystemConfig(selector='hierarchical',coverage_rows=64,length_policy='fixed',fixed_k=4,
        injection_layers=(27,),inject_tokens=64,prefill_precision='bfloat16',max_new_tokens=128)
    noinj=replace(cfg,inject_tokens=0)
    eos,vocab=tokenizer.eos_token_id,len(tokenizer);references={}
    # References from prior TRAIN streams may be reused only when precision,
    # context, and output limit match. Fresh confirmation references are new.
    oldrefs=json.loads((ROOT/'exp/module_opt/out_v1_20261018/references.json').read_text())
    def reference(item,n):
        key=f"{item['split']}:{item['source_index']}:{n}"
        if key not in references:
            if item['split']!='confirmation' and key in oldrefs:
                references[key]=oldrefs[key]
            else:
                with BaselineEngine(teacher,student,cfg,vocab_size=vocab) as engine:
                    references[key]=engine.reference(context(item,n),eos)['generated']
            write(args.output/'references.json',references)
        return references[key]
    def run(item,n,mapper=query,config=cfg,policy=None,collector=None,limit=128,cost=None):
        with RescueEngine(teacher,student,replace(config,max_new_tokens=limit),mapper,policy or oldrisk,
            cost or initial_cost,vocab_size=vocab,draft_observer=collector,
            round_observer=collector.round if collector else None) as engine:
            if collector:collector.engine=engine
            result=engine.generate(context(item,n),eos)
        if collector:collector.engine=None
        result.update(source_index=item['source_index'],split=item['split'],length=n,
            output_agrees=result['generated']==reference(item,n)[:limit],accounting=accounting(result))
        if not result['output_agrees']:
            append(args.output/'mismatch.jsonl',result);raise AssertionError('Output differs from matched Target')
        return result
    def collect(label,mapper,records=48,policy=None):
        payloads=[]
        for item in splits['fit']:
            for n in (8192,16384):
                collector=ArchitectureCollector(student,max_records=records);collector.eos_token_id=eos
                config=replace(cfg,length_policy='risk') if policy is not None else cfg
                row=run(item,n,mapper,config,policy,collector,cost=cost if policy is not None else None)
                row.update(mode=label,records=len(collector.records),budget_frames=len(collector.budget_frames))
                append(args.output/'collection.jsonl',row)
                payload=dict(phase=label,source_index=item['source_index'],length=n,document_sha256=item['document_sha256'],
                    records=collector.records,budget_frames=collector.budget_frames)
                file=args.cache/f'{label}_{item["source_index"]}_{n}.pt';torch.save(payload,file)
                append(args.output/'bundles.jsonl',dict(phase=label,path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
                payloads.append(payload)
                print('architecture collect',label,item['source_index'],n,len(collector.records),len(collector.budget_frames),flush=True)
        return payloads
    common=collect('initial_fixed4',query)
    full=copy.deepcopy(query)
    compact=NativeResidualMapper(layers=[27],teacher_layers=[query.teacher_layers[query.layers.index(27)]],
        source_dim=query.config['source_dim'],value_dim=query.config['value_dim'],query_dim=query.config['query_dim'],
        rank=16,residual_rank=8,gate_hidden=16,seed=612).to(teacher.device)
    full,_=fit_mapper(student,full,common,args.output,'full_stage1',epochs=4)
    compact,_=fit_mapper(student,compact,common,args.output,'compact_stage1',epochs=4)
    for label,model in (('full',full),('compact',compact)):
        own=collect(f'{label}_recollect',model)
        model,result=fit_mapper(student,model,common+own,args.output,f'{label}_final',epochs=2)
        if label=='full':full=model
        else:compact=model
        del own
    del common
    candidates=dict(native=(noinj,query),query=(cfg,query),full=(cfg,full),compact=(cfg,compact))
    tune=splits['tune'];warm=splits['warmup'][0]
    for n in (8192,16384):
        for config,mapper in candidates.values():run(warm,n,mapper,config,limit=8)
    mrows=[]
    for item in tune:
        for n in (8192,16384):
            names=list(candidates);random.Random(612+item['source_index']+n).shuffle(names)
            for name in names:
                config,mapper=candidates[name];row=run(item,n,mapper,config);row['mode']=name
                mrows.append(row);append(args.output/'mapper_tune.jsonl',row)
            print('architecture mapper tune',item['source_index'],n,flush=True)
    def geometric(rows,name,base):
        control={(r['source_index'],r['length']):r for r in rows if r['mode']==base}
        return math.exp(sum(math.log(r['total_tps']/control[r['source_index'],r['length']]['total_tps']) for r in rows if r['mode']==name)/len(control))
    selected=max(('query','full','compact'),key=lambda name:geometric(mrows,name,'query'))
    deployed=candidates[selected][1]
    write(args.output/'mapper_choice.json',dict(chosen=selected,chosen_overall=max(candidates,key=lambda name:geometric(mrows,name,'query')),
        scores={name:geometric(mrows,name,'query') for name in candidates},
        parameter_counts={name:sum(p.numel() for p in mapper.parameters()) for name,(_,mapper) in candidates.items()},
        selection='Train tune rolling geometric whole-request TPS; native no-injection separately eligible overall'))
    print('architecture mapper locked',selected,flush=True)
    # Length fitting is collected after final mapper weights are frozen.
    calibration_docs=[dict(ids=context(warm,n),source_index=f'{warm["source_index"]}:{n}') for n in (8192,16384)]
    cost,calibration=calibrate(teacher,student,deployed,oldrisk,replace(cfg,max_new_tokens=32),calibration_docs,eos,vocab)
    write(args.output/'cost.json',cost);write(args.output/'calibration.json',calibration)
    budget0=collect('budget_fixed4',deployed,records=0)
    policy,_=fit_budget(budget0,cost,args.output,'budget_stage1')
    policy=policy.to(teacher.device);policy.exploration=.25
    budget1=collect('budget_explore',deployed,records=0,policy=policy)
    policy,_=fit_budget(budget0+budget1,cost,args.output,'budget_final')
    policy=policy.to(teacher.device)
    policies={}
    for name,margin in (('step0',0.),('step5',.05)):
        m=copy.deepcopy(policy);m.margin=margin;policies[name]=m
    prior=copy.deepcopy(policy);prior.prior_weight=1.;prior.margin=.02
    policies['prior']=prior
    lrows=[]
    for n in (8192,16384):
        for name in ('fixed4',*policies):run(warm,n,deployed,replace(cfg,length_policy='fixed' if name=='fixed4' else 'risk'),policies.get(name),limit=8,cost=cost)
    for item in tune:
        for n in (8192,16384):
            names=['fixed4',*policies];random.Random(913+item['source_index']+n).shuffle(names)
            for name in names:
                row=run(item,n,deployed,replace(cfg,length_policy='fixed' if name=='fixed4' else 'risk'),policies.get(name),cost=cost)
                row['mode']=name;lrows.append(row);append(args.output/'length_tune.jsonl',row)
            print('architecture length tune',item['source_index'],n,flush=True)
    base=[r for r in lrows if r['mode']=='fixed4']
    rounds=sum(r['blocks'] for r in base);waste=sum(r['proposed']-r['accepted'] for r in base)
    eligible=[name for name in ('step0','step5') if sum(r['blocks'] for r in lrows if r['mode']==name)<=rounds*1.05 and
              sum(r['proposed']-r['accepted'] for r in lrows if r['mode']==name)<=waste]
    length_choice=max(eligible or ('step0','step5'),key=lambda name:geometric(lrows,name,'fixed4'))
    policy=policies[length_choice];torch.save(policy.payload(),args.output/'budget_selected.pt')
    torch.save(prior.payload(),args.output/'budget_prior.pt')
    write(args.output/'length_choice.json',dict(chosen=length_choice,eligible=eligible,
        scores={name:geometric(lrows,name,'fixed4') for name in ('fixed4',*policies)},
        note='Positive step candidate kept for confirmation even if fixed4 wins; not automatically promoted'))
    modes={**{name:(config,mapper,None) for name,(config,mapper) in candidates.items()},
        'step':(replace(cfg,length_policy='risk'),deployed,policy),'prior':(replace(cfg,length_policy='risk'),deployed,prior)}
    checkpoints=[args.output/'full_final.pt',args.output/'compact_final.pt',args.output/'budget_selected.pt',args.output/'budget_prior.pt']
    locked=dict(modes={name:asdict(c) for name,(c,_,_) in modes.items()},selected_mapper=selected,selected_length=length_choice,
        checkpoints_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in checkpoints},
        confirmation_ids=[r['source_index'] for r in splits['confirmation']])
    write(args.output/'locked.json',locked);print('architecture all choices locked',selected,length_choice,flush=True)
    for name,mapper in (('full',full),('compact',compact)):
        directory=args.output/f'quality_{name}';directory.mkdir()
        probe(teacher,student,tokenizer,query,mapper,splits['confirmation'],
            dict(hier512=noinj,m2_old=cfg,hier512_goal=cfg),directory,24)
    for n in (8192,16384):
        for config,mapper,model in modes.values():run(warm,n,mapper,config,model,limit=8,cost=cost)
    for item in splits['confirmation']:
        for n in (8192,16384):
            for repeat in range(2):
                names=list(modes);random.Random(7712+item['source_index']+n+repeat).shuffle(names)
                for name in names:
                    config,mapper,model=modes[name]
                    row=run(item,n,mapper,config,model,cost=cost);row.update(mode=name,repeat=repeat)
                    append(args.output/'runtime.jsonl',row)
                print('architecture confirmation',item['source_index'],n,repeat,flush=True)
    write(args.output/'COMPLETE.json',dict(passed=True,confirmation_requests=72))


if __name__=='__main__':main()
