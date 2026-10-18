"""Predefined M2-only follow-up: open residual gate and full-span sampling."""
import argparse
from dataclasses import replace,asdict
import hashlib
import json
from pathlib import Path
import random

import torch

from .architecture_training import ArchitectureCollector,fit_mapper
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
from .policy import RiskModel


def spread(values,count):
    if count<=0:return []
    if len(values)<=count:return values
    if count==1:return [values[len(values)//2]]
    return [values[round(j*(len(values)-1)/(count-1))] for j in range(count)]


def sample_span(records,monitor=False,count=48):
    if monitor:return spread(list(range(len(records))),count)
    wrong=[i for i,r in enumerate(records) if int(r['student_logits'][0,:r['target'].numel()].argmax())!=int(r['target'].argmax())]
    correct=[i for i in range(len(records)) if i not in wrong]
    indices=spread(wrong,count//2)+spread(correct,count//2)
    if len(indices)<min(count,len(records)):
        remaining=[i for i in range(len(records)) if i not in indices]
        indices+=spread(remaining,min(count,len(records))-len(indices))
    return sorted(indices)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('/mnt/8T/cl/rescue_experiment_cache/module_arch_warm_v1'))
    args=p.parse_args()
    if args.output.exists() or args.cache.exists():p.error('Use new output and tensor cache')
    assert (args.base/'COMPLETE.json').exists()
    args.output.mkdir(parents=True);args.cache.mkdir(parents=True)
    sources={}
    for path in [*sorted((ROOT/'rescue').glob('*.py')),*sorted((ROOT/'tests').glob('*.py')),
                 ROOT/'exp/selective_kv/bridge.py',ROOT/'exp/attention_horizon/selection.py']:
        name=str(path.relative_to(ROOT));sources[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        dest=args.output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    base=json.loads((args.base/'protocol.json').read_text());data=Path(base['args']['data'])
    plan=ROOT/'exp/module_arch/warm_gate_plan.json'
    write(args.output/'protocol.json',dict(base=str(args.base),source_sha256=sources,
        plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),data_sha256=base['data_manifest']['data_sha256'],
        length_plan_sha256=hashlib.sha256((ROOT/'exp/module_arch/length_speed_plan.json').read_bytes()).hexdigest(),
        lengths=[8192,16384],max_new_tokens=128,repeats=2,seed=713,
        scope='M2 supplement and separate matched M3 speed endpoint, same three documents; not independent replication or warm mapper/length joint result'))
    (args.output/'PLAN.json').write_bytes(plan.read_bytes())
    torch.set_num_threads(4);torch.manual_seed(713)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    teacher,student,tokenizer=load_models(efficient=True)
    _,risk,cost=load_modules(teacher.device,cost=args.base/'cost.json')
    def load(path):return mapper_from_payload(torch.load(path,map_location='cpu',weights_only=False),teacher.device).eval().requires_grad_(False)
    query=load(ROOT/'exp/module_opt/out_v1_20261018/mapper_query.pt');full=load(args.base/'full_final.pt')
    pack_projections(teacher);pack_projections(student)
    cfg=SystemConfig(selector='hierarchical',coverage_rows=64,length_policy='fixed',fixed_k=4,
        injection_layers=(27,),inject_tokens=64,prefill_precision='bfloat16',max_new_tokens=128)
    splits={s:[json.loads(l) for l in (data/f'{s}.jsonl').read_text().splitlines()] for s in base['data_manifest']['data_sha256']}
    refs=json.loads((args.base/'references.json').read_text());eos,vocab=tokenizer.eos_token_id,len(tokenizer)
    def run(item,n,mapper,config=cfg,collector=None,limit=128,policy=None):
        with RescueEngine(teacher,student,replace(config,max_new_tokens=limit),mapper,policy or risk,cost,vocab_size=vocab,
            draft_observer=collector,round_observer=collector.round if collector else None) as engine:
            if collector:collector.engine=engine
            r=engine.generate(context(item,n),eos)
        if collector:collector.engine=None
        r.update(source_index=item['source_index'],split=item['split'],length=n,
            output_agrees=r['generated']==refs[f"{item['split']}:{item['source_index']}:{n}"][:limit],accounting=accounting(r))
        assert r['output_agrees']
        return r
    def collect(label,mapper):
        result=[]
        for item in splits['fit']:
            for n in (8192,16384):
                collector=ArchitectureCollector(student,max_records=256);collector.eos_token_id=eos
                r=run(item,n,mapper,collector=collector);r['mode']=label;append(args.output/'collection.jsonl',r)
                raw=dict(phase=label,source_index=item['source_index'],length=n,records=collector.records,budget_frames=collector.budget_frames)
                file=args.cache/f'{label}_{item["source_index"]}_{n}.pt';torch.save(raw,file)
                indices=sample_span(collector.records,monitor=item['source_index']==236)
                chosen=[collector.records[i] for i in indices]
                append(args.output/'bundles.jsonl',dict(phase=label,path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                    selected_indices=indices,raw_records=len(collector.records),chosen_records=len(chosen),
                    chosen_wrong=sum(int(r['student_logits'][0,:r['target'].numel()].argmax())!=int(r['target'].argmax()) for r in chosen),
                    min_prefix=min(r['prefix'] for r in chosen),max_prefix=max(r['prefix'] for r in chosen)))
                result.append(dict(raw,records=chosen));print('warm collect',label,item['source_index'],n,len(collector.records),len(chosen),flush=True)
        return result
    common=collect('full_span_query',query)
    mapper=NativeResidualMapper(layers=[27],teacher_layers=[query.teacher_layers[query.layers.index(27)]],
        source_dim=query.config['source_dim'],value_dim=query.config['value_dim'],query_dim=query.config['query_dim'],
        rank=16,residual_rank=8,gate_hidden=16,seed=713).to(teacher.device)
    with torch.no_grad():mapper.gates[0][-1].bias.fill_(0.)
    mapper,_=fit_mapper(student,mapper,common,args.output,'warm_stage1',epochs=4)
    own=collect('full_span_recollect',mapper)
    mapper,_=fit_mapper(student,mapper,common+own,args.output,'warm_final',epochs=2)
    del common,own
    compact=load(args.base/'compact_final.pt')
    fast=RiskModel.load(args.base/'budget_final.pt',teacher.device);fast.margin=0.
    constrained=RiskModel.load(args.base/'budget_selected.pt',teacher.device)
    modes=dict(native=(replace(cfg,inject_tokens=0),query,None),query=(cfg,query,None),full=(cfg,full,None),warm=(cfg,mapper,None),
        compact_fixed=(cfg,compact,None),length_fast=(replace(cfg,length_policy='risk'),compact,fast),
        length_constrained=(replace(cfg,length_policy='risk'),compact,constrained))
    write(args.output/'locked.json',dict(modes={name:asdict(c) for name,(c,_,_) in modes.items()},
        mapper_sha256=hashlib.sha256((args.output/'warm_final.pt').read_bytes()).hexdigest(),
        confirmation_ids=[r['source_index'] for r in splits['confirmation']]))
    directory=args.output/'quality';directory.mkdir()
    probe(teacher,student,tokenizer,query,mapper,splits['confirmation'],
        dict(hier512=replace(cfg,inject_tokens=0),m2_old=cfg,hier512_goal=cfg),directory,24)
    warm=splits['warmup'][0]
    for n in (8192,16384):
        for config,m,policy in modes.values():run(warm,n,m,config,limit=8,policy=policy)
    for item in splits['confirmation']:
        for n in (8192,16384):
            for repeat in range(2):
                names=list(modes);random.Random(713+item['source_index']+n+repeat).shuffle(names)
                for name in names:
                    config,m,policy=modes[name];r=run(item,n,m,config,policy=policy);r.update(mode=name,repeat=repeat);append(args.output/'runtime.jsonl',r)
                print('warm confirmation',item['source_index'],n,repeat,flush=True)
    write(args.output/'COMPLETE.json',dict(passed=True,confirmation_requests=12*len(modes)))


if __name__=='__main__':main()
