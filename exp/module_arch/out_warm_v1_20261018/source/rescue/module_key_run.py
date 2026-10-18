"""M2 read-conditioned direction and separate matched M3 Pareto endpoints."""
import argparse
from dataclasses import replace,asdict
import hashlib
import json
from pathlib import Path
import random
import torch

from .compare_baselines import append,write
from .config import SystemConfig
from .engine import RescueEngine
from .loading import ROOT,load_models,load_modules
from .long_data import context
from .module_opt_training import train_mapper
from .module_study import probe
from .objectives import accounting
from .packing import pack_projections
from .query_mapper import mapper_from_payload
from .policy import RiskModel,FeedbackRisk


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Use fresh output')
    assert (args.base/'COMPLETE.json').exists()
    args.output.mkdir(parents=True)
    torch.set_num_threads(4);torch.manual_seed(181)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    protocol=json.loads((args.base/'protocol.json').read_text());data=Path(protocol['args']['data'])
    sources={}
    for path in [*sorted((ROOT/'rescue').glob('*.py')),*sorted((ROOT/'tests').glob('*.py'))]:
        name=str(path.relative_to(ROOT));sources[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        dest=args.output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    write(args.output/'protocol.json',dict(base=str(args.base),source_sha256=sources,
        plan_sha256=hashlib.sha256(Path('exp/module_opt/key_attention_plan.json').read_bytes()).hexdigest(),
        length_pareto_plan_sha256=hashlib.sha256(Path('exp/module_opt/length_pareto_plan.json').read_bytes()).hexdigest(),
        scope='M2 plan preceded base confirmation; extra M3 Pareto endpoints use training-only counts, same confirmation cohort, no combined key-mapper/length claim',
        data_sha256=protocol['data_manifest']['data_sha256'],lengths=[8192,16384],repeats=2,max_new_tokens=128))
    teacher,student,tokenizer=load_models(efficient=True)
    old,risk,cost=load_modules(teacher.device,mapper=ROOT/'exp/long_system/out_joint_train_v2_20261017/mapper_joint.pt',
        risk=ROOT/'exp/long_system/out_joint_train_v2_20261017/risk_injected.pt',cost=args.base/'cost.json')
    pack_projections(teacher);pack_projections(student)
    student.requires_grad_(False)
    payloads=[]
    for line in (args.base/'bundles.jsonl').read_text().splitlines():
        b=json.loads(line)
        if b['phase']!='mapper':continue
        path=Path(b['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==b['sha256']
        payloads.append(torch.load(path,weights_only=False))
    keyed,promoted=train_mapper(student,old,payloads,args.output,attention_features=student.config.num_key_value_heads,projected=True)
    del payloads
    direction=mapper_from_payload(torch.load(args.base/'mapper_query.pt',map_location='cpu',weights_only=False),teacher.device).eval().requires_grad_(False)
    chosen=json.loads((args.base/'selector_choice.json').read_text())['config'];chosen['injection_layers']=tuple(chosen['injection_layers'])
    cfg=SystemConfig(**chosen)
    light=RiskModel.load(args.base/'risk_light.pt',teacher.device)
    prior=json.loads((args.base/'risk_training.json').read_text())['prior']
    rates=json.loads((args.base/'length_choice.json').read_text())['rates']
    risk_cfg=replace(cfg,inject_tokens=64,length_policy='risk')
    modes=dict(native=(cfg,old,risk),old=(replace(cfg,inject_tokens=64),old,risk),
        direction=(replace(cfg,inject_tokens=64),direction,risk),keyed=(replace(cfg,inject_tokens=64),keyed,risk),
        old_risk=(risk_cfg,old,risk),
        feedback_ratio=(risk_cfg,old,FeedbackRisk(light,prior,strength=8,rule='ratio')),
        feedback_surplus=(risk_cfg,old,FeedbackRisk(light,prior,strength=8,rule='surplus',rates=rates)))
    confirmation=[json.loads(l) for l in (data/'confirmation.jsonl').read_text().splitlines()]
    warm=json.loads((data/'warmup.jsonl').read_text().splitlines()[0])
    refs=json.loads((args.base/'references.json').read_text())
    write(args.output/'locked.json',dict(modes={name:asdict(c) for name,(c,_,_) in modes.items()},
        promoted_on_monitor=promoted,confirmation_ids=[r['source_index'] for r in confirmation],
        mapper_sha256=hashlib.sha256((args.output/'mapper_query.pt').read_bytes()).hexdigest()))
    probe(teacher,student,tokenizer,old,keyed,confirmation,
        dict(full_native=replace(cfg,selector='full'),hier512=cfg,m2_old=replace(cfg,inject_tokens=64),hier512_goal=replace(cfg,inject_tokens=64)),args.output,24)
    def run(item,n,name,limit=128):
        c,m,policy=modes[name]
        with RescueEngine(teacher,student,replace(c,max_new_tokens=limit),m,policy,cost,vocab_size=len(tokenizer)) as engine:
            r=engine.generate(context(item,n),tokenizer.eos_token_id)
        reference=refs[f"{item['split']}:{item['source_index']}:{n}"]
        r.update(mode=name,source_index=item['source_index'],split=item['split'],length=n,
            output_agrees=r['generated']==reference[:limit],accounting=accounting(r))
        assert r['output_agrees']
        return r
    for n in (8192,16384):
        for name in modes:run(warm,n,name,8)
    for item in confirmation:
        for n in (8192,16384):
            for repeat in range(2):
                names=list(modes);random.Random(181+item['source_index']+n+repeat).shuffle(names)
                for name in names:
                    row=run(item,n,name);row['repeat']=repeat;append(args.output/'runtime.jsonl',row)
                print('key confirmation complete',item['source_index'],n,repeat,flush=True)
    write(args.output/'COMPLETE.json',dict(passed=True,requests=len(modes)*3*2*2))


if __name__=='__main__':main()
