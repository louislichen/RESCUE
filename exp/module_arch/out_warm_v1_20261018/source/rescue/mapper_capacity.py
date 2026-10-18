"""Training-monitor oracle diagnostic: route capacity versus mapper direction.

The per-query oracle sees the Teacher target and cannot be deployed. It adapts
only the selected last-layer V values, with each component bounded by native
token RMS, without modifying keys, queries, history, or base model weights.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from exp.selective_kv.bridge import ValueMapper
from .loading import DEFAULT_STUDENT
from .objectives import distribution_quality
from .rollout_training import replay


class OracleValues(torch.nn.Module):
    def __init__(self,native):
        super().__init__();self.raw=torch.nn.Parameter(torch.zeros_like(native))

    def correction(self,relative,source,native,query,features,alpha=1.):
        scale=native.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-5)
        normalized=self.raw.tanh()
        return native+scale*normalized,torch.ones_like(scale),normalized


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,default=Path('exp/long_system/out_joint_train_v2_20261017'))
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--steps',type=int,default=24)
    args=p.parse_args();torch.set_num_threads(4);torch.manual_seed(58)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    target=args.study/'capacity.json'
    if target.exists(): p.error('Refuse to overwrite diagnostic')
    student=AutoModelForCausalLM.from_pretrained(DEFAULT_STUDENT,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='eager').cuda().eval().requires_grad_(False)
    def load(path):
        v=torch.load(path,map_location='cpu',weights_only=False)
        model=ValueMapper(**v['config']).cuda();model.load_state_dict(v['state_dict'])
        return model.eval().requires_grad_(False)
    old,new=load(args.training/'mapper_joint.pt'),load(args.study/'mapper_goal.pt')
    records=[]
    for line in (args.training/'bundles.jsonl').read_text().splitlines():
        b=json.loads(line)
        if b['source_index']!=236 or not b['inject_tokens']: continue
        path=Path(b['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==b['sha256']
        records.extend(torch.load(path,weights_only=False)['records'])
    rows=[]
    for i,r in enumerate(records):
        target_logits=r['target'].cuda()
        with torch.no_grad():
            native=replay(student,old,r,strength=0.)[0]
            nq=distribution_quality(native,target_logits)
            scales={}
            for label,model in (('old',old),('goal',new)):
                scales[label]={str(a):distribution_quality(replay(student,model,r,strength=a)[0],target_logits)
                               for a in (.25,.5,1.,2.)}
        result=dict(query=i,query_index=r['query_index'],prefix=r['prefix'],native=nq,scaled_mapper=scales)
        if not nq['top1_matches_target']:
            model=OracleValues(r['native'].cuda())
            optimizer=torch.optim.Adam(model.parameters(),lr=.1)
            best=nq
            for step in range(args.steps):
                optimizer.zero_grad(set_to_none=True)
                logits,penalty=replay(student,model,r)
                q=distribution_quality(logits,target_logits)
                if q['teacher_argmax_margin']>best['teacher_argmax_margin']: best=q
                if best['top1_matches_target']: break
                loss=F.cross_entropy(logits[None],target_logits.argmax()[None])+.01*penalty
                loss.backward();optimizer.step()
            with torch.no_grad():
                q=distribution_quality(replay(student,model,r)[0],target_logits)
                if q['teacher_argmax_margin']>best['teacher_argmax_margin']: best=q
            result.update(oracle=best,oracle_steps=step+1)
        else: result.update(oracle=nq,oracle_steps=0)
        rows.append(result)
    result=dict(scope='Training-monitor per-query privileged oracle; not learned mapper performance, deployable gating, or system evidence',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        monitor_document=236,queries=len(rows),native_matches=sum(r['native']['top1_matches_target'] for r in rows),
        bounded_value_oracle_matches=sum(r['oracle']['top1_matches_target'] for r in rows),
        strengths={label:{str(a):sum(r['scaled_mapper'][label][str(a)]['top1_matches_target'] for r in rows)
            for a in (.25,.5,1.,2.)} for label in ('old','goal')},
        learned_scale_oracle={label:sum(r['native']['top1_matches_target'] or any(
            v['top1_matches_target'] for v in r['scaled_mapper'][label].values()) for r in rows) for label in ('old','goal')},
        limits='Optimistic target-aware local V feasibility only; selected positions and frozen old-rollout entering states are fixed. No historical backprop or content-source attribution.',rows=rows)
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__': main()
