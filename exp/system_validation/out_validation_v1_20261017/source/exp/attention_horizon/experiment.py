"""Shared models and true greedy SD with target-referenced full-key probes."""
import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'exp/attn_adapter_transfer'))
from common import load_model_and_tokenizer, tensor_ids
from run_attention_oracle import core_forward, legacy_cache, logits_of
from exp.attention_selection.attention import AttentionController
from transformers.models.qwen2.modeling_qwen2 import repeat_kv
from selection import ReadState, select, feature_vector, mandatory


def read_rows(path):
    path=Path(path)
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def write(path,value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def append(path,value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value,ensure_ascii=False)+'\n')


def sync():
    torch.cuda.synchronize()


def trim(cache,n):
    return tuple((k[:,:,:n],v[:,:,:n]) for k,v in legacy_cache(cache))


def confidence(logits):
    lp=F.log_softmax(logits.float(),-1)
    p=lp.exp()
    top=lp.topk(2).values
    return torch.stack((-(p*lp).sum()/math.log(len(p)),p.max(),top[0]-top[1]))


def metrics(logits,target):
    p,q=F.log_softmax(target,-1),F.log_softmax(logits,-1)
    return dict(kl=float((p.exp()*(p-q)).sum()),teacher_ce=float(-q[target.argmax()]),
                match=int(logits.argmax()==target.argmax()))


class Experiment:
    def __init__(self,args):
        self.a=args
        self.out=Path(args.output)
        self.out.mkdir(parents=True,exist_ok=True)
        self.data={s:read_rows(Path(args.data)/f'{s}.jsonl') for s in ('train','validation','test')}
        manifest=json.loads((Path(args.data)/'manifest.json').read_text())
        protocol=dict(vars(args),manifest=manifest,teacher_dtype='float32',student_dtype='bfloat16',
                      probe='Actual last-layer Q; full native student K; no extra model forward; charged to timing',
                      source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
        path=self.out/'protocol.json'
        if path.exists():
            old=json.loads(path.read_text())
            for key,value in vars(args).items():
                if key not in ('phase','seed') and old.get(key)!=value:
                    raise ValueError(f'Resume changed {key}')
            assert old['manifest']==manifest
        else:
            write(path,protocol)
        self.tt,self.teacher=load_model_and_tokenizer(args.teacher,torch.bfloat16,args.device)
        self.st,self.student=load_model_and_tokenizer(args.student,torch.bfloat16,args.device)
        self.teacher.to(dtype=torch.float32)
        self.teacher.requires_grad_(False)
        self.student.requires_grad_(False)
        self.tc=AttentionController(self.teacher).install()
        self.sc=AttentionController(self.student).install()
        self.sc.reference_observer=self.observe
        self.vocab=min(len(self.tt),len(self.st),self.teacher.config.vocab_size,self.student.config.vocab_size)
        self.shadow=None
        self.injection=None
        self.mapper=None
        self.risks={}
        self.cost=None

    def observe(self,index,module,q,k,v,mask,positions):
        if index!=27:
            return
        keys=repeat_kv(k,module.num_key_value_groups)
        score=torch.matmul(q[:,:,-1:],keys.transpose(-1,-2))/math.sqrt(module.head_dim)
        if mask is not None:
            score=score+mask[:,:,-1:]
        else:
            score=score.masked_fill(torch.arange(k.shape[-2],device=q.device)>positions[0,-1],torch.finfo(score.dtype).min)
        self.shadow=F.softmax(score.float(),-1)[0,:,0].detach()

    def forward(self,model,token,cache):
        return core_forward(model,tensor_ids([token],self.a.device),cache)

    def logit(self,model,out):
        return logits_of(model,out)[0,:self.vocab]

    def prefill(self,item):
        if self.injection is not None:
            self.injection.clear()
        self.sc.enabled=False
        self.tc.capture=self.sc.capture=True
        sync();tick=time.perf_counter()
        tout=core_forward(self.teacher,tensor_ids(item['ids'][:-1],self.a.device))
        sout=core_forward(self.student,tensor_ids(item['ids'][:-1],self.a.device))
        tc,sc=legacy_cache(tout.past_key_values),legacy_cache(sout.past_key_values)
        tr,sr=self.tc.rows[27],self.sc.rows[27]
        conf=confidence(self.logit(self.student,sout))
        sync();seconds=time.perf_counter()-tick
        self.tc.capture=False
        return tc,sc,tr,sr,conf,seconds

    def banks(self,tc,sc,tr,sr):
        assert tr.shape[-1]==sr.shape[-1]==tc[-1][1].shape[-2]==sc[-1][1].shape[-2]
        return ReadState(tr,tc[-1][1][0]),ReadState(sr,sc[-1][1][0])

    def make_selection(self,rule,t,s,previous=None):
        return select(rule,t,s,self.a.budget,self.a.recent,previous,self.a.swaps)

    def bind_injection(self,tc,selection,t):
        if self.injection is None:
            return
        remote=selection[~torch.isin(selection,mandatory(t.n,self.a.recent,selection.device))]
        order=torch.argsort(t.a.mean(0)[remote],descending=True,stable=True)
        positions=remote[order[:128]].sort().values
        source=torch.stack([tc[l][1][0,:,positions].transpose(0,1).reshape(len(positions),512)
                            for l in self.mapper.teacher_layers]).to(torch.bfloat16).float()
        self.injection.bind(self.mapper,positions,source,t.a.mean(0),'gated',.5)

    def load_mapper(self):
        if self.mapper is not None:
            return
        sys.path.insert(0,str(ROOT/'exp/selective_kv'))
        from bridge import Injection,ValueMapper
        payload=torch.load(ROOT/'exp/selective_kv/out_pilot/performance_gated.pt',map_location='cpu',weights_only=False)
        self.mapper=ValueMapper(**payload['config']).to(self.a.device).eval()
        self.mapper.load_state_dict(payload['state_dict'])
        self.injection=Injection(self.sc)

    @torch.no_grad()
    def reference(self,item):
        self.tc.capture=False
        sync();tick=time.perf_counter()
        out=core_forward(self.teacher,tensor_ids(item['ids'][:-1],self.a.device))
        cache=legacy_cache(out.past_key_values)
        sync();prefill=time.perf_counter()-tick
        current=item['ids'][-1];generated=[]
        tick=time.perf_counter()
        for _ in range(self.a.max_new_tokens):
            out=self.forward(self.teacher,current,cache)
            cache=legacy_cache(out.past_key_values)
            current=int(self.logit(self.teacher,out).argmax())
            generated.append(current)
            if current==self.st.eos_token_id:break
        sync();decode=time.perf_counter()-tick
        return dict(source_index=item['source_index'],method='teacher_greedy',generated=generated,tokens=len(generated),
                    prefill_seconds=prefill,decode_seconds=decode,total_seconds=prefill+decode,
                    decode_tps=len(generated)/decode,total_tps=len(generated)/(prefill+decode),output_agrees=True)

    @torch.no_grad()
    def offline(self,split):
        path=self.out/f'offline_{split}.jsonl'
        done={r['source_index'] for r in read_rows(path)}
        rules=['teacher','student','union','read_teacher','read_joint025','read_joint1']
        for index,item in enumerate(self.data[split]):
            if item['source_index'] in done:continue
            tc,sc,tr,sr,conf,_=self.prefill(item)
            current=item['ids'][-1]
            for step in range(33):
                t,s=self.banks(tc,sc,tr,sr)
                self.tc.capture=True
                tout=self.forward(self.teacher,current,tc)
                target=self.logit(self.teacher,tout)
                if step in (0,8,24):
                    for rule in rules:
                        sync();tick=time.perf_counter()
                        chosen=self.make_selection(rule,t,s)
                        sync();planning=time.perf_counter()-tick
                        self.sc.configure(chosen)
                        sout=self.forward(self.student,current,sc)
                        append(path,dict(source_index=item['source_index'],step=step,rule=rule,
                            teacher_read_error=float(t.error(chosen)),student_read_error=float(s.error(chosen)),
                            planning_seconds=planning,**metrics(self.logit(self.student,sout),target)))
                self.sc.configure(None)
                sout=self.forward(self.student,current,sc)
                if step in (0,8,24):
                    append(path,dict(source_index=item['source_index'],step=step,rule='native_full',
                        teacher_read_error=0.,student_read_error=0.,planning_seconds=0.,
                        **metrics(self.logit(self.student,sout),target)))
                tc,sc=legacy_cache(tout.past_key_values),legacy_cache(sout.past_key_values)
                tr,sr=self.tc.rows[27],self.shadow
                current=int(target.argmax())
                if current==self.st.eos_token_id:break
            self.tc.capture=False
            print(f'offline {split} {index+1}/{len(self.data[split])}',flush=True)


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',default='all',choices=['verify','offline','collect','fit','validate','test','deployment','all'])
    p.add_argument('--data',default='exp/attention_horizon/data')
    p.add_argument('--output',default='exp/attention_horizon/out')
    p.add_argument('--teacher',default='/mnt/8T/cl/models/Qwen2-7B-Instruct')
    p.add_argument('--student',default='/mnt/8T/cl/models/Qwen2-1.5B-Instruct')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--budget',type=int,default=512)
    p.add_argument('--recent',type=int,default=64)
    p.add_argument('--swaps',type=int,default=4)
    p.add_argument('--max-new-tokens',type=int,default=64)
    p.add_argument('--seed',type=int,default=42)
    return p
