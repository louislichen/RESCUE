"""Synthetic selector microbenchmark; not a model-quality or SD evaluation."""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F


@torch.no_grad()
def prepare(values, attention, candidates=1024, recent=64):
    n=len(attention[0]);device=attention[0].device
    anchors=torch.cat((torch.tensor([0],device=device),torch.arange(n-recent,n,device=device)))
    free=torch.arange(1,n-recent,device=device)
    pool=free[(attention[0][free]+attention[1][free]).topk(min(candidates,len(free))).indices]
    similarities=[];coverages=[];weights=[]
    for v,a in zip(values,attention):
        x=F.normalize(v,dim=-1)
        similarities.append((x@x[pool].T).clamp_min_(0))
        coverages.append((x@x[anchors].T).clamp_min_(0).max(-1).values)
        # Evaluate remote-history coverage; compulsory anchors may already
        # represent some of it. Model spaces remain separate.
        w=a.clone();w[anchors]=0;w=w/w.sum()
        weights.append(w)
    return similarities,coverages,weights


@torch.no_grad()
def greedy(similarities,initial,weights,slots=447,batch=1):
    coverage=[x.clone() for x in initial]
    k=similarities[0].shape[1]
    used=torch.zeros(k,device=similarities[0].device,dtype=torch.bool)
    result=[]
    for offset in range(0,slots,batch):
        gains=sum(((s-c[:,None]).clamp_min(0)*w[:,None]).sum(0)
                  for s,c,w in zip(similarities,coverage,weights))
        gains.masked_fill_(used,-torch.inf)
        count=min(batch,slots-offset)
        indices=gains.topk(count,sorted=False).indices
        used.scatter_(0,indices,True)
        for m,s in enumerate(similarities):
            coverage[m]=torch.maximum(coverage[m],s.index_select(1,indices).max(1).values)
        result.append(indices)
    return torch.cat(result)


def timed(call):
    torch.cuda.synchronize();start=time.perf_counter()
    result=call()
    torch.cuda.synchronize()
    return result,(time.perf_counter()-start)*1000


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='exp/attention_coverage/out_benchmark/summary.json')
    p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    measurements=[]
    for n in (2048,16384):
        values=[torch.randn(n,d,device='cuda',dtype=torch.float32) for d in (512,256)]
        attention=[(torch.randn(n,device='cuda')*2).softmax(0) for _ in range(2)]
        prepared,_=timed(lambda:prepare(values,attention))
        prep_times=[]
        for _ in range(args.repeats):
            prepared,elapsed=timed(lambda:prepare(values,attention));prep_times.append(elapsed)
        for batch in (1,16,32):
            greedy(*prepared,batch=batch);torch.cuda.synchronize()
            times=[]
            for _ in range(args.repeats):
                chosen,elapsed=timed(lambda:greedy(*prepared,batch=batch))
                assert chosen.unique().numel()==447
                times.append(elapsed)
            row=dict(n=n,candidates=1024,additional_tokens=447,teacher_dim=512,student_dim=256,
                     batch=batch,rounds=(447+batch-1)//batch,prepare_ms=prep_times,selection_ms=times,
                     prepare_median_ms=sorted(prep_times)[len(prep_times)//2],
                     selection_median_ms=sorted(times)[len(times)//2],
                     two_similarity_matrices_mib=2*n*1024*4/1024**2)
            row['total_median_ms']=row['prepare_median_ms']+row['selection_median_ms']
            measurements.append(row);print(json.dumps(row),flush=True)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(kind='synthetic selector microbenchmark',gpu=torch.cuda.get_device_name(0),
                 dtype='float32',tf32=False,torch=torch.__version__,repeats=args.repeats,
                 notes=['No models loaded, no acceptance/quality measurements.',
                        'Includes cosine-matrix preparation and selection; excludes attention acquisition and model execution.',
                        'Batch=1 evaluates exact marginal gains within the fixed pool.',
                        'Batch>1 is an approximation: within-batch redundancy is not recomputed.',
                        'All 447 iterations remain on GPU, without per-iteration CPU argmax synchronization.'],
                 measurements=measurements)
    path.write_text(json.dumps(payload,indent=2)+'\n')


if __name__=='__main__':main()
