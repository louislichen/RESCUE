"""Real-GPU correctness smoke for terminal elision and corrected Root geometry.

One training document, no inferential speed claim or learned policy selection.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from classic.model_classic import SPModel
from baseline_profiler import exact_input,load_texts
from csgb_same_state import release_request_caches
from run_csgb_controller import Harness,atomic


class FixedBudget:
    def __init__(self,budget):self.budget=budget
    def choose(self,x,explore=False):
        if explore:raise ValueError('Smoke must not explore')
        return self.budget


def main():
    ap=argparse.ArgumentParser()
    for key in ('base','draft','source','out'):ap.add_argument('--'+key,required=True)
    args=ap.parse_args()
    if torch.cuda.device_count()!=1:raise RuntimeError('One visible GPU required')
    out=Path(args.out);out.mkdir(exist_ok=False)
    torch.manual_seed(42);np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    model=SPModel.from_pretrained(base_model_path=args.base,draft_model_path=args.draft,
        torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map={'':0}).eval()
    model.base_model.float();model.csgb_attention_every_round=True;model.csgb_root_coverage=True
    h=Harness(model);text=load_texts(args.source,1)[0]
    atomic(out/'protocol.json',{'purpose':'correctness smoke, not efficacy','documents':[0],
        'contexts':[4096,8192],'output_limits':[1,8,128],'budgets':[8,24],'terminal_modes':[False,True],
        'repeats':1,'attention_every_round':True,'correct_root_coverage':True,
        'target_dtype':'float32','draft_dtype':'float16','cache_refresh_every':4,
        'source_sha256':hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    cases=0
    for c in (4096,8192):
        ids=exact_input(model.tokenizer,text,c,torch,torch.device('cuda:0'))
        for limit in (1,8,128):
            release_request_caches(model)
            with torch.inference_mode():reference=model.ar_generate(ids,max_new_tokens=limit,verbose=False,return_metrics=True)
            atomic(out/f'reference_{c}_{limit}.json',reference)
            order=[(b,mode) for b in (8,24) for mode in (False,True)]
            random.Random(999+c+limit).shuffle(order)
            for b,mode in order:
                release_request_caches(model);h.reset('csgb',FixedBudget(b),False,0.,clean=True)
                model.csgb_skip_terminal_draft=mode
                model.csgb_terminal_commit_callback=h.finalize_terminal if mode else None
                torch.cuda.synchronize();start=time.perf_counter()
                with torch.inference_mode():
                    result=model.spgenerate(ids,nodes=b,max_depth=10,threshold=.7,
                        max_new_tokens=limit,temperature=0,use_specextend=True,
                        retrieval_chunk_size=32,retrieve_top_k=32,retrieve_every_n_steps=4,
                        retrieval_num_queries=1,exact_target_attention=True,return_output_ids=True,
                        verbose=False,output_result_line=False)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                verified=len(result['raw_accept_length_list'])
                if result['output_token_ids']!=reference['output_token_ids']:raise RuntimeError('Canonical token mismatch')
                if len(h.rows)!=verified:raise RuntimeError('Last verified round missing')
                if sum(r['committed'] for r in h.rows)!=result['raw_total_generated']:raise RuntimeError('Raw commitment accounting mismatch')
                if mode and (h.pending is not None or not h.rows[-1].get('terminal_commit')):
                    raise RuntimeError('Terminal callback did not finalize exactly once')
                if not mode and h.pending is None:raise RuntimeError('Legacy compatibility path changed')
                row={'context':c,'limit':limit,'budget':b,'skip_terminal_draft':mode,
                     'seconds':elapsed,'tps':result['total_generated']/elapsed,'exact':True,
                     'result':result,'rounds':h.rows,'last_pending':h.pending,
                     'performance_claim':False}
                atomic(out/f'case_{c}_{limit}_{b}_{int(mode)}.json',row);cases+=1
                print(json.dumps({'context':c,'limit':limit,'budget':b,'terminal':mode,'exact':True,
                                  'verified_rounds':verified,'recorded_rounds':len(h.rows)}),flush=True)
    atomic(out/'DONE.json',{'passed':True,'cases':cases,'performance_claim':False})


if __name__=='__main__':main()
