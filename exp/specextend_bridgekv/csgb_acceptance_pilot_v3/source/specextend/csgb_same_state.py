"""Replay fixed8 to a locked round, intervene once, stop after verification.

Full active Target/Draft KV fingerprints are computed before the measured
intervention. This is an instrumented one-round mechanism audit, never TPS.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from classic.model_classic import SPModel
from baseline_profiler import exact_input,load_texts
from run_csgb_controller import Harness,atomic


class AuditStop(Exception):pass


def release_request_caches(model):
    """Drop this process's previous-request KV before allocating a new one.

    Every replay starts from scratch; retaining the previous full Target cache
    during RHS allocation otherwise creates an unnecessary peak. Never touches
    model parameters or other processes.
    """
    for name in ('past_key_values','past_key_values_data','current_length_data',
                 'draft_stable_kv','full_draft_kv','attn_scores','attn_scores_final',
                 'cached_chunks','chunks','selected_chunks'):
        setattr(model,name,None)
    model.draft_model.model.past_key_position_ids=None
    gc.collect()
    torch.cuda.empty_cache()


def fingerprint(model,ids,harness):
    digest=hashlib.sha256()
    def blob(tag,data):
        digest.update(tag+str(len(data)).encode()+b':'+data)
    def feed(value):
        if isinstance(value,torch.Tensor):
            a=value.detach().contiguous().cpu().numpy()
            blob(b'T',str((a.shape,a.dtype.str)).encode());blob(b'B',a.tobytes())
        elif isinstance(value,np.ndarray):
            blob(b'N',str((value.shape,value.dtype.str)).encode());blob(b'B',value.tobytes())
        elif isinstance(value,(tuple,list)):
            digest.update(b'L'+str(len(value)).encode()+b':')
            for item in value:feed(item)
        else:blob(b'J',json.dumps(value,sort_keys=True).encode())
    feed(ids)
    for name in ['total_seq_len','seq_len_total_old','evicted','selected_chunks','cached_chunks',
                 'chunks','retrieval_condition','timestep','num_chunks_old','recent_start','recent_end']:
        feed(name);feed(getattr(model,name,None))
    feed(model.draft_stable_kv)
    for pair in model.full_draft_kv:
        for kv in pair:feed(kv[:,:,:model.total_seq_len,:])
    for pair in model.past_key_values:
        for kv in pair:feed(kv.data[:,:,:int(kv.current_length.item()),:])
    feed(model.draft_model.model.past_key_position_ids)
    feed(model.attn_scores_final);feed(harness.anchor);feed(harness.latest_attention)
    return digest.hexdigest()


class Schedule:
    def __init__(self,harness):self.h=harness
    def choose(self,x,explore=False):
        return self.h.intervention_budget if self.h.round_index==self.h.intervention_round else 8


class InterventionHarness(Harness):
    def configure(self,round_index,budget,context):
        self.reset('csgb',Schedule(self),False,0.)
        self.intervention_round=round_index;self.intervention_budget=budget;self.context=context
        self.before_hash=None;self.observation=None

    def draft(self,model,input_ids,nodes,threshold,max_depth):
        if self.round_index==self.intervention_round+1:
            torch.cuda.synchronize();end=time.perf_counter()
            row=dict(self.pending)
            row['seconds']=end-row.pop('start')
            row['committed']=int(input_ids.shape[-1])-row['prefix']
            row['state_sha256']=self.before_hash
            row['attention_observed_timestep']=getattr(self,'observed_timestep',None)
            # The last token is the next Target root, not yet committed.
            row['committed_output_ids']=input_ids[0,self.context:-1].cpu().tolist()
            self.observation=row
            raise AuditStop()
        if self.round_index==self.intervention_round:
            torch.cuda.synchronize()
            self.before_hash=fingerprint(model,input_ids,self)
            self.observed_timestep=getattr(model,'csgb_attention_observed_timestep',None)
        return super().draft(model,input_ids,nodes,threshold,max_depth)


def main():
    ap=argparse.ArgumentParser()
    for key in ('base','draft','source','out'):ap.add_argument('--'+key,required=True)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--fresh-attention',action='store_true',
                    help='Observe verification attention every round; cache refresh stays every 4 rounds')
    ap.add_argument('--training-documents',type=int,default=2)
    ap.add_argument('--document-ids',type=int,nargs='+',help='Explicit training-only shard, IDs 0..23')
    ap.add_argument('--intervention-rounds',type=int,nargs='+',default=[5,8,12])
    args=ap.parse_args()
    if not 1 <= args.training_documents <= 24:
        ap.error('Only the first 1..24 training documents may be used')
    if args.document_ids is not None and (len(set(args.document_ids))!=len(args.document_ids) or min(args.document_ids)<0 or max(args.document_ids)>=24):
        ap.error('Shard IDs must be unique training document IDs 0..23')
    if args.smoke and args.document_ids is not None:
        ap.error('Smoke and explicit shard cannot be combined')
    if not args.intervention_rounds or min(args.intervention_rounds)<5 or len(set(args.intervention_rounds))!=len(args.intervention_rounds):
        ap.error('Intervention rounds must be unique and at least 5')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    if torch.cuda.device_count()!=1:raise RuntimeError('Exactly one visible GPU required')
    torch.manual_seed(42);np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    model=SPModel.from_pretrained(base_model_path=args.base,draft_model_path=args.draft,
        torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map={'':0}).eval()
    model.base_model.float();model.csgb_attention_every_round=args.fresh_attention
    h=InterventionHarness(model)
    docs=[0] if args.smoke else (args.document_ids if args.document_ids is not None else list(range(args.training_documents)));contexts=[4096] if args.smoke else [4096,8192]
    rounds=[5] if args.smoke else args.intervention_rounds;budgets=[8,32] if args.smoke else [8,16,24,32]
    repeats=1 if args.smoke else 2
    protocol={'phase':'same_state_mechanism_audit_not_tps','documents':docs,'contexts':contexts,
        'intervention_rounds':rounds,'budgets':budgets,'repeats':repeats,'baseline_replay_budget':8,
        'target_dtype':'float32','draft_dtype':'float16','max_depth':10,'threshold':.7,
        'attention_observation_every_round':args.fresh_attention,'cache_refresh_every_rounds':4,
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu_name':torch.cuda.get_device_name(0),
        'gpu_uuid':str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable')),
        'timing':'one instrumented round, fingerprint excluded; not request TPS',
        'request_cache_release':'drop own previous-request KV and allocator cache before each independent replay, outside timer',
        'source_sha256':hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    atomic(out/'protocol.json',protocol);texts=load_texts(args.source,max(docs)+1)
    failures=[];state_hashes={};cases=[]
    for d in docs:
        for c in contexts:
            release_request_caches(model)
            ids=exact_input(model.tokenizer,texts[d],c,torch,torch.device('cuda:0'))
            with torch.inference_mode():
                ref=model.ar_generate(ids,max_new_tokens=128,verbose=False,return_metrics=True)
            atomic(out/f'reference_{c}_{d}.json',ref)
            for j in rounds:
                for repeat in range(repeats):
                    order=list(budgets);random.Random(4400+d*100+c+j+repeat).shuffle(order)
                    for b in order:cases.append((d,c,j,repeat,b))
    # Warm every intervention shape before collecting paired timing. Save,
    # but do not analyze, warmup outcomes as measured cases.
    warmups=[(docs[0],c,rounds[0],-1,b) for c in contexts for b in budgets]
    for d,c,j,repeat,b in warmups+cases:
        release_request_caches(model)
        ids=exact_input(model.tokenizer,texts[d],c,torch,torch.device('cuda:0'));h.configure(j,b,c)
        try:
            with torch.inference_mode():
                model.spgenerate(ids,nodes=8,max_depth=10,threshold=.7,max_new_tokens=128,
                    temperature=0,use_specextend=True,retrieval_chunk_size=32,retrieve_top_k=32,
                    retrieve_every_n_steps=4,retrieval_num_queries=1,exact_target_attention=True,
                    return_output_ids=True,verbose=False,output_result_line=False)
        except AuditStop:pass
        row=h.observation
        if row is None:raise RuntimeError('Intervention was not reached; do not silently drop cases')
        key=f'{c}_{d}_{j}';previous=state_hashes.setdefault(key,row['state_sha256'])
        if row['state_sha256']!=previous:failures.append(['state_mismatch',d,c,j,repeat,b])
        ref=json.loads((out/f'reference_{c}_{d}.json').read_text())['output_token_ids']
        output=row['committed_output_ids']
        if output!=ref[:len(output)]:failures.append(['canonical_mismatch',d,c,j,repeat,b])
        if row['x'] is None:failures.append(['missing_features',d,c,j,repeat,b])
        row.update(document=d,context=c,intervention_round=j,repeat=repeat)
        phase='warmup' if repeat<0 else 'case'
        atomic(out/f'{phase}_{c}_{d}_{j}_{repeat}_{b}.json',row)
        print(json.dumps({'document':d,'context':c,'round':j,'repeat':repeat,'budget':b,
            'committed':row['committed'],'seconds':row['seconds'],'failures':len(failures)}),flush=True)
    atomic(out/'AUDIT_DONE.json',{'passed':not failures,'failures':failures,'cases':len(cases),
                               'state_groups':len(state_hashes),'production_timing':False})


if __name__=='__main__':main()
