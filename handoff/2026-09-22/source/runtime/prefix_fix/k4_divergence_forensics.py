"""RESCUE 002 / STOP-C forensics: compare batched vs incremental target paths at divergence."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefix_seqloss import PrefixSequenceBackend, load_backend_module
from real_k4_runner import EOS_ID, K, StudentDraft, TeacherTarget, build_draft_condition, build_target

def topk(logits,k=5):
    v,i=torch.topk(logits,k); return {"ids":[int(x) for x in i.tolist()],"logits":[round(float(x),6) for x in v.tolist()],"margin_top1_top2":float(v[0]-v[1])}
def target_logits_batched(model,device,ids,vocab):
    t=torch.tensor([ids],device=device,dtype=torch.long); m=torch.ones((1,len(ids)),device=device,dtype=torch.long)
    with torch.no_grad(): out=model(input_ids=t,attention_mask=m,use_cache=True,output_attentions=False)
    return out.logits[0,-1,:vocab].float()
def target_logits_incremental(model,device,ids,vocab):
    t=torch.tensor([ids[:-1]],device=device,dtype=torch.long); m=torch.ones((1,len(ids)-1),device=device,dtype=torch.long)
    with torch.no_grad():
        pre=model(input_ids=t,attention_mask=m,use_cache=True,output_attentions=False)
        nxt=torch.tensor([[ids[-1]]],device=device,dtype=torch.long); m2=torch.ones((1,len(ids)),device=device,dtype=torch.long)
        out=model(input_ids=nxt,attention_mask=m2,past_key_values=pre.past_key_values,use_cache=True,output_attentions=False)
    return out.logits[0,-1,:vocab].float()
def main():
    ap=argparse.ArgumentParser()
    for n in ("root","source-dir","source-lock","teacher-model","student-model","dataset-path","validation-ledger","validation-prior-cache","checkpoint","out"): ap.add_argument("--"+n,required=True)
    ap.add_argument("--target-dtype",default="bfloat16"); ap.add_argument("--target-max-memory",default="8GiB"); ap.add_argument("--device",default="cuda:0"); ap.add_argument("--max-new-tokens",type=int,default=16); ap.add_argument("--worst",type=int,default=6)
    args=ap.parse_args()
    backend_mod,runner_mod=load_backend_module(args.source_dir,args.source_lock)
    backend=backend_mod.QwenAdapterBackend(args.teacher_model,args.student_model,args.device,source_lock=args.source_lock,load_teacher=False)
    wrap=PrefixSequenceBackend(backend); payload=torch.load(args.checkpoint,map_location=backend.device,weights_only=False); wrap.adapter.load_state_dict(payload["adapter"])
    model,device,dmap=build_target(args); vocab=min(int(backend.student.config.vocab_size),int(model.config.vocab_size))
    from fixed_k4_protocol import fixed_k4_decode, greedy_reference
    rows=runner_mod.load_rows(Path(args.validation_ledger),Path(args.dataset_path)); priors=torch.load(Path(args.validation_prior_cache),map_location="cpu",weights_only=True)
    out={"probe":"k4_divergence_forensics","k":K,"target_dtype":args.target_dtype,"common_vocab_limit":vocab,"target_device_map":dmap,"records":[]}
    checked=0
    for row in rows:
        if checked>=args.worst: break
        target,*_=build_draft_condition(args,backend,wrap,row,priors[row["canonical_source_id"]]); prompt_ids=target.input_ids
        ref=greedy_reference(prompt_ids,TeacherTarget(model,device,vocab),EOS_ID,args.max_new_tokens)
        spec=fixed_k4_decode(prompt_ids,StudentDraft(backend,vocab),TeacherTarget(model,device,vocab),EOS_ID,k=K,max_new_tokens=args.max_new_tokens).token_ids
        d=next((i for i,(a,b) in enumerate(zip(spec,ref)) if a!=b),None)
        if d is None and len(spec)!=len(ref): d=min(len(spec),len(ref))
        if d is None: continue
        checked+=1; prefix=list(prompt_ids)+list(spec[:d]); lb=target_logits_batched(model,device,prefix,vocab); li=target_logits_incremental(model,device,prefix,vocab)
        out["records"].append({"canonical_source_id":row["canonical_source_id"],"source_index":row["source_index"],"prompt_len":len(prompt_ids),"divergence_index":d,"speculative_next":int(spec[d]) if d<len(spec) else None,"reference_next":int(ref[d]) if d<len(ref) else None,"batched":topk(lb),"incremental":topk(li),"batched_argmax":int(torch.argmax(lb)),"incremental_argmax":int(torch.argmax(li)),"paths_agree":bool(int(torch.argmax(lb))==int(torch.argmax(li))),"max_abs_logit_diff":float((lb-li).abs().max())})
    Path(args.out).write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); return 0
if __name__=="__main__": raise SystemExit(main())
