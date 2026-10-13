#!/usr/bin/env python3
"""Controlled full-prefix Q/QK/QV/QKV soft-output and relation distillation."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch

from run_compare import (aggregate, append, core_forward, legacy_cache, load_model_and_tokenizer,
                         logits_of, paired_interval, prefix_cache, tensor_ids, trim_cache, write_json)
from qkv_methods import AttentionController, ProjectionUpdates, distribution_kl, relation_kl, upper_forward


@torch.no_grad()
def replay(student,frame,controller,args):
    controller.capture=False
    cache=prefix_cache(student,frame["ids"],args.device)
    reference=frame["reference"]
    generated,blocks,last=[],[],frame["ids"][-1]
    while len(generated)<len(reference):
        position=len(generated)
        draft=[];token=last
        for _ in range(min(args.draft_k,args.max_new_tokens-position)):
            out=core_forward(student,tensor_ids([token],args.device),cache)
            cache=legacy_cache(out.past_key_values)
            token=int(logits_of(student,out).argmax(-1));draft.append(token)
            if token==args.eos_id:
                break
        accepted=0
        for i,token in enumerate(draft):
            if position+i>=len(reference) or token!=reference[position+i]:
                break
            accepted+=1
            if token==args.eos_id:
                break
        blocks.append(dict(position=position,proposed=len(draft),accepted=accepted))
        generated.extend(draft[:accepted])
        if len(generated)==len(reference):
            break
        correction=reference[len(generated)]
        if accepted<len(draft):
            cache=trim_cache(cache,len(frame["ids"])+position+accepted)
        else:
            cache=legacy_cache(core_forward(student,tensor_ids([draft[-1]],args.device),cache).past_key_values)
        generated.append(correction);last=correction
        assert cache[0][0].shape[-2]==len(frame["ids"])+len(generated)-1
    assert generated==reference
    return dict(cursor=frame["cursor"],accepted=sum(b["accepted"] for b in blocks),
                proposed=sum(b["proposed"] for b in blocks),blocks=blocks,generated_ids=generated,seconds=0.0)


@torch.no_grad()
def evaluate(student,frames,controller,args,path=None):
    rows=[]
    started=time.perf_counter()
    for frame in frames:
        row=replay(student,frame,controller,args);rows.append(row)
        if path is not None:
            append(path,row)
    score=aggregate(rows);score.pop("total_seconds")
    score["prototype_student_seconds"]=time.perf_counter()-started
    return score,rows


@torch.no_grad()
def evaluate_window(student,frames,controller,args):
    hits=tokens=0;kl=0.0
    for frame in frames:
        logits,_,_=upper_forward(student,frame,controller)
        labels=torch.as_tensor(frame["reference"],device=student.device)
        n=labels.numel();tokens+=n;hits+=int((logits.argmax(-1)==labels).sum())
        kl+=float(distribution_kl(logits,frame["teacher_logits"],args.vocab_size))*n
    return dict(tokens=tokens,hits=hits,match=hits/tokens,output_kl=kl/tokens)


def validate(student,frame,controller,args,network=None):
    """Compare cached frozen lower layers to a full model, including gradients."""
    controller.projection(None)
    ids=frame["ids"][:-1]+frame["inputs"]
    start=len(frame["ids"])-1
    with torch.no_grad():
        base_out=core_forward(student,tensor_ids(ids,args.device))
        base=student.lm_head(base_out.last_hidden_state[:,start:])[0].float()
        cached,_,_=upper_forward(student,frame,controller)
        torch.testing.assert_close(base,cached,rtol=0,atol=0)
        original_prefix=prefix_cache(student,frame["ids"],args.device)
    own_network=network is None
    if own_network:
        network=ProjectionUpdates(student,controller.layers,"qkv")
    controller.projection(network)
    result=dict(frozen_lower_full_model_max_logit_error=float((base-cached).abs().max()))
    if own_network:
        with torch.no_grad():
            zero,_,_=upper_forward(student,frame,controller)
            torch.testing.assert_close(base,zero,rtol=0,atol=0)
            result["zero_updates_max_logit_error"]=float((base-zero).abs().max())
            for p in network.parameters():
                p.normal_(0,1e-4)
    fast,attn,values=upper_forward(student,frame,controller,capture=True)
    loss=distribution_kl(fast,frame["teacher_logits"],args.vocab_size)
    loss+=0.05*relation_kl(attn,frame["attention"])+0.05*relation_kl(values,frame["value_relation"])
    fast_grads=torch.autograd.grad(loss,tuple(network.parameters()))
    controller.capture=True;controller.query_start=start
    controller.attention_rows={};controller.value_rows={}
    out=core_forward(student,tensor_ids(ids,args.device))
    full=student.lm_head(out.last_hidden_state[:,start:])[0].float()
    full_attn=torch.stack(list(controller.attention_rows.values()))
    full_values=torch.stack(list(controller.value_rows.values()))
    full_loss=distribution_kl(full,frame["teacher_logits"],args.vocab_size)
    full_loss+=0.05*relation_kl(full_attn,frame["attention"])+0.05*relation_kl(full_values,frame["value_relation"])
    full_grads=torch.autograd.grad(full_loss,tuple(network.parameters()))
    torch.testing.assert_close(fast,full,rtol=0,atol=0)
    for a,b in zip(fast_grads,full_grads):
        torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-6)
    result["nonzero_fast_full_max_logit_error"]=float((fast-full).abs().max())
    result["nonzero_fast_full_max_gradient_error"]=max(float((a-b).abs().max()) for a,b in zip(fast_grads,full_grads))
    result["gradient_norms"]={name:float(grad.norm()) for (name,_),grad in zip(network.named_parameters(),fast_grads)}
    assert all(v>0 for v in result["gradient_norms"].values())
    controller.capture=False
    with torch.no_grad():
        changed=prefix_cache(student,frame["ids"],args.device)
        first=controller.layers[0]
        result["first_updated_layer_prompt_key_change"]=float((changed[first][0]-original_prefix[first][0]).abs().max())
        result["first_updated_layer_prompt_value_change"]=float((changed[first][1]-original_prefix[first][1]).abs().max())
        if own_network:
            assert result["first_updated_layer_prompt_key_change"]>0 and result["first_updated_layer_prompt_value_change"]>0
    controller.projection(None)
    return result


def train(method,student,frames,controller,args,output,baseline):
    projections,objective=method.split("_")
    torch.manual_seed(args.seed)
    network=ProjectionUpdates(student,controller.layers,projections)
    controller.projection(network)
    optimizer=torch.optim.Adam(network.parameters(),lr=args.q_lr)
    best_score,best_state,best_epoch=baseline,None,0
    history=[]
    for epoch in range(1,args.epochs+1):
        ordering=list(range(len(frames["train"])));random.Random(args.seed+epoch).shuffle(ordering)
        started=time.perf_counter();totals=dict(output_kl=0.0,attention_kl=0.0,value_kl=0.0);tokens=0
        for step,index in enumerate(ordering):
            frame=frames["train"][index]
            optimizer.zero_grad(set_to_none=True)
            logits,attention,values=upper_forward(student,frame,controller,capture=objective=="relations")
            output_loss=distribution_kl(logits,frame["teacher_logits"],args.vocab_size)
            attention_loss=relation_kl(attention,frame["attention"]) if objective=="relations" else logits.new_zeros(())
            value_loss=relation_kl(values,frame["value_relation"]) if objective=="relations" else logits.new_zeros(())
            loss=output_loss+args.relation_weights["attention"]*attention_loss+args.relation_weights["value"]*value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step()
            count=len(frame["reference"]);tokens+=count
            for name,value in zip(totals,(output_loss,attention_loss,value_loss)):
                totals[name]+=float(value.detach())*count
            if (step+1)%16==0 or step+1==len(ordering):
                print(f"train {method} epoch={epoch} {step+1}/{len(ordering)} KL={totals['output_kl']/tokens:.4f} seconds={time.perf_counter()-started:.1f}",flush=True)
        network.eval()
        score,_=evaluate(student,frames["calibration"],controller,args,output/f"cal_{method}_epoch{epoch}.jsonl")
        window=evaluate_window(student,frames["calibration"],controller,args)
        score["fixed_context"]=window
        row=dict(method=method,epoch=epoch,train={k:v/tokens for k,v in totals.items()},calibration=score,
                 seconds=time.perf_counter()-started)
        history.append(row);append(output/"training.jsonl",row)
        print("epoch_result "+json.dumps(row),flush=True)
        key=lambda s:(s["accept_rate"],-s["blocks"],-s["fixed_context"]["output_kl"])
        if key(score)>key(best_score):
            best_score,best_epoch=score,epoch
            best_state={k:v.detach().cpu().clone() for k,v in network.state_dict().items()}
    endpoint={k:v.detach().cpu().clone() for k,v in network.state_dict().items()}
    torch.save(dict(method=method,state=endpoint,epoch=args.epochs),output/f"{method}_final.pt")
    checkpoint=dict(method=method,state=best_state,selected_epoch=best_epoch,calibration=best_score,
                    parameter_count=sum(p.numel() for p in network.parameters()))
    torch.save(checkpoint,output/f"{method}.pt")
    if method=="qkv_relations":
        result=validate(student,frames["calibration"][0],controller,args,network)
        write_json(output/"trained_validation.json",result)
    controller.projection(None)
    return {k:v for k,v in checkpoint.items() if k!="state"}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepared",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--epochs",type=int,default=3)
    parser.add_argument("--methods",nargs="+",default=[p+"_"+s for p in ("q","qk","qv","qkv") for s in ("output","relations")])
    parser.add_argument("--validate-only",action="store_true")
    cli=parser.parse_args()
    prepared,output=Path(cli.prepared),Path(cli.output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/"summary.json").exists():
        raise FileExistsError("Training experiment complete")
    config=json.loads((prepared/"config.json").read_text());config.update(vars(cli))
    config["source_sha256"]={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in ("run_qkv.py","qkv_methods.py")}
    args=argparse.Namespace(**config)
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    write_json(output/"config.json",config)
    _,student=load_model_and_tokenizer(args.student,torch.bfloat16,args.device)
    student.requires_grad_(False)
    layers=list(range(student.config.num_hidden_layers-4,student.config.num_hidden_layers))
    controller=AttentionController(student,layers,28);controller.install()
    manifest=json.loads((prepared/"manifest.json").read_text())
    frames={split:[torch.load(r["path"],map_location="cpu",weights_only=False) for r in manifest if r["split"]==split]
            for split in ("train","calibration","test")}
    validation=validate(student,frames["train"][0],controller,args)
    write_json(output/"validation.json",validation)
    print("validation "+json.dumps(validation),flush=True)
    if args.validate_only:
        controller.restore();return
    controller.projection(None)
    baseline,_=evaluate(student,frames["calibration"],controller,args,output/"cal_baseline.jsonl")
    baseline["fixed_context"]=evaluate_window(student,frames["calibration"],controller,args)
    print("baseline_calibration "+json.dumps(baseline),flush=True)
    selections={}
    for method in args.methods:
        checkpoint_path=output/f"{method}.pt"
        if checkpoint_path.exists():
            saved=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
            selections[method]={k:v for k,v in saved.items() if k!="state"}
        else:
            selections[method]=train(method,student,frames,controller,args,output,baseline)
        write_json(output/"selection.json",selections)
    summaries,records,windows={},{},{}
    checks=dict(all_outputs_match_frozen_reference=True,full_prefix_recomputed_for_every_projection_set=True,
                test_labels_not_used_for_checkpoint_selection=True,checkpoint_first_question_replay={})
    for method in ["baseline"]+args.methods:
        network=None
        if method!="baseline":
            checkpoint=torch.load(output/f"{method}.pt",map_location="cpu",weights_only=False)
            if checkpoint["state"] is not None:
                network=ProjectionUpdates(student,layers,method.split("_")[0])
                network.load_state_dict(checkpoint["state"])
        controller.projection(network)
        score,rows=evaluate(student,frames["test"],controller,args,output/f"test_{method}.jsonl")
        windows[method]=evaluate_window(student,frames["test"],controller,args)
        if method!="baseline":
            score["delta_accept_rate"]=score["accept_rate"]-summaries["baseline"]["accept_rate"]
            score["paired_95pct_delta_interval"]=paired_interval(rows,records["baseline"],args.seed)
            score["selected_epoch"]=selections[method]["selected_epoch"]
            score["parameter_count"]=selections[method]["parameter_count"]
        repeated=replay(student,frames["test"][0],controller,args)
        assert all(repeated[k]==rows[0][k] for k in ("cursor","proposed","accepted","blocks","generated_ids"))
        checks["checkpoint_first_question_replay"][method]=True
        summaries[method],records[method]=score,rows
        print("test_result "+method+" "+json.dumps(score),flush=True)
        write_json(output/"summary.partial.json",dict(methods=summaries,selection=selections))
    comparisons={}
    pairs=[(p+"_relations",p+"_output") for p in ("q","qk","qv","qkv")]
    pairs += [(p+"_"+s,"q_"+s) for p in ("qk","qv","qkv") for s in ("output","relations")]
    pairs += [("qv_"+s,"qk_"+s) for s in ("output","relations")]
    for a,b in pairs:
        if a in records and b in records:
            comparisons[a+"_vs_"+b]=dict(delta_accept_rate=summaries[a]["accept_rate"]-summaries[b]["accept_rate"],
                paired_95pct_delta_interval=paired_interval(records[a],records[b],args.seed))
    if args.max_new_tokens==32 and args.test_count==50:
        old=[json.loads(s) for s in (Path(__file__).with_name("out_fixed_teacher_audit")/"test_baseline.jsonl").read_text().splitlines()]
        for a,b in zip(records["baseline"],old):
            assert all(a[k]==b[k] for k in ("cursor","proposed","accepted","blocks","generated_ids"))
        checks["previous_baseline_exact"]=True
    controller.restore()
    write_json(output/"verification.json",checks)
    write_json(output/"summary.json",dict(config=config,methods=summaries,selection=selections,fixed_context=windows,
                                          paired_controls=comparisons,verification=checks,
                                          peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9))
    (output/"summary.partial.json").unlink(missing_ok=True)


if __name__=="__main__":
    main()
