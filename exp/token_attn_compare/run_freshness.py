#!/usr/bin/env python3
"""Offline fresh-attention oracle with real draft prefixes and cache rollback."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from run_compare import (aggregate, append, core_forward, legacy_cache, load_model_and_tokenizer,
                         logits_of, paired_interval, tensor_ids, trim_cache, write_json)
from qkv_methods import AttentionController, move_cache


@torch.no_grad()
def replay_prior(teacher, student, frame, controller, args, route):
    strength,source,scope=route["strength"],route["source"],route["scope"]
    controller.configure_prior(strength,frame["docs"],scope)
    dynamic=source=="dynamic" and strength!=0
    sc=move_cache(frame["student_prefix_cache"],args.device)
    tc=move_cache(frame["teacher_prefix_cache"],args.device) if dynamic else None
    static=frame["attention"][:,0,:len(frame["ids"])].to(args.device)
    docs=torch.as_tensor(frame["docs"],device=args.device)
    generator=torch.Generator(device="cpu").manual_seed(args.seed+frame["cursor"])
    permutation=torch.randperm(len(frame["docs"]),generator=generator).to(args.device)
    calls=0

    def step(token):
        nonlocal sc,tc,calls
        if dynamic:
            out=core_forward(teacher,tensor_ids([token],args.device),tc,attentions=True)
            tc=legacy_cache(out.past_key_values)
            rows=torch.stack([out.attentions[i][0,:,-1,:].float().mean(0) for i in controller.layers])
            calls+=1
        else:
            rows=static
        if route.get("shuffle",False):
            rows=rows.clone()
            rows[:,docs]=rows[:,docs[permutation]]
        controller.prior_rows=rows
        out=core_forward(student,tensor_ids([token],args.device),sc)
        sc=legacy_cache(out.past_key_values)
        return int(logits_of(student,out).argmax(-1))

    generated,blocks,last=[],[],frame["ids"][-1]
    reference=frame["reference"]
    while len(generated)<len(reference):
        position=len(generated)
        k=min(args.draft_k,args.max_new_tokens-position)
        draft=[]
        token=last
        for _ in range(k):
            token=step(token)
            draft.append(token)
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
            length=len(frame["ids"])+position+accepted
            sc=trim_cache(sc,length)
            if dynamic:
                tc=trim_cache(tc,length)
        else:
            step(draft[-1])
        generated.append(correction)
        last=correction
        expected=len(frame["ids"])+len(generated)-1
        assert sc[0][0].shape[-2]==expected
        if dynamic:
            assert tc[0][0].shape[-2]==expected
    assert generated==reference
    return dict(cursor=frame["cursor"],accepted=sum(b["accepted"] for b in blocks),
                proposed=sum(b["proposed"] for b in blocks),blocks=blocks,generated_ids=generated,
                teacher_attention_calls=calls,seconds=0.0)


def select(candidates,source,scope):
    rows=[r for r in candidates if r["route"]["source"]==source and r["route"]["scope"]==scope]
    return max(rows,key=lambda r:(r["score"]["accept_rate"],-r["score"]["blocks"],-r["route"]["strength"]))


@torch.no_grad()
def validate_fresh_queries(teacher,student,frame,controller,args):
    tc=move_cache(frame["teacher_prefix_cache"],args.device)
    max_error=0.0
    first_rows=None
    for position,token in enumerate(frame["inputs"][:5]):
        out=core_forward(teacher,tensor_ids([token],args.device),tc,attentions=True)
        tc=legacy_cache(out.past_key_values)
        rows=torch.stack([out.attentions[i][0,:,-1,:].float().mean(0) for i in controller.layers])
        stored=frame["attention"][:,position,:rows.shape[-1]].to(args.device)
        torch.testing.assert_close(rows,stored,rtol=0,atol=0)
        max_error=max(max_error,float((rows-stored).abs().max()))
        if position==0:
            first_rows=rows
    cache=move_cache(frame["student_prefix_cache"],args.device)
    logits=[]
    controller.configure_prior(0.5,frame["docs"])
    for rows in (first_rows,frame["attention"][:,0,:len(frame["ids"])]):
        controller.prior_rows=rows.to(args.device)
        logits.append(logits_of(student,core_forward(student,tensor_ids(frame["ids"][-1:],args.device),cache)))
    torch.testing.assert_close(logits[0],logits[1],rtol=0,atol=0)
    controller.projection(None)
    return dict(current_query_rows_equal_prepared_teacher_rows=True,
                cached_fresh_attention_max_error=max_error,
                first_query_static_dynamic_max_logit_error=float((logits[0]-logits[1]).abs().max()),
                teacher_queries_checked=min(5,len(frame["inputs"])))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepared",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--strengths",type=float,nargs="+",default=[0,0.125,0.25,0.5,1])
    parser.add_argument("--resume",action="store_true")
    cli=parser.parse_args()
    prepared,output=Path(cli.prepared),Path(cli.output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/"summary.json").exists():
        raise FileExistsError("Freshness experiment complete")
    config=json.loads((prepared/"config.json").read_text())
    config.update(vars(cli))
    config["source_sha256"]={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in ("run_freshness.py","qkv_methods.py")}
    if (output/"config.json").exists():
        if not cli.resume:
            raise FileExistsError("Existing run: use --resume to reuse completed records")
        previous=json.loads((output/"config.json").read_text())
        for key in config:
            if key not in ("source_sha256","resume","recovery_history"):
                assert previous[key]==config[key],("Resume configuration changed",key)
        assert previous["source_sha256"]["qkv_methods.py"]==config["source_sha256"]["qkv_methods.py"]
        config["recovery_history"]=previous.get("recovery_history",[])+[
            dict(previous_source_sha256=previous["source_sha256"],reason="reuse completed calibration and test records")]
    args=argparse.Namespace(**config)
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    write_json(output/"config.json",config)
    _,teacher=load_model_and_tokenizer(args.teacher,torch.bfloat16,args.device)
    _,student=load_model_and_tokenizer(args.student,torch.bfloat16,args.device)
    teacher.requires_grad_(False);student.requires_grad_(False)
    layers=list(range(student.config.num_hidden_layers-4,student.config.num_hidden_layers))
    controller=AttentionController(student,layers,teacher.config.num_attention_heads);controller.install()
    manifest=json.loads((prepared/"manifest.json").read_text())
    frames={split:[torch.load(r["path"],map_location="cpu",weights_only=False) for r in manifest if r["split"]==split]
            for split in ("calibration","test")}
    validation=validate_fresh_queries(teacher,student,frames["calibration"][0],controller,args)
    write_json(output/"validation.json",validation)
    print("validation "+json.dumps(validation),flush=True)
    memo={}

    def evaluate(split,route):
        key=(split,route["source"],route["scope"],route["strength"],route.get("shuffle",False))
        if route["strength"]==0:
            key=(split,"zero")
        if key not in memo:
            started=time.perf_counter()
            rows=[]
            for i,frame in enumerate(frames[split]):
                rows.append(replay_prior(teacher,student,frame,controller,args,route))
                if (i+1)%10==0:
                    print(f"{split} {json.dumps(route)} {i+1}/{len(frames[split])}",flush=True)
            score=aggregate(rows);score.pop("total_seconds")
            score["teacher_attention_calls"]=sum(r["teacher_attention_calls"] for r in rows)
            score["prototype_seconds"]=time.perf_counter()-started
            memo[key]=(score,rows)
        return memo[key]

    candidates=[]
    saved_calibration={}
    if cli.resume and (output/"calibration.jsonl").exists():
        for line in (output/"calibration.jsonl").read_text().splitlines():
            row=json.loads(line)
            key=tuple(row["route"][k] for k in ("source","scope","strength"))
            assert key not in saved_calibration,("Duplicate calibration route",key)
            saved_calibration[key]=row
    for source,scope in (("static","document"),("dynamic","document"),("dynamic","full")):
        for strength in args.strengths:
            route=dict(source=source,scope=scope,strength=strength)
            if (source,scope,strength) in saved_calibration:
                candidates.append(saved_calibration[(source,scope,strength)])
                continue
            score,_=evaluate("calibration",route)
            row=dict(route=route,score=score);candidates.append(row)
            append(output/"calibration.jsonl",row)
            print("calibration_result "+json.dumps(row),flush=True)
    static=select(candidates,"static","document")["route"]
    fresh=select(candidates,"dynamic","document")["route"]
    full=select(candidates,"dynamic","full")["route"]
    choices={"baseline":dict(source="static",scope="document",strength=0),
             "static_selected":static,"dynamic_selected":fresh,
             "static_at_dynamic_strength":dict(fresh,source="static"),
             "dynamic_at_static_strength":dict(static,source="dynamic"),
             "dynamic_shuffled_selected":dict(fresh,shuffle=True),
             "static_unit":dict(static,strength=1),"dynamic_unit":dict(fresh,strength=1),
             "dynamic_shuffled_unit":dict(fresh,strength=1,shuffle=True),
             "dynamic_full_selected":full,"dynamic_full_unit":dict(full,strength=1),
             "dynamic_full_shuffled_selected":dict(full,shuffle=True)}
    if cli.resume and (output/"selection.json").exists():
        assert choices==json.loads((output/"selection.json").read_text()),"Calibration selection changed"
    write_json(output/"selection.json",choices)
    records,summary={},{}
    saved_summary={}
    if cli.resume and (output/"summary.partial.json").exists():
        saved_summary=json.loads((output/"summary.partial.json").read_text())["methods"]
    for name,route in choices.items():
        path=output/f"test_{name}.jsonl"
        if cli.resume and name in saved_summary:
            rows=[json.loads(s) for s in path.read_text().splitlines()]
            assert [r["cursor"] for r in rows]==[f["cursor"] for f in frames["test"]]
            assert all(r["generated_ids"]==f["reference"] for r,f in zip(rows,frames["test"]))
            score=saved_summary[name]
            for key,value in aggregate(rows).items():
                if key!="total_seconds":
                    assert score[key]==value,(name,key)
            print("resumed_test "+name,flush=True)
        else:
            score,rows=evaluate("test",route);score=dict(score)
            # Replace an incomplete file left before its summary was committed.
            path.write_text("".join(json.dumps(row)+"\n" for row in rows))
        if name!="baseline":
            score["delta_accept_rate"]=score["accept_rate"]-summary["baseline"]["accept_rate"]
            score["paired_95pct_delta_interval"]=paired_interval(rows,records["baseline"],args.seed)
        records[name],summary[name]=rows,score
        print("test_result "+name+" "+json.dumps(score),flush=True)
        write_json(output/"summary.partial.json",dict(methods=summary,selection=choices))
    pairs=[("dynamic_selected","static_at_dynamic_strength"),("dynamic_at_static_strength","static_selected"),
           ("dynamic_unit","static_unit"),("dynamic_selected","dynamic_shuffled_selected"),
           ("dynamic_unit","dynamic_shuffled_unit"),("dynamic_full_selected","dynamic_full_shuffled_selected")]
    comparisons={a+"_vs_"+b:dict(delta_accept_rate=summary[a]["accept_rate"]-summary[b]["accept_rate"],
                                paired_95pct_delta_interval=paired_interval(records[a],records[b],args.seed)) for a,b in pairs}
    verification=dict(all_outputs_match_frozen_reference=True,teacher_and_student_rollback_lengths_checked=True,
                      same_strength_freshness_comparisons=True,teacher_lookahead_labels_not_used=True)
    if args.max_new_tokens==32 and args.test_count==50:
        old=[json.loads(s) for s in (Path(__file__).with_name("out_fixed_teacher_audit")/"test_baseline.jsonl").read_text().splitlines()]
        for a,b in zip(records["baseline"],old):
            assert all(a[k]==b[k] for k in ("cursor","proposed","accepted","blocks","generated_ids"))
        verification["previous_baseline_exact"]=True
    controller.restore()
    write_json(output/"summary.json",dict(config=config,selection=choices,methods=summary,paired_controls=comparisons,
                                          verification=verification,evaluation="offline fresh teacher oracle; no latency claim"))
    (output/"summary.partial.json").unlink(missing_ok=True)


if __name__=="__main__":
    main()
