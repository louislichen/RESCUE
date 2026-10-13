#!/usr/bin/env python3
"""Freeze teacher trajectories, soft labels, current attention/value relations."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from run_compare import (append, core_forward, encoded_sample, legacy_cache,
    load_model_and_tokenizer, logits_of, prefix_cache, read_jsonl, tensor_ids, write_json)
from qkv_methods import move_cache


@torch.no_grad()
def collect(teacher, student, item, tokenizer, layers, args, fixed, keep_cache):
    tc0 = prefix_cache(teacher, item["ids"], args.device)
    tc = tc0
    inputs, reference, attention, values, logits = [], [], [], [], []
    token = item["ids"][-1]
    for position in range(args.max_new_tokens):
        inputs.append(token)
        out = core_forward(teacher, tensor_ids([token], args.device), tc, attentions=True)
        scores = logits_of(teacher, out)[0]
        token = int(scores.argmax())
        if fixed is not None and (position >= len(fixed) or token != fixed[position]):
            raise RuntimeError(f"Frozen teacher reference differs at {item['cursor']}/{position}")
        if token >= len(tokenizer):
            raise RuntimeError("Teacher generated a padded vocabulary entry")
        logits.append(scores[:len(tokenizer)].to(torch.bfloat16).cpu())
        reference.append(token)
        tc = legacy_cache(out.past_key_values)
        attention.append(torch.stack([out.attentions[i][0, :, -1, :].float().mean(0) for i in layers]).cpu())
        relations = []
        for i in layers:
            v = tc[i][1][0].float()
            relations.append(((v[:, -1:, :]@v.transpose(1, 2))/math.sqrt(v.shape[-1])).softmax(-1).mean(0)[0])
        values.append(torch.stack(relations).cpu())
        if token == tokenizer.eos_token_id:
            break
    full_ids = item["ids"][:-1]+inputs
    full = student.model(input_ids=tensor_ids(full_ids, args.device), use_cache=False,
                         output_hidden_states=True, return_dict=True)
    hidden = full.hidden_states[layers[0]].detach().cpu()
    base_ids = student.lm_head(full.last_hidden_state[:, len(item["ids"])-1:, :])[0].argmax(-1).cpu()
    total = len(full_ids)
    frame = dict(item, inputs=inputs, reference=reference, lower_hidden=hidden,
                 teacher_logits=torch.stack(logits), baseline_window_ids=base_ids,
                 attention=torch.stack([F.pad(r,(0,total-r.shape[-1])) for r in attention],dim=1),
                 value_relation=torch.stack([F.pad(r,(0,total-r.shape[-1])) for r in values],dim=1))
    if keep_cache:
        frame["teacher_prefix_cache"] = move_cache(tc0,"cpu")
        frame["student_prefix_cache"] = move_cache(prefix_cache(student,item["ids"],args.device),"cpu")
    return frame


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--cache-dir",required=True)
    parser.add_argument("--train-count",type=int,default=128)
    parser.add_argument("--cal-count",type=int,default=32)
    parser.add_argument("--test-count",type=int,default=50)
    parser.add_argument("--max-new-tokens",type=int,default=32)
    cli=parser.parse_args()
    previous=Path(__file__).resolve().parent
    config=json.loads((previous/"out_native_q128_e3/config.json").read_text())
    config.update(vars(cli))
    config.update(train_tokens=cli.max_new_tokens,output_loss="KL(teacher||student), shared tokenizer vocabulary, T=1",
                  relation_weights=dict(attention=0.05,value=0.05),prefix_scope="all prompt and generation tokens",
                  train_policy="single-query teacher greedy trajectories",projection_sets=["q","qk","qv","qkv"],
                  supervision=["output","relations"])
    args=argparse.Namespace(**config)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    out,cache=Path(args.output),Path(args.cache_dir)
    out.mkdir(parents=True,exist_ok=True)
    cache.mkdir(parents=True,exist_ok=True)
    if (out/"prepared.json").exists():
        raise FileExistsError("Preparation already complete")
    config["prepare_source_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if hashlib.sha256(Path(args.samples).read_bytes()).hexdigest()!=config["data_sha256"]:
        raise RuntimeError("Frozen dataset changed")
    teacher_tokenizer,teacher=load_model_and_tokenizer(args.teacher,torch.bfloat16,args.device)
    tokenizer,student=load_model_and_tokenizer(args.student,torch.bfloat16,args.device)
    teacher.requires_grad_(False); student.requires_grad_(False)
    config["vocab_size"],config["eos_id"]=len(tokenizer),tokenizer.eos_token_id
    write_json(out/"config.json",config)
    layers=list(range(student.config.num_hidden_layers-4,student.config.num_hidden_layers))
    samples=list(read_jsonl(args.samples))
    fixed={r["cursor"]:r["generated_ids"] for r in read_jsonl(str(previous/"out_fixed_teacher_audit/teacher_reference.jsonl"))}
    manifest=[]
    for split,start,count in (("train",0,args.train_count),("calibration",args.cal_start,args.cal_count),("test",args.test_start,args.test_count)):
        started=time.perf_counter()
        for index in range(count):
            cursor=start+index
            path=cache/f"{split}_{cursor}.pt"
            if path.exists():
                frame=torch.load(path,map_location="cpu",weights_only=False)
            else:
                item=encoded_sample(samples[cursor],cursor,teacher_tokenizer,tokenizer,args)
                frame=collect(teacher,student,item,tokenizer,layers,args,
                              fixed[cursor][:args.max_new_tokens] if split=="test" else None,split!="train")
                temporary=path.with_suffix(".tmp")
                torch.save(frame,temporary); temporary.replace(path)
            manifest.append(dict(split=split,cursor=cursor,path=str(path),prompt_tokens=len(frame["ids"]),
                                 labels=len(frame["reference"]),reference=frame["reference"]))
            if (index+1)%4==0 or index+1==count:
                print(f"prepare {split} {index+1}/{count} seconds={time.perf_counter()-started:.1f}",flush=True)
    write_json(out/"manifest.json",manifest)
    write_json(out/"prepared.json",dict(frames=len(manifest),tokens={s:sum(r["labels"] for r in manifest if r["split"]==s)
                                                                 for s in ("train","calibration","test")},
                                           all_test_references_reused=True,peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9))


if __name__=="__main__":
    main()
