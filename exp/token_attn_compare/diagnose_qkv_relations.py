#!/usr/bin/env python3
"""Measure held-out token-relation fit after checkpoint selection is complete."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from run_compare import load_model_and_tokenizer, write_json
from qkv_methods import AttentionController, ProjectionUpdates, distribution_kl, relation_kl, upper_forward


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepared",required=True)
    parser.add_argument("--qkv",required=True)
    args=parser.parse_args()
    prepared,output=Path(args.prepared),Path(args.qkv)
    summary=json.loads((output/"summary.json").read_text())
    config=summary["config"]
    torch.set_num_threads(4)
    _,student=load_model_and_tokenizer(config["student"],torch.bfloat16,config["device"])
    student.requires_grad_(False)
    layers=list(range(student.config.num_hidden_layers-4,student.config.num_hidden_layers))
    controller=AttentionController(student,layers,28);controller.install()
    manifest=json.loads((prepared/"manifest.json").read_text())
    frames=[torch.load(r["path"],map_location="cpu",weights_only=False) for r in manifest if r["split"]=="test"]
    result={}
    for method in summary["methods"]:
        network=ProjectionUpdates(student,layers,"q" if method=="baseline" else method.split("_")[0])
        if method!="baseline":
            state=torch.load(output/f"{method}.pt",map_location="cpu",weights_only=False)["state"]
            if state is not None:
                network.load_state_dict(state)
        controller.projection(network)
        totals=dict(output_kl=0.0,attention_kl=0.0,value_relation_kl=0.0)
        hits=tokens=0
        per_layer=torch.zeros(2,len(layers),dtype=torch.float64)
        for frame in frames:
            logits,attention,values=upper_forward(student,frame,controller,capture=True)
            n=len(frame["reference"]);tokens+=n
            hits+=int((logits.argmax(-1)==torch.tensor(frame["reference"],device=student.device)).sum())
            totals["output_kl"]+=float(distribution_kl(logits,frame["teacher_logits"],config["vocab_size"]))*n
            totals["attention_kl"]+=float(relation_kl(attention,frame["attention"]))*n
            totals["value_relation_kl"]+=float(relation_kl(values,frame["value_relation"]))*n
            for i in range(len(layers)):
                per_layer[0,i]+=float(relation_kl(attention[i],frame["attention"][i]))*n
                per_layer[1,i]+=float(relation_kl(values[i],frame["value_relation"][i]))*n
        reference=summary["fixed_context"][method]
        assert tokens==reference["tokens"] and hits==reference["hits"]
        assert abs(totals["output_kl"]/tokens-reference["output_kl"])<1e-7
        result[method]=dict(tokens=tokens,hits=hits,**{k:v/tokens for k,v in totals.items()},
                            per_layer_attention_kl=(per_layer[0]/tokens).tolist(),
                            per_layer_value_relation_kl=(per_layer[1]/tokens).tolist())
        print(method,json.dumps(result[method]),flush=True)
    controller.restore()
    write_json(output/"relation_diagnostics.json",dict(methods=result,layers=layers,
        all_fixed_context_logits_metrics_match=True,used_for_selection=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__=="__main__":
    main()
