#!/usr/bin/env python3
"""Audit saved experiment artifacts without loading models or rerunning training."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepared",required=True)
    parser.add_argument("--freshness",required=True)
    parser.add_argument("--qkv",required=True)
    args=parser.parse_args()
    prepared=Path(args.prepared)
    config=read(prepared/"config.json")
    manifest=read(prepared/"manifest.json")
    expected={s:list(range(start,start+count)) for s,start,count in
              (("train",0,config["train_count"]),("calibration",config["cal_start"],config["cal_count"]),
               ("test",config["test_start"],config["test_count"]))}
    for split,indices in expected.items():
        assert [r["cursor"] for r in manifest if r["split"]==split]==indices
    for a,b in (("train","calibration"),("train","test"),("calibration","test")):
        assert not set(expected[a])&set(expected[b])
    source_samples=rows(Path(config["samples"]))
    prompt_hashes={}
    question_sets={}
    for split,indices in expected.items():
        prompt_hashes[split]={hashlib.sha256(source_samples[i]["prompt"].encode()).hexdigest() for i in indices}
        question_sets[split]={source_samples[i].get("question","") for i in indices}-{ "" }
        assert len(prompt_hashes[split])==len(indices)
    for a,b in (("train","calibration"),("train","test"),("calibration","test")):
        assert not prompt_hashes[a]&prompt_hashes[b]
        assert not question_sets[a]&question_sets[b]
    refs={r["cursor"]:r["reference"] for r in manifest if r["split"]=="test"}
    source=Path(__file__).resolve().parent
    assert hashlib.sha256(Path(config["samples"]).read_bytes()).hexdigest()==config["data_sha256"]
    assert hashlib.sha256((source/"prepare_full_experiment.py").read_bytes()).hexdigest()==config["prepare_source_sha256"]
    result=dict(split_cursors_disjoint=True,prompts_and_nonempty_questions_disjoint=True,
                data_and_prepare_source_sha256_match=True,experiments={})
    for experiment,path in (("freshness",Path(args.freshness)),("qkv",Path(args.qkv))):
        summary=read(path/"summary.json")
        for name,digest in summary["config"]["source_sha256"].items():
            assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,(experiment,name)
        if experiment=="freshness":
            calibration=rows(path/"calibration.jsonl")
            expected_routes={(source,scope,strength) for source,scope in
                             (("static","document"),("dynamic","document"),("dynamic","full"))
                             for strength in summary["config"]["strengths"]}
            assert len(calibration)==len(expected_routes)
            assert {tuple(r["route"][k] for k in ("source","scope","strength")) for r in calibration}==expected_routes
            for name,source_name,scope in (("static_selected","static","document"),
                                           ("dynamic_selected","dynamic","document"),
                                           ("dynamic_full_selected","dynamic","full")):
                eligible=[r for r in calibration if r["route"]["source"]==source_name and r["route"]["scope"]==scope]
                best=max(eligible,key=lambda r:(r["score"]["accept_rate"],-r["score"]["blocks"],-r["route"]["strength"]))
                assert best["route"]==summary["selection"][name]
            assert len(summary["methods"])==12
        else:
            import torch
            torch.set_num_threads(4)
            methods=[p+"_"+s for p in ("q","qk","qv","qkv") for s in ("output","relations")]
            assert set(summary["methods"])=={"baseline",*methods}
            history=rows(path/"training.jsonl")
            assert len(history)==len(methods)*summary["config"]["epochs"]
            base_records=rows(path/"cal_baseline.jsonl")
            assert [r["cursor"] for r in base_records]==expected["calibration"]
            base_accept=sum(r["accepted"] for r in base_records)/sum(r["proposed"] for r in base_records)
            base_blocks=sum(len(r["blocks"]) for r in base_records)
            for method in methods:
                epochs=[r for r in history if r["method"]==method]
                assert [r["epoch"] for r in epochs]==list(range(1,summary["config"]["epochs"]+1))
                for epoch in epochs:
                    records=rows(path/f"cal_{method}_epoch{epoch['epoch']}.jsonl")
                    assert [r["cursor"] for r in records]==expected["calibration"]
                    assert sum(r["accepted"] for r in records)==epoch["calibration"]["accepted"]
                    assert sum(r["proposed"] for r in records)==epoch["calibration"]["proposed"]
                selected=summary["selection"][method]
                assert (selected["calibration"]["accept_rate"],-selected["calibration"]["blocks"])>=(base_accept,-base_blocks)
                candidates=[(r["epoch"],r["calibration"]) for r in epochs]
                # Epoch zero remains eligible; its score is equal across all routes.
                if selected["selected_epoch"]==0:
                    candidates.append((0,selected["calibration"]))
                best=max(candidates,key=lambda pair:(pair[1]["accept_rate"],-pair[1]["blocks"],
                                                     -pair[1]["fixed_context"]["output_kl"],-pair[0]))
                assert best[0]==selected["selected_epoch"]
                checkpoint=torch.load(path/f"{method}.pt",map_location="cpu",weights_only=False)
                endpoint=torch.load(path/f"{method}_final.pt",map_location="cpu",weights_only=False)
                assert checkpoint["method"]==endpoint["method"]==method
                assert checkpoint["selected_epoch"]==selected["selected_epoch"]
                assert endpoint["epoch"]==summary["config"]["epochs"]
                assert set(endpoint["state"])=={"weights."+p for p in method.split("_")[0]}
                assert sum(t.numel() for t in endpoint["state"].values())==selected["parameter_count"]
                assert all(torch.isfinite(t).all() and torch.count_nonzero(t)>0 for t in endpoint["state"].values())
                if selected["selected_epoch"]==0:
                    assert checkpoint["state"] is None
                else:
                    assert checkpoint["state"] is not None
                    assert all(torch.isfinite(t).all() for t in checkpoint["state"].values())
                if selected["selected_epoch"]==endpoint["epoch"]:
                    assert all(torch.equal(t,endpoint["state"][k]) for k,t in checkpoint["state"].items())
            for filename in ("validation.json","trained_validation.json"):
                validation=read(path/filename)
                assert validation["frozen_lower_full_model_max_logit_error"]==0
                assert validation["nonzero_fast_full_max_logit_error"]==0
                assert validation["nonzero_fast_full_max_gradient_error"]<1e-6
                assert validation["first_updated_layer_prompt_key_change"]>0
                assert validation["first_updated_layer_prompt_value_change"]>0
        all_records={}
        for method,score in summary["methods"].items():
            records=rows(path/f"test_{method}.jsonl")
            assert [r["cursor"] for r in records]==expected["test"]
            assert all(r["generated_ids"]==refs[r["cursor"]] for r in records)
            for record in records:
                assert sum(b["accepted"] for b in record["blocks"])==record["accepted"]
                assert sum(b["proposed"] for b in record["blocks"])==record["proposed"]
                assert all(0<=b["accepted"]<=b["proposed"]<=config["draft_k"] for b in record["blocks"])
            accepted=sum(r["accepted"] for r in records);proposed=sum(r["proposed"] for r in records)
            assert accepted==score["accepted"] and proposed==score["proposed"]
            assert abs(accepted/proposed-score["accept_rate"])<1e-15
            assert sum(len(r["blocks"]) for r in records)==score["blocks"]
            if experiment=="freshness":
                route=summary["selection"][method]
                for record in records:
                    calls=0
                    if route["source"]=="dynamic" and route["strength"]!=0:
                        calls=record["proposed"]+sum(b["accepted"]==b["proposed"] and
                              b["position"]+b["accepted"]<len(record["generated_ids"]) for b in record["blocks"])
                    assert calls==record["teacher_attention_calls"]
            all_records[method]=records
        if config["max_new_tokens"]==32 and config["test_count"]==50:
            previous=rows(source/"out_fixed_teacher_audit/test_baseline.jsonl")
            assert len(previous)==len(all_records["baseline"])
            for a,b in zip(previous,all_records["baseline"]):
                assert all(a[k]==b[k] for k in ("cursor","proposed","accepted","blocks","generated_ids"))
        result["experiments"][experiment]=dict(methods=len(all_records),questions_per_method=len(refs),
            source_sha256_match=True,record_aggregates_exact=True,all_outputs_match_frozen_reference=True,
            baseline_matches_previous_audit=True)
    (prepared/"artifact_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
