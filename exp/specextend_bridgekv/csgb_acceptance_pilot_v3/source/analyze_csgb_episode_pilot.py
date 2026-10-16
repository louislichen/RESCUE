"""CPU-only episode-pilot audit, document statistics and scientific figures.

Fit0..15 / internal training-monitor16..23, actual4096 Tokens only. Never
select a policy/baseline on monitor outcomes or infer full-study success.
Heavy plotting imports are delayed; pure audits/tests require only stdlib.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import inspect
import math
from pathlib import Path
import random
import statistics
import sys
import shutil

ENGINE = Path(__file__).resolve().parent/"specextend"
if str(ENGINE) not in sys.path:
    sys.path.insert(0,str(ENGINE))
from csgb_episode_policy import BUDGETS, Decision, EpisodePolicy, episode_gradient
from csgb_episode_harness import (audit_decisions, parameter_sha, rng_state,
                                  validate_feature_chronology, validate_request_accounting)
import csgb_episode_runner as runner

FIT, MONITOR, STREAMS, REPEATS = tuple(range(16)), tuple(range(16,24)), tuple(range(3)), (0,1)
MODES = ("csgb","constant","history_native")
FIXED = tuple(f"fixed{b}_native" for b in BUDGETS)
POLICIES = FIXED+MODES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    def reject(value):
        raise ValueError("Nonfinite JSON literal: "+value)
    return json.loads(Path(path).read_text(),parse_constant=reject)


def finite(value, *, positive=False):
    if type(value) not in (int,float) or not math.isfinite(value) or (positive and value<=0):
        raise ValueError("Invalid finite pilot measurement")
    return value


def close(a,b):
    return math.isclose(finite(a),finite(b),rel_tol=1e-12,abs_tol=1e-12)


def paired_interval(values):
    values=list(values)
    if not values or any(not math.isfinite(finite(v)) for v in values):
        raise ValueError("Finite document differences required")
    if len(values)<2:return None
    rng=random.Random(42)
    draws=sorted(statistics.fmean(rng.choices(values,k=len(values))) for _ in range(10000))
    def q(frac):
        p=frac*(len(draws)-1);lo=int(p);hi=math.ceil(p)
        return draws[lo]+(draws[hi]-draws[lo])*(p-lo)
    return [q(.025),q(.975)]


def audit_episode_record(record, reference, *, eos_token_id, expected_parameters=None):
    """Replay canonical arithmetic, feature chronology, parameter SHA and RNG.

    Does not claim independently reconstructed GPU attention or model-weight
    provenance. The root auditor separately binds source/input/weight hashes.
    """
    mode,phase,d=record.get("policy"),record.get("phase"),record.get("document")
    if (mode not in POLICIES and mode!="fixed8_fresh" or type(d) is not int
            or type(record.get("context")) is not int or record["context"]!=4096
            or phase not in {"calibration","fit_warmup","monitor_warmup","train","monitor","smoke_fixed","smoke_train"}
            or record.get("uncontended_at_request_boundaries") is not True
            or not isinstance(record.get("input_sha256"),str) or len(record["input_sha256"])!=64
            or any(c not in "0123456789abcdef" for c in record["input_sha256"])):
        raise ValueError("Wrong request phase/policy/context/identity")
    # Additional strict schema/type boundaries; arithmetic and stochastic
    # trajectory below are also replayed by this analyzer, not just trusted.
    runner.validate_episode_record(record,reference,eos_token_id=eos_token_id)
    e=record["episode"]
    if (e.get("phase")!=phase or e.get("policy")!=mode or e.get("parameters_unchanged") is not True
            or e.get("explicit_round_sync") is not False or e.get("round_updates") is not False
            or e.get("baseline_or_sgd_performed") is not False
            or e.get("whole_request_timing_supplied_by_caller") is not True):
        raise ValueError("Episode phase or boundary-only/no-round-learning contract changed")
    checked=validate_request_accounting(e["result"],reference,e["rows"],context=4096,seconds=e["seconds"],
                                        max_new_tokens=128,eos_token_id=eos_token_id,mode=mode)
    chronology=validate_feature_chronology(e["rows"],mode)
    ref=reference["output_token_ids"] if isinstance(reference,dict) else reference
    if e.get("reference_output_token_ids")!=ref or any(not close(e.get(k),checked[k]) for k in
            ("useful_n","seconds","request_tps","scored_acceptance")) or e.get("exact") is not True:
        raise ValueError("Saved episode metrics/canonical reference differ from independent arithmetic")
    before,after=record.get("parameters_before"),record.get("parameters_after")
    if mode in MODES:
        p=EpisodePolicy.from_state(before)
        if (p.mode!=mode or p.temperature!=1. or expected_parameters is not None and before!=expected_parameters
                or e.get("parameter_sha256")!=parameter_sha(p) or e.get("parameter_version")!=p.parameter_version
                or type(record.get("request_seed")) is not int or record["request_seed"]<0
                or e["rng_before"]!=rng_state(random.Random(record["request_seed"]))):
            raise ValueError("Unfrozen/wrong-mode policy or substituted initial action stream")
        audit_decisions(p,e["decisions"],e["rows"],e["rng_before"],e["rng_after"])
        if phase in {"train","smoke_train"}:
            rho,baseline=finite(record["rho"],positive=True),finite(record["baseline_return"])
            g=episode_gradient(p,[Decision.from_state(d) for d in e["decisions"]],
                useful_n=checked["useful_n"],elapsed_s=checked["seconds"],rho=rho,baseline=baseline)
            if (record.get("learning_rate")!=.0005 or record.get("gradient")!=[list(v) for v in g.values]
                    or record.get("episode_return")!=checked["useful_n"]-rho*checked["seconds"]
                    or after!=p.sgd(g,.0005).to_state() or finite(record["update_wall_seconds"])<0):
                raise ValueError("External episode return/summed score gradient/SGD differs")
        elif before!=after:raise ValueError("Frozen monitor/warmup changed parameters")
    elif (record.get("request_seed") is not None or before is not None or after is not None or e.get("parameter_sha256") is not None
          or e.get("parameter_version") is not None or e.get("decisions")!=[]
          or e.get("rng_before") is not None or e.get("rng_after") is not None):
        raise ValueError("Native fixed/calibration cannot hide a sampled actor")
    identity={k:record.get(k) for k in ("document","context","phase","pass_index","stream","timing_repeat","request_seed")}
    if any(any(row.get("provenance",{}).get(k)!=v for k,v in identity.items()) for row in e["rows"]):
        raise ValueError("Round phase was relabeled")
    trajectory={"output":e["result"]["output_token_ids"],"accept":e["result"]["accept_length_list"],
                "raw_accept":e["result"]["raw_accept_length_list"],
                "budgets":[r["budget"] for r in e["rows"]],"decisions":e["decisions"]}
    return {"document":d,"context":4096,"policy":mode,"phase":phase,
            "stream":record.get("stream"),"timing_repeat":record.get("timing_repeat"),
            "pass_index":record.get("pass_index"),"request_seed":record["request_seed"],
            "input_sha256":record["input_sha256"],"tps":checked["request_tps"],
            "acceptance":checked["scored_acceptance"],"seconds":checked["seconds"],"output_tokens":checked["useful_n"],
            "decision_ms":1000*statistics.fmean(r["decision_wall_seconds"] for r in e["rows"]),
            "trajectory_sha256":hashlib.sha256(json.dumps(trajectory,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
            "chronology":chronology,"rounds":copy.deepcopy(e["rows"])}


def aggregate_monitor(records, baseline_budget):
    """Median two timing repeats per stream, then equal mean of three streams.

    Incomplete whole document units are excluded explicitly, never individual
    unfavorable cells. Statistics are PARTIAL whenever any planned cell misses.
    """
    if type(baseline_budget) is not int or baseline_budget not in BUDGETS:
        raise ValueError("Need the fit-only frozen Native baseline, never monitor selection")
    expected={(d,p,s,r) for d in MONITOR for p in POLICIES for s in STREAMS for r in REPEATS}
    found={}
    for row in records:
        key=(row.get("document"),row.get("policy"),row.get("stream"),row.get("timing_repeat"))
        if (any(type(row.get(k)) is not int for k in ("document","stream","timing_repeat"))
                or key not in expected or key in found or row.get("phase")!="monitor" or row.get("context")!=4096):
            raise ValueError("Duplicate/unregistered monitor request")
        for m in ("tps","acceptance","decision_ms"):
            if finite(row[m],positive=m=="tps")<0:raise ValueError("Negative metric")
        found[key]=row
    complete_docs=[d for d in MONITOR if all((d,p,s,r) in found for p in POLICIES for s in STREAMS for r in REPEATS)]
    cells,stream_values={},{}
    for d in complete_docs:
        hashes={found[d,p,s,r]["input_sha256"] for p in POLICIES for s in STREAMS for r in REPEATS}
        if len(hashes)!=1:raise ValueError("Unpaired monitor prompt inputs")
        for p in POLICIES:
            seeds=[]
            for s in STREAMS:
                rows=[found[d,p,s,r] for r in REPEATS]
                if (rows[0]["request_seed"]!=rows[1]["request_seed"]
                        or rows[0]["trajectory_sha256"]!=rows[1]["trajectory_sha256"]):
                    raise ValueError("A timing repeat changed its action stream or discrete trajectory")
                seeds.append(rows[0]["request_seed"])
                stream_values[d,p,s]={m:statistics.median(row[m] for row in rows) for m in ("tps","acceptance","decision_ms")}
            if p in MODES and len(set(seeds))!=3:raise ValueError("Three monitor action streams must be independent")
            cells[d,p]={m:statistics.fmean(stream_values[d,p,s][m] for s in STREAMS) for m in ("tps","acceptance","decision_ms")}
    comparisons={}
    for b in BUDGETS:
        base=f"fixed{b}_native";comparisons[base]={}
        for p in POLICIES:
            estimates={}
            for m in ("acceptance","tps"):
                values=[100*(cells[d,p][m]/cells[d,base][m]-1) if m=="tps" else cells[d,p][m]-cells[d,base][m] for d in complete_docs]
                estimates[m]={"mean":None if not values else statistics.fmean(values),
                    "ci95":None if not values else paired_interval(values),"documents":len(values),
                    "per_document":dict(zip(map(str,complete_docs),values))}
            comparisons[base][p]=estimates
    variability={p:{str(d):{m:{"streams":[stream_values[d,p,s][m] for s in STREAMS],
         "sample_sd":statistics.stdev(stream_values[d,p,s][m] for s in STREAMS),
         "mcse_three_stream_mean":statistics.stdev(stream_values[d,p,s][m] for s in STREAMS)/math.sqrt(3)}
         for m in ("acceptance","tps")} for d in complete_docs} for p in POLICIES}
    return {"label":"PARTIAL TRAINING-ONLY FEASIBILITY PILOT","context":4096,
        "monitor_matrix_complete":set(found)==expected,"measured_requests":len(found),"expected_requests":336,
        "complete_document_units":complete_docs,"incomplete_document_units":[d for d in MONITOR if d not in complete_docs],
        "missing_cells":[list(v) for v in sorted(expected-set(found))],"primary_baseline":f"fixed{baseline_budget}_native",
        "baseline_selected_on_monitor":False,"comparisons_all_fixed":comparisons,"action_stream_variability":variability,
        "absolute_metrics":{p:{m:{"mean":None if not complete_docs else statistics.fmean(cells[d,p][m] for d in complete_docs),
            "ci95":None if not complete_docs else paired_interval([cells[d,p][m] for d in complete_docs]),
            "per_document":{str(d):cells[d,p][m] for d in complete_docs}} for m in ("acceptance","tps","decision_ms")} for p in POLICIES},
        "aggregation":"Within-stream median of2 timing repeats, then equal mean3 action streams; paired document bootstrap10000 seed42",
        "randomness_scope":"Document CI conditional on one fitted policy and3 observed streams; stream MCSE descriptive, not training-RNG inference",
        "goal_achieved":False,"full_study_complete":False,"confirmation_authorized":False}


def identities(smoke=False):
    if smoke:
        return {("smoke_fixed",0,p,None,None,None) for p in FIXED}|{("smoke_train",0,p,0,None,None) for p in MODES}
    return ({("fit_warmup",0,p,None,None,-1) for p in FIXED}
        |{("calibration",d,p,None,None,None) for d in FIT for p in FIXED}
        |{("train",d,p,epoch,None,None) for d in FIT for p in MODES for epoch in (0,1)}
        |{("monitor_warmup",d,p,None,0,-1) for d in MONITOR for p in POLICIES}
        |{("monitor",d,p,None,s,r) for d in MONITOR for p in POLICIES for s in STREAMS for r in REPEATS})


def filename(key):
    phase,d,p,epoch,stream,repeat=key
    return f"{phase}_4096_{d}_{p}_p{epoch}_s{stream}_r{repeat}.json"


def fit_calibration(rows):
    expected={(d,p) for d in FIT for p in FIXED};by_key={}
    for row in rows:
        key=row["document"],row["policy"]
        if row["phase"]!="calibration" or key not in expected or key in by_key:raise ValueError("Invalid fit-only calibration")
        by_key[key]=row
    if set(by_key)!=expected:raise ValueError("Need all64 fit-only Native fixed calibration requests")
    rho=sum(by_key[d,"fixed8_native"]["output_tokens"] for d in FIT)/math.fsum(by_key[d,"fixed8_native"]["seconds"] for d in FIT)
    means={p:statistics.fmean(by_key[d,p]["tps"] for d in FIT) for p in FIXED}
    best=max(FIXED,key=lambda p:(means[p],-int(p[5:].split("_")[0])))
    return {"rho":rho,"baseline_returns":{str(d):by_key[d,"fixed8_native"]["output_tokens"]-rho*by_key[d,"fixed8_native"]["seconds"] for d in FIT},
            "fixed_mean_document_tps":means,"best_fixed":best}


def partial_training_summary(records):
    """Descriptive phase-specific values only, never an incomplete policy gate."""
    metrics={};stages={}
    for mode in POLICIES:
        observed=[r for r in records if r["policy"]==mode]
        phase=next((p for p in ("train","calibration","fit_warmup") if any(r["phase"]==p for r in observed)),None)
        chosen=[r for r in observed if r["phase"]==phase];stages[mode]={"phase":phase,"requests":len(chosen)}
        metrics[mode]={m:{"mean":statistics.fmean(r[m] for r in chosen) if chosen else None,"ci95":None,
                         "per_document":{}} for m in ("acceptance","tps","decision_ms")}
    return {"label":"PARTIAL TRAINING/CALIBRATION ONLY — NOT MONITOR EFFICACY","context":4096,
        "smoke":False,"smoke_passed":False,"analysis_complete":False,"pre_freeze_partial":True,
        "measured_requests":len(records),"expected_requests":556,"complete_document_units":[],
        "incomplete_document_units":list(MONITOR),"stages":stages,"absolute_metrics":metrics,
        "continuation_gate":None,"goal_achieved":False,"full_study_complete":False,
        "confirmation_authorized":False,"confidence_intervals":None,
        "warning":"Different evolving fit phases are descriptive only; no baseline comparison or monitor gate computed"}


def continuation_gate(summary):
    if not summary["analysis_complete"]:return {"evaluated":False,"passed":False,"reason":"PARTIAL pilot"}
    primary=summary["comparisons_all_fixed"][summary["primary_baseline"]]["csgb"]
    docs=summary["complete_document_units"];m=summary["absolute_metrics"]
    values=[100*(m["csgb"]["tps"]["per_document"][str(d)]/m["constant"]["tps"]["per_document"][str(d)]-1) for d in docs]
    constant={"mean":statistics.fmean(values),"ci95":paired_interval(values),"documents":len(docs)}
    positive=lambda v:v["mean"]>0 and v["ci95"] is not None and v["ci95"][0]>0
    return {"evaluated":True,"passed":all(positive(v) for v in primary.values()) and positive(constant),
            "csgb_vs_fit_best_fixed":primary,"csgb_vs_feature_matched_constant_tps_percent":constant,
            "scope":"Internal training-only4K feasibility continuation; NOT full-study success"}


def audit_root(root, *, smoke=False, partial=False):
    """Complete actual CPU replay; never opens document24+ or chooses on monitor."""
    root=Path(root).resolve();evidence={}
    def read(name):
        path=root/name
        if path.is_symlink() or path.resolve().parent!=root:raise ValueError("Aliased pilot raw file")
        data=load(path);evidence[str(path)]=sha(path);return data
    protocol=read("protocol.json");method=protocol["pilot_protocol"]
    if (protocol.get("runtime_version")!=1 or protocol.get("phase")!=("episode_smoke" if smoke else "episode_train_only_pilot")
            or protocol.get("smoke") is not smoke
            or any(protocol.get(k) is not False for k in ("production_speed_claim","development_opened","confirmation_opened"))):
        raise ValueError("Wrong runtime/scope or an unsupported production-speed claim")
    # The runner preflight is CPU-only and verifies the exact staged engine.
    known_hashes={method[k] for k in ("source_sha256","scaler_sha256","checkpoint_receipt_sha256")}
    if len(protocol["input_manifest"])!=4:raise ValueError("Expected exactly source/scaler/checkpoint/method inputs")
    if len(known_hashes)!=3 or any(list(protocol["input_manifest"].values()).count(h)!=1 for h in known_hashes):
        raise ValueError("Source/scaler/checkpoint inputs must each uniquely match their frozen digest")
    method_paths=[p for p,h in protocol["input_manifest"].items() if h not in known_hashes]
    if len(method_paths)!=1 or load(method_paths[0])!=method:raise ValueError("Missing exact frozen pilot protocol input")
    runner.validate_pilot_protocol(method_paths[0])
    immutable={}
    for mapping in (protocol["source_manifest"],protocol["input_manifest"],protocol["checkpoint_manifest"],protocol["scaler_evidence"]):
        for name,h in mapping.items():
            if name in immutable and immutable[name]!=h:raise ValueError("Conflicting immutable source/input hash")
            immutable[name]=h
    runner.verify_manifest(immutable);evidence.update(immutable)
    # Import the registered CPU provenance helpers before verifying actual
    # loaded-module identities. A fresh CLI must not rely on test imports.
    from csgb_online_evaluate_migrated import validate_checkpoint_receipt
    runner.verify_loaded_sources(protocol["source_manifest"])
    for module in (runner,sys.modules["csgb_episode_policy"],sys.modules["csgb_episode_harness"]):
        path=str(Path(module.__file__).resolve())
        if protocol["source_manifest"].get(path)!=sha(path):raise ValueError("Actually imported episode code is outside the frozen closure")
    if (protocol["source_manifest"]!=method["source_manifest"]
            or protocol["gpu_uuid"]!=method["hardware"]["expected_gpu_uuid"]
            or protocol["physical_gpu_index"]!=method["hardware"]["physical_gpu_index"]):
        raise ValueError("Runtime source or physical worker differs from the frozen method")
    scale_paths=[p for p,h in protocol["input_manifest"].items() if h==method["scaler_sha256"]]
    if len(scale_paths)!=1:raise ValueError("Missing unique fit16 scaler")
    scaler,scale_evidence=runner.validate_pilot_scaler(scale_paths[0],method["source_sha256"])
    if scale_evidence!=protocol["scaler_evidence"]:raise ValueError("Runtime scaler evidence differs from its full fit-only closure")
    receipt_path=next(p for p,h in protocol["input_manifest"].items() if h==method["checkpoint_receipt_sha256"])
    receipt=load(receipt_path)
    mapping=receipt.get("file_sha256",{})
    if protocol["checkpoint_manifest"]!=mapping or len(mapping)!=14:raise ValueError("Runtime checkpoint manifest must equal the receipt's complete14 inputs")
    folders={Path(p).parent for p in mapping}
    roles={p.name:p for p in folders}
    if len(folders)!=2 or set(roles)!={"vicuna-7b-v1.5-16k","vicuna-68m"}:raise ValueError("Checkpoint roles changed")
    validate_checkpoint_receipt(receipt_path,roles["vicuna-7b-v1.5-16k"],roles["vicuna-68m"],
        method["checkpoint_training_root"],current_versions=receipt.get("versions"))
    expected=identities(smoke);names={filename(key):key for key in expected}
    metadata={"protocol.json","DONE.json"}|({"calibration.json","training_order.json","frozen_policies.json"} if not smoke else set())
    allowed=metadata|set(names)|{f"reference_4096_{d}.json" for d in ((0,) if smoke else FIT+MONITOR)}
    actual={p.name for p in root.glob("*.json")}
    if not actual<=allowed:raise ValueError("Unexpected pilot data file; never parse unregistered documents/phases")
    pending=[p.name for p in root.iterdir() if p.name.endswith(".pending")]
    if pending and not partial:raise ValueError("Partial publication requires explicit --partial")
    if smoke and partial:raise ValueError("A partial smoke cannot authorize a pilot")
    records,raw_records,references=[],{},{}
    for name,key in sorted(names.items()):
        if name not in actual:continue
        row=read(name);phase,d,p,epoch,s,r=key
        if tuple(row.get(k) for k in ("phase","document","policy","pass_index","stream","timing_repeat"))!=key:
            raise ValueError("Filename/request identity mismatch")
        if row.get("gpu_uuid")!=protocol["gpu_uuid"] or any(type(row.get(k)) is not int for k in ("schema","document","context")) or row["schema"]!=1:
            raise ValueError("Request GPU/schema identity mismatch")
        if d not in references:references[d]=read(f"reference_4096_{d}.json")
        record=audit_episode_record(row,references[d],eos_token_id=row["eos_token_id"])
        if p in MODES and row["request_seed"]!=runner.request_seed(method,phase,p,d,pass_index=epoch,stream=s):
            raise ValueError("Action seed differs from the frozen per-request identity")
        records.append(record);raw_records[key]=row
    found=set(raw_records)
    if not partial and found!=expected:raise ValueError("Incomplete pilot/smoke request matrix")
    if len({r["eos_token_id"] for r in raw_records.values()})!=1:raise ValueError("Tokenizer EOS changed across requests")
    calibration=None;frozen=None;training=[]
    if not smoke and partial and ("frozen_policies.json" not in actual or
            any(k not in found for k in expected if k[0] in {"calibration","train"})):
        if any(k[0] in {"monitor","monitor_warmup"} for k in found):raise ValueError("Monitor was opened before complete training/final freeze")
        if "DONE.json" in actual:raise ValueError("A partial training run cannot claim terminal data completion")
        summary=partial_training_summary(records)
        for name,h in evidence.items():
            if sha(name)!=h:raise ValueError("Input/source changed during partial CPU analysis")
        return {"summary":summary,"records":records,"training":[],"protocol":protocol,
                "input_sha256":evidence,"root":str(root),"pending":pending}
    if smoke:
        if found!=expected:raise ValueError("All seven technical smoke requests are required")
        first=raw_records["smoke_fixed",0,"fixed8_native",None,None,None]["episode"]
        rho=first["useful_n"]/first["seconds"]
        for mode in MODES:
            row=raw_records["smoke_train",0,mode,0,None,None]
            if row["parameters_before"]!=runner.make_policy(mode,scaler).to_state() or row["rho"]!=rho or row["baseline_return"]!=0.:
                raise ValueError("Smoke must start from zero parameters and independent Native8 calibration")
        summary={"label":"TECHNICAL SMOKE — NOT EFFICACY","smoke":True,"documents":[0],"context":4096,
            "requests":7,"smoke_passed":True,"analysis_complete":True,"goal_achieved":False,
            "full_study_complete":False,"confirmation_authorized":False,"confidence_intervals":None,
            "absolute_metrics":{r["policy"]:{m:{"mean":r[m],"ci95":None,"per_document":{"0":r[m]}} for m in ("tps","acceptance","decision_ms")} for r in records}}
    else:
        calibration=fit_calibration([r for r in records if r["phase"]=="calibration"])
        stored=read("calibration.json")
        if any(stored.get(k)!=v for k,v in calibration.items()) or stored.get("monitor_used") is not False or stored.get("baseline_frozen_before_training") is not True:
            raise ValueError("Baseline/rho must be derived once from fit0..15 only")
        expected_calibration={str(root/filename(k)) for k in expected if k[0]=="calibration"}
        if ({str(root/Path(p).name) for p in stored["request_sha256"]}!=expected_calibration
                or len(stored["request_sha256"])!=64):raise ValueError("Calibration manifest must bind all64 fit-only requests")
        for name,h in stored["request_sha256"].items():
            if evidence.get(str(root/Path(name).name))!=h:raise ValueError("Calibration raw SHA mismatch")
        current={m:runner.make_policy(m,scaler).to_state() for m in MODES}
        order=[]
        for epoch in (0,1):
            for mode,d in runner.training_order(method,epoch):
                key="train",d,mode,epoch,None,None
                if key not in raw_records:raise ValueError("Full two-pass training must finish before monitor analysis")
                row=raw_records[key];order.append([mode,d])
                if (row["parameters_before"]!=current[mode] or row["rho"]!=calibration["rho"]
                        or row["baseline_return"]!=calibration["baseline_returns"][str(d)]):
                    raise ValueError("Training parameter chain or action-independent baseline changed")
                current[mode]=row["parameters_after"]
                training.append({"policy":mode,"document":d,"pass_index":epoch,"step":len(training),
                    "tps":row["episode"]["request_tps"],"episode_return":row["episode_return"],
                    "advantage":row["episode_return"]-row["baseline_return"],"update_wall_seconds":row["update_wall_seconds"]})
        if read("training_order.json")["passes"]!=[[list(v) for v in runner.training_order(method,p)] for p in (0,1)]:
            raise ValueError("Recorded training order differs from its predeclared shuffle")
        frozen=read("frozen_policies.json")
        if (frozen.get("models")!=current or frozen.get("final_only_no_selection") is not True or frozen.get("passes")!=2
                or frozen.get("fit_documents")!=list(FIT) or frozen.get("monitor_documents_used_for_training") is not False
                or frozen.get("calibration_sha256")!=sha(root/"calibration.json")
                or frozen.get("scaler_sha256")!=method["scaler_sha256"]
                or frozen.get("source_manifest")!=protocol["source_manifest"] or frozen.get("input_manifest")!=protocol["input_manifest"]
                or any(frozen.get(k) is not False for k in ("development_opened","confirmation_opened"))):raise ValueError("Wrong final-only checkpoint freeze")
        prefreeze={"protocol.json","calibration.json","training_order.json"}|{f"reference_4096_{d}.json" for d in FIT}|{filename(k) for k in expected if k[0] in {"fit_warmup","calibration","train"}}
        if ({Path(p).name for p in frozen["training_evidence"]}!=prefreeze
                or len(frozen["training_evidence"])!=len(prefreeze)):raise ValueError("Incomplete/duplicate final training evidence")
        for name,h in frozen["training_evidence"].items():
            local=root/Path(name).name
            if not local.is_file() or sha(local)!=h:raise ValueError("Changed frozen training evidence")
            evidence[str(local)]=h
        for key,row in raw_records.items():
            if key[0] in {"monitor","monitor_warmup"} and row["policy"] in MODES:
                if row["parameters_before"]!=current[row["policy"]] or row["parameters_after"]!=current[row["policy"]]:
                    raise ValueError("Monitor selected/updated another epoch or parameter state")
        summary=aggregate_monitor([r for r in records if r["phase"]=="monitor"],int(calibration["best_fixed"][5:].split("_")[0]))
        summary.update(smoke=False,analysis_complete=found==expected and not pending,smoke_passed=False,
            fit_calibration=calibration,training_requests=96,fit_documents=list(FIT),monitor_documents=list(MONITOR))
    if "DONE.json" in actual:
        done=read("DONE.json")
        if (done.get("raw_data_complete") is not True or done.get("smoke") is not smoke
                or done.get("requests_including_warmups")!=len(expected) or found!=expected
                or any(done.get(k) is not False for k in ("study_complete","production_speed_claim","development_opened","confirmation_opened"))):
            raise ValueError("DONE does not describe the complete registered phase")
        if (smoke and (done.get("no_final_models") is not True or done.get("monitor_opened") is not False)
                or not smoke and (any(done.get(k)!=v for k,v in {"calibration_requests":64,"training_requests":96,"monitor_requests":336,"warmup_requests":60}.items())
                    or done.get("frozen_policies_sha256")!=sha(root/"frozen_policies.json"))):
            raise ValueError("Terminal phase counts/freeze or smoke scope changed")
        mapping={Path(name).name:h for name,h in done["raw_sha256"].items()}
        if len(mapping)!=len(done["raw_sha256"]) or set(mapping)!=actual-{"DONE.json"}:
            raise ValueError("Incomplete/ambiguous final raw manifest")
        for name,h in mapping.items():
            if sha(root/name)!=h:raise ValueError("Changed raw bytes after terminal completion")
            evidence[str(root/name)]=h
    elif not partial:raise ValueError("Missing terminal completion marker")
    else:summary["analysis_complete"]=False
    if not smoke:summary["continuation_gate"]=continuation_gate(summary)
    for name,h in evidence.items():
        if sha(name)!=h:raise ValueError("Input/source changed during CPU analysis")
    return {"summary":summary,"records":records,"training":training,"protocol":protocol,
            "input_sha256":evidence,"root":str(root),"pending":pending}


def render_figures(audit, out):
    """Render only recorded values; uncertainty unit is the paired document."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap
    summary=audit["summary"];smoke=summary["smoke"];pre=summary.get("pre_freeze_partial",False)
    caption=("TECHNICAL SMOKE: doc0 only, n=1, no CI; NOT efficacy" if smoke else
        "PARTIAL FIT/CALIBRATION: evolving observed phases only; no comparator, no CI" if pre else
        f"PARTIAL STUDY: 4K historical training-monitor, n={len(summary['complete_document_units'])}/8 docs; "
        "median2 timing repeats → mean3 action streams; paired-doc95% bootstrap CI")
    if audit.get("synthetic_qa"):caption="SYNTHETIC CPU QA — NOT EXPERIMENT DATA | "+caption
    figures=[]
    def save(fig,name):
        width=max(60,int(fig.get_figwidth()*18))
        footer=textwrap.fill(caption,width)+"\n"+textwrap.fill(
            "Eager FP32 Target / FP16 Draft; server-reported 48GB RTX4090; instrumented, NOT production speed",width)
        fig.text(.5,.015,footer,ha="center",va="bottom",fontsize=7)
        for ax in fig.axes:
            title_width=max(32,int(fig.get_figwidth()*ax.get_position().width*14))
            ax.set_title(textwrap.fill(ax.get_title(),title_width))
        fig.tight_layout(rect=(0,.20,1,1))
        for suffix in ("png","svg"):fig.savefig(out/f"{name}.{suffix}",dpi=170)
        plt.close(fig);figures.append(name)
    for metric,name,ylabel in (("acceptance","acceptance","Accepted Draft tokens / scored round" if smoke or pre else "Δ accepted Draft tokens / scored round"),
                                ("tps","request_tps","Whole-request useful output tokens/s" if smoke or pre else "Δ whole-request TPS (%)")):
        fig,ax=plt.subplots(figsize=(9,4.5));values=[];errors=[]
        for p in POLICIES:
            row=(summary["absolute_metrics"][p][metric] if smoke or pre else summary["comparisons_all_fixed"][summary["primary_baseline"]][p][metric])
            value=row["mean"];ci=row["ci95"];values.append(float("nan") if value is None else value)
            errors.append((0.,0.) if value is None or ci is None else (max(0.,value-ci[0]),max(0.,ci[1]-value)))
        ax.bar(range(7),values,color=["#8795a1"]*4+["#167a9c","#d79137","#9365a3"])
        ax.errorbar(range(7),values,yerr=list(zip(*errors)),fmt="none",ecolor="black",capsize=4)
        ax.set(xticks=range(7),xticklabels=POLICIES,ylabel=ylabel,
               title=("TECHNICAL SMOKE — " if smoke else "PARTIAL FIT STAGES — " if pre else "TRAINING-ONLY MONITOR — ")+name.replace("_"," ")+" (actual4096 prompt tokens)")
        ax.tick_params(axis="x",rotation=25);ax.axhline(0,color="black",lw=.7);ax.grid(axis="y",alpha=.2)
        if pre:ax.text(.02,.97,"PARTIAL evolving fit/calibration phases; no comparison",transform=ax.transAxes,va="top",fontsize=8)
        elif not smoke:ax.text(.02,.97,"Fit-only frozen comparator: "+summary["primary_baseline"],transform=ax.transAxes,va="top",fontsize=8)
        save(fig,name)
    rows=[r for r in audit["records"] if r["policy"]=="csgb" and r["phase"]==("smoke_train" if smoke else "train" if pre else "monitor") and (smoke or pre or r["timing_repeat"]==0)]
    eligible=[r for request in rows for r in request["rounds"] if r.get("raw_x") is not None]
    fig,ax=plt.subplots(figsize=(7,4.6))
    if eligible:
        scatter=ax.scatter([r["raw_x"][1] for r in eligible],[r["raw_x"][2] for r in eligible],
            c=[r["budget"] for r in eligible],cmap="viridis",vmin=8,vmax=32,s=18,alpha=.6)
        fig.colorbar(scatter,ax=ax,ticks=BUDGETS,label="Observed sampled proposal-node budget")
    else:ax.text(.5,.5,"No eligible C/D rounds recorded",transform=ax.transAxes,ha="center")
    ax.set(xlabel="Attention Coverage C (next Root cache)",ylabel="Cache-anchored normalized JS Shift D",
           xlim=(-.02,1.02),title=f"Observed CSGB actions, {len(eligible)} eligible rounds; NOT inferred decision boundary")
    if eligible:
        lo=min(r["raw_x"][2] for r in eligible);hi=max(r["raw_x"][2] for r in eligible)
        padding=max(.005,(hi-lo)*.12);ax.set_ylim(max(0.,lo-padding),min(1.,hi+padding))
        ax.text(.02,.98,"D zoomed to observations; theoretical normalized JS range [0,1]",transform=ax.transAxes,va="top",fontsize=7)
    else:ax.set_ylim(0.,1.)
    save(fig,"observed_budget_map")
    fig,ax=plt.subplots(figsize=(9,4.5));ys=[];es=[]
    for p in POLICIES:
        row=summary["absolute_metrics"][p]["decision_ms"];value=row["mean"];ci=row["ci95"]
        ys.append(float("nan") if value is None else value)
        es.append((0.,0.) if value is None or ci is None else (max(0.,value-ci[0]),max(0.,ci[1]-value)))
    ax.bar(range(7),ys,color="#53768c");ax.errorbar(range(7),ys,yerr=list(zip(*es)),fmt="none",ecolor="black",capsize=4)
    ax.set(xticks=range(7),xticklabels=POLICIES,ylabel="Feature/decision wall (ms/round)",
           title="Measured decision region includes implicit host/GPU waits; not pure arithmetic latency")
    ax.tick_params(axis="x",rotation=25);save(fig,"decision_latency")
    if not smoke and not pre:
        fig,axes=plt.subplots(2,1,figsize=(8,6),sharex=True)
        for mode in MODES:
            points=[r for r in audit["training"] if r["policy"]==mode]
            axes[0].plot(range(1,len(points)+1),[r["episode_return"] for r in points],label=mode,alpha=.8)
            axes[1].plot(range(1,len(points)+1),[r["advantage"] for r in points],label=mode,alpha=.8)
        axes[0].set(ylabel="Observed N − ρT",title="Fit0..15 learning trace: evolving policy, NOT held-out efficacy")
        axes[1].set(xlabel="Whole-request SGD step per policy (16docs ×2 fixed passes)",ylabel="Return − independent Native8 baseline")
        axes[0].legend();save(fig,"training_learning_curve")
        fig,axes=plt.subplots(1,2,figsize=(10,4.5))
        for ax,metric in zip(axes,("acceptance","tps")):
            for i,mode in enumerate(MODES):
                values=[v[metric]["sample_sd"] for v in summary["action_stream_variability"][mode].values()]
                ax.scatter([i]*len(values),values,alpha=.65)
            ax.set(xticks=range(3),xticklabels=MODES,ylabel="Within-doc SD across3 action streams",
                   title="Sampling variability: "+metric)
            ax.tick_params(axis="x",rotation=15)
        save(fig,"action_stream_variability")
    return figures


_FIGURE_RENDERER=render_figures


def report_text(audit, figures):
    s=audit["summary"];smoke=s["smoke"]
    text=["# "+s["label"],"","This is a 4K-only technical/feasibility experiment, not completion of the 4K+8K study.",
          "No development or confirmation documents were opened. Native V is unchanged; the pilot evaluates module3 only.",""]
    if s.get("pre_freeze_partial"):
        text += [f"PARTIAL: {s['measured_requests']}/556 request records are available. Fit calibration/training/final freeze is incomplete; no monitor comparison, fitted-policy success, CI or continuation gate is calculated.",
            "Per-policy latest observed fit/calibration phase and request counts: "+json.dumps(s["stages"]),
            "Absolute plots are descriptive stage diagnostics only. Evolving training requests and fixed calibration requests must not be interpreted as a paired policy comparison.",""]
    elif smoke:
        text += ["All7 registered requests from doc0 passed canonical Token equality, useful-N/terminal accounting, attention-feature chronology, parameter and action-RNG replay, and external summed-score SGD auditing.",
            "This n=1 technical smoke has no confidence interval, trained final checkpoint, monitor result or efficacy claim.",""]
    else:
        complete=s["complete_document_units"]
        text += [f"Fit: historical training docs0..15 (64 fixed-arm calibrations,96 SGD requests,2 fixed passes). Internal monitor: historical training docs16..23, {len(complete)}/8 complete paired document units.",
            f"Monitor requests: {s['measured_requests']}/336. Incomplete document units excluded in full: {s['incomplete_document_units']}. Missing cells are saved in summary.json. {'Full registered pilot phase analyzed.' if s['analysis_complete'] else 'PARTIAL data: not a final pilot result.'}",
            "One predeclared training seed; no epoch/seed selection. Final parameters are frozen, but stochastic softmax action RNG progresses and is replayed. Three action streams are not three independently trained models.",
            "Each fit document/fixed arm has only1 calibration timing for rho/baseline selection. Training runs in mode blocks (32 updates each), one seed, so time drift is a limitation. Monitor randomizes the7 policy order for each document/stream/repeat on the registered physical GPU. Request-boundary process checks establish only this GPU's observed process isolation, not whole-machine isolation.",
            f"Primary comparator: {s['primary_baseline']}, chosen only on fit16 equal-document mean request TPS (ties smaller budget). Rho and action-independent document baselines also use fit Native8 only. All four fixed comparisons are retained; monitor never selects a winner.",
            "Each stream first takes the median of2 timing repeats, then the3 stream values are averaged within each document. Paired bootstrap resamples8 documents (or the explicitly partial completed subset),10000 draws, seed42, linear2.5/97.5 percentiles. These intervals are conditional on this fitted model and observed action streams, not a training-seed confidence interval. Stream SD/MCSE is descriptive only.",""]
        comp=s["comparisons_all_fixed"][s["primary_baseline"]]["csgb"]
        for metric in ("acceptance","tps"):
            v=comp[metric];text.append(f"CSGB {metric} paired difference: mean={v['mean']},95%CI={v['ci95']},n={v['documents']} documents (TPS units percent; acceptance units tokens/scored round).")
        text += ["","Pilot continuation gate: "+json.dumps(s["continuation_gate"])+". Requires positive mean and95%CI lower bound for CSGB acceptance and TPS against fit-selected bestfixed, plus TPS against feature-matched constant. This never establishes generalization, an8K gain or the full strict study gate.",""]
    text += ["Whole-request time includes prefill, fresh attention/features where applicable, Draft/Target work, terminal round and closing synchronization. Canonical AR generation, JSON/audits and external request-end SGD are outside that timer. No explicit per-round synchronization/SGD is used, but item/tolist/D2H and host decisions still cause implicit waits.",
        "Target uses eager attention in FP32; Draft is FP16. Hardware is a server-reported48GB RTX4090. These are instrumented implementation measurements, NOT production-speed evidence. Native fixed paths retain their native attention/cache cadence; attention methods pay their actual fresh-feature cost.",
        "The CPU auditor checks saved chronology, sampling and accounting, not a reconstruction of full GPU attention matrices. Runtime engine/model/input/source manifests bind that limitation explicitly.","",
        "## Saved artifacts","",f"Raw results: [{audit['root']}]({audit['root']})", "[Summary](summary.json) · [Input and artifact hashes](manifest.json) · [Full audit source](analysis_source.py) · [Standalone rerender source](figure_source.py)",""]
    text += [f"[{name} rendered PNG]({name}.png) · [vector SVG]({name}.svg)" for name in figures]
    return "\n\n".join(text)+"\n"


def write_analysis(audit, out):
    """Exclusive artifact publication. Failure leaves a visible partial directory."""
    out=Path(out).resolve()
    if out.exists():raise ValueError("Analysis output exists; never overwrite partial/completed analysis")
    for name,h in audit["input_sha256"].items():
        if sha(name)!=h:raise ValueError("Raw/source changed before analysis publication")
    out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve();shutil.copyfile(source,out/"analysis_source.py")
    figures=render_figures(audit,out)
    (out/"REPORT.md").write_text(report_text(audit,figures))
    # True only after user-viewable figures and a written report exist.
    summary_text=json.dumps(audit["summary"],indent=2,allow_nan=False)+"\n"
    (out/"plot_data.json").write_text(json.dumps({"summary":audit["summary"],"training":audit["training"],
        "observed_rounds":[r for r in audit["records"] if r["phase"] in {"train","monitor","smoke_fixed","smoke_train"}],
        "synthetic_qa":audit.get("synthetic_qa",False)},indent=2,allow_nan=False)+"\n")
    figure_source=("\"\"\"Standalone render from saved plot_data.json; no engine/model imports.\"\"\"\n"
        "import argparse, json\nfrom pathlib import Path\n"+f"BUDGETS={tuple(BUDGETS)!r}\nMODES={MODES!r}\nPOLICIES={POLICIES!r}\n"+
        inspect.getsource(_FIGURE_RENDERER)+"\nif __name__=='__main__':\n"
        "    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path(__file__).with_name('plot_data.json'));p.add_argument('--out',type=Path,required=True);a=p.parse_args()\n"
        "    d=json.loads(a.data.read_text());a.out.mkdir(parents=True,exist_ok=False)\n"
        "    render_figures({'summary':d['summary'],'training':d['training'],'records':d['observed_rounds'],'synthetic_qa':d.get('synthetic_qa',False)},a.out)\n")
    (out/"figure_source.py").write_text(figure_source)
    for name,h in audit["input_sha256"].items():
        if sha(name)!=h:raise ValueError("Raw/source changed during analysis publication")
    artifacts={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    artifacts["summary.json"]=hashlib.sha256(summary_text.encode()).hexdigest()
    manifest={"input_sha256":audit["input_sha256"],"artifacts":artifacts,
              "analysis_source_sha256":sha(source),"raw_root":audit["root"],
              "figures":figures,"goal_achieved":False,"production_speed_claim":False}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    # Queue-visible smoke authorization is the LAST publication operation.
    with (out/"summary.json").open("x") as handle:handle.write(summary_text)
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--smoke",action="store_true");p.add_argument("--partial",action="store_true")
    args=p.parse_args(argv)
    audit=audit_root(args.raw,smoke=args.smoke,partial=args.partial)
    manifest=write_analysis(audit,args.out)
    print(json.dumps({"analysis_complete":audit["summary"]["analysis_complete"],
        "smoke_passed":audit["summary"]["smoke_passed"],"goal_achieved":False,
        "report":str(args.out.resolve()/"REPORT.md"),"figures":manifest["figures"]},allow_nan=False))


if __name__=="__main__":main()
