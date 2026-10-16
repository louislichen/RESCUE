"""Isolated train-only4K episodic pilot. CPU contracts precede any GPU import.

Only fit documents0..15 train parameters;16..23 monitor the final two-pass
checkpoint. No development/confirmation loader or automatic next-stage exists.
Whole-request timers include prefill/features/controller/verification, exclude
reference/cleanup/serialization/SGD, and use explicit boundary synchronization
only (engine/feature implicit synchronization remains). Not production speed.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

from csgb_episode_harness import (EpisodeHarness, audit_decisions, rng_state,
    validate_feature_chronology, validate_request_accounting)
from csgb_episode_policy import BUDGETS, Decision, EpisodePolicy, episode_gradient
from csgb_online_train import exclusive_json

CONTEXT = 4096
FIT_DOCUMENTS = tuple(range(16))
MONITOR_DOCUMENTS = tuple(range(16,24))
MODES = ("csgb","constant","history_native")
FIXED = tuple(f"fixed{b}_native" for b in BUDGETS)
POLICIES = FIXED+MODES
SETTINGS = {"model_seed":42,"target_dtype":"float32","draft_dtype":"float16",
            "max_new_tokens":128,"max_depth":10,"threshold":.7,
            "cache_refresh_every":4,"chunk_size":32,"retrieve_top_k":32,
            "corrected_root_coverage":True,"terminal_draft_elision":True,"tf32":False}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def _finite(value, *, positive=False):
    if type(value) not in (int,float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError("Finite episode scalar required")
    return value


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum: raise ValueError("Invalid integer identity")
    return value


def verify_manifest(mapping):
    if not isinstance(mapping,dict) or not mapping: raise ValueError("Missing immutable source/input manifest")
    for name,wanted in mapping.items():
        path = Path(name)
        if (not path.is_absolute() or str(path.resolve()) != name or path.is_symlink()
                or not isinstance(wanted,str) or re.fullmatch(r"[0-9a-f]{64}",wanted) is None
                or sha(path) != wanted):
            raise ValueError("Missing, aliased or changed immutable evidence: "+str(name))
    return dict(mapping)


def verify_loaded_sources(mapping):
    """A same-named module elsewhere must not bypass the file inventory."""
    names = ("csgb_episode_policy","csgb_episode_harness","csgb_online_harness",
        "csgb_cache_geometry","csgb_online_train","csgb_online_policy",
        "csgb_online_evaluate_migrated","csgb_migration_provenance")
    for name in names:
        module = sys.modules.get(name)
        actual = None if module is None else getattr(module,"__file__",None)
        if actual is None or mapping.get(str(Path(actual).resolve())) != sha(actual):
            raise ValueError("Actually imported module is outside the frozen source closure: "+name)


def load_train24(path):
    # This must be an isolated24-record input, not a prefix-loader of full48.
    # Do not parse record24+ even when the caller accidentally supplies full48.
    with Path(path).open() as handle:
        rows = [json.loads(handle.readline()) for _ in range(24)]
        extra = handle.read(1)
    if extra or any(not isinstance(r,dict) or not isinstance(r.get("text"),str) or not r["text"] for r in rows):
        raise ValueError("Only the isolated24 training records are permitted")
    return [r["text"] for r in rows]


def validate_pilot_protocol(path):
    p = load(path)
    expected = {"schema":1,"kind":"csgb_episode_train_only_pilot_v1",
        "fit_documents":list(FIT_DOCUMENTS),"monitor_documents":list(MONITOR_DOCUMENTS),
        "contexts":[CONTEXT],"development_opened":False,"confirmation_opened":False,
        "no_automatic_development_or_confirmation_launch":True,"native_v":True,
        "no_epoch_or_seed_selection":True}
    method = {"budgets":list(BUDGETS),"modes":list(MODES),"temperature":1.,
              "learning_rate":.0005,"passes":2,"initial_weights":"all_zero",
              "gradient":"episode_return_minus_independent_baseline_times_sum_log_policy_scores",
              "round_count_normalization":False}
    monitor = {"policies":list(POLICIES),"action_streams":3,"timing_repeats":2,"warmups_per_policy":1}
    if (any(p.get(k) != v or type(p.get(k)) is not type(v) for k,v in expected.items())
            or any(p.get("method",{}).get(k) != v or type(p["method"].get(k)) is not type(v) for k,v in method.items())
            or any(p.get("monitor",{}).get(k) != v or type(p["monitor"].get(k)) is not type(v) for k,v in monitor.items())
            or p.get("settings") != SETTINGS
            or any(type(p["settings"][k]) is not type(v) for k,v in SETTINGS.items())):
        raise ValueError("Changed episodic pilot objective/split/settings")
    for name in ("seed","order_seed","stream_seed"): _integer(p["method"].get(name),1)
    for name in ("source_sha256","scaler_sha256","checkpoint_receipt_sha256"):
        if not isinstance(p.get(name),str) or re.fullmatch(r"[0-9a-f]{64}",p[name]) is None:
            raise ValueError("Protocol lacks immutable input hash")
    if (not isinstance(p.get("checkpoint_training_root"),str)
            or not Path(p["checkpoint_training_root"]).is_absolute()):
        raise ValueError("Checkpoint observation needs its genuine original training root")
    hardware = p.get("hardware",{})
    if (type(hardware.get("physical_gpu_index")) is not int or hardware["physical_gpu_index"] not in range(8)
            or not isinstance(hardware.get("expected_gpu_uuid"),str)
            or re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",hardware["expected_gpu_uuid"]) is None):
        raise ValueError("Explicit physical worker identity required")
    verify_manifest(p.get("source_manifest"))
    code = Path(__file__).parent.resolve()
    actual = {str(q.resolve()):sha(q) for q in code.rglob("*.py")}
    engine = {name:wanted for name,wanted in p["source_manifest"].items() if Path(name).is_relative_to(code)}
    if actual != engine:
        raise ValueError("Source manifest must cover the entire actual staged Python engine")
    return p


def validate_pilot_scaler(path, source_sha256):
    s = load(path)
    expected = {"schema":1,"purpose":"episode fit16 fixed population scale","context":CONTEXT,
        "fit_documents":list(FIT_DOCUMENTS),"monitor_documents_used":False,
        "development_opened":False,"confirmation_opened":False,"source_sha256":source_sha256}
    if any(s.get(k) != v or type(s.get(k)) is not type(v) for k,v in expected.items()):
        raise ValueError("Scaler must use fit16 only, never old24-document statistics")
    _integer(s.get("count"),16)
    if not isinstance(s.get("mean"),list) or not isinstance(s.get("scale"),list):
        raise ValueError("Explicit three-feature population statistics required")
    # Core validates dimensions/finite means/positive standard deviations.
    EpisodePolicy("csgb",((0.,)*4,)*4,1.,s["mean"],s["scale"])
    evidence = verify_manifest(s.get("input_sha256"))
    sources = verify_manifest(s.get("source_manifest"))
    if any(k in evidence and evidence[k] != v for k,v in sources.items()):
        raise ValueError("Contradictory scaler raw/source evidence")
    evidence.update(sources)
    # Recompute only fit raw FEATURES. Prior seconds/rho/rewards never enter
    # this scale or the current pilot's new complete-request calibration.
    observed,counts,rounds,terminal = [],{},0,0
    for doc in FIT_DOCUMENTS:
        names = [n for n in s["input_sha256"] if Path(n).name == f"calibration_4096_{doc}_fixed8_fresh.json"]
        refs = [n for n in s["input_sha256"] if Path(n).name == f"reference_4096_{doc}.json"]
        if len(names) != 1 or len(refs) != 1: raise ValueError("Missing unique fit-only feature/reference source")
        row,reference = load(names[0]),load(refs[0])
        if ((row.get("document"),row.get("context"),row.get("policy"),row.get("phase")) !=
                (doc,CONTEXT,"fixed8_fresh","calibration") or row.get("exact") is not True
                or row["result"]["output_token_ids"] != reference["output_token_ids"]):
            raise ValueError("Scaler source is not a canonical fit-only Fixed8 feature request")
        counts[str(doc)] = 0; rounds += len(row["rounds"])
        for j,r in enumerate(row["rounds"]):
            raw = r["raw_x"]
            if raw is not None:
                EpisodePolicy("csgb",((0.,)*4,)*4,1.,(0.,)*3,(1.,)*3).transform(raw)
                if (j < 2 or r.get("proposal") != j or r.get("attention_observed_timestep") != j-1
                        or r.get("anchor_timestep") != 4*((j-2)//4)
                        or r.get("terminal_commit") is not (j == len(row["rounds"])-1)):
                    raise ValueError("Scaler feature chronology/terminal membership differs")
                observed.append(raw[1:]); counts[str(doc)] += 1; terminal += int(r["terminal_commit"])
        if counts[str(doc)] < 1: raise ValueError("Every fit document must contribute features")
    columns = list(zip(*observed))
    means = [math.fsum(v)/len(v) for v in columns]
    scales = [math.sqrt(math.fsum((v-m)**2 for v in col)/len(col)) for col,m in zip(columns,means)]
    if (s["mean"] != means or s["scale"] != scales or s["count"] != len(observed)
            or s.get("per_document_count",counts) != counts
            or s.get("raw_total_verified_rounds",rounds) != rounds
            or s.get("final_verified_rounds_included",terminal) != terminal
            or s.get("original_timing_or_rho_reused") is not False):
        raise ValueError("Scaler does not equal the complete fit-only population statistics")
    return s,evidence


def request_seed(protocol, phase, mode, document, *, pass_index=None, stream=None):
    """Repeat intentionally absent: same monitor stream resets exactly."""
    if type(document) is not int or mode not in MODES or phase not in ("train","monitor","monitor_warmup","smoke_train"):
        raise ValueError("Only learned actual requests have action-stream seeds")
    if phase in ("train","smoke_train"):
        if document not in FIT_DOCUMENTS or type(pass_index) is not int or pass_index not in (0,1) or stream is not None:
            raise ValueError("Training stream outside locked fit/pass")
        if phase == "smoke_train" and (document != 0 or pass_index != 0):
            raise ValueError("Disposable smoke uses doc0 once")
    elif (document not in MONITOR_DOCUMENTS or pass_index is not None
            or type(stream) is not int or stream not in range(3)):
        raise ValueError("Monitoring stream outside locked monitor set")
    identity = [protocol["method"]["stream_seed"],phase,mode,document,pass_index,stream]
    return int.from_bytes(hashlib.sha256(json.dumps(identity,separators=(",",":")).encode()).digest()[:8],"big")


def training_order(protocol, pass_index):
    if type(pass_index) is not int or pass_index not in (0,1): raise ValueError("Only two frozen passes")
    order = []
    for mode in MODES:
        docs = list(FIT_DOCUMENTS)
        random.Random(protocol["method"]["order_seed"]+100*pass_index+MODES.index(mode)).shuffle(docs)
        order += [(mode,d) for d in docs]
    return order


def make_policy(mode, scaler):
    if mode not in MODES: raise ValueError("Unregistered learned policy")
    width = {"csgb":4,"constant":1,"history_native":3}[mode]
    return EpisodePolicy(mode,((0.,)*width,)*4,1.,
        scaler["mean"] if mode == "csgb" else None,scaler["scale"] if mode == "csgb" else None)


def validate_episode_record(row, reference, *, eos_token_id=None, protocol=None):
    """Independent pure replay; does not trust Harness.finish or a DONE flag."""
    phase,doc,mode = row.get("phase"),row.get("document"),row.get("policy")
    if (type(row.get("schema")) is not int or row["schema"] != 1 or mode not in POLICIES
            or type(doc) is not int or type(row.get("context")) is not int or row["context"] != CONTEXT
            or phase not in ("fit_warmup","monitor_warmup","calibration","train","monitor","smoke_fixed","smoke_train")
            or (doc not in FIT_DOCUMENTS if phase in ("fit_warmup","calibration","train","smoke_fixed","smoke_train")
                else doc not in MONITOR_DOCUMENTS)
            or row.get("uncontended_at_request_boundaries") is not True):
        raise ValueError("Request split/identity/physical boundary contract changed")
    if (phase in ("calibration","fit_warmup","smoke_fixed") and mode not in FIXED
            or phase in ("train","smoke_train") and mode not in MODES
            or phase in ("fit_warmup","smoke_fixed","smoke_train") and doc != 0):
        raise ValueError("Wrong phase-specific policy/document")
    pindex,stream,repeat = (row.get(k) for k in ("pass_index","stream","timing_repeat"))
    if phase in ("train","smoke_train"):
        if type(pindex) is not int or pindex not in (0,1) or stream is not None or repeat is not None:
            raise ValueError("Only predeclared training passes")
        if phase == "smoke_train" and pindex != 0: raise ValueError("Single disposable smoke update")
    elif phase in ("monitor","monitor_warmup"):
        if (pindex is not None or type(stream) is not int or stream not in range(3)
                or type(repeat) is not int or (repeat not in (0,1) if phase == "monitor" else repeat != -1 or stream != 0)):
            raise ValueError("Wrong frozen monitor action-stream/repeat")
    elif pindex is not None or stream is not None or repeat != (-1 if phase == "fit_warmup" else None):
        raise ValueError("Calibration/warmup must not hide action-stream/epoch selection")
    e = row["episode"]
    for name,wanted in {"phase":phase,"policy":mode,"parameters_unchanged":True,
        "explicit_round_sync":False,"round_updates":False,"whole_request_timing_supplied_by_caller":True,
        "baseline_or_sgd_performed":False}.items():
        if e.get(name) != wanted or type(e.get(name)) is not type(wanted):
            raise ValueError("In-request update/synchronization or phase mismatch")
    checked = validate_request_accounting(e["result"],reference,e["rows"],context=CONTEXT,
        seconds=e["seconds"],max_new_tokens=128,eos_token_id=eos_token_id,mode=mode)
    if any(e.get(k) != v for k,v in checked.items()) or e.get("reference_output_token_ids") != reference["output_token_ids"]:
        raise ValueError("Serialized whole-request numerator/time/acceptance differs")
    validate_feature_chronology(e["rows"],mode)
    identity = {k:row[k] for k in ("document","context","phase","pass_index","stream","timing_repeat","request_seed")}
    if any(not isinstance(r.get("provenance"),dict) or any(r["provenance"].get(k) != v for k,v in identity.items()) for r in e["rows"]):
        raise ValueError("Verified round provenance differs from its request identity")
    before,after = row.get("parameters_before"),row.get("parameters_after")
    if mode in MODES:
        p = EpisodePolicy.from_state(before)
        if (p.mode != mode or p.temperature != 1. or e["parameter_sha256"] != p.parameter_sha256
                or e["parameter_version"] != p.parameter_version):
            raise ValueError("Wrong episode parameter identity/temperature")
        _integer(row.get("request_seed"))
        if e["rng_before"] != rng_state(random.Random(row["request_seed"])):
            raise ValueError("Initial RNG differs from the recorded request-local seed")
        if protocol is not None and row["request_seed"] != request_seed(protocol,phase,mode,doc,pass_index=pindex,stream=stream):
            raise ValueError("Action stream differs from the preregistered identity")
        audit_decisions(p,e["decisions"],e["rows"],e["rng_before"],e["rng_after"])
        if phase in ("train","smoke_train"):
            _integer(row.get("pass_index")); _finite(row.get("baseline_return"))
            if row["pass_index"] not in (0,1): raise ValueError("Extra epoch is forbidden")
            rho = _finite(row.get("rho"),positive=True)
            expected_return = checked["useful_n"]-rho*checked["seconds"]
            gradient = episode_gradient(p,[Decision.from_state(d) for d in e["decisions"]],
                useful_n=checked["useful_n"],elapsed_s=checked["seconds"],rho=rho,baseline=row["baseline_return"])
            if (row.get("episode_return") != expected_return or row.get("gradient") != [list(v) for v in gradient.values]
                    or row.get("learning_rate") != .0005 or p.sgd(gradient,.0005).to_state() != after):
                raise ValueError("Whole-request return/summed gradient/external SGD mismatch")
            _finite(row.get("update_wall_seconds"))
            if row["update_wall_seconds"] < 0: raise ValueError("Negative external SGD wall time")
        elif before != after:
            raise ValueError("Frozen monitor/warmup changed parameters")
    elif (row.get("request_seed") is not None or before is not None or after is not None or e["decisions"] or e["rng_before"] is not None
          or e["rng_after"] is not None or e["parameter_sha256"] is not None or e["parameter_version"] is not None):
        raise ValueError("Native fixed unexpectedly has parameter/RNG/score state")
    if phase not in ("train","smoke_train") and any(row.get(k) is not None for k in
            ("baseline_return","rho","episode_return","gradient","learning_rate","update_wall_seconds")):
        raise ValueError("Frozen requests must not receive reward/baseline/SGD labels")
    return checked


def calibrate_fit(rows):
    expected = {(d,m) for d in FIT_DOCUMENTS for m in FIXED}
    seen,by_mode,canonical = {},{m:[] for m in FIXED},{}
    for row in rows:
        key = row.get("document"),row.get("policy")
        if key not in expected or key in seen or row.get("phase") != "calibration":
            raise ValueError("Calibration must cover fit16 x four Native fixed arms exactly once")
        e = row["episode"]; n = _integer(e.get("useful_n"),1); t = _finite(e.get("seconds"),positive=True)
        reference = {"output_token_ids":e["reference_output_token_ids"]}
        validate_episode_record(row,reference,eos_token_id=row.get("eos_token_id"))
        if e.get("exact") is not True or n != len(e["reference_output_token_ids"]):
            raise ValueError("Calibration must preserve useful canonical N")
        if key[0] in canonical and canonical[key[0]] != reference:
            raise ValueError("Calibration arms disagree on canonical output")
        canonical[key[0]] = reference
        seen[key] = (n,t); by_mode[key[1]].append(n/t)
    if set(seen) != expected: raise ValueError("Incomplete fit-only fixed calibration")
    fixed8 = [seen[d,"fixed8_native"] for d in FIT_DOCUMENTS]
    rho = sum(n for n,_ in fixed8)/math.fsum(t for _,t in fixed8)
    means = {m:math.fsum(v)/len(v) for m,v in by_mode.items()}
    best = max(FIXED,key=lambda m:(means[m],-int(m[5:].split("_")[0])))
    return {"schema":1,"context":CONTEXT,"fit_documents":list(FIT_DOCUMENTS),"rho":rho,
        "rho_source":"sum usefulN / sum whole-request seconds, NativeFixed8 fit16 only",
        "baseline_returns":{str(d):seen[d,"fixed8_native"][0]-rho*seen[d,"fixed8_native"][1] for d in FIT_DOCUMENTS},
        "fixed_mean_document_tps":means,"best_fixed":best,"tie_break":"smaller budget",
        "monitor_used":False,"baseline_frozen_before_training":True}


def timed_request(generate, *, synchronize, clock=time.perf_counter):
    """Caller supplies the entire spgenerate only; finish/audits remain outside."""
    synchronize(); start = clock()
    result = generate()
    synchronize(); elapsed = _finite(clock()-start,positive=True)
    return result,elapsed


def apply_episode_update(policy, episode, calibration, document, *, clock=time.perf_counter):
    """Explicit request-EXTERNAL ascent, never modify the sampled parameters."""
    if type(document) is not int or document not in FIT_DOCUMENTS:
        raise ValueError("Only fit documents may update parameters")
    rho = _finite(calibration["rho"],positive=True)
    baseline = _finite(calibration["baseline_returns"][str(document)])
    start = clock()
    gradient = episode_gradient(policy,[Decision.from_state(d) for d in episode["decisions"]],
        useful_n=episode["useful_n"],elapsed_s=episode["seconds"],rho=rho,baseline=baseline)
    updated = policy.sgd(gradient,.0005)
    elapsed = _finite(clock()-start)
    if elapsed < 0: raise ValueError("External update clock moved backwards")
    return updated,{"parameters_after":updated.to_state(),"baseline_return":baseline,"rho":rho,
        "episode_return":episode["useful_n"]-rho*episode["seconds"],
        "gradient":[list(r) for r in gradient.values],"learning_rate":.0005,"update_wall_seconds":elapsed}


def assert_gpu_boundary(expected_uuid, *, query=None, pid=None):
    """Fail closed on errors/warnings/malformed output, never touch other jobs."""
    uuid_pattern = r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
    if not isinstance(expected_uuid,str) or re.fullmatch(uuid_pattern,expected_uuid) is None:
        raise ValueError("Invalid physical UUID")
    query = subprocess.run if query is None else query
    result = query(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"],
                   capture_output=True,text=True,check=True,timeout=15)
    if result.stderr != "": raise RuntimeError("GPU observation emitted stderr")
    pairs = []
    for line in result.stdout.splitlines():
        match = re.fullmatch(r"\s*("+uuid_pattern+r")\s*,\s*([0-9]+)\s*",line)
        if match is None: raise RuntimeError("Malformed GPU process observation")
        pairs.append((match[1],int(match[2])))
    own = os.getpid() if pid is None else pid
    own_uuids = {u for u,p in pairs if p == own}
    if own_uuids != {expected_uuid} or {p for u,p in pairs if u == expected_uuid} != {own}:
        raise RuntimeError("Physical GPU contention/identity failure; no process will be interrupted")
    return expected_uuid


def wait_for_preload_gpu(args):
    """Read-only empty-GPU gate after CPU verification, before any CUDA call."""
    guard = Path(__file__).resolve().parents[1]/"csgb_episode_gpu_guard.py"
    return subprocess.run([sys.executable,str(guard),"--index",str(args.physical_gpu_index),
        "--uuid",args.expected_gpu_uuid],check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("source","scaler","protocol","base","draft","checkpoint-receipt","out","expected-gpu-uuid"):
        ap.add_argument("--"+name,required=True)
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--physical-gpu-index",type=int,choices=range(8),required=True)
    ap.add_argument("--validate-only",action="store_true")
    ap.add_argument("--smoke",action="store_true",help="Disposable doc0 seven-request pipeline test; never freeze or monitor")
    args = ap.parse_args()
    protocol = validate_pilot_protocol(args.protocol)
    if (args.seed != protocol["method"]["seed"] or protocol["hardware"] !=
            {"physical_gpu_index":args.physical_gpu_index,"expected_gpu_uuid":args.expected_gpu_uuid}
            or os.environ.get("CUDA_VISIBLE_DEVICES") != args.expected_gpu_uuid):
        raise ValueError("CLI seed/physical GPU differs from the frozen protocol")
    for name,key in ((args.source,"source_sha256"),(args.scaler,"scaler_sha256"),
                     (args.checkpoint_receipt,"checkpoint_receipt_sha256")):
        if sha(name) != protocol[key]: raise ValueError("CLI input differs from the frozen protocol")
    texts = load_train24(args.source)
    scaler,scaler_evidence = validate_pilot_scaler(args.scaler,protocol["source_sha256"])
    from csgb_online_evaluate_migrated import validate_physical_gpu, validate_checkpoint_receipt
    verify_loaded_sources(protocol["source_manifest"])
    validate_physical_gpu(args.physical_gpu_index,args.expected_gpu_uuid)
    receipt,weights = validate_checkpoint_receipt(args.checkpoint_receipt,args.base,args.draft,protocol["checkpoint_training_root"])
    out = Path(args.out)
    if out.exists() or out.is_symlink(): raise FileExistsError("Never resume/overwrite a partial pilot output")
    inputs = {str(Path(p).resolve()):sha(p) for p in (args.source,args.scaler,args.protocol,args.checkpoint_receipt)}
    immutable = {**protocol["source_manifest"],**inputs,**weights,**scaler_evidence}
    if args.validate_only:
        print(json.dumps({"read_only_preflight_passed":True,"cuda_imported":False,"output_created":False,
            "smoke":args.smoke,"fit_documents":[0] if args.smoke else list(FIT_DOCUMENTS),
            "monitor_documents":[] if args.smoke else list(MONITOR_DOCUMENTS),"context":CONTEXT}))
        return
    import numpy as np
    import torch
    from baseline_profiler import exact_input
    from classic.model_classic import SPModel
    from csgb_same_state import release_request_caches
    wait_for_preload_gpu(args)
    if torch.cuda.device_count() != 1: raise RuntimeError("Exactly one visible physical GPU per paired pilot")
    probe = torch.empty(1,device="cuda:0"); assert_gpu_boundary(args.expected_gpu_uuid); del probe
    torch.manual_seed(42); np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = SPModel.from_pretrained(base_model_path=args.base,draft_model_path=args.draft,
        torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map={"":0}).eval()
    assert_gpu_boundary(args.expected_gpu_uuid)
    model.base_model.float()
    if {str(p.device) for p in model.parameters()} != {"cuda:0"}: raise RuntimeError("Components left the proved worker GPU")
    assert_gpu_boundary(args.expected_gpu_uuid)
    h = EpisodeHarness(model)
    out.mkdir(exist_ok=False)
    exclusive_json(out/"protocol.json",{"schema":1,"runtime_version":1,
        "phase":"episode_smoke" if args.smoke else "episode_train_only_pilot","smoke":args.smoke,
        "pilot_protocol":protocol,"input_manifest":inputs,"checkpoint_manifest":weights,
        "scaler_evidence":scaler_evidence,"gpu_uuid":args.expected_gpu_uuid,
        "physical_gpu_index":args.physical_gpu_index,"source_manifest":protocol["source_manifest"],
        "timing":"whole request; explicit boundary sync only; engine/feature implicit sync remains; external SGD excluded",
        "production_speed_claim":False,"development_opened":False,"confirmation_opened":False})
    references,inputs_cache = {},{}

    def get_reference(doc):
        if doc not in FIT_DOCUMENTS+MONITOR_DOCUMENTS: raise ValueError("No dev/confirmation data path")
        if doc not in references:
            ids = exact_input(model.tokenizer,texts[doc],CONTEXT,torch,torch.device("cuda:0"))
            if int(ids.shape[-1]) != CONTEXT: raise RuntimeError("Actual prompt length is not4096")
            release_request_caches(model); assert_gpu_boundary(args.expected_gpu_uuid)
            with torch.inference_mode(): reference = model.ar_generate(ids,max_new_tokens=128,verbose=False,return_metrics=True)
            assert_gpu_boundary(args.expected_gpu_uuid)
            inputs_cache[doc] = (ids,hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest())
            references[doc] = reference
            exclusive_json(out/f"reference_{CONTEXT}_{doc}.json",reference)
        return inputs_cache[doc],references[doc]

    count = 0
    def run(doc,mode,phase,policy=None,*,pass_index=None,stream=None,timing_repeat=None,calibration=None):
        nonlocal count
        (ids,input_hash),reference = get_reference(doc)
        seed = None if policy is None else request_seed(protocol,phase,mode,doc,pass_index=pass_index,stream=stream)
        rng = None if seed is None else random.Random(seed)
        before = None if policy is None else policy.to_state()
        release_request_caches(model)
        h.reset(mode,policy,rng,phase=phase,context=CONTEXT,max_new_tokens=128,eos_token_id=model.tokenizer.eos_token_id,
                provenance={"document":doc,"context":CONTEXT,"phase":phase,"pass_index":pass_index,
                            "stream":stream,"timing_repeat":timing_repeat,"request_seed":seed})
        assert_gpu_boundary(args.expected_gpu_uuid)
        def generate():
            with torch.inference_mode():
                return model.spgenerate(ids,nodes=8,max_depth=10,threshold=.7,max_new_tokens=128,
                    temperature=0,use_specextend=True,retrieval_chunk_size=32,retrieve_top_k=32,
                    retrieve_every_n_steps=4,retrieval_num_queries=1,exact_target_attention=True,
                    return_output_ids=True,verbose=False,output_result_line=False)
        result,seconds = timed_request(generate,synchronize=torch.cuda.synchronize)
        assert_gpu_boundary(args.expected_gpu_uuid)
        episode = h.finish(result,reference,seconds)
        row = {"schema":1,"document":doc,"context":CONTEXT,"policy":mode,"phase":phase,
            "pass_index":pass_index,"stream":stream,"timing_repeat":timing_repeat,"request_seed":seed,
            "gpu_uuid":args.expected_gpu_uuid,"uncontended_at_request_boundaries":True,"input_sha256":input_hash,
            "eos_token_id":model.tokenizer.eos_token_id,"episode":episode,"parameters_before":before,
            "parameters_after":before,"baseline_return":None,"rho":None,"episode_return":None,
            "gradient":None,"learning_rate":None,"update_wall_seconds":None}
        updated = policy
        if phase in ("train","smoke_train"):
            updated,update = apply_episode_update(policy,episode,calibration,doc)
            row.update(update)
        validate_episode_record(row,reference,eos_token_id=model.tokenizer.eos_token_id,protocol=protocol)
        name = f"{phase}_{CONTEXT}_{doc}_{mode}_p{pass_index}_s{stream}_r{timing_repeat}.json"
        exclusive_json(out/name,row); count += 1
        print(json.dumps({"phase":phase,"document":doc,"policy":mode,"pass_index":pass_index,
            "stream":stream,"timing_repeat":timing_repeat,"exact":True}),flush=True)
        return row,updated

    if args.smoke:
        smoke_rows = [run(0,mode,"smoke_fixed")[0] for mode in FIXED]
        native8 = smoke_rows[0]["episode"]
        smoke_calibration = {"rho":native8["useful_n"]/native8["seconds"],"baseline_returns":{"0":0.}}
        for mode in MODES:
            run(0,mode,"smoke_train",make_policy(mode,scaler),pass_index=0,calibration=smoke_calibration)
        verify_manifest(immutable)
        exclusive_json(out/"DONE.json",{"raw_data_complete":True,"smoke":True,"requests_including_warmups":7,
            "raw_sha256":{str(p.resolve()):sha(p) for p in out.glob("*.json")},
            "no_final_models":True,"monitor_opened":False,"development_opened":False,"confirmation_opened":False,
            "study_complete":False,"figures_and_report_still_required":True,"production_speed_claim":False})
        return

    # Fixed calibration warms each arm once, without paying for fresh features.
    # It is saved separately and never contributes to rho/baseline/bestfixed.
    for mode in FIXED: run(0,mode,"fit_warmup",timing_repeat=-1)
    calibration_rows = [run(doc,mode,"calibration")[0] for doc in FIT_DOCUMENTS for mode in FIXED]
    calibration = calibrate_fit(calibration_rows)
    exclusive_json(out/"calibration.json",{**calibration,"request_sha256":{str(p.resolve()):sha(p) for p in out.glob("calibration_*.json")}})
    policies = {mode:make_policy(mode,scaler) for mode in MODES}
    exclusive_json(out/"training_order.json",{"passes":[training_order(protocol,p) for p in (0,1)]})
    for pass_index in (0,1):
        for mode,doc in training_order(protocol,pass_index):
            _,policies[mode] = run(doc,mode,"train",policies[mode],pass_index=pass_index,calibration=calibration)
    frozen = {"schema":1,"context":CONTEXT,"fit_documents":list(FIT_DOCUMENTS),
        "monitor_documents_used_for_training":False,"passes":2,"final_only_no_selection":True,
        "models":{mode:p.to_state() for mode,p in policies.items()},"calibration_sha256":sha(out/"calibration.json"),
        "scaler_sha256":sha(args.scaler),"source_manifest":protocol["source_manifest"],"input_manifest":inputs,
        "training_evidence":{str(p.resolve()):sha(p) for p in out.glob("*.json")},
        "development_opened":False,"confirmation_opened":False}
    verify_manifest(immutable)
    exclusive_json(out/"frozen_policies.json",frozen)
    for doc in MONITOR_DOCUMENTS:
        # Every policy/doc receives a disposable warmup; learned warmups also
        # sample softmax from a separate request stream, never mutate policies.
        for mode in POLICIES: run(doc,mode,"monitor_warmup",policies.get(mode),stream=0,timing_repeat=-1)
        paired_paths = {}
        for stream in range(3):
            for repeat in range(2):
                order = list(POLICIES)
                random.Random(protocol["method"]["order_seed"]+doc*100+stream*10+repeat).shuffle(order)
                for mode in order:
                    row,_ = run(doc,mode,"monitor",policies.get(mode),stream=stream,timing_repeat=repeat)
                    e = row["episode"]
                    fingerprint = (e["decisions"],e["result"]["output_token_ids"],
                        e["result"]["raw_accept_length_list"],e["result"]["accept_length_list"],
                        [r["budget"] for r in e["rows"]])
                    key = mode,stream
                    if key in paired_paths and paired_paths[key] != fingerprint:
                        raise RuntimeError("Same action stream did not replay across timing repeats")
                    paired_paths[key] = fingerprint
    if any(p.to_state() != frozen["models"][m] for m,p in policies.items()): raise RuntimeError("Frozen monitor parameters changed")
    verify_manifest(immutable)
    raw = {str(p.resolve()):sha(p) for p in out.glob("*.json")}
    exclusive_json(out/"DONE.json",{"raw_data_complete":True,"smoke":False,"requests_including_warmups":count,
        "calibration_requests":64,"training_requests":96,"monitor_requests":336,"warmup_requests":60,
        "raw_sha256":raw,"frozen_policies_sha256":sha(out/"frozen_policies.json"),
        "development_opened":False,"confirmation_opened":False,"study_complete":False,
        "figures_and_report_still_required":True,"production_speed_claim":False})


if __name__ == "__main__": main()
