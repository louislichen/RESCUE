"""94-request, sealed single-coefficient cost intervention; no model training.

The development file is neither opened nor hashed before calibration is sealed.
Only student_step changes. All historical verification/cache coefficients remain
untouched and must not be described as newly validated cost-model components.
"""
import argparse
import copy
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import time
import uuid

import torch

from .draft_graph_benchmark import HookOwners, owner_signature, record
from .draft_graph_smoke import COST_SHA256, native_digest, runtime_snapshot, tensor_digest
from .round_overhead_run import (MAPPER_SHA256, POLICY_SHA256, append, check_pair,
                                 context, gpu_snapshot, model_files, require, sha256, write)
from .specextend_match_pilot import check_output


ARMS=('oldrisk','newrisk','fixed4','retrieval32_k4_graph')
LENGTHS=(4096,8192)
DOCUMENTS=(271,272,288)
SEED=20261022
CALIBRATION_SHA='d0bf671ea6e54f81533def13a812fb00f4b27d9199fbf2e0b1d2a8623c90d5ad'
CALIBRATION_DOCUMENT_SHA='b0398d17c8500c6b24997927377e4a431042794a8a22aaaf56a9d7921ec16ad9'
DEVELOPMENT_SHA='85b2eb1a81d1617b2d158c069c420c452459a8bdb90da128f257220cc27ded09'
PREREGISTRATION='exp/draft_graph/cost_recalibration_20261020_preregistration.json'
PIPELINE='exp/draft_graph/run_cost_recalibration_20261020_v2.sh'
EXPECTED=dict(calibration_reference_requests=2,calibration_prime_requests=2,
    calibration_profile_requests=4,calibration_owners=2,reference_requests=6,
    correctness_requests=24,prime_requests=8,timing_requests=48,owners=8)


def progress(stage,counts,**fields):
    print(json.dumps(dict(stage=stage,counts=dict(counts),**fields)),flush=True)


def source_inventory(root):
    paths=list((root/'rescue').glob('*.py'))+[
        root/'exp/attention_horizon/selection.py',root/'exp/selective_kv/bridge.py',
        root/PREREGISTRATION,root/PIPELINE,root/'tests/test_rescue_draft_cost_recalibration.py',
        root/'tests/test_rescue_draft_cost_recalibration_report.py']
    require(all(p.is_file() for p in paths),'Missing preregistered runner/source/test/pipeline')
    return {str(p.relative_to(root)):sha256(p) for p in sorted(set(paths))}


def check_source_bindings(root):
    """Prove Python executes the frozen tree, not another PYTHONPATH namespace."""
    root=root.resolve();receipt={}
    modules=('rescue.engine','rescue.config','rescue.selection','rescue.query_mapper',
             'rescue.step_budget','rescue.loading','exp.attention_horizon.selection',
             'exp.selective_kv.bridge')
    for name in modules:
        module=importlib.import_module(name);filename=getattr(module,'__file__',None)
        require(filename is not None,'Imported source has no file binding: '+name)
        actual=Path(filename).resolve();expected=root/Path(*name.split('.')).with_suffix('.py')
        require(actual==expected,'Imported source escaped frozen root: '+name)
        receipt[name]=dict(path=str(actual),source_sha256=sha256(actual))
    return receipt


def read_calibration(path,expected_sha=CALIBRATION_SHA,document_sha=CALIBRATION_DOCUMENT_SHA):
    require(sha256(path)==expected_sha,'Calibration file SHA mismatch')
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(len(rows)==1 and rows[0].get('source_index')==246 and rows[0].get('split')=='train',
            'Require exactly original-corpus train document246')
    require(rows[0].get('document_sha256')==document_sha,'Calibration document SHA mismatch')
    for n in LENGTHS:context(rows[0],n)
    return rows[0]


def validate_seal(output,expected_manifest_sha=None):
    path=output/'seal_manifest.json'
    require(path.is_file(),'Cost must be sealed before development file access')
    if expected_manifest_sha is not None:
        require(sha256(path)==expected_manifest_sha,'Cost seal manifest changed')
    manifest=json.loads(path.read_text())
    required={'cost_old.json','cost_new.json','fit_receipt.json','COST_SEALED.json'}
    require(set(manifest.get('file_sha256',{}))==required,'Incomplete immutable cost seal')
    for name,digest in manifest['file_sha256'].items():
        require(sha256(output/name)==digest,'Sealed cost/receipt changed: '+name)
    sealed=json.loads((output/'COST_SEALED.json').read_text())
    require(sealed.get('seal_execution_index')==8,'Cost was not sealed after exactly8 calibration requests')
    return sealed


def read_development(path,output,manifest_sha,expected_sha=DEVELOPMENT_SHA):
    validate_seal(output,manifest_sha)  # before even hashing/opening the file
    digest=sha256(path)
    require(digest==expected_sha,'Development cohort SHA mismatch; never reselect')
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(len(rows)==3 and {r.get('source_index') for r in rows}==set(DOCUMENTS),
            'Require exactly locked development documents271/272/288')
    require(all(r.get('split')=='train' for r in rows),'Development is reused original train corpus, never test')
    require(len({r.get('document_sha256') for r in rows})==3,'Development document duplication')
    for row in rows:
        require(isinstance(row.get('document_sha256'),str) and len(row['document_sha256'])==64,'Missing development document digest')
        for n in LENGTHS:context(row,n)
    return {r['source_index']:r for r in rows},digest


def fit_student_cost(old_cost,profiles):
    """Repeat0 pooled sample median; repeat1 is only a repeatability gate."""
    keyed={}
    for row in profiles:
        key=row.get('length'),row.get('repeat')
        require(key not in keyed and key[0] in LENGTHS and key[1] in (0,1),'Unexpected/duplicate calibration profile')
        require(row.get('source_index')==246 and row.get('mode')=='oldrisk','Calibration policy/document changed')
        delta=row.get('draft_runtime',{}).get('request',{})
        require(delta.get('captures')==0 and delta.get('capture_host_seconds')==0 and delta.get('replays')==row.get('draft_forwards'),
                'Calibration warm capture/replay gate failed; do not fit')
        samples=row.get('cost_samples',{}).get('student_step')
        require(isinstance(samples,list) and len(samples)==row.get('proposed') and samples and
            all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in samples),
            'Invalid raw per-Draft-step calibration samples')
        keyed[key]=samples
    require(set(keyed)=={(n,r) for n in LENGTHS for r in (0,1)},'Incomplete4-profile calibration matrix')
    fitting=[v for n in LENGTHS for v in keyed[n,0]]
    measured=statistics.median(fitting)
    validation={}
    for n in LENGTHS:
        median=statistics.median(keyed[n,1]);error=abs(measured-median)/median
        validation[str(n)]=dict(sample_count=len(keyed[n,1]),median=median,relative_error=error,passed=error<=.20)
    require(all(v['passed'] for v in validation.values()),'Repeat1 context-median relative error exceeds20%; no fit/seal permitted')
    require(type(old_cost.get('student_step')) in (int,float) and old_cost['student_step']>0,'Missing historical student_step')
    new_cost=copy.deepcopy(old_cost);new_cost['student_step']=measured
    require({k:v for k,v in new_cost.items() if k!='student_step'}=={k:v for k,v in old_cost.items() if k!='student_step'},
            'A coefficient other than student_step changed')
    return new_cost,dict(sample_count=len(fitting),median=measured),validation


def seal_cost(output,old_path,new_cost,fit,validation,policy_metadata):
    require(not (output/'seal_manifest.json').exists(),'Cost already sealed; never refit')
    with (output/'cost_old.json').open('xb') as handle:
        handle.write(old_path.read_bytes());handle.flush();os.fsync(handle.fileno())
    write(output/'cost_new.json',new_cost)
    receipt=dict(fit=fit,validation=validation,fit_repeats=[0],validation_repeats=[1],
        validation_denominator='Repeat1 context sample median',seal_execution_index=8,
        old_sha256=sha256(old_path),new_sha256=sha256(output/'cost_new.json'),policy_metadata=policy_metadata,
        calibration_file_sha256=CALIBRATION_SHA,calibration_document_sha256=CALIBRATION_DOCUMENT_SHA,
        original_corpus_role='train, reused calibration document; repeat1 is NOT independent held-out validation',
        unchanged_coefficients={k:v for k,v in new_cost.items() if k!='student_step'},
        legacy_verify_coefficients_unchanged=True,
        limitation='verify_beta/fixed_round/cache_update and every other field remain historical, not recalibrated; profile includes model/argmax host synchronization and required M2 summaries, ends before M3 topk/features/controller')
    write(output/'fit_receipt.json',receipt)
    write(output/'COST_SEALED.json',dict(receipt,fit_receipt_sha256=sha256(output/'fit_receipt.json')))
    names=('cost_old.json','cost_new.json','fit_receipt.json','COST_SEALED.json')
    write(output/'seal_manifest.json',dict(file_sha256={name:sha256(output/name) for name in names},
        development_unopened=True,seal_execution_index=8))
    return sha256(output/'seal_manifest.json')


def schedule(seed=SEED):
    result=[]
    for n in LENGTHS:
        rng=random.Random(seed+n)
        for repeat in range(2):
            docs=list(DOCUMENTS);rng.shuffle(docs)
            for index in docs:
                arms=list(ARMS);rng.shuffle(arms)
                result.extend(dict(mode=arm,source_index=index,length=n,repeat=repeat) for arm in arms)
    return result


def arm_config(config,arm,profile=False):
    require(arm in ARMS,'Unknown preregistered arm')
    changes=dict(length_policy='fixed' if arm in ('fixed4','retrieval32_k4_graph') else 'risk',
                 fixed_k=4,profile=profile)
    if arm=='retrieval32_k4_graph':changes.update(inject_tokens=0,injection_strength=0.)
    return replace(config,**changes)


def detach_quality_observer(engine):
    """Break the observer/Engine cycle before closing request-owned GPU buffers."""
    observer=getattr(engine,'round_observer',None)
    if getattr(observer,'engine',None) is engine:observer.engine=None


class QualityObserver:
    """Observe real V changes without turning fixed4's M3 summaries back on."""
    step_budget=False

    def __init__(self):
        self.engine=None;self.rounds=[];self.nonzero_injection=False

    def __call__(self,packet):
        engine=self.engine;vector=packet.get('budget_features')
        vector=None if vector is None else vector.detach().float().cpu().tolist()
        evidence=[];injector=getattr(engine,'injector',None)
        for layer,active in (injector.active.items() if injector is not None else []):
            indices=active[1];absolute=engine.working.positions[indices]
            working=engine.working.values[layer][0,:,indices].transpose(0,1).flatten(1).contiguous()
            native=engine.native_s.values[layer][0,:,absolute].transpose(0,1).flatten(1).contiguous().to(working.dtype)
            matches=torch.equal(native,active[3].to(working.dtype))
            require(matches,'Injection proof gathered a different Native position/value')
            different=int(torch.count_nonzero(working!=native))
            self.nonzero_injection|=different>0
            evidence.append(dict(layer=layer,positions=absolute.detach().cpu().tolist(),
                working_value=tensor_digest(working),native_value=tensor_digest(native),
                different_elements=different,native_matches_memo=matches))
        self.rounds.append(dict(round_index=packet['round_index'],prefix=packet['prefix'],
            progress=packet['progress'],remaining=packet['remaining'],selected=packet['selected'].detach().cpu().tolist(),
            budget_features=vector,draft=list(packet['draft']),checks=list(packet['checks']),accepted=packet['accepted'],
            injection_evidence=evidence,summary_count=len(injector.summaries) if injector is not None else 0))


@contextmanager
def record_draft_steps():
    from .attention import DraftAttention
    original=DraftAttention.step;records=[]
    def step(owner,token,native,logits=True):
        result=original(owner,token,native,logits=logits)
        records.append(dict(logits=logits,query=tensor_digest(owner.last_query),
            summary_requested=bool(getattr(owner,'record_injection_summary',False)),
            summary_count=len(owner.injector.summaries) if owner.injector is not None else 0))
        return result
    DraftAttention.step=step
    try:yield records
    finally:DraftAttention.step=original


def run(args):
    from .baselines import BaselineEngine
    from .config import SystemConfig
    from .draft_graph import ResidentDraftRuntime
    from .engine import RescueEngine
    from .loading import load_models
    from .packing import pack_projections
    from .query_mapper import mapper_from_payload
    from .step_budget import StepBudget

    root=Path(__file__).resolve().parents[1];output=args.output.resolve()
    require(not output.exists(),'Fresh isolated output required; never resume a partial intervention')
    for name in ('teacher','student'):
        require(not output.is_relative_to(getattr(args,name).resolve()),'Output overlaps immutable model root')
    output.mkdir(parents=True)
    counts=dict.fromkeys(EXPECTED,0);seen=set();group=None;live=[];seal_sha=None
    previous_limit=ResidentDraftRuntime.MAX_ENTRIES
    try:
        for p in (args.teacher,args.student,args.mapper,args.policy,args.cost,args.calibration_file):
            require(p.is_absolute() and p.exists(),'Explicit existing absolute calibration/model paths required')
        require(args.development_file.is_absolute(),'Development path must be explicit; it is not opened before seal')
        require(sha256(args.mapper)==MAPPER_SHA256 and sha256(args.policy)==POLICY_SHA256 and sha256(args.cost)==COST_SHA256,
                'Frozen Mapper/policy/cost asset mismatch')
        calibration=read_calibration(args.calibration_file)
        cal_inputs={n:context(calibration,n) for n in LENGTHS}
        sources=source_inventory(root);bindings=check_source_bindings(root)
        prereg=json.loads((root/PREREGISTRATION).read_text())
        for name in sources:
            target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,target)
        asset_paths=sorted(set(model_files(args.teacher)+model_files(args.student)+[args.mapper,args.policy,args.cost,args.calibration_file]))
        assets={str(p):sha256(p) for p in asset_paths}  # deliberately excludes unopened development file
        old_cost=json.loads(args.cost.read_text())
        write(output/'calibration_cost_payload.json',old_cost)  # retain raw coefficients even if calibration fails
        payload=torch.load(args.policy,map_location='cpu',weights_only=True)
        require(payload.get('kind')=='step_budget' and 'seed' in payload,'Expected frozen StepBudget including its original seed')
        policy_metadata={k:payload[k] for k in ('kind','hidden','margin','prior_weight','seed')}
        policy_metadata['state_dict_sha256']={k:tensor_digest(v)['sha256'] for k,v in payload['state_dict'].items()}
        torch.set_num_threads(4);torch.manual_seed(SEED)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        environment=gpu_snapshot()
        require(prereg['gpu_uuid'] in environment.get('nvidia_smi_gpu','') or prereg['gpu_uuid']==environment.get('gpu_uuid'),
                'Unexpected physical GPU UUID')
        teacher,student,tokenizer=load_models(str(args.teacher),str(args.student),'cuda:0',target_dtype=torch.float32,efficient=True)
        for model in (teacher,student):
            model.eval().requires_grad_(False)
            require(all(p.device==torch.device('cuda:0') for p in model.parameters()),'Model escaped single visible cuda:0')
        pack_projections(teacher);pack_projections(student);group=HookOwners(teacher,student)
        mapper_payload=torch.load(args.mapper,map_location='cpu',weights_only=True)
        def mapper():return mapper_from_payload(mapper_payload,'cuda:0').eval().requires_grad_(False)
        def policy():
            result=StepBudget.from_payload(payload,'cuda:0').eval().requires_grad_(False)
            require(result.margin==payload['margin'] and result.prior_weight==payload['prior_weight'] and result.seed==payload['seed'],
                    'Loaded policy hyperparameters changed')
            return result
        ResidentDraftRuntime.MAX_ENTRIES=128
        config=SystemConfig(selector='hierarchical',coverage_rows=64,refresh_every=4,refresh_policy='periodic',
            injection_layers=(27,),inject_tokens=64,injection_strength=.5,prefill_precision='bfloat16',
            max_new_tokens=128,max_draft=12,length_policy='risk',schedule='serial',profile=False,
            draft_execution='cuda_graph',coverage_execution='eager')
        calibration_config=replace(config,profile=True)
        calibration_mapper,calibration_policy=mapper(),policy()
        cal_protocol=dict(schema=1,expected_counts=EXPECTED,expected_requests=94,source_sha256=sources,
            source_bindings=bindings,
            preregistration_sha256=sources[PREREGISTRATION],calibration_file_sha256=CALIBRATION_SHA,
            calibration_document_sha256=CALIBRATION_DOCUMENT_SHA,calibration_role='Original corpus train; reused doc246, not held-out',
            calibration_config=asdict(calibration_config),environment=environment,assets_sha256=assets,
            mapper_sha256=MAPPER_SHA256,policy_sha256=POLICY_SHA256,old_cost_sha256=COST_SHA256,
            mapper_frozen=True,policy_frozen=True,no_training=True,no_model_training=True,
            policy_metadata=policy_metadata,development_content_opened=False,fit_rule='Repeat0 pooled per-step median only; repeat1 each-context relative-error gate20%',
            args={k:str(v) for k,v in vars(args).items()})
        write(output/'protocol_calibration.json',cal_protocol);progress('calibration_protocol_frozen',counts)

        def seed_request():
            torch.manual_seed(policy_metadata['seed']);random.seed(policy_metadata['seed'])
        def make(arm,cost,observer=None,profile=False):
            group.idle()
            if arm=='retrieval32_k4_graph':
                engine=BaselineEngine(teacher,student,arm_config(config,arm,profile),
                    vocab_size=len(tokenizer),top_chunks=32,round_observer=observer)
            else:
                cfg=arm_config(config,arm,profile)
                engine=RescueEngine(teacher,student,cfg,mapper(),None if arm=='fixed4' else policy(),cost,
                    vocab_size=len(tokenizer),round_observer=observer)
            group.add(engine);live.append(engine)
            if observer is not None:observer.engine=engine
            return engine
        def close(engine):
            detach_quality_observer(engine)
            group.close(engine);live.remove(engine)
        def enrich(row,engine,arm,index,n,reference):
            row.update(mode=arm,source_index=index,length=n,execution_index=len(seen),
                output_agrees=row['generated']==reference,runtime_stats=runtime_snapshot(engine),
                owner_signature=owner_signature(engine),m3_disabled=arm in ('fixed4','retrieval32_k4_graph'),
                m3_summary_policy='off' if arm in ('fixed4','retrieval32_k4_graph') else 'real frozen StepBudget first-two-token summaries',
                policy_seed=policy_metadata['seed'])
            return row
        def generated(engine,ids):
            gpu_snapshot();seed_request();start=time.perf_counter()
            row=group.generate(engine,ids,tokenizer.eos_token_id)
            return row,time.perf_counter()-start
        def graph_gate(row,cold=False,zero_capture=False):
            delta=row['draft_runtime']['request']
            require(delta['replays']==row['draft_forwards'] and delta['replays']>0,'Graph replay/work mismatch')
            if cold:require(delta['captures']>0 and delta['capture_host_seconds']>0,'Fresh owner borrowed graphs')
            if zero_capture:require(delta['captures']==0 and delta['capture_host_seconds']==0,'Calibration warm capture forbidden; no fit')
        def ar(index,n,ids,name,counter):
            gpu_snapshot();seed_request();group.idle()
            with BaselineEngine(teacher,student,replace(config,length_policy='fixed',draft_execution='eager'),vocab_size=len(tokenizer)) as engine:
                row=engine.reference(ids,tokenizer.eos_token_id)
            row.update(request_id=uuid.uuid4().hex,source_index=index,length=n,mode='canonical_ar',execution_index=len(seen))
            record(output,name,row,counts,counter,seen);gpu_snapshot()
            return row['generated']
        def finish_owner(owner,engine,filename,counter):
            owner['final_runtime_stats']=runtime_snapshot(engine)
            start=time.perf_counter();close(engine);end=time.perf_counter()
            owner.update(close_seconds=end-start,owner_lifetime_wall_seconds=end-owner.pop('owner_start'),
                closed_runtime_stats=runtime_snapshot(engine),engine_closed=engine.closed,hook_restored=True)
            require(owner['engine_closed'] and owner['closed_runtime_stats']['entries']==0,'Graph owner did not release')
            if counter=='owners':
                seconds=owner['constructor_and_detach_seconds']+owner['cold_prime_outer_seconds']+owner['warm_outer_seconds_sum']+owner['close_seconds']
                owner.update(accounted_service_seconds=seconds,accounted_service_tokens=owner['prime_tokens']+owner['warm_tokens'],
                    accounted_service_tps=(owner['prime_tokens']+owner['warm_tokens'])/seconds,
                    scope='Non-contiguous measured constructor+cold+6warm+close service sum, not production wall')
            append(output/filename,owner);counts[counter]+=1

        cal_refs={str(n):ar(246,n,cal_inputs[n],'calibration_reference.jsonl','calibration_reference_requests') for n in LENGTHS}
        write(output/'calibration_references.json',cal_refs)
        calibration_profiles=[]
        for n in LENGTHS:
            gpu_snapshot();tick=time.perf_counter()
            engine=RescueEngine(teacher,student,calibration_config,calibration_mapper,calibration_policy,old_cost,vocab_size=len(tokenizer))
            group.add(engine);live.append(engine);constructor=time.perf_counter()-tick
            owner_id=uuid.uuid4().hex
            try:
                prime,elapsed=generated(engine,cal_inputs[n]);enrich(prime,engine,'oldrisk',246,n,cal_refs[str(n)])
                prime.update(phase='calibration_cold_prime',owner_id=owner_id,performance_eligible=False,
                    profile_enabled=True,measured_generate_seconds=elapsed,constructor_and_detach_seconds=constructor)
                record(output,'calibration_prime.jsonl',prime,counts,'calibration_prime_requests',seen)
                check_output(prime,cal_refs[str(n)],tokenizer.eos_token_id);graph_gate(prime,cold=True)
                owner=dict(mode='oldrisk',length=n,owner_id=owner_id,owner_start=tick,owner_signature=prime['owner_signature'],
                    constructor_and_detach_seconds=constructor,prime_request_id=prime['request_id'],profile_request_ids=[],
                    cold_prime_profile_seconds=elapsed,warm_profile_seconds_sum=0.,calibration_cost_only=True)
                for repeat in range(2):
                    row,elapsed=generated(engine,cal_inputs[n]);enrich(row,engine,'oldrisk',246,n,cal_refs[str(n)])
                    row.update(phase='calibration_profile',repeat=repeat,owner_id=owner_id,performance_eligible=False,
                        profile_enabled=True,profile_outer_seconds=elapsed,fit_eligible=repeat==0,validation_only=repeat==1)
                    record(output,'calibration_profiles.jsonl',row,counts,'calibration_profile_requests',seen)
                    check_output(row,cal_refs[str(n)],tokenizer.eos_token_id);check_pair(prime,row);graph_gate(row,zero_capture=True)
                    require(row['owner_signature']==owner['owner_signature'],'Calibration persistent owner changed')
                    calibration_profiles.append(row);owner['profile_request_ids'].append(row['request_id']);owner['warm_profile_seconds_sum']+=elapsed
                    gpu_snapshot();progress('calibration_profile_complete',counts,length=n,repeat=repeat)
                finish_owner(owner,engine,'calibration_owners.jsonl','calibration_owners')
            finally:
                if engine in live:close(engine)
        require(len(seen)==8,'Cost must be fit only after all8 calibration requests')
        require(source_inventory(root)==sources and {str(p):sha256(p) for p in asset_paths}==assets,'Source/assets changed before fitting')
        new_cost,fit,validation=fit_student_cost(old_cost,calibration_profiles)
        seal_sha=seal_cost(output,args.cost,new_cost,fit,validation,policy_metadata)
        validate_seal(output,seal_sha);progress('cost_sealed_before_development_access',counts,student_step=new_cost['student_step'],seal_sha256=seal_sha)

        documents,development_sha=read_development(args.development_file,output,seal_sha)
        assets[str(args.development_file)]=development_sha;asset_paths.append(args.development_file)
        inputs={(i,n):context(documents[i],n) for i in DOCUMENTS for n in LENGTHS}
        plan=schedule();configs={arm:asdict(arm_config(config,arm)) for arm in ARMS}
        write(output/'protocol.json',dict(cal_protocol,modes=list(ARMS),source_indices=list(DOCUMENTS),lengths=list(LENGTHS),repeats=2,
            max_new_tokens=128,configs=configs,new_cost_sha256=sha256(output/'cost_new.json'),development_file_sha256=development_sha,
            development_roles={str(i):documents[i]['split'] for i in DOCUMENTS},development_content_opened_after_seal=True,
            calibration_seal_sha256=seal_sha,cost_sealed_sha256=sha256(output/'COST_SEALED.json'),
            seal_execution_index=8,assets_sha256=assets,warm_order=plan,retrieval_top_chunks=32,
            input_sha256={f'{i}:{n}':hashlib.sha256(json.dumps(ids).encode()).hexdigest() for (i,n),ids in inputs.items()},
            document_sha256={str(i):documents[i]['document_sha256'] for i in DOCUMENTS},
            scope='Three previously used development documents from original train corpus, NOT held-out; one cost coefficient only; no full SpecExtend superiority claim',
            policy_metadata=policy_metadata,graph_max_entries=128,mapper_frozen=True,policy_frozen=True,
            no_training=True,no_model_training=True,primary='Unprofiled full-generate outer, includes legal captures/hook handoff/prefill/reset/cleanup; model loading excluded',
            fixed4='Real mandatory nonzero M2; risk=None, fixed config, no summary requests/features/controller even during quality'))
        refs={f'{i}:{n}':ar(i,n,inputs[i,n],'reference.jsonl','reference_requests') for n in LENGTHS for i in DOCUMENTS}
        write(output/'references.json',refs);quality={}
        for n in LENGTHS:
            for i in DOCUMENTS:
                for arm in ARMS:
                    validate_seal(output,seal_sha);observer=QualityObserver();engine=make(arm,new_cost if arm=='newrisk' else old_cost,observer)
                    try:
                        with record_draft_steps() as steps:
                            row,_=generated(engine,inputs[i,n])
                        enrich(row,engine,arm,i,n,refs[f'{i}:{n}'])
                        row.update(phase='correctness',owner_id=uuid.uuid4().hex,performance_eligible=False,
                            observed_rounds=observer.rounds,queries=steps,nonzero_injection=observer.nonzero_injection,
                            native_t=native_digest(engine.native_t),native_s=native_digest(engine.native_s))
                        record(output,'correctness.jsonl',row,counts,'correctness_requests',seen)
                        check_output(row,refs[f'{i}:{n}'],tokenizer.eos_token_id);graph_gate(row,cold=True)
                        require(len(steps)==row['draft_forwards'],'Missing every-step Query evidence')
                        if arm=='retrieval32_k4_graph':require(not observer.nonzero_injection,'Retrieval baseline injected Mapper content')
                        else:require(observer.nonzero_injection,'Mandatory M2 produced no actual nonzero working-V correction')
                        if arm in ('fixed4','retrieval32_k4_graph'):
                            require(all(not q['summary_requested'] and q['summary_count']==0 for q in steps) and
                                all(r['budget_features'] is None and r['summary_count']==0 for r in observer.rounds),
                                'Fixed/baseline arm accidentally enabled M3 summaries/features')
                        quality[arm,i,n]=row;progress('development_quality_complete',counts,mode=arm,source_index=i,length=n)
                    finally:close(engine)
        require(source_inventory(root)==sources and {str(p):sha256(p) for p in asset_paths}==assets,'Source/assets changed before timing')
        write(output/'QUALITY_PASSED.json',dict(passed=True,requests=24,per_arm_canonical=True,
            cross_arm_trajectories_not_required=True,actual_nonzero_fixed4_and_risk=True,fixed4_m3_summary_disabled=True))
        for n in LENGTHS:
            active={};ledgers={};context_start=time.perf_counter()
            try:
                order=list(ARMS);random.Random(SEED+n+1).shuffle(order)
                for arm in order:
                    validate_seal(output,seal_sha);gpu_snapshot();tick=time.perf_counter()
                    engine=make(arm,new_cost if arm=='newrisk' else old_cost);constructor=time.perf_counter()-tick;active[arm]=engine
                    row,elapsed=generated(engine,inputs[271,n]);enrich(row,engine,arm,271,n,refs[f'271:{n}'])
                    owner_id=uuid.uuid4().hex;row.update(phase='cold_prime',owner_id=owner_id,performance_eligible=True,
                        outer_seconds=elapsed,outer_tps=row['tokens']/elapsed,constructor_and_detach_seconds=constructor,
                        cold_complete_outer_seconds=constructor+elapsed)
                    record(output,'prime.jsonl',row,counts,'prime_requests',seen)
                    check_output(row,refs[f'271:{n}'],tokenizer.eos_token_id);check_pair(quality[arm,271,n],row);graph_gate(row,cold=True)
                    ledgers[arm]=dict(mode=arm,length=n,owner_id=owner_id,owner_start=tick,owner_signature=row['owner_signature'],
                        constructor_and_detach_seconds=constructor,prime_request_id=row['request_id'],prime_tokens=row['tokens'],
                        cold_prime_outer_seconds=elapsed,warm_tokens=0,warm_outer_seconds_sum=0.,warm_request_ids=[])
                for item in (r for r in plan if r['length']==n):
                    arm,i=item['mode'],item['source_index'];owner=ledgers[arm];engine=active[arm]
                    validate_seal(output,seal_sha);row,elapsed=generated(engine,inputs[i,n]);enrich(row,engine,arm,i,n,refs[f'{i}:{n}'])
                    row.update(**item,phase='warm',owner_id=owner['owner_id'],performance_eligible=True,
                        outer_seconds=elapsed,outer_tps=row['tokens']/elapsed)
                    record(output,'runtime.jsonl',row,counts,'timing_requests',seen)
                    check_output(row,refs[f'{i}:{n}'],tokenizer.eos_token_id);check_pair(quality[arm,i,n],row);graph_gate(row)
                    require(row['owner_signature']==owner['owner_signature'],'Timing owner/cache pointers changed')
                    require('timeline' not in row and 'cost_samples' not in row,'Profile data leaked into primary TPS')
                    owner['warm_tokens']+=row['tokens'];owner['warm_outer_seconds_sum']+=elapsed;owner['warm_request_ids'].append(row['request_id'])
                    gpu_snapshot();progress('development_warm_complete',counts,**item)
                for arm,engine in active.items():finish_owner(ledgers[arm],engine,'owners.jsonl','owners')
                append(output/'contexts.jsonl',dict(length=n,context_wall_seconds=time.perf_counter()-context_start,
                    tokens=sum(o['prime_tokens']+o['warm_tokens'] for o in ledgers.values()),scope='All4interleaved arms plus audit I/O and close, diagnostic only'))
            finally:
                for engine in active.values():
                    if engine in live:close(engine)
        require(counts==EXPECTED and len(seen)==94,'Incomplete94-request intervention')
        validate_seal(output,seal_sha)
        require(source_inventory(root)==sources and {str(p):sha256(p) for p in asset_paths}==assets,'Frozen source/assets changed')
        require(check_source_bindings(root)==bindings,'Imported source bindings changed')
        group.idle();gpu_snapshot()
        write(output/'COMPLETE.json',dict(passed=True,counts=counts,expected_counts=EXPECTED,logical_requests=94,
            sources_unchanged=True,assets_unchanged=True,cost_seal_unchanged=True,calibration_seal_sha256=seal_sha,
            analysis_required=True,scope='Reused3-document development cost intervention; not a held-out/SOTA/full-cost-model validation'))
        progress('intervention_complete',counts)
    except BaseException as error:
        cleanup=[]
        if group is not None:
            for engine in list(live):
                try:
                    detach_quality_observer(engine);group.close(engine)
                except BaseException as failure:cleanup.append(repr(failure))
        write(output/'FAILURE.json',dict(passed=False,error_type=type(error).__name__,error=str(error),counts=counts,
            cost_sealed=seal_sha is not None,cleanup_errors=cleanup,scope='Preserve partial data; no refit/reselection/restart'))
        raise
    finally:ResidentDraftRuntime.MAX_ENTRIES=previous_limit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('teacher','student','mapper','policy','cost','calibration-file','development-file','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())


if __name__=='__main__':main()
