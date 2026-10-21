"""Frozen-Mapper six-context development curve; no deployment/restart mechanism.

All model code comes from the explicitly pinned old capsule. This wrapper only
orchestrates798 new calls; it never calls an older study's run/execute_study,
reads new corpus data, tunes a policy, or interprets curve success as final GO.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import copy
from dataclasses import asdict
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import random
import shutil
import time
from types import SimpleNamespace
import uuid

LENGTHS = (2048,4096,6144,8192,12288,16384)
OLD_LENGTHS = (4096,8192)
NEW_LENGTHS = tuple(n for n in LENGTHS if n not in OLD_LENGTHS)
DOCS = (211,213,220,226,233,236,246,248,253,271,272,288,303,306,327,604)
MODES = ('rescue_graph','retrieval32_k4_graph','tree32_n8')
SEED = 20261028
EXPECTED = dict(reference_requests=64,own_quality_requests=128,baseline_quality_requests=12,
                quality_owners=140,prime_requests=18,timing_requests=576,owners=18)
TOTAL = 798
ORDER = ((0,1,2),(1,2,0),(2,0,1),(0,2,1),(2,1,0),(1,0,2))
OLD_ROOT = Path('/mnt/8T/cl/rescue_experiment_cache/frozen_devcheck_20261020_v1')
SOURCE_ROOT = OLD_ROOT/'capsule/after'
OUTPUT_ROOT = Path('/mnt/8T/cl/rescue_experiment_cache/context_curve_20261020_v1')
EXECUTION_OUTPUT = OUTPUT_ROOT/'results'
SOURCE_MANIFEST_SHA = '002545801b8a15fdb94afb083c61dd76633e423ec828f7a3d7e2c1fc6fbaf40a'
HELPER_SHA = 'ce4f06ab60e34bf27b448dafe5f4574332557cda9e8b9733d697e8a4208a0069'
ASSET_SHA = dict(mapper='d08889e796ca2527e23241ff34a072ef640e2fdf3fe9a2f9ba5439975af04d9f',
    policy='3c333a470119c236afcc1308c18592842a535bee0b8f83c4fb4da359ff17eff5',
    cost='d92c843a84676c0cd6adaf9364bbf7e8c771d56548f1016b5720a3e04f371fa1')


def require(ok,message):
    if not ok: raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def fingerprint(ids): return hashlib.sha256(json.dumps(ids).encode()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as f: json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def append(path,value):
    with Path(path).open('a') as f: f.write(json.dumps(value,allow_nan=False)+'\n')


def rows(path):
    raw=Path(path).read_bytes()
    require(raw and raw.endswith(b'\n') and all(v.strip() for v in raw.splitlines()),'Incomplete JSONL')
    values=[json.loads(v) for v in raw.splitlines()]
    require(all(isinstance(v,dict) for v in values),'JSONL must contain objects')
    return values


def no_links(path):
    p=Path(path)
    require(p.is_absolute() and '..' not in p.parts and not any(v.is_symlink() for v in (p,*p.parents)),
            'Noncanonical or symlink path')


def pinned_inventory(root,manifest):
    no_links(root);require(isinstance(manifest,dict) and bool(manifest),'Empty frozen inventory')
    for name,digest in manifest.items():
        p=PurePosixPath(name)
        require(p.as_posix()==name and not p.is_absolute() and '..' not in p.parts and '\\' not in name,
                'Unsafe frozen relative path')
        path=root/name;no_links(path)
        require(path.is_file() and sha(path)==digest,'Frozen bytes changed: '+name)
    return manifest


def validate_protocol(p):
    expected=dict(schema=1,status='FROZEN_BEFORE_EXECUTION',seed=SEED,lengths=list(LENGTHS),source_indices=list(DOCS),
        modes=list(MODES),repeats=2,max_new_tokens=128,graph_max_entries=128,new_generation_requests=TOTAL,
        reused_reference_requests=32,reused_own_quality_requests=64,prime_source_index=271,retrieval_quality_source_index=211,
        source_root=str(SOURCE_ROOT),source_manifest_sha256=SOURCE_MANIFEST_SHA,old_root=str(OLD_ROOT),output_root=str(OUTPUT_ROOT),execution_output=str(EXECUTION_OUTPUT),
        owner_order=[[MODES[i] for i in order] for order in ORDER],**EXPECTED)
    require(all(p.get(k)==v for k,v in expected.items()),'Frozen context-curve protocol changed')
    require(all(p.get(k) is True for k in ('no_training','no_search','no_new_confirmation_access','paused_test_untouched','no_restart_or_partial_reuse'))
        and p.get('goal_complete') is False,'Curve scope/safety changed')
    f=p.get('frozen',{})
    require(all(f.get(k+'_sha256')==v for k,v in ASSET_SHA.items()) and
        all(f.get(k)==v for k,v in dict(teacher='Qwen2-7B-Instruct',student='Qwen2-1.5B-Instruct',target_dtype='float32',
            draft_dtype='bfloat16',prefix_dtype='bfloat16',tf32=False).items()),'Frozen assets/precision changed')
    require(f.get('retrieval')==dict(top_chunks=32,fixed_k=4,backend='cuda_graph') and
        f.get('tree')==dict(top_chunks=32,nodes=8,max_depth=10,threshold=.7,flash_prefill=True,hybrid_tree=True),'Representative baselines changed')


def validate_sequence(sequence):
    require(isinstance(sequence,list) and len(sequence)==32 and
        all(set(v)=={'source_index','repeat'} and type(v['source_index']) is int and type(v['repeat']) is int for v in sequence) and
        {(v['source_index'],v['repeat']) for v in sequence}=={(i,r) for i in DOCS for r in (0,1)},'Wrong sealed32-request sequence')


def arms():
    return {
        MODES[0]:dict(mode=MODES[0],family='rescue',backend='cuda_graph'),
        MODES[1]:dict(mode=MODES[1],family='retrieval',backend='cuda_graph',top_chunks=32,fixed_k=4),
        MODES[2]:dict(mode=MODES[2],family='full_tree',backend='upstream_flash_hybrid',top_chunks=32,nodes=8,max_depth=10,threshold=.7)}


def canonical(ids,eos,vocab):
    require(isinstance(ids,list) and 0<len(ids)<=128 and all(type(v) is int and 0<=v<vocab for v in ids),'Bad canonical token IDs')
    require((len(ids)==128 or ids[-1]==eos) and (eos not in ids or ids[-1]==eos and eos not in ids[:-1]),'Invalid canonical EOS/cap')


def exact_inputs(documents,prime,old):
    require(list(documents)==list(DOCS) and len(documents)==16 and prime==documents[271],
            'Exact old16 documents / identical cold-warm271 required')
    result={}
    for i,row in documents.items():
        prefix,suffix,body=(row[k] for k in ('prefix_ids','suffix_ids','document_ids'))
        require(len(prefix)+len(suffix)==30 and len(body)==16354,'Need saved natural body16354+prompt30, no padding')
        for n in LENGTHS:
            ids=prefix+body[:n-30]+suffix
            require(len(ids)==n and all(type(v) is int and v>=0 for v in ids),'Invalid exact natural input')
            result[i,n]=ids
            if n in OLD_LENGTHS:require(fingerprint(ids)==old['input_sha256'][f'{i}:{n}'],'4K/8K old exact input changed')
    return result


def load_helper(old_root):
    path=old_root/'tools/frozen_devcheck.py';no_links(path)
    require(sha(path)==HELPER_SHA,'Frozen external helper changed')
    spec=importlib.util.spec_from_file_location('_curve_frozen_helper',path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    return helper


def prepare(args):
    """Validate sealed old evidence/source before corpus/model bytes or output writes."""
    require(args.old_root==OLD_ROOT and args.source_root==SOURCE_ROOT and args.output==EXECUTION_OUTPUT,'Only new isolated registered roots allowed')
    for k,v in vars(args).items():
        if k not in ('prior_manifest_sha256','prior_verified_sha256'):no_links(v)
    require(not args.output.exists(),'Never resume or overwrite partial experiment')
    prereg=read(args.preregistration);validate_protocol(prereg)
    manifest_path=args.prior_seal_root/'FINAL_ARTIFACTS_MANIFEST.json'
    verified_path=args.prior_seal_root/'VERIFIED_COMPLETE.json'
    require(sha(manifest_path)==args.prior_manifest_sha256 and sha(verified_path)==args.prior_verified_sha256,'Old complete-seal pins differ')
    manifest=read(manifest_path);seal=read(verified_path)
    require(seal.get('status')=='VERIFIED_COMPLETE' and seal.get('passed') is True and
        seal.get('final_artifacts_manifest_sha256')==args.prior_manifest_sha256,'Old devcheck not completely sealed')
    pins=manifest['file_sha256']
    require(isinstance(pins,dict) and pins,'Missing old sealed file pins')
    required=('results/protocol.json','results/COMPLETE.json','results/QUALITY_PASSED.json','results/GATE_PASSED.json',
        'results/own_quality.jsonl','results/quality_owners.jsonl','results/reference.jsonl','results/references.json',
        'results/data/documents.jsonl','results/data/prime.json','tools/frozen_devcheck.py','LAUNCH_MANIFEST.json','capsule/LAUNCH_MANIFEST.json',
        'PIPELINE_EXIT.json','GPU_PREFLIGHT.json','DEPENDENCY_PREFLIGHT.json')
    consumed={name:pins[name] for name in required};pinned_inventory(args.old_root,consumed)
    old=read(args.old_root/'results/protocol.json');done=read(args.old_root/'results/COMPLETE.json')
    require(done.get('passed') is True and done.get('logical_requests')==130 and done.get('counts')==
        dict(own_quality_requests=64,quality_owners=64,prime_requests=2,timing_requests=64,owners=2) and
        all(done.get(k) is True for k in ('sources_unchanged','assets_unchanged','dependencies_unchanged','quality_gate_passed','all_canonical','owners_closed')),
        'Old130 matrix incomplete')
    require(read(args.old_root/'PIPELINE_EXIT.json').get('exit_code')==0,'Old process did not terminate normally')
    require(read(args.old_root/'results/QUALITY_PASSED.json').get('passed') is True,'Old quality gate missing')
    require(read(args.old_root/'results/GATE_PASSED.json').get('passed') is True,'Old frozen-source gate missing')
    source_manifest=args.old_root/'LAUNCH_MANIFEST.json'
    require(source_manifest.is_file() and sha(source_manifest)==SOURCE_MANIFEST_SHA,'Frozen original source manifest missing/changed')
    source_meta=read(source_manifest);capsule_manifest=args.old_root/'capsule/LAUNCH_MANIFEST.json'
    require(sha(capsule_manifest)==source_meta['capsule_manifest_sha256'],'Inherited capsule manifest changed')
    capsule=read(capsule_manifest);source_pins=capsule['source_sha256']
    require(isinstance(source_pins,dict) and len(source_pins)>=152,'Incomplete original source inventory')
    pinned_inventory(args.source_root,source_pins)
    require(all(source_pins.get(n)==v for n,v in old['source_sha256'].items()),'Scientific source differs from prior run')
    helper=load_helper(args.old_root);helper.load_gate(args.source_root)
    helper.idle_gpu()  # No CUDA allocation or model/document loading before idle gate.
    for role,digest in ASSET_SHA.items():require(sha(getattr(args,role))==digest,'Frozen '+role+' changed')
    for role in ('teacher','student','mapper','policy','cost'):
        require(str(getattr(args,role))==old['args'][role],'Different original asset path')
    assets={str(p):sha(p) for p in helper.models(args.teacher)+helper.models(args.student)+[args.mapper,args.policy,args.cost]}
    require(all(old['assets_sha256'].get(k)==v for k,v in assets.items()),'Original model/tokenizer/asset bytes changed')
    docpath=args.old_root/'results/data/documents.jsonl';primepath=args.old_root/'results/data/prime.json'
    document_rows=rows(docpath);documents={v['source_index']:v for v in document_rows}
    require(len(document_rows)==len(documents)==16 and sha(docpath)==old['selected_documents_sha256'] and
        sha(primepath)==old['prime_document_sha256'],'Saved old document bytes changed')
    prime=read(primepath);inputs=exact_inputs(documents,prime,old)
    sequence=copy.deepcopy(old['shared_warm_sequence']);validate_sequence(sequence)
    references=read(args.old_root/'results/references.json');reference_rows=rows(args.old_root/'results/reference.jsonl')
    require(set(references)=={f'{i}:{n}' for n in OLD_LENGTHS for i in DOCS} and len(reference_rows)==32 and
        references=={f"{r['source_index']}:{r['length']}":r['generated'] for r in reference_rows},'Old32 AR incomplete')
    for ids in references.values():canonical(ids,old['tokenizer_eos_token_id'],old['vocab_size'])
    quality=rows(args.old_root/'results/own_quality.jsonl')
    require(len(quality)==64 and {(v['source_index'],v['length'],v['mode']) for v in quality}==
        {(i,n,m) for i in DOCS for n in OLD_LENGTHS for m in ('rescue_eager','rescue_graph')},'Old64 own quality incomplete')
    return dict(helper=helper,old=old,source_pins=source_pins,assets=assets,inputs=inputs,sequence=sequence,
        references=references,reference_rows=reference_rows,old_quality=quality,old_pins=consumed,
        entry_sha256=sha(Path(__file__)),preregistration_sha256=sha(args.preregistration))


def record(output,filename,row,p,counter):
    request=row.get('request_id')
    require(isinstance(request,str) and len(request)==32 and request not in p['seen'],'Duplicate/invalid generation request ID')
    row['execution_index']=len(p['seen']);append(output/filename,row)
    p['seen'].add(request);p['counts'][counter]+=1


def execute_matrix(adapter,p,output):
    """The same798-call production control loop is exercised by CPU stubs."""
    refs=copy.deepcopy(p['references']);lookup=arms();own_quality={};sentinels={};repeated={}
    for row in p['old_quality']:
        adapter.validate_quality(row,refs[f"{row['source_index']}:{row['length']}"],lookup[MODES[0]])
        if row['mode']==MODES[0]:own_quality[row['source_index'],row['length']]=row
    for n in OLD_LENGTHS:
        for i in DOCS:
            pair=[next(v for v in p['old_quality'] if (v['source_index'],v['length'],v['mode'])==(i,n,m)) for m in ('rescue_eager',MODES[0])]
            adapter.compare_quality(*pair,lookup[MODES[0]])
    live=None
    try:
        for n in NEW_LENGTHS:
            for i in DOCS:
                item=dict(mode='canonical_ar',source_index=i,length=n,phase='reference',performance_eligible=False)
                row=adapter.reference(p['inputs'][i,n],item)
                record(output,'reference.jsonl',row,p,'reference_requests')
                canonical(row['generated'],p['old']['tokenizer_eos_token_id'],p['old']['vocab_size'])
                refs[f'{i}:{n}']=row['generated']
        require(len(refs)==96,'Incomplete96 canonical references')
        write(output/'references.json',refs)
        write(output/'reference_provenance.json',dict(new_reference_requests=64,reused_reference_requests=32,
            refs={k:dict(origin='reused' if int(k.split(':')[1]) in OLD_LENGTHS else 'new',input_sha256=fingerprint(p['inputs'][tuple(map(int,k.split(':')))])) for k in refs},
            reused_rows_sha256=sha(output/'reused_reference.jsonl'),no_new_prime_ar=True))
        for n in LENGTHS:
            quality_plan=[]
            if n in NEW_LENGTHS:
                quality_plan.extend((lookup[MODES[0]],i,'own_quality','own_quality_requests',backend) for i in DOCS for backend in ('eager','cuda_graph'))
            quality_plan.extend((lookup[MODES[1]],211,'baseline_quality','baseline_quality_requests',backend) for backend in ('eager','cuda_graph'))
            pair=[]
            for arm,i,phase,counter,backend in quality_plan:
                actual=copy.deepcopy(arm);actual['backend']=backend
                if backend=='eager':actual['mode']='rescue_eager' if arm['family']=='rescue' else 'retrieval32_k4_eager'
                owner_id=uuid.uuid4().hex;live=adapter.make(actual,True)
                try:
                    item=dict(mode=actual['mode'],source_index=i,length=n,phase=phase,owner_id=owner_id,performance_eligible=False)
                    row,_=adapter.generate(live,actual,p['inputs'][i,n],refs[f'{i}:{n}'],item,True)
                    record(output,phase+'.jsonl',row,p,counter)
                    adapter.validate_quality(row,refs[f'{i}:{n}'],actual)
                    closed=adapter.close(live);live=None
                    append(output/'quality_owners.jsonl',dict(closed,owner_id=owner_id,mode=actual['mode'],length=n,source_index=i,request_id=row['request_id'],owner_signature=row['owner_signature']))
                    p['counts']['quality_owners']+=1;pair.append(row)
                    if backend=='cuda_graph':
                        adapter.compare_quality(*pair,arm);pair=[]
                        if arm['family']=='rescue':own_quality[i,n]=row
                        else:sentinels[n]=row
                finally:
                    if live is not None:adapter.close(live);live=None
        require(p['counts']['quality_owners']==140 and len(own_quality)==96 and len(sentinels)==6 and live is None,'Incomplete backend quality gate')
        adapter.assert_unchanged()
        write(output/'QUALITY_PASSED.json',dict(passed=True,counts=copy.deepcopy(p['counts']),all96_own_pairs=True,new_own_pairs=64,reused_own_pairs=32,
            retrieval_sentinel_docs_per_context=1,retrieval_sentinel_source_index=211,quality_owners_closed=True,nonzero_mapper=True))
        for n,order in zip(LENGTHS,ORDER):
            for index in order:
                arm=lookup[MODES[index]];owner_id=uuid.uuid4().hex
                append(output/'gpu_diagnostics.jsonl',dict(phase='owner_start',owner_id=owner_id,mode=arm['mode'],length=n,execution_index=len(p['seen']),telemetry=adapter.telemetry()))
                started=time.perf_counter();live=adapter.make(arm,False);constructor=time.perf_counter()-started
                owner=dict(mode=arm['mode'],arm=arm,length=n,owner_id=owner_id,constructor_and_detach_seconds=constructor,
                    warm_tokens=0,warm_outer_seconds_sum=0.,warm_request_ids=[],maximum_live_owners=1)
                try:
                    requests=[dict(source_index=271,phase='cold_prime')]+[dict(v,phase='warm',warm_index=j) for j,v in enumerate(p['sequence'])]
                    for request in requests:
                        cold=request['phase']=='cold_prime';i=request['source_index'];ref=refs[f'{i}:{n}']
                        item=dict(request,mode=arm['mode'],length=n,owner_id=owner_id,performance_eligible=True)
                        row,elapsed=adapter.generate(live,arm,p['inputs'][i,n],ref,item,False)
                        row.update(outer_seconds=elapsed,outer_tps=row['tokens']/elapsed,observer_enabled=False,profile_enabled=False,
                            mapper_executed=arm['family']=='rescue',controller_executed=arm['family']=='rescue')
                        if cold:row.update(constructor_and_detach_seconds=constructor,cold_complete_outer_seconds=constructor+elapsed)
                        record(output,'prime.jsonl' if cold else 'runtime.jsonl',row,p,'prime_requests' if cold else 'timing_requests')
                        adapter.validate_timing(row,ref,arm,cold)
                        require(math.isfinite(elapsed) and elapsed>0 and 'timeline' not in row and 'cost_samples' not in row,'Invalid/profiled primary timing')
                        key=(arm['mode'],i,n)
                        if key in repeated:adapter.compare_timing(repeated[key],row,arm)
                        else:repeated[key]=row
                        if arm['family']=='rescue':adapter.compare_timing(own_quality[i,n],row,arm)
                        elif arm['family']=='retrieval' and i==211:adapter.compare_timing(sentinels[n],row,arm)
                        if cold:owner.update(owner_signature=row['owner_signature'],prime_request_id=row['request_id'],prime_tokens=row['tokens'],cold_prime_outer_seconds=elapsed)
                        else:
                            require(row['owner_signature']==owner['owner_signature'],'Persistent owner/cache identity changed')
                            owner['warm_tokens']+=row['tokens'];owner['warm_outer_seconds_sum']+=elapsed;owner['warm_request_ids'].append(row['request_id'])
                    closed=adapter.close(live);live=None;end=time.perf_counter()
                    seconds=constructor+owner['cold_prime_outer_seconds']+owner['warm_outer_seconds_sum']+closed['close_seconds']
                    owner.update(closed,owner_lifetime_wall_seconds=end-started,accounted_service_seconds=seconds,
                        accounted_service_tokens=owner['prime_tokens']+owner['warm_tokens'],accounted_service_tps=(owner['prime_tokens']+owner['warm_tokens'])/seconds)
                    append(output/'owners.jsonl',owner);p['counts']['owners']+=1
                    append(output/'gpu_diagnostics.jsonl',dict(phase='owner_end',owner_id=owner_id,mode=arm['mode'],length=n,execution_index=len(p['seen']),telemetry=adapter.telemetry()))
                finally:
                    if live is not None:adapter.close(live);live=None
        require(p['counts']==EXPECTED and len(p['seen'])==TOTAL,'Incomplete798 matrix')
        adapter.assert_unchanged()
        return dict(passed=True,counts=p['counts'],expected_counts=EXPECTED,logical_requests=TOTAL,
            all_canonical=True,quality_gate_passed=True,owners_closed=True,sources_unchanged=True,assets_unchanged=True,
            dependencies_unchanged=True,reused_reference_requests=32,reused_own_quality_requests=64,
            analysis_required_locally=True,fresh_confirmation=False,goal_complete=False)
    finally:
        if live is not None:adapter.close(live)


class RuntimeAdapter:
    """Only GPU-specific operations; no algorithm override or fallback."""
    def __init__(self,args,p):
        self.args=args;self.p=p;self.h=p['helper'];self.r=self.h._runtime(args.source_root);r=self.r;torch=r.torch
        self.previous=r.draft_graph.ResidentDraftRuntime.MAX_ENTRIES;r.draft_graph.ResidentDraftRuntime.MAX_ENTRIES=128
        self.group=None;self.live=[];self.setup_complete=False
        try:self.setup()
        except BaseException:self.finish();raise

    def setup(self):
        r=self.r;torch=r.torch;p=self.p;args=self.args
        self.baseline=r.baseline_selection;self.legacy=r.fair_confirmation;self.q=r.draft_cost_recalibration;self.common=r.round_overhead_run
        self.baselines=importlib.import_module('rescue.baselines')
        require(Path(self.baselines.__file__).resolve()==args.source_root/'rescue/baselines.py','Foreign baseline module')
        torch.set_num_threads(4);torch.manual_seed(SEED);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.environment=self.common.gpu_snapshot()
        for key in ('nvidia_smi_gpu','torch','cuda','target_dtype','draft_dtype','prefix_dtype','tf32'):
            require(self.environment[key]==p['old']['environment'][key],'Different GPU/dependency/precision: '+key)
        self.teacher,self.student,self.tokenizer=r.loading.load_models(str(args.teacher),str(args.student),'cuda:0',target_dtype=torch.float32,efficient=True)
        for model,dtype in ((self.teacher,torch.float32),(self.student,torch.bfloat16)):
            model.eval().requires_grad_(False);require(all(v.device==torch.device('cuda:0') and v.dtype==dtype for v in model.parameters()),'Model dtype/device changed')
        require(self.tokenizer.eos_token_id==p['old']['tokenizer_eos_token_id'] and len(self.tokenizer)==p['old']['vocab_size'],'Tokenizer changed')
        started=time.perf_counter();self.template=r.specextend_qwen.build(self.teacher,self.student,str(args.teacher));self.template.prefill_precision='bfloat16'
        r.packing.share_packed_projections(self.teacher,self.template.base_model);r.packing.share_packed_projections(self.student,self.template.draft_model)
        self.preparation=time.perf_counter()-started;self.group=r.draft_graph_benchmark.HookOwners(self.teacher,self.student)
        self.mapper=torch.load(args.mapper,map_location='cpu',weights_only=True);self.policy=torch.load(args.policy,map_location='cpu',weights_only=True);self.cost=read(args.cost)
        require(self.policy.get('kind')=='step_budget' and 'seed' in self.policy,'Frozen causal policy payload required')
        self.policy_meta={k:self.policy[k] for k in ('kind','hidden','margin','prior_weight','seed')}
        self.imports=self.h.bindings(r,args.source_root);self.tree_bindings=self.baseline.check_full_bindings(args.source_root);self.flash=self.baseline.flash_receipt()
        require(self.flash==p['old']['flash_attention'],'Flash dependency changed')
        self.setup_complete=True

    def stats(self,e):
        if isinstance(e,self.baseline.FullOwner):return e.stats()
        return dict(self.r.draft_graph_smoke.runtime_snapshot(e),max_graph_entries=e.draft_runtime.max_entries if e.draft_runtime is not None else None)

    def signature(self,e):
        return e.signature() if isinstance(e,self.baseline.FullOwner) else self.r.draft_graph_benchmark.owner_signature(e)

    def make(self,arm,quality):
        r=self.r;self.group.idle();require(not self.live and not self.group.engines,'At most one live private owner')
        observer=self.q.QualityObserver() if quality else None
        if arm['family']=='full_tree':e=self.baseline.FullOwner(self.template,self.args.teacher,arm)
        elif arm['family']=='retrieval':e=self.baselines.BaselineEngine(self.teacher,self.student,self.baseline.linear_config(arm),vocab_size=len(self.tokenizer),top_chunks=arm['top_chunks'],round_observer=observer)
        else:
            mapper=r.query_mapper.mapper_from_payload(self.mapper,'cuda:0').eval().requires_grad_(False)
            policy=r.step_budget.StepBudget.from_payload(self.policy,'cuda:0').eval().requires_grad_(False)
            require(all(getattr(policy,k)==self.policy[k] for k in ('margin','prior_weight','seed')),'Frozen policy settings changed')
            e=r.engine.RescueEngine(self.teacher,self.student,self.legacy.own_config(arm['backend']),mapper,policy,self.cost,vocab_size=len(self.tokenizer),round_observer=observer)
        try:
            if arm['family']!='full_tree':self.group.add(e)
        except BaseException:
            self.q.detach_quality_observer(e);e.close();raise
        self.live.append(e)
        if observer is not None:observer.engine=e
        return e

    def close(self,e):
        final=self.stats(e);started=time.perf_counter();self.q.detach_quality_observer(e)
        try:
            if isinstance(e,self.baseline.FullOwner):self.group.idle();e.close();self.group.idle()
            else:self.group.close(e)
        finally:
            if e in self.live:self.live.remove(e)
        elapsed=time.perf_counter()-started;closed=self.stats(e)
        require(e.closed and closed['entries']==0 and not self.group.engines,'Owner retained graph/hooks')
        return dict(close_seconds=elapsed,final_runtime_stats=final,closed_runtime_stats=closed,engine_closed=True,hook_restored=True)

    def before(self):
        snap=self.common.gpu_snapshot()
        require(snap['nvidia_smi_gpu']==self.environment['nvidia_smi_gpu'],'Physical GPU changed')
        self.r.torch.manual_seed(self.policy_meta['seed']);random.seed(self.policy_meta['seed']);self.group.idle()
        return snap

    def reference(self,ids,item):
        require(not self.live and not self.group.engines,'Reference overlapped serving owner')
        snap=self.before()
        with self.baselines.BaselineEngine(self.teacher,self.student,self.baseline.linear_config(arms()[MODES[1]]),vocab_size=len(self.tokenizer)) as e:
            row=e.reference(ids,self.tokenizer.eos_token_id)
        row.update(item,request_id=uuid.uuid4().hex,gpu_snapshot_before=snap)
        return row

    def generate(self,e,arm,ids,reference,item,quality):
        snap=self.before();started=time.perf_counter()
        def call():return e.generate(ids) if arm['family']=='full_tree' else self.group.generate(e,ids,self.tokenizer.eos_token_id)
        if quality:
            with self.q.record_draft_steps() as steps:raw=call()
        else:raw=call()
        elapsed=time.perf_counter()-started
        if arm['family']=='full_tree':
            raw['request_id']=uuid.uuid4().hex
            append(self.args.output/'tree_raw.jsonl',dict(request_id=raw['request_id'],execution_index=len(self.p['seen']),**item,outer_seconds=elapsed,result=raw))
            row=self.baseline.normalize_tree(raw,arm,reference,self.tokenizer.eos_token_id)
        else:row=raw
        row.update(item,arm=arm,output_agrees=row['generated']==reference,runtime_stats=self.stats(e),owner_signature=self.signature(e),gpu_snapshot_before=snap)
        if quality:
            observer=e.round_observer
            row.update(observed_rounds=observer.rounds,queries=steps,nonzero_injection=observer.nonzero_injection,
                native_t=self.r.draft_graph_smoke.native_digest(e.native_t),native_s=self.r.draft_graph_smoke.native_digest(e.native_s))
        return row,elapsed

    def validate_quality(self,row,ref,arm):
        self.legacy.check_quality_row(row,ref,arm['family']=='rescue',arm,self.tokenizer.eos_token_id)
        # Reused own eager rows carry their original arm; check_quality_row owns EOS/trajectory validation.
        if row['mode'].endswith('_graph'):
            require(row['draft_runtime']['request']['captures']>0 and row['draft_runtime']['request']['replays']==row['draft_forwards']>0,'Fresh quality Graph missing')

    def compare_quality(self,eager,graph,arm):
        if arm['family']=='rescue':self.legacy.own_quality_pair(eager,graph)
        else:self.legacy.baseline_quality_pair(eager,graph,arm)

    def validate_timing(self,row,ref,arm,cold):
        if arm['family']=='rescue':self.legacy.check_own_output(row,ref,self.tokenizer.eos_token_id)
        else:self.baseline.check_output(row,ref,arm,self.tokenizer.eos_token_id)
        if arm['backend']=='cuda_graph':
            require(row['draft_runtime']['request']['replays']==row['draft_forwards']>0 and row['runtime_stats']['max_graph_entries']==128,'Actual Graph replay/bound missing')
            if cold:require(row['draft_runtime']['request']['captures']>0,'Fresh timing owner borrowed Graphs')

    def compare_timing(self,left,right,arm):
        if arm['family']=='rescue':self.common.check_pair(left,right)
        else:self.baseline.check_reproducible(left,right,arm)

    def telemetry(self):return self.r.baseline_horizon.gpu_telemetry()

    def assert_unchanged(self):
        require(self.h.bindings(self.r,self.args.source_root)==self.imports and self.baseline.check_full_bindings(self.args.source_root)==self.tree_bindings and
            self.baseline.flash_receipt()==self.flash,'Imported source/Flash changed')
        require({p:sha(Path(p)) for p in self.p['assets']}==self.p['assets'],'Model/tokenizer/Mapper/policy/cost changed')
        pinned_inventory(self.args.source_root,self.p['source_pins'])
        require(sha(Path(__file__))==self.p['entry_sha256'] and sha(self.args.preregistration)==self.p['preregistration_sha256'],'Curve entry/protocol changed')
        self.common.gpu_snapshot()

    def finish(self):
        for e in list(self.live):self.close(e)
        self.r.draft_graph.ResidentDraftRuntime.MAX_ENTRIES=self.previous


def run(args):
    p=prepare(args);output=args.output;output.mkdir(parents=True);p.update(counts=dict.fromkeys(EXPECTED,0),seen=set())
    adapter=None
    try:
        for name in ('results/reference.jsonl','results/own_quality.jsonl','results/data/documents.jsonl','results/data/prime.json'):
            destination=output/({'results/reference.jsonl':'reused_reference.jsonl','results/own_quality.jsonl':'reused_own_quality.jsonl'}.get(name,Path(name).name))
            shutil.copyfile(args.old_root/name,destination);require(sha(destination)==p['old_pins'][name],'Reused bytes changed while copying')
        shutil.copyfile(args.preregistration,output/'PREREGISTRATION.json');shutil.copyfile(Path(__file__),output/'CURVE_SOURCE.py')
        (output/'prior_local_seal').mkdir()
        for name in ('FINAL_ARTIFACTS_MANIFEST.json','VERIFIED_COMPLETE.json'):
            shutil.copyfile(args.prior_seal_root/name,output/'prior_local_seal'/name)
            require(sha(output/'prior_local_seal'/name)==sha(args.prior_seal_root/name),'Prior seal copy changed')
        direct_copies={'results/reference.jsonl','results/own_quality.jsonl','results/data/documents.jsonl','results/data/prime.json'}
        for name,digest in p['old_pins'].items():
            if name in direct_copies:continue
            destination=output/'prior_consumed'/name;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(args.old_root/name,destination);require(sha(destination)==digest,'Prior consumed proof copy changed')
        for name,digest in p['source_pins'].items():
            destination=output/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(args.source_root/name,destination);require(sha(destination)==digest,'Frozen source snapshot changed')
        write(output/'REUSE_RECEIPT.json',dict(old_root=str(args.old_root),prior_manifest_sha256=args.prior_manifest_sha256,
            prior_verified_sha256=args.prior_verified_sha256,prior_local_seal='prior_local_seal',source_manifest_sha256=SOURCE_MANIFEST_SHA,
            prior_consumed_root='prior_consumed',source_snapshot='source',
            old_artifact_pins=p['old_pins'],reused_reference_requests=32,reused_own_quality_requests=64,new_generation_requests=TOTAL,
            canonical_inputs_all96={f'{i}:{n}':fingerprint(ids) for (i,n),ids in p['inputs'].items()},
            no_new_confirmation_access=True,old_quality_not_new_execution=True))
        adapter=RuntimeAdapter(args,p)
        configs={k:asdict(adapter.legacy.own_config()) if a['family']=='rescue' else
            asdict(adapter.baseline.linear_config(a)) if a['family']=='retrieval' else adapter.baseline.tree_policy(a) for k,a in arms().items()}
        write(output/'protocol.json',dict(schema=1,status='FROZEN_CONTEXT_CURVE',seed=SEED,source_indices=list(DOCS),lengths=list(LENGTHS),repeats=2,
            max_new_tokens=128,modes=list(MODES),arms=list(arms().values()),configs=configs,expected_counts=EXPECTED,expected_requests=TOTAL,
            shared_warm_sequence=p['sequence'],owner_order=[dict(length=n,mode=MODES[i]) for n,order in zip(LENGTHS,ORDER) for i in order],
            input_sha256={f'{i}:{n}':fingerprint(ids) for (i,n),ids in p['inputs'].items()},environment=adapter.environment,
            source_sha256=p['source_pins'],source_bindings=adapter.imports,tree_source_bindings=adapter.tree_bindings,flash_attention=adapter.flash,
            assets_sha256=p['assets'],policy_metadata=adapter.policy_meta,tokenizer_eos_token_id=adapter.tokenizer.eos_token_id,vocab_size=len(adapter.tokenizer),
            graph_max_entries=128,max_live_owners=1,model_preparation_seconds=adapter.preparation,template_prefill_precision='bfloat16',
            entry_source_sha256=p['entry_sha256'],preregistration_sha256=p['preregistration_sha256'],
            mapper_sha256=ASSET_SHA['mapper'],policy_sha256=ASSET_SHA['policy'],cost_sha256=ASSET_SHA['cost'],
            reused_reference_requests=32,reused_own_quality_requests=64,reuse_receipt_sha256=sha(output/'REUSE_RECEIPT.json'),
            primary='All32 warm whole-generate outer, all captures/overshoot charged; constructor/cold/close separately',
            scope='Descriptive reused16-doc six-context curve; fixed representative controls, NOT all-menu optimization/fresh confirmation',
            no_training=True,no_search=True,fresh_confirmation=False,no_new_confirmation_access=True,paused_test_untouched=True,goal_complete=False,
            args={k:str(v) for k,v in vars(args).items()}))
        completion=execute_matrix(adapter,p,output)
        require(sha(args.prior_seal_root/'FINAL_ARTIFACTS_MANIFEST.json')==args.prior_manifest_sha256 and
            sha(args.prior_seal_root/'VERIFIED_COMPLETE.json')==args.prior_verified_sha256,'Prior seal changed')
        pinned_inventory(args.old_root,p['old_pins'])
        write(output/'COMPLETE.json',completion);return completion
    except BaseException as error:
        write(output/'FAILURE.json',dict(error_type=type(error).__name__,error=str(error),counts=p['counts'],logical_requests=len(p['seen']),
            preserve_partial=True,no_automatic_retry=True,no_replacement=True,goal_complete=False));raise
    finally:
        if adapter is not None:adapter.finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('old-root','source-root','prior-seal-root','teacher','student','mapper','policy','cost','preregistration','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--prior-manifest-sha256',required=True)
    parser.add_argument('--prior-verified-sha256',required=True)
    print(json.dumps(run(parser.parse_args()),indent=2))


if __name__=='__main__':main()
