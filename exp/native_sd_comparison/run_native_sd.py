"""Frozen six-context comparison with genuine full-context ordinary SD.

Only orchestration and Native-specific validation are new. The sealed runtime,
Mapper, causal2/4 policy and old cost stay untouched. No corpus is opened and no
old timing is reused. Eager/Eager is the algorithm comparison; Graph/Native is
an explicitly backend-confounded deployment comparison, never a Native Graph
or optimal-fixed-k claim.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import time
import uuid

LENGTHS=(2048,4096,6144,8192,12288,16384)
DOCS=(211,213,220,226,233,236,246,248,253,271,272,288,303,306,327,604)
MODES=('native_eager','rescue_eager','rescue_graph')
ORDER=((0,1,2),(1,2,0),(2,0,1),(0,2,1),(2,1,0),(1,0,2))
SEED=20261028
EXPECTED=dict(native_quality_requests=96,quality_owners=96,prime_requests=18,timing_requests=576,owners=18)
TOTAL=690
ROOT=Path('/mnt/8T/cl/rescue_experiment_cache/native_sd_comparison_20261020_v1')
CURVE_ROOT=Path('/mnt/8T/cl/rescue_experiment_cache/context_curve_20261020_v2')
OLD_ROOT=Path('/mnt/8T/cl/rescue_experiment_cache/frozen_devcheck_20261020_v1')
SOURCE_ROOT=OLD_ROOT/'capsule/after'
SOURCE_MANIFEST_SHA='002545801b8a15fdb94afb083c61dd76633e423ec828f7a3d7e2c1fc6fbaf40a'
HELPER_SHA='ce4f06ab60e34bf27b448dafe5f4574332557cda9e8b9733d697e8a4208a0069'
CURVE_ENTRY_SHA='1ef58e790894757e7a8a8a5dd328fd57cc45a6525d7ca6bb9436b96dabec1be3'
ASSET_SHA=dict(mapper='d08889e796ca2527e23241ff34a072ef640e2fdf3fe9a2f9ba5439975af04d9f',
    policy='3c333a470119c236afcc1308c18592842a535bee0b8f83c4fb4da359ff17eff5',
    cost='d92c843a84676c0cd6adaf9364bbf7e8c771d56548f1016b5720a3e04f371fa1')
ZERO_WORK=('working_gather_layers','working_full_gather_layers','working_gathered_position_layers',
    'working_incremental_binds','selector_refreshes','target_probe_rounds','draft_shadow_probes','projected_positions','rpc_calls')


def require(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def fingerprint(ids):return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def append(path,value):
    with Path(path).open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')


def progress(stage,p,**fields):
    print(json.dumps(dict(stage=stage,counts=dict(p['counts']),logical_requests=len(p['seen']),**fields)),flush=True)


def rows(path):
    raw=Path(path).read_bytes()
    require(raw and raw.endswith(b'\n') and all(v.strip() for v in raw.splitlines()),'Incomplete JSONL')
    result=[json.loads(v) for v in raw.splitlines()]
    require(all(isinstance(v,dict) for v in result),'JSONL must contain objects');return result


def no_links(path):
    p=Path(path)
    require(p.is_absolute() and '..' not in p.parts and not any(v.is_symlink() for v in (p,*p.parents)),
        'Noncanonical or symlink path')


def pinned_inventory(root,pins):
    no_links(root);require(isinstance(pins,dict) and pins,'Missing frozen inventory')
    for name,digest in pins.items():
        rel=PurePosixPath(name)
        require(rel.as_posix()==name and not rel.is_absolute() and '..' not in rel.parts and '\\' not in name,'Unsafe frozen relative path')
        path=root/name;no_links(path);require(path.is_file() and sha(path)==digest,'Frozen bytes changed: '+name)


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def arms():
    return {MODES[0]:dict(mode=MODES[0],family='native',backend='eager',fixed_k=4,full_context=True),
        MODES[1]:dict(mode=MODES[1],family='rescue',backend='eager'),
        MODES[2]:dict(mode=MODES[2],family='rescue',backend='cuda_graph')}


def validate_protocol(p):
    expected=dict(schema=1,status='FROZEN_BEFORE_EXECUTION',seed=SEED,lengths=list(LENGTHS),source_indices=list(DOCS),
        modes=list(MODES),repeats=2,max_new_tokens=128,graph_max_entries=128,new_generation_requests=TOTAL,
        reused_reference_requests=96,reused_own_quality_requests=192,prime_source_index=271,
        source_root=str(SOURCE_ROOT),source_manifest_sha256=SOURCE_MANIFEST_SHA,old_root=str(OLD_ROOT),curve_root=str(CURVE_ROOT),
        output_root=str(ROOT),execution_output=str(ROOT/'results'),owner_order=[[MODES[i] for i in order] for order in ORDER],**EXPECTED)
    require(all(p.get(k)==v for k,v in expected.items()),'Frozen Native SD protocol changed')
    require(all(p.get(k) is True for k in ('no_training','no_search','no_new_confirmation_access','paused_test_untouched','no_restart_or_partial_reuse')) and
        p.get('goal_complete') is False,'Scope/safety changed')
    f=p.get('frozen',{})
    require(all(f.get(k+'_sha256')==v for k,v in ASSET_SHA.items()) and all(f.get(k)==v for k,v in
        dict(teacher='Qwen2-7B-Instruct',student='Qwen2-1.5B-Instruct',target_dtype='float32',draft_dtype='bfloat16',prefix_dtype='bfloat16',tf32=False).items()),'Frozen assets/precision changed')
    require(f.get('native')==dict(fixed_k=4,backend='eager',full_context=True),'Ordinary SD must be fixed4 full context Eager')


def canonical(ids,eos,vocab):
    require(isinstance(ids,list) and 0<len(ids)<=128 and all(type(v) is int and 0<=v<vocab for v in ids),'Invalid canonical token IDs')
    require((len(ids)==128 or ids[-1]==eos) and (eos not in ids or ids[-1]==eos and eos not in ids[:-1]),'Invalid canonical EOS/cap')


def validate_sequence(seq):
    require(isinstance(seq,list) and len(seq)==32 and all(set(v)=={'source_index','repeat'} and
        type(v['source_index']) is int and type(v['repeat']) is int for v in seq) and
        {(v['source_index'],v['repeat']) for v in seq}=={(i,r) for i in DOCS for r in (0,1)},'Changed sealed32 sequence')


def prepare(args):
    """Read-only provenance checks; no output creation, model import or CUDA."""
    require(args.curve_root==CURVE_ROOT and args.old_root==OLD_ROOT and args.source_root==SOURCE_ROOT and args.output==ROOT/'results','Only registered isolated roots allowed')
    for key,value in vars(args).items():
        if isinstance(value,Path):no_links(value)
    require(not args.output.exists(),'Never resume or overwrite partial experiment')
    validate_protocol(read(args.preregistration))
    manifest_path=args.prior_seal_root/'FINAL_ARTIFACTS_MANIFEST.json';verified_path=args.prior_seal_root/'VERIFIED_COMPLETE.json'
    require(sha(manifest_path)==args.prior_manifest_sha256 and sha(verified_path)==args.prior_verified_sha256,'Prior complete-seal SHA changed')
    manifest=read(manifest_path);verified=read(verified_path)
    require(verified.get('status')=='VERIFIED_COMPLETE' and verified.get('passed') is True and verified.get('six_lengths_complete') is True and
        verified.get('final_artifacts_manifest_sha256')==args.prior_manifest_sha256,'Prior six-context experiment not verified complete')
    pins=manifest['file_sha256']
    names=('LAUNCH_MANIFEST.json','PIPELINE_EXIT.json','GPU_PREFLIGHT.json','DEPENDENCY_PREFLIGHT.json',
        'tools/run_context_curve.py','results/protocol.json','results/COMPLETE.json','results/QUALITY_PASSED.json',
        'results/REUSE_RECEIPT.json','results/reference.jsonl','results/reused_reference.jsonl','results/references.json',
        'results/own_quality.jsonl','results/reused_own_quality.jsonl','results/quality_owners.jsonl',
        'results/documents.jsonl','results/prime.json')
    consumed={name:pins['terminal/'+name] for name in names};pinned_inventory(args.curve_root,consumed)
    require(consumed['tools/run_context_curve.py']==CURVE_ENTRY_SHA,'Prior adapter is not the frozen executed v2')
    curve=read(args.curve_root/'results/protocol.json');done=read(args.curve_root/'results/COMPLETE.json')
    require(done.get('passed') is True and done.get('logical_requests')==798 and done.get('counts')==dict(reference_requests=64,
        own_quality_requests=128,baseline_quality_requests=12,quality_owners=140,prime_requests=18,timing_requests=576,owners=18) and
        all(done.get(k) is True for k in ('all_canonical','quality_gate_passed','owners_closed','sources_unchanged','assets_unchanged','dependencies_unchanged')),'Prior798 matrix incomplete')
    require(read(args.curve_root/'PIPELINE_EXIT.json').get('exit_code')==0 and read(args.curve_root/'results/QUALITY_PASSED.json').get('passed') is True,'Prior execution/quality gate missing')
    require(curve.get('lengths')==list(LENGTHS) and curve.get('source_indices')==list(DOCS) and curve.get('repeats')==2 and curve.get('max_new_tokens')==128,'Prior cohort/settings changed')
    original=read(args.curve_root/'results/REUSE_RECEIPT.json')['old_artifact_pins']
    old_names=('LAUNCH_MANIFEST.json','capsule/LAUNCH_MANIFEST.json','results/protocol.json','tools/frozen_devcheck.py')
    old_pins={name:original[name] for name in old_names};pinned_inventory(args.old_root,old_pins)
    require(old_pins['LAUNCH_MANIFEST.json']==SOURCE_MANIFEST_SHA and old_pins['tools/frozen_devcheck.py']==HELPER_SHA,'Original execution/helper pins changed')
    source_pins=read(args.old_root/'capsule/LAUNCH_MANIFEST.json')['source_sha256'];pinned_inventory(args.source_root,source_pins)
    require(all(source_pins.get(k)==v for k,v in curve['source_sha256'].items()),'Prior scientific sources changed')
    helper=load(args.old_root/'tools/frozen_devcheck.py','_native_sd_frozen_helper');helper.load_gate(args.source_root);helper.idle_gpu()
    for role,digest in ASSET_SHA.items():require(sha(getattr(args,role))==digest,'Frozen '+role+' changed')
    for role in ('teacher','student','mapper','policy','cost'):require(str(getattr(args,role))==curve['args'][role],'Original asset path changed')
    assets={str(p):sha(p) for p in helper.models(args.teacher)+helper.models(args.student)+[args.mapper,args.policy,args.cost]}
    require(all(curve['assets_sha256'].get(k)==v for k,v in assets.items()),'Model/tokenizer/asset bytes changed')
    document_rows=rows(args.curve_root/'results/documents.jsonl');documents={v['source_index']:v for v in document_rows}
    prime=read(args.curve_root/'results/prime.json')
    require(len(document_rows)==16 and list(documents)==list(DOCS) and prime==documents[271],'Original natural16/cold271 changed')
    inputs={}
    for i,row in documents.items():
        prefix,suffix,body=(row[k] for k in ('prefix_ids','suffix_ids','document_ids'))
        require(len(prefix)+len(suffix)==30 and len(body)==16354,'Natural16354+prompt30 required; no padding')
        for n in LENGTHS:
            ids=prefix+body[:n-30]+suffix;require(len(ids)==n and all(type(v) is int and v>=0 for v in ids),'Bad exact inputs')
            require(fingerprint(ids)==curve['input_sha256'][f'{i}:{n}'],'Exact old96 inputs changed');inputs[i,n]=ids
    references=read(args.curve_root/'results/references.json')
    reference_rows=rows(args.curve_root/'results/reference.jsonl')+rows(args.curve_root/'results/reused_reference.jsonl')
    require(len(reference_rows)==96 and len({(r['source_index'],r['length']) for r in reference_rows})==96 and
        set(references)=={f'{i}:{n}' for i in DOCS for n in LENGTHS} and
        references=={f"{r['source_index']}:{r['length']}":r['generated'] for r in reference_rows},'Incomplete/repeated96 AR')
    for ids in references.values():canonical(ids,curve['tokenizer_eos_token_id'],curve['vocab_size'])
    quality=rows(args.curve_root/'results/own_quality.jsonl')+rows(args.curve_root/'results/reused_own_quality.jsonl')
    require(len(quality)==192 and {(v['source_index'],v['length'],v['mode']) for v in quality}==
        {(i,n,m) for i in DOCS for n in LENGTHS for m in MODES[1:]},'Incomplete/repeated192 own quality')
    sequence=copy.deepcopy(curve['shared_warm_sequence']);validate_sequence(sequence)
    curve_module=load(args.curve_root/'tools/run_context_curve.py','_native_sd_frozen_curve_adapter')
    return dict(helper=helper,curve_module=curve_module,old=curve,source_pins=source_pins,assets=assets,documents=documents,prime=prime,
        inputs=inputs,sequence=sequence,references=references,reference_rows=reference_rows,old_quality=quality,curve_pins=consumed,old_pins=old_pins,
        entry_sha256=sha(Path(__file__)),preregistration_sha256=sha(args.preregistration))


def native_config(config_class):
    return config_class(selector='full',length_policy='fixed',fixed_k=4,max_draft=12,max_new_tokens=128,
        refresh_every=4,inject_tokens=0,injection_strength=0.,injection_layers=(),prefill_precision='bfloat16',
        schedule='serial',profile=False,working_mode='packed',optimized_execution=True,deferred_probes=True,
        attention_backend='sdpa',draft_execution='eager',coverage_execution='eager')


def check_native(row,reference,eos,quality=False):
    canonical(reference,eos,row.get('vocab_size',max(reference)+1))
    require(row.get('generated')==reference and row.get('output_agrees') is True and row.get('tokens')==len(reference),'Native canonical output mismatch')
    require(row.get('baseline_algorithm')=='native_full_context' and row.get('execution_backend')=='eager' and
        'draft_runtime' not in row,'Ordinary SD silently became retrieval/Graph')
    require(all(type(row.get(k)) is int and row[k]==0 for k in ZERO_WORK),'Native performed intervention-only work')
    state=row.get('native_state',{})
    require(state==dict(top_chunks=None,working=False,injector=False,draft_runtime=False,target_hooks=0,round_observer_enabled=quality),'Native owner retained intervention components')
    trace=row.get('trace',[]);require(row.get('blocks')==len(trace)>0,'Native round trace missing')
    require(row.get('accepted')==sum(t['accepted'] for t in trace) and row.get('proposed')==sum(t['proposed'] for t in trace),'Native accept/proposal mismatch')
    progress=0
    for j,t in enumerate(trace):
        require(t.get('round_index')==j and t.get('terminal') is (j==len(trace)-1),'Native terminal/round order changed')
        require(type(t.get('cap')) is int and t['cap']==min(4,128-progress) and type(t.get('proposed')) is int and
            1<=t['proposed']<=t['cap'] and type(t.get('accepted')) is int and 0<=t['accepted']<=t['proposed'],'Native fixed4/cap invalid')
        require(t.get('prefix')==row['length']-1+progress and t.get('workset_tokens')==t['prefix'],'Native did not read full committed prefix')
        if not t['terminal']:progress+=t['accepted']+1
        else:require(progress+t['accepted']<=len(reference)<=progress+t['accepted']+1,'Native terminal committed count invalid')
    require(row.get('draft_forwards')==row['proposed']+sum(not t['terminal'] and t['accepted']==t['proposed'] for t in trace),'Native catchup forwards changed')
    if quality:
        require(row.get('nonzero_injection') is False and row.get('native_t') and row.get('native_s'),'Native quality state missing/injected')
        observed=row.get('observed_rounds',[]);queries=row.get('queries',[])
        require(len(observed)==len(trace) and len(queries)==row['draft_forwards'],'Native quality round/Query evidence missing')
        for t,o in zip(trace,observed):
            require(all(o.get(k)==t[k] for k in ('round_index','prefix','accepted')) and o.get('selected')==list(range(t['prefix'])) and
                o.get('budget_features') is None and o.get('injection_evidence')==[] and o.get('summary_count')==0,'Native observer selected/intervened state changed')
            require(len(o.get('draft',[]))==t['proposed'] and len(o.get('checks',[]))==t['proposed']+1,'Native quality drafts/checks missing')
            count=0
            for token,check in zip(o['draft'],o['checks']):
                if token!=check:break
                count+=1
                if token==eos:break
            require(count==t['accepted'],'Native observed accept count differs')
        require(all(q.get('summary_requested') is False and q.get('summary_count')==0 and q.get('query') for q in queries),'Native quality collected M3 summaries')


def record(output,filename,row,p,counter):
    # Persist first, including malformed/duplicate output, then fail without retry.
    row['execution_index']=len(p['seen']);append(output/filename,row);p['counts'][counter]+=1
    request=row.get('request_id');require(isinstance(request,str) and len(request)==32 and request not in p['seen'],'Duplicate/invalid generation UUID')
    p['seen'].add(request)


def execute_matrix(adapter,p,output):
    lookup=arms();refs=p['references'];own_quality={};native_quality={};repeated={};live=None
    for row in p['old_quality']:
        arm=lookup[row['mode']];adapter.validate_quality(row,refs[f"{row['source_index']}:{row['length']}"],arm)
        own_quality[row['mode'],row['source_index'],row['length']]=row
    require(len(own_quality)==192,'Missing own proof cells')
    for n in LENGTHS:
        for i in DOCS:adapter.compare_own_quality(own_quality[MODES[1],i,n],own_quality[MODES[2],i,n])
    try:
        for n in LENGTHS:
            for i in DOCS:
                arm=lookup[MODES[0]];owner_id=uuid.uuid4().hex;live=adapter.make(arm,True)
                try:
                    item=dict(mode=arm['mode'],source_index=i,length=n,phase='native_quality',owner_id=owner_id,performance_eligible=False)
                    row,_=adapter.generate(live,arm,p['inputs'][i,n],refs[f'{i}:{n}'],item,True)
                    record(output,'native_quality.jsonl',row,p,'native_quality_requests');adapter.validate_quality(row,refs[f'{i}:{n}'],arm)
                    closed=adapter.close(live);live=None
                    append(output/'quality_owners.jsonl',dict(closed,owner_id=owner_id,mode=arm['mode'],length=n,source_index=i,request_id=row['request_id'],owner_signature=row['owner_signature']))
                    p['counts']['quality_owners']+=1;native_quality[i,n]=row
                finally:
                    if live is not None:adapter.close(live);live=None
            progress('native_quality_context_complete',p,length=n)
        require(p['counts']['quality_owners']==96 and len(native_quality)==96 and live is None,'Native quality matrix incomplete')
        adapter.assert_unchanged();write(output/'QUALITY_PASSED.json',dict(passed=True,native_quality_requests=96,quality_owners_closed=True,
            reused_own_quality_requests=192,reused_reference_requests=96,all96_native_full_context=True,own_nonzero_mapper=True,
            own_all96_eager_graph_pairs=True,counts=copy.deepcopy(p['counts'])))
        progress('quality_passed',p)
        for n,order in zip(LENGTHS,ORDER):
            for index in order:
                arm=lookup[MODES[index]];owner_id=uuid.uuid4().hex
                append(output/'gpu_diagnostics.jsonl',dict(phase='owner_start',owner_id=owner_id,mode=arm['mode'],length=n,execution_index=len(p['seen']),telemetry=adapter.telemetry()))
                started=time.perf_counter();live=adapter.make(arm,False);constructor=time.perf_counter()-started
                owner=dict(mode=arm['mode'],arm=arm,length=n,owner_id=owner_id,constructor_and_detach_seconds=constructor,
                    warm_tokens=0,warm_outer_seconds_sum=0.,warm_request_ids=[],maximum_live_owners=1)
                try:
                    plan=[dict(source_index=271,phase='cold_prime')]+[dict(v,phase='warm',warm_index=j) for j,v in enumerate(p['sequence'])]
                    for request in plan:
                        cold=request['phase']=='cold_prime';i=request['source_index'];ref=refs[f'{i}:{n}']
                        item=dict(request,mode=arm['mode'],length=n,owner_id=owner_id,performance_eligible=True)
                        row,elapsed=adapter.generate(live,arm,p['inputs'][i,n],ref,item,False)
                        row.update(outer_seconds=elapsed,outer_tps=row['tokens']/elapsed,observer_enabled=False,profile_enabled=False,
                            mapper_executed=arm['family']=='rescue',controller_executed=arm['family']=='rescue')
                        if cold:row.update(constructor_and_detach_seconds=constructor,cold_complete_outer_seconds=constructor+elapsed)
                        record(output,'prime.jsonl' if cold else 'runtime.jsonl',row,p,'prime_requests' if cold else 'timing_requests')
                        adapter.validate_timing(row,ref,arm,cold)
                        require(math.isfinite(elapsed) and elapsed>0 and 'timeline' not in row and 'cost_samples' not in row,'Invalid/profiled timing')
                        proof=native_quality[i,n] if arm['family']=='native' else own_quality[arm['mode'],i,n]
                        adapter.compare_timing(proof,row,arm)
                        key=(arm['mode'],i,n)
                        if key in repeated:adapter.compare_timing(repeated[key],row,arm)
                        else:repeated[key]=row
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
                    progress('timing_owner_complete',p,length=n,mode=arm['mode'])
                finally:
                    if live is not None:adapter.close(live);live=None
        require(p['counts']==EXPECTED and len(p['seen'])==TOTAL,'Incomplete690 request matrix');adapter.assert_unchanged()
        return dict(passed=True,counts=p['counts'],expected_counts=EXPECTED,logical_requests=TOTAL,all_canonical=True,quality_gate_passed=True,
            owners_closed=True,sources_unchanged=True,assets_unchanged=True,dependencies_unchanged=True,reused_reference_requests=96,
            reused_own_quality_requests=192,analysis_required_locally=True,fresh_confirmation=False,goal_complete=False)
    finally:
        if live is not None:adapter.close(live)


class RuntimeAdapter:
    """Reuse the pinned curve runtime, with Native-only lifecycle branches."""
    def __init__(self,args,p):
        self.args=args;self.p=p;self.base=p['curve_module'].RuntimeAdapter(args,p)
        self.r=self.base.r

    def native_state(self,e):
        self.base.group.idle()
        state=dict(top_chunks=e.top_chunks,working=e.working is not None,injector=e.injector is not None,
            draft_runtime=e.draft_runtime is not None,target_hooks=len(e.observation.originals),round_observer_enabled=e.round_observer is not None)
        require(state==dict(top_chunks=None,working=False,injector=False,draft_runtime=False,target_hooks=0,
            round_observer_enabled=e.round_observer is not None),'Native acquired intervention/runtime/hooks')
        return state

    def signature(self,e):
        if not isinstance(e,self.base.baselines.BaselineEngine):return self.base.signature(e)
        self.native_state(e);require(e.native_t is not None and e.native_s is not None,'Native cache not allocated')
        result={}
        for name in ('native_t','native_s'):
            cache=getattr(e,name);result[name]=dict(identity=id(cache),key_ptr=cache.key_storage.data_ptr(),value_ptr=cache.value_storage.data_ptr(),capacity=cache.capacity)
        return dict(result,working=None,injector=None,runtime=None)

    def make(self,arm,quality):
        if arm['family']!='native':return self.base.make(arm,quality)
        b=self.base;b.group.idle();require(not b.live and not b.group.engines,'At most one private owner')
        observer=b.q.QualityObserver() if quality else None
        e=b.baselines.BaselineEngine(b.teacher,b.student,native_config(__import__('rescue.config',fromlist=['SystemConfig']).SystemConfig),
            vocab_size=len(b.tokenizer),top_chunks=None,round_observer=observer)
        b.live.append(e)
        if observer is not None:observer.engine=e
        self.native_state(e);return e

    def close(self,e):
        if not isinstance(e,self.base.baselines.BaselineEngine):return self.base.close(e)
        b=self.base;self.native_state(e);final=b.stats(e);started=time.perf_counter();b.q.detach_quality_observer(e)
        try:e.close();b.group.idle()
        finally:
            if e in b.live:b.live.remove(e)
        elapsed=time.perf_counter()-started;closed=b.stats(e)
        require(e.closed and closed['entries']==0 and not b.group.engines,'Native owner retained hooks/graphs')
        return dict(close_seconds=elapsed,final_runtime_stats=final,closed_runtime_stats=closed,engine_closed=True,hook_restored=True)

    def generate(self,e,arm,ids,reference,item,quality):
        if arm['family']!='native':return self.base.generate(e,arm,ids,reference,item,quality)
        b=self.base;require(b.live==[e] and not b.group.engines,'Unregistered/overlapping Native owner')
        self.native_state(e);snap=b.before();started=time.perf_counter()
        if quality:
            with b.q.record_draft_steps() as steps:row=e.generate(ids,b.tokenizer.eos_token_id)
        else:row=e.generate(ids,b.tokenizer.eos_token_id)
        elapsed=time.perf_counter()-started
        row.update(item,arm=arm,output_agrees=row['generated']==reference,runtime_stats=b.stats(e),owner_signature=self.signature(e),
            gpu_snapshot_before=snap,native_state=self.native_state(e))
        if quality:row.update(observed_rounds=e.round_observer.rounds,queries=steps,nonzero_injection=e.round_observer.nonzero_injection,
            native_t=self.r.draft_graph_smoke.native_digest(e.native_t),native_s=self.r.draft_graph_smoke.native_digest(e.native_s))
        return row,elapsed

    def validate_quality(self,row,reference,arm):
        if arm['family']=='native':check_native(row,reference,self.base.tokenizer.eos_token_id,True)
        else:self.base.validate_quality(row,reference,arm)

    def compare_own_quality(self,eager,graph):self.base.legacy.own_quality_pair(eager,graph)

    def validate_timing(self,row,reference,arm,cold):
        if arm['family']=='native':check_native(row,reference,self.base.tokenizer.eos_token_id,False)
        else:self.base.validate_timing(row,reference,arm,cold)

    def compare_timing(self,left,right,arm):
        self.base.common.check_pair(left,right)
        keys=ZERO_WORK if arm['family']=='native' else tuple(k for k in ZERO_WORK if k!='target_probe_rounds')
        for k in keys:require(left.get(k)==right.get(k),'Repeated/quality work counter changed: '+k)

    def telemetry(self):return self.base.telemetry()

    def assert_unchanged(self):
        b=self.base;b.group.idle();require(not b.live,'Owner still live at gate')
        require(b.h.bindings(self.r,self.args.source_root)==b.imports and b.baseline.check_full_bindings(self.args.source_root)==b.tree_bindings and b.baseline.flash_receipt()==b.flash,'Imported frozen runtime changed')
        require({p:sha(Path(p)) for p in self.p['assets']}==self.p['assets'],'Frozen assets changed')
        pinned_inventory(self.args.source_root,self.p['source_pins']);pinned_inventory(self.args.curve_root,self.p['curve_pins']);pinned_inventory(self.args.old_root,self.p['old_pins'])
        require(sha(Path(__file__))==self.p['entry_sha256'] and sha(self.args.preregistration)==self.p['preregistration_sha256'],'Entry/protocol changed')
        require(sha(self.args.prior_seal_root/'FINAL_ARTIFACTS_MANIFEST.json')==self.args.prior_manifest_sha256 and
            sha(self.args.prior_seal_root/'VERIFIED_COMPLETE.json')==self.args.prior_verified_sha256,'Prior seal changed')
        b.common.gpu_snapshot()

    def finish(self):
        for e in list(self.base.live):self.close(e)
        self.base.finish()


def run(args):
    p=prepare(args)
    if getattr(args,'validate_only',False):return dict(status='VALIDATED_NOT_LAUNCHED',new_generation_requests=690,reused_reference_requests=96,reused_own_quality_requests=192)
    output=args.output;output.mkdir(parents=True);p.update(counts=dict.fromkeys(EXPECTED,0),seen=set());adapter=None
    try:
        for filename,values in (('reused_reference.jsonl',p['reference_rows']),('reused_own_quality.jsonl',p['old_quality'])):
            for row in values:append(output/filename,row)
        write(output/'references.json',p['references'])
        for name in ('documents.jsonl','prime.json'):shutil.copyfile(args.curve_root/'results'/name,output/name)
        shutil.copyfile(args.preregistration,output/'PREREGISTRATION.json');shutil.copyfile(Path(__file__),output/'NATIVE_SD_SOURCE.py')
        (output/'prior_local_seal').mkdir()
        for name in ('FINAL_ARTIFACTS_MANIFEST.json','VERIFIED_COMPLETE.json'):shutil.copyfile(args.prior_seal_root/name,output/'prior_local_seal'/name)
        for root_label,source,pins in (('curve',args.curve_root,p['curve_pins']),('old',args.old_root,p['old_pins'])):
            for name,digest in pins.items():
                dest=output/'prior_consumed'/root_label/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,dest);require(sha(dest)==digest,'Prior proof changed copying')
        for name,digest in p['source_pins'].items():
            dest=output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(args.source_root/name,dest);require(sha(dest)==digest,'Frozen source changed copying')
        write(output/'REUSE_RECEIPT.json',dict(curve_root=str(args.curve_root),old_root=str(args.old_root),prior_manifest_sha256=args.prior_manifest_sha256,
            prior_verified_sha256=args.prior_verified_sha256,curve_artifact_pins=p['curve_pins'],old_artifact_pins=p['old_pins'],source_snapshot='source',
            prior_consumed_root='prior_consumed',reused_reference_requests=96,reused_own_quality_requests=192,new_generation_requests=690,
            canonical_inputs_all96={f'{i}:{n}':fingerprint(ids) for (i,n),ids in p['inputs'].items()},no_old_timing_reused=True,no_new_confirmation_access=True))
        adapter=RuntimeAdapter(args,p);b=adapter.base
        configs={m:asdict(native_config(__import__('rescue.config',fromlist=['SystemConfig']).SystemConfig)) if a['family']=='native' else asdict(b.legacy.own_config(a['backend'])) for m,a in arms().items()}
        write(output/'protocol.json',dict(schema=1,status='FROZEN_NATIVE_SD_COMPARISON',seed=SEED,source_indices=list(DOCS),lengths=list(LENGTHS),repeats=2,
            max_new_tokens=128,modes=list(MODES),arms=list(arms().values()),configs=configs,expected_counts=EXPECTED,expected_requests=TOTAL,
            shared_warm_sequence=p['sequence'],owner_order=[dict(length=n,mode=MODES[i]) for n,order in zip(LENGTHS,ORDER) for i in order],
            input_sha256={f'{i}:{n}':fingerprint(ids) for (i,n),ids in p['inputs'].items()},environment=b.environment,source_sha256=p['source_pins'],
            source_bindings=b.imports,tree_source_bindings=b.tree_bindings,flash_attention=b.flash,assets_sha256=p['assets'],policy_metadata=b.policy_meta,
            tokenizer_eos_token_id=b.tokenizer.eos_token_id,vocab_size=len(b.tokenizer),graph_max_entries=128,max_live_owners=1,
            model_preparation_seconds=b.preparation,template_prefill_precision='bfloat16',entry_source_sha256=p['entry_sha256'],preregistration_sha256=p['preregistration_sha256'],
            mapper_sha256=ASSET_SHA['mapper'],policy_sha256=ASSET_SHA['policy'],cost_sha256=ASSET_SHA['cost'],
            reused_reference_requests=96,reused_own_quality_requests=192,reuse_receipt_sha256=sha(output/'REUSE_RECEIPT.json'),
            primary='All32 warm whole-generate outer, every capture charged; constructor/cold/close separately',
            interpretation='Eager/Eager algorithm-package comparison; Graph/Native combines algorithm and execution backend, not Native Graph/optimal k',
            scope='Reused16-doc descriptive six-context study, no old timing splice, not final confirmation',
            no_training=True,no_search=True,fresh_confirmation=False,no_new_confirmation_access=True,paused_test_untouched=True,goal_complete=False,
            args={k:str(v) for k,v in vars(args).items()}))
        complete=execute_matrix(adapter,p,output);write(output/'COMPLETE.json',complete);return complete
    except BaseException as error:
        write(output/'FAILURE.json',dict(error_type=type(error).__name__,error=str(error),counts=p['counts'],logical_requests=len(p['seen']),
            preserve_partial=True,no_automatic_retry=True,no_replacement=True,goal_complete=False));raise
    finally:
        if adapter is not None:adapter.finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('curve-root','old-root','source-root','prior-seal-root','teacher','student','mapper','policy','cost','preregistration','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--prior-manifest-sha256',required=True);parser.add_argument('--prior-verified-sha256',required=True)
    parser.add_argument('--validate-only',action='store_true');print(json.dumps(run(parser.parse_args()),indent=2))


if __name__=='__main__':main()
