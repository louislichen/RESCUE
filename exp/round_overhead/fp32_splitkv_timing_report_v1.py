"""Portable matched four-arm timing report, no runtime/model imports.

Only actual saved whole-generate clocks are analyzed. All request-time JIT and
Graph capture costs remain charged; document CIs do not cover owner/clock drift.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import textwrap

LENGTHS=(2048,4096,6144,8192,12288,16384)
DOCS=(211,213,220,226,233,236,246,248,253,271,272,288,303,306,327,604)
MODES=('rescue_graph','retrieval32_k4_graph')
VARIANTS=('original','candidate')
WORK=('tokens','accepted','proposed','blocks','draft_forwards','selector_refreshes',
    'working_gather_layers','working_full_gather_layers','working_gathered_position_layers',
    'working_incremental_binds','draft_shadow_probes','projected_positions','rpc_calls')
OPTIONAL=('target_probe_rounds','selection_similarity_elements','injection_condition_steps',
    'static_affine_positions','working_restored_value_positions','rpc_bytes_sent')
TRACE=('round_index','prefix','cap','proposed','accepted','terminal','workset_tokens')
RUNTIME=('captures','replays','resident_steps','capture_host_seconds','shadow_updates')
ASSETS=dict(mapper='d08889e796ca2527e23241ff34a072ef640e2fdf3fe9a2f9ba5439975af04d9f',
    policy='3c333a470119c236afcc1308c18592842a535bee0b8f83c4fb4da359ff17eff5',
    cost='d92c843a84676c0cd6adaf9364bbf7e8c771d56548f1016b5720a3e04f371fa1')
PRIOR_MANIFEST='70e46d52c3f3f57b1efa8d1633b665f345d54d7656529c9b09e2d25e185bf41e'
PRIOR_VERIFIED='c5bf8390ec5349c6d7ce29a688d3b9b13d95b6fd11f006e46a7522b9be42dffc'
OPERATOR_TERMINAL='184674b6a991601378fb3e10ee4b7b4125477efad1400fd21599a5ac98e8c7b6'
SOURCES={
 'capsule/exp/round_overhead/fp32_splitkv_candidate_v1.py':'9f133cc824915afd09d9ad25a1b16e6cd721522e1f583cabd5e102235358cfd2',
 'capsule/tests/test_fp32_splitkv_candidate_v1.py':'d05aab7170f75451f1af4e93aac9a895c25e9b3f15ae9c59d11e2b6117bed8c0',
 'capsule/exp/round_overhead/fp32_splitkv_cuda_gate_v1.py':'62957e77c8d26b61dd89fc7692189ff0f46d61f3d77b9f6131b996eae26de98d',
 'capsule/tests/test_fp32_splitkv_cuda_gate_v1.py':'ae1f6665852f99f947c95ade0c7f766186fb80615b1260911a5f8b27aada78ef',
 'capsule/exp/round_overhead/fp32_splitkv_cuda_gate_v2.py':'1d4b340786851eb41c80489fb6c3b775101ba8125e97d50e81903d4536a52549',
 'capsule/tests/test_fp32_splitkv_cuda_gate_v2.py':'525ad7ade08a3fff6c8b593c88d8d32bc4d7f3399de35025f48146b4f1ff02b5',
 'capsule/exp/round_overhead/fp32_splitkv_cuda_gate_v3.py':'77501b123927d4ce0606da9017b14d8adaa84ce90059285b70735d8de1f3d39e',
 'capsule/tests/test_fp32_splitkv_cuda_gate_v3.py':'fd94317f0c1514a459f531c615ea3a39124d12c4e72f2700090c7549c4e336cb',
 'capsule/exp/round_overhead/fp32_splitkv_target_adapter_v1.py':'78092c417c10e25aa69ac9055fda59394e4349d021a277cfecfb43757e0e63a5',
 'capsule/tests/test_fp32_splitkv_target_adapter_v1.py':'a76c3eaa7630c33a8ff343a06978b6a1f96721a10d15744a12c210cfbe16367f',
 'capsule/exp/round_overhead/fp32_splitkv_model_gate_v1.py':'e91f613d760f9b94c9c67241fecdc39442d0e917b4ca4de21d75b400a3276bf3',
 'capsule/tests/test_fp32_splitkv_model_gate_v1.py':'ff8affd94fecccd6b311c4745ba2704f7448838d7f6cdb6d10cc0ec96f1e2d93',
 'capsule/exp/round_overhead/fp32_splitkv_timing_v1.py':'65832cea19b111740bf711e9d609e0bca61a92925a5ddc6183b388defffc62b2',
 'capsule/tests/test_fp32_splitkv_timing_v1.py':'ee4bce8fba8101ecb6a0ee3e2d3c1cdb661d23c13a81adaadc61510a2fff6907',
 'launch.sh':'011fc694a74f844d5d490c3c398133085f6d5f6c44b3974eba73a4b818da1a83'}
FIXED_PINS={'PREREGISTRATION.json':'2601405a52627fc3b2df4d8947a448381fbec1e042a680f111c82d77893972d0',
    'SOURCE_SHA256SUMS':'6f0bdc74aeabacd53abb08ab735f7b87b3daa20e9987b8e78512b9601eed0960'}
PROOF_FILES=('FINAL_ARTIFACTS_MANIFEST.json','VERIFIED_COMPLETE.json',
    'terminal/results/references.json','terminal/results/runtime.jsonl','terminal/results/protocol.json',
    'terminal/results/documents.jsonl','terminal/results/COMPLETE.json','terminal/results/QUALITY_PASSED.json',
    'terminal/results/REUSE_RECEIPT.json','terminal/tools/run_context_curve.py')
ARM_IDS=('original_rescue','candidate_rescue','original_retrieval','candidate_retrieval')
ORDER=((0,1,2,3),(3,2,1,0),(1,3,0,2),(2,0,3,1),(0,2,1,3),(3,1,2,0))
DRAWS=10000
SEED=20261031
CONTRASTS={
 'primary_candidate_rescue_vs_candidate_retrieval':('candidate_rescue','candidate_retrieval'),
 'original_rescue_vs_original_retrieval':('original_rescue','original_retrieval'),
 'rescue_execution_gain':('candidate_rescue','original_rescue'),
 'retrieval_execution_gain':('candidate_retrieval','original_retrieval')}
MODEL_PINS={'FINAL_ARTIFACTS_MANIFEST.json':'013273e977a8b4aeb97b3ceff845bc9b03ffbe0ab7f4752307c65c5b598f7320',
 'VERIFIED_COMPLETE.json':'44174149ed3b8b36984fb0ec51581933c61c863884d112ff9e1297170e83c328',
 'results/TERMINAL.json':'ae9f5870b37097699c791ee1fa139bc9de7fe0f3b50ab70d1666fd2deb9c8993',
 'PIPELINE_EXIT.json':'8c8d01cfcc5b443958a22f0d93f792f4abaf7a0e6a028ae8b5d642252bd5b42d',
 'TERMINAL_EXTERNAL_BARRIER.json':'d7646b2f8e05550be7929574d0257f6dad7e531223406c69f127b1650841bd83'}
FIGURES=('acceptance_length','acceptance_rate','warm_tps','relative_tps_gain','execution_gains',
 'finite_owner_costs','graph_mechanisms','target_scope_calls','round_work')
# Frozen only after the actual terminal collection is READY.
RUN_PINS={
 'results/PRIME.jsonl':'65c86b2b2861606e5d227c210a2f025a7d13e58f582a7d255205fc4efe882646',
 'results/RUNTIME.jsonl':'3c6086cdfceeda8372d9d616afd999f749146a69fae635d35a18e47c6acde000',
 'results/OWNERS.jsonl':'404030a8a05f05d5c41712d1f3d6f51ca1fa5b9d02619da2c2981d88bf8e4bc1',
 'results/WORK_DIFFERENCES.jsonl':'6892898302e1f538600f5aa7af1abfefb88ad668a5573600b8691fcc5fdc75d5',
 'results/TERMINAL.json':'e56ffb5f51c7d399cdce4b930c6a09a9f0265fffbc5cc93540f296efb3d64b1f',
 'PIPELINE_EXIT.json':'58d042d2584e756c5ea00d50e481ee85bb0896c2db5350b26a897caf5feee470',
 'COLLECTION_RECEIPT.json':'4b62a06fe49c28c962e7b99dd618b3a077b5c574d5d7e098992605d50fd8506c'}


def require(value,message):
    if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def issha(value):return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def uid(value):return type(value) is str and len(value)==32 and all(c in '0123456789abcdef' for c in value)
def finite(value):return type(value) in (int,float) and math.isfinite(value)
def near(a,b):return finite(a) and finite(b) and abs(a-b)<=1e-9*max(1,abs(a),abs(b))
def safe(root,name):
    p=PurePosixPath(name)
    require(p.parts and not p.is_absolute() and p.as_posix()==name and '..' not in p.parts and '\\' not in name,'Unsafe evidence path')
    path=Path(root).joinpath(*p.parts)
    require(not any(p.is_symlink() for p in (path,*path.parents)),'Symlink evidence forbidden')
    return path
def jsonl(path):
    data=Path(path).read_bytes();require(not data or data.endswith(b'\n'),'Truncated JSONL')
    rows=[json.loads(line) for line in data.splitlines()]
    require(all(type(r) is dict for r in rows),'Object JSONL rows required');return rows
def descriptors():
    return {a:dict(arm_id=a,variant='candidate' if a.startswith('candidate') else 'original',
        method='rescue' if a.endswith('rescue') else 'retrieval',mode=MODES[0] if a.endswith('rescue') else MODES[1]) for a in ARM_IDS}
def owner_order():
    return [(n,ARM_IDS[i]) for n,order in zip(LENGTHS,ORDER) for i in order]
def semantic_trace(row):return [{k:r[k] for k in TRACE} for r in row['trace']]
def difference(left,right):
    return dict(work_fields={k:dict(original=left.get(k),candidate=right.get(k)) for k in (*WORK,*OPTIONAL)
        if (k in left)!=(k in right) or left.get(k)!=right.get(k)},trajectory_equal=semantic_trace(left)==semantic_trace(right))
def digest(value):
    return type(value) is dict and issha(value.get('sha256')) and type(value.get('dtype')) is str and type(value.get('shape')) is list and all(type(x) is int and x>=0 for x in value['shape'])


def paired_statistics(rows):
    """One document-cluster draw shared across all contexts and four arms.

    Both repetitions are pooled within document before resampling. Cold rows
    never enter this estimator; no capture cost or slow request is subtracted.
    """
    import numpy as np
    indices=np.random.default_rng(SEED).integers(0,len(DOCS),size=(DRAWS,len(DOCS)),dtype=np.int64)
    output=dict(seed=SEED,draws=DRAWS,document_order=list(DOCS),repeats=2,common_document_resample=True,
        bootstrap_index_sha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
        pointwise_not_simultaneous=True,owner_clock_drift_not_covered=True,contexts={},missing_contexts=[])
    lookup={}
    for row in rows:
        require(row.get('phase')=='warm' and row.get('performance_eligible') is True,'Only actual warm eligible rows enter primary statistics')
        key=tuple(row.get(k) for k in ('length','arm_id','source_index','repeat'))
        require(key not in lookup and key[0] in LENGTHS and key[1] in ARM_IDS and key[2] in DOCS and type(key[3]) is int and key[3] in (0,1),'Duplicate/foreign warm statistical row')
        require(all(finite(row.get(k)) and row[k]>0 for k in ('tokens','outer_seconds','blocks','proposed')) and
            finite(row.get('accepted')) and 0<=row['accepted']<=row['proposed'],'Invalid raw statistical quantities')
        lookup[key]=row
    interval=lambda array:[float(x) for x in np.percentile(array,(2.5,97.5),method='linear')]
    for n in LENGTHS:
        keys=[(n,a,d,r) for a in ARM_IDS for d in DOCS for r in (0,1)]
        if not all(key in lookup for key in keys):output['missing_contexts'].append(n);continue
        arms={};samples={}
        for arm in ARM_IDS:
            arrays={field:np.asarray([sum(lookup[n,arm,d,r][field] for r in (0,1)) for d in DOCS],dtype=float)
                for field in ('tokens','outer_seconds','accepted','blocks','proposed','draft_forwards','prefill_seconds','decode_seconds','cleanup_seconds')}
            totals={key:float(value.sum()) for key,value in arrays.items()};metric=dict(documents=16,warm_requests=32,totals=totals)
            one={}
            for label,num,den in (('tps','tokens','outer_seconds'),('acceptance_length','accepted','blocks'),('acceptance_rate','accepted','proposed'),
                ('rounds_per_request','blocks',None),('draft_forwards_per_request','draft_forwards',None),
                ('whole_generate_seconds_per_round','outer_seconds','blocks'),('prefill_seconds_per_request','prefill_seconds',None),
                ('decode_seconds_per_request','decode_seconds',None)):
                if den is not None:
                    estimate=totals[num]/totals[den];bootstrap=arrays[num][indices].sum(axis=1)/arrays[den][indices].sum(axis=1)
                else:
                    estimate=totals[num]/32;bootstrap=arrays[num][indices].sum(axis=1)/32
                metric[label]=dict(point=estimate,ci95=interval(bootstrap));one[label]=bootstrap
            arms[arm]=metric;samples[arm]=one
        contrasts={}
        for label,(num,den) in CONTRASTS.items():
            estimate=100*(arms[num]['tps']['point']/arms[den]['tps']['point']-1)
            bootstrap=100*(samples[num]['tps']/samples[den]['tps']-1)
            contrasts[label]=dict(numerator=num,denominator=den,point_percent=estimate,ci95_percent=interval(bootstrap))
        primary=contrasts['primary_candidate_rescue_vs_candidate_retrieval'];original=contrasts['original_rescue_vs_original_retrieval']
        interaction=100*((arms['candidate_rescue']['tps']['point']/arms['original_rescue']['tps']['point'])/
            (arms['candidate_retrieval']['tps']['point']/arms['original_retrieval']['tps']['point'])-1)
        output['contexts'][str(n)]=dict(arms=arms,contrasts=contrasts,
            primary_point_exceeds10_percent=primary['point_percent']>10,primary_ci_lower_positive=primary['ci95_percent'][0]>0,
            threshold_statistics_pass=primary['point_percent']>10 and primary['ci95_percent'][0]>0,
            descriptive_interaction=dict(ratio_of_execution_gains_percent=interaction,
                primary_minus_original_percentage_points=primary['point_percent']-original['point_percent'],
                confidence_interval=None,not_an_extra_success_gate=True))
    output['all_six_statistical_gates_pass']=len(output['contexts'])==6 and all(c['threshold_statistics_pass'] for c in output['contexts'].values())
    return output


def row_metrics(row,reference):
    own=row['method']=='rescue';steps=row.get('trace',[])
    require(row.get('generated')==reference and row.get('tokens')==len(reference) and row.get('output_agrees') is True,'Frozen original canonical Tokens differ')
    require(row.get('observer_enabled') is False and row.get('profile_enabled') is False and row.get('scope_metadata_append_kept') is True and
        row.get('all_request_jit_and_captures_charged') is True and all(k not in row for k in
            ('observed_rounds','queries','native_t','native_s','timeline','cost_samples')),'Timing gained observer/digest/profiler or removed true cost')
    require(finite(row.get('outer_seconds')) and row['outer_seconds']>0 and near(row.get('outer_tps'),row['tokens']/row['outer_seconds']),'Whole-generate clock inconsistent')
    require(steps and row.get('blocks')==len(steps),'Actual verification trace missing')
    for i,step in enumerate(steps):
        prefix,cap,proposed,accepted=[step.get(k) for k in ('prefix','cap','proposed','accepted')]
        require(all(type(x) is int for x in (prefix,cap,proposed,accepted)),'Noninteger trace')
        remaining=128-(prefix-(row['length']-1))
        require(cap in ((min(2,remaining),min(4,remaining)) if own else (min(4,remaining),)) and
            1<=proposed<=cap<=4 and 0<=accepted<=proposed and step.get('round_index')==i and
            step.get('terminal') is (i==len(steps)-1) and prefix==(row['length']-1 if not i else steps[i-1]['prefix']+steps[i-1]['accepted']+1) and
            type(step.get('workset_tokens')) is int and 0<step['workset_tokens']<=(512 if own else 1024),'Causal trace cap/prefix/workset differs')
    require(row.get('accepted')==sum(r['accepted'] for r in steps) and row.get('proposed')==sum(r['proposed'] for r in steps) and
        near(row.get('acceptance_rate'),row['accepted']/row['proposed']),'Acceptance totals differ')
    last=steps[-1];progress=last['prefix']-(row['length']-1)
    require(row['tokens'] in (progress+last['accepted'],progress+last['accepted']+1),'Terminal output accounting differs')
    require(row.get('draft_forwards')==row['proposed']+sum(s['accepted']==s['proposed'] and not s['terminal'] for s in steps),'Draft proposal/catchup accounting differs')
    require(row.get('performance_eligible') is (row['phase']=='warm'),'Cold/primary eligibility changed')
    require(all(finite(row.get(k)) and row[k]>=0 for k in ('prefill_seconds','decode_seconds','cleanup_seconds')) and
        near(sum(row[k] for k in ('prefill_seconds','decode_seconds','cleanup_seconds')),row.get('total_seconds')) and
        row['outer_seconds']+1e-6>=row['total_seconds'],'Stage boundaries differ from whole-generate clock')
    return {k:row[k] for k in ('tokens','outer_seconds','accepted','blocks','proposed','draft_forwards')}


def graph_step(row,previous=None):
    stats=row.get('runtime_stats',{});runtime=row.get('draft_runtime',{});delta=runtime.get('request',{});keys=stats.get('graph_keys',[])
    require(stats.get('mode')==runtime.get('mode')=='cuda_graph' and stats.get('max_graph_entries')==128 and
        all(type(delta.get(k)) is int and delta[k]>=0 for k in RUNTIME if k!='capture_host_seconds') and
        finite(delta.get('capture_host_seconds')) and delta['capture_host_seconds']>=0,'Graph runtime/delta scope differs')
    encoded={json.dumps(k,sort_keys=True) for k in keys};previous=previous or dict(keys=set(),totals=dict.fromkeys(RUNTIME,0))
    require(stats.get('entries')==stats.get('captures')==len(keys)==len(encoded)<=128 and previous['keys']<=encoded and
        len(encoded-previous['keys'])==delta['captures'] and delta['replays']==delta['shadow_updates']==row['draft_forwards']>0 and
        delta['resident_steps']==0,'Actual private Graph capture/replay ledger differs')
    totals={k:previous['totals'][k]+delta[k] for k in RUNTIME}
    require(all(near(totals[k],stats.get(k)) and near(totals[k],runtime.get('lifetime',{}).get(k)) for k in RUNTIME),
        'Lifetime Graph counts/capture costs not preserved')
    signature=row.get('owner_signature',{})
    for role in ('native_t','native_s','working'):
        require(all(type(signature.get(role,{}).get(k)) is int and signature[role][k]>0 for k in ('identity','key_ptr','value_ptr','capacity')),'Actual private cache pointers missing')
    require(signature.get('runtime',{}).get('bound_ptrs')==[signature[role][k] for role in ('native_s','working') for k in ('key_ptr','value_ptr')],
        'Graph not bound to persistent owner KV')
    return dict(keys=encoded,totals=totals)

def scope_counts(summary,variant,closed=True):
    require(summary.get('state')==('closed' if closed else ('active' if variant=='candidate' else 'disabled')) and
        summary.get('enabled') is (variant=='candidate'),'Target scope lifecycle differs')
    require(summary.get('cpu_metadata_only') is True and summary.get('tensor_readbacks')==summary.get('tensor_allocations_by_adapter')==0 and
        summary.get('performance_gate')==summary.get('token_gate')=='NOT_RUN','Adapter metadata-only scope changed')
    records=summary.get('records',[])
    require(type(records) is list and summary.get('calls')==len(records),'Scope record counter differs')
    count={action:[0]*28 for action in ('candidate','original')}
    for row in records:
        require(type(row.get('layer')) is int and 0<=row['layer']<28 and row.get('action') in count and row.get('outcome')=='returned',
            'Missing actual Target-layer return')
        count[row['action']][row['layer']]+=1
    for action in count:require(sum(count[action])==summary.get(action+'_calls'),'Scope action counter differs')
    if variant=='original':require(not records,'Original variant unexpectedly replaced Target attention')
    else:
        require(len(records)%28==0 and min(count['candidate'])>0 and len(set(count['candidate']))==1,'All28 real candidate layers required')
        for offset in range(0,len(records),28):
            block=records[offset:offset+28]
            require([r['layer'] for r in block]==list(range(28)) and
                len({(r['action'],r['query_tokens'],r['key_tokens']) for r in block})==1,'Target-layer block metadata differs')
    return count

def historical_proof(root):
    root=Path(root).absolute()
    require(sha(safe(root,PROOF_FILES[0]))==PRIOR_MANIFEST and sha(safe(root,PROOF_FILES[1]))==PRIOR_VERIFIED,'Original six-length parent seal changed')
    manifest=read(safe(root,PROOF_FILES[0]));verified=read(safe(root,PROOF_FILES[1]))
    require(verified.get('passed') is True and verified.get('six_lengths_complete') is True and
        verified.get('final_artifacts_manifest_sha256')==PRIOR_MANIFEST,'Old curve not sealed complete')
    for name in PROOF_FILES[2:]:require(sha(safe(root,name))==manifest['file_sha256'][name],'Parent-manifest proof mismatch: '+name)
    protocol=read(safe(root,'terminal/results/protocol.json'))
    require(len(protocol.get('source_sha256',{}))==167 and protocol.get('mapper_sha256')==ASSETS['mapper'] and
        protocol.get('policy_sha256')==ASSETS['policy'] and protocol.get('cost_sha256')==ASSETS['cost'],'Old frozen source/assets differ')
    require(read(safe(root,'terminal/results/COMPLETE.json')).get('passed') is True and
        read(safe(root,'terminal/results/COMPLETE.json')).get('logical_requests')==798,'Old canonical execution incomplete')
    references=read(safe(root,'terminal/results/references.json'));history=jsonl(safe(root,'terminal/results/runtime.jsonl'))
    documents={r['source_index']:r for r in jsonl(safe(root,'terminal/results/documents.jsonl'))}
    return dict(protocol=protocol,references=references,history=history,documents=documents,
        pins={name:sha(safe(root,name)) for name in PROOF_FILES})


def model_proof(root):
    root=Path(root).absolute()
    for name,pin in MODEL_PINS.items():require(sha(safe(root,name))==pin,'Sealed evaluated-model proof changed: '+name)
    verified=read(safe(root,'VERIFIED_COMPLETE.json'));terminal=read(safe(root,'results/TERMINAL.json'))
    require(verified.get('quality_gate_complete') is True and verified.get('canonical_tokens_exact') is True and
        verified.get('model_generation_calls')==verified.get('private_owners_closed')==48 and
        verified.get('performance_measured') is False and verified.get('final_artifacts_manifest_sha256')==MODEL_PINS['FINAL_ARTIFACTS_MANIFEST.json'],
        'Prior model quality seal is not the evaluated48-call pass')
    require(terminal.get('status')=='TARGET_TOKEN_GATE_PASSED' and terminal.get('passed') is True and terminal.get('actual_cuda') is True and
        terminal.get('frozen_canonical_token_exact') is True and all(terminal.get(k)==48 for k in
        ('attempted_calls','returned_calls','completed_calls','validated_calls','private_owners_closed')) and
        terminal.get('performance_measured') is False,'Actual prior model Token gate incomplete')
    return {name:sha(safe(root,name)) for name in MODEL_PINS}


def equal_work(left,right):
    return left.get('generated')==right.get('generated') and semantic_trace(left)==semantic_trace(right) and all(
        k in left and k in right and left[k]==right[k] for k in WORK) and all(
        (k in left)==(k in right) and (k not in left or left[k]==right[k]) for k in OPTIONAL)


def collection_check(root,warnings):
    path=safe(root,'COLLECTION_RECEIPT.json')
    if not path.is_file():
        warnings.append('No saved collection receipt; remote terminal/process barriers are not established by this report.')
        return dict(present=False,signed_attestation=False)
    receipt=read(path);hashes=receipt.get('file_sha256',{});sizes=receipt.get('file_size_bytes',{})
    require(type(hashes) is dict and hashes and set(hashes)==set(sizes) and len(hashes)==receipt.get('collected_files') and
        all(issha(h) and sha(safe(root,name))==h and safe(root,name).stat().st_size==sizes[name] for name,h in hashes.items()),
        'Collection local SHA/size inventory mismatch')
    for name in ('remote_read_only_barrier_before','remote_read_only_barrier_after'):
        barrier=receipt.get(name,{})
        require(barrier.get('file_sha256')==hashes and barrier.get('file_size_bytes')==sizes,'Collection before/after full inventory mismatch')
        external=barrier.get('external_terminal_barrier',{})
        require(external.get('status')=='TERMINAL_EXTERNAL_BARRIER_PASSED' and external.get('passed') is True and
            all(external.get(k) is True for k in ('tmux_absent','model_process_absent','model_children_absent','gpu_compute_idle')) and
            external.get('root')==receipt.get('remote_root') and
            external.get('terminal_sha256')==sha(safe(root,'results/TERMINAL.json')) and
            all(external.get(k,{}).get('exit_code')==1 for k in ('tmux_observation','process_observation','children_observation')) and
            external.get('gpu_observation',{}).get('exit_code')==0 and external['gpu_observation'].get('stdout')=='' and
            external.get('identity_observation',{}).get('exit_code')==0 and
            external['identity_observation'].get('stdout','').strip()=='GPU-9db4f7b5-1447-9a2a-5c1b-80379d909fd9',
            'Saved external terminal/process/GPU barrier incomplete')
    require(receipt.get('scp_exit_codes') and all(type(x) is int and x==0 for x in receipt['scp_exit_codes']) and
        receipt.get('local_matches_both_remote_inventories') is True and receipt.get('weights_transferred') is False and
        receipt.get('compiled_cache_transferred') is False,'Collection scope/transport incomplete')
    return dict(present=True,local_files_rehashed=len(hashes),double_barrier_matches_local_bytes=True,saved_external_barriers_checked=True,
        signed_attestation=False,remote_reobserved=False,provenance='MAIN_AGENT_TOOL_OBSERVATION_TRANSCRIPTION')


def analyze(root,curve_proof_root,model_proof_root):
    root=Path(root).absolute();issues=[];warnings=[]
    result=dict(status='PARTIAL_MATCHED_TIMING',planned_calls=792,planned_warm_calls=768,planned_cold_calls=24,planned_owners=24,
        valid_warm_rows=[],valid_cold_rows=[],owner_diagnostics=[],work_differences=[],issues=issues,warnings=warnings,
        all_outputs_canonical=False,counts={},statistics=None,complete_raw_evidence=False,scientific_gate_pass=False,
        goal_complete=False,goal_status='ACTIVE',fresh_confirmation=False,production_speed_claim=False)
    try:
        for name,pin in {**SOURCES,**FIXED_PINS,**RUN_PINS}.items():require(sha(safe(root,name))==pin,'Pinned source/raw changed: '+name)
        sums={line.split('  ',1)[1]:line.split('  ',1)[0] for line in safe(root,'SOURCE_SHA256SUMS').read_text().splitlines()}
        require(sums=={n.removeprefix('capsule/') if n.startswith('capsule/') else '../launch.sh':h for n,h in SOURCES.items()},'Fifteen-source inventory differs')
        old=historical_proof(curve_proof_root);model_pins=model_proof(model_proof_root);protocol=old['protocol']
        pre=read(safe(root,'PREREGISTRATION.json'));plan=read(safe(root,'results/PLAN.json'));terminal=read(safe(root,'results/TERMINAL.json'))
        exitrow=read(safe(root,'PIPELINE_EXIT.json'));reuse=read(safe(root,'results/REUSE_RECEIPT.json'));env=read(safe(root,'results/ENVIRONMENT.json'))
        result['synthetic_program_test_only']=pre.get('synthetic_program_test_only') is True
        for obj in (pre,plan):
            require(obj.get('lengths')==list(LENGTHS) and obj.get('source_indices')==list(DOCS) and obj.get('arm_ids')==list(ARM_IDS) and
                obj.get('repeats')==2 and obj.get('max_new_tokens')==obj.get('graph_max_entries')==128 and
                obj.get('owner_order')==[[ARM_IDS[i] for i in order] for order in ORDER] and
                all(obj.get(k) is False for k in ('quality_observer','native_digest','profiler')) and
                obj.get('all_request_jit_and_captures_charged') is True and obj.get('target_adapter_metadata_append_kept') is True,
                'Frozen balanced four-arm792 protocol differs')
        require(pre.get('generation_calls')==792 and pre.get('private_owners')==24 and pre.get('statistics',{}).get('bootstrap_replicates')==DRAWS and
            pre['statistics'].get('bootstrap_seed')==SEED and pre.get('model_terminal_sha256')==MODEL_PINS['results/TERMINAL.json'] and
            pre.get('terminal_barrier_sha256')==MODEL_PINS['TERMINAL_EXTERNAL_BARRIER.json'] and
            pre.get('runner_sha256')==SOURCES['capsule/exp/round_overhead/fp32_splitkv_timing_v1.py'],'Statistical/predecessor source contract differs')
        require(sha(safe(root,'inputs/MODEL_TERMINAL_BARRIER.json'))==MODEL_PINS['TERMINAL_EXTERNAL_BARRIER.json'],'Pinned prelaunch external model barrier differs')
        require(read(safe(root,'results/PREREGISTRATION.json'))==pre and
            exitrow.get('terminal_sha256')==sha(safe(root,'results/TERMINAL.json')) and exitrow.get('analysis_required') is True and
            exitrow.get('goal_complete') is False,'Exit/terminal/preregistration binding differs')
        require(terminal.get('runner_sha256')==SOURCES['capsule/exp/round_overhead/fp32_splitkv_timing_v1.py'] and
            terminal.get('goal_complete') is False,'Timing runner/goal binding differs')
        sequence=protocol['shared_warm_sequence']
        require(len(sequence)==32 and {(r['source_index'],r['repeat']) for r in sequence}=={(d,r) for d in DOCS for r in (0,1)} and
            reuse.get('shared_warm_sequence')==sequence and reuse.get('old_source_sha256')==protocol['source_sha256'] and
            reuse.get('assets_sha256')==protocol['assets_sha256'] and reuse.get('model_terminal_sha256')==MODEL_PINS['results/TERMINAL.json'] and
            reuse.get('terminal_barrier_sha256')==MODEL_PINS['TERMINAL_EXTERNAL_BARRIER.json'] and reuse.get('no_new_ar') is True and
            reuse.get('no_new_corpus') is True,'Unchanged assets/source/sequence/reuse chain differs')
        external=('exp/round_overhead/fp32_splitkv_model_gate_v1.py','tests/test_fp32_splitkv_model_gate_v1.py',
            'exp/round_overhead/fp32_splitkv_target_adapter_v1.py','tests/test_fp32_splitkv_target_adapter_v1.py',
            'exp/round_overhead/fp32_splitkv_candidate_v1.py')
        old_reuse=read(safe(curve_proof_root,'terminal/results/REUSE_RECEIPT.json'))
        manifest=read(safe(curve_proof_root,'FINAL_ARTIFACTS_MANIFEST.json'))['file_sha256']
        curve_names=('results/protocol.json','results/COMPLETE.json','results/QUALITY_PASSED.json','results/REUSE_RECEIPT.json',
            'results/references.json','results/documents.jsonl','results/prime.json','results/runtime.jsonl','results/own_quality.jsonl',
            'results/reused_own_quality.jsonl','PIPELINE_EXIT.json','LAUNCH_MANIFEST.json','tools/run_context_curve.py')
        require(reuse.get('external_source_sha256')=={name:SOURCES['capsule/'+name] for name in external} and
            reuse.get('old_artifact_pins')==old_reuse['old_artifact_pins'] and
            reuse.get('curve_artifact_pins')=={name:manifest['terminal/'+name] for name in curve_names},'Consumed frozen-file/source ledgers differ')
        require(env.get('observers') is False and env.get('profiler') is False and env.get('max_live_owners')==1 and
            env.get('max_new_tokens')==env.get('graph_max_entries')==128 and env.get('scope_metadata_append_kept') is True and
            env.get('configs')=={mode:protocol['configs'][mode] for mode in MODES} and env.get('policy_metadata')==protocol['policy_metadata'] and
            env.get('source_bindings')==protocol['source_bindings'] and env.get('tree_source_bindings')==protocol['tree_source_bindings'] and
            env.get('flash_attention')==protocol['flash_attention'],'Frozen runtime/assets/configuration changed')
        require(all(env.get('environment',{}).get(k)==protocol['environment'][k] for k in
            ('gpu_name','gpu_uuid','torch','cuda','target_dtype','draft_dtype','prefix_dtype','tf32')),'Frozen GPU/backend/precision changed')
        references=read(safe(root,'results/references.json'));inputs=read(safe(root,'results/INPUTS.json'))
        require(references==old['references'] and set(inputs)=={f'{d}:{n}' for d in DOCS for n in LENGTHS},'Old canonical reference/input matrix changed')
        for d in DOCS:
            saved=old['documents'][d]
            for n in LENGTHS:
                key=f'{d}:{n}';ids=saved['prefix_ids']+saved['document_ids'][:n-30]+saved['suffix_ids']
                require(len(ids)==n and inputs[key]==ids and hashlib.sha256(json.dumps(ids).encode()).hexdigest()==protocol['input_sha256'][key],
                    'Original saved input differs: '+key)
        cold=jsonl(safe(root,'results/PRIME.jsonl'));warm=jsonl(safe(root,'results/RUNTIME.jsonl'));owners=jsonl(safe(root,'results/OWNERS.jsonl'))
        combined=sorted(cold+warm,key=lambda r:r.get('execution_index',-1));require(len(combined)<=792 and len(owners)<=24,'Too many calls/owners')
        require([r.get('execution_index') for r in combined]==list(range(len(combined))),'Returned raw call order is not an exact prefix')
        planned=[]
        for n,arm in owner_order():
            planned.extend([dict(length=n,arm_id=arm,source_index=271,phase='cold_prime')]+[dict(r,length=n,arm_id=arm,phase='warm') for r in sequence])
        request_ids=set();owner_rows={};repeated={};actual_by_key={};valid_cold=[];valid_warm=[];owner_ids=set()
        for i,row in enumerate(combined):
            descriptor=planned[i];arm=descriptors()[descriptor['arm_id']]
            try:
                require(all(row.get(k)==v and type(row.get(k)) is type(v) for k,v in {**descriptor,**arm}.items()),'Call context/arm/order differs')
                require(uid(row.get('request_id')) and row['request_id'] not in request_ids and uid(row.get('owner_id')),'Missing/reused request/owner identity')
                request_ids.add(row['request_id']);row_metrics(row,references[f"{row['source_index']}:{row['length']}"])
                key=(row['length'],row['arm_id']);entry=owner_rows.setdefault(key,dict(rows=[],ledger=None,signature=row['owner_signature'],owner_id=row['owner_id']))
                require(row['owner_id']==entry['owner_id'] and row['owner_signature']==entry['signature'],'Persistent owner/cache identity changed')
                if not entry['rows']:require(row['owner_id'] not in owner_ids,'Owner reused across arms/contexts');owner_ids.add(row['owner_id'])
                entry['ledger']=graph_step(row,entry['ledger']);entry['rows'].append(row)
                if row['phase']=='cold_prime':require(row['draft_runtime']['request']['captures']>0,'Cold owner borrowed prewarmed Graph execution')
                history=[h for h in old['history'] if (h['length'],h['source_index'],h['mode'],h.get('repeat'))==
                    (row['length'],row['source_index'],row['mode'],row.get('repeat',0))]
                if row['variant']=='original':require(len(history)==1 and equal_work(history[0],row),'Original work differs from matching frozen historical row')
                repeat_key=(row['length'],row['arm_id'],row['source_index'])
                if repeat_key in repeated:require(equal_work(repeated[repeat_key],row),'Repeated candidate/original trajectory/work changed')
                else:repeated[repeat_key]=row
                if row['phase']=='warm':actual_by_key[row['length'],row['arm_id'],row['source_index'],row['repeat']]=row;valid_warm.append(row)
                else:valid_cold.append(row)
            except (ValueError,KeyError,TypeError,IndexError,ZeroDivisionError) as error:issues.append(f'call{i:03d}: '+str(error))
        for i,owner in enumerate(owners):
            n,arm_id=owner_order()[i]
            try:
                require((owner.get('length'),owner.get('arm_id'))==(n,arm_id),'Owner order/matrix differs')
                entry=owner_rows[n,arm_id];requests=entry['rows'];prime=[r for r in requests if r['phase']=='cold_prime'];timed=[r for r in requests if r['phase']=='warm']
                close=owner.get('close_receipt',{})
                require(len(prime)==1 and len(timed)==32 and owner.get('owner_id')==entry['owner_id'] and owner.get('owner_signature')==entry['signature'] and
                    owner.get('primary_error') is None and owner.get('cleanup_errors')==[] and close.get('engine_closed') is True and close.get('hook_restored') is True and
                    close.get('closed_runtime_stats',{}).get('entries')==0 and close.get('final_runtime_stats')==requests[-1]['runtime_stats'],'Owner lifecycle/Graph restoration incomplete')
                require(owner.get('warm_request_ids')==[r['request_id'] for r in timed] and owner.get('primary_warm_requests')==32 and
                    owner.get('max_live_owners')==1 and owner.get('all_captures_charged') is True,'Owner warm-request/capture accounting differs')
                constructor,cold_seconds,close_seconds=owner.get('constructor_scope_seconds'),owner.get('cold_prime_seconds'),close.get('scope_and_close_seconds')
                require(all(finite(v) and v>=0 for v in (constructor,cold_seconds,close_seconds)) and near(cold_seconds,prime[0]['outer_seconds']) and
                    near(owner.get('cold_complete_seconds'),constructor+cold_seconds) and near(owner.get('warm_outer_seconds_sum'),sum(r['outer_seconds'] for r in timed)) and
                    owner.get('warm_tokens')==sum(r['tokens'] for r in timed) and owner.get('cold_prime_tokens')==prime[0]['tokens'] and
                    near(owner.get('warm_tps'),sum(r['tokens'] for r in timed)/sum(r['outer_seconds'] for r in timed)),'Constructor/cold/warm clock accounting differs')
                service=constructor+cold_seconds+sum(r['outer_seconds'] for r in timed)+close_seconds;tokens=sum(r['tokens'] for r in requests)
                require(near(owner.get('accounted_service_seconds'),service) and owner.get('accounted_service_tokens')==tokens and near(owner.get('accounted_service_tps'),tokens/service),
                    'Full finite-owner service clock accounting differs')
                counts=scope_counts(owner.get('scope_summary',{}),descriptors()[arm_id]['variant'])
                qn=Counter((r['action'],r['query_tokens'],r['key_tokens']) for r in owner['scope_summary']['records'])
                result['owner_diagnostics'].append(dict(length=n,arm_id=arm_id,owner_id=owner['owner_id'],constructor_seconds=constructor,
                    cold_seconds=cold_seconds,close_seconds=close_seconds,warm_seconds=sum(r['outer_seconds'] for r in timed),finite_service_seconds=service,
                    finite_service_tps=tokens/service,cold_complete_seconds=constructor+cold_seconds,
                    warm_captures=sum(r['draft_runtime']['request']['captures'] for r in timed),cold_captures=prime[0]['draft_runtime']['request']['captures'],
                    warm_capture_host_seconds=sum(r['draft_runtime']['request']['capture_host_seconds'] for r in timed),
                    cold_capture_host_seconds=prime[0]['draft_runtime']['request']['capture_host_seconds'],
                    graph_replays=sum(r['draft_runtime']['request']['replays'] for r in requests),graph_entries=requests[-1]['runtime_stats']['entries'],
                    target_layer_calls=counts,scope_qn_distribution=[dict(action=action,query_tokens=q,key_tokens=k,calls=count)
                        for (action,q,k),count in sorted(qn.items())]))
            except (ValueError,KeyError,TypeError,IndexError,ZeroDivisionError) as error:issues.append(f'owner{i:02d}: '+str(error))
        differences=jsonl(safe(root,'results/WORK_DIFFERENCES.jsonl')) if safe(root,'results/WORK_DIFFERENCES.jsonl').is_file() else []
        for i,saved in enumerate(differences):
            n=LENGTHS[i//64];method=('rescue','retrieval')[(i%64)//32];d=DOCS[(i%32)//2];repeat=i%2
            require((saved.get('length'),saved.get('method'),saved.get('source_index'),saved.get('repeat'))==(n,method,d,repeat),'Work-difference matrix/order differs')
            computed=difference(actual_by_key[n,'original_'+method,d,repeat],actual_by_key[n,'candidate_'+method,d,repeat])
            require(saved==dict(length=n,method=method,source_index=d,repeat=repeat,**computed,candidate_equals_original_required=False,
                fullstate_equality_not_measured=True),'Saved old/new work-difference evidence differs')
            result['work_differences'].append(saved)
        closed_keys={(r['length'],r['arm_id']) for r in result['owner_diagnostics']}
        eligible=[r for r in valid_warm if (r['length'],r['arm_id']) in closed_keys]
        stats=paired_statistics(eligible)
        counts=dict(prime_requests=len(cold),timing_requests=len(warm),owners=len(owners))
        result.update(valid_warm_rows=[{k:r[k] for k in ('length','arm_id','source_index','repeat','phase','performance_eligible','tokens','outer_seconds',
            'accepted','blocks','proposed','draft_forwards','prefill_seconds','decode_seconds','cleanup_seconds')} for r in eligible],
            valid_cold_rows=[dict(length=r['length'],arm_id=r['arm_id'],tokens=r['tokens'],outer_seconds=r['outer_seconds']) for r in valid_cold],
            counts=counts,statistics=stats,terminal=terminal,original_proof_sha256=old['pins'],model_proof_sha256=model_pins,
            used_source_sha256={**SOURCES,**FIXED_PINS,**RUN_PINS},collection=collection_check(root,warnings))
        complete=(not issues and counts==dict(prime_requests=24,timing_requests=768,owners=24) and len(valid_warm)==768 and len(valid_cold)==24 and
            len(result['owner_diagnostics'])==24 and len(differences)==384 and terminal.get('status')=='TIMING_RAW_COMPLETE_ANALYSIS_REQUIRED' and
            terminal.get('passed') is True and terminal.get('actual_cuda') is True and terminal.get('performance_measured') is True and
            terminal.get('performance_gate')=='ANALYSIS_REQUIRED' and terminal.get('all_old_canonical_tokens') is True and
            terminal.get('original_historical_work_equal') is True and terminal.get('candidate_repeated_work_equal') is True and
            terminal.get('observers') is terminal.get('profiler') is False and terminal.get('paired_document_count')==16 and
            terminal.get('repeats')==2 and terminal.get('owner_failure_receipts')==[] and
            all(terminal.get(k)==792 for k in ('attempted_calls','returned_calls','validated_calls','logical_requests')) and
            terminal.get('counts')==counts and exitrow.get('exit_code')==0)
        result.update(status='COMPLETE_MATCHED_DEVELOPMENT_TIMING' if complete else 'PARTIAL_MATCHED_TIMING',
            complete_raw_evidence=complete,all_outputs_canonical=complete,scientific_gate_pass=complete and stats['all_six_statistical_gates_pass'])
    except (OSError,ValueError,TypeError,KeyError,IndexError,ZeroDivisionError) as error:issues.append(str(error))
    return result


def figures(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':12,'axes.labelsize':12,'axes.titlesize':13,'xtick.labelsize':12,'ytick.labelsize':12})
    tag='COMPLETE matched development timing' if summary['complete_raw_evidence'] else 'PARTIAL — completed balanced contexts only'
    if summary.get('synthetic_program_test_only'):tag='SYNTHETIC CPU MOCK — '+tag
    cells=(summary.get('statistics') or {}).get('contexts',{});owners=summary['owner_diagnostics']
    labels=[f'{n//1024}K\n{n}' for n in LENGTHS]
    colors={'original_rescue':'#555555','candidate_rescue':'#b66c30','original_retrieval':'#9a9a9a','candidate_retrieval':'#4a785d'}
    names={'original_rescue':'Original Target · RESCUE','candidate_rescue':'Candidate Target · RESCUE',
        'original_retrieval':'Original Target · retrieval Top32/k4','candidate_retrieval':'Candidate Target · retrieval Top32/k4'}
    styles={'original_rescue':'--','candidate_rescue':'-','original_retrieval':':','candidate_retrieval':'-'}
    caption=(f"Verified {len(owners)}/24 owners, {len(summary.get('valid_warm_rows',[]))}/768 warm rows; planned16 reused docs ×2 repeats, cold271 +32warm/owner; H100.\n"
        'Qwen2-7B/1.5B; Target FP32, prefix/Draft BF16, TF32 off; cap128/EOS; Mapper/policy/oldcost frozen; all true captures charged.\n'
        'Warm CIs: paired whole-document bootstrap, 10000 draws / seed20261031, pointwise95%; not simultaneous or fresh confirmation.\n'
        'One nonconcurrent owner block/arm/context: CIs do not cover owner/clock drift. No fully-prewarmed production or SOTA claim.')
    def save(fig,name,title,mechanism=False,bottom=.22,top=.92):
        suffix='\nMechanism/owner points are descriptive; no independently repeated-owner CI.' if mechanism else ''
        fig.suptitle(tag+'\n'+'\n'.join(textwrap.wrap(title,width=115)),fontsize=14);fig.text(.5,.012,caption+suffix,ha='center',fontsize=12)
        fig.tight_layout(rect=(0,bottom,1,top));fig.savefig(output/(name+'.png'),dpi=170);fig.savefig(output/(name+'.svg'));plt.close(fig)
    def axis(ax,ylabel):
        ax.set_xticks(range(6),labels);ax.set_xlim(-.2,5.2);ax.set_xlabel('Actual input context length (Tokens; K=1024)')
        ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.25)
        if not cells:ax.text(.5,.5,'No complete balanced timing context',ha='center',transform=ax.transAxes)
    def points(ax,arm,metric,label=None,offset=0):
        rows=[(LENGTHS.index(int(n)),c['arms'][arm][metric]) for n,c in cells.items()]
        x=[r[0]+offset for r in rows];p=[r[1]['point'] for r in rows]
        ax.plot(x,p,marker='o',linestyle=styles[arm],color=colors[arm],label=label or names[arm])
        for value,(_,r) in zip(x,rows):
            lo,hi=r['ci95'];ax.vlines(value,lo,hi,color=colors[arm]);ax.hlines((lo,hi),value-.035,value+.035,color=colors[arm])
    for metric,name,ylabel,title in (
        ('acceptance_length','acceptance_length','Accepted draft Tokens / verification round','Acceptance length (quality), separately from throughput'),
        ('acceptance_rate','acceptance_rate','Accepted / proposed draft Tokens','Acceptance rate (quality), separately from throughput'),
        ('tps','warm_tps','Whole-generate warm throughput (Tokens/s)','Actual warm whole-generate TPS; four matched arms, no ordinary-SD curve')):
        fig,ax=plt.subplots(figsize=(14,7.5))
        for i,arm in enumerate(ARM_IDS):points(ax,arm,metric,offset=(i-1.5)*.02)
        axis(ax,ylabel);ax.set_ylim(bottom=0)
        if metric=='acceptance_rate':ax.set_ylim(top=1)
        ax.legend(ncols=2,fontsize=12);save(fig,name,title)
    fig,ax=plt.subplots(figsize=(14,7.5))
    for key,color,style,label,offset in (
        ('primary_candidate_rescue_vs_candidate_retrieval','#b66c30','-','Primary: candidate RESCUE / candidate retrieval',.018),
        ('original_rescue_vs_original_retrieval','#777777','--','Remeasured control: original RESCUE / original retrieval',-.018)):
        rows=[(LENGTHS.index(int(n)),c['contrasts'][key]) for n,c in cells.items()]
        x=[i+offset for i,r in rows];ax.plot(x,[r['point_percent'] for i,r in rows],marker='o',color=color,linestyle=style,label=label)
        for value,(_,r) in zip(x,rows):
            lo,hi=r['ci95_percent'];ax.vlines(value,lo,hi,color=color);ax.hlines((lo,hi),value-.035,value+.035,color=color)
    ax.axhline(0,color='#555555',linewidth=1);ax.axhline(10,color='#9a9a9a',linewidth=1,linestyle=':')
    axis(ax,'Relative whole-generate warm TPS gain (%)');ax.legend(fontsize=12)
    save(fig,'relative_tps_gain','Primary fair comparison: both RESCUE and strong retrieval baseline receive the same new Target execution')
    fig,axes=plt.subplots(1,2,figsize=(15,7.5))
    for ax,method,contrast,color in zip(axes,('RESCUE','retrieval Top32/k4'),('rescue_execution_gain','retrieval_execution_gain'),('#b66c30','#4a785d')):
        rows=[(LENGTHS.index(int(n)),c['contrasts'][contrast]) for n,c in cells.items()]
        ax.plot([i for i,r in rows],[r['point_percent'] for i,r in rows],marker='o',color=color)
        for x,r in rows:
            lo,hi=r['ci95_percent'];ax.vlines(x,lo,hi,color=color);ax.hlines((lo,hi),x-.04,x+.04,color=color)
        ax.axhline(0,color='#555555',linewidth=1);axis(ax,'Candidate / original TPS gain (%)');ax.set_title(method)
    save(fig,'execution_gains','Execution-only gain within each frozen algorithm; not a cross-batch comparison')
    fig,axes=plt.subplots(1,2,figsize=(16,8))
    for i,arm in enumerate(ARM_IDS):
        rows=sorted((r for r in owners if r['arm_id']==arm),key=lambda r:r['length']);x=[LENGTHS.index(r['length'])+(i-1.5)*.17 for r in rows]
        axes[0].bar(x,[r['finite_service_tps'] for r in rows],width=.16,color=colors[arm],label=names[arm])
        axes[1].bar(x,[r['constructor_seconds']+r['cold_seconds']+r['close_seconds'] for r in rows],width=.16,color=colors[arm],label=names[arm])
    axis(axes[0],'Finite accounted owner throughput (Tokens/s)');axis(axes[1],'Constructor + cold prime + scope/close (s)')
    for ax in axes:ax.set_xlim(-.5,5.5)
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(.5,.87),ncols=2,fontsize=12)
    axes[0].set_ylim(bottom=0);axes[1].set_ylim(bottom=0)
    save(fig,'finite_owner_costs','Finite service includes constructor + cold +32warm +close; overhead panel excludes warm only for display',True,bottom=.26,top=.77)
    fig,axes=plt.subplots(2,2,figsize=(16,11))
    for ax,key,ylabel in zip(axes.flat,('warm_captures','warm_capture_host_seconds','graph_replays','graph_entries'),
        ('Warm late-capture count','Charged warm capture host time (s)','Cold + warm Graph replay count','Final persistent Graph entries (count)')):
        for i,arm in enumerate(ARM_IDS):
            rows=sorted((r for r in owners if r['arm_id']==arm),key=lambda r:r['length'])
            ax.plot([LENGTHS.index(r['length']) for r in rows],[r[key] for r in rows],marker='o',color=colors[arm],linestyle=styles[arm],label=names[arm])
        axis(ax,ylabel);maximum=max((r[key] for r in owners),default=0)
        ax.set_ylim(0,maximum*1.06 if maximum>0 else 1)
    axes[0,0].legend(fontsize=12);save(fig,'graph_mechanisms','Actual Graph ledgers and capture intervals; no capture costs subtracted and no GPU-busy inference',True,bottom=.22)
    fig,axes=plt.subplots(2,2,figsize=(16,12))
    maximum=max(1,max((v for r in owners for v in r['target_layer_calls']['candidate']),default=1))
    for ax,method in zip(axes[0],('rescue','retrieval')):
        matrix=np.full((6,28),np.nan)
        for row in owners:
            if row['arm_id']=='candidate_'+method:matrix[LENGTHS.index(row['length'])]=row['target_layer_calls']['candidate']
        image=ax.imshow(matrix,aspect='auto',vmin=0,vmax=maximum,cmap=plt.get_cmap('YlGnBu').with_extremes(bad='#eeeeee'))
        fig.colorbar(image,ax=ax,label='Candidate attention calls / Target layer')
        ax.set_title('Candidate '+method+' · cold +32warm');ax.set_yticks(range(6),[f'{n//1024}K / {n}' for n in LENGTHS]);ax.set_xticks((0,7,14,21,27));ax.set_xlabel('Target layer index (0–27)')
    for offset,method,color in ((-.18,'rescue','#b66c30'),(.18,'retrieval','#4a785d')):
        rows=[r for r in owners if r['arm_id']=='candidate_'+method]
        qcounts=[sum(v['calls'] for r in rows for v in r['scope_qn_distribution'] if v['action']=='candidate' and v['query_tokens']==q) for q in (2,3,4,5)]
        axes[1,0].bar(np.arange(4)+offset,qcounts,width=.34,color=color,label=method)
        boxes=[];positions=[]
        for row in sorted(rows,key=lambda r:r['length']):
            data=[]
            for item in row['scope_qn_distribution']:
                if item['action']=='candidate':data.extend([item['key_tokens']-row['length']]*(item['calls']//28))
            if data:boxes.append(data);positions.append(LENGTHS.index(row['length'])+offset)
        if boxes:
            artists=axes[1,1].boxplot(boxes,positions=positions,widths=.30,patch_artist=True,whis=(0,100),manage_ticks=False,showfliers=False)
            for box in artists['boxes']:box.set_facecolor(color);box.set_alpha(.6)
    axes[1,0].set_xticks(range(4),('2','3','4','5'));axes[1,0].set_xlabel('Candidate actual query length Q (Tokens)');axes[1,0].set_ylabel('Target attention calls, all28 layers');axes[1,0].legend(fontsize=12)
    axes[1,1].set_xticks(range(6),labels);axes[1,1].set_xlim(-.6,5.6);axes[1,1].set_xlabel('Actual input context (Tokens; K=1024)');axes[1,1].set_ylabel('Candidate key span N − input length (keys)')
    axes[1,1].set_title('Box: median/IQR; whiskers: observed span, NOT CI')
    save(fig,'target_scope_calls','Actual scoped cold +warm call metadata; N=input label+offset; original unwrapped owners have no scope records',True,bottom=.22)
    fig,axes=plt.subplots(2,2,figsize=(16,11))
    for ax,metric,ylabel in zip(axes.flat,('rounds_per_request','draft_forwards_per_request','whole_generate_seconds_per_round','prefill_seconds_per_request'),
        ('Verification rounds / request','Draft forwards / request','Whole-generate seconds / round','Recorded prefill seconds / request')):
        for arm in ARM_IDS:points(ax,arm,metric)
        axis(ax,ylabel);ax.set_ylim(bottom=0)
    axes[0,0].legend(fontsize=12)
    save(fig,'round_work','Round/work descriptors and logged clocks; seconds/round includes prefill/captures, not isolated Target verification cost')


def write_report(summary,root,curve_proof_root,model_proof_root,output):
    root,curve_proof_root,model_proof_root,output=map(lambda p:Path(p).absolute(),(root,curve_proof_root,model_proof_root,output))
    require(not output.exists(),'Fresh analysis output required; preserve previous outputs')
    historical_proof(curve_proof_root);model_proof(model_proof_root)
    output.mkdir(parents=True);shutil.copyfile(__file__,output/'analysis_entry.py')
    for source,names,subdir in ((curve_proof_root,PROOF_FILES,'curve'),(model_proof_root,tuple(MODEL_PINS),'model')):
        for name in names:
            target=output/'proof'/subdir/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(safe(source,name),target)
    figures(summary,output)
    (output/'summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2,allow_nan=False)+'\n')
    stats=summary.get('statistics') or dict(contexts={},missing_contexts=list(LENGTHS))
    lines=['# Matched four-arm FP32 split-KV development timing','',f"Status: `{summary['status']}`.",'',
        f"Saved counts: `{summary.get('counts',{})}`; raw evidence complete: `{summary['complete_raw_evidence']}`; "+
        f"preregistered scientific gate: `{summary['scientific_gate_pass']}`.",'',
        'Primary comparison: candidate RESCUE versus candidate retrieval Top32/fixed-k4 Graph. Both receive the same new Target execution. '+
        'The original RESCUE/original retrieval comparison is separately remeasured in this same batch, alongside each method’s candidate/original execution gain. '+
        'There is no ordinary-SD arm and no division by timings from the earlier context curve. Retrieval Top32/k4 is the frozen strong representative, not a claim that it is globally optimal or the original paper implementation.','',
        'Primary TPS is the sum of saved warm output Tokens divided by the sum of actual frozen-curve whole-generate elapsed seconds per arm/context. '+
        'Each private owner serves cold document271 then the same32 warm calls (16 reused development documents, two repeats). All true request-time Triton JIT, Graph captures, warmups and restoration stay charged. '+
        'The Target proxy’s metadata append is retained. The timer begins after outside-generate telemetry/seed/idle and ends before row normalization. '+
        'Shared model loading/preparation and service infrastructure are not included. This is not fully-prewarmed production or network-service TPS.','',
        'Acceptance length (accepted draft Tokens / verification rounds), acceptance rate (accepted / proposed) and TPS are separately plotted. '+
        'Bootstrap resamples whole paired documents, keeping both repeats and all four arms together, with one shared index across all six contexts: '+
        '10000 draws, seed20261031, pointwise95% percentile intervals. These are not simultaneous intervals. '+
        'One nonconcurrent owner block per context/arm, despite balanced fixed order, means document CIs do not capture owner-level noise, clock drift or block-order effects. '+
        'The16 documents were already used; this is not fresh confirmation, general stochastic distribution equivalence, production or SOTA evidence.','',
        'The preregistered gate requires every one of the six primary point gains to exceed10%, each corresponding pointwise95% CI lower bound to exceed0, '+
        'and all792 evaluated outputs to exactly equal the original sealed canonical Tokens. The CI is not required to have lower bound>10%; '+
        'that stronger claim is not made. Raw completion alone is not scientific success. Rendered inspection and independent audit remain required.','',
        'Mapper, policy and old cost table stay byte-frozen; Target tensors are FP32 with TF32 flags disabled, prefix and Draft BF16. '+
        'This is not an intermediate-state bitwise or compiled-instruction equivalence claim. The separately preserved operator V1/V2 failures and V3 analytic smoke are not relabeled. '+
        'The48-call model Token gate is an authenticated predecessor, not timing evidence. This792-call run checks its own saved outputs against the old canonical references.','']
    if summary.get('synthetic_program_test_only'):lines.insert(2,'SYNTHETIC CPU MOCK ONLY — no measured GPU/system result.\n')
    if not summary['complete_raw_evidence']:lines+=['PARTIAL: figures use only fully balanced, independently validated contexts with closed owners; missing contexts/calls are not promoted to success.','']
    lines+=['## Primary fair comparison','',
        '|Actual input Tokens|Candidate RESCUE TPS|Candidate retrieval TPS|Relative gain %|Paired95% CI %|Point>10|CI lower>0|Statistical gate|',
        '|---:|---:|---:|---:|---|---|---|---|']
    for n in LENGTHS:
        cell=stats['contexts'].get(str(n))
        if not cell:lines.append(f'|{n}|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|False|False|False|');continue
        c=cell['contrasts']['primary_candidate_rescue_vs_candidate_retrieval']
        lines.append(f"|{n}|{cell['arms']['candidate_rescue']['tps']['point']:.6f}|{cell['arms']['candidate_retrieval']['tps']['point']:.6f}|"+
            f"{c['point_percent']:.6f}|[{c['ci95_percent'][0]:.6f},{c['ci95_percent'][1]:.6f}]|{cell['primary_point_exceeds10_percent']}|{cell['primary_ci_lower_positive']}|{cell['threshold_statistics_pass']}|")
    lines+=['','## All four prespecified throughput contrasts','',
        '|Actual input Tokens|Contrast|Point gain %|Paired95% CI %|','|---:|---|---:|---|']
    for n,cell in stats['contexts'].items():
        for name,c in cell['contrasts'].items():lines.append(f"|{n}|{name}|{c['point_percent']:.6f}|[{c['ci95_percent'][0]:.6f},{c['ci95_percent'][1]:.6f}]|")
    lines+=['','## Acceptance descriptors (not interchangeable with TPS)','',
        '|Input Tokens|Arm|Accepted/round|Paired95% CI|Accepted/proposed|Paired95% CI|','|---:|---|---:|---|---:|---|']
    for n,cell in stats['contexts'].items():
        for arm,r in cell['arms'].items():
            a,b=r['acceptance_length'],r['acceptance_rate']
            lines.append(f"|{n}|{arm}|{a['point']:.9f}|{a['ci95']}|{b['point']:.9f}|{b['ci95']}|")
    lines+=['','## Descriptive interaction (not an additional success gate)','',
        'Multiplicative interaction is `[(candidate_rescue/original_rescue)/(candidate_retrieval/original_retrieval)-1]×100`. '+
        'The second descriptor is primary relative gain minus original relative gain, in percentage points. Neither adds a new hard gate.','',
        '|Input Tokens|Ratio-of-execution-gains interaction %|Primary minus original relative gain (pp)|','|---:|---:|---:|']
    for n,cell in stats['contexts'].items():
        d=cell['descriptive_interaction'];lines.append(f"|{n}|{d['ratio_of_execution_gains_percent']:.6f}|{d['primary_minus_original_percentage_points']:.6f}|")
    lines+=['','## Finite owner clocks and captures','',
        'Finite service TPS includes constructor/scope setup + cold prime +32warm +scope/owner close. Constructor/close are outside the primary warm clock but are not hidden: '+
        'they are reported here and included in finite service. One owner per arm/context provides no independently repeated-owner CI. Capture host intervals are recorded subintervals, '+
        'not GPU-busy time or a removable speed upper bound. Cold means first request in a fresh private owner, not independent process/model cold start: '+
        'the single process/cache can retain compiled kernels from preceding owners. Compilation is charged wherever it actually occurs, never reset or subtracted for a hypothetical deployment. '+
        'Seconds/round includes prefill and capture work, not isolated Target verification cost.','',
        '|Input Tokens|Arm|Constructor s|Cold s|Warm32 s|Scope/close s|Finite service TPS|Warm late captures|Charged warm capture s|',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['owner_diagnostics']:lines.append(f"|{r['length']}|{r['arm_id']}|{r['constructor_seconds']:.6f}|{r['cold_seconds']:.6f}|"+
        f"{r['warm_seconds']:.6f}|{r['close_seconds']:.6f}|{r['finite_service_tps']:.6f}|{r['warm_captures']}|{r['warm_capture_host_seconds']:.6f}|")
    lines+=['','## Mechanism and unchanged-work evidence','',
        'Original variants must match the old historical work at matching document/context/repeat; every variant must match its own repeated trajectory. '+
        'Candidate-versus-original differences are descriptive and allowed, provided canonical output Tokens agree. Scope call counts are actual returned proxy invocations, not kernel timings; '+
        'the original unwrapped variants deliberately have no proxy records. Full hidden-state equality is not measured by this timing run.','',
        '|Context|Method|Paired rows|Changed-work rows|Different semantic trajectories|','|---:|---|---:|---:|---:|']
    for n in LENGTHS:
        for method in ('rescue','retrieval'):
            subset=[r for r in summary['work_differences'] if (r['length'],r['method'])==(n,method)]
            lines.append(f"|{n}|{method}|{len(subset)}|{sum(bool(r['work_fields']) for r in subset)}|{sum(not r['trajectory_equal'] for r in subset)}|")
    lines+=['','## Figures','']
    for name in FIGURES:lines += [f'[{name} PNG]({name}.png) · [SVG]({name}.svg)',f'![{name}]({name}.png)','']
    lines+=['## Evidence issues','']+(['- '+s for s in summary['issues']] or ['None in saved-data checks.'])
    lines+=['','## Limitations and collection provenance','']+(['- '+s for s in summary['warnings']] or [])
    lines+=['A saved collection receipt is a transcription of tool observations checked against local SHA/size inventories. '+
        'This report does not re-observe the remote machine, certify a signed attestation, transfer weights, or claim a full remote compiled-cache inventory.','',
        '## Portable source and raw results','', '[Saved single-file source](analysis_entry.py). '+
        'Reproduce with `python analysis_entry.py /absolute/raw/root --curve-proof-root /absolute/analysis/proof/curve '+
        '--model-proof-root /absolute/analysis/proof/model --output /absolute/fresh/output`. Only stdlib+NumPy/Matplotlib are required; no runtime/repo imports.','']
    for name in ('results/PRIME.jsonl','results/RUNTIME.jsonl','results/OWNERS.jsonl','results/WORK_DIFFERENCES.jsonl','results/TERMINAL.json','COLLECTION_RECEIPT.json'):
        lines.append(f'[{name}]({os.path.relpath(root/name,output)})')
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    receipt=dict(status=summary['status'],scientific_gate_pass=summary['scientific_gate_pass'],goal_complete=False,render_inspection_required=True,
        file_sha256={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()})
    (output/'ANALYSIS_RECEIPT.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--curve-proof-root',type=Path,required=True);parser.add_argument('--model-proof-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    summary=analyze(args.root,args.curve_proof_root,args.model_proof_root)
    write_report(summary,args.root,args.curve_proof_root,args.model_proof_root,args.output)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('valid_warm_rows','valid_cold_rows','owner_diagnostics',
        'work_differences','used_source_sha256','statistics')},indent=2))


if __name__=='__main__':main()

