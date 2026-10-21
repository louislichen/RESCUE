"""Portable saved-data quality-only report for the frozen 48-call model gate.

No Torch/Triton/runtime/repository imports or new model calls. Instrumented
engine seconds are intentionally not interpreted as performance measurements.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil

LENGTHS=(2048,4096,6144,8192,12288,16384)
DOCS=(271,303)
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
 'launch.sh':'0251d94e739d6e66b4eba7be075364257dc5a063a661026ca96f2c24f4605815'}
FIXED_PINS={'PREREGISTRATION.json':'ecc6ccebca2d75d0feb9bf15c8d2611b58d136b1b58341dc110f1e7568349555',
    'SOURCE_SHA256SUMS':'6266c3efb911a14ba1fac0722b3434fd639d8f0a3729d674e7b7952ce1d3f9b4'}
PROOF_FILES=('FINAL_ARTIFACTS_MANIFEST.json','VERIFIED_COMPLETE.json',
    'terminal/results/references.json','terminal/results/runtime.jsonl','terminal/results/protocol.json',
    'terminal/results/documents.jsonl','terminal/results/COMPLETE.json','terminal/results/QUALITY_PASSED.json',
    'terminal/results/REUSE_RECEIPT.json','terminal/tools/run_context_curve.py')
FIGURES=('acceptance_quality','paired_quality_changes','candidate_layer_calls','owner_lifecycle','trajectory_diagnostics')
# Filled only from the actual terminal collection, never a planned success.
RUN_PINS={
    'results/QUALITY.jsonl':'b87f961918d536a6ace67067c003f06bd264c60fcfc65b87aa0d17ac71dac161',
    'results/TERMINAL.json':'ae9f5870b37097699c791ee1fa139bc9de7fe0f3b50ab70d1666fd2deb9c8993',
    'PIPELINE_EXIT.json':'8c8d01cfcc5b443958a22f0d93f792f4abaf7a0e6a028ae8b5d642252bd5b42d',
    'COLLECTION_RECEIPT.json':'6ebf68838362b4862d260d6ec5830d47e863c6aab1a5a23e8947d0791b489110'}


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
def expected():
    return [dict(length=n,source_index=d,mode=m,variant=v,execution_index=i) for i,(n,d,m,v) in enumerate(
        (n,d,m,v) for n in LENGTHS for d in DOCS for m in MODES for v in VARIANTS)]
def semantic_trace(row):return [{k:r[k] for k in TRACE} for r in row['trace']]
def difference(left,right):
    return dict(work_fields={k:dict(original=left.get(k),candidate=right.get(k)) for k in (*WORK,*OPTIONAL)
        if (k in left)!=(k in right) or left.get(k)!=right.get(k)},trajectory_equal=semantic_trace(left)==semantic_trace(right),
        observed_rounds_equal=left.get('observed_rounds')==right.get('observed_rounds'),
        queries_equal=left.get('queries')==right.get('queries'),native_t_equal=left.get('native_t')==right.get('native_t'),
        native_s_equal=left.get('native_s')==right.get('native_s'))
def digest(value):
    return type(value) is dict and issha(value.get('sha256')) and type(value.get('dtype')) is str and type(value.get('shape')) is list and all(type(x) is int and x>=0 for x in value['shape'])


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


def quality_row(row,reference):
    """Recompute saved greedy/Graph/observer accounting, never model arithmetic."""
    own=row['mode']==MODES[0];steps=row.get('trace',[])
    require(reference and row.get('generated')==reference and row.get('tokens')==len(reference) and row.get('output_agrees') is True,'Frozen canonical Token output differs')
    require(row.get('performance_eligible') is False and row.get('observer_enabled') is True and row.get('profile_enabled') is False and
        row.get('all_timing_fields_are_instrumented') is True and 'timeline' not in row and 'cost_samples' not in row,'Quality-only/instrumented scope differs')
    require(type(steps) is list and steps and row.get('blocks')==len(steps),'Missing actual round trace')
    for i,step in enumerate(steps):
        prefix,cap,proposed,accepted=[step.get(k) for k in ('prefix','cap','proposed','accepted')]
        require(all(type(x) is int for x in (prefix,cap,proposed,accepted)),'Noninteger round trace')
        remaining=128-(prefix-(row['length']-1))
        require(cap in ((min(2,remaining),min(4,remaining)) if own else (min(4,remaining),)) and
            1<=proposed<=cap<=4 and 0<=accepted<=proposed and step.get('round_index')==i and
            step.get('terminal') is (i==len(steps)-1) and prefix==(row['length']-1 if not i else steps[i-1]['prefix']+steps[i-1]['accepted']+1) and
            type(step.get('workset_tokens')) is int and 0<step['workset_tokens']<=(512 if own else 1024),'Causal cap/prefix/workset differs')
    require(row.get('accepted')==sum(s['accepted'] for s in steps) and row.get('proposed')==sum(s['proposed'] for s in steps) and
        near(row.get('acceptance_rate'),row['accepted']/row['proposed']),'Acceptance accounting differs')
    last=steps[-1];progress=last['prefix']-(row['length']-1)
    require(row['tokens'] in (progress+last['accepted'],progress+last['accepted']+1),'Terminal commit count differs')
    forwards=row['proposed']+sum(s['accepted']==s['proposed'] and not s['terminal'] for s in steps)
    require(row.get('draft_forwards')==forwards,'Proposal/catchup work differs')
    runtime=row.get('draft_runtime',{});delta=runtime.get('request',{});stats=row.get('runtime_stats',{});keys=stats.get('graph_keys',[])
    require(runtime.get('mode')==stats.get('mode')=='cuda_graph' and stats.get('max_graph_entries')==128 and
        0<len(keys)<=128 and len({json.dumps(k,sort_keys=True) for k in keys})==stats.get('entries')==stats.get('captures')==len(keys),'Private Graph inventory differs')
    require(all(finite(delta.get(k)) and delta[k]>=0 and near(delta[k],stats.get(k)) for k in RUNTIME) and
        delta['captures']>0 and delta['replays']==delta['shadow_updates']==forwards and delta['resident_steps']==0,'Fresh-owner Graph work differs')
    required=set()
    for step in steps:
        required.update((step['workset_tokens']+j,True,own and j<=2) for j in range(1,step['proposed']+1))
        if step['accepted']==step['proposed'] and not step['terminal']:required.add((step['workset_tokens']+step['proposed']+1,False,False))
    require(required<={tuple(key[:3]) for key in keys},'Graph shape keys do not cover actual proposal/catchup trajectory')
    signature=row.get('owner_signature',{})
    for role in ('native_t','native_s','working'):
        require(all(type(signature.get(role,{}).get(k)) is int and signature[role][k]>0 for k in ('identity','key_ptr','value_ptr','capacity')),'Missing private cache identity')
    require(signature.get('runtime',{}).get('bound_ptrs')==[signature[role][k] for role in ('native_s','working') for k in ('key_ptr','value_ptr')],'Graph not bound to owner KV')
    rounds=row.get('observed_rounds',[]);queries=row.get('queries',[]);committed=[];cursor=0;nonzero=False
    require(len(rounds)==len(steps) and len(queries)==forwards,'Missing every round/query observation')
    for role in ('native_t','native_s'):
        cache=row.get(role,{});lengths,layers=cache.get('lengths',[]),cache.get('layers',[])
        require(len(lengths)==len(layers)==28,'Missing full28-layer valid-span Native digests')
        for n,layer in zip(lengths,layers):
            require(type(n) is int and n>=row['length']-1 and all(digest(layer.get(k)) and len(layer[k]['shape'])==4 and layer[k]['shape'][2]==n for k in ('key','value')),'Native digest valid-span differs')
    for step,observed in zip(steps,rounds):
        selected=observed.get('selected',[]);draft=observed.get('draft',[]);checks=observed.get('checks',[])
        require(selected==sorted(set(selected)) and len(selected)==step['workset_tokens'] and
            all(type(p) is int and 0<=p<step['prefix'] for p in selected) and
            observed.get('progress')==len(committed) and observed.get('remaining')==128-len(committed) and
            all(observed.get(k)==step[k] for k in ('round_index','prefix','accepted')) and len(draft)==step['proposed'] and len(checks)==len(draft)+1,
            'Observed selection/round/output state differs')
        accepted=0
        for a,b in zip(draft,checks):
            if a!=b:break
            accepted+=1
        require(accepted==step['accepted'],'Saved contiguous greedy agreement differs')
        committed.extend(draft[:accepted])
        if len(committed)<row['tokens']:committed.append(checks[accepted])
        features=observed.get('budget_features')
        require(features is None or len(features)==12 and all(finite(x) for x in features),'Actual M3 feature shape differs')
        for evidence in observed.get('injection_evidence',[]):
            working,native,positions,count=[evidence.get(k) for k in ('working_value','native_value','positions','different_elements')]
            require(own and evidence.get('layer')==27 and positions and len(set(positions))==len(positions) and set(positions)<=set(selected) and
                digest(working) and digest(native) and working['shape']==native['shape'] and working['dtype']==native['dtype']=='torch.bfloat16' and
                len(working['shape'])==2 and working['shape'][0]==len(positions) and type(count) is int and 0<=count<=math.prod(working['shape']) and
                evidence.get('native_matches_memo') is True and (count>0)==(working['sha256']!=native['sha256']),'Genuine same-BF16 V injection evidence differs')
            nonzero|=count>0
        for depth in range(step['proposed']):
            q=queries[cursor];cursor+=1
            require(digest(q.get('query')) and q.get('logits') is True and q.get('summary_requested') is (own and depth<2),'Proposal query/summary differs')
        if step['accepted']==step['proposed'] and not step['terminal']:
            q=queries[cursor];cursor+=1
            require(digest(q.get('query')) and q.get('logits') is False and q.get('summary_requested') is False,'Catchup query/summary differs')
    require(committed==row['generated'] and cursor==len(queries),'Saved observations do not reconstruct exact output')
    require(row.get('nonzero_injection') is own and nonzero is own,'Frozen nonzero Mapper evidence differs')
    return dict(acceptance_length=row['accepted']/row['blocks'],acceptance_rate=row['accepted']/row['proposed'],
        tokens=row['tokens'],accepted=row['accepted'],blocks=row['blocks'],proposed=row['proposed'],draft_forwards=forwards)


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


def collection_check(root,warnings):
    path=safe(root,'COLLECTION_RECEIPT.json')
    if not path.is_file():
        warnings.append('No saved collection receipt; remote process absence/barriers are not independently established by this report.')
        return dict(present=False,signed_attestation=False,remote_reobserved=False)
    receipt=read(path);hashes=receipt.get('file_sha256',{});sizes=receipt.get('file_size_bytes',{})
    require(type(hashes) is dict and hashes and set(hashes)==set(sizes) and len(hashes)==receipt.get('collected_files') and
        all(issha(h) and sha(safe(root,n))==h and safe(root,n).stat().st_size==sizes[n] for n,h in hashes.items()),
        'Collection local-byte inventory mismatch')
    for key in ('remote_read_only_barrier_before','remote_read_only_barrier_after'):
        barrier=receipt.get(key,{})
        require(barrier.get('file_sha256')==hashes and barrier.get('file_size_bytes')==sizes,'Collection double barrier mismatch')
    require(receipt.get('scp_exit_codes')==[0,0] and receipt.get('local_matches_both_remote_inventories') is True and
        receipt.get('compiled_cache_transferred') is False and receipt.get('weights_transferred') is False and
        receipt.get('performance_measured') is False and receipt.get('goal_complete') is False,'Collection scope/SCP did not complete')
    process=receipt.get('process_observations',{})
    require(all(process.get(k) is True for k in ('passed','gpu_compute_idle','model_children_absent','model_process_absent','model_tmux_absent')) and
        process.get('model_terminal_sha256')==sha(safe(root,'results/TERMINAL.json')),'Recorded process-absence barrier differs')
    return dict(present=True,local_files_rehashed=len(hashes),double_barrier_matches_local_bytes=True,
        signed_attestation=False,remote_reobserved=False,provenance='MAIN_AGENT_TOOL_OBSERVATION_TRANSCRIPTION')


def analyze(root,proof_root):
    root=Path(root).absolute();issues=[];warnings=[]
    result=dict(status='PARTIAL_TARGET_TOKEN_GATE',planned_calls=48,quality_rows_saved=0,generation_returns_saved=0,
        owners_closed=0,valid_quality_rows=0,valid_rows=[],lifecycle=[],differences=[],scope_layers=[],
        metrics=[],paired_changes=[],issues=issues,warnings=warnings,tps='NOT_MEASURED',latency='NOT_MEASURED',
        timing_fields_instrumented_not_performance=True,confidence_intervals=None,documents=list(DOCS),
        contexts=list(LENGTHS),fresh_confirmation=False,distribution_equivalence_claim=False,goal_complete=False,
        goal_status='ACTIVE',model_token_scope_complete=False)
    try:
        for name,pin in {**SOURCES,**FIXED_PINS,**RUN_PINS}.items():require(sha(safe(root,name))==pin,'Pinned source/raw changed: '+name)
        sums={line.split('  ',1)[1]:line.split('  ',1)[0] for line in safe(root,'SOURCE_SHA256SUMS').read_text().splitlines()}
        require(sums=={n.removeprefix('capsule/') if n.startswith('capsule/') else '../launch.sh':h for n,h in SOURCES.items()},'Thirteen-source inventory differs')
        prior=historical_proof(proof_root);protocol=prior['protocol'];pre=read(safe(root,'PREREGISTRATION.json'))
        plan=read(safe(root,'results/PLAN.json'));terminal=read(safe(root,'results/TERMINAL.json'));exitrow=read(safe(root,'PIPELINE_EXIT.json'))
        result['synthetic_program_test_only']=pre.get('synthetic_program_test_only') is True
        for obj in (pre,plan):
            require(obj.get('lengths')==list(LENGTHS) and obj.get('source_indices')==list(DOCS) and obj.get('modes')==list(MODES) and
                obj.get('variants')==list(VARIANTS) and obj.get('max_new_tokens')==obj.get('graph_max_entries')==128 and
                obj.get('quality_observer') is True and obj.get('performance_measured') is False,'Frozen48-call scope changed')
        require(plan.get('planned_generation_calls')==plan.get('planned_private_owners')==48 and
            plan.get('candidate_intermediate_bitwise_required') is False and plan.get('candidate_trajectory_equal_required') is False and
            plan.get('original_historical_work_equal_required') is True and plan.get('target_dtype')=='float32' and plan.get('draft_dtype')=='bfloat16' and
            plan.get('tf32') is False,'Quality gate numerical/precision contract changed')
        require(pre.get('runner_source_sha256')==SOURCES['capsule/exp/round_overhead/fp32_splitkv_model_gate_v1.py'] and
            pre.get('operator_terminal_sha256')==OPERATOR_TERMINAL,'Predecessor/operator runner pin changed')
        require(terminal.get('runner_sha256')==SOURCES['capsule/exp/round_overhead/fp32_splitkv_model_gate_v1.py'] and
            terminal.get('performance_measured') is False and terminal.get('performance_gate')=='NOT_RUN' and terminal.get('goal_complete') is False,
            'Terminal source/no-performance contract differs')
        require(exitrow.get('terminal_sha256')==sha(safe(root,'results/TERMINAL.json')) and
            exitrow.get('performance_measured') is False and exitrow.get('goal_complete') is False,'Outer exit/terminal binding differs')
        reuse=read(safe(root,'results/REUSE_RECEIPT.json'))
        require(reuse.get('source_sha256')==protocol['source_sha256'] and reuse.get('assets_sha256')==protocol['assets_sha256'] and
            reuse.get('mapper_policy_cost_unchanged')==ASSETS and reuse.get('prior_manifest_sha256')==PRIOR_MANIFEST and
            reuse.get('prior_verified_sha256')==PRIOR_VERIFIED and reuse.get('operator_terminal_sha256')==OPERATOR_TERMINAL and
            reuse.get('no_new_ar') is True and reuse.get('no_new_corpus') is True and reuse.get('performance_eligible') is False,'Original source/assets/reuse chain differs')
        require(reuse.get('external_source_sha256')=={n.removeprefix('capsule/'):SOURCES[n] for n in SOURCES if
            n in ('capsule/exp/round_overhead/fp32_splitkv_candidate_v1.py','capsule/exp/round_overhead/fp32_splitkv_cuda_gate_v3.py',
            'capsule/tests/test_fp32_splitkv_cuda_gate_v3.py','capsule/exp/round_overhead/fp32_splitkv_target_adapter_v1.py','capsule/tests/test_fp32_splitkv_target_adapter_v1.py')},
            'External candidate/adapter source inventory differs')
        references=read(safe(root,'results/references.json'));inputs=read(safe(root,'results/INPUTS.json'))
        keys={f'{d}:{n}' for n in LENGTHS for d in DOCS}
        require(set(references)==set(inputs)==keys,'Exact12 input/reference matrix differs')
        for key in keys:
            d,n=map(int,key.split(':'));old=prior['documents'][d]
            ids=old['prefix_ids']+old['document_ids'][:n-30]+old['suffix_ids']
            require(len(inputs[key])==n and inputs[key]==ids and hashlib.sha256(json.dumps(ids).encode()).hexdigest()==protocol['input_sha256'][key] and
                references[key]==prior['references'][key],'Old canonical/input identity differs: '+key)
        runtime=read(safe(root,'results/RUNTIME.json'))
        require(runtime.get('maximum_live_owners')==1 and runtime.get('original_graph_classes') is True and
            runtime.get('performance_eligible') is False and runtime.get('quality_observer') is True and
            runtime.get('target_dtype')=='float32' and runtime.get('draft_dtype')=='bfloat16' and
            runtime.get('policy_metadata')==protocol['policy_metadata'] and runtime.get('configs')=={mode:protocol['configs'][mode] for mode in MODES} and
            runtime.get('source_bindings')==protocol['source_bindings'] and runtime.get('tree_source_bindings')==protocol['tree_source_bindings'] and
            runtime.get('flash_attention')==protocol['flash_attention'],'Frozen model/runtime preparation differs')
        require(all(runtime.get('environment',{}).get(k)==protocol['environment'][k] for k in
            ('gpu_name','gpu_uuid','torch','cuda','target_dtype','draft_dtype','prefix_dtype','tf32')),'Frozen GPU/runtime/precision environment differs')
        rows=jsonl(safe(root,'results/QUALITY.jsonl'));require(len(rows)<=48,'More than48 quality calls')
        returned=sorted(safe(root,'results').glob('GENERATION_RETURNED_*.json'))
        result.update(quality_rows_saved=len(rows),generation_returns_saved=len(returned),
            original_proof_sha256=prior['pins'],terminal=terminal,collection=collection_check(root,warnings))
        raw_ids=set();owner_ids=set();valid=[];rows_by_index={};returned_indices=[];owner_paths=[]
        for i,row in enumerate(rows):
            descriptor=expected()[i];life=dict(descriptor,quality_saved=True,returned_saved=False,canonical_exact=False,quality_verified=False,owner_closed=False)
            result['lifecycle'].append(life)
            try:
                require(all(type(row.get(k)) is type(v) and row[k]==v for k,v in descriptor.items()),'Quality call order/matrix differs')
                require(uid(row.get('owner_id')) and uid(row.get('request_id')) and row['owner_id'] not in owner_ids and row['request_id'] not in raw_ids,'Owner/request identity reused')
                owner_ids.add(row['owner_id']);raw_ids.add(row['request_id']);rows_by_index[i]=row
                life['canonical_exact']=row.get('generated')==references[f"{row['source_index']}:{row['length']}"]
                returned_path=safe(root,f'results/GENERATION_RETURNED_{i:03d}.json');early=read(returned_path);life['returned_saved']=True
                expected_row={k:v for k,v in row.items() if k not in ('target_attention_scope','all_timing_fields_are_instrumented')}
                require(early.get('performance_eligible') is False and early.get('row')==expected_row,'Returned-generation row not bound to QUALITY')
                pre_scope=early.get('target_scope_at_return',{});scope=row.get('target_attention_scope',{})
                require({k:v for k,v in pre_scope.items() if k!='state'}=={k:v for k,v in scope.items() if k!='state'},'Returned scope counters changed')
                scope_counts(pre_scope,row['variant'],False);counts=scope_counts(scope,row['variant'])
                metric=quality_row(row,references[f"{row['source_index']}:{row['length']}"])
                if row['variant']=='original':
                    historic=[r for r in prior['history'] if all(r.get(k)==row[k] for k in ('length','source_index','mode'))]
                    require(len(historic)==2 and {r['repeat'] for r in historic}=={0,1} and all(
                        r.get('generated')==row['generated'] and semantic_trace(r)==semantic_trace(row) and
                        all(k in r and k in row and r[k]==row[k] for k in WORK) and
                        all((k in r)==(k in row) and (k not in r or r[k]==row[k]) for k in OPTIONAL) for r in historic),
                        'Original variant historical work differs')
                life['quality_verified']=True;valid.append(dict(descriptor,**metric));result['scope_layers'].append(dict(descriptor,**counts))
                owner_path=safe(root,'results/OWNER_'+row['owner_id']+'.json');owner=read(owner_path)
                require(all(owner.get(k)==row[k] for k in ('owner_id','request_id','mode','variant','length','source_index')) and
                    owner.get('engine_closed') is True and owner.get('hook_restored') is True and owner.get('performance_eligible') is False and
                    owner.get('closed_runtime_stats',{}).get('entries')==0 and owner.get('final_runtime_stats')==row['runtime_stats'],
                    'Closed private owner/restore evidence differs')
                life['owner_closed']=True;owner_paths.append(owner_path)
            except (OSError,ValueError,TypeError,KeyError,IndexError,ZeroDivisionError) as error:issues.append(f'call{i:03d}: '+str(error))
        for path in returned:
            index=int(path.stem.rsplit('_',1)[1]);require(index not in returned_indices and 0<=index<48,'Invalid/duplicate generation-return index')
            returned_indices.append(index)
            if index not in rows_by_index:
                early=read(path);row=early.get('row',{});descriptor=expected()[index]
                result['lifecycle'].append(dict(descriptor,quality_saved=False,returned_saved=True,
                    canonical_exact=row.get('generated')==references[f"{descriptor['source_index']}:{descriptor['length']}"],quality_verified=False,owner_closed=False))
        require(returned_indices==list(range(len(returned))),'Generation-return ledger is not a completed prefix')
        for i in range(1,len(rows),2):
            left,right=rows[i-1:i+1]
            path=safe(root,f"results/DIFF_{right['length']}_{right['source_index']}_{right['mode']}.json")
            if path.is_file():
                computed=difference(left,right);require(read(path)==computed,'Saved work/trajectory DIFF differs')
                changed={}
                for a,b in zip(left.get('observed_rounds',[]),right.get('observed_rounds',[])):
                    for key in a.keys()|b.keys():
                        if a.get(key)!=b.get(key):changed[key]=changed.get(key,0)+1
                result['differences'].append(dict(length=right['length'],source_index=right['source_index'],mode=right['mode'],
                    observed_round_fields_changed=changed,**computed))
        result.update(valid_rows=valid,valid_quality_rows=len(valid),owners_closed=len(owner_paths),
            actual_owner_files=len(list(safe(root,'results').glob('OWNER_*.json'))),raw_quality_sha256=sha(safe(root,'results/QUALITY.jsonl')),
            used_source_sha256={**SOURCES,**FIXED_PINS,**RUN_PINS})
        counters=[terminal.get(k,0) for k in ('attempted_calls','returned_calls','completed_calls','validated_calls','private_owners_closed')]
        require(all(type(n) is int and 0<=n<=48 for n in counters) and counters==sorted(counters,reverse=True),'Terminal lifecycle counters are invalid')
        require(counters[1]==len(returned) and counters[2]==len(rows) and counters[4]==result['actual_owner_files'],
            'Terminal lifecycle counts differ from saved return/QUALITY/OWNER files')
        for n in LENGTHS:
            for mode in MODES:
                for variant in VARIANTS:
                    subset=[r for r in valid if (r['length'],r['mode'],r['variant'])==(n,mode,variant)]
                    if subset:result['metrics'].append(dict(length=n,mode=mode,variant=variant,documents=[r['source_index'] for r in subset],
                        accepted=sum(r['accepted'] for r in subset),blocks=sum(r['blocks'] for r in subset),proposed=sum(r['proposed'] for r in subset),
                        acceptance_length=sum(r['accepted'] for r in subset)/sum(r['blocks'] for r in subset),
                        acceptance_rate=sum(r['accepted'] for r in subset)/sum(r['proposed'] for r in subset)))
                for doc in DOCS:
                    pair={r['variant']:r for r in valid if (r['length'],r['mode'],r['source_index'])==(n,mode,doc)}
                    if set(pair)==set(VARIANTS):result['paired_changes'].append(dict(length=n,mode=mode,source_index=doc,
                        **{key:pair['candidate'][key]-pair['original'][key] for key in ('acceptance_length','acceptance_rate')}))
        complete=(not issues and len(rows)==len(returned)==len(valid)==len(owner_paths)==result['actual_owner_files']==48 and len(result['differences'])==24 and
            terminal.get('status')=='TARGET_TOKEN_GATE_PASSED' and terminal.get('passed') is True and terminal.get('actual_cuda') is True and
            all(terminal.get(k)==48 for k in ('completed_calls','attempted_calls','returned_calls','validated_calls','private_owners_closed')) and
            terminal.get('frozen_canonical_token_exact') is True and terminal.get('token_gate')=='FROZEN_CANONICAL_TOKEN_EXACT_PASSED' and
            all(terminal.get(k) is False for k in ('candidate_intermediate_bitwise_claim','candidate_trajectory_equal_claim','production_speed_claim',
                'distribution_equivalence_claim','fresh_confirmation')) and exitrow.get('exit_code')==0)
        result.update(status='COMPLETE_FROZEN_TARGET_TOKEN_GATE_ONLY' if complete else 'PARTIAL_TARGET_TOKEN_GATE',model_token_scope_complete=complete)
        result['mechanism_counts']=dict(candidate_target_calls=sum(sum(r['candidate']) for r in result['scope_layers']),
            scoped_original_fallback_calls=sum(sum(r['original']) for r in result['scope_layers']),
            paired_trajectories_equal=sum(r['trajectory_equal'] for r in result['differences']),
            pairs_with_work_counter_changes=sum(bool(r['work_fields']) for r in result['differences']),
            paired_target_native_digest_equal=sum(r['native_t_equal'] for r in result['differences']),
            paired_draft_native_digest_equal=sum(r['native_s_equal'] for r in result['differences']),
            paired_draft_query_digest_equal=sum(r['queries_equal'] for r in result['differences']))
    except (OSError,ValueError,TypeError,KeyError,IndexError,ZeroDivisionError) as error:issues.append(str(error))
    return result


def figures(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    plt.rcParams.update({'font.size':12,'axes.labelsize':12,'axes.titlesize':13,'xtick.labelsize':12,'ytick.labelsize':12})
    tag='COMPLETE: evaluated frozen-Token gate only' if summary['model_token_scope_complete'] else 'PARTIAL: completed quality subset only'
    if summary.get('synthetic_program_test_only'):tag='SYNTHETIC CPU SCHEMA FIXTURE — '+tag
    caption=(f"{summary['valid_quality_rows']}/48 valid quality rows; 2 reused historical documents (271,303), 1 request per variant / document / arm / length.\n"
        'Qwen2-7B /1.5B; Target FP32, Draft BF16, TF32 off; fixed Mapper / policy / cost; cap128 / EOS; both arms use Draft Graph.\n'
        'No stable general CI from 2 historical documents. TPS / latency NOT_MEASURED; all engine clocks are instrumented, not performance.\n'
        'Absent points / gray cells are NOT_RUN or unverified. No fresh-confirmation, distribution-equivalence or speed claim.')
    def save(fig,name,title,bottom=.20):
        fig.suptitle(tag+'\n'+title,fontsize=14);fig.text(.5,.012,caption,ha='center',fontsize=12)
        fig.tight_layout(rect=(0,bottom,1,.93));fig.savefig(output/(name+'.png'),dpi=170);fig.savefig(output/(name+'.svg'));plt.close(fig)
    labels=[f'{n//1024}K\n{n}' for n in LENGTHS];colors={'original':'#216882','candidate':'#b66c30'}
    fig,axes=plt.subplots(2,2,figsize=(15,10),squeeze=False)
    for i,mode in enumerate(MODES):
        for j,(metric,ylabel) in enumerate((('acceptance_length','Accepted draft Tokens / verification round'),('acceptance_rate','Accepted / proposed draft Tokens'))):
            ax=axes[i,j]
            for variant in VARIANTS:
                for doc in DOCS:
                    rows=sorted((r for r in summary['valid_rows'] if (r['mode'],r['variant'],r['source_index'])==(mode,variant,doc)),key=lambda r:r['length'])
                    ax.plot([LENGTHS.index(r['length'])+(-.025 if variant=='original' else .025) for r in rows],[r[metric] for r in rows],
                        marker='o' if doc==271 else 's',linestyle='-' if doc==271 else '--',color=colors[variant],
                        markerfacecolor='none' if variant=='original' else colors[variant],label=f'{variant}, doc{doc}',alpha=.85)
            ax.set_title(mode);ax.set_xticks(range(6),labels);ax.set_xlim(-.2,5.2);ax.set_ylim(bottom=0)
            if metric=='acceptance_rate':ax.set_ylim(0,1)
            ax.set_ylabel(ylabel);ax.set_xlabel('Actual input context length (Tokens; K=1024)');ax.grid(axis='y',alpha=.25)
            if i==j==0:ax.legend(ncols=2,fontsize=12)
            if not summary['valid_rows']:ax.text(.5,.5,'No valid completed quality rows',ha='center',transform=ax.transAxes)
    save(fig,'acceptance_quality','Acceptance length and rate are separate quantities; paired document points, not a throughput result')
    fig,axes=plt.subplots(2,2,figsize=(15,10),squeeze=False)
    for i,mode in enumerate(MODES):
        for j,(metric,ylabel) in enumerate((('acceptance_length','Δ accepted draft Tokens / round'),('acceptance_rate','Δ accepted / proposed draft Tokens'))):
            ax=axes[i,j]
            for n in LENGTHS:
                rows=[r for r in summary['paired_changes'] if (r['length'],r['mode'])==(n,mode)]
                if not rows:continue
                values=[r[metric] for r in rows];center=sum(values)/len(values);x=LENGTHS.index(n)
                ax.errorbar([x],[center],yerr=[[center-min(values)],[max(values)-center]],fmt='D',color='#555555',capsize=5,markersize=6)
                for row in rows:ax.scatter(x+(-.045 if row['source_index']==271 else .045),row[metric],
                    marker='o' if row['source_index']==271 else 's',s=55,color=colors['candidate'])
            ax.axhline(0,color='#555555',linewidth=1);ax.set_title(mode);ax.set_xticks(range(6),labels);ax.set_xlim(-.2,5.2)
            ax.set_ylabel(ylabel);ax.set_xlabel('Actual input context length (Tokens; K=1024)');ax.grid(axis='y',alpha=.25)
            if not summary['paired_changes']:ax.text(.5,.5,'No complete original / candidate quality pairs',ha='center',transform=ax.transAxes)
    save(fig,'paired_quality_changes','Candidate − original: diamond = mean paired change; whisker = observed document min–max, NOT confidence interval')
    fig,axes=plt.subplots(1,2,figsize=(16,9.5),squeeze=False)
    maximum=max(1,max((int(x) for row in summary['scope_layers'] if row['variant']=='candidate' for x in row['candidate']),default=1))
    for ax,mode in zip(axes[0],MODES):
        matrix=np.full((12,28),np.nan)
        for row in summary['scope_layers']:
            if row['mode']==mode and row['variant']=='candidate':matrix[LENGTHS.index(row['length'])*2+DOCS.index(row['source_index'])]=row['candidate']
        cmap=plt.get_cmap('YlGnBu').with_extremes(bad='#eeeeee')
        image=ax.imshow(matrix,aspect='auto',vmin=0,vmax=maximum,cmap=cmap);fig.colorbar(image,ax=ax,label='Actual candidate call count / Target layer')
        ax.set_title(mode);ax.set_xticks((0,7,14,21,27));ax.set_xlabel('Target layer index (0–27)')
        ax.set_yticks(range(12),[f'{n//1024}K doc{d}' for n in LENGTHS for d in DOCS])
    save(fig,'candidate_layer_calls','Only genuinely replaced Target attention calls; original scopes disabled, original fallback calls excluded')
    lifecycle={r['execution_index']:r for r in summary['lifecycle']}
    fields=('returned_saved','quality_saved','canonical_exact','quality_verified','owner_closed')
    fig,axes=plt.subplots(1,2,figsize=(16,13),squeeze=False)
    for ax,mode in zip(axes[0],MODES):
        planned=[r for r in expected() if r['mode']==mode];matrix=np.zeros((24,5))
        for i,descriptor in enumerate(planned):
            row=lifecycle.get(descriptor['execution_index'])
            for j,key in enumerate(fields):matrix[i,j]=0 if row is None else (1 if row.get(key) else 2)
        ax.imshow(matrix,cmap=ListedColormap(('#eeeeee','#bdd7c5','#d98e66')),vmin=0,vmax=2,aspect='auto')
        ax.set_title(mode);ax.set_yticks(range(24),[f"{r['length']//1024}K d{r['source_index']} {r['variant']}" for r in planned])
        ax.set_xticks(range(5),('Returned','QUALITY\nsaved','Canonical\nexact','Quality\nverified','Closed +\nrestored'));ax.xaxis.tick_top();ax.tick_params(length=0)
        for i in range(24):
            for j in range(5):ax.text(j,i,('—','YES','NO')[int(matrix[i,j])],ha='center',va='center',fontsize=12)
    save(fig,'owner_lifecycle','One fresh quality owner per call: returned ≠ validated ≠ closed; gray means no saved evidence',.15)
    flags=('trajectory_equal','observed_rounds_equal','queries_equal','native_t_equal','native_s_equal')
    fig,axes=plt.subplots(1,2,figsize=(16,9.5),squeeze=False)
    for ax,mode in zip(axes[0],MODES):
        matrix=np.zeros((12,5));lookup={(r['length'],r['source_index']):r for r in summary['differences'] if r['mode']==mode}
        for i,(n,d) in enumerate((n,d) for n in LENGTHS for d in DOCS):
            row=lookup.get((n,d))
            for j,key in enumerate(flags):matrix[i,j]=0 if row is None else (1 if row[key] else 2)
        ax.imshow(matrix,cmap=ListedColormap(('#eeeeee','#bdd7c5','#d7b78e')),vmin=0,vmax=2,aspect='auto')
        ax.set_title(mode);ax.set_yticks(range(12),[f'{n//1024}K doc{d}' for n in LENGTHS for d in DOCS])
        ax.set_xticks(range(5),('Semantic\ntrajectory','Observed\nrounds','Draft\nqueries','Target\nNative KV','Draft\nNative KV'));ax.xaxis.tick_top();ax.tick_params(length=0)
        for i in range(12):
            for j in range(5):ax.text(j,i,('—','SAME','DIFF')[int(matrix[i,j])],ha='center',va='center',fontsize=12)
    save(fig,'trajectory_diagnostics','Candidate versus original saved intermediate trajectories; DIFF is permitted if canonical Tokens are exact')


def write_report(summary,root,proof_root,output):
    root,proof_root,output=map(lambda p:Path(p).absolute(),(root,proof_root,output))
    require(not output.exists(),'Fresh analysis directory required')
    output.mkdir(parents=True);shutil.copyfile(__file__,output/'analysis_entry.py')
    historical_proof(proof_root)
    for name in PROOF_FILES:
        target=output/'proof'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(safe(proof_root,name),target)
    figures(summary,output)
    (output/'summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2,allow_nan=False)+'\n')
    lines=['# FP32 split-KV evaluated model Token gate — quality only','',f"Status: `{summary['status']}`.",'',
        'This is a quality gate using two reused historical documents (271 and303), six exact contexts and two frozen Graph arms. '+
        'It is not a speed experiment: TPS and latency are NOT_MEASURED. The expensive quality observers and synchronization instrument every request; '+
        'internal engine time fields are retained only in raw data and are not production measurements.','',
        f"Saved QUALITY rows: {summary['quality_rows_saved']}/48; raw generation returns: {summary['generation_returns_saved']}/48; "+
        f"independently validated quality rows: {summary['valid_quality_rows']}/48; owner close/restore records verified: {summary['owners_closed']}/48.",'',
        'A complete pass requires all48 fresh owners, unchanged original historical work on both old repeats, canonical Token equality, '+
        'actual candidate use on every Target layer, saved returned/QUALITY/DIFF correspondence and terminal/exit linkage. '+
        'Partial or returned-but-not-validated rows are never promoted to a completed Token gate.','',
        'Acceptance length is accepted draft Tokens divided by verification rounds; acceptance rate is accepted divided by proposed draft Tokens. '+
        'The figures show the two document points separately rather than a misleading general confidence interval. '+
        'Paired-change diamonds are the mean of the observed document changes; whiskers show their minimum and maximum, NOT a confidence interval. '+
        'There is one request per variant / arm / document / context.','',
        'Both arms are kept: RESCUE with its frozen nonzero V-only Mapper/policy/cost, and retrieval Top32 fixed-k4 Graph. '+
        'This candidate changes shared Target execution for both arms, not the Mapper or policy. Intermediate floats, queries, cache digests and '+
        'candidate trajectories are permitted to differ; every evaluated final Token must still equal the original sealed Target reference. '+
        'A pass on these two reused documents is not distribution-wide stochastic equivalence or fresh confirmation.','',
        'The historical proof is copied from the sealed798-call context curve and authenticated against its parent manifest and VERIFIED_COMPLETE hashes. '+
        'No new AR reference, corpus read, training, search or GPU execution occurs in this report. '+
        'The independent V1/V2 operator failures remain preserved; V3 was only a candidate-analytic smoke and is not itself a model or speed proof.','']
    if summary['model_token_scope_complete']:
        m=summary['mechanism_counts']
        lines+=['All 48 evaluated outputs match the old frozen canonical Tokens exactly. This completes only this evaluated quality scope; the >10% strong-baseline speed goal remains active.','',
            f"The candidate actually ran {m['candidate_target_calls']} scoped Target attention calls across all 28 layers; "+
            f"{m['scoped_original_fallback_calls']} scoped calls used the original fallback. These are call counts, not timing or kernel-speed measurements.",'',
            f"All {m['paired_trajectories_equal']}/24 candidate/original semantic round trajectories agree, with {m['pairs_with_work_counter_changes']} pairs having changed work counters. "+
            'Acceptance length and rate changes are zero in every evaluated pair. Target Native-KV byte digests differ in all 24 pairs, while Draft Native-KV and Draft-query digests agree in all 24. '+
            'Changed intermediate digests are not measured numerical distances; they neither invalidate the observed final-Token pass nor prove general equivalence.','']
    else:lines+=['PARTIAL: only the plotted independently verified subset is available. Missing/invalid lifecycle evidence is explicitly shown; no final quality or performance success is asserted.','']
    lines+=['## Pooled quality descriptors (not generalization estimates)','',
        '|Actual context Tokens|Arm|Variant|Observed docs|Accepted / round|Accepted / proposed|','|---:|---|---|---|---:|---:|']
    for r in summary['metrics']:lines.append(f"|{r['length']}|{r['mode']}|{r['variant']}|{r['documents']}|{r['acceptance_length']:.9f}|{r['acceptance_rate']:.9f}|")
    lines+=['','## Work and trajectory diagnostics','',
        '|Context|Document|Arm|Changed work counters|Semantic trajectory|Observed rounds|Draft queries|Target Native KV|Draft Native KV|Observed field changes|',
        '|---:|---:|---|---|---|---|---|---|---|---|']
    for r in summary['differences']:
        state=lambda key:'SAME' if r[key] else 'DIFF'
        fields=', '.join(f'{k}: {v} rounds' for k,v in r['observed_round_fields_changed'].items()) or 'None'
        lines.append(f"|{r['length']}|{r['source_index']}|{r['mode']}|{', '.join(r['work_fields']) or 'None'}|"+'|'.join(state(k) for k in ('trajectory_equal','observed_rounds_equal','queries_equal','native_t_equal','native_s_equal'))+f'|{fields}|')
    lines+=['','## Figures','']
    for name in FIGURES:lines += [f'[{name} PNG]({name}.png) · [SVG]({name}.svg)',f'![{name}]({name}.png)','']
    lines+=['## Issues','']+(['- '+x for x in summary['issues']] or ['None in saved-data checks.'])
    lines+=['','## Limitations and collection provenance','']+(['- '+x for x in summary['warnings']] or [])
    lines+=['Collection receipt, when present, is a main-agent transcription of tool observations and checked against local bytes. '+
        'This report does not re-observe the remote machine or claim signed attestation, transferred model weights, or complete remote compiled-cache inventory.','',
        '## Raw and portable analysis','', '[Saved single-file analysis and figure source](analysis_entry.py). '+
        'Reproduce using `python analysis_entry.py /absolute/raw/root --proof-root /absolute/analysis_v1/proof --output /absolute/fresh/output`. '+
        'Only stdlib plus NumPy/Matplotlib are required; no repository/runtime imports.','',
        f'[Raw QUALITY]({os.path.relpath(root/"results/QUALITY.jsonl",output)}) · '+
        f'[Terminal]({os.path.relpath(root/"results/TERMINAL.json",output)}) · '+
        f'[Collection receipt]({os.path.relpath(root/"COLLECTION_RECEIPT.json",output)}) · [Copied historical proof](proof/)','']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    receipt=dict(status=summary['status'],model_token_scope_complete=summary['model_token_scope_complete'],performance_measured=False,goal_complete=False,
        render_inspection_required=True,file_sha256={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()})
    (output/'ANALYSIS_RECEIPT.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--proof-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();summary=analyze(args.root,args.proof_root);write_report(summary,args.root,args.proof_root,args.output)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('valid_rows','lifecycle','differences','scope_layers','used_source_sha256')},indent=2))


if __name__=='__main__':main()
