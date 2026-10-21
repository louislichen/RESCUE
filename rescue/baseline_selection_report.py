"""Saved-data audit of the finite538-request Qwen2 baseline menu.

No models or remote tools are used. Selection uses the exact pooled ratio of
six warm requests, not the mean of per-request or per-document throughput.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

DOCS=(271,272,288)
LENGTHS=(4096,8192)
BACKENDS=('eager','resident','graph')
SEED=20261023
EXPECTED=dict(reference_requests=6,prime_requests=76,timing_requests=456,owners=76)
DATA_SHA='85b2eb1a81d1617b2d158c069c420c452459a8bdb90da128f257220cc27ded09'
TRACE_FIELDS=('round_index','prefix','cap','proposed','accepted','terminal','workset_tokens')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def positive(v):return type(v) in (int,float) and math.isfinite(v) and v>0
def nonnegative(v):return type(v) in (int,float) and math.isfinite(v) and v>=0
def near(a,b):return nonnegative(a) and nonnegative(b) and math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-8)


def menu():
    result=[]
    for top in (16,32):
        for k in (1,2,4,8,12):
            for backend in BACKENDS:
                result.append(dict(mode=f'retrieval{top}_k{k}_{backend}',family='retrieval',top_chunks=top,fixed_k=k,backend='cuda_graph' if backend=='graph' else backend))
    for top in (16,32):
        for nodes in (8,32,50,64):
            result.append(dict(mode=f'tree{top}_n{nodes}',family='full_tree',top_chunks=top,nodes=nodes,max_depth=10,threshold=.7,backend='upstream_flash_hybrid'))
    return result


def registered_schedule():
    owners=[];warm=[]
    for length in LENGTHS:
        rng=random.Random(SEED+length);modes=[a['mode'] for a in menu()];rng.shuffle(modes)
        for mode in modes:
            owners.append(dict(mode=mode,length=length))
            for repeat in (0,1):
                docs=list(DOCS);rng.shuffle(docs)
                warm.extend(dict(mode=mode,length=length,source_index=d,repeat=repeat) for d in docs)
    return owners,warm


def pool_document(rows):
    if len(rows)!=2 or {r.get('repeat') for r in rows}!={0,1} or len({(r.get('mode'),r.get('length'),r.get('source_index')) for r in rows})!=1:
        raise ValueError('Both repeats of one document/context/arm required; no imputation')
    tokens=sum(r['tokens'] for r in rows);seconds=sum(r['outer_seconds'] for r in rows)
    if not positive(tokens) or not positive(seconds):raise ValueError('Invalid token/time sample')
    return dict(tokens=tokens,seconds=seconds,tps=tokens/seconds,
        accepted=sum(r['accepted'] for r in rows),blocks=sum(r['blocks'] for r in rows),
        proposed=None if any(r.get('proposed') is None for r in rows) else sum(r['proposed'] for r in rows),
        captures=sum(r.get('draft_runtime',{}).get('request',{}).get('captures',0) for r in rows),
        capture_seconds=sum(r.get('draft_runtime',{}).get('request',{}).get('capture_host_seconds',0.) for r in rows))


def pooled_bootstrap(cells, *, other=None, seed=SEED, draws=10000):
    """Resample whole document token/time pairs, then recompute the ratio."""
    import numpy as np
    if draws!=10000:raise ValueError('Frozen10000 document bootstrap draws required')
    if not cells:return dict(n=0,point=None,ci95=None)
    if any(not positive(c.get('tokens')) or not positive(c.get('seconds')) for c in cells):raise ValueError('Invalid document token/time pair')
    tokens=np.array([c['tokens'] for c in cells],float);seconds=np.array([c['seconds'] for c in cells],float)
    point=float(tokens.sum()/seconds.sum())
    if other is not None:
        if len(other)!=len(cells) or any(not positive(c.get('tokens')) or not positive(c.get('seconds')) for c in other):raise ValueError('Paired documents required')
        ot=np.array([c['tokens'] for c in other],float);os=np.array([c['seconds'] for c in other],float)
        point=100*(point/(ot.sum()/os.sum())-1)
    result=dict(n=len(cells),point=point,ci95=None)
    if len(cells)>1:
        idx=np.random.default_rng(seed).integers(0,len(cells),size=(draws,len(cells)))
        ratio=tokens[idx].sum(1)/seconds[idx].sum(1)
        if other is not None:ratio=100*(ratio/(ot[idx].sum(1)/os[idx].sum(1))-1)
        result['ci95']=[float(v) for v in np.percentile(ratio,[2.5,97.5])]
    return result


def quality_interval(cells, denominator):
    """Whole-document ratio CI, allowing a genuine zero acceptance count."""
    import numpy as np
    if not cells:return None
    numerator=np.array([c['accepted'] for c in cells],float);den=np.array([c[denominator] for c in cells],float)
    if len(cells)<2:return None
    idx=np.random.default_rng(SEED).integers(0,len(cells),size=(10000,len(cells)))
    values=numerator[idx].sum(1)/den[idx].sum(1)
    return [float(v) for v in np.percentile(values,[2.5,97.5])]


def select_winners(metrics, arms):
    """Exact primary score, exact equality ties broken lexicographically."""
    selected={};family={};service={}
    for length in LENGTHS:
        values=metrics.get(str(length),{});selected[str(length)]={};family[str(length)]={};service[str(length)]={}
        for group in ('retrieval/eager','retrieval/resident','retrieval/graph','full_tree'):
            eligible=[a['mode'] for a in arms if (a['family']=='full_tree' if group=='full_tree' else a['family']=='retrieval' and a['backend']==('cuda_graph' if group.endswith('/graph') else group.split('/')[1])) and a['mode'] in values and values[a['mode']].get('documents')==3]
            if eligible:selected[str(length)][group]=min(eligible,key=lambda m:(-values[m]['pooled_tps'],m))
        for group in ('retrieval','full_tree'):
            eligible=[a['mode'] for a in arms if a['family']==group and a['mode'] in values and values[a['mode']].get('documents')==3]
            if eligible:
                family[str(length)][group]=min(eligible,key=lambda m:(-values[m]['pooled_tps'],m))
                service_eligible=[m for m in eligible if positive(values[m].get('accounted_tps'))]
                if service_eligible:service[str(length)][group]=min(service_eligible,key=lambda m:(-values[m]['accounted_tps'],m))
    return dict(selected=selected,overall_family=family,descriptive_service=service)


def linear_signature(row):return [[t.get(k) for k in TRACE_FIELDS] for t in row.get('trace',[])]


def row_errors(row, reference, arm):
    errors=[]
    if not isinstance(reference,list) or not reference or row.get('generated')!=reference or row.get('tokens')!=len(reference) or not 0<len(reference)<=128 or row.get('output_agrees') is not True:
        errors.append('Canonical Target output/token mismatch')
    if not positive(row.get('outer_seconds')) or not near(row.get('outer_tps'),row.get('tokens',0)/row['outer_seconds'] if positive(row.get('outer_seconds')) else -1) or row.get('performance_eligible') is not True or 'timeline' in row or 'cost_samples' in row:
        errors.append('Invalid/profile-contaminated full-generate outer timing')
    if positive(row.get('total_seconds')) and row.get('outer_seconds',0)+1e-6<row['total_seconds']:errors.append('Outer clock excludes part of generation')
    if arm['family']=='full_tree':
        upstream=row.get('upstream',{});accepted=upstream.get('accept_length_list',[])
        raw=row.get('raw_total_generated');overshoot=row.get('generation_overshoot')
        scored=row.get('scored_generated_before_eos',row.get('generated'))
        if (not isinstance(accepted,list) or not accepted or not all(type(v) is int and v>=0 for v in accepted) or
            row.get('accepted')!=sum(accepted) or row.get('blocks')!=len(accepted) or row.get('proposed') is not None or
            type(raw) is not int or raw!=sum(accepted)+len(accepted) or type(overshoot) is not int or
            not isinstance(scored,list) or raw-len(scored)!=overshoot or overshoot<0 or
            upstream.get('raw_total_generated')!=raw or upstream.get('generation_overshoot')!=overshoot or
            upstream.get('output_token_ids')!=scored or upstream.get('total_generated')!=len(scored) or
            scored[:len(reference or [])]!=reference or row.get('raw_tail_ids_available') is not False):
            errors.append('Tree raw acceptance/scored prefix/overshoot accounting differs')
        if row.get('acceptance_rate') is not None:errors.append('Invented full-tree proposal acceptance rate')
        if row.get('wrapper_generated')!=row.get('generated') or row.get('wrapper_tokens')!=row.get('tokens'):
            errors.append('Tree wrapper output was altered by client-prefix normalization')
        expected_policy=dict(nodes=arm['nodes'],top_chunks=arm['top_chunks'],max_depth=10,threshold=.7,
            chunk_size=32,refresh_every=4,flash_prefill=True,hybrid_tree=True,prefill_precision='bfloat16',mapper=False,controller=False)
        expected_flags=dict(target_use_flash_prefill=True,target_use_hybrid_tree_attn=True,draft_use_flash_prefill=True,
            use_retrieval_cache=True,retrieve_top_k=arm['top_chunks'],retrieval_chunk_size=32,retrieve_every_n_steps=4)
        if row.get('tree_policy')!=expected_policy or row.get('observed_full_config')!=expected_flags:
            errors.append('Actual full-tree Flash/hybrid/retrieval configuration differs')
    else:
        trace=row.get('trace',[])
        if not isinstance(trace,list) or not trace or row.get('blocks')!=len(trace):return errors+['Missing linear trace']
        if row.get('accepted')!=sum(t.get('accepted',0) for t in trace) or row.get('proposed')!=sum(t.get('proposed',0) for t in trace):errors.append('Linear acceptance/proposal totals differ')
        previous=None
        for i,t in enumerate(trace):
            prefix,cap,proposed,accepted=(t.get(k) for k in ('prefix','cap','proposed','accepted'))
            if not all(type(v) is int for v in (prefix,cap,proposed,accepted)):
                errors.append('Invalid linear integer trace');continue
            remaining=128-(prefix-(row.get('length',0)-1))
            if (cap!=min(arm['fixed_k'],remaining) or not 1<=proposed<=cap<=12 or not 0<=accepted<=proposed or
                t.get('round_index')!=i or t.get('terminal') is not (i==len(trace)-1) or
                (i==0 and prefix!=row.get('length',0)-1) or
                (previous is not None and prefix!=previous['prefix']+previous['accepted']+1)):
                errors.append('Fixed-k/EOS/committed-prefix trajectory differs')
            previous=t
        if not near(row.get('acceptance_rate'),row.get('accepted',0)/max(1,row.get('proposed',0))):errors.append('Linear acceptance rate differs')
        if row.get('projected_positions')!=0 or row.get('draft_shadow_probes')!=0 or row.get('execution_backend')!=arm['backend'] or row.get('baseline_algorithm')!='qwen_chunk_retrieval':
            errors.append('Linear baseline performed extra Mapper/probe or different execution')
        if any(not 0<t.get('workset_tokens',0)<=arm['top_chunks']*32 for t in trace):errors.append('Actual retrieval working set differs')
        if arm['backend']!='eager':
            delta=row.get('draft_runtime',{}).get('request',{})
            if arm['backend']=='cuda_graph' and (delta.get('replays')!=row.get('draft_forwards') or not positive(delta.get('replays'))):errors.append('Graph replay/work mismatch')
            if arm['backend']=='resident' and (delta.get('resident_steps')!=row.get('draft_forwards') or delta.get('captures')!=0):errors.append('Resident work/capture mismatch')
            if not nonnegative(delta.get('capture_host_seconds')) or type(delta.get('captures')) is not int or delta['captures']<0:errors.append('Invalid graph capture ledger')
            if row.get('draft_runtime',{}).get('retain_query') is not False or row.get('draft_runtime',{}).get('max_graph_entries')!=128:errors.append('Baseline retained unused Query or changed graph limit')
    return errors


def _analyze(directory):
    """Locked runner schema integration follows before any COMPLETE output."""
    directory=Path(directory);problems=[]
    def read(name,default):
        try:
            value=json.loads((directory/name).read_text())
            if not isinstance(value,type(default)):raise ValueError('Wrong JSON container')
            return value
        except (OSError,ValueError):problems.append('Missing/malformed '+name);return default
    def rows(name):
        result=[]
        try:lines=(directory/name).read_text().splitlines()
        except OSError:problems.append('Missing '+name);return result
        for i,line in enumerate(lines,1):
            if not line.strip():continue
            try:
                value=json.loads(line)
                if not isinstance(value,dict):raise ValueError('Not an object')
                result.append(value)
            except ValueError:problems.append(f'Malformed {name}:{i}')
        return result
    protocol=read('protocol.json',{});canonical=rows('reference.jsonl');prime=rows('prime.jsonl');warm=rows('runtime.jsonl');owners=rows('owners.jsonl')
    arms=protocol.get('arms',[]);expected={a['mode']:a for a in menu()}
    valid_arms=isinstance(arms,list) and len(arms)==38 and all(isinstance(a,dict) and a.get('mode') in expected for a in arms)
    if not valid_arms or len({a.get('mode') for a in arms if isinstance(a,dict)})!=38:
        problems.append('Frozen38-arm menu incomplete or duplicated');arms=menu()
    amap={a['mode']:a for a in arms}
    for mode,arm in expected.items():
        if any(amap.get(mode,{}).get(k)!=v for k,v in arm.items()):problems.append('Unregistered arm descriptor: '+mode)
    expected_owner_plan,expected_warm_plan=registered_schedule()
    if protocol.get('owner_order')!=expected_owner_plan or protocol.get('warm_order')!=expected_warm_plan:
        problems.append('Declared request order differs from frozen seeded schedule')
    if (protocol.get('source_indices')!=list(DOCS) or protocol.get('lengths')!=list(LENGTHS) or
        protocol.get('repeats')!=2 or protocol.get('max_new_tokens')!=128 or
        protocol.get('expected_counts')!=EXPECTED or protocol.get('seed')!=SEED):problems.append('Frozen538-request data/context/repeat plan differs')
    if protocol.get('no_training') is not True or protocol.get('data_file_sha256')!=DATA_SHA:
        problems.append('Original train-tune cohort/no-training declaration differs')
    if (protocol.get('original_corpus_roles')!={str(d):'train' for d in DOCS} or protocol.get('mapper_executed') is not False or
        protocol.get('controller_executed') is not False or protocol.get('observer') is not False or protocol.get('profile') is not False or
        protocol.get('graph_max_entries')!=128 or protocol.get('max_live_owners')!=1 or
        protocol.get('retrieval_retain_query') is not False or protocol.get('template_prefill_precision')!='bfloat16'):
        problems.append('Baseline-only/lean single-owner/BF16 shared prefill contract differs')
    for mode,arm in expected.items():
        config=protocol.get('configs',{}).get(mode,{})
        fixed=dict(length_policy='fixed',fixed_k=arm.get('fixed_k'),max_draft=12,max_new_tokens=128,
            refresh_every=4,inject_tokens=0,injection_strength=0.,injection_layers=[],prefill_precision='bfloat16',
            schedule='serial',profile=False,working_mode='packed',optimized_execution=True,deferred_probes=True,
            attention_backend='sdpa',draft_execution=arm['backend'],coverage_execution='eager') if arm['family']=='retrieval' else dict(
            nodes=arm['nodes'],top_chunks=arm['top_chunks'],max_depth=10,threshold=.7,chunk_size=32,refresh_every=4,
            flash_prefill=True,hybrid_tree=True,prefill_precision='bfloat16',mapper=False,controller=False)
        if any(config.get(k)!=v for k,v in fixed.items()):problems.append('Configured algorithm differs: '+mode)
    environment=protocol.get('environment',{})
    if ('H100' not in environment.get('gpu_name','') or environment.get('total_memory',0)<80_000_000_000 or
        environment.get('target_dtype')!='torch.float32' or environment.get('draft_dtype')!='torch.bfloat16' or
        environment.get('prefix_dtype')!='torch.bfloat16' or environment.get('tf32') is not False):problems.append('H10080GB/FP32Target/BF16Draft+prefix/TF32off environment differs')
    inventory=protocol.get('source_sha256',{});source=(directory/'source').resolve()
    if not inventory:problems.append('Missing immutable source inventory')
    for name,digest in inventory.items():
        path=(source/name).resolve()
        if not path.is_relative_to(source) or not path.is_file() or sha(path)!=digest:problems.append('Source hash mismatch: '+name)
    if not protocol.get('source_bindings') or not protocol.get('tree_source_bindings'):problems.append('Actual import provenance missing')
    for name,value in protocol.get('source_bindings',{}).items():
        relative=name.replace('.','/')+'.py'
        if not str(value.get('path','')).endswith('/'+relative) or inventory.get(relative)!=value.get('sha256'):
            problems.append('Actual module binding differs: '+name)
    for name,value in protocol.get('tree_source_bindings',{}).items():
        if 'sha256' in value and inventory.get(value.get('path'))!=value['sha256']:
            problems.append('Archived tree module binding differs: '+name)
    flash=protocol.get('flash_attention',{})
    if flash.get('version')!='2.8.3' or not str(flash.get('files',{}).get('flash_attn_2_cuda',{}).get('path','')).endswith('.so'):
        problems.append('Actual official Flash2.8.3 binary receipt missing')
    manifest=directory.parent/'EVIDENCE_MANIFEST.json'
    if manifest.is_file():
        try:
            values=json.loads(manifest.read_text())['file_sha256'];root=directory.parent.resolve()
            for name,digest in values.items():
                path=(root/name).resolve()
                if not path.is_relative_to(root) or not path.is_file() or sha(path)!=digest:problems.append('Outer evidence hash mismatch: '+name)
        except (OSError,KeyError,ValueError,TypeError):problems.append('Malformed outer evidence manifest')
    metadata_valid=not problems
    refs=read('references.json',{});reference_map={};bad_refs=set()
    for row in canonical:
        key=row.get('source_index'),row.get('length');values=row.get('generated')
        if key not in {(d,n) for d in DOCS for n in LENGTHS} or key in reference_map or not isinstance(values,list) or not 0<len(values)<=128 or not all(type(v) is int and v>=0 for v in values) or refs.get(f'{key[0]}:{key[1]}')!=values:
            problems.append('Canonical AR missing/duplicate/inconsistent');bad_refs.add(key)
        reference_map[key]=values
    if len(reference_map)!=6:problems.append('Six independent AR cells incomplete')
    all_requests=canonical+prime+warm;ids=[r.get('request_id') for r in all_requests];indices=[r.get('execution_index') for r in all_requests]
    if any(not isinstance(i,str) or not i for i in ids) or len(set(ids))!=len(ids):problems.append('Logical request UUID missing/reused');metadata_valid=False
    index_valid=all(type(i) is int and 0<=i<538 for i in indices) and len(set(indices))==len(indices)
    if not index_valid or sorted(indices)!=list(range(538)):
        problems.append('538-request execution ledger incomplete or duplicated')
        if not index_valid:metadata_valid=False
    if index_valid and canonical and any(r['execution_index']>=6 for r in canonical):problems.append('Canonical AR did not precede all timing');metadata_valid=False
    pmap={};wmap={};bad_owners=set();bad_warm=set();signatures={}
    for phase,values in (('cold_prime',prime),('warm',warm)):
        for row in values:
            mode=row.get('mode');length=row.get('length');doc=row.get('source_index');ownerkey=mode,length
            arm=amap.get(mode);errors=[]
            if arm is None:
                problems.append('Unknown timed arm: '+str(mode));continue
            errors+=row_errors(row,reference_map.get((doc,length)),arm)
            if row.get('arm')!=arm or row.get('phase')!=phase or length not in LENGTHS or (doc,length) in bad_refs:
                errors.append('Arm/phase/context/canonical identity differs')
            if any(row.get(k) is not False for k in ('observer_enabled','profile_enabled','mapper_executed','controller_executed')):
                errors.append('Primary request enabled Mapper/controller/observer/profile')
            if phase=='cold_prime':
                if doc!=271 or ownerkey in pmap:errors.append('Cold prime not uniquely doc271')
                if arm['family']=='retrieval' and arm['backend']=='cuda_graph':
                    delta=row.get('draft_runtime',{}).get('request',{});stats=row.get('runtime_stats',{})
                    if not positive(delta.get('captures')) or any(not near(stats.get(k),delta.get(k)) for k in ('captures','replays','capture_host_seconds')):errors.append('Fresh graph owner borrowed a prior capture')
                pmap[ownerkey]=row
            else:
                key=mode,length,doc,row.get('repeat')
                if doc not in DOCS or row.get('repeat') not in (0,1) or key in wmap:errors.append('Warm matrix missing/duplicate registered key')
                wmap[key]=row
            # Runtime keys and timers may differ across backends; logical
            # operations must not. Tree equality is within each actual policy.
            logical=(linear_signature(row),[row.get(k) for k in ('accepted','proposed','blocks','draft_forwards','selector_refreshes','working_gather_layers','working_full_gather_layers','working_gathered_position_layers','working_incremental_binds','target_probe_rounds','draft_shadow_probes','projected_positions','rpc_calls')]) if arm['family']=='retrieval' else (row.get('upstream',{}).get('accept_length_list'),row.get('raw_total_generated'),row.get('generation_overshoot'),row.get('scored_generated_before_eos'))
            strategy=(arm['family'],arm['top_chunks'],arm.get('fixed_k',arm.get('nodes')),doc,length)
            if strategy in signatures and signatures[strategy]!=logical:errors.append('Same strategy repeat/backend logical trajectory differs')
            signatures[strategy]=logical
            if errors:
                problems.extend(f'{phase} {mode}/{doc}/{length}: '+e for e in errors)
                if phase=='cold_prime':bad_owners.add(ownerkey)
                else:bad_warm.add((mode,length,doc,row.get('repeat')))
    expected_owners={(m,n) for m in expected for n in LENGTHS}
    expected_warm={(m,n,d,r) for m,n in expected_owners for d in DOCS for r in (0,1)}
    if set(pmap)!=expected_owners:problems.append('76 cold-prime cells incomplete')
    if set(wmap)!=expected_warm:problems.append('456 warm cells incomplete')
    actual_warm=[{k:r.get(k) for k in ('mode','length','source_index','repeat')} for r in warm]
    if protocol.get('warm_order')!=actual_warm:problems.append('Warm order differs from frozen plan')
    actual_owner=[{k:r.get(k) for k in ('mode','length')} for r in prime]
    if protocol.get('owner_order')!=actual_owner:problems.append('Private owner block order differs from frozen plan')
    if index_valid:
        for i,row in enumerate(prime):
            group=[r for r in warm if r.get('owner_id')==row.get('owner_id')]
            if row['execution_index']!=6+7*i or [r['execution_index'] for r in group]!=list(range(row['execution_index']+1,row['execution_index']+7)):
                problems.append('Serving owners are not disjoint contiguous prime+6warm blocks');break
    owner_map={};service={}
    for owner in owners:
        key=owner.get('mode'),owner.get('length');first=pmap.get(key,{})
        group=sorted([r for r in warm if (r.get('mode'),r.get('length'))==key],key=lambda r:r.get('execution_index',-1));samples=[first]+group
        errors=[]
        if key not in expected_owners or key in owner_map or not owner.get('owner_id') or len(group)!=6 or owner.get('prime_request_id')!=first.get('request_id') or owner.get('warm_request_ids')!=[r.get('request_id') for r in group] or owner.get('maximum_live_owners')!=1:errors.append('Private owner UUID/6-warm/single-live linkage differs')
        if not owner.get('owner_signature') or any(r.get('owner_id')!=owner.get('owner_id') or r.get('owner_signature')!=owner.get('owner_signature') for r in samples):errors.append('Persistent private owner changed')
        constructor,close=owner.get('constructor_and_detach_seconds'),owner.get('close_seconds');warm_seconds=sum(r.get('outer_seconds',0) for r in group)
        active=(constructor or 0)+first.get('outer_seconds',0)+warm_seconds+(close or 0);tokens=first.get('tokens',0)+sum(r.get('tokens',0) for r in group)
        checks=dict(cold_prime_outer_seconds=first.get('outer_seconds'),warm_outer_seconds_sum=warm_seconds,prime_tokens=first.get('tokens'),warm_tokens=sum(r.get('tokens',0) for r in group),accounted_service_seconds=active,accounted_service_tokens=tokens,accounted_service_tps=tokens/active if positive(active) else -1)
        if not nonnegative(constructor) or not nonnegative(close) or any(not near(owner.get(k),v) for k,v in checks.items()) or not near(first.get('cold_complete_outer_seconds'),(constructor or 0)+first.get('outer_seconds',0)):errors.append('Constructor/cold/6warm/close cost accounting differs')
        if owner.get('engine_closed') is not True or owner.get('hook_restored') is not True or owner.get('closed_runtime_stats',{}).get('entries')!=0 or owner.get('final_runtime_stats')!=(group[-1].get('runtime_stats') if group else None):errors.append('Owner lifecycle/graph release not proved')
        arm=amap.get(key[0],{})
        if arm.get('family')=='retrieval' and arm.get('backend')!='eager':
            fields=('captures','replays','resident_steps','capture_host_seconds','shadow_updates');previous={k:0 for k in fields}
            for row in samples:
                stats=row.get('runtime_stats',{});delta=row.get('draft_runtime',{}).get('request',{})
                if any(not near(stats.get(k),previous[k]+delta.get(k,-1)) for k in fields):errors.append('Lifetime runtime counters discontinuous')
                previous={k:stats.get(k,-1) for k in fields}
        if errors:problems.extend('owner '+str(key)+': '+e for e in errors);bad_owners.add(key)
        elif key not in bad_owners:service[key]=dict(cold_tps=first['tokens']/(constructor+first['outer_seconds']),accounted_tps=tokens/active,constructor_seconds=constructor,close_seconds=close)
        owner_map[key]=owner
    bad_owners.update(expected_owners-set(owner_map))
    if len(owner_map)!=76 or len({o.get('owner_id') for o in owners})!=76:problems.append('76 owner receipts incomplete or reused')
    raw_tree=rows('tree_raw.jsonl');raw_map={r.get('request_id'):r for r in raw_tree}
    tree_rows=[r for r in prime+warm if r.get('arm',{}).get('family')=='full_tree']
    if len(raw_tree)!=112 or len(raw_map)!=112 or set(raw_map)!={r.get('request_id') for r in tree_rows}:problems.append('112 upstream raw tree records incomplete/duplicated')
    for row in tree_rows:
        raw=raw_map.get(row.get('request_id'),{});value=raw.get('result',{})
        if (raw.get('execution_index')!=row.get('execution_index') or raw.get('owner_id')!=row.get('owner_id') or
            raw.get('outer_seconds')!=row.get('outer_seconds') or value.get('upstream')!=row.get('upstream') or
            value.get('generated')!=row.get('wrapper_generated') or value.get('tokens')!=row.get('wrapper_tokens')):
            problems.append('Original raw tree output differs from normalized evidence')
    observed=dict(reference_requests=len(canonical),prime_requests=len(prime),timing_requests=len(warm),owners=len(owners));complete=read('COMPLETE.json',{})
    if (observed!=EXPECTED or complete.get('passed') is not True or complete.get('counts')!=EXPECTED or complete.get('logical_requests')!=538 or
        any(complete.get(k) is not True for k in ('sources_unchanged','assets_unchanged','dependencies_unchanged','all_canonical','backend_trajectories_equal')) or
        complete.get('owner_closed')!=76 or complete.get('goal_complete') is not False):problems.append('538-request completion/immutability/exactness receipt missing')
    if (directory/'FAILURE.json').is_file():problems.append('Runner failure marker exists')
    metrics={};rankings={};usable={}
    for length in LENGTHS:
        metrics[str(length)]={};usable[str(length)]={}
        for mode,arm in amap.items():
            docs=[d for d in DOCS if metadata_valid and (mode,length) not in bad_owners and all((mode,length,d,r) in wmap and (mode,length,d,r) not in bad_warm for r in (0,1))]
            usable[str(length)][mode]=docs
            if not docs:continue
            cells=[pool_document([wmap[mode,length,d,r] for r in (0,1)]) for d in docs];estimate=pooled_bootstrap(cells)
            accepted=sum(c['accepted'] for c in cells);blocks=sum(c['blocks'] for c in cells);proposed=None if arm['family']=='full_tree' else sum(c['proposed'] for c in cells)
            repeat1=[wmap[mode,length,d,1] for d in docs]
            actual_rows=[r for r in warm if r.get('mode')==mode and r.get('length')==length and r.get('source_index') in docs]
            actual_tokens=sum(r['tokens'] for r in actual_rows);actual_seconds=sum(r['outer_seconds'] for r in actual_rows)
            metrics[str(length)][mode]=dict(documents=len(docs),pooled_tps=actual_tokens/actual_seconds,pooled_tps_ci95=estimate['ci95'],
                mean_document_tps=sum(c['tps'] for c in cells)/len(cells),per_document={str(d):c for d,c in zip(docs,cells)},
                warm_tokens=actual_tokens,warm_seconds=actual_seconds,
                acceptance_length=accepted/blocks,acceptance_length_ci95=quality_interval(cells,'blocks'),
                acceptance_rate=None if proposed is None else accepted/proposed,
                acceptance_rate_ci95=None if proposed is None else quality_interval(cells,'proposed'),
                acceptance_scope='RAW including overshoot' if arm['family']=='full_tree' else 'Linear accepted/proposed',
                warm_captures=sum(c['captures'] for c in cells),warm_capture_seconds=sum(c['capture_seconds'] for c in cells),
                observed_repeat1=dict(tps=sum(r['tokens'] for r in repeat1)/sum(r['outer_seconds'] for r in repeat1),
                    requests=[dict(source_index=r['source_index'],tokens=r['tokens'],outer_seconds=r['outer_seconds'],captures=r.get('draft_runtime',{}).get('request',{}).get('captures',0)) for r in repeat1]),
                **service.get((mode,length),{}))
        rankings[str(length)]=sorted([m for m,v in metrics[str(length)].items() if v['documents']==3],key=lambda m:(-metrics[str(length)][m]['pooled_tps'],m))
    winners=select_winners(metrics,arms)
    frozen=read('FROZEN_SELECTION.json',{})
    if (frozen.get('source_sha256')!=inventory or frozen.get('assets_sha256')!=protocol.get('assets_sha256') or
        frozen.get('protocol_sha256')!=(sha(directory/'protocol.json') if (directory/'protocol.json').is_file() else None) or
        frozen.get('complete_request_count')!=538 or frozen.get('no_new_confirmation_opened') is not True or
        complete.get('frozen_selection_sha256')!=(sha(directory/'FROZEN_SELECTION.json') if (directory/'FROZEN_SELECTION.json').is_file() else None)):
        problems.append('Frozen selection not bound to complete exact source/data/protocol')
    for length in LENGTHS:
        for group,mode in winners['selected'][str(length)].items():
            receipt=frozen.get('per_context',{}).get(str(length),{}).get('groups',{}).get(group,{})
            values=metrics[str(length)][mode]
            requests_for_mode=[r for r in warm if r.get('mode')==mode and r.get('length')==length]
            if receipt.get('mode')!=mode or receipt.get('arm')!=amap[mode] or not near(receipt.get('warm_tokens'),values['warm_tokens']) or not near(receipt.get('warm_seconds'),values['warm_seconds']) or receipt.get('warm_tps')!=values['pooled_tps'] or receipt.get('request_ids')!=[r['request_id'] for r in requests_for_mode]:problems.append(f'{length}/{group}: frozen winner differs from exact primary score')
        record=frozen.get('per_context',{}).get(str(length),{})
        if [r.get('mode') for r in record.get('ranking',[])]!=rankings[str(length)]:problems.append(str(length)+': complete frozen ranking differs from exact primary scores')
        if rankings[str(length)] and record.get('overall',{}).get('mode')!=rankings[str(length)][0]:problems.append(str(length)+': overall primary winner differs')
        for family,mode in winners['overall_family'][str(length)].items():
            if record.get('overall_family_winner',{}).get(family,{}).get('mode')!=mode:problems.append(str(length)+': overall family primary winner differs')
    if sum(len(v) for v in winners['selected'].values())!=8:problems.append('Eight backend/family winner candidates cannot be verified')
    return dict(status='COMPLETE' if not problems else 'PARTIAL',problems=problems,arms=arms,metrics=metrics,rankings=rankings,winners=winners,
        observed_counts=observed,winner_status='VERIFIED_FROZEN' if not problems else 'PROVISIONAL_ONLY',
        usable_documents=usable,bootstrap_seed=SEED,bootstrap_draws=10000,
        scope='Development menu selection on3 previously used original-train documents; not held-out,production,Vicuna or global-best superiority',
        raw_sha256={p.name:sha(p) for p in directory.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl')})


def analyze(directory):
    try:return _analyze(directory)
    except (TypeError,KeyError,AttributeError,ValueError,ZeroDivisionError,OverflowError) as error:
        path=Path(directory)
        return dict(status='PARTIAL',problems=['Malformed nested evidence; no performance admitted: '+type(error).__name__+': '+str(error)],
            arms=menu(),metrics={},rankings={},winners={},winner_status='PROVISIONAL_ONLY',observed_counts={},
            usable_documents={},bootstrap_seed=SEED,bootstrap_draws=10000,
            scope='Incomplete/malformed developer menu only; no frozen-winner,held-out or research-goal claim',
            raw_sha256={p.name:sha(p) for p in path.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl')} if path.is_dir() else {})


def plot(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    names=[];arms=summary['arms'];colors=dict(eager='#62798c',resident='#c69440',graph='#2c8b7b')
    title=f"{summary['status']} ·H10080GB ·38-config developer menu ·up to3 reused train docs ·4K/8K ·128 output ·2 repeats"
    def save(fig,name):
        fig.suptitle(title,fontsize=12);fig.tight_layout(rect=(0,.045,1,.95));fig.text(.5,.012,'Selection-biased development data only; document bootstrap10,000 draws. All legal captures charged; no steady-state claim.',ha='center',fontsize=9)
        fig.savefig(output/(name+'.png'),dpi=170);fig.savefig(output/(name+'.svg'));plt.close(fig);names.append(name)
    fig,axes=plt.subplots(3,2,figsize=(13,11))
    for col,n in enumerate(LENGTHS):
        values=summary['metrics'].get(str(n),{})
        for top,color in ((16,'#62798c'),(32,'#2c8b7b')):
            xs=[];lengths=[];rates=[];length_ci=[];rate_ci=[]
            for k in (1,2,4,8,12):
                v=values.get(f'retrieval{top}_k{k}_eager')
                if v:
                    xs.append(k);lengths.append(v['acceptance_length']);rates.append(v['acceptance_rate']);length_ci.append(v.get('acceptance_length_ci95'));rate_ci.append(v.get('acceptance_rate_ci95'))
            def errors(values,intervals):return np.array([[v-ci[0] if ci else 0. for v,ci in zip(values,intervals)],[ci[1]-v if ci else 0. for v,ci in zip(values,intervals)]])
            axes[0,col].errorbar(xs,lengths,yerr=errors(lengths,length_ci),marker='o',color=color,label=f'Top{top}',capsize=2)
            axes[1,col].errorbar(xs,rates,yerr=errors(rates,rate_ci),marker='o',color=color,label=f'Top{top}',capsize=2)
            ns=[];raw=[];raw_ci=[]
            for nodes in (8,32,50,64):
                v=values.get(f'tree{top}_n{nodes}')
                if v:ns.append(nodes);raw.append(v['acceptance_length']);raw_ci.append(v.get('acceptance_length_ci95'))
            axes[2,col].errorbar(ns,raw,yerr=errors(raw,raw_ci),marker='o',color=color,label=f'Top{top}',capsize=2)
        for row in range(3):axes[row,col].set_title(f'{n} exact input tokens');axes[row,col].legend(fontsize=8)
        axes[0,col].set_ylabel('Linear accepted Draft / round');axes[1,col].set_ylabel('Linear accepted / proposed');axes[2,col].set_ylabel('Tree RAW accepted / round\n(includes overshoot work)')
        axes[0,col].set_xlabel('Fixed draft k');axes[1,col].set_xlabel('Fixed draft k');axes[2,col].set_xlabel('Tree nodes')
    save(fig,'quality')
    for metric,name,ylabel in (('pooled_tps','tps','Six-warm pooled outer tokens/s'),('warm_captures','captures','New warm Graph captures'),('warm_capture_seconds','capture_time','New warm Graph capture host seconds'),('cold_tps','cold','Constructor+cold prime tokens/s'),('accounted_tps','service','Constructor+prime+6warm+close tokens/s')):
        fig,axes=plt.subplots(3,2,figsize=(13,11))
        for col,n in enumerate(LENGTHS):
            values=summary['metrics'].get(str(n),{})
            for row,top in enumerate((16,32)):
                ax=axes[row,col]
                for backend in BACKENDS:
                    xs=[];ys=[];lo=[];hi=[]
                    for k in (1,2,4,8,12):
                        v=values.get(f'retrieval{top}_k{k}_{backend}')
                        if v and v.get(metric) is not None:
                            xs.append(k);ys.append(v[metric]);ci=v.get('pooled_tps_ci95') if name=='tps' else None
                            lo.append(v[metric]-ci[0] if ci else 0.);hi.append(ci[1]-v[metric] if ci else 0.)
                    ax.errorbar(xs,ys,yerr=[lo,hi] if name=='tps' else None,marker='o',color=colors[backend],label=backend,capsize=2)
                ax.set_title(f'{n} ·Retrieval Top{top}');ax.set_xlabel('Fixed draft k');ax.set_ylabel(ylabel);ax.legend(fontsize=8);ax.set_ylim(bottom=0)
            ax=axes[2,col]
            for top,color in ((16,'#62798c'),(32,'#2c8b7b')):
                xs=[];ys=[];lo=[];hi=[]
                for nodes in (8,32,50,64):
                    v=values.get(f'tree{top}_n{nodes}')
                    if v and v.get(metric) is not None:
                        xs.append(nodes);ys.append(v[metric]);ci=v.get('pooled_tps_ci95') if name=='tps' else None
                        lo.append(v[metric]-ci[0] if ci else 0.);hi.append(ci[1]-v[metric] if ci else 0.)
                ax.errorbar(xs,ys,yerr=[lo,hi] if name=='tps' else None,marker='o',color=color,label=f'Top{top}',capsize=2)
            ax.set_title(f'{n} ·Full Qwen2 tree port');ax.set_xlabel('Tree nodes');ax.set_ylabel(ylabel);ax.legend(fontsize=8);ax.set_ylim(bottom=0)
        save(fig,name)
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    group_names=('retrieval/eager','retrieval/resident','retrieval/graph','full_tree')
    for column,length in enumerate(LENGTHS):
        chosen=summary.get('winners',{}).get('selected',{}).get(str(length),{})
        labels=[]
        for i,group in enumerate(group_names):
            mode=chosen.get(group);value=summary['metrics'].get(str(length),{}).get(mode,{})
            descriptor=next((a for a in arms if a['mode']==mode),{})
            short=('tree' if group=='full_tree' else group.split('/')[1])
            labels.append(short+'\n'+(f'T{descriptor.get("top_chunks")}/'+('n'+str(descriptor.get('nodes')) if group=='full_tree' else 'k'+str(descriptor.get('fixed_k'))) if descriptor else 'unavailable'))
            if not value:continue
            ci=value.get('pooled_tps_ci95');point=value['pooled_tps'];err=None if ci is None else np.array([[point-ci[0]],[ci[1]-point]])
            axes[0,column].bar(i,point,color=['#62798c','#c69440','#2c8b7b','#9b6b9e'][i],yerr=err,capsize=3)
            observations=value.get('observed_repeat1',{}).get('requests',[])
            for j,doc in enumerate(DOCS):
                row=next((r for r in observations if r['source_index']==doc),None)
                if row:
                    x=i+(j-1)*.13;y=row['tokens']/row['outer_seconds']
                    axes[1,column].scatter(x,y,color=['#62798c','#c69440','#2c8b7b'][j],marker=['o','s','^'][j],label='doc'+str(doc) if i==0 else None)
                    axes[1,column].annotate('c'+str(row['captures']),(x,y),xytext=(2,3),textcoords='offset points',fontsize=7)
        for ax in axes[:,column]:ax.set_xticks(range(4),labels);ax.set_ylim(bottom=0)
        axes[0,column].set_title(f'{length} ·'+summary.get('winner_status','PROVISIONAL_ONLY')+' candidates');axes[0,column].set_ylabel('Six-warm pooled outer tokens/s ·95% CI')
        axes[1,column].set_title(f'{length} ·All observed repeat1 requests ·c=new captures');axes[1,column].set_ylabel('Single-request outer tokens/s ·NOT steady state')
        if chosen:axes[1,column].legend(fontsize=8)
    save(fig,'selected_repeat1')
    return names


def write_report(directory,output):
    directory,output=Path(directory),Path(output)
    if output.exists():raise FileExistsError('Fresh analysis directory required')
    output.mkdir(parents=True);summary=analyze(directory);figures=plot(summary,output)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');shutil.copyfile(__file__,output/'analysis_source.py')
    winner_name='VERIFIED_WINNERS.json' if summary['status']=='COMPLETE' else 'PROVISIONAL_RANKING.json'
    (output/winner_name).write_text(json.dumps(dict(status=summary.get('winner_status','PROVISIONAL_ONLY'),
        authoritative=summary['status']=='COMPLETE',winners=summary['winners'],
        note='Verified only for this complete frozen developer menu; no new confirmation or research-goal success' if summary['status']=='COMPLETE' else 'Incomplete evidence: never treat these provisional candidates as frozen winners'),indent=2)+'\n')
    text=['# '+summary['status']+': Qwen2 finite baseline menu','',summary['scope'],'',
        'Primary rank = sum client tokens / sum whole-generate outer seconds across all six warm requests. Captures remain charged. Exactly equal scores use lexical mode tie-break; mean-document TPS and active-service TPS never replace this criterion.',
        'Per context, retain one winner for each retrieval backend plus one full-tree winner (eight candidates total). This is developer selection, not independent evidence of superiority; a larger stream could change capture amortization and backend ordering.',
        'Intervals resample three whole document token/time pairs and recompute pooled ratios. Repeats and contexts are not independent units. Tree quality is RAW accepted/round including overshoot; proposed counts, acceptance rates and unexposed tail IDs are not invented.',
        'All backends of the same retrieval geometry must match logical trajectories; quality panels use representative eager observations, with document-level ratio intervals. Complete full-tree intervals are shown separately.',
        'Every owner executes one separately charged doc271 cold prime followed by six warm requests; primary excludes the prime but includes every later capture. Constructor/cold/service views are secondary. Observed repeat1 includes all three documents and annotates captures; none are discarded or renamed steady-state.',
        'Source scope: Qwen2 FP32 Target and BF16 Draft/prefix, with the archived full-tree Flash/hybrid port and FP32 attention shim. Not original Vicuna/FP16 results. Serial owner blocks retain clock-drift limitations.','',
        '## Full primary rankings','']
    for n in LENGTHS:
        text+=['### '+str(n)+' input tokens','','| Rank | Mode | Six-warm pooled TPS | Document95% CI | Accepted/round | Warm captures |','|---:|---|---:|---|---:|---:|']
        for rank,mode in enumerate(summary.get('rankings',{}).get(str(n),[]),1):
            v=summary['metrics'][str(n)][mode];ci=v.get('pooled_tps_ci95');s='none' if ci is None else f'[{ci[0]:.4f},{ci[1]:.4f}]'
            text.append(f'| {rank} | {mode} | {v["pooled_tps"]:.4f} | {s} | {v["acceptance_length"]:.4f} | {v["warm_captures"]} |')
    text+=['','## Audit issues','']+(['- '+p for p in summary['problems']] or ['None. The frozen menu evidence is complete; the research goal is not thereby complete.'])
    text+=['','## Figures and raw evidence','']+['- ['+name+']('+name+'.png) ([SVG]('+name+'.svg))' for name in figures]
    text+=['- [Runnable standalone source](analysis_source.py)','- [Summary and raw SHA ledger](summary.json)','- [Winner verification status]('+winner_name+')']
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix in ('.json','.jsonl'):text.append('- ['+path.name+'](../'+path.name+')')
    (output/'REPORT.md').write_text('\n'.join(text)+'\n')
    (output/'ANALYSIS_COMPLETE.json').write_text(json.dumps(dict(status=summary['status'],artifacts_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()},rendered_inspection_required=True),indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=write_report(a.directory,a.output);print(json.dumps(dict(status=result['status'],problems=result['problems'],winners=result['winners']),indent=2))


if __name__=='__main__':main()
