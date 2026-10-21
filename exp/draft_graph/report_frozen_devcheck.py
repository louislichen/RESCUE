"""Portable frozen-RESCUE check on the already-used16-document development trace.

Reuses disclosed frozen study/record checking helpers, not an independent audit.
No models, original corpus, SSH or training imports. Whole-document bootstrap
intervals are conditional on the historical, non-concurrent owner blocks.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys

SUPPORT_FILES={'analysis_support.py':'rescue/baseline_selection_report.py',
    'analysis_horizon_support.py':'rescue/baseline_horizon_report.py',
    'analysis_fair_support.py':'rescue/fair_confirmation_report.py',
    'analysis_union_support.py':'rescue/fair_union_confirmation_report.py'}
SUPPORT_PATHS={}
for saved,relative in SUPPORT_FILES.items():
    source=Path(__file__).with_name(saved)
    if not source.is_file():source=Path(__file__).resolve().parents[2]/relative
    SUPPORT_PATHS[saved]=source
    spec=importlib.util.spec_from_file_location(saved[:-3],source);module=importlib.util.module_from_spec(spec)
    sys.modules[saved[:-3]]=module;spec.loader.exec_module(module)
f=sys.modules['analysis_union_support'];s=f.support
LENGTHS=(4096,8192);DOCS=tuple(f.horizon.DOCS);SEED=20261027;DRAWS=10000
EXPECTED=dict(own_quality_requests=64,quality_owners=64,prime_requests=2,timing_requests=64,owners=2)
TOTAL=130;OWN=dict(f.OWN);ASSETS=dict(f.ASSETS)
FIGURES=('acceptance','tps','contrasts','cold_captures','latency_rounds','actions_selection')
positive,nonnegative,near=f.positive,f.nonnegative,f.near


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(tokens,eos,vocab):
    return (isinstance(tokens,list) and 0<len(tokens)<=128 and all(type(t) is int and 0<=t<vocab for t in tokens) and
        (len(tokens)==128 or tokens[-1]==eos) and (eos not in tokens or tokens[-1]==eos and eos not in tokens[:-1]))


def bootstrap(cells,challengers,draws=DRAWS,seed=SEED):
    """Repeat-pool each doc first; one synchronized draw for both contexts/all arms."""
    import numpy as np
    if seed!=SEED or draws!=DRAWS:raise ValueError('Frozen development bootstrap seed/draws changed')
    result=dict(seed=seed,draws=draws,contexts={},contrasts=[],per_context_n={},formal_success=False,
        document_index_shared_across_available_contexts=False,interval_scope='conditional on observed cross-study owner blocks; not a fresh confirmation')
    for n,arms in cells.items():
        count=len(arms.get('rescue_graph',[]));result['per_context_n'][n]=count
        if not count:continue
        index=np.random.default_rng(seed).integers(0,count,size=(draws,count));values={}
        for mode,rows in arms.items():
            if len(rows)!=count:raise ValueError('Unpaired document cells')
            tokens=np.array([r['tokens'] for r in rows],float);seconds=np.array([r['seconds'] for r in rows],float)
            sample=tokens[index].sum(1)/seconds[index].sum(1)
            accepted=np.array([r['accepted'] for r in rows],float);blocks=np.array([r['blocks'] for r in rows],float)
            length=accepted.sum()/blocks.sum();quality=accepted[index].sum(1)/blocks[index].sum(1)
            proposed=None if any(r.get('proposed') is None for r in rows) else np.array([r['proposed'] for r in rows],float)
            values[mode]=dict(documents=count,tokens=int(tokens.sum()),seconds=float(seconds.sum()),pooled_tps=float(tokens.sum()/seconds.sum()),
                tps_ci95=[float(v) for v in np.percentile(sample,[2.5,97.5])] if count>1 else None,
                acceptance_length=float(length),acceptance_length_ci95=[float(v) for v in np.percentile(quality,[2.5,97.5])] if count>1 else None,
                acceptance_rate=None if proposed is None else float(accepted.sum()/proposed.sum()),
                acceptance_rate_ci95=None if proposed is None or count<2 else [float(v) for v in np.percentile(accepted[index].sum(1)/proposed[index].sum(1),[2.5,97.5])],
                raw_tree_unit=proposed is None,accepted=int(accepted.sum()),blocks=int(blocks.sum()))
            values[mode]['_sample']=sample
        for mode in challengers[n]:
            point=100*(values['rescue_graph']['pooled_tps']/values[mode]['pooled_tps']-1)
            sample=100*(values['rescue_graph']['_sample']/values[mode]['_sample']-1)
            result['contrasts'].append(dict(length=int(n),baseline_mode=mode,documents=count,point_percent=point,
                ci95=[float(v) for v in np.percentile(sample,[2.5,97.5])] if count>1 else None,
                nonconcurrent_cross_study=True,development_only=True))
        for value in values.values():value.pop('_sample')
        result['contexts'][n]=values
    return result


def own_config_errors(cfg):
    wanted=dict(selector='hierarchical',budget=512,recent=64,coverage_rows=64,coverage_candidates=1024,coverage_batch=16,
        coarse_chunk_size=32,refresh_every=4,refresh_policy='periodic',injection_layers=[27],inject_tokens=64,injection_strength=.5,
        draft_execution='cuda_graph',coverage_execution='eager',max_new_tokens=128,max_draft=12,working_mode='packed',
        optimized_execution=True,deferred_probes=True,attention_backend='sdpa',prefill_precision='bfloat16',schedule='serial',profile=False)
    wanted['length_policy']='risk'
    return ['Frozen own configuration differs: '+k for k,v in wanted.items() if cfg.get(k)!=v]


def _analyze(directory):
    e=f.Evidence(directory);p=e.json('protocol.json')
    try:choices=f.normalized_choices(p.get('union_freeze',{}).get('choices',{}))
    except (ValueError,TypeError,KeyError) as error:
        return dict(schema=1,status='PARTIAL',goal_complete=False,problems=[str(error)],statistics={},observed_counts={},expected_counts=EXPECTED,expected_requests=TOTAL)
    m=sum(len(v) for v in choices.values());studies=f.audit_union_chain(e,p,choices)
    if not studies:return dict(schema=1,status='PARTIAL',goal_complete=False,problems=e.problems,statistics={},comparisons=m)
    old=studies['stage2'];oldp=old['protocol'];base=f.Evidence(old['root']);oldrefs=base.json(old['result']+'/references.json')
    refs=e.json('references.json');reference=base.rows(old['result']+'/reference.jsonl');prime_refs=e.json('prime_references.json')
    reuse=e.json('reference_reuse.json');e.pin('reference_reuse.json',p.get('reference_reuse_sha256'))
    first=studies['stage1'];first_base=f.Evidence(first['root'])
    if refs!=oldrefs or len(reference)!=32:e.problem('Reused canonical32 reference bytes/rows differ from sealed stage2')
    expected_reuse=dict(status='SEALED_OLD_REFERENCE_REUSE',reused_reference_requests=34,generated_reference_requests=0,
        references_sha256=sha(base.path(old['result']+'/references.json')),reference_rows_sha256=sha(base.path(old['result']+'/reference.jsonl')),
        stage2_protocol_sha256=old['protocol_sha256'],stage1_protocol_sha256=first['protocol_sha256'],
        stage1_references_sha256=sha(first_base.path(first['result']+'/references.json')),input_sha256=oldp.get('input_sha256'),
        prime_input_sha256={str(n):first['protocol'].get('input_sha256',{}).get(f'271:{n}') for n in LENGTHS})
    if any(reuse.get(k)!=v for k,v in expected_reuse.items()) or p.get('reused_reference_requests')!=34 or p.get('generated_reference_requests')!=0:
        e.problem('Thirty-four reused references are not bound to sealed models/inputs/protocols; no new AR permitted')
    if p.get('prime_input_sha256')!=expected_reuse['prime_input_sha256']:e.problem('Paid prime input SHA differs from sealed stage1')
    if (p.get('source_indices')!=list(DOCS) or p.get('lengths')!=list(LENGTHS) or p.get('repeats')!=2 or p.get('max_new_tokens')!=128 or
        p.get('shared_warm_sequence')!=oldp.get('shared_warm_sequence') or p.get('union_freeze',{}).get('M')!=m or p.get('seed')!=oldp.get('seed')):e.problem('Old16 cohort/context or historical shared32 sequence changed')
    if p.get('expected_counts')!=EXPECTED or p.get('expected_requests')!=TOTAL:e.problem('New logical130 count plan changed; copied AR must not be counted as new generation')
    if any(p.get(k)!=v for k,v in ASSETS.items()) or any(p.get(k) is not True for k in ('mapper_frozen','policy_frozen','no_training')):
        e.problem('Frozen nonzero Mapper/old-cost policy assets or no-training contract changed')
    if p.get('fresh_confirmation') is not False or p.get('no_new_confirmation_access') is not True or p.get('no_search') is not True or p.get('goal_complete') is not False:
        e.problem('Development-only/no-new-data/no-search truth flags differ')
    env=p.get('environment',{})
    if 'H100' not in env.get('gpu_name','') or env.get('total_memory',0)<80_000_000_000 or env.get('target_dtype')!='torch.float32' or env.get('draft_dtype')!='torch.bfloat16' or env.get('prefix_dtype')!='torch.bfloat16' or env.get('tf32') is not False:
        e.problem('Single H10080GB/FP32 Target/BF16 Draft+prefix/TF32off receipt differs')
    for problem in own_config_errors(p.get('configs',{}).get('rescue_graph',{})):e.problem(problem)
    eager=dict(p.get('configs',{}).get('rescue_graph',{}),draft_execution='eager')
    if p.get('configs',{}).get('rescue_eager')!=eager:e.problem('Quality eager config differs beyond execution mode')
    source=p.get('source_sha256',{})
    if not source:e.problem('Frozen complete scientific source inventory absent')
    for name,digest in source.items():e.pin('source/'+name,digest)
    e.pin(p.get('entry_source_file','__missing__'),p.get('entry_source_sha256'))
    args=p.get('args',{});assets=p.get('assets_sha256',{})
    for role in ('teacher','student'):
        if not isinstance(args.get(role),str) or args[role]!=first['protocol'].get('args',{}).get(role) or args[role]!=oldp.get('args',{}).get(role):e.problem('Frozen model root differs from both studies: '+role)
    for name,digest in ASSETS.items():
        role=name.removesuffix('_sha256')
        if assets.get(args.get(role))!=digest:e.problem('Frozen asset path/SHA missing: '+role)
    bindings=p.get('source_bindings',{})
    for name,binding in bindings.items():
        relative=name.replace('.','/')+'.py'
        if not isinstance(binding,dict) or source.get(relative)!=binding.get('sha256') or not str(binding.get('path','')).endswith('/'+relative):e.problem('Imported source binding absent from source freeze')
    if not {'rescue.engine','rescue.query_mapper','rescue.step_budget','rescue.draft_graph'}<=set(bindings):e.problem('Mandatory actual own runtime source bindings missing')
    outer=f.Evidence(e.root.parent);inventory=outer.json('EVIDENCE_MANIFEST.json').get('file_sha256',{})
    if not inventory:outer.problem('Exported raw/source SHA inventory absent')
    for name,digest in inventory.items():outer.pin(name,digest)
    e.problems.extend('Export: '+v for v in outer.problems)
    # Re-read only portable, previously tokenized old inputs; never original
    # Parquet or old runtime/outcome metadata. They must be byte-identical to
    # the already-sealed stage2 data and produce precisely its input hashes.
    data_name=p.get('data_manifest','data/DATASET_MANIFEST.json');e.pin(data_name,p.get('data_manifest_sha256'))
    saved_manifest=e.json(data_name);old_manifest=base.json(old['result']+'/'+oldp.get('data_manifest','data/DATASET_MANIFEST.json'))
    if saved_manifest!=old_manifest:e.problem('Copied old-data manifest differs from sealed stage2')
    documents=e.rows('data/documents.jsonl');old_docs=base.rows(old['result']+'/data/documents.jsonl')
    paid=e.json('data/prime.json');old_paid=base.json(old['result']+'/data/prime.json')
    if documents!=old_docs or paid!=old_paid or [r.get('source_index') for r in documents]!=list(DOCS):e.problem('Selected old tokenized documents/paid271 changed')
    inputs={}
    for row in documents:
        prefix,body,suffix=(row.get(k,[]) for k in ('prefix_ids','document_ids','suffix_ids'))
        if not all(isinstance(v,list) and all(type(t) is int and t>=0 for t in v) for v in (prefix,body,suffix)) or len(body)<8192:
            e.problem('Old document token arrays cannot supply exact4K/8K');continue
        for n in LENGTHS:
            ids=prefix+body[:n-len(prefix)-len(suffix)]+suffix
            if len(ids)!=n:e.problem('Actual input length differs from nominal context')
    # Historical helper canonicalizes input IDs with its own JSON encoding.
    for row in documents:
        for n in LENGTHS:
            prefix,body,suffix=(row[k] for k in ('prefix_ids','document_ids','suffix_ids'))
            inputs[f"{row['source_index']}:{n}"]=f.horizon.input_sha(prefix+body[:n-len(prefix)-len(suffix)]+suffix)
    if p.get('input_sha256')!=inputs or oldp.get('input_sha256')!=inputs:e.problem('Own4K/8K inputs do not match old baseline inputs')
    old_data_paths=oldp.get('data_source_paths',{})
    if p.get('data_source_paths')!=old_data_paths:e.problem('Old metadata source layout differs')
    for name,path in old_data_paths.items():
        old_file=base.path(old['result']+'/'+path);e.pin(path,sha(old_file) if old_file.is_file() else None)
    eos,vocab=p.get('tokenizer_eos_token_id'),p.get('vocab_size')
    if type(eos) is not int or type(vocab) is not int or not 0<=eos<vocab or p.get('target_layers')!=28 or p.get('draft_layers')!=28 or p.get('graph_max_entries')!=128:
        e.problem('Target EOS/vocabulary/all28layers/Graph128 metadata invalid');eos,vocab=-1,0
    for row in reference:
        key=f"{row.get('source_index')}:{row.get('length')}"
        if refs.get(key)!=row.get('generated') or not canonical(row.get('generated'),eos,vocab):e.problem('Reused canonical Target output/EOS boundary invalid')
    if set(refs)!={f'{d}:{n}' for d in DOCS for n in LENGTHS} or len({(v.get('source_index'),v.get('length')) for v in reference})!=32:e.problem('Unique32 reused canonical cells incomplete')
    q=e.rows('own_quality.jsonl');primes=e.rows('prime.jsonl');warm=e.rows('runtime.jsonl');owners=e.rows('owners.jsonl');qo=e.rows('quality_owners.jsonl')
    observed=dict(own_quality_requests=len(q),quality_owners=len(qo),prime_requests=len(primes),timing_requests=len(warm),owners=len(owners))
    if observed!=EXPECTED or len(qo)!=64:e.problem('Complete130 requests/64 quality owners/2 timing owners required')
    if (any(not isinstance(v.get('owner_id'),str) or not v['owner_id'] for v in qo+owners) or
        len({v.get('owner_id') for v in qo})!=len(qo) or len({v.get('owner_id') for v in owners})!=len(owners) or {v.get('owner_id') for v in qo}&{v.get('owner_id') for v in owners}):e.problem('Fresh quality/timing owner UUIDs missing/reused')
    requests=q+primes+warm;ids=[r.get('request_id') for r in requests];indices=[r.get('execution_index') for r in requests]
    valid_index=all(type(v) is int for v in indices)
    if len(set(ids))!=len(ids) or any(not isinstance(v,str) or not v for v in ids):e.problem('New request UUID reused/missing')
    if not valid_index or sorted(indices)!=list(range(TOTAL)):e.problem('New130 execution index ledger incomplete')
    oq={};qowners={v.get('request_id'):v for v in qo}
    bad=set()
    for row in q:
        doc,n,mode=row.get('source_index'),row.get('length'),row.get('mode');key=doc,n,mode;ref=refs.get(f'{doc}:{n}')
        problems=[]
        if key in oq or doc not in DOCS or n not in LENGTHS or mode not in ('rescue_eager','rescue_graph'):problems.append('Own quality matrix unregistered/duplicate')
        backend='eager' if mode=='rescue_eager' else 'cuda_graph'
        if row.get('phase')!='own_quality' or row.get('arm')!=dict(mode=mode,family='rescue',backend=backend):problems.append('Own quality declared method differs')
        if backend=='cuda_graph' and (row.get('runtime_stats',{}).get('max_graph_entries')!=128 or row.get('draft_runtime',{}).get('mode')!='cuda_graph'):problems.append('Own quality actual Graph128 mode differs')
        oq[key]=row
        if row.get('generated')!=ref or row.get('tokens')!=len(ref or []) or row.get('output_agrees') is not True or row.get('performance_eligible') is not False:problems.append('Quality differs from AR or marked timed')
        problems+=s.observation_errors(row)+s.quality_pair_errors(row,row)+s.linear_trace_errors(row,own=True)+s.injection_errors(row)
        owner=qowners.get(row.get('request_id'),{})
        if owner.get('owner_id')!=row.get('owner_id') or owner.get('engine_closed') is not True or owner.get('hook_restored') is not True or owner.get('closed_runtime_stats',{}).get('entries')!=0 or owner.get('final_runtime_stats')!=row.get('runtime_stats'):
            problems.append('Quality owner was not closed with exact runtime snapshot')
        for problem in problems:e.problem(f'Quality {doc}/{n}/{mode}: '+problem)
        if problems:bad.add((doc,n))
    for n in LENGTHS:
        for doc in DOCS:
            a,b=oq.get((doc,n,'rescue_eager')),oq.get((doc,n,'rescue_graph'))
            if a is None or b is None:e.problem('Own eager/Graph pair missing');bad.add((doc,n));continue
            errors=s.quality_pair_errors(a,b)
            for problem in errors:e.problem(f'Pair {doc}/{n}: '+problem)
            if errors:bad.add((doc,n))
    if valid_index and q and (min(v['execution_index'] for v in q)!=0 or max(v['execution_index'] for v in q)!=63 or
        primes and min(v['execution_index'] for v in primes)!=64):e.problem('All64 quality owners must close before timing starts')
    quality_gate=e.json('QUALITY_PASSED.json')
    if quality_gate.get('passed') is not True:e.problem('Raw-checked quality stage has no successful gate')
    wm={};pm={};owner_map={v.get('owner_id'):v for v in owners}
    for phase,rows in (('cold_prime',primes),('warm',warm)):
        for row in rows:
            doc,n=row.get('source_index'),row.get('length');ref=prime_refs.get(str(n)) if phase=='cold_prime' else refs.get(f'{doc}:{n}')
            errors=s.timing_errors(row,ref,OWN,'rescue_graph')
            if row.get('mode')!='rescue_graph' or row.get('arm')!=OWN or n not in LENGTHS or row.get('phase')!=phase or any(row.get(k) is not False for k in ('observer_enabled','profile_enabled')) or any(row.get(k) is not True for k in ('mapper_executed','controller_executed')):
                errors.append('Timed own method is unfrozen/instrumented')
            if phase=='warm':
                key=n,doc,row.get('repeat')
                if key in wm or doc not in DOCS or row.get('repeat') not in (0,1):errors.append('Own warm matrix unregistered/duplicate')
                wm[key]=row;target=oq.get((doc,n,'rescue_graph'))
                if target is None or s.trace_signature(target)!=s.trace_signature(row) or any(target.get(k)!=row.get(k) for k in ('accepted','proposed','blocks','draft_forwards','selector_refreshes','working_gather_layers','projected_positions')):errors.append('Timing trajectory differs from same-document quality')
            else:
                if n in pm or doc!=271:errors.append('Paid unique old271 prime missing/changed')
                pm[n]=row;delta=row.get('draft_runtime',{}).get('request',{});life=row.get('runtime_stats',{})
                if not positive(delta.get('captures')) or any(not near(delta.get(k),life.get(k)) for k in ('captures','replays','resident_steps','capture_host_seconds')):errors.append('Cold timing owner borrowed quality graphs')
            for problem in errors:e.problem(f'{phase} {doc}/{n}: '+problem)
            if errors and phase=='warm':bad.add((doc,n))
    sequence=oldp.get('shared_warm_sequence',[]);plan=p.get('owner_order',[])
    if len(plan)!=2 or {(v.get('mode'),v.get('length')) for v in plan}!={('rescue_graph',n) for n in LENGTHS}:e.problem('Exactly two private own owners must be pre-frozen')
    actual_prime=[dict(mode=v.get('mode'),length=v.get('length')) for v in primes]
    if actual_prime!=plan:e.problem('Actual own owner block order differs')
    for j,prime in enumerate(primes):
        block=[v for v in warm if v.get('owner_id')==prime.get('owner_id')];owner=owner_map.get(prime.get('owner_id'),{})
        if [{k:v.get(k) for k in ('source_index','repeat')} for v in block]!=sequence or [v.get('warm_index') for v in block]!=list(range(32)):
            e.problem('Actual32 request sequence differs from sealed baseline horizon')
        if valid_index and (prime.get('execution_index')!=64+33*j or [v.get('execution_index') for v in block]!=list(range(65+33*j,97+33*j))):e.problem('Owner block is not one cold plus32 consecutive warm')
        for problem in s.lifecycle_errors(owner,[prime,*block]):e.problem('Owner: '+problem)
        if owner.get('maximum_live_owners')!=1:e.problem('Overlapping private owners')
    telemetry=e.rows('gpu_diagnostics.jsonl') if (e.root/'gpu_diagnostics.jsonl').is_file() else []
    expected_telemetry=[(phase,n,64+33*j+(33 if phase=='owner_end' else 0)) for j,n in enumerate(LENGTHS) for phase in ('owner_start','owner_end')]
    if [(v.get('phase'),v.get('length'),v.get('execution_index')) for v in telemetry]!=expected_telemetry or any(v.get('mode')!='rescue_graph' or v.get('owner_id') not in owner_map for v in telemetry):e.problem('Four read-only timing-owner telemetry boundary records required')
    if set(wm)!={(n,d,r) for n in LENGTHS for d in DOCS for r in (0,1)} or set(pm)!=set(LENGTHS):e.problem('Own cold/warm matrix incomplete')
    baseline={};baseline_bad=set()
    for row in old['raw']:
        n,mode,doc,rep=row.get('length'),row.get('mode'),row.get('source_index'),row.get('repeat')
        if mode not in choices.get(str(n),{}):continue
        key=n,mode,doc,rep
        if key in baseline or doc not in DOCS or rep not in (0,1):e.problem('Retained stage2 baseline cell duplicate/unregistered');baseline_bad.add((n,doc))
        baseline[key]=row;ref=refs.get(f'{doc}:{n}')
        if row.get('generated')!=ref or row.get('tokens')!=len(ref or []) or row.get('output_agrees') is not True or not positive(row.get('outer_seconds')) or not nonnegative(row.get('accepted')) or not positive(row.get('blocks')):
            e.problem('Retained baseline canonical/primary/acceptance record invalid');baseline_bad.add((n,doc))
    for n in LENGTHS:
        for mode in choices[str(n)]:
            ordered=[r for r in old['raw'] if r.get('length')==n and r.get('mode')==mode]
            if [{k:r.get(k) for k in ('source_index','repeat')} for r in ordered]!=sequence:e.problem('Retained baseline shared32 order differs')
    per_context={};cells={};challengers={}
    for n in LENGTHS:
        modes=list(choices[str(n)]);challengers[str(n)]=modes;docs=[d for d in DOCS if (d,n) not in bad and (n,d) not in baseline_bad and all((n,d,r) in wm for r in (0,1)) and all((n,mode,d,r) in baseline for mode in modes for r in (0,1))]
        per_context[str(n)]=docs
        if not docs:continue
        cells[str(n)]={'rescue_graph':[f.pool_document([wm[n,d,r] for r in (0,1)]) for d in docs]}
        for mode in modes:cells[str(n)][mode]=[f.pool_document([baseline[n,mode,d,r] for r in (0,1)]) for d in docs]
    done=e.json('COMPLETE.json')
    if done.get('passed') is not True or done.get('counts')!=EXPECTED or done.get('logical_requests')!=TOTAL or done.get('goal_complete') is not False or any(done.get(k) is not True for k in ('sources_unchanged','assets_unchanged','dependencies_unchanged','quality_gate_passed','all_canonical','owners_closed','data_unchanged','dual_study_freeze_unchanged')):
        e.problem('Terminal immutable130-request completion receipt invalid')
    if (e.root/'FAILURE.json').exists():e.problem('Run failed; preserved subset only')
    status='COMPLETE' if not e.problems else 'PARTIAL';stats=bootstrap(cells,challengers)
    stats['document_index_shared_across_available_contexts']=bool(cells) and len({tuple(per_context[n]) for n in cells})==1
    feasible=status=='COMPLETE' and len(stats['contrasts'])==m and all(v['point_percent']>0 for v in stats['contrasts'])
    actions={};selected={}
    for n in LENGTHS:
        counts={}
        for row in warm:
            if row.get('length')==n:
                for step in row.get('trace',[]):counts[str(step.get('cap'))]=counts.get(str(step.get('cap')),0)+1
        actions[str(n)]=counts;hist=None
        for row in q:
            if row.get('length')==n and row.get('mode')=='rescue_graph':
                if hist is None:hist=[0]*20
                for step in row.get('observed_rounds',[]):
                    for pos in step.get('selected',[]):
                        if type(pos) is int and 0<=pos<step.get('prefix',0):hist[min(19,int(20*pos/step['prefix']))]+=1
        selected[str(n)]=hist
    return dict(schema=1,status=status,goal_complete=False,fresh_confirmation=False,synthetic_program_test_only=p.get('synthetic_program_test_only') is True,development_feasible=feasible,problems=e.problems,statistics=stats,comparisons=m,
        observed_counts=observed,expected_counts=EXPECTED,expected_requests=TOTAL,observed_requests=len(requests),reused_reference_requests=34,
        source_indices=list(DOCS),paired_complete_documents_by_context=per_context,frozen_choices=choices,
        diagnostics=dict(prime=primes,owners=owners,warm=warm,actions=actions,quality_selected_position_histogram=selected,gpu_telemetry=telemetry),raw_sha256={v.name:sha(v) for v in e.root.glob('*.json*') if v.is_file()},
        limitations=['Already-used16 development documents:12 historical corpus training documents and4 previously reused validation documents, not fresh held-out confirmation. No training or parameter search.',
            'RESCUE and baseline measurements are separate historical owner blocks, not contemporaneously randomized; clock drift is not removed.',
            'CIs condition on observed document/capture history and do not cover owner-block or hardware drift; no formal goal or fresh-confirmation gate.',
            'Finite one-paid-prime plus32-warm trace, all legal captures charged; not production or fully-prewarmed steady-state speed.',
            'Full-tree acceptance is raw accepted path tokens per verification round, including charged overshoot; not a node acceptance rate.',
            'Reuses the disclosed frozen report record/proof checking helpers; this is not a new independent raw audit.'])


def analyze(directory):
    try:return _analyze(directory)
    except (OSError,ValueError,KeyError,TypeError,IndexError,AttributeError,OverflowError) as error:
        return dict(schema=1,status='PARTIAL',goal_complete=False,fresh_confirmation=False,problems=['Malformed/missing saved evidence: '+type(error).__name__+': '+str(error)],statistics={},observed_counts={},expected_counts=EXPECTED,expected_requests=TOTAL)


def figures(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    partial=summary.get('status')!='COMPLETE';stats=summary.get('statistics',{});contexts=stats.get('contexts',{});title='PARTIAL — ' if partial else ''
    if summary.get('synthetic_program_test_only'):title='SYNTHETIC PROGRAM TEST — '+title
    counts=stats.get('per_context_n',{});caption=f'Planned old16 development docs; paired completed4K n={counts.get("4096",0)},8K n={counts.get("8192",0)}; ≤128 tokens,2 repeats; conditional bootstrap95%.\nHistorical baseline vs new own blocks, not contemporaneous. One paid cold +32 warm when complete; all captures charged; no fresh-confirmation/goal claim.'
    def save(fig,name):
        fig.suptitle(title+name.replace('_',' ').title()+' — frozen development check',fontsize=12);fig.text(.5,.015,caption,ha='center',fontsize=7)
        fig.tight_layout(rect=(0,.11,1,.94));fig.savefig(output/(name+'.png'),dpi=160);fig.savefig(output/(name+'.svg'));plt.close(fig)
    def error(ax,x,value,ci,**kw):
        if ci is not None:
            ax.vlines(x,ci[0],ci[1],**kw);ax.hlines(ci,[x-.06]*2,[x+.06]*2,**kw)
        ax.plot(x,value,marker='o',linestyle='none',**kw)
    for name,metric in (('acceptance','acceptance_length'),('tps','pooled_tps')):
        fig,axes=plt.subplots(1,2,figsize=(13,5),squeeze=False)
        for col,n in enumerate(LENGTHS):
            ax=axes[0,col];values=contexts.get(str(n),{});modes=list(values)
            for j,mode in enumerate(modes):
                value=values[mode];error(ax,j,value[metric],value.get('acceptance_length_ci95' if name=='acceptance' else 'tps_ci95'),color='#167d8d' if mode=='rescue_graph' else '#b76b32')
            ax.set_xticks(range(len(modes)),[v.replace('retrieval','R').replace('rescue_graph','RESCUE') for v in modes],rotation=55,ha='right',fontsize=7)
            ax.set_title(f'{n//1024}K actual ({n} input tokens); n={stats.get("per_context_n",{}).get(str(n),0)}');ax.grid(axis='y',alpha=.25)
            ax.set_ylabel('Accepted Draft/path tokens per round\n(tree: raw counts, including overshoot)' if name=='acceptance' else 'Whole-generate outer tokens / second',fontsize=9)
            ax.set_ylim(bottom=0)
            if not modes:ax.text(.5,.5,'No complete paired document subset',ha='center',transform=ax.transAxes)
        save(fig,name)
    fig,axes=plt.subplots(1,2,figsize=(12,5),squeeze=False)
    for col,n in enumerate(LENGTHS):
        ax=axes[0,col];rows=[v for v in stats.get('contrasts',[]) if v['length']==n]
        for j,row in enumerate(rows):
            if row.get('ci95') is not None:
                ax.hlines(j,*row['ci95'],color='#167d8d');ax.vlines(row['ci95'],[j-.06]*2,[j+.06]*2,color='#167d8d')
            ax.plot(row['point_percent'],j,marker='o',linestyle='none',color='#167d8d')
        ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(len(rows)),[v['baseline_mode'] for v in rows],fontsize=7);ax.set_title(f'{n//1024}K actual');ax.set_xlabel('RESCUE relative pooled TPS difference (%)');ax.grid(axis='x',alpha=.2)
    save(fig,'contrasts')
    diag=summary.get('diagnostics',{});primes=diag.get('prime',[]);owners=diag.get('owners',[]);warm=diag.get('warm',[])
    fig,axes=plt.subplots(2,3,figsize=(13,7));axes=axes.reshape(-1)
    for col,(label,key) in enumerate((('Cold constructor + generate (s)','cold_complete_outer_seconds'),('Warm captures (count)','captures'),('Warm capture host time (s)','capture_host_seconds'))):
        for j,n in enumerate(LENGTHS):
            if col==0:value=next((v.get(key) for v in primes if v.get('length')==n),None)
            else:
                observed=[v for v in warm if v.get('length')==n]
                value=sum(v.get('draft_runtime',{}).get('request',{}).get(key,0) for v in observed) if observed else None
            if value is not None:axes[col].bar(j,value,color='#167d8d')
            else:axes[col].text(j,.02,'No data',ha='center',rotation=90,transform=axes[col].get_xaxis_transform(),fontsize=7)
        axes[col].set_xticks([0,1],['4K','8K']);axes[col].set_ylabel(label);axes[col].grid(axis='y',alpha=.2);axes[col].set_ylim(bottom=0)
    for j,n in enumerate(LENGTHS):
        values=[v.get('prefill_seconds') for v in warm if v.get('length')==n and nonnegative(v.get('prefill_seconds'))]
        if values:axes[3].bar(j,sum(values)/len(values),color='#167d8d')
    axes[3].set_xticks([0,1],['4K','8K']);axes[3].set_ylabel('Saved engine prefill / request (s)');axes[3].grid(axis='y',alpha=.2);axes[3].set_ylim(bottom=0)
    telemetry=diag.get('gpu_telemetry',[])
    for col,key,label in ((4,'temperature.gpu','Recorded GPU temperature (°C)'),(5,'clocks.current.sm','Recorded SM clock (MHz)')):
        xs=[];ys=[]
        for j,row in enumerate(telemetry):
            try:value=float(row.get('telemetry',{}).get('values',{}).get(key))
            except (ValueError,TypeError):continue
            if math.isfinite(value):xs.append(j);ys.append(value)
        if ys:axes[col].plot(xs,ys,'o-',color='#b76b32')
        else:axes[col].text(.5,.5,'No numeric telemetry recorded',ha='center',transform=axes[col].transAxes,fontsize=8)
        axes[col].set_xticks(range(len(telemetry)),[f"{v.get('length',0)//1024}K\n{v.get('phase','').replace('owner_','')}" for v in telemetry],fontsize=7)
        axes[col].set_ylabel(label);axes[col].set_xlabel('Outside primary clock; no drift correction');axes[col].grid(axis='y',alpha=.2)
    save(fig,'cold_captures')
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for col,(label,key) in enumerate((('Whole decode / verification round (s)\nNOT isolated Target latency','decode_seconds'),('Mean verification rounds / request','blocks'),('Accounted service TPS\nconstructor+cold+32warm+close','accounted_service_tps'))):
        for j,n in enumerate(LENGTHS):
            values=[v.get(key) for v in (owners if col==2 else warm) if v.get('length')==n and nonnegative(v.get(key))]
            if values:
                value=sum(values)/sum(v.get('blocks',0) for v in warm if v.get('length')==n) if col==0 else sum(values)/len(values)
                axes[col].bar(j,value,color='#167d8d')
        axes[col].set_xticks([0,1],['4K','8K']);axes[col].set_ylabel(label);axes[col].grid(axis='y',alpha=.2)
    save(fig,'latency_rounds')
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for col,n in enumerate(LENGTHS):
        actions=diag.get('actions',{}).get(str(n),{});keys=sorted(actions,key=lambda v:int(v))
        axes[0,col].bar(keys,[actions[v] for v in keys],color='#167d8d');axes[0,col].set_title(f'{n//1024}K actual warm actions');axes[0,col].set_xlabel('Actual proposal cap (tail-limited included)');axes[0,col].set_ylabel('Observed verification rounds')
        if not keys:axes[0,col].text(.5,.5,'No measured warm actions',ha='center',transform=axes[0,col].transAxes)
        hist=diag.get('quality_selected_position_histogram',{}).get(str(n),[])
        if hist:axes[1,col].bar((np.arange(20)+.5)/20,hist,width=.045,color='#b76b32')
        else:axes[1,col].text(.5,.5,'No saved Graph-quality positions',ha='center',transform=axes[1,col].transAxes)
        axes[1,col].set_xlabel('Selected absolute position / committed prefix');axes[1,col].set_ylabel('Selected positions in Graph quality\n(quality records only;\nnot timing repeats)',fontsize=9);axes[1,col].set_ylim(bottom=0)
    save(fig,'actions_selection');return list(FIGURES)


def write_report(summary,directory,output):
    output=Path(output)
    if output.exists():raise FileExistsError('Fresh analysis output required; never overwrite')
    output.mkdir(parents=True);source=Path(__file__);shutil.copyfile(source,output/'analysis_source.py')
    for saved,relative in SUPPORT_FILES.items():
        shutil.copyfile(SUPPORT_PATHS[saved],output/saved)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n');rendered=figures(summary,output)
    lines=['# '+summary.get('status','PARTIAL')+' — frozen RESCUE old-development check','',
        'Not fresh confirmation; no goal completion, new winner freeze, training or parameter search.','',
        'New generation:64 own eager/Graph quality +2 cold +64 warm =130;34 old references (32 canonical AR +2 paid-prime reference outputs) and all retained baseline records are reused.','',
        'Primary: repeat-pool token/time within each document, then pooled token/time throughput. Pointwise paired document-bootstrap95% intervals are conditional on historical owner blocks; no multiplicity-adjusted success claim.','']
    if summary.get('synthetic_program_test_only'):lines[2:2]=['SYNTHETIC PROGRAM TEST ONLY: fabricated records, no real model/GPU results.','']
    lines+=['- '+v for v in summary.get('limitations',[])]
    lines+=['',f"Development preliminary go/no-go: {summary.get('development_feasible',False)}. This requires COMPLETE quality/exactness/nonzero/source/owner gates and strictly positive pooled-TPS points against ALL retained challengers at both contexts. Descriptive CIs do not gate this preliminary rule. It never completes the goal or substitutes for fresh confirmation."]
    lines+=['','## Every retained comparison','','|Context|Frozen baseline|Paired docs|TPS difference|Conditional95% CI|','|---|---|---:|---:|---|']
    for row in summary.get('statistics',{}).get('contrasts',[]):lines.append(f"|{row['length']}|{row['baseline_mode']}|{row['documents']}|{row['point_percent']:.3f}%|{row.get('ci95')}|")
    lines+=['','## Charged own owner costs','','|Context|Constructor (s)|Paid prime generate (s)|All32 warm generate (s)|Close (s)|Accounted service TPS|','|---|---:|---:|---:|---:|---:|']
    for row in summary.get('diagnostics',{}).get('owners',[]):lines.append('|'+ '|'.join(str(row.get(k)) for k in ('length','constructor_and_detach_seconds','cold_prime_outer_seconds','warm_outer_seconds_sum','close_seconds','accounted_service_tps'))+'|')
    lines+=['','## Saved quality, performance and mechanism figures','']
    for name in rendered:lines+=[f'[{name} PNG]({name}.png) · [SVG]({name}.svg)',f'![{name}]({name}.png)','']
    lines+=['## Evidence issues','']+(['- '+v for v in summary.get('problems',[])] or ['None.'])
    lines+=['','## Saved sources and raw records','', '[Analysis](analysis_source.py) · [Frozen union support](analysis_union_support.py) · [Fair support](analysis_fair_support.py) · [Horizon support](analysis_horizon_support.py) · [Baseline support](analysis_support.py)','']
    for name in ('protocol.json','COMPLETE.json','QUALITY_PASSED.json','own_quality.jsonl','quality_owners.jsonl','prime.jsonl','runtime.jsonl','owners.jsonl','gpu_diagnostics.jsonl','reference_reuse.json','reference.jsonl','references.json','prime_references.json','data/DATASET_MANIFEST.json'):
        if (Path(directory)/name).is_file():lines.append(f'[{name}]({os.path.relpath(Path(directory)/name,output)})')
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    receipt=dict(status=summary.get('status','PARTIAL'),goal_complete=False,fresh_confirmation=False,render_required=True,rendered_figures=rendered,
        file_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()},independent_audit=False,scope='Development-only cross-study feasibility check; actual root figure inspection still required')
    (output/'ANALYSIS_COMPLETE.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();summary=analyze(args.directory);write_report(summary,args.directory,args.output)
    print(json.dumps(dict(status=summary.get('status'),goal_complete=False,problems=summary.get('problems')),indent=2))


if __name__=='__main__':main()
