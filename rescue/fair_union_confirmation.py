"""Fresh confirmation against the immutable stage1+stage2 contender union.

This module does not select a method or a document by observed outcomes. Both
completed development studies and their final source/data gate must be verified
before any fresh confirmation content is read or hashed.
"""
import argparse
import copy
from dataclasses import asdict,replace
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import shutil
import time
import uuid

import torch

from . import baseline_selection as baseline
from . import fair_confirmation as legacy
from .draft_cost_recalibration import QualityObserver,detach_quality_observer,record_draft_steps
from .draft_graph_benchmark import HookOwners,owner_signature,record,check_output as check_own_output
from .draft_graph_smoke import COST_SHA256,native_digest,runtime_snapshot
from .round_overhead_run import MAPPER_SHA256,POLICY_SHA256,append,check_pair,context,gpu_snapshot,model_files,require,sha256,write

SEED=20261026
LENGTHS=(4096,8192)
REPEATS=2
OUR_MODE='rescue_graph'
GROUPS=('retrieval/eager','retrieval/resident','retrieval/graph','full_tree')
PREREGISTRATION='exp/draft_graph/fair_union_confirmation_20261020_preregistration.json'
PIPELINE='exp/draft_graph/run_fair_union_confirmation_20261020_v1.sh'


def progress(stage,counts,**fields):print(json.dumps(dict(stage=stage,counts=dict(counts),**fields)),flush=True)


def group_of(arm):
    return 'full_tree' if arm['family']=='full_tree' else 'retrieval/'+('graph' if arm['backend']=='cuda_graph' else arm['backend'])


def frozen_choices(selection):
    """Retain original-group order and stage1 first, never re-rank a union."""
    registered={a['mode']:a for a in baseline.arms()};choices={}
    for n in LENGTHS:
        cell=selection.get('per_context',{}).get(str(n),{});groups=cell.get('union_by_group',{})
        require(set(groups)==set(GROUPS),'Missing an immutable contender group')
        choices[str(n)]={}
        for name in GROUPS:
            entries=groups[name];require(isinstance(entries,list) and 1<=len(entries)<=2,'Expected stage1 then optional distinct stage2 contender')
            seen=set()
            for item in entries:
                arm=item.get('arm');mode=item.get('mode')
                require(mode in registered and arm==registered[mode] and group_of(arm)==name and mode not in seen,
                    'Unknown, duplicated or misgrouped frozen contender')
                seen.add(mode);choices[str(n)][mode]=copy.deepcopy(arm)
        require(list(choices[str(n)])==cell.get('union_modes'),'Union order/membership differs from frozen receipt')
    validate_choices(choices)
    return choices


def validate_choices(choices):
    registered={a['mode']:a for a in baseline.arms()}
    require(set(choices)=={str(n) for n in LENGTHS},'Both frozen contexts required')
    for n in LENGTHS:
        values=choices[str(n)];require(isinstance(values,dict) and 4<=len(values)<=8,'Four to eight contenders per context required')
        require(all(mode in registered and arm==registered[mode] for mode,arm in values.items()),'Union contains an unregistered arm')
        groups=[group_of(a) for a in values.values()]
        require(all(1<=groups.count(g)<=2 for g in GROUPS) and groups==sorted(groups,key=GROUPS.index),
            'Contender group order/count changed')
    return sum(len(c) for c in choices.values())


def extra_geometries(choices):
    validate_choices(choices);result=[]
    for n in LENGTHS:
        arms=choices[str(n)].values()
        seen={(a['top_chunks'],a['fixed_k']) for a in arms if a['family']=='retrieval' and a['backend']=='eager'}
        for arm in arms:
            if arm['family']!='retrieval' or arm['backend']=='eager':continue
            geometry=arm['top_chunks'],arm['fixed_k']
            if geometry not in seen:
                result.append(dict(length=n,top_chunks=geometry[0],fixed_k=geometry[1]));seen.add(geometry)
    require(0<=len(result)<=8,'Matching eager geometry reference count outside frozen bound')
    return result


def expected_counts(choices):
    m=validate_choices(choices);owners=m+2
    return dict(reference_requests=32,own_quality_requests=64,baseline_quality_requests=m,
        matching_quality_requests=len(extra_geometries(choices)),prime_requests=owners,timing_requests=32*owners,owners=owners)


def scientific_requests(choices):
    return sum(v for k,v in expected_counts(choices).items() if k!='owners')


def timing_schedule(choices,documents,seed=SEED):
    validate_choices(choices);require(len(documents)==16 and len(set(documents))==16,'Exactly16 fixed fresh documents required')
    rng=random.Random(seed);sequence=[]
    for repeat in range(REPEATS):
        order=list(documents);rng.shuffle(order);sequence.extend(dict(source_index=i,repeat=repeat) for i in order)
    owners=[];warm=[]
    for n in LENGTHS:
        modes=[OUR_MODE]+list(choices[str(n)]);random.Random(seed+n).shuffle(modes)
        for mode in modes:
            owners.append(dict(mode=mode,length=n));warm.extend(dict(mode=mode,length=n,**item) for item in sequence)
    return sequence,owners,warm


def source_bindings(root):
    receipt=legacy.source_bindings(root)
    for name in ('rescue.fair_union_confirmation','rescue.fair_union_confirmation_data','rescue.fair_union_confirmation_report'):
        module=importlib.import_module(name);path=Path(module.__file__).resolve();expected=root/Path(*name.split('.')).with_suffix('.py')
        require(path==expected,'Union confirmation imported a foreign module: '+name)
        receipt[name]=dict(path=str(path),sha256=sha256(path))
    return receipt


def validate_minimum_sources(root,inventory):
    tree=list((root/'exp/specextend_bridgekv/specextend').rglob('*.py'))
    require(len(tree)>=27,'Complete unchanged tree source required before data access')
    required=list((root/'rescue').glob('*.py'))+tree+[
        root/'exp/attention_horizon/selection.py',root/'exp/selective_kv/bridge.py',root/PREREGISTRATION,root/PIPELINE,
        *[root/'tests'/('test_rescue_fair_union_confirmation'+suffix+'.py') for suffix in ('','_data','_report')]]
    require(all(p.is_file() and str(p.relative_to(root)) in inventory for p in required),'Source freeze omits union runner/data/report/tests/tree/launcher')
    for name,digest in inventory.items():
        path=(root/name).resolve();require(path.is_relative_to(root) and path.is_file() and sha256(path)==digest,'Frozen source changed: '+name)


def execute_study(args,prepared):
    from .draft_graph import ResidentDraftRuntime
    previous=ResidentDraftRuntime.MAX_ENTRIES;ResidentDraftRuntime.MAX_ENTRIES=128
    try:return _execute_study(args,prepared)
    finally:ResidentDraftRuntime.MAX_ENTRIES=previous


def _execute_study(args,prepared):
    from .baselines import BaselineEngine
    from .engine import RescueEngine
    from .loading import load_models
    from .packing import share_packed_projections
    from .query_mapper import mapper_from_payload
    from .specextend_qwen import build
    from .step_budget import StepBudget
    root=Path(__file__).resolve().parents[1];output=args.output.resolve();choices=prepared['choices'];documents=prepared['documents'];indices=sorted(documents)
    counts,seen=prepared['counts'],prepared['seen'];sources,assets,paths=prepared['sources'],prepared['assets'],prepared['asset_paths']
    expected=expected_counts(choices);total=scientific_requests(choices);m=validate_choices(choices);extra=extra_geometries(choices)
    sequence,owner_plan,warm_plan=timing_schedule(choices,indices)
    arms={str(n):[dict(mode=OUR_MODE,family='rescue',backend='cuda_graph')]+list(choices[str(n)].values()) for n in LENGTHS}
    lookup={n:{a['mode']:a for a in values} for n,values in arms.items()}
    inputs={(i,n):context(documents[i],n) for n in LENGTHS for i in indices}
    old=prepared['baseline_protocol'];horizon=prepared['horizon_protocol']
    prime_inputs,prime_refs=legacy.read_prime(args.prime_file,old,prepared['baseline_references'])
    torch.set_num_threads(4);torch.manual_seed(SEED);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    environment=gpu_snapshot()
    fields=('nvidia_smi_gpu','torch','cuda','target_dtype','draft_dtype','prefix_dtype','tf32')
    require(all(environment[k]==old['environment'][k]==horizon['environment'][k] for k in fields),'Same frozen GPU/runtime/precision as both development studies required')
    teacher,student,tokenizer=load_models(str(args.teacher),str(args.student),'cuda:0',target_dtype=torch.float32,efficient=True)
    for model,dtype in ((teacher,torch.float32),(student,torch.bfloat16)):
        model.eval().requires_grad_(False);require(all(p.device==torch.device('cuda:0') and p.dtype==dtype for p in model.parameters()),'Frozen model dtype/device changed')
    tick=time.perf_counter();template=build(teacher,student,str(args.teacher));template.prefill_precision='bfloat16'
    share_packed_projections(teacher,template.base_model);share_packed_projections(student,template.draft_model);preparation=time.perf_counter()-tick
    bindings=source_bindings(root);tree_bindings=baseline.check_full_bindings(root);flash=baseline.flash_receipt()
    require(flash==old['flash_attention']==horizon['flash_attention'],'Official Flash dependency differs from a completed study')
    mapper_payload=torch.load(args.mapper,map_location='cpu',weights_only=True);policy_payload=torch.load(args.policy,map_location='cpu',weights_only=True)
    cost=json.loads(args.cost.read_text());require(policy_payload.get('kind')=='step_budget' and 'seed' in policy_payload,'Frozen seeded causal StepBudget required')
    policy_metadata={k:policy_payload[k] for k in ('kind','hidden','margin','prior_weight','seed')}
    group=HookOwners(teacher,student);live=[]
    protocol=dict(prepared['metadata'],schema=1,seed=SEED,source_indices=indices,lengths=list(LENGTHS),repeats=2,
        arms=arms,arms_by_context=arms,modes_by_context={n:list(v) for n,v in lookup.items()},frozen_choices=choices,comparisons=m,
        configs={n:{a['mode']:asdict(legacy.own_config()) if a['family']=='rescue' else asdict(baseline.linear_config(a)) if a['family']=='retrieval' else baseline.tree_policy(a) for a in values} for n,values in arms.items()},
        expected_counts=expected,expected_requests=total,extra_eager_reference_geometries=extra,shared_warm_sequence=sequence,owner_order=owner_plan,warm_order=warm_plan,
        first_sentinel_source_index=indices[0],max_new_tokens=128,tokenizer_eos_token_id=tokenizer.eos_token_id,vocab_size=len(tokenizer),
        target_layers=teacher.config.num_hidden_layers,draft_layers=student.config.num_hidden_layers,source_sha256=sources,source_bindings=bindings,
        tree_source_bindings=tree_bindings,assets_sha256=assets,environment=environment,flash_attention=flash,
        mapper_sha256=MAPPER_SHA256,policy_sha256=POLICY_SHA256,cost_sha256=COST_SHA256,mapper_frozen=True,policy_frozen=True,no_training=True,
        policy_metadata=policy_metadata,model_preparation_seconds=preparation,template_prefill_precision='bfloat16',max_live_owners=1,graph_max_entries=128,
        input_sha256={f'{i}:{n}':hashlib.sha256(json.dumps(ids).encode()).hexdigest() for (i,n),ids in inputs.items()},
        prime_source_index=271,prime_input_sha256={str(n):hashlib.sha256(json.dumps(prime_inputs[n]).encode()).hexdigest() for n in LENGTHS},
        prime_reference_provenance='Verified stage1 canonical, same exact old271 inputs/models/tokenizer/precision; no new AR/free warmup',
        primary='Whole-generate outer of all32warm including every legal capture, prefill/reset/handoff/cleanup; constructor/cold/close separate',
        scope='Fresh fixed16doc confirmation versus every immutable stage1+stage2 union contender; no selection/replacement/capture filtering',
        args={k:str(v) for k,v in vars(args).items()})
    write(output/'protocol.json',protocol);write(output/'prime_references.json',prime_refs);progress('union_protocol_frozen',counts,comparisons=m,expected_requests=total)
    def make(arm,observer=None,own_backend='cuda_graph'):
        group.idle();require(not live and not group.engines,'Only one private serving owner may be live')
        if arm['family']=='full_tree':engine=baseline.FullOwner(template,args.teacher,arm)
        elif arm['family']=='retrieval':engine=BaselineEngine(teacher,student,baseline.linear_config(arm),vocab_size=len(tokenizer),top_chunks=arm['top_chunks'],round_observer=observer)
        else:
            mapper=mapper_from_payload(mapper_payload,'cuda:0').eval().requires_grad_(False)
            policy=StepBudget.from_payload(policy_payload,'cuda:0').eval().requires_grad_(False)
            require(all(getattr(policy,k)==policy_payload[k] for k in ('margin','prior_weight','seed')),'Policy settings changed')
            engine=RescueEngine(teacher,student,legacy.own_config(own_backend),mapper,policy,cost,vocab_size=len(tokenizer),round_observer=observer)
        if arm['family']!='full_tree':
            try:group.add(engine)
            except BaseException as error:
                try:detach_quality_observer(engine);engine.close()
                finally:append(output/'construction_failures.jsonl',dict(mode=arm['mode'],error=repr(error),engine_closed=engine.closed,logical_requests=len(seen)))
                raise
        live.append(engine)
        if observer is not None:observer.engine=engine
        return engine
    def close(engine):
        detach_quality_observer(engine)
        if isinstance(engine,baseline.FullOwner):group.idle();engine.close();group.idle()
        else:group.close(engine)
        live.remove(engine)
    def stats(engine):
        return engine.stats() if isinstance(engine,baseline.FullOwner) else dict(runtime_snapshot(engine),max_graph_entries=engine.draft_runtime.max_entries if engine.draft_runtime is not None else None)
    def signature(engine):return engine.signature() if isinstance(engine,baseline.FullOwner) else owner_signature(engine)
    def generate(engine,arm,ids,reference,request):
        gpu_snapshot();torch.manual_seed(policy_metadata['seed']);random.seed(policy_metadata['seed']);group.idle();tick=time.perf_counter()
        raw=engine.generate(ids) if arm['family']=='full_tree' else group.generate(engine,ids,tokenizer.eos_token_id);elapsed=time.perf_counter()-tick
        if arm['family']=='full_tree':
            raw['request_id']=uuid.uuid4().hex;append(output/'tree_raw.jsonl',dict(request_id=raw['request_id'],execution_index=len(seen),**request,outer_seconds=elapsed,result=raw))
            row=baseline.normalize_tree(raw,arm,reference,tokenizer.eos_token_id)
        else:row=raw
        row.update(request,arm=arm,execution_index=len(seen),output_agrees=row['generated']==reference,runtime_stats=stats(engine),owner_signature=signature(engine))
        return row,elapsed
    def check(row,reference,arm):
        if arm['family']=='rescue':check_own_output(row,reference,tokenizer.eos_token_id)
        else:baseline.check_output(row,reference,arm,tokenizer.eos_token_id)
        if arm.get('backend')=='cuda_graph':require(row['draft_runtime']['request']['replays']==row['draft_forwards']>0,'Graph replay/work mismatch')
    try:
        refs={}
        for n in LENGTHS:
            for i in indices:
                gpu_snapshot();group.idle();torch.manual_seed(policy_metadata['seed'])
                with BaselineEngine(teacher,student,replace(legacy.own_config('eager'),length_policy='fixed',inject_tokens=0,injection_strength=0.),vocab_size=len(tokenizer)) as engine:
                    row=engine.reference(inputs[i,n],tokenizer.eos_token_id)
                del engine
                row.update(request_id=uuid.uuid4().hex,mode='canonical_ar',source_index=i,length=n,phase='reference',execution_index=len(seen),performance_eligible=False)
                record(output,'reference.jsonl',row,counts,'reference_requests',seen);refs[f'{i}:{n}']=row['generated']
        write(output/'references.json',refs);quality_own={};quality_baseline={};matching={}
        def quality(arm,i,n,filename,counter,phase,own_backend='cuda_graph',targets=None):
            observer=None if arm['family']=='full_tree' else QualityObserver();engine=make(arm,observer,own_backend);owner_id=uuid.uuid4().hex
            try:
                request=dict(mode=arm['mode'],source_index=i,length=n,phase=phase,owner_id=owner_id,performance_eligible=False)
                if observer is None:row,_=generate(engine,arm,inputs[i,n],refs[f'{i}:{n}'],request)
                else:
                    with record_draft_steps() as steps:row,_=generate(engine,arm,inputs[i,n],refs[f'{i}:{n}'],request)
                    row.update(observed_rounds=observer.rounds,queries=steps,nonzero_injection=observer.nonzero_injection,native_t=native_digest(engine.native_t),native_s=native_digest(engine.native_s))
                row.update(equivalence_targets=targets or []);record(output,filename,row,counts,counter,seen)
                legacy.check_quality_row(row,refs[f'{i}:{n}'],arm['family']=='rescue',arm,tokenizer.eos_token_id);check(row,refs[f'{i}:{n}'],arm)
                if arm.get('backend')=='cuda_graph':require(row['draft_runtime']['request']['captures']>0,'Quality owner borrowed graphs')
                final=stats(engine);tick=time.perf_counter();close(engine);close_seconds=time.perf_counter()-tick
                require(engine.closed and stats(engine)['entries']==0,'Quality owner did not release graph/hook ownership')
                append(output/'quality_owners.jsonl',dict(owner_id=owner_id,mode=arm['mode'],length=n,source_index=i,request_id=row['request_id'],
                    owner_signature=row['owner_signature'],close_seconds=close_seconds,final_runtime_stats=final,closed_runtime_stats=stats(engine),engine_closed=True,hook_restored=True))
                progress('quality_request_complete',counts,mode=arm['mode'],source_index=i,length=n,phase=phase);return row
            finally:
                if engine in live:close(engine)
        for n in LENGTHS:
            for i in indices:
                eager=quality(dict(mode='rescue_eager',family='rescue',backend='eager'),i,n,'own_quality.jsonl','own_quality_requests','own_quality','eager')
                graph=quality(lookup[str(n)][OUR_MODE],i,n,'own_quality.jsonl','own_quality_requests','own_quality')
                legacy.own_quality_pair(eager,graph);quality_own[i,n]=graph
            for arm in choices[str(n)].values():
                row=quality(arm,indices[0],n,'baseline_quality.jsonl','baseline_quality_requests','baseline_sentinel');quality_baseline[arm['mode'],n]=row
                if arm['family']=='retrieval' and arm['backend']=='eager':matching[n,arm['top_chunks'],arm['fixed_k']]=row
            for geometry in (g for g in extra if g['length']==n):
                top,k=geometry['top_chunks'],geometry['fixed_k'];arm=dict(mode=f'matching_retrieval{top}_k{k}_eager',family='retrieval',backend='eager',top_chunks=top,fixed_k=k)
                targets=[a['mode'] for a in choices[str(n)].values() if a['family']=='retrieval' and (a['top_chunks'],a['fixed_k'])==(top,k)]
                matching[n,top,k]=quality(arm,indices[0],n,'matching_quality.jsonl','matching_quality_requests','matching_geometry_eager',targets=targets)
            for arm in choices[str(n)].values():
                if arm['family']=='retrieval' and arm['backend']!='eager':legacy.baseline_quality_pair(matching[n,arm['top_chunks'],arm['fixed_k']],quality_baseline[arm['mode'],n],arm)
        require(not live and not group.engines,'Quality owner survived into timing')
        write(output/'QUALITY_PASSED.json',dict(passed=True,own_all32_pairs=True,baseline_matching_geometry=True,actual_same_bfloat16_nonzero_mapper=True,
            quality_owners_closed=True,counts=dict(counts),comparisons=m,matching_geometry_count=len(extra)))
        progress('quality_gate_passed',counts);repeated={}
        require({str(p):sha256(p) for p in paths}==assets,'Frozen assets changed before timing')
        for entry in owner_plan:
            mode,n=entry['mode'],entry['length'];arm=lookup[str(n)][mode];gpu_snapshot();tick=time.perf_counter();engine=make(arm);constructor=time.perf_counter()-tick;owner_id=uuid.uuid4().hex
            owner=dict(mode=mode,arm=arm,length=n,owner_id=owner_id,owner_start=tick,constructor_and_detach_seconds=constructor,warm_tokens=0,warm_outer_seconds_sum=0.,warm_request_ids=[],maximum_live_owners=1)
            try:
                requests=[dict(mode=mode,length=n,source_index=271,phase='cold_prime')]+[dict(r,phase='warm',warm_index=j) for j,r in enumerate(x for x in warm_plan if x['mode']==mode and x['length']==n)]
                for item in requests:
                    cold=item['phase']=='cold_prime';i=item['source_index'];reference=prime_refs[str(n)] if cold else refs[f'{i}:{n}']
                    row,elapsed=generate(engine,arm,prime_inputs[n] if cold else inputs[i,n],reference,dict(item,owner_id=owner_id,performance_eligible=True))
                    row.update(outer_seconds=elapsed,outer_tps=row['tokens']/elapsed,observer_enabled=False,profile_enabled=False,
                        mapper_executed=arm['family']=='rescue',controller_executed=arm['family']=='rescue')
                    if cold:row.update(constructor_and_detach_seconds=constructor,cold_complete_outer_seconds=constructor+elapsed)
                    record(output,'prime.jsonl' if cold else 'runtime.jsonl',row,counts,'prime_requests' if cold else 'timing_requests',seen)
                    check(row,reference,arm);require(math.isfinite(elapsed) and elapsed>0 and 'timeline' not in row and 'cost_samples' not in row,'Invalid/profiled primary clock')
                    if cold:
                        owner.update(owner_signature=row['owner_signature'],prime_request_id=row['request_id'],prime_tokens=row['tokens'],cold_prime_outer_seconds=elapsed)
                        if arm.get('backend')=='cuda_graph':require(row['draft_runtime']['request']['captures']>0,'Timing owner borrowed quality graphs')
                    else:
                        require(row['owner_signature']==owner['owner_signature'],'Persistent private owner/cache pointers changed')
                        if arm['family']=='rescue':check_pair(quality_own[i,n],row)
                        else:
                            key=(arm['family'],arm['top_chunks'],arm.get('fixed_k',arm.get('nodes')),i,n)
                            if key in repeated:baseline.check_reproducible(repeated[key],row,arm)
                            else:repeated[key]=row
                            if i==indices[0]:baseline.check_reproducible(quality_baseline[mode,n],row,arm)
                        owner['warm_tokens']+=row['tokens'];owner['warm_outer_seconds_sum']+=elapsed;owner['warm_request_ids'].append(row['request_id'])
                    progress('cold_prime_complete' if cold else 'warm_complete',counts,**item)
                final=stats(engine);tick=time.perf_counter();close(engine);end=time.perf_counter();closed=stats(engine)
                require(engine.closed and closed['entries']==0,'Timing owner failed to release graphs/hooks')
                seconds=constructor+owner['cold_prime_outer_seconds']+owner['warm_outer_seconds_sum']+end-tick
                owner.update(close_seconds=end-tick,owner_lifetime_wall_seconds=end-owner.pop('owner_start'),final_runtime_stats=final,closed_runtime_stats=closed,
                    engine_closed=True,hook_restored=True,accounted_service_seconds=seconds,accounted_service_tokens=owner['prime_tokens']+owner['warm_tokens'],
                    accounted_service_tps=(owner['prime_tokens']+owner['warm_tokens'])/seconds)
                append(output/'owners.jsonl',owner);counts['owners']+=1;progress('owner_complete',counts,mode=mode,length=n)
            finally:
                if engine in live:close(engine)
        require(counts==expected and len(seen)==total,'Incomplete immutable union confirmation matrix')
        require(source_bindings(root)==bindings and baseline.check_full_bindings(root)==tree_bindings and baseline.flash_receipt()==flash and
            {str(p):sha256(p) for p in paths}==assets,'Frozen imports/dependency/assets changed')
        for name,digest in sources.items():require(sha256(root/name)==digest,'Frozen source changed: '+name)
        group.idle();gpu_snapshot()
        return dict(passed=True,counts=counts,expected_counts=expected,logical_requests=total,comparisons=m,matching_geometry_count=len(extra),
            sources_unchanged=True,assets_unchanged=True,dependencies_unchanged=True,quality_gate_passed=True,all_canonical=True,owners_closed=True,
            analysis_required_locally=True,goal_complete=False)
    finally:
        for engine in list(live):
            try:close(engine)
            except BaseException:detach_quality_observer(engine);engine.close()


def copy_development_evidence(arguments,output,name):
    """Copy a verified manifest and all raw/source/figure bytes unchanged."""
    require(name in ('stage1_evidence','stage2_evidence'),'Unregistered portable evidence root')
    source_root=Path(arguments['final_manifest']).resolve().parent;destination=output/name;index={};relative_paths={}
    manifest=json.loads(Path(arguments['final_manifest']).read_text())
    for relative,digest in manifest['file_sha256'].items():
        part=Path(relative);require(not part.is_absolute() and '..' not in part.parts,'Unsafe development artifact path')
        source=(source_root/part).resolve();require(source.is_relative_to(source_root) and sha256(source)==digest,'Development artifact changed while copying')
        target=destination/part;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        require(sha256(target)==digest,'Portable development artifact copy differs')
        index[str(source)]=dict(saved_path=str(target.relative_to(output)),sha256=digest)
    for key in ('baseline_results','analysis_dir','verified_complete','final_manifest'):
        source=Path(arguments[key]).resolve();require(source.is_relative_to(source_root),'Development proof escapes manifest root')
        relative=source.relative_to(source_root);relative_paths[key]=str(relative)
        if source.is_file():
            target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copyfile(source,target)
            digest=sha256(source);require(sha256(target)==digest,'Portable development proof differs')
            index[str(source)]=dict(saved_path=str(target.relative_to(output)),sha256=digest)
    write(destination/'PIN_INDEX.json',dict(original_to_saved=index))
    return dict(root=name,pin_index=name+'/PIN_INDEX.json',pin_index_sha256=sha256(destination/'PIN_INDEX.json'),**relative_paths)


def run(args):
    from .fair_union_confirmation_data import validate_source_freeze,verify_dataset_manifest
    root=Path(__file__).resolve().parents[1];output=args.output.resolve()
    require(not output.exists(),'Fresh isolated union confirmation output only; never resume/reselect')
    for name,value in vars(args).items():require(isinstance(value,Path) and value.is_absolute(),'All paths must be explicit absolute paths: '+name)
    require(not any(output==p or output.is_relative_to(p) or p.is_relative_to(output) for p in (args.teacher.resolve(),args.student.resolve())),
        'Output overlaps frozen models')
    output.mkdir(parents=True);counts={};seen=set()
    try:
        # In particular, no stat/hash/open of fresh dataset/manifest before this
        # helper verifies BOTH complete studies, figures, union and final source.
        gate,_=validate_source_freeze(args.freeze_receipt)
        require(Path(gate['source_root']).resolve()==root,'Frozen source root differs from runtime tree')
        validate_minimum_sources(root,gate['source_sha256'])
        proof=gate['union_freeze'];choices=copy.deepcopy(proof['choices']);m=validate_choices(choices)
        expected=expected_counts(choices);total=scientific_requests(choices);extra=extra_geometries(choices)
        require(proof.get('status')=='UNION_VERIFIED' and proof['M']==m and proof['R']==len(extra) and
            proof['extra_eager_reference_geometries']==extra and proof['expected_counts']==expected and proof['scientific_requests']==total,
            'Frozen variable-M request matrix/geometries differ')
        counts.update(dict.fromkeys(expected,0))
        write(output/'GATE_PASSED.json',dict(passed=True,union_freeze=proof,source_freeze_sha256=sha256(args.freeze_receipt),
            expected_counts=expected,expected_requests=total,comparisons=m,extra_eager_reference_geometries=extra,confirmation_content_opened=False))
        stage1_evidence=copy_development_evidence(gate['baseline_arguments'],output,'stage1_evidence')
        stage2_evidence=copy_development_evidence(gate['horizon_arguments'],output,'stage2_evidence')
        require(args.dataset_manifest.resolve()==args.data_file.resolve().parent/'DATASET_MANIFEST.json','Prepared data/manifest roots differ')
        manifest=verify_dataset_manifest(args.dataset_manifest.parent)
        require(args.data_file.resolve()==args.dataset_manifest.resolve().parent/manifest['dataset_file'],'Explicit prepared dataset path differs')
        require(manifest['union_freeze']==proof and manifest['source_freeze_sha256']==sha256(args.freeze_receipt) and manifest['source_sha256']==gate['source_sha256'],
            'Fresh cohort is not bound to this exact dual-study/source gate')
        documents={r['source_index']:r for r in (json.loads(line) for line in args.data_file.read_text().splitlines() if line.strip())}
        require(len(documents)==16 and sorted(documents)==[r['source_index'] for r in manifest['selected']],'Prepared16 identities/order changed')
        shutil.copytree(args.dataset_manifest.parent,output/'data');verify_dataset_manifest(output/'data')
        old_root=Path(gate['baseline_arguments']['baseline_results']);horizon_root=Path(gate['horizon_arguments']['baseline_results'])
        old=json.loads((old_root/'protocol.json').read_text());old_refs=json.loads((old_root/'references.json').read_text())
        horizon=json.loads((horizon_root/'protocol.json').read_text())
        require(frozen_choices(json.loads((horizon_root/'FROZEN_SELECTION.json').read_text()))==choices,'Retained union changed after freeze')
        require(sha256(args.mapper)==MAPPER_SHA256 and sha256(args.policy)==POLICY_SHA256 and sha256(args.cost)==COST_SHA256,
            'Mandatory real nonzero Mapper/frozen StepBudget/old cost changed')
        paths=sorted(set(model_files(args.teacher)+model_files(args.student)+[args.mapper,args.policy,args.cost,args.prime_file,
            args.freeze_receipt,args.dataset_manifest,args.data_file,Path(gate['final_protocol']['path']),Path(gate['exclusion_receipt']['path'])]))
        assets={str(p):sha256(p) for p in paths}
        for path in model_files(args.teacher)+model_files(args.student)+[args.prime_file]:
            require(old['assets_sha256'].get(str(path))==assets[str(path)]==horizon['assets_sha256'].get(str(path)),
                'Models/tokenizer/old prime differ from a verified development study')
        require(Path(gate['tokenizer_root']).resolve()==args.teacher.resolve(),'Prepared tokenizer is not frozen Target tokenizer')
        for name,digest in gate['tokenizer_sha256'].items():require(sha256(args.teacher/name)==digest,'Frozen tokenizer changed')
        for name,digest in gate['source_sha256'].items():
            target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,target)
            require(sha256(target)==digest,'Source snapshot copy differs')
        prepared=dict(choices=choices,documents=documents,counts=counts,seen=seen,sources=gate['source_sha256'],asset_paths=paths,assets=assets,
            baseline_protocol=old,horizon_protocol=horizon,baseline_references=old_refs,
            metadata=dict(stage1_evidence=stage1_evidence,stage2_evidence=stage2_evidence,union_freeze=proof,
                source_freeze_sha256=sha256(args.freeze_receipt),dataset_manifest_sha256=sha256(args.dataset_manifest),dataset_sha256=manifest['dataset_sha256'],
                dataset_manifest='data/DATASET_MANIFEST.json',portable_data_root='data',selection_rule=manifest['selection_rule'],
                final_protocol_sha256=gate['final_protocol']['sha256']))
        completion=execute_study(args,prepared)
        validate_source_freeze(args.freeze_receipt);verify_dataset_manifest(args.dataset_manifest.parent);verify_dataset_manifest(output/'data')
        write(output/'COMPLETE.json',dict(completion,data_unchanged=True,dual_study_freeze_unchanged=True))
        progress('union_confirmation_execution_complete',counts)
    except BaseException as error:
        write(output/'FAILURE.json',dict(passed=False,error_type=type(error).__name__,error=str(error),counts=counts,logical_requests=len(seen),
            scope='Preserve partial union confirmation; no candidate/data replacement, tuning, resume or silent restart'))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('teacher','student','mapper','policy','cost','freeze-receipt','dataset-manifest','data-file','prime-file','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())


if __name__=='__main__':main()
