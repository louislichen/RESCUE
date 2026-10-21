"""Fresh16-document confirmation of a frozen nonzero RESCUE against8 choices.

This runner never selects an algorithm or a document by observed performance.
The strict completed baseline, rendered inspection and final source/data gate
must be authenticated before even hashing the new confirmation dataset.
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
from .draft_cost_recalibration import (QualityObserver,detach_quality_observer,record_draft_steps)
from .draft_graph_benchmark import HookOwners,owner_signature,record,check_output as check_own_output
from .draft_graph_smoke import COST_SHA256,native_digest,runtime_snapshot
from .round_overhead_run import (MAPPER_SHA256,POLICY_SHA256,append,check_pair,context,
    gpu_snapshot,model_files,require,sha256,write)

SEED=20261024
LENGTHS=(4096,8192)
REPEATS=2
OUR_MODE='rescue_graph'
PRIME_SHA256='85b2eb1a81d1617b2d158c069c420c452459a8bdb90da128f257220cc27ded09'
GROUPS=('retrieval/eager','retrieval/resident','retrieval/graph','full_tree')


def progress(stage,counts,**extra):
    print(json.dumps(dict(stage=stage,counts=dict(counts),**extra)),flush=True)


def frozen_choices(selection):
    """Only eight formally frozen developer choices; never re-rank here."""
    registered={a['mode']:a for a in baseline.arms()};choices={}
    for n in LENGTHS:
        groups=selection.get('per_context',{}).get(str(n),{}).get('groups',{})
        require(set(groups)==set(GROUPS),'Missing one of four frozen baseline groups')
        choices[str(n)]={}
        for group in GROUPS:
            value=groups[group];mode=value.get('mode');arm=value.get('arm')
            require(mode in registered and arm==registered[mode],'Frozen baseline not in original38-arm menu')
            actual_group='full_tree' if arm['family']=='full_tree' else 'retrieval/'+('graph' if arm['backend']=='cuda_graph' else arm['backend'])
            require(actual_group==group,'Frozen baseline group/backend mismatch')
            choices[str(n)][group]=copy.deepcopy(arm)
    return choices


def extra_geometries(choices):
    result=[]
    for n in LENGTHS:
        groups=choices[str(n)];eager=groups['retrieval/eager'];seen={(eager['top_chunks'],eager['fixed_k'])}
        for group in ('retrieval/resident','retrieval/graph'):
            arm=groups[group];geometry=arm['top_chunks'],arm['fixed_k']
            if geometry not in seen:
                result.append(dict(length=n,top_chunks=geometry[0],fixed_k=geometry[1]));seen.add(geometry)
    require(0<=len(result)<=4,'Invalid deterministic matching-reference count')
    return result


def expected_counts(choices):
    r=len(extra_geometries(choices))
    return dict(reference_requests=32,own_quality_requests=64,baseline_quality_requests=8,
        matching_quality_requests=r,prime_requests=10,timing_requests=320,owners=10)


def timing_schedule(choices,documents,seed=SEED):
    require(len(documents)==16 and len(set(documents))==16,'Exactly16 fresh registered documents required')
    owners=[];warm=[]
    for n in LENGTHS:
        rng=random.Random(seed+n);modes=[OUR_MODE]+[choices[str(n)][g]['mode'] for g in GROUPS];rng.shuffle(modes)
        require(len(set(modes))==5,'Timing modes duplicated')
        for mode in modes:
            owners.append(dict(mode=mode,length=n))
            for repeat in range(REPEATS):
                order=list(documents);rng.shuffle(order)
                warm.extend(dict(mode=mode,length=n,source_index=i,repeat=repeat) for i in order)
    return owners,warm


def own_config(mode='cuda_graph'):
    from .config import SystemConfig
    return SystemConfig(selector='hierarchical',coverage_rows=64,refresh_every=4,
        refresh_policy='periodic',injection_layers=(27,),inject_tokens=64,injection_strength=.5,
        prefill_precision='bfloat16',max_new_tokens=128,max_draft=12,length_policy='risk',
        schedule='serial',profile=False,draft_execution=mode,coverage_execution='eager')


def own_quality_pair(eager,graph):
    check_pair(eager,graph)
    for name in ('observed_rounds','queries','native_t','native_s'):
        require(eager.get(name) and eager[name]==graph.get(name),'Own eager/Graph changed '+name)
    require(eager.get('nonzero_injection') is True and graph.get('nonzero_injection') is True,
            'Own frozen Mapper must produce a real same-dtype working-V change')


def baseline_quality_pair(eager,row,arm):
    baseline.check_reproducible(eager,row,arm)
    for name in ('observed_rounds','queries','native_t','native_s'):
        require(eager.get(name) and eager[name]==row.get(name),'Matching retrieval geometry changed '+name)
    require(eager.get('nonzero_injection') is False and row.get('nonzero_injection') is False,
            'Baseline unexpectedly injected Mapper content')


def read_prime(path,baseline_protocol,baseline_references):
    require(sha256(path)==PRIME_SHA256,'Old271 prime file changed')
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    matches=[r for r in rows if r.get('source_index')==271]
    require(len(matches)==1 and matches[0].get('split')=='train','Fixed old271 prime must not be replaced')
    item=matches[0];inputs={};refs={}
    for n in LENGTHS:
        ids=context(item,n);digest=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        require(baseline_protocol.get('input_sha256',{}).get(f'271:{n}')==digest,'Old271 prime input differs from verified baseline AR')
        reference=baseline_references.get(f'271:{n}')
        require(isinstance(reference,list) and len(reference)==128,'Reuse exact old271 canonical128 output, no free new reference')
        inputs[n]=ids;refs[str(n)]=reference
    return inputs,refs


def check_quality_row(row,reference,own,arm,eos):
    if own:check_own_output(row,reference,eos)
    else:baseline.check_output(row,reference,arm,eos)
    if own or arm['family']=='retrieval':
        require(len(row.get('queries',[]))==row['draft_forwards'],'Missing every actual Draft Query digest')
        require(row.get('native_t') and row.get('native_s'),'Missing full effective Native KV digests')
        for observed in row['observed_rounds']:
            vector=observed.get('budget_features')
            require(vector is None or len(vector)==12 and all(math.isfinite(v) for v in vector),'Invalid actual M3 feature vector')
        if own:require(row.get('nonzero_injection') is True,'Frozen nonzero Mapper made no actual same-BF16 V change')
        else:
            require(row.get('nonzero_injection') is False,'Retrieval baseline injected content')
            require(all(not q['summary_requested'] and q['summary_count']==0 for q in row['queries']),
                    'Baseline quality accidentally enabled M3 summaries')


def source_bindings(root):
    receipt=baseline.source_bindings(root)
    for name in ('rescue.fair_confirmation','rescue.fair_confirmation_data','rescue.query_mapper','rescue.step_budget',
                 'rescue.selection','rescue.injection','rescue.features','rescue.attention_summary'):
        module=importlib.import_module(name);path=Path(module.__file__).resolve();expected=root/Path(*name.split('.')).with_suffix('.py')
        require(path==expected,'Confirmation imported a foreign source namespace: '+name)
        receipt[name]=dict(path=str(path),sha256=sha256(path))
    return receipt


def execute_study(args,prepared):
    """GPU work starts only after the caller authenticates both freeze/data gates."""
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

    output=args.output.resolve();choices=prepared['choices'];documents=prepared['documents']
    indices=sorted(documents);counts=prepared['counts'];seen=prepared['seen'];expected=expected_counts(choices)
    source_root=Path(__file__).resolve().parents[1];sources=prepared['sources'];asset_paths=prepared['asset_paths'];assets=prepared['assets']
    inputs={(i,n):context(documents[i],n) for n in LENGTHS for i in indices}
    owner_plan,warm_plan=timing_schedule(choices,indices)
    descriptors={str(n):[dict(mode=OUR_MODE,family='rescue',backend='cuda_graph')]+[choices[str(n)][g] for g in GROUPS] for n in LENGTHS}
    lookups={str(n):{a['mode']:a for a in descriptors[str(n)]} for n in LENGTHS}
    extra=extra_geometries(choices)
    old_inputs,old_refs=read_prime(args.prime_file,prepared['baseline_protocol'],prepared['baseline_references'])
    torch.set_num_threads(4);torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    environment=gpu_snapshot()
    require(environment['nvidia_smi_gpu']==prepared['baseline_protocol']['environment']['nvidia_smi_gpu'],
            'Confirmation must use the same frozen physical GPU as baseline selection')
    require(all(environment[k]==prepared['baseline_protocol']['environment'][k] for k in ('torch','cuda','target_dtype','draft_dtype','prefix_dtype','tf32')),
            'Frozen precision/runtime environment changed')
    teacher,student,tokenizer=load_models(str(args.teacher),str(args.student),'cuda:0',target_dtype=torch.float32,efficient=True)
    for model,dtype in ((teacher,torch.float32),(student,torch.bfloat16)):
        model.eval().requires_grad_(False)
        require(all(p.device==torch.device('cuda:0') and p.dtype==dtype for p in model.parameters()),'Frozen model dtype/device changed')
    tick=time.perf_counter();template=build(teacher,student,str(args.teacher));template.prefill_precision='bfloat16'
    share_packed_projections(teacher,template.base_model);share_packed_projections(student,template.draft_model)
    preparation_seconds=time.perf_counter()-tick
    bindings=source_bindings(source_root);tree_bindings=baseline.check_full_bindings(source_root);flash=baseline.flash_receipt()
    require(flash==prepared['baseline_protocol']['flash_attention'],'FlashAttention binary/Python dependency differs from verified baseline')
    group=HookOwners(teacher,student);live=[]
    mapper_payload=torch.load(args.mapper,map_location='cpu',weights_only=True)
    policy_payload=torch.load(args.policy,map_location='cpu',weights_only=True);cost=json.loads(args.cost.read_text())
    require(policy_payload.get('kind')=='step_budget' and 'seed' in policy_payload,'Frozen causal StepBudget payload required')
    policy_metadata={k:policy_payload[k] for k in ('kind','hidden','margin','prior_weight','seed')}
    metadata=dict(prepared['metadata'],schema=1,seed=SEED,source_indices=indices,lengths=list(LENGTHS),repeats=2,
        arms=descriptors,arms_by_context=descriptors,modes_by_context={n:[a['mode'] for a in v] for n,v in descriptors.items()},
        configs={str(n):{a['mode']:asdict(own_config()) if a['family']=='rescue' else asdict(baseline.linear_config(a)) if a['family']=='retrieval' else baseline.tree_policy(a) for a in descriptors[str(n)]} for n in LENGTHS},
        expected_counts=expected,expected_requests=434+len(extra),extra_eager_reference_geometries=extra,
        max_new_tokens=128,owner_order=owner_plan,warm_order=warm_plan,first_sentinel_source_index=indices[0],
        tokenizer_eos_token_id=tokenizer.eos_token_id,vocab_size=len(tokenizer),
        target_layers=teacher.config.num_hidden_layers,draft_layers=student.config.num_hidden_layers,
        source_sha256=sources,source_bindings=bindings,tree_source_bindings=tree_bindings,assets_sha256=assets,
        environment=environment,flash_attention=flash,mapper_sha256=MAPPER_SHA256,policy_sha256=POLICY_SHA256,cost_sha256=COST_SHA256,
        mapper_frozen=True,policy_frozen=True,no_training=True,policy_metadata=policy_metadata,
        model_preparation_seconds=preparation_seconds,template_prefill_precision='bfloat16',max_live_owners=1,
        graph_max_entries=128,
        input_sha256={f'{i}:{n}':hashlib.sha256(json.dumps(ids).encode()).hexdigest() for (i,n),ids in inputs.items()},
        prime_source_index=271,prime_input_sha256={str(n):hashlib.sha256(json.dumps(old_inputs[n]).encode()).hexdigest() for n in LENGTHS},
        prime_reference_provenance='Exact same verified baseline old271 input/weights/tokenizer/precision; no extra AR or free warmup',
        primary='32warm per mode/context full-generate outer including every legal capture/prefill/reset/hook handoff/cleanup; private constructor/coldprime/close separately reported',
        scope='One fixed fresh16-document confirmation, all8 frozen challengers retained; no test-aware tuning, replacement or censoring',
        args={k:str(v) for k,v in vars(args).items()})
    write(output/'protocol.json',metadata);write(output/'prime_references.json',old_refs)
    progress('confirmation_protocol_frozen',counts,expected_requests=434+len(extra))

    def make(arm,observer=None,own_backend='cuda_graph'):
        group.idle();require(not live and not group.engines,'Only one serving owner may be alive')
        def register(engine):
            try:group.add(engine)
            except BaseException as error:
                try:detach_quality_observer(engine);engine.close()
                finally:append(output/'construction_failures.jsonl',dict(mode=arm['mode'],error=repr(error),engine_closed=engine.closed,logical_requests=len(seen)))
                raise
        if arm['family']=='full_tree':engine=baseline.FullOwner(template,args.teacher,arm)
        elif arm['family']=='retrieval':
            engine=BaselineEngine(teacher,student,baseline.linear_config(arm),vocab_size=len(tokenizer),top_chunks=arm['top_chunks'],round_observer=observer)
            register(engine)
        else:
            mapper=mapper_from_payload(mapper_payload,'cuda:0').eval().requires_grad_(False)
            policy=StepBudget.from_payload(policy_payload,'cuda:0').eval().requires_grad_(False)
            require(policy.margin==policy_payload['margin'] and policy.prior_weight==policy_payload['prior_weight'] and policy.seed==policy_payload['seed'],'Policy hyperparameters changed')
            engine=RescueEngine(teacher,student,own_config(own_backend),mapper,policy,cost,vocab_size=len(tokenizer),round_observer=observer)
            register(engine)
        live.append(engine)
        if observer is not None:observer.engine=engine
        return engine
    def close(engine):
        detach_quality_observer(engine)
        if isinstance(engine,baseline.FullOwner):group.idle();engine.close();group.idle()
        else:group.close(engine)
        live.remove(engine)
    def stats(engine):
        if isinstance(engine,baseline.FullOwner):return engine.stats()
        return dict(runtime_snapshot(engine),max_graph_entries=engine.draft_runtime.max_entries if engine.draft_runtime is not None else None)
    def signature(engine):return engine.signature() if isinstance(engine,baseline.FullOwner) else owner_signature(engine)
    def generate(engine,arm,ids,reference,request):
        gpu_snapshot();torch.manual_seed(policy_metadata['seed']);random.seed(policy_metadata['seed']);group.idle()
        start=time.perf_counter()
        raw=engine.generate(ids) if arm['family']=='full_tree' else group.generate(engine,ids,tokenizer.eos_token_id)
        elapsed=time.perf_counter()-start
        if arm['family']=='full_tree':
            raw['request_id']=uuid.uuid4().hex
            append(output/'tree_raw.jsonl',dict(request_id=raw['request_id'],execution_index=len(seen),**request,outer_seconds=elapsed,result=raw))
            row=baseline.normalize_tree(raw,arm,reference,tokenizer.eos_token_id)
        else:row=raw
        row.update(request,arm=arm,execution_index=len(seen),output_agrees=row['generated']==reference,
            runtime_stats=stats(engine),owner_signature=signature(engine))
        return row,elapsed
    def check(row,reference,arm):
        if arm['family']=='rescue':check_own_output(row,reference,tokenizer.eos_token_id)
        else:baseline.check_output(row,reference,arm,tokenizer.eos_token_id)
        if arm.get('backend')=='cuda_graph':
            require(row['draft_runtime']['request']['replays']==row['draft_forwards']>0,'Actual graph replay work differs')

    try:
        refs={}
        for n in LENGTHS:
            for i in indices:
                gpu_snapshot();group.idle();torch.manual_seed(policy_metadata['seed'])
                with BaselineEngine(teacher,student,replace(own_config('eager'),length_policy='fixed',inject_tokens=0,injection_strength=0.),vocab_size=len(tokenizer)) as engine:
                    row=engine.reference(inputs[i,n],tokenizer.eos_token_id)
                del engine
                row.update(request_id=uuid.uuid4().hex,mode='canonical_ar',source_index=i,length=n,phase='reference',execution_index=len(seen),performance_eligible=False)
                record(output,'reference.jsonl',row,counts,'reference_requests',seen);refs[f'{i}:{n}']=row['generated']
        write(output/'references.json',refs);quality_own={};quality_baseline={};matching={}
        def quality(arm,i,n,filename,counter,phase,own_backend='cuda_graph',targets=None):
            observer=None if arm['family']=='full_tree' else QualityObserver();engine=make(arm,observer,own_backend)
            owner_id=uuid.uuid4().hex
            try:
                request=dict(mode=arm['mode'],source_index=i,length=n,phase=phase,owner_id=owner_id,performance_eligible=False)
                if observer is None:row,_=generate(engine,arm,inputs[i,n],refs[f'{i}:{n}'],request)
                else:
                    with record_draft_steps() as steps:row,_=generate(engine,arm,inputs[i,n],refs[f'{i}:{n}'],request)
                    row.update(observed_rounds=observer.rounds,queries=steps,nonzero_injection=observer.nonzero_injection,
                        native_t=native_digest(engine.native_t),native_s=native_digest(engine.native_s))
                row.update(equivalence_targets=targets or [])
                record(output,filename,row,counts,counter,seen)
                check_quality_row(row,refs[f'{i}:{n}'],arm['family']=='rescue',arm,tokenizer.eos_token_id)
                check(row,refs[f'{i}:{n}'],arm)
                if arm.get('backend')=='cuda_graph':require(row['draft_runtime']['request']['captures']>0,'Quality owner borrowed graphs')
                final=stats(engine);close_start=time.perf_counter();close(engine);close_seconds=time.perf_counter()-close_start
                require(engine.closed and stats(engine)['entries']==0,'Quality owner did not release graphs/hooks')
                append(output/'quality_owners.jsonl',dict(owner_id=owner_id,mode=arm['mode'],length=n,source_index=i,
                    request_id=row['request_id'],owner_signature=row['owner_signature'],close_seconds=close_seconds,
                    final_runtime_stats=final,closed_runtime_stats=stats(engine),engine_closed=engine.closed,hook_restored=True))
                progress('quality_request_complete',counts,mode=arm['mode'],source_index=i,length=n,phase=phase)
                return row
            finally:
                if engine in live:close(engine)
        for n in LENGTHS:
            for i in indices:
                eager_arm=dict(mode='rescue_eager',family='rescue',backend='eager')
                eager=quality(eager_arm,i,n,'own_quality.jsonl','own_quality_requests','own_quality','eager')
                graph=quality(lookups[str(n)][OUR_MODE],i,n,'own_quality.jsonl','own_quality_requests','own_quality')
                own_quality_pair(eager,graph);quality_own[i,n]=graph
            for group_name in GROUPS:
                arm=choices[str(n)][group_name]
                row=quality(arm,indices[0],n,'baseline_quality.jsonl','baseline_quality_requests','baseline_sentinel')
                quality_baseline[arm['mode'],n]=row
                if group_name=='retrieval/eager':matching[n,arm['top_chunks'],arm['fixed_k']]=row
            for geometry in (g for g in extra if g['length']==n):
                top,k=geometry['top_chunks'],geometry['fixed_k']
                arm=dict(mode=f'matching_retrieval{top}_k{k}_eager',family='retrieval',backend='eager',top_chunks=top,fixed_k=k)
                targets=[a['mode'] for a in choices[str(n)].values() if a['family']=='retrieval' and a['top_chunks']==top and a['fixed_k']==k]
                matching[n,top,k]=quality(arm,indices[0],n,'matching_quality.jsonl','matching_quality_requests','matching_geometry_eager',targets=targets)
            for group_name in ('retrieval/resident','retrieval/graph'):
                arm=choices[str(n)][group_name]
                baseline_quality_pair(matching[n,arm['top_chunks'],arm['fixed_k']],quality_baseline[arm['mode'],n],arm)
        require(not live and not group.engines,'Quality owners survive into timing')
        write(output/'QUALITY_PASSED.json',dict(passed=True,own_all32_pairs=True,baseline_matching_geometry=True,
            actual_same_bfloat16_nonzero_mapper=True,quality_owners_closed=True,counts=dict(counts)))
        progress('quality_gate_passed',counts)
        repeated={}
        for entry in owner_plan:
            mode,n=entry['mode'],entry['length'];arm=lookups[str(n)][mode]
            gpu_snapshot();tick=time.perf_counter();engine=make(arm);constructor=time.perf_counter()-tick;owner_id=uuid.uuid4().hex
            owner=dict(mode=mode,arm=arm,length=n,owner_id=owner_id,owner_start=tick,constructor_and_detach_seconds=constructor,
                warm_tokens=0,warm_outer_seconds_sum=0.,warm_request_ids=[],maximum_live_owners=1)
            try:
                requests=[dict(mode=mode,length=n,source_index=271,phase='cold_prime')]+[dict(r,phase='warm') for r in warm_plan if r['mode']==mode and r['length']==n]
                for item in requests:
                    cold=item['phase']=='cold_prime';i=item['source_index'];reference=old_refs[str(n)] if cold else refs[f'{i}:{n}']
                    row,elapsed=generate(engine,arm,old_inputs[n] if cold else inputs[i,n],reference,dict(item,owner_id=owner_id,performance_eligible=True))
                    row.update(outer_seconds=elapsed,outer_tps=row['tokens']/elapsed,observer_enabled=False,profile_enabled=False)
                    row.update(mapper_executed=arm['family']=='rescue',controller_executed=arm['family']=='rescue')
                    if cold:row.update(constructor_and_detach_seconds=constructor,cold_complete_outer_seconds=constructor+elapsed)
                    record(output,'prime.jsonl' if cold else 'runtime.jsonl',row,counts,'prime_requests' if cold else 'timing_requests',seen)
                    check(row,reference,arm);require('timeline' not in row and 'cost_samples' not in row,'Profile contaminated primary clock')
                    if cold:
                        owner.update(owner_signature=row['owner_signature'],prime_request_id=row['request_id'],prime_tokens=row['tokens'],cold_prime_outer_seconds=elapsed)
                        if arm.get('backend')=='cuda_graph':require(row['draft_runtime']['request']['captures']>0,'Fresh timing owner borrowed quality graphs')
                    else:
                        require(row['owner_signature']==owner['owner_signature'],'Private owner/cache pointers changed')
                        if arm['family']=='rescue':check_pair(quality_own[i,n],row)
                        else:
                            key=mode,i,n
                            if key in repeated:baseline.check_reproducible(repeated[key],row,arm)
                            else:repeated[key]=row
                            if i==indices[0]:baseline.check_reproducible(quality_baseline[mode,n],row,arm)
                        owner['warm_tokens']+=row['tokens'];owner['warm_outer_seconds_sum']+=elapsed;owner['warm_request_ids'].append(row['request_id'])
                    progress('cold_prime_complete' if cold else 'warm_complete',counts,**item)
                final=stats(engine);start=time.perf_counter();close(engine);end=time.perf_counter()
                seconds=constructor+owner['cold_prime_outer_seconds']+owner['warm_outer_seconds_sum']+end-start
                owner.update(close_seconds=end-start,owner_lifetime_wall_seconds=end-owner.pop('owner_start'),final_runtime_stats=final,
                    closed_runtime_stats=stats(engine),engine_closed=engine.closed,hook_restored=True,accounted_service_seconds=seconds,
                    accounted_service_tokens=owner['prime_tokens']+owner['warm_tokens'],accounted_service_tps=(owner['prime_tokens']+owner['warm_tokens'])/seconds)
                require(engine.closed and stats(engine)['entries']==0,'Timing owner did not release graph/cache ownership')
                append(output/'owners.jsonl',owner);counts['owners']+=1;progress('owner_complete',counts,mode=mode,length=n)
            finally:
                if engine in live:close(engine)
        require(counts==expected and len(seen)==434+len(extra),'Incomplete frozen confirmation matrix')
        require(source_bindings(source_root)==bindings and baseline.check_full_bindings(source_root)==tree_bindings and
            baseline.flash_receipt()==flash and {str(p):sha256(p) for p in asset_paths}==assets,'Frozen sources/dependency/assets changed')
        for name,digest in sources.items():require(sha256(source_root/name)==digest,'Frozen source changed: '+name)
        group.idle();gpu_snapshot()
        return dict(passed=True,counts=counts,expected_counts=expected,logical_requests=len(seen),
            sources_unchanged=True,assets_unchanged=True,quality_gate_passed=True,all_canonical=True,owners_closed=True,
            analysis_required_locally=True,goal_complete=False)
    finally:
        for engine in list(live):
            try:close(engine)
            except BaseException:detach_quality_observer(engine);engine.close()


def validate_minimum_sources(root,inventory):
    root=root.resolve();tree=list((root/'exp/specextend_bridgekv/specextend').rglob('*.py'))
    require(len(tree)>=27,'Complete archived tree source must precede confirmation access')
    required=list((root/'rescue').glob('*.py'))+tree+[
        root/'exp/attention_horizon/selection.py',root/'exp/selective_kv/bridge.py',
        root/'tests/test_rescue_fair_confirmation.py',root/'tests/test_rescue_fair_confirmation_data.py',
        root/'tests/test_rescue_fair_confirmation_report.py',
        root/'exp/draft_graph/fair_confirmation_20261020_preregistration.json',
        root/'exp/draft_graph/run_fair_confirmation_20261020_v1.sh']
    require(all(p.is_file() and str(p.relative_to(root)) in inventory for p in required),
            'Source freeze omits mandatory runner/report/test/tree/launcher dependencies')
    for name,digest in inventory.items():
        path=(root/name).resolve();require(path.is_relative_to(root) and path.is_file() and sha256(path)==digest,'Source freeze changed: '+name)


def copy_baseline_evidence(gate,output):
    arguments=gate['baseline_arguments'];root=Path(arguments['final_manifest']).resolve().parent
    manifest=json.loads(Path(arguments['final_manifest']).read_text());destination=output/'baseline_evidence';pins={}
    for name,digest in manifest['file_sha256'].items():
        source=(root/name).resolve();require(source.is_relative_to(root) and sha256(source)==digest,'Baseline artifact changed while copying')
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        require(sha256(target)==digest,'Portable baseline copy changed bytes')
        pins[str(source)]=dict(saved_path=str(target.relative_to(output)),sha256=digest)
    paths={}
    for key in ('baseline_results','analysis_dir','verified_complete','final_manifest'):
        path=Path(arguments[key]).resolve();relative=path.relative_to(root)
        paths[key]=str(relative)
        if path.is_file():
            target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copyfile(path,target)
            pins[str(path)]=dict(saved_path=str(target.relative_to(output)),sha256=sha256(target))
    write(destination/'PIN_INDEX.json',dict(original_to_saved=pins))
    return dict(root='baseline_evidence',pin_index='baseline_evidence/PIN_INDEX.json',pin_index_sha256=sha256(destination/'PIN_INDEX.json'),**paths)


def run(args):
    from .fair_confirmation_data import validate_source_freeze,verify_dataset_manifest
    root=Path(__file__).resolve().parents[1];output=args.output.resolve()
    require(not output.exists(),'Fresh confirmation output only; never resume/reselect')
    for name in ('teacher','student','freeze_receipt','dataset_manifest','data_file','prime_file','mapper','policy','cost'):
        require(getattr(args,name).is_absolute(),'All confirmation asset paths must be explicit absolute paths')
    require(not output.is_relative_to(args.teacher.resolve()) and not output.is_relative_to(args.student.resolve()),'Output overlaps model assets')
    output.mkdir(parents=True);counts={};seen=set()
    try:
        # Absolutely no stat/hash/open of new dataset/manifest before this gate.
        gate,_=validate_source_freeze(args.freeze_receipt)
        require(Path(gate['source_root']).resolve()==root,'Frozen source root differs from actual runtime tree')
        validate_minimum_sources(root,gate['source_sha256'])
        proof=gate['baseline_freeze'];choices=proof['choices']
        require(extra_geometries(choices)==proof['extra_eager_reference_geometries'],'R differs from frozen geometry-only references')
        counts.update(dict.fromkeys(expected_counts(choices),0))
        require(434+len(extra_geometries(choices))==proof['scientific_requests'],'Frozen request count changed')
        write(output/'GATE_PASSED.json',dict(passed=True,source_freeze_sha256=sha256(args.freeze_receipt),baseline_freeze=proof,
            expected_counts=expected_counts(choices),expected_requests=proof['scientific_requests'],confirmation_content_opened=False))
        baseline_evidence=copy_baseline_evidence(gate,output)
        # The first new-dataset read in this runner happens strictly after gate.
        require(args.dataset_manifest.resolve()==args.data_file.resolve().parent/'DATASET_MANIFEST.json','Dataset and manifest roots differ')
        manifest=verify_dataset_manifest(args.dataset_manifest.parent)
        require(args.data_file.resolve()==args.dataset_manifest.resolve().parent/manifest['dataset_file'],'Explicit dataset path differs')
        require(manifest['source_freeze_sha256']==sha256(args.freeze_receipt) and manifest['source_sha256']==gate['source_sha256'],
                'Prepared cohort is not bound to this exact final source freeze')
        documents={r['source_index']:r for r in (json.loads(line) for line in args.data_file.read_text().splitlines() if line.strip())}
        require(len(documents)==16 and sorted(documents)==[r['source_index'] for r in manifest['selected']],'Prepared16 document order changed')
        shutil.copytree(args.dataset_manifest.parent,output/'data')
        verify_dataset_manifest(output/'data')
        baseline_results=Path(gate['baseline_arguments']['baseline_results'])
        baseline_protocol=json.loads((baseline_results/'protocol.json').read_text())
        baseline_references=json.loads((baseline_results/'references.json').read_text())
        require(sha256(args.mapper)==MAPPER_SHA256 and sha256(args.policy)==POLICY_SHA256 and sha256(args.cost)==COST_SHA256,
                'Mandatory nonzero Mapper/old-cost StepBudget assets changed')
        paths=sorted(set(model_files(args.teacher)+model_files(args.student)+[
            args.mapper,args.policy,args.cost,args.prime_file,args.freeze_receipt,args.dataset_manifest,args.data_file,
            Path(gate['final_protocol']['path']),Path(gate['exclusion_receipt']['path'])]))
        assets={str(p):sha256(p) for p in paths}
        for path in model_files(args.teacher)+model_files(args.student)+[args.prime_file]:
            require(baseline_protocol['assets_sha256'].get(str(path))==assets[str(path)],'Models/tokenizer/old prime not identical to verified baseline canonical source')
        require(Path(gate['tokenizer_root']).resolve()==args.teacher.resolve(),'Prepared tokenizer is not the frozen Target tokenizer')
        for name,digest in gate['tokenizer_sha256'].items():require(sha256(args.teacher/name)==digest,'Prepared tokenizer changed')
        for name,digest in gate['source_sha256'].items():
            target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,target)
        prepared=dict(choices=choices,documents=documents,counts=counts,seen=seen,sources=gate['source_sha256'],
            asset_paths=paths,assets=assets,baseline_protocol=baseline_protocol,baseline_references=baseline_references,
            metadata=dict(baseline_evidence=baseline_evidence,baseline_freeze=proof,source_freeze_sha256=sha256(args.freeze_receipt),
                dataset_manifest_sha256=sha256(args.dataset_manifest),dataset_sha256=manifest['dataset_sha256'],
                dataset_manifest='data/DATASET_MANIFEST.json',selection_rule=manifest['selection_rule'],
                frozen_choices=choices,portable_data_root='data',final_protocol_sha256=gate['final_protocol']['sha256']))
        completion=execute_study(args,prepared)
        validate_source_freeze(args.freeze_receipt);verify_dataset_manifest(args.dataset_manifest.parent)
        write(output/'COMPLETE.json',completion)
        progress('confirmation_execution_complete',counts)
    except BaseException as error:
        write(output/'FAILURE.json',dict(passed=False,error_type=type(error).__name__,error=str(error),counts=counts,
            logical_requests=len(seen),scope='Preserve partial confirmation; never replace data, alter frozen choices or silently restart'))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('teacher','student','mapper','policy','cost','freeze-receipt','dataset-manifest','data-file','prime-file','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())


if __name__=='__main__':main()
