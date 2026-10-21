"""2540-request,32-warm horizon-matched baseline development selection.

Only predeclared, already-used data are permitted. Stage1's six-warm choices
remain intact; stage2 freezes their deduplicated union with32-warm winners.
No new confirmation data, Mapper or length-policy search is performed.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import time
import uuid

import torch

from . import baseline_selection as base
from .draft_graph_benchmark import HookOwners,owner_signature,record
from .draft_graph_smoke import runtime_snapshot
from .round_overhead_run import append,context,gpu_snapshot,model_files,require,sha256,write

SEED=20261025
LENGTHS=(4096,8192)
REPEATS=2
GROUPS=('retrieval/eager','retrieval/resident','retrieval/graph','full_tree')
EXPECTED=dict(reference_requests=32,prime_requests=76,timing_requests=2432,owners=76)
PREREGISTRATION='exp/draft_graph/baseline_horizon_20261020_preregistration.json'
PIPELINE='exp/draft_graph/run_baseline_horizon_20261020_v1.sh'
SOURCE_NAMES=dict(train_file='exp/long_system/data/train.jsonl',
    opt_confirmation_file='exp/module_opt/data_v1_20261018/confirmation.jsonl',
    arch_confirmation_file='exp/module_arch/data_v1_20261018/confirmation.jsonl',
    prime_file='exp/module_opt/data_v1_20261018/tune.jsonl')
FUTURE_ANCHOR=dict(mapper_sha256='d08889e796ca2527e23241ff34a072ef640e2fdf3fe9a2f9ba5439975af04d9f',
    policy_sha256='3c333a470119c236afcc1308c18592842a535bee0b8f83c4fb4da359ff17eff5',
    cost_sha256='d92c843a84676c0cd6adaf9364bbf7e8c771d56548f1016b5720a3e04f371fa1',
    draft_execution='cuda_graph',coverage_execution='eager',injection_layers=[27],inject_tokens=64,injection_strength=.5)


def progress(stage,counts,**fields):
    print(json.dumps(dict(stage=stage,counts=dict(counts),**fields)),flush=True)


def schedule(documents,seed=SEED):
    require(len(documents)==16 and len(set(documents))==16,'Exactly16 locked used documents required')
    rng=random.Random(seed);sequence=[]
    for repeat in range(REPEATS):
        indices=list(documents);rng.shuffle(indices)
        sequence.extend(dict(source_index=i,repeat=repeat) for i in indices)
    owners=[];warm=[]
    for n in LENGTHS:
        modes=[a['mode'] for a in base.arms()];random.Random(seed+n).shuffle(modes)
        for mode in modes:
            owners.append(dict(mode=mode,length=n))
            warm.extend(dict(mode=mode,length=n,**item) for item in sequence)
    return sequence,owners,warm


def group_of(arm):
    return 'full_tree' if arm['family']=='full_tree' else 'retrieval/'+('graph' if arm['backend']=='cuda_graph' else arm['backend'])


def select_winners(rows,stage1,descriptors=None):
    descriptors=base.arms() if descriptors is None else descriptors;lookup={a['mode']:a for a in descriptors};result={}
    for n in LENGTHS:
        candidates=[]
        for mode,arm in lookup.items():
            samples=[r for r in rows if r.get('mode')==mode and r.get('length')==n]
            require(len(samples)==32 and len({(r.get('source_index'),r.get('repeat')) for r in samples})==32 and
                {r.get('repeat') for r in samples}=={0,1} and len({r.get('source_index') for r in samples})==16,
                'Cannot freeze incomplete32-warm baseline cell')
            require(all(r.get('phase')=='warm' and r.get('performance_eligible') is True and
                type(r.get('outer_seconds')) in (int,float) and math.isfinite(r['outer_seconds']) and r['outer_seconds']>0 for r in samples),
                'Winner requires unprofiled positive outer clocks, no capture filtering')
            def pooled(values):
                tokens=sum(r['tokens'] for r in values);seconds=sum(r['outer_seconds'] for r in values)
                return dict(tokens=tokens,seconds=seconds,tps=tokens/seconds,
                    captures=sum(r.get('draft_runtime',{}).get('request',{}).get('captures',0) for r in values),
                    capture_seconds=sum(r.get('draft_runtime',{}).get('request',{}).get('capture_host_seconds',0.) for r in values))
            metric=pooled(samples)
            candidates.append(dict(mode=mode,arm=copy.deepcopy(arm),warm_requests=32,warm_tokens=metric['tokens'],
                warm_seconds=metric['seconds'],warm_tps=metric['tps'],request_ids=[r['request_id'] for r in samples],
                prefix6=pooled(samples[:6]),prefix32=metric))
        def best(values):return min(values,key=lambda v:(-v['warm_tps'],v['mode']))
        groups={g:best([r for r in candidates if group_of(r['arm'])==g]) for g in GROUPS}
        original=stage1.get('per_context',{}).get(str(n),{}).get('groups',{})
        require(set(original)==set(GROUPS),'Stage1 four original choices must remain intact')
        union={};modes=[]
        for group in GROUPS:
            require(original[group].get('arm')==lookup.get(original[group].get('mode')) and
                group_of(original[group]['arm'])==group,'Original stage1 candidate changed')
            union[group]=[copy.deepcopy(original[group])]
            if original[group]['mode']!=groups[group]['mode']:union[group].append(copy.deepcopy(groups[group]))
            for item in union[group]:
                if item['mode'] not in modes:modes.append(item['mode'])
        result[str(n)]=dict(groups=groups,stage1_groups=copy.deepcopy(original),union_by_group=union,union_modes=modes,
            ranking=sorted(candidates,key=lambda r:(-r['warm_tps'],r['mode'])),overall=best(candidates),
            overall_family_winner={family:best([r for r in candidates if r['arm']['family']==family]) for family in ('retrieval','full_tree')})
    return dict(per_context=result,stage2_group_candidates=8,
        union_candidates=sum(len(v['union_modes']) for v in result.values()),
        selection_score='SUM client tokens / SUM full-generate outer seconds of ALL32 warm requests',
        exact_tie_break='lexicographically smallest mode, exact float equality only',seed=SEED,
        diagnostics='First6 requests of the SAME32 sequence, not stage1 original3docs x2; never substitute for primary32',
        scope='Reused16-document development selection; preserve stage1+stage2 challengers, not independent confirmation')


def source_inventory(root,stage1_protocol):
    tree=root/'exp/specextend_bridgekv/specextend';tree_files=list(tree.rglob('*.py'))
    require(len(tree_files)>=27,'Complete unchanged upstream tree source required')
    paths=list((root/'rescue').glob('*.py'))+list((root/'tests').glob('test_rescue*.py'))+tree_files+[
        root/'exp/attention_horizon/selection.py',root/'exp/selective_kv/bridge.py',root/PREREGISTRATION,root/PIPELINE,
        root/'rescue/baseline_horizon_data.py',root/'rescue/baseline_horizon_report.py',
        root/'tests/test_rescue_baseline_horizon.py',root/'tests/test_rescue_baseline_horizon_data.py',root/'tests/test_rescue_baseline_horizon_report.py']
    require(all(p.is_file() for p in paths),'Missing final horizon runner/data/report/tests/preregistration/launcher')
    inventory={str(p.relative_to(root)):sha256(p) for p in sorted(set(paths))}
    previous=stage1_protocol['source_sha256']
    for name,digest in previous.items():
        if name.startswith('rescue/') or name in ('exp/attention_horizon/selection.py','exp/selective_kv/bridge.py'):
            require(inventory.get(name)==digest,'A shared stage1 algorithm source is missing or changed: '+name)
    require({p:sha for p,sha in inventory.items() if p.startswith('exp/specextend_bridgekv/specextend/')}==
            {p:sha for p,sha in previous.items() if p.startswith('exp/specextend_bridgekv/specextend/')},'Tree source differs from stage1')
    return inventory


def source_bindings(root):
    receipt=base.source_bindings(root)
    for name in ('rescue.baseline_horizon','rescue.baseline_horizon_data'):
        module=importlib.import_module(name);actual=Path(module.__file__).resolve();expected=root/Path(*name.split('.')).with_suffix('.py')
        require(actual==expected,'Horizon imported a foreign module: '+name);receipt[name]=dict(path=str(actual),sha256=sha256(actual))
    return receipt


def validate_preregistration(prereg):
    """Fail before data/model loading if the predeclared horizon was altered."""
    from . import baseline_horizon_data as data
    require(prereg.get('schema')==1 and prereg.get('status')=='FROZEN_BEFORE_EXECUTION' and
        prereg.get('seed')==SEED,'Wrong frozen horizon protocol')
    require(all(prereg.get(k) is True for k in ('no_training','no_mapper_or_policy_search',
        'no_new_confirmation_access','paused_test_untouched')),'Horizon safety scope changed')
    common=prereg.get('common',{});expected=dict(teacher='Qwen2-7B-Instruct',student='Qwen2-1.5B-Instruct',input_tokens=list(LENGTHS),max_new_tokens=128,
        chunk_size=32,refresh_every=4,mapper=False,controller=False,profile=False,observer=False,
        graph_max_entries=128,target_dtype='float32',student_dtype='bfloat16',target_prefill_precision='bfloat16',tf32=False)
    require(all(common.get(k)==v for k,v in expected.items()),'Frozen execution configuration changed')
    d=prereg.get('data',{})
    require(d.get('metadata_source_sha256')==data.SOURCE_SHA256 and d.get('train_indices')==list(data.TRAIN_INDICES) and
        d.get('old_opt_confirmation_indices')==list(data.OPT_INDICES) and d.get('old_arch_confirmation_indices')==[604] and
        d.get('documents')==16 and d.get('repeats')==2,'Frozen old-data cohort changed')
    execution=prereg.get('execution',{})
    require(all(execution.get(k)==v for k,v in EXPECTED.items()) and
        execution.get('total_scientific_requests')==2540 and execution.get('upstream_tree_raw_records')==528,
        'Frozen request matrix changed')
    arms=prereg.get('arms',{});retrieval=arms.get('retrieval',{});tree=arms.get('full_tree',{})
    require(arms.get('total')==38 and retrieval==dict(top_chunks=[16,32],fixed_k=[1,2,4,8,12],backends=['eager','resident','graph'],count=30) and
        all(tree.get(k)==v for k,v in dict(top_chunks=[16,32],nodes=[8,32,50,64],max_depth=10,threshold=.7,count=8).items()) and
        prereg.get('selection',{}).get('groups')==list(GROUPS),
        'Frozen baseline groups changed')
    require(all(prereg.get('frozen_rescue_anchor_for_later_confirmation',{}).get(k)==v for k,v in FUTURE_ANCHOR.items()),
        'Frozen future method anchor changed during baseline selection')
    return True


def gpu_telemetry():
    """Read-only drift diagnostics; never change clocks/power or filter blocks."""
    fields=('uuid','temperature.gpu','clocks.current.sm','clocks.current.memory',
        'power.draw','power.limit','utilization.gpu','memory.used','memory.total')
    result=subprocess.run(['nvidia-smi','--query-gpu='+','.join(fields),'--format=csv,noheader,nounits'],
        check=True,text=True,capture_output=True)
    lines=result.stdout.strip().splitlines();require(len(lines)==1,'Telemetry requires exactly one physical GPU')
    values=[part.strip() for part in lines[0].split(',')];require(len(values)==len(fields),'Malformed GPU telemetry')
    return dict(query_fields=list(fields),values=dict(zip(fields,values)),raw_csv=result.stdout,
        wall_time_unix=time.time(),read_only=True,primary_clock_included=False)


def copy_stage1_evidence(args,output):
    """Preserve every verified artifact byte, with a portable absolute-pin map."""
    original=Path(args.stage1_manifest).resolve();root=original.parent
    manifest=json.loads(original.read_text());destination=output/'stage1_evidence';pins={};paths={}
    for name,digest in manifest['file_sha256'].items():
        relative=Path(name);require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe stage1 artifact path')
        source=(root/relative).resolve();require(source.is_relative_to(root) and sha256(source)==digest,'Stage1 artifact changed during copy')
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        require(sha256(target)==digest,'Stage1 portable artifact copy differs')
        pins[str(source)]=dict(saved_path=str(target.relative_to(output)),sha256=digest)
    for key,attribute in (('baseline_results','stage1_results'),('analysis_dir','stage1_analysis'),
                          ('verified_complete','stage1_verified'),('final_manifest','stage1_manifest')):
        source=getattr(args,attribute).resolve();require(source.is_relative_to(root),'Stage1 proof escapes manifest root')
        relative=source.relative_to(root);paths[key]=str(relative)
        if source.is_file():
            target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copyfile(source,target)
            digest=sha256(source);require(sha256(target)==digest,'Stage1 proof copy differs')
            pins[str(source)]=dict(saved_path=str(target.relative_to(output)),sha256=digest)
    write(destination/'PIN_INDEX.json',dict(original_to_saved=pins))
    return dict(root='stage1_evidence',pin_index='stage1_evidence/PIN_INDEX.json',
        pin_index_sha256=sha256(destination/'PIN_INDEX.json'),**paths)


def run(args):
    from .baseline_horizon_data import load_horizon_documents,validate_horizon_manifest,verify_horizon_sources
    from .fair_confirmation_data import validate_baseline_freeze,_tokenizer_inventory
    root=Path(__file__).resolve().parents[1];output=args.output.resolve()
    require(not output.exists(),'Fresh isolated horizon output only; never resume/reselect')
    for name,value in vars(args).items():require(isinstance(value,Path) and value.is_absolute(),'All paths must be explicit absolute paths: '+name)
    protected=[args.teacher.resolve(),args.student.resolve(),args.stage1_manifest.resolve().parent]
    require(not any(output==p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected),
        'New output overlaps model/stage1 evidence')
    output.mkdir(parents=True);counts=dict.fromkeys(EXPECTED,0);seen=set()
    try:
        # No cohort file is statted, hashed or parsed before the authoritative stage1 gate.
        proof=validate_baseline_freeze(args.stage1_results,args.stage1_analysis,args.stage1_verified,args.stage1_manifest)
        old=json.loads((args.stage1_results/'protocol.json').read_text())
        stage1=json.loads((args.stage1_results/'FROZEN_SELECTION.json').read_text())
        old_refs=json.loads((args.stage1_results/'references.json').read_text())
        prereg=json.loads((root/PREREGISTRATION).read_text());validate_preregistration(prereg)
        require(sha256(args.stage1_results/'FROZEN_SELECTION.json')==proof['selection_freeze_sha256'],
            'Stage1 choices changed after gate')
        require(old.get('arms')==base.arms(),'Stage1 baseline menu differs; do not substitute candidates')
        sources=source_inventory(root,old);bindings=source_bindings(root)
        metadata_paths={relative:getattr(args,attribute).resolve() for attribute,relative in SOURCE_NAMES.items()}
        paths=sorted(set(model_files(args.teacher)+model_files(args.student)+list(metadata_paths.values())))
        assets={str(p):sha256(p) for p in paths}
        for path in model_files(args.teacher)+model_files(args.student)+[args.prime_file.resolve()]:
            require(old['assets_sha256'].get(str(path))==assets[str(path)],'Frozen models/tokenizer/prime differ from stage1')
        tokenizer_inventory=_tokenizer_inventory(args.teacher)
        documents,prime,manifest=load_horizon_documents(metadata_paths,
            expected_source_sha256=prereg['data']['metadata_source_sha256'],expected_train_indices=prereg['data']['train_indices'],
            tokenizer_sha256=tokenizer_inventory)
        validate_horizon_manifest(manifest,documents,prime)
        copied_paths={}
        for name,path in metadata_paths.items():
            target=output/'data'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
            require(sha256(target)==manifest['source_sha256'][name],'Copied old metadata bytes changed')
            copied_paths[name]=target
        write(output/'data/DATASET_MANIFEST.json',manifest)
        for row in documents.values():append(output/'data/documents.jsonl',row)
        write(output/'data/prime.json',prime)
        verify_horizon_sources(manifest,copied_paths)
        portable=copy_stage1_evidence(args,output)
        portable_root=output/portable['root']
        require(validate_baseline_freeze(*(portable_root/portable[k] for k in
            ('baseline_results','analysis_dir','verified_complete','final_manifest')))==proof,'Portable stage1 chain differs')
        for name,digest in sources.items():
            target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,target)
            require(sha256(target)==digest,'Frozen source copy differs')
        metadata=dict(preregistration=PREREGISTRATION,preregistration_sha256=sources[PREREGISTRATION],
            stage1_evidence=portable,stage1_freeze=proof,stage1_selection_sha256=proof['selection_freeze_sha256'],
            data_manifest='data/DATASET_MANIFEST.json',data_manifest_sha256=sha256(output/'data/DATASET_MANIFEST.json'),
            data_source_paths={name:str(path.relative_to(output)) for name,path in copied_paths.items()},
            selected_documents='data/documents.jsonl',prime_document='data/prime.json',
            selected_documents_sha256=sha256(output/'data/documents.jsonl'),prime_document_sha256=sha256(output/'data/prime.json'),
            data_receipt_sha256=manifest['receipt_sha256'],tokenizer_sha256=tokenizer_inventory,
            frozen_rescue_anchor_for_later_confirmation=prereg['frozen_rescue_anchor_for_later_confirmation'],
            no_new_confirmation_access=True,paused_test_untouched=True)
        write(output/'GATE_PASSED.json',dict(passed=True,stage1_freeze=proof,source_sha256=sources,
            data_manifest_sha256=metadata['data_manifest_sha256'],old_metadata_sources=manifest['source_sha256'],
            expected_counts=EXPECTED,expected_requests=2540,no_new_confirmation_opened=True))
        require(source_bindings(root)==bindings,'Imports changed before model generation')
        prepared=dict(documents=documents,prime=prime,sources=sources,assets=assets,asset_paths=paths,
            counts=counts,seen=seen,stage1_protocol=old,stage1_selection=stage1,stage1_references=old_refs,metadata=metadata)
        completion=execute_study(args,prepared)
        require(validate_baseline_freeze(args.stage1_results,args.stage1_analysis,args.stage1_verified,args.stage1_manifest)==proof,
            'Stage1 proof changed during horizon experiment')
        verify_horizon_sources(manifest,metadata_paths);verify_horizon_sources(manifest,copied_paths)
        require({str(p):sha256(p) for p in paths}==assets and source_bindings(root)==bindings,'Frozen assets/imports changed after generation')
        for name,digest in sources.items():require(sha256(root/name)==digest,'Frozen source changed after generation: '+name)
        require(_tokenizer_inventory(args.teacher)==tokenizer_inventory,'Tokenizer changed after generation')
        write(output/'COMPLETE.json',dict(completion,data_unchanged=True,stage1_verified_unchanged=True,
            data_manifest_sha256=metadata['data_manifest_sha256'],stage1_selection_sha256=proof['selection_freeze_sha256']))
        progress('horizon_execution_complete',counts)
    except BaseException as error:
        write(output/'FAILURE.json',dict(passed=False,error_type=type(error).__name__,error=str(error),counts=counts,
            logical_requests=len(seen),scope='Preserve partial horizon study; never replace data/candidates or silently restart'))
        raise


def execute_study(args,prepared):
    from .baselines import BaselineEngine
    from .loading import load_models
    from .packing import share_packed_projections
    from .specextend_qwen import build
    output=args.output.resolve();root=Path(__file__).resolve().parents[1];documents=prepared['documents'];indices=sorted(documents)
    sources=prepared['sources'];assets=prepared['assets'];paths=prepared['asset_paths'];counts=prepared['counts'];seen=prepared['seen']
    old=prepared['stage1_protocol'];stage1=prepared['stage1_selection'];descriptors=base.arms();lookup={a['mode']:a for a in descriptors}
    sequence,owner_plan,warm_plan=schedule(indices)
    inputs={(i,n):context(documents[i],n) for n in LENGTHS for i in indices};prime={n:context(prepared['prime'],n) for n in LENGTHS}
    prime_refs={str(n):prepared['stage1_references'][f'271:{n}'] for n in LENGTHS}
    for n in LENGTHS:
        require(hashlib.sha256(json.dumps(prime[n]).encode()).hexdigest()==old['input_sha256'][f'271:{n}'] and len(prime_refs[str(n)])==128,
                'Old271 prime input or canonical128 output changed')
    torch.set_num_threads(4);torch.manual_seed(SEED);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    environment=gpu_snapshot()
    require(environment['nvidia_smi_gpu']==old['environment']['nvidia_smi_gpu'] and all(environment[k]==old['environment'][k] for k in
        ('torch','cuda','target_dtype','draft_dtype','prefix_dtype','tf32')),'Same physical GPU and precision/runtime as stage1 required')
    teacher,student,tokenizer=load_models(str(args.teacher),str(args.student),'cuda:0',target_dtype=torch.float32,efficient=True)
    for model,dtype in ((teacher,torch.float32),(student,torch.bfloat16)):
        model.eval().requires_grad_(False)
        require(all(p.device==torch.device('cuda:0') and p.dtype==dtype for p in model.parameters()),'Single-GPU frozen dtype changed')
    tick=time.perf_counter();template=build(teacher,student,str(args.teacher));template.prefill_precision='bfloat16'
    share_packed_projections(teacher,template.base_model);share_packed_projections(student,template.draft_model)
    preparation_seconds=time.perf_counter()-tick;group=HookOwners(teacher,student);active=None
    bindings=source_bindings(root);tree_bindings=base.check_full_bindings(root);flash=base.flash_receipt()
    require(flash==old['flash_attention'],'Official Flash binary/Python receipt changed from stage1')
    protocol=dict(prepared['metadata'],schema=1,seed=SEED,source_indices=indices,lengths=list(LENGTHS),repeats=2,
        max_new_tokens=128,arms=descriptors,modes=[a['mode'] for a in descriptors],
        configs={a['mode']:asdict(base.linear_config(a)) if a['family']=='retrieval' else base.tree_policy(a) for a in descriptors},
        expected_counts=EXPECTED,expected_requests=2540,shared_warm_sequence=sequence,owner_order=owner_plan,warm_order=warm_plan,
        source_sha256=sources,source_bindings=bindings,tree_source_bindings=tree_bindings,assets_sha256=assets,environment=environment,
        flash_attention=flash,tokenizer_eos_token_id=tokenizer.eos_token_id,vocab_size=len(tokenizer),graph_max_entries=128,
        no_training=True,mapper_executed=False,controller_executed=False,observer=False,profile=False,max_live_owners=1,
        retrieval_retain_query=False,template_prefill_precision='bfloat16',model_preparation_seconds=preparation_seconds,
        input_sha256={f'{i}:{n}':hashlib.sha256(json.dumps(ids).encode()).hexdigest() for (i,n),ids in inputs.items()},
        document_sha256={str(i):documents[i]['document_sha256'] for i in indices},prime_source_index=271,
        prime_input_sha256={str(n):hashlib.sha256(json.dumps(prime[n]).encode()).hexdigest() for n in LENGTHS},
        prime_in_warm_cohort=271 in indices,prime_overlap_scope='Intentional old271 cold prime may also appear in the already-used16doc warm cohort, identically for all38 configs; not fresh confirmation',
        selection_score='ALL32 warm client tokens / ALL32 full-generate outer seconds; exact float max with lexical ties',
        prefix6_diagnostic='First6 of the shared32 sequence, NOT the original stage1 three-doc x2 cohort; secondary only',
        primary_clock='Whole generate outer, including all lazy captures, prefill/reset/input/handoff/cleanup; common model loading excluded uniformly',
        gpu_diagnostics='Read-only temperature, SM/memory clocks, power, utilization and memory at each owner start/end;152 records outside generate/constructor/close clocks, never filter slow blocks',
        scope='Already-used16doc development; horizon-matched selection and union retention only, no new confirmation/paused outcomes',
        args={k:str(v) for k,v in vars(args).items()})
    write(output/'protocol.json',protocol);write(output/'prime_references.json',prime_refs);progress('protocol_frozen',counts,expected_requests=2540)
    def stats(engine):
        if isinstance(engine,base.FullOwner):return engine.stats()
        return dict(runtime_snapshot(engine),max_graph_entries=engine.draft_runtime.max_entries if engine.draft_runtime is not None else None)
    def close(engine):
        if isinstance(engine,base.FullOwner):group.idle();engine.close();group.idle()
        else:group.close(engine)
    refs={};warm_rows=[];trajectories={}
    try:
        for n in LENGTHS:
            for i in indices:
                gpu_snapshot();group.idle();torch.manual_seed(SEED);random.seed(SEED)
                with BaselineEngine(teacher,student,base.linear_config(descriptors[0]),vocab_size=len(tokenizer)) as engine:
                    row=engine.reference(inputs[i,n],tokenizer.eos_token_id)
                del engine
                row.update(request_id=uuid.uuid4().hex,mode='canonical_ar',source_index=i,length=n,phase='reference',execution_index=len(seen),performance_eligible=False)
                record(output,'reference.jsonl',row,counts,'reference_requests',seen)
                require(0<len(row['generated'])<=128,'Invalid canonical output')
                if i==271:require(row['generated']==prime_refs[str(n)],'Independent271 AR differs from frozen stage1 canonical')
                refs[f'{i}:{n}']=row['generated'];progress('reference_complete',counts,source_index=i,length=n)
        write(output/'references.json',refs)
        require({str(p):sha256(p) for p in paths}==assets,'Frozen data/models changed before timing')
        for entry in owner_plan:
            mode,n=entry['mode'],entry['length'];arm=lookup[mode]
            gpu_snapshot();group.idle();require(active is None and not group.engines,'At most one serving owner is allowed')
            owner_id=uuid.uuid4().hex;telemetry=gpu_telemetry()
            append(output/'gpu_diagnostics.jsonl',dict(phase='owner_start',owner_id=owner_id,mode=mode,length=n,
                execution_index=len(seen),telemetry=telemetry))
            tick=time.perf_counter()
            if arm['family']=='full_tree':active=base.FullOwner(template,args.teacher,arm)
            else:
                active=BaselineEngine(teacher,student,base.linear_config(arm),vocab_size=len(tokenizer),top_chunks=arm['top_chunks'])
                try:group.add(active)
                except BaseException:active.close();raise
            constructor=time.perf_counter()-tick
            owner=dict(mode=mode,arm=arm,length=n,owner_id=owner_id,owner_start=tick,constructor_and_detach_seconds=constructor,
                warm_tokens=0,warm_outer_seconds_sum=0.,warm_request_ids=[],maximum_live_owners=1)
            try:
                requests=[dict(mode=mode,length=n,source_index=271,phase='cold_prime')]+[
                    dict(item,phase='warm',warm_index=index) for index,item in enumerate(warm_plan_item for warm_plan_item in warm_plan if warm_plan_item['mode']==mode and warm_plan_item['length']==n)]
                for request in requests:
                    cold=request['phase']=='cold_prime';i=request['source_index'];reference=prime_refs[str(n)] if cold else refs[f'{i}:{n}']
                    gpu_snapshot();group.idle();torch.manual_seed(SEED);random.seed(SEED);start=time.perf_counter()
                    raw=active.generate(prime[n] if cold else inputs[i,n]) if arm['family']=='full_tree' else group.generate(active,prime[n] if cold else inputs[i,n],tokenizer.eos_token_id)
                    elapsed=time.perf_counter()-start
                    if arm['family']=='full_tree':
                        raw['request_id']=uuid.uuid4().hex
                        append(output/'tree_raw.jsonl',dict(request_id=raw['request_id'],execution_index=len(seen),owner_id=owner_id,**request,outer_seconds=elapsed,result=raw))
                        row=base.normalize_tree(raw,arm,reference,tokenizer.eos_token_id);signature=active.signature()
                    else:row=raw;signature=owner_signature(active)
                    row.update(request,arm=arm,execution_index=len(seen),owner_id=owner_id,output_agrees=row['generated']==reference,
                        outer_seconds=elapsed,outer_tps=row['tokens']/elapsed,runtime_stats=stats(active),owner_signature=signature,
                        performance_eligible=True,observer_enabled=False,profile_enabled=False,mapper_executed=False,controller_executed=False)
                    if cold:row.update(constructor_and_detach_seconds=constructor,cold_complete_outer_seconds=constructor+elapsed)
                    record(output,'prime.jsonl' if cold else 'runtime.jsonl',row,counts,'prime_requests' if cold else 'timing_requests',seen)
                    base.check_output(row,reference,arm,tokenizer.eos_token_id)
                    require(math.isfinite(elapsed) and elapsed>0 and 'timeline' not in row and 'cost_samples' not in row,'Invalid/profile-contaminated primary clock')
                    key=(arm['family'],arm['top_chunks'],arm.get('fixed_k',arm.get('nodes')),i,n)
                    if key in trajectories:base.check_reproducible(trajectories[key],row,arm)
                    else:trajectories[key]=row
                    if arm['family']=='retrieval' and arm['backend']=='cuda_graph':
                        delta=row['draft_runtime']['request'];require(delta['replays']==row['draft_forwards']>0,'Actual Graph replay/work mismatch')
                        if cold:require(delta['captures']>0,'Fresh owner borrowed graphs')
                        require(active.draft_runtime.max_entries==128 and not active.draft_runtime.retain_query,'Wrong actual graph limit or unused Query work')
                    if cold:owner.update(owner_signature=signature,prime_request_id=row['request_id'],prime_tokens=row['tokens'],cold_prime_outer_seconds=elapsed)
                    else:
                        require(signature==owner['owner_signature'],'Persistent owner/cache pointers changed')
                        warm_rows.append(row);owner['warm_tokens']+=row['tokens'];owner['warm_outer_seconds_sum']+=elapsed;owner['warm_request_ids'].append(row['request_id'])
                    progress('cold_prime_complete' if cold else 'warm_complete',counts,**request)
                final=stats(active);start=time.perf_counter();close(active);end=time.perf_counter();closed=stats(active)
                require(active.closed and closed['entries']==0,'Owner failed to release graphs/hooks')
                seconds=constructor+owner['cold_prime_outer_seconds']+owner['warm_outer_seconds_sum']+end-start
                owner.update(final_runtime_stats=final,closed_runtime_stats=closed,engine_closed=True,hook_restored=True,
                    close_seconds=end-start,owner_lifetime_wall_seconds=end-owner.pop('owner_start'),accounted_service_seconds=seconds,
                    accounted_service_tokens=owner['prime_tokens']+owner['warm_tokens'],accounted_service_tps=(owner['prime_tokens']+owner['warm_tokens'])/seconds)
                append(output/'owners.jsonl',owner);counts['owners']+=1;active=None;group.idle();progress('owner_complete',counts,mode=mode,length=n)
                append(output/'gpu_diagnostics.jsonl',dict(phase='owner_end',owner_id=owner_id,mode=mode,length=n,
                    execution_index=len(seen),telemetry=gpu_telemetry()))
            finally:
                if active is not None:close(active);active=None
        require(counts==EXPECTED and len(seen)==2540,'Incomplete2540-request study; cannot freeze partial winners')
        require(source_bindings(root)==bindings and base.check_full_bindings(root)==tree_bindings and base.flash_receipt()==flash and
            {str(p):sha256(p) for p in paths}==assets,'Frozen imports/dependency/assets changed')
        for name,digest in sources.items():require(sha256(root/name)==digest,'Frozen source changed: '+name)
        group.idle();gpu_snapshot()
        frozen=select_winners(warm_rows,stage1,descriptors)
        write(output/'FROZEN_SELECTION.json',dict(frozen,source_sha256=sources,assets_sha256=assets,
            protocol_sha256=sha256(output/'protocol.json'),complete_request_count=2540,
            stage1_selection_sha256=prepared['metadata']['stage1_selection_sha256'],no_new_confirmation_opened=True))
        return dict(passed=True,counts=counts,expected_counts=EXPECTED,logical_requests=2540,owner_closed=76,
            sources_unchanged=True,assets_unchanged=True,dependencies_unchanged=True,all_canonical=True,backend_trajectories_equal=True,
            frozen_selection_sha256=sha256(output/'FROZEN_SELECTION.json'),analysis_required_locally=True,goal_complete=False)
    finally:
        if active is not None:
            try:close(active)
            except BaseException:active.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('teacher','student','train-file','opt-confirmation-file','arch-confirmation-file','prime-file',
                 'stage1-results','stage1-analysis','stage1-verified','stage1-manifest','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())


if __name__=='__main__':main()
