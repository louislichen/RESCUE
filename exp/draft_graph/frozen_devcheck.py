"""Frozen RESCUE feasibility check on the sixteen already-used stage2 documents.

This external wrapper never selects, trains, changes rescue source, opens fresh
Parquet, generates free AR references, or executes the fresh-confirmation runner.
Both genuinely complete study seals are mandatory before old data/model access.
Scientific source is imported only from an explicitly pinned frozen source root.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import copy
from dataclasses import asdict
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
from types import SimpleNamespace
import uuid

LENGTHS = (4096, 8192)
MODE = 'rescue_graph'
GPU = 'GPU-9db4f7b5-1447-9a2a-5c1b-80379d909fd9'
EXPECTED = dict(own_quality_requests=64, quality_owners=64, prime_requests=2,
                timing_requests=64, owners=2)
TOTAL = 130
PROOF_KEYS = tuple(s+'_'+v for s in ('stage1','stage2') for v in ('results','analysis','verified','manifest'))
ARG_KEYS = (*PROOF_KEYS, 'source_root', 'teacher', 'student', 'mapper', 'policy', 'cost', 'prime_file', 'output')
ASSET_SHA = dict(mapper='d08889e796ca2527e23241ff34a072ef640e2fdf3fe9a2f9ba5439975af04d9f',
    policy='3c333a470119c236afcc1308c18592842a535bee0b8f83c4fb4da359ff17eff5',
    cost='d92c843a84676c0cd6adaf9364bbf7e8c771d56548f1016b5720a3e04f371fa1')
GATE_SOURCE = {
    'rescue/__init__.py':'89eba56e38c2689930a5bf1ba4e0221ada1da541dc16e4a759156fa9dd7fe11a',
    'rescue/fair_union_confirmation_data.py':'9ef8b39565303661e865f7a97a6d9bdcbad7838c41fbb7c11cc0c5864654446e',
    'rescue/fair_confirmation_data.py':'0224aa0e03ec3c3867a43dbb1a05142f1886f22872b4a004447e4d20fa6ebba4',
    'rescue/baseline_horizon_data.py':'1b4699a9427e6584c5476f3e2ee0ccf3bdab20b75ca5fee3ed23eb107a04d4cd'}


def require(ok, message):
    if not ok: raise RuntimeError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda:handle.read(8*1024*1024),b''):digest.update(part)
    return digest.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n')


def append(path, value):
    with Path(path).open('a') as handle:handle.write(json.dumps(value,allow_nan=False)+'\n')


def rows(path):
    raw=Path(path).read_bytes()
    require(raw and raw.endswith(b'\n') and all(v.strip() for v in raw.splitlines()),'Incomplete old JSONL')
    return [json.loads(v) for v in raw.splitlines()]


def no_links(path):
    path=Path(path)
    require(not any(p.is_symlink() for p in reversed((path,*path.parents))),'Symlink in explicit input/output path')


def idle_gpu():
    """No CUDA import/context, clock changes, process interruption or fallback."""
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).strip()
    values=[r.split(',') for r in raw.splitlines() if r.strip()]
    require(len(values)==1 and len(values[0])==4 and values[0][1].strip()==GPU and 'H100' in values[0][2] and
        int(values[0][3])>=80000,'Require the same exact single H10080GB')
    jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).strip()
    require(not jobs,'GPU occupied; no process was interrupted')
    return dict(nvidia_smi_gpu=raw,nvidia_smi_compute=jobs,gpu_uuid=GPU,idle=True)


def load_gate(root):
    """Authenticate the four stdlib modules before importing any of their code."""
    no_links(root);require(root.is_dir(),'Explicit frozen source root missing')
    for name,digest in GATE_SOURCE.items():
        path=root/name;no_links(path);require(path.is_file() and sha(path)==digest,'Frozen gate source changed: '+name)
    for name,module in list(sys.modules.items()):
        if name=='rescue' or name.startswith('rescue.'):
            actual=Path(module.__file__).resolve()
            require(actual.is_relative_to(root/'rescue'),'Foreign rescue namespace already imported')
    sys.path.insert(0,str(root))
    data=importlib.import_module('rescue.fair_union_confirmation_data')
    for name in GATE_SOURCE:
        module=importlib.import_module('rescue' if name=='rescue/__init__.py' else name[:-3].replace('/','.'))
        require(Path(module.__file__).resolve()==root/name,'Frozen gate imported a foreign module')
    return data


def proof_arguments(args): return {name:getattr(args,name) for name in PROOF_KEYS}


def canonical(ids,eos):
    require(isinstance(ids,list) and 0<len(ids)<=128 and all(type(v) is int and v>=0 for v in ids),'Invalid reused canonical output')
    require(len(ids)==128 or ids[-1]==eos,'Short reused canonical output lacks EOS')
    require(eos not in ids or ids[-1]==eos and eos not in ids[:-1],'Reused canonical EOS is not first EOS')


def validate_sequence(sequence,indices):
    require(isinstance(sequence,list) and len(sequence)==32 and
        all(set(v)=={'source_index','repeat'} and v['source_index'] in indices and type(v['repeat']) is int for v in sequence) and
        {(v['source_index'],v['repeat']) for v in sequence}=={(i,r) for i in indices for r in (0,1)},
        'Must copy the exact shared stage2 thirty-two request sequence')


def _runtime(root):
    """Load, but never edit or call the run()/execute_study() of frozen studies."""
    names=('fair_confirmation','baseline_selection','baseline_horizon','draft_cost_recalibration','draft_graph_benchmark',
        'draft_graph_smoke','round_overhead_run','engine','loading','packing','query_mapper','step_budget','draft_graph','specextend_qwen')
    modules={n:importlib.import_module('rescue.'+n) for n in names}
    for name,module in modules.items():require(Path(module.__file__).resolve()==root/'rescue'/(name+'.py'),'Foreign frozen runtime module')
    return SimpleNamespace(**modules,torch=importlib.import_module('torch'))


def bindings(runtime,root):
    values=runtime.fair_confirmation.source_bindings(root)
    for name,module in vars(runtime).items():
        if name=='torch':continue
        path=Path(module.__file__).resolve();require(path==root/'rescue'/(name+'.py'),'Frozen import escaped its source root')
        values['rescue.'+name]=dict(path=str(path),sha256=sha(path))
    return values


def models(root):
    paths=list(root.rglob('*'));require(not any(p.is_symlink() for p in paths),'Symlink in frozen model inventory')
    result=sorted(p for p in paths if p.is_file() and p.suffix in {'.json','.model','.safetensors','.bin','.txt'} and
        not any(v.startswith('.') for v in p.relative_to(root).parts))
    require((root/'config.json').is_file() and any(p.suffix in {'.safetensors','.bin'} for p in result),'Frozen model assets missing')
    return result


def copy_evidence(arguments,output,label):
    """Only files whitelisted by the already-validated final manifest."""
    source=Path(arguments['final_manifest']).parent;target=output/label;manifest=read(arguments['final_manifest']);index={}
    for relative,digest in manifest['file_sha256'].items():
        part=Path(relative);require(not part.is_absolute() and '..' not in part.parts,'Unsafe sealed evidence path')
        path=source/part;no_links(path);require(sha(path)==digest,'Development evidence changed while copying')
        dest=target/part;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        require(sha(dest)==digest,'Copied development evidence changed');index[str(path)]=dict(saved_path=str(dest.relative_to(output)),sha256=digest)
    relative_paths={}
    for key,path in arguments.items():
        path=Path(path);require(path.is_relative_to(source),'Proof escapes its final manifest root')
        relative_paths[key]=str(path.relative_to(source))
        if path.is_file():
            no_links(path);dest=target/path.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():shutil.copyfile(path,dest)
            require(sha(dest)==sha(path),'Copied seal changed');index[str(path)]=dict(saved_path=str(dest.relative_to(output)),sha256=sha(path))
    write(target/'PIN_INDEX.json',dict(original_to_saved=index))
    return dict(root=label,pin_index=label+'/PIN_INDEX.json',pin_index_sha256=sha(target/'PIN_INDEX.json'),**relative_paths)


def run(args):
    require(set(vars(args))==set(ARG_KEYS),'Exactly the sixteen explicit frozen-devcheck paths are required')
    for name,path in vars(args).items():
        require(isinstance(path,Path) and path.is_absolute() and '..' not in path.parts,'All arguments must be absolute canonical paths: '+name)
        no_links(path)
    output=args.output;require(not output.exists(),'Fresh isolated output only; never resume/retry/reuse partial data')
    protected=(args.source_root,args.teacher,args.student,args.stage1_manifest.parent,args.stage2_manifest.parent)
    require(not any(output==p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected),'Output overlaps frozen source/evidence/models')
    data=load_gate(args.source_root)
    proof=data.validate_union_freeze(**proof_arguments(args))
    require(proof.get('status')=='UNION_VERIFIED','Both real completed and sealed studies are mandatory')
    gpu=idle_gpu()  # No old document/model bytes and no writes before this gate.
    first=read(args.stage1_results/'protocol.json');second=read(args.stage2_results/'protocol.json')
    require(gpu['nvidia_smi_gpu']==first['environment']['nvidia_smi_gpu']==second['environment']['nvidia_smi_gpu'],'Different physical development GPU')
    sources=data._sources(args.source_root,second['source_sha256'])
    datafile=args.stage2_results/'data/documents.jsonl';manifestfile=args.stage2_results/'data/DATASET_MANIFEST.json';primefile=args.stage2_results/'data/prime.json'
    require(second.get('selected_documents')=='data/documents.jsonl' and second.get('data_manifest')=='data/DATASET_MANIFEST.json' and
        second.get('prime_document')=='data/prime.json' and sha(datafile)==second['selected_documents_sha256'] and
        sha(manifestfile)==second['data_manifest_sha256'] and sha(primefile)==second['prime_document_sha256'],'Old stage2 data bytes changed')
    docrows=rows(datafile);documents={r['source_index']:r for r in docrows};indices=second['source_indices']
    require(len(documents)==len(docrows)==16 and list(documents)==indices and indices==sorted(indices),'Fixed sixteen old documents/order changed')
    prime=read(primefile);receipt=read(manifestfile)
    data.horizon.validate_horizon_manifest(receipt,documents,prime)
    oldpaths={name:args.stage2_results/path for name,path in second['data_source_paths'].items()}
    data.horizon.verify_horizon_sources(receipt,oldpaths)
    sequence=copy.deepcopy(second['shared_warm_sequence']);validate_sequence(sequence,indices)
    references=read(args.stage2_results/'references.json');reference_rows=rows(args.stage2_results/'reference.jsonl')
    require(len(reference_rows)==32 and len({(r['source_index'],r['length']) for r in reference_rows})==32 and
        references=={f"{r['source_index']}:{r['length']}":r['generated'] for r in reference_rows} and
        set(references)=={f'{i}:{n}' for i in indices for n in LENGTHS},'Sealed thirty-two AR references are incomplete')
    for ids in references.values():canonical(ids,second['tokenizer_eos_token_id'])
    for role,digest in ASSET_SHA.items():require(sha(getattr(args,role))==digest,'Frozen '+role+' changed; no retraining or search permitted')
    paths=sorted(set(models(args.teacher)+models(args.student)+list(oldpaths.values())+[args.mapper,args.policy,args.cost,args.prime_file,
        datafile,manifestfile,primefile,args.stage2_results/'references.json',args.stage2_results/'reference.jsonl',args.stage1_results/'references.json']))
    assets={str(p):sha(p) for p in paths}
    for path in models(args.teacher)+models(args.student)+[args.prime_file]:
        require(first['assets_sha256'].get(str(path))==assets[str(path)]==second['assets_sha256'].get(str(path)),'Model/tokenizer/old271 prime differs from sealed studies')
    require(str(args.teacher)==first['args']['teacher']==second['args']['teacher'] and str(args.student)==first['args']['student']==second['args']['student'],'Different frozen model roots')
    own_source=Path(__file__).resolve();own_sha=sha(own_source);counts=dict.fromkeys(EXPECTED,0);seen=set()
    output.mkdir(parents=True)
    try:
        portable={}
        for stage in ('stage1','stage2'):
            arguments={k:getattr(args,stage+'_'+v) for k,v in (('baseline_results','results'),('analysis_dir','analysis'),('verified_complete','verified'),('final_manifest','manifest'))}
            portable[stage+'_evidence']=copy_evidence(arguments,output,stage+'_evidence')
        for name,digest in sources.items():
            dest=output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(args.source_root/name,dest);require(sha(dest)==digest,'Frozen source copy changed')
        (output/'data').mkdir()
        for name,path in (('documents.jsonl',datafile),('DATASET_MANIFEST.json',manifestfile),('prime.json',primefile)):
            shutil.copyfile(path,output/'data'/name);require(sha(output/'data'/name)==assets[str(path)],'Old data snapshot changed')
        for name,path in oldpaths.items():
            dest=output/'data'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
            require(sha(dest)==assets[str(path)],'Copied old cohort changed')
        shutil.copyfile(own_source,output/'FROZEN_DEVCHECK_SOURCE.py');require(sha(output/'FROZEN_DEVCHECK_SOURCE.py')==own_sha,'External entry snapshot changed')
        shutil.copyfile(args.stage2_results/'reference.jsonl',output/'reference.jsonl')
        require(sha(output/'reference.jsonl')==assets[str(args.stage2_results/'reference.jsonl')],'Copied reused AR rows changed')
        write(output/'GATE_PASSED.json',dict(passed=True,union_freeze=proof,source_sha256=sources,entry_source_sha256=own_sha,
            expected_counts=EXPECTED,expected_requests=TOTAL,old_data_only=True,no_new_confirmation_opened=True))
        prepared=dict(data=data,proof=proof,documents=documents,prime=prime,references=references,
            stage1_references=read(args.stage1_results/'references.json'),first=first,second=second,sequence=sequence,
            sources=sources,assets=assets,asset_paths=paths,counts=counts,seen=seen,entry_path=own_source,entry_sha256=own_sha,portable=portable,
            metadata=dict(data_manifest='data/DATASET_MANIFEST.json',data_manifest_sha256=sha(manifestfile),
                selected_documents='data/documents.jsonl',selected_documents_sha256=sha(datafile),prime_document='data/prime.json',prime_document_sha256=sha(primefile),
                data_source_paths={name:'data/'+name for name in oldpaths},data_receipt_sha256=receipt['receipt_sha256']))
        completion=execute_study(args,prepared)
        require(completion.get('passed') is True and completion.get('counts')==EXPECTED and completion.get('logical_requests')==TOTAL,
            'Incomplete frozen development generation matrix')
        require(data.validate_union_freeze(**proof_arguments(args))==proof,'Dual completed-study proof changed')
        require(data._sources(args.source_root,second['source_sha256'])==sources and sha(own_source)==own_sha and
            {str(p):sha(p) for p in paths}==assets,'Frozen source/assets/old data changed after generation')
        write(output/'COMPLETE.json',dict(completion,data_unchanged=True,dual_study_freeze_unchanged=True,entry_source_unchanged=True))
        return completion
    except BaseException as error:
        write(output/'FAILURE.json',dict(passed=False,error_type=type(error).__name__,error=str(error),counts=counts,
            logical_requests=len(seen),preserve_partial=True,no_automatic_retry=True,no_replacement=True,fresh_confirmation=False,goal_complete=False))
        raise


def execute_study(args,prepared):
    runtime=_runtime(args.source_root);cls=runtime.draft_graph.ResidentDraftRuntime;previous=cls.MAX_ENTRIES;cls.MAX_ENTRIES=128
    try:return _execute_study(args,prepared,runtime)
    finally:cls.MAX_ENTRIES=previous


def _execute_study(args,p,r):
    torch=r.torch;legacy=r.fair_confirmation;bench=r.draft_graph_benchmark;quality=r.draft_cost_recalibration
    smoke=r.draft_graph_smoke;common=r.round_overhead_run;output=args.output;counts=p['counts'];seen=p['seen']
    documents=p['documents'];indices=p['second']['source_indices'];sequence=p['sequence'];validate_sequence(sequence,indices)
    inputs={(i,n):common.context(documents[i],n) for n in LENGTHS for i in indices}
    require({f'{i}:{n}':hashlib.sha256(json.dumps(ids).encode()).hexdigest() for (i,n),ids in inputs.items()}==p['second']['input_sha256'],'Exact reused inputs changed')
    prime_inputs,prime_refs=legacy.read_prime(args.prime_file,p['first'],p['stage1_references'])
    torch.set_num_threads(4);torch.manual_seed(p['second']['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    environment=common.gpu_snapshot()
    for key in ('nvidia_smi_gpu','torch','cuda','target_dtype','draft_dtype','prefix_dtype','tf32'):
        require(environment[key]==p['first']['environment'][key]==p['second']['environment'][key],'Frozen GPU/runtime/precision changed: '+key)
    teacher,student,tokenizer=r.loading.load_models(str(args.teacher),str(args.student),'cuda:0',target_dtype=torch.float32,efficient=True)
    for model,dtype in ((teacher,torch.float32),(student,torch.bfloat16)):
        model.eval().requires_grad_(False);require(all(v.device==torch.device('cuda:0') and v.dtype==dtype for v in model.parameters()),'Frozen model precision/device changed')
    require(tokenizer.eos_token_id==p['second']['tokenizer_eos_token_id'] and len(tokenizer)==p['second']['vocab_size'],'Tokenizer/EOS/vocabulary changed')
    tick=time.perf_counter();template=r.specextend_qwen.build(teacher,student,str(args.teacher));template.prefill_precision='bfloat16'
    r.packing.share_packed_projections(teacher,template.base_model);r.packing.share_packed_projections(student,template.draft_model);preparation=time.perf_counter()-tick
    imports=bindings(r,args.source_root);flash=r.baseline_selection.flash_receipt()
    require(flash==p['first']['flash_attention']==p['second']['flash_attention'],'Actual frozen Flash dependency changed')
    mapper_payload=torch.load(args.mapper,map_location='cpu',weights_only=True);policy_payload=torch.load(args.policy,map_location='cpu',weights_only=True)
    require(policy_payload.get('kind')=='step_budget' and 'seed' in policy_payload,'Existing seeded StepBudget required')
    policy_meta={k:policy_payload[k] for k in ('kind','hidden','margin','prior_weight','seed')};cost=read(args.cost)
    configs={mode:asdict(legacy.own_config(backend)) for mode,backend in (('rescue_eager','eager'),(MODE,'cuda_graph'))}
    reference_reuse=dict(status='SEALED_OLD_REFERENCE_REUSE',reused_reference_requests=34,generated_reference_requests=0,
        reference_rows_are_reused_not_new_generation_calls=True,
        references_sha256=sha(args.stage2_results/'references.json'),reference_rows_sha256=sha(args.stage2_results/'reference.jsonl'),
        stage2_protocol_sha256=sha(args.stage2_results/'protocol.json'),stage1_protocol_sha256=sha(args.stage1_results/'protocol.json'),
        stage1_references_sha256=sha(args.stage1_results/'references.json'),input_sha256=p['second']['input_sha256'],
        prime_input_sha256={str(n):p['first']['input_sha256'][f'271:{n}'] for n in LENGTHS},scope='Already-used development documents, no new AR/free warmup or fresh confirmation')
    write(output/'reference_reuse.json',reference_reuse);write(output/'references.json',p['references']);write(output/'prime_references.json',prime_refs)
    protocol=dict(schema=1,status='FROZEN_OLD_DEVELOPMENT_CHECK',fresh_confirmation=False,no_new_confirmation_access=True,no_training=True,no_search=True,goal_complete=False,
        seed=p['second']['seed'],source_indices=indices,lengths=list(LENGTHS),repeats=2,max_new_tokens=128,shared_warm_sequence=copy.deepcopy(sequence),
        sequence_provenance='Exact COPY of sealed stage2.shared_warm_sequence, no randomization or selection here',
        owner_order=[dict(mode=MODE,length=n) for n in LENGTHS],configs=configs,expected_counts=EXPECTED,expected_requests=TOTAL,
        frozen_choices=p['proof']['choices'],comparisons=p['proof']['M'],
        reused_reference_requests=34,generated_reference_requests=0,reference_reuse_sha256=sha(output/'reference_reuse.json'),
        input_sha256=p['second']['input_sha256'],prime_source_index=271,prime_input_sha256={str(n):hashlib.sha256(json.dumps(prime_inputs[n]).encode()).hexdigest() for n in LENGTHS},
        source_sha256=p['sources'],source_bindings=imports,entry_source_file='FROZEN_DEVCHECK_SOURCE.py',entry_source_sha256=p['entry_sha256'],
        assets_sha256=p['assets'],environment=environment,flash_attention=flash,mapper_sha256=ASSET_SHA['mapper'],policy_sha256=ASSET_SHA['policy'],cost_sha256=ASSET_SHA['cost'],
        mapper_frozen=True,policy_frozen=True,policy_metadata=policy_meta,union_freeze=p['proof'],graph_max_entries=128,max_live_owners=1,
        tokenizer_eos_token_id=tokenizer.eos_token_id,vocab_size=len(tokenizer),target_layers=teacher.config.num_hidden_layers,draft_layers=student.config.num_hidden_layers,
        template_prefill_precision='bfloat16',model_preparation_seconds=preparation,**p['portable'],**p['metadata'],
        baseline_comparison_scope='Historical sealed stage2 warm clocks, separate sessions; development feasibility only, not fresh confirmation or a causal speed claim',
        primary='All32warm full-generate outer per context, including captures, prefill/reset/handoff/cleanup; constructor/cold/close separate',
        args={k:str(v) for k,v in vars(args).items()})
    write(output/'protocol.json',protocol)
    group=bench.HookOwners(teacher,student);live=[];quality_rows={}
    def make(backend,observer=None):
        group.idle();require(not live and not group.engines,'At most one live private Engine')
        mapper=r.query_mapper.mapper_from_payload(mapper_payload,'cuda:0').eval().requires_grad_(False)
        policy=r.step_budget.StepBudget.from_payload(policy_payload,'cuda:0').eval().requires_grad_(False)
        require(all(getattr(policy,k)==policy_payload[k] for k in ('margin','prior_weight','seed')),'Frozen policy parameters changed')
        engine=r.engine.RescueEngine(teacher,student,legacy.own_config(backend),mapper,policy,cost,vocab_size=len(tokenizer),round_observer=observer)
        try:group.add(engine)
        except BaseException:
            quality.detach_quality_observer(engine);engine.close();raise
        live.append(engine)
        if observer is not None:observer.engine=engine
        return engine
    def close(engine):
        quality.detach_quality_observer(engine)
        try:group.close(engine)
        finally:
            if engine in live:live.remove(engine)
    def stats(engine):
        return dict(smoke.runtime_snapshot(engine),max_graph_entries=engine.draft_runtime.max_entries if engine.draft_runtime is not None else None)
    def generate(engine,ids,reference,item):
        common.gpu_snapshot()
        torch.manual_seed(policy_meta['seed']);random.seed(policy_meta['seed']);group.idle();tick=time.perf_counter()
        row=group.generate(engine,ids,tokenizer.eos_token_id);elapsed=time.perf_counter()-tick
        row.update(item,arm=dict(mode=item['mode'],family='rescue',backend=engine.config.draft_execution),execution_index=len(seen),
            output_agrees=row['generated']==reference,runtime_stats=stats(engine),owner_signature=bench.owner_signature(engine))
        return row,elapsed
    try:
        for n in LENGTHS:
            for i in indices:
                pair=[]
                for mode,backend in (('rescue_eager','eager'),(MODE,'cuda_graph')):
                    observer=quality.QualityObserver();engine=make(backend,observer);owner_id=uuid.uuid4().hex
                    try:
                        with quality.record_draft_steps() as steps:
                            row,_=generate(engine,inputs[i,n],p['references'][f'{i}:{n}'],dict(mode=mode,source_index=i,length=n,phase='own_quality',owner_id=owner_id,performance_eligible=False))
                        row.update(observed_rounds=observer.rounds,queries=steps,nonzero_injection=observer.nonzero_injection,native_t=smoke.native_digest(engine.native_t),native_s=smoke.native_digest(engine.native_s))
                        bench.record(output,'own_quality.jsonl',row,counts,'own_quality_requests',seen)
                        legacy.check_quality_row(row,p['references'][f'{i}:{n}'],True,row['arm'],tokenizer.eos_token_id)
                        if backend=='cuda_graph':require(row['draft_runtime']['request']['captures']>0 and row['draft_runtime']['request']['replays']==row['draft_forwards']>0,'Fresh quality Graph work missing')
                        final=stats(engine);tick=time.perf_counter();close(engine);elapsed=time.perf_counter()-tick
                        require(engine.closed and stats(engine)['entries']==0,'Quality Engine retained graphs/hooks')
                        append(output/'quality_owners.jsonl',dict(owner_id=owner_id,mode=mode,length=n,source_index=i,request_id=row['request_id'],owner_signature=row['owner_signature'],
                            close_seconds=elapsed,final_runtime_stats=final,closed_runtime_stats=stats(engine),engine_closed=True,hook_restored=True))
                        counts['quality_owners']+=1;pair.append(row)
                    finally:
                        if engine in live:close(engine)
                legacy.own_quality_pair(*pair);quality_rows[i,n]=pair[1]
        require(counts['own_quality_requests']==counts['quality_owners']==64 and not live and not group.engines,'Quality must finish and close before timing')
        write(output/'QUALITY_PASSED.json',dict(passed=True,own_all32_pairs=True,actual_same_bfloat16_nonzero_mapper=True,quality_owners_closed=True,counts=dict(counts)))
        require({str(v):sha(v) for v in p['asset_paths']}==p['assets'],'Assets/data changed before timing')
        for n in LENGTHS:
            common.gpu_snapshot();owner_id=uuid.uuid4().hex
            append(output/'gpu_diagnostics.jsonl',dict(phase='owner_start',owner_id=owner_id,mode=MODE,length=n,execution_index=len(seen),telemetry=r.baseline_horizon.gpu_telemetry()))
            tick=time.perf_counter();engine=make('cuda_graph');constructor=time.perf_counter()-tick
            owner=dict(mode=MODE,length=n,owner_id=owner_id,owner_start=tick,constructor_and_detach_seconds=constructor,warm_tokens=0,warm_outer_seconds_sum=0.,warm_request_ids=[],maximum_live_owners=1)
            try:
                requests=[dict(mode=MODE,length=n,source_index=271,phase='cold_prime')]+[dict(v,mode=MODE,length=n,phase='warm',warm_index=j) for j,v in enumerate(sequence)]
                for item in requests:
                    cold=item['phase']=='cold_prime';i=item['source_index'];ref=prime_refs[str(n)] if cold else p['references'][f'{i}:{n}']
                    row,elapsed=generate(engine,prime_inputs[n] if cold else inputs[i,n],ref,dict(item,owner_id=owner_id,performance_eligible=True))
                    row.update(outer_seconds=elapsed,outer_tps=row['tokens']/elapsed,observer_enabled=False,profile_enabled=False,mapper_executed=True,controller_executed=True)
                    if cold:row.update(constructor_and_detach_seconds=constructor,cold_complete_outer_seconds=constructor+elapsed)
                    bench.record(output,'prime.jsonl' if cold else 'runtime.jsonl',row,counts,'prime_requests' if cold else 'timing_requests',seen)
                    bench.check_output(row,ref,tokenizer.eos_token_id)
                    require(math.isfinite(elapsed) and elapsed>0 and 'timeline' not in row and 'cost_samples' not in row and row['draft_runtime']['request']['replays']==row['draft_forwards']>0,'Primary timing profiled or not real Graph replay')
                    if cold:
                        require(row['draft_runtime']['request']['captures']>0,'Timing Engine borrowed quality Graphs')
                        owner.update(owner_signature=row['owner_signature'],prime_request_id=row['request_id'],prime_tokens=row['tokens'],cold_prime_outer_seconds=elapsed)
                    else:
                        require(row['owner_signature']==owner['owner_signature'],'Persistent owner/cache changed')
                        common.check_pair(quality_rows[i,n],row);owner['warm_tokens']+=row['tokens'];owner['warm_outer_seconds_sum']+=elapsed;owner['warm_request_ids'].append(row['request_id'])
                final=stats(engine);tick=time.perf_counter();close(engine);end=time.perf_counter();closed=stats(engine)
                require(engine.closed and closed['entries']==0,'Timing Engine retained graphs/hooks')
                seconds=constructor+owner['cold_prime_outer_seconds']+owner['warm_outer_seconds_sum']+end-tick
                owner.update(close_seconds=end-tick,owner_lifetime_wall_seconds=end-owner.pop('owner_start'),final_runtime_stats=final,closed_runtime_stats=closed,
                    engine_closed=True,hook_restored=True,accounted_service_seconds=seconds,accounted_service_tokens=owner['prime_tokens']+owner['warm_tokens'],accounted_service_tps=(owner['prime_tokens']+owner['warm_tokens'])/seconds)
                append(output/'owners.jsonl',owner);counts['owners']+=1
                append(output/'gpu_diagnostics.jsonl',dict(phase='owner_end',owner_id=owner_id,mode=MODE,length=n,execution_index=len(seen),telemetry=r.baseline_horizon.gpu_telemetry()))
            finally:
                if engine in live:close(engine)
        require(counts==EXPECTED and len(seen)==TOTAL,'Incomplete fixed130 generation matrix')
        require(bindings(r,args.source_root)==imports and r.baseline_selection.flash_receipt()==flash and
            {str(v):sha(v) for v in p['asset_paths']}==p['assets'] and sha(p['entry_path'])==p['entry_sha256'],'Frozen imports/dependency/assets/entry changed')
        for name,digest in p['sources'].items():require(sha(args.source_root/name)==digest,'Frozen scientific source changed')
        group.idle();common.gpu_snapshot()
        return dict(passed=True,counts=counts,expected_counts=EXPECTED,logical_requests=TOTAL,sources_unchanged=True,assets_unchanged=True,dependencies_unchanged=True,
            quality_gate_passed=True,all_canonical=True,owners_closed=True,analysis_required_locally=True,fresh_confirmation=False,no_training=True,no_search=True,goal_complete=False)
    finally:
        for engine in list(live):
            try:close(engine)
            except BaseException:quality.detach_quality_observer(engine);engine.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ARG_KEYS:parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    args=parser.parse_args();print(json.dumps(run(args),indent=2))


if __name__=='__main__':main()
