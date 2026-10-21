"""Frozen 538-request baseline-only development selection, no training.

One private serving owner at a time. Every requested visible token is checked
against a separate same-precision Target AR reference. All lazy captures and
upstream tree overshoot work remain charged to the unprofiled outer request.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time
from types import MethodType
import uuid

import torch

from .draft_graph_benchmark import HookOwners, owner_signature, record
from .draft_graph_smoke import COST_SHA256, runtime_snapshot
from .round_overhead_run import (MAPPER_SHA256, POLICY_SHA256, append,
    check_pair, context, gpu_snapshot, model_files, require, sha256, write)
from .specextend_match_pilot import check_full_bindings


SEED=20261023
LENGTHS=(4096,8192)
DOCUMENTS=(271,272,288)
REPEATS=2
DATA_SHA256='85b2eb1a81d1617b2d158c069c420c452459a8bdb90da128f257220cc27ded09'
PREREGISTRATION='exp/draft_graph/baseline_selection_20261020_preregistration.json'
PIPELINE='exp/draft_graph/run_baseline_selection_20261020_v1.sh'
EXPECTED=dict(reference_requests=6,prime_requests=76,timing_requests=456,owners=76)


def progress(stage,counts,**extra):
    print(json.dumps(dict(stage=stage,counts=dict(counts),**extra)),flush=True)


def arms():
    result=[]
    for top in (16,32):
        for k in (1,2,4,8,12):
            for backend in ('eager','resident','graph'):
                result.append(dict(mode=f'retrieval{top}_k{k}_{backend}',family='retrieval',
                    top_chunks=top,fixed_k=k,backend='cuda_graph' if backend=='graph' else backend))
    for top in (16,32):
        for nodes in (8,32,50,64):
            result.append(dict(mode=f'tree{top}_n{nodes}',family='full_tree',top_chunks=top,
                nodes=nodes,backend='upstream_flash_hybrid',max_depth=10,threshold=.7))
    return result


def schedule(seed=SEED):
    """One owner block contains a separately billed prime and all6 warm runs."""
    owner_order=[];warm_order=[]
    for n in LENGTHS:
        rng=random.Random(seed+n);modes=[a['mode'] for a in arms()];rng.shuffle(modes)
        for mode in modes:
            owner_order.append(dict(mode=mode,length=n))
            for repeat in range(REPEATS):
                documents=list(DOCUMENTS);rng.shuffle(documents)
                warm_order.extend(dict(mode=mode,length=n,source_index=i,repeat=repeat) for i in documents)
    return owner_order,warm_order


def read_documents(path,expected_sha=DATA_SHA256):
    require(sha256(path)==expected_sha,'Locked reused development file SHA mismatch')
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(len(rows)==3 and {r.get('source_index') for r in rows}==set(DOCUMENTS),
            'Require exactly locked documents271/272/288, no reselect')
    require(all(r.get('split')=='train' for r in rows),'Original corpus train only; never open confirmation/test')
    require(len({r.get('document_sha256') for r in rows})==3,'Duplicated development document')
    for row in rows:
        digest=row.get('document_sha256','')
        require(len(digest)==64 and all(v in '0123456789abcdef' for v in digest),'Missing document digest')
        for n in LENGTHS:context(row,n)
    return {r['source_index']:r for r in rows}


def source_inventory(root):
    tree=root/'exp/specextend_bridgekv/specextend'
    require((tree/'classic/model_classic.py').is_file(),'Stage the complete frozen upstream tree source explicitly')
    paths=list((root/'rescue').glob('*.py'))+list(tree.rglob('*.py'))+[
        root/'exp/attention_horizon/selection.py',root/'exp/selective_kv/bridge.py',
        root/PREREGISTRATION,root/PIPELINE,root/'tests/test_rescue_baseline_selection.py',
        root/'tests/test_rescue_baseline_selection_report.py']
    require(all(p.is_file() for p in paths),'Missing frozen runner/report/test/dependency source')
    return {str(p.relative_to(root)):sha256(p) for p in sorted(set(paths))}


def source_bindings(root):
    root=root.resolve();receipt={}
    names=('rescue.baseline_selection','rescue.baselines','rescue.engine','rescue.config',
        'rescue.attention','rescue.loading','rescue.packing','rescue.draft_graph',
        'rescue.specextend_qwen','exp.attention_horizon.selection','exp.selective_kv.bridge')
    for name in names:
        module=importlib.import_module(name);filename=getattr(module,'__file__',None)
        expected=root/Path(*name.split('.')).with_suffix('.py')
        require(filename is not None and Path(filename).resolve()==expected,
                'Python imported a foreign source namespace: '+name)
        receipt[name]=dict(path=str(expected),sha256=sha256(expected))
    return receipt


def flash_receipt():
    """Read-only provenance of the real installed CUDA extension, not a stub."""
    package=importlib.import_module('flash_attn')
    require(getattr(package,'__version__',None)=='2.8.3','Require frozen official FlashAttention2.8.3 environment')
    interface=importlib.import_module('flash_attn.flash_attn_interface')
    extension=importlib.import_module('flash_attn_2_cuda')
    files={}
    for name,module in (('flash_attn',package),('flash_attn.flash_attn_interface',interface),('flash_attn_2_cuda',extension)):
        filename=getattr(module,'__file__',None)
        require(filename is not None and Path(filename).is_file(),'Missing installed Flash source/extension')
        path=Path(filename).resolve()
        if name=='flash_attn_2_cuda':require(path.suffix=='.so','Flash CUDA extension is not a real binary shared object')
        files[name]=dict(path=str(path),sha256=sha256(path))
    return dict(version=package.__version__,files=files,read_only=True,
        scope='Installed isolated official dependency reused; no installation or shared-environment mutation')


def linear_config(arm):
    from .config import SystemConfig
    require(arm['family']=='retrieval','Not a retrieval arm')
    return SystemConfig(length_policy='fixed',fixed_k=arm['fixed_k'],max_draft=12,max_new_tokens=128,
        refresh_every=4,inject_tokens=0,injection_strength=0.,injection_layers=(),
        prefill_precision='bfloat16',schedule='serial',profile=False,
        working_mode='packed',optimized_execution=True,deferred_probes=True,
        attention_backend='sdpa',draft_execution=arm['backend'],coverage_execution='eager')


def tree_policy(arm):
    return dict(nodes=arm['nodes'],top_chunks=arm['top_chunks'],max_depth=arm['max_depth'],
        threshold=arm['threshold'],chunk_size=32,refresh_every=4,flash_prefill=True,hybrid_tree=True,
        prefill_precision='bfloat16',mapper=False,controller=False)


def normalize_tree(result,arm,reference,eos=None):
    """Keep the raw tree result even if the later canonical gate rejects it."""
    row=dict(result);upstream=row.get('upstream',{});scored=list(upstream.get('output_token_ids',[]))
    visible=scored[:128]
    if eos is not None and eos in visible:visible=visible[:visible.index(eos)+1]
    accepted=list(upstream.get('accept_length_list',[]))
    row.update(wrapper_generated=result.get('generated'),wrapper_tokens=result.get('tokens'),
        generated=visible,tokens=len(visible),scored_generated_before_eos=scored,
        raw_total_generated=upstream.get('raw_total_generated'),generation_overshoot=upstream.get('generation_overshoot'),
        raw_tail_ids_available=False,output_agrees=visible==reference,tree_policy=tree_policy(arm),
        blocks=len(accepted),accepted=sum(accepted),proposed=None,acceptance_rate=None,
        output_policy='Requested client prefix plus first EOS; entire raw tree interval charged; unexposed tail IDs/proposed counts never invented')
    return row


def check_output(row,reference,arm,eos=None):
    require(row.get('generated')==reference and row.get('output_agrees') is True,
            'Client-visible output differs from independent same-precision Target AR')
    require(0<row.get('tokens',0)==len(reference)<=128,'Invalid requested output count')
    if len(reference)<128:require(eos is not None and reference[-1]==eos,'Short output without canonical EOS')
    if eos is not None and eos in reference:
        require(reference[-1]==eos and eos not in reference[:-1],'Wrong canonical EOS boundary')
    if arm['family']=='retrieval':
        trace=row.get('trace',[])
        require(row.get('blocks')==len(trace)>0,'Missing linear round trace')
        require(row.get('accepted')==sum(r['accepted'] for r in trace) and
            row.get('proposed')==sum(r['proposed'] for r in trace),'Linear acceptance counters differ')
        progress=0
        for i,r in enumerate(trace):
            require(type(r['cap']) is int and r['cap']==min(arm['fixed_k'],128-progress) and
                1<=r['proposed']<=r['cap']<=arm['fixed_k']<=12 and
                0<=r['accepted']<=r['proposed'],'Invalid configured linear cap/acceptance')
            require(r['round_index']==i and r['terminal'] is (i==len(trace)-1),'Invalid linear terminal trace')
            require(r['prefix']==row['length']-1+progress,'Linear committed prefix progression changed')
            require(0<r['workset_tokens']<=arm['top_chunks']*32,'Wrong actual retrieval workset size')
            progress+=r['accepted']+1
        require(row.get('projected_positions')==0 and row.get('draft_shadow_probes')==0,
                'Baseline accidentally performed Mapper/probe work')
        require(row.get('execution_backend')==arm['backend'],'Linear execution backend changed')
        require(row.get('baseline_algorithm')=='qwen_chunk_retrieval','Unexpected baseline algorithm')
    else:
        upstream=row.get('upstream',{});scored=row.get('scored_generated_before_eos',[])
        raw=row.get('raw_total_generated');overshoot=row.get('generation_overshoot');accepted=upstream.get('accept_length_list',[])
        require(scored and len(scored)<=128 and upstream.get('total_generated')==len(scored),
                'Tree scored client-prefix accounting changed')
        require(row.get('wrapper_generated')==row.get('generated') and row.get('wrapper_tokens')==row.get('tokens'),
                'Archived wrapper/client-visible EOS accounting changed')
        require(type(raw) is int and type(overshoot) is int and raw>=len(scored) and overshoot==raw-len(scored),
                'Tree raw/overshoot accounting changed')
        require(accepted and all(type(v) is int and 0<=v<=arm['max_depth'] for v in accepted),
                'Missing/invalid raw tree acceptance counters')
        require(raw==sum(accepted)+len(accepted),'Tree raw committed-token count differs from per-round counters')
        require(row.get('raw_tail_ids_available') is False and row.get('proposed') is None and row.get('acceptance_rate') is None,
                'Unexposed tree tails/proposals must not be invented')
        require(row.get('tree_policy')==tree_policy(arm),'Configured tree policy changed')
        expected=dict(target_use_flash_prefill=True,target_use_hybrid_tree_attn=True,draft_use_flash_prefill=True,
            use_retrieval_cache=True,retrieve_top_k=arm['top_chunks'],retrieval_chunk_size=32,retrieve_every_n_steps=4)
        require(row.get('observed_full_config')==expected,'Actual Flash/hybrid/retrieval flags changed')


def check_reproducible(previous,row,arm):
    if arm['family']=='retrieval':
        check_pair(previous,row)
        for name in ('working_gather_layers','working_full_gather_layers','working_gathered_position_layers',
                     'working_incremental_binds','target_probe_rounds','draft_shadow_probes','projected_positions','rpc_calls'):
            require(previous.get(name)==row.get(name),'Linear repeated/backend work counter changed '+name)
    else:
        for name in ('generated','tokens','raw_total_generated','generation_overshoot','accepted','blocks','scored_generated_before_eos'):
            require(previous.get(name)==row.get(name),'Tree repeated document changed '+name)
        require(previous['upstream']['accept_length_list']==row['upstream']['accept_length_list'],
                'Tree repeated document changed raw accepted-per-round trajectory')


def select_winners(rows,descriptors=None):
    descriptors=arms() if descriptors is None else descriptors;lookup={a['mode']:a for a in descriptors}
    result={}
    for n in LENGTHS:
        candidates=[]
        for mode,arm in lookup.items():
            samples=[r for r in rows if r.get('length')==n and r.get('mode')==mode]
            require(len(samples)==6 and {(r.get('source_index'),r.get('repeat')) for r in samples}==
                {(i,r) for i in DOCUMENTS for r in range(REPEATS)},'Incomplete warm selection matrix')
            require(all(r.get('phase')=='warm' and r.get('performance_eligible') is True and
                type(r.get('outer_seconds')) in (int,float) and math.isfinite(r['outer_seconds']) and r['outer_seconds']>0
                for r in samples),'Selection may use only finite unprofiled warm request clocks')
            tokens=sum(r['tokens'] for r in samples);seconds=sum(r['outer_seconds'] for r in samples)
            candidates.append(dict(mode=mode,arm=arm,warm_tokens=tokens,warm_seconds=seconds,
                warm_tps=tokens/seconds,request_ids=[r['request_id'] for r in samples]))
        def best(c):return sorted(c,key=lambda r:(-r['warm_tps'],r['mode']))[0]
        groups={}
        for backend in ('eager','resident','cuda_graph'):
            name='graph' if backend=='cuda_graph' else backend
            groups['retrieval/'+name]=best([r for r in candidates if r['arm']['family']=='retrieval' and r['arm']['backend']==backend])
        groups['full_tree']=best([r for r in candidates if r['arm']['family']=='full_tree'])
        result[str(n)]=dict(groups=groups,overall=best(candidates),overall_family_winner={
            family:best([r for r in candidates if r['arm']['family']==family]) for family in ('retrieval','full_tree')},
            ranking=sorted(candidates,key=lambda r:(-r['warm_tps'],r['mode'])))
    return dict(per_context=result,group_candidates=8,selection_score='SUM warm client tokens / SUM warm full-generate outer seconds',
        exact_tie_break='lexicographically smallest mode; no tolerance',seed=SEED,
        scope='Development selection only; preserve all3 backend winners plus full tree for future confirmation; finite7request service score is secondary')


class FullOwner:
    """Private archived request state; immutable Qwen2 wrapper weights shared."""
    def __init__(self,template,teacher_path,arm):
        from classic.model_classic import SPModel
        self.arm=arm;self.closed=False
        require(template.prefill_precision=='bfloat16','Shared prefill closure must be BF16 too')
        self.engine=SPModel(template.base_model,str(teacher_path),template.draft_model).eval()
        self.engine.init_caches=MethodType(template.init_caches.__func__,self.engine)
        self.engine.prefill_precision='bfloat16'
        require(inspect.signature(self.engine.spgenerate).parameters['threshold'].default==.7,'Archived threshold changed')
        require(self.engine.attn_bridge_mapper is None and self.engine.attn_bridge_config=={} and
            self.engine.bridge_collector is None,'Full-tree baseline must not inject Mapper content')

    def generate(self,ids):
        from .specextend_qwen import generate
        result=generate(self.engine,ids,max_new_tokens=128,nodes=self.arm['nodes'],
            top_chunks=self.arm['top_chunks'],max_depth=self.arm['max_depth'])
        result['observed_full_config']={name:getattr(self.engine,name,None) for name in (
            'target_use_flash_prefill','target_use_hybrid_tree_attn','draft_use_flash_prefill',
            'use_retrieval_cache','retrieve_top_k','retrieval_chunk_size','retrieve_every_n_steps')}
        return result

    def signature(self):
        return dict(owner_identity=id(self),engine_identity=id(self.engine),
            base_model_identity=id(self.engine.base_model),draft_model_identity=id(self.engine.draft_model),
            prefill_precision=self.engine.prefill_precision)

    def stats(self):
        return dict(mode='upstream_flash_hybrid',entries=0,captures=0,replays=0,
            resident_steps=0,capture_host_seconds=0.,closed=self.closed,
            scope='Upstream full tree, not ResidentDraftRuntime; per-request caches follow archived implementation')

    def close(self):
        try:torch.cuda.synchronize()
        finally:
            if self.engine is not None:self.engine.init_caches=None
            self.engine=None;self.closed=True


def run(args):
    from .baselines import BaselineEngine
    from .loading import load_models
    from .packing import share_packed_projections
    from .specextend_qwen import build

    root=Path(__file__).resolve().parents[1];output=args.output.resolve()
    require(not output.exists(),'Fresh isolated output only; never resume partial selection')
    for name in ('teacher','student'):
        require(not output.is_relative_to(getattr(args,name).resolve()),'Output overlaps frozen model root')
    output.mkdir(parents=True);counts=dict.fromkeys(EXPECTED,0);seen=set();group=None;active=None
    try:
        for name in ('teacher','student','data_file'):
            path=getattr(args,name);require(path.is_absolute() and path.exists(),'Explicit existing absolute inputs required')
        documents=read_documents(args.data_file)
        inputs={(i,n):context(documents[i],n) for n in LENGTHS for i in DOCUMENTS}
        descriptors=arms();lookup={a['mode']:a for a in descriptors};owner_plan,warm_plan=schedule()
        sources=source_inventory(root);bindings=source_bindings(root);prereg=json.loads((root/PREREGISTRATION).read_text())
        for name in sources:
            destination=output/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,destination)
        paths=sorted(set(model_files(args.teacher)+model_files(args.student)+[args.data_file]));assets={str(p):sha256(p) for p in paths}
        torch.set_num_threads(4);torch.manual_seed(SEED)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        environment=gpu_snapshot()
        require(prereg['gpu_uuid'] in environment.get('nvidia_smi_gpu','') or prereg['gpu_uuid']==environment.get('gpu_uuid'),
                'Wrong physical GPU')
        teacher,student,tokenizer=load_models(str(args.teacher),str(args.student),'cuda:0',target_dtype=torch.float32,efficient=True)
        for model,dtype in ((teacher,torch.float32),(student,torch.bfloat16)):
            model.eval().requires_grad_(False)
            require(all(p.device==torch.device('cuda:0') and p.dtype==dtype for p in model.parameters()),'Frozen dtype/device changed')
        tick=time.perf_counter();template=build(teacher,student,str(args.teacher))
        template.prefill_precision='bfloat16'
        share_packed_projections(teacher,template.base_model);share_packed_projections(student,template.draft_model)
        preparation_seconds=time.perf_counter()-tick
        tree_bindings=check_full_bindings(root);dependency=flash_receipt();group=HookOwners(teacher,student)
        configs={a['mode']:asdict(linear_config(a)) if a['family']=='retrieval' else tree_policy(a) for a in descriptors}
        write(output/'protocol.json',dict(schema=1,seed=SEED,source_indices=list(DOCUMENTS),lengths=list(LENGTHS),repeats=REPEATS,
            max_new_tokens=128,arms=descriptors,modes=[a['mode'] for a in descriptors],configs=configs,
            expected_counts=EXPECTED,expected_requests=538,owner_order=owner_plan,warm_order=warm_plan,
            no_training=True,mapper_executed=False,controller_executed=False,data_file_sha256=DATA_SHA256,
            development_file_sha256=DATA_SHA256,
            original_corpus_roles={str(i):documents[i]['split'] for i in DOCUMENTS},
            document_sha256={str(i):documents[i]['document_sha256'] for i in DOCUMENTS},
            input_sha256={f'{i}:{n}':hashlib.sha256(json.dumps(ids).encode()).hexdigest() for (i,n),ids in inputs.items()},
            source_sha256=sources,source_bindings=bindings,tree_source_bindings=tree_bindings,
            assets_sha256=assets,environment=environment,flash_attention=dependency,
            preregistration_sha256=sources[PREREGISTRATION],
            graph_max_entries=128,max_live_owners=1,observer=False,profile=False,
            retrieval_retain_query=False,template_prefill_precision=template.prefill_precision,
            model_preparation_seconds=preparation_seconds,model_preparation_scope='Shared loading, architecture wrappers and identical packed operators excluded for all arms; private construction/close separately charged',
            output_policy='All client-visible requested tokens and EOS checked; tree raw overshoot is charged and unexposed tail IDs are unavailable',
            selection_score='Six warm requests SUM client tokens / SUM outer seconds; exact float maximum, lexical mode tie',
            selection_groups=['retrieval/eager','retrieval/resident','retrieval/graph','full_tree'],
            owner_timing='Single owner at a time; seeded blocks, separately billed doc271 cold prime then6warm; block clock drift is a limitation',
            primary_clock='Full generate outer including hook handoff, prefill/reset, every legal lazy capture and request cleanup; no observer or profile',
            frozen_rescue_anchor_for_future_confirmation=dict(mapper_sha256=MAPPER_SHA256,policy_sha256=POLICY_SHA256,cost_sha256=COST_SHA256,
                note='Metadata only: no Mapper/policy/cost is loaded or executed in this baseline-only study'),
            scope='Three reused original-train development documents; 38-arm winner selection bias; no held-out/production/original-Vicuna/full-SpecExtend superiority claim',
            args={k:str(v) for k,v in vars(args).items()}))
        progress('protocol_frozen',counts,arms=len(descriptors),expected_requests=538)
        references={}
        native_cfg=linear_config(descriptors[0])
        for n in LENGTHS:
            for i in DOCUMENTS:
                gpu_snapshot();group.idle();torch.manual_seed(SEED);random.seed(SEED)
                with BaselineEngine(teacher,student,native_cfg,vocab_size=len(tokenizer)) as engine:
                    row=engine.reference(inputs[i,n],tokenizer.eos_token_id)
                del engine  # closed NativeCache buffers must not remain live across the grid
                row.update(request_id=uuid.uuid4().hex,mode='canonical_ar',source_index=i,length=n,
                    phase='reference',execution_index=len(seen),performance_eligible=False)
                record(output,'reference.jsonl',row,counts,'reference_requests',seen)
                require(0<len(row['generated'])<=128,'Invalid canonical request length')
                references[f'{i}:{n}']=row['generated'];progress('reference_complete',counts,source_index=i,length=n)
        write(output/'references.json',references)
        require(source_inventory(root)==sources and {str(p):sha256(p) for p in paths}==assets,'Frozen sources/assets changed before timing')
        warm_rows=[];trajectories={};context_start=None;context_tokens=0;previous_length=None
        for entry in owner_plan:
            n,mode=entry['length'],entry['mode'];arm=lookup[mode]
            if n!=previous_length:
                if previous_length is not None:
                    append(output/'contexts.jsonl',dict(length=previous_length,context_wall_seconds=time.perf_counter()-context_start,
                        tokens=context_tokens,scope='All38 serial owner blocks plus audit I/O, diagnostic not per-arm production wall'))
                context_start=time.perf_counter();context_tokens=0;previous_length=n
            gpu_snapshot();group.idle();require(active is None and not group.engines,'More than one private serving owner')
            tick=time.perf_counter()
            if arm['family']=='full_tree':active=FullOwner(template,args.teacher,arm)
            else:
                active=BaselineEngine(teacher,student,linear_config(arm),vocab_size=len(tokenizer),top_chunks=arm['top_chunks'])
                group.add(active)
                require(active.round_observer is None and active.draft_observer is None,'Timing owner enabled an observer')
            constructor=time.perf_counter()-tick;owner_id=uuid.uuid4().hex
            owner=dict(mode=mode,arm=arm,length=n,owner_id=owner_id,owner_start=tick,constructor_and_detach_seconds=constructor,
                warm_tokens=0,warm_outer_seconds_sum=0.,warm_request_ids=[],maximum_live_owners=1)
            try:
                requests=[dict(mode=mode,length=n,source_index=271,phase='cold_prime')]+[
                    dict(r,phase='warm') for r in warm_plan if r['mode']==mode and r['length']==n]
                for request in requests:
                    i=request['source_index'];gpu_snapshot();group.idle();torch.manual_seed(SEED);random.seed(SEED)
                    start=time.perf_counter()
                    if arm['family']=='full_tree':raw=active.generate(inputs[i,n])
                    else:raw=group.generate(active,inputs[i,n],tokenizer.eos_token_id)
                    elapsed=time.perf_counter()-start
                    # Preserve upstream raw output before parsing/validating tree counters.
                    if arm['family']=='full_tree':
                        raw['request_id']=uuid.uuid4().hex
                        append(output/'tree_raw.jsonl',dict(request_id=raw['request_id'],execution_index=len(seen),
                            owner_id=owner_id,**request,outer_seconds=elapsed,result=raw))
                        row=normalize_tree(raw,arm,references[f'{i}:{n}'],tokenizer.eos_token_id)
                        signature=active.signature();stats=active.stats()
                    else:row=raw;signature=owner_signature(active);stats=runtime_snapshot(active)
                    row.update(request,arm=arm,request_id=row.get('request_id',uuid.uuid4().hex),execution_index=len(seen),owner_id=owner_id,
                        output_agrees=row['generated']==references[f'{i}:{n}'],performance_eligible=True,
                        outer_seconds=elapsed,outer_tps=row['tokens']/elapsed,runtime_stats=stats,owner_signature=signature,
                        observer_enabled=False,profile_enabled=False,mapper_executed=False,controller_executed=False)
                    cold=request['phase']=='cold_prime'
                    if cold:row.update(constructor_and_detach_seconds=constructor,cold_complete_outer_seconds=constructor+elapsed)
                    record(output,'prime.jsonl' if cold else 'runtime.jsonl',row,counts,'prime_requests' if cold else 'timing_requests',seen)
                    check_output(row,references[f'{i}:{n}'],arm,tokenizer.eos_token_id)
                    require(math.isfinite(elapsed) and elapsed>0,'Invalid primary outer clock')
                    require('timeline' not in row and 'cost_samples' not in row,'Profile leaked into primary selection')
                    key=(arm['family'],arm['top_chunks'],arm.get('fixed_k',arm.get('nodes')),i,n)
                    if key in trajectories:check_reproducible(trajectories[key],row,arm)
                    else:trajectories[key]=row
                    if arm['family']=='retrieval' and arm['backend']=='cuda_graph':
                        delta=row['draft_runtime']['request']
                        require(delta['replays']==row['draft_forwards']>0,'Graph replay count differs from actual Draft work')
                        if cold:require(delta['captures']>0 and delta['capture_host_seconds']>0,'Fresh timing owner borrowed graphs')
                        require(not active.draft_runtime.retain_query,'Timing baseline retained unused Query clones')
                    if cold:
                        owner.update(owner_signature=signature,prime_request_id=row['request_id'],prime_tokens=row['tokens'],
                            cold_prime_outer_seconds=elapsed)
                    else:
                        require(signature==owner['owner_signature'],'Persistent private owner/cache pointers changed')
                        warm_rows.append(row);owner['warm_tokens']+=row['tokens'];owner['warm_outer_seconds_sum']+=elapsed
                        owner['warm_request_ids'].append(row['request_id'])
                    context_tokens+=row['tokens'];group.idle()
                    progress('cold_prime_complete' if cold else 'warm_complete',counts,**request,
                        captures=row.get('draft_runtime',{}).get('request',{}).get('captures',0))
                owner['final_runtime_stats']=active.stats() if arm['family']=='full_tree' else runtime_snapshot(active)
                close_start=time.perf_counter()
                if arm['family']=='full_tree':active.close()
                else:group.close(active)
                close_end=time.perf_counter();closed_stats=active.stats() if arm['family']=='full_tree' else runtime_snapshot(active)
                require(active.closed and closed_stats['entries']==0,'Serving owner did not close/release graphs')
                seconds=constructor+owner['cold_prime_outer_seconds']+owner['warm_outer_seconds_sum']+close_end-close_start
                owner.update(close_seconds=close_end-close_start,owner_lifetime_wall_seconds=close_end-owner.pop('owner_start'),
                    closed_runtime_stats=closed_stats,engine_closed=True,hook_restored=True,
                    accounted_service_seconds=seconds,accounted_service_tokens=owner['prime_tokens']+owner['warm_tokens'],
                    accounted_service_tps=(owner['prime_tokens']+owner['warm_tokens'])/seconds,
                    scope='Constructor+cold prime+6warm+close measured service sum; audit I/O omitted, not production wall')
                append(output/'owners.jsonl',owner);counts['owners']+=1;active=None
                require(not group.engines,'Closed retrieval owner retained in hook registry');group.idle()
                progress('owner_complete',counts,mode=mode,length=n)
            except BaseException:
                # Failure is fatal. Release only this experiment's current owner; never restart/skip it.
                if active is not None:
                    try:
                        if arm['family']=='full_tree':active.close()
                        else:group.close(active)
                    finally:active=None
                raise
        append(output/'contexts.jsonl',dict(length=previous_length,context_wall_seconds=time.perf_counter()-context_start,
            tokens=context_tokens,scope='All38 serial owner blocks plus audit I/O, diagnostic only'))
        require(counts==EXPECTED and len(seen)==538,'Incomplete538-request selection; never freeze partial winners')
        require(source_inventory(root)==sources and source_bindings(root)==bindings and check_full_bindings(root)==tree_bindings and
            flash_receipt()==dependency and {str(p):sha256(p) for p in paths}==assets,'Frozen code/dependency/model/data changed')
        group.idle();gpu_snapshot()
        selection=select_winners(warm_rows,descriptors)
        write(output/'FROZEN_SELECTION.json',dict(selection,source_sha256=sources,assets_sha256=assets,
            protocol_sha256=sha256(output/'protocol.json'),complete_request_count=538,no_new_confirmation_opened=True))
        write(output/'COMPLETE.json',dict(passed=True,counts=counts,expected_counts=EXPECTED,logical_requests=538,
            sources_unchanged=True,assets_unchanged=True,dependencies_unchanged=True,all_canonical=True,
            backend_trajectories_equal=True,owner_closed=76,frozen_selection_sha256=sha256(output/'FROZEN_SELECTION.json'),
            analysis_required_locally=True,goal_complete=False,scope='Development grid selection, not final independent confirmation'))
        progress('selection_complete',counts)
    except BaseException as error:
        cleanup=[]
        if active is not None:
            try:
                if isinstance(active,FullOwner):active.close()
                elif group is not None:group.close(active)
            except BaseException as failure:
                cleanup.append(repr(failure))
                # A constructor/handoff guard can fail before registry entry.
                # Close only our own object to restore its installed hooks.
                try:active.close()
                except BaseException as failure:cleanup.append(repr(failure))
            finally:active=None
        write(output/'FAILURE.json',dict(passed=False,error_type=type(error).__name__,error=str(error),counts=counts,
            logical_requests=len(seen),cleanup_errors=cleanup,
            scope='Preserve partial root; no restart, data cleanup, candidate omission or winner freezing'))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('teacher','student','data-file','output'):parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())


if __name__=='__main__':main()
