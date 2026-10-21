"""22-request, fixed-algorithm coverage-Graph execution pilot on doc303.

Unprofiled timing finishes before four diagnostic requests. Profiling is never
included in primary throughput or the measured active-service denominator.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import uuid

import torch

from .draft_graph_benchmark import HookOwners, owner_signature, record
from .draft_graph_smoke import COST_SHA256, native_digest, record_queries, runtime_snapshot
from .round_overhead_run import (MAPPER_SHA256, POLICY_SHA256, append, check_pair,
    context, gpu_snapshot, model_files, require, sha256, write)
from .specextend_match_pilot import Observer, check_output, check_quality, read_document


MODES = ('draft_graph', 'draft_coverage_graph')
LENGTHS = (4096, 8192)
SEED = 20261022
EXPECTED = dict(reference_requests=2, correctness_requests=4, prime_requests=4,
                timing_requests=8, profile_requests=4, owners=4)


def progress(stage, counts, **fields):
    print(json.dumps(dict(stage=stage, counts=dict(counts), **fields)), flush=True)


def schedule(seed=SEED):
    result = []
    for length in LENGTHS:
        rng = random.Random(seed+length)
        for repeat in range(2):
            modes = list(MODES); rng.shuffle(modes)
            result.extend(dict(mode=m, source_index=303, length=length, repeat=repeat) for m in modes)
    return result


def source_inventory(root):
    paths = list((root/'rescue').glob('*.py'))+[
        root/'exp/attention_horizon/selection.py', root/'exp/selective_kv/bridge.py',
        root/'tests/test_rescue_coverage_graph_pilot.py']
    require(all(p.is_file() for p in paths), 'Missing frozen coverage pilot dependency/source/test')
    return {str(p.relative_to(root)):sha256(p) for p in sorted(set(paths))}


def coverage_snapshot(engine):
    runtime = getattr(engine, 'coverage_runtime', None)
    if runtime is None:
        return dict(mode='eager', owner_identity=None, max_entries=0, entries=0,
            captures=0, replays=0, capture_host_seconds=0., eager_calls=0,
            input_copies=0, shape_keys=[])
    return dict(**runtime.stats(), owner_identity=id(runtime),
        max_entries=runtime.max_entries, shape_keys=list(runtime._entries))


def private_signature(engine):
    signature = owner_signature(engine)
    runtime = getattr(engine, 'coverage_runtime', None)
    signature['coverage_runtime_identity'] = id(runtime) if runtime is not None else None
    return signature


@contextmanager
def profile_only(engine):
    """Temporarily enable only diagnostic timing, not a learned decision."""
    original = engine.config
    require(original.profile is False, 'Unprofiled owner must start profile=False')
    engine.config = replace(original, profile=True)
    try:
        yield
    finally:
        engine.config = original


def finish_owner_record(record_, close_seconds, end_time):
    """No profile time/tokens are included in accounted serving throughput."""
    result = dict(record_)
    result.update(close_seconds=close_seconds,
        owner_lifetime_wall_seconds=end_time-result.pop('owner_start'), hook_restored=True)
    seconds = (result['constructor_and_detach_seconds']+result['cold_prime_outer_seconds']+
               result['warm_outer_seconds_sum']+close_seconds)
    result.update(accounted_service_seconds=seconds,
        accounted_service_tokens=result['prime_tokens']+result['warm_tokens'],
        accounted_service_tps=(result['prime_tokens']+result['warm_tokens'])/seconds,
        profile_excluded_from_service=True,
        accounting_scope='Measured constructor/detach+cold+2 unprofiled warm+close only; excludes profile execution; not a contiguous owner session')
    return result


def run(args):
    from .attention import DraftAttention
    from .baselines import BaselineEngine
    from .config import SystemConfig
    from .draft_graph import ResidentDraftRuntime
    from .engine import RescueEngine
    from .loading import load_models
    from .packing import pack_projections
    from .query_mapper import mapper_from_payload
    from .step_budget import StepBudget

    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    require(not output.exists(), 'Fresh coverage pilot output required; never reuse a partial directory')
    output.mkdir(parents=True)
    counts = dict.fromkeys(EXPECTED, 0); seen = set(); group = None
    live = []; active = {}; records = {}; contexts = []
    old_limit = ResidentDraftRuntime.MAX_ENTRIES
    try:
        require(sha256(args.mapper) == MAPPER_SHA256 and sha256(args.policy) == POLICY_SHA256 and
                sha256(args.cost) == COST_SHA256, 'Expected unchanged Mapper/StepBudget/cost')
        item = read_document(args.data_file); inputs = {n:context(item,n) for n in LENGTHS}
        sources = source_inventory(root)
        for name in sources:
            path = output/'source'/name; path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root/name,path)
        paths = sorted(set(model_files(args.teacher)+model_files(args.student)+[
            args.mapper,args.policy,args.cost]+args.data_file))
        assets = {str(p):sha256(p) for p in paths}
        torch.set_num_threads(4); torch.manual_seed(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        environment = gpu_snapshot()
        teacher,student,tokenizer = load_models(str(args.teacher),str(args.student),'cuda:0',
            target_dtype=torch.float32,efficient=True)
        pack_projections(teacher); pack_projections(student)
        mapper = mapper_from_payload(torch.load(args.mapper,map_location='cpu',weights_only=True),'cuda:0')
        policy = StepBudget.from_payload(torch.load(args.policy,map_location='cpu',weights_only=True),'cuda:0')
        for model in (teacher,student,mapper,policy):
            model.eval().requires_grad_(False)
        cost = json.loads(args.cost.read_text()); group = HookOwners(teacher,student)
        ResidentDraftRuntime.MAX_ENTRIES = 128
        config = SystemConfig(selector='hierarchical',coverage_rows=64,refresh_every=4,
            refresh_policy='periodic',injection_layers=(27,),inject_tokens=64,injection_strength=.5,
            prefill_precision='bfloat16',max_new_tokens=128,max_draft=12,length_policy='risk',
            schedule='serial',profile=False,draft_execution='cuda_graph',coverage_execution='eager')
        configs = dict(draft_graph=config,draft_coverage_graph=replace(config,coverage_execution='cuda_graph'))
        plan = schedule()
        write(output/'protocol.json',dict(schema=1,scope='Coverage execution-only n1 reused-dev gap pilot; not final SpecExtend/held-out evidence',
            source_index=303,document_sha256=item['document_sha256'],lengths=list(LENGTHS),max_new_tokens=128,
            repeats=2,modes=list(MODES),expected_counts=EXPECTED,expected_requests=22,
            configs={m:asdict(c) for m,c in configs.items()},source_sha256=sources,assets_sha256=assets,
            mapper_sha256=MAPPER_SHA256,policy_sha256=POLICY_SHA256,cost_sha256=COST_SHA256,
            mapper_frozen=True,policy_frozen=True,no_training=True,environment=environment,
            draft_graph_max_entries=128,coverage_graph_max_entries=64,retain_query=True,
            input_sha256={str(n):hashlib.sha256(json.dumps(v).encode()).hexdigest() for n,v in inputs.items()},
            warm_order=plan,seed=SEED,confidence_intervals=None,
            primary='2 unprofiled warm full-generate outer requests per mode/context; pool client tokens/time; captures included',
            profiling='Exactly4 diagnostic requests only AFTER all8 unprofiled warm requests; same warmed owners, temporary profile=True, no observer; never primary TPS data',
            service_accounting='Constructor/detach+cold+2 unprofiled warm+close only. Profile time/tokens separately recorded and EXCLUDED. Owner lifetime includes other arms/context/profiling/I/O, not production throughput.',
            shape_policy='Actual coverage support/pool shapes and exact selector outputs; no padding, altered budget, approximate scores or silent fallback. Every legal cold/warm/profile capture remains recorded.',
            args={k:[str(p) for p in v] if isinstance(v,list) else str(v) for k,v in vars(args).items()}))
        progress('protocol_frozen',counts,expected_counts=EXPECTED)

        def make(mode, observer=None):
            group.idle()
            engine = RescueEngine(teacher,student,configs[mode],mapper,policy,cost,
                vocab_size=len(tokenizer),round_observer=observer)
            group.add(engine); live.append(engine)
            return engine

        def close(engine):
            group.close(engine)
            if engine in live:
                live.remove(engine)

        def enrich(row, engine, mode, length, reference):
            row.update(mode=mode,source_index=303,length=length,output_agrees=row['generated']==reference,
                execution_index=len(seen),
                runtime_stats=runtime_snapshot(engine),coverage_runtime_stats=coverage_snapshot(engine),
                owner_signature=private_signature(engine))
            return row

        def graph_gate(row, cold=False):
            require(row['draft_runtime']['request']['replays']>0,'Draft Graph did not replay')
            if cold:
                require(row['draft_runtime']['request']['captures']>0,'Timing Draft Graph borrowed a quality graph')
            if row['mode']=='draft_coverage_graph':
                require(row['coverage_runtime']['request']['replays']>0,'Coverage Graph did not replay real selection')
                if cold:
                    require(row['coverage_runtime']['request']['captures']>0,'Timing Coverage Graph borrowed a quality graph')

        references = {}
        for length in LENGTHS:
            gpu_snapshot(); group.idle()
            with BaselineEngine(teacher,student,replace(config,length_policy='fixed',draft_execution='eager'),vocab_size=len(tokenizer)) as engine:
                row = engine.reference(inputs[length],tokenizer.eos_token_id)
            row.update(request_id=uuid.uuid4().hex,mode='canonical_ar',source_index=303,length=length,
                execution_index=len(seen))
            record(output,'reference.jsonl',row,counts,'reference_requests',seen)
            references[str(length)] = row['generated']; progress('reference_complete',counts,length=length)
        write(output/'references.json',references)
        quality = {}
        for length in LENGTHS:
            baseline = None
            for mode in MODES:
                gpu_snapshot(); observer = Observer(); engine = make(mode,observer)
                try:
                    with record_queries(DraftAttention) as queries:
                        row = group.generate(engine,inputs[length],tokenizer.eos_token_id)
                    enrich(row,engine,mode,length,references[str(length)])
                    row.update(phase='correctness',owner_id=uuid.uuid4().hex,performance_eligible=False,
                        observed_rounds=observer.rounds,queries=queries,native_t=native_digest(engine.native_t),
                        native_s=native_digest(engine.native_s),nonzero_injection=observer.nonzero_injection)
                    record(output,'correctness.jsonl',row,counts,'correctness_requests',seen)
                    check_output(row,references[str(length)],tokenizer.eos_token_id)
                    require(observer.nonzero_injection and len(queries)==row['draft_forwards'],'Missing nonzero Mapper/every-step Query evidence')
                    graph_gate(row,cold=True)
                    if baseline is None:
                        baseline = row
                    else:
                        check_quality(baseline,row)
                    quality[mode,length] = row
                    progress('correctness_complete',counts,mode=mode,length=length)
                finally:
                    close(engine)
        require(source_inventory(root)==sources and {str(p):sha256(p) for p in paths}==assets,'Source/assets changed before timing')
        write(output/'QUALITY_PASSED.json',dict(passed=True,counts=dict(counts),
            exact_selected_features_cap_draft_checks_native_queries=True,actual_nonzero=True))
        progress('quality_gate_passed',counts)

        # Keep the four timing owners alive until AFTER both contexts' warm
        # timings, so diagnostic profiling reuses warm graphs, not fresh ones.
        for length in LENGTHS:
            context_start = time.perf_counter(); order=list(MODES);random.Random(SEED+length+7).shuffle(order)
            for mode in order:
                gpu_snapshot(); tick=time.perf_counter(); engine=make(mode);constructor=time.perf_counter()-tick
                key=mode,length;active[key]=engine;owner_id=uuid.uuid4().hex
                start=time.perf_counter();row=group.generate(engine,inputs[length],tokenizer.eos_token_id);end=time.perf_counter()
                enrich(row,engine,mode,length,references[str(length)])
                row.update(phase='cold_prime',owner_id=owner_id,performance_eligible=True,
                    outer_seconds=end-start,outer_tps=row['tokens']/(end-start),
                    constructor_and_detach_seconds=constructor,cold_complete_outer_seconds=constructor+end-start)
                record(output,'prime.jsonl',row,counts,'prime_requests',seen)
                check_output(row,references[str(length)],tokenizer.eos_token_id);check_pair(quality[key],row);graph_gate(row,cold=True)
                records[key]=dict(mode=mode,length=length,owner_id=owner_id,owner_start=tick,
                    owner_signature=row['owner_signature'],constructor_and_detach_seconds=constructor,
                    prime_request_id=row['request_id'],prime_tokens=row['tokens'],cold_prime_outer_seconds=row['outer_seconds'],
                    warm_tokens=0,warm_outer_seconds_sum=0.,warm_request_ids=[],profile_request_ids=[],profile_outer_seconds_sum=0.,profile_tokens=0)
                progress('cold_prime_complete',counts,mode=mode,length=length,
                    draft_captures=row['draft_runtime']['request']['captures'],coverage_captures=row.get('coverage_runtime',{}).get('request',{}).get('captures',0))
            for item_ in (r for r in plan if r['length']==length):
                key=item_['mode'],length;engine=active[key];owner=records[key]
                gpu_snapshot();start=time.perf_counter();row=group.generate(engine,inputs[length],tokenizer.eos_token_id);end=time.perf_counter()
                enrich(row,engine,key[0],length,references[str(length)])
                row.update(**item_,phase='warm',owner_id=owner['owner_id'],performance_eligible=True,
                    outer_seconds=end-start,outer_tps=row['tokens']/(end-start))
                record(output,'runtime.jsonl',row,counts,'timing_requests',seen)
                check_output(row,references[str(length)],tokenizer.eos_token_id);check_pair(quality[key],row);graph_gate(row)
                require(row['owner_signature']==owner['owner_signature'],'Private cache/Draft/Coverage runtime owner changed')
                owner['warm_tokens']+=row['tokens'];owner['warm_outer_seconds_sum']+=row['outer_seconds'];owner['warm_request_ids'].append(row['request_id'])
                progress('warm_complete',counts,**item_)
            contexts.append(dict(length=length,unprofiled_block_wall_seconds=time.perf_counter()-context_start,
                timing_client_tokens=sum(r['prime_tokens']+r['warm_tokens'] for k,r in records.items() if k[1]==length),
                scope='This context constructor+cold+warm block plus audit I/O; excludes deferred closes and all later diagnostic profiles; not per-mode serving wall'))
        require(counts['timing_requests']==8,'All unprofiled warm timings must finish before diagnostics')
        progress('unprofiled_timing_complete',counts)
        for length in LENGTHS:
            for mode in MODES:
                key=mode,length;engine=active[key];owner=records[key]
                gpu_snapshot();start=time.perf_counter()
                with profile_only(engine):
                    row=group.generate(engine,inputs[length],tokenizer.eos_token_id)
                end=time.perf_counter()
                enrich(row,engine,mode,length,references[str(length)])
                row.update(phase='profile',owner_id=owner['owner_id'],performance_eligible=False,
                    diagnostic_only=True,profile_outer_seconds=end-start,profile_enabled=True)
                record(output,'profiles.jsonl',row,counts,'profile_requests',seen)
                check_output(row,references[str(length)],tokenizer.eos_token_id);check_pair(quality[key],row);graph_gate(row)
                require(row['owner_signature']==owner['owner_signature'],'Diagnostic run changed persistent graph/cache owners')
                require(row.get('timeline') is not None and row.get('cost_samples') is not None,'Missing requested diagnostic timeline/cost samples')
                owner['profile_request_ids'].append(row['request_id']);owner['profile_outer_seconds_sum']+=end-start;owner['profile_tokens']+=row['tokens']
                progress('profile_complete',counts,mode=mode,length=length)
        for key,engine in active.items():
            records[key].update(final_runtime_stats=runtime_snapshot(engine),
                final_coverage_runtime_stats=coverage_snapshot(engine))
            start=time.perf_counter();close(engine);end=time.perf_counter()
            owner=finish_owner_record(records[key],end-start,end)
            owner.update(closed_runtime_stats=runtime_snapshot(engine),
                closed_coverage_runtime_stats=coverage_snapshot(engine),engine_closed=engine.closed)
            require(owner['engine_closed'] and owner['closed_runtime_stats']['entries']==0 and
                owner['closed_coverage_runtime_stats']['entries']==0,'Closed owner retained a live graph entry')
            append(output/'owners.jsonl',owner);counts['owners']+=1
        for row in contexts:
            append(output/'contexts.jsonl',row)
        require(counts==EXPECTED and len(seen)==22,'Registered22-request pilot incomplete')
        require(source_inventory(root)==sources and {str(p):sha256(p) for p in paths}==assets,'Source/assets changed during pilot')
        group.idle();gpu_snapshot()
        write(output/'COMPLETE.json',dict(passed=True,counts=counts,expected_counts=EXPECTED,
            sources_unchanged=True,assets_unchanged=True,analysis_required=True,
            primary_timing_requests=8,diagnostic_profile_requests=4,
            profile_excluded_from_service=True,scope='n1 coverage execution pilot only; no held-out/SpecExtend win or document-CI claim'))
        progress('pilot_complete',counts)
    except BaseException as error:
        cleanup=[]
        if group is not None:
            for engine in list(live):
                try:
                    group.close(engine)
                except BaseException as failure:
                    cleanup.append(repr(failure))
        write(output/'FAILURE.json',dict(passed=False,error_type=type(error).__name__,error=str(error),
            counts=counts,cleanup_errors=cleanup,scope='Preserve partial evidence; no duplicate, restart or automatic cleanup'))
        raise
    finally:
        ResidentDraftRuntime.MAX_ENTRIES=old_limit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('teacher','student','mapper','policy','cost','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--data-file',type=Path,action='append',required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
