"""Paired, persistent-owner execution benchmark; frozen models, no training.

Only existing development documents are used. Three execution modes retain
the same nonzero Mapper and 2/4 policy. A shared-model hook handoff is included
in every outer request clock; graph/cache owners survive between requests.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import importlib
import json
from pathlib import Path
import random
import shutil
import sys
import time
import uuid

import torch

from .draft_graph_smoke import (MODES, COST_SHA256, sources, native_digest,
                                record_queries, runtime_snapshot)
from .round_overhead_run import (MAPPER_SHA256, POLICY_SHA256, SOURCE_INDICES,
    LENGTHS, REPEATS, CorrectnessObserver, alias_package, append, check_pair,
    context, gpu_snapshot, model_files, read_documents, require, sha256, write)


EXPECTED = dict(reference_requests=12, correctness_requests=36,
                prime_requests=6, timing_requests=108, owners=6)
SEED = 20261020


def progress(stage, counts, **fields):
    print(json.dumps(dict(stage=stage, counts=dict(counts), **fields)), flush=True)


def record(output, name, row, counts, counter, seen):
    """Persist even an invalid result before failing the strict UUID gate."""
    append(output/name, row); counts[counter] += 1
    identity = row.get('request_id')
    require(isinstance(identity, str) and identity and identity not in seen,
            'Missing/duplicate logical request UUID')
    seen.add(identity)


def schedule(seed=SEED):
    """Each document/repeat has three adjacent, randomly ordered modes."""
    plan = []
    for length in LENGTHS:
        rng = random.Random(seed+length)
        for repeat in range(REPEATS):
            indices = list(SOURCE_INDICES); rng.shuffle(indices)
            for index in indices:
                modes = list(MODES); rng.shuffle(modes)
                plan.extend(dict(mode=m, source_index=index, length=length,
                                 repeat=repeat) for m in modes)
    return plan


class HookOwners:
    """Serially lease the original model hooks to one persistent Engine.

    Constructors otherwise stack TargetObservation wrappers on a shared
    Teacher. Do not keep any installed wrapper between requests. Installation,
    restoration and the guards here are charged to outer request timing.
    """
    def __init__(self, teacher, student):
        self.teacher, self.student = teacher, student
        self.modules = [teacher.model.layers[-1].self_attn]+[
            layer.self_attn for layer in student.model.layers]
        self.originals = [module.forward for module in self.modules]
        self.engines = []

    def idle(self):
        require(all(module.forward is original for module, original in
                    zip(self.modules, self.originals)), 'Shared model hook escaped its original efficient callable')
        require(all(not owner.observation.originals for owner in self.engines),
                'Inactive persistent owner retains Target hooks')

    def add(self, engine):
        require(engine.teacher is self.teacher and engine.student is self.student,
                'Persistent owner uses different base model objects')
        require(all(not e.observation.originals for e in self.engines), 'Nested Target hook owner')
        bindings = engine.observation.originals
        require(len(bindings) == 1 and bindings[0][0] is self.modules[0] and
                bindings[0][1] is self.originals[0], 'Constructor nested or replaced the original Target callable')
        engine.observation.restore()
        self.engines.append(engine)
        self.idle()

    def generate(self, engine, ids, eos):
        self.idle()
        require(engine in self.engines and not engine.closed, 'Unregistered/closed persistent owner')
        engine.observation.install()
        try:
            return engine.generate(ids, eos)
        finally:
            engine.observation.restore()
            self.idle()

    def close(self, engine):
        self.idle()
        try:
            engine.close()
        finally:
            self.idle()
            if engine in self.engines:
                self.engines.remove(engine)


def owner_signature(engine):
    """Physical addresses are diagnostics, not a cross-mode equality test."""
    require(engine.native_t is not None and engine.native_s is not None and engine.working is not None,
            'Owner has no allocated cache')
    result = {}
    for name in ('native_t', 'native_s', 'working'):
        cache = getattr(engine, name)
        result[name] = dict(identity=id(cache), key_ptr=cache.key_storage.data_ptr(),
                            value_ptr=cache.value_storage.data_ptr(), capacity=cache.capacity)
    runtime = getattr(engine, 'draft_runtime', None)
    if runtime is not None:
        expected = (engine.native_s.key_storage.data_ptr(), engine.native_s.value_storage.data_ptr(),
                    engine.working.key_storage.data_ptr(), engine.working.value_storage.data_ptr())
        require(tuple(runtime._owners) == expected and runtime.native is engine.native_s and
                runtime.working is engine.working and runtime.injector is engine.injector,
                'Draft runtime/cache/injector ownership changed')
        result['runtime'] = dict(identity=id(runtime), bound_ptrs=list(runtime._owners))
    return result


def check_output(row, reference, eos=None):
    require(row.get('generated') == reference and row.get('output_agrees') is True,
            'Generated tokens differ from the independent Target reference')
    require(0 < row['tokens'] == len(reference) <= 128 and row['blocks'] == len(row['trace']),
            'Invalid max128/EOS/round accounting')
    if eos is not None and eos in reference:
        require(reference[-1] == eos and eos not in reference[:-1], 'EOS boundary changed')
    if row['tokens'] < 128:
        require(eos is not None and reference[-1] == eos, 'Short output without an EOS termination')
    require(row['accepted'] == sum(t['accepted'] for t in row['trace']) and
            row['proposed'] == sum(t['proposed'] for t in row['trace']), 'Acceptance accounting mismatch')
    for i, trace in enumerate(row['trace']):
        require(trace['round_index'] == i and trace['terminal'] is (i == row['blocks']-1),
                'Invalid round/terminal sequence')
        require(1 <= trace['proposed'] <= trace['cap'] <= 4 and
                0 <= trace['accepted'] <= trace['proposed'], 'Invalid causal 2/4 proposal trajectory')


def frozen_assets(paths):
    return {str(path): sha256(path) for path in paths}


def check_bindings(old, before, root):
    alias = old['engine'].__package__
    for name, module in list(sys.modules.items()):
        if name == alias or name.startswith(alias+'.'):
            require(Path(module.__file__).resolve().is_relative_to(before/'rescue'), 'Old alias escaped its frozen source root')
    for part in ('engine', 'config', 'query_mapper', 'step_budget', 'features', 'selection'):
        require(Path(old[part].__file__).resolve() == before/'rescue'/f'{part}.py', 'Old module binding mismatch')
        module = importlib.import_module('rescue.'+part)
        require(Path(module.__file__).resolve() == root/'rescue'/f'{part}.py', 'New module binding mismatch')
    for relative in ('exp/attention_horizon/selection.py', 'exp/selective_kv/bridge.py'):
        module = importlib.import_module(relative[:-3].replace('/', '.'))
        require(Path(module.__file__).resolve() == root/relative and
                sha256(module.__file__) == sha256(before/relative), 'Shared exp dependency differs')


def run(args):
    from .attention import DraftAttention
    from .baselines import BaselineEngine
    from .config import SystemConfig
    from .engine import RescueEngine
    from .loading import load_models
    from .packing import pack_projections
    from .query_mapper import mapper_from_payload
    from .step_budget import StepBudget

    root, output = Path(__file__).resolve().parents[1], args.output.resolve()
    require(not output.exists(), 'Fresh-only output; never resume/overwrite a partial benchmark')
    for name in ('baseline_root', 'teacher', 'student'):
        require(not output.is_relative_to(getattr(args, name).resolve()), 'Output overlaps an immutable source/model root')
    output.mkdir(parents=True)
    counts = dict.fromkeys(EXPECTED, 0); group = None; seen = set()
    try:
        paths = [args.baseline_root, args.teacher, args.student, args.mapper, args.policy, args.cost, *args.data_file]
        require(all(p.is_absolute() and p.exists() for p in paths), 'All assets/source/data paths must be explicit, existing and absolute')
        before = args.baseline_root.resolve()
        require(before != root, 'Baseline and treatment source roots must differ')
        require(sha256(args.mapper) == MAPPER_SHA256 and sha256(args.policy) == POLICY_SHA256,
                'Frozen nonzero Mapper/StepBudget checkpoint mismatch')
        require(sha256(args.cost) == COST_SHA256, 'Frozen costs changed; do not recalibrate')
        inventories = sources(before, root)
        old = alias_package(before, '_rescue_draft_graph_benchmark_before')
        check_bindings(old, before, root)
        documents = read_documents(args.data_file)
        inputs = {(index, length): context(documents[index], length)
                  for index in SOURCE_INDICES for length in LENGTHS}
        asset_paths = sorted(set(model_files(args.teacher)+model_files(args.student)+[
            args.mapper, args.policy, args.cost, *args.data_file]))
        assets = frozen_assets(asset_paths)
        for role, inventory in inventories.items():
            owner = before if role == 'baseline_eager' else root
            for name in inventory:
                target = output/'source'/role/name; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(owner/name, target)
        cost = json.loads(args.cost.read_text())
        mapper_payload = torch.load(args.mapper, map_location='cpu', weights_only=True)
        policy_payload = torch.load(args.policy, map_location='cpu', weights_only=True)
        require(policy_payload.get('kind') == 'step_budget', 'Expected frozen causal StepBudget')
        torch.set_num_threads(4); torch.manual_seed(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        environment = gpu_snapshot()
        teacher, student, tokenizer = load_models(str(args.teacher), str(args.student), 'cuda:0',
                                                   target_dtype=torch.float32, efficient=True)
        pack_projections(teacher); pack_projections(student)
        group = HookOwners(teacher, student)
        config = SystemConfig(selector='hierarchical', coverage_rows=64, refresh_every=4,
            refresh_policy='periodic', injection_layers=(27,), inject_tokens=64,
            prefill_precision='bfloat16', max_new_tokens=128, max_draft=12,
            length_policy='risk', schedule='serial', profile=False)
        baseline_config = old['config'].SystemConfig(**{k: v for k, v in asdict(config).items()
                                                      if k not in ('draft_execution', 'coverage_execution')})
        configs = dict(baseline_eager=baseline_config, resident=replace(config, draft_execution='resident'),
                       cuda_graph=replace(config, draft_execution='cuda_graph'))
        engines = dict(baseline_eager=old['engine'].RescueEngine, resident=RescueEngine, cuda_graph=RescueEngine)
        mappers = {mode: (old['query_mapper'].mapper_from_payload if mode == 'baseline_eager' else mapper_from_payload)(
                          mapper_payload, 'cuda:0') for mode in MODES}
        policies = {mode: (old['step_budget'].StepBudget if mode == 'baseline_eager' else StepBudget).from_payload(
                           policy_payload, 'cuda:0') for mode in MODES}
        for model in (teacher, student, *mappers.values(), *policies.values()):
            model.eval().requires_grad_(False)
            require(all(p.device == torch.device('cuda:0') and not p.requires_grad for p in model.parameters()),
                    'Models/checkpoints must be frozen on the one visible GPU')
        plan = schedule()
        write(output/'protocol.json', dict(schema=1, scope='Execution-only development benchmark; no new algorithm/training/test resume or SpecExtend claim',
            source_indices=list(SOURCE_INDICES), lengths=list(LENGTHS), repeats=REPEATS, modes=list(MODES),
            max_new_tokens=128, max_draft=12, config=asdict(config), seed=SEED,
            environment=environment, source_sha256=inventories, assets_sha256=assets,
            mapper_sha256=MAPPER_SHA256, policy_sha256=POLICY_SHA256, cost_sha256=COST_SHA256,
            mapper_nonzero=True, mapper_frozen=True, policy_frozen=True, training_epochs=0, no_training=True,
            expected_counts=EXPECTED, expected_requests=162, timing_order=plan, context_order=list(LENGTHS),
            cold_prime=dict(source_index=303, requests_per_owner=1, retained_in_accounted_service=True),
            request_buffer_policy='Persistent per-context per-mode owners; same-capacity reset, no graph borrowed from correctness',
            hook_policy='Single active shared-Teacher hook owner: install/generate/restore and identity guards included in outer timing',
            primary='Warm outer generate including symmetric hook handoff; per-document paired 3-repeat pool',
            accounted_service='Constructor/detach + cold-prime outer +18 warm outer +close; measured non-contiguous service sum, NOT production/continuous session wall',
            context_wall='All three modes plus guard/audit I/O; diagnostic only, never per-mode session TPS',
            baseline='Previously measured round_overhead v1 after source tree, not bare24f63f9',
            input_sha256={f'{i}:{n}': hashlib.sha256(json.dumps(v).encode()).hexdigest() for (i,n),v in inputs.items()},
            document_sha256={str(i): documents[i]['document_sha256'] for i in SOURCE_INDICES},
            args={k: [str(p) for p in v] if isinstance(v, list) else str(v) for k,v in vars(args).items()}))
        progress('protocol_frozen', counts, expected_counts=EXPECTED)

        def new_engine(mode, observer=None):
            group.idle()
            engine = engines[mode](teacher, student, configs[mode], mappers[mode], policies[mode], cost,
                vocab_size=len(tokenizer), round_observer=observer)
            group.add(engine)
            return engine

        references, quality_base = {}, {}
        graph_keys = {length: set() for length in LENGTHS}
        # No persistent timing owner exists until all 48 independent quality
        # requests have completed and the source/asset gate passes again.
        for length in LENGTHS:
            for index in SOURCE_INDICES:
                gpu_snapshot(); group.idle()
                with BaselineEngine(teacher, student, replace(config, length_policy='fixed'), vocab_size=len(tokenizer)) as engine:
                    reference = engine.reference(inputs[index, length], tokenizer.eos_token_id)
                reference.update(request_id=uuid.uuid4().hex, mode='canonical_ar', source_index=index, length=length)
                record(output, 'reference.jsonl', reference, counts, 'reference_requests', seen)
                require(0 < len(reference['generated']) <= 128, 'Invalid independent AR output length')
                references[f'{index}:{length}'] = reference['generated']; group.idle()
                progress('reference_complete', counts, source_index=index, length=length)
        write(output/'references.json', references)
        for length in LENGTHS:
            for index in SOURCE_INDICES:
                base = None
                for mode in MODES:
                    gpu_snapshot(); observer = CorrectnessObserver(); engine = new_engine(mode, observer)
                    attention_class = old['engine'].DraftAttention if mode == 'baseline_eager' else DraftAttention
                    try:
                        with record_queries(attention_class) as queries:
                            row = group.generate(engine, inputs[index, length], tokenizer.eos_token_id)
                        row.update(mode=mode, source_index=index, length=length, owner_id=uuid.uuid4().hex,
                            output_agrees=row['generated'] == references[f'{index}:{length}'],
                            observed_rounds=observer.rounds, nonzero_injection=observer.nonzero_injection,
                            queries=queries, native_t=native_digest(engine.native_t), native_s=native_digest(engine.native_s),
                            runtime_stats=runtime_snapshot(engine), owner_signature=owner_signature(engine))
                        record(output, 'correctness.jsonl', row, counts, 'correctness_requests', seen)
                        check_output(row, references[f'{index}:{length}'], tokenizer.eos_token_id)
                        require(observer.nonzero_injection and len(queries) == row['draft_forwards'], 'Missing real nonzero injection/query evidence')
                        if mode == 'cuda_graph':
                            require(row['runtime_stats']['captures'] > 0 and row['runtime_stats']['replays'] > 0,
                                    'Correctness did not exercise the graph')
                            graph_keys[length].update(json.dumps(v, sort_keys=True) for v in row['runtime_stats']['graph_keys'])
                        if base is not None:
                            check_pair(base, row)
                            for name in ('observed_rounds', 'queries', 'native_t', 'native_s'):
                                require(base[name] == row[name], 'Exact instrumented '+name+' mismatch')
                        else:
                            base = row
                            quality_base[index, length] = row
                        progress('correctness_complete', counts, mode=mode, source_index=index, length=length)
                    finally:
                        group.close(engine)
        require(sources(before, root) == inventories and frozen_assets(asset_paths) == assets,
                'Source or assets changed before timing')
        write(output/'QUALITY_PASSED.json', dict(passed=True, requests=48, exact_native_and_queries=True, actual_nonzero=True,
            graph_key_union={str(n): [json.loads(v) for v in sorted(values)] for n, values in graph_keys.items()}))
        progress('quality_gate_passed', counts)

        benchmark_start = time.perf_counter()
        for length in LENGTHS:
            context_start = time.perf_counter(); active, records = {}, {}
            prime_order = list(MODES); random.Random(SEED+length+1).shuffle(prime_order)
            try:
                for mode in prime_order:
                    gpu_snapshot(); constructor_start = time.perf_counter()
                    engine = new_engine(mode); constructor_seconds = time.perf_counter()-constructor_start
                    active[mode] = engine; owner_id = uuid.uuid4().hex
                    start = time.perf_counter(); row = group.generate(engine, inputs[303, length], tokenizer.eos_token_id)
                    end = time.perf_counter(); signature = owner_signature(engine)
                    row.update(mode=mode, source_index=303, length=length, phase='cold_prime', owner_id=owner_id,
                        output_agrees=row['generated'] == references[f'303:{length}'],
                        outer_seconds=end-start, outer_tps=row['tokens']/(end-start),
                        constructor_and_detach_seconds=constructor_seconds,
                        cold_complete_outer_seconds=constructor_seconds+end-start,
                        runtime_stats=runtime_snapshot(engine), owner_signature=signature,
                        required_graph_keys=[json.loads(v) for v in sorted(graph_keys[length])] if mode == 'cuda_graph' else [])
                    record(output, 'prime.jsonl', row, counts, 'prime_requests', seen)
                    records[mode] = dict(owner_id=owner_id, mode=mode, length=length,
                        constructor_and_detach_seconds=constructor_seconds, prime_request_id=row['request_id'],
                        cold_prime_outer_seconds=row['outer_seconds'], cold_complete_outer_seconds=row['cold_complete_outer_seconds'],
                        prime_tokens=row['tokens'], warm_tokens=0, warm_outer_seconds_sum=0., warm_request_ids=[],
                        owner_signature=signature, prime_runtime_stats=row['runtime_stats'], owner_start=constructor_start)
                    check_output(row, references[f'303:{length}'], tokenizer.eos_token_id)
                    check_pair(quality_base[303, length], row)
                    require(engine.round_observer is None and engine.draft_observer is None and not engine.config.profile,
                            'Timing owner contains an observer/profile')
                    if mode == 'cuda_graph':
                        delta = row['draft_runtime']['request']
                        require(delta['captures'] > 0 and delta['replays'] > 0, 'Timing owner borrowed a quality graph')
                        actual_keys = {json.dumps(v, sort_keys=True) for v in row['runtime_stats']['graph_keys']}
                        require(graph_keys[length] <= actual_keys,
                                'The sole registered doc303 cold prime does not cover observed quality graph-key shapes; no extra warmup permitted')
                    progress('cold_prime_complete', counts, mode=mode, length=length,
                             captures=row.get('draft_runtime', {}).get('request', {}).get('captures', 0))
                for item in (v for v in plan if v['length'] == length):
                    mode, index, repeat = item['mode'], item['source_index'], item['repeat']
                    gpu_snapshot(); engine = active[mode]; owner_record = records[mode]
                    require(owner_signature(engine) == owner_record['owner_signature'], 'Persistent owner/capacity changed before warm request')
                    start = time.perf_counter(); row = group.generate(engine, inputs[index, length], tokenizer.eos_token_id)
                    end = time.perf_counter()
                    row.update(**item, phase='warm', owner_id=owner_record['owner_id'],
                        output_agrees=row['generated'] == references[f'{index}:{length}'],
                        outer_seconds=end-start, outer_tps=row['tokens']/(end-start),
                        runtime_stats=runtime_snapshot(engine), owner_signature=owner_signature(engine))
                    record(output, 'runtime.jsonl', row, counts, 'timing_requests', seen)
                    owner_record['warm_tokens'] += row['tokens']; owner_record['warm_outer_seconds_sum'] += row['outer_seconds']
                    owner_record['warm_request_ids'].append(row['request_id'])
                    check_output(row, references[f'{index}:{length}'], tokenizer.eos_token_id)
                    check_pair(quality_base[index, length], row)
                    require(row['owner_signature'] == owner_record['owner_signature'], 'Persistent cache/runtime owner changed')
                    if mode == 'cuda_graph':
                        delta = row['draft_runtime']['request']
                        require(delta['captures'] == 0 and delta['replays'] > 0,
                                'Warm request captured a new graph or failed to replay; no silent recapture/warmup')
                        require(row['runtime_stats']['graph_keys'] == owner_record['prime_runtime_stats']['graph_keys'],
                                'Graph shape/signature set changed after the one registered cold prime')
                    gpu_snapshot()
                    progress('warm_request_complete', counts, **item)
            finally:
                for mode, engine in active.items():
                    final_runtime = runtime_snapshot(engine)
                    start = time.perf_counter(); group.close(engine); end = time.perf_counter()
                    if mode in records:
                        owner_record = records[mode]; owner_record['close_seconds'] = end-start
                        owner_record['owner_lifetime_wall_seconds'] = end-owner_record.pop('owner_start')
                        seconds = (owner_record['constructor_and_detach_seconds']+owner_record['cold_prime_outer_seconds']+
                                   owner_record['warm_outer_seconds_sum']+owner_record['close_seconds'])
                        owner_record.update(accounted_service_seconds=seconds,
                            accounted_service_tps=(owner_record['prime_tokens']+owner_record['warm_tokens'])/seconds,
                            accounting_scope='Measured non-contiguous active service sum, not production session wall',
                            final_runtime_stats=final_runtime, closed_runtime_stats=runtime_snapshot(engine), hook_restored=True)
                        append(output/'owners.jsonl', owner_record); counts['owners'] += 1
            append(output/'contexts.jsonl', dict(length=length, context_wall_seconds=time.perf_counter()-context_start,
                tokens=sum(r['prime_tokens']+r['warm_tokens'] for r in records.values()),
                mode_count=len(records), scope='All interleaved arms and audit/guard I/O, diagnostic wall only'))
            progress('context_complete', counts, length=length)
        require(counts == EXPECTED, 'Registered 162-request matrix incomplete')
        require(len(seen) == 162, 'Registered logical requests are missing/reused')
        require(sources(before, root) == inventories and frozen_assets(asset_paths) == assets, 'Source/assets changed during benchmark')
        group.idle(); gpu_snapshot()
        write(output/'COMPLETE.json', dict(passed=True, counts=counts, **counts, expected_counts=EXPECTED,
            expected_requests=162, sources_unchanged=True, assets_unchanged=True,
            benchmark_wall_seconds=time.perf_counter()-benchmark_start,
            analysis_required=True, scope='Six existing development documents, no held-out or SpecExtend success claim'))
        progress('benchmark_complete', counts)
    except BaseException as error:
        cleanup = []
        if group is not None:
            for engine in list(group.engines):
                try:
                    group.close(engine)
                except BaseException as close_error:
                    cleanup.append(repr(close_error))
        write(output/'FAILURE.json', dict(passed=False, error_type=type(error).__name__, error=str(error),
            counts=counts, cleanup_errors=cleanup, scope='Do not resume or silently reuse partial results'))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-root', 'teacher', 'student', 'mapper', 'policy', 'cost', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--data-file', type=Path, action='append', required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
