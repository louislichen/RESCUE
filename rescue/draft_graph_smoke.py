"""One-document, fixed-weight resident/CUDA-graph pilot; never a full test."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import importlib
import json
from pathlib import Path
import random
import shutil
import time
import uuid

import torch

from .round_overhead_run import (MAPPER_SHA256, POLICY_SHA256, CorrectnessObserver,
    alias_package, append, check_pair, context, gpu_snapshot, model_files,
    require, sha256, source_inventory, trace_signature, write)


MODES = ('baseline_eager', 'resident', 'cuda_graph')
COST_SHA256 = 'd92c843a84676c0cd6adaf9364bbf7e8c771d56548f1016b5720a3e04f371fa1'
BASELINE_PINS = {
    'attention.py': 'f85ef63dae607996e77f4f77be893bf9ad6ca4a88c784c3ec8bf3c0d360f6aa9',
    'config.py': '716b017666b041ebe89aa5cfa58fd37dd699d49acfae567a83ffef2567121c66',
    'engine.py': 'e3237c531e5a4db6f016f02986bf9c1a8ac5b08026b87a94194e1c7f3ac233e7',
    'features.py': 'eb6b98f27f62693c57c4932aafb673315837c9513f8313290eae0f1773072b73',
    'selection.py': '498b73b2d9a0382f4194862d13e66ade44c0846d34b6a0a725b54ac920e98db1',
    'step_budget.py': '8cfe0993a1e6e5cab0de3faee5c51c95a72fdf67b72ef16b69a84240dfaa6db9',
    'attention_summary.py': 'b1436b5b34beb2b453298419ad39a69319cf2a7ce526bcf095f1cc09a751db2e',
}


def tensor_digest(tensor):
    value = tensor.detach().contiguous().cpu()
    return dict(dtype=str(value.dtype), shape=list(value.shape),
                sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())


def native_digest(native):
    return dict(lengths=list(native.lengths), layers=[dict(key=tensor_digest(native[i][0]),
                value=tensor_digest(native[i][1])) for i in range(len(native))])


@contextmanager
def record_queries(attention_class):
    """Quality-only wrapper, outside graph capture; never used for timing."""
    original, records = attention_class.step, []

    def step(owner, token, native, logits=True):
        result = original(owner, token, native, logits=logits)
        records.append(dict(logits=logits, query=tensor_digest(owner.last_query)))
        return result

    attention_class.step = step
    try:
        yield records
    finally:
        attention_class.step = original


def runtime_snapshot(engine):
    runtime = getattr(engine, 'draft_runtime', None)
    if runtime is None:
        return dict(mode='eager', captures=0, replays=0, resident_steps=0,
                    capture_host_seconds=0., entries=0, shadow_updates=0, graph_keys=[])
    return dict(**runtime.stats(), graph_keys=[list(key) for key in runtime._entries])


def check_output(row, reference):
    require(row['generated'] == reference and row['output_agrees'], 'Canonical output mismatch')
    require(row['tokens'] == 32 and row['blocks'] == len(row['trace']), 'Unexpected registered output/round count')
    require(row['accepted'] == sum(t['accepted'] for t in row['trace']), 'Acceptance accounting mismatch')
    require(row['proposed'] == sum(t['proposed'] for t in row['trace']), 'Proposal accounting mismatch')


def sources(before, after):
    inventories = dict(baseline_eager=source_inventory(before), new=source_inventory(after))
    for name, digest in BASELINE_PINS.items():
        require(inventories['baseline_eager'].get('rescue/'+name) == digest,
                'Baseline must be the previously measured v1 after tree: '+name)
    for name, digest in inventories['baseline_eager'].items():
        if name not in ('rescue/engine.py', 'rescue/attention.py', 'rescue/config.py', 'rescue/__main__.py'):
            require(inventories['new'].get(name) == digest, 'Unexpected shared runtime change: '+name)
    return inventories


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
    require(not output.exists(), 'Fresh output required; never reuse a partial pilot')
    for name in ('baseline_root', 'teacher', 'student'):
        require(not output.is_relative_to(getattr(args, name).resolve()), 'Output overlaps an immutable source/model root')
    output.mkdir(parents=True)
    counts = dict(reference=0, correctness=0, timing=0, sessions=0)
    try:
        for value in vars(args).values():
            if isinstance(value, Path):
                require(value.is_absolute() and value.exists(), 'Paths must be explicit existing absolute paths')
        before = args.baseline_root.resolve()
        require(before != root, 'Baseline root must be isolated from new source')
        require(sha256(args.mapper) == MAPPER_SHA256 and sha256(args.policy) == POLICY_SHA256,
                'Frozen mapper/policy checkpoint mismatch')
        require(sha256(args.cost) == COST_SHA256, 'Do not recalibrate/replace the registered cost model')
        inventories = sources(before, root)
        old = alias_package(before, '_rescue_draft_graph_smoke_before')
        for name in ('engine', 'config', 'step_budget', 'query_mapper', 'features', 'selection'):
            require(Path(old[name].__file__).resolve() == before/'rescue'/f'{name}.py', 'Baseline alias escaped source root')
        alias = old['engine'].__package__
        import sys
        for name, module in list(sys.modules.items()):
            if name == alias or name.startswith(alias+'.'):
                require(Path(module.__file__).resolve().is_relative_to(before/'rescue'), 'Baseline submodule escaped source root')
        for relative in ('exp/attention_horizon/selection.py', 'exp/selective_kv/bridge.py'):
            module = importlib.import_module(relative[:-3].replace('/', '.'))
            require(Path(module.__file__).resolve() == root/relative and
                    sha256(module.__file__) == sha256(before/relative), 'Shared exp source binding mismatch')
        docs = [json.loads(line) for line in args.data_file.read_text().splitlines() if line.strip()]
        docs = [r for r in docs if r['source_index'] == 303]
        require(len(docs) == 1 and docs[0].get('split') != 'test', 'Exactly registered development doc303 is required')
        item = docs[0]; ids = context(item, 4096)
        cost = json.loads(args.cost.read_text())
        mapper_payload = torch.load(args.mapper, map_location='cpu', weights_only=True)
        policy_payload = torch.load(args.policy, map_location='cpu', weights_only=True)
        require(policy_payload.get('kind') == 'step_budget', 'Expected fixed StepBudget checkpoint')
        assets_paths = sorted(set(model_files(args.teacher)+model_files(args.student)+[
            args.mapper, args.policy, args.cost, args.data_file]))
        assets = {str(p): sha256(p) for p in assets_paths}
        for arm, inventory in inventories.items():
            owner = before if arm == 'baseline_eager' else root
            for relative in inventory:
                target = output/'source'/arm/relative; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(owner/relative, target)
        torch.set_num_threads(4); torch.manual_seed(619)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        environment = gpu_snapshot()
        teacher, student, tokenizer = load_models(str(args.teacher), str(args.student), 'cuda:0',
                                                   target_dtype=torch.float32, efficient=True)
        pack_projections(teacher); pack_projections(student)
        config = SystemConfig(selector='hierarchical', coverage_rows=64, refresh_every=4,
            refresh_policy='periodic', injection_layers=(27,), inject_tokens=64,
            prefill_precision='bfloat16', max_new_tokens=32, max_draft=12,
            length_policy='risk', schedule='serial', profile=False)
        baseline_config = old['config'].SystemConfig(**{k: v for k, v in asdict(config).items()
                                                      if k not in ('draft_execution', 'coverage_execution')})
        configs = dict(baseline_eager=baseline_config, resident=replace(config, draft_execution='resident'),
                       cuda_graph=replace(config, draft_execution='cuda_graph'))
        engines = dict(baseline_eager=old['engine'].RescueEngine, resident=RescueEngine, cuda_graph=RescueEngine)
        mappers = dict(baseline_eager=old['query_mapper'].mapper_from_payload(mapper_payload, 'cuda:0'),
                       resident=mapper_from_payload(mapper_payload, 'cuda:0'), cuda_graph=mapper_from_payload(mapper_payload, 'cuda:0'))
        policies = dict(baseline_eager=old['step_budget'].StepBudget.from_payload(policy_payload, 'cuda:0'),
                        resident=StepBudget.from_payload(policy_payload, 'cuda:0'),
                        cuda_graph=StepBudget.from_payload(policy_payload, 'cuda:0'))
        for model in (teacher, student, *mappers.values(), *policies.values()):
            model.eval().requires_grad_(False)
            require(all(p.device == torch.device('cuda:0') and not p.requires_grad for p in model.parameters()), 'Frozen single-GPU ownership mismatch')
        order = []
        for repeat in range(2):
            modes = list(MODES); random.Random(619+repeat).shuffle(modes)
            order.extend(dict(repeat=repeat, mode=mode) for mode in modes)
        write(output/'protocol.json', dict(schema=1, scope='One-document execution pilot, no training/held-out/generalization or SpecExtend claim',
            source_index=303, length=4096, max_new_tokens=32, repeats=2, modes=list(MODES), config=asdict(config),
            environment=environment, source_sha256=inventories, assets_sha256=assets,
            mapper_sha256=MAPPER_SHA256, policy_sha256=POLICY_SHA256, cost_sha256=COST_SHA256,
            mapper_frozen=True, training_epochs=0, no_training=True, session_order=order,
            baseline='Previously measured round_overhead v1 after tree, not bare 24f63f9',
            allowed_source_changes='Draft runtime integration in engine/attention/config and CLI --draft-execution wiring; no algorithm or weight change',
            input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(), document_sha256=item['document_sha256'],
            expected_requests=dict(reference=1, correctness=3, timing=12),
            cold_primary='Outer constructor + generate, no graph borrowed from correctness; close deferred until warm request',
            warm_primary='Outer generate using same engine/cache/graph owner',
            session_primary='Wall clock from before constructor until close after warm; includes intervening raw flush and safety checks',
            inner_timing='Engine total_seconds retained as secondary, not complete cold primary',
            ci='No document-level confidence interval or significance estimate: one document, two timing repeats',
            args={k: str(v) for k, v in vars(args).items()}))
        reference_config = replace(config, length_policy='fixed')
        gpu_snapshot()
        with BaselineEngine(teacher, student, reference_config, vocab_size=len(tokenizer)) as engine:
            reference_row = engine.reference(ids, tokenizer.eos_token_id)
        reference_row.update(request_id=uuid.uuid4().hex, mode='canonical_ar', source_index=303, length=4096)
        append(output/'reference.jsonl', reference_row); counts['reference'] += 1
        reference = reference_row['generated']; require(len(reference) == 32, 'Registered output did not reach32; do not pad/reselect')
        write(output/'reference.json', reference)

        def new_engine(mode, observer=None):
            return engines[mode](teacher, student, configs[mode], mappers[mode], policies[mode], cost,
                                 vocab_size=len(tokenizer), round_observer=observer)

        quality = {}
        for mode in MODES:
            gpu_snapshot(); observer = CorrectnessObserver(); engine = new_engine(mode, observer)
            attention_class = old['engine'].DraftAttention if mode == 'baseline_eager' else DraftAttention
            try:
                with record_queries(attention_class) as queries:
                    row = engine.generate(ids, tokenizer.eos_token_id)
                row.update(mode=mode, source_index=303, length=4096,
                    output_agrees=row['generated'] == reference, observed_rounds=observer.rounds,
                    nonzero_injection=observer.nonzero_injection, queries=queries,
                    native_t=native_digest(engine.native_t), native_s=native_digest(engine.native_s),
                    runtime_stats=runtime_snapshot(engine))
                append(output/'correctness.jsonl', row); counts['correctness'] += 1
                check_output(row, reference); require(observer.nonzero_injection, 'No actual nonzero V correction observed')
                require(len(queries) == row['draft_forwards'], 'Missing Draft query snapshots')
                if mode == 'cuda_graph':
                    require(row['runtime_stats']['captures'] > 0 and row['runtime_stats']['replays'] > 0, 'Graph was not exercised')
                if quality:
                    base = quality['baseline_eager']; check_pair(base, row)
                    for name in ('observed_rounds', 'queries', 'native_t', 'native_s'):
                        require(base[name] == row[name], 'Instrumented '+name+' differs')
                quality[mode] = row
            finally:
                engine.close()
        require(sources(before, root) == inventories and {str(p): sha256(p) for p in assets_paths} == assets,
                'Source/assets changed before timing')
        write(output/'QUALITY_PASSED.json', dict(passed=True, requests=3, exact_native_and_queries=True, actual_nonzero=True))
        for entry in order:
            mode, repeat = entry['mode'], entry['repeat']; gpu_snapshot()
            session_start = time.perf_counter(); engine = new_engine(mode)
            constructor_end = time.perf_counter(); rows = []
            try:
                for phase in ('cold', 'warm'):
                    start = session_start if phase == 'cold' else time.perf_counter()
                    row = engine.generate(ids, tokenizer.eos_token_id)
                    end = time.perf_counter()
                    row.update(mode=mode, repeat=repeat, phase=phase, source_index=303, length=4096,
                        outer_seconds=end-start, outer_tps=row['tokens']/(end-start),
                        constructor_seconds=constructor_end-session_start if phase == 'cold' else 0.,
                        output_agrees=row['generated'] == reference, runtime_stats=runtime_snapshot(engine))
                    append(output/'runtime.jsonl', row); counts['timing'] += 1; rows.append(row)
                    check_output(row, reference); check_pair(quality['baseline_eager'], row)
                    if mode == 'cuda_graph':
                        delta = row['draft_runtime']['request']
                        require(delta['replays'] > 0 and (delta['captures'] > 0 if phase == 'cold' else delta['captures'] == 0),
                                'Cold graph was reused or warm graph unexpectedly recaptured')
            finally:
                close_start = time.perf_counter(); engine.close(); session_end = time.perf_counter()
            require(len(rows) == 2, 'Session did not complete both requests')
            append(output/'sessions.jsonl', dict(mode=mode, repeat=repeat, tokens=sum(r['tokens'] for r in rows),
                request_ids=[r['request_id'] for r in rows], session_seconds=session_end-session_start,
                session_tps=sum(r['tokens'] for r in rows)/(session_end-session_start),
                constructor_seconds=constructor_end-session_start, close_seconds=session_end-close_start,
                active_sum_seconds=sum(r['outer_seconds'] for r in rows)+session_end-close_start))
            counts['sessions'] += 1; gpu_snapshot()
        require(counts == dict(reference=1, correctness=3, timing=12, sessions=6), 'Incomplete pilot matrix')
        require(sources(before, root) == inventories and {str(p): sha256(p) for p in assets_paths} == assets, 'Sources/assets changed')
        write(output/'COMPLETE.json', dict(passed=True, counts=counts, sources_unchanged=True, assets_unchanged=True,
            analysis_required=True, scope='Only one-doc4K32-token pilot; no six-doc8K conclusion'))
    except BaseException as error:
        write(output/'FAILURE.json', dict(passed=False, error_type=type(error).__name__, error=str(error), counts=counts))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-root', 'teacher', 'student', 'mapper', 'policy', 'cost', 'data-file', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
