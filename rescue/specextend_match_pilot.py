"""One reused document: matched RESCUE/retrieval/full-tree gap pilot.

No training, test selection or baseline search. Every legal Graph capture is
charged to the request. A COMPLETE run is not a final SpecExtend/SOTA claim.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import random
import shutil
import sys
import time
from types import MethodType
import uuid

import torch

from .draft_graph_benchmark import HookOwners, owner_signature, record
from .draft_graph_smoke import COST_SHA256, native_digest, record_queries, runtime_snapshot
from .round_overhead_run import (MAPPER_SHA256, POLICY_SHA256, append,
    check_pair, context, gpu_snapshot, model_files, require, sha256, write)


LENGTHS = (4096, 8192)
SOURCE_INDEX = 303
REPEATS = 2
ARMS = ('rescue_graph', 'retrieval32_k4_eager', 'retrieval32_k4_resident',
        'retrieval32_k4_graph', 'full32_n8')
QUALITY_ARMS = ('rescue_eager',)+ARMS
EXPECTED = dict(reference_requests=2, correctness_requests=12, prime_requests=10,
                timing_requests=20, owners=10)
SEED = 20261021
TREE_POLICY = dict(nodes=8, top_chunks=32, max_depth=10, threshold=.7,
                   chunk_size=32, refresh_every=4, flash_prefill=True, hybrid_tree=True)


def progress(stage, counts, **extra):
    print(json.dumps(dict(stage=stage, counts=dict(counts), **extra)), flush=True)


def schedule(seed=SEED):
    plan = []
    for length in LENGTHS:
        rng = random.Random(seed+length)
        for repeat in range(REPEATS):
            order = list(ARMS); rng.shuffle(order)
            plan.extend(dict(mode=m, source_index=SOURCE_INDEX, length=length, repeat=repeat) for m in order)
    return plan


def read_document(paths):
    matches = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            # Other documents in a frozen development input file are not run.
            require(item.get('split') != 'test', 'Never open a test input file in this pilot')
            if item.get('source_index') == SOURCE_INDEX:
                matches.append(item)
    require(len(matches) == 1, 'Exactly one non-test doc303 is required')
    item = matches[0]
    digest = item.get('document_sha256', '')
    require(len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'Missing document SHA256')
    for length in LENGTHS:
        context(item, length)
    return item


def source_inventory(root):
    paths = list((root/'rescue').glob('*.py'))+list((root/'exp/specextend_bridgekv/specextend').rglob('*.py'))
    paths += [root/'exp/attention_horizon/selection.py', root/'exp/selective_kv/bridge.py',
              root/'tests/test_rescue_specextend_match_pilot.py']
    require((root/'exp/specextend_bridgekv/specextend/classic/model_classic.py').is_file(), 'Frozen full-tree source must be staged explicitly')
    require(all(p.is_file() for p in paths), 'Missing source/dependency/test file')
    return {str(p.relative_to(root)): sha256(p) for p in sorted(set(paths))}


def check_full_bindings(root):
    root = Path(root).resolve()
    source = (root/'exp/specextend_bridgekv/specextend').resolve()
    found = {}
    for name, module in list(sys.modules.items()):
        if any(name == prefix or name.startswith(prefix+'.') for prefix in ('classic', 'shared')):
            filename = getattr(module, '__file__', None)
            if filename is None:
                # Upstream classic/shared are PEP420 namespace packages.
                paths = list(getattr(module, '__path__', []))
                require(paths and all(Path(p).resolve().is_relative_to(source) for p in paths),
                        'Full SpecExtend namespace escaped its staged source: '+name)
                found[name] = dict(namespace_paths=[str(Path(p).resolve().relative_to(root)) for p in paths])
            else:
                require(Path(filename).resolve().is_relative_to(source),
                        'Full SpecExtend module escaped its staged source: '+name)
                found[name] = dict(path=str(Path(filename).resolve().relative_to(root)), sha256=sha256(filename))
    require('classic.model_classic' in found and 'shared.modeling_llama_kv_target' in found,
            'Full-tree implementation was not imported from the staged source')
    return found


def check_output(row, reference, eos=None):
    require(row.get('generated') == reference and row.get('output_agrees') is True,
            'Client-visible output differs from the independent Target AR')
    require(0 < row.get('tokens', 0) == len(reference) <= 128, 'Invalid requested output accounting')
    if len(reference) < 128:
        require(eos is not None and reference[-1] == eos, 'Short output without canonical EOS')
    if eos is not None and eos in reference:
        require(reference[-1] == eos and eos not in reference[:-1], 'EOS boundary differs')
    if row.get('mode') != 'full32_n8':
        require(row.get('blocks') == len(row.get('trace', [])) > 0, 'Missing linear round trace')
        require(row['accepted'] == sum(r['accepted'] for r in row['trace']) and
                row['proposed'] == sum(r['proposed'] for r in row['trace']), 'Linear acceptance accounting differs')
        for i, trace in enumerate(row['trace']):
            require(1 <= trace['proposed'] <= trace['cap'] <= 4 and 0 <= trace['accepted'] <= trace['proposed'], 'Invalid linear cap/acceptance')
            require(trace['round_index'] == i and trace['terminal'] is (i == row['blocks']-1), 'Invalid linear terminal sequence')


def normalize_full(result, reference, eos=None):
    """Honor the archived client-prefix cap without hiding a wrong token.

    The archived SPModel exposes the scored prefix, raw *count* and overshoot,
    not unscored tail token IDs. Its entire compute interval remains charged.
    """
    row = dict(result)
    upstream = row.get('upstream', {})
    scored = list(upstream.get('output_token_ids', []))
    require(scored and len(scored) <= 128, 'Full tree did not return its registered scored prefix')
    raw_count, overshoot = upstream.get('raw_total_generated'), upstream.get('generation_overshoot')
    require(type(raw_count) is int and type(overshoot) is int and raw_count >= len(scored) and
            overshoot == raw_count-len(scored) and upstream.get('total_generated') == len(scored),
            'Full-tree raw/overshoot/scored accounting differs')
    visible = scored[:128]
    if eos is not None and eos in visible:
        visible = visible[:visible.index(eos)+1]
    require(row.get('generated') == visible, 'Full wrapper and registered EOS/client cap differ')
    row.update(request_id=uuid.uuid4().hex, mode='full32_n8', generated=visible,
        tokens=len(visible), scored_generated_before_eos=scored, raw_total_generated=raw_count,
        generation_overshoot=overshoot, raw_tail_ids_available=False,
        output_agrees=visible == reference, tree_policy=TREE_POLICY,
        blocks=len(upstream.get('accept_length_list', [])),
        accepted=sum(upstream.get('accept_length_list', [])), proposed=None,
        output_policy='Archived max128 client-prefix cap plus first EOS; all returned client tokens checked; full overshoot work charged, tail IDs not exposed')
    require(row['blocks'] > 0 and all(type(v) is int and v >= 0 for v in upstream['accept_length_list']), 'Missing full-tree acceptance counters')
    return row


class Observer:
    step_budget = True

    def __init__(self):
        self.rounds = []
        self.nonzero_injection = False

    def __call__(self, packet):
        vector = packet.get('budget_features')
        vector = None if vector is None else vector.detach().float().cpu().tolist()
        self.nonzero_injection |= vector is not None and vector[-1] > 0
        self.rounds.append(dict(round_index=packet['round_index'], prefix=packet['prefix'],
            progress=packet['progress'], remaining=packet['remaining'],
            selected=packet['selected'].detach().cpu().tolist(), budget_features=vector,
            draft=list(packet['draft']), checks=list(packet['checks']), accepted=packet['accepted']))


def check_quality(base, row):
    check_pair(base, row)
    for name in ('observed_rounds', 'queries', 'native_t', 'native_s'):
        require(base.get(name) and base[name] == row.get(name), 'Matched execution changed '+name)


class FullOwner:
    """Private SPModel request state, shared immutable architecture weights."""
    def __init__(self, template, teacher_path):
        from classic.model_classic import SPModel
        self.engine = SPModel(template.base_model, str(teacher_path), template.draft_model).eval()
        self.engine.init_caches = MethodType(template.init_caches.__func__, self.engine)
        self.engine.prefill_precision = 'bfloat16'
        require(inspect.signature(self.engine.spgenerate).parameters['threshold'].default == .7,
                'Frozen full-tree threshold default changed')
        require(self.engine.attn_bridge_mapper is None and self.engine.attn_bridge_config == {} and
                self.engine.bridge_collector is None, 'Full-tree baseline must not inject RESCUE content')

    def generate(self, ids):
        from .specextend_qwen import generate
        result = generate(self.engine, ids, max_new_tokens=128, nodes=8, top_chunks=32, max_depth=10)
        observed = {name: getattr(self.engine, name, None) for name in (
            'target_use_flash_prefill', 'target_use_hybrid_tree_attn', 'draft_use_flash_prefill',
            'use_retrieval_cache', 'retrieve_top_k', 'retrieval_chunk_size', 'retrieve_every_n_steps')}
        require(observed == dict(target_use_flash_prefill=True, target_use_hybrid_tree_attn=True,
            draft_use_flash_prefill=True, use_retrieval_cache=True, retrieve_top_k=32,
            retrieval_chunk_size=32, retrieve_every_n_steps=4), 'Full tree silently changed retrieval/Flash/hybrid policy')
        result['observed_full_config'] = observed
        return result

    def close(self):
        torch.cuda.synchronize()
        # Break the per-owner bound-method cycle; release request caches while
        # the immutable shared model wrappers remain alive in the template.
        self.engine.init_caches = None
        self.engine = None


def run(args):
    from .attention import DraftAttention
    from .baselines import BaselineEngine
    from .config import SystemConfig
    from .draft_graph import ResidentDraftRuntime
    from .engine import RescueEngine
    from .loading import load_models
    from .packing import share_packed_projections
    from .query_mapper import mapper_from_payload
    from .specextend_qwen import build
    from .step_budget import StepBudget

    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    require(not output.exists(), 'Fresh pilot output required; never resume/overwrite partial evidence')
    output.mkdir(parents=True)
    counts = dict.fromkeys(EXPECTED, 0); seen = set(); group = None
    owners_to_close = []
    old_limit = ResidentDraftRuntime.MAX_ENTRIES
    try:
        require(sha256(args.mapper) == MAPPER_SHA256 and sha256(args.policy) == POLICY_SHA256 and
                sha256(args.cost) == COST_SHA256, 'Expected frozen nonzero Mapper/StepBudget/cost')
        item = read_document(args.data_file)
        inputs = {n: context(item, n) for n in LENGTHS}
        sources = source_inventory(root)
        for name in sources:
            dest = output/'source'/name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root/name, dest)
        paths = sorted(set(model_files(args.teacher)+model_files(args.student)+[
            args.mapper, args.policy, args.cost]+args.data_file))
        assets = {str(p): sha256(p) for p in paths}
        torch.set_num_threads(4); torch.manual_seed(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        environment = gpu_snapshot()
        teacher, student, tokenizer = load_models(str(args.teacher), str(args.student), 'cuda:0',
                                                  target_dtype=torch.float32, efficient=True)
        # Construct the architecture wrappers before packing changes MLP state
        # names, then share exactly the same packed operators and model weights.
        tick = time.perf_counter(); template = build(teacher, student, str(args.teacher))
        template.prefill_precision = 'bfloat16'
        share_packed_projections(teacher, template.base_model)
        share_packed_projections(student, template.draft_model)
        preparation_seconds = time.perf_counter()-tick
        binding = check_full_bindings(root)
        group = HookOwners(teacher, student)
        mapper = mapper_from_payload(torch.load(args.mapper, map_location='cpu', weights_only=True), 'cuda:0')
        policy = StepBudget.from_payload(torch.load(args.policy, map_location='cpu', weights_only=True), 'cuda:0')
        for model in (teacher, student, mapper, policy):
            model.eval().requires_grad_(False)
        cost = json.loads(args.cost.read_text())
        ResidentDraftRuntime.MAX_ENTRIES = 128
        config = SystemConfig(selector='hierarchical', coverage_rows=64, refresh_every=4,
            refresh_policy='periodic', injection_layers=(27,), inject_tokens=64,
            injection_strength=.5, prefill_precision='bfloat16', max_new_tokens=128,
            max_draft=12, length_policy='risk', schedule='serial', profile=False)
        fixed = replace(config, length_policy='fixed', fixed_k=4, inject_tokens=0)
        plan = schedule()
        write(output/'protocol.json', dict(schema=1, scope='Matched gap pilot, n=1 reused development document; NOT final goal or held-out proof',
            source_index=303, document_sha256=item['document_sha256'], lengths=list(LENGTHS), max_new_tokens=128,
            repeats=2, arms=list(ARMS), quality_arms=list(QUALITY_ARMS), expected_counts=EXPECTED,
            expected_requests=44, source_sha256=sources, assets_sha256=assets,
            mapper_sha256=MAPPER_SHA256, policy_sha256=POLICY_SHA256, cost_sha256=COST_SHA256,
            no_training=True, mapper_frozen=True, policy_frozen=True, config=asdict(config),
            retrieval=dict(top_chunks=32, chunk_size=32, fixed_k=4, refresh_every=4, mapper=False),
            tree=TREE_POLICY, tree_source_bindings=binding, environment=environment,
            graph_max_entries=128, rescue_retain_query=True,
            retrieval_retain_query=dict(correctness=True, timing=False),
            input_sha256={str(n): hashlib.sha256(json.dumps(v).encode()).hexdigest() for n,v in inputs.items()},
            model_preparation_seconds=preparation_seconds, model_preparation_scope='Shared model loading, architecture wrappers and packed projection preparation excluded identically; each private serving owner constructor/detach is charged separately',
            output_policy='All128 requested client-prefix tokens plus EOS checked against independent AR; archived full tree may compute an overshoot, retained in upstream raw counts and entirely charged; unexposed tail IDs are not invented',
            graph_timing='Every capture, including legal new warm shapes, charged to full-generate outer; no graph borrowed from quality',
            warm_order=plan, seed=SEED, confidence_intervals=None,
            timing='Outer hook install+complete generate+restore; includes prefill/capture/request cleanup; same-owner2 warm repeats after separately measured cold prime',
            accounting='Constructor/detach+cold prime+2 warm outer+close measured service sum; interleaved wall contains other arms and audit I/O, not production TPS',
            args={k: [str(p) for p in v] if isinstance(v, list) else str(v) for k,v in vars(args).items()}))
        progress('protocol_frozen', counts, expected_counts=EXPECTED)

        def make_owner(mode, observer=None):
            group.idle()
            if mode == 'full32_n8':
                return FullOwner(template, args.teacher)
            if mode.startswith('rescue_'):
                engine = RescueEngine(teacher, student, replace(config, draft_execution='eager' if mode == 'rescue_eager' else 'cuda_graph'),
                    mapper, policy, cost, vocab_size=len(tokenizer), round_observer=observer)
            else:
                backend = mode.rsplit('_', 1)[1]
                engine = BaselineEngine(teacher, student, replace(fixed, draft_execution='cuda_graph' if backend == 'graph' else backend),
                    vocab_size=len(tokenizer), top_chunks=32, round_observer=observer)
            group.add(engine)
            return engine

        def close(owner):
            if isinstance(owner, FullOwner):
                group.idle(); owner.close(); group.idle()
            else:
                group.close(owner)

        def generate(owner, mode, length, reference):
            group.idle()
            if mode == 'full32_n8':
                row = normalize_full(owner.generate(inputs[length]), reference, tokenizer.eos_token_id)
                group.idle()
            else:
                row = group.generate(owner, inputs[length], tokenizer.eos_token_id)
                row.update(mode=mode, output_agrees=row['generated'] == reference)
            row.update(mode=mode, source_index=303, length=length)
            return row

        references, quality = {}, {}
        for length in LENGTHS:
            gpu_snapshot(); group.idle()
            with BaselineEngine(teacher, student, fixed, vocab_size=len(tokenizer)) as engine:
                row = engine.reference(inputs[length], tokenizer.eos_token_id)
            row.update(request_id=uuid.uuid4().hex, mode='canonical_ar', source_index=303, length=length)
            record(output, 'reference.jsonl', row, counts, 'reference_requests', seen)
            references[str(length)] = row['generated']
            progress('reference_complete', counts, length=length)
        write(output/'references.json', references)
        for length in LENGTHS:
            family_base = {}
            for mode in QUALITY_ARMS:
                gpu_snapshot(); observer = None if mode == 'full32_n8' else Observer()
                owner = make_owner(mode, observer); owners_to_close.append(owner)
                try:
                    if mode == 'full32_n8':
                        row = generate(owner, mode, length, references[str(length)])
                    else:
                        with record_queries(DraftAttention) as queries:
                            row = generate(owner, mode, length, references[str(length)])
                        row.update(observed_rounds=observer.rounds, queries=queries,
                            native_t=native_digest(owner.native_t), native_s=native_digest(owner.native_s),
                            nonzero_injection=observer.nonzero_injection, runtime_stats=runtime_snapshot(owner),
                            owner_signature=owner_signature(owner))
                    row.update(owner_id=uuid.uuid4().hex, phase='correctness')
                    record(output, 'correctness.jsonl', row, counts, 'correctness_requests', seen)
                    check_output(row, references[str(length)], tokenizer.eos_token_id)
                    if mode != 'full32_n8':
                        require(len(row['queries']) == row['draft_forwards'], 'Missing every-step Query evidence')
                        family = 'rescue' if mode.startswith('rescue') else 'retrieval'
                        if family in family_base:
                            check_quality(family_base[family], row)
                        else:
                            family_base[family] = row
                        if family == 'rescue':
                            require(row['nonzero_injection'], 'RESCUE must preserve actual nonzero Mapper correction')
                        if mode.endswith('graph'):
                            require(row['runtime_stats']['captures'] > 0 and row['runtime_stats']['replays'] > 0, 'Quality did not exercise real Graphs')
                    quality[mode, length] = row
                    progress('correctness_complete', counts, mode=mode, length=length)
                finally:
                    close(owner); owners_to_close.remove(owner)
        require(source_inventory(root) == sources and {str(p):sha256(p) for p in paths} == assets, 'Source/assets changed before timing')
        write(output/'QUALITY_PASSED.json', dict(passed=True, counts=dict(counts), exact_all_client_tokens=True,
            exact_linear_queries_native_and_trajectories=True, actual_nonzero_rescue=True))
        progress('quality_gate_passed', counts)

        for length in LENGTHS:
            context_start = time.perf_counter(); active, records = {}, {}
            order = list(ARMS); random.Random(SEED+length+9).shuffle(order)
            try:
                for mode in order:
                    gpu_snapshot(); tick = time.perf_counter(); owner = make_owner(mode)
                    constructor_seconds = time.perf_counter()-tick
                    active[mode] = owner; owners_to_close.append(owner); owner_id = uuid.uuid4().hex
                    start = time.perf_counter(); row = generate(owner, mode, length, references[str(length)]); end = time.perf_counter()
                    row.update(phase='cold_prime', owner_id=owner_id, outer_seconds=end-start, outer_tps=row['tokens']/(end-start),
                        constructor_and_detach_seconds=constructor_seconds, cold_complete_outer_seconds=constructor_seconds+end-start)
                    if mode != 'full32_n8':
                        row.update(runtime_stats=runtime_snapshot(owner), owner_signature=owner_signature(owner))
                    record(output, 'prime.jsonl', row, counts, 'prime_requests', seen)
                    check_output(row, references[str(length)], tokenizer.eos_token_id)
                    if mode != 'full32_n8':
                        check_pair(quality[mode,length], row)
                        if mode.endswith('graph'):
                            require(row['draft_runtime']['request']['captures'] > 0, 'Timing Graph owner borrowed a correctness graph')
                    records[mode] = dict(mode=mode, length=length, owner_id=owner_id,
                        constructor_and_detach_seconds=constructor_seconds, prime_request_id=row['request_id'],
                        cold_prime_outer_seconds=row['outer_seconds'], prime_tokens=row['tokens'],
                        warm_tokens=0, warm_outer_seconds_sum=0., warm_request_ids=[],
                        owner_signature=row.get('owner_signature'), owner_start=tick)
                    progress('cold_prime_complete', counts, mode=mode, length=length,
                        captures=row.get('draft_runtime', {}).get('request', {}).get('captures', 0))
                for entry in (v for v in plan if v['length'] == length):
                    mode = entry['mode']; owner = active[mode]; owner_record = records[mode]
                    gpu_snapshot(); start = time.perf_counter(); row = generate(owner, mode, length, references[str(length)]); end = time.perf_counter()
                    row.update(**entry, phase='warm', owner_id=owner_record['owner_id'], outer_seconds=end-start, outer_tps=row['tokens']/(end-start))
                    if mode != 'full32_n8':
                        row.update(runtime_stats=runtime_snapshot(owner), owner_signature=owner_signature(owner))
                    record(output, 'runtime.jsonl', row, counts, 'timing_requests', seen)
                    check_output(row, references[str(length)], tokenizer.eos_token_id)
                    if mode != 'full32_n8':
                        check_pair(quality[mode,length], row)
                        require(row['owner_signature'] == owner_record['owner_signature'], 'Persistent linear cache/Graph owner changed')
                        if mode.endswith('graph'):
                            require(row['draft_runtime']['request']['replays'] > 0, 'Warm Graph did not replay')
                    owner_record['warm_tokens'] += row['tokens']; owner_record['warm_outer_seconds_sum'] += row['outer_seconds']
                    owner_record['warm_request_ids'].append(row['request_id'])
                    progress('warm_complete', counts, **entry,
                        captures=row.get('draft_runtime', {}).get('request', {}).get('captures', 0))
            finally:
                for mode, owner in active.items():
                    start = time.perf_counter(); close(owner); end = time.perf_counter()
                    owners_to_close.remove(owner)
                    if mode in records:
                        owner_record = records[mode]
                        owner_record.update(close_seconds=end-start,
                            owner_lifetime_wall_seconds=end-owner_record.pop('owner_start'), hook_restored=True)
                        seconds = (owner_record['constructor_and_detach_seconds']+owner_record['cold_prime_outer_seconds']+
                                   owner_record['warm_outer_seconds_sum']+owner_record['close_seconds'])
                        owner_record.update(accounted_service_seconds=seconds,
                            accounted_service_tps=(owner_record['prime_tokens']+owner_record['warm_tokens'])/seconds)
                        append(output/'owners.jsonl', owner_record); counts['owners'] += 1
            append(output/'contexts.jsonl', dict(length=length, context_wall_seconds=time.perf_counter()-context_start,
                tokens=sum(v['prime_tokens']+v['warm_tokens'] for v in records.values()), scope='Five interleaved arms plus audit I/O, not per-mode production wall'))
            progress('context_complete', counts, length=length)
        require(counts == EXPECTED and len(seen) == 44, 'Registered44-request pilot incomplete')
        require(source_inventory(root) == sources and {str(p):sha256(p) for p in paths} == assets, 'Source/assets changed during pilot')
        group.idle(); check_full_bindings(root); gpu_snapshot()
        write(output/'COMPLETE.json', dict(passed=True, counts=counts, expected_counts=EXPECTED,
            sources_unchanged=True, assets_unchanged=True, analysis_required=True,
            scope='n1 reused-dev4K/8K128 gap pilot; no document CI, held-out, original Vicuna or final SpecExtend/SOTA success claim'))
        progress('pilot_complete', counts)
    except BaseException as error:
        cleanup_errors = []
        for owner in list(owners_to_close):
            try:
                if isinstance(owner, FullOwner):
                    owner.close()
                elif group is not None:
                    group.close(owner)
            except BaseException as cleanup:
                cleanup_errors.append(repr(cleanup))
        write(output/'FAILURE.json', dict(passed=False, error_type=type(error).__name__, error=str(error),
            counts=counts, cleanup_errors=cleanup_errors, scope='Preserve partial evidence; no automatic resume or duplicate'))
        raise
    finally:
        ResidentDraftRuntime.MAX_ENTRIES = old_limit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('teacher', 'student', 'mapper', 'policy', 'cost', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--data-file', type=Path, action='append', required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
