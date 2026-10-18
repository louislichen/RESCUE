"""Development-selected RESCUE vs SpecExtend only; frozen held-out test matrix."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import torch

from .baselines import BaselineEngine
from .calibration import calibrate
from .compare_baselines import append, write
from .config import SystemConfig
from .engine import RescueEngine
from .loading import ROOT, DEFAULT_TEACHER, DEFAULT_MAPPER, load_models, load_modules
from .long_data import context
from .specextend_qwen import SOURCE, build, generate as full_generate
from .policy import RiskModel
from .packing import share_packed_projections


def summary(rows):
    result = {}
    for length in sorted({r['length'] for r in rows}):
        selected = [r for r in rows if r['length'] == length]
        modes = {}
        for mode in sorted({r['mode'] for r in selected}):
            group = [r for r in selected if r['mode'] == mode]
            tokens = sum(r['tokens'] for r in group)
            modes[mode] = dict(requests=len(group), tokens=tokens,
                total_tps=tokens/sum(r['total_seconds'] for r in group),
                exact_outputs=sum(r['output_agrees'] for r in group),
                blocks=sum(r.get('blocks', len(r.get('upstream', {}).get('accept_length_list', []))) for r in group))
        comparisons = {}
        for candidate in ('rescue_positive', 'rescue_guarded', 'rescue_dynamic'):
            for baseline in ('retrieval', 'full_specextend'):
                first = [r for r in selected if r['mode'] == candidate]
                second = [r for r in selected if r['mode'] == baseline]
                differences = []
                for doc in sorted({r['source_index'] for r in first}):
                    a = statistics.median(r['total_tps'] for r in first if r['source_index'] == doc)
                    b = statistics.median(r['total_tps'] for r in second if r['source_index'] == doc)
                    differences.append(100*(a/b-1))
                rng = random.Random(42)
                draws = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(5000))
                comparisons[candidate+'_vs_'+baseline] = dict(mean_percent=statistics.mean(differences),
                    ci95=[draws[124], draws[4874]], documents=len(differences))
        result[str(length)] = dict(modes=modes, comparisons=comparisons)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--training', type=Path, default=Path('exp/long_system/out_joint_train_v2_20261017'))
    p.add_argument('--development-documents', type=int, default=3)
    p.add_argument('--test-documents', type=int, default=8)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--lengths', type=int, nargs='+', default=[2048, 4096, 8192, 16384])
    p.add_argument('--pack-projections', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a fresh output directory')
    args.output.mkdir(parents=True)
    sources = [*sorted((ROOT/'rescue').glob('*.py')), *sorted((ROOT/'tests').glob('*.py')),
        *sorted(SOURCE.rglob('*.py')), ROOT/'exp/attention_horizon/selection.py', ROOT/'exp/selective_kv/bridge.py']
    hashes = {}
    for path in sources:
        relative = path.relative_to(ROOT)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        dest = args.output/'source'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    directory = ROOT/'exp/long_system/data'
    training = [json.loads(l) for l in (directory/'train.jsonl').read_text().splitlines()]
    development = [json.loads(l) for l in (directory/'development.jsonl').read_text().splitlines()][:args.development_documents]
    base = SystemConfig(selector='hierarchical', coverage_rows=64, length_policy='fixed', fixed_k=4,
        injection_layers=(27,), inject_tokens=64, max_new_tokens=args.max_new_tokens,
        prefill_precision='bfloat16')
    candidates = dict(h64_k4=base, h64_k8=replace(base, fixed_k=8),
        h256_k4=replace(base, coverage_rows=256),
        adapt_k4=replace(base, refresh_policy='adaptive', attention_drift=.2, coverage_floor=.75),
        adapt_k8=replace(base, refresh_policy='adaptive', attention_drift=.2, coverage_floor=.75, fixed_k=8),
        budget1024=replace(base, budget=1024, coverage_candidates=1536),
        noinj_k4=replace(base, inject_tokens=0), noinj_k8=replace(base, inject_tokens=0, fixed_k=8),
        dynamic=replace(base, length_policy='risk'))
    # Both the lighter retrieval control and the complete tree-engine port are
    # selected on development documents, independently of RESCUE's choice.
    retrieval = {f'retrieval{chunks}_k{k}': (chunks, k) for chunks in (16, 32) for k in (2, 4, 8)}
    full = {f'full{chunks}_n{nodes}': (chunks, nodes) for chunks in (16, 32) for nodes in (8, 32)}
    weight_paths = [args.training/'mapper_joint.pt', args.training/'risk_injected.pt']
    write(args.output/'protocol.json', dict(scope='RESCUE vs SpecExtend retrieval and complete Qwen2 tree-engine port; no other baselines',
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, source_sha256=hashes,
        data_sha256={name: hashlib.sha256((directory/(name+'.jsonl')).read_bytes()).hexdigest()
                     for name in ('train', 'development', 'test')},
        checkpoint_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in weight_paths},
        configs={name: asdict(cfg) for name, cfg in candidates.items()}, retrieval=retrieval, full_tree=full,
        development_indices=[r['source_index'] for r in development], tuning_lengths=[8192, 16384],
        selection='Pooled whole-request TPS on development at 8K/16K, positive and zero-allowed RESCUE retained',
        target_dtype='float32', draft_dtype='bfloat16', prefill='BF16 Target prefix excluding final prompt token; FP32 Target decoding',
        tf32=False, full_port='QKV biases, RoPE base, GQA heads/native dtype, common usable vocab; FP32 efficient-SDPA prefix/LSE instead of unsupported FP32 Flash',
        fairness='Same Target/Draft weights, usable vocabulary, matched precision/prompt/output cap and common Target prefill; SpecExtend keeps upstream Draft window-512 Flash prefill and tree/hybrid flow',
        target_reference='Independent same-precision Target AR, correctness only, not another timed performance baseline',
        timing='Fresh buffers and complete request including prefill/control/cleanup; model loading/references/compilation warmups excluded',
        repeats='Timing repeats are not independent samples; paired bootstrap over documents',
        success_gate='>=10% paired whole-request TPS improvement and CI lower bound >0 at both 8K and16K; not a general SOTA claim'))
    torch.set_num_threads(4); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    teacher, student, tokenizer = load_models(efficient=True)
    mapper, _, _ = load_modules(teacher.device, mapper=weight_paths[0])
    risk = RiskModel.load(weight_paths[1], teacher.device)
    full_engine = build(teacher, student, DEFAULT_TEACHER)
    full_engine.prefill_precision = base.prefill_precision
    packing = None
    if args.pack_projections:
        torch.cuda.synchronize()
        tick = time.perf_counter()
        before = torch.cuda.memory_allocated()
        share_packed_projections(teacher, full_engine.base_model)
        share_packed_projections(student, full_engine.draft_model)
        torch.cuda.synchronize()
        packing = dict(seconds=time.perf_counter()-tick, allocated_delta_bytes=torch.cuda.memory_allocated()-before,
                       scope='Common preparation before request timing, applied to both engines')
    write(args.output/'environment.json', dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        teacher_dtype=str(teacher.dtype), draft_dtype=str(student.dtype), packing=packing))
    calibration = [dict(source_index=training[0]['source_index'], ids=context(training[0], n)) for n in (4096, 16384)]
    cost, cost_rows = calibrate(teacher, student, mapper, risk, replace(base, max_new_tokens=32),
        calibration, tokenizer.eos_token_id, len(tokenizer))
    write(args.output/'cost.json', cost); write(args.output/'cost_records.json', cost_rows)
    references = {}
    def reference(item, length):
        key = item['split']+':'+str(item['source_index'])+':'+str(length)
        if key not in references:
            with BaselineEngine(teacher, student, base, vocab_size=len(tokenizer)) as engine:
                references[key] = engine.reference(context(item, length), tokenizer.eos_token_id)['generated']
            write(args.output/'references.json', references)
        return references[key]
    def run(item, length, name, output_limit=None):
        ids = context(item, length)
        limit = args.max_new_tokens if output_limit is None else output_limit
        if name in candidates:
            with RescueEngine(teacher, student, replace(candidates[name], max_new_tokens=limit), mapper, risk, cost, vocab_size=len(tokenizer)) as engine:
                result = engine.generate(ids, tokenizer.eos_token_id)
        elif name in retrieval:
            chunks, k = retrieval[name]
            with BaselineEngine(teacher, student, replace(base, fixed_k=k, max_new_tokens=limit), top_chunks=chunks, vocab_size=len(tokenizer)) as engine:
                result = engine.generate(ids, tokenizer.eos_token_id)
        else:
            chunks, nodes = full[name]
            result = full_generate(full_engine, ids, limit, nodes=nodes, top_chunks=chunks)
        result.update(source_index=item['source_index'], length=length, split=item['split'], candidate=name,
            output_agrees=result['generated'] == reference(item, length)[:limit])
        if not result['output_agrees']:
            append(args.output/'mismatches.jsonl', result)
            raise AssertionError(f'Output mismatch {item["split"]} {item["source_index"]} {length} {name}')
        return result
    names = list(candidates)+list(retrieval)+list(full)
    # Train-only warmups also exercise each tree shape before development timing.
    for length in (8192, 16384):
        for name in names:
            run(training[0], length, name, output_limit=8)
        print('warmups complete', length, flush=True)
    tuning = []
    for item in development:
        for length in (8192, 16384):
            order = list(names); random.Random(42+item['source_index']*1000+length).shuffle(order)
            for name in order:
                result = run(item, length, name)
                tuning.append(result); append(args.output/'development.jsonl', result)
            print('development complete', item['source_index'], length, flush=True)
    scores = {name: sum(r['tokens'] for r in tuning if r['candidate'] == name)/
                    sum(r['total_seconds'] for r in tuning if r['candidate'] == name) for name in names}
    positive = max((name for name, cfg in candidates.items() if cfg.inject_tokens), key=scores.get)
    guarded = max(candidates, key=scores.get)
    chosen = dict(rescue_positive=positive, rescue_guarded=guarded, rescue_dynamic='dynamic',
        retrieval=max(retrieval, key=scores.get), full_specextend=max(full, key=scores.get))
    write(args.output/'selection.json', dict(chosen=chosen, development_total_tps=scores))
    print('locked', chosen, scores, flush=True)
    # Only now load independent test documents. No further configuration changes.
    test = [json.loads(l) for l in (directory/'test.jsonl').read_text().splitlines()][:args.test_documents]
    write(args.output/'test_matrix.json', dict(source_indices=[r['source_index'] for r in test],
        lengths=args.lengths, repeats=args.repeats, modes=chosen))
    results = []
    for item in test:
        for length in args.lengths:
            for repeat in range(args.repeats):
                order = list(chosen); random.Random(4242+item['source_index']*1000+length+repeat).shuffle(order)
                for mode in order:
                    result = run(item, length, chosen[mode])
                    result.update(mode=mode, repeat=repeat)
                    results.append(result); append(args.output/'runtime.jsonl', result)
                print('test complete', item['source_index'], length, repeat, flush=True)
    write(args.output/'summary.json', summary(results))


if __name__ == '__main__':
    main()
