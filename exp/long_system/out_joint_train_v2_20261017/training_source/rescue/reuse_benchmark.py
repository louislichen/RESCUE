"""Paired execution-reuse and training-selected injection-route benchmark."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import statistics

import torch

from .baselines import BaselineEngine
from .compare_baselines import append, write
from .config import SystemConfig
from .engine import RescueEngine
from .loading import ROOT, DEFAULT_MAPPER, DEFAULT_RISK, load_models, load_modules


def summarize(rows):
    result, by_doc = dict(modes={}, comparisons={}), {}
    for mode in sorted({r['mode'] for r in rows}):
        group = [r for r in rows if r['mode'] == mode]
        tokens = sum(r['tokens'] for r in group)
        proposed, accepted = (sum(r.get(k, 0) for r in group) for k in ('proposed', 'accepted'))
        totals = dict(requests=len(group), tokens=tokens,
            decode_tps=tokens/sum(r['decode_seconds'] for r in group),
            total_tps=tokens/sum(r['total_seconds'] for r in group),
            proposed=proposed, accepted=accepted, acceptance_rate=accepted/max(1, proposed))
        for key in ('blocks', 'draft_forwards', 'draft_shadow_probes', 'static_affine_positions',
                    'projected_positions', 'working_full_gather_layers', 'working_gathered_position_layers',
                    'working_incremental_binds', 'working_restored_value_positions'):
            totals[key] = sum(r.get(key, 0) for r in group)
        result['modes'][mode] = totals
        by_doc[mode] = {d: {key: statistics.median(r[key] for r in group if r['source_index'] == d)
                           for key in ('total_tps', 'decode_tps')}
                        for d in sorted({r['source_index'] for r in group})}
    pairs = [('new_risk', m) for m in ('previous_risk', 'native', 'specextend16', 'specextend32')]
    pairs += [('new_fixed', m) for m in ('previous_fixed', 'native', 'specextend16', 'specextend32')]
    pairs += [('routed_fixed', m) for m in ('new_fixed', 'coverage_noinj', 'native', 'specextend16')]
    pairs += [('teacher_inj', 'teacher_noinj'), ('new_fixed', 'coverage_noinj')]
    for candidate, baseline in pairs:
        comparison = {}
        for metric in ('total_tps', 'decode_tps'):
            differences = [100*(by_doc[candidate][d][metric]/by_doc[baseline][d][metric]-1)
                           for d in sorted(by_doc[candidate])]
            rng = random.Random(42)
            draws = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(5000))
            comparison[metric] = dict(mean_percent=statistics.mean(differences),
                                     ci95=[draws[124], draws[4874]], documents=len(differences))
        result['comparisons'][candidate+'_vs_'+baseline] = comparison
    result.update(measured_runs=len(rows), exact_outputs=sum(r['output_agrees'] for r in rows))
    return result


def routes(base):
    return dict(
        original=base,
        last2_128=replace(base, injection_layers=(26, 27)),
        last1_64=replace(base, injection_layers=(27,), inject_tokens=64),
        last2_64=replace(base, injection_layers=(26, 27), inject_tokens=64),
        all4_64_weak=replace(base, inject_tokens=64, injection_strength=.25),
        last2_64_weak=replace(base, injection_layers=(26, 27), inject_tokens=64, injection_strength=.25),
        joint_last2_64=replace(base, injection_layers=(26, 27), inject_tokens=64, injection_rule='joint'),
        joint_all4_128=replace(base, injection_rule='joint'),
        none=replace(base, inject_tokens=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--documents', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--training-documents', type=int, default=8)
    args = parser.parse_args()
    if min(args.documents, args.repeats, args.max_new_tokens) < 1 or args.training_documents < 2:
        parser.error('Positive counts and at least two training documents required')
    if args.output.exists():
        parser.error('Use a new output directory')
    args.output.mkdir(parents=True)
    data = ROOT/'exp/attention_horizon/data'
    train = [json.loads(s) for s in (data/'train.jsonl').read_text().splitlines()][:args.training_documents]
    validation = [json.loads(s) for s in (data/'validation.jsonl').read_text().splitlines()][:args.documents]
    assert len(train) == args.training_documents and len(validation) == args.documents
    assert not {r['source_index'] for r in train} & {r['source_index'] for r in validation}
    source_files = sorted((ROOT/'rescue').glob('*.py'))+sorted((ROOT/'tests').glob('*.py'))+[
        ROOT/'exp/attention_horizon/selection.py', ROOT/'exp/selective_kv/bridge.py']
    def hashes(paths):
        return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for path in source_files:
        destination = args.output/'source'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    cost_path = ROOT/'exp/system_validation/out_validation_v1_20261017/calibrated_cost.json'
    base = SystemConfig(max_new_tokens=args.max_new_tokens)
    previous = replace(base, deferred_probes=False, memoized_gate_affine=False, working_mode='layerwise')
    execution_configs = dict(previous=previous,
        probe=replace(previous, deferred_probes=True),
        affine=replace(previous, deferred_probes=True, memoized_gate_affine=True),
        packed=base, incremental=replace(base, working_mode='incremental'))
    write(args.output/'protocol.json', dict(schema=1,
        scope='Existing short Qwen validation split, execution reuse and optional train-selected static injection route',
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        train_indices=[r['source_index'] for r in train], validation_indices=[r['source_index'] for r in validation],
        source_sha256=hashes(source_files), artifact_sha256=hashes([DEFAULT_MAPPER, DEFAULT_RISK, cost_path]),
        data_sha256=hashes([data/'train.jsonl', data/'validation.jsonl']),
        execution_candidates={k: asdict(v) for k, v in execution_configs.items()},
        execution_selection='Pooled total TPS on first two training documents, two repeats, choose affine/packed/incremental',
        route_selection='Pooled total TPS on all eight training documents, one repeat; positive and zero-allowed choices retained',
        fixed_k='4 inherited from previous training-only fixed horizon selection; not validation tuned',
        fairness='Native/retrieval use same packed native storage and chosen working-cache mode; native has no probes or mapper',
        cost='Same frozen pre-reuse cost and risk weights in previous/new risk comparisons; no retraining',
        timing='Model loads/reference/warmup excluded; fresh request buffers/prefill/control/draft/verify/cleanup included',
        order_seed=42, target_dtype='float32', draft_dtype='bfloat16', tf32=False))
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    teacher, student, tokenizer = load_models()
    mapper, risk, _ = load_modules(teacher.device)
    cost = json.loads(cost_path.read_text())
    write(args.output/'environment.json', dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        teacher_layers=teacher.config.num_hidden_layers, student_layers=student.config.num_hidden_layers,
        input_lengths={r['source_index']: len(r['ids']) for r in validation}))
    references = {}
    with BaselineEngine(teacher, student, replace(base, length_policy='fixed'), vocab_size=len(tokenizer)) as engine:
        for item in train+validation:
            references[item['source_index']] = engine.reference(item['ids'], tokenizer.eos_token_id)['generated']
    write(args.output/'references.json', references)
    def check(result, item):
        result.update(source_index=item['source_index'],
                      output_agrees=result['generated'] == references[item['source_index']])
        assert result['output_agrees'], item['source_index']
        return result
    def system(item, cfg):
        with RescueEngine(teacher, student, cfg, mapper, risk, cost, vocab_size=len(tokenizer)) as engine:
            return check(engine.generate(item['ids'], tokenizer.eos_token_id), item)
    for cfg in execution_configs.values():
        system(train[0], cfg)
    execution_rows = []
    for item in train[:2]:
        for repeat in range(2):
            order = list(execution_configs)
            random.Random(42+item['source_index']*1000+repeat).shuffle(order)
            for mode in order:
                result = system(item, execution_configs[mode])
                result.update(mode=mode, repeat=repeat)
                append(args.output/'execution_training.jsonl', result)
                execution_rows.append(result)
            print('execution training', item['source_index'], repeat, flush=True)
    scores = {m: sum(r['tokens'] for r in execution_rows if r['mode'] == m)/
                 sum(r['total_seconds'] for r in execution_rows if r['mode'] == m) for m in execution_configs}
    chosen_mode = max(('affine', 'packed', 'incremental'), key=scores.get)
    new = execution_configs[chosen_mode]
    write(args.output/'execution_selection.json', dict(chosen=chosen_mode, total_tps=scores, config=asdict(new)))
    print('execution locked', chosen_mode, scores, flush=True)
    fixed = replace(new, length_policy='fixed', fixed_k=4)
    candidates = routes(fixed)
    write(args.output/'route_candidates.json', {k: asdict(v) for k, v in candidates.items()})
    for name in candidates:
        system(train[0], candidates[name])
    route_rows = []
    for item in train:
        order = list(candidates)
        random.Random(420+item['source_index']*1000).shuffle(order)
        for name in order:
            result = system(item, candidates[name])
            result.update(mode=name, repeat=0)
            append(args.output/'route_training.jsonl', result)
            route_rows.append(result)
        print('route training', item['source_index'], flush=True)
    route_scores = {m: sum(r['tokens'] for r in route_rows if r['mode'] == m)/
                      sum(r['total_seconds'] for r in route_rows if r['mode'] == m) for m in candidates}
    positive = max((m for m in candidates if m != 'none'), key=route_scores.get)
    guarded = max(candidates, key=route_scores.get)
    write(args.output/'route_selection.json', dict(positive=positive, guarded=guarded, total_tps=route_scores,
        positive_config=asdict(candidates[positive]), guarded_config=asdict(candidates[guarded])))
    print('routes locked', positive, guarded, route_scores, flush=True)
    configs = dict(previous_risk=previous, new_risk=new,
        previous_fixed=replace(previous, length_policy='fixed', fixed_k=4), new_fixed=fixed,
        routed_fixed=candidates[positive], coverage_noinj=candidates['none'],
        teacher_inj=replace(fixed, selector='teacher'), teacher_noinj=replace(fixed, selector='teacher', inject_tokens=0))
    write(args.output/'mode_configs.json', {k: asdict(v) for k, v in configs.items()})
    modes = (*configs, 'native', 'specextend16', 'specextend32', 'target_ar')
    def run(item, mode):
        if mode in configs:
            return system(item, configs[mode])
        chunks = dict(specextend16=16, specextend32=32).get(mode)
        with BaselineEngine(teacher, student, fixed, vocab_size=len(tokenizer), top_chunks=chunks) as engine:
            if mode == 'target_ar':
                result = engine.reference(item['ids'], tokenizer.eos_token_id)
                result['tokens'] = len(result['generated'])
            else:
                result = engine.generate(item['ids'], tokenizer.eos_token_id)
        return check(result, item)
    for mode in modes:
        run(train[0], mode)
    rows = []
    for item in validation:
        for repeat in range(args.repeats):
            order = list(modes)
            random.Random(42+item['source_index']*1000+repeat).shuffle(order)
            for mode in order:
                result = run(item, mode)
                result.update(mode=mode, repeat=repeat)
                append(args.output/'runtime.jsonl', result)
                rows.append(result)
            print('validation paired', item['source_index'], repeat, 'all exact', flush=True)
    summary = summarize(rows)
    write(args.output/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
