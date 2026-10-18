"""Matched native SD / SpecExtend retrieval / complete RESCUE comparison.

Select each fixed horizon on training data, then lock it before validation.
SpecExtend controls port only retrieval to Qwen2 linear SD, not its full engine.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import statistics

import torch

from .baselines import BaselineEngine
from .config import SystemConfig
from .engine import RescueEngine
from .loading import ROOT, DEFAULT_MAPPER, DEFAULT_RISK, load_models, load_modules


def write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')


def append(path, payload):
    with path.open('a') as handle:
        handle.write(json.dumps(payload)+'\n')


def summary(rows):
    by_mode, result = {}, dict(modes={}, comparisons={})
    for mode in sorted({r['mode'] for r in rows}):
        group = [r for r in rows if r['mode'] == mode]
        n = sum(r['tokens'] for r in group)
        blocks = sum(r.get('blocks', 0) for r in group)
        result['modes'][mode] = dict(requests=len(group), tokens=n,
            decode_tps=n/sum(r['decode_seconds'] for r in group),
            total_tps=n/sum(r['total_seconds'] for r in group),
            proposed=sum(r.get('proposed', 0) for r in group),
            accepted=sum(r.get('accepted', 0) for r in group), blocks=blocks)
        m = result['modes'][mode]
        m['acceptance_rate'] = m['accepted']/max(1, m['proposed'])
        m['accepted_per_round'] = m['accepted']/max(1, blocks)
        by_mode[mode] = {d: {metric: statistics.median(r[metric] for r in group if r['source_index'] == d)
                            for metric in ('total_tps', 'decode_tps')}
                         for d in sorted({r['source_index'] for r in group})}
    for baseline in sorted(set(by_mode)-{'rescue'}):
        result['comparisons'][baseline] = {}
        for metric in ('total_tps', 'decode_tps'):
            differences = [100*(by_mode['rescue'][d][metric]/by_mode[baseline][d][metric]-1)
                           for d in sorted(by_mode[baseline])]
            rng = random.Random(42)
            draws = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(5000))
            result['comparisons'][baseline][metric] = dict(mean_percent=statistics.mean(differences),
                ci95=[draws[124], draws[4874]], documents=len(differences))
    result.update(measured_runs=len(rows), exact_outputs=sum(r['output_agrees'] for r in rows))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cost', type=Path, default=ROOT/'exp/system_validation/out_validation_v1_20261017/calibrated_cost.json')
    parser.add_argument('--documents', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--training-documents', type=int, default=2)
    parser.add_argument('--training-repeats', type=int, default=2)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    args = parser.parse_args()
    if min(args.documents, args.repeats, args.training_documents, args.training_repeats, args.max_new_tokens) < 1:
        parser.error('Positive counts required')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use an empty output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = ROOT/'exp/attention_horizon/data'
    train = [json.loads(line) for line in (data/'train.jsonl').read_text().splitlines()][:args.training_documents]
    documents = [json.loads(line) for line in (data/'validation.jsonl').read_text().splitlines()][:args.documents]
    if len(train) != args.training_documents or len(documents) != args.documents:
        raise ValueError('Insufficient documents')
    assert not {d['source_index'] for d in train} & {d['source_index'] for d in documents}
    base = SystemConfig(max_new_tokens=args.max_new_tokens)
    fixed = replace(base, selector='full', length_policy='fixed', inject_tokens=0, schedule='serial')
    menu = (1, 2, 4, 8, 12)
    families = dict(native=None, specextend16=16, specextend32=32)
    sources = sorted((ROOT/'rescue').glob('*.py'))+[
        ROOT/'exp/attention_horizon/selection.py', ROOT/'exp/selective_kv/bridge.py',
        ROOT/'exp/specextend_bridgekv/specextend/classic/model_classic.py',
        ROOT/'exp/specextend_bridgekv/specextend/classic/utils_classic.py',
        ROOT/'tests/test_rescue_baselines.py', ROOT/'tests/test_rescue_system.py']
    def hashes(paths):
        return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for path in sources:
        dest = args.output/'source'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    write(args.output/'protocol.json', dict(schema=1,
        scope='Matched Qwen2 backbone; SpecExtend retrieval port only, not upstream full-system reproduction',
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        config=asdict(base), fixed_config=asdict(fixed), menu=menu,
        train_indices=[r['source_index'] for r in train], validation_indices=[r['source_index'] for r in documents],
        source_sha256=hashes(sources), artifact_sha256=hashes([DEFAULT_MAPPER, DEFAULT_RISK, args.cost]),
        data_sha256=hashes([data/'train.jsonl', data/'validation.jsonl']),
        selection_metric='Pooled training total TPS over two timing repeats; no validation tuning',
        sdpa='Same Draft decode SDPA, Target eager FP32 and Draft BF16 in all SD modes',
        retrieval='chunk32, newest mandatory, mean last-layer Target heads, refresh every4 verification rounds, recent initialization',
        retrieval_scope='Linear draft only; no upstream Flash prefill or hybrid tree attention',
        timing='Fresh request buffers and hook setup included; model/module load and engine constructor excluded equally',
        existing_validation=True, order_seed=42))
    teacher, student, tokenizer = load_models()
    mapper, risk, _ = load_modules(teacher.device)
    cost = json.loads(args.cost.read_text())
    write(args.output/'cost.json', cost)
    write(args.output/'environment.json', dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        target_dtype=str(teacher.dtype), draft_dtype=str(student.dtype), tf32=False,
        target_layers=teacher.config.num_hidden_layers, draft_layers=student.config.num_hidden_layers,
        input_lengths={r['source_index']: len(r['ids']) for r in documents}))

    def run(item, mode, k=4):
        if mode == 'rescue':
            with RescueEngine(teacher, student, base, mapper, risk, cost, vocab_size=len(tokenizer)) as engine:
                return engine.generate(item['ids'], tokenizer.eos_token_id)
        with BaselineEngine(teacher, student, replace(fixed, fixed_k=k),
                            vocab_size=len(tokenizer), top_chunks=families.get(mode)) as engine:
            if mode == 'target_ar':
                result = engine.reference(item['ids'], tokenizer.eos_token_id)
                result['tokens'] = len(result['generated'])
                return result
            return engine.generate(item['ids'], tokenizer.eos_token_id)

    references = {}
    for item in train+documents:
        references[item['source_index']] = run(item, 'target_ar')['generated']
    write(args.output/'references.json', references)
    for mode in (*families, 'rescue', 'target_ar'):
        warm = run(train[0], mode)
        assert warm['generated'] == references[train[0]['source_index']]
        print('warmup', mode, flush=True)
    training = []
    candidates = [(mode, k) for mode in families for k in menu]
    for item in train:
        for repeat in range(args.training_repeats):
            order = list(candidates)
            random.Random(42+item['source_index']*1000+repeat).shuffle(order)
            for mode, k in order:
                result = run(item, mode, k)
                result.update(mode=mode, k=k, source_index=item['source_index'], repeat=repeat,
                              output_agrees=result['generated'] == references[item['source_index']])
                append(args.output/'training.jsonl', result)
                training.append(result)
                if not result['output_agrees']:
                    raise AssertionError(f'Training output mismatch: {mode}, {k}')
            print('training paired', item['source_index'], repeat, flush=True)
    selected = {}
    for mode in families:
        scores = {k: sum(r['tokens'] for r in training if r['mode'] == mode and r['k'] == k)/
                     sum(r['total_seconds'] for r in training if r['mode'] == mode and r['k'] == k) for k in menu}
        selected[mode] = dict(k=max(menu, key=lambda k: scores[k]), training_total_tps=scores)
    # Written before the first validation timing; chosen lengths stay frozen.
    write(args.output/'selected_lengths.json', selected)
    print('training-only selected lengths', selected, flush=True)
    modes = ('native', 'native_k4', 'specextend16', 'specextend32', 'rescue', 'target_ar')
    rows = []
    for item in documents:
        for repeat in range(args.repeats):
            order = list(modes)
            random.Random(42+item['source_index']*1000+repeat).shuffle(order)
            for mode in order:
                k = selected[mode]['k'] if mode in selected else 4
                result = run(item, 'native' if mode == 'native_k4' else mode, k)
                result.update(mode=mode, k=k if mode != 'rescue' else None,
                              source_index=item['source_index'], repeat=repeat,
                              output_agrees=result['generated'] == references[item['source_index']])
                append(args.output/'runtime.jsonl', result)
                rows.append(result)
                if not result['output_agrees']:
                    raise AssertionError(f'Validation output mismatch: {mode}, {item["source_index"]}')
            print('validation paired', item['source_index'], repeat, 'all exact', flush=True)
    result = summary(rows)
    write(args.output/'summary.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
