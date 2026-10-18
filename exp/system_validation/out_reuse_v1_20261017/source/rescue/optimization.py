"""Training-only system calibration, then frozen paired pipeline validation."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import statistics
from unittest.mock import patch

import torch
import torch.nn.functional as F

from . import scheduler
from .baselines import BaselineEngine
from .calibration import calibrate
from .compare_baselines import append, write
from .config import SystemConfig
from .engine import RescueEngine
from .loading import ROOT, DEFAULT_MAPPER, DEFAULT_RISK, load_models, load_modules
from .profiling import instrument


def fit_hazards(records):
    """Two-parameter, document-balanced censored competing-risk calibration."""
    logits = torch.tensor([r['logits'] for r in records])
    labels = torch.tensor([r['labels'] for r in records])
    mask = torch.tensor([r['mask'] for r in records])
    counts = Counter(r['source_index'] for r in records)
    weights = torch.tensor([1/counts[r['source_index']] for r in records])[:, None]*mask
    logscale = torch.nn.Parameter(torch.zeros(()))
    bias = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([logscale, bias], lr=.03)
    accept = torch.tensor([1., 0., 0.])
    def nll(values):
        errors = F.cross_entropy(values.flatten(0, 1), labels.flatten(), reduction='none').reshape_as(mask)
        return (errors*weights).sum()/weights.sum()
    before = float(nll(logits))
    for _ in range(300):
        optimizer.zero_grad()
        adjusted = logits*logscale.exp()+bias*accept
        loss = nll(adjusted)+.01*(logscale.square()+bias.square())
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        adjusted = logits*logscale.exp()+bias*accept
        return dict(scale=float(logscale.exp()), accept_bias=float(bias),
                    training_nll_before=before, training_nll_after=float(nll(adjusted)),
                    frames=len(records), documents=sorted(counts), regularization=.01,
                    scope='Fit only, two scalar parameters; no validation-set fitting or held-out calibration claim')


def compare(rows):
    modes = sorted({r['mode'] for r in rows})
    pooled, medians = {}, {}
    for mode in modes:
        g = [r for r in rows if r['mode'] == mode]
        count = sum(r['tokens'] for r in g)
        proposed = sum(r.get('proposed', 0) for r in g)
        accepted = sum(r.get('accepted', 0) for r in g)
        pooled[mode] = dict(requests=len(g), tokens=count,
            decode_tps=count/sum(r['decode_seconds'] for r in g),
            total_tps=count/sum(r['total_seconds'] for r in g),
            blocks=sum(r.get('blocks', 0) for r in g), proposed=proposed, accepted=accepted,
            acceptance_rate=accepted/max(1, proposed), draft_forwards=sum(r.get('draft_forwards', 0) for r in g))
        medians[mode] = {d: {m: statistics.median(r[m] for r in g if r['source_index'] == d)
                            for m in ('decode_tps', 'total_tps')}
                         for d in sorted({r['source_index'] for r in g})}
    comparisons = {}
    for candidate in ('optimized_oldrisk', 'optimized_fixed', 'optimized_calibrated'):
        comparisons[candidate] = {}
        for baseline in ('legacy', 'native', 'specextend16', 'specextend32', 'target_ar'):
            comparisons[candidate][baseline] = {}
            for metric in ('decode_tps', 'total_tps'):
                diff = [100*(medians[candidate][d][metric]/medians[baseline][d][metric]-1)
                        for d in sorted(medians[candidate])]
                rng = random.Random(42)
                draws = sorted(statistics.mean(rng.choices(diff, k=len(diff))) for _ in range(5000))
                comparisons[candidate][baseline][metric] = dict(mean_percent=statistics.mean(diff),
                    ci95=[draws[124], draws[4874]], documents=len(diff))
    return dict(modes=pooled, comparisons=comparisons, measured_runs=len(rows),
                exact_outputs=sum(r['output_agrees'] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--documents', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    args = parser.parse_args()
    if min(args.documents, args.repeats, args.max_new_tokens) < 1:
        parser.error('Positive counts required')
    if args.output.exists():
        parser.error('Use a new output directory')
    args.output.mkdir(parents=True)
    source_files = sorted((ROOT/'rescue').glob('*.py'))+sorted((ROOT/'tests').glob('*.py'))+[
        ROOT/'exp/attention_horizon/selection.py', ROOT/'exp/selective_kv/bridge.py']
    def hashes(paths):
        return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for path in source_files:
        dest = args.output/'source'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    data = ROOT/'exp/attention_horizon/data'
    training = [json.loads(line) for line in (data/'train.jsonl').read_text().splitlines()][:8]
    fit, tune = training[:4], training[4:]
    validation = [json.loads(line) for line in (data/'validation.jsonl').read_text().splitlines()][:args.documents]
    assert len(fit) == len(tune) == 4 and len(validation) == args.documents
    assert not {r['source_index'] for r in training} & {r['source_index'] for r in validation}
    base = SystemConfig(max_new_tokens=args.max_new_tokens)
    old_cost_path = ROOT/'exp/system_validation/out_validation_v1_20261017/calibrated_cost.json'
    old_cost = json.loads(old_cost_path.read_text())
    write(args.output/'protocol.json', dict(schema=1, args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        config=asdict(base), fit_indices=[r['source_index'] for r in fit], tune_indices=[r['source_index'] for r in tune],
        validation_indices=[r['source_index'] for r in validation], source_sha256=hashes(source_files),
        artifact_sha256=hashes([DEFAULT_MAPPER, DEFAULT_RISK, old_cost_path]),
        data_sha256=hashes([data/'train.jsonl', data/'validation.jsonl']),
        fixed_menu=[1, 2, 4, 8, 12], order_seed=42,
        scope='Existing Qwen validation split; training-only system/cost/hazard calibration, no long-context/full SpecExtend claim',
        native_fairness='Common GQA/direct-gather/catchup optimizations also enabled for native and retrieval baselines',
        profiling='Separate instrumented training requests excluded from throughput matrix'))
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    teacher, student, tokenizer = load_models()
    mapper, risk, _ = load_modules(teacher.device)
    write(args.output/'environment.json', dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        target_dtype=str(teacher.dtype), draft_dtype=str(student.dtype), tf32=False,
        teacher_layers=teacher.config.num_hidden_layers, student_layers=student.config.num_hidden_layers))
    references = {}
    with BaselineEngine(teacher, student, replace(base, length_policy='fixed'), vocab_size=len(tokenizer)) as engine:
        for item in training+validation:
            references[item['source_index']] = engine.reference(item['ids'], tokenizer.eos_token_id)['generated']
    write(args.output/'references.json', references)
    def checked(result, item):
        result['source_index'] = item['source_index']
        result['output_agrees'] = result['generated'] == references[item['source_index']]
        if not result['output_agrees']:
            raise AssertionError(f'Target mismatch: {item["source_index"]}')
        return result

    # Diagnostic controlled measurements; no validation-set timing/tuning yet.
    diagnostic = []
    for mode in ('legacy', 'optimized', 'coverage_fixed', 'injection_fixed'):
        cfg = replace(base, optimized_execution=mode != 'legacy',
                      length_policy='fixed' if mode.endswith('fixed') else 'risk',
                      inject_tokens=0 if mode == 'coverage_fixed' else 128)
        for item in fit[:2]:
            with RescueEngine(teacher, student, cfg, mapper, risk, old_cost, vocab_size=len(tokenizer)) as engine:
                with instrument(engine) as timeline:
                    result = checked(engine.generate(item['ids'], tokenizer.eos_token_id), item)
                diagnostic.append(dict(mode=mode, result=result, spans=timeline.summary()))
        write(args.output/'diagnostic_profiles.json', diagnostic)
        print('diagnostic', mode, flush=True)

    cost, cost_records = calibrate(teacher, student, mapper, risk, base, fit[:2], tokenizer.eos_token_id, len(tokenizer))
    write(args.output/'cost.json', cost)
    write(args.output/'cost_records.json', cost_records)
    print('new cost', cost, flush=True)
    # Fixed long-enough drafts expose at-risk hazards; later rejected tokens
    # are censored rather than incorrectly labeled as additional failures.
    fitting = []
    for item in fit:
        vectors = []
        def observe(model, vector, *rest):
            vectors.append(vector.detach().clone())
            remaining = rest[2]
            return min(8, remaining), 0.
        with RescueEngine(teacher, student, base, mapper, risk, cost, vocab_size=len(tokenizer)) as engine:
            with patch.object(scheduler, 'choose', observe):
                result = checked(engine.generate(item['ids'], tokenizer.eos_token_id), item)
        append(args.output/'fit_requests.jsonl', result)
        assert len(vectors) == len(result['trace'])
        with torch.no_grad():
            stacked = torch.stack(vectors)
            logits = risk(stacked).cpu().tolist()
            features = stacked.cpu().tolist()
        progress = 0
        for vector, logit, row in zip(features, logits, result['trace']):
            labels, mask = [0]*12, [False]*12
            for j in range(row['accepted']):
                mask[j] = True
            if row['accepted'] < row['proposed']:
                labels[row['accepted']], mask[row['accepted']] = 2, True
            if (row['terminal'] and result['generated'][-1] == tokenizer.eos_token_id and
                row['accepted'] > 0 and progress+row['accepted'] == result['tokens']):
                labels[row['accepted']-1] = 1
            fitting.append(dict(source_index=item['source_index'], features=vector, logits=logit,
                                labels=labels, mask=mask, proposed=row['proposed'], accepted=row['accepted']))
            progress += row['accepted']+1
        print('fit hazards', item['source_index'], flush=True)
    write(args.output/'hazard_frames.json', fitting)
    calibration = fit_hazards(fitting)
    calibration['base_checkpoint_sha256'] = hashlib.sha256(DEFAULT_RISK.read_bytes()).hexdigest()
    write(args.output/'risk_calibration.json', calibration)
    adjusted_risk = risk.calibrated(calibration['scale'], calibration['accept_bias'])
    print('risk calibration', calibration, flush=True)

    fixed_cfg = replace(base, length_policy='fixed')
    def run(item, mode, k=4):
        if mode in ('native', 'specextend16', 'specextend32', 'target_ar'):
            chunks = dict(specextend16=16, specextend32=32).get(mode)
            with BaselineEngine(teacher, student, replace(fixed_cfg, fixed_k=k),
                                vocab_size=len(tokenizer), top_chunks=chunks) as engine:
                if mode == 'target_ar':
                    result = engine.reference(item['ids'], tokenizer.eos_token_id)
                    result['tokens'] = len(result['generated'])
                else:
                    result = engine.generate(item['ids'], tokenizer.eos_token_id)
        else:
            cfg = replace(base, optimized_execution=mode != 'legacy',
                          length_policy='fixed' if mode == 'optimized_fixed' else 'risk', fixed_k=k)
            network = adjusted_risk if mode == 'optimized_calibrated' else risk
            with RescueEngine(teacher, student, cfg, mapper, network,
                              old_cost if mode in ('legacy', 'optimized_oldrisk') else cost,
                              vocab_size=len(tokenizer)) as engine:
                result = engine.generate(item['ids'], tokenizer.eos_token_id)
        result['mode'] = mode
        return checked(result, item)

    candidates = [(mode, k) for mode in ('native', 'specextend16', 'specextend32', 'optimized_fixed')
                  for k in (1, 2, 4, 8, 12)]+[(mode, 4) for mode in ('legacy', 'optimized_oldrisk', 'optimized_calibrated')]
    for mode in ('native', 'specextend16', 'specextend32', 'legacy', 'optimized_oldrisk', 'optimized_fixed', 'optimized_calibrated', 'target_ar'):
        run(tune[0], mode)
    tuning = []
    for item in tune:
        for repeat in range(1):
            order = list(candidates)
            random.Random(42+item['source_index']*1000+repeat).shuffle(order)
            for mode, k in order:
                result = run(item, mode, k)
                result.update(k=k, repeat=repeat)
                append(args.output/'tuning.jsonl', result)
                tuning.append(result)
            print('tuning paired', item['source_index'], repeat, flush=True)
    selected = {}
    for mode in ('native', 'specextend16', 'specextend32', 'optimized_fixed'):
        scores = {k: sum(r['tokens'] for r in tuning if r['mode'] == mode and r['k'] == k)/
                     sum(r['total_seconds'] for r in tuning if r['mode'] == mode and r['k'] == k) for k in (1, 2, 4, 8, 12)}
        selected[mode] = dict(k=max(scores, key=scores.get), tuning_total_tps=scores)
    write(args.output/'selected_lengths.json', selected)
    print('locked lengths', selected, flush=True)
    rows = []
    modes = ('native', 'specextend16', 'specextend32', 'legacy', 'optimized_oldrisk', 'optimized_fixed', 'optimized_calibrated', 'target_ar')
    for item in validation:
        for repeat in range(args.repeats):
            order = list(modes)
            random.Random(42+item['source_index']*1000+repeat).shuffle(order)
            for mode in order:
                k = selected[mode]['k'] if mode in selected else 4
                result = run(item, mode, k)
                result.update(k=k if mode in selected else None, repeat=repeat)
                append(args.output/'runtime.jsonl', result)
                rows.append(result)
            print('validation paired', item['source_index'], repeat, flush=True)
    result = compare(rows)
    write(args.output/'summary.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
