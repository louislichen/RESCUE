"""System regression/performance matrix on an explicitly recorded dataset split."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import torch

from .calibration import calibrate
from .config import SystemConfig
from .engine import RescueEngine
from .loading import ROOT, DEFAULT_RISK, load_models, load_modules


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def prototype(teacher, student, tokenizer, mapper, risk, cost, config, item):
    # Import adapters only for the legacy comparison; the actual system does
    # not depend on these mutable global import paths or experiment harnesses.
    sys.path.insert(0, str(ROOT/'exp/attention_horizon'))
    from experiment import Experiment
    from exp.attention_selection.attention import AttentionController
    from exp.selective_kv.bridge import Injection
    from exp.attention_pipeline.planner import LengthPlanner
    from exp.attention_pipeline.runtime import speculative
    e = Experiment.__new__(Experiment)
    e.a = SimpleNamespace(device=str(teacher.device), max_new_tokens=config.max_new_tokens,
                          budget=config.budget, recent=config.recent, swaps=4)
    e.teacher, e.student, e.tt, e.st = teacher, student, tokenizer, tokenizer
    e.tc, e.sc = AttentionController(teacher).install(), AttentionController(student).install()
    e.sc.reference_observer = e.observe
    e.mapper, e.injection, e.shadow, e.vocab = mapper, Injection(e.sc), None, len(tokenizer)
    e.risks, e.cost = {'attention_read': risk}, cost
    try:
        with LengthPlanner() as planner:
            result = speculative(e, item, 'serial_cached', planner, torch.cuda.Stream(device=teacher.device))
        return result
    finally:
        e.injection.clear()
        for controller in (e.tc, e.sc):
            for module, original in controller.originals:
                module.forward = original


def summarize(rows, modes, references):
    result = dict(modes={}, comparisons={}, measured_runs=len(rows),
                  exact_outputs=sum(r['output_agrees'] for r in rows))
    by_mode = {}
    for mode in modes:
        group = [r for r in rows if r['mode'] == mode]
        by_mode[mode] = {}
        for d in {r['source_index'] for r in group}:
            dr = [r for r in group if r['source_index'] == d]
            by_mode[mode][d] = dict(total_tps=statistics.median(r['total_tps'] for r in dr),
                                   decode_tps=statistics.median(r['decode_tps'] for r in dr))
        result['modes'][mode] = dict(
            decode_tps=sum(r['tokens'] for r in group)/sum(r['decode_seconds'] for r in group),
            total_tps=sum(r['tokens'] for r in group)/sum(r['total_seconds'] for r in group),
            acceptance_rate=sum(r['accepted'] for r in group)/max(1, sum(r['proposed'] for r in group)),
            blocks=sum(r['blocks'] for r in group), proposed=sum(r['proposed'] for r in group),
            projected_positions=sum(r.get('projected_positions', 0) for r in group),
            working_gather_layers=sum(r.get('working_gather_layers', 0) for r in group),
            length_fallbacks=sum(bool(b.get('fallback')) for r in group for b in r['trace']))
    reference_rows = list(references.values())
    result['target_ar'] = {key: sum(len(r['generated']) for r in reference_rows)/sum(r[time_key] for r in reference_rows)
                           for key, time_key in [('decode_tps', 'decode_seconds'), ('total_tps', 'total_seconds')]}
    baseline = 'prototype'
    for mode in modes:
        if mode == baseline:
            continue
        result['comparisons'][mode] = {}
        for metric in ('decode_tps', 'total_tps'):
            differences = [100*(by_mode[mode][d][metric]/by_mode[baseline][d][metric]-1) for d in sorted(by_mode[mode])]
            rng = random.Random(42)
            draws = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(5000))
            result['comparisons'][mode][metric] = dict(mean_percent=statistics.mean(differences),
                ci95=[draws[124], draws[4874]], documents=len(differences))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'validation', 'test'], default='validation')
    parser.add_argument('--documents', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--calibration-documents', type=int, default=2)
    parser.add_argument('--modes', nargs='+', default=['prototype', 'compat', 'fast', 'coverage', 'rpc'])
    args = parser.parse_args()
    if min(args.documents, args.repeats, args.max_new_tokens, args.calibration_documents) < 1:
        parser.error('Counts must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new output directory; results are never overwritten')
    allowed = {'prototype', 'compat', 'fast', 'coverage', 'rpc'}
    if not set(args.modes) <= allowed or 'prototype' not in args.modes or len(set(args.modes)) != len(args.modes):
        parser.error('Choose distinct known modes including prototype')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    teacher, student, tokenizer = load_models()
    mapper, risk, old_cost = load_modules(teacher.device)
    data = ROOT/'exp/attention_horizon/data'
    documents = [json.loads(line) for line in (data/f'{args.split}.jsonl').read_text().splitlines()][:args.documents]
    training = [json.loads(line) for line in (data/'train.jsonl').read_text().splitlines()][:args.calibration_documents]
    if len(documents) != args.documents or len(training) != args.calibration_documents:
        raise ValueError('Not enough documents in requested split')
    base = SystemConfig(max_new_tokens=args.max_new_tokens)
    source_files = sorted((ROOT/'rescue').glob('*.py'))+[
        ROOT/'exp/attention_horizon/selection.py', ROOT/'exp/attention_horizon/runtime.py',
        ROOT/'exp/attention_horizon/experiment.py', ROOT/'exp/attention_selection/attention.py',
        ROOT/'exp/selective_kv/bridge.py', ROOT/'exp/attention_pipeline/runtime.py',
        ROOT/'exp/attention_pipeline/planner.py', ROOT/'exp/attention_pipeline/cached_injection.py']
    for p in source_files:
        target = args.output/'source'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    protocol = dict(schema=1, scope='Existing split system benchmark; not a fresh paper confirmation set',
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, config=asdict(base),
        source_indices=[d['source_index'] for d in documents], calibration_sources=[d['source_index'] for d in training],
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        data_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (data/'train.jsonl', data/f'{args.split}.jsonl')},
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, target_dtype='float32', draft_dtype='bfloat16',
        tf32=False, order_seed=42)
    write(args.output/'protocol.json', protocol)
    cost, calibration = calibrate(teacher, student, mapper, risk, base, training, tokenizer.eos_token_id, len(tokenizer))
    write(args.output/'calibrated_cost.json', cost)
    write(args.output/'calibration.json', calibration)
    print('training-only cost calibration complete', cost, flush=True)
    references = {}
    with RescueEngine(teacher, student, base, mapper, risk, cost, vocab_size=len(tokenizer)) as engine:
        for item in documents:
            references[item['source_index']] = engine.reference(item['ids'], tokenizer.eos_token_id)
    write(args.output/'references.json', references)
    process = None
    try:
        endpoint = None
        if 'rpc' in args.modes:
            ready_path = (args.output/'length_server_ready.json').resolve()
            process = subprocess.Popen([sys.executable, '-m', 'rescue.length_server', '--checkpoint', str(DEFAULT_RISK),
                                        '--ready-file', str(ready_path)], stdout=subprocess.DEVNULL)
            deadline = time.monotonic()+30
            while not ready_path.exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('Length service failed to become ready')
                time.sleep(.05)
            ready = json.loads(ready_path.read_text())
            endpoint = (ready['host'], ready['port'])
        configs = dict(
            compat=replace(base, selector='read_joint', refresh_every=1, attention_backend='eager'),
            fast=replace(base, selector='read_joint', refresh_every=1),
            coverage=base, rpc=replace(base, length_endpoint=endpoint))
        engines = {}
        try:
            # Controllers own their model hooks; instantiate each engine only
            # during its request, restore hooks before another mode takes over.
            def run(item, mode):
                if mode == 'prototype':
                    return prototype(teacher, student, tokenizer, mapper, risk, old_cost, base, item)
                cfg = configs[mode]
                with RescueEngine(teacher, student, cfg, mapper, risk,
                                  old_cost if mode in ('compat', 'fast') else cost,
                                  memoize=mode != 'compat', vocab_size=len(tokenizer)) as engine:
                    return engine.generate(item['ids'], tokenizer.eos_token_id)
            for mode in args.modes:
                run(training[0], mode)
                print('warmup', mode, flush=True)
            rows = []
            for item in documents:
                for repeat in range(args.repeats):
                    order = list(args.modes)
                    random.Random(42+item['source_index']*1000+repeat).shuffle(order)
                    for mode in order:
                        result = run(item, mode)
                        result.update(mode=mode, source_index=item['source_index'], repeat=repeat,
                                      output_agrees=result['generated'] == references[item['source_index']]['generated'])
                        with (args.output/'runtime.jsonl').open('a') as handle:
                            handle.write(json.dumps(result)+'\n')
                        rows.append(result)
                        if not result['output_agrees']:
                            raise AssertionError(f'Target output mismatch: {mode} {item["source_index"]}')
                    print('paired', item['source_index'], repeat, 'outputs exact', flush=True)
            summary = summarize(rows, args.modes, references)
            write(args.output/'summary.json', summary)
            print(json.dumps(summary, indent=2), flush=True)
        finally:
            for engine in engines.values():
                engine.close()
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    main()
