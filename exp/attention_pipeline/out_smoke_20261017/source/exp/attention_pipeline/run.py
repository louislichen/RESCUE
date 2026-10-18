"""Small TRAINING-DOCUMENT execution smoke; no held-out quality claim."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'exp/attention_horizon'))

import torch
from experiment import Experiment, parser as engine_parser, write, append
import runtime as legacy
from risk import load
from exp.selective_kv.bridge import Injection
from exp.attention_pipeline.planner import LengthPlanner
from exp.attention_pipeline.runtime import speculative


MODES = ('legacy_serial', 'serial_original', 'serial_cached', 'async_cached')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'exp/attention_pipeline/out_smoke_20261017')
    parser.add_argument('--documents', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--planning-spent-ms', type=float, default=0.)
    parser.add_argument('--delivery-delay-ms', type=float, default=0.)
    args = parser.parse_args()
    if args.documents < 1 or args.repeats < 1 or args.max_new_tokens < 1 or min(args.planning_spent_ms, args.delivery_delay_ms) < 0:
        parser.error('Counts must be positive; times must be nonnegative')
    if (args.output/'runtime.jsonl').exists():
        parser.error('Use a new output directory; do not overwrite a completed or partial run')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    engine_args = engine_parser().parse_args([
        '--phase', 'verify', '--output', str(args.output/'engine'),
        '--max-new-tokens', str(args.max_new_tokens)])
    e = Experiment(engine_args)
    artifacts = ROOT/'exp/attention_horizon/out'
    e.cost = json.loads((artifacts/'cost_model.json').read_text())
    risk_path = artifacts/json.loads((artifacts/'risk_selection.json').read_text())['attention_read']['path']
    e.risks['attention_read'] = load(risk_path, e.a.device)[0]
    e.load_mapper()
    documents = e.data['train'][:args.documents]
    if len(documents) != args.documents:
        raise ValueError('Requested more documents than the training split contains')
    source_files = list(Path(__file__).parent.glob('*.py'))+[
        ROOT/'exp/attention_horizon/runtime.py', ROOT/'exp/attention_horizon/experiment.py',
        ROOT/'exp/attention_horizon/selection.py', ROOT/'exp/attention_horizon/risk.py',
        ROOT/'exp/attention_selection/attention.py', ROOT/'exp/selective_kv/bridge.py']
    write(args.output/'protocol.json', dict(
        schema=1, scope='Training-document execution smoke, not held-out efficacy or distributed benchmark',
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source_indices=[d['source_index'] for d in documents], modes=MODES,
        torch=torch.__version__, gpu=torch.cuda.get_device_name(), teacher_dtype='float32', student_dtype='bfloat16',
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        artifact_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         [risk_path, artifacts/'cost_model.json', ROOT/'exp/selective_kv/out_pilot/performance_gated.pt']},
        planning='New serial/async controls use identical fixed spent cost input, historical reference and frozen checkpoints; legacy keeps measured spent',
        asynchronous='Local thread and CUDA stream. Wait at first-token deadline; no extra Draft token. No RPC/network transport.'))
    control_stream = torch.cuda.Stream(device=e.a.device)
    rows = []
    references = {d['source_index']: e.reference(d) for d in documents}
    write(args.output/'references.json', references)
    with LengthPlanner() as planner:
        def run(item, mode):
            if mode == 'legacy_serial':
                e.injection.clear()
                e.injection = Injection(e.sc)
                r = legacy.speculative(e, item, mode, dict(rule='read_joint1', risk='attention_read', inject=True))
                r['mode'] = mode
                return r
            return speculative(e, item, mode, planner, control_stream,
                               planning_spent=args.planning_spent_ms/1000,
                               delivery_delay=args.delivery_delay_ms/1000)
        for mode in MODES:
            run(documents[0], mode)
            print('warmup', mode, flush=True)
        for item in documents:
            for repeat in range(args.repeats):
                order = list(MODES)
                random.Random(42+1000*item['source_index']+repeat).shuffle(order)
                paired = {}
                for mode in order:
                    r = run(item, mode)
                    r.update(repeat=repeat, output_agrees=r['generated'] == references[item['source_index']]['generated'])
                    append(args.output/'runtime.jsonl', r)
                    rows.append(r)
                    paired[mode] = r
                    if not r['output_agrees']:
                        raise AssertionError(f'Target mismatch: {mode}')
                # Execution-order and static-cache controls must preserve every
                # cap/proposed/accepted decision, not just the final output.
                trace = lambda r: [(b['cap'], b['proposed'], b['accepted']) for b in r['trace']]
                base = trace(paired['serial_original'])
                for mode in ('serial_cached', 'async_cached'):
                    if trace(paired[mode]) != base:
                        raise AssertionError(f'Pipeline trajectory mismatch: {mode}')
                print('paired', item['source_index'], repeat, 'exact outputs and matching new-control trajectories', flush=True)
    summary = dict(scope='Two-document default smoke on training data; no inferential CI or held-out quality claim',
                   measured_runs=len(rows), exact_outputs=sum(r['output_agrees'] for r in rows),
                   new_control_trajectories_match=True, modes={})
    for mode in MODES:
        selected = [r for r in rows if r['mode'] == mode]
        by_document = defaultdict(list)
        for r in selected:
            by_document[r['source_index']].append(r)
        summary['modes'][mode] = dict(
            pooled_decode_tps=sum(r['tokens'] for r in selected)/sum(r['decode_seconds'] for r in selected),
            pooled_total_tps=sum(r['tokens'] for r in selected)/sum(r['total_seconds'] for r in selected),
            median_request_decode_seconds={str(d): statistics.median(r['decode_seconds'] for r in group)
                                           for d, group in by_document.items()},
            rounds=sum(r['blocks'] for r in selected),
            decision_wait_seconds=sum(r.get('decision_wait_seconds', 0.) for r in selected))
    write(args.output/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
