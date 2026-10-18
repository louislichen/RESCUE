"""CPU-only accounting and implementation audit of saved system measurements.

Counterfactuals below are accounting estimates, not measured speedups. No model
generation, training, original source modification, or GPU work is performed.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

import torch


ROOT = Path(__file__).resolve().parents[2]


def analyze(directory):
    protocol = json.loads((directory/'protocol.json').read_text())
    for path, digest in protocol['source_sha256'].items():
        assert hashlib.sha256((directory/'source'/path).read_bytes()).hexdigest() == digest
    rows = [json.loads(line) for line in (directory/'runtime.jsonl').read_text().splitlines()]
    summary = json.loads((directory/'summary.json').read_text())
    assert len(rows) == summary['exact_outputs'] == summary['measured_runs']
    old_cost = json.loads((ROOT/'exp/system_validation/out_validation_v1_20261017/calibrated_cost.json').read_text())
    new_cost = json.loads((directory/'cost.json').read_text())
    mapper_path = ROOT/'exp/selective_kv/out_pilot/performance_gated.pt'
    mapper = torch.load(mapper_path, map_location='cpu', weights_only=False)
    count = sum(v.numel() for v in mapper['state_dict'].values())
    size = sum(v.numel()*v.element_size() for v in mapper['state_dict'].values())
    result = dict(scope='Saved paired records and inspected implementation; no new GPU evaluation',
        output_directory=str(directory.relative_to(ROOT)),
        mapper=dict(config=mapper['config'], parameters=count, weight_bytes=size,
                    weight_mib=size/2**20, dtypes=sorted({str(v.dtype) for v in mapper['state_dict'].values()}),
                    sha256=hashlib.sha256(mapper_path.read_bytes()).hexdigest()), modes={})
    for mode in summary['modes']:
        group = [r for r in rows if r['mode'] == mode]
        decode, prefill, total = (sum(r[k] for r in group) for k in ('decode_seconds', 'prefill_seconds', 'total_seconds'))
        blocks, steps = sum(r.get('blocks', 0) for r in group), sum(r.get('draft_forwards', 0) for r in group)
        result['modes'][mode] = dict(requests=len(group), decode_seconds=decode, prefill_seconds=prefill,
            total_seconds=total, blocks=blocks, steps=steps,
            tokens_per_round=sum(r['tokens'] for r in group)/blocks if blocks else None,
            decode_ms_per_round=decode*1000/blocks if blocks else None,
            length_caps=dict(Counter(t['cap'] for r in group for t in r.get('trace', []))))
        if mode.startswith('optimized'):
            values = result['modes'][mode]
            cost = old_cost if mode == 'optimized_oldrisk' else new_cost
            predictions, errors = [], []
            for row in group:
                pred = cost['student_step']*row['draft_forwards']
                for t in row['trace']:
                    n, k = t['prefix']/2048, t['proposed']
                    pred += sum(a*b for a, b in zip(cost['verify_beta'], [1., n, k, n*k]))
                    pred += cost.get('fixed_round', 0.)+cost.get('cache_update', 0.)
                predictions.append(pred)
                errors.append(100*(pred/row['decode_seconds']-1))
            values['cost_replay'] = dict(predicted_seconds=sum(predictions), observed_seconds=decode,
                pooled_error_percent=100*(sum(predictions)/decode-1),
                request_mape_percent=statistics.mean(abs(x) for x in errors),
                request_error_range_percent=[min(errors), max(errors)],
                scope='Uses realized lengths and catchup counts; excludes risk-prediction error and does not validate candidate ranking')
            values['full_key_decode_probes_now'] = steps
            values['full_key_decode_probes_if_deferred'] = blocks-len(group)
            values['deferred_probe_reduction_percent'] = 100*(1-(blocks-len(group))/steps)
            values['query_token_layer_gate_evaluations'] = steps*len(mapper['config']['layers'])
    native, fixed = result['modes']['native'], result['modes']['optimized_fixed']
    base_cycle = native['decode_seconds']/native['blocks']
    own_cycle = fixed['decode_seconds']/fixed['blocks']
    extra_rounds = (fixed['blocks']-native['blocks'])*base_cycle
    extra_cycle = fixed['blocks']*(own_cycle-base_cycle)
    gap = fixed['decode_seconds']-native['decode_seconds']
    assert abs(gap-extra_rounds-extra_cycle) < 1e-10
    result['fixed_vs_native_accounting'] = dict(decode_gap_seconds=gap,
        extra_round_count_component_seconds=extra_rounds,
        higher_average_cycle_component_seconds=extra_cycle,
        higher_average_cycle_component_fraction=extra_cycle/gap,
        break_even_extra_emitted_tokens_per_round_percent=100*(own_cycle*summary['modes']['native']['decode_tps']/fixed['tokens_per_round']-1),
        scope='Algebraic decomposition of saved totals, not causal per-module attribution; break-even holds other terms constant')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT/'exp/system_validation/out_optimization_v1_20261017')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path')
    result = analyze(args.directory)
    result['analysis_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'modes'}, ensure_ascii=False, indent=2))
    for mode in ('optimized_oldrisk', 'optimized_fixed', 'optimized_calibrated'):
        print(mode, json.dumps(result['modes'][mode]['cost_replay']))


if __name__ == '__main__':
    main()
