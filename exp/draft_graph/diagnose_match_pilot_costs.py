"""Describe the old n=1 matched pilot; never run models or choose parameters.

The pinned pilot already completed. This post-hoc decomposition is not a new
independent experiment, a causal stage attribution, or the pending confirmation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil


PINS = {
    'runtime.jsonl': 'cf72869e1a060719938bb1e7fc0bad17e738ea604542a5da76ad296f8eae2d5c',
    'reference.jsonl': '8d74e2991a5cf374bbc9c227c3a3fec82f1d485afeeb0e424172c246e0e25630',
    'protocol.json': '502f73a9108cd605be626b230adb39f5a4c36c352098cbacf34a58ab67154306',
    'COMPLETE.json': 'e89c4c62bc39e7f15f70bc1fe4ab5142a5935eafde8c36faede75e22c88f3749',
    'correctness.jsonl': '9283e19cf5656f68b0356ca8751d09c8b2fab75c7b34f5943a68565d5088ff2c',
}
MODES = ('rescue_graph', 'retrieval32_k4_graph')
LENGTHS = (4096, 8192)
FIELDS = ('tokens', 'outer_seconds', 'prefill_seconds', 'decode_seconds',
          'blocks', 'accepted', 'proposed', 'draft_forwards')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(raw):
    require(raw.endswith(b'\n') and all(x.strip() for x in raw.splitlines()), 'Incomplete JSONL')
    return [json.loads(x) for x in raw.splitlines()]


def analyze(directory):
    root = Path(directory).resolve()
    raw = {name: (root/name).read_bytes() for name in PINS}
    require({n: hashlib.sha256(b).hexdigest() for n, b in raw.items()} == PINS,
            'Only the exact archived matched pilot may be described')
    protocol, complete = json.loads(raw['protocol.json']), json.loads(raw['COMPLETE.json'])
    require(complete.get('passed') is True and complete.get('sources_unchanged') is True and
            complete.get('assets_unchanged') is True and protocol.get('max_new_tokens') == 128,
            'Missing immutable completed pilot')
    references = rows(raw['reference.jsonl'])
    require(len(references) == 2 and {r['length'] for r in references} == set(LENGTHS) and
            all(r['source_index'] == 303 and len(r['generated']) == 128 for r in references),
            'Changed exact4K/8K doc303 reference matrix')
    reference = {r['length']: r['generated'] for r in references}
    records = rows(raw['runtime.jsonl'])
    require(len(records) == 20 and len({r['request_id'] for r in records}) == 20 and
            all(r['generated'] == reference[r['length']] and r['tokens'] == 128 and
                r['source_index'] == 303 and r['output_agrees'] is True and r['phase'] == 'warm'
                for r in records), 'Pilot outputs differ from canonical reference')
    cells, decomposition = {}, {}
    for length in LENGTHS:
        cells[str(length)] = {}
        for mode in MODES:
            group = sorted((r for r in records if r['mode'] == mode and r['length'] == length),
                           key=lambda r: r['repeat'])
            require([r['repeat'] for r in group] == [0, 1], 'Missing paired timing repeat')
            require(all(type(r[k]) in (int, float) and math.isfinite(r[k]) and r[k] > 0
                        for r in group for k in FIELDS), 'Invalid measured point')
            sums = {k: sum(r[k] for r in group) for k in FIELDS}
            require(all(r['draft_runtime']['request']['captures'] == 0 for r in group),
                    'This exact old warm pilot is no longer capture-free')
            cell = dict(sums=sums, repeats=2, tps=sums['tokens']/sums['outer_seconds'],
                        accepted_per_round=sums['accepted']/sums['blocks'],
                        rounds_per_request=sums['blocks']/2,
                        decode_ms_per_round=1000*sums['decode_seconds']/sums['blocks'])
            for key, fn in (
                ('tps_range', lambda r: r['tokens']/r['outer_seconds']),
                ('acceptance_range', lambda r: r['accepted']/r['blocks']),
                ('rounds_range', lambda r: r['blocks']),
                ('decode_ms_per_round_range', lambda r: 1000*r['decode_seconds']/r['blocks']),
            ):
                values = [fn(r) for r in group]
                cell[key] = [min(values), max(values)]
            cells[str(length)][mode] = cell
        own, base = (cells[str(length)][mode] for mode in MODES)
        round_ratio = own['sums']['blocks']/base['sums']['blocks']
        average_ratio = own['decode_ms_per_round']/base['decode_ms_per_round']
        decode_ratio = own['sums']['decode_seconds']/base['sums']['decode_seconds']
        require(math.isclose(round_ratio*average_ratio, decode_ratio, rel_tol=1e-12),
                'Decode factorization does not close')
        decomposition[str(length)] = dict(
            outer_gap_seconds=own['sums']['outer_seconds']-base['sums']['outer_seconds'],
            prefill_gap_seconds=own['sums']['prefill_seconds']-base['sums']['prefill_seconds'],
            decode_gap_seconds=own['sums']['decode_seconds']-base['sums']['decode_seconds'],
            rounds_ratio=round_ratio, mean_decode_per_round_ratio=average_ratio,
            total_decode_ratio=decode_ratio, tps_effect_percent=100*(own['tps']/base['tps']-1))
    require(all(sha(root/name) == digest for name, digest in PINS.items()), 'Inputs changed during analysis')
    return dict(status='COMPLETE_POSTHOC_DESCRIPTIVE_NOT_CONFIRMATION', independent_documents=1,
                source_index=303, repeats=2, exact_context_lengths=list(LENGTHS), max_new_tokens=128,
                raw_sha256=PINS, cells=cells, decomposition=decomposition, goal_complete=False,
                interval='Observed two-repeat ranges only; no document-level95% CI possible at n=1',
                interpretation='Tdecode = rounds * mean measured decode time per round is an accounting identity, not causal attribution. Mean includes all decode-side work, not isolated Target latency. Different cache, Mapper, selection and length policies are confounded. Both repetitions are reused development; no fresh test or parameter search.')


def render(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors = ('#235b8d', '#cc7832')
    footer = ('POST-HOC; n=1 reused doc303, 2 paired timing repeats; exact4K/8K, cap128. H100, Qwen2 7B/1.5B;\n'
              'FP32 Target/BF16 Draft+prefill; real frozen V-only Mapper in RESCUE. Error bars: observed repeat range, NOT95%CI.')

    def bars(ax, field, bounds, ylabel):
        for j, mode in enumerate(MODES):
            values = [summary['cells'][str(n)][mode][field] for n in LENGTHS]
            ranges = [summary['cells'][str(n)][mode][bounds] for n in LENGTHS]
            errors = np.array([[v-b[0] for v, b in zip(values, ranges)],
                               [b[1]-v for v, b in zip(values, ranges)]])
            positions = np.arange(2)+(j-.5)*.32
            ax.bar(positions, values, width=.30, yerr=np.maximum(errors, 0), capsize=4,
                   color=colors[j], label=('RESCUE Graph', 'Retrieval Top32/k4 Graph')[j])
            for x, value in zip(positions, values):
                ax.text(x, value*1.03, f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks([0, 1], ['4096 tokens', '8192 tokens'])
        ax.set_ylabel(ylabel)
        maximum = max(summary['cells'][str(n)][mode][bounds][1]
                      for n in LENGTHS for mode in MODES)
        ax.set_ylim(0, maximum*1.20)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)

    for name, field, bounds, ylabel, title in (
        ('quality', 'accepted_per_round', 'acceptance_range', 'Recorded accepted Draft tokens / round', 'Quality: accepted Draft tokens'),
        ('tps', 'tps', 'tps_range', 'Warm whole-generate tokens / second', 'System performance: measured warm TPS'),
    ):
        fig, ax = plt.subplots(figsize=(9, 5))
        bars(ax, field, bounds, ylabel)
        fig.suptitle(title, y=.99)
        fig.legend(*ax.get_legend_handles_labels(), loc='upper center',
                   bbox_to_anchor=(.5, .935), ncol=2, fontsize=9)
        fig.text(.5, .015, footer, ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .1, 1, .85))
        for suffix in ('png', 'svg'):
            fig.savefig(output/(name+'.'+suffix), dpi=170)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.1))
    bars(axes[0], 'rounds_per_request', 'rounds_range', 'Verification rounds / request')
    bars(axes[1], 'decode_ms_per_round', 'decode_ms_per_round_range', 'Mean whole-decode milliseconds / round')
    axes[0].set_title('Number of rounds')
    axes[1].set_title('Average cost per round (NOT isolated Target latency)')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.5, .935), ncol=2, fontsize=8)
    fig.suptitle('Decode seconds = rounds × average decode cost per round; identity, NOT causal attribution', y=.99)
    fig.text(.5, .015, footer, ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .11, 1, .85))
    for suffix in ('png', 'svg'):
        fig.savefig(output/('decode_costs.'+suffix), dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.results)  # authenticate before any output creation
    require(not args.output.exists(), 'Fresh output only; never overwrite prior analysis')
    args.output.mkdir(parents=True)
    shutil.copyfile(Path(__file__), args.output/'analysis_source.py')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    render(summary, args.output)
    d = summary['decomposition']['8192']
    report = f'''# Post-hoc diagnostic, not new confirmation: old matched doc303 pilot

One reused development document, exact4096/8192 input tokens,128 output cap,
two timing repeats; no document-level95% CI can be estimated. Bars use only
observed repeat ranges. Same H100/Qwen2-7B/1.5B/precision; RESCUE retains real
nonzero V-only Mapper and the frozen causal2/4 policy. The retrieval comparator
is the old Top32/k4 Graph pilot, NOT the pending finite-menu selected winner.

At8K, pooled two-request prefill differs by {d['prefill_gap_seconds']:+.6f}s,
decode differs by {d['decode_gap_seconds']:+.6f}s, whole-generate differs by
{d['outer_gap_seconds']:+.6f}s. RESCUE executes84 verification rounds versus76
and has mean measured whole-decode cost67.6168ms versus64.5577ms per round.
Thus {d['total_decode_ratio']:.6f} = {d['rounds_ratio']:.6f} ×
{d['mean_decode_per_round_ratio']:.6f}. This is a descriptive accounting identity:
it does not isolate Target computation, establish causal Mapper harm, or prove
that changing either factor alone would realize the corresponding speed gain.

The future optimization question must distinguish fewer verification rounds
from cheaper rounds, not merely faster Draft kernels. No parameter was changed,
no candidate document was read, and no extra GPU experiment was executed.
The current frozen final protocol is unchanged. No comparison to other cohorts,
longer service traces, original Vicuna,4090 or production throughput is made.

- [Quality](quality.png) · [TPS](tps.png) · [Round/cost mechanism](decode_costs.png)
- [Analysis source](analysis_source.py) · [Summary and input pins](summary.json)
- [Raw warm requests]({args.results.resolve()}/runtime.jsonl); exact file SHA are in summary.json.
'''
    (args.output/'REPORT.md').write_text(report)
    print(json.dumps(dict(status=summary['status'], output=str(args.output), goal_complete=False)))


if __name__ == '__main__':
    main()
