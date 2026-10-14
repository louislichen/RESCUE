"""Report the exploratory source-content necessity checks separately."""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / 'out_pilot'


def rows(name):
    return [json.loads(line) for line in (OUT/name).read_text().splitlines()]


def main():
    offline, runtime = rows('source_controls_offline.jsonl'), rows('source_controls_runtime.jsonl')
    methods = ['aligned', 'mean', 'zero']
    docs = sorted({r['source_index'] for r in runtime})
    assert len(docs) == 30 and len(runtime) == 90 and len(offline) == 354
    assert len({(r['source_index'], r['method']) for r in runtime}) == len(runtime)
    assert len({(r['source_index'], r['step'], r['method']) for r in offline}) == len(offline)
    for doc in docs:
        assert {r['method'] for r in runtime if r['source_index'] == doc} == set(methods)
    rng = np.random.default_rng(42)
    boot = rng.integers(0, 30, (5000, 30))
    off, run, summary = {}, {}, dict(methods={}, comparisons={})
    for method in methods:
        off[method] = np.array([[np.mean([r[k] for r in offline if r['source_index'] == doc and r['method'] == method])
                                for k in ('kl', 'match')] for doc in docs])
        rr = [next(r for r in runtime if r['source_index'] == doc and r['method'] == method) for doc in docs]
        run[method] = np.array([[r['accepted'], r['proposed'], r['blocks'], r['tokens'], r['decode_seconds']]
                               for r in rr])
        total = run[method].sum(0)
        summary['methods'][method] = dict(kl=float(off[method][:, 0].mean()), match=float(off[method][:, 1].mean()),
            acceptance=float(total[0]/total[1]), mean_accepted=float(total[0]/total[2]), decode_tps=float(total[3]/total[4]),
            output_agreements=sum(r['output_agrees'] for r in rr),
            gate_mean=float(np.mean([r['gate_mean'] for r in rr])))
    for method in ['mean', 'zero']:
        a, b = run['aligned'][boot].sum(1), run[method][boot].sum(1)
        diff = (off['aligned']-off[method])[boot].mean(1)
        summary['comparisons'][f'aligned_vs_{method}'] = dict(
            delta_kl=summary['methods']['aligned']['kl']-summary['methods'][method]['kl'],
            delta_kl_ci=np.quantile(diff[:, 0], [.025, .975]).tolist(),
            delta_acceptance=summary['methods']['aligned']['acceptance']-summary['methods'][method]['acceptance'],
            delta_acceptance_ci=np.quantile(a[:, 0]/a[:, 1]-b[:, 0]/b[:, 1], [.025, .975]).tolist())
    primary = {r['source_index']: r for r in rows('runtime_test.jsonl') if r['method'] == 'performance_gated'}
    repeated = [r for r in runtime if r['method'] == 'aligned']
    assert all(all(r[k] == primary[r['source_index']][k] for k in ['accepted', 'proposed', 'blocks', 'generated']) for r in repeated)
    summary['checks'] = dict(offline_rows=len(offline), runtime_rows=len(runtime),
                            exact_aligned_acceptance_trace_reproduction=True,
                            sd_output_agreements=sum(r['output_agrees'] for r in runtime), sd_total=len(runtime))
    (OUT/'source_controls_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    text = ['# Frozen source-content ablations', '',
            'Exploratory checks added after the primary shuffled-content result. No model, strength, '
            'checkpoint, candidate positions, or test-set tuning changed. Both mapper and gate receive '
            'either aligned teacher V, per-layer mean teacher V over the selected remote positions, or zero '
            'teacher V. Teacher-attention position selection remains available to all controls. '
            'Mean retains document-level teacher content; zero eliminates teacher V content input but is '
            'out of the trained source distribution. These controls do not replace a separately trained '
            'native-only adapter comparison.', '',
            '| Source | Output KL | Next-token agreement % | Actual acceptance % | Decode TPS | Mean gate |',
            '|---|---:|---:|---:|---:|---:|']
    for method in methods:
        r = summary['methods'][method]
        text.append(f"| {method} | {r['kl']:.4f} | {100*r['match']:.2f} | {100*r['acceptance']:.2f} | "
                    f"{r['decode_tps']:.2f} | {r['gate_mean']:.3f} |")
    text += ['', '| Comparison | ΔKL [95% CI] | ΔAcceptance pp [95% CI] |', '|---|---:|---:|']
    for name, r in summary['comparisons'].items():
        k, a = r['delta_kl_ci'], r['delta_acceptance_ci']
        text.append(f"| {name} | {r['delta_kl']:+.4f} [{k[0]:+.4f}, {k[1]:+.4f}] | "
                    f"{100*r['delta_acceptance']:+.2f} [{100*a[0]:+.2f}, {100*a[1]:+.2f}] |")
    text += ['', 'Paired document bootstrap: 5,000 resamples, 95% intervals, no multiplicity correction. '
             'Aligned acceptance/proposal/block counts and generated outputs exactly reproduce the primary run. '
             f"Output agreement: {summary['checks']['sd_output_agreements']}/{len(runtime)}. "
             'TPS comparisons here use this supplementary session only.', '', '[Primary pilot](report.md).']
    (OUT/'source_controls_report.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
