"""Document-paired uncertainty, integrity checks, and a standalone pilot report."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
OUT = HERE / 'out_pilot'


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def interval(values):
    return np.quantile(values, [.025, .975]).tolist()


def main():
    offline = rows(OUT/'offline_test.jsonl')
    runtime = rows(OUT/'runtime_test.jsonl')
    config = json.loads((OUT/'test_methods.json').read_text())
    protocol = json.loads((OUT/'protocol.json').read_text())
    selected = json.loads((OUT/'selected.json').read_text())
    docs = sorted({r['source_index'] for r in runtime})
    assert len(docs) == 30
    assert len({(r['source_index'], r['method']) for r in runtime}) == len(runtime)
    assert len({(r['source_index'], r['step'], r['method']) for r in offline}) == len(offline)
    for doc in docs:
        assert {r['method'] for r in runtime if r['source_index'] == doc} == set(config) | {'teacher_greedy'}
        patterns = [{r['step'] for r in offline if r['source_index'] == doc and r['method'] == method}
                    for method in config]
        assert all(p == patterns[0] and p for p in patterns)
    agreements = sum(r['output_agrees'] for r in runtime if r['method'] != 'teacher_greedy')
    sd_total = sum(r['method'] != 'teacher_greedy' for r in runtime)
    completeness = dict(test_documents=len(docs), offline_rows=len(offline), runtime_rows=len(runtime),
                        unique_and_complete=True, sd_output_agreements=agreements, sd_total=sd_total,
                        failures=[dict(source_index=r['source_index'], method=r['method'])
                                  for r in runtime if not r['output_agrees']])
    write(OUT/'completeness.json', completeness)
    rng = np.random.default_rng(42)
    boot = rng.integers(0, len(docs), (5000, len(docs)))
    off_arrays = {}
    runtime_arrays = {}
    summary = dict(protocol=protocol, selected=selected, completeness=completeness,
                   offline={}, runtime={}, comparisons={})
    for method in config:
        values = np.array([[np.mean([r[key] for r in offline if r['source_index'] == doc and r['method'] == method])
                            for key in ('kl', 'teacher_ce', 'match')] for doc in docs])
        off_arrays[method] = values
        result = dict(zip(('kl', 'teacher_ce', 'match'), values.mean(0).tolist()))
        gates = [r['gate'] for r in offline if r['method'] == method]
        result['mean_gate'] = float(np.mean([g['mean'] for g in gates]))
        result['mean_within_query_gate_std'] = float(np.mean([g['std'] for g in gates]))
        summary['offline'][method] = result
    for method in list(config) + ['teacher_greedy']:
        records = [next(r for r in runtime if r['source_index'] == doc and r['method'] == method) for doc in docs]
        values = np.array([[r.get('accepted', 0), r.get('proposed', 0), r.get('blocks', 0),
                            r['tokens'], r['decode_seconds'], r['total_seconds']] for r in records])
        runtime_arrays[method] = values
        totals = values.sum(0)
        result = dict(acceptance=float(totals[0]/max(1, totals[1])), mean_accepted=float(totals[0]/max(1, totals[2])),
                      decode_tps=float(totals[3]/totals[4]), total_tps=float(totals[3]/totals[5]),
                      output_agreements=sum(r['output_agrees'] for r in records),
                      mean_gate=float(np.mean([r.get('gate_mean', 0) for r in records])))
        if method != 'teacher_greedy':
            result['stage_seconds_per_doc'] = {key: float(np.mean([r['stage_seconds'][key] for r in records]))
                                               for key in records[0]['stage_seconds']}
        summary['runtime'][method] = result
    base = 'teacher_token'
    for method in config:
        a, b = runtime_arrays[method], runtime_arrays[base]
        ab, bb = a[boot].sum(1), b[boot].sum(1)
        acceptance_delta = ab[:, 0]/ab[:, 1] - bb[:, 0]/bb[:, 1]
        accepted_delta = ab[:, 0]/ab[:, 2] - bb[:, 0]/bb[:, 2]
        ratio = (ab[:, 3]/ab[:, 4]) / (bb[:, 3]/bb[:, 4])
        off_delta = (off_arrays[method]-off_arrays[base])[boot].mean(1)
        point = summary['runtime'][method]
        baseline = summary['runtime'][base]
        summary['comparisons'][method] = dict(
            delta_kl=summary['offline'][method]['kl']-summary['offline'][base]['kl'],
            delta_kl_ci=interval(off_delta[:, 0]),
            delta_match=summary['offline'][method]['match']-summary['offline'][base]['match'],
            delta_match_ci=interval(off_delta[:, 2]),
            delta_acceptance=point['acceptance']-baseline['acceptance'],
            delta_acceptance_ci=interval(acceptance_delta),
            delta_mean_accepted=point['mean_accepted']-baseline['mean_accepted'],
            delta_mean_accepted_ci=interval(accepted_delta),
            decode_tps_ratio=point['decode_tps']/baseline['decode_tps'], decode_tps_ratio_ci=interval(ratio))
    for comparison in [('performance_gated', 'performance_fixed'),
                       ('performance_gated', 'gated_constant'), ('performance_gated', 'gated_shuffled'),
                       ('performance_fixed', 'native_full'), ('performance_gated', 'native_full')]:
        first, second = comparison
        a, b = runtime_arrays[first], runtime_arrays[second]
        ab, bb = a[boot].sum(1), b[boot].sum(1)
        difference = (off_arrays[first]-off_arrays[second])[boot].mean(1)
        summary['comparisons'][f'{first}_vs_{second}'] = dict(
            delta_kl=summary['offline'][first]['kl']-summary['offline'][second]['kl'],
            delta_kl_ci=interval(difference[:, 0]),
            delta_acceptance=summary['runtime'][first]['acceptance']-summary['runtime'][second]['acceptance'],
            delta_acceptance_ci=interval(ab[:, 0]/ab[:, 1] - bb[:, 0]/bb[:, 1]))
    write(OUT/'summary.json', summary)
    report = ['# Selective V injection: first pilot', '',
              'Qwen2-7B → 1.5B, frozen models, FP32 teacher / BF16 student, H100 eager attention. '
              '24 train / 12 validation / 30 new test documents; 600–2048 input tokens. '
              '512-position working cache, 128 remote V corrections in four final student layers. '
              'Teacher signals come only from a completed prefix; no fresh teacher forward per draft token.', '',
              '## Validation choices', '',
              'Checkpoint and strength use validation output KL + 0.25 teacher-argmax CE. '
              'Tables retain forced nonzero diagnostic strengths; the zero-allowed choices are reported below.', '',
              '| Method | Forced strength | Zero-allowed strength | Chosen epoch |', '|---|---:|---:|---:|']
    for name, choice in selected.items():
        history_path = OUT / ('train_reconstruction.json' if name == 'reconstruction' else f'train_{name}.json')
        history = json.loads(history_path.read_text())
        best = min(history, key=lambda r: r['objective'])
        report.append(f"| {name} | {choice['forced']['alpha']} | {choice['guarded']['alpha']} | {best['epoch']} |")
    report += ['', '## Held-out output and actual decoding', '',
               'KL and next-token agreement are document-equal means on teacher-forced states. '
               'Actual acceptance is total accepted draft tokens / total proposed tokens. '
               'TPS is total output tokens / total timed seconds. All SD routes receive the same '
               'completed-prefix attention acquisition; mapper and gate costs are included.', '',
               '| Method | Output KL ↓ | Next-token agreement % | Actual acceptance % | Accepted/block | Decode TPS | Total TPS |',
               '|---|---:|---:|---:|---:|---:|---:|']
    for method in config:
        o, r = summary['offline'][method], summary['runtime'][method]
        report.append(f"| {method} | {o['kl']:.4f} | {100*o['match']:.2f} | {100*r['acceptance']:.2f} | "
                      f"{r['mean_accepted']:.3f} | {r['decode_tps']:.2f} | {r['total_tps']:.2f} |")
    r = summary['runtime']['teacher_greedy']
    report.append(f"| teacher_greedy | — | — | — | — | {r['decode_tps']:.2f} | {r['total_tps']:.2f} |")
    report += ['', '## Paired changes versus native teacher-token selection', '',
               'Document bootstrap, 5,000 paired resamples, 95% intervals; no multiplicity correction.', '',
               '| Method | ΔKL [95% CI] | ΔAcceptance pp [95% CI] | Decode TPS ratio [95% CI] |',
               '|---|---:|---:|---:|']
    for method in config:
        c = summary['comparisons'][method]
        k, a, t = c['delta_kl_ci'], c['delta_acceptance_ci'], c['decode_tps_ratio_ci']
        report.append(f"| {method} | {c['delta_kl']:+.4f} [{k[0]:+.4f}, {k[1]:+.4f}] | "
                      f"{100*c['delta_acceptance']:+.2f} [{100*a[0]:+.2f}, {100*a[1]:+.2f}] | "
                      f"{c['decode_tps_ratio']:.3f} [{t[0]:.3f}, {t[1]:.3f}] |")
    report += ['', '## Gate and content controls', '',
               'gated_constant uses the same performance-gated mapper with a constant validation mean gate. '
               'gated_shuffled retains unshuffled teacher content as gate input and unchanged attention locations, '
               'shuffling only mapped content directions. Later query states and realized gates can change '
               'through upstream content corrections; gates are not replayed. This tests position-content '
               'alignment, not complete removal of all teacher content from the gate.', '',
               '| Comparison | ΔKL [95% CI] | ΔAcceptance pp [95% CI] |', '|---|---:|---:|']
    for first, second in [('performance_gated', 'performance_fixed'), ('performance_gated', 'gated_constant'),
                           ('performance_gated', 'gated_shuffled'), ('performance_fixed', 'native_full'),
                           ('performance_gated', 'native_full')]:
        c = summary['comparisons'][f'{first}_vs_{second}']
        k, a = c['delta_kl_ci'], c['delta_acceptance_ci']
        report.append(f"| {first} − {second} | {c['delta_kl']:+.4f} [{k[0]:+.4f}, {k[1]:+.4f}] | "
                      f"{100*c['delta_acceptance']:+.2f} [{100*a[0]:+.2f}, {100*a[1]:+.2f}] |")
    report += ['', '## Integrity and limits', '',
               f"- Complete unique records: {len(offline)} offline, {len(runtime)} runtime. "
               f"SD output agreement with the separately generated FP32 teacher: {agreements}/{sd_total}.",
               '- Zero gate exactly reproduces the native selection output; backing K/V are unchanged by the '
               'read transformation; only selected remote positions are corrected. Gate derivative finite-difference '
               'check passed. Frozen student weights receive no gradients during performance training.',
               '- Output supervision uses the shared tokenizer vocabulary, not full unshared padded logits. '
               'Greedy acceptance is evaluated, not directly differentiated. Test documents are new but training '
               'and validation are reused from stage two.',
               '- This is V-only content injection, not K+V injection. K is not mapped and Q/K projection weights '
               'remain frozen; later query activations can respond to earlier V corrections. '
               'Four late layers, one source pairing, one rank, one seed, and about 2K context do not establish '
               'general long-context gains or validate layer pairing as optimal.',
               '- Full teacher and student backing caches and full student prefill remain. This is not a storage '
               'compression or prefill-bypass result; timings are for eager linear validation, not optimized SpecExtend.',
               '', '[Method and reproduction](../README.md). Raw results: offline_test.jsonl, runtime_test.jsonl.']
    (OUT/'report.md').write_text('\n'.join(report)+'\n')
    labels = ['reconstruction', 'performance_fixed', 'performance_gated', 'gated_constant', 'gated_shuffled']
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(labels))
    for ax, metric, ci, scale, title in [
            (axes[0], 'delta_kl', 'delta_kl_ci', 1, 'Output KL change vs native selection (lower is better)'),
            (axes[1], 'delta_acceptance', 'delta_acceptance_ci', 100, 'Actual acceptance change (percentage points)')]:
        points = np.array([summary['comparisons'][m][metric]*scale for m in labels])
        bounds = np.array([summary['comparisons'][m][ci] for m in labels])*scale
        ax.errorbar(x, points, yerr=np.stack((points-bounds[:, 0], bounds[:, 1]-points)), fmt='o', capsize=4)
        ax.axhline(0, color='gray', linewidth=1)
        ax.set_xticks(x, ['Recon', 'Output\nfixed', 'Output\ngated', 'Same map\nconstant', 'Shuffled\ncontent'])
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=.25)
    figure.tight_layout()
    figure.savefig(OUT/'pilot_comparison.png', dpi=180)
    figure.savefig(OUT/'pilot_comparison.pdf')
    print(json.dumps(dict(completeness=completeness, offline=summary['offline'], runtime=summary['runtime']), indent=2))


if __name__ == '__main__':
    main()
