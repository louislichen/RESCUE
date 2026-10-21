"""Supplementary measured-context lines from the saved frozen devcheck report.

No new experiment, winner selection, interpolation samples or statistic changes.
The full frozen report retains every challenger; this view highlights two modes
already retained at both contexts, never chosen by observing RESCUE performance.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

MODES = ('retrieval32_k4_graph', 'tree32_n8')
REQUESTED_CONTEXTS = (2048, 4096, 6144, 8192, 12288, 16384)
LABELS = {'rescue_graph': 'RESCUE (frozen Mapper + causal 2/4)',
          'retrieval32_k4_graph': 'Cross-model retrieval port: Top32 / k4 / Graph',
          'tree32_n8': 'Full-tree Qwen2 port: Top32 / 8 nodes'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(value, message):
    if not value:
        raise RuntimeError(message)


def render(analysis, output):
    analysis, output = Path(analysis).resolve(), Path(output).resolve()
    need(analysis.is_dir() and not analysis.is_symlink() and not output.exists(),
         'Existing saved analysis and fresh plot output required')
    receipt = json.loads((analysis / 'ANALYSIS_COMPLETE.json').read_text())
    summary = json.loads((analysis / 'summary.json').read_text())
    need(receipt['file_sha256']['summary.json'] == sha(analysis / 'summary.json'), 'Saved summary SHA changed')
    need(not summary.get('synthetic_program_test_only'), 'Never present synthetic fixtures as measured curves')
    need(summary.get('fresh_confirmation') is False and summary.get('goal_complete') is False,
         'Keep development-only scope')
    stats = summary['statistics']
    contexts = sorted(int(n) for n, arms in stats['contexts'].items()
                      if all(m in arms for m in ('rescue_graph', *MODES)))
    need(contexts and all(n in (4096, 8192) for n in contexts), 'Only actually measured fixed-protocol contexts allowed')
    contrast = {(r['length'], r['baseline_mode']): r for r in stats['contrasts']}
    need(all((n, m) in contrast for n in contexts for m in MODES), 'Missing paired-document contrasts')
    before = sha(analysis / 'summary.json')
    source = Path(__file__).resolve()
    output.mkdir(parents=True)
    shutil.copyfile(source, output / 'plot_source.py')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'axes.unicode_minus': True})
    x = np.array(contexts, float) / 1024
    # The original two-context experiment can be complete while the user's
    # newly requested six-context curve is still only a measured subset.
    partial = summary['status'] != 'COMPLETE' or tuple(contexts) != REQUESTED_CONTEXTS
    prefix = 'PARTIAL — ' if partial else ''
    nlabel = ', '.join(f'{n//1024}K: n={stats["contexts"][str(n)]["rescue_graph"]["documents"]}' for n in contexts)
    note = ('H100 80GB; Qwen2-7B/1.5B; Target FP32 / Draft+prefix BF16; TF32 off.\n'
            + nlabel + ' paired reused-development docs, 2 repeats/doc, max128/EOS; frozen nonzero Mapper + causal2/4 oldcost.\n'
            'Whole-generate warm TPS; all lazy captures charged. 95% paired-document percentile CIs (10,000 draws).\n'
            'PARTIAL six-length extension: only 4K/8K measured in this stage; no invented 2K/6K/12K/16K points.\n'
            'Historical owner blocks; not fresh confirmation/production. Lines connect measured points only.')
    products = []

    def save(fig, stem):
        fig.text(.02, .02, note, va='bottom', fontsize=8)
        fig.tight_layout(rect=(0, .23, 1, 1))
        for ext in ('.png', '.svg'):
            fig.savefig(output / (stem + ext), dpi=180)
        plt.close(fig)
        products.append(stem)

    for mode, stem, color in ((MODES[0], 'length_tps_gain', '#176b87'),
                              (MODES[1], 'length_tps_gain_full_tree', '#9b5b17')):
        points = [contrast[n, mode] for n in contexts]
        y = np.array([r['point_percent'] for r in points], float)
        intervals = [r.get('ci95') for r in points]
        fig, ax = plt.subplots(figsize=(8.3, 5.8))
        ax.axhline(0, color='#666666', linewidth=1, linestyle='--', label='Equal TPS')
        ax.plot(x, y, '-o', color=color, linewidth=2, markersize=7)
        for xi, yi, bounds in zip(x, y, intervals):
            if bounds is not None:
                ax.vlines(xi, bounds[0], bounds[1], color=color, linewidth=1.8)
                ax.hlines(bounds, xi - .07, xi + .07, color=color, linewidth=1.5)
            ax.annotate(f'{yi:+.2f}%', (xi, yi), xytext=(9, 9), textcoords='offset points',
                        fontsize=10, color=color)
        all_y = [0., *y, *(v for bounds in intervals if bounds is not None for v in bounds)]
        span = max(max(all_y) - min(all_y), 2.)
        ax.set_ylim(min(all_y) - .15 * span, max(all_y) + .24 * span)
        ax.set_xlim(min(x) - .65, max(x) + .85)
        ax.set_xticks(x, [f'{n//1024}K\n({n:,} tokens)' for n in contexts])
        ax.set_xlabel('Actual input context length')
        ax.set_ylabel('RESCUE / baseline whole-request TPS − 1 (%)')
        ax.set_title(prefix + 'TPS gain by input length\nvs ' + LABELS[mode], pad=12)
        ax.grid(axis='y', alpha=.2)
        ax.legend(loc='best', frameon=False)
        save(fig, stem)

    fig, ax = plt.subplots(figsize=(8.3, 5.8))
    quality_extent = []
    for mode, color, marker in (('rescue_graph', '#176b87', 'o'), (MODES[0], '#a56216', 's'),
                                 (MODES[1], '#6c5797', '^')):
        rows = [stats['contexts'][str(n)][mode] for n in contexts]
        y = [r['acceptance_length'] for r in rows]
        quality_extent.extend(y)
        ax.plot(x, y, '-' + marker, color=color, label=LABELS[mode], linewidth=1.8)
        for xi, row in zip(x, rows):
            bounds = row.get('acceptance_length_ci95')
            if bounds is not None:
                quality_extent.extend(bounds)
                ax.vlines(xi, bounds[0], bounds[1], color=color, linewidth=1.4)
                ax.hlines(bounds, xi - .06, xi + .06, color=color, linewidth=1.2)
    ax.set_xticks(x, [f'{n//1024}K\n({n:,} tokens)' for n in contexts])
    ax.set_xlim(min(x) - .65, max(x) + .85)
    ax.set_xlabel('Actual input context length')
    ax.set_ylabel('Accepted Draft tokens / verification round')
    ax.set_title(prefix + 'Acceptance quality by input length\nTree counts include charged overshoot; not a tree-node acceptance rate', pad=12)
    quality_span = max(max(quality_extent) - min(quality_extent), .5)
    # Keep the three-line legend above the measured points and their whiskers.
    ax.set_ylim(min(quality_extent) - .08 * quality_span, max(quality_extent) + .45 * quality_span)
    ax.grid(axis='y', alpha=.2)
    ax.legend(loc='best', frameon=False, fontsize=8)
    save(fig, 'length_acceptance')
    need(sha(analysis / 'summary.json') == before, 'Saved primary summary changed while rendering')
    lines = ['# ' + prefix + 'Measured input-length curves', '',
             'Supplementary view of the unchanged frozen-development report; no added GPU requests, new statistics or winner selection.', '',
             'Only exact4096/8192 were measured here. The original matched-development experiment is complete, but this is only2/6 points of the newly requested2K/4K/6K/8K/12K/16K extension. Lines are visual connectors, not interpolation measurements or evidence at other lengths.', '',
             'The primary speed figure compares the retained Top32/k4 Graph cross-model-retrieval port. A separate full-tree curve avoids replacing the faster opponent with a weaker one. All12 retained comparisons remain in the primary report.', '',
             'Quality and whole-request TPS are separate. 95% intervals use16 whole paired documents after repeat pooling, not32 independent requests. Historical nonconcurrent owner-block/capture-path and reused-development limits remain.', '']
    for stem in products:
        lines.append(f'[{stem} PNG]({stem}.png) · [SVG]({stem}.svg)')
    lines += ['', f'Primary report: {analysis / "REPORT.md"}', f'Summary SHA256: {before}',
              'Raw results: ' + str(analysis.parent / 'runtime.jsonl'), 'Source: plot_source.py', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))
    final = dict(status='PARTIAL_MEASURED_SUBSET' if partial else 'COMPLETE_DEVELOPMENT_DISPLAY_ONLY',
                 created_at_utc=datetime.now(timezone.utc).isoformat(), primary_summary_sha256=before,
                 plot_source_sha256=sha(source), measured_contexts=contexts, no_new_measurements=True,
                 requested_contexts=list(REQUESTED_CONTEXTS), original_experiment_status=summary['status'],
                 no_new_selection=True, no_new_statistics=True, goal_complete=False,
                 figures=products, file_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()})
    (output / 'PLOT_MANIFEST.json').write_text(json.dumps(final, indent=2, sort_keys=True) + '\n')
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.analysis, args.output), indent=2))
