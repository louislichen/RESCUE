"""Standalone long-context comparison figures from the audited summary."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    args = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = json.loads((args.directory/'summary.json').read_text())
    lengths = sorted(map(int, result))
    names = dict(rescue_positive='RESCUE nonzero (dev selected)', rescue_guarded='RESCUE zero-allowed',
        rescue_dynamic='RESCUE injection-aware length', retrieval='SpecExtend retrieval (Qwen port)',
        full_specextend='SpecExtend tree (Qwen port)')
    colors = dict(rescue_positive='#136b9b', rescue_guarded='#589abd', rescue_dynamic='#269580',
        retrieval='#dd8b2b', full_specextend='#ad5278')
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for mode, label in names.items():
        values = [result[str(n)]['modes'][mode]['total_tps'] for n in lengths]
        axes[0].plot(lengths, values, marker='o', label=label, color=colors[mode], linewidth=1.5)
    axes[0].set_ylabel('Whole-request throughput (tokens/s)')
    axes[0].set_title('Matched Target/Draft weights and precision')
    for candidate, baseline, label, color, offset in (
        ('rescue_positive', 'retrieval', 'Nonzero vs retrieval', colors['rescue_positive'], -.025),
        ('rescue_dynamic', 'retrieval', 'Dynamic vs retrieval', colors['rescue_dynamic'], 0.),
        ('rescue_positive', 'full_specextend', 'Nonzero vs tree', colors['full_specextend'], .025)):
        comparison = [result[str(n)]['comparisons'][candidate+'_vs_'+baseline] for n in lengths]
        means = [r['mean_percent'] for r in comparison]
        errors = [[r['mean_percent']-r['ci95'][0] for r in comparison],
                  [r['ci95'][1]-r['mean_percent'] for r in comparison]]
        axes[1].errorbar([n*(2**offset) for n in lengths], means, yerr=errors,
            marker='o', capsize=3, label=label, color=color, linewidth=1.3)
    axes[1].axhline(0, color='#777777', linewidth=.8)
    axes[1].axhline(10, color='#777777', linewidth=.8, linestyle='--', label='Registered +10% threshold')
    axes[1].set_ylabel('Document-paired TPS change (%)')
    axes[1].set_title('95% document bootstrap intervals')
    for ax in axes:
        ax.set_xscale('log', base=2)
        ax.set_xticks(lengths, [f'{n//1024}K' for n in lengths])
        ax.set_xlabel('Input tokens')
        ax.grid(axis='y', alpha=.2)
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.18), fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(bottom=.36)
    for extension in ('png', 'pdf'):
        fig.savefig(args.directory/f'long_context_throughput.{extension}', dpi=180, bbox_inches='tight')


if __name__ == '__main__':
    main()
