"""Audit and plot a paired before/after execution-overhead experiment.

This analyzer never starts model inference. Incomplete matrices use complete
three-repeat pairs only, and every plot/report is explicitly marked PARTIAL.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics


MODES = ('before', 'after')
TRACE_FIELDS = ('round_index', 'cap', 'proposed', 'accepted', 'prefix', 'terminal', 'workset_tokens')


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def finite_number(value, *, positive=False):
    return (type(value) in (int, float) and math.isfinite(value) and
            (value > 0 if positive else value >= 0))


def integer(value, *, minimum=0):
    return type(value) is int and value >= minimum


def bootstrap(values, samples=10000, seed=42):
    """Resample documents, not requests, repeats, contexts, or query rows."""
    if not values:
        return None
    import numpy as np
    array = np.asarray(values, dtype=np.float64)
    draws = np.random.default_rng(seed).integers(0, len(array), size=(samples, len(array)))
    means = array[draws].mean(axis=1)
    return dict(mean=float(array.mean()), ci95=[float(v) for v in np.quantile(means, [.025, .975])],
                n_documents=len(values))


def trace_signature(row):
    return [[entry.get(field) for field in TRACE_FIELDS] for entry in row.get('trace', [])]


def reference_for(references, row):
    key = f"{row['source_index']}:{row['length']}"
    value = references.get(key)
    if isinstance(value, dict):
        value = value.get('generated')
    return value


def source_problems(directory, inventories):
    problems = []
    for mode in MODES:
        inventory = inventories.get(mode, {})
        if not isinstance(inventory, dict) or not inventory:
            problems.append(mode+': missing frozen source inventory')
            continue
        root = (directory/'source'/mode).resolve()
        for relative, expected_hash in inventory.items():
            target = (root/relative).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                problems.append(mode+': source path escapes root: '+relative)
                continue
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != expected_hash:
                problems.append(mode+': missing or hash-mismatched source: '+relative)
    return problems


def correctness_problems(row, reference, output_limit):
    issues = []
    if not all(row.get(k) is True for k in ('passed', 'output_agrees', 'trajectory_equal', 'rounds_equal')):
        issues.append('correctness booleans did not pass')
    if row.get('before', {}).get('rounds') != row.get('after', {}).get('rounds'):
        issues.append('saved selected positions/features/draft/checks differ')
    if row.get('before', {}).get('trace_signature') != row.get('after', {}).get('trace_signature'):
        issues.append('saved trace signatures differ')
    for mode in MODES:
        part = row.get(mode, {})
        if part.get('generated') != reference:
            issues.append(mode+': canonical output mismatch')
        rounds, signature = part.get('rounds'), part.get('trace_signature')
        if not isinstance(rounds, list) or not rounds or not isinstance(signature, list) or len(signature) != len(rounds):
            issues.append(mode+': missing round/signature evidence')
            continue
        observed_nonzero = False
        for index, packet in enumerate(rounds):
            vector = packet.get('budget_features')
            if vector is not None:
                if not isinstance(vector, list) or len(vector) != 12 or not all(
                        type(v) in (int, float) and math.isfinite(v) for v in vector):
                    issues.append(mode+': malformed causal feature vector')
                else:
                    observed_nonzero |= vector[-1] > 0
            prefix, progress = packet.get('prefix'), packet.get('progress')
            selected = packet.get('selected')
            if (packet.get('round_index') != index or not integer(prefix, minimum=row['length']-1) or
                    not integer(progress) or prefix != row['length']-1+progress or
                    packet.get('remaining') != output_limit-progress or not isinstance(selected, list) or
                    not selected or not all(integer(v) and v < prefix for v in selected) or selected != sorted(set(selected))):
                issues.append(mode+': invalid saved prefix/position state')
            draft, checks = packet.get('draft'), packet.get('checks')
            if (not isinstance(draft, list) or not draft or not isinstance(checks, list) or len(checks) != len(draft)+1 or
                    not all(integer(v) for v in draft+checks)):
                issues.append(mode+': invalid draft/check sequence')
                continue
            leading = next((i for i, (a, b) in enumerate(zip(draft, checks)) if a != b), len(draft))
            if packet.get('accepted') != leading:
                issues.append(mode+': accepted count disagrees with saved checks')
        if row.get('nonzero_injection', {}).get(mode) is not True or not observed_nonzero:
            issues.append(mode+': no saved actual nonzero-injection evidence')
    return issues


def row_problems(row, references, output_limit):
    issues = []
    generated = row.get('generated')
    if (not isinstance(generated, list) or not generated or
            not all(integer(v) for v in generated) or len(generated) > output_limit):
        issues.append('invalid generated token list or output limit')
    reference = reference_for(references, row)
    if row.get('output_agrees') is not True or generated != reference:
        issues.append('saved output is not equal to canonical Target reference')
    if not integer(row.get('tokens'), minimum=1) or row.get('tokens') != len(generated or []):
        issues.append('tokens does not match saved output length')
    for field in ('prefill_seconds', 'decode_seconds', 'cleanup_seconds', 'total_seconds'):
        if not finite_number(row.get(field), positive=field in ('decode_seconds', 'total_seconds')):
            issues.append('invalid '+field)
    if (all(finite_number(row.get(k), positive=k in ('decode_seconds', 'total_seconds'))
            for k in ('prefill_seconds', 'decode_seconds', 'cleanup_seconds', 'total_seconds')) and
            integer(row.get('tokens'), minimum=1)):
        if not math.isclose(sum(row[k] for k in ('prefill_seconds', 'decode_seconds', 'cleanup_seconds')),
                            row['total_seconds'], rel_tol=1e-6, abs_tol=1e-6):
            issues.append('prefill/decode/cleanup do not sum to total_seconds')
        for metric, seconds in (('total_tps', 'total_seconds'), ('decode_tps', 'decode_seconds')):
            if not finite_number(row.get(metric), positive=True) or not math.isclose(
                    row[metric], row['tokens']/row[seconds], rel_tol=1e-6, abs_tol=1e-7):
                issues.append(metric+' is inconsistent with tokens/time')
    trace = row.get('trace')
    if not isinstance(trace, list) or not trace:
        issues.append('missing per-round trace')
        return issues
    if not integer(row.get('blocks'), minimum=1) or row.get('blocks') != len(trace):
        issues.append('blocks does not match trace length')
    valid_trace = True
    expected_prefix = row['length']-1  # The current unverified root is excluded from prefill.
    for index, entry in enumerate(trace):
        cap, proposed, accepted = (entry.get(k) for k in ('cap', 'proposed', 'accepted'))
        if not (integer(cap, minimum=1) and integer(proposed, minimum=1) and
                integer(accepted) and accepted <= proposed <= cap <= 4):
            valid_trace = False
        if (entry.get('round_index') != index or not integer(entry.get('prefix'), minimum=row['length']-1) or
                entry.get('prefix') != expected_prefix or
                type(entry.get('terminal')) is not bool or not integer(entry.get('workset_tokens'), minimum=1)):
            valid_trace = False
        if entry.get('terminal') is not (index == len(trace)-1):
            valid_trace = False
        if integer(accepted):
            expected_prefix += accepted+1
    if not valid_trace:
        issues.append('invalid cap/proposed/accepted trace')
    else:
        for field in ('proposed', 'accepted'):
            if not integer(row.get(field)) or row.get(field) != sum(entry[field] for entry in trace):
                issues.append(field+' total differs from trace')
        if integer(row.get('accepted')) and integer(row.get('blocks'), minimum=1) and integer(row.get('tokens')):
            if not row['accepted']+row['blocks']-1 <= row['tokens'] <= row['accepted']+row['blocks']:
                issues.append('output length does not match accepted tokens plus committed roots/bonuses')
    return issues


def summarize(directory, *, samples=10000, seed=42):
    directory = Path(directory)
    protocol = read_json(directory/'protocol.json')
    references = read_json(directory/'references.json')
    ids = protocol.get('source_indices', protocol.get('document_ids'))
    lengths = protocol.get('lengths')
    repeats = protocol.get('repeats')
    if (not isinstance(ids, list) or not ids or len(ids) != len(set(ids)) or
            any(not integer(v) for v in ids)):
        raise ValueError('Protocol requires unique integer source_indices')
    if lengths != [4096, 8192] or repeats != 3:
        raise ValueError('This frozen analysis requires lengths [4096,8192] and three repeats')
    config = protocol.get('config', {})
    output_limit = protocol.get('max_new_tokens', config.get('max_new_tokens'))
    max_draft = protocol.get('max_draft', config.get('max_draft'))
    if output_limit != 128 or max_draft != 12:
        raise ValueError('This analysis requires max_new_tokens=128 and max_draft=12')
    expected = {(mode, doc, n, repeat) for mode in MODES for doc in ids
                for n in lengths for repeat in range(repeats)}
    rows = read_jsonl(directory/'runtime.jsonl')
    indexed, invalid = {}, []
    for row in rows:
        key = tuple(row.get(k) for k in ('mode', 'source_index', 'length', 'repeat'))
        if key not in expected:
            raise ValueError('Runtime row outside protocol matrix: '+repr(key))
        if key in indexed:
            raise ValueError('Duplicate timing row: '+repr(key))
        indexed[key] = row
        for problem in row_problems(row, references, output_limit):
            invalid.append(dict(key=list(key), problem=problem))
    bad_keys = {tuple(v['key']) for v in invalid}
    comparisons, invariant_failures = {}, []
    for doc in ids:
        for n in lengths:
            for repeat in range(repeats):
                bk, ak = ('before', doc, n, repeat), ('after', doc, n, repeat)
                if bk in indexed and ak in indexed:
                    same = (indexed[bk].get('generated') == indexed[ak].get('generated') and
                            trace_signature(indexed[bk]) == trace_signature(indexed[ak]))
                    comparisons[(doc, n, repeat)] = same
                    if not same:
                        invariant_failures.append([doc, n, repeat])
    correctness = read_jsonl(directory/'correctness.jsonl')
    correctness_index = {}
    for row in correctness:
        key = (row.get('source_index'), row.get('length'))
        if key not in {(doc, n) for doc in ids for n in lengths} or key in correctness_index:
            raise ValueError('Duplicate or out-of-matrix correctness pair: '+repr(key))
        correctness_index[key] = row
    correctness_bad = []
    for key, row in correctness_index.items():
        ref = references.get(f'{key[0]}:{key[1]}')
        if isinstance(ref, dict):
            ref = ref.get('generated')
        problems = correctness_problems(row, ref, output_limit)
        if problems:
            correctness_bad.append(dict(key=list(key), problems=problems))
    complete_path = directory/'COMPLETE.json'
    if not complete_path.exists():
        complete_path = directory/'COMPLETE'
    receipt = read_json(complete_path) if complete_path.exists() else {}
    metadata = dict(six_documents=len(ids) == 6,
                    nonzero_mapper=protocol.get('mapper_nonzero') is True,
                    frozen_mapper=protocol.get('mapper_frozen') is True,
                    cold_request=protocol.get('request_buffer_policy') == 'cold_request')
    environment = protocol.get('environment', {})
    metadata['h100_80gb'] = ('H100' in environment.get('gpu_name', '') and
        ('80GB' in environment.get('gpu_name', '').upper() or
         finite_number(environment.get('total_memory')) and 75*1024**3 <= environment['total_memory'] <= 85*1024**3))
    source_failures = source_problems(directory, protocol.get('source_sha256', {}))
    gates = dict(full_timing_matrix=set(indexed) == expected,
                 all_timing_outputs_and_accounting_valid=bool(rows) and not invalid,
                 cap_acceptance_trajectory_invariant=bool(comparisons) and not invariant_failures,
                 full_correctness_matrix=len(correctness_index) == len(ids)*len(lengths),
                 nonzero_correctness_exact=bool(correctness_index) and not correctness_bad,
                 completion_receipt=receipt.get('passed') is True,
                 frozen_sources_hash_match=not source_failures,
                 assets_and_sources_unchanged=receipt.get('assets_unchanged') is True and receipt.get('source_unchanged') is True,
                 no_failure_marker=not (directory/'FAILURE.json').exists(),
                 **metadata)
    expected_reference_keys = {f'{doc}:{n}' for doc in ids for n in lengths}
    gates['full_reference_matrix'] = expected_reference_keys <= set(references)
    for key, expected_count in (('expected_timing_requests', len(expected)), ('timing_requests', len(expected)),
                                ('correctness_pairs', len(ids)*len(lengths)),
                                ('reference_requests', len(ids)*len(lengths)), ('warmup_requests', 4)):
        if receipt.get(key) != expected_count:
            gates['completion_receipt'] = False
    per_document, by_context = [], {}
    for n in lengths:
        docs = []
        for doc in ids:
            keys = [(mode, doc, n, repeat) for mode in MODES for repeat in range(repeats)]
            # A missing repeat is never replaced by a smaller-repeat median.
            if any(k not in indexed or k in bad_keys for k in keys) or any(
                    not comparisons.get((doc, n, repeat), False) for repeat in range(repeats)):
                continue
            item = dict(source_index=doc, length=n, modes={})
            for mode in MODES:
                rr = [indexed[(mode, doc, n, repeat)] for repeat in range(repeats)]
                item['modes'][mode] = {field: statistics.median(r[field] for r in rr)
                    for field in ('total_tps', 'decode_tps', 'prefill_seconds', 'decode_seconds',
                                  'cleanup_seconds', 'total_seconds')}
                item['modes'][mode]['acceptance_length'] = statistics.median(r['accepted']/r['blocks'] for r in rr)
                item['modes'][mode]['acceptance_rate'] = statistics.median(r['accepted']/max(1, r['proposed']) for r in rr)
                item['modes'][mode]['blocks'] = statistics.median(r['blocks'] for r in rr)
            before, after = (item['modes'][m] for m in MODES)
            item['tps_change_percent'] = (after['total_tps']/before['total_tps']-1)*100
            item['acceptance_length_delta'] = after['acceptance_length']-before['acceptance_length']
            docs.append(item)
        per_document.extend(docs)
        fields = ('total_tps', 'decode_tps', 'acceptance_length', 'acceptance_rate', 'prefill_seconds',
                  'decode_seconds', 'cleanup_seconds', 'total_seconds', 'blocks')
        by_context[str(n)] = dict(n_documents=len(docs), expected_documents=len(ids),
            modes={mode: {field: bootstrap([r['modes'][mode][field] for r in docs], samples, seed)
                          for field in fields} for mode in MODES},
            tps_change_percent=bootstrap([r['tps_change_percent'] for r in docs], samples, seed),
            acceptance_length_delta=bootstrap([r['acceptance_length_delta'] for r in docs], samples, seed))
    status = 'COMPLETE' if all(gates.values()) else 'PARTIAL'
    speed_gates = {n: bool(status == 'COMPLETE' and value['n_documents'] == len(ids) and
        value['tps_change_percent'] is not None and value['tps_change_percent']['ci95'][0] > 0)
        for n, value in by_context.items()}
    return dict(status=status, gates=gates, protocol=protocol, expected_timing_rows=len(expected),
        observed_timing_rows=len(rows), missing_timing_rows=[list(k) for k in sorted(expected-set(indexed))],
        invalid_rows=invalid, invariant_failures=invariant_failures, correctness_failures=correctness_bad,
        source_failures=source_failures,
        per_document=per_document, contexts=by_context,
        speed_ci_lower_gt_zero=speed_gates, both_contexts_speed_gate=all(speed_gates.values()),
        bootstrap=dict(samples=samples, seed=seed,
        unit='document; within each document/context/mode take the median of all three TPS repeats'),
        scope='H100 80GB execution-only before/after; frozen nonzero mapper; 2/4 draft decisions (capacity 12), output limit 128; cold requests; '
              'not independent held-out generalization, not RTX 4090, not a SpecExtend comparison')


def render_figures(summary, destination, diagnostic_rows=()):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'svg.fonttype': 'none'})
    destination = Path(destination)
    contexts = summary['contexts']
    ns = sorted(map(int, contexts))
    colors = {'before': '#738392', 'after': '#137f9c'}
    counts = ', '.join(f'{n//1024}K n={contexts[str(n)]["n_documents"]}/6 docs' for n in ns)
    caption = (f'{summary["status"]} | H100 80GB | frozen nonzero Mapper | 2/4 drafts / output limit 128 | cold request\n'
               f'{counts}; median of 3 repeats; 95% document bootstrap. Development reuse; not 4090/SpecExtend.')

    def finish(fig, stem):
        fig.text(.5, .01, caption, ha='center', va='bottom', fontsize=8)
        fig.tight_layout(rect=(0, .15, 1, .97))
        for extension in ('png', 'svg'):
            fig.savefig(destination/f'{stem}.{extension}', dpi=180, bbox_inches='tight')
        plt.close(fig)

    def series(ax, field, ylabel):
        x = np.arange(len(ns), dtype=float)
        for mode, offset in (('before', -.17), ('after', .17)):
            values = [contexts[str(n)]['modes'][mode][field] for n in ns]
            means = [v['mean'] if v else np.nan for v in values]
            errors = [[v['mean']-v['ci95'][0] if v else 0. for v in values],
                      [v['ci95'][1]-v['mean'] if v else 0. for v in values]]
            ax.bar(x+offset, means, .32, color=colors[mode], label=mode,
                   yerr=errors, error_kw=dict(capsize=3, linewidth=1))
        ax.set_xticks(x, [f'{n//1024}K ({n} input tokens)' for n in ns])
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=.2)
        ax.legend()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    series(ax, 'acceptance_length', 'Accepted Draft tokens / verification round')
    ax.set_title('Quality: acceptance length (separate from throughput)')
    finish(fig, 'acceptance_length')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    series(axes[0], 'total_tps', 'Whole-request tokens / second')
    axes[0].set_title('Includes prefill, control, Draft, verify and cleanup')
    values = [contexts[str(n)]['tps_change_percent'] for n in ns]
    for i, value in enumerate(values):
        if value:
            axes[1].errorbar(i, value['mean'], yerr=[[value['mean']-value['ci95'][0]],
                [value['ci95'][1]-value['mean']]], fmt='o', color=colors['after'], capsize=4)
    axes[1].axhline(0., color='#777', linewidth=.8)
    axes[1].set_xticks(range(len(ns)), [f'{n//1024}K' for n in ns])
    axes[1].set_ylabel('After / before TPS change (%)')
    axes[1].set_title('Paired documents, 95% CI')
    axes[1].grid(axis='y', alpha=.2)
    finish(fig, 'whole_request_tps')

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, field, title in zip(axes, ('prefill_seconds', 'decode_seconds', 'cleanup_seconds'),
                                ('Prefill', 'Decode', 'Cleanup')):
        series(ax, field, 'Seconds / complete request')
        ax.set_title(title)
    finish(fig, 'stage_latency')

    fig, ax = plt.subplots(figsize=(8, 4.5))
    docs = summary['protocol'].get('source_indices', summary['protocol'].get('document_ids'))
    for n, offset, color in ((4096, -.12, '#7959a2'), (8192, .12, '#137f9c')):
        rows = [r for r in summary['per_document'] if r['length'] == n]
        ax.scatter([docs.index(r['source_index'])+offset for r in rows],
                   [r['tps_change_percent'] for r in rows], label=f'{n//1024}K', color=color)
    ax.axhline(0., color='#777', linewidth=.8)
    ax.set_xticks(range(len(docs)), [str(d) for d in docs])
    ax.set_xlabel('Document source index (one dot per complete 3-repeat pair)')
    ax.set_ylabel('Whole-request TPS change (%)')
    ax.set_title('Per-document paired changes; no missing-pair imputation')
    ax.grid(axis='y', alpha=.2)
    ax.legend()
    finish(fig, 'per_document_tps_delta')

    # Optional instrumented requests are never mixed into primary timing.
    stages = sorted({stage for r in diagnostic_rows for stage in r.get('timeline', {}).get('stages', {})})
    if stages:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, metric in zip(axes, ('host_seconds', 'stream_seconds')):
            for mode, offset in (('before', -.18), ('after', .18)):
                values = [statistics.mean(r['timeline']['stages'][stage].get(metric, 0.)
                    for r in diagnostic_rows if r.get('mode') == mode and stage in r.get('timeline', {}).get('stages', {}))
                    if any(r.get('mode') == mode and stage in r.get('timeline', {}).get('stages', {}) for r in diagnostic_rows)
                    else 0. for stage in stages]
                ax.bar(np.arange(len(stages))+offset, values, .34, label=mode, color=colors[mode])
            ax.set_xticks(range(len(stages)), stages, rotation=55, ha='right')
            ax.set_ylabel('Seconds / instrumented request (descriptive mean)')
            ax.set_title(metric+'; intervals overlap, do not sum')
            ax.legend()
        finish(fig, 'instrumented_stages')


def write_report(directory, destination, *, samples=10000, seed=42):
    directory, destination = Path(directory), Path(destination)
    if destination.exists():
        raise ValueError('Use a fresh analysis directory; do not overwrite an earlier report')
    summary = summarize(directory, samples=samples, seed=seed)
    destination.mkdir(parents=True)
    diagnostics = read_jsonl(directory/'diagnostics.jsonl')
    render_figures(summary, destination, diagnostics)
    (destination/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    shutil.copyfile(__file__, destination/'analysis_source.py')
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    lines = [f'# {summary["status"]}: paired execution-overhead comparison', '', summary['scope']+'.', '',
        'Only the three authorized overhead changes are compared; mapper/length weights and algorithm settings remain frozen. '
        'This is not evidence of superiority over native SD, SpecExtend, or Target AR.', '',
        f'Measured rows: {summary["observed_timing_rows"]}/{summary["expected_timing_rows"]}. '
        'The statistical unit is a document, not a timing repeat or query. 4K/8K are analyzed separately; '
        'their nested prefixes are not independent documents. All three timing repeats are required in both arms before a document/context enters a figure.', '',
        '| Context | Complete paired documents | Mean whole-request TPS change, 95% CI | Acceptance-length delta, 95% CI |',
        '|---|---:|---:|---:|']
    for n, value in summary['contexts'].items():
        def interval(result):
            return ('not available' if result is None else
                    f'{result["mean"]:+.4f} [{result["ci95"][0]:+.4f}, {result["ci95"][1]:+.4f}]')
        lines.append(f'| {int(n)//1024}K ({n} tokens) | {value["n_documents"]}/6 | '
                     f'{interval(value["tps_change_percent"])}% | {interval(value["acceptance_length_delta"])} tokens/round |')
    lines += ['', '## Scientific conclusion (distinct from completion)', '',
        'COMPLETE means the registered data, correctness checks and reports are complete; it does not mean acceleration is established. '
        'The strict speed criterion is a positive lower endpoint of the paired document 95% CI at both contexts.', '']
    for n, passed in summary['speed_ci_lower_gt_zero'].items():
        lines.append(f'- {int(n)//1024}K speed CI lower > 0: {"PASS (within this reused-development pilot only)" if passed else "NOT ESTABLISHED"}.')
    lines += [f'- Both-context speed criterion: {"PASS within this pilot" if summary["both_contexts_speed_gate"] else "NOT MET"}.', '',
        'The controller still makes causal 2-versus-4 Draft-token decisions after two proposals. '
        'The shared max_draft=12 is buffer capacity, not a measured twelve-token proposal policy. '
        'The three execution changes are tested jointly; this run does not isolate the contribution of any one change.', '',
        '## Gates', '']
    lines += [f'- {key}: {"PASS" if passed else "NOT PROVEN / FAIL"}' for key, passed in summary['gates'].items()]
    lines += ['', 'Incomplete rows and invalid/changed pairs are not imputed. See summary.json for all missing cells and failures. '
        'A PARTIAL report is not completion or a successful acceleration claim. Confidence intervals are descriptive document bootstrap '
        'intervals at fixed training weights; no training-seed uncertainty or multiplicity correction.', '',
        '## Figures and evidence', '',
        '- [Acceptance length](acceptance_length.png) ([editable SVG](acceptance_length.svg))',
        '- [Whole-request TPS](whole_request_tps.png) ([editable SVG](whole_request_tps.svg))',
        '- [Stage latency](stage_latency.png) ([editable SVG](stage_latency.svg))',
        '- [Per-document changes](per_document_tps_delta.png) ([editable SVG](per_document_tps_delta.svg))',
        '- [Analysis summary](summary.json)', '- [Figure/analysis source](analysis_source.py)',
        '- Raw requests, references, protocol and correctness records are in the parent experiment directory.',
        '', f'Analysis-source SHA256: `{source_hash}`.', '',
        'Request timings include actual prefill/control/Draft/verification/cleanup. Stage host/stream diagnostics, when available, '
        'are separate instrumented observations; overlapping intervals cannot be summed into independent kernel costs. '
        'CPU synthetic tests validate this program, not H100 performance.']
    if diagnostics:
        lines += ['', '[Instrumented stage diagnostics](instrumented_stages.png), separate from primary timed requests.']
    for filename in ('protocol.json', 'references.json', 'runtime.jsonl', 'correctness.jsonl', 'COMPLETE.json', 'COMPLETE', 'diagnostics.jsonl'):
        path = directory/filename
        if path.exists():
            import os
            target = Path(os.path.relpath(path, destination)).as_posix()
            lines.append(f'- [{filename}]({target})')
    (destination/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error('Use at least 100 bootstrap draws')
    summary = write_report(args.directory, args.output or args.directory/'analysis',
                           samples=args.bootstrap, seed=args.seed)
    print(json.dumps(dict(status=summary['status'], gates=summary['gates'], contexts=summary['contexts']), indent=2))
    return 0 if summary['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
