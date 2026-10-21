"""Audit and visualize the one-document Draft execution pilot, without models."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics


MODES = ('baseline_eager', 'resident', 'cuda_graph')
TRACE_FIELDS = ('round_index', 'prefix', 'cap', 'proposed', 'accepted', 'terminal', 'workset_tokens')


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def signature(row):
    return [[t.get(k) for k in TRACE_FIELDS] for t in row.get('trace', [])]


def row_errors(row, reference):
    issues = []
    if row.get('generated') != reference or row.get('tokens') != 32 or row.get('output_agrees') is not True:
        issues.append('Canonical tokens/output accounting mismatch')
    trace = row.get('trace', [])
    if not trace or row.get('blocks') != len(trace):
        issues.append('Missing/inconsistent round trace')
    if row.get('accepted') != sum(t.get('accepted', 0) for t in trace) or row.get('proposed') != sum(t.get('proposed', 0) for t in trace):
        issues.append('Accepted/proposed accounting mismatch')
    for i, t in enumerate(trace):
        cap, proposed, accepted = (t.get(k) for k in ('cap', 'proposed', 'accepted'))
        if not all(type(v) is int for v in (cap, proposed, accepted)) or not (1 <= proposed <= cap <= 4 and 0 <= accepted <= proposed):
            issues.append('Invalid bounded 2/4 trajectory')
        if t.get('round_index') != i or t.get('terminal') is not (i == len(trace)-1):
            issues.append('Invalid round/terminal sequence')
    for key in ('total_seconds', 'decode_seconds', 'total_tps', 'decode_tps'):
        if not positive(row.get(key)):
            issues.append('Invalid '+key)
    if positive(row.get('total_seconds')) and not math.isclose(row.get('total_tps', 0), 32/row['total_seconds'], rel_tol=1e-6):
        issues.append('Engine TPS accounting mismatch')
    return issues


def analyze(directory):
    protocol = read(directory/'protocol.json', {})
    reference = read(directory/'reference.json', [])
    quality, runtime, sessions = [rows(directory/name) for name in ('correctness.jsonl', 'runtime.jsonl', 'sessions.jsonl')]
    problems = []
    if protocol.get('source_index') != 303 or protocol.get('length') != 4096 or protocol.get('max_new_tokens') != 32 or protocol.get('repeats') != 2:
        problems.append('Unregistered document/context/output/repeat settings')
    if len(reference) != 32 or not all(type(v) is int and v >= 0 for v in reference):
        problems.append('Missing canonical32-token reference')
    canonical = rows(directory/'reference.jsonl')
    if len(canonical) != 1 or canonical[0].get('generated') != reference:
        problems.append('Independent AR request is missing/inconsistent')
    for role in ('baseline_eager', 'new'):
        inventory = protocol.get('source_sha256', {}).get(role, {})
        if not inventory:
            problems.append('Missing frozen '+role+' source inventory')
        source_root = (directory/'source'/role).resolve()
        for name, digest in inventory.items():
            path = (source_root/name).resolve()
            if not path.is_relative_to(source_root) or not path.is_file() or sha(path) != digest:
                problems.append('Frozen source mismatch: '+role+'/'+name)
    by_quality = {r.get('mode'): r for r in quality}
    if len(quality) != 3 or set(by_quality) != set(MODES):
        problems.append('Correctness matrix is incomplete/duplicated')
    base = by_quality.get('baseline_eager', {})
    for mode, row in by_quality.items():
        problems.extend(mode+': '+v for v in row_errors(row, reference))
        if signature(row) != signature(base):
            problems.append(mode+': correctness acceptance/depth trajectory differs')
        for key in ('observed_rounds', 'queries', 'native_t', 'native_s'):
            if not row.get(key) or row.get(key) != base.get(key):
                problems.append(mode+': exact '+key+' evidence differs/is missing')
        if len(row.get('queries', [])) != row.get('draft_forwards'):
            problems.append(mode+': Draft query count mismatch')
        if row.get('nonzero_injection') is not True or not any(
                isinstance(r.get('budget_features'), list) and len(r['budget_features']) == 12 and r['budget_features'][-1] > 0
                for r in row.get('observed_rounds', [])):
            problems.append(mode+': no actual nonzero-injection evidence')
        if mode == 'cuda_graph' and not all(row.get('runtime_stats', {}).get(k, 0) > 0 for k in ('captures', 'replays')):
            problems.append('Correctness did not exercise CUDA graphs')
    expected = {(mode, repeat, phase) for mode in MODES for repeat in range(2) for phase in ('cold', 'warm')}
    indexed, invalid = {}, set()
    for row in runtime:
        key = (row.get('mode'), row.get('repeat'), row.get('phase'))
        errors = row_errors(row, reference)
        if key not in expected or key in indexed:
            errors.append('Unexpected/duplicate timing key')
        if signature(row) != signature(base):
            errors.append('Timing trajectory differs from instrumented baseline')
        if not positive(row.get('outer_seconds')) or not positive(row.get('outer_tps')) or (
                positive(row.get('outer_seconds')) and not math.isclose(row.get('outer_tps', 0), 32/row['outer_seconds'], rel_tol=1e-6)):
            errors.append('Primary outer timing/accounting mismatch')
        if row.get('mode') == 'cuda_graph':
            stats = row.get('draft_runtime', {}).get('request', {})
            if stats.get('replays', 0) <= 0 or (stats.get('captures', 0) <= 0 if row.get('phase') == 'cold' else stats.get('captures') != 0):
                errors.append('Cold graph reuse/warm recapture or no graph replay')
        if errors:
            problems.extend(str(key)+': '+v for v in errors); invalid.add(key)
        indexed[key] = row
    session_map = {}
    for session in sessions:
        key = (session.get('mode'), session.get('repeat'))
        cold = indexed.get((*key, 'cold')); warm = indexed.get((*key, 'warm'))
        errors = []
        if key in session_map or key[0] not in MODES or key[1] not in (0, 1) or cold is None or warm is None:
            errors.append('Missing/duplicate session/cold/warm relation')
        if not positive(session.get('session_seconds')) or not positive(session.get('session_tps')) or session.get('tokens') != 64:
            errors.append('Invalid session wall-clock accounting')
        elif not math.isclose(session['session_tps'], 64/session['session_seconds'], rel_tol=1e-6):
            errors.append('Session throughput inconsistent with64 tokens')
        if cold and warm:
            if session.get('request_ids') != [cold.get('request_id'), warm.get('request_id')]:
                errors.append('Session UUID linkage mismatch')
            active = cold.get('outer_seconds', 0)+warm.get('outer_seconds', 0)+session.get('close_seconds', 0)
            if not math.isclose(session.get('active_sum_seconds', -1), active, rel_tol=1e-6) or session.get('session_seconds', -1)+1e-6 < active:
                errors.append('Session does not include constructor/generate/close')
        if errors:
            problems.extend(str(key)+': '+v for v in errors)
            invalid.update({(*key, 'cold'), (*key, 'warm')})
        session_map[key] = session
    missing = expected-set(indexed)
    if missing or len(runtime) != 12 or len(sessions) != 6:
        problems.append('Registered12 timing requests/6 sessions incomplete')
    uuids = [r.get('request_id') for r in canonical+quality+runtime]
    if len(set(uuids)) != len(uuids) or None in uuids:
        problems.append('Missing/reused request UUID')
    receipt = read(directory/'COMPLETE.json', {})
    if receipt.get('passed') is not True or receipt.get('counts') != dict(reference=1, correctness=3, timing=12, sessions=6) or not (
            receipt.get('sources_unchanged') and receipt.get('assets_unchanged')):
        problems.append('Missing/invalid completion and immutable-asset receipt')
    if (directory/'FAILURE.json').exists():
        problems.append('Runner failure marker exists')
    # A partial plot uses only whole three-mode cold/warm sessions from the
    # same repeat. Missing pairs are never replaced or pooled asymmetrically.
    usable = [rep for rep in range(2) if all((m, rep, p) in indexed and (m, rep, p) not in invalid
              for m in MODES for p in ('cold', 'warm')) and all((m, rep) in session_map for m in MODES)]
    metrics = {}
    for mode in MODES:
        selected = [indexed[mode, rep, phase] for rep in usable for phase in ('cold', 'warm')]
        if not selected:
            continue
        m = dict(requests=len(selected), timing_repeats=len(usable), documents=1,
                 accepted_per_round=sum(r['accepted'] for r in selected)/sum(r['blocks'] for r in selected),
                 acceptance_rate=sum(r['accepted'] for r in selected)/sum(r['proposed'] for r in selected),
                 cold={}, warm={})
        for phase in ('cold', 'warm'):
            group = [indexed[mode, rep, phase] for rep in usable]
            values = [r['outer_tps'] for r in group]
            m[phase] = dict(outer_tps=32*len(group)/sum(r['outer_seconds'] for r in group),
                repeat_tps=values, repeat_range=[min(values), max(values)],
                inner_tps=32*len(group)/sum(r['total_seconds'] for r in group),
                constructor_seconds=statistics.mean(r.get('constructor_seconds', 0.) for r in group),
                capture_host_seconds=statistics.mean(r.get('draft_runtime', {}).get('request', {}).get('capture_host_seconds', 0.) for r in group))
        group = [session_map[mode, rep] for rep in usable]; values = [s['session_tps'] for s in group]
        m['session'] = dict(tps=64*len(group)/sum(s['session_seconds'] for s in group), repeat_tps=values,
            repeat_range=[min(values), max(values)], close_seconds=statistics.mean(s['close_seconds'] for s in group))
        metrics[mode] = m
    return dict(status='COMPLETE' if not problems else 'PARTIAL', problems=problems,
        observed_counts=dict(reference=len(canonical), correctness=len(quality), timing=len(runtime), sessions=len(sessions)),
        usable_repeats=usable, metrics=metrics, documents=1, confidence_intervals=None,
        inference='H100 80GB, frozen nonzero V Mapper, causal 2/4 Draft lengths; one reused development document, 4K input, 32-token outputs, two timing repeats. No document CI, significance, generalization or SpecExtend conclusion.',
        raw_sha256={p.name: sha(p) for p in directory.glob('*.json*') if p.is_file()})


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    metrics = summary['metrics']; modes = [m for m in MODES if m in metrics]
    label = summary['status']+' · H100 80GB · frozen nonzero V Mapper · 2/4 Draft lengths\ndoc303 · 4096 input tokens · 32 output tokens · n=1 document'
    note = 'Two timing repeats; bars span timing ranges, NOT 95% CI. Only complete matched repeats shown.'
    figures = []

    def finish(fig, name):
        fig.suptitle(label, fontsize=10); fig.text(.5, .015, note, ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .06, 1, .94))
        for suffix in ('png', 'svg'):
            fig.savefig(output/f'{name}.{suffix}', dpi=170)
        plt.close(fig); figures.append(name)

    if not modes:
        for name in ('acceptance', 'cold_warm_tps', 'session_tps', 'latency'):
            fig, ax = plt.subplots(figsize=(8, 4)); ax.axis('off')
            ax.text(.5, .5, 'PARTIAL: no complete matched three-mode timing repeat', ha='center')
            finish(fig, name)
        return figures
    x = np.arange(len(modes))
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(x, [metrics[m]['accepted_per_round'] for m in modes])
    ax.set(xticks=x, xticklabels=modes, ylabel='Accepted Draft tokens / verification round', ylim=(0, None))
    finish(fig, 'acceptance')
    fig, ax = plt.subplots(figsize=(8, 4))
    for offset, phase in ((-.18, 'cold'), (.18, 'warm')):
        values = [metrics[m][phase]['outer_tps'] for m in modes]
        ax.bar(x+offset, values, width=.34, label=phase+' outer')
        for i, m in enumerate(modes):
            lo, hi = metrics[m][phase]['repeat_range']; ax.plot([i+offset]*2, [lo, hi], color='black', linewidth=1.5)
    ax.set(xticks=x, xticklabels=modes, ylabel='Outer throughput (tokens/s)'); ax.legend(title='Cold: constructor + generate\nWarm: reused engine', fontsize=9, title_fontsize=8)
    finish(fig, 'cold_warm_tps')
    fig, ax = plt.subplots(figsize=(8, 4)); values = [metrics[m]['session']['tps'] for m in modes]; ax.bar(x, values)
    for i, m in enumerate(modes):
        lo, hi = metrics[m]['session']['repeat_range']; ax.plot([i]*2, [lo, hi], color='black', linewidth=1.5)
    ax.set(xticks=x, xticklabels=modes, ylabel='64 tokens / complete cold+warm session seconds')
    finish(fig, 'session_tps')
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, (name, getter) in enumerate((('constructor', lambda m: m['cold']['constructor_seconds']),
            ('cold capture', lambda m: m['cold']['capture_host_seconds']),
            ('warm capture', lambda m: m['warm']['capture_host_seconds']), ('close', lambda m: m['session']['close_seconds']))):
        ax.bar(x+(j-1.5)*.18, [getter(metrics[m]) for m in modes], width=.17, label=name)
    ax.set(xticks=x, xticklabels=modes, ylabel='Seconds (non-additive diagnostic intervals)'); ax.legend(fontsize=8)
    finish(fig, 'latency')
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    if args.output.exists():
        parser.error('Fresh analysis output required; do not overwrite partial evidence')
    args.output.mkdir(parents=True)
    summary = analyze(args.directory); figures = plot(summary, args.output)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    shutil.copyfile(__file__, args.output/'analysis_source.py')
    report = ['# '+summary['status']+': one-document resident/CUDA-graph pilot', '', summary['inference'], '',
        'Fixed precision: Target FP32; Draft and prefix prefill BF16; TF32 disabled. Same frozen nonzero Mapper and causal 2/4 policy in all modes.',
        'Cold primary is outer constructor+generate; warm primary is outer generate. Engine totalTPS is secondary.',
        'Session wall runs constructor→cold→raw flush/safety checks→warm→close; it includes capture, teardown and intervening audit I/O, with no hidden subtraction.',
        'All weights,2/4 decisions,each-token EOS,selection and nonzeroV injection are fixed. No retraining or search.', '',
        '| Mode | Cold outerTPS | Warm outerTPS | Complete-sessionTPS | Accepted Draft tokens/round |', '|---|---:|---:|---:|---:|']
    for mode, m in summary['metrics'].items():
        report.append(f"| {mode} | {m['cold']['outer_tps']:.4f} | {m['warm']['outer_tps']:.4f} | {m['session']['tps']:.4f} | {m['accepted_per_round']:.4f} |")
    report += ['', 'n=1 document; two timing repeats, not two independent samples. No 95% document CI or significance estimate. Timing-range bars are not confidence intervals.',
        'Cold correctness graphs are not reused by timing engines. Warm replay does not by itself establish cold or amortized end-to-end acceleration.',
        'This16-request pilot is not the formal six-document4K/8K experiment and does not establish a generalization or SpecExtend win.', '', '## Audit issues', '']
    report += ['- '+v for v in summary['problems']] or ['None; registered pilot evidence complete.']
    report += ['', '## Figures and raw evidence', '']
    for name in figures:
        report.append(f'- [{name}]({name}.png) ([SVG source]({name}.svg))')
    report += ['- [Analysis code](analysis_source.py)', '- [Summary](summary.json)']
    for name in ('protocol.json', 'reference.jsonl', 'reference.json', 'correctness.jsonl', 'runtime.jsonl', 'sessions.jsonl', 'COMPLETE.json', 'FAILURE.json'):
        if (args.directory/name).exists():
            report.append(f'- [{name}](../{name})')
    (args.output/'REPORT.md').write_text('\n'.join(report)+'\n')
    manifest = {p.name: sha(p) for p in args.output.iterdir() if p.is_file()}
    (args.output/'ANALYSIS_COMPLETE.json').write_text(json.dumps(dict(status=summary['status'], artifacts_sha256=manifest,
        inspected_render_required=True, scope='Artifact generation/audit, not GPU execution or scientific success'), indent=2)+'\n')
    print(json.dumps(dict(status=summary['status'], metrics=summary['metrics'], problems=summary['problems']), indent=2))


if __name__ == '__main__':
    main()
