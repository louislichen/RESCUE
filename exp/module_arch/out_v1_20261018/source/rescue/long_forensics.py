"""CPU-only accounting of the frozen long-context comparison.

Phase differences are descriptive; they do not isolate individual modules.
This analysis neither selects a new configuration nor reruns GPU inference.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def analyze(directory):
    raw = (directory / 'runtime.jsonl').read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    protocol = json.loads((directory / 'protocol.json').read_text())
    matrix = json.loads((directory / 'test_matrix.json').read_text())
    choices = json.loads((directory / 'selection.json').read_text())['chosen']
    assert len(rows) == (len(matrix['source_indices']) * len(matrix['lengths'])
                         * matrix['repeats'] * len(matrix['modes']))
    assert all(row['output_agrees'] for row in rows)
    result = {}
    for length in matrix['lengths']:
        modes = {}
        for mode, candidate in choices.items():
            group = [r for r in rows if r['length'] == length and r['mode'] == mode]
            assert len(group) == len(matrix['source_indices']) * matrix['repeats']
            entry = dict(candidate=candidate, requests=len(group),
                output_tokens=sum(r['tokens'] for r in group),
                mean_seconds=statistics.mean(r['total_seconds'] for r in group),
                total_tps=sum(r['tokens'] for r in group) / sum(r['total_seconds'] for r in group))
            if 'trace' in group[0]:
                trace = [t for r in group for t in r['trace']]
                blocks = sum(r['blocks'] for r in group)
                assert len(trace) == blocks
                for field in ('prefill_seconds', 'decode_seconds', 'cleanup_seconds',
                              'draft_forwards', 'selector_refreshes', 'draft_shadow_probes',
                              'projected_positions', 'selection_similarity_elements',
                              'injection_condition_steps', 'working_gathered_position_layers'):
                    if field in group[0]:
                        entry['mean_' + field] = statistics.mean(r[field] for r in group)
                proposed = sum(r['proposed'] for r in group)
                accepted = sum(r['accepted'] for r in group)
                entry.update(mean_rounds=blocks / len(group),
                    accepted=accepted, proposed=proposed,
                    accepted_per_round=accepted / blocks,
                    proposed_per_round=proposed / blocks,
                    acceptance_rate=accepted / proposed,
                    mean_decode_seconds_per_round=sum(r['decode_seconds'] for r in group) / blocks,
                    proposed_histogram=dict(sorted(Counter(t['proposed'] for t in trace).items())),
                    cap_histogram=dict(sorted(Counter(t['cap'] for t in trace).items())))
            else:
                entry.update(mean_upstream_rounds=statistics.mean(
                    len(r['upstream']['accept_length_list']) for r in group),
                    total_generation_overshoot=sum(r['upstream']['generation_overshoot'] for r in group))
            modes[mode] = entry
        differences = {}
        for first, second in (('rescue_positive', 'retrieval'), ('rescue_dynamic', 'retrieval'),
                              ('rescue_guarded', 'rescue_positive')):
            a, b = modes[first], modes[second]
            differences[first + '_minus_' + second] = {
                field: a[field] - b[field] for field in (
                    'mean_seconds', 'mean_prefill_seconds', 'mean_decode_seconds',
                    'mean_cleanup_seconds', 'mean_rounds', 'mean_draft_forwards',
                    'mean_selector_refreshes')}
            delta = differences[first + '_minus_' + second]
            assert abs(delta['mean_seconds'] - sum(delta[f] for f in (
                'mean_prefill_seconds', 'mean_decode_seconds', 'mean_cleanup_seconds'))) < 1e-8
            # Symmetric product decomposition of D=rounds * average round cost.
            # These are accounting terms, not timings for individual modules.
            ra, rb = a['mean_rounds'], b['mean_rounds']
            ca, cb = a['mean_decode_seconds_per_round'], b['mean_decode_seconds_per_round']
            delta['decode_round_count_term_seconds'] = (ra - rb) * (ca + cb) / 2
            delta['decode_mean_round_cost_term_seconds'] = (ca - cb) * (ra + rb) / 2
            assert abs(delta['mean_decode_seconds'] - delta['decode_round_count_term_seconds']
                       - delta['decode_mean_round_cost_term_seconds']) < 1e-8
        result[str(length)] = dict(modes=modes, descriptive_differences=differences)
    configs = protocol['configs']
    return dict(scope='Descriptive accounting only; not new GPU evidence or isolated causal attribution',
        runtime_sha256=hashlib.sha256(raw).hexdigest(),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        candidate_configs={m:configs[c] for m,c in choices.items() if c in configs},
        warning='Dynamic and development-selected fixed configurations can differ in budget/refresh as well as length. Tree counters and overshoot follow the upstream algorithm.',
        lengths=result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    args = p.parse_args()
    result = analyze(args.directory)
    (args.directory / 'forensics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: {m: round(v['total_tps'], 3) for m,v in r['modes'].items()}
                      for k,r in result['lengths'].items()}, indent=2))


if __name__ == '__main__':
    main()
