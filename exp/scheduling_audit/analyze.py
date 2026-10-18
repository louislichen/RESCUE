"""Recompute scheduling diagnostics from saved requests; no model/GPU imports.

All time subtraction is fixed-trace arithmetic, not an optimized runtime result.
The original experiment files and frozen CSGB archive are read only.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile


ROOT = Path(__file__).resolve().parents[2]
CSGB = ROOT / 'exp/specextend_bridgekv/csgb_acceptance_pilot_v3'


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def runtime_metrics(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['method']].append(row)
    result = {}
    for method, records in groups.items():
        tokens = sum(r['tokens'] for r in records)
        seconds = math.fsum(r['decode_seconds'] for r in records)
        stages = sorted({k for r in records for k in r.get('stage_seconds', {})})
        stage_seconds = {k: math.fsum(r.get('stage_seconds', {}).get(k, 0.) for r in records)
                         for k in stages}
        blocks = sum(r.get('blocks', 0) for r in records)
        proposed = sum(r.get('proposed', 0) for r in records)
        accepted = sum(r.get('accepted', 0) for r in records)
        # Only the horizon trace records terminal status. Do not infer it for
        # the older V pilot, whose trace uses a different schema.
        traces = [b for r in records for b in r.get('trace', [])]
        catchup = (sum(not b['terminal'] and b['accepted'] == b['proposed'] for b in traces)
                   if traces and all('terminal' in b for b in traces) else None)
        result[method] = dict(
            requests=len(records), tokens=tokens, blocks=blocks, proposed=proposed,
            accepted=accepted, decode_seconds=seconds, pooled_decode_tps=tokens/seconds,
            ms_per_output=seconds*1000/tokens,
            stage_ms_per_output={k: v*1000/tokens for k, v in stage_seconds.items()},
            unallocated_ms_per_output=((seconds-math.fsum(stage_seconds.values()))*1000/tokens
                                      if stages else None),
            accepted_per_round=accepted/blocks if blocks else None,
            draft_catchup_forwards=catchup,
            student_forwards=proposed+catchup if catchup is not None else None,
            saved_exact_outputs=sum(r.get('output_agrees') is True for r in records))
    return result


def csgb_metrics():
    groups = defaultdict(list)
    seen = set()
    with tarfile.open(CSGB / 'raw_results.tar.gz', 'r:gz') as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith('.json') or 'monitor_' not in member.name:
                continue
            with archive.extractfile(member) as handle:
                record = json.load(handle)
            if record.get('phase') != 'monitor':
                continue
            key = tuple(record[k] for k in ('document', 'policy', 'stream', 'timing_repeat'))
            if key in seen:
                raise ValueError(f'Duplicate monitor request: {key}')
            seen.add(key)
            groups[record['policy']].append(record)
    summary = json.loads((CSGB / 'analysis/summary.json').read_text())
    if len(seen) != summary['expected_requests'] or len(groups) != 7:
        raise ValueError('Incomplete registered monitor matrix')
    output, doc_tps = {}, {}
    for mode, records in sorted(groups.items()):
        episodes = [r['episode'] for r in records]
        rounds = [v for e in episodes for v in e['rows']]
        seconds = math.fsum(e['seconds'] for e in episodes)
        decision = math.fsum(v['decision_wall_seconds'] for v in rounds)
        useful = sum(e['useful_n'] for e in episodes)
        by_doc_stream = defaultdict(list)
        for record in records:
            e = record['episode']
            recorded_decision = math.fsum(v['decision_wall_seconds'] for v in e['rows'])
            by_doc_stream[record['document'], record['stream']].append(
                (e['useful_n']/e['seconds'], e['useful_n']/(e['seconds']-recorded_decision)))
        documents = sorted({d for d, _ in by_doc_stream})
        if documents != list(range(16, 24)) or any(len(v) != 2 for v in by_doc_stream.values()):
            raise ValueError('Wrong paired document/stream/repeat units')
        doc_tps[mode] = {d: tuple(statistics.mean(
            statistics.median(v[j] for v in by_doc_stream[d, s]) for s in range(3))
            for j in range(2)) for d in documents}
        probabilities = [d['probabilities'] for e in episodes for d in e['decisions']
                         if d['raw_x'] is not None]
        output[mode] = dict(
            requests=len(records), rounds=len(rounds), useful_tokens=useful,
            whole_request_seconds=seconds, summed_decision_wall_seconds=decision,
            decision_fraction_percent=100*decision/seconds,
            whole_request_ms_divided_by_rounds=1000*seconds/len(rounds),
            # This includes amortized prefill; it is NOT an isolated round timer.
            useful_tokens_per_round=useful/len(rounds),
            action_counts=dict(sorted(Counter(v['budget'] for v in rounds).items())),
            mean_nonstartup_probabilities=([statistics.mean(p[j] for p in probabilities)
                                            for j in range(4)] if probabilities else None),
            document_equal_request_tps=statistics.mean(doc_tps[mode][d][0] for d in documents))
        if not math.isclose(output[mode]['document_equal_request_tps'],
                            summary['absolute_metrics'][mode]['tps']['mean'], abs_tol=1e-10):
            raise ValueError('Aggregation no longer matches the archived report')
    base = doc_tps['fixed8_native']
    actual = statistics.mean(100*(doc_tps['csgb'][d][0]/base[d][0]-1) for d in base)
    subtracted = statistics.mean(100*(doc_tps['csgb'][d][1]/base[d][0]-1) for d in base)
    expected = summary['comparisons_all_fixed']['fixed8_native']['csgb']['tps']['mean']
    if not math.isclose(actual, expected, abs_tol=1e-10):
        raise ValueError('Primary paired TPS differs from the saved report')
    return dict(modes=output, paired_tps_change_percent=actual,
                fixed_trace_subtract_recorded_decision_percent=subtracted,
                subtraction_scope='Arithmetic only. Same outputs/actions/other times; no new CI or GPU run. '
                'Decision wall time includes device waits and excludes extra Target attention collection.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('audit.json'))
    args = parser.parse_args()
    horizon_path = ROOT / 'exp/attention_horizon/out/runtime_test.jsonl'
    pilot_path = ROOT / 'exp/selective_kv/out_pilot/runtime_test.jsonl'
    completion_path = ROOT / 'exp/attention_horizon/out/completeness.json'
    completion = json.loads(completion_path.read_text())
    source_matches = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
                      for name, digest in completion['current_source_sha256'].items()}
    if not all(source_matches.values()):
        raise ValueError('Current horizon source differs from final completeness inventory')
    horizon = runtime_metrics(read_jsonl(horizon_path))
    pilot = runtime_metrics(read_jsonl(pilot_path))
    sequential, joint = horizon['C_sequential'], horizon['C_joint']
    planning = lambda r: sum(r['stage_ms_per_output'][k] for k in ('selection', 'decision'))
    planning_difference = planning(joint)-planning(sequential)
    result = dict(
        schema=1, date='2026-10-17',
        scope='Saved-record CPU accounting and source inspection support. No model-quality/GPU rerun.',
        input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (horizon_path, pilot_path, completion_path,
                                CSGB/'raw_results.tar.gz', CSGB/'analysis/summary.json')},
        horizon_source_matches_final_inventory=source_matches,
        horizon=horizon, selective_v_pilot=pilot, csgb=csgb_metrics(),
        horizon_planning_arithmetic=dict(
            sequential_ms_per_output=planning(sequential), joint_ms_per_output=planning(joint),
            extra_planning_ms_per_output=planning_difference,
            total_time_difference_ms_per_output=joint['ms_per_output']-sequential['ms_per_output'],
            fixed_trace_time_difference_after_subtraction_ms_per_output=(
                joint['ms_per_output']-planning_difference-sequential['ms_per_output']),
            scope='Arithmetic only; extra planning removed on saved traces, not a runtime result.'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),
                         horizon_planning=result['horizon_planning_arithmetic'],
                         csgb_paired_tps_percent=result['csgb']['paired_tps_change_percent'],
                         csgb_fixed_trace_subtraction_percent=result['csgb']['fixed_trace_subtract_recorded_decision_percent']),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
