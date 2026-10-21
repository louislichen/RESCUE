"""Independent saved-raw timing audit: stdlib + lazy NumPy, never GPU/runtime.

Prepare only until the parent confirms collection READY. No runner/reporter or
shared validation module is imported. Paired intervals resample whole documents
with both repeats/all four arms retained, using one shared seed/index matrix.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath


LENGTHS = (2048, 4096, 6144, 8192, 12288, 16384)
DOCS = (211, 213, 220, 226, 233, 236, 246, 248, 253, 271, 272, 288, 303, 306, 327, 604)
ARMS = ('original_rescue', 'candidate_rescue', 'original_retrieval', 'candidate_retrieval')
ORDER = ((0, 1, 2, 3), (3, 2, 1, 0), (1, 3, 0, 2), (2, 0, 3, 1), (0, 2, 1, 3), (3, 1, 2, 0))
CONTRASTS = (('candidate_rescue', 'candidate_retrieval'), ('candidate_rescue', 'original_rescue'),
             ('candidate_retrieval', 'original_retrieval'), ('original_rescue', 'original_retrieval'))
WORK = ('tokens', 'accepted', 'proposed', 'blocks', 'draft_forwards', 'selector_refreshes',
        'working_gather_layers', 'working_full_gather_layers', 'working_gathered_position_layers',
        'working_incremental_binds', 'draft_shadow_probes', 'projected_positions', 'rpc_calls')
OPTIONAL = ('target_probe_rounds', 'selection_similarity_elements', 'injection_condition_steps',
            'static_affine_positions', 'working_restored_value_positions', 'rpc_bytes_sent')
TRACE = ('round_index', 'prefix', 'cap', 'proposed', 'accepted', 'terminal', 'workset_tokens')
GRAPH = ('captures', 'replays', 'resident_steps', 'shadow_updates', 'capture_host_seconds')
PREREG_SHA = '2601405a52627fc3b2df4d8947a448381fbec1e042a680f111c82d77893972d0'
SUMS_SHA = '6f0bdc74aeabacd53abb08ab735f7b87b3daa20e9987b8e78512b9601eed0960'
CURVE_MANIFEST = '70e46d52c3f3f57b1efa8d1633b665f345d54d7656529c9b09e2d25e185bf41e'
CURVE_VERIFIED = 'c5bf8390ec5349c6d7ce29a688d3b9b13d95b6fd11f006e46a7522b9be42dffc'
MODEL_MANIFEST = '013273e977a8b4aeb97b3ceff845bc9b03ffbe0ab7f4752307c65c5b598f7320'
MODEL_VERIFIED = '44174149ed3b8b36984fb0ec51581933c61c863884d112ff9e1297170e83c328'


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1 << 20), b''):
            result.update(part)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def rows(path):
    path = Path(path)
    if not path.is_file() or not path.stat().st_size:
        return []
    data = path.read_bytes()
    if not data.endswith(b'\n'):
        raise ValueError('Incomplete JSONL: '+str(path))
    return [json.loads(line) for line in data.splitlines()]


def same_work(left, right):
    return (left['generated'] == right['generated'] and
        [{k: r[k] for k in TRACE} for r in left['trace']] == [{k: r[k] for k in TRACE} for r in right['trace']] and
        all(k in left and k in right and left[k] == right[k] for k in WORK) and
        all((k in left) == (k in right) and left.get(k) == right.get(k) for k in OPTIONAL))


def difference(left, right):
    fields = {k: dict(original=left.get(k), candidate=right.get(k)) for k in (*WORK, *OPTIONAL)
              if (k in left) != (k in right) or left.get(k) != right.get(k)}
    return dict(work_fields=fields,
        trajectory_equal=[{k: r[k] for k in TRACE} for r in left['trace']] == [{k: r[k] for k in TRACE} for r in right['trace']])


def statistics(warm):
    """Independent registered pooled TPS + paired whole-document intervals."""
    import numpy as np
    indices = np.random.default_rng(20261031).integers(0, 16, size=(10000, 16), dtype=np.int64)
    index_sha = hashlib.sha256(indices.tobytes()).hexdigest()
    result, all_primary = [], True
    for n in LENGTHS:
        points, bootstrap, per_document = {}, {}, {}
        for arm in ARMS:
            vectors = []
            for document in DOCS:
                selected = [r for r in warm if (r['length'], r['arm_id'], r['source_index']) == (n, arm, document)]
                if len(selected) != 2 or {r['repeat'] for r in selected} != {0, 1}:
                    raise ValueError('Statistics require all paired2 repeats for each16 documents')
                vectors.append([sum(r[k] for r in selected) for k in ('tokens', 'outer_seconds', 'accepted', 'proposed', 'blocks')])
            array = np.array(vectors, dtype=np.float64)
            totals, resampled = array.sum(axis=0), array[indices].sum(axis=1)
            if not bool(np.isfinite(array).all()) or not bool((array[:, 1] > 0).all()):
                raise ValueError('Finite positive actual warm clock required')
            points[arm] = dict(tokens=float(totals[0]), outer_seconds=float(totals[1]), tps=float(totals[0]/totals[1]),
                accepted_per_round=float(totals[2]/totals[4]), acceptance_rate=float(totals[2]/totals[3]),
                accepted=float(totals[2]), proposed=float(totals[3]), blocks=float(totals[4]))
            bootstrap[arm] = resampled[:, 0]/resampled[:, 1]
            per_document[arm] = [dict(source_index=d, tokens=float(a[0]), outer_seconds=float(a[1]),
                                     accepted=float(a[2]), proposed=float(a[3]), blocks=float(a[4])) for d, a in zip(DOCS, array)]
        contrasts = []
        for numerator, denominator in CONTRASTS:
            value = 100*(points[numerator]['tps']/points[denominator]['tps']-1)
            samples = 100*(bootstrap[numerator]/bootstrap[denominator]-1)
            lower, upper = np.quantile(samples, [.025, .975])
            contrasts.append(dict(numerator=numerator, denominator=denominator, relative_tps_percent=float(value),
                paired95_ci_percent=[float(lower), float(upper)]))
        primary = contrasts[0]
        passed = primary['relative_tps_percent'] > 10 and primary['paired95_ci_percent'][0] > 0
        all_primary = all_primary and passed
        interaction = 100*((points['candidate_rescue']['tps']/points['original_rescue']['tps'])/
                           (points['candidate_retrieval']['tps']/points['original_retrieval']['tps'])-1)
        result.append(dict(length=n, pooled=points, per_document=per_document, contrasts=contrasts,
            primary_numeric_gate_passed=bool(passed), multiplicative_interaction_percent=float(interaction),
            primary_minus_original_relative_gain_pp=contrasts[0]['relative_tps_percent']-contrasts[3]['relative_tps_percent']))
    return dict(bootstrap_seed=20261031, bootstrap_draws=10000, paired_document_count=16, repeats=2,
        document_order=list(DOCS), shared_bootstrap_index_sha256=index_sha, numpy_version=np.__version__,
        pointwise_not_simultaneous=True, all_six_primary_numeric_gates=bool(all_primary), lengths=result)


def audit(root, curve, model):
    checks, pins = [], {}
    def check(value, label):
        checks.append(dict(check=label, passed=bool(value)))
    def pin(path, expected, label):
        digest = sha(path); pins[str(path.resolve())] = digest
        check(digest == expected, label)
    pin(root/'PREREGISTRATION.json', PREREG_SHA, 'registered792 protocol exact bytes')
    pin(root/'SOURCE_SHA256SUMS', SUMS_SHA, 'registered15 source inventory exact bytes')
    pre = read(root/'PREREGISTRATION.json')
    lines = (root/'SOURCE_SHA256SUMS').read_text().splitlines()
    check(len(lines) == 15, 'all15 source/launch artifacts')
    for line in lines:
        digest, name = line.split(None, 1)
        check(name == '../launch.sh' or not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts, 'safe capsule source path')
        pin(root/'launch.sh' if name == '../launch.sh' else root/'capsule'/name, digest, 'new source byte pin:'+name)
    pin(curve/'FINAL_ARTIFACTS_MANIFEST.json', CURVE_MANIFEST, 'old798 complete manifest unchanged')
    pin(curve/'VERIFIED_COMPLETE.json', CURVE_VERIFIED, 'old798 verified completion unchanged')
    pin(model/'FINAL_ARTIFACTS_MANIFEST.json', MODEL_MANIFEST, 'completed48 Token-gate manifest unchanged')
    pin(model/'VERIFIED_COMPLETE.json', MODEL_VERIFIED, 'completed48 Token-gate seal unchanged')
    old_manifest = read(curve/'FINAL_ARTIFACTS_MANIFEST.json')['file_sha256']
    old_root = curve/'terminal/results'
    for name in ('protocol.json', 'references.json', 'documents.jsonl', 'runtime.jsonl', 'REUSE_RECEIPT.json'):
        pin(old_root/name, old_manifest['terminal/results/'+name], 'sealed original canonical/input/work:'+name)
    old, old_ref = read(old_root/'protocol.json'), read(old_root/'references.json')
    old_documents = {r['source_index']: r for r in rows(old_root/'documents.jsonl')}
    historical = {(r['length'], r['source_index'], r['mode'], r['repeat']): r for r in rows(old_root/'runtime.jsonl')}
    pin(model/'results/TERMINAL.json', pre['model_terminal_sha256'], 'actual prior48 Token PASS')
    pin(model/'TERMINAL_EXTERNAL_BARRIER.json', pre['terminal_barrier_sha256'], 'registered prior actual external terminal barrier')
    mt = read(model/'results/TERMINAL.json')
    check(mt['passed'] is True and mt['actual_cuda'] is True and mt['completed_calls'] == mt['private_owners_closed'] == 48,
          'prior model result is actual48, not analytic operator promotion')
    result = root/'results'
    terminal, pipeline = read(result/'TERMINAL.json'), read(root/'PIPELINE_EXIT.json')
    pin(result/'TERMINAL.json', pipeline['terminal_sha256'], 'actual timing pipeline binds terminal raw bytes')
    check(pipeline['preserve_partial'] is True and pipeline['automatic_retry'] is False and pipeline['goal_complete'] is False,
          'preserved terminal data no retry/goal promotion')
    check(terminal['runner_sha256'] == pre['runner_sha256'], 'actual terminal frozen timing runner')
    reuse = read(result/'REUSE_RECEIPT.json')
    check(reuse['old_artifact_pins'] == read(old_root/'REUSE_RECEIPT.json')['old_artifact_pins'],
          'full older source/reference reuse chain unchanged')
    check(reuse['old_source_sha256'] == old['source_sha256'] and len(reuse['old_source_sha256']) == 167,
          'all original167 source fingerprints reused')
    for name, digest in reuse['old_source_sha256'].items():
        pin(old_root/'source'/name, digest, 'local saved original runtime bytes:'+name)
    check(reuse['assets_sha256'] == old['assets_sha256'] and reuse['no_new_ar'] is True and reuse['no_new_corpus'] is True,
          'saved full asset fingerprints same/no corpus or new AR; weights not locally rehashed')
    environment = read(result/'ENVIRONMENT.json')
    check(environment['environment']['gpu_uuid'].removeprefix('GPU-') == pre['gpu_uuid'].removeprefix('GPU-')
          and environment['environment']['torch'] == pre['torch']
          and environment['environment']['cuda'] == '12.8' and environment['environment']['tf32'] is False
          and environment['environment']['target_dtype'] == 'torch.float32'
          and environment['environment']['draft_dtype'] == environment['environment']['prefix_dtype'] == 'torch.bfloat16',
          'actual fixed H100/CUDA/Torch/TargetFP32 DraftBF16 environment')
    check(environment['source_bindings'] == old['source_bindings'] and environment['tree_source_bindings'] == old['tree_source_bindings']
          and environment['flash_attention'] == old['flash_attention'] and environment['observers'] is False
          and environment['profiler'] is False, 'authentic original loaded runtime/dependency bindings without instrumentation')
    sequence = old['shared_warm_sequence']
    check(sequence == reuse['shared_warm_sequence'] and len(sequence) == 32
          and {(r['source_index'], r['repeat']) for r in sequence} == {(d, r) for d in DOCS for r in (0, 1)}, 'same complete original32 warm sequence')
    refs, inputs = read(result/'references.json'), read(result/'INPUTS.json')
    for n in LENGTHS:
        for d in DOCS:
            key, doc = f'{d}:{n}', old_documents[d]
            check(refs[key] == old_ref[key], key+' exact original canonical, no candidate self-reference')
            reconstructed = doc['prefix_ids']+doc['document_ids'][:n-len(doc['prefix_ids'])-len(doc['suffix_ids'])]+doc['suffix_ids']
            check(inputs[key] == reconstructed and len(inputs[key]) == n
                  and hashlib.sha256(json.dumps(inputs[key]).encode()).hexdigest() == old['input_sha256'][key], key+' exact saved input SHA')
    prime, warm, owners = [rows(result/name) for name in ('PRIME.jsonl', 'RUNTIME.jsonl', 'OWNERS.jsonl')]
    actual = sorted(prime+warm, key=lambda r: r['execution_index'])
    expected = []
    for n, order in zip(LENGTHS, ORDER):
        for index in order:
            expected.extend((n, ARMS[index], r['source_index'], r.get('repeat'), r['phase']) for r in
                [dict(source_index=271, phase='cold_prime')]+[dict(r, phase='warm') for r in sequence])
    check([(r['length'], r['arm_id'], r['source_index'], r.get('repeat'), r['phase']) for r in actual] == expected[:len(actual)]
          and [r['execution_index'] for r in actual] == list(range(len(actual))), 'actual exact ordered792 prefix, no duplicate/missing calls')
    check(len(actual) == terminal['returned_calls'] == terminal['counts']['prime_requests']+terminal['counts']['timing_requests']
          and len(prime) == terminal['counts']['prime_requests'] and len(warm) == terminal['counts']['timing_requests'], 'all actual returned/counter records bind')
    check(len({r['request_id'] for r in actual}) == len(actual), 'all792 request UUIDs distinct')
    signatures, ledgers, repeated, by_owner, all_canonical = {}, {}, {}, {}, True
    late_captures, late_seconds, timing_cells = Counter(), Counter(), []
    for row in actual:
        label, owner = str(row['execution_index']), row['owner_id']
        n, arm, d = row['length'], row['arm_id'], row['source_index']
        mode = 'rescue_graph' if arm.endswith('rescue') else 'retrieval32_k4_graph'
        canonical = row['generated'] == refs[f'{d}:{n}']
        all_canonical = all_canonical and canonical
        check(canonical and row['tokens'] == len(row['generated']), label+' exact frozen original Target Tokens')
        check(row['observer_enabled'] is False and row['profile_enabled'] is False
              and row['performance_eligible'] is (row['phase'] == 'warm')
              and not any(k in row for k in ('observed_rounds', 'queries', 'native_t', 'native_s', 'timeline', 'cost_samples')),
              label+' real observer-free timing, never mixed quality')
        elapsed = row['outer_seconds']
        check(math.isfinite(elapsed) and elapsed > 0 and math.isclose(row['outer_tps'], row['tokens']/elapsed, rel_tol=1e-12), label+' actual whole-generate clock ratio')
        check(row['all_request_jit_and_captures_charged'] is True and row['scope_metadata_append_kept'] is True,
              label+' real capture/JIT/metadata path not removed')
        signatures.setdefault(owner, row['owner_signature'])
        check(row['owner_signature'] == signatures[owner], label+' private owner/cache addresses stable')
        by_owner.setdefault(owner, []).append(row)
        stats, delta = row['runtime_stats'], row['draft_runtime']['request']
        keys = {json.dumps(k, sort_keys=True) for k in stats['graph_keys']}
        previous = ledgers.get(owner, dict(keys=set(), totals=dict.fromkeys(GRAPH, 0)))
        check(stats['mode'] == 'cuda_graph' and stats['max_graph_entries'] == 128
              and stats['entries'] == stats['captures'] == len(keys) <= 128 and previous['keys'] <= keys
              and len(keys-previous['keys']) == delta['captures'], label+' actual cumulative Graph key/new-capture ledger')
        check(delta['replays'] == delta['shadow_updates'] == row['draft_forwards'] > 0 and delta['resident_steps'] == 0,
              label+' actual every Draft forward Graph replay/shadow update')
        totals = {k: previous['totals'][k]+delta[k] for k in GRAPH}
        check(all(math.isclose(totals[k], stats[k], rel_tol=1e-10, abs_tol=1e-8)
                  and math.isclose(totals[k], row['draft_runtime']['lifetime'][k], rel_tol=1e-10, abs_tol=1e-8) for k in GRAPH),
              label+' Graph request deltas sum to lifetime, capture seconds never subtracted')
        ledgers[owner] = dict(keys=keys, totals=totals)
        if row['phase'] == 'cold_prime':
            check(delta['captures'] > 0 and len(by_owner[owner]) == 1, label+' fresh prime has actual Graph captures')
        else:
            late_captures[n, arm] += delta['captures']; late_seconds[n, arm] += delta['capture_host_seconds']
        if arm.startswith('original'):
            check(same_work(historical[n, d, mode, row.get('repeat', 0)], row), label+' original work exactly historical matched repeat')
        repeat_key = n, arm, d
        if repeat_key in repeated:
            check(same_work(repeated[repeat_key], row), label+' candidate/original own repeated logical work stable')
        repeated[repeat_key] = row
    check(len({r['owner_id'] for r in owners}) == len(owners), 'distinct closed persistent owners')
    for owner in owners:
        label, identity = owner['arm_id'], owner['owner_id']
        data = by_owner.get(identity, [])
        closed, summary = owner['close_receipt'], owner['scope_summary']
        check(closed.get('engine_closed') is True and closed.get('hook_restored') is True
              and closed['closed_runtime_stats']['entries'] == 0 and not owner['cleanup_errors'] and owner['primary_error'] is None,
              label+' actual successful private owner/lease/Graph release')
        check(len(data) == 33 and data[0]['phase'] == 'cold_prime'
              and [(r['source_index'], r['repeat']) for r in data[1:]] == [(r['source_index'], r['repeat']) for r in sequence], label+' exactlycold+32 shared warm calls')
        if len(data) != 33:
            continue
        selected = data[1:]
        tokens, seconds = sum(r['tokens'] for r in selected), sum(r['outer_seconds'] for r in selected)
        check(owner['warm_tokens'] == tokens and math.isclose(owner['warm_outer_seconds_sum'], seconds, rel_tol=1e-12)
              and math.isclose(owner['warm_tps'], tokens/seconds, rel_tol=1e-12)
              and owner['warm_request_ids'] == [r['request_id'] for r in selected], label+' owner primary TPS independently pooled no cold mixing')
        service = owner['constructor_scope_seconds']+data[0]['outer_seconds']+seconds+closed['scope_and_close_seconds']
        service_tokens = tokens+data[0]['tokens']
        check(math.isclose(owner['accounted_service_seconds'], service, rel_tol=1e-12)
              and owner['accounted_service_tokens'] == service_tokens
              and math.isclose(owner['accounted_service_tps'], service_tokens/service, rel_tol=1e-12), label+' finite accounted-service supplement includes exact four components')
        check(summary['state'] == 'closed' and summary['enabled'] is label.startswith('candidate'), label+' whole-owner Target scope/lease restore guards closed')
        records = summary['records']
        if label.startswith('original'):
            check(summary['calls'] == summary['candidate_calls'] == summary['original_calls'] == 0 and not records, label+' original scope identity/no wrapping')
        else:
            check(len(records) == summary['calls'] and len(records) % 28 == 0 and summary['candidate_calls'] > 0, label+' actual candidate whole-owner metadata')
            for i in range(0, len(records), 28):
                block = records[i:i+28]
                check([r['layer'] for r in block] == list(range(28))
                      and len({(r['action'], r['query_tokens'], r['key_tokens']) for r in block}) == 1
                      and all(r['outcome'] == 'returned' for r in block), label+f' fullTarget28 ordered callblock{i//28}')
            per_layer = [sum(r['layer'] == layer and r['action'] == 'candidate' for r in records) for layer in range(28)]
            check(min(per_layer) > 0 and len(set(per_layer)) == 1 and sum(per_layer) == summary['candidate_calls'], label+' nonzero equal candidate layer coverage')
        timing_cells.append(dict(length=owner['length'], arm_id=label, primary_tokens=tokens, primary_seconds=seconds,
            primary_tps=tokens/seconds, accounted_service_tps=service_tokens/service,
            warm_captures=late_captures[owner['length'], label], warm_capture_host_seconds=late_seconds[owner['length'], label],
            candidate_target_layer_calls=summary['candidate_calls'], original_target_layer_calls=summary['original_calls']))
    differences = rows(result/'WORK_DIFFERENCES.jsonl')
    warm_map = {(r['length'], r['arm_id'], r['source_index'], r['repeat']): r for r in warm}
    for item in differences:
        n, method, d, repeat = (item[k] for k in ('length', 'method', 'source_index', 'repeat'))
        diff = difference(warm_map[n, 'original_'+method, d, repeat], warm_map[n, 'candidate_'+method, d, repeat])
        check(all(item[k] == v for k, v in diff.items()) and item['candidate_equals_original_required'] is False
              and item['fullstate_equality_not_measured'] is True, 'independent candidate/original work diagnostic:'+str((n, method, d, repeat)))
    complete = terminal.get('passed') is True
    if complete:
        check(pipeline['exit_code'] == 0 and terminal['status'] == 'TIMING_RAW_COMPLETE_ANALYSIS_REQUIRED'
              and terminal['actual_cuda'] is True and terminal['performance_gate'] == 'ANALYSIS_REQUIRED'
              and terminal['counts'] == dict(prime_requests=24, timing_requests=768, owners=24)
              and terminal['attempted_calls'] == terminal['returned_calls'] == terminal['validated_calls'] == 792
              and len(owners) == 24 and len(differences) == 384, 'actual complete792 matrix with24closed owners, not performance success')
    collection = None
    collection_path = root/'COLLECTION_RECEIPT.json'
    if collection_path.is_file():
        evidence = read(collection_path); collection = dict(sha256=sha(collection_path), record=evidence)
        files = evidence.get('file_sha256', evidence.get('collected_file_sha256', {}))
        sizes = evidence.get('file_size_bytes', {})
        check(evidence['status'] == 'READ_ONLY_TERMINAL_COLLECTION_VERIFIED' and evidence['collected_files'] == 34
              and evidence['remote_root'] == pre['root'] and evidence['remote_writes'] is False
              and evidence['automatic_retry'] is False and evidence['compiled_cache_transferred'] is False
              and evidence['weights_transferred'] is False, 'actual read-only34-file selection, no remote writes/cache/weights transfer')
        check(len(files) == len(sizes) == 34 and files.keys() == sizes.keys(), 'actual collection complete34 hash and size inventories')
        for name, digest in files.items():
            pin(root/name, digest, 'collected raw local byte pin:'+name)
            if name in sizes:
                check((root/name).stat().st_size == sizes[name], 'collected local size:'+name)
        first, last = evidence.get('remote_read_only_barrier_before'), evidence.get('remote_read_only_barrier_after')
        check(first is not None and last is not None, 'actual recorded double read-only transport barriers')
        if first is not None and last is not None:
            check(first['file_sha256'] == last['file_sha256'] == files and first['file_size_bytes'] == last['file_size_bytes'] == sizes
                  and sum(sizes.values()) == 83318496, 'actual before/after/local all34 SHA and sizes agree,83318496bytes')
            for phase, barrier in (('before', first['external_terminal_barrier']), ('after', last['external_terminal_barrier'])):
                check(barrier['passed'] is True and barrier['status'] == 'TERMINAL_EXTERNAL_BARRIER_PASSED'
                      and barrier['root'] == pre['root'] and barrier['terminal_sha256'] == sha(result/'TERMINAL.json')
                      and all(barrier[k] is True for k in ('tmux_absent', 'model_process_absent', 'model_children_absent', 'gpu_compute_idle')),
                      phase+' actual external terminal observations bind correct root and final terminal SHA')
                check(all(barrier[k]['exit_code'] == 1 for k in ('tmux_observation', 'process_observation', 'children_observation'))
                      and barrier['gpu_observation']['exit_code'] == 0 and barrier['gpu_observation']['stdout'] == ''
                      and barrier['identity_observation']['exit_code'] == 0
                      and barrier['identity_observation']['stdout'].strip() == pre['gpu_uuid'],
                      phase+' actual tmux/threePIDs/children absent and GPUempty fixedUUID')
            check(read(root/'COLLECTION_BEFORE.json') == first and read(root/'COLLECTION_AFTER.json') == last,
                  'separately saved complete transport-barrier documents match receipt')
        check(bool(evidence.get('scp_exit_codes')) and all(c == 0 for c in evidence['scp_exit_codes']), 'all actual collection transfers exited0')
    metrics = statistics(warm) if complete else None
    all_primary = bool(metrics and metrics['all_six_primary_numeric_gates'] and all_canonical)
    problems = [r['check'] for r in checks if not r['passed']]
    return dict(status='AUDIT_PASS' if not problems else 'AUDIT_FAIL', checks=len(checks), problems=problems,
        actual_terminal_status=terminal['status'], actual_calls=len(actual), actual_warm_calls=len(warm), actual_cold_primes=len(prime),
        closed_owner_records=len(owners), all_canonical=all_canonical, statistics=metrics,
        all_six_registered_numeric_gates_with_canonical=all_primary,
        raw_complete=complete, experiment_complete=False, goal_complete=False, production_speed_claim=False,
        owner_metrics=timing_cells, source_and_artifact_pins=pins, audit_source_sha256=sha(Path(__file__)),
        collection=collection, collection_provenance_status='RECORDED_LOCAL_BYTES_CHECKED' if collection else 'MISSING',
        detail_checks=checks, limitations=['Saved-raw CPU audit only; no SSH/GPU/runtime or new model call.',
            'Primary denominator is the frozen whole-generate timer, excluding before() telemetry/seed guard and post-return normalization/logging; no profiler/quality clocks are mixed.',
            'Every real capture/JIT remains in its request; no captures are subtracted to predict a fully prewarmed speed.',
            'Finite accounted-service supplement sums constructor/scope+cold+32warm+scope/close; it is not all owner-lifetime idle/validation/logging wall time.',
            'One owner block per context/arm and fixed balanced nonconcurrent order; document-paired CIs do not cover owner/clock drift.',
            'Reused16 development documents/two repeats; no fresh-confirmation, production, general stochastic or SOTA claim.',
            'Original weights were not transferred/rehashed locally. Saved asset fingerprints are verified against sealed history.',
            'Restoration/process evidence binds authenticated guard receipts and saved tool observations, not independent new inspection of remote object identities or a signed attestation.',
            'Raw completion/numeric gates alone are not final experiment or overall-goal completion; plots, report, saved rerun, visual inspection and seal are separate.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--curve', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    value = audit(args.root.resolve(strict=True), args.curve.resolve(strict=True), args.model.resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False); handle.write('\n')
    print(json.dumps({k: value[k] for k in ('status', 'checks', 'problems', 'actual_calls', 'raw_complete')}))
    return 0 if not value['problems'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
