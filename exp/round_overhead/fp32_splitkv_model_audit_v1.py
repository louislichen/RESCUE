"""Independent stdlib saved-record audit; no runner/report/Torch imports.

Verifies only the registered two-document, six-context model Token gate. Timing
fields are instrumented quality diagnostics and are never used to compute TPS.
Source/lease/restoration claims bind saved receipts to frozen guards, not new
local observation of remote objects. Weights were not transferred/rehashable.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


LENGTHS = (2048, 4096, 6144, 8192, 12288, 16384)
DOCS = (271, 303)
MODES = ('rescue_graph', 'retrieval32_k4_graph')
VARIANTS = ('original', 'candidate')
PRIOR_MANIFEST = '70e46d52c3f3f57b1efa8d1633b665f345d54d7656529c9b09e2d25e185bf41e'
PRIOR_VERIFIED = 'c5bf8390ec5349c6d7ce29a688d3b9b13d95b6fd11f006e46a7522b9be42dffc'
PREREG_SHA = 'ecc6ccebca2d75d0feb9bf15c8d2611b58d136b1b58341dc110f1e7568349555'
SUMS_SHA = '6266c3efb911a14ba1fac0722b3434fd639d8f0a3729d674e7b7952ce1d3f9b4'
OPERATOR_SHA = '184674b6a991601378fb3e10ee4b7b4125477efad1400fd21599a5ac98e8c7b6'
WORK = ('tokens', 'accepted', 'proposed', 'blocks', 'draft_forwards', 'selector_refreshes',
        'working_gather_layers', 'working_full_gather_layers', 'working_gathered_position_layers',
        'working_incremental_binds', 'draft_shadow_probes', 'projected_positions', 'rpc_calls')
OPTIONAL = ('target_probe_rounds', 'selection_similarity_elements', 'injection_condition_steps',
            'static_affine_positions', 'working_restored_value_positions', 'rpc_bytes_sent')
TRACE = ('round_index', 'prefix', 'cap', 'proposed', 'accepted', 'terminal', 'workset_tokens')


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
    if not path.is_file() or path.stat().st_size == 0:
        return []
    data = path.read_bytes()
    if not data.endswith(b'\n'):
        raise ValueError('Incomplete saved JSONL: '+str(path))
    return [json.loads(line) for line in data.splitlines()]


def cell(row):
    return row['length'], row['source_index'], row['mode'], row['variant']


def trace(row):
    return [{key: value[key] for key in TRACE} for value in row['trace']]


def difference(left, right):
    return dict(work_fields={key: dict(original=left.get(key), candidate=right.get(key))
                    for key in (*WORK, *OPTIONAL)
                    if (key in left) != (key in right) or left.get(key) != right.get(key)},
                trajectory_equal=trace(left) == trace(right),
                observed_rounds_equal=left.get('observed_rounds') == right.get('observed_rounds'),
                queries_equal=left.get('queries') == right.get('queries'),
                native_t_equal=left.get('native_t') == right.get('native_t'),
                native_s_equal=left.get('native_s') == right.get('native_s'))


def audit(root, prior, operator):
    checks, pins = [], {}
    def check(ok, label):
        checks.append(dict(check=label, passed=bool(ok)))
    def pin(path, expected, label):
        actual = sha(path)
        pins[str(path.resolve())] = actual
        check(actual == expected, label)
    expected = [(n, d, m, v) for n in LENGTHS for d in DOCS for m in MODES for v in VARIANTS]
    pin(root/'PREREGISTRATION.json', PREREG_SHA, 'registered exact48 protocol bytes')
    pin(root/'SOURCE_SHA256SUMS', SUMS_SHA, 'registered13-source inventory bytes')
    pre = read(root/'PREREGISTRATION.json')
    source_lines = (root/'SOURCE_SHA256SUMS').read_text().splitlines()
    check(len(source_lines) == 13, 'all13 new source/launch entries')
    for line in source_lines:
        digest, name = line.split(None, 1)
        check(name == '../launch.sh' or ('..' not in PurePosixPath(name).parts and not name.startswith('/')), 'safe registered source path')
        pin(root/'launch.sh' if name == '../launch.sh' else root/'capsule'/name, digest, 'new frozen byte pin:'+name)
    pin(prior/'FINAL_ARTIFACTS_MANIFEST.json', PRIOR_MANIFEST, 'original798 complete manifest')
    pin(prior/'VERIFIED_COMPLETE.json', PRIOR_VERIFIED, 'original798 verified completion')
    old_manifest = read(prior/'FINAL_ARTIFACTS_MANIFEST.json')['file_sha256']
    old_results = prior/'terminal/results'
    for name in ('protocol.json', 'references.json', 'documents.jsonl', 'runtime.jsonl', 'COMPLETE.json'):
        pin(old_results/name, old_manifest['terminal/results/'+name], 'sealed historical input/ref/work:'+name)
    old = read(old_results/'protocol.json')
    old_refs = read(old_results/'references.json')
    old_docs = {r['source_index']: r for r in rows(old_results/'documents.jsonl')}
    old_work = rows(old_results/'runtime.jsonl')
    check(read(old_results/'COMPLETE.json')['logical_requests'] == 798, 'original798 not new candidate AR')
    pin(operator/'results/TERMINAL.json', OPERATOR_SHA, 'prior actual operator analytic-only terminal')
    op = read(operator/'results/TERMINAL.json')
    pin(operator/'results/CASES.jsonl', op['cases_sha256'], 'prior complete112 analytic operator raw')
    check(op['actual_cuda'] is True and op['status'] == 'CANDIDATE_ANALYTIC_SMOKE_PASSED'
          and op['completed_cases'] == 112 and op['numerical_pass'] is False
          and op['token_gate'] == 'NOT_RUN', 'operator smoke is actual but not Token/equivalence proof')
    result = root/'results'
    terminal, pipeline = read(result/'TERMINAL.json'), read(root/'PIPELINE_EXIT.json')
    pin(result/'TERMINAL.json', pipeline['terminal_sha256'], 'actual pipeline exit binds terminal bytes')
    check(pipeline['status'] == 'MODEL_TOKEN_PIPELINE_EXIT' and pipeline['exit_code'] in (0, 1)
          and pipeline['preserve_partial'] is True and pipeline['automatic_retry'] is False
          and pipeline['performance_measured'] is False and pipeline['goal_complete'] is False,
          'terminal pipeline preserves result and never claims speed/goal')
    check(terminal['performance_measured'] is False and terminal['performance_gate'] == 'NOT_RUN'
          and terminal['goal_complete'] is False, 'instrumented quality never TPS')
    check(terminal['runner_sha256'] == pre['runner_source_sha256'], 'actual terminal frozen runner binding')
    reuse = read(result/'REUSE_RECEIPT.json')
    check(reuse['prior_manifest_sha256'] == PRIOR_MANIFEST and reuse['prior_verified_sha256'] == PRIOR_VERIFIED
          and reuse['operator_terminal_sha256'] == OPERATOR_SHA and reuse['no_new_ar'] is True
          and reuse['no_new_corpus'] is True, 'complete original canonical reuse chain')
    check(reuse['source_sha256'] == old['source_sha256'] and len(reuse['source_sha256']) == 167,
          'all167 original runtime-source fingerprints identical')
    for name, digest in reuse['source_sha256'].items():
        pin(old_results/'source'/name, digest, 'local saved original runtime bytes:'+name)
    check(reuse['assets_sha256'] == old['assets_sha256'], 'saved full asset SHA inventory unchanged, not local weight rehash')
    references, inputs = read(result/'references.json'), read(result/'INPUTS.json')
    wanted = {f'{d}:{n}' for n in LENGTHS for d in DOCS}
    check(set(references) == set(inputs) == wanted, 'exact12 reused input/reference cells')
    for n in LENGTHS:
        for d in DOCS:
            key, doc = f'{d}:{n}', old_docs[d]
            reconstructed = doc['prefix_ids']+doc['document_ids'][:n-len(doc['prefix_ids'])-len(doc['suffix_ids'])]+doc['suffix_ids']
            check(inputs[key] == reconstructed and len(inputs[key]) == n
                  and hashlib.sha256(json.dumps(inputs[key]).encode()).hexdigest() == old['input_sha256'][key], key+' exact original tokens/input SHA')
            check(references[key] == old_refs[key], key+' canonical unchanged, not candidate self-reference')
    quality = rows(result/'QUALITY.jsonl')
    returned_paths = sorted(result.glob('GENERATION_RETURNED_*.json'))
    returned = [read(path) for path in returned_paths]
    owner_paths = sorted(result.glob('OWNER_*.json'))
    owners = [read(path) for path in owner_paths]
    qn, rn = len(quality), len(returned)
    attempted, validated, closed = [terminal.get(k, 0) for k in ('attempted_calls', 'validated_calls', 'private_owners_closed')]
    check(0 <= validated <= qn <= rn <= attempted <= 48 and closed <= attempted, 'actual attempt/return/save/validated/closed ordering')
    check(terminal.get('completed_calls', 0) == qn and terminal.get('returned_calls', 0) == rn
          and closed == len(owners), 'saved actual counts bind raw files')
    check([cell(r) for r in quality] == expected[:qn] and [r['execution_index'] for r in quality] == list(range(qn)),
          'unique actual quality matrix exact registered prefix')
    check([cell(r['row']) for r in returned] == expected[:rn]
          and [r['row']['execution_index'] for r in returned] == list(range(rn)), 'raw returned exact registered prefix')
    check(len({r['owner_id'] for r in owners}) == len(owners)
          and {cell(r) for r in owners} == set(expected[:closed]), 'all distinct private-owner closed records exact prefix')
    owner_map = {r['owner_id']: r for r in owners}
    q_uuid, cells, changed, aggregate = set(), {}, [], []
    for i, row in enumerate(quality):
        label = str(cell(row))
        ref = references[f'{row["source_index"]}:{row["length"]}']
        agrees = row['generated'] == ref
        check(row['performance_eligible'] is False and row['observer_enabled'] is True
              and row['profile_enabled'] is False and row['all_timing_fields_are_instrumented'] is True,
              label+' quality-only clocks not benchmark')
        check(row['tokens'] == len(row['generated']) and 0 < len(row['generated']) <= 128
              and all(type(t) is int and 0 <= t < old['vocab_size'] for t in row['generated']), label+' actual Token and count validity')
        check(row['request_id'] not in q_uuid and type(row['request_id']) is str, label+' unique request identity')
        q_uuid.add(row['request_id'])
        check(all(row[k] == value for k, value in returned[i]['row'].items()), label+' raw persisted before restore/validation agrees quality')
        snapshot, scope = returned[i]['target_scope_at_return'], row['target_attention_scope']
        check(scope['state'] == 'closed' and scope['enabled'] is (row['variant'] == 'candidate')
              and snapshot['records'] == scope['records'] and snapshot['calls'] == scope['calls'], label+' actual scope completed restoration guard')
        if row['variant'] == 'original':
            check(scope['calls'] == scope['candidate_calls'] == scope['original_calls'] == 0 and not scope['records'], label+' original exact unwrapped scope')
            hist = [r for r in old_work if (r['length'], r['source_index'], r['mode']) == cell(row)[:3]]
            check(len(hist) == 2 and all(trace(r) == trace(row) and all(r[k] == row[k] for k in WORK)
                  and all((k in r) == (k in row) and r.get(k) == row.get(k) for k in OPTIONAL) for r in hist),
                  label+' original logical work equals both saved historical repeats')
        else:
            records = scope['records']
            check(scope['calls'] == len(records) and len(records) % 28 == 0 and scope['candidate_calls'] > 0,
                  label+' actual candidate execution metadata present')
            for start in range(0, len(records), 28):
                block = records[start:start+28]
                check([r['layer'] for r in block] == list(range(28))
                      and len({(r['action'], r['query_tokens'], r['key_tokens']) for r in block}) == 1
                      and all(r['outcome'] == 'returned' for r in block), label+f' full Target28 block{start//28}')
            counts = [sum(r['action'] == 'candidate' and r['layer'] == layer for r in records) for layer in range(28)]
            check(min(counts) > 0 and len(set(counts)) == 1 and sum(counts) == scope['candidate_calls'], label+' equal nonzero candidate coverage every Target layer')
            for record in records:
                if record['action'] == 'candidate':
                    check(2 <= record['query_tokens'] <= 5 and record['fallback_reason'] is None,
                          label+' only eligible candidate short-query route')
            original = cells.get((*cell(row)[:3], 'original'))
            if original is not None:
                diff_path = result/f'DIFF_{row["length"]}_{row["source_index"]}_{row["mode"]}.json'
                diff = difference(original, row)
                if diff_path.is_file():
                    check(read(diff_path) == diff, label+' independent work/trajectory/native/Q diagnostics exact')
                changed.append(dict(cell=cell(row), token_exact=agrees, **diff))
        check(row['nonzero_injection'] is (row['mode'] == 'rescue_graph')
              and len(row['queries']) == row['draft_forwards'] and row['native_t'] and row['native_s'], label+' quality/actual nonzero Mapper and full Q/cache records')
        runtime, request = row['runtime_stats'], row['draft_runtime']['request']
        check(runtime['max_graph_entries'] == 128 and runtime['entries'] <= 128
              and row['draft_runtime']['mode'] == 'cuda_graph' and request['captures'] > 0
              and request['replays'] == row['draft_forwards'] > 0, label+' fresh actual Draft Graph execution, no quality owner reuse')
        owner = owner_map.get(row['owner_id'])
        check(owner is not None and owner['engine_closed'] is True and owner['hook_restored'] is True
              and owner['closed_runtime_stats']['entries'] == 0 and cell(owner) == cell(row), label+' owned Graph/hook closure guard')
        if owner is not None and not owner.get('failed_request_cleanup'):
            check(owner['request_id'] == row['request_id'] and owner['final_runtime_stats'] == runtime,
                  label+' final owned cache/Graph ledger binds request')
        cells[cell(row)] = row
        aggregate.append(dict(cell=cell(row), canonical_token_exact=agrees, tokens=row['tokens'],
            accepted=row['accepted'], proposed=row['proposed'], blocks=row['blocks'],
            target_candidate_calls=scope['candidate_calls'], target_original_calls=scope['original_calls']))
    for owner in owners:
        check(owner['engine_closed'] is True and owner['hook_restored'] is True
              and owner['closed_runtime_stats']['entries'] == 0 and owner['performance_eligible'] is False,
              'all closed owners release graphs/hooks:'+owner['owner_id'])
    all_exact = all(r['canonical_token_exact'] for r in aggregate)
    passed = terminal.get('passed') is True
    if passed:
        check(pipeline['exit_code'] == 0 and terminal['actual_cuda'] is True
              and terminal['status'] == 'TARGET_TOKEN_GATE_PASSED'
              and attempted == rn == qn == validated == closed == 48 and all_exact
              and terminal['token_gate'] == 'FROZEN_CANONICAL_TOKEN_EXACT_PASSED', 'actual complete48 frozen-Token-only gate')
    else:
        check(pipeline['exit_code'] == 1 and terminal['status'] == 'FAIL'
              and terminal['token_gate'] == 'NOT_PASSED', 'preserved partial failure never promoted')
    collection_path = root/'COLLECTION_RECEIPT.json'
    collection = None
    if collection_path.is_file():
        collection = dict(sha256=sha(collection_path), record=read(collection_path))
        evidence = collection['record']
        check(evidence['status'] == 'TERMINAL_MODEL_GATE_COLLECTED_READ_ONLY'
              and evidence['collected_files'] == 148 and evidence['remote_root'] == pre['root'],
              'exact148-file successful read-only collection receipt')
        check(len(evidence['file_sha256']) == len(evidence['file_size_bytes']) == 148
              and evidence['file_sha256'].keys() == evidence['file_size_bytes'].keys(),
              'all148 collected hashes and byte sizes recorded')
        for name, digest in evidence['file_sha256'].items():
            check(not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts,
                  'safe collected relative path:'+name)
            pin(root/name, digest, 'collector local bytes:'+name)
            check((root/name).stat().st_size == evidence['file_size_bytes'][name], 'collector actual local byte size:'+name)
        before, after = [evidence.get(k) for k in ('remote_read_only_barrier_before', 'remote_read_only_barrier_after')]
        check(before is not None and after is not None, 'saved collector actually contains both transport barriers')
        if before is not None and after is not None:
            check(before['file_sha256'] == after['file_sha256'] == evidence['file_sha256']
                  and before['file_size_bytes'] == after['file_size_bytes'] == evidence['file_size_bytes'],
                  'complete148 recorded remote before/after barriers equal actual local SHA and size inventories')
        check(bool(evidence.get('scp_exit_codes')) and all(code == 0 for code in evidence['scp_exit_codes']),
              'all recorded read-only transfer commands succeeded')
        observation = read(root/'TERMINAL_EXTERNAL_BARRIER.json')
        pins[str((root/'TERMINAL_EXTERNAL_BARRIER.json').resolve())] = sha(root/'TERMINAL_EXTERNAL_BARRIER.json')
        check(observation == evidence['process_observations'] and observation['status'] == 'TERMINAL_EXTERNAL_BARRIER_PASSED'
              and observation['passed'] is True and observation['model_root'] == pre['root']
              and observation['model_terminal_sha256'] == sha(result/'TERMINAL.json'),
              'separate actual external terminal barrier binds correct model root and final terminal bytes')
        check(all(observation[k] is True for k in ('model_tmux_absent', 'model_process_absent', 'model_children_absent', 'gpu_compute_idle'))
              and all(observation[k]['exit_code'] == 1 for k in ('tmux_observation', 'process_observation', 'children_observation'))
              and observation['gpu_observation']['exit_code'] == 0 and observation['gpu_observation']['stdout'] == ''
              and observation['identity_observation']['exit_code'] == 0
              and observation['identity_observation']['stdout'].strip() == observation['gpu_uuid'] == pre['gpu_uuid'],
              'saved actual SSH tmux/ps/child rc1 and GPU compute empty fixed UUID observations')
    problems = [c['check'] for c in checks if not c['passed']]
    return dict(status='AUDIT_PASS' if not problems else 'AUDIT_FAIL', checks=len(checks), problems=problems,
        actual_terminal_status=terminal['status'], actual_attempted_calls=attempted, actual_returned_calls=rn,
        actual_quality_rows=qn, actual_validated_calls=validated, actual_closed_owners=closed,
        all_saved_quality_tokens_exact=all_exact, model_token_gate_complete=passed and qn == 48 and all_exact,
        new_ar_calls=0, performance_gate='NOT_RUN', performance_measured=False, goal_complete=False,
        experiment_complete=False, distribution_equivalence_claim=False, fresh_confirmation=False,
        source_and_artifact_pins=pins, audit_source_sha256=sha(Path(__file__)),
        collection=collection, collection_provenance_status='SAVED_OBSERVATIONS_CHECKED' if collection else 'MISSING_NOT_INDEPENDENTLY_VERIFIED',
        per_cell_quality=aggregate, candidate_work_diagnostics=changed, detail_checks=checks,
        limitations=['Independent stdlib audit of saved raw, not new remote/CUDA/model execution.',
            'Only two reused documents, six registered contexts and two methods; no population/stochastic/generalization/fresh-confirmation claim.',
            'All timings instrumented; no throughput was calculated or validated.',
            'Source/lease/restoration proofs bind saved guard outcomes to authenticated source; no new local measurement of remote forward identities or hidden tensors.',
            'Recorded asset fingerprints were checked against sealed history; model weights were not transferred and were not locally rehashed.',
            'Collection/process observations, when present, are tool-output transcriptions, not signed attestations; no new SSH checks were performed.',
            'A successful observed Token gate does not establish intermediate/reference floating-point equivalence or the overall optimization goal.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--prior', required=True, type=Path)
    parser.add_argument('--operator', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    receipt = audit(args.root.resolve(strict=True), args.prior.resolve(strict=True), args.operator.resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False); handle.write('\n')
    print(json.dumps({k: receipt[k] for k in ('status', 'checks', 'problems', 'actual_quality_rows', 'model_token_gate_complete')}))
    return 0 if not receipt['problems'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
