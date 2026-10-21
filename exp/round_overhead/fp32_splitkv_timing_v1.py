"""Default PLAN_ONLY: frozen four-arm792-call paired development timing.

Six exact lengths, old16 documents x2 repeats, original/candidate Target
attention for BOTH RESCUE and fixed retrievalTop32/k4 Graph.24 private owners
each execute cold271 then the same sealed32 warm requests. No new references,
corpus, training, policy/Mapper/cost search, observers, profiler or Native digest.
The original curve generate timer is unchanged. All true request-time JIT and
Graph captures stay charged; scope clone/lease and restoration are separately
charged to constructor/close. Existing candidate metadata append is NOT removed.
Prior48-call Token PASS and a separately pinned external terminal barrier are
mandatory. No remote launcher, retries, cleanup or partial-root resumption.
"""
import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import types
import uuid


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path('/mnt/8T/cl/rescue_experiment_cache/fp32_splitkv_model_20261021_v1')
OLD_ROOT = Path('/mnt/8T/cl/rescue_experiment_cache/frozen_devcheck_20261020_v1')
SOURCE_ROOT = OLD_ROOT/'capsule/after'
CURVE_ROOT = Path('/mnt/8T/cl/rescue_experiment_cache/context_curve_20261020_v2')
MODEL_HELPER_SHA = 'e91f613d760f9b94c9c67241fecdc39442d0e917b4ca4de21d75b400a3276bf3'
MODEL_TEST_SHA = 'ff8affd94fecccd6b311c4745ba2704f7448838d7f6cdb6d10cc0ec96f1e2d93'
TARGET_ADAPTER_TEST_SHA = 'a76c3eaa7630c33a8ff343a06978b6a1f96721a10d15744a12c210cfbe16367f'
LENGTHS = (2048, 4096, 6144, 8192, 12288, 16384)
DOCS = (211, 213, 220, 226, 233, 236, 246, 248, 253, 271, 272, 288, 303, 306, 327, 604)
ARM_IDS = ('original_rescue', 'candidate_rescue', 'original_retrieval', 'candidate_retrieval')
ORDER = ((0, 1, 2, 3), (3, 2, 1, 0), (1, 3, 0, 2), (2, 0, 3, 1), (0, 2, 1, 3), (3, 1, 2, 0))
EXPECTED = dict(prime_requests=24, timing_requests=768, owners=24)
TOTAL = 792
RUNTIME_KEYS = ('captures', 'replays', 'resident_steps', 'shadow_updates', 'capture_host_seconds')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for data in iter(lambda: source.read(8 << 20), b''):
            result.update(data)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def append(stream, value):
    stream.write(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')
    stream.flush()
    os.fsync(stream.fileno())


def arms():
    return {arm_id: dict(arm_id=arm_id, variant='candidate' if arm_id.startswith('candidate') else 'original',
        method='rescue' if arm_id.endswith('rescue') else 'retrieval',
        arm=dict(mode='rescue_graph', family='rescue', backend='cuda_graph') if arm_id.endswith('rescue') else
            dict(mode='retrieval32_k4_graph', family='retrieval', backend='cuda_graph', top_chunks=32, fixed_k=4))
        for arm_id in ARM_IDS}


def plan():
    return dict(status='PLAN_ONLY', passed=False, actual_cuda=False, new_generation_calls=0,
        lengths=list(LENGTHS), source_indices=list(DOCS), repeats=2, arm_ids=list(ARM_IDS), arms=arms(),
        owner_order=[[ARM_IDS[index] for index in order] for order in ORDER], expected_counts=EXPECTED,
        planned_generation_calls=TOTAL, planned_private_owners=24, max_new_tokens=128, graph_max_entries=128,
        cold_prime_source_index=271, quality_observer=False, native_digest=False, profiler=False,
        candidate_scope='One scoped Target+owned HookOwners lease per entire persistent owner: cold+32warm',
        primary='Sum warm output tokens / sum frozen-curve whole-generate elapsed; document-paired analysis, n16',
        timer_boundary='Frozen RuntimeAdapter elapsed starts after before() telemetry/seed/idle and includes HookOwners.generate; row normalization outside',
        all_request_jit_and_captures_charged=True, constructor_scope_clone_and_close_reported=True,
        target_adapter_metadata_append_kept=True, operator_or_model_gate_is_not_speed_evidence=True,
        prior_model_token_pass_required=True, external_model_process_absence_barrier_required=True,
        token_gate='NOT_RUN', performance_gate='NOT_RUN', performance_measured=False, goal_complete=False,
        original_history_work_equal_required=True, candidate_repeat_trajectory_equal_required=True,
        candidate_equals_original_trajectory_required=False, fresh_confirmation=False, no_training=True,
        no_search=True, no_new_references=True, no_new_corpus=True, no_retry=True,
        model_helper_sha256=MODEL_HELPER_SHA, model_test_sha256=MODEL_TEST_SHA)


def load_contract():
    path = ROOT/'exp/round_overhead/fp32_splitkv_model_gate_v1.py'
    require(sha(path) == MODEL_HELPER_SHA and sha(ROOT/'tests/test_fp32_splitkv_model_gate_v1.py') == MODEL_TEST_SHA,
            'Frozen model contract/proof source changed')
    name = '_splitkv_timing_frozen_model_contract'
    require(name not in sys.modules, 'Fresh private contract namespace required')
    module = types.ModuleType(name)
    module.__file__, module.__package__ = str(path), ''
    sys.modules[name] = module
    try:
        exec(compile(path.read_bytes(), str(path), 'exec', dont_inherit=True), module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def validate_terminal_barrier(barrier, model_terminal_sha256, gpu_uuid):
    expected = dict(schema=1, status='TERMINAL_EXTERNAL_BARRIER_PASSED', passed=True,
        model_root=str(MODEL_ROOT), model_terminal_sha256=model_terminal_sha256,
        model_process_absent=True, model_tmux_absent=True, model_children_absent=True,
        gpu_compute_idle=True, gpu_uuid=gpu_uuid)
    require(all(type(barrier.get(key)) is type(value) and barrier[key] == value for key, value in expected.items()),
            'Pinned external model process/tmux/children/GPU terminal barrier required')


def validate_model_terminal(terminal):
    expected = dict(status='TARGET_TOKEN_GATE_PASSED', passed=True, actual_cuda=True, completed_calls=48,
        attempted_calls=48, returned_calls=48, validated_calls=48, private_owners_closed=48,
        frozen_canonical_token_exact=True, token_gate='FROZEN_CANONICAL_TOKEN_EXACT_PASSED',
        performance_gate='NOT_RUN', performance_measured=False, runner_sha256=MODEL_HELPER_SHA)
    require(all(type(terminal.get(key)) is type(value) and terminal[key] == value for key, value in expected.items()),
            'Actual complete48-call frozen-canonical Token gate required')


def prepare(args):
    contract = load_contract()
    output = contract.validate_paths(args)
    require(not output.is_relative_to(MODEL_ROOT) and not args.cache_dir.is_relative_to(MODEL_ROOT), 'Never write old model gate')
    for path in (args.preregistration, args.prior_seal_root, args.model_terminal, args.terminal_barrier):
        require(path is not None, 'Explicit protocol/seals/model terminal/external barrier required')
        contract.no_links(path)
    require(args.model_terminal == MODEL_ROOT/'results/TERMINAL.json', 'Only frozen completed model gate allowed')
    require(sha(args.model_terminal) == args.model_terminal_sha256, 'Model terminal SHA changed')
    terminal = read(args.model_terminal)
    validate_model_terminal(terminal)
    pipeline = read(MODEL_ROOT/'PIPELINE_EXIT.json')
    require(pipeline['exit_code'] == 0 and pipeline['terminal_sha256'] == args.model_terminal_sha256, 'Model pipeline not terminal-success')
    require(sha(args.terminal_barrier) == args.terminal_barrier_sha256, 'External terminal barrier SHA changed')
    validate_terminal_barrier(read(args.terminal_barrier), args.model_terminal_sha256, contract.GPU_UUID)
    pre = read(args.preregistration)
    expected = dict(status='FROZEN_BEFORE_EXECUTION', lengths=list(LENGTHS), source_indices=list(DOCS), repeats=2,
        arm_ids=list(ARM_IDS), owner_order=[[ARM_IDS[i] for i in order] for order in ORDER], expected_counts=EXPECTED,
        generation_calls=TOTAL, private_owners=24, max_new_tokens=128, graph_max_entries=128,
        cold_prime_source_index=271, quality_observer=False, native_digest=False, profiler=False,
        execution_output=str(output), old_root=str(OLD_ROOT), source_root=str(SOURCE_ROOT), curve_root=str(CURVE_ROOT),
        model_terminal_sha256=args.model_terminal_sha256, terminal_barrier_sha256=args.terminal_barrier_sha256,
        model_helper_sha256=MODEL_HELPER_SHA, runner_sha256=sha(Path(__file__)),
        target_adapter_source_sha256=contract.TARGET_ADAPTER_SHA, candidate_source_sha256=contract.CANDIDATE_SHA)
    require(all(type(pre.get(key)) is type(value) and pre[key] == value for key, value in expected.items()), 'Frozen792-call timing protocol changed')
    for name, digest in (('FINAL_ARTIFACTS_MANIFEST.json', contract.PRIOR_MANIFEST), ('VERIFIED_COMPLETE.json', contract.PRIOR_VERIFIED)):
        require(sha(args.prior_seal_root/name) == digest, 'Old complete curve seal changed')
    sealed = read(args.prior_seal_root/'VERIFIED_COMPLETE.json')
    require(sealed['passed'] is True and sealed['six_lengths_complete'] is True and
            sealed['final_artifacts_manifest_sha256'] == contract.PRIOR_MANIFEST, 'Old curve incomplete')
    full = read(args.prior_seal_root/'FINAL_ARTIFACTS_MANIFEST.json')['file_sha256']
    needed = ('results/protocol.json', 'results/COMPLETE.json', 'results/QUALITY_PASSED.json', 'results/REUSE_RECEIPT.json',
        'results/references.json', 'results/documents.jsonl', 'results/prime.json', 'results/runtime.jsonl',
        'results/own_quality.jsonl', 'results/reused_own_quality.jsonl', 'PIPELINE_EXIT.json', 'LAUNCH_MANIFEST.json', 'tools/run_context_curve.py')
    curve_pins = {name: full['terminal/'+name] for name in needed}
    contract.inventory(CURVE_ROOT, curve_pins)
    old = read(CURVE_ROOT/'results/protocol.json')
    require(read(CURVE_ROOT/'results/COMPLETE.json')['passed'] is True and
            read(CURVE_ROOT/'results/COMPLETE.json')['logical_requests'] == 798 and
            read(CURVE_ROOT/'PIPELINE_EXIT.json')['exit_code'] == 0, 'Frozen798 curve did not complete')
    require(old['lengths'] == list(LENGTHS) and old['source_indices'] == list(DOCS), 'Original context/doc selection changed')
    require(sha(CURVE_ROOT/'tools/run_context_curve.py') == contract.CURVE_SHA, 'Original RuntimeAdapter changed')
    source_pins = old['source_sha256']
    require(len(source_pins) == 167, 'Full167 original scientific source files required')
    contract.inventory(SOURCE_ROOT, source_pins)
    old_pins = read(CURVE_ROOT/'results/REUSE_RECEIPT.json')['old_artifact_pins']
    contract.inventory(OLD_ROOT, old_pins)
    require(sha(OLD_ROOT/'tools/frozen_devcheck.py') == contract.HELPER_SHA, 'Original source-import helper changed')
    for name, digest in old['assets_sha256'].items():
        contract.no_links(Path(name))
        require(sha(Path(name)) == digest, 'Original model/tokenizer/Mapper/policy/cost asset changed')
    for role, digest in contract.ASSETS.items():
        require(sha(Path(old['args'][role])) == digest, 'Mapper/policy/cost checkpoint changed')
    external_pins = {'exp/round_overhead/fp32_splitkv_model_gate_v1.py': MODEL_HELPER_SHA,
        'tests/test_fp32_splitkv_model_gate_v1.py': MODEL_TEST_SHA,
        'exp/round_overhead/fp32_splitkv_target_adapter_v1.py': contract.TARGET_ADAPTER_SHA,
        'tests/test_fp32_splitkv_target_adapter_v1.py': TARGET_ADAPTER_TEST_SHA,
        'exp/round_overhead/fp32_splitkv_candidate_v1.py': contract.CANDIDATE_SHA}
    contract.inventory(ROOT, external_pins)
    documents = contract.rows(CURVE_ROOT/'results/documents.jsonl')
    require([row['source_index'] for row in documents] == list(DOCS), 'Saved16-document order changed')
    references = read(CURVE_ROOT/'results/references.json')
    inputs = {}
    eos, vocab = old['tokenizer_eos_token_id'], old['vocab_size']
    for row in documents:
        document = row['source_index']
        left, body, right = (row[key] for key in ('prefix_ids', 'document_ids', 'suffix_ids'))
        require(len(left)+len(right) == 30 and len(body) == 16354, 'Natural saved body/prompt changed')
        for n in LENGTHS:
            ids = left+body[:n-30]+right
            require(len(ids) == n and hashlib.sha256(json.dumps(ids).encode()).hexdigest() == old['input_sha256'][f'{document}:{n}'], 'Exact input SHA changed')
            ref = references[f'{document}:{n}']
            require(type(ref) is list and 0 < len(ref) <= 128 and all(type(v) is int and 0 <= v < vocab for v in ref) and
                (len(ref) == 128 or ref[-1] == eos) and (eos not in ref or ref[-1] == eos and eos not in ref[:-1]), 'Invalid old canonical EOS/cap')
            inputs[document, n] = ids
    sequence = old['shared_warm_sequence']
    require(type(sequence) is list and len(sequence) == 32 and all(set(row) == {'source_index', 'repeat'} for row in sequence)
            and {(row['source_index'], row['repeat']) for row in sequence} == {(document, repeat) for document in DOCS for repeat in (0, 1)}, 'Sealed32 warm sequence changed')
    historical = {}
    for row in contract.rows(CURVE_ROOT/'results/runtime.jsonl'):
        if row['mode'] in ('rescue_graph', 'retrieval32_k4_graph'):
            key = row['length'], row['source_index'], row['mode'], row['repeat']
            require(key not in historical, 'Duplicate historical work row')
            historical[key] = row
    require(len(historical) == 384, 'All two-method old384 warm work rows required')
    quality = contract.rows(MODEL_ROOT/'results/QUALITY.jsonl')
    expected_quality = {(n, document, method, variant) for n in LENGTHS for document in (271, 303)
        for method in ('rescue_graph', 'retrieval32_k4_graph') for variant in ('original', 'candidate')}
    observed_quality = set()
    for row in quality:
        key = row['length'], row['source_index'], row['mode'], row['variant']
        require(key not in observed_quality and key in expected_quality and row['generated'] == references[f'{key[1]}:{key[0]}'], 'Actual model gate outputs differ from OLD references')
        contract.check_scope(row['target_attention_scope'], row['variant'])
        observed_quality.add(key)
    require(observed_quality == expected_quality, 'Actual48 model quality rows incomplete')
    return dict(contract=contract, old=old, source_pins=source_pins, old_pins=old_pins, curve_pins=curve_pins,
        assets=old['assets_sha256'], external_pins=external_pins, inputs=inputs, references=references,
        historic=historical, sequence=sequence, entry_sha256=contract.CURVE_SHA,
        preregistration_sha256=sha(args.preregistration), runner_sha256=sha(Path(__file__)),
        model_terminal_sha256=args.model_terminal_sha256, terminal_barrier_sha256=args.terminal_barrier_sha256,
        seen=set(), counts=dict.fromkeys(EXPECTED, 0), attempted_calls=0, returned_calls=0, validated_calls=0)


def graph_ledger(row, previous=None):
    stats, delta = row['runtime_stats'], row['draft_runtime']['request']
    keys = {json.dumps(key, sort_keys=True) for key in stats['graph_keys']}
    previous = previous or dict(keys=set(), totals=dict.fromkeys(RUNTIME_KEYS, 0))
    require(stats['mode'] == 'cuda_graph' and stats['max_graph_entries'] == 128 and
        stats['entries'] == stats['captures'] == len(keys) <= 128 and previous['keys'] <= keys and
        len(keys-previous['keys']) == delta['captures'], 'Graph entry/capture ledger changed')
    require(delta['replays'] == delta['shadow_updates'] == row['draft_forwards'] > 0 and delta['resident_steps'] == 0, 'Actual Graph replay ledger changed')
    totals = {key: previous['totals'][key]+delta[key] for key in RUNTIME_KEYS}
    require(all(math.isclose(totals[key], stats[key], rel_tol=1e-10, abs_tol=1e-8) and
                math.isclose(totals[key], row['draft_runtime']['lifetime'][key], rel_tol=1e-10, abs_tol=1e-8)
                for key in RUNTIME_KEYS), 'Graph lifetime/capture costs not fully accounted')
    return dict(keys=keys, totals=totals)


def execute_matrix(adapter, prepared, output):
    lookup, repeated, pairs = arms(), {}, {}
    with (output/'PRIME.jsonl').open('x') as prime_stream, (output/'RUNTIME.jsonl').open('x') as warm_stream, \
         (output/'OWNERS.jsonl').open('x') as owner_stream:
        for n, order in zip(LENGTHS, ORDER):
            for index in order:
                descriptor = lookup[ARM_IDS[index]]
                owner_id = uuid.uuid4().hex
                handle = None
                try:
                    with adapter.owner(descriptor) as handle:
                        signature, ledger = None, None
                        requests = [dict(source_index=271, phase='cold_prime')]+[dict(row, phase='warm') for row in prepared['sequence']]
                        warm_tokens, warm_seconds, cold_seconds, cold_tokens, warm_ids = 0, 0., None, None, []
                        for request in requests:
                            document = request['source_index']
                            item = dict(request, arm_id=descriptor['arm_id'], variant=descriptor['variant'],
                                method=descriptor['method'], mode=descriptor['arm']['mode'], length=n, owner_id=owner_id,
                                performance_eligible=request['phase'] == 'warm', observer_enabled=False, profile_enabled=False,
                                execution_index=prepared['returned_calls'])
                            prepared['attempted_calls'] += 1
                            row, elapsed = adapter.generate(handle, descriptor, prepared['inputs'][document, n],
                                prepared['references'][f'{document}:{n}'], item)
                            prepared['returned_calls'] += 1
                            row.update(outer_seconds=elapsed, outer_tps=row['tokens']/elapsed if elapsed > 0 else None,
                                all_request_jit_and_captures_charged=True, scope_metadata_append_kept=True)
                            # Preserve returned output before canonical/work/ledger validation.
                            append(prime_stream if request['phase'] == 'cold_prime' else warm_stream, row)
                            key = 'prime_requests' if request['phase'] == 'cold_prime' else 'timing_requests'
                            prepared['counts'][key] += 1
                            require(math.isfinite(elapsed) and elapsed > 0, 'Invalid unprofiled generate duration')
                            reference = prepared['references'][f'{document}:{n}']
                            require(row['generated'] == reference, 'Frozen OLD canonical Token mismatch')
                            require(type(row.get('request_id')) is str and row['request_id'] not in prepared['seen'], 'Duplicate/missing request identity')
                            prepared['seen'].add(row['request_id'])
                            adapter.validate(row, reference, descriptor, request['phase'] == 'cold_prime')
                            ledger = graph_ledger(row, ledger)
                            signature = row['owner_signature'] if signature is None else signature
                            require(row['owner_signature'] == signature, 'Persistent owner/cache addresses changed')
                            historical = prepared['historic'][n, document, descriptor['arm']['mode'], request.get('repeat', 0)]
                            if descriptor['variant'] == 'original':
                                adapter.compare_work(historical, row)
                            repeat_key = n, descriptor['arm_id'], document
                            if repeat_key in repeated:
                                adapter.compare_work(repeated[repeat_key], row)
                            else:
                                repeated[repeat_key] = row
                            pairs[n, descriptor['arm_id'], document, request.get('repeat', -1)] = row
                            prepared['validated_calls'] += 1
                            if request['phase'] == 'cold_prime':
                                cold_seconds = elapsed
                                cold_tokens = row['tokens']
                            else:
                                warm_tokens += row['tokens']; warm_seconds += elapsed; warm_ids.append(row['request_id'])
                        handle.result = dict(owner_id=owner_id, arm_id=descriptor['arm_id'], variant=descriptor['variant'],
                            method=descriptor['method'], mode=descriptor['arm']['mode'], length=n,
                            constructor_scope_seconds=handle.constructor_seconds, cold_prime_seconds=cold_seconds,
                            cold_prime_tokens=cold_tokens,
                            cold_complete_seconds=handle.constructor_seconds+cold_seconds,
                            warm_tokens=warm_tokens, warm_outer_seconds_sum=warm_seconds,
                            warm_tps=warm_tokens/warm_seconds, warm_request_ids=warm_ids, owner_signature=signature,
                            primary_warm_requests=32, max_live_owners=1, all_captures_charged=True)
                finally:
                    if handle is not None:
                        active_error = sys.exc_info()[1]
                        owner_result = dict(getattr(handle, 'result', {}))
                        if owner_result:
                            service = owner_result['constructor_scope_seconds']+owner_result['cold_prime_seconds']+owner_result['warm_outer_seconds_sum']+handle.close_receipt['scope_and_close_seconds']
                            tokens = owner_result['cold_prime_tokens']+owner_result['warm_tokens']
                            owner_result.update(accounted_service_seconds=service, accounted_service_tokens=tokens,
                                                accounted_service_tps=tokens/service)
                        append(owner_stream, dict(owner_result, owner_id=owner_id, arm_id=descriptor['arm_id'],
                            length=n, close_receipt=handle.close_receipt, scope_summary=handle.scope_summary,
                            primary_error=getattr(handle, 'primary_error', None), cleanup_errors=getattr(handle, 'cleanup_errors', [])))
                        closed = handle.close_receipt.get('engine_closed') is True and handle.close_receipt.get('hook_restored') is True
                        if closed:
                            prepared['counts']['owners'] += 1
                        elif active_error is None:
                            raise ValueError('Persistent owner did not close/restore')
        require(prepared['counts'] == EXPECTED and len(prepared['seen']) == prepared['attempted_calls'] ==
                prepared['returned_calls'] == prepared['validated_calls'] == TOTAL, 'Incomplete792-call timing matrix')
        with (output/'WORK_DIFFERENCES.jsonl').open('x') as stream:
            for n in LENGTHS:
                for method in ('rescue', 'retrieval'):
                    for document in DOCS:
                        for repeat in (0, 1):
                            old = pairs[n, 'original_'+method, document, repeat]
                            new = pairs[n, 'candidate_'+method, document, repeat]
                            difference = adapter.work_difference(old, new)
                            append(stream, dict(length=n, method=method, source_index=document, repeat=repeat,
                                **difference, candidate_equals_original_required=False, fullstate_equality_not_measured=True))
        adapter.assert_unchanged()
        require(adapter.actual_cuda is True, 'Actual CUDA generation required')
        return dict(status='TIMING_RAW_COMPLETE_ANALYSIS_REQUIRED', passed=True, actual_cuda=True,
            counts=prepared['counts'], attempted_calls=TOTAL, returned_calls=TOTAL, validated_calls=TOTAL,
            logical_requests=TOTAL, all_old_canonical_tokens=True, original_historical_work_equal=True,
            candidate_repeated_work_equal=True, observers=False, profiler=False, performance_measured=True,
            performance_gate='ANALYSIS_REQUIRED', paired_document_count=16, repeats=2,
            goal_complete=False, fresh_confirmation=False, production_speed_claim=False,
            figures_and_paired_analysis_required=True)


class RuntimeAdapter:
    def __init__(self, args, prepared):
        self.args, self.prepared, self.contract = args, prepared, prepared['contract']
        self.shared = self.contract.RuntimeAdapter(args, prepared)
        self.base, self.external = self.shared.base, self.shared.external
        self.actual_cuda = False
        self.owner_failures = []

    @contextmanager
    def owner(self, descriptor):
        base = self.base
        started = time.perf_counter()
        engine, scope, snapshot, original = None, None, None, None
        handle = types.SimpleNamespace(close_receipt={}, scope_summary={}, primary_error=None, cleanup_errors=[])
        close_started = None
        try:
            engine = base.make(descriptor['arm'], False)
            handle.engine = engine
            original = tuple(layer.self_attn.forward for layer in base.teacher.model.layers)
            snapshot = self.contract.hook_snapshot(base.group, base.teacher, base.student, engine)
            scope = self.external.TargetAttentionScope(base.teacher, enabled=descriptor['variant'] == 'candidate', cache_dir=self.args.cache_dir)
            handle.scope = scope
            with base.r.torch.no_grad(), scope:
                lease = self.contract.target_hook_lease(base.group, engine, snapshot) if descriptor['variant'] == 'candidate' else nullcontext()
                with lease:
                    handle.constructor_seconds = time.perf_counter()-started
                    try:
                        yield handle
                    finally:
                        close_started = time.perf_counter()
        except BaseException as error:
            handle.primary_error = dict(error_type=type(error).__name__, error=str(error))
            raise
        finally:
            try:
                if original is not None:
                    require(all(layer.self_attn.forward is before for layer, before in zip(base.teacher.model.layers, original)), 'Owner Target scope did not restore original28 forwards')
                if snapshot is not None:
                    require(base.group.originals is snapshot['originals'] and tuple(base.group.originals) == snapshot['original_values'] and
                            base.group.modules is snapshot['modules'] and tuple(base.group.modules) == snapshot['module_values'], 'Owner hook-group metadata changed')
                base.group.idle()
                if scope is not None:
                    handle.scope_summary = scope.stats()
                    self.contract.check_scope(handle.scope_summary, descriptor['variant'])
            except BaseException as error:
                handle.cleanup_errors.append(dict(stage='scope_or_hook_guard', error_type=type(error).__name__, error=str(error)))
            finally:
                if engine is not None:
                    try:
                        handle.close_receipt = base.close(engine)
                    except BaseException as error:
                        handle.cleanup_errors.append(dict(stage='owner_close', error_type=type(error).__name__, error=str(error)))
                handle.close_receipt['scope_and_close_seconds'] = time.perf_counter()-(close_started or started)
            if handle.primary_error or handle.cleanup_errors:
                self.owner_failures.append(dict(arm_id=descriptor['arm_id'], primary_error=handle.primary_error,
                    cleanup_errors=handle.cleanup_errors, close_receipt=handle.close_receipt))
            if handle.cleanup_errors and handle.primary_error is None:
                raise RuntimeError('Persistent owner cleanup failed: '+json.dumps(handle.cleanup_errors, sort_keys=True))

    def generate(self, handle, descriptor, ids, reference, item):
        row, elapsed = self.base.generate(handle.engine, descriptor['arm'], ids, reference, item, False)
        self.actual_cuda = True
        return row, elapsed

    def validate(self, row, reference, descriptor, cold):
        self.base.validate_timing(row, reference, descriptor['arm'], cold)
        require('observed_rounds' not in row and 'queries' not in row and 'native_t' not in row and 'native_s' not in row
                and 'timeline' not in row and 'cost_samples' not in row, 'Unprofiled timing acquired observer/digest/profiler payload')

    def compare_work(self, left, right):
        self.base.common.check_pair(left, right)
        require(all(key in left and key in right and left[key] == right[key] for key in self.contract.WORK), 'Logical work changed')
        require(all((key in left) == (key in right) and (key not in left or left[key] == right[key]) for key in self.contract.OPTIONAL), 'Optional work changed')

    def work_difference(self, left, right):
        difference = self.contract.work_difference(left, right)
        return {key: difference[key] for key in ('work_fields', 'trajectory_equal')}

    def assert_unchanged(self):
        # Reuse only the frozen serving runtime's own invariant check. The old
        #48 runner's assertion refers to its operator-terminal CLI, not ours.
        self.base.assert_unchanged()
        self.contract.inventory(CURVE_ROOT, self.prepared['curve_pins'])
        self.contract.inventory(OLD_ROOT, self.prepared['old_pins'])
        self.contract.inventory(ROOT, self.prepared['external_pins'])
        require(sha(Path(__file__)) == self.prepared['runner_sha256'] and
                sha(self.args.model_terminal) == self.prepared['model_terminal_sha256'] and
                sha(self.args.terminal_barrier) == self.prepared['terminal_barrier_sha256'], 'Prior model/barrier evidence changed')

    def finish(self):
        self.shared.finish()


def run(args):
    prepared = prepare(args)
    output = args.output
    output.mkdir(mode=0o700, exist_ok=False)
    adapter = None
    result = dict(status='FAIL', passed=False, actual_cuda=False, goal_complete=False,
        performance_gate='NOT_PASSED', preserve_partial=True, automatic_retry=False)
    try:
        write(output/'PLAN.json', plan())
        write(output/'PREREGISTRATION.json', read(args.preregistration))
        write(output/'REUSE_RECEIPT.json', dict(old_source_sha256=prepared['source_pins'], old_artifact_pins=prepared['old_pins'],
            curve_artifact_pins=prepared['curve_pins'], assets_sha256=prepared['assets'], external_source_sha256=prepared['external_pins'],
            model_terminal_sha256=prepared['model_terminal_sha256'], terminal_barrier_sha256=prepared['terminal_barrier_sha256'],
            shared_warm_sequence=prepared['sequence'], no_new_ar=True, no_new_corpus=True))
        write(output/'references.json', prepared['references'])
        write(output/'INPUTS.json', {f'{document}:{n}': value for (document, n), value in prepared['inputs'].items()})
        adapter = RuntimeAdapter(args, prepared)
        base = adapter.base
        write(output/'ENVIRONMENT.json', dict(environment=base.environment, policy_metadata=base.policy_meta,
            source_bindings=base.imports, tree_source_bindings=base.tree_bindings, flash_attention=base.flash,
            configs={mode: asdict(base.legacy.own_config('cuda_graph')) if arm['family'] == 'rescue'
                     else asdict(base.baseline.linear_config(arm)) for mode, arm in prepared['contract'].arms().items()},
            max_live_owners=1, max_new_tokens=128, graph_max_entries=128, observers=False, profiler=False,
            timer_boundary=plan()['timer_boundary'], scope_metadata_append_kept=True))
        result.update(execute_matrix(adapter, prepared, output))
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error))
    finally:
        if adapter is not None:
            result.update(actual_cuda=adapter.actual_cuda, owner_failure_receipts=adapter.owner_failures)
            try:
                adapter.finish()
            except BaseException as error:
                result.update(status='FAIL', passed=False, close_error=type(error).__name__+': '+str(error))
        result.update(counts=prepared['counts'], attempted_calls=prepared['attempted_calls'], returned_calls=prepared['returned_calls'],
            validated_calls=prepared['validated_calls'], runner_sha256=prepared['runner_sha256'])
        write(output/'TERMINAL.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-timing', action='store_true')
    for name in ('output', 'cache-dir', 'preregistration', 'prior-seal-root', 'model-terminal', 'terminal-barrier'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--model-terminal-sha256')
    parser.add_argument('--terminal-barrier-sha256')
    args = parser.parse_args(argv)
    if not args.execute_timing:
        print(json.dumps(plan(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    try:
        require(not any(name.split('.')[0] in ('torch', 'triton', 'rescue') for name in sys.modules), 'Fresh owned subprocess required')
        result = run(args)
    except BaseException as error:
        print('EXECUTION_REFUSED: '+type(error).__name__+': '+str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
