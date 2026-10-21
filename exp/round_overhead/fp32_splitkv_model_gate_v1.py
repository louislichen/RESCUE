"""Default PLAN_ONLY. Explicit48-call frozen-canonical Target Token gate.

Six exact lengths x old documents271/303 x two Graph arms x original/candidate.
Every call has a fresh private quality owner; no AR references are generated.
The pinned historical798-call canonical tokens, inputs,167 source files and
full model/Mapper/policy/cost inventory are verified before loading models.
Only candidate owners temporarily enable the private Target attention scope.
Intermediate floats/actions may differ; both outputs must equal the OLD refs.
Quality observers instrument both variants; their clocks are NOT performance.
No remote launcher, training, retry, cleanup, corpus read or confirmation data.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import types
import uuid


ROOT = Path(__file__).resolve().parents[2]
OLD_ROOT = Path('/mnt/8T/cl/rescue_experiment_cache/frozen_devcheck_20261020_v1')
SOURCE_ROOT = OLD_ROOT/'capsule/after'
CURVE_ROOT = Path('/mnt/8T/cl/rescue_experiment_cache/context_curve_20261020_v2')
LENGTHS = (2048, 4096, 6144, 8192, 12288, 16384)
DOCS = (271, 303)
MODES = ('rescue_graph', 'retrieval32_k4_graph')
VARIANTS = ('original', 'candidate')
TOTAL = 48
GPU_UUID = 'GPU-9db4f7b5-1447-9a2a-5c1b-80379d909fd9'
CURVE_SHA = '1ef58e790894757e7a8a8a5dd328fd57cc45a6525d7ca6bb9436b96dabec1be3'
HELPER_SHA = 'ce4f06ab60e34bf27b448dafe5f4574332557cda9e8b9733d697e8a4208a0069'
PRIOR_MANIFEST = '70e46d52c3f3f57b1efa8d1633b665f345d54d7656529c9b09e2d25e185bf41e'
PRIOR_VERIFIED = 'c5bf8390ec5349c6d7ce29a688d3b9b13d95b6fd11f006e46a7522b9be42dffc'
CANDIDATE_SHA = '9f133cc824915afd09d9ad25a1b16e6cd721522e1f583cabd5e102235358cfd2'
OPERATOR_GATE_SHA = '77501b123927d4ce0606da9017b14d8adaa84ce90059285b70735d8de1f3d39e'
OPERATOR_TEST_SHA = 'fd94317f0c1514a459f531c615ea3a39124d12c4e72f2700090c7549c4e336cb'
# Adapter author freezes this exact public API before any model execution.
TARGET_ADAPTER_SHA = '78092c417c10e25aa69ac9055fda59394e4349d021a277cfecfb43757e0e63a5'
TARGET_ADAPTER_TEST_SHA = 'a76c3eaa7630c33a8ff343a06978b6a1f96721a10d15744a12c210cfbe16367f'
ASSETS = dict(mapper='d08889e796ca2527e23241ff34a072ef640e2fdf3fe9a2f9ba5439975af04d9f',
    policy='3c333a470119c236afcc1308c18592842a535bee0b8f83c4fb4da359ff17eff5',
    cost='d92c843a84676c0cd6adaf9364bbf7e8c771d56548f1016b5720a3e04f371fa1')
WORK = ('tokens', 'accepted', 'proposed', 'blocks', 'draft_forwards', 'selector_refreshes',
    'working_gather_layers', 'working_full_gather_layers', 'working_gathered_position_layers',
    'working_incremental_binds', 'draft_shadow_probes', 'projected_positions', 'rpc_calls')
OPTIONAL = ('target_probe_rounds', 'selection_similarity_elements', 'injection_condition_steps',
    'static_affine_positions', 'working_restored_value_positions', 'rpc_bytes_sent')
TRACE_KEYS = ('round_index', 'prefix', 'cap', 'proposed', 'accepted', 'terminal', 'workset_tokens')


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


def rows(path):
    raw = Path(path).read_bytes()
    require(raw and raw.endswith(b'\n'), 'Complete saved JSONL required')
    result = [json.loads(line) for line in raw.splitlines()]
    require(all(type(row) is dict for row in result), 'JSONL objects required')
    return result


def no_links(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and
            not any(item.is_symlink() for item in (path, *path.parents)), 'Absolute no-symlink path required')


def inventory(root, pins):
    no_links(root)
    require(type(pins) is dict and pins, 'Nonempty pinned inventory required')
    for name, digest in pins.items():
        part = PurePosixPath(name)
        require(part.parts and not part.is_absolute() and part.as_posix() == name and
                '..' not in part.parts and '\\' not in name, 'Unsafe inventory path')
        target = root/name
        no_links(target)
        require(target.is_file() and sha(target) == digest, 'Frozen file changed: '+name)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def arms():
    return {MODES[0]: dict(mode=MODES[0], family='rescue', backend='cuda_graph'),
            MODES[1]: dict(mode=MODES[1], family='retrieval', backend='cuda_graph', top_chunks=32, fixed_k=4)}


def plan():
    return dict(status='PLAN_ONLY', passed=False, actual_cuda=False, new_model_calls=0,
        lengths=list(LENGTHS), source_indices=list(DOCS), modes=list(MODES), variants=list(VARIANTS),
        max_new_tokens=128, graph_max_entries=128, planned_generation_calls=TOTAL, planned_private_owners=TOTAL,
        quality_observer=True, performance_measured=False, performance_gate='NOT_RUN', token_gate='NOT_RUN',
        canonical_reference='Frozen original798-call refs; never candidate-self-generated or new AR',
        candidate_intermediate_bitwise_required=False, candidate_trajectory_equal_required=False,
        original_historical_work_equal_required=True, nonzero_mapper_required_in_both_rescue_variants=True,
        scope='Two already-used documents, six lengths, both arms; no distribution-wide/fresh-confirmation claim',
        old_root=str(OLD_ROOT), source_root=str(SOURCE_ROOT), curve_root=str(CURVE_ROOT),
        candidate_source_sha256=CANDIDATE_SHA, operator_gate_source_sha256=OPERATOR_GATE_SHA,
        operator_gate_test_source_sha256=OPERATOR_TEST_SHA, target_adapter_source_sha256=TARGET_ADAPTER_SHA,
        target_adapter_test_source_sha256=TARGET_ADAPTER_TEST_SHA,
        gpu_uuid=GPU_UUID, target_dtype='float32', draft_dtype='bfloat16', tf32=False,
        goal_complete=False, fresh_confirmation=False, no_training=True, no_search=True, no_retry=True)


def owned_directory(path):
    no_links(path)
    path = Path(path)
    require(path.resolve(strict=True) == path, 'Canonical existing directory required')
    value = path.stat()
    require(stat.S_ISDIR(value.st_mode) and value.st_uid == os.geteuid() and not value.st_mode & 0o022,
            'Owned non-group/world-writable directory required')
    require(path not in (Path('/'), Path.home().resolve()), 'Broad directory forbidden')
    return path


def validate_paths(args):
    require(args.output is not None and args.cache_dir is not None, 'Explicit output/cache required')
    output = Path(args.output)
    no_links(output)
    owned_directory(output.parent)
    require(not os.path.lexists(output), 'Fresh nonexistent output required; preserve partial roots')
    cache = owned_directory(args.cache_dir)
    require(not any(cache.iterdir()), 'Fresh empty cache required')
    require(not output.is_relative_to(cache) and not cache.is_relative_to(output), 'Disjoint output/cache required')
    for protected in (ROOT, OLD_ROOT, CURVE_ROOT):
        require(not output.is_relative_to(protected) and not cache.is_relative_to(protected), 'No frozen/source writes allowed')
    require(os.environ.get('TRITON_CACHE_DIR') == str(cache) and os.environ.get('CUDA_CACHE_DISABLE') == '1',
            'Owned cache environment must already be set')
    require('TRITON_CACHE_MANAGER' not in os.environ, 'Custom cache manager forbidden')
    return output


def validate_operator_gate(receipt, expected_hash):
    require(type(expected_hash) is str and len(expected_hash) == 64, 'Explicit prior terminal SHA required')
    expected = dict(status='CANDIDATE_ANALYTIC_SMOKE_PASSED', passed=True, actual_cuda=True,
        candidate_analytic_smoke_passed=True, passed_scope='CANDIDATE_ANALYTIC_ONLY', completed_cases=112,
        attempted_cases=112, synchronized_cases=112, reference_returned_cases=112, candidate_returned_cases=112,
        numerical_pass=False, quality_gate='NOT_RUN', performance_gate='NOT_RUN', model_gate='NOT_RUN', token_gate='NOT_RUN',
        gpu_uuid=GPU_UUID, torch_version='2.7.1+cu128', torch_cuda_version='12.8', triton_version='3.3.1', tf32=False)
    require(all(type(receipt.get(key)) is type(value) and receipt[key] == value for key, value in expected.items()),
            'Actual complete v3 analytic operator gate required; not a Token/numerical gate')
    pins = receipt.get('source_pins', {})
    require(pins.get('candidate_sha256') == CANDIDATE_SHA and pins.get('gate_sha256') == OPERATOR_GATE_SHA
            and pins.get('contract_version') == 3 and pins.get('fixture_version') == 2,
            'Prior operator source/contract changed')


def prepare(args):
    output = validate_paths(args)
    require(TARGET_ADAPTER_SHA is not None, 'Target adapter not yet frozen; execution refused')
    for path in (args.preregistration, args.prior_seal_root, args.operator_terminal):
        require(path is not None, 'Explicit preregistration/seal/prior operator terminal required')
        no_links(path)
    pre = read(args.preregistration)
    expected = dict(status='FROZEN_BEFORE_EXECUTION', lengths=list(LENGTHS), source_indices=list(DOCS),
        modes=list(MODES), variants=list(VARIANTS), max_new_tokens=128, graph_max_entries=128,
        generation_calls=TOTAL, private_owners=TOTAL, quality_observer=True, performance_measured=False,
        execution_output=str(output), old_root=str(OLD_ROOT), source_root=str(SOURCE_ROOT), curve_root=str(CURVE_ROOT),
        operator_terminal_sha256=args.operator_terminal_sha256, target_adapter_source_sha256=TARGET_ADAPTER_SHA,
        runner_source_sha256=sha(Path(__file__)), candidate_source_sha256=CANDIDATE_SHA)
    require(all(pre.get(key) == value and type(pre.get(key)) is type(value) for key, value in expected.items()),
            'Frozen48-call model gate protocol changed')
    require(sha(args.operator_terminal) == args.operator_terminal_sha256, 'Prior v3 terminal SHA changed')
    gate = read(args.operator_terminal)
    validate_operator_gate(gate, args.operator_terminal_sha256)
    require(sha(args.operator_terminal.parent/'CASES.jsonl') == gate['cases_sha256'], 'Prior112 raw operator cases changed')
    require(len(rows(args.operator_terminal.parent/'CASES.jsonl')) == 112, 'Prior112 raw cases missing')
    for name, digest in (('FINAL_ARTIFACTS_MANIFEST.json', PRIOR_MANIFEST), ('VERIFIED_COMPLETE.json', PRIOR_VERIFIED)):
        require(sha(args.prior_seal_root/name) == digest, 'Complete old curve seal changed')
    seal = read(args.prior_seal_root/'VERIFIED_COMPLETE.json')
    require(seal['passed'] is True and seal['six_lengths_complete'] is True and
            seal['final_artifacts_manifest_sha256'] == PRIOR_MANIFEST, 'Old six-length curve incomplete')
    full = read(args.prior_seal_root/'FINAL_ARTIFACTS_MANIFEST.json')['file_sha256']
    needed = ('results/protocol.json', 'results/COMPLETE.json', 'results/QUALITY_PASSED.json', 'results/REUSE_RECEIPT.json',
        'results/references.json', 'results/documents.jsonl', 'results/prime.json', 'results/runtime.jsonl',
        'results/own_quality.jsonl', 'results/reused_own_quality.jsonl', 'PIPELINE_EXIT.json', 'LAUNCH_MANIFEST.json', 'tools/run_context_curve.py')
    curve_pins = {name: full['terminal/'+name] for name in needed}
    inventory(CURVE_ROOT, curve_pins)
    old = read(CURVE_ROOT/'results/protocol.json')
    complete = read(CURVE_ROOT/'results/COMPLETE.json')
    require(complete['passed'] is True and complete['logical_requests'] == 798 and
            read(CURVE_ROOT/'PIPELINE_EXIT.json')['exit_code'] == 0, 'Frozen798-call execution incomplete')
    require(sha(CURVE_ROOT/'tools/run_context_curve.py') == CURVE_SHA, 'Frozen RuntimeAdapter changed')
    source_pins = old['source_sha256']
    require(len(source_pins) == 167, 'Complete original167-file source inventory required')
    inventory(SOURCE_ROOT, source_pins)
    old_pins = read(CURVE_ROOT/'results/REUSE_RECEIPT.json')['old_artifact_pins']
    inventory(OLD_ROOT, old_pins)
    require(sha(OLD_ROOT/'tools/frozen_devcheck.py') == HELPER_SHA, 'Frozen source import helper changed')
    for role, digest in ASSETS.items():
        require(sha(Path(old['args'][role])) == digest, 'Frozen Mapper/policy/cost changed')
    for path, digest in old['assets_sha256'].items():
        no_links(Path(path))
        require(sha(Path(path)) == digest, 'Frozen full asset inventory changed: '+path)
    external = {'exp/round_overhead/fp32_splitkv_candidate_v1.py': CANDIDATE_SHA,
        'exp/round_overhead/fp32_splitkv_cuda_gate_v3.py': OPERATOR_GATE_SHA,
        'tests/test_fp32_splitkv_cuda_gate_v3.py': OPERATOR_TEST_SHA,
        'exp/round_overhead/fp32_splitkv_target_adapter_v1.py': TARGET_ADAPTER_SHA,
        'tests/test_fp32_splitkv_target_adapter_v1.py': TARGET_ADAPTER_TEST_SHA}
    inventory(ROOT, external)
    documents = {row['source_index']: row for row in rows(CURVE_ROOT/'results/documents.jsonl')}
    references = read(CURVE_ROOT/'results/references.json')
    inputs = {}
    for n in LENGTHS:
        for document in DOCS:
            item = documents[document]
            left, right, body = (item[key] for key in ('prefix_ids', 'suffix_ids', 'document_ids'))
            require(len(left)+len(right) == 30 and len(body) == 16354, 'Natural document shape changed')
            ids = left+body[:n-30]+right
            require(len(ids) == n and hashlib.sha256(json.dumps(ids).encode()).hexdigest() == old['input_sha256'][f'{document}:{n}'],
                    'Old exact input SHA changed')
            ref = references[f'{document}:{n}']
            require(type(ref) is list and 0 < len(ref) <= 128 and all(type(token) is int for token in ref), 'Old canonical tokens invalid')
            inputs[document, n] = ids
    historical = rows(CURVE_ROOT/'results/runtime.jsonl')
    lookup = {}
    for n in LENGTHS:
        for document in DOCS:
            for mode in MODES:
                matches = [row for row in historical if (row['length'], row['source_index'], row['mode']) == (n, document, mode)]
                require(len(matches) == 2 and {row['repeat'] for row in matches} == {0, 1}, 'Frozen historical paired work missing')
                lookup[n, document, mode] = matches[0]
    return dict(old=old, inputs=inputs, references=references, historic=lookup, source_pins=source_pins,
        old_pins=old_pins, curve_pins=curve_pins, assets=old['assets_sha256'], external_pins=external,
        entry_sha256=CURVE_SHA, preregistration_sha256=sha(args.preregistration), runner_sha256=sha(Path(__file__)),
        operator_terminal_sha256=args.operator_terminal_sha256, seen=set())


def load_private(path, name):
    require(name not in sys.modules, 'Fresh private import namespace required')
    module = types.ModuleType(name)
    module.__file__, module.__package__ = str(path), name.rpartition('.')[0]
    sys.modules[name] = module
    try:
        exec(compile(Path(path).read_bytes(), str(path), 'exec', dont_inherit=True), module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def work_difference(left, right):
    changes = {key: dict(original=left.get(key), candidate=right.get(key)) for key in (*WORK, *OPTIONAL)
               if (key in left) != (key in right) or left.get(key) != right.get(key)}
    trajectory = lambda row: [{key: item[key] for key in TRACE_KEYS} for item in row['trace']]
    return dict(work_fields=changes, trajectory_equal=trajectory(left) == trajectory(right),
        observed_rounds_equal=left.get('observed_rounds') == right.get('observed_rounds'),
        queries_equal=left.get('queries') == right.get('queries'), native_t_equal=left.get('native_t') == right.get('native_t'),
        native_s_equal=left.get('native_s') == right.get('native_s'))


def hook_snapshot(group, teacher, student, engine):
    """Authenticate this private group's idle ownership before scope entry."""
    group.idle()
    modules = (teacher.model.layers[-1].self_attn, *(layer.self_attn for layer in student.model.layers))
    require(type(group.modules) is list and tuple(group.modules) == modules and type(group.originals) is list
            and len(group.originals) == len(modules) and group.engines == [engine]
            and not engine.observation.originals, 'Only one authentic idle private hook owner may lease Target')
    return dict(modules=group.modules, module_values=tuple(group.modules), originals=group.originals,
                original_values=tuple(group.originals))


@contextmanager
def target_hook_lease(group, engine, snapshot):
    """Lease only last-Target expected identity, not any model computation.

Entered AFTER enabled scope installs its28 private forwards. Original HookOwners
idle/install/restore checks still execute. Exit restores metadata BEFORE scope
restores the actual bound methods; caller then checks original group.idle().
"""
    originals, old = snapshot['originals'], snapshot['original_values']
    require(group.modules is snapshot['modules'] and tuple(group.modules) == snapshot['module_values']
            and group.originals is originals and tuple(originals) == old and group.engines == [engine]
            and not engine.observation.originals, 'Private hook ownership changed before scope lease')
    private = group.modules[0].forward
    require(private is not old[0] and all(module.forward is original for module, original in
            zip(group.modules[1:], old[1:])), 'Only scoped Target may differ; Draft hooks must remain original')
    originals[0] = private
    try:
        group.idle()
        yield
    finally:
        try:
            # Frozen group.generate normally restores this in its own finally;
            # also restore a partial install failure on this sole owned engine.
            if engine.observation.originals:
                engine.observation.restore()
            group.idle()
            require(group.modules is snapshot['modules'] and tuple(group.modules) == snapshot['module_values']
                    and group.originals is originals and tuple(originals) == (private, *old[1:])
                    and group.engines == [engine], 'Private hook metadata changed during scope lease')
        finally:
            group.originals = originals
            originals[:] = old


def check_scope(summary, variant):
    require(summary.get('state') == 'closed' and summary.get('enabled') is (variant == 'candidate'), 'Scope not restored/closed')
    records = summary.get('records', [])
    if variant == 'original':
        require(summary.get('calls') == summary.get('candidate_calls') == summary.get('original_calls') == 0 and not records,
                'Original owner unexpectedly wrapped Target attention')
        return
    require(summary.get('candidate_calls', 0) > 0 and summary['calls'] == len(records) and len(records) % 28 == 0,
            'Actual28-layer candidate calls required')
    require(all(row.get('outcome') == 'returned' for row in records), 'Target attention did not return')
    for offset in range(0, len(records), 28):
        block = records[offset:offset+28]
        require([row['layer'] for row in block] == list(range(28)) and
                len({(row['action'], row['query_tokens'], row['key_tokens']) for row in block}) == 1,
                'Target attention layer-call coverage changed')
    counts = {layer: sum(row['action'] == 'candidate' and row['layer'] == layer for row in records) for layer in range(28)}
    require(min(counts.values()) > 0 and len(set(counts.values())) == 1 and sum(counts.values()) == summary['candidate_calls'],
            'Candidate must execute equally on all28 Target layers')


def execute_matrix(adapter, prepared, output):
    completed, owners = 0, 0
    live = None
    records_path = output/'QUALITY.jsonl'
    with records_path.open('x') as stream:
        try:
            for n in LENGTHS:
                for document in DOCS:
                    for mode, arm in arms().items():
                        original = None
                        for variant in VARIANTS:
                            owner_id = uuid.uuid4().hex
                            live = adapter.make(arm)
                            try:
                                item = dict(length=n, source_index=document, mode=mode, variant=variant,
                                    owner_id=owner_id, phase='target_token_quality', performance_eligible=False,
                                    observer_enabled=True, profile_enabled=False, execution_index=completed)
                                reference = prepared['references'][f'{document}:{n}']
                                prepared['attempted_calls'] = prepared.get('attempted_calls', 0)+1
                                row, summary = adapter.generate(live, arm, prepared['inputs'][document, n], reference, item, variant)
                                # Preserve the returned raw result BEFORE any comparison can fail.
                                stream.write(json.dumps(dict(row, target_attention_scope=summary), sort_keys=True, allow_nan=False)+'\n')
                                stream.flush(); os.fsync(stream.fileno())
                                completed += 1
                                prepared['completed_calls'] = completed
                                require(row.get('generated') == reference, 'Frozen original canonical Token mismatch')
                                request = row.get('request_id')
                                require(type(request) is str and request not in prepared['seen'], 'Duplicate/missing request identity')
                                prepared['seen'].add(request)
                                adapter.validate(row, reference, arm)
                                check_scope(summary, variant)
                                if variant == 'original':
                                    adapter.compare_historical(prepared['historic'][n, document, mode], row)
                                    original = row
                                else:
                                    write(output/f'DIFF_{n}_{document}_{mode}.json', work_difference(original, row))
                                prepared['validated_calls'] = prepared.get('validated_calls', 0)+1
                                closed = adapter.close(live)
                                live = None
                                require(closed.get('engine_closed') is True and closed.get('hook_restored') is True,
                                        'Quality owner retained Graph/hook state')
                                write(output/f'OWNER_{owner_id}.json', dict(closed, owner_id=owner_id, mode=mode,
                                    variant=variant, length=n, source_index=document, request_id=request, performance_eligible=False))
                                owners += 1
                                prepared['closed_owners'] = owners
                            finally:
                                if live is not None:
                                    closed = adapter.close(live)
                                    live = None
                                    require(closed.get('engine_closed') is True and closed.get('hook_restored') is True,
                                            'Failed request retained its private owner/hooks')
                                    owners += 1
                                    prepared['closed_owners'] = owners
                                    write(output/f'OWNER_{owner_id}.json', dict(closed, owner_id=owner_id, mode=mode,
                                        variant=variant, length=n, source_index=document, failed_request_cleanup=True,
                                        performance_eligible=False))
            require(completed == owners == len(prepared['seen']) == prepared.get('attempted_calls') ==
                    prepared.get('returned_calls') == prepared.get('validated_calls') == TOTAL,
                    'Incomplete48 attempted/returned/validated/closed Token gate')
            require(adapter.actual_cuda is True, 'Actual CUDA generation not established')
            adapter.assert_unchanged()
            return dict(status='TARGET_TOKEN_GATE_PASSED', passed=True, actual_cuda=adapter.actual_cuda, completed_calls=completed,
                attempted_calls=prepared['attempted_calls'], returned_calls=prepared['returned_calls'], validated_calls=prepared['validated_calls'],
                private_owners_closed=owners, frozen_canonical_token_exact=True, token_gate='FROZEN_CANONICAL_TOKEN_EXACT_PASSED',
                quality_scope='Only two already-used documents x six exact contexts x two frozen Graph arms',
                candidate_intermediate_bitwise_claim=False, candidate_trajectory_equal_claim=False,
                performance_gate='NOT_RUN', performance_measured=False, production_speed_claim=False,
                distribution_equivalence_claim=False, fresh_confirmation=False, goal_complete=False)
        finally:
            if live is not None:
                adapter.close(live)


class RuntimeAdapter:
    def __init__(self, args, prepared):
        self.args, self.prepared = args, prepared
        self.actual_cuda = False
        helper = load_private(OLD_ROOT/'tools/frozen_devcheck.py', '_splitkv_model_frozen_helper')
        helper.load_gate(SOURCE_ROOT)
        helper.idle_gpu()
        prepared['helper'] = helper
        curve = load_private(CURVE_ROOT/'tools/run_context_curve.py', '_splitkv_model_frozen_curve')
        self.base = None
        # Keep the adapter at its authentic capsule/exp/round_overhead path.
        sys.path.insert(0, str(ROOT))
        self.external = load_private(ROOT/'exp/round_overhead/fp32_splitkv_target_adapter_v1.py', '_splitkv_target_adapter')
        runtime_args = argparse.Namespace(**vars(args), old_root=OLD_ROOT, source_root=SOURCE_ROOT, curve_root=CURVE_ROOT,
            **{role: Path(prepared['old']['args'][role]) for role in ('teacher', 'student', 'mapper', 'policy', 'cost')})
        triton = importlib.import_module('triton')
        require(triton.__version__ == '3.3.1', 'Pinned Triton3.3.1 required')
        try:
            self.base = curve.RuntimeAdapter(runtime_args, prepared)
        except BaseException:
            self.finish()
            raise

    def make(self, arm):
        return self.base.make(arm, True)

    def generate(self, engine, arm, ids, reference, item, variant):
        base = self.base
        original = tuple(layer.self_attn.forward for layer in base.teacher.model.layers)
        snapshot = hook_snapshot(base.group, base.teacher, base.student, engine)
        scope = self.external.TargetAttentionScope(base.teacher, enabled=variant == 'candidate', cache_dir=self.args.cache_dir)
        def call():
            row, _instrumented_clock = base.generate(engine, arm, ids, reference, item, True)
            self.prepared['returned_calls'] = self.prepared.get('returned_calls', 0)+1
            # Save return BEFORE synchronization, lease/Scope restoration or
            # validation can fail. This is not another generation request.
            write(self.args.output/f'GENERATION_RETURNED_{item["execution_index"]:03d}.json',
                  dict(row=row, target_scope_at_return=scope.stats(), performance_eligible=False))
            return row
        try:
            with base.r.torch.no_grad(), scope:
                if variant == 'candidate':
                    with target_hook_lease(base.group, engine, snapshot):
                        row = call()
                else:
                    row = call()
                base.r.torch.cuda.synchronize(0)
                self.actual_cuda = True
            row.update(all_timing_fields_are_instrumented=True, performance_eligible=False)
            return row, scope.stats()
        finally:
            require(all(layer.self_attn.forward is before for layer, before in zip(base.teacher.model.layers, original)),
                    'Original Target bound forwards were not restored')
            require(base.group.originals is snapshot['originals'] and tuple(base.group.originals) == snapshot['original_values']
                    and base.group.modules is snapshot['modules'] and tuple(base.group.modules) == snapshot['module_values'],
                    'Private group metadata was not restored')
            base.group.idle()

    def validate(self, row, reference, arm):
        self.base.validate_quality(row, reference, arm)
        require(row.get('nonzero_injection') is (arm['family'] == 'rescue'), 'Actual frozen Mapper nonzero evidence changed')
        require('timeline' not in row and 'cost_samples' not in row, 'No performance profiling allowed')

    def compare_historical(self, old, row):
        self.base.common.check_pair(old, row)
        require(all(key in old and key in row and old[key] == row[key] for key in WORK), 'Original historical work changed')
        require(all((key in old) == (key in row) and (key not in old or old[key] == row[key]) for key in OPTIONAL),
                'Original optional historical work changed')

    def close(self, engine):
        return self.base.close(engine)

    def assert_unchanged(self):
        self.base.assert_unchanged()
        inventory(CURVE_ROOT, self.prepared['curve_pins'])
        inventory(OLD_ROOT, self.prepared['old_pins'])
        inventory(ROOT, self.prepared['external_pins'])
        require(sha(Path(__file__)) == self.prepared['runner_sha256'] and
                sha(self.args.operator_terminal) == self.prepared['operator_terminal_sha256'], 'Runner/prior operator receipt changed')

    def finish(self):
        if self.base is not None:
            self.base.finish()


def run(args):
    prepared = prepare(args)
    output = args.output
    output.mkdir(mode=0o700, exist_ok=False)
    adapter = None
    receipt = dict(status='FAIL', passed=False, actual_cuda=False, performance_measured=False,
        performance_gate='NOT_RUN', token_gate='NOT_PASSED', goal_complete=False, preserve_partial=True,
        automatic_retry=False, planned_calls=TOTAL, runner_sha256=prepared['runner_sha256'])
    try:
        write(output/'PLAN.json', plan())
        write(output/'PREREGISTRATION.json', read(args.preregistration))
        write(output/'REUSE_RECEIPT.json', dict(old_root=str(OLD_ROOT), source_root=str(SOURCE_ROOT), curve_root=str(CURVE_ROOT),
            source_sha256=prepared['source_pins'], old_artifact_pins=prepared['old_pins'], curve_artifact_pins=prepared['curve_pins'],
            assets_sha256=prepared['assets'], external_source_sha256=prepared['external_pins'],
            prior_manifest_sha256=PRIOR_MANIFEST, prior_verified_sha256=PRIOR_VERIFIED,
            operator_terminal_sha256=prepared['operator_terminal_sha256'], no_new_ar=True, no_new_corpus=True,
            mapper_policy_cost_unchanged=ASSETS, performance_eligible=False))
        write(output/'INPUTS.json', {f'{document}:{n}': prepared['inputs'][document, n] for n in LENGTHS for document in DOCS})
        write(output/'references.json', {f'{document}:{n}': prepared['references'][f'{document}:{n}'] for n in LENGTHS for document in DOCS})
        adapter = RuntimeAdapter(args, prepared)
        write(output/'RUNTIME.json', dict(environment=adapter.base.environment, policy_metadata=adapter.base.policy_meta,
            source_bindings=adapter.base.imports, tree_source_bindings=adapter.base.tree_bindings,
            flash_attention=adapter.base.flash, target_dtype='float32', draft_dtype='bfloat16', quality_observer=True,
            configs={mode: asdict(adapter.base.legacy.own_config('cuda_graph')) if arm['family'] == 'rescue'
                     else asdict(adapter.base.baseline.linear_config(arm)) for mode, arm in arms().items()},
            maximum_live_owners=1, original_graph_classes=True, performance_eligible=False))
        receipt.update(execute_matrix(adapter, prepared, output))
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error), completed_rows=
            len(rows(output/'QUALITY.jsonl')) if (output/'QUALITY.jsonl').is_file() and (output/'QUALITY.jsonl').stat().st_size else 0)
    finally:
        if adapter is not None:
            receipt.update(actual_cuda=adapter.actual_cuda, completed_calls=prepared.get('completed_calls', 0),
                           private_owners_closed=prepared.get('closed_owners', 0), attempted_calls=prepared.get('attempted_calls', 0),
                           returned_calls=prepared.get('returned_calls', 0), validated_calls=prepared.get('validated_calls', 0))
            try:
                adapter.finish()
            except BaseException as error:
                receipt.update(status='FAIL', passed=False, token_gate='NOT_PASSED', close_error=type(error).__name__+': '+str(error))
        write(output/'TERMINAL.json', receipt)
    return receipt


def fresh_process():
    return not any(name.split('.')[0] in ('torch', 'triton', 'rescue') for name in sys.modules)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-model', action='store_true')
    for name in ('output', 'cache-dir', 'preregistration', 'prior-seal-root', 'operator-terminal'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--operator-terminal-sha256')
    args = parser.parse_args(argv)
    if not args.execute_model:
        print(json.dumps(plan(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    try:
        require(fresh_process(), 'Fresh owned subprocess without Torch/Triton/rescue required')
        receipt = run(args)
    except BaseException as error:
        print('EXECUTION_REFUSED: '+type(error).__name__+': '+str(error), file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
