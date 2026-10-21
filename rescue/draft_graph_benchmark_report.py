"""CPU-only, partial-safe audit of the six-document persistent Draft benchmark.

This is not a benchmark runner. It reads saved evidence, never loads a model,
and uses documents (not requests or repeats) as the statistical unit.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics


MODES = ('baseline_eager', 'resident', 'cuda_graph')
DOCS = (303, 306, 327, 604, 613, 614)
LENGTHS = (4096, 8192)
REPEATS = 3
EXPECTED = dict(reference_requests=12, correctness_requests=36,
                prime_requests=6, timing_requests=108, owners=6)
TRACE_FIELDS = ('round_index', 'prefix', 'cap', 'proposed', 'accepted',
                'terminal', 'workset_tokens')
ASSETS = dict(mapper_sha256='d08889e796ca2527e23241ff34a072ef640e2fdf3fe9a2f9ba5439975af04d9f',
              policy_sha256='3c333a470119c236afcc1308c18592842a535bee0b8f83c4fb4da359ff17eff5',
              cost_sha256='d92c843a84676c0cd6adaf9364bbf7e8c771d56548f1016b5720a3e04f371fa1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def near(left, right):
    return nonnegative(left) and nonnegative(right) and math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-8)


def signature(row):
    return [[r.get(k) for k in TRACE_FIELDS] for r in row.get('trace', [])]


def bootstrap(values, seed=42, draws=10000):
    """Percentile CI for a mean of document-paired values; n=1 has no CI."""
    import numpy as np
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(n=0, mean=None, ci95=None)
    result = dict(n=len(values), mean=float(values.mean()), ci95=None)
    if len(values) >= 2:
        sampled = np.random.default_rng(seed).choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
        result['ci95'] = [float(v) for v in np.percentile(sampled, [2.5, 97.5])]
    return result


def row_errors(row, reference):
    """Audit algorithmic counts, bounded 2/4 control and saved Target tokens."""
    errors = []
    tokens = row.get('tokens')
    if type(row.get('length')) is not int or row['length'] not in LENGTHS:
        return ['Invalid exact input context length']
    if not isinstance(reference, list) or not reference or row.get('generated') != reference or tokens != len(reference) or not (1 <= len(reference) <= 128) or row.get('output_agrees') is not True:
        errors.append('Canonical output/token accounting mismatch')
    trace = row.get('trace')
    if not isinstance(trace, list) or not trace or not all(isinstance(v, dict) for v in trace) or row.get('blocks') != len(trace):
        return errors+['Missing/inconsistent round trace']
    if not all(type(r.get(k)) is int for r in trace for k in ('prefix', 'cap', 'proposed', 'accepted')):
        return errors+['Missing/noninteger round accounting fields']
    if row.get('accepted') != sum(r.get('accepted', 0) for r in trace) or row.get('proposed') != sum(r.get('proposed', 0) for r in trace):
        errors.append('Accepted/proposed totals differ from trace')
    previous = None
    for i, r in enumerate(trace):
        cap, proposed, accepted = (r.get(k) for k in ('cap', 'proposed', 'accepted'))
        prefix = r.get('prefix')
        if not all(type(v) is int for v in (cap, proposed, accepted, prefix)) or not (1 <= proposed <= cap <= 4 and 0 <= accepted <= proposed):
            errors.append('Invalid bounded 2/4 trajectory')
            continue
        if i == 0 and prefix != row.get('length', 0)-1:
            errors.append('First native prefix must be exact input length minus one')
        if previous and prefix != previous['prefix']+previous['accepted']+1:
            errors.append('Nonterminal committed-prefix progression mismatch')
        remaining = 128-(prefix-(row.get('length', 0)-1))
        if cap not in (min(2, remaining), min(4, remaining)):
            errors.append('Depth is not frozen 2/4 with remaining-output clipping')
        if r.get('round_index') != i or r.get('terminal') is not (i == len(trace)-1):
            errors.append('Round index/terminal mismatch')
        previous = r
    if type(tokens) is int and previous and type(previous.get('prefix')) is int:
        committed_before = previous['prefix']-(row.get('length', 0)-1)
        if not (committed_before < tokens <= committed_before+previous.get('accepted', 0)+1):
            errors.append('Terminal commit count does not explain output length')
    for key in ('prefill_seconds', 'decode_seconds', 'total_seconds', 'total_tps', 'decode_tps'):
        if not positive(row.get(key)):
            errors.append('Invalid '+key)
    if not nonnegative(row.get('cleanup_seconds')):
        errors.append('Invalid cleanup_seconds')
    if type(tokens) is int and positive(row.get('total_seconds')) and not near(row.get('total_tps'), tokens/row['total_seconds']):
        errors.append('Engine total TPS mismatch')
    if type(tokens) is int and positive(row.get('decode_seconds')) and not near(row.get('decode_tps'), tokens/row['decode_seconds']):
        errors.append('Engine decode TPS mismatch')
    if positive(row.get('total_seconds')) and all(nonnegative(row.get(k)) for k in ('prefill_seconds', 'decode_seconds', 'cleanup_seconds')) and not near(row['total_seconds'], sum(row[k] for k in ('prefill_seconds', 'decode_seconds', 'cleanup_seconds'))):
        errors.append('Engine stage sum mismatch')
    if type(row.get('draft_forwards')) is not int or row.get('draft_forwards', 0) < row.get('proposed', 0):
        errors.append('Missing/invalid Draft forward accounting')
    return errors


def digest_valid(value):
    return isinstance(value, dict) and isinstance(value.get('dtype'), str) and isinstance(value.get('shape'), list) and all(type(v) is int and v > 0 for v in value['shape']) and isinstance(value.get('sha256'), str) and len(value['sha256']) == 64 and all(c in '0123456789abcdef' for c in value['sha256'])


def quality_errors(row):
    errors = []
    observed = row.get('observed_rounds', [])
    if len(observed) != row.get('blocks'):
        errors.append('Missing full-round selection/features/check evidence')
    nonzero = False
    reconstructed = []
    for i, r in enumerate(observed):
        features = r.get('budget_features')
        if features is None and len(r.get('draft', [])) < 2:
            # EOS/remaining=1 can stop before the second-token controller.
            pass
        elif not isinstance(features, list) or len(features) != 12 or not all(type(v) in (int, float) and math.isfinite(v) for v in features):
            errors.append('Missing/invalid frozen twelve-dimensional features')
        else:
            nonzero |= features[-1] > 0
        selected = r.get('selected')
        if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected) or not all(type(v) is int and 0 <= v < r.get('prefix', 0) for v in selected):
            errors.append('Missing/invalid absolute selected-position evidence')
        if i >= len(row.get('trace', [])):
            continue
        trace = row['trace'][i]
        if any(r.get(k) != trace.get(k) for k in ('round_index', 'prefix', 'accepted')) or len(r.get('draft', [])) != trace.get('proposed') or len(r.get('checks', [])) != trace.get('proposed', -1)+1:
            errors.append('Observed round and acceptance trace disagree')
        draft, checks = r.get('draft', []), r.get('checks', [])
        accepted = 0
        for a, b in zip(draft, checks):
            if a != b:
                break
            accepted += 1
        if accepted != r.get('accepted'):
            errors.append('Acceptance is not the saved token-prefix match length')
        if accepted < len(checks):
            reconstructed.extend(draft[:accepted]+[checks[accepted]])
    if reconstructed[:row.get('tokens', 0)] != row.get('generated'):
        errors.append('Instrumented committed token sequence differs from canonical output')
    if row.get('nonzero_injection') is not True or not nonzero:
        errors.append('Missing actual nonzero Mapper injection evidence')
    queries = row.get('queries', [])
    if not queries or len(queries) != row.get('draft_forwards') or not all(isinstance(v, dict) and type(v.get('logits')) is bool and digest_valid(v.get('query')) for v in queries):
        errors.append('Missing/invalid every-Draft-step Query digest')
    for role in ('native_t', 'native_s'):
        cache = row.get(role, {})
        lengths, layers = cache.get('lengths', []), cache.get('layers', [])
        if len(lengths) != 28 or len(layers) != 28 or not all(type(n) is int and n >= row.get('length', 0)-1 for n in lengths):
            errors.append('Missing all28 valid Native KV layers: '+role)
            continue
        for n, layer in zip(lengths, layers):
            if not isinstance(layer, dict) or any(not digest_valid(layer.get(k)) or len(layer[k]['shape']) != 4 or layer[k]['shape'][2] != n for k in ('key', 'value')):
                errors.append('Invalid valid-span Native KV digest: '+role)
    if row.get('mode') == 'cuda_graph' and not all(row.get('runtime_stats', {}).get(k, 0) > 0 for k in ('captures', 'replays')):
        errors.append('Correctness did not exercise real Graph capture and replay')
    return errors


def request_runtime_errors(row, cold=False):
    errors = []
    if not positive(row.get('outer_seconds')) or not near(row.get('outer_tps'), row.get('tokens', 0)/row['outer_seconds'] if positive(row.get('outer_seconds')) else -1):
        errors.append('Invalid full-generate outer timing/TPS')
    if positive(row.get('total_seconds')) and positive(row.get('outer_seconds')) and row['outer_seconds']+1e-6 < row['total_seconds']:
        errors.append('Outer clock excludes part of generate')
    mode = row.get('mode')
    stats = row.get('draft_runtime', {}).get('request', {})
    if mode == 'cuda_graph':
        if stats.get('replays') != row.get('draft_forwards') or not positive(stats.get('replays')):
            errors.append('Graph replay count does not match Draft forwards')
        if cold:
            if not positive(stats.get('captures')) or not positive(stats.get('capture_host_seconds')):
                errors.append('Cold owner did not pay real capture cost')
        elif stats.get('captures') != 0 or stats.get('capture_host_seconds') != 0:
            errors.append('Warm Graph request recaptured')
    if mode == 'resident' and (stats.get('resident_steps') != row.get('draft_forwards') or stats.get('captures') != 0):
        errors.append('Resident step/capture accounting mismatch')
    return errors


def analyze(directory, *, bootstrap_draws=10000):
    directory = Path(directory)
    problems = []

    def read(name, default):
        try:
            value = json.loads((directory/name).read_text())
            if not isinstance(value, type(default)):
                raise ValueError('Unexpected JSON container')
            return value
        except (OSError, ValueError):
            problems.append('Missing/malformed '+name)
            return default

    def rows(name):
        try:
            values = [json.loads(v) for v in (directory/name).read_text().splitlines() if v.strip()]
            if not all(isinstance(v, dict) for v in values):
                raise ValueError('Nonobject JSONL row')
            return values
        except (OSError, ValueError):
            problems.append('Missing/malformed '+name)
            return []

    protocol = read('protocol.json', {})
    references = read('references.json', {})
    canonical, quality, primes, runtime, owners, contexts = [rows(name) for name in (
        'reference.jsonl', 'correctness.jsonl', 'prime.jsonl', 'runtime.jsonl', 'owners.jsonl', 'contexts.jsonl')]
    if protocol.get('source_indices') != list(DOCS) or protocol.get('lengths') != list(LENGTHS) or protocol.get('max_new_tokens') != 128 or protocol.get('repeats') != REPEATS or protocol.get('modes') != list(MODES) or protocol.get('expected_counts') != EXPECTED:
        problems.append('Unregistered document/context/output/repeat/mode matrix')
    config = protocol.get('config', {})
    fixed = dict(max_new_tokens=128, max_draft=12, attention_backend='sdpa', schedule='serial',
                 working_mode='packed', length_policy='risk', optimized_execution=True,
                 deferred_probes=True, injection_layers=[27], inject_tokens=64, injection_strength=0.5,
                 profile=False, prefill_precision='bfloat16')
    if any(config.get(k) != v for k, v in fixed.items()):
        problems.append('Frozen nonzero Mapper/StepBudget execution configuration differs')
    env = protocol.get('environment', {})
    if 'H100' not in env.get('gpu_name', '') or env.get('total_memory', 0) < 80_000_000_000 or env.get('target_dtype') != 'torch.float32' or env.get('draft_dtype') != 'torch.bfloat16' or env.get('prefix_dtype') != 'torch.bfloat16' or env.get('tf32') is not False:
        problems.append('Unregistered H10080GB/FP32Target/BF16Draft precision environment')
    if protocol.get('no_training') is not True or protocol.get('mapper_frozen') is not True or protocol.get('training_epochs') != 0 or any(protocol.get(k) != v for k, v in ASSETS.items()):
        problems.append('Frozen Mapper/StepBudget/cost assets or no-training declaration mismatch')
    assets = protocol.get('assets_sha256', {})
    for name, key in (('full_final.pt', 'mapper_sha256'), ('budget_selected.pt', 'policy_sha256'), ('cost.json', 'cost_sha256')):
        matches = [digest for path, digest in assets.items() if Path(path).name == name]
        if matches != [ASSETS[key]]:
            problems.append('Frozen asset inventory does not bind '+name)
    source_counts = {}
    for role in ('baseline_eager', 'new'):
        inventory = protocol.get('source_sha256', {}).get(role, {})
        source_counts[role] = len(inventory)
        if not inventory:
            problems.append('Missing frozen '+role+' source inventory')
        source_root = (directory/'source'/role).resolve()
        for name, digest in inventory.items():
            path = (source_root/name).resolve()
            if not path.is_relative_to(source_root) or not path.is_file() or sha(path) != digest:
                problems.append('Frozen source hash mismatch: '+role+'/'+name)
    baseline_sources = protocol.get('source_sha256', {}).get('baseline_eager', {})
    new_sources = protocol.get('source_sha256', {}).get('new', {})
    allowed_changes = {'rescue/engine.py', 'rescue/attention.py', 'rescue/config.py', 'rescue/__main__.py'}
    if any(new_sources.get(name) != digest for name, digest in baseline_sources.items() if name not in allowed_changes):
        problems.append('Unregistered shared algorithm/source change outside Draft runtime integration')
    # The local evidence archive may add an outer immutable manifest after the
    # runner finishes. Verify it when present without requiring the runner to
    # create a self-referential manifest before producing its own analysis.
    manifest_path = directory.parent/'EVIDENCE_MANIFEST.json'
    manifest_count = None
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
            inventory = manifest.get('file_sha256', manifest.get('files', manifest.get('sha256', manifest)))
            if isinstance(inventory, list):
                inventory = {v['path']: v['sha256'] for v in inventory}
            manifest_count = len(inventory)
            root = directory.parent.resolve()
            for name, digest in inventory.items():
                path = (root/name).resolve()
                if not path.is_relative_to(root) or not path.is_file() or sha(path) != digest:
                    problems.append('Archive evidence hash mismatch: '+name)
        except (OSError, ValueError, KeyError, AttributeError, TypeError):
            problems.append('Malformed outer EVIDENCE_MANIFEST.json')

    cells = {(doc, length) for doc in DOCS for length in LENGTHS}
    reference_map, bad_cells = {}, set()
    for row in canonical:
        key = row.get('source_index'), row.get('length')
        tokens = row.get('generated')
        if key not in cells or key in reference_map or row.get('mode') != 'canonical_ar' or not isinstance(tokens, list) or not (1 <= len(tokens) <= 128) or not all(type(v) is int and v >= 0 for v in tokens) or references.get(':'.join(map(str, key))) != tokens:
            problems.append('Missing/duplicate/inconsistent canonical AR cell: '+str(key)); bad_cells.add(key)
        reference_map[key] = tokens
    if set(reference_map) != cells or len(canonical) != 12 or set(references) != {f'{d}:{l}' for d, l in cells}:
        problems.append('Canonical12-cell matrix incomplete')

    qmap, bad_quality = {}, set()
    expected_quality = {(d, l, m) for d, l in cells for m in MODES}
    for row in quality:
        key = row.get('source_index'), row.get('length'), row.get('mode')
        errors = row_errors(row, reference_map.get(key[:2]))+quality_errors(row)
        if key not in expected_quality or key in qmap:
            errors.append('Unexpected/duplicate correctness cell')
        if errors:
            problems.extend('quality '+str(key)+': '+v for v in errors); bad_quality.add(key)
        qmap[key] = row
    for d, l in cells:
        base = qmap.get((d, l, 'baseline_eager'))
        for m in MODES:
            key = d, l, m
            row = qmap.get(key)
            if base is None or row is None:
                bad_quality.add(key); continue
            for field in ('observed_rounds', 'queries', 'native_t', 'native_s'):
                if row.get(field) != base.get(field):
                    problems.append('quality '+str(key)+': '+field+' differs from eager'); bad_quality.add(key)
            if signature(row) != signature(base):
                problems.append('quality '+str(key)+': trajectory differs from eager'); bad_quality.add(key)
    if set(qmap) != expected_quality or len(quality) != 36:
        problems.append('Correctness36-request matrix incomplete')

    owner_map, bad_owners = {}, set()
    expected_owners = {(m, l) for m in MODES for l in LENGTHS}
    for owner in owners:
        key = owner.get('mode'), owner.get('length')
        if key not in expected_owners or key in owner_map or not isinstance(owner.get('owner_id'), str):
            problems.append('Unexpected/duplicate/missing persistent owner'); bad_owners.add(key)
        owner_map[key] = owner
    if set(owner_map) != expected_owners or len(owners) != 6 or len({o.get('owner_id') for o in owners}) != len(owners):
        problems.append('Six independent context/mode owners missing/duplicated')

    prime_map = {}
    for row in primes:
        key = row.get('mode'), row.get('length')
        qkey = 303, row.get('length'), row.get('mode')
        errors = row_errors(row, reference_map.get(qkey[:2]))+request_runtime_errors(row, cold=True)
        if key not in expected_owners or key in prime_map or row.get('source_index') != 303 or row.get('phase') != 'cold_prime':
            errors.append('Unexpected/duplicate cold-prime cell')
        if qkey in bad_quality or signature(row) != signature(qmap.get(qkey, {})):
            errors.append('Cold-prime trajectory differs from quality evidence')
        for field in ('tokens', 'accepted', 'proposed', 'blocks', 'draft_forwards', 'selector_refreshes'):
            if row.get(field) != qmap.get(qkey, {}).get(field):
                errors.append('Cold-prime '+field+' differs from quality evidence')
        owner = owner_map.get(key, {})
        if row.get('owner_id') != owner.get('owner_id') or row.get('request_id') != owner.get('prime_request_id'):
            errors.append('Cold owner/UUID linkage mismatch')
        constructor = owner.get('constructor_and_detach_seconds')
        if not nonnegative(constructor) or not near(row.get('cold_complete_outer_seconds'), constructor+row.get('outer_seconds', 0)):
            errors.append('Cold complete TPS must include constructor and capture')
        if row.get('owner_signature') != owner.get('owner_signature') or not row.get('owner_signature'):
            errors.append('Cold-prime cache/runtime pointer signature missing/different')
        if row.get('mode') == 'cuda_graph':
            required = row.get('required_graph_keys', [])
            actual = row.get('runtime_stats', {}).get('graph_keys', [])
            if not required or not {json.dumps(v, sort_keys=True) for v in required} <= {json.dumps(v, sort_keys=True) for v in actual}:
                errors.append('Registered cold prime did not cover all quality graph keys')
        if errors:
            problems.extend('prime '+str(key)+': '+v for v in errors); bad_owners.add(key)
        prime_map[key] = row
    if set(prime_map) != expected_owners or len(primes) != 6:
        problems.append('Six cold primes missing/duplicated')

    expected_warm = {(d, l, r, m) for d, l in cells for r in range(REPEATS) for m in MODES}
    indexed, invalid = {}, set()
    for row in runtime:
        key = row.get('source_index'), row.get('length'), row.get('repeat'), row.get('mode')
        qkey = key[0], key[1], key[3]
        errors = row_errors(row, reference_map.get(key[:2]))+request_runtime_errors(row)
        if key not in expected_warm or key in indexed or row.get('phase') != 'warm':
            errors.append('Unexpected/duplicate warm timing cell')
        if signature(row) != signature(qmap.get(qkey, {})):
            errors.append('Warm acceptance/depth trajectory differs from correctness')
        for field in ('tokens', 'accepted', 'proposed', 'blocks', 'draft_forwards', 'selector_refreshes'):
            if row.get(field) != qmap.get(qkey, {}).get(field):
                errors.append('Warm '+field+' differs from quality evidence')
        if key[:2] in bad_cells or qkey in bad_quality:
            errors.append('Canonical/correctness evidence for this warm cell is invalid')
        owner = owner_map.get((key[3], key[1]), {})
        if row.get('owner_id') != owner.get('owner_id'):
            errors.append('Warm request changed persistent owner')
        if not row.get('owner_signature') or row.get('owner_signature') != owner.get('owner_signature'):
            errors.append('Warm request cache/runtime pointer signature changed')
        if errors:
            problems.extend('warm '+str(key)+': '+v for v in errors); invalid.add(key)
        indexed[key] = row
    if set(indexed) != expected_warm or len(runtime) != 108:
        problems.append('Registered108 warm timing matrix incomplete')
    actual_order = [{k: r.get(k) for k in ('source_index', 'length', 'repeat', 'mode')} for r in runtime]
    if protocol.get('timing_order') != actual_order:
        problems.append('Warm request order differs from frozen randomized plan')
    for i in range(0, len(runtime), 3):
        block = runtime[i:i+3]
        if len(block) != 3 or len({(r.get('source_index'), r.get('length'), r.get('repeat')) for r in block}) != 1 or {r.get('mode') for r in block} != set(MODES):
            problems.append('Modes are not interleaved within document/context/repeat'); break

    service = {}
    for key, owner in owner_map.items():
        mode, length = key
        group = [r for r in runtime if (r.get('mode'), r.get('length')) == key]
        prime = prime_map.get(key, {})
        errors = []
        if len(group) != 18 or owner.get('warm_request_ids') != [r.get('request_id') for r in group]:
            errors.append('Persistent owner warm UUID sequence is incomplete/inconsistent')
        warm_seconds = sum(r.get('outer_seconds', 0) for r in group)
        prime_seconds = prime.get('outer_seconds', 0)
        constructor = owner.get('constructor_and_detach_seconds', -1)
        close = owner.get('close_seconds', -1)
        active = constructor+prime_seconds+warm_seconds+close
        warm_tokens = sum(r.get('tokens', 0) for r in group)
        prime_tokens = prime.get('tokens', 0)
        for field, target in (('warm_outer_seconds_sum', warm_seconds), ('cold_prime_outer_seconds', prime_seconds),
                              ('accounted_service_seconds', active), ('warm_tokens', warm_tokens), ('prime_tokens', prime_tokens)):
            if not near(owner.get(field), target):
                errors.append('Owner '+field+' accounting mismatch')
        if not nonnegative(constructor) or not nonnegative(close) or not positive(active) or not near(owner.get('accounted_service_tps'), (warm_tokens+prime_tokens)/active if positive(active) else -1) or not positive(owner.get('owner_lifetime_wall_seconds')) or owner.get('owner_lifetime_wall_seconds', 0)+1e-6 < active:
            errors.append('Invalid owner active sum/lifetime/close accounting')
        if mode == 'cuda_graph':
            cold_captures = prime.get('runtime_stats', {}).get('captures')
            previous_replays = prime.get('runtime_stats', {}).get('replays', 0)
            for row in group:
                stats = row.get('runtime_stats', {})
                if stats.get('captures') != cold_captures or stats.get('replays') != previous_replays+row.get('draft_forwards', 0):
                    errors.append('Graph lifetime counters do not prove same-owner reuse without capture')
                if stats.get('graph_keys') != prime.get('runtime_stats', {}).get('graph_keys'):
                    errors.append('Warm Graph key set differs from the cold owner')
                previous_replays = stats.get('replays', 0)
        if owner.get('hook_restored') is not True:
            errors.append('Owner did not restore shared Target hook on close')
        if mode in ('resident', 'cuda_graph') and owner.get('closed_runtime_stats', {}).get('entries') != 0:
            errors.append('Closed runtime retained graph/resident entries')
        # Correctness Graph owners must not seed timing owners; all quality
        # shapes for this context must be covered by its one registered prime.
        if mode == 'cuda_graph':
            union = {json.dumps(key_, sort_keys=True) for q in quality if q.get('mode') == mode and q.get('length') == length for key_ in q.get('runtime_stats', {}).get('graph_keys', [])}
            actual = {json.dumps(v, sort_keys=True) for v in prime.get('runtime_stats', {}).get('graph_keys', [])}
            if not union or not union <= actual:
                errors.append('Cold Graph owner fails the actual correctness graph-key union')
        if errors:
            problems.extend('owner '+str(key)+': '+v for v in errors); bad_owners.add(key)
        if not errors and key not in bad_owners:
            service[f'{mode}:{length}'] = dict(cold_complete_tps=prime_tokens/(constructor+prime_seconds),
                constructor_seconds=constructor, cold_prime_seconds=prime_seconds,
                cold_capture_seconds=prime.get('draft_runtime', {}).get('request', {}).get('capture_host_seconds', 0),
                accounted_service_tps=(prime_tokens+warm_tokens)/active,
                accounted_service_seconds=active, close_seconds=close,
                owner_lifetime_wall_seconds=owner.get('owner_lifetime_wall_seconds'))
    for key in indexed:
        if (key[3], key[1]) in bad_owners:
            invalid.add(key)

    context_map = {}
    for context in contexts:
        length = context.get('length')
        if length not in LENGTHS or length in context_map or not positive(context.get('context_wall_seconds')):
            problems.append('Invalid/duplicate context wall record')
        group = [r for r in primes+runtime if r.get('length') == length]
        if context.get('tokens') != sum(r.get('tokens', 0) for r in group) or context.get('mode_count') != 3:
            problems.append('Context wall token/mode accounting mismatch')
        active = sum(v.get('accounted_service_seconds', 0) for k, v in owner_map.items() if k[1] == length)
        if positive(context.get('context_wall_seconds')) and context['context_wall_seconds']+1e-6 < active:
            problems.append('Context wall shorter than measured active service')
        context_map[length] = context
    if set(context_map) != set(LENGTHS) or len(contexts) != 2:
        problems.append('Two context experiment-wall records missing')
    uuids = [r.get('request_id') for r in canonical+quality+primes+runtime]
    if len(set(uuids)) != len(uuids) or any(not isinstance(v, str) or not v for v in uuids):
        problems.append('Missing/reused request UUID across AR, quality, prime and timing')
    quality_owners = [r.get('owner_id') for r in quality]
    timing_owners = {r.get('owner_id') for r in owners}
    if any(not isinstance(v, str) or not v for v in quality_owners) or len(set(quality_owners)) != len(quality_owners) or set(quality_owners) & timing_owners:
        problems.append('Correctness owners are missing/reused by timing owners')
    receipt = read('COMPLETE.json', {})
    observed_counts = dict(reference_requests=len(canonical), correctness_requests=len(quality),
                           prime_requests=len(primes), timing_requests=len(runtime), owners=len(owners))
    if receipt.get('passed') is not True or receipt.get('counts') != EXPECTED or observed_counts != EXPECTED or receipt.get('assets_unchanged') is not True or receipt.get('sources_unchanged') is not True:
        problems.append('Missing/invalid completion counts or immutable assets/sources receipt')
    if (directory/'FAILURE.json').exists():
        problems.append('Runner FAILURE marker exists')

    # Only an entire three-repeat, three-mode document is plotted. Neither
    # missing pairs nor failed correctness cells are silently imputed.
    per_document = []
    for d, l in sorted(cells):
        keys = [(d, l, r, m) for r in range(REPEATS) for m in MODES]
        if any(k not in indexed or k in invalid for k in keys):
            continue
        for m in MODES:
            group = [indexed[d, l, r, m] for r in range(REPEATS)]
            metric = dict(source_index=d, length=l, mode=m, repeats=REPEATS,
                tokens=sum(r['tokens'] for r in group), outer_seconds=sum(r['outer_seconds'] for r in group),
                outer_tps=sum(r['tokens'] for r in group)/sum(r['outer_seconds'] for r in group),
                accepted_per_round=sum(r['accepted'] for r in group)/sum(r['blocks'] for r in group),
                acceptance_rate=sum(r['accepted'] for r in group)/sum(r['proposed'] for r in group))
            for field in ('prefill_seconds', 'decode_seconds', 'cleanup_seconds', 'total_seconds', 'draft_forwards', 'blocks', 'proposed', 'accepted', 'selector_refreshes', 'working_gathered_position_layers', 'projected_positions', 'selection_similarity_elements'):
                values = [r[field] for r in group if nonnegative(r.get(field))]
                metric[field] = statistics.mean(values) if len(values) == REPEATS else None
            metric['capture_host_seconds'] = statistics.mean(r.get('draft_runtime', {}).get('request', {}).get('capture_host_seconds', 0) for r in group)
            metric['captures'] = sum(r.get('draft_runtime', {}).get('request', {}).get('captures', 0) for r in group)
            metric['replays'] = sum(r.get('draft_runtime', {}).get('request', {}).get('replays', 0) for r in group)
            per_document.append(metric)
    by_context = {}
    for length in LENGTHS:
        data = [r for r in per_document if r['length'] == length]
        by_mode = {m: {r['source_index']: r for r in data if r['mode'] == m} for m in MODES}
        common = sorted(set.intersection(*(set(v) for v in by_mode.values())))
        result = dict(documents=common, n_documents=len(common), modes={}, comparisons={})
        for mode, values in by_mode.items():
            rows_ = [values[d] for d in common]
            result['modes'][mode] = dict(tps=bootstrap([r['outer_tps'] for r in rows_], draws=bootstrap_draws),
                acceptance=bootstrap([r['accepted_per_round'] for r in rows_], draws=bootstrap_draws),
                pooled_tps=sum(r['tokens'] for r in rows_)/sum(r['outer_seconds'] for r in rows_) if rows_ else None)
        for mode, baseline in (('resident', 'baseline_eager'), ('cuda_graph', 'baseline_eager'), ('cuda_graph', 'resident')):
            a, b = by_mode[mode], by_mode[baseline]
            result['comparisons'][mode+'_vs_'+baseline] = dict(
                relative_percent=bootstrap([100*(a[d]['outer_tps']/b[d]['outer_tps']-1) for d in common], draws=bootstrap_draws),
                tps_delta=bootstrap([a[d]['outer_tps']-b[d]['outer_tps'] for d in common], draws=bootstrap_draws),
                acceptance_delta=bootstrap([a[d]['accepted_per_round']-b[d]['accepted_per_round'] for d in common], draws=bootstrap_draws))
        by_context[str(length)] = result
    complete = not problems
    gates = {}
    for length in LENGTHS:
        stat = by_context[str(length)]['comparisons']['cuda_graph_vs_baseline_eager']['relative_percent']
        gates[str(length)] = bool(complete and stat['n'] == 6 and stat['mean'] is not None and stat['mean'] > 0 and stat['ci95'] is not None and stat['ci95'][0] > 0)
    return dict(status='COMPLETE' if complete else 'PARTIAL', problems=problems,
        observed_counts=observed_counts, expected_counts=EXPECTED, contexts=by_context,
        per_document=per_document, service=service, context_wall_records=contexts,
        strict_warm_speed_gate=dict(by_context=gates, passed=all(gates.values()),
            rule='Exactness and complete evidence; cuda_graph versus baseline_eager mean paired warm TPS change and95% document-bootstrap lower bound>0 at BOTH4096 and8192.'),
        source_counts=source_counts, archive_manifest_files=manifest_count,
        bootstrap=dict(unit='document', draws=bootstrap_draws, seed=42, within_document='sum(tokens across3 repeats)/sum(full-generate outer_seconds across3 repeats)'),
        scope='H10080GB;6 reused development documents, NOT held-out; fixed nonzero V-only Mapper and causal2/4 StepBudget,12 is buffer capacity; FP32Target/BF16Draft+prefill, TF32off;128 output limit. No retraining,4090,SpecExtend comparison or independent-generalization claim.',
        raw_sha256={p.name: sha(p) for p in directory.glob('*.json*') if p.is_file()})


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    output = Path(output)
    figures = []
    labels = ['eager', 'resident', 'CUDA Graph']
    title = summary['status']+' · H10080GB · frozen nonzero V Mapper +2/4 StepBudget'
    caption = 'Exact4K/8K input;128 output limit;6 reused dev documents×3 repeats; document-bootstrap95%CI. NOT held-out/4090/SpecExtend.'

    def finish(fig, name, note=caption):
        fig.suptitle(title, fontsize=10)
        fig.text(.5, .015, note, ha='center', fontsize=7.3, wrap=True)
        fig.tight_layout(rect=(0, .075, 1, .94))
        for suffix in ('png', 'svg'):
            fig.savefig(output/f'{name}.{suffix}', dpi=170)
        plt.close(fig); figures.append(name)

    def bars(ax, stats):
        for i, stat in enumerate(stats):
            if stat['mean'] is None:
                continue
            ax.bar(i, stat['mean'], color=['#63778b', '#d4963b', '#32867c'][i])
            if stat['ci95']:
                ax.plot([i, i], stat['ci95'], color='black', linewidth=1.5)
        ax.set_xticks(range(3), labels)

    def legend(ax, size=8):
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=size)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for col, length in enumerate(LENGTHS):
        data = summary['contexts'][str(length)]
        bars(axes[0, col], [data['modes'][m]['acceptance'] for m in MODES])
        axes[0, col].set(title=f'{length} tokens · n={data["n_documents"]}', ylabel='Accepted Draft tokens / verification round')
        for i, mode in enumerate(MODES[1:]):
            stat = data['comparisons'][mode+'_vs_baseline_eager']['acceptance_delta']
            if stat['mean'] is not None:
                axes[1, col].plot(i, stat['mean'], 'o', color='#32867c')
                if stat['ci95']:
                    axes[1, col].plot([i, i], stat['ci95'], color='black')
        axes[1, col].axhline(0, color='gray', linewidth=.8)
        axes[1, col].set(xticks=[0, 1], xticklabels=labels[1:], ylabel='Paired acceptance-length delta vs eager')
    finish(fig, 'acceptance')
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for col, length in enumerate(LENGTHS):
        data = summary['contexts'][str(length)]
        bars(axes[0, col], [data['modes'][m]['tps'] for m in MODES])
        axes[0, col].set(title=f'{length} tokens · n={data["n_documents"]}', ylabel='Mean document full-generate outer TPS')
        for i, mode in enumerate(MODES[1:]):
            stat = data['comparisons'][mode+'_vs_baseline_eager']['relative_percent']
            if stat['mean'] is not None:
                axes[1, col].plot(i, stat['mean'], 'o', color='#32867c')
                if stat['ci95']:
                    axes[1, col].plot([i, i], stat['ci95'], color='black')
        axes[1, col].axhline(0, color='gray', linewidth=.8)
        axes[1, col].set(xticks=[0, 1], xticklabels=labels[1:], ylabel='Paired TPS change vs eager (%)')
    finish(fig, 'warm_tps')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, length in zip(axes, LENGTHS):
        data = [r for r in summary['per_document'] if r['length'] == length]
        for mode in MODES[1:]:
            base = {r['source_index']: r['outer_tps'] for r in data if r['mode'] == MODES[0]}
            group = [r for r in data if r['mode'] == mode]
            ax.plot([str(r['source_index']) for r in group], [100*(r['outer_tps']/base[r['source_index']]-1) for r in group], 'o-', label=mode)
        ax.axhline(0, color='gray', linewidth=.8); ax.set(title=f'{length} input tokens', xlabel='Reused development document', ylabel='Paired warm TPS change (%)'); legend(ax)
    finish(fig, 'per_document_delta')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, length in zip(axes, LENGTHS):
        for offset, field, name, color in ((-.18, 'cold_complete_tps', 'Cold complete (doc303)', '#63778b'),
                                          (.18, 'accounted_service_tps', 'Accounted service (prime+18 warm)', '#32867c')):
            for i, mode in enumerate(MODES):
                value = summary['service'].get(f'{mode}:{length}', {}).get(field)
                if value is not None:
                    ax.bar(i+offset, value, width=.33, color=color, label=name if i == 0 else None)
        ax.set(xticks=range(3), xticklabels=labels, title=f'{length} input tokens', ylabel='Measured throughput\n(tokens/s)'); legend(ax, 6.7)
    finish(fig, 'cold_and_accounted_service', 'Cold: doc303 once, including capture. Accounted: that prime +6 documents×3 warm repeats +constructor/close.\nNon-contiguous service sum, NOT per-mode wall/production TPS. One prime per mode/context; no cold CI.')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, length in zip(axes, LENGTHS):
        for j, field in enumerate(('prefill_seconds', 'decode_seconds', 'cleanup_seconds')):
            for i, mode in enumerate(MODES):
                values = [r[field] for r in summary['per_document'] if r['length'] == length and r['mode'] == mode and r[field] is not None]
                if values:
                    ax.bar(i+(j-1)*.23, statistics.mean(values)*1000, width=.22,
                           color=['#63778b', '#d4963b', '#32867c'][j], label=field.replace('_seconds', '') if i == 0 else None)
        ax.set(xticks=range(3), xticklabels=labels, title=f'{length} tokens', ylabel='Warm request stage\nlatency (ms)'); legend(ax)
    finish(fig, 'stage_latency', caption+' Prefill includes setup; coarse stages do not identify Mapper/selector/verify kernel costs.')
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    for col, length in enumerate(LENGTHS):
        for j, field in enumerate(('constructor_seconds', 'cold_capture_seconds', 'close_seconds')):
            for i, mode in enumerate(MODES):
                value = summary['service'].get(f'{mode}:{length}', {}).get(field)
                if value is not None:
                    axes[0, col].bar(i+(j-1)*.23, value*1000, width=.22,
                                    color=['#63778b', '#d4963b', '#32867c'][j], label=field.replace('_seconds', '') if i == 0 else None)
        axes[0, col].set(xticks=range(3), xticklabels=labels, title=f'{length} tokens', ylabel='Cold/owner intervals\n(ms; nested capture)'); legend(axes[0, col], 7)
        for j, field in enumerate(('draft_forwards', 'blocks', 'proposed', 'accepted')):
            for i, mode in enumerate(MODES):
                values = [r[field] for r in summary['per_document'] if r['length'] == length and r['mode'] == mode and r[field] is not None]
                if values:
                    axes[1, col].bar(i+(j-1.5)*.18, statistics.mean(values), width=.17,
                                    color=['#63778b', '#d4963b', '#32867c', '#9b6b9f'][j], label=field if i == 0 else None)
        axes[1, col].set(xticks=range(3), xticklabels=labels, ylabel='Warm request\nwork count'); legend(axes[1, col], 7)
        for j, field in enumerate(('working_gathered_position_layers', 'projected_positions', 'selection_similarity_elements')):
            for i, mode in enumerate(MODES):
                values = [r[field] for r in summary['per_document'] if r['length'] == length and r['mode'] == mode and r[field] is not None]
                if values:
                    axes[2, col].bar(i+(j-1)*.23, statistics.mean(values), width=.22,
                                    color=['#63778b', '#d4963b', '#32867c'][j], label=field.replace('_', ' ') if i == 0 else None)
        axes[2, col].set(xticks=range(3), xticklabels=labels, ylabel='Warm cache/selection\nwork (log scale)', yscale='log'); legend(axes[2, col], 6.7)
    finish(fig, 'runtime_mechanisms', caption+' Capture is nested inside cold generate, not an additive stage; every eligible warm Graph request has zero captures.')
    return figures


def write_report(directory, output, summary, figures):
    directory, output = Path(directory), Path(output)
    report = ['# '+summary['status']+': persistent Draft resident/CUDA Graph benchmark', '', summary['scope'], '',
        'COMPLETE means the registered data, exactness, immutable evidence and reporting gates are complete; it does not mean the speed hypothesis passed.',
        'Primary metric: full-generate outer seconds, including prefill and per-request hook handoff, excluding shared model loading. Every timed owner cold-primes independently before warm reuse; no correctness-stage Graph is borrowed.',
        'Each document/mode pools actual generated tokens and outer time across all three repeats. Document paired differences are then averaged and percentile-bootstrapped (10,000 draws, seed42); requests/repeats are not independent samples.', '',
        '| Input | Eligible documents | resident vs eager TPS % [95% CI] | Graph vs eager TPS % [95% CI] | Strict Graph warm gate |', '|---|---:|---|---|---|']
    def text_stat(stat):
        if stat['mean'] is None:
            return 'not available'
        ci = stat['ci95']
        return f'{stat["mean"]:+.4f}%'+(f' [{ci[0]:+.4f}, {ci[1]:+.4f}]' if ci else ' [no document CI]')
    for length in LENGTHS:
        data = summary['contexts'][str(length)]
        report.append(f'| {length} | {data["n_documents"]} | '+text_stat(data['comparisons']['resident_vs_baseline_eager']['relative_percent'])+' | '+text_stat(data['comparisons']['cuda_graph_vs_baseline_eager']['relative_percent'])+' | '+str(summary['strict_warm_speed_gate']['by_context'][str(length)])+' |')
    report += ['', 'Both-context preregistered warm speed gate: **'+str(summary['strict_warm_speed_gate']['passed'])+'**. '+summary['strict_warm_speed_gate']['rule'], '',
        '## Cold cost and amortization', '',
        '| Input | Mode | Cold complete TPS | Accounted-service TPS | Constructor ms | Cold capture ms | Close ms |', '|---|---|---:|---:|---:|---:|---:|']
    for key, row in summary['service'].items():
        mode, length = key.split(':')
        report.append(f'| {length} | {mode} | {row["cold_complete_tps"]:.4f} | {row["accounted_service_tps"]:.4f} | {row["constructor_seconds"]*1000:.3f} | {row["cold_capture_seconds"]*1000:.3f} | {row["close_seconds"]*1000:.3f} |')
    report += ['', 'Cold complete TPS includes constructor/detach plus the complete first generate (including capture/warmup/restoration). Capture is nested in generate and is not subtracted.',
        'Accounted-service TPS includes constructor/detach+cold prime+18 warm generate intervals+close. These are non-contiguous measured service intervals, NOT per-mode contiguous wall or production throughput. Owner lifetime contains the other arms and audit I/O; context wall contains all three arms and audit I/O. Neither is an arm speedup estimator.',
        'Cold prime is only doc303 once per mode/context; there is no cold document CI. Six reused development documents are not a new independent held-out sample. No claim against SpecExtend, native full-context SD, Target AR or4090 is made.', '',
        '## Audit', '', 'Observed counts: `'+json.dumps(summary['observed_counts'], sort_keys=True)+'`.',
        'All28 valid Native KV layers, each Draft Query digest, selected positions,12 causal features when two Draft tokens are proposed, token checks,2/4 trajectories, per-token output equality and real nonzero injection are required on the separate correctness matrix. EOS or remaining=1 may stop before a controller vector exists. Timing trajectories and tokens must match that matrix. All warm Graph requests must replay without capture.',
        'Saved source files and optional outer archive manifest are rehashed here. Asset hashes are bound by the frozen protocol and runner immutable-assets receipt; this CPU analysis does not reread unavailable remote model weights or rerun CUDA.', '']
    report += ['- '+v for v in summary['problems']] if summary['problems'] else ['No audit issues. Registered evidence complete; speed conclusion is assessed separately above.']
    report += ['', '## Figures, analysis and raw evidence', '']
    for name in figures:
        report += [f'- [{name}]({name}.png) ([SVG]({name}.svg))']
    report += ['- [Analysis source](analysis_source.py)', '- [Summary](summary.json)']
    for name in ('protocol.json', 'references.json', 'reference.jsonl', 'correctness.jsonl', 'prime.jsonl', 'runtime.jsonl', 'owners.jsonl', 'contexts.jsonl', 'COMPLETE.json', 'FAILURE.json'):
        if (directory/name).exists():
            import os
            report.append(f'- [{name}]({os.path.relpath(directory/name, output)})')
    (output/'REPORT.md').write_text('\n'.join(report)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Fresh analysis output required; do not overwrite any evidence')
    args.output.mkdir(parents=True)
    summary = analyze(args.directory)
    figures = plot(summary, args.output)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    shutil.copyfile(__file__, args.output/'analysis_source.py')
    write_report(args.directory, args.output, summary, figures)
    artifacts = {p.name: sha(p) for p in args.output.iterdir() if p.is_file()}
    (args.output/'ANALYSIS_COMPLETE.json').write_text(json.dumps(dict(status=summary['status'],
        artifacts_sha256=artifacts, inspected_render_required=True,
        scope='CPU saved-record audit and report generation; COMPLETE is not speed success'), indent=2)+'\n')
    print(json.dumps(dict(status=summary['status'], strict_warm_speed_gate=summary['strict_warm_speed_gate'], problems=summary['problems']), indent=2))
    return 0 if summary['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
