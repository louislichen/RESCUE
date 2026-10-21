"""Read-only, partial-safe analysis of the frozen six-input-length curve.

No model, Torch, original corpus, SSH, GPU, or training imports. This is a
development characterization against two fixed representative ports, not a
new best-baseline search, production benchmark, or fresh confirmation gate.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys

LENGTHS = (2048, 4096, 6144, 8192, 12288, 16384)
NEW_LENGTHS = (2048, 6144, 12288, 16384)
DOCS = (211, 213, 220, 226, 233, 236, 246, 248, 253, 271, 272, 288, 303, 306, 327, 604)
MODES = ('rescue_graph', 'retrieval32_k4_graph', 'tree32_n8')
LABELS = {'rescue_graph': 'RESCUE: frozen Mapper + causal 2/4',
          'retrieval32_k4_graph': 'Fixed retrieval Top32 / k4 / Graph port',
          'tree32_n8': 'Fixed complete-tree Top32 / nodes8 port'}
COLORS = dict(zip(MODES, ('#087e8b', '#c46c24', '#7256a2')))
OWNER_MODES = (
    MODES,
    (MODES[1], MODES[2], MODES[0]),
    (MODES[2], MODES[0], MODES[1]),
    (MODES[0], MODES[2], MODES[1]),
    (MODES[2], MODES[1], MODES[0]),
    (MODES[1], MODES[0], MODES[2]),
)
SEED, DRAWS = 20261028, 10000
EXPECTED = dict(reference_requests=64, own_quality_requests=128,
                baseline_quality_requests=12, quality_owners=140, prime_requests=18,
                timing_requests=576, owners=18)
TOTAL = 798
FIGURES = ('tps_gain_retrieval', 'tps', 'acceptance', 'tps_gain_tree',
           'latency', 'captures_cold_service', 'actions_selection', 'gpu_telemetry')
SUPPORT_FILES = {
    'analysis_support.py': 'rescue/baseline_selection_report.py',
    'analysis_horizon_support.py': 'rescue/baseline_horizon_report.py',
    'analysis_fair_support.py': 'rescue/fair_confirmation_report.py',
    'analysis_union_support.py': 'rescue/fair_union_confirmation_report.py',
    'analysis_devcheck_support.py': 'exp/draft_graph/report_frozen_devcheck.py',
}
SUPPORT_PATHS = {}
for saved, relative in SUPPORT_FILES.items():
    source = Path(__file__).with_name(saved)
    if not source.is_file():
        source = Path(__file__).resolve().parents[2] / relative
    SUPPORT_PATHS[saved] = source
    spec = importlib.util.spec_from_file_location(saved[:-3], source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[saved[:-3]] = module
    spec.loader.exec_module(module)
fair = sys.modules['analysis_fair_support']
union = sys.modules['analysis_union_support']
devcheck = sys.modules['analysis_devcheck_support']
ASSETS = dict(fair.ASSETS)
positive, nonnegative, near = fair.positive, fair.nonnegative, fair.near


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def paired_statistics(cells, *, seed=SEED, draws=DRAWS):
    """All points/arms use exactly the same whole-document resample matrix.

    Each already-pooled row includes its source_index. Contexts with no paired
    observations are omitted, not assigned zeros. In partial output the caller
    supplies only the common, completion-defined document intersection.
    """
    import numpy as np
    if seed != SEED or draws != DRAWS:
        raise ValueError('Frozen seed20261028 and10000 draws required')
    result = dict(seed=seed, draws=draws, contexts={}, contrasts=[],
                  document_order=[], document_index_shared_across_all_contexts_and_arms=True,
                  bootstrap_index_sha256=None, formal_success=False,
                  interval='pointwise two-sided95 document percentile bootstrap; no simultaneous claim',
                  estimand='ratio of pooled client tokens to whole-generate outer seconds; repeat pool within document first')
    if not cells:
        return result
    sequences = [tuple(row.get('source_index') for row in rows)
                 for arms in cells.values() for rows in arms.values()]
    if not sequences or len(set(sequences)) != 1:
        raise ValueError('Every context and arm must have identical ordered document identities')
    documents = sequences[0]
    if (not documents or len(set(documents)) != len(documents) or
            any(doc not in DOCS for doc in documents)):
        raise ValueError('Unique registered document identities required')
    n = len(documents)
    index = np.random.default_rng(seed).integers(0, n, size=(draws, n), dtype=np.int64)
    result['document_order'] = list(documents)
    result['bootstrap_index_sha256'] = hashlib.sha256(index.astype('<i8').tobytes()).hexdigest()
    for length in map(str, LENGTHS):
        if length not in cells:
            continue
        arms = cells[length]
        if set(arms) != set(MODES):
            raise ValueError('Exactly the three frozen representative arms required')
        values, samples = {}, {}
        for mode in MODES:
            rows = arms[mode]
            if any(not positive(row.get('tokens')) or not positive(row.get('seconds')) or
                   not nonnegative(row.get('accepted')) or not positive(row.get('blocks')) for row in rows):
                raise ValueError('Finite positive paired token/time/round cells required')
            token, seconds, accepted, blocks = (
                np.asarray([row[field] for row in rows], dtype=float)
                for field in ('tokens', 'seconds', 'accepted', 'blocks'))
            sample = token[index].sum(1) / seconds[index].sum(1)
            quality_sample = accepted[index].sum(1) / blocks[index].sum(1)
            proposed = None if mode == 'tree32_n8' else np.asarray([row['proposed'] for row in rows], dtype=float)
            if proposed is not None and (np.any(proposed <= 0) or np.any(accepted > proposed)):
                raise ValueError('Linear proposed/accepted counts invalid')
            values[mode] = dict(documents=n, tokens=int(token.sum()), seconds=float(seconds.sum()),
                pooled_tps=float(token.sum()/seconds.sum()),
                tps_ci95=None if n < 2 else np.percentile(sample, [2.5, 97.5]).tolist(),
                accepted=int(accepted.sum()), blocks=int(blocks.sum()),
                acceptance_length=float(accepted.sum()/blocks.sum()),
                acceptance_length_ci95=None if n < 2 else np.percentile(quality_sample, [2.5, 97.5]).tolist(),
                proposed=None if proposed is None else int(proposed.sum()),
                acceptance_rate=None if proposed is None else float(accepted.sum()/proposed.sum()),
                acceptance_rate_ci95=None if proposed is None or n < 2 else np.percentile(
                    accepted[index].sum(1)/proposed[index].sum(1), [2.5, 97.5]).tolist(),
                tree_raw_path_counts_including_charged_overshoot=mode == 'tree32_n8')
            samples[mode] = sample
        for baseline in MODES[1:]:
            point = 100*(values[MODES[0]]['pooled_tps']/values[baseline]['pooled_tps']-1)
            sample = 100*(samples[MODES[0]]/samples[baseline]-1)
            result['contrasts'].append(dict(length=int(length), baseline_mode=baseline,
                documents=n, point_percent=point,
                ci95=None if n < 2 else np.percentile(sample, [2.5, 97.5]).tolist(),
                fixed_representative_not_best_at_each_length=True))
        result['contexts'][length] = values
    return result


def pooled_cells(warm, valid_documents_by_context):
    """Completion-only common intersection; never choose using speed or EOS."""
    active = [n for n in map(str, LENGTHS) if valid_documents_by_context.get(n)]
    documents = [doc for doc in DOCS if active and all(doc in valid_documents_by_context[n] for n in active)]
    if not documents:
        return {}, documents
    lookup = {(row['length'], row['mode'], row['source_index'], row['repeat']): row for row in warm}
    cells = {}
    for n in active:
        cells[n] = {}
        for mode in MODES:
            cells[n][mode] = [dict(fair.pool_document([lookup[int(n), mode, doc, repeat]
                                for repeat in (0, 1)]), source_index=doc) for doc in documents]
    return cells, documents


class Evidence(union.Evidence):
    """Consume only regular saved local paths; never dereference remote args."""
    def path(self, name):
        if (not isinstance(name, str) or not name or '\\' in name or '\x00' in name or
                Path(name).is_absolute() or '..' in Path(name).parts or
                Path(name).as_posix() != name):
            raise ValueError('Noncanonical relative saved path')
        original = self.root.absolute()/name
        if any(v.is_symlink() for v in (original, *original.parents)):
            raise ValueError('Symlink saved evidence forbidden')
        return super().path(name)

    def pin(self, name, digest):
        try:
            path = self.path(name)
            ok = (isinstance(digest, str) and len(digest) == 64 and
                  all(c in '0123456789abcdef' for c in digest) and path.is_file() and sha(path) == digest)
        except (OSError, ValueError, TypeError):ok = False
        if not ok:self.problem('Missing/changed/unsafe SHA-pinned saved evidence: '+str(name))
        return ok


def descriptors():
    return {MODES[0]: dict(mode=MODES[0], family='rescue', backend='cuda_graph'),
        MODES[1]: dict(mode=MODES[1], family='retrieval', backend='cuda_graph', top_chunks=32, fixed_k=4),
        MODES[2]: dict(mode=MODES[2], family='full_tree', backend='upstream_flash_hybrid',
                       top_chunks=32, nodes=8, max_depth=10, threshold=.7)}


def canonical(ids, eos, vocab):
    return (isinstance(ids, list) and 0 < len(ids) <= 128 and
        all(type(v) is int and 0 <= v < vocab for v in ids) and
        (len(ids) == 128 or ids[-1] == eos) and
        (eos not in ids or ids[-1] == eos and eos not in ids[:-1]))


def work_errors(left, right, arm):
    fields = ('generated', 'tokens', 'accepted', 'blocks')
    if arm['family'] == 'full_tree':
        fields += ('raw_total_generated', 'generation_overshoot', 'scored_generated_before_eos')
        same = left.get('upstream', {}).get('accept_length_list') == right.get('upstream', {}).get('accept_length_list')
    else:
        fields += ('proposed', 'draft_forwards', 'selector_refreshes', 'working_gather_layers',
                   'working_full_gather_layers', 'working_gathered_position_layers',
                   'working_incremental_binds', 'target_probe_rounds', 'draft_shadow_probes', 'projected_positions', 'rpc_calls')
        same = fair.trace_signature(left) == fair.trace_signature(right)
    return [] if same and all(left.get(k) == right.get(k) for k in fields) else ['Same-input repeated/backend logical trajectory or work counters changed']


RUNTIME = ('captures', 'replays', 'resident_steps', 'shadow_updates', 'capture_host_seconds')


def runtime_errors(row, arm, *, quality=False):
    """Check actual uncropped shape keys and binding-copy/capture ledgers."""
    if arm['family'] == 'full_tree':return []
    errors = []
    signature = row.get('owner_signature', {})
    for name in ('native_t', 'native_s', 'working'):
        cache = signature.get(name, {})
        if any(type(cache.get(k)) is not int or cache[k] <= 0 for k in ('identity', 'key_ptr', 'value_ptr', 'capacity')):
            errors.append('Missing actual private cache identity/pointers/capacity')
        if name != 'working' and cache.get('capacity', 0) < row.get('length', 0)+128:
            errors.append('Native cache capacity cannot hold actual input/output')
    stats = row.get('runtime_stats', {});backend = arm['backend']
    if backend == 'eager':
        if 'runtime' in signature or stats.get('entries') != 0:errors.append('Eager quality owns a hidden Graph runtime')
        return errors
    runtime = row.get('draft_runtime', {});delta = runtime.get('request', {})
    pointers = [signature.get(name, {}).get(k) for name in ('native_s', 'working') for k in ('key_ptr', 'value_ptr')]
    if (signature.get('runtime', {}).get('bound_ptrs') != pointers or runtime.get('mode') != backend or
            stats.get('max_graph_entries') != 128):errors.append('Runtime/backend/bound-pointer ownership differs')
    if arm['family'] == 'retrieval' and (runtime.get('max_graph_entries') != 128 or runtime.get('retain_query') is not quality):
        errors.append('Retrieval actual Graph limit/query ownership differs')
    if any(not nonnegative(delta.get(k)) or k != 'capture_host_seconds' and type(delta[k]) is not int for k in RUNTIME):
        errors.append('Missing/nonfinite runtime delta counters')
    if delta.get('shadow_updates') != row.get('draft_forwards') or delta.get('replays') != row.get('draft_forwards') or delta.get('resident_steps') != 0:
        errors.append('Graph replay/binding-copy work omitted or silently fell back')
    keys = stats.get('graph_keys', [])
    if (not isinstance(keys, list) or any(not isinstance(k, list) or len(k) != 4 or type(k[0]) is not int or
            k[0] <= 0 or type(k[1]) is not bool or type(k[2]) is not bool for k in keys)):
        errors.append('Actual nonpadded Graph shape keys malformed');keys = []
    if (stats.get('entries') != len(keys) or len({json_sha(k) for k in keys}) != len(keys) or
            len(keys) > 128 or stats.get('captures') != len(keys)):
        errors.append('Persistent Graph entries/captures/unique-shape ledger differs')
    required = set()
    for step in row.get('trace', []):
        required.update((step['workset_tokens']+j, True, arm['family'] == 'rescue' and j <= 2) for j in range(1, step['proposed']+1))
        if step['accepted'] == step['proposed'] and not step['terminal']:
            required.add((step['workset_tokens']+step['proposed']+1, False, False))
    if not required <= {tuple(k[:3]) for k in keys}:errors.append('Actual work/logits/summary shapes absent from Graph owner')
    if arm['family'] == 'retrieval' and any(k[2] is not False or k[3] != [[], [], []] for k in keys):
        errors.append('Retrieval Graph silently binds Mapper or summary state')
    if quality or row.get('phase') == 'cold_prime':
        if not positive(delta.get('captures')) or any(not near(delta.get(k), stats.get(k)) for k in RUNTIME):
            errors.append('Fresh quality/timing owner borrowed prior Graph work')
    return errors


def _prior(e, p):
    reuse = e.json('REUSE_RECEIPT.json')
    e.pin('REUSE_RECEIPT.json', p.get('reuse_receipt_sha256'))
    root = reuse.get('prior_local_seal')
    if root != 'prior_local_seal' or reuse.get('prior_consumed_root') != 'prior_consumed' or reuse.get('source_snapshot') != 'source':
        e.problem('Portable minimal prior-proof paths changed')
    args = p.get('args', {})
    for name, field in (('FINAL_ARTIFACTS_MANIFEST.json', 'prior_manifest_sha256'), ('VERIFIED_COMPLETE.json', 'prior_verified_sha256')):
        digest = reuse.get(field)
        e.pin('prior_local_seal/'+name, digest)
        if args.get(field) != digest:e.problem('Prior seal SHA is not bound to frozen entry arguments')
    seal = e.json('prior_local_seal/VERIFIED_COMPLETE.json')
    manifest = e.json('prior_local_seal/FINAL_ARTIFACTS_MANIFEST.json')
    if (seal.get('status') != 'VERIFIED_COMPLETE' or seal.get('passed') is not True or
            seal.get('final_artifacts_manifest_sha256') != reuse.get('prior_manifest_sha256')):
        e.problem('Prior devcheck complete independent seal/manifest linkage absent')
    pins = reuse.get('old_artifact_pins', {})
    inventory = manifest.get('file_sha256', {})
    required = {'results/protocol.json', 'results/COMPLETE.json', 'results/QUALITY_PASSED.json', 'results/GATE_PASSED.json',
        'results/quality_owners.jsonl', 'results/reference.jsonl', 'results/references.json', 'results/own_quality.jsonl',
        'results/data/documents.jsonl', 'results/data/prime.json', 'tools/frozen_devcheck.py', 'LAUNCH_MANIFEST.json',
        'capsule/LAUNCH_MANIFEST.json', 'PIPELINE_EXIT.json', 'GPU_PREFLIGHT.json', 'DEPENDENCY_PREFLIGHT.json'}
    if not required <= set(pins) or any(inventory.get(name) != digest for name, digest in pins.items()):
        e.problem('Consumed prior subset not pinned by complete prior artifact inventory')
    remap = {'results/reference.jsonl': 'reused_reference.jsonl', 'results/own_quality.jsonl': 'reused_own_quality.jsonl',
             'results/data/documents.jsonl': 'documents.jsonl', 'results/data/prime.json': 'prime.json'}
    for name, digest in pins.items():e.pin(remap.get(name, 'prior_consumed/'+name), digest)
    old = e.json('prior_consumed/results/protocol.json')
    done = e.json('prior_consumed/results/COMPLETE.json')
    if (done.get('passed') is not True or done.get('logical_requests') != 130 or done.get('counts') !=
            dict(own_quality_requests=64, quality_owners=64, prime_requests=2, timing_requests=64, owners=2) or
            e.json('prior_consumed/PIPELINE_EXIT.json').get('exit_code') != 0):
        e.problem('Prior complete130 execution/exit proof invalid')
    if (reuse.get('reused_reference_requests') != 32 or reuse.get('reused_own_quality_requests') != 64 or
            reuse.get('new_generation_requests') != TOTAL or reuse.get('no_new_confirmation_access') is not True or
            reuse.get('old_quality_not_new_execution') is not True):e.problem('Prior reuse/new-call accounting differs')
    old_refs = e.json('prior_consumed/results/references.json')
    return old, old_refs, reuse


def _analyze(directory):
    e = Evidence(Path(directory))
    p = e.json('protocol.json')
    prereg = e.json('PREREGISTRATION.json')
    e.pin('PREREGISTRATION.json', p.get('preregistration_sha256'))
    e.pin('CURVE_SOURCE.py', p.get('entry_source_sha256'))
    wanted = dict(schema=1, seed=SEED, source_indices=list(DOCS), lengths=list(LENGTHS),
                  repeats=2, max_new_tokens=128, modes=list(MODES), graph_max_entries=128)
    if any(p.get(k) != value or prereg.get(k) != value for k, value in wanted.items()):
        e.problem('Six exact input lengths /16 fixed old docs / repeats / cap / Graph bound changed')
    if p.get('status') != 'FROZEN_CONTEXT_CURVE' or prereg.get('status') != 'FROZEN_BEFORE_EXECUTION':
        e.problem('Protocol was not frozen before execution')
    if p.get('expected_counts') != EXPECTED or p.get('expected_requests') != TOTAL:
        e.problem('New798-call and140-quality-owner accounting changed')
    if (any(p.get(k) is not True for k in ('no_training', 'no_search', 'no_new_confirmation_access', 'paused_test_untouched')) or
            p.get('fresh_confirmation') is not False or p.get('goal_complete') is not False):
        e.problem('Development-only frozen no-training/no-search/safety contract changed')
    arms = descriptors()
    if p.get('arms') != list(arms.values()):e.problem('Fixed representative three arms changed')
    owner_plan = [dict(length=n, mode=mode) for n, modes in zip(LENGTHS, OWNER_MODES) for mode in modes]
    if p.get('owner_order') != owner_plan:e.problem('Pre-frozen counterbalanced18-owner block order changed')
    for error in devcheck.own_config_errors(p.get('configs', {}).get(MODES[0], {})):e.problem(error)
    retrieval_config = dict(length_policy='fixed', fixed_k=4, max_draft=12, max_new_tokens=128,
        refresh_every=4, inject_tokens=0, injection_strength=0., injection_layers=[], prefill_precision='bfloat16',
        schedule='serial', profile=False, working_mode='packed', optimized_execution=True, deferred_probes=True,
        attention_backend='sdpa', draft_execution='cuda_graph', coverage_execution='eager')
    if any(p.get('configs', {}).get(MODES[1], {}).get(k) != v for k, v in retrieval_config.items()):
        e.problem('Fixed retrieval k4/Graph no-Mapper execution config changed')
    tree_config = dict(nodes=8, top_chunks=32, max_depth=10, threshold=.7, chunk_size=32, refresh_every=4,
        flash_prefill=True, hybrid_tree=True, prefill_precision='bfloat16', mapper=False, controller=False)
    if p.get('configs', {}).get(MODES[2]) != tree_config:e.problem('Fixed complete-tree config changed')
    if any(p.get(k) != v for k, v in ASSETS.items()):e.problem('Frozen Mapper/policy/old-cost asset hashes changed')
    old, old_refs, reuse = _prior(e, p)
    sources = p.get('source_sha256', {})
    if not sources or any(sources.get(k) != v for k, v in old.get('source_sha256', {}).items()):
        e.problem('Scientific source differs from prior nonzero-Mapper study')
    for name, digest in sources.items():e.pin('source/'+name, digest)
    bindings = p.get('source_bindings', {})
    if not {'rescue.engine', 'rescue.query_mapper', 'rescue.step_budget', 'rescue.draft_graph'} <= set(bindings):
        e.problem('Actual imported core source bindings missing')
    for name, binding in bindings.items():
        relative = name.replace('.', '/')+'.py'
        if (not isinstance(binding, dict) or sources.get(relative) != binding.get('sha256') or
                not str(binding.get('path', '')).endswith('/'+relative)):e.problem('Imported runtime binding not source-pinned: '+name)
    if p.get('tree_source_bindings') != old.get('tree_source_bindings') and old.get('tree_source_bindings') is not None:
        e.problem('Tree imported source bindings differ from inherited freeze')
    for name, binding in p.get('tree_source_bindings', {}).items():
        if isinstance(binding, dict) and 'namespace_paths' in binding:
            if not binding['namespace_paths'] or any(not str(v).startswith('exp/specextend_bridgekv/specextend/') for v in binding['namespace_paths']):
                e.problem('Tree namespace binding escaped inherited source root: '+name)
        elif not isinstance(binding, dict) or sources.get(binding.get('path')) != binding.get('sha256'):
            e.problem('Tree imported source binding not source-pinned: '+name)
    if not {'classic.model_classic', 'shared.modeling_llama_kv_target'} <= set(p.get('tree_source_bindings', {})):
        e.problem('Actual complete-tree implementation bindings missing')
    assets = p.get('assets_sha256', {})
    if not assets or any(old.get('assets_sha256', {}).get(k) != v for k, v in assets.items()):
        e.problem('Inherited models/tokenizer/assets receipt differs')
    for role in ('mapper', 'policy', 'cost'):
        if assets.get(p.get('args', {}).get(role)) != ASSETS[role+'_sha256']:e.problem('Frozen asset path/SHA absent: '+role)
    env = p.get('environment', {})
    if (env.get('nvidia_smi_gpu') != old.get('environment', {}).get('nvidia_smi_gpu') or
            'H100' not in env.get('gpu_name', '') or env.get('total_memory', 0) < 80_000_000_000 or
            env.get('target_dtype') != 'torch.float32' or env.get('draft_dtype') != 'torch.bfloat16' or
            env.get('prefix_dtype') != 'torch.bfloat16' or env.get('tf32') is not False):
        e.problem('Same single H10080GB / FP32Target / BF16Draft-prefix / TF32off contract differs')
    if p.get('flash_attention') != old.get('flash_attention'):e.problem('Flash dependency receipt changed')
    for error in _source_export_errors(e):e.problem(error)
    docs = e.rows('documents.jsonl');prime_document = e.json('prime.json')
    if [row.get('source_index') for row in docs] != list(DOCS) or prime_document != next((r for r in docs if r.get('source_index') == 271), None):
        e.problem('Saved old16 cohort order / identical paid-warm271 input changed')
    inputs = {}
    for row in docs:
        doc = row.get('source_index')
        prefix, suffix, body = (row.get(k, []) for k in ('prefix_ids', 'suffix_ids', 'document_ids'))
        if (not all(isinstance(v, list) and all(type(t) is int and t >= 0 for t in v) for v in (prefix, suffix, body)) or
                len(prefix)+len(suffix) != 30 or len(body) != 16354):
            e.problem('Saved natural body16354+prompt30 required; no padding / new Parquet reads');continue
        for n in LENGTHS:
            tokens = prefix+body[:n-30]+suffix
            if len(tokens) != n:e.problem('Input is not exact measured context length')
            inputs[f'{doc}:{n}'] = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
    if p.get('input_sha256') != inputs or reuse.get('canonical_inputs_all96') != inputs:
        e.problem('Actual96 prompt/input SHA map differs from immutable data')
    if any(inputs.get(k) != v for k, v in old.get('input_sha256', {}).items()):e.problem('Old4K/8K exact prompts changed')
    refs = e.json('references.json');newref = e.rows('reference.jsonl');oldref = e.rows('reused_reference.jsonl')
    eos, vocab = p.get('tokenizer_eos_token_id'), p.get('vocab_size')
    if type(eos) is not int or type(vocab) is not int or not 0 <= eos < vocab:e.problem('Tokenizer EOS/vocabulary invalid');eos, vocab = -1, 0
    reference_bad = set()
    for row in oldref+newref:
        doc, n = row.get('source_index'), row.get('length');key = f'{doc}:{n}'
        if refs.get(key) != row.get('generated') or not canonical(row.get('generated'), eos, vocab):
            e.problem('Canonical AR reference/output/EOS invalid: '+key);reference_bad.add((doc, n))
    if (len(oldref) != 32 or {(r.get('source_index'), r.get('length')) for r in oldref} != {(d, n) for n in (4096, 8192) for d in DOCS} or
            any(refs.get(k) != v for k, v in old_refs.items())):e.problem('Reused32 sealed canonical AR references changed')
    if len(newref) != 64 or {(r.get('source_index'), r.get('length')) for r in newref} != {(d, n) for n in NEW_LENGTHS for d in DOCS}:
        e.problem('New64 AR reference matrix incomplete')
    if set(refs) != {f'{d}:{n}' for n in LENGTHS for d in DOCS}:e.problem('Combined96 canonical cells incomplete')
    provenance = e.json('reference_provenance.json')
    expected_provenance = {k:dict(origin='reused' if int(k.split(':')[1]) in (4096, 8192) else 'new', input_sha256=v) for k, v in inputs.items()}
    if (provenance.get('refs') != expected_provenance or provenance.get('new_reference_requests') != 64 or
            provenance.get('reused_reference_requests') != 32 or provenance.get('no_new_prime_ar') is not True or
            provenance.get('reused_rows_sha256') != sha(e.path('reused_reference.jsonl'))):e.problem('New-vs-reused AR provenance differs')
    return _records(e, p, old, refs, newref, reference_bad, arms)


def _source_export_errors(e):
    """Require a read-only collection inventory, without trusting original paths."""
    errors = []
    if (e.root/'EVIDENCE_MANIFEST.json').is_file():
        base, prefix = e, ''
    elif (e.root.parent/'EVIDENCE_MANIFEST.json').is_file():
        base, prefix = Evidence(e.root.parent), e.root.name+'/'
    else:
        return ['Saved raw/source export SHA inventory missing']
    manifest = base.json('EVIDENCE_MANIFEST.json')
    pins = manifest.get('file_sha256', {})
    if not pins:return ['Saved raw/source export SHA inventory missing']
    required = {'protocol.json', 'CURVE_SOURCE.py', 'PREREGISTRATION.json', 'REUSE_RECEIPT.json', 'documents.jsonl', 'prime.json',
        'reused_reference.jsonl', 'reused_own_quality.jsonl', 'prior_local_seal/FINAL_ARTIFACTS_MANIFEST.json', 'prior_local_seal/VERIFIED_COMPLETE.json'}
    for name in ('COMPLETE.json', 'QUALITY_PASSED.json', 'FAILURE.json', 'reference.jsonl', 'references.json', 'reference_provenance.json',
                 'own_quality.jsonl', 'baseline_quality.jsonl', 'quality_owners.jsonl', 'prime.jsonl', 'runtime.jsonl',
                 'owners.jsonl', 'gpu_diagnostics.jsonl', 'tree_raw.jsonl'):
        if (e.root/name).is_file():required.add(name)
    for snapshot in ('source', 'prior_consumed'):
        if (e.root/snapshot).is_dir():
            required.update(path.relative_to(e.root).as_posix() for path in (e.root/snapshot).rglob('*') if path.is_file())
    if not {prefix+name for name in required} <= set(pins):errors.append('Export inventory omits protocol/source/reuse/raw/data/proof')
    for name, digest in pins.items():base.pin(name, digest)
    if base is not e:errors += ['Export: '+error for error in base.problems]
    return errors


def _records(e, p, old, refs, reference, reference_bad, arms):
    metadata_invalid = bool(e.problems)
    own = e.rows('own_quality.jsonl');reused = e.rows('reused_own_quality.jsonl')
    baseline = e.rows('baseline_quality.jsonl');quality_owners = e.rows('quality_owners.jsonl')
    old_owners = e.rows('prior_consumed/results/quality_owners.jsonl')
    prime = e.rows('prime.jsonl');warm = e.rows('runtime.jsonl');owners = e.rows('owners.jsonl')
    telemetry = e.rows('gpu_diagnostics.jsonl', optional=True)
    tree_raw = e.rows('tree_raw.jsonl', optional=True)
    counts = dict(reference_requests=len(reference), own_quality_requests=len(own), baseline_quality_requests=len(baseline),
        quality_owners=len(quality_owners), prime_requests=len(prime), timing_requests=len(warm), owners=len(owners))
    if counts != EXPECTED:e.problem('Complete798 generation matrix /140quality owners /18timing owners not yet present')
    all_requests = reference+own+baseline+prime+warm
    ids, indices = [r.get('request_id') for r in all_requests], [r.get('execution_index') for r in all_requests]
    if (any(not isinstance(v, str) or len(v) != 32 or any(c not in '0123456789abcdef' for c in v) for v in ids) or len(set(ids)) != len(ids) or
            any(type(v) is not int for v in indices) or sorted(indices) != list(range(len(all_requests)))):
        e.problem('New request UUID/execution-index bijection invalid; no raw records may be omitted/reused')
        metadata_invalid = True
    if set(ids) & {r.get('request_id') for r in reused+e.rows('reused_reference.jsonl')}:
        e.problem('New requests reuse historical request IDs');metadata_invalid = True
    for row in all_requests:
        if row.get('gpu_snapshot_before', {}).get('nvidia_smi_gpu') != p.get('environment', {}).get('nvidia_smi_gpu'):
            e.problem('Generation did not record same physical GPU/precision before call');metadata_invalid = True
    for row in reference:
        if row.get('phase') != 'reference' or row.get('mode') != 'canonical_ar' or row.get('performance_eligible') is not False:
            e.problem('New AR phase/mode/timing eligibility changed');metadata_invalid = True
    qmap, bad = {}, set(reference_bad)
    qo = {row.get('request_id'): row for row in quality_owners}
    old_qo = {row.get('request_id'): row for row in old_owners}
    seen_owner_ids = [row.get('owner_id') for row in quality_owners+owners]
    if any(not isinstance(v, str) or not v for v in seen_owner_ids) or len(set(seen_owner_ids)) != len(seen_owner_ids):
        e.problem('Private quality/timing owner IDs missing or reused');metadata_invalid = True
    if len(qo) != len(quality_owners) or len(old_qo) != 64:e.problem('Quality owner/request bijection incomplete')
    for rows, historical in ((own, False), (reused, True), (baseline, False)):
        for row in rows:
            doc, n, mode = row.get('source_index'), row.get('length'), row.get('mode')
            is_own = mode in ('rescue_eager', MODES[0]);is_graph = mode in (MODES[0], MODES[1])
            key = doc, n, mode;errors = []
            family = 'rescue' if is_own else 'retrieval';expected_mode = MODES[0] if is_own else MODES[1]
            arm = dict(arms[expected_mode], mode=mode, backend='cuda_graph' if is_graph else 'eager')
            expected_n = (4096, 8192) if historical else NEW_LENGTHS if is_own else LENGTHS
            if (key in qmap or doc not in (DOCS if is_own else (211,)) or n not in expected_n or
                    mode not in ('rescue_eager', MODES[0], 'retrieval32_k4_eager', MODES[1]) or row.get('arm') != arm):
                errors.append('Unregistered or duplicate own/retrieval quality cell')
            qmap[key] = row
            if (row.get('phase') != ('own_quality' if is_own else 'baseline_quality') or row.get('performance_eligible') is not False or
                    row.get('generated') != refs.get(f'{doc}:{n}') or row.get('output_agrees') is not True or
                    row.get('tokens') != len(refs.get(f'{doc}:{n}', []))):errors.append('Quality is not exact untimed canonical output')
            errors += fair.observation_errors(row)+fair.quality_pair_errors(row, row)+fair.linear_trace_errors(row, own=is_own, fixed_k=None if is_own else 4)
            errors += runtime_errors(row, arm, quality=True)
            if is_own:errors += fair.injection_errors(row)
            if is_graph and (row.get('runtime_stats', {}).get('max_graph_entries') != 128 or
                    row.get('draft_runtime', {}).get('request', {}).get('replays') != row.get('draft_forwards') or
                    not positive(row.get('draft_runtime', {}).get('request', {}).get('captures'))):
                errors.append('Fresh private quality Graph128 capture/replay missing')
            owner = (old_qo if historical else qo).get(row.get('request_id'), {})
            if (owner.get('owner_id') != row.get('owner_id') or owner.get('engine_closed') is not True or
                    owner.get('hook_restored') is not True or owner.get('closed_runtime_stats', {}).get('entries') != 0 or
                    owner.get('final_runtime_stats') != row.get('runtime_stats')):errors.append('Quality owner not completely closed with matching final snapshot')
            for error in errors:e.problem(f'Quality{doc}/{n}/{mode}: '+error)
            if errors:bad.add((doc, n))
    for n in LENGTHS:
        for doc in DOCS:
            for eager_mode, graph_mode in (('rescue_eager', MODES[0]), ('retrieval32_k4_eager', MODES[1])):
                if graph_mode == MODES[1] and doc != 211:continue
                eager, graph = qmap.get((doc, n, eager_mode)), qmap.get((doc, n, graph_mode))
                errors = fair.quality_pair_errors(eager, graph)+work_errors(eager, graph, arms[graph_mode]) if eager is not None and graph is not None else ['Missing pre-frozen eager/Graph quality pair']
                if errors:
                    bad.add((doc, n))
                    for error in errors:e.problem(f'Backend{doc}/{n}/{graph_mode}: '+error)
    if len(reused) != 64:e.problem('All64 historical own quality records required, not new generation')
    gate = e.json('QUALITY_PASSED.json')
    if (gate.get('passed') is not True or gate.get('all96_own_pairs') is not True or
            gate.get('new_own_pairs') != 64 or gate.get('reused_own_pairs') != 32 or
            gate.get('quality_owners_closed') is not True or gate.get('nonzero_mapper') is not True):
        e.problem('Untimed all96 own backend pairs /6retrieval sentinel pairs gate absent')
    quality_count = 64+128+12
    qindices = sorted(row.get('execution_index', -1) for row in own+baseline)
    if (len(reference) == 64 and sorted(r.get('execution_index', -1) for r in reference) != list(range(64)) or
            len(own)+len(baseline) == 140 and qindices != list(range(64, quality_count))):
        e.problem('All AR then140 closed quality requests must precede timed owners');metadata_invalid = True
    # Every observed prefix of an owner block is checked. A partially written
    # block can contribute only complete two-repeat, three-arm document pairs.
    wm, pm, om = {}, {}, {row.get('owner_id'): row for row in owners}
    rawmap = {row.get('request_id'): row for row in tree_raw}
    if len(rawmap) != len(tree_raw):e.problem('Duplicate raw tree request IDs');metadata_invalid = True
    observed_tree = 0
    repeated = {}
    for phase, records in (('cold_prime', prime), ('warm', warm)):
        for row in records:
            n, doc, mode = row.get('length'), row.get('source_index'), row.get('mode')
            errors = []
            arm = arms.get(mode)
            if arm is None or n not in LENGTHS:errors.append('Unregistered timed arm/context');arm = arms[MODES[0]]
            errors += fair.timing_errors(row, refs.get(f'{doc}:{n}'), arm, MODES[0])
            errors += runtime_errors(row, arm)
            if (row.get('arm') != arm or row.get('phase') != phase or
                    any(row.get(k) is not False for k in ('observer_enabled', 'profile_enabled')) or
                    any(row.get(k) is not (mode == MODES[0]) for k in ('mapper_executed', 'controller_executed'))):
                errors.append('Primary method/execution/observer/Mapper flags differ')
            if phase == 'warm':
                key = n, mode, doc, row.get('repeat')
                if key in wm or doc not in DOCS or row.get('repeat') not in (0, 1):errors.append('Warm matrix duplicate/unregistered repeat')
                wm[key] = row
                quality = qmap.get((doc, n, mode))
                if (mode == MODES[0] or mode == MODES[1] and doc == 211) and quality is not None:
                    errors += work_errors(quality, row, arm)
            else:
                key = n, mode
                if key in pm or doc != 271:errors.append('Each private timing owner requires one separately billed old271 prime')
                pm[key] = row
                if (not near(row.get('cold_complete_outer_seconds'), row.get('constructor_and_detach_seconds', -1)+row.get('outer_seconds', -1))):
                    errors.append('Cold constructor+generate complete interval differs')
                if arm.get('backend') == 'cuda_graph':
                    delta, life = row.get('draft_runtime', {}).get('request', {}), row.get('runtime_stats', {})
                    if not positive(delta.get('captures')) or any(not near(delta.get(k), life.get(k)) for k in ('captures', 'replays', 'resident_steps', 'capture_host_seconds')):
                        errors.append('Cold timing owner borrowed quality graphs or lifetime work')
            same_input = mode, doc, n
            if same_input in repeated:errors += work_errors(repeated[same_input], row, arm)
            repeated[same_input] = row
            if mode == 'tree32_n8':
                observed_tree += 1
                raw = rawmap.get(row.get('request_id'), {});result = raw.get('result', {})
                if (raw.get('execution_index') != row.get('execution_index') or raw.get('outer_seconds') != row.get('outer_seconds') or
                        any(raw.get(k) != row.get(k) for k in ('phase', 'mode', 'length', 'source_index', 'owner_id')) or
                        result.get('upstream') != row.get('upstream') or result.get('generated') != row.get('wrapper_generated') or
                        result.get('tokens') != row.get('wrapper_tokens')):errors.append('Normalized tree row not bound to separately saved raw overshoot/output interval')
            for error in errors:e.problem(f'{phase}{doc}/{n}/{mode}: '+error)
            if errors:bad.add((doc, n))
    if observed_tree != len(tree_raw):e.problem('Raw tree requests omitted or extra')
    sequence = p.get('shared_warm_sequence', [])
    if (len(sequence) != 32 or {(r.get('source_index'), r.get('repeat')) for r in sequence} != {(d, r) for d in DOCS for r in (0, 1)} or
            sequence != old.get('shared_warm_sequence')):e.problem('Historical shared32 request order changed');metadata_invalid = True
    plan = [dict(length=n, mode=mode) for n, modes in zip(LENGTHS, OWNER_MODES) for mode in modes]
    if [dict(length=r.get('length'), mode=r.get('mode')) for r in prime] != plan[:len(prime)]:
        e.problem('Observed owner blocks are not the pre-frozen counterbalanced prefix');metadata_invalid = True
    for j, cold in enumerate(prime):
        block = [r for r in warm if r.get('owner_id') == cold.get('owner_id')]
        owner = om.get(cold.get('owner_id'))
        if ([{k:r.get(k) for k in ('source_index', 'repeat')} for r in block] != sequence[:len(block)] or
                [r.get('warm_index') for r in block] != list(range(len(block))) or
                cold.get('execution_index') != quality_count+33*j or
                [r.get('execution_index') for r in block] != list(range(quality_count+33*j+1, quality_count+33*j+1+len(block)))):
            e.problem('Each owner must execute cold then original32 contiguous warm order');metadata_invalid = True
        if owner is not None:
            for error in fair.lifecycle_errors(owner, [cold, *block]):e.problem('Owner: '+error);metadata_invalid = True
            if owner.get('maximum_live_owners') != 1:e.problem('Private serving owners overlapped');metadata_invalid = True
            if (owner.get('prime_tokens') != cold.get('tokens') or owner.get('accounted_service_tokens') != owner.get('warm_tokens', 0)+cold.get('tokens', 0) or
                    not near(owner.get('cold_prime_outer_seconds'), cold.get('outer_seconds')) or
                    not near(owner.get('constructor_and_detach_seconds'), cold.get('constructor_and_detach_seconds')) or
                    not nonnegative(owner.get('owner_lifetime_wall_seconds')) or
                    owner.get('owner_lifetime_wall_seconds', -1)+1e-8 < owner.get('accounted_service_seconds', 0)):
                e.problem('Owner cold/tokens/wall-lifetime accounting differs');metadata_invalid = True
            if cold.get('mode') != MODES[2]:
                totals = dict.fromkeys(RUNTIME, 0);previous_keys = set()
                for row in [cold, *block]:
                    current = {json_sha(key) for key in row.get('runtime_stats', {}).get('graph_keys', [])}
                    delta = row.get('draft_runtime', {}).get('request', {})
                    if not previous_keys <= current or len(current-previous_keys) != delta.get('captures'):
                        e.problem('Persistent Graph keys evicted/reset or late captures omitted');metadata_invalid = True
                    previous_keys = current
                    for name in RUNTIME:
                        totals[name] += delta.get(name, 0)
                        if not near(row.get('runtime_stats', {}).get(name), totals[name]) or not near(row.get('draft_runtime', {}).get('lifetime', {}).get(name), totals[name]):
                            e.problem('Cold/late-capture lifetime runtime ledger differs');metadata_invalid = True
        elif len(block) == 32:e.problem('Finished32-warm owner has no closing ledger')
    expected_telemetry = [(phase, row['length'], row['mode'], quality_count+33*j+(33 if phase == 'owner_end' else 0))
        for j, row in enumerate(plan) for phase in ('owner_start', 'owner_end')]
    if [(r.get('phase'), r.get('length'), r.get('mode'), r.get('execution_index')) for r in telemetry] != expected_telemetry[:len(telemetry)]:
        e.problem('Telemetry owner-boundary order differs');metadata_invalid = True
    if len(telemetry) != 36:e.problem('Complete36 outside-clock owner-boundary diagnostics required')
    valid_docs = {}
    for n in LENGTHS:
        valid_docs[str(n)] = [doc for doc in DOCS if (doc, n) not in bad and not metadata_invalid and
            all((n, mode, doc, repeat) in wm for mode in MODES for repeat in (0, 1)) and
            all((n, mode) in pm for mode in MODES)]
    cells, common_docs = pooled_cells(warm, valid_docs)
    done = e.json('COMPLETE.json')
    if (done.get('passed') is not True or done.get('counts') != EXPECTED or done.get('expected_counts') != EXPECTED or
            done.get('logical_requests') != TOTAL or done.get('goal_complete') is not False or done.get('fresh_confirmation') is not False or
            any(done.get(k) is not True for k in ('all_canonical', 'quality_gate_passed', 'owners_closed', 'sources_unchanged', 'assets_unchanged', 'dependencies_unchanged'))):
        e.problem('Complete immutable798-call terminal receipt missing/invalid')
    if (e.root/'FAILURE.json').exists():e.problem('Experiment failed; only preserved completed matched subset may be shown')
    actions, selected = {}, {}
    for n in LENGTHS:
        counts0 = {}
        for row in warm:
            if row.get('length') == n and row.get('mode') == MODES[0]:
                for step in row.get('trace', []):counts0[str(step.get('cap'))] = counts0.get(str(step.get('cap')), 0)+1
        actions[str(n)] = counts0
        hist = None
        for row in own+reused:
            if row.get('length') == n and row.get('mode') == MODES[0]:
                if hist is None:hist = [0]*20
                for step in row.get('observed_rounds', []):
                    for pos in step.get('selected', []):
                        if type(pos) is int and 0 <= pos < step.get('prefix', 0):hist[min(19, int(20*pos/step['prefix']))] += 1
        selected[str(n)] = hist
    return dict(schema=1, status='COMPLETE' if not e.problems else 'PARTIAL', goal_complete=False,
        fresh_confirmation=False, synthetic_program_test_only=p.get('synthetic_program_test_only') is True,
        problems=e.problems, statistics=paired_statistics(cells), per_document_cells=cells,
        paired_complete_documents_by_context=valid_docs, common_completion_only_documents=common_docs,
        observed_counts=counts, expected_counts=EXPECTED, expected_requests=TOTAL, observed_requests=len(all_requests),
        reused_reference_requests=32, reused_own_quality_requests=64, generated_reference_requests=len(reference),
        source_indices=list(DOCS), diagnostics=dict(warm=warm, prime=prime, owners=owners, gpu_telemetry=telemetry,
            actions=actions, quality_selected_position_histogram=selected),
        raw_sha256={path.name:sha(path) for path in e.root.glob('*.json*') if path.is_file()},
        limitations=[
            'Already-used16 development documents; same original corpus12 train +4 previously reused validation documents. Not fresh confirmation.',
            'Frozen Mapper, causal2/4 policy and old cost are not retuned; these are fixed representative ports, not best-per-length controls.',
            'Counterbalanced private owner blocks are not interleaved or concurrent. Document intervals do not cover owner-block or hardware-clock drift.',
            'All32 warm requests follow one paid cold prime per owner. Every legal lazy Graph capture and full-tree overshoot is charged; not production/fully-prewarmed throughput.',
            'Acceptance is accepted Draft/path tokens per round. Tree proposals/unexposed tails are unavailable; no tree node acceptance rate is invented.',
            'Prior full study seal is authenticated by externally frozen argument SHA and consumed-file manifest subset. This report rehashes only the saved consumed prior subset, not the entire prior archive.',
            'Uses shared frozen record validators, not an independent raw audit or newly executed tensor equivalence proof.',
            'Partial primary curves use the common completion-defined document intersection across observed three-arm contexts, with no outcome-based selection or missing-value imputation.',
            'The original4K/8K retained12-opponent final acceptance criteria are unchanged; these descriptive pointwise CIs cannot satisfy that gate.'])


def analyze(directory):
    try:return _analyze(directory)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, OverflowError) as error:
        return dict(schema=1, status='PARTIAL', goal_complete=False, fresh_confirmation=False,
            problems=['Malformed/missing saved evidence: '+type(error).__name__+': '+str(error)],
            statistics={}, observed_counts={}, expected_counts=EXPECTED, expected_requests=TOTAL)


def figures(summary, output):
    """Only actual paired cells are plotted, with gaps instead of imputation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    output = Path(output)
    stats = summary.get('statistics', {})
    contexts = stats.get('contexts', {})
    status = 'PARTIAL — ' if summary.get('status') != 'COMPLETE' else ''
    if summary.get('synthetic_program_test_only'):
        status = 'SYNTHETIC PROGRAM TEST — ' + status
    n = len(stats.get('document_order', []))
    caption = (f'Already-used development docs; common paired n={n}/16; 2 repeats/document; output cap128/EOS. '
        'Single H10080GB; Qwen2-7B/1.5B; Mapper, policy and old cost frozen.\n'
        'Pointwise paired95% document bootstrap; owner blocks are not concurrent, CIs do not cover clock drift. '
        'Fixed representative ports, not length-wise best / official original / production / fresh confirmation.')
    def save(fig, name, title):
        fig.suptitle(status+title, fontsize=12)
        extra = ('\nAcceptance: tree raw path counts include charged overshoot; NOT node acceptance rate.'
                 if name == 'acceptance' else '')
        fig.text(.5, .012, caption+extra, ha='center', fontsize=7)
        for ax in fig.axes:
            finite = any(np.isfinite(np.asarray(line.get_ydata(), dtype=float)).any() for line in ax.lines)
            if not finite and not ax.images and not ax.collections:
                ax.text(.5, .5, 'No saved observations', ha='center', va='center', transform=ax.transAxes, fontsize=9)
        fig.tight_layout(rect=(0, .105, 1, .93))
        fig.savefig(output/(name+'.png'), dpi=160)
        fig.savefig(output/(name+'.svg'))
        plt.close(fig)
    def axis(ax, ylabel):
        ax.set_xticks([v/1024 for v in LENGTHS], [f'{v//1024}K\n{v}' for v in LENGTHS])
        ax.set_xlabel('Actual input context length (tokens; K =1024)')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.22)
    def measured_line(ax, values, cis, *, color, label):
        # NaNs break segments over missing registered contexts, not fabricated points.
        xs = np.asarray(LENGTHS)/1024
        ys = [values.get(n, math.nan) for n in LENGTHS]
        ax.plot(xs, ys, 'o-', color=color, label=label, linewidth=1.7)
        for x, length in zip(xs, LENGTHS):
            ci = cis.get(length)
            if ci is not None:
                ax.vlines(x, ci[0], ci[1], color=color)
                ax.hlines(ci, x-.09, x+.09, color=color)
    for baseline, stem in ((MODES[1], 'tps_gain_retrieval'), (MODES[2], 'tps_gain_tree')):
        rows = [r for r in stats.get('contrasts', []) if r['baseline_mode'] == baseline]
        fig, ax = plt.subplots(figsize=(11, 5.2))
        measured_line(ax, {r['length']: r['point_percent'] for r in rows},
                      {r['length']: r.get('ci95') for r in rows}, color=COLORS[MODES[0]],
                      label='RESCUE relative to '+LABELS[baseline])
        ax.axhline(0, color='black', linewidth=.9)
        axis(ax, 'Relative whole-generate TPS difference (%)')
        ax.legend(fontsize=8)
        save(fig, stem, 'Input-length TPS benefit against fixed '+('retrieval Graph' if baseline == MODES[1] else 'complete-tree')+' port')
    for metric, interval, stem, label in (
            ('pooled_tps', 'tps_ci95', 'tps', 'Whole-generate client tokens / second'),
            ('acceptance_length', 'acceptance_length_ci95', 'acceptance',
             'Accepted Draft / path tokens\nper verification round')):
        fig, ax = plt.subplots(figsize=(11, 5.7))
        for mode in MODES:
            measured_line(ax, {int(n): a[mode][metric] for n, a in contexts.items()},
                          {int(n): a[mode].get(interval) for n, a in contexts.items()},
                          color=COLORS[mode], label=LABELS[mode])
        axis(ax, label);ax.set_ylim(bottom=0);ax.legend(fontsize=8)
        save(fig, stem, 'Measured '+('system throughput' if stem == 'tps' else 'acceptance, separately from speed')+' vs input context')
    diag = summary.get('diagnostics', {})
    warm, primes, owners = (diag.get(k, []) for k in ('warm', 'prime', 'owners'))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for ax, key, label in zip(axes, ('prefill_seconds', 'decode_seconds', 'blocks'),
            ('Saved engine prefill seconds / request', 'Whole decode seconds / verification round\nNOT isolated Target kernel latency', 'Mean verification rounds / request')):
        for mode in MODES:
            values = {}
            for n0 in LENGTHS:
                rows = [r for r in warm if r.get('mode') == mode and r.get('length') == n0]
                values0 = [r.get(key) for r in rows]
                if values0 and all(nonnegative(v) for v in values0):
                    values[n0] = sum(values0)/(sum(r['blocks'] for r in rows) if key == 'decode_seconds' else len(rows))
            measured_line(ax, values, {}, color=COLORS[mode], label=mode)
        axis(ax, label);ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=7)
    save(fig, 'latency', 'Saved stage / round diagnostics (all observed timing records; partial counts disclosed)')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    specs = (('Warm new Graph captures (count)', 'captures'),
             ('Warm capture host seconds', 'capture_host_seconds'),
             ('Cold constructor + whole prime generate (s)', 'cold_complete_outer_seconds'),
             ('Accounted service TPS\nconstructor + paid prime +32 warm + close', 'accounted_service_tps'))
    for ax, (label, key) in zip(axes.flat, specs):
        for mode in MODES:
            values = {}
            for n0 in LENGTHS:
                if key in ('captures', 'capture_host_seconds'):
                    rows = [r for r in warm if r.get('mode') == mode and r.get('length') == n0]
                    observed = [r.get('draft_runtime', {}).get('request', {}).get(key) for r in rows]
                    if mode == 'tree32_n8':
                        continue  # No invented zero Graph statistics for the tree engine.
                    if observed and all(nonnegative(v) for v in observed):values[n0] = sum(observed)
                else:
                    rows = [r for r in (primes if key.startswith('cold_') else owners)
                            if r.get('mode') == mode and r.get('length') == n0]
                    if len(rows) == 1 and nonnegative(rows[0].get(key)):values[n0] = rows[0][key]
            measured_line(ax, values, {}, color=COLORS[mode], label=mode)
        axis(ax, label);ax.set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=7)
    save(fig, 'captures_cold_service', 'Captured work, paid cold start and finite-trace service (not primary warm TPS)')
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    actions = diag.get('actions', {})
    for cap, color in (('1', '#bbbbbb'), ('2', COLORS[MODES[0]]), ('3', '#c8a77d'), ('4', COLORS[MODES[1]])):
        values = {n0: actions[str(n0)][cap] for n0 in LENGTHS if cap in actions.get(str(n0), {})}
        measured_line(axes[0], values, {}, color=color, label='Actual cap '+cap)
    axis(axes[0], 'Observed own timing verification rounds');axes[0].legend(fontsize=8)
    selected = diag.get('quality_selected_position_histogram', {})
    for n0 in LENGTHS:
        hist = selected.get(str(n0))
        if hist is not None and sum(hist):
            axes[1].plot((np.arange(20)+.5)/20, np.asarray(hist)/sum(hist), label=f'{n0} tokens')
    axes[1].set_xlabel('Selected absolute position / committed prefix')
    axes[1].set_ylabel('Selected-position fraction\nGraph-quality requests only, not timing repeats')
    axes[1].set_xlim(0, 1);axes[1].set_ylim(bottom=0);axes[1].grid(alpha=.2)
    handles, _ = axes[1].get_legend_handles_labels()
    if handles:axes[1].legend(ncol=3, fontsize=7)
    else:axes[1].text(.5, .5, 'No saved quality selection histogram', transform=axes[1].transAxes, ha='center')
    save(fig, 'actions_selection', 'Actual proposal caps and saved quality selection positions (counts not reward causality)')
    telemetry = diag.get('gpu_telemetry', [])
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    for ax, key, label in zip(axes, ('temperature.gpu', 'clocks.current.sm'), ('GPU temperature (°C)', 'SM clock (MHz)')):
        xs, ys = [], []
        for i, row in enumerate(telemetry):
            try:value = float(row.get('telemetry', {}).get('values', {}).get(key))
            except (ValueError, TypeError):continue
            if math.isfinite(value):xs.append(i);ys.append(value)
        if ys:ax.plot(xs, ys, 'o-', markersize=3, color=COLORS[MODES[0]])
        else:ax.text(.5, .5, 'No numeric telemetry recorded', transform=ax.transAxes, ha='center')
        ax.set_ylabel(label);ax.grid(alpha=.2)
        ax.set_xticks(range(len(telemetry)), [f"{r.get('length')}\n{r.get('mode', '').replace('retrieval32_k4_graph', 'retr').replace('rescue_graph', 'own').replace('tree32_n8', 'tree')}\n{r.get('phase', '').replace('owner_', '')}" for r in telemetry], fontsize=5, rotation=60)
    axes[-1].set_xlabel('Saved owner-boundary order, outside primary clock; not drift correction')
    save(fig, 'gpu_telemetry', 'Read-only GPU boundary diagnostics')
    return list(FIGURES)


def write_report(summary, directory, output):
    output, directory = Path(output), Path(directory)
    if output.exists():raise FileExistsError('Fresh report directory required; never overwrite saved outputs')
    output.mkdir(parents=True)
    shutil.copyfile(Path(__file__), output/'analysis_source.py')
    for saved, source in SUPPORT_PATHS.items():shutil.copyfile(source, output/saved)
    (output/'summary.json').write_text(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False)+'\n')
    rendered = figures(summary, output)
    lines = ['# '+summary.get('status', 'PARTIAL')+' — frozen input-context benefit curves', '',
        'Fixed Mapper, frozen causal2/4 policy and old cost. No training, search, replacement of documents or new fresh-confirmation claim.', '',
        'This characterization uses already-seen development documents and fixed representative Qwen2 ports. It does not identify the fastest opponent at each new input length, does not replace the retained12-opponent4K/8K test, and cannot complete the end-to-end goal.', '',
        'New generation plan:64 AR +128 own quality +12 retrieval quality +18 paid cold +576 warm =798. Old32 canonical AR and64 own quality requests are proof reuse, not new calls. Input labels are actual token counts, not nominal bins.', '',
        'Primary: pool both timing repeats within each document first; whole-arm TPS = sum(client tokens)/sum(whole-generate outer seconds). Benefit =100×(RESCUE TPS / fixed-opponent TPS −1). Shared seed20261028,10000 document resamples across all plotted contexts and arms. Error bars are pointwise two-sided95% percentile intervals, not simultaneous final gates.', '']
    if summary.get('synthetic_program_test_only'):lines[2:2] = ['SYNTHETIC PROGRAM TEST ONLY — fabricated records, no actual model / GPU measurements.', '']
    lines += ['- '+value for value in summary.get('limitations', [])]
    lines += ['', 'Completed matched documents by actual context: '+json.dumps(summary.get('paired_complete_documents_by_context', {}), sort_keys=True)+'.',
              'Common completion-only document intersection used by the plotted curves: '+str(summary.get('statistics', {}).get('document_order', []))+'. No missing points are imputed; line segments break over missing registered contexts.', '',
              '## Relative TPS benefit', '', '|Actual input tokens|Fixed opponent|Paired docs|TPS difference|Pointwise95% CI|', '|---:|---|---:|---:|---|']
    for row in summary.get('statistics', {}).get('contrasts', []):
        lines.append(f"|{row['length']}|{row['baseline_mode']}|{row['documents']}|{row['point_percent']:.3f}%|{row.get('ci95')}|")
    lines += ['', '## Saved visualizations', '']
    for name in rendered:lines += [f'[{name} PNG]({name}.png) · [SVG]({name}.svg)', f'![{name}]({name}.png)', '']
    lines += ['## Evidence issues', '']+(['- '+v for v in summary.get('problems', [])] or ['None.'])
    lines += ['', '## Raw records and saved reproducible sources', '', '[Analysis source](analysis_source.py)', '']
    for name in SUPPORT_FILES:lines.append(f'[{name}]({name})')
    for name in ('protocol.json', 'COMPLETE.json', 'QUALITY_PASSED.json', 'reference.jsonl', 'references.json',
                 'own_quality.jsonl', 'baseline_quality.jsonl', 'quality_owners.jsonl', 'prime.jsonl', 'runtime.jsonl',
                 'owners.jsonl', 'gpu_diagnostics.jsonl', 'reference_reuse.json', 'data/DATASET_MANIFEST.json'):
        if (directory/name).is_file():lines.append(f'[{name}]({os.path.relpath(directory/name, output)})')
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    receipt = dict(status=summary.get('status', 'PARTIAL'), goal_complete=False, fresh_confirmation=False,
        rendered_figures=rendered, root_actual_render_inspection_required=True, independent_audit=False,
        file_sha256={path.name: sha(path) for path in output.iterdir() if path.is_file()})
    (output/'ANALYSIS_COMPLETE.json').write_text(json.dumps(receipt, sort_keys=True, indent=2)+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.directory)
    write_report(summary, args.directory, args.output)
    print(json.dumps(dict(status=summary.get('status'), goal_complete=False, problems=summary.get('problems')), indent=2))


if __name__ == '__main__':main()
