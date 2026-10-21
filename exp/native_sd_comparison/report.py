"""Local, partial-safe ordinary-SD comparison: no Torch, model or SSH imports.

Three *newly timed* arms, never a ratio against historical timings. Shared
frozen tensor/trajectory validators are saved beside the report; this program
is not represented as an independently implemented raw-evidence audit.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys

LENGTHS = (2048, 4096, 6144, 8192, 12288, 16384)
DOCS = (211, 213, 220, 226, 233, 236, 246, 248, 253, 271, 272, 288, 303, 306, 327, 604)
MODES = ('native_eager', 'rescue_eager', 'rescue_graph')
SEED, DRAWS = 20261028, 10000
EXPECTED = dict(native_quality_requests=96, quality_owners=96, prime_requests=18,
                timing_requests=576, owners=18)
TOTAL = 690
ORDER = ((0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2))
RUNTIME = ('captures', 'replays', 'resident_steps', 'shadow_updates', 'capture_host_seconds')
WORK = ('tokens', 'proposed', 'accepted', 'blocks', 'draft_forwards',
        'selector_refreshes', 'working_gather_layers', 'working_full_gather_layers',
        'working_gathered_position_layers', 'working_incremental_binds',
        'draft_shadow_probes', 'projected_positions', 'rpc_calls')
NATIVE_ZERO = ('selector_refreshes', 'working_gather_layers', 'working_full_gather_layers',
               'working_gathered_position_layers', 'working_incremental_binds',
               'target_probe_rounds', 'draft_shadow_probes', 'projected_positions', 'rpc_calls')
LABELS = dict(native_eager='Native SD: full KV, fixed k4, eager',
              rescue_eager='RESCUE: frozen modules, eager',
              rescue_graph='RESCUE: frozen modules, Draft Graph')
COLORS = dict(zip(MODES, ('#555555', '#087e8b', '#bd6926')))
CONTRASTS = (('matched_eager', 'rescue_eager', 'native_eager'),
             ('deployment_combination', 'rescue_graph', 'native_eager'),
             ('draft_graph_execution', 'rescue_graph', 'rescue_eager'))
FIGURES = ('tps_gain_native', 'tps', 'acceptance_length', 'acceptance_rate',
           'graph_execution_gain', 'latency_work', 'captures_cold_service',
           'actions_selection', 'gpu_telemetry')
SUPPORT_NAME = 'analysis_fair_support.py'
SUPPORT = Path(__file__).with_name(SUPPORT_NAME)
if not SUPPORT.is_file():
    SUPPORT = Path(__file__).resolve().parents[2]/'rescue/fair_confirmation_report.py'
_spec = importlib.util.spec_from_file_location('_native_sd_saved_fair', SUPPORT)
fair = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fair)
positive, nonnegative, near = fair.positive, fair.nonnegative, fair.near


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


class Evidence:
    """Only regular local descendants; saved remote paths are never opened."""
    def __init__(self, root):
        self.root, self.problems = Path(root), []

    def problem(self, text):
        if text not in self.problems:
            self.problems.append(text)

    def path(self, name):
        p = PurePosixPath(name)
        if (not isinstance(name, str) or not p.parts or p.as_posix() != name or
                p.is_absolute() or '..' in p.parts or '\\' in name or '\x00' in name):
            raise ValueError('Unsafe saved relative path')
        path = self.root.absolute()/Path(*p.parts)
        if any(v.is_symlink() for v in (path, *path.parents)):
            raise ValueError('Symlink saved evidence forbidden')
        return path

    def json(self, name):
        try:
            value = json.loads(self.path(name).read_bytes())
            if not isinstance(value, dict):
                raise ValueError('Expected object')
            return value
        except (OSError, ValueError, TypeError):
            self.problem('Missing/malformed '+name)
            return {}

    def rows(self, name, optional=False):
        try:
            raw = self.path(name).read_bytes()
        except OSError:
            if not optional:
                self.problem('Missing '+name)
            return []
        rows = []
        if raw and not raw.endswith(b'\n'):
            self.problem('Truncated final JSONL record: '+name)
        for i, line in enumerate(raw.splitlines()):
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError('Expected object')
                rows.append(value)
            except (ValueError, TypeError):
                self.problem(f'Malformed {name} record {i}')
        return rows

    def pin(self, name, digest):
        try:
            good = (isinstance(digest, str) and len(digest) == 64 and
                    all(c in '0123456789abcdef' for c in digest) and sha(self.path(name)) == digest)
        except (OSError, ValueError, TypeError):
            good = False
        if not good:
            self.problem('Missing/changed saved SHA evidence: '+str(name))
        return good


def canonical(ids, eos, vocab):
    return (isinstance(ids, list) and 0 < len(ids) <= 128 and
            all(type(v) is int and 0 <= v < vocab for v in ids) and
            (len(ids) == 128 or ids[-1] == eos) and
            (eos not in ids or ids[-1] == eos and eos not in ids[:-1]))


def arms():
    return {MODES[0]: dict(mode=MODES[0], family='native', backend='eager', fixed_k=4, full_context=True),
            MODES[1]: dict(mode=MODES[1], family='rescue', backend='eager'),
            MODES[2]: dict(mode=MODES[2], family='rescue', backend='cuda_graph')}


def work_errors(left, right):
    """Optional counters compare presence too: missing is not an invented zero."""
    errors = []
    if (left.get('generated') != right.get('generated') or
            fair.trace_signature(left) != fair.trace_signature(right) or
            any(k not in left or k not in right or left[k] != right[k] for k in WORK)):
        errors.append('Same-input logical trajectory/work counters differ')
    for key in ('target_probe_rounds', 'selection_similarity_elements', 'injection_condition_steps',
                'static_affine_positions', 'working_restored_value_positions', 'rpc_bytes_sent'):
        if (key in left) != (key in right) or key in left and left[key] != right[key]:
            errors.append('Same-input optional counter missing/changed: '+key)
    return errors


def native_errors(row, quality=False):
    errors = []
    state = row.get('native_state', {})
    wanted = dict(top_chunks=None, working=False, injector=False, draft_runtime=False,
                  target_hooks=0, round_observer_enabled=quality)
    if state != wanted or row.get('baseline_algorithm') != 'native_full_context' or row.get('execution_backend') != 'eager':
        errors.append('Native is not unmodified full-context eager SD without intervention state')
    signature = row.get('owner_signature', {})
    if any(signature.get(k, 'MISSING') is not None for k in ('working', 'injector', 'runtime')):
        errors.append('Native owns a hidden working KV, injector or Graph runtime')
    if 'draft_runtime' in row or any(row.get(k) != 0 for k in NATIVE_ZERO):
        errors.append('Native executed selector/working KV/probe/Mapper/Graph/RPC work')
    if any(k in row and row[k] != 0 for k in ('selection_similarity_elements', 'injection_condition_steps',
            'static_affine_positions', 'working_restored_value_positions', 'rpc_bytes_sent')):
        errors.append('Native has nonzero intervention-only work')
    for step in row.get('trace', []):
        if step.get('workset_tokens') != step.get('prefix'):
            errors.append('Native Draft did not read the complete committed prefix')
    if quality:
        if row.get('nonzero_injection') is not False:
            errors.append('Native quality falsely contains Mapper injection')
        for step in row.get('observed_rounds', []):
            if (step.get('selected') != list(range(step.get('prefix', 0))) or
                    step.get('budget_features') is not None or step.get('summary_count') != 0 or
                    step.get('injection_evidence', []) != []):
                errors.append('Native observed full-context/no-controller state differs')
        if any(q.get('summary_requested') is not False or q.get('summary_count') != 0 for q in row.get('queries', [])):
            errors.append('Native computes controller summaries')
    return errors


def runtime_errors(row, mode, quality=False):
    errors = []
    signature, stats = row.get('owner_signature', {}), row.get('runtime_stats', {})
    for name in ('native_t', 'native_s'):
        cache = signature.get(name, {})
        if (any(type(cache.get(k)) is not int or cache[k] <= 0 for k in ('identity', 'key_ptr', 'value_ptr', 'capacity')) or
                cache.get('capacity', 0) < row.get('length', 0)+128):
            errors.append('Actual private Native KV capacity/pointers missing')
    if mode == MODES[0]:
        errors += native_errors(row, quality)
    elif not isinstance(signature.get('working'), dict):
        errors.append('RESCUE actual working KV owner missing')
    if mode != MODES[2]:
        if ('draft_runtime' in row or stats.get('mode') != 'eager' or stats.get('entries') != 0 or
                any(stats.get(k) != 0 for k in RUNTIME)):
            errors.append('Eager arm contains hidden Graph work')
        if mode == MODES[1] and 'runtime' in signature:
            errors.append('Eager RESCUE owner binds hidden runtime')
        return errors
    runtime, delta = row.get('draft_runtime', {}), row.get('draft_runtime', {}).get('request', {})
    pointers = [signature.get(name, {}).get(k) for name in ('native_s', 'working') for k in ('key_ptr', 'value_ptr')]
    if (runtime.get('mode') != 'cuda_graph' or signature.get('runtime', {}).get('bound_ptrs') != pointers or
            stats.get('mode') != 'cuda_graph' or stats.get('max_graph_entries') != 128):
        errors.append('Graph backend/bound private cache/entry limit differs')
    if any(not nonnegative(delta.get(k)) or k != 'capture_host_seconds' and type(delta.get(k)) is not int for k in RUNTIME):
        return errors+['Missing real Graph request work counters']
    if (delta.get('replays') != row.get('draft_forwards') or delta.get('shadow_updates') != row.get('draft_forwards') or
            delta.get('resident_steps') != 0 or (delta['captures'] > 0) != (delta['capture_host_seconds'] > 0)):
        errors.append('Graph replay/binding work or capture count/time differs')
    keys = stats.get('graph_keys', [])
    if (not isinstance(keys, list) or any(not isinstance(k, list) or len(k) != 4 or type(k[0]) is not int or
            k[0] <= 0 or type(k[1]) is not bool or type(k[2]) is not bool for k in keys)):
        return errors+['Malformed actual Graph shape key ledger']
    if (stats.get('entries') != len(keys) or len({json_sha(k) for k in keys}) != len(keys) or
            len(keys) > 128 or stats.get('captures') != len(keys)):
        errors.append('Graph unique entries/captures differ')
    required = set()
    for step in row.get('trace', []):
        required.update((step['workset_tokens']+j, True, j <= 2) for j in range(1, step['proposed']+1))
        if step['accepted'] == step['proposed'] and not step['terminal']:
            required.add((step['workset_tokens']+step['proposed']+1, False, False))
    if not required <= {tuple(k[:3]) for k in keys}:
        errors.append('Actual per-step work/logit/summary shape missing from Graph ledger')
    if quality or row.get('phase') == 'cold_prime':
        if not positive(delta['captures']) or any(not near(delta[k], stats.get(k)) for k in RUNTIME):
            errors.append('Fresh owner borrowed historical Graph capture/replay work')
    return errors


def record_errors(row, reference, mode, quality=False):
    errors = []
    if (row.get('generated') != reference or not reference or row.get('tokens') != len(reference) or
            row.get('output_agrees') is not True or row.get('performance_eligible') is not (not quality)):
        errors.append('Request differs from canonical Target AR or clock eligibility')
    errors += fair.linear_trace_errors(row, own=mode != MODES[0], fixed_k=4 if mode == MODES[0] else None)
    errors += runtime_errors(row, mode, quality)
    if quality:
        errors += fair.observation_errors(row)+fair.quality_pair_errors(row, row)
        if mode != MODES[0]:
            errors += fair.injection_errors(row)
    else:
        if (not positive(row.get('outer_seconds')) or not near(row.get('outer_tps'),
                row.get('tokens', 0)/row['outer_seconds'] if positive(row.get('outer_seconds')) else -1) or
                not positive(row.get('total_seconds')) or row['outer_seconds']+1e-6 < row['total_seconds'] or
                'timeline' in row or 'cost_samples' in row or row.get('observer_enabled') is not False or
                row.get('profile_enabled') is not False or row.get('mapper_executed') is not (mode != MODES[0]) or
                row.get('controller_executed') is not (mode != MODES[0])):
            errors.append('Profiled/incomplete outer clock or wrong timed module flags')
        for key in ('prefill_seconds', 'decode_seconds', 'cleanup_seconds'):
            if not nonnegative(row.get(key)):
                errors.append('Missing/nonfinite measured stage interval: '+key)
        if not near(sum(row.get(k, -1) for k in ('prefill_seconds', 'decode_seconds', 'cleanup_seconds')), row.get('total_seconds')):
            errors.append('Saved stage intervals do not sum to generate total')
    return errors


def paired_statistics(cells, seed=SEED, draws=DRAWS):
    """Two repeats pool within document; one resample matrix for every point."""
    import numpy as np
    if (seed, draws) != (SEED, DRAWS):
        raise ValueError('Frozen seed20261028 /10000 draws required')
    result = dict(seed=seed, draws=draws, document_order=[], contexts={}, contrasts=[],
        common_document_resample=True, bootstrap_index_sha256=None, formal_success=False,
        interval='Pointwise two-sided95% document percentile bootstrap, not simultaneous/final gates')
    if not cells:
        return result
    sequences = [tuple(r['source_index'] for r in rows) for group in cells.values() for rows in group.values()]
    if len(set(sequences)) != 1 or not sequences[0] or len(set(sequences[0])) != len(sequences[0]) or any(d not in DOCS for d in sequences[0]):
        raise ValueError('Same ordered unique registered documents required for every context/arm')
    n = len(sequences[0]);result['document_order'] = list(sequences[0])
    index = np.random.default_rng(seed).integers(0, n, size=(draws, n), dtype=np.int64)
    result['bootstrap_index_sha256'] = hashlib.sha256(index.astype('<i8').tobytes()).hexdigest()
    for length in map(str, LENGTHS):
        if length not in cells:
            continue
        group = cells[length]
        if set(group) != set(MODES):
            raise ValueError('Exactly three newly timed arms required')
        samples, values = {}, {}
        for mode in MODES:
            rows = group[mode]
            if any(not positive(r.get('tokens')) or not positive(r.get('seconds')) or not positive(r.get('blocks')) or
                   not positive(r.get('proposed')) or not nonnegative(r.get('accepted')) or r['accepted'] > r['proposed'] for r in rows):
                raise ValueError('Positive actual count/time cells required')
            token, seconds, accepted, rounds, proposed = (np.asarray([r[k] for r in rows], dtype=float)
                for k in ('tokens', 'seconds', 'accepted', 'blocks', 'proposed'))
            metric = dict(documents=n, tokens=int(token.sum()), seconds=float(seconds.sum()),
                          accepted=int(accepted.sum()), blocks=int(rounds.sum()), proposed=int(proposed.sum()))
            for key, numerator, denominator in (('pooled_tps', token, seconds),
                    ('acceptance_length', accepted, rounds), ('acceptance_rate', accepted, proposed)):
                sample = numerator[index].sum(1)/denominator[index].sum(1)
                metric[key] = float(numerator.sum()/denominator.sum())
                metric[key+'_ci95'] = None if n < 2 else np.percentile(sample, [2.5, 97.5]).tolist()
                if key == 'pooled_tps':
                    samples[mode] = sample
            metric['tps_ci95'] = metric.pop('pooled_tps_ci95')
            values[mode] = metric
        result['contexts'][length] = values
        for name, numerator, denominator in CONTRASTS:
            sample = 100*(samples[numerator]/samples[denominator]-1)
            result['contrasts'].append(dict(name=name, length=int(length), own_mode=numerator,
                baseline_mode=denominator, documents=n,
                point_percent=100*(values[numerator]['pooled_tps']/values[denominator]['pooled_tps']-1),
                ci95=None if n < 2 else np.percentile(sample, [2.5, 97.5]).tolist()))
    return result


def pooled_cells(warm, complete):
    active = [str(n) for n in LENGTHS if complete.get(str(n))]
    common = [d for d in DOCS if active and all(d in complete[n] for n in active)]
    lookup = {(r['length'], r['mode'], r['source_index'], r['repeat']): r for r in warm}
    cells = {}
    if common:
        for n in active:
            cells[n] = {m: [dict(fair.pool_document([lookup[int(n), m, d, j] for j in (0, 1)]), source_index=d)
                            for d in common] for m in MODES}
    return cells


def export_errors(e):
    if (e.root/'EVIDENCE_MANIFEST.json').is_file():
        base, prefix = e, ''
    else:
        base, prefix = Evidence(e.root.parent), e.root.name+'/'
    manifest = base.json('EVIDENCE_MANIFEST.json').get('file_sha256', {})
    if not manifest:
        return ['Read-only exported source/raw inventory missing']
    files = [p.relative_to(e.root).as_posix() for p in e.root.rglob('*') if p.is_file() and
             not p.is_relative_to(e.root/'analysis') and p.name != 'EVIDENCE_MANIFEST.json']
    # Output analysis directories are deliberately not part of the raw inventory.
    files = [n for n in files if not any(part.startswith('analysis') for part in Path(n).parts[:-1])]
    errors = []
    if not {prefix+n for n in files} <= set(manifest):
        errors.append('Export inventory omits saved raw/source/proof files')
    for name, digest in manifest.items():
        base.pin(name, digest)
    errors += base.problems if base is not e else []
    return errors


def prior_errors(e, p, reuse, refs, reference_rows, quality_rows):
    """Authenticate the consumed subset, not a rehash of the whole old archive."""
    before = len(e.problems)
    args = p.get('args', {})
    for name, key in (('FINAL_ARTIFACTS_MANIFEST.json', 'prior_manifest_sha256'),
                      ('VERIFIED_COMPLETE.json', 'prior_verified_sha256')):
        digest = reuse.get(key)
        e.pin('prior_local_seal/'+name, digest)
        if args.get(key) != digest:
            e.problem('Prior external seal SHA differs from frozen entry arguments')
    seal=e.json('prior_local_seal/VERIFIED_COMPLETE.json')
    manifest=e.json('prior_local_seal/FINAL_ARTIFACTS_MANIFEST.json')
    if (seal.get('status') != 'VERIFIED_COMPLETE' or seal.get('passed') is not True or
            seal.get('six_lengths_complete') is not True or
            seal.get('final_artifacts_manifest_sha256') != reuse.get('prior_manifest_sha256')):
        e.problem('Prior six-context completion seal invalid')
    inventory=manifest.get('file_sha256', {})
    curvepins=reuse.get('curve_artifact_pins', {});oldpins=reuse.get('old_artifact_pins', {})
    required={'results/protocol.json','results/COMPLETE.json','results/QUALITY_PASSED.json',
        'results/REUSE_RECEIPT.json','results/reference.jsonl','results/reused_reference.jsonl',
        'results/references.json','results/own_quality.jsonl','results/reused_own_quality.jsonl',
        'results/quality_owners.jsonl','results/documents.jsonl','results/prime.json',
        'PIPELINE_EXIT.json','LAUNCH_MANIFEST.json','GPU_PREFLIGHT.json','DEPENDENCY_PREFLIGHT.json',
        'tools/run_context_curve.py'}
    if not required <= set(curvepins) or any(inventory.get('terminal/'+n) != digest for n,digest in curvepins.items()):
        e.problem('Consumed prior curve subset not bound to complete seal inventory')
    for label,pins in (('curve',curvepins),('old',oldpins)):
        if not pins:e.problem('Empty consumed proof pins: '+label)
        for name,digest in pins.items():e.pin('prior_consumed/'+label+'/'+name,digest)
    oldreuse=e.json('prior_consumed/curve/results/REUSE_RECEIPT.json')
    if (any(oldreuse.get('old_artifact_pins',{}).get(n)!=v for n,v in oldpins.items()) or
            oldpins.get('LAUNCH_MANIFEST.json') != '002545801b8a15fdb94afb083c61dd76633e423ec828f7a3d7e2c1fc6fbaf40a'):
        e.problem('Original capsule/helper linkage changed')
    old=e.json('prior_consumed/curve/results/protocol.json')
    done=e.json('prior_consumed/curve/results/COMPLETE.json')
    if (done.get('passed') is not True or done.get('logical_requests')!=798 or
            e.json('prior_consumed/curve/PIPELINE_EXIT.json').get('exit_code')!=0 or
            e.json('prior_consumed/curve/results/QUALITY_PASSED.json').get('passed') is not True):
        e.problem('Prior798 execution/quality/normal termination invalid')
    expected_refs=(e.rows('prior_consumed/curve/results/reference.jsonl')+
                   e.rows('prior_consumed/curve/results/reused_reference.jsonl'))
    expected_quality=(e.rows('prior_consumed/curve/results/own_quality.jsonl')+
                      e.rows('prior_consumed/curve/results/reused_own_quality.jsonl'))
    if (reference_rows!=expected_refs or quality_rows!=expected_quality or
            refs!=e.json('prior_consumed/curve/results/references.json') or
            p.get('shared_warm_sequence')!=old.get('shared_warm_sequence')):
        e.problem('Reused reference/quality/original request-order bytes differ')
    if (not old.get('assets_sha256') or p.get('assets_sha256')!=old.get('assets_sha256') or
            p.get('configs',{}).get(MODES[2])!=old.get('configs',{}).get(MODES[2])):
        e.problem('Actual frozen model/Mapper/policy/cost/config differs from sealed prior')
    for role,digest in fair.ASSETS.items():
        if p.get(role)!=digest or p.get('assets_sha256',{}).get(args.get(role.removesuffix('_sha256')))!=digest:
            e.problem('Actual mandatory asset-path SHA differs: '+role)
    source=e.json('prior_consumed/old/capsule/LAUNCH_MANIFEST.json').get('source_sha256',{})
    if not source or source!=p.get('source_sha256'):
        e.problem('Frozen runtime source inventory differs from original capsule')
    for name,digest in source.items():e.pin('source/'+name,digest)
    documents=e.rows('documents.jsonl');prime=e.json('prime.json');inputs={}
    if ([r.get('source_index') for r in documents]!=list(DOCS) or
            prime!=next((r for r in documents if r.get('source_index')==271),None)):
        e.problem('Same16 natural documents / identical cold-warm271 required')
    for row in documents:
        left,right,body=(row.get(k,[]) for k in ('prefix_ids','suffix_ids','document_ids'))
        if (not all(isinstance(v,list) and all(type(t) is int and t>=0 for t in v) for v in (left,right,body)) or
                len(left)+len(right)!=30 or len(body)!=16354):
            e.problem('Natural body16354 + prompt30 changed; no padding allowed');continue
        for n in LENGTHS:
            ids=left+body[:n-30]+right
            inputs[f"{row['source_index']}:{n}"]=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    if (len(inputs)!=96 or inputs!=p.get('input_sha256') or inputs!=old.get('input_sha256') or
            inputs!=reuse.get('canonical_inputs_all96') or reuse.get('no_old_timing_reused') is not True):
        e.problem('Exact96 input fingerprints changed or historical timing reused')
    return len(e.problems)>before


def terminal_errors(e):
    """Runner completion alone cannot pass a failed stage/terminal export."""
    parent=Evidence(e.root.parent)
    pipeline=parent.json('PIPELINE_EXIT.json');collection=parent.json('COLLECTION.json')
    try:complete_digest=sha(e.path('COMPLETE.json'))
    except OSError:complete_digest=None
    if (pipeline.get('exit_code')!=0 or complete_digest is None or
            pipeline.get('complete_sha256')!=complete_digest):
        parent.problem('Authoritative outer stage did not exit0 with same raw COMPLETE')
    if collection.get('remote_writes') is not False:
        parent.problem('Read-only terminal collection receipt missing')
    for key in ('handles_before','handles_after'):
        handles=collection.get(key,{})
        if (handles.get('tmux_present') is not False or handles.get('execution_processes')!=[] or
                handles.get('exit')!=pipeline):
            parent.problem('Actual experiment handles not closed before/after export')
    pins=collection.get('original_file_sha256',{})
    if (not isinstance(pins,dict) or 'PIPELINE_EXIT.json' not in pins or
            e.root.name+'/COMPLETE.json' not in pins or e.root.name+'/protocol.json' not in pins):
        parent.problem('Terminal collection omits outer-exit/raw complete/protocol pins')
        pins={}
    for name,digest in pins.items():parent.pin(name,digest)
    return ['Terminal: '+v for v in parent.problems]


def _analyze(directory):
    e = Evidence(directory);p = e.json('protocol.json');prereg = e.json('PREREGISTRATION.json')
    wanted = dict(schema=1, seed=SEED, lengths=list(LENGTHS), source_indices=list(DOCS),
                  modes=list(MODES), repeats=2, max_new_tokens=128, graph_max_entries=128)
    if any(p.get(k) != v or prereg.get(k) != v for k, v in wanted.items()):
        e.problem('Six exact contexts /16 old documents / three arms / repeat/output/Graph bounds changed')
    if prereg.get('status') != 'FROZEN_BEFORE_EXECUTION' or p.get('expected_counts') != EXPECTED or p.get('expected_requests') != TOTAL:
        e.problem('Frozen690-call execution protocol/counts missing')
    if p.get('status')!='FROZEN_NATIVE_SD_COMPARISON' or p.get('arms')!=list(arms().values()):
        e.problem('Frozen three-arm ordinary-SD descriptors changed')
    if (any(p.get(k) is not True for k in ('no_training', 'no_search', 'no_new_confirmation_access', 'paused_test_untouched')) or
            p.get('fresh_confirmation') is not False or p.get('goal_complete') is not False):
        e.problem('Frozen development-only safety/method scope changed')
    e.pin('PREREGISTRATION.json', p.get('preregistration_sha256'))
    e.pin('NATIVE_SD_SOURCE.py', p.get('entry_source_sha256'))
    for role, digest in fair.ASSETS.items():
        if prereg.get('frozen', {}).get(role) != digest:
            e.problem('Fixed Mapper/policy/cost bytes changed: '+role)
    env = p.get('environment', {})
    if (env.get('gpu_name') != 'NVIDIA H100' or env.get('target_dtype') != 'torch.float32' or
            env.get('draft_dtype') != 'torch.bfloat16' or env.get('prefix_dtype') != 'torch.bfloat16' or
            env.get('tf32') is not False or env.get('visible_devices') != '0'):
        e.problem('Actual single-H100 / identical precision scope changed')
    e.problems.extend(x for x in export_errors(e) if x not in e.problems)
    plan = [dict(length=n, mode=MODES[i]) for n, order in zip(LENGTHS, ORDER) for i in order]
    if p.get('owner_order') != plan:
        e.problem('Pre-frozen counterbalanced18-owner order changed')
    sequence = p.get('shared_warm_sequence', [])
    if (len(sequence) != 32 or any(set(r) != {'source_index', 'repeat'} for r in sequence) or
            {(r.get('source_index'), r.get('repeat')) for r in sequence} != {(d, j) for d in DOCS for j in (0, 1)}):
        e.problem('Common32-request warm order changed')
    configs = p.get('configs', {})
    eager, graph = configs.get(MODES[1], {}), configs.get(MODES[2], {})
    if (not eager or not graph or {k:v for k,v in eager.items() if k != 'draft_execution'} !=
            {k:v for k,v in graph.items() if k != 'draft_execution'} or
            eager.get('draft_execution') != 'eager' or graph.get('draft_execution') != 'cuda_graph'):
        e.problem('Own eager/Graph arms differ beyond execution backend')
    fixed = dict(selector='hierarchical', budget=512, recent=64, inject_tokens=64,
        injection_strength=.5, injection_layers=[27], schedule='serial', length_policy='risk',
        max_new_tokens=128, coverage_execution='eager', attention_backend='sdpa')
    if any(eager.get(k) != v for k, v in fixed.items()):
        e.problem('Mandatory frozen attention/Mapper/causal-policy settings changed')
    native = configs.get(MODES[0], {})
    if any(native.get(k) != v for k, v in dict(length_policy='fixed', fixed_k=4,
            max_new_tokens=128, inject_tokens=0, draft_execution='eager', attention_backend='sdpa').items()):
        e.problem('Native fixed-k4 eager/SDPA control settings changed')
    metadata_bad = bool(e.problems)
    refs = e.json('references.json');rr = e.rows('reused_reference.jsonl');own = e.rows('reused_own_quality.jsonl')
    eos, vocab = p.get('tokenizer_eos_token_id'), p.get('vocab_size')
    if type(eos) is not int or type(vocab) is not int or not 0 <= eos < vocab:
        e.problem('Canonical tokenizer EOS/vocabulary invalid');eos, vocab = -1, 0;metadata_bad = True
    wanted_refs = {f'{d}:{n}' for d in DOCS for n in LENGTHS}
    if (set(refs) != wanted_refs or len(rr) != 96 or
            len({(r.get('source_index'), r.get('length')) for r in rr}) != 96 or
            {f"{r.get('source_index')}:{r.get('length')}":r.get('generated') for r in rr} != refs or
            any(not canonical(v, eos, vocab) for v in refs.values())):
        e.problem('All96 sealed canonical AR references missing/changed');metadata_bad = True
    reuse = e.json('REUSE_RECEIPT.json')
    e.pin('REUSE_RECEIPT.json', p.get('reuse_receipt_sha256'))
    if (reuse.get('reused_reference_requests') != 96 or reuse.get('reused_own_quality_requests') != 192 or
            reuse.get('new_generation_requests') != TOTAL or reuse.get('no_new_confirmation_access') is not True):
        e.problem('Reuse-versus-new690-call accounting changed');metadata_bad = True
    metadata_bad |= prior_errors(e,p,reuse,refs,rr,own)
    quality = e.rows('native_quality.jsonl');qo = e.rows('quality_owners.jsonl')
    prime = e.rows('prime.jsonl');warm = e.rows('runtime.jsonl');owners = e.rows('owners.jsonl')
    telemetry = e.rows('gpu_diagnostics.jsonl', optional=True)
    counts = dict(native_quality_requests=len(quality), quality_owners=len(qo), prime_requests=len(prime),
                  timing_requests=len(warm), owners=len(owners))
    if counts != EXPECTED:
        e.problem('Incomplete690-call /96quality-owner /18timing-owner matrix')
    new = quality+prime+warm
    ids, ix = [r.get('request_id') for r in new], [r.get('execution_index') for r in new]
    if (any(not isinstance(v, str) or len(v) != 32 or any(c not in '0123456789abcdef' for c in v) for v in ids) or
            len(set(ids)) != len(ids) or any(type(v) is not int for v in ix) or sorted(ix) != list(range(len(new))) or
            set(ids) & {r.get('request_id') for r in own+rr}):
        e.problem('New request/execution bijection or separation from historical proof invalid');metadata_bad = True
    for row in new:
        if row.get('gpu_snapshot_before', {}).get('nvidia_smi_gpu') != env.get('nvidia_smi_gpu'):
            e.problem('Same physical GPU not checked before generation');metadata_bad = True
    qmap, bad = {}, set()
    qowners = {r.get('request_id'):r for r in qo}
    allownerids = [r.get('owner_id') for r in qo+owners]
    if len(set(allownerids)) != len(allownerids) or any(not isinstance(v, str) or not v for v in allownerids) or len(qowners) != len(qo):
        e.problem('Private quality/timing owner identities missing/duplicated');metadata_bad = True
    for row in own+quality:
        n, d, m = row.get('length'), row.get('source_index'), row.get('mode')
        key = n, d, m;errors = []
        if key in qmap or n not in LENGTHS or d not in DOCS or m not in MODES or (m == MODES[0]) is not (row in quality):
            errors.append('Duplicate/unregistered reused-own or new-native quality cell')
        qmap[key] = row
        errors += record_errors(row, refs.get(f'{d}:{n}'), m, True)
        if m == MODES[0]:
            owner = qowners.get(row.get('request_id'), {})
            if (row.get('phase') != 'native_quality' or owner.get('owner_id') != row.get('owner_id') or
                    owner.get('owner_signature') != row.get('owner_signature') or owner.get('engine_closed') is not True or
                    owner.get('hook_restored') is not True or owner.get('closed_runtime_stats', {}).get('entries') != 0 or
                    owner.get('final_runtime_stats') != row.get('runtime_stats')):
                errors.append('New private Native quality owner not fully closed')
        for error in errors:
            e.problem(f'Quality {n}/{d}/{m}: '+error)
        if errors:
            bad.add((n, d))
    if len(own) != 192 or len({(r.get('source_index'), r.get('length'), r.get('mode')) for r in own}) != 192:
        e.problem('All192 sealed eager/Graph own quality records required');metadata_bad = True
    for n in LENGTHS:
        for d in DOCS:
            pair = [qmap.get((n, d, m)) for m in MODES[1:]]
            errors = fair.quality_pair_errors(*pair)+work_errors(*pair) if all(pair) else ['Missing sealed own eager/Graph pair']
            if errors:
                bad.add((n, d))
                for error in errors:
                    e.problem(f'Backend {n}/{d}: '+error)
    gate = e.json('QUALITY_PASSED.json')
    if gate.get('passed') is not True or gate.get('quality_owners_closed') is not True:
        e.problem('Native quality gate not passed before timing');metadata_bad = True
    if len(quality) == 96 and [r.get('execution_index') for r in quality] != list(range(96)):
        e.problem('Native quality must be first96 new calls');metadata_bad = True
    wm, pm = {}, {}
    repeated = {}
    for phase, rows in (('cold_prime', prime), ('warm', warm)):
        for row in rows:
            n, d, m = row.get('length'), row.get('source_index'), row.get('mode')
            errors = record_errors(row, refs.get(f'{d}:{n}'), m)
            if m not in MODES or n not in LENGTHS or d not in DOCS or row.get('phase') != phase:
                errors.append('Unregistered timing arm/context/document/phase')
            if phase == 'warm':
                key = n, m, d, row.get('repeat')
                if key in wm or row.get('repeat') not in (0, 1):
                    errors.append('Duplicate/unregistered timing repeat')
                wm[key] = row
            else:
                key = n, m
                if key in pm or d != 271 or not near(row.get('cold_complete_outer_seconds'),
                        row.get('constructor_and_detach_seconds', -1)+row.get('outer_seconds', -1)):
                    errors.append('Missing separately billed same-context cold271 prime')
                pm[key] = row
            counterpart = qmap.get((n, d, m))
            errors += work_errors(counterpart, row) if counterpart else ['Missing correct quality proof for timing cell']
            if (n, d, m) in repeated:
                errors += work_errors(repeated[n, d, m], row)
            repeated[n, d, m] = row
            for error in errors:
                e.problem(f'{phase} {n}/{d}/{m}: '+error)
            if errors:
                bad.add((n, d))
    om = {r.get('owner_id'):r for r in owners};owner_bad = set()
    if [dict(length=r.get('length'), mode=r.get('mode')) for r in prime] != plan[:len(prime)]:
        e.problem('Observed owner blocks are not registered order prefix');metadata_bad = True
    for j, cold in enumerate(prime):
        n, m = cold.get('length'), cold.get('mode')
        block = [r for r in warm if r.get('owner_id') == cold.get('owner_id')]
        owner = om.get(cold.get('owner_id'))
        errors = []
        if ([{k:r.get(k) for k in ('source_index', 'repeat')} for r in block] != sequence[:len(block)] or
                [r.get('warm_index') for r in block] != list(range(len(block))) or
                cold.get('execution_index') != 96+33*j or
                [r.get('execution_index') for r in block] != list(range(97+33*j, 97+33*j+len(block)))):
            errors.append('Owner does not follow cold + original32 contiguous warm sequence')
        if owner:
            errors += fair.lifecycle_errors(owner, [cold, *block])
            if (owner.get('maximum_live_owners') != 1 or owner.get('mode') != m or owner.get('length') != n or
                    owner.get('accounted_service_tokens') != cold.get('tokens', 0)+sum(r.get('tokens', 0) for r in block) or
                    not nonnegative(owner.get('owner_lifetime_wall_seconds')) or
                    owner.get('owner_lifetime_wall_seconds', -1)+1e-8 < owner.get('accounted_service_seconds', 0)):
                errors.append('Owner identity/nonoverlap/finite service accounting differs')
        elif len(block) == 32:
            errors.append('Finished32-warm owner has no closing ledger')
        if m == MODES[2]:
            previous, totals = set(), dict.fromkeys(RUNTIME, 0)
            for row in [cold, *block]:
                keys = {json_sha(k) for k in row.get('runtime_stats', {}).get('graph_keys', [])}
                delta = row.get('draft_runtime', {}).get('request', {})
                if not previous <= keys or len(keys-previous) != delta.get('captures'):
                    errors.append('Graph eviction/reinitialization or omitted late capture')
                previous = keys
                for key in RUNTIME:
                    if not nonnegative(delta.get(key)):
                        errors.append('Missing Graph request delta: '+key)
                        continue
                    totals[key] += delta[key]
                    if not near(row.get('runtime_stats', {}).get(key), totals[key]) or not near(
                            row.get('draft_runtime', {}).get('lifetime', {}).get(key), totals[key]):
                        errors.append('Graph cumulative cold/warm work ledger differs')
        if errors:
            owner_bad.add((n, m))
            for error in errors:
                e.problem(f'Owner {n}/{m}: '+error)
    for n in LENGTHS:
        for d in DOCS:
            for repeat in (0, 1):
                pair = [wm.get((n, m, d, repeat)) for m in MODES[1:]]
                if all(pair) and work_errors(*pair):
                    bad.add((n, d));e.problem(f'Timing own eager/Graph logical trajectory differs: {n}/{d}/{repeat}')
    complete = {str(n):[d for d in DOCS if not metadata_bad and (n, d) not in bad and
                    all((n, m, d, j) in wm for m in MODES for j in (0, 1)) and
                    all((n, m) in pm and (n, m) not in owner_bad for m in MODES)] for n in LENGTHS}
    if len(owners) != 18 or len(om) != len(owners):
        e.problem('Complete18 closing owner ledgers required')
    expected_tel = [(phase, q['length'], q['mode'], 96+33*j+(33 if phase == 'owner_end' else 0))
                    for j, q in enumerate(plan) for phase in ('owner_start', 'owner_end')]
    if [(r.get('phase'), r.get('length'), r.get('mode'), r.get('execution_index')) for r in telemetry] != expected_tel[:len(telemetry)] or len(telemetry) != 36:
        e.problem('Complete36 outside-clock owner-boundary diagnostics required')
    done = e.json('COMPLETE.json')
    if (done.get('passed') is not True or done.get('counts') != EXPECTED or done.get('logical_requests') != TOTAL or
            done.get('goal_complete') is not False or done.get('fresh_confirmation') is not False or any(done.get(k) is not True for k in
                ('all_canonical', 'quality_gate_passed', 'owners_closed', 'sources_unchanged', 'assets_unchanged', 'dependencies_unchanged'))):
        e.problem('Complete690-call unchanged/closed/canonical terminal receipt missing')
    if (e.root/'FAILURE.json').exists():
        e.problem('Preserved experiment failure; completed subset is PARTIAL only')
    for error in terminal_errors(e):e.problem(error)
    cells = pooled_cells(warm, complete)
    return dict(schema=1, status='PARTIAL' if e.problems else 'COMPLETE', goal_complete=False,
        fresh_confirmation=False, synthetic_program_test_only=p.get('synthetic_program_test_only') is True,
        problems=e.problems, statistics=paired_statistics(cells), per_document_cells=cells,
        paired_complete_documents_by_context=complete, observed_counts=counts,
        expected_counts=EXPECTED, observed_requests=len(new), expected_requests=TOTAL,
        reused_reference_requests=len(rr), reused_own_quality_requests=len(own),
        diagnostics=dict(warm=warm, prime=prime, owners=owners, gpu_telemetry=telemetry,
                         reused_quality=own), raw_sha256={x.name:sha(x) for x in e.root.glob('*.json*') if x.is_file()},
        limitations=[
            'Already-reused16 development documents, two repeats pooled within document; no fresh confirmation or general/SOTA claim.',
            'Native SD is full native Draft KV, fixedk4, no attention probe/selector/working KV/Mapper/controller. Fixedk4 is transparent, not a search for the optimal native draft length.',
            'Primary algorithm-matched contrast is eager RESCUE / eager Native with identical attention and precision. Graph RESCUE / eager Native is a deployed implementation combination, not a matched-Graph or algorithm-only effect.',
            'All576 warm timings and18 cold primes are new in this batch. No historical timing is spliced into a numerator or denominator.',
            'Single H10080GB, Qwen2-7B/1.5B, Target FP32, Draft and prefill BF16, TF32 disabled; Mapper, causal2/4 policy and old cost frozen.',
            'One paid cold271 prime precedes32 warm requests for each private owner. Every legal late Graph capture remains in whole-generate outer time. Constructor/cold/close/accounted service are separate; not fully-prewarmed or production throughput.',
            'Owner order is counterbalanced across contexts, not interleaved within a context or independently replicated hardware sessions. Document bootstrap intervals do not cover owner-block thermal/clock drift.',
            'One shared10000xN whole-document bootstrap matrix, seed20261028, across every context and arm. Pointwise95% intervals are not simultaneous final success gates.',
            'Shared frozen record validators inspect saved KV/Q/injection hashes and traces; this analysis is not independently implemented tensor verification or a new GPU forward.',
            'Partial plots use completion-only common document intersections across available contexts/three arms; no speed/EOS filtering or missing-value imputation.'])


def analyze(directory):
    try:
        return _analyze(Path(directory))
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, OverflowError) as error:
        return dict(schema=1, status='PARTIAL', goal_complete=False, fresh_confirmation=False,
            problems=['Malformed/missing saved evidence: '+type(error).__name__+': '+str(error)],
            statistics={}, observed_counts={}, expected_counts=EXPECTED, expected_requests=TOTAL)


def figures(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    output = Path(output);stats = summary.get('statistics', {});contexts = stats.get('contexts', {})
    prefix = 'PARTIAL — ' if summary.get('status') != 'COMPLETE' else ''
    if summary.get('synthetic_program_test_only'):
        prefix = 'SYNTHETIC PROGRAM TEST — '+prefix
    caption = (f"Reused development docs; common paired n={len(stats.get('document_order', []))}/16; two repeats/doc; cap128/EOS. "
        'Actual contexts, K=1024; fixed Mapper/policy/old cost; H10080GB/Qwen2-7B+1.5B.\n'
        'Primary TPS/acceptance: pointwise95% document bootstrap; mechanism/thermal panels: descriptive logged observations, not paired inference.\n'
        'Native fixedk4 is not tuned; Graph vs Native includes execution-backend differences. '
        'Finite paid-cold +32-warm trace, late captures charged; not production/fresh confirmation.')
    def save(fig, name, title):
        fig.suptitle(prefix+title, fontsize=12)
        fig.text(.5, .012, caption, ha='center', fontsize=7)
        for ax in fig.axes:
            finite = any(np.isfinite(np.asarray(line.get_ydata(), dtype=float)).any() for line in ax.lines)
            if not finite and not ax.images and not ax.collections:
                ax.text(.5, .5, 'No saved observations', transform=ax.transAxes, ha='center')
        fig.tight_layout(rect=(0, .12, 1, .92))
        fig.savefig(output/(name+'.png'), dpi=160);fig.savefig(output/(name+'.svg'));plt.close(fig)
    def axis(ax, label):
        ax.set_xticks(np.asarray(LENGTHS)/1024, [f'{n//1024}K\n{n}' for n in LENGTHS])
        ax.set_xlabel('Actual input context (tokens; K=1024)');ax.set_ylabel(label);ax.grid(alpha=.2)
    def line(ax, values, cis=None, color=None, label=None, marker='o'):
        ax.plot(np.asarray(LENGTHS)/1024, [values.get(n, math.nan) for n in LENGTHS],
                marker+'-', color=color, label=label, linewidth=1.7)
        for n, ci in (cis or {}).items():
            if ci is not None:
                x=n/1024;ax.vlines(x, *ci, color=color);ax.hlines(ci, x-.08, x+.08, color=color)
    for name, subset, title in (
            ('tps_gain_native', CONTRASTS[:2], 'Whole-generate TPS benefit relative to ordinary full-KV SD'),
            ('graph_execution_gain', CONTRASTS[2:], 'Same frozen RESCUE method: Draft Graph / eager execution')):
        fig, ax = plt.subplots(figsize=(12.5, 5.5))
        for contrast, own, baseline in subset:
            rows = [r for r in stats.get('contrasts', []) if r['name'] == contrast]
            line(ax, {r['length']:r['point_percent'] for r in rows}, {r['length']:r['ci95'] for r in rows},
                 COLORS[own], label=('Matched eager RESCUE / Native eager' if contrast == 'matched_eager' else
                 'Deployment combination: Graph RESCUE / Native eager' if contrast == 'deployment_combination' else
                 'Graph RESCUE / eager RESCUE'))
        ax.axhline(0, color='#777777', linewidth=.8);axis(ax, 'Relative pooled whole-generate TPS difference (%)')
        ax.legend(fontsize=8);save(fig, name, title)
    for stem, key, ci, label in (
            ('tps', 'pooled_tps', 'tps_ci95', 'Client tokens / whole-generate outer second'),
            ('acceptance_length', 'acceptance_length', 'acceptance_length_ci95', 'Accepted Draft tokens / verification round'),
            ('acceptance_rate', 'acceptance_rate', 'acceptance_rate_ci95', 'Accepted / actually proposed Draft tokens')):
        fig, ax = plt.subplots(figsize=(12.5, 5.5))
        for m in MODES:
            line(ax, {int(n):v[m][key] for n,v in contexts.items()},
                 {int(n):v[m][ci] for n,v in contexts.items()}, COLORS[m], LABELS[m])
        axis(ax, label);ax.set_ylim(bottom=0);ax.margins(y=.18);ax.legend(fontsize=8)
        save(fig, stem, 'Measured '+stem.replace('_', ' ')+' vs input context')
    diag=summary.get('diagnostics', {});warm=diag.get('warm', []);prime=diag.get('prime', []);owners=diag.get('owners', [])
    def measured(rows, mode, key, how='mean'):
        values={}
        for n in LENGTHS:
            records=[r for r in rows if r.get('length')==n and r.get('mode')==mode]
            vals=[r.get(key) for r in records]
            if vals and all(nonnegative(v) for v in vals):
                values[n]=sum(vals)/(len(vals) if how=='mean' else 1)
        return values
    fig, axes=plt.subplots(2, 3, figsize=(16, 8))
    specs=(('prefill_seconds','Prefill seconds / request'),('decode_seconds','Decode seconds / request'),
           ('blocks','Verification rounds / request'),('draft_forwards','Actual Draft forwards / request'),
           ('selector_refreshes','Selection refreshes / request'),('projected_positions','Mapped positions / request'))
    for ax,(key,label) in zip(axes.flat,specs):
        for m in MODES:line(ax,measured(warm,m,key),color=COLORS[m],label=LABELS[m])
        axis(ax,label);ax.set_ylim(bottom=0)
    axes[0,0].legend(fontsize=7);save(fig,'latency_work','Saved stage/work diagnostics: all observed rows (not paired speed estimates)')
    fig,axes=plt.subplots(2,2,figsize=(14,8))
    for ax,key,label in zip(axes.flat,('captures','capture_host_seconds','cold_complete_outer_seconds','accounted_service_tps'),
            ('Warm Graph captures: actual count','Warm Graph capture host seconds',
             'Constructor + paid cold prime generate (s)','Accounted service TPS: constructor + cold +32warm +close')):
        for m in MODES:
            if key in RUNTIME:
                vals={}
                for n in LENGTHS:
                    rows=[r for r in warm if r.get('length')==n and r.get('mode')==m]
                    observed=[r.get('draft_runtime',{}).get('request',{}).get(key) for r in rows]
                    # Eager zeros are only displayed if explicitly recorded in runtime_stats.
                    if m != MODES[2]:observed=[r.get('runtime_stats',{}).get(key) for r in rows]
                    if observed and all(nonnegative(v) for v in observed):vals[n]=sum(observed)
            else:vals=measured(prime if key.startswith('cold_') else owners,m,key)
            line(ax,vals,color=COLORS[m],label=LABELS[m])
        axis(ax,label);ax.set_ylim(bottom=0)
    axes[0,0].legend(fontsize=7);save(fig,'captures_cold_service','All captures, cold starts and finite-trace accounted service')
    fig,axes=plt.subplots(2,1,figsize=(13,8))
    for m in MODES:
        for cap in (1,2,3,4):
            values={}
            for n in LENGTHS:
                rows=[r for r in warm if r.get('length')==n and r.get('mode')==m]
                if rows:values[n]=sum(s.get('cap')==cap for r in rows for s in r.get('trace',[]))
            if any(values.values()):line(axes[0],values,color=COLORS[m],label=f'{m}, cap{cap}',marker={1:'v',2:'s',3:'^',4:'o'}[cap])
    axis(axes[0],'Observed verification rounds by actual cap')
    handles,_=axes[0].get_legend_handles_labels()
    if handles:axes[0].legend(fontsize=7,ncol=3)
    for n in LENGTHS:
        hist=np.zeros(20,dtype=int);found=False
        for r in diag.get('reused_quality',[]):
            if r.get('length')!=n or r.get('mode')!=MODES[2]:continue
            for s in r.get('observed_rounds',[]):
                for p0 in s.get('selected',[]):hist[min(19,int(20*p0/s['prefix']))]+=1;found=True
        if found:axes[1].plot((np.arange(20)+.5)/20,hist/hist.sum(),label=f'{n} tokens')
    axes[1].set_xlabel('Selected position / committed prefix; reused quality only, not timing repeats')
    axes[1].set_ylabel('Selected-position fraction');axes[1].set_xlim(0,1);axes[1].grid(alpha=.2)
    handles,_=axes[1].get_legend_handles_labels()
    if handles:axes[1].legend(ncol=3,fontsize=7)
    save(fig,'actions_selection','Actual proposal caps and frozen selection-position diagnostics')
    telemetry=diag.get('gpu_telemetry',[]);fig,axes=plt.subplots(2,1,figsize=(15,7))
    for ax,key,label in zip(axes,('temperature.gpu','clocks.current.sm'),('GPU temperature (degrees C)','SM clock (MHz)')):
        xs,ys=[],[]
        for i,r in enumerate(telemetry):
            try:v=float(r.get('telemetry',{}).get('values',{}).get(key))
            except (ValueError,TypeError):continue
            if math.isfinite(v):xs.append(i);ys.append(v)
        if ys:ax.plot(xs,ys,'o-',markersize=3,color=COLORS[MODES[1]])
        ax.set_ylabel(label);ax.grid(alpha=.2)
        ax.set_xticks(range(len(telemetry)),[f"{r.get('length')}\n{r.get('mode','').replace('native_eager','native').replace('rescue_eager','own-E').replace('rescue_graph','own-G')}\n{r.get('phase','').replace('owner_','')}" for r in telemetry],rotation=60,fontsize=5)
    axes[-1].set_xlabel('Saved owner-boundary order, outside primary clock; diagnostics, not drift correction')
    save(fig,'gpu_telemetry','Observed physical-GPU boundary diagnostics')
    return list(FIGURES)


def write_report(summary, directory, output):
    directory,output=Path(directory),Path(output)
    if output.exists():raise FileExistsError('Fresh analysis output only; preserve prior artifacts')
    output.mkdir(parents=True)
    shutil.copyfile(Path(__file__),output/'analysis_entry.py');shutil.copyfile(SUPPORT,output/SUPPORT_NAME)
    (output/'summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2,allow_nan=False)+'\n')
    rendered=figures(summary,output)
    lines=['# '+summary.get('status','PARTIAL')+' — ordinary full-KV SD comparison','',
        'Frozen Mapper and causal2/4 policy, all three timing arms newly measured in this batch. No historical timing ratio, training or policy search.','',
        'Primary: eager RESCUE / eager native SD. Deployment combination: Graph RESCUE / eager native SD. Graph/eager RESCUE is a separate implementation diagnostic, not a third algorithm. Native fixedk4 is not a tuned-best or matched-Graph baseline.','',
        'New calls:96 Native quality +18 separately billed cold primes +576 warm timings =690. Reused96 AR references and192 eager/Graph own quality records are proof reuse, not new calls.','',
        'Two repeats pool within each document. Whole-arm TPS is sum(actual client tokens)/sum(complete-generate outer seconds). Benefit =100×(TPS numerator/TPS denominator−1). Every contrast and context shares10000 whole-document resamples, seed20261028; intervals are pointwise95%, not simultaneous final gates.','']
    if summary.get('synthetic_program_test_only'):lines[2:2]=['SYNTHETIC PROGRAM TEST ONLY — fabricated records, no measured model performance.','']
    lines+=['- '+s for s in summary.get('limitations',[])]+['',
        'Completed three-arm paired documents by exact context: '+json.dumps(summary.get('paired_complete_documents_by_context',{}),sort_keys=True)+'.',
        'Plotted common document intersection: '+str(summary.get('statistics',{}).get('document_order',[]))+'. Missing measurements are omitted, not assigned zeros.','',
        '|Input tokens|Contrast|Paired docs|TPS difference|Pointwise95% CI|','|---:|---|---:|---:|---|']
    for r in summary.get('statistics',{}).get('contrasts',[]):
        lines.append(f"|{r['length']}|{r['name']}|{r['documents']}|{r['point_percent']:.3f}%|{r['ci95']}|")
    lines+=['','## Separate measured metrics','', '|Input tokens|Mode|Whole-generate TPS|Accepted Draft/round|Draft acceptance rate|','|---:|---|---:|---:|---:|']
    for n,group in summary.get('statistics',{}).get('contexts',{}).items():
        for m,r in group.items():lines.append(f"|{n}|{m}|{r['pooled_tps']:.4f}|{r['acceptance_length']:.4f}|{r['acceptance_rate']:.4f}|")
    lines+=['','## Figures','']
    for name in rendered:lines += [f'[{name} PNG]({name}.png) · [SVG]({name}.svg)',f'![{name}]({name}.png)','']
    lines+=['## Evidence issues','']+(['- '+s for s in summary.get('problems',[])] or ['None.'])
    lines+=['','## Saved raw results and executable figure source','',
        '[Saved CLI and figure source](analysis_entry.py) · [Shared frozen evidence validators](analysis_fair_support.py)','',
        'Reproduce without Torch, network, models or original repository: `python analysis_entry.py /absolute/saved/results --output /absolute/new/report-directory`.','']
    for name in ('protocol.json','PREREGISTRATION.json','COMPLETE.json','QUALITY_PASSED.json','references.json',
                 'reused_reference.jsonl','reused_own_quality.jsonl','native_quality.jsonl','quality_owners.jsonl',
                 'prime.jsonl','runtime.jsonl','owners.jsonl','gpu_diagnostics.jsonl','REUSE_RECEIPT.json'):
        if (directory/name).is_file():lines.append(f'[{name}]({os.path.relpath(directory/name,output)})')
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    receipt=dict(status=summary.get('status','PARTIAL'),goal_complete=False,fresh_confirmation=False,
        rendered_figures=rendered,root_actual_render_inspection_required=True,independent_audit=False,
        file_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    (output/'ANALYSIS_COMPLETE.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();summary=analyze(args.directory);write_report(summary,args.directory,args.output)
    print(json.dumps(dict(status=summary['status'],goal_complete=False,problems=summary.get('problems')),indent=2))


if __name__=='__main__':main()
