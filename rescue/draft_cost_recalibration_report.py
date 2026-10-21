"""CPU-only audit and figures for the frozen 94-request cost intervention.

Documents, not rounds or repeats, are the independent statistical units. Only
student_step is recalibrated; this is not a full M3 cost model or held-out test.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

if __package__:
    from . import draft_graph_benchmark_report as _support
else:
    import analysis_support as _support

positive, nonnegative, near = _support.positive, _support.nonnegative, _support.near
signature, digest_valid = _support.signature, _support.digest_valid
DOCS = (271, 272, 288)
LENGTHS = (4096, 8192)
MODES = ('oldrisk', 'newrisk', 'fixed4', 'retrieval32_k4_graph')
EXPECTED = dict(calibration_reference_requests=2, calibration_prime_requests=2,
    calibration_profile_requests=4, reference_requests=6, correctness_requests=24,
    prime_requests=8, timing_requests=48, owners=8, calibration_owners=2)
ASSETS = dict(mapper_sha256=_support.ASSETS['mapper_sha256'],
    policy_sha256=_support.ASSETS['policy_sha256'],
    old_cost_sha256=_support.ASSETS['cost_sha256'])
REQUEST_FILES = dict(calibration_reference_requests='calibration_reference.jsonl',
    calibration_prime_requests='calibration_prime.jsonl',
    calibration_profile_requests='calibration_profiles.jsonl', reference_requests='reference.jsonl',
    correctness_requests='correctness.jsonl', prime_requests='prime.jsonl',
    timing_requests='runtime.jsonl')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_statistics(values, *, seed=20261020, draws=10000):
    """Bootstrap the document-paired mean; no pseudo-replication or n1 CI."""
    import numpy as np
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or draws != 10000:
        raise ValueError('Finite document values and frozen10000 draws required')
    result = dict(n=len(values), mean=None, ci95=None)
    if len(values):
        result['mean'] = float(values.mean())
    if len(values) >= 2:
        samples = np.random.default_rng(seed).choice(values, size=(draws, len(values)), replace=True).mean(1)
        result['ci95'] = [float(x) for x in np.percentile(samples, [2.5, 97.5])]
    return result


def pooled_document(rows):
    """Only a complete two-repeat, one-document cell is a performance sample."""
    if len(rows) != 2 or {r.get('repeat') for r in rows} != {0, 1}:
        raise ValueError('Exactly both repeats required; no imputation')
    if len({(r.get('source_index'), r.get('length'), r.get('mode')) for r in rows}) != 1:
        raise ValueError('Cannot pool different documents/contexts/arms')
    seconds, tokens = sum(r['outer_seconds'] for r in rows), sum(r['tokens'] for r in rows)
    if not positive(seconds) or not positive(tokens):
        raise ValueError('Positive saved outer time/tokens required')
    blocks, proposed = sum(r['blocks'] for r in rows), sum(r['proposed'] for r in rows)
    return dict(tps=tokens/seconds, tokens=tokens, outer_seconds=seconds,
        acceptance_length=sum(r['accepted'] for r in rows)/blocks,
        acceptance_rate=sum(r['accepted'] for r in rows)/proposed,
        blocks=blocks, proposed=proposed,
        cap_counts={str(cap):sum(t['cap'] == cap for r in rows for t in r['trace'])
                    for cap in (1, 2, 3, 4)},
        captures=sum(r['draft_runtime']['request']['captures'] for r in rows),
        capture_seconds=sum(r['draft_runtime']['request']['capture_host_seconds'] for r in rows),
        prefill_seconds=sum(r['prefill_seconds'] for r in rows)/2,
        decode_seconds=sum(r['decode_seconds'] for r in rows)/2,
        cleanup_seconds=sum(r['cleanup_seconds'] for r in rows)/2)


def calibration_summary(profiles):
    """Reconstruct the intervention using repeat0 only; repeat1 is validation."""
    fit, validation, problems = [], {}, []
    for length in LENGTHS:
        for repeat in (0, 1):
            group = [r for r in profiles if r.get('length') == length and r.get('repeat') == repeat]
            if len(group) != 1:
                problems.append(f'Calibration profile {length}/{repeat} missing or duplicated')
                continue
            row = group[0]; samples = row.get('cost_samples', {}).get('student_step', [])
            if not isinstance(samples, list) or not samples or not all(positive(v) for v in samples) or len(samples) != row.get('proposed'):
                problems.append(f'Calibration profile {length}/{repeat} invalid per-step samples')
                continue
            delta = row.get('draft_runtime', {}).get('request', {})
            if delta.get('captures') != 0 or delta.get('capture_host_seconds') != 0:
                problems.append(f'Calibration profile {length}/{repeat} includes capture cost')
            if repeat == 0:
                fit.extend(samples)
            else:
                validation[str(length)] = dict(sample_count=len(samples), median=statistics.median(samples))
    value = statistics.median(fit) if fit else None
    if positive(value):
        for key, row in validation.items():
            row['relative_error'] = abs(value-row['median'])/row['median']
            row['passed'] = row['relative_error'] <= .20
            if not row['passed']:
                problems.append(key+': repeat1 timing validation exceeds20%')
    if len(validation) != 2 or not positive(value):
        problems.append('Training-only calibration fit/validation incomplete')
    return dict(fit=dict(sample_count=len(fit), median=value), validation=validation), problems


def graph_errors(row, *, cold=False):
    errors = []; delta = row.get('draft_runtime', {}).get('request', {})
    stats = row.get('runtime_stats', {})
    if row.get('draft_runtime', {}).get('mode') != 'cuda_graph' or delta.get('replays') != row.get('draft_forwards') or not positive(delta.get('replays')):
        errors.append('Real Draft replay count differs from actual forwards')
    if any(type(delta.get(k)) is not int or delta[k] < 0 for k in ('captures', 'replays')) or not nonnegative(delta.get('capture_host_seconds')):
        errors.append('Invalid capture/replay counters')
    if cold and (not positive(delta.get('captures')) or not positive(delta.get('capture_host_seconds'))):
        errors.append('Fresh owner did not pay capture cost')
    if cold and any(not near(stats.get(k),delta.get(k)) for k in ('captures','replays','capture_host_seconds')):
        errors.append('Fresh owner borrowed pre-existing graph state')
    if delta.get('captures') == 0 and delta.get('capture_host_seconds') != 0:
        errors.append('Capture cost without a new graph')
    if positive(delta.get('captures')) and not positive(delta.get('capture_host_seconds')):
        errors.append('New capture cost missing')
    if type(stats.get('entries')) is not int or not 1 <= stats.get('entries', 0) <= 128 or stats.get('entries') != len(stats.get('graph_keys', [])) or stats.get('captures') != stats.get('entries'):
        errors.append('Lifetime Graph shapes/entries/captures differ')
    if positive(row.get('total_seconds')) and delta.get('capture_host_seconds', 0) > row['total_seconds']+1e-5:
        errors.append('Capture exceeds request interval')
    return errors


def quality_errors(row):
    """Audit observations without pretending different policies share states."""
    errors = []; observations = row.get('observed_rounds', [])
    if len(observations) != row.get('blocks'):
        errors.append('Missing every-round quality observations')
    reconstructed = []; feature_nonzero = False; value_nonzero = False
    for i, observed in enumerate(observations):
        if i >= len(row.get('trace', [])):
            continue
        trace = row['trace'][i]
        selected = observed.get('selected', [])
        if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected) or not all(type(x) is int and 0 <= x < observed.get('prefix', 0) for x in selected):
            errors.append('Invalid absolute selection observations')
        if any(observed.get(k) != trace.get(k) for k in ('round_index', 'prefix', 'accepted')):
            errors.append('Observed round disagrees with trace')
        draft, checks = observed.get('draft', []), observed.get('checks', [])
        if len(draft) != trace.get('proposed') or len(checks) != trace.get('proposed', -2)+1:
            errors.append('Missing saved proposal/Target checks')
        accepted = 0
        for a, b in zip(draft, checks):
            if a != b:
                break
            accepted += 1
        if accepted != observed.get('accepted'):
            errors.append('Saved acceptance is not matching prefix')
        if accepted < len(checks):
            reconstructed.extend(draft[:accepted]+[checks[accepted]])
        vector = observed.get('budget_features')
        if row.get('mode') in ('oldrisk', 'newrisk'):
            if vector is None and len(draft) < 2:
                pass
            elif not isinstance(vector, list) or len(vector) != 12 or not all(type(x) in (int, float) and math.isfinite(x) for x in vector):
                errors.append('Missing actual twelve-dimensional M3 feature')
            else:
                feature_nonzero |= vector[-1] > 0
        elif vector is not None:
            errors.append('M3-off fixed arm acquired M3 features')
        evidence = observed.get('injection_evidence', [])
        if not isinstance(evidence, list):
            errors.append('Malformed raw injection evidence'); evidence = []
        for item in evidence:
            working, native = item.get('working_value', {}), item.get('native_value', {})
            positions = item.get('positions', [])
            count = item.get('different_elements')
            valid = (item.get('layer') == 27 and isinstance(positions, list) and positions and
                len(set(positions)) == len(positions) and all(v in selected for v in positions) and
                digest_valid(working) and digest_valid(native) and working['shape'] == native['shape'] and
                working['dtype'] == native['dtype'] == 'torch.bfloat16' and
                len(working['shape'])==2 and working['shape'][0]==len(positions) and
                type(count) is int and 0<=count<=math.prod(working['shape']) and item.get('native_matches_memo') is True)
            if not valid:
                errors.append('Invalid actual working-V/native-V evidence')
            elif count > 0:
                value_nonzero = True
                if working['sha256'] == native['sha256']:
                    errors.append('Nonzero V difference contradicts identical digests')
            elif working['sha256']!=native['sha256']:
                errors.append('Zero V difference contradicts different digests')
        if row.get('mode') == 'retrieval32_k4_graph' and evidence:
            errors.append('Retrieval unexpectedly acquired injection evidence')
    if reconstructed[:row.get('tokens', 0)] != row.get('generated'):
        errors.append('Observed commits differ from canonical output')
    if row.get('mode') in ('oldrisk', 'newrisk') and not feature_nonzero:
        errors.append('Missing actual Mapper-summary M3 feature')
    if row.get('mode') != 'retrieval32_k4_graph' and (not value_nonzero or row.get('nonzero_injection') is not True):
        errors.append('Missing genuine nonzero Mapper evidence')
    if row.get('mode') == 'retrieval32_k4_graph' and row.get('nonzero_injection') is not False:
        errors.append('Retrieval arm unexpectedly injected Mapper information')
    queries = row.get('queries', [])
    if len(queries) != row.get('draft_forwards') or not queries or not all(type(q.get('logits')) is bool and type(q.get('summary_requested')) is bool and type(q.get('summary_count')) is int and q['summary_count']>=0 and digest_valid(q.get('query')) for q in queries):
        errors.append('Missing every-step Query digest')
    if row.get('mode') in ('fixed4', 'retrieval32_k4_graph') and (row.get('m3_disabled') is not True or any(q.get('summary_requested') is not False or q.get('summary_count') != 0 for q in queries)):
        errors.append('Fixed control secretly performed M3 summaries')
    if row.get('mode') in ('oldrisk','newrisk'):
        cursor=0
        for trace in row.get('trace',[]):
            for depth in range(trace.get('proposed',0)):
                if cursor>=len(queries) or queries[cursor].get('summary_requested') is not (depth<2) or queries[cursor].get('logits') is not True:
                    errors.append('Real risk policy first-two-summary schedule not preserved')
                cursor+=1
            if trace.get('accepted')==trace.get('proposed') and not trace.get('terminal'):
                if cursor>=len(queries) or queries[cursor].get('summary_requested') is not False or queries[cursor].get('logits') is not False:
                    errors.append('Nonterminal catchup Query/summary boundary changed')
                cursor+=1
        if cursor!=len(queries):errors.append('Query count cannot explain real proposal/catchup schedule')
    for role in ('native_t', 'native_s'):
        native = row.get(role, {}); lengths = native.get('lengths', []); layers = native.get('layers', [])
        if len(lengths) != 28 or len(layers) != 28:
            errors.append('Missing all28 Native layers: '+role)
            continue
        for length, layer in zip(lengths, layers):
            if type(length) is not int or length < row.get('length', 0)-1 or any(not digest_valid(layer.get(k)) or len(layer[k]['shape']) != 4 or layer[k]['shape'][2] != length for k in ('key', 'value')):
                errors.append('Invalid all-layer Native KV valid-span digest')
    return errors


def _analyze(directory):
    """Filled from the locked runner schema; invalid cells are never imputed."""
    directory = Path(directory); problems = []
    def read(name, default):
        try:
            value = json.loads((directory/name).read_text())
            if not isinstance(value, type(default)):
                raise ValueError('Wrong JSON container')
            return value
        except (OSError, ValueError):
            problems.append('Missing/malformed '+name); return default
    def rows(name):
        result = []
        try:
            lines = (directory/name).read_text().splitlines()
        except OSError:
            problems.append('Missing '+name); return result
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('Not an object')
                result.append(row)
            except ValueError:
                problems.append(f'Malformed {name}:{number}; valid rows preserved')
        return result
    data = {name:rows(filename) for name,filename in REQUEST_FILES.items()}
    owners, calibration_owners = rows('owners.jsonl'), rows('calibration_owners.jsonl')
    protocol = read('protocol.json', {}); calibration_protocol = read('protocol_calibration.json', {})
    old_cost, new_cost, seal = (read(name,{}) for name in ('cost_old.json','cost_new.json','COST_SEALED.json'))
    calibration, calibration_errors = calibration_summary(data['calibration_profile_requests'])
    problems.extend(calibration_errors)
    if protocol.get('source_indices') != list(DOCS) or protocol.get('lengths') != list(LENGTHS) or protocol.get('modes') != list(MODES) or protocol.get('repeats') != 2 or protocol.get('max_new_tokens') != 128 or protocol.get('expected_counts') != EXPECTED:
        problems.append('Registered94-request developer matrix differs')
    for key,value in ASSETS.items():
        if protocol.get(key) != value or calibration_protocol.get(key) != value:
            problems.append('Frozen asset pin differs: '+key)
    if any(protocol.get(k) is not True for k in ('mapper_frozen','policy_frozen','no_training')) or calibration_protocol.get('no_training') is not True:
        problems.append('Mapper/policy/no-training freeze declaration missing')
    environment=protocol.get('environment',{})
    if ('H100' not in environment.get('gpu_name','') or environment.get('total_memory',0)<80_000_000_000 or
        environment.get('target_dtype') != 'torch.float32' or environment.get('draft_dtype') != 'torch.bfloat16' or
        environment.get('prefix_dtype') != 'torch.bfloat16' or environment.get('tf32') is not False):
        problems.append('H10080GB/FP32Target/BF16Draft+prefix/TF32off environment differs')
    fixed=dict(selector='hierarchical',budget=512,recent=64,coverage_rows=64,coverage_batch=16,
        coverage_candidates=1024,coarse_chunk_size=32,refresh_every=4,refresh_policy='periodic',
        prefill_precision='bfloat16',
        max_new_tokens=128,max_draft=12,schedule='serial',profile=False,
        draft_execution='cuda_graph',coverage_execution='eager')
    configs=protocol.get('configs',{})
    for mode in MODES:
        config=configs.get(mode,{})
        if any(config.get(k)!=v for k,v in fixed.items()) or config.get('length_policy') != ('risk' if mode in ('oldrisk','newrisk') else 'fixed') or (mode not in ('oldrisk','newrisk') and config.get('fixed_k')!=4):
            problems.append(mode+': frozen configuration differs')
        injection=dict(inject_tokens=0,injection_strength=0.) if mode=='retrieval32_k4_graph' else dict(injection_layers=[27],inject_tokens=64,injection_strength=.5)
        if any(config.get(k)!=v for k,v in injection.items()):
            problems.append(mode+': Mapper enabled/disabled configuration differs')
    if configs.get('oldrisk') != configs.get('newrisk'):
        problems.append('Old/new risk differ beyond external student cost')
    if protocol.get('retrieval_top_chunks')!=32 or protocol.get('graph_max_entries')!=128:
        problems.append('Retrieval Top32 or private Graph entry limit differs')
    if protocol.get('policy_metadata') != calibration_protocol.get('policy_metadata') or not protocol.get('policy_metadata'):
        problems.append('Policy weights/hyperparameters changed after calibration')
    for current in (protocol,calibration_protocol):
        inventory=current.get('source_sha256',{})
        if not inventory:problems.append('Frozen source inventory missing')
        root=(directory/'source').resolve()
        for name,digest in inventory.items():
            path=(root/name).resolve()
            if not path.is_relative_to(root) or not path.is_file() or sha(path)!=digest:
                problems.append('Source hash mismatch: '+name)
    outer=directory.parent/'EVIDENCE_MANIFEST.json'
    if outer.is_file():
        try:
            evidence=json.loads(outer.read_text())['file_sha256'];root=directory.parent.resolve()
            for name,digest in evidence.items():
                path=(root/name).resolve()
                if not path.is_relative_to(root) or not path.is_file() or sha(path)!=digest:
                    problems.append('Immutable outer evidence hash mismatch: '+name)
        except (OSError,ValueError,TypeError,KeyError):problems.append('Malformed outer evidence manifest')
    if not old_cost or set(old_cost)!=set(new_cost) or any(old_cost[k]!=new_cost[k] for k in old_cost if k!='student_step'):
        problems.append('Recalibration altered fields other than student_step')
    if not positive(new_cost.get('student_step')) or new_cost.get('student_step') != calibration['fit']['median']:
        problems.append('New student_step is not exact pooled repeat0 median')
    if (directory/'cost_old.json').is_file() and sha(directory/'cost_old.json')!=ASSETS['old_cost_sha256']:
        problems.append('Old cost payload is not the frozen byte-exact input')
    fit_receipt=read('fit_receipt.json',{});seal_manifest=read('seal_manifest.json',{})
    sealed_names=('cost_old.json','cost_new.json','fit_receipt.json','COST_SEALED.json')
    sealed=seal_manifest.get('file_sha256',{})
    if set(sealed)!=set(sealed_names):problems.append('Incomplete immutable cost seal inventory')
    for name in sealed_names:
        if not (directory/name).is_file() or sealed.get(name)!=sha(directory/name):
            problems.append('Cost seal hash mismatch: '+name)
    if (seal_manifest.get('development_unopened') is not True or seal_manifest.get('seal_execution_index')!=8 or
        fit_receipt.get('unchanged_coefficients')!={k:v for k,v in old_cost.items() if k!='student_step'} or
        fit_receipt.get('fit_repeats')!=[0] or fit_receipt.get('validation_repeats')!=[1] or
        fit_receipt.get('calibration_file_sha256')!='d0bf671ea6e54f81533def13a812fb00f4b27d9199fbf2e0b1d2a8623c90d5ad' or
        fit_receipt.get('calibration_document_sha256')!='b0398d17c8500c6b24997927377e4a431042794a8a22aaaf56a9d7921ec16ad9' or
        fit_receipt.get('policy_metadata')!=protocol.get('policy_metadata') or
        seal!={**fit_receipt,'fit_receipt_sha256':sha(directory/'fit_receipt.json') if (directory/'fit_receipt.json').is_file() else None}):
        problems.append('Cost seal does not prove immutable single-coefficient fit before developer access')
    if (seal.get('seal_execution_index')!=8 or seal.get('old_sha256')!=ASSETS['old_cost_sha256'] or
        seal.get('new_sha256')!= (sha(directory/'cost_new.json') if (directory/'cost_new.json').is_file() else None) or
        seal.get('fit_receipt_sha256')!= (sha(directory/'fit_receipt.json') if (directory/'fit_receipt.json').is_file() else None) or
        seal.get('fit')!=calibration['fit'] or seal.get('validation')!=calibration['validation'] or
        seal.get('legacy_verify_coefficients_unchanged') is not True):
        problems.append('Cost seal does not bind independently reconstructed fit/validation')
    if protocol.get('calibration_seal_sha256') != (sha(directory/'seal_manifest.json') if (directory/'seal_manifest.json').is_file() else None):
        problems.append('Development protocol is not bound to prior cost seal')
    if protocol.get('new_cost_sha256')!=seal.get('new_sha256') or protocol.get('development_file_sha256')!='85b2eb1a81d1617b2d158c069c420c452459a8bdb90da128f257220cc27ded09':
        problems.append('New cost or registered tune cohort not frozen')
    if calibration_protocol.get('calibration_file_sha256')!='d0bf671ea6e54f81533def13a812fb00f4b27d9199fbf2e0b1d2a8623c90d5ad':
        problems.append('Registered train246 file not frozen')
    if (calibration_protocol.get('calibration_document_sha256')!='b0398d17c8500c6b24997927377e4a431042794a8a22aaaf56a9d7921ec16ad9' or
        calibration_protocol.get('development_content_opened') is not False or
        protocol.get('development_content_opened_after_seal') is not True or
        protocol.get('development_roles')!={str(d):'train' for d in DOCS}):
        problems.append('Calibration/developer corpus identity or prior-seal boundary differs')
    metadata_valid=not problems
    all_requests=sum(data.values(),[])
    indices=[r.get('execution_index') for r in all_requests];uuids=[r.get('request_id') for r in all_requests]
    order_valid=all(type(i) is int and 0<=i<94 for i in indices) and len(set(indices))==len(indices)
    if not order_valid or sorted(indices)!=list(range(94)):
        problems.append('Missing/duplicate94-request execution indices')
        if not order_valid:metadata_valid=False
    if any(not isinstance(i,str) or not i for i in uuids) or len(set(uuids))!=len(uuids):
        problems.append('Request UUID missing or reused');metadata_valid=False
    calibration_requests=sum((data[k] for k in ('calibration_reference_requests','calibration_prime_requests','calibration_profile_requests')),[])
    developer_requests=sum((data[k] for k in ('reference_requests','correctness_requests','prime_requests','timing_requests')),[])
    if order_valid and (any(r['execution_index']>=8 for r in calibration_requests) or any(r['execution_index']<8 for r in developer_requests)):
        problems.append('Developer request preceded the cost seal');metadata_valid=False
    references=read('references.json',{});refmap={};bad_cells=set()
    for label,values,docs in (('calibration',data['calibration_reference_requests'],(246,)),('developer',data['reference_requests'],DOCS)):
        cells=set()
        for row in values:
            key=row.get('source_index'),row.get('length');tokens=row.get('generated')
            if key not in {(d,n) for d in docs for n in LENGTHS} or key in cells or not isinstance(tokens,list) or not 0<len(tokens)<=128 or not all(type(v) is int and v>=0 for v in tokens):
                problems.append(label+': invalid/duplicate canonical AR cell');bad_cells.add(key)
            cells.add(key);refmap[key]=tokens
            if label=='developer' and references.get(f'{key[0]}:{key[1]}')!=tokens:
                problems.append('References JSON differs from independent AR');bad_cells.add(key)
        if cells!={(d,n) for d in docs for n in LENGTHS}:problems.append(label+': incomplete independent AR matrix')
    calibration_prime={r.get('length'):r for r in data['calibration_prime_requests']}
    if len(calibration_prime)!=2 or set(calibration_prime)!=set(LENGTHS):
        problems.append('Calibration cold-prime matrix incomplete');metadata_valid=False
    for row in data['calibration_prime_requests']+data['calibration_profile_requests']:
        errors=_support.row_errors(row,refmap.get((246,row.get('length'))))+graph_errors(row,cold=row in data['calibration_prime_requests'])
        base=calibration_prime.get(row.get('length'),{})
        if row.get('source_index')!=246 or row.get('mode')!='oldrisk' or row.get('performance_eligible') is not False:
            errors.append('Calibration profile leaked into developer timing')
        if row.get('policy_seed')!=protocol.get('policy_metadata',{}).get('seed'):
            errors.append('Calibration policy seed not frozen')
        if not row.get('timeline',{}).get('stages') or row.get('profile_enabled') is not True:
            errors.append('Calibration did not record actual instrumented engine costs')
        if row in data['calibration_profile_requests'] and ('outer_seconds' in row or 'outer_tps' in row or not positive(row.get('profile_outer_seconds'))):
            errors.append('Calibration profiles mislabeled as primary system timing')
        if signature(row)!=signature(base) or any(row.get(k)!=base.get(k) for k in ('tokens','accepted','proposed','blocks','draft_forwards','selector_refreshes')):
            errors.append('Calibration policy/trajectory changed between identical repeats')
        if errors:
            problems.extend('calibration: '+e for e in errors);metadata_valid=False
    qmap={};bad_quality=set();expected_quality={(m,d,n) for m in MODES for d in DOCS for n in LENGTHS}
    for row in data['correctness_requests']:
        key=row.get('mode'),row.get('source_index'),row.get('length')
        errors=_support.row_errors(row,refmap.get(key[1:]))+quality_errors(row)+graph_errors(row,cold=True)
        if key not in expected_quality or key in qmap or row.get('performance_eligible') is not False:
            errors.append('Duplicate/unregistered/performance-eligible quality cell')
        if row.get('mode') in ('fixed4','retrieval32_k4_graph') and any(t.get('cap')!=min(4,128-(t.get('prefix',0)-(row.get('length',0)-1))) for t in row.get('trace',[])):
            errors.append('Fixed4 arm performed a dynamic cap decision')
        if errors:problems.extend('quality '+str(key)+': '+e for e in errors);bad_quality.add(key)
        qmap[key]=row
    if set(qmap)!=expected_quality:problems.append('Developer24 quality cells incomplete')
    if len({r.get('owner_id') for r in data['correctness_requests']})!=24 or any(not r.get('owner_id') for r in data['correctness_requests']):
        problems.append('Quality owners missing or reused');bad_quality.update(expected_quality)
    warm_map={};invalid=set();prime_map={};bad_owners=set()
    def performance_errors(row, cold=False):
        key=row.get('mode'),row.get('source_index'),row.get('length')
        errors=_support.row_errors(row,refmap.get(key[1:]))+graph_errors(row,cold=cold)
        base=qmap.get(key,{})
        if key not in expected_quality or key in bad_quality or signature(row)!=signature(base) or any(row.get(k)!=base.get(k) for k in ('tokens','accepted','proposed','blocks','draft_forwards','selector_refreshes','projected_positions','static_affine_positions','selection_similarity_elements')):
            errors.append('Own-policy quality/work/trajectory differs from timing')
        if row.get('performance_eligible') is not True or 'timeline' in row or 'cost_samples' in row or not positive(row.get('outer_seconds')) or not near(row.get('outer_tps'),row.get('tokens',0)/row['outer_seconds'] if positive(row.get('outer_seconds')) else -1) or row.get('outer_seconds',0)+1e-6<row.get('total_seconds',0):
            errors.append('Primary timing contains profiling or excludes real generation')
        if row.get('policy_seed')!=protocol.get('policy_metadata',{}).get('seed'):
            errors.append('Per-request policy seed differs')
        return key,errors
    for row in data['prime_requests']:
        key,errors=performance_errors(row,cold=True);ownerkey=key[0],key[2]
        if key[1]!=271 or ownerkey in prime_map:errors.append('Cold owner did not uniquely prime doc271')
        if errors:problems.extend('prime '+str(key)+': '+e for e in errors);bad_owners.add(ownerkey)
        prime_map[ownerkey]=row
    for row in data['timing_requests']:
        key,errors=performance_errors(row);identity=(*key,row.get('repeat'))
        if row.get('repeat') not in (0,1) or identity in warm_map:errors.append('Missing/duplicate registered warm repeat')
        if errors:problems.extend('warm '+str(identity)+': '+e for e in errors);invalid.add(identity)
        warm_map[identity]=row
    expected_warm={(*key,r) for key in expected_quality for r in (0,1)}
    if set(warm_map)!=expected_warm:problems.append('Developer48 warm requests incomplete')
    order=[{k:r.get(k) for k in ('mode','source_index','length','repeat')} for r in data['timing_requests']]
    if protocol.get('warm_order')!=order:
        problems.append('Warm ordering differs from frozen interleaved schedule')
    for offset in range(0,len(order),4):
        block=order[offset:offset+4]
        if len(block)!=4 or {r['mode'] for r in block}!=set(MODES) or len({(r['source_index'],r['length'],r['repeat']) for r in block})!=1:
            problems.append('Warm requests not adjacent document-paired arm blocks');break
    if order_valid and data['correctness_requests'] and data['prime_requests'] and max(r['execution_index'] for r in data['correctness_requests'])>=min(r['execution_index'] for r in data['prime_requests']):
        problems.append('Timing started before all quality gates passed');metadata_valid=False
    service={};owner_map={};quality_owners={r.get('owner_id') for r in data['correctness_requests']}
    all_calibration_ids={r.get('owner_id') for r in calibration_requests if r.get('owner_id')}
    for owner in owners:
        key=owner.get('mode'),owner.get('length');prime=prime_map.get(key,{})
        warm=sorted([r for r in data['timing_requests'] if (r.get('mode'),r.get('length'))==key],key=lambda r:r.get('execution_index',-1))
        errors=[];samples=[prime]+warm
        if key not in {(m,n) for m in MODES for n in LENGTHS} or key in owner_map or not owner.get('owner_id') or owner['owner_id'] in quality_owners|all_calibration_ids:
            errors.append('Private owner reused quality/calibration state or is unregistered')
        if len(warm)!=6 or owner.get('prime_request_id')!=prime.get('request_id') or owner.get('warm_request_ids')!=[r.get('request_id') for r in warm]:
            errors.append('Persistent owner UUID matrix incomplete')
        if not owner.get('owner_signature') or any(r.get('owner_id')!=owner.get('owner_id') or r.get('owner_signature')!=owner.get('owner_signature') for r in samples):
            errors.append('Persistent Native/Working/Graph owner changed')
        previous=dict(captures=0,replays=0,capture_host_seconds=0.)
        for row in samples:
            stats=row.get('runtime_stats',{});delta=row.get('draft_runtime',{}).get('request',{})
            if any(not near(stats.get(k),previous[k]+delta.get(k,-1)) for k in previous):errors.append('Lifetime Graph counters are discontinuous')
            previous={k:stats.get(k,-1) for k in previous}
        constructor,close=owner.get('constructor_and_detach_seconds'),owner.get('close_seconds')
        active=(constructor if nonnegative(constructor) else 0)+prime.get('outer_seconds',0)+sum(r.get('outer_seconds',0) for r in warm)+(close if nonnegative(close) else 0)
        tokens=prime.get('tokens',0)+sum(r.get('tokens',0) for r in warm)
        checks=dict(cold_prime_outer_seconds=prime.get('outer_seconds'),warm_outer_seconds_sum=sum(r.get('outer_seconds',0) for r in warm),prime_tokens=prime.get('tokens'),warm_tokens=sum(r.get('tokens',0) for r in warm),accounted_service_seconds=active,accounted_service_tokens=tokens,accounted_service_tps=tokens/active if positive(active) else -1)
        if any(not near(owner.get(k),v) for k,v in checks.items()) or not nonnegative(constructor) or not nonnegative(close):errors.append('Constructor/cold/warm/close accounting differs')
        if not near(prime.get('cold_complete_outer_seconds'),(constructor or 0)+prime.get('outer_seconds',0)):errors.append('Cold-complete generation cost excludes constructor')
        closed=owner.get('closed_runtime_stats',{})
        if owner.get('engine_closed') is not True or owner.get('hook_restored') is not True or closed.get('entries')!=0 or closed.get('graph_keys')!=[] or owner.get('final_runtime_stats')!=(warm[-1].get('runtime_stats') if warm else None) or any(not near(closed.get(k),previous[k]) for k in previous):errors.append('Owner graph cleanup/final counters not proved')
        if errors:problems.extend('owner '+str(key)+': '+e for e in errors);bad_owners.add(key)
        elif key not in bad_owners:service[key]=dict(cold_tps=prime['tokens']/(constructor+prime['outer_seconds']),accounted_tps=tokens/active,constructor_seconds=constructor,cold_prime_seconds=prime['outer_seconds'],close_seconds=close)
        owner_map[key]=owner
    if len(owner_map)!=8 or len({o.get('owner_id') for o in owners})!=8:problems.append('Eight private developer owners incomplete/reused')
    bad_owners.update({(m,n) for m in MODES for n in LENGTHS}-set(owner_map))
    for length in LENGTHS:
        group=[o for o in owners if o.get('length')==length]
        for cache in ('native_t','native_s','working'):
            pointers=[o.get('owner_signature',{}).get(cache,{}).get('key_ptr') for o in group]
            if any(type(v) is not int or v<=0 for v in pointers) or len(set(pointers))!=len(pointers):
                problems.append(str(length)+': simultaneous developer owners share '+cache+' storage')
                bad_owners.update((m,length) for m in MODES)
    # The calibration ledger is separate; its instrumented profile intervals
    # are never admitted to developer performance or serving-accounting sums.
    if len(calibration_owners)!=2 or {o.get('length') for o in calibration_owners}!=set(LENGTHS):
        problems.append('Two calibration owner lifecycle receipts incomplete')
    for owner in calibration_owners:
        errors=[];length=owner.get('length');prime=calibration_prime.get(length,{})
        profiles=sorted([r for r in data['calibration_profile_requests'] if r.get('length')==length],key=lambda r:r.get('repeat',-1))
        if (owner.get('calibration_cost_only') is not True or owner.get('prime_request_id')!=prime.get('request_id') or
            owner.get('profile_request_ids')!=[r.get('request_id') for r in profiles] or len(profiles)!=2 or
            owner.get('owner_id') in quality_owners or owner.get('owner_id') in {o.get('owner_id') for o in owners}):
            errors.append('Calibration owner UUID/scope linked to developer data')
        samples=[prime]+profiles
        if not owner.get('owner_signature') or any(r.get('owner_id')!=owner.get('owner_id') or r.get('owner_signature')!=owner.get('owner_signature') for r in samples):
            errors.append('Calibration private caches/Graph ownership changed')
        previous=dict(captures=0,replays=0,capture_host_seconds=0.)
        for row in samples:
            delta=row.get('draft_runtime',{}).get('request',{});stats=row.get('runtime_stats',{})
            if any(not near(stats.get(k),previous[k]+delta.get(k,-1)) for k in previous):errors.append('Calibration lifetime counters discontinuous')
            previous={k:stats.get(k,-1) for k in previous}
        if (not near(owner.get('cold_prime_profile_seconds'),prime.get('measured_generate_seconds')) or
            not near(owner.get('warm_profile_seconds_sum'),sum(r.get('profile_outer_seconds',0) for r in profiles)) or
            not nonnegative(owner.get('constructor_and_detach_seconds')) or not nonnegative(owner.get('close_seconds'))):
            errors.append('Calibration instrumented cost ledger differs')
        closed=owner.get('closed_runtime_stats',{})
        if (owner.get('engine_closed') is not True or owner.get('hook_restored') is not True or
            closed.get('entries')!=0 or closed.get('graph_keys')!=[] or
            owner.get('final_runtime_stats')!=(profiles[-1].get('runtime_stats') if profiles else None) or
            any(not near(closed.get(k),previous[k]) for k in previous)):
            errors.append('Calibration owner graph cleanup not proved')
        if errors:problems.extend('calibration owner: '+e for e in errors);metadata_valid=False
    observed={**{name:len(value) for name,value in data.items()},'owners':len(owners),'calibration_owners':len(calibration_owners)}
    complete=read('COMPLETE.json',{})
    if observed!=EXPECTED or complete.get('passed') is not True or complete.get('counts')!=EXPECTED or complete.get('logical_requests')!=94 or complete.get('sources_unchanged') is not True or complete.get('assets_unchanged') is not True or complete.get('cost_seal_unchanged') is not True or complete.get('calibration_seal_sha256')!=protocol.get('calibration_seal_sha256'):
        problems.append('Missing/inconsistent94-request completion and unchanged-source/assets receipt')
    if (directory/'FAILURE.json').is_file():problems.append('Runner failure marker exists')
    usable={};metrics={};comparisons={}
    for length in LENGTHS:
        docs=[]
        for doc in DOCS:
            if metadata_valid and (doc,length) not in bad_cells and all((m,doc,length) in qmap and (m,doc,length) not in bad_quality and (m,length) not in bad_owners and all((m,doc,length,r) in warm_map and (m,doc,length,r) not in invalid for r in (0,1)) for m in MODES):docs.append(doc)
        usable[str(length)]=docs;metrics[str(length)]={};comparisons[str(length)]={}
        pooled={}
        for mode in MODES:
            cells={doc:pooled_document([warm_map[mode,doc,length,r] for r in (0,1)]) for doc in docs}
            pooled[mode]=cells
            if not cells:continue
            tps=paired_statistics([c['tps'] for c in cells.values()]);acc=paired_statistics([c['acceptance_length'] for c in cells.values()])
            value=dict(documents=len(cells),per_document={str(k):v for k,v in cells.items()},
                mean_document_tps=tps['mean'],mean_document_tps_ci95=tps['ci95'],
                pooled_tps=sum(c['tokens'] for c in cells.values())/sum(c['outer_seconds'] for c in cells.values()),
                acceptance_length=acc['mean'],acceptance_length_ci95=acc['ci95'],
                acceptance_rate=sum(c['acceptance_rate'] for c in cells.values())/len(cells),
                cap_counts={str(k):sum(c['cap_counts'][str(k)] for c in cells.values()) for k in (1,2,3,4)},
                warm_captures=sum(c['captures'] for c in cells.values()),
                warm_capture_seconds=sum(c['capture_seconds'] for c in cells.values()),
                **{k:sum(c[k] for c in cells.values())/len(cells) for k in ('prefill_seconds','decode_seconds','cleanup_seconds')},
                **service.get((mode,length),{}))
            metrics[str(length)][mode]=value
        for control in ('oldrisk','fixed4','retrieval32_k4_graph'):
            if not docs:continue
            percent=[100*(pooled['newrisk'][d]['tps']/pooled[control][d]['tps']-1) for d in docs]
            delta=[pooled['newrisk'][d]['tps']-pooled[control][d]['tps'] for d in docs]
            acceptance=[pooled['newrisk'][d]['acceptance_length']-pooled[control][d]['acceptance_length'] for d in docs]
            comparisons[str(length)][control]=dict(relative_percent=paired_statistics(percent),tps_delta=paired_statistics(delta),acceptance_delta=paired_statistics(acceptance),
                strict_speed_gate=len(docs)==3 and paired_statistics(percent)['mean']>0 and paired_statistics(percent)['ci95'][0]>0)
    result = dict(status='COMPLETE' if not problems else 'PARTIAL', problems=problems,
        observed_counts=observed, calibration=calibration, metrics=metrics, comparisons=comparisons, diagnostics={},
        usable_documents=usable,
        scope='Development only: three previously used tune documents from original train corpus; not held-out,4090,Vicuna or best-baseline proof',
        documents=3, bootstrap_draws=10000,
        raw_sha256={p.name:sha(p) for p in directory.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl')})
    return result


def analyze(directory):
    """Malformed nested records fail closed, while still allowing PARTIAL plots."""
    try:
        return _analyze(directory)
    except (TypeError,KeyError,AttributeError,ValueError,ZeroDivisionError,OverflowError) as error:
        path=Path(directory)
        return dict(status='PARTIAL',problems=['Malformed nested evidence; no performance admitted: '+type(error).__name__+': '+str(error)],
            observed_counts={},calibration={},metrics={},comparisons={},diagnostics={},
            usable_documents={str(n):[] for n in LENGTHS},documents=3,bootstrap_draws=10000,
            scope='Incomplete or malformed developer evidence only; no held-out/full-cost-model/SpecExtend success claim',
            raw_sha256={p.name:sha(p) for p in path.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl')} if path.is_dir() else {})


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors = ['#62798c','#2c8b7b','#c69440','#9b6b9e']
    labels = ['Old cost risk','New step cost risk','Mapper + fixed4','Retrieval32/k4 Graph']
    header = f"{summary['status']} · H10080GB ·3 reused train-tune documents ·4K/8K ·128 output limit ·2 repeats"
    names = []
    def save(fig, name):
        fig.suptitle(header,fontsize=12);fig.tight_layout(rect=(0,.04,1,.95))
        fig.text(.5,.01,'Development only. Where shown: document bootstrap95% CI,10,000 draws. Calibration/cold/service are descriptive.',ha='center',fontsize=9)
        fig.savefig(output/(name+'.png'),dpi=170);fig.savefig(output/(name+'.svg'));plt.close(fig);names.append(name)
    for metric,name,ylabel in (('acceptance_length','acceptance','Accepted Draft tokens / verification round'),('mean_document_tps','tps','Whole-generate outer tokens/s')):
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for column,length in enumerate(LENGTHS):
            ax=axes[0,column]
            available=summary.get('metrics',{}).get(str(length),{})
            for i,mode in enumerate(MODES):
                cell=available.get(mode)
                if cell:
                    value=cell[metric];ci=cell.get(metric+'_ci95')
                    err=None if ci is None else np.array([[value-ci[0]],[ci[1]-value]])
                    ax.bar(i,value,color=colors[i],yerr=err,capsize=3)
            ax.set_xticks(range(4),labels,rotation=15,ha='right');ax.set_ylabel(ylabel);ax.set_title(f'{length} exact input tokens ·n={len(summary["usable_documents"][str(length)])}');ax.set_ylim(bottom=0)
            ax=axes[1,column]
            for i,control in enumerate(('oldrisk','fixed4','retrieval32_k4_graph')):
                cell=summary.get('comparisons',{}).get(str(length),{}).get(control,{})
                effect=cell.get('relative_percent' if name=='tps' else 'acceptance_delta',{})
                value,ci=effect.get('mean'),effect.get('ci95')
                if value is not None:
                    err=None if ci is None else np.array([[value-ci[0]],[ci[1]-value]])
                    ax.bar(i,value,color=[colors[0],colors[2],colors[3]][i],yerr=err,capsize=3)
            ax.axhline(0,color='black',linewidth=.7)
            ax.set_xticks(range(3),['vs old risk','vs Mapper fixed4','vs retrieval Graph'],rotation=15,ha='right')
            ax.set_title(f'{length} ·New risk paired document effects')
            ax.set_ylabel('TPS relative change (%)' if name=='tps' else 'Accepted Draft/round difference')
        save(fig,name)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    fit=summary.get('calibration',{}).get('fit',{}).get('median')
    for ax,length in zip(axes[0],LENGTHS):
        validation=summary.get('calibration',{}).get('validation',{}).get(str(length),{})
        if positive(fit) and positive(validation.get('median')):
            ax.bar([0,1],[1000*fit,1000*validation['median']],color=['#2c8b7b','#c69440'])
            ax.set_title(f'{length} ·validation relative error {100*validation["relative_error"]:.2f}%')
        ax.set_xticks([0,1],['Pooled repeat0 fit','Repeat1 validation']);ax.set_ylabel('Instrumented student-step median (ms)')
    for ax,length in zip(axes[1],LENGTHS):
        cells=summary.get('metrics',{}).get(str(length),{})
        for i,mode in enumerate(MODES):
            cell=cells.get(mode)
            if cell:
                ax.bar(i,cell.get('warm_captures',0),color=colors[i])
        ax.set_xticks(range(4),labels,rotation=15,ha='right');ax.set_title(f'{length} ·All legal warm captures included');ax.set_ylabel('New warm Graph captures')
    save(fig,'calibration_capture')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for column,length in enumerate(LENGTHS):
        cells=summary.get('metrics',{}).get(str(length),{})
        for i,mode in enumerate(MODES):
            cell=cells.get(mode)
            if not cell:continue
            counts=cell.get('cap_counts',{});total=sum(counts.values())
            if total:
                axes[0,column].bar(i,sum(v for k,v in counts.items() if k=='4')/total,color=colors[i])
            for j,(field,label) in enumerate((('cold_tps','Cold complete'),('accounted_tps','Accounted service'))):
                value=cell.get(field)
                if positive(value):axes[1,column].bar(i+(j-.5)*.34,value,width=.32,color=colors[i],alpha=1. if j==0 else .45,label=label if i==0 else None)
        axes[0,column].set_title(f'{length} ·Actual cap4 fraction');axes[0,column].set_ylabel('Fraction of verification rounds');axes[0,column].set_ylim(0,1)
        axes[1,column].set_title(f'{length} ·Constructor/cold/service costs');axes[1,column].set_ylabel('Tokens/s (measured non-contiguous service sum)')
        if cells:axes[1,column].legend(fontsize=8)
        for ax in axes[:,column]:ax.set_xticks(range(4),labels,rotation=15,ha='right')
    save(fig,'actions_service')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,length in zip(axes,LENGTHS):
        cells=summary.get('metrics',{}).get(str(length),{})
        for i,mode in enumerate(MODES):
            cell=cells.get(mode)
            if not cell:continue
            bottom=0.
            for field,color,label in (('prefill_seconds','#62798c','Prefill'),('decode_seconds','#2c8b7b','Decode'),('cleanup_seconds','#c69440','Cleanup')):
                value=cell.get(field,0);ax.bar(i,value,bottom=bottom,color=color,label=label if i==0 else None);bottom+=value
        ax.set_xticks(range(4),labels,rotation=15,ha='right');ax.set_title(f'{length} ·Unprofiled engine stage means');ax.set_ylabel('Seconds / request')
        if cells:ax.legend(fontsize=8)
    save(fig,'latency')
    return names


def write_report(directory, output):
    directory,output=Path(directory),Path(output)
    if output.exists():raise FileExistsError('Fresh analysis output required')
    output.mkdir(parents=True);summary=analyze(directory);figures=plot(summary,output)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    shutil.copyfile(__file__,output/'analysis_source.py');shutil.copyfile(_support.__file__,output/'analysis_support.py')
    lines=['# '+summary['status']+': Draft student-step cost intervention','',summary['scope'],'',
        'Only student_step changes. Verify coefficients, fixed-round and cache-update costs remain historical approximations. Instrumented student-step timing does not include the policy network, topk or feature construction; this is not a full M3 cost calibration.',
        'Frozen nonzero V Mapper and StepBudget weights/hyperparameters; FP32 Target,BF16 Draft/prefix,TF32 disabled. Fixed4 keeps Mapper but M3 is genuinely off; retrieval keeps no Mapper.',
        'Warm primary = per-document pooled tokens/full-generate outer seconds over two repeats. Graph capture costs are included, not censored. All calibration profiles are excluded from developer TPS. Cold and accounted active service are separate non-contiguous measured sums, not production wall throughput.',
        'Per-context confidence intervals use three document-paired units,10,000 bootstrap draws. This previously used train-tune cohort is not held-out; small-sample intervals do not establish broad superiority or the best SpecExtend baseline.','',
        '| Context | Mode | Mean document TPS | Accepted Draft/round | Cold TPS | Accounted TPS |','|---|---|---:|---:|---:|---:|']
    for length in LENGTHS:
        for mode,cell in summary['metrics'].get(str(length),{}).items():
            def fmt(v):return f'{v:.4f}' if type(v) in (int,float) else 'unavailable'
            lines.append(f'| {length} | {mode} | {fmt(cell.get("mean_document_tps"))} | {fmt(cell.get("acceptance_length"))} | {fmt(cell.get("cold_tps"))} | {fmt(cell.get("accounted_tps"))} |')
    lines+=['','## Paired developer effects','',
        'Each effect first pools both repeats within each document, then averages the three paired document effects. These are not ratios of cross-document pooled TPS. A positive lower95% bound is only a developer gate, not independent confirmation.','',
        '| Input | New risk versus | Mean TPS change | Paired95% CI | Point+lowerCI positive, n3 |',
        '|---|---|---:|---|---|']
    for length in LENGTHS:
        for control,cell in summary['comparisons'].get(str(length),{}).items():
            effect=cell['relative_percent'];interval=effect['ci95']
            label='unavailable' if interval is None else f'[{interval[0]:+.3f}%, {interval[1]:+.3f}%]'
            lines.append(f'| {length} | {control} | {effect["mean"]:+.3f}% | {label} | {cell["strict_speed_gate"]} |')
    lines+=['','Calibration repeat0 pooled student median: `'+str(summary.get('calibration',{}).get('fit',{}).get('median'))+'` seconds. Repeat1 is a same-document repeatability check only; it never enters the fit.']
    lines+=['','## Audit issues','']+(['- '+v for v in summary['problems']] or ['None. The registered developer experiment evidence is complete; the larger research objective is not thereby complete.'])
    lines+=['','## Figures and reproducible evidence','']
    lines+=['- ['+name+']('+name+'.png) ([SVG]('+name+'.svg))' for name in figures]
    lines+=['- [Runnable analysis source](analysis_source.py)','- [Saved standalone support](analysis_support.py), SHA256 `'+sha(output/'analysis_support.py')+'`.',
        '- [Audit and raw SHA summary](summary.json)','- Re-run: `python analysis_source.py .. --output ../analysis_rerun` (fresh directory).']
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix in ('.json','.jsonl'):lines.append('- ['+path.name+'](../'+path.name+')')
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (output/'ANALYSIS_COMPLETE.json').write_text(json.dumps(dict(status=summary['status'],
        artifacts_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()},
        rendered_inspection_required=True,scope='Saved-data analysis only; not a model run or held-out scientific success'),indent=2)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=write_report(args.directory,args.output)
    print(json.dumps(dict(status=result['status'],problems=result['problems'],comparisons=result['comparisons']),indent=2))


if __name__ == '__main__':main()
