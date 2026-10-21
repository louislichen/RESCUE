"""Saved-record audit and three figures for the doc303 matched gap pilot.

This CPU analysis never loads a model. n=1 document cannot support a document
confidence interval, a held-out claim, or completing the larger SpecExtend goal.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics


LENGTHS = (4096, 8192)
ARMS = ('rescue_graph', 'retrieval32_k4_eager', 'retrieval32_k4_resident',
        'retrieval32_k4_graph', 'full32_n8')
QUALITY_ARMS = ('rescue_eager',)+ARMS
EXPECTED = dict(reference_requests=2, correctness_requests=12, prime_requests=10,
                timing_requests=20, owners=10)
PINS = dict(mapper_sha256='d08889e796ca2527e23241ff34a072ef640e2fdf3fe9a2f9ba5439975af04d9f',
            policy_sha256='3c333a470119c236afcc1308c18592842a535bee0b8f83c4fb4da359ff17eff5',
            cost_sha256='d92c843a84676c0cd6adaf9364bbf7e8c771d56548f1016b5720a3e04f371fa1')
TRACE = ('round_index', 'prefix', 'cap', 'proposed', 'accepted', 'terminal', 'workset_tokens')
TREE = dict(nodes=8, top_chunks=32, max_depth=10, threshold=.7,
            chunk_size=32, refresh_every=4, flash_prefill=True, hybrid_tree=True)
FULL_FLAGS = dict(target_use_flash_prefill=True, target_use_hybrid_tree_attn=True,
    draft_use_flash_prefill=True, use_retrieval_cache=True, retrieve_top_k=32,
    retrieval_chunk_size=32, retrieve_every_n_steps=4)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def positive(value):
    return nonnegative(value) and value > 0


def near(a, b):
    return nonnegative(a) and nonnegative(b) and math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-8)


def signature(row):
    return [[v.get(k) for k in TRACE] for v in row.get('trace', [])]


def digest_valid(value):
    return isinstance(value, dict) and isinstance(value.get('dtype'), str) and isinstance(value.get('shape'), list) and all(type(v) is int and v > 0 for v in value['shape']) and isinstance(value.get('sha256'), str) and len(value['sha256']) == 64 and all(v in '0123456789abcdef' for v in value['sha256'])


def row_errors(row, reference):
    issues = []
    if row.get('generated') != reference or row.get('output_agrees') is not True or not isinstance(reference, list) or not 0 < len(reference) <= 128 or row.get('tokens') != len(reference):
        issues.append('Canonical client-token equality/count mismatch')
    if row.get('source_index') != 303 or row.get('length') not in LENGTHS:
        issues.append('Unregistered document/context')
    if row.get('mode') == 'full32_n8':
        upstream = row.get('upstream', {})
        scored = row.get('scored_generated_before_eos', [])
        lengths = upstream.get('accept_length_list', [])
        raw, overshoot = upstream.get('raw_total_generated'), upstream.get('generation_overshoot')
        if not isinstance(scored, list) or not 0 < len(scored) <= 128 or upstream.get('output_token_ids') != scored or row.get('generated') != scored[:row.get('tokens', 0)]:
            issues.append('Full-tree scored-prefix evidence mismatch')
        if not isinstance(lengths, list) or not lengths or not all(type(v) is int and v >= 0 for v in lengths):
            issues.append('Missing/invalid raw full-tree accepted-length list')
        elif row.get('blocks') != len(lengths) or row.get('accepted') != sum(lengths) or raw != sum(lengths)+len(lengths):
            issues.append('Full-tree raw accepted/round/commit accounting mismatch')
        if type(raw) is not int or type(overshoot) is not int or raw < len(scored) or overshoot != raw-len(scored) or upstream.get('total_generated') != len(scored) or row.get('raw_total_generated') != raw or row.get('generation_overshoot') != overshoot:
            issues.append('Full-tree raw count/overshoot/scored accounting mismatch')
        if row.get('raw_tail_ids_available') is not False or row.get('tree_policy') != TREE or row.get('observed_full_config') != FULL_FLAGS:
            issues.append('Full-tree provenance/policy/observed Flash-hybrid flags mismatch')
        if row.get('proposed') is not None:
            issues.append('Full-tree proposal count/rate must not be invented')
    else:
        trace = row.get('trace', [])
        if not trace or not all(isinstance(v, dict) for v in trace) or row.get('blocks') != len(trace):
            return issues+['Missing/inconsistent linear round trace']
        if any(type(v.get(k)) is not int for v in trace for k in ('prefix', 'cap', 'accepted', 'proposed')):
            return issues+['Noninteger linear trace counts']
        if row.get('accepted') != sum(v['accepted'] for v in trace) or row.get('proposed') != sum(v['proposed'] for v in trace):
            issues.append('Linear accepted/proposed trace totals mismatch')
        for i, value in enumerate(trace):
            if not 1 <= value['proposed'] <= value['cap'] <= 4 or not 0 <= value['accepted'] <= value['proposed']:
                issues.append('Invalid linear2/4 proposal or acceptance')
            if value.get('round_index') != i or value.get('terminal') is not (i == len(trace)-1):
                issues.append('Linear round/terminal mismatch')
            if i == 0 and value['prefix'] != row.get('length', 0)-1:
                issues.append('Initial native prefix is not input length minus one')
            if i and value['prefix'] != trace[i-1]['prefix']+trace[i-1]['accepted']+1:
                issues.append('Committed prefix progression mismatch')
        for field in ('prefill_seconds', 'decode_seconds'):
            if not positive(row.get(field)):
                issues.append('Missing/invalid linear '+field)
        if not nonnegative(row.get('cleanup_seconds')):
            issues.append('Invalid linear cleanup_seconds')
    if not positive(row.get('total_seconds')) or not near(row.get('total_tps'), row.get('tokens', 0)/row['total_seconds'] if positive(row.get('total_seconds')) else -1):
        issues.append('Invalid engine whole-request time/TPS')
    return issues


def quality_errors(row):
    issues = []
    queries = row.get('queries', [])
    if not queries or len(queries) != row.get('draft_forwards') or any(not digest_valid(v.get('query')) for v in queries):
        issues.append('Missing every-step Query hash evidence')
    for role in ('native_t', 'native_s'):
        cache = row.get(role, {}); lengths, layers = cache.get('lengths', []), cache.get('layers', [])
        if len(lengths) != 28 or len(layers) != 28:
            issues.append('Missing all28 valid Native KV layers: '+role); continue
        for n, layer in zip(lengths, layers):
            if type(n) is not int or n < row['length']-1 or any(not digest_valid(layer.get(k)) or len(layer[k]['shape']) != 4 or layer[k]['shape'][2] != n for k in ('key', 'value')):
                issues.append('Invalid Native KV valid-span digest: '+role)
    rounds = row.get('observed_rounds', [])
    if len(rounds) != row.get('blocks'):
        issues.append('Missing full-round observer evidence')
    emitted, peaks = [], []
    for i, obs in enumerate(rounds):
        if i >= len(row.get('trace', [])):
            continue
        trace = row['trace'][i]
        if any(obs.get(k) != trace.get(k) for k in ('round_index','prefix','accepted')) or len(obs.get('draft', [])) != trace.get('proposed') or len(obs.get('checks', [])) != trace.get('proposed', -1)+1:
            issues.append('Observer/linear trace mismatch')
        selected = obs.get('selected', [])
        if not selected or not all(type(v) is int and 0 <= v < obs.get('prefix', 0) for v in selected) or len(set(selected)) != len(selected):
            issues.append('Invalid absolute selected-position evidence')
        draft, checks = obs.get('draft', []), obs.get('checks', [])
        accepted = 0
        for a, b in zip(draft, checks):
            if a != b:
                break
            accepted += 1
        if accepted != obs.get('accepted'):
            issues.append('Observer acceptance is not the saved token-prefix match')
        if accepted < len(checks):
            emitted.extend(draft[:accepted]+[checks[accepted]])
        features = obs.get('budget_features')
        if row.get('mode', '').startswith('rescue'):
            if features is None and len(draft) < 2:
                continue
            if not isinstance(features, list) or len(features) != 12 or not all(type(v) in (int,float) and math.isfinite(v) for v in features):
                issues.append('Missing valid causal12 features after two RESCUE Draft tokens')
            else:
                peaks.append(features[-1])
        elif features is not None:
            issues.append('Retrieval was given a learned controller feature vector')
    if emitted[:row.get('tokens',0)] != row.get('generated'):
        issues.append('Saved observer commits differ from client output')
    if row.get('mode', '').startswith('rescue'):
        if row.get('nonzero_injection') is not True or not peaks or max(peaks) <= 0:
            issues.append('Missing actual nonzero Mapper correction')
    elif row.get('nonzero_injection') is not False:
        issues.append('Retrieval baseline must have no Mapper correction')
    if row.get('mode', '').endswith('graph') and not all(row.get('runtime_stats',{}).get(k,0) > 0 for k in ('captures','replays')):
        issues.append('Quality did not exercise real Graph capture/replay')
    return issues


def analyze(directory):
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
        if not (directory/name).is_file():
            problems.append('Missing '+name); return []
        values = []
        for line_no, line in enumerate((directory/name).read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError('Nonobject row')
                values.append(value)
            except ValueError:
                problems.append(f'Malformed {name}:{line_no}; valid earlier rows retained')
        return values

    protocol = read('protocol.json', {})
    references = read('references.json', {})
    canonical, quality, primes, runtime, owners, contexts = [rows(name) for name in (
        'reference.jsonl','correctness.jsonl','prime.jsonl','runtime.jsonl','owners.jsonl','contexts.jsonl')]
    if protocol.get('source_index') != 303 or protocol.get('lengths') != list(LENGTHS) or protocol.get('max_new_tokens') != 128 or protocol.get('repeats') != 2 or protocol.get('arms') != list(ARMS) or protocol.get('quality_arms') != list(QUALITY_ARMS) or protocol.get('expected_counts') != EXPECTED or protocol.get('expected_requests') != 44:
        problems.append('Registered44-request/doc303/context/output matrix mismatch')
    if protocol.get('tree') != TREE or protocol.get('retrieval') != dict(top_chunks=32,chunk_size=32,fixed_k=4,refresh_every=4,mapper=False):
        problems.append('Frozen retrieval/tree algorithm differs')
    if protocol.get('graph_max_entries') != 128 or protocol.get('rescue_retain_query') is not True or protocol.get('retrieval_retain_query') != dict(correctness=True,timing=False):
        problems.append('Graph entry budget/Query-retention fairness differs')
    if any(protocol.get(k) != v for k,v in PINS.items()) or any(protocol.get(k) is not True for k in ('no_training','mapper_frozen','policy_frozen')):
        problems.append('Frozen Mapper/StepBudget/cost or no-training declaration differs')
    assets = protocol.get('assets_sha256', {})
    for name,key in (('full_final.pt','mapper_sha256'),('budget_selected.pt','policy_sha256'),('cost.json','cost_sha256')):
        if [v for p,v in assets.items() if Path(p).name == name] != [PINS[key]]:
            problems.append('Asset inventory does not bind '+name)
    config = protocol.get('config', {})
    fixed = dict(selector='hierarchical',coverage_rows=64,refresh_every=4,injection_layers=[27],
        inject_tokens=64,injection_strength=.5,prefill_precision='bfloat16',max_new_tokens=128,
        max_draft=12,length_policy='risk',schedule='serial',profile=False)
    if any(config.get(k) != v for k,v in fixed.items()):
        problems.append('RESCUE frozen algorithm/config differs')
    env = protocol.get('environment', {})
    if 'H100' not in env.get('gpu_name','') or env.get('total_memory',0) < 80_000_000_000 or env.get('target_dtype') != 'torch.float32' or env.get('draft_dtype') != 'torch.bfloat16' or env.get('prefix_dtype') != 'torch.bfloat16' or env.get('tf32') is not False:
        problems.append('H10080GB/FP32Target/BF16Draft+prefill/TF32off mismatch')
    inventory = protocol.get('source_sha256', {}); source = (directory/'source').resolve()
    if not inventory:
        problems.append('Missing saved source inventory')
    for name,digest in inventory.items():
        path = (source/name).resolve()
        if not path.is_relative_to(source) or not path.is_file() or sha(path) != digest:
            problems.append('Frozen source SHA mismatch: '+name)
    bindings = protocol.get('tree_source_bindings', {})
    for required in ('classic.model_classic','shared.modeling_llama_kv_target'):
        bound = bindings.get(required, {})
        if not isinstance(bound.get('path'),str) or not bound['path'].startswith('exp/specextend_bridgekv/specextend/') or inventory.get(bound['path']) != bound.get('sha256'):
            problems.append('Full-tree frozen import binding missing/foreign: '+required)
    outer_manifest_count = None
    manifest_path = directory.parent/'EVIDENCE_MANIFEST.json'
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())['file_sha256']; outer_manifest_count = len(manifest)
            root = directory.parent.resolve()
            for name,digest in manifest.items():
                path = (root/name).resolve()
                if not path.is_relative_to(root) or not path.is_file() or sha(path) != digest:
                    problems.append('Outer archive SHA mismatch: '+name)
        except (KeyError,TypeError,ValueError,OSError):
            problems.append('Malformed outer EVIDENCE_MANIFEST.json')

    refmap = {}; invalid_refs = set()
    for row in canonical:
        length = row.get('length'); tokens = row.get('generated')
        if length in refmap or length not in LENGTHS or row.get('source_index') != 303 or row.get('mode') != 'canonical_ar' or not isinstance(tokens,list) or not 0 < len(tokens) <= 128 or not all(type(v) is int and v >= 0 for v in tokens) or references.get(str(length)) != tokens:
            problems.append('Missing/duplicate/inconsistent independent AR cell: '+str(length)); invalid_refs.add(length)
        refmap[length] = tokens
    if len(canonical) != 2 or set(refmap) != set(LENGTHS) or set(references) != {'4096','8192'}:
        problems.append('Two independent AR references incomplete')
    qmap = {}; bad_q = set(); expected_q = {(m,l) for m in QUALITY_ARMS for l in LENGTHS}
    for row in quality:
        key = row.get('mode'),row.get('length')
        errors = row_errors(row,refmap.get(key[1]))
        if key not in expected_q or key in qmap or row.get('phase') != 'correctness':
            errors.append('Unexpected/duplicate quality cell')
        if key[0] != 'full32_n8':
            errors += quality_errors(row)
        if errors:
            problems.extend('quality '+str(key)+': '+e for e in errors); bad_q.add(key)
        qmap[key] = row
    if len(quality) != 12 or set(qmap) != expected_q:
        problems.append('Twelve quality requests incomplete')
    for length in LENGTHS:
        for family in (('rescue_eager','rescue_graph'), ('retrieval32_k4_eager','retrieval32_k4_resident','retrieval32_k4_graph')):
            base = qmap.get((family[0],length), {})
            for mode in family:
                key=mode,length; row=qmap.get(key,{})
                if signature(row) != signature(base) or any(row.get(k) != base.get(k) for k in ('observed_rounds','queries','native_t','native_s','draft_forwards','selector_refreshes')):
                    problems.append('Matched family quality trajectory/KV/Query differs: '+str(key)); bad_q.add(key)

    owner_map = {}; bad_owners=set(); expected_owners={(m,l) for m in ARMS for l in LENGTHS}
    for owner in owners:
        key=owner.get('mode'),owner.get('length')
        if key in owner_map or key not in expected_owners or not owner.get('owner_id'):
            problems.append('Unexpected/duplicate/missing timing owner'); bad_owners.add(key)
        owner_map[key]=owner
    if len(owners) != 10 or set(owner_map) != expected_owners or len({o.get('owner_id') for o in owners}) != len(owners):
        problems.append('Ten private timing owners incomplete/duplicated')
    prime_map={}; warm_map={}; bad_warm=set()

    def timing_errors(row, cold=False):
        mode,length=row.get('mode'),row.get('length'); key=mode,length
        errors=row_errors(row,refmap.get(length))
        if length in invalid_refs or key in bad_q:
            errors.append('Canonical/family quality proof invalid')
        owner=owner_map.get(key,{})
        if row.get('owner_id') != owner.get('owner_id'):
            errors.append('Timing owner linkage mismatch')
        if not positive(row.get('outer_seconds')) or not near(row.get('outer_tps'),row.get('tokens',0)/row['outer_seconds'] if positive(row.get('outer_seconds')) else -1) or (positive(row.get('outer_seconds')) and positive(row.get('total_seconds')) and row['outer_seconds']+1e-6 < row['total_seconds']):
            errors.append('Whole-generate outer clock/TPS mismatch')
        if mode != 'full32_n8':
            quality=qmap.get(key,{})
            if signature(row) != signature(quality) or any(row.get(k) != quality.get(k) for k in ('tokens','accepted','proposed','blocks','draft_forwards','selector_refreshes')):
                errors.append('Timing linear algorithm differs from its quality proof')
            if not row.get('owner_signature') or row.get('owner_signature') != owner.get('owner_signature'):
                errors.append('Private Native/working/Graph owner signature changed')
            if mode != 'rescue_graph' and (row.get('projected_positions') != 0 or row.get('draft_shadow_probes') != 0):
                errors.append('Retrieval timing includes Mapper/probe work')
            stats=row.get('draft_runtime',{}).get('request',{})
            if mode.endswith('graph'):
                if stats.get('replays') != row.get('draft_forwards') or not positive(stats.get('replays')):
                    errors.append('Graph replays differ from logical Draft forwards')
                if not all(nonnegative(stats.get(k)) for k in ('captures','capture_host_seconds')) or (cold and (not positive(stats.get('captures')) or not positive(stats.get('capture_host_seconds')))):
                    errors.append('Invalid/unpaid real Graph capture evidence')
                # Legal new warm shapes are explicitly allowed, not censored.
            if mode == 'retrieval32_k4_resident' and (stats.get('resident_steps') != row.get('draft_forwards') or stats.get('captures') != 0):
                errors.append('Resident step/no-capture accounting differs')
            if mode in ('retrieval32_k4_resident','retrieval32_k4_graph') and (row.get('draft_runtime',{}).get('retain_query') is not False or row.get('draft_runtime',{}).get('max_graph_entries') != 128):
                errors.append('Retrieval timing retained unneeded Queries or wrong graph budget')
        if cold and (not nonnegative(owner.get('constructor_and_detach_seconds')) or not near(row.get('cold_complete_outer_seconds'),owner.get('constructor_and_detach_seconds',0)+row.get('outer_seconds',0))):
            errors.append('Cold complete clock excludes constructor')
        return errors

    for row in primes:
        key=row.get('mode'),row.get('length'); errors=timing_errors(row,True)
        if key not in expected_owners or key in prime_map or row.get('phase') != 'cold_prime' or row.get('request_id') != owner_map.get(key,{}).get('prime_request_id'):
            errors.append('Unexpected/duplicate/unlinked cold prime')
        if errors:
            problems.extend('prime '+str(key)+': '+e for e in errors); bad_owners.add(key)
        prime_map[key]=row
    if len(primes) != 10 or set(prime_map) != expected_owners:
        problems.append('Ten cold-prime requests incomplete')
    expected_warm={(m,l,r) for m in ARMS for l in LENGTHS for r in range(2)}
    for row in runtime:
        key=row.get('mode'),row.get('length'),row.get('repeat'); errors=timing_errors(row)
        if key not in expected_warm or key in warm_map or row.get('phase') != 'warm':
            errors.append('Unexpected/duplicate warm request')
        if errors:
            problems.extend('warm '+str(key)+': '+e for e in errors); bad_warm.add(key)
        warm_map[key]=row
    if len(runtime) != 20 or set(warm_map) != expected_warm:
        problems.append('Twenty warm timing requests incomplete')
    order=[{k:r.get(k) for k in ('mode','source_index','length','repeat')} for r in runtime]
    if protocol.get('warm_order') != order:
        problems.append('Warm order differs from its frozen crossover plan')

    for key,owner in owner_map.items():
        group=[r for r in runtime if (r.get('mode'),r.get('length')) == key]; prime=prime_map.get(key,{})
        errors=[]
        if owner.get('warm_request_ids') != [r.get('request_id') for r in group]:
            errors.append('Warm UUID order does not match the owner record')
        total=owner.get('constructor_and_detach_seconds',-1)+prime.get('outer_seconds',0)+sum(r.get('outer_seconds',0) for r in group)+owner.get('close_seconds',-1)
        tokens=prime.get('tokens',0)+sum(r.get('tokens',0) for r in group)
        checks=(('cold_prime_outer_seconds',prime.get('outer_seconds',0)),('prime_tokens',prime.get('tokens',0)),
            ('warm_tokens',sum(r.get('tokens',0) for r in group)),('warm_outer_seconds_sum',sum(r.get('outer_seconds',0) for r in group)),('accounted_service_seconds',total))
        if any(not near(owner.get(k),v) for k,v in checks) or not nonnegative(owner.get('close_seconds')) or not positive(total) or not near(owner.get('accounted_service_tps'),tokens/total if positive(total) else -1) or not positive(owner.get('owner_lifetime_wall_seconds')) or owner.get('owner_lifetime_wall_seconds',0)+1e-6 < total or owner.get('hook_restored') is not True:
            errors.append('Owner cold/warm/close/accounted-service ledger mismatch')
        if key[0].endswith('graph'):
            before=prime.get('runtime_stats',{})
            for row in group:
                after=row.get('runtime_stats',{}); delta=row.get('draft_runtime',{}).get('request',{})
                for field in ('captures','replays','resident_steps','capture_host_seconds','shadow_updates'):
                    if not near(after.get(field),before.get(field,0)+delta.get(field,0)):
                        errors.append('Same-owner Graph lifetime/request counter mismatch')
                before=after
        if errors:
            problems.extend('owner '+str(key)+': '+e for e in errors); bad_owners.add(key)
    ids=[r.get('request_id') for r in canonical+quality+primes+runtime]
    if any(not isinstance(v,str) or not v for v in ids) or len(set(ids)) != len(ids):
        problems.append('Missing/reused request UUID across phases')
    qowners=[r.get('owner_id') for r in quality]
    if len(set(qowners)) != len(qowners) or None in qowners or set(qowners)&{o.get('owner_id') for o in owners}:
        problems.append('Quality/timing owners are missing/reused')
    context_map={}
    for row in contexts:
        length=row.get('length'); group=[r for r in primes+runtime if r.get('length') == length]
        if length in context_map or length not in LENGTHS or not positive(row.get('context_wall_seconds')) or row.get('tokens') != sum(r.get('tokens',0) for r in group):
            problems.append('Context diagnostic-wall accounting mismatch')
        context_map[length]=row
    if len(contexts) != 2 or set(context_map) != set(LENGTHS):
        problems.append('Two context diagnostic-wall records incomplete')
    observed=dict(reference_requests=len(canonical),correctness_requests=len(quality),prime_requests=len(primes),timing_requests=len(runtime),owners=len(owners))
    receipt=read('COMPLETE.json',{})
    if receipt.get('passed') is not True or receipt.get('counts') != EXPECTED or observed != EXPECTED or receipt.get('sources_unchanged') is not True or receipt.get('assets_unchanged') is not True:
        problems.append('Missing/invalid COMPLETE matrix and source/asset receipt')
    if (directory/'FAILURE.json').exists():
        problems.append('Runner FAILURE marker exists')
    metrics={}; usable={}; comparisons={}
    for length in LENGTHS:
        repeats=[r for r in range(2) if all((m,length,r) in warm_map and (m,length,r) not in bad_warm and (m,length) not in bad_owners and (m,length) in prime_map for m in ARMS)]
        usable[str(length)]=repeats; metrics[str(length)]={}
        for mode in ARMS:
            if not repeats:
                continue
            group=[warm_map[mode,length,r] for r in repeats]; owner=owner_map[mode,length]; prime=prime_map[mode,length]
            values=[r['outer_tps'] for r in group]
            metric=dict(warm_tps=sum(r['tokens'] for r in group)/sum(r['outer_seconds'] for r in group),
                repeat_tps=values,timing_range=[min(values),max(values)],warm_repeats=len(repeats),
                acceptance_length=sum(r['accepted'] for r in group)/sum(r['blocks'] for r in group),
                acceptance_rate=None if mode == 'full32_n8' else sum(r['accepted'] for r in group)/sum(r['proposed'] for r in group),
                cold_complete_tps=prime['tokens']/(owner['constructor_and_detach_seconds']+prime['outer_seconds']),
                accounted_service_tps=owner['accounted_service_tps'] if repeats == [0,1] else None,
                constructor_seconds=owner['constructor_and_detach_seconds'],close_seconds=owner['close_seconds'])
            for field in ('prefill_seconds','decode_seconds','cleanup_seconds','total_seconds','blocks','accepted','draft_forwards','working_gathered_position_layers','projected_positions','selection_similarity_elements'):
                values=[r[field] for r in group if nonnegative(r.get(field))]
                metric[field]=statistics.mean(values) if len(values) == len(group) else None
            metric['cold_capture_seconds']=prime.get('draft_runtime',{}).get('request',{}).get('capture_host_seconds',0)
            metric['cold_captures']=prime.get('draft_runtime',{}).get('request',{}).get('captures',0)
            metric['warm_capture_seconds']=statistics.mean(r.get('draft_runtime',{}).get('request',{}).get('capture_host_seconds',0) for r in group)
            metric['warm_captures']=sum(r.get('draft_runtime',{}).get('request',{}).get('captures',0) for r in group)
            metric['warm_replays']=sum(r.get('draft_runtime',{}).get('request',{}).get('replays',0) for r in group)
            metric['generation_overshoot']=sum(r.get('generation_overshoot',0) for r in group)
            metric['eos_tail_tokens']=sum(len(r.get('scored_generated_before_eos',r['generated']))-r['tokens'] for r in group)
            metrics[str(length)][mode]=metric
        if repeats:
            data=metrics[str(length)]; rescue=data['rescue_graph']['warm_tps']
            fastest=max(ARMS[1:4],key=lambda m:data[m]['warm_tps'])
            comparisons[str(length)]=dict(fastest_observed_retrieval=fastest,
                rescue_vs_percent={m:100*(rescue/data[m]['warm_tps']-1) for m in ARMS[1:]},
                pointwise_beats_both=rescue>data[fastest]['warm_tps'] and rescue>data['full32_n8']['warm_tps'])
    return dict(status='COMPLETE' if not problems else 'PARTIAL', goal_status='PARTIAL: gap pilot only; formal SpecExtend superiority not established',
        problems=problems,documents=1,confidence_intervals=None,observed_counts=observed,expected_counts=EXPECTED,
        usable_repeats=usable,metrics=metrics,comparisons=comparisons,source_files=len(inventory),
        outer_manifest_files=outer_manifest_count,scope='H10080GB;doc303 reused development;exact4K/8K input;128 requested output limit;2 timing repeats, NOT independent documents. No held-out,4090,original Vicuna/FP16 or SOTA claim.',
        raw_sha256={p.name:sha(p) for p in directory.glob('*.json*') if p.is_file()})


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output); figures=[]
    names=['RESCUE\nGraph','Retrieval\neager','Retrieval\nresident','Retrieval\nGraph','Full tree\nn8']
    mode_colors=['#32867c','#63778b','#d4963b','#9b6b9f','#658b50']
    feature_colors=['#63778b','#d4963b','#32867c','#9b6b9f']
    title=summary['status']+' pilot · H10080GB · doc303 (n=1 reused-dev document)'
    caption='Exact4K/8K inputs;128 output limit;2 timing repeats are NOT2 documents. Timing-range bars are NOT confidence intervals. Formal SpecExtend goal remains PARTIAL.'

    def finish(fig,name,note=caption):
        fig.suptitle(title,fontsize=10);fig.text(.5,.012,note,ha='center',fontsize=7.3,wrap=True)
        fig.tight_layout(rect=(0,.075,1,.94))
        for ext in ('png','svg'):
            fig.savefig(output/f'{name}.{ext}',dpi=170)
        plt.close(fig);figures.append(name)

    def decorate(ax,length,ylabel):
        ax.set(xticks=range(5),xticklabels=names,title=f'{length} tokens · matched repeats={len(summary["usable_repeats"][str(length)])}',ylabel=ylabel)

    def legend(ax,size=7):
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=size)

    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for ax,length in zip(axes,LENGTHS):
        for i,mode in enumerate(ARMS):
            value=summary['metrics'][str(length)].get(mode)
            if value:
                ax.bar(i,value['acceptance_length'],color=mode_colors[i])
        decorate(ax,length,'Accepted Draft tokens\n/ verification round')
    finish(fig,'quality',caption+'\nFull-tree uses raw upstream accepted Draft/round, INCLUDING overshoot/EOS tail computation; no invented tree acceptance rate.')
    fig,axes=plt.subplots(3,2,figsize=(12,11))
    for col,length in enumerate(LENGTHS):
        for row,(field,label) in enumerate((('warm_tps','Warm full-generate\nouter TPS'),('cold_complete_tps','Cold constructor+prime\nTPS'),('accounted_service_tps','3-request accounted\nservice TPS'))):
            for i,mode in enumerate(ARMS):
                value=summary['metrics'][str(length)].get(mode)
                if value and value[field] is not None:
                    axes[row,col].bar(i,value[field],color=mode_colors[i])
                    if row == 0:
                        axes[row,col].plot([i,i],value['timing_range'],color='black',linewidth=1.3)
            decorate(axes[row,col],length,label)
    finish(fig,'tps',caption+'\nCold includes every capture. Accounted service sums constructor+cold+2 warm+close; non-contiguous intervals, NOT per-mode production/session wall.')
    fig,axes=plt.subplots(4,2,figsize=(12,14))
    for col,length in enumerate(LENGTHS):
        data=summary['metrics'][str(length)]
        for j,field in enumerate(('prefill_seconds','decode_seconds','cleanup_seconds')):
            for i,mode in enumerate(ARMS):
                value=data.get(mode,{}).get(field)
                if value is not None:
                    axes[0,col].bar(i+(j-1)*.22,value*1000,width=.21,color=feature_colors[j],label=field.replace('_seconds','') if i == 0 else None)
        if 'full32_n8' in data:
            axes[0,col].bar(4,data['full32_n8']['total_seconds']*1000,width=.6,color='#658b50',hatch='//',label='full-tree total (not split)')
        decorate(axes[0,col],length,'Warm stage/total\nlatency (ms)');legend(axes[0,col])
        for j,field in enumerate(('constructor_seconds','cold_capture_seconds','close_seconds')):
            for i,mode in enumerate(ARMS):
                value=data.get(mode,{}).get(field)
                if value is not None:
                    axes[1,col].bar(i+(j-1)*.22,value*1000,width=.21,color=feature_colors[j],label=field.replace('_seconds','') if i == 0 else None)
        decorate(axes[1,col],length,'Cold/owner intervals\n(ms; capture nested)');legend(axes[1,col])
        for i,mode in enumerate(ARMS):
            value=data.get(mode)
            if value:
                axes[2,col].bar(i,value['warm_capture_seconds']*1000,color=mode_colors[i])
                if mode.endswith('graph'):
                    axes[2,col].text(i,.97,f"capture={value['warm_captures']}\nreplay={value['warm_replays']}",transform=axes[2,col].get_xaxis_transform(),ha='center',va='top',fontsize=7)
        decorate(axes[2,col],length,'Mean warm capture\ninterval (ms)')
        for j,field in enumerate(('blocks','draft_forwards','working_gathered_position_layers','projected_positions')):
            for i,mode in enumerate(ARMS):
                value=data.get(mode,{}).get(field)
                if value is not None and value > 0:
                    axes[3,col].bar(i+(j-1.5)*.17,value,width=.16,color=feature_colors[j],label=field.replace('_',' ') if i == 0 else None)
        decorate(axes[3,col],length,'Mean request work\ncount (log scale)');axes[3,col].set_yscale('log');legend(axes[3,col],6.5)
    finish(fig,'latency_captures',caption+'\nFull-tree split is unavailable, not zero. Capture intervals are nested in generate, not additive/subtracted; legal new warm captures remain charged.')
    return figures


def write_report(directory,output,summary,figures):
    import os
    directory,output=Path(directory),Path(output)
    report=['# '+summary['status']+': matched SpecExtend gap pilot','',summary['scope'],'',
        '**Formal goal: '+summary['goal_status']+'.** COMPLETE only certifies this small registered pilot and its report; it cannot establish document-level significance or superiority of the general method.',
        'Five timing arms use the same model weights, usable vocabulary,precision,prompt and requested output cap. RESCUE keeps its frozen nonzero Mapper and causal2/4 StepBudget. Retrieval has no Mapper/controller, fixed k4,top32×chunk32,refresh4; it gets persistent private buffers and eager/resident/Graph execution. Full tree retains n8,depth10,threshold0.7,Flash prefill and hybrid attention in the Qwen2 architecture port.',
        'Warm primary TPS pools actual client tokens/time over the two matched repeats. Error bars show observed per-request timing ranges only, NOT95%CI. n=1 document and two context prefixes do not become independent documents.', '',
        '| Input | Arm | Warm outer TPS | Cold complete TPS | Accounted-service TPS | Accepted Draft/round | Warm captures |', '|---|---|---:|---:|---:|---:|---:|']
    for length in LENGTHS:
        for mode,m in summary['metrics'][str(length)].items():
            service='n/a' if m['accounted_service_tps'] is None else f'{m["accounted_service_tps"]:.4f}'
            report.append(f'| {length} | {mode} | {m["warm_tps"]:.4f} | {m["cold_complete_tps"]:.4f} | {service} | {m["acceptance_length"]:.4f} | {m["warm_captures"]} |')
    report += ['', '## Descriptive gaps, not a final success gate', '']
    for length,comp in summary['comparisons'].items():
        report.append(f'- {length}: fastest **observed in this pilot** retrieval arm is `{comp["fastest_observed_retrieval"]}`; RESCUE warm TPS changes: '+', '.join(f'{m} {v:+.3f}%' for m,v in comp['rescue_vs_percent'].items())+'.')
    report += ['', 'No baseline or method search was performed. k4/n8 were historical development choices at different contexts/output caps; this pilot does not establish their global optimality or replace a train-only baseline selection stage.', '',
        '## Accounting and evidence limits', '',
        'Cold complete TPS includes constructor/detach plus the complete cold generate. Every lazy Graph capture,warmup and restoration remains inside generate. Legal new warm shapes are counted, not censored. Accounted-service TPS includes constructor+cold+two warm requests+close; it is a non-contiguous measured service sum, not interleaved owner lifetime or production session throughput. Shared model loading/architecture-wrapper/packed-projection preparation is excluded in all arms and separately recorded.',
        'Full-tree client-prefix/EOS cap is registered before measurement. Every returned client token must match independent AR; any wrong token inside the requested prefix fails, rather than being clipped away. Upstream raw generated count,raw accepted-length list and overshoot are retained, and all computation time is charged. Tail token IDs not exposed by upstream are not invented. Raw overshoot (`raw_count-scored_prefix_length`) and EOS-tail removal (`scored_prefix_length-client_tokens`) are different quantities.',
        'Full-tree acceptance is raw upstream accepted Draft tokens per verification round, including cap/EOS-tail computation. It is not a128-token-only clamped acceptance measure; no proposed-tree-node count or acceptance rate is invented. Full-tree prefill/decode splits are unavailable, not zero.',
        'Quality checks compare eager↔Graph within RESCUE and eager↔resident↔Graph within retrieval: all valid28-layer Native K/V hashes,each Draft Query,selected positions,causal features where present,and draft/check/cap/acceptance trajectories. RESCUE additionally requires observed nonzero Mapper correction. Full tree has canonical-client-output and upstream-count evidence, NOT the same KV/Query equivalence audit across architectures.',
        'Source files and optional outer evidence manifest are rehashed by this CPU program. Remote weight files are not reopened locally: their frozen hashes and immutable-assets receipt are checked. No CUDA computation is re-executed by this analysis.', '',
        '| Input | Arm | Raw128-cap overshoot tokens (warm sum) | EOS-tail tokens (warm sum) | Cold captures |', '|---|---|---:|---:|---:|']
    for length in LENGTHS:
        for mode,m in summary['metrics'][str(length)].items():
            report.append(f'| {length} | {mode} | {m["generation_overshoot"]} | {m["eos_tail_tokens"]} | {m["cold_captures"]} |')
    report += ['', '## Audit issues', '', 'Observed counts: `'+json.dumps(summary['observed_counts'],sort_keys=True)+'`.']
    report += ['- '+v for v in summary['problems']] if summary['problems'] else ['No audit issues in the registered pilot. Formal goal remains PARTIAL.']
    report += ['', '## Figures, source and raw results', '']
    for name in figures:
        report.append(f'- [{name}]({name}.png) ([SVG]({name}.svg))')
    report += ['- [Analysis source](analysis_source.py)','- [Summary](summary.json)']
    for name in ('protocol.json','references.json','reference.jsonl','correctness.jsonl','prime.jsonl','runtime.jsonl','owners.jsonl','contexts.jsonl','COMPLETE.json','FAILURE.json'):
        if (directory/name).exists():
            report.append(f'- [{name}]({os.path.relpath(directory/name,output)})')
    (output/'REPORT.md').write_text('\n'.join(report)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():
        parser.error('Fresh analysis directory required; no evidence overwrite')
    args.output.mkdir(parents=True)
    summary=analyze(args.directory);figures=plot(summary,args.output)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    shutil.copyfile(__file__,args.output/'analysis_source.py')
    write_report(args.directory,args.output,summary,figures)
    manifest={p.name:sha(p) for p in args.output.iterdir() if p.is_file()}
    (args.output/'ANALYSIS_COMPLETE.json').write_text(json.dumps(dict(status=summary['status'],
        goal_status=summary['goal_status'],artifacts_sha256=manifest,inspected_render_required=True),indent=2)+'\n')
    print(json.dumps(dict(status=summary['status'],goal_status=summary['goal_status'],
        comparisons=summary['comparisons'],problems=summary['problems']),indent=2))
    return 0 if summary['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
