"""Saved-record audit/figures for the 22-request, one-document M1 Graph pilot.

No model is loaded. Profile requests are diagnostics only, never performance
samples or part of the measured active-service numerator/denominator.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

if __package__:
    from . import draft_graph_benchmark_report as _analysis_support
else:
    # The saved analysis artifact is rerunnable without the repository/package.
    import analysis_support as _analysis_support

ASSETS = _analysis_support.ASSETS
near, nonnegative, positive = (_analysis_support.near, _analysis_support.nonnegative,
                              _analysis_support.positive)
quality_errors, row_errors, signature = (_analysis_support.quality_errors,
    _analysis_support.row_errors, _analysis_support.signature)


MODES = ('draft_graph', 'draft_coverage_graph')
LENGTHS = (4096, 8192)
EXPECTED = dict(reference_requests=2, correctness_requests=4, prime_requests=4,
                timing_requests=8, profile_requests=4, owners=4)
WORK = ('tokens', 'accepted', 'proposed', 'blocks', 'draft_forwards', 'selector_refreshes',
        'projected_positions', 'static_affine_positions', 'selection_similarity_elements',
        'working_gathered_position_layers')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def graph_errors(row, cold=False):
    errors = []
    for family, snapshot, expected_replays, limit in (
            ('draft_runtime', 'runtime_stats', row.get('draft_forwards'), 128),
            ('coverage_runtime', 'coverage_runtime_stats', row.get('selector_refreshes'), 64)):
        active = family == 'draft_runtime' or row.get('mode') == 'draft_coverage_graph'
        stats = row.get(snapshot, {})
        if not active:
            if family in row or any(stats.get(k) != 0 for k in ('entries', 'captures', 'replays', 'capture_host_seconds')):
                errors.append('Eager coverage unexpectedly used Graph execution')
            continue
        delta = row.get(family, {}).get('request', {})
        if delta.get('replays') != expected_replays or not positive(delta.get('replays')):
            errors.append(family+': replay count differs from actual work')
        for value in (delta, stats):
            if any(type(value.get(k)) is not int or value[k] < 0 for k in ('captures', 'replays')) or not nonnegative(value.get('capture_host_seconds')):
                errors.append(family+': invalid capture/replay counters')
        if cold and (not positive(delta.get('captures')) or not positive(delta.get('capture_host_seconds'))):
            errors.append(family+': fresh quality/cold owner did not capture')
        if delta.get('captures') == 0 and delta.get('capture_host_seconds') != 0:
            errors.append(family+': capture cost without a new capture')
        if positive(delta.get('captures')) and not positive(delta.get('capture_host_seconds')):
            errors.append(family+': capture cost was not charged')
        keys = stats.get('graph_keys' if family == 'draft_runtime' else 'shape_keys', [])
        if type(stats.get('entries')) is not int or not 1 <= stats['entries'] <= limit or len(keys) != stats.get('entries') or stats.get('entries') != stats.get('captures'):
            errors.append(family+': actual shape/capture/entry accounting differs')
        if family == 'coverage_runtime':
            if stats.get('max_entries') != limit or stats.get('owner_identity') != row.get('owner_signature', {}).get('coverage_runtime_identity'):
                errors.append('Coverage Graph owner/entry-limit identity differs')
    capture = sum(row.get(f, {}).get('request', {}).get('capture_host_seconds', 0.)
                  for f in ('draft_runtime', 'coverage_runtime'))
    if positive(row.get('total_seconds')) and capture > row['total_seconds']+1e-5:
        errors.append('Capture host intervals exceed the full request; cost accounting invalid')
    return errors


def analyze(directory):
    directory = Path(directory); problems = []

    def read(name, default):
        try:
            value = json.loads((directory/name).read_text())
            if not isinstance(value, type(default)):
                raise ValueError('Wrong container')
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
                    raise ValueError('Nonobject row')
                result.append(row)
            except ValueError:
                problems.append(f'Malformed {name}:{number}; valid rows retained')
        return result

    protocol = read('protocol.json', {}); references = read('references.json', {})
    canonical, quality, primes, runtime, profiles, owners, contexts = [rows(name) for name in (
        'reference.jsonl', 'correctness.jsonl', 'prime.jsonl', 'runtime.jsonl', 'profiles.jsonl', 'owners.jsonl', 'contexts.jsonl')]
    if protocol.get('source_index') != 303 or protocol.get('lengths') != list(LENGTHS) or protocol.get('max_new_tokens') != 128 or protocol.get('repeats') != 2 or protocol.get('modes') != list(MODES) or protocol.get('expected_counts') != EXPECTED or protocol.get('expected_requests') != 22:
        problems.append('Unregistered22-request/doc303/4K8K/output128-limit protocol')
    if any(protocol.get(k) != v for k,v in ASSETS.items()) or any(protocol.get(k) is not True for k in ('mapper_frozen','policy_frozen','no_training')):
        problems.append('Frozen nonzero Mapper/StepBudget/cost or no-training declaration differs')
    assets = protocol.get('assets_sha256', {})
    for name, key in (('full_final.pt','mapper_sha256'),('budget_selected.pt','policy_sha256'),('cost.json','cost_sha256')):
        if [digest for path,digest in assets.items() if Path(path).name == name] != [ASSETS[key]]:
            problems.append('Immutable asset inventory does not bind '+name)
    fixed = dict(selector='hierarchical', budget=512, recent=64, coverage_rows=64,
        coverage_batch=16, refresh_every=4, refresh_policy='periodic', injection_layers=[27],
        inject_tokens=64, injection_strength=.5, prefill_precision='bfloat16',
        max_new_tokens=128, max_draft=12, length_policy='risk', schedule='serial', profile=False,
        draft_execution='cuda_graph')
    for mode in MODES:
        config = protocol.get('configs', {}).get(mode, {})
        if any(config.get(k) != v for k,v in fixed.items()) or config.get('coverage_execution') != ('cuda_graph' if mode == MODES[1] else 'eager'):
            problems.append(mode+': frozen algorithm or execution configuration differs')
    configs=protocol.get('configs',{})
    if {k:v for k,v in configs.get(MODES[0],{}).items() if k != 'coverage_execution'} != {k:v for k,v in configs.get(MODES[1],{}).items() if k != 'coverage_execution'}:
        problems.append('The two configurations differ beyond coverage execution')
    if protocol.get('draft_graph_max_entries') != 128 or protocol.get('coverage_graph_max_entries') != 64 or protocol.get('retain_query') is not True:
        problems.append('Frozen Graph limits/query-retention policy differ')
    env = protocol.get('environment', {})
    if 'H100' not in env.get('gpu_name','') or env.get('total_memory',0) < 80_000_000_000 or env.get('target_dtype') != 'torch.float32' or env.get('draft_dtype') != 'torch.bfloat16' or env.get('prefix_dtype') != 'torch.bfloat16' or env.get('tf32') is not False:
        problems.append('H10080GB/FP32Target/BF16Draft+prefill/TF32off environment differs')
    inventory = protocol.get('source_sha256', {})
    if not inventory:
        problems.append('Missing frozen source inventory')
    source_root = (directory/'source').resolve()
    for name, digest in inventory.items():
        path = (source_root/name).resolve()
        if not path.is_relative_to(source_root) or not path.is_file() or sha(path) != digest:
            problems.append('Frozen source hash mismatch: '+name)
    manifest=directory.parent/'EVIDENCE_MANIFEST.json'
    if manifest.is_file():
        try:
            evidence=json.loads(manifest.read_text())
            evidence=evidence.get('file_sha256',evidence.get('files',evidence.get('sha256',evidence)))
            if isinstance(evidence,list):
                evidence={v['path']:v['sha256'] for v in evidence}
            for name,digest in evidence.items():
                path=(directory.parent/name).resolve()
                if not path.is_relative_to(directory.parent.resolve()) or not path.is_file() or sha(path) != digest:
                    problems.append('Outer immutable evidence hash mismatch: '+name)
        except (OSError,ValueError,TypeError,KeyError,AttributeError):
            problems.append('Malformed outer immutable evidence manifest')
    metadata_trusted = not problems

    reference_map, bad_contexts = {}, set()
    for row in canonical:
        length, values = row.get('length'), row.get('generated')
        if length not in LENGTHS or length in reference_map or row.get('source_index') != 303 or row.get('mode') != 'canonical_ar' or not isinstance(values,list) or not 0 < len(values) <= 128 or not all(type(v) is int and v >= 0 for v in values) or references.get(str(length)) != values:
            problems.append('Independent AR/reference cell missing/duplicated/invalid'); bad_contexts.add(length)
        reference_map[length] = values
    if set(reference_map) != set(LENGTHS) or len(canonical) != 2:
        problems.append('Independent AR matrix incomplete')
        bad_contexts.update(set(LENGTHS)-set(reference_map))
    expected_keys = {(m,n) for m in MODES for n in LENGTHS}
    qmap, bad_quality = {}, set()
    for row in quality:
        key = row.get('mode'), row.get('length')
        errors = row_errors(row, reference_map.get(key[1]))+quality_errors(row)+graph_errors(row, cold=True)
        if key not in expected_keys or key in qmap or row.get('phase') != 'correctness' or row.get('performance_eligible') is not False:
            errors.append('Unexpected/duplicated/performance-eligible correctness cell')
        if errors:
            problems.extend('quality '+str(key)+': '+e for e in errors); bad_quality.add(key)
        qmap[key] = row
    if set(qmap) != expected_keys or len(quality) != 4:
        problems.append('Four-mode/context correctness matrix incomplete')
    for length in LENGTHS:
        base, other = (qmap.get((m,length), {}) for m in MODES)
        if not base or not other or any(base.get(k) != other.get(k) for k in ('generated','observed_rounds','queries','native_t','native_s')) or signature(base) != signature(other):
            problems.append(str(length)+': modes changed exact selected/features/query/Native/trajectory evidence')
            bad_contexts.add(length)
        if any((m,length) in bad_quality for m in MODES):
            bad_contexts.add(length)

    def consistent(row, *, phase, cold=False, profile=False):
        key = row.get('mode'), row.get('length')
        errors = row_errors(row, reference_map.get(key[1]))+graph_errors(row,cold=cold)
        base = qmap.get(key,{})
        if key not in expected_keys or row.get('phase') != phase or row.get('source_index') != 303:
            errors.append('Wrong phase/mode/context')
        if signature(row) != signature(base) or any(row.get(k) != base.get(k) for k in WORK):
            errors.append('Work counts/depth/acceptance differ from instrumented correctness')
        if key[1] in bad_contexts or key in bad_quality:
            errors.append('Invalid canonical/correctness evidence')
        if profile:
            if row.get('performance_eligible') is not False or row.get('diagnostic_only') is not True or row.get('profile_enabled') is not True or 'outer_seconds' in row or 'outer_tps' in row or not positive(row.get('profile_outer_seconds')):
                errors.append('Diagnostic profile leaked into primary timing or has invalid scope')
            stages = row.get('timeline',{}).get('stages',{})
            if not stages or not isinstance(row.get('cost_samples'),dict) or any(not isinstance(v,dict) or type(v.get('calls')) is not int or v['calls'] < 1 or not nonnegative(v.get('host_seconds')) or not nonnegative(v.get('stream_seconds')) for v in stages.values()):
                errors.append('Missing/invalid separate diagnostic host/stream stages')
            for name,count in (('selection',row.get('blocks')),('target_verify',row.get('blocks'))):
                if stages.get(name,{}).get('calls') != count:
                    errors.append('Diagnostic stage calls differ from actual round work: '+name)
            costs=row.get('cost_samples',{})
            for name,count in (('student_step',row.get('proposed')),('verify',row.get('blocks')),('overhead',row.get('blocks'))):
                if not isinstance(costs.get(name),list) or len(costs[name]) != count:
                    errors.append('Diagnostic cost-sample count differs from work: '+name)
        else:
            if row.get('performance_eligible') is not True or 'timeline' in row or 'cost_samples' in row or not positive(row.get('outer_seconds')) or not near(row.get('outer_tps'),row.get('tokens',0)/row['outer_seconds'] if positive(row.get('outer_seconds')) else -1) or row.get('outer_seconds',0)+1e-6 < row.get('total_seconds',0):
                errors.append('Primary sample is profiled, incomplete, or mis-accounted')
        return key, errors

    prime_map, profile_map, owner_map = {}, {}, {}
    bad_owners = set()
    for name, values, indexed, phase in (('prime',primes,prime_map,'cold_prime'),('profile',profiles,profile_map,'profile')):
        for row in values:
            key, errors = consistent(row,phase=phase,cold=name=='prime',profile=name=='profile')
            if key in indexed:
                errors.append('Duplicate '+name+' owner cell')
            if errors:
                problems.extend(name+' '+str(key)+': '+e for e in errors); bad_owners.add(key)
            indexed[key] = row
        if set(indexed) != expected_keys or len(values) != 4:
            problems.append('Four '+name+' requests incomplete')
            bad_owners.update(expected_keys-set(indexed))
    warm_map, invalid_warm = {}, set()
    expected_warm = {(m,n,r) for m,n in expected_keys for r in range(2)}
    for row in runtime:
        key, errors = consistent(row,phase='warm')
        identity = (*key,row.get('repeat'))
        if identity not in expected_warm or identity in warm_map:
            errors.append('Unexpected/duplicate warm cell')
        if errors:
            problems.extend('warm '+str(identity)+': '+e for e in errors); invalid_warm.add(identity)
        warm_map[identity] = row
    if set(warm_map) != expected_warm or len(runtime) != 8:
        problems.append('Eight unprofiled warm requests incomplete')
    order = [{k:r.get(k) for k in ('mode','source_index','length','repeat')} for r in runtime]
    if protocol.get('warm_order') != order:
        problems.append('Unprofiled warm order differs from frozen interleaved plan')
        present={json.dumps(r,sort_keys=True) for r in order}
        retained=[r for r in protocol.get('warm_order',[]) if json.dumps(r,sort_keys=True) in present]
        if retained != order:
            bad_contexts.update(LENGTHS)
    for i in range(0,len(runtime),2):
        block = runtime[i:i+2]
        if len(block) != 2 or {r.get('mode') for r in block} != set(MODES) or len({(r.get('length'),r.get('repeat')) for r in block}) != 1:
            problems.append('Warm modes are not paired/interleaved'); break
    all_requests = canonical+quality+primes+runtime+profiles
    uuids = [r.get('request_id') for r in all_requests]
    if any(not isinstance(v,str) or not v for v in uuids) or len(set(uuids)) != len(uuids):
        problems.append('Logical request UUID missing/reused')
        bad_contexts.update(LENGTHS)
    indices=[r.get('execution_index') for r in all_requests]
    index_valid=all(type(v) is int and 0 <= v < 22 for v in indices) and len(set(indices)) == len(indices)
    if not index_valid or sorted(indices) != list(range(22)):
        problems.append('Missing/duplicated monotonic22-request execution indices')
        if not index_valid:
            bad_contexts.update(LENGTHS)
    if index_valid and runtime and profiles and quality and primes and canonical:
        if max(r['execution_index'] for r in runtime) >= min(r['execution_index'] for r in profiles) or max(r['execution_index'] for r in quality) >= min(r['execution_index'] for r in primes) or max(r['execution_index'] for r in canonical) >= min(r['execution_index'] for r in quality):
            problems.append('Profiles ran before warm timing ended or quality gate followed timing')
            bad_contexts.update(LENGTHS)
        if [r['execution_index'] for r in runtime] != sorted(r['execution_index'] for r in runtime):
            problems.append('Warm file order is not monotonic execution order')
    quality_ids = {r.get('owner_id') for r in quality}
    if len(quality_ids) != 4 or None in quality_ids:
        problems.append('Correctness request owners are missing/reused')
    service = {}
    for owner in owners:
        key = owner.get('mode'),owner.get('length'); prime=prime_map.get(key,{}); profile=profile_map.get(key,{})
        group = [r for r in runtime if (r.get('mode'),r.get('length')) == key]
        errors = []
        if key not in expected_keys or key in owner_map or not owner.get('owner_id') or owner.get('owner_id') in quality_ids:
            errors.append('Duplicate/unregistered owner or borrowed quality owner')
        if len(group) != 2 or owner.get('prime_request_id') != prime.get('request_id') or owner.get('warm_request_ids') != [r.get('request_id') for r in group] or owner.get('profile_request_ids') != [profile.get('request_id')]:
            errors.append('Owner prime/warm/profile UUID linkage differs')
        samples = [prime,*group,profile]
        if not owner.get('owner_signature') or any(r.get('owner_id') != owner.get('owner_id') or r.get('owner_signature') != owner['owner_signature'] for r in samples):
            errors.append('Persistent private cache/Draft/Coverage owner changed')
        constructor=owner.get('constructor_and_detach_seconds',-1); close=owner.get('close_seconds',-1)
        prime_seconds=prime.get('outer_seconds',0); warm_seconds=sum(r.get('outer_seconds',0) for r in group)
        active=constructor+prime_seconds+warm_seconds+close
        tokens=prime.get('tokens',0)+sum(r.get('tokens',0) for r in group)
        checks = dict(cold_prime_outer_seconds=prime_seconds,warm_outer_seconds_sum=warm_seconds,
            prime_tokens=prime.get('tokens',0),warm_tokens=sum(r.get('tokens',0) for r in group),
            profile_tokens=profile.get('tokens',0),profile_outer_seconds_sum=profile.get('profile_outer_seconds',0),
            accounted_service_seconds=active,accounted_service_tokens=tokens,
            accounted_service_tps=tokens/active if positive(active) else -1)
        if any(not near(owner.get(k),v) for k,v in checks.items()) or not nonnegative(constructor) or not nonnegative(close) or owner.get('profile_excluded_from_service') is not True or owner.get('hook_restored') is not True or not positive(owner.get('owner_lifetime_wall_seconds')) or owner.get('owner_lifetime_wall_seconds',0)+1e-6 < active+profile.get('profile_outer_seconds',0):
            errors.append('Owner service accounting includes/excludes wrong costs or profile')
        if not near(prime.get('constructor_and_detach_seconds'),constructor) or not near(prime.get('cold_complete_outer_seconds'),constructor+prime_seconds):
            errors.append('Cold complete constructor/prime accounting differs')
        for family,snapshot in (('draft_runtime','runtime_stats'),('coverage_runtime','coverage_runtime_stats')):
            if family == 'coverage_runtime' and key[0] == MODES[0]:
                continue
            previous = dict(captures=0,replays=0,capture_host_seconds=0.)
            for row in samples:
                stats=row.get(snapshot,{});delta=row.get(family,{}).get('request',{})
                if any(not near(stats.get(k),previous[k]+delta.get(k,-1)) for k in previous):
                    errors.append(family+': lifetime counters fail prime→warm→profile continuity')
                previous={k:stats.get(k,-1) for k in previous}
        for final,closed,snapshot in (('final_runtime_stats','closed_runtime_stats','runtime_stats'),
                ('final_coverage_runtime_stats','closed_coverage_runtime_stats','coverage_runtime_stats')):
            if owner.get(final) != profile.get(snapshot):
                errors.append('Final graph counters differ from the last diagnostic request')
            if not isinstance(owner.get(closed),dict) or owner[closed].get('entries') != 0:
                errors.append('Closed owner retained graph entries')
            elif any(not near(owner[closed].get(k),profile.get(snapshot,{}).get(k)) for k in ('captures','replays','capture_host_seconds')):
                errors.append('Close changed lifetime graph counters')
        if owner.get('engine_closed') is not True:
            errors.append('Engine did not close')
        if errors:
            problems.extend('owner '+str(key)+': '+e for e in errors);bad_owners.add(key)
        owner_map[key]=owner
        if not errors and key not in bad_owners:
            service[key]=dict(cold_tps=prime.get('tokens',0)/(constructor+prime_seconds),
                accounted_tps=tokens/active,cold_prime_seconds=prime_seconds,constructor_seconds=constructor,
                close_seconds=close,profile_excluded_seconds=profile.get('profile_outer_seconds',0))
    if set(owner_map) != expected_keys or len(owners) != 4:
        problems.append('Four persistent-owner accounting records incomplete')
        bad_owners.update(expected_keys-set(owner_map))
    timing_ids = [r.get('owner_id') for r in owners]
    if len(set(timing_ids)) != len(timing_ids):
        problems.append('Two timing owners share identity')
    for cache in ('native_t','native_s','working'):
        pointers=[o.get('owner_signature',{}).get(cache,{}).get('key_ptr') for o in owners]
        if any(type(v) is not int or v <= 0 for v in pointers) or len(set(pointers)) != len(pointers):
            problems.append('Timing owners share or omit private '+cache+' storage')
    context_map={r.get('length'):r for r in contexts}
    for length in LENGTHS:
        context=context_map.get(length,{})
        samples=[r for r in primes+runtime if r.get('length') == length]
        minimum=sum(r.get('outer_seconds',0) for r in samples)+sum(o.get('constructor_and_detach_seconds',0) for o in owners if o.get('length') == length)
        if not positive(context.get('unprofiled_block_wall_seconds')) or context.get('unprofiled_block_wall_seconds',0)+1e-6 < minimum or context.get('timing_client_tokens') != sum(r.get('tokens',0) for r in samples):
            problems.append(str(length)+': real unprofiled block wall/token diagnostic differs')
    observed=dict(reference_requests=len(canonical),correctness_requests=len(quality),prime_requests=len(primes),
                  timing_requests=len(runtime),profile_requests=len(profiles),owners=len(owners))
    receipt=read('COMPLETE.json',{})
    if observed != EXPECTED or receipt.get('passed') is not True or receipt.get('counts') != EXPECTED or receipt.get('sources_unchanged') is not True or receipt.get('assets_unchanged') is not True or receipt.get('profile_excluded_from_service') is not True:
        problems.append('Missing/inconsistent22-request completion/immutability/profile-exclusion receipt')
    if (directory/'FAILURE.json').exists():
        problems.append('Runner failure marker exists')

    metrics, diagnostics, usable = {}, {}, {}
    for length in LENGTHS:
        matched=[repeat for repeat in range(2) if metadata_trusted and length not in bad_contexts and
            all((m,length,repeat) in warm_map and (m,length,repeat) not in invalid_warm and (m,length) not in bad_owners for m in MODES)]
        usable[str(length)]=matched; metrics[str(length)]={};diagnostics[str(length)]={}
        for mode in MODES:
            key=mode,length;prime=prime_map.get(key,{});profile=profile_map.get(key,{})
            group=[warm_map[mode,length,r] for r in matched]
            if not group:
                continue
            metric=dict(warm_tps=sum(r['tokens'] for r in group)/sum(r['outer_seconds'] for r in group),
                repeat_tps=[r['outer_tps'] for r in group],acceptance_length=sum(r['accepted'] for r in group)/sum(r['blocks'] for r in group),
                acceptance_rate=sum(r['accepted'] for r in group)/sum(r['proposed'] for r in group),
                repeats=len(matched),documents=1,**service[key])
            metric['repeat_range']=[min(metric['repeat_tps']),max(metric['repeat_tps'])]
            metrics[str(length)][mode]=metric
            diagnostics[str(length)][mode]=dict(
                coverage_cold_captures=prime.get('coverage_runtime',{}).get('request',{}).get('captures',0),
                coverage_warm_captures=sum(r.get('coverage_runtime',{}).get('request',{}).get('captures',0) for r in group),
                coverage_profile_captures=profile.get('coverage_runtime',{}).get('request',{}).get('captures',0),
                coverage_warm_capture_seconds=sum(r.get('coverage_runtime',{}).get('request',{}).get('capture_host_seconds',0.) for r in group),
                profile_stages=profile.get('timeline',{}).get('stages',{}),
                profile_scope=profile.get('timeline',{}).get('scope',''),
                profile_seconds=profile.get('profile_outer_seconds'),warm_work_counts={k:statistics.mean(r[k] for r in group) for k in WORK})
        if set(metrics[str(length)]) == set(MODES):
            base,variant=(metrics[str(length)][m] for m in MODES)
            variant['warm_change_percent']=100*(variant['warm_tps']/base['warm_tps']-1)
            variant['paired_repeat_change_percent']=[100*(warm_map[MODES[1],length,r]['outer_tps']/warm_map[MODES[0],length,r]['outer_tps']-1) for r in matched]
    return dict(status='COMPLETE' if not problems else 'PARTIAL',problems=problems,observed_counts=observed,
        usable_repeats=usable,metrics=metrics,diagnostics=diagnostics,documents=1,confidence_intervals=None,
        scope='H10080GB; doc303 reused development; 4K/8K exact inputs,128 output limit; frozen nonzero V Mapper and causal2/4 policy; Draft Graph in both arms. No document CI, significance, held-out or SpecExtend-win claim.',
        profile_scope='Four separately profiled requests occur after unprofiled timing by frozen runner control flow; per-owner counters also prove profiles follow both warm requests. Profile tokens/time never enter primary or service metrics.',
        raw_sha256={p.name:sha(p) for p in directory.glob('*.json*') if p.is_file()})


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    output=Path(output);names=[];colors=('#63778b','#32867c');labels=('Draft Graph','+ Coverage Graph')
    caption=summary['status']+' · H10080GB · frozen nonzero V Mapper +2/4 policy · n=1 doc303 ·128 output limit'
    footer='Registered two repeats; timing ranges are NOT 95% CI. Profile requests are diagnostics only.'

    def finish(fig,name):
        fig.suptitle(caption,fontsize=11);fig.text(.5,.01,footer,ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.04,1,.95))
        for suffix in ('png','svg'):
            fig.savefig(output/(name+'.'+suffix),dpi=165)
        plt.close(fig);names.append(name)

    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,length in zip(axes,LENGTHS):
        values=summary['metrics'].get(str(length),{})
        if set(values) != set(MODES):
            ax.text(.5,.5,'PARTIAL: no valid matched repeat',ha='center',transform=ax.transAxes);ax.set_axis_off();continue
        ax.bar(range(2),[values[m]['acceptance_length'] for m in MODES],color=colors)
        ax.set(xticks=range(2),xticklabels=labels,ylabel='Accepted Draft tokens / verification round',title=f'{length} input tokens')
    finish(fig,'acceptance')
    fig,axes=plt.subplots(3,2,figsize=(12,11))
    for column,length in enumerate(LENGTHS):
        values=summary['metrics'].get(str(length),{})
        for index,(key,title) in enumerate((('warm_tps','Unprofiled warm outer TPS'),('cold_tps','Cold constructor + generate TPS'),('accounted_tps','Accounted active-service TPS'))):
            ax=axes[index,column]
            if set(values) != set(MODES):
                ax.text(.5,.5,'PARTIAL: no valid matched repeat',ha='center',transform=ax.transAxes);ax.set_axis_off();continue
            ax.bar(range(2),[values[m][key] for m in MODES],color=colors)
            if index == 0:
                for i,m in enumerate(MODES):
                    ax.plot([i,i],values[m]['repeat_range'],color='black',linewidth=2)
                title+=f'\npaired pooled change {values[MODES[1]]["warm_change_percent"]:+.2f}% (n=1)'
            ax.set(xticks=range(2),xticklabels=labels,ylabel='Tokens/s',title=f'{length} tokens · '+title)
    finish(fig,'tps')
    fig,axes=plt.subplots(3,2,figsize=(14,13))
    for column,length in enumerate(LENGTHS):
        values=summary['diagnostics'].get(str(length),{})
        if set(values) != set(MODES):
            for ax in axes[:,column]:
                ax.text(.5,.5,'PARTIAL: no valid matched diagnostics',ha='center',transform=ax.transAxes);ax.set_axis_off()
            continue
        x=np.arange(2);ax=axes[0,column]
        for offset,(key,label) in zip((-.22,0,.22),(('coverage_cold_captures','Cold'),('coverage_warm_captures','Warm sum'),('coverage_profile_captures','Profile only'))):
            ax.bar(x+offset,[values[m][key] for m in MODES],width=.21,label=label)
        ax.set(xticks=x,xticklabels=labels,ylabel='New M1 Graph captures',title=f'{length} input tokens');ax.legend(fontsize=8)
        stages=sorted({s for m in MODES for s in values[m]['profile_stages']})
        for index,field in enumerate(('host_seconds','stream_seconds'),1):
            ax=axes[index,column];y=np.arange(len(stages))
            for offset,m,color,label in zip((-.17,.17),MODES,colors,labels):
                ax.barh(y+offset,[values[m]['profile_stages'].get(s,{}).get(field,0.) for s in stages],height=.32,color=color,label=label)
            ax.set(yticks=y,yticklabels=stages,xlabel='Seconds (diagnostic, non-additive)',title=f'{length} tokens · profile '+field.replace('_',' '));ax.legend(fontsize=8)
    finish(fig,'diagnostics')
    return names


def write_report(directory, output):
    directory,output=Path(directory),Path(output)
    if output.exists():
        raise FileExistsError('Fresh analysis directory required')
    output.mkdir(parents=True);summary=analyze(directory);figures=plot(summary,output)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    shutil.copyfile(__file__,output/'analysis_source.py')
    shutil.copyfile(_analysis_support.__file__,output/'analysis_support.py')
    text=['# '+summary['status']+': coverage-Graph execution pilot','',summary['scope'],'',
        'Precision: Target FP32; Draft and prefix prefill BF16; TF32 disabled. Both arms keep the identical frozen nonzero Mapper, attention selection and causal 2/4 actions.',
        'Only the greedy coverage inner-loop execution changes. Actual support/pool shapes are used, without padding or approximate scores.',
        'Warm primary pools tokens/full-generate outer seconds across two unprofiled repeats. All legal new cold/warm captures remain charged.',
        'Cold includes constructor/detach plus first generate. Accounted service includes constructor+cold+two warm+close; it is a measured non-contiguous service sum, not production wall throughput.',
        'Close is measured after diagnostic requests. Their execution intervals and tokens are excluded, but any diagnostic-induced teardown effects are not separately isolated.',
        summary['profile_scope'],'Host and stream stage intervals can overlap/include queueing; they are not independent additive GPU-kernel costs.','',
        '| Input | Mode | Warm outer TPS | Cold TPS | Accounted-service TPS | Accepted Draft/round |','|---|---|---:|---:|---:|---:|']
    for length in LENGTHS:
        for mode,value in summary['metrics'].get(str(length),{}).items():
            text.append(f'| {length} | {mode} | {value["warm_tps"]:.4f} | {value["cold_tps"]:.4f} | {value["accounted_tps"]:.4f} | {value["acceptance_length"]:.4f} |')
    text+=['','n=1 reused document, two timing repeats. No 95% document confidence interval, significance, formal SpecExtend win or held-out-generalization conclusion. Timing ranges are not confidence intervals.','',
           'Observed counts: `'+json.dumps(summary['observed_counts'],sort_keys=True)+'`.','', '## Audit issues','']
    text+=['- '+v for v in summary['problems']] or ['None. The registered pilot evidence is complete; this is not the larger research goal.']
    text+=['','## Figures and saved evidence','']
    for name in figures:
        text.append(f'- [{name}]({name}.png) ([SVG]({name}.svg))')
    text+=['- [Analysis code](analysis_source.py)',
        '- [Exact saved analysis support](analysis_support.py), SHA256 `'+sha(output/'analysis_support.py')+'`.',
        '- Re-run from this analysis directory: `python analysis_source.py .. --output ../analysis_rerun` (fresh output required).',
        '- [Raw SHA/audit summary](summary.json)']
    for name in ('protocol.json','reference.jsonl','references.json','correctness.jsonl','prime.jsonl','runtime.jsonl','profiles.jsonl','owners.jsonl','contexts.jsonl','COMPLETE.json','FAILURE.json'):
        if (directory/name).is_file():
            text.append(f'- [{name}](../{name})')
    (output/'REPORT.md').write_text('\n'.join(text)+'\n')
    (output/'ANALYSIS_COMPLETE.json').write_text(json.dumps(dict(status=summary['status'],
        artifacts_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()},
        rendered_inspection_required=True,scope='Saved-data analysis, not a model run or scientific success'),indent=2)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();summary=write_report(args.directory,args.output)
    print(json.dumps(dict(status=summary['status'],metrics=summary['metrics'],problems=summary['problems']),indent=2))


if __name__ == '__main__':
    main()
