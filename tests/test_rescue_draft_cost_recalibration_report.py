"""Synthetic CPU accounting tests, not model or GPU performance results."""
import copy
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from rescue.config import SystemConfig
from rescue import draft_cost_recalibration_report as cost_report
from rescue.draft_cost_recalibration_report import (
    DOCS, EXPECTED, LENGTHS, MODES, analyze, calibration_summary, paired_statistics,
    pooled_document, write_report,
)


def saved_cell(doc=271, length=4096, mode='oldrisk'):
    """Two unequal synthetic timings make pooling differ from mean TPS."""
    result = []
    for repeat, seconds, tokens, blocks, proposed, accepted in (
            (0, 1., 8, 2, 4, 2), (1, 3., 12, 4, 8, 6)):
        result.append(dict(source_index=doc, length=length, mode=mode,
            repeat=repeat, outer_seconds=seconds, tokens=tokens,
            blocks=blocks, proposed=proposed, accepted=accepted,
            trace=[dict(cap=2)]*blocks,
            draft_runtime=dict(request=dict(captures=repeat,
                capture_host_seconds=.05*repeat)),
            prefill_seconds=.1*(repeat+1),
            decode_seconds=seconds-.1*(repeat+1)-.1,
            cleanup_seconds=.1))
    return result


def profiles(fit_4k=(1.,)*5, fit_8k=(9.,),
             validation_4k=(1., 1.), validation_8k=(1., 1.)):
    """Repeat 0 is fit; repeat 1 is an independent timing validation."""
    result = []
    for length, fitting, validation in (
            (4096, fit_4k, validation_4k),
            (8192, fit_8k, validation_8k)):
        for repeat, samples in enumerate((fitting, validation)):
            result.append(dict(length=length, repeat=repeat,
                proposed=len(samples),
                cost_samples=dict(student_step=list(samples)),
                draft_runtime=dict(request=dict(captures=0,
                    capture_host_seconds=0.))))
    return result


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def save_rows(path, values):
    path.write_text(''.join(json.dumps(v, allow_nan=False)+'\n' for v in values))


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reseal_synthetic_cost(directory):
    """Refresh only synthetic seal hashes after a deliberate cost mutation."""
    root = Path(directory)
    receipt = json.loads((root/'fit_receipt.json').read_text())
    receipt['new_sha256'] = file_sha(root/'cost_new.json')
    save_json(root/'fit_receipt.json', receipt)
    save_json(root/'COST_SEALED.json', dict(receipt,
        fit_receipt_sha256=file_sha(root/'fit_receipt.json')))
    names = ('cost_old.json', 'cost_new.json', 'fit_receipt.json', 'COST_SEALED.json')
    save_json(root/'seal_manifest.json', dict(
        file_sha256={name: file_sha(root/name) for name in names},
        development_unopened=True, seal_execution_index=8))
    if (root/'protocol.json').is_file():
        protocol = json.loads((root/'protocol.json').read_text())
        protocol.update(new_cost_sha256=file_sha(root/'cost_new.json'),
            calibration_seal_sha256=file_sha(root/'seal_manifest.json'),
            cost_sealed_sha256=file_sha(root/'COST_SEALED.json'))
        save_json(root/'protocol.json', protocol)
    if (root/'COMPLETE.json').is_file():
        complete = json.loads((root/'COMPLETE.json').read_text())
        complete['calibration_seal_sha256'] = file_sha(root/'seal_manifest.json')
        save_json(root/'COMPLETE.json', complete)


def full_fixture(directory):
    """A complete synthetic 94-request schema, never benchmark measurements."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    old_cost = dict(student_step=.02, verify_beta=[.02, 0., .0003, 0.],
                    cache_update=.001, fixed_round=.002)
    save_json(root/'cost_old.json', old_cost)
    save_json(root/'cost_new.json', dict(old_cost, student_step=.01))
    pins = dict(cost_report.ASSETS, old_cost_sha256=file_sha(root/'cost_old.json'))
    metadata = dict(kind='step_budget', hidden=32, margin=.01,
        prior_weight=.5, seed=101, state_dict_sha256={'synthetic': 'd'*64})
    reference = list(range(128))
    keys = [[513, True, True], [514, True, False]]
    source = root/'source/rescue/synthetic.py'
    source.parent.mkdir(parents=True)
    source.write_text('# Synthetic fixture only, not an executed model source.\n')
    source_hashes = {'rescue/synthetic.py': file_sha(source)}
    streams = {name: [] for name in cost_report.REQUEST_FILES.values()}
    owners, cal_owners = [], []
    logical_index = 0
    pointer_index = 0

    def register(filename, row):
        nonlocal logical_index
        row.update(execution_index=logical_index, request_id=f'synthetic-request:{logical_index}')
        logical_index += 1
        streams[filename].append(row)
        return row

    def digest(shape, value='a', dtype='torch.bfloat16'):
        return dict(shape=shape, dtype=dtype, sha256=value*64)

    def pointers():
        nonlocal pointer_index
        pointer_index += 1
        start = 10000*pointer_index
        return {name: dict(identity=start+i, key_ptr=start+100+i,
                          value_ptr=start+200+i, capacity=16384)
                for i, name in enumerate(('native_t', 'native_s', 'working'))}

    def runtime(count, cold=False):
        delta = dict(captures=2 if cold else 0,
            capture_host_seconds=.25 if cold else 0., replays=128,
            resident_steps=0, shadow_updates=128)
        lifetime = dict(mode='cuda_graph', entries=2, captures=2,
            capture_host_seconds=.25, replays=128*count, resident_steps=0,
            shadow_updates=128*count, graph_keys=copy.deepcopy(keys))
        return dict(draft_runtime=dict(mode='cuda_graph', request=delta,
                    lifetime=copy.deepcopy(lifetime)), runtime_stats=lifetime)

    def row(mode, doc, length, seconds=4., count=1, cold=False):
        total = seconds-.005
        trace = [dict(round_index=i, prefix=length-1+4*i, cap=4,
            proposed=4, accepted=3, terminal=i == 31, workset_tokens=512)
            for i in range(32)]
        result = dict(mode=mode, source_index=doc, length=length,
            generated=reference.copy(), tokens=128, output_agrees=True,
            blocks=32, proposed=128, accepted=96, draft_forwards=128,
            selector_refreshes=8, trace=trace, prefill_seconds=.2,
            decode_seconds=total-.21, cleanup_seconds=.01,
            total_seconds=total, total_tps=128/total,
            decode_tps=128/(total-.21), projected_positions=0 if mode=='retrieval32_k4_graph' else 64,
            static_affine_positions=0 if mode=='retrieval32_k4_graph' else 64,
            selection_similarity_elements=0 if mode=='retrieval32_k4_graph' else 4096,
            working_gathered_position_layers=512*28*32,
            m3_disabled=mode in ('fixed4', 'retrieval32_k4_graph'), policy_seed=101)
        result.update(runtime(count, cold))
        return result

    def quality_row(mode, doc, length):
        result = row(mode, doc, length, cold=True)
        risk = mode in ('oldrisk', 'newrisk')
        injection = mode != 'retrieval32_k4_graph'
        observations, queries = [], []
        for i in range(32):
            proof = [dict(layer=27, positions=[0, 1],
                working_value=digest([2, 256], 'b'),
                native_value=digest([2, 256], 'c'),
                different_elements=1, native_matches_memo=True)] if injection else []
            observations.append(dict(round_index=i, prefix=length-1+4*i,
                progress=4*i, remaining=128-4*i, selected=[0, 1],
                budget_features=[0.]*11+[.1] if risk else None,
                draft=[4*i, 4*i+1, 4*i+2, 10000],
                checks=[4*i, 4*i+1, 4*i+2, 4*i+3, 0], accepted=3,
                injection_evidence=proof, summary_count=0 if not risk else 1))
            for depth in range(4):
                requested = risk and depth < 2
                queries.append(dict(logits=True, query=digest([1, 12, 1, 128]),
                    summary_requested=requested, summary_count=int(requested)))
        result.update(phase='correctness', performance_eligible=False,
            owner_id=f'quality:{mode}:{doc}:{length}', owner_signature=pointers(),
            observed_rounds=observations, queries=queries, nonzero_injection=injection,
            native_t=dict(lengths=[length+128]*28, layers=[dict(
                key=digest([1, 4, length+128, 128], dtype='torch.float32'),
                value=digest([1, 4, length+128, 128], dtype='torch.float32')) for _ in range(28)]),
            native_s=dict(lengths=[length+127]*28, layers=[dict(
                key=digest([1, 2, length+127, 128]),
                value=digest([1, 2, length+127, 128])) for _ in range(28)]))
        return result

    def finish_stats(value):
        closed = copy.deepcopy(value)
        closed.update(entries=0, graph_keys=[])
        return closed

    # Exactly eight requests precede the immutable seal: 2 AR, 2 primes,
    # 4 instrumented profiles, with one persistent calibration owner/context.
    for length in LENGTHS:
        register('calibration_reference.jsonl', dict(mode='canonical_ar',
            source_index=246, length=length, generated=reference.copy()))
    for length in LENGTHS:
        owner_id = f'calibration-owner:{length}'
        signature = pointers()
        prime = row('oldrisk', 246, length, 6., cold=True)
        prime.update(phase='calibration_cold_prime', owner_id=owner_id,
            owner_signature=signature, performance_eligible=False, profile_enabled=True,
            timeline=dict(stages={'draft_first': dict(host_seconds=.1)}),
            cost_samples=dict(student_step=[.01]*128), measured_generate_seconds=6.,
            constructor_and_detach_seconds=.02)
        register('calibration_prime.jsonl', prime)
        group = []
        for repeat in (0, 1):
            sample = .011 if length == 8192 and repeat == 1 else .01
            value = row('oldrisk', 246, length, 4., repeat+2)
            value.update(phase='calibration_profile', repeat=repeat, owner_id=owner_id,
                owner_signature=signature, performance_eligible=False,
                profile_enabled=True, profile_outer_seconds=4.,
                fit_eligible=repeat == 0, validation_only=repeat == 1,
                timeline=dict(stages={'draft_first': dict(host_seconds=.1)}),
                cost_samples=dict(student_step=[sample]*128))
            register('calibration_profiles.jsonl', value)
            group.append(value)
        cal_owners.append(dict(mode='oldrisk', length=length, owner_id=owner_id,
            owner_signature=signature, calibration_cost_only=True,
            constructor_and_detach_seconds=.02, close_seconds=.001,
            prime_request_id=prime['request_id'],
            profile_request_ids=[r['request_id'] for r in group],
            cold_prime_profile_seconds=6., warm_profile_seconds_sum=8.,
            final_runtime_stats=copy.deepcopy(group[-1]['runtime_stats']),
            closed_runtime_stats=finish_stats(group[-1]['runtime_stats']),
            engine_closed=True, hook_restored=True, owner_lifetime_wall_seconds=15.))
    fit = dict(sample_count=256, median=.01)
    validation = {str(length): dict(sample_count=128,
        median=.011 if length == 8192 else .01,
        relative_error=abs(.01-(.011 if length == 8192 else .01))/(.011 if length == 8192 else .01),
        passed=True) for length in LENGTHS}
    receipt = dict(fit=fit, validation=validation, fit_repeats=[0], validation_repeats=[1],
        validation_denominator='Repeat1 context sample median', seal_execution_index=8,
        old_sha256=pins['old_cost_sha256'], new_sha256=file_sha(root/'cost_new.json'),
        policy_metadata=metadata,
        calibration_file_sha256='d0bf671ea6e54f81533def13a812fb00f4b27d9199fbf2e0b1d2a8623c90d5ad',
        calibration_document_sha256='b0398d17c8500c6b24997927377e4a431042794a8a22aaaf56a9d7921ec16ad9',
        unchanged_coefficients={k:v for k,v in old_cost.items() if k != 'student_step'},
        legacy_verify_coefficients_unchanged=True)
    save_json(root/'fit_receipt.json', receipt)
    reseal_synthetic_cost(root)
    environment = dict(gpu_name='NVIDIA H100', total_memory=85_000_000_000,
        target_dtype='torch.float32', draft_dtype='torch.bfloat16',
        prefix_dtype='torch.bfloat16', tf32=False)
    common = dict(schema=1, expected_counts=EXPECTED, expected_requests=94,
        **pins, no_training=True, no_model_training=True,
        mapper_frozen=True, policy_frozen=True, policy_metadata=metadata,
        source_sha256=source_hashes, environment=environment,
        calibration_file_sha256=receipt['calibration_file_sha256'],
        calibration_document_sha256=receipt['calibration_document_sha256'],
        development_content_opened=False)
    config = SystemConfig(selector='hierarchical', coverage_rows=64, refresh_every=4,
        refresh_policy='periodic', injection_layers=(27,), inject_tokens=64,
        injection_strength=.5, prefill_precision='bfloat16', max_new_tokens=128,
        max_draft=12, length_policy='risk', schedule='serial', profile=False,
        draft_execution='cuda_graph', coverage_execution='eager')
    calibration_protocol = dict(common, calibration_config=asdict(replace(config, profile=True)))
    save_json(root/'protocol_calibration.json', calibration_protocol)
    for length in LENGTHS:
        for doc in DOCS:
            register('reference.jsonl', dict(mode='canonical_ar', source_index=doc,
                length=length, generated=reference.copy()))
    for length in LENGTHS:
        for doc in DOCS:
            for mode in MODES:
                register('correctness.jsonl', quality_row(mode, doc, length))
    plan = [dict(mode=mode, source_index=doc, length=length, repeat=repeat)
            for length in LENGTHS for repeat in (0, 1) for doc in DOCS for mode in MODES]
    for length in LENGTHS:
        owners_by_mode, prime_by_mode, warm_by_mode = {}, {}, {mode: [] for mode in MODES}
        for mode in MODES:
            owner_id = f'timing-owner:{mode}:{length}'
            signature = pointers()
            prime = row(mode, 271, length, 6., cold=True)
            prime.update(phase='cold_prime', owner_id=owner_id,
                owner_signature=signature, performance_eligible=True,
                outer_seconds=6., outer_tps=128/6.,
                constructor_and_detach_seconds=.02, cold_complete_outer_seconds=6.02)
            register('prime.jsonl', prime)
            prime_by_mode[mode] = prime
            owners_by_mode[mode] = owner_id, signature
        for entry in (p for p in plan if p['length'] == length):
            mode, doc, repeat = entry['mode'], entry['source_index'], entry['repeat']
            scale = (1., 1.1, 1.2)[DOCS.index(doc)]
            factor = ((.9, .8, .7)[DOCS.index(doc)] if mode == 'newrisk'
                      else 1.05 if mode == 'fixed4'
                      else .6 if mode == 'retrieval32_k4_graph' else 1.)
            seconds = (4., 8.)[repeat]*scale*factor
            count = len(warm_by_mode[mode])+2
            value = row(mode, doc, length, seconds, count)
            owner_id, signature = owners_by_mode[mode]
            value.update(**entry, phase='warm', owner_id=owner_id,
                owner_signature=signature, performance_eligible=True,
                outer_seconds=seconds, outer_tps=128/seconds)
            register('runtime.jsonl', value)
            warm_by_mode[mode].append(value)
        for mode in MODES:
            prime, group = prime_by_mode[mode], warm_by_mode[mode]
            owner_id, signature = owners_by_mode[mode]
            warm_seconds = sum(v['outer_seconds'] for v in group)
            service = .02+6.+warm_seconds+.001
            owners.append(dict(mode=mode, length=length, owner_id=owner_id,
                owner_signature=signature, constructor_and_detach_seconds=.02,
                cold_prime_outer_seconds=6., prime_request_id=prime['request_id'],
                warm_request_ids=[v['request_id'] for v in group],
                warm_outer_seconds_sum=warm_seconds, prime_tokens=128,
                warm_tokens=6*128, close_seconds=.001,
                accounted_service_seconds=service, accounted_service_tokens=7*128,
                accounted_service_tps=7*128/service, owner_lifetime_wall_seconds=service*5,
                final_runtime_stats=copy.deepcopy(group[-1]['runtime_stats']),
                closed_runtime_stats=finish_stats(group[-1]['runtime_stats']),
                engine_closed=True, hook_restored=True))
    configs = {}
    for mode in MODES:
        changes = dict(length_policy='risk' if mode in ('oldrisk', 'newrisk') else 'fixed', fixed_k=4)
        if mode == 'retrieval32_k4_graph':
            changes.update(inject_tokens=0, injection_strength=0.)
        configs[mode] = asdict(replace(config, **changes))
    protocol = dict(common, modes=list(MODES), source_indices=list(DOCS),
        lengths=list(LENGTHS), repeats=2, max_new_tokens=128, configs=configs,
        graph_max_entries=128, retrieval_top_chunks=32,
        new_cost_sha256=file_sha(root/'cost_new.json'),
        calibration_seal_sha256=file_sha(root/'seal_manifest.json'),
        cost_sealed_sha256=file_sha(root/'COST_SEALED.json'), seal_execution_index=8,
        development_file_sha256='85b2eb1a81d1617b2d158c069c420c452459a8bdb90da128f257220cc27ded09',
        development_content_opened_after_seal=True,
        development_roles={str(d): 'train' for d in DOCS},
        document_sha256={str(d): hashlib.sha256(f'synthetic-doc:{d}'.encode()).hexdigest() for d in DOCS},
        warm_order=plan)
    save_json(root/'protocol.json', protocol)
    save_json(root/'references.json', {f'{d}:{n}': reference.copy() for d in DOCS for n in LENGTHS})
    for filename, values in streams.items():
        save_rows(root/filename, values)
    save_rows(root/'owners.jsonl', owners)
    save_rows(root/'calibration_owners.jsonl', cal_owners)
    save_json(root/'COMPLETE.json', dict(passed=True, counts=EXPECTED,
        expected_counts=EXPECTED, logical_requests=94, sources_unchanged=True,
        assets_unchanged=True, cost_seal_unchanged=True,
        calibration_seal_sha256=file_sha(root/'seal_manifest.json'), analysis_required=True))
    assert logical_index == 94
    return dict(streams=streams, owners=owners, calibration_owners=cal_owners,
        protocol=protocol, asset_patch={'old_cost_sha256': pins['old_cost_sha256']})


@contextmanager
def synthetic_evidence(directory):
    evidence = full_fixture(directory)
    with patch.dict(cost_report.ASSETS, evidence['asset_patch']):
        yield evidence


class DraftCostReportTests(unittest.TestCase):
    def test_pool_tokens_over_time_not_the_mean_of_request_tps(self):
        rows = saved_cell()
        result = pooled_document(rows)
        self.assertEqual(result['tokens'], 20)
        self.assertEqual(result['outer_seconds'], 4.)
        self.assertEqual(result['tps'], 5.)
        self.assertNotEqual(result['tps'], (8/1.+12/3.)/2)
        self.assertAlmostEqual(result['acceptance_length'], 8/6)
        self.assertAlmostEqual(result['acceptance_rate'], 8/12)
        self.assertEqual(result['cap_counts'], {'1': 0, '2': 6, '3': 0, '4': 0})
        self.assertEqual(result['captures'], 1)
        self.assertAlmostEqual(result['capture_seconds'], .05)

    def test_no_missing_repeat_or_cross_document_context_arm_imputation(self):
        original = saved_cell()
        invalid = [original[:1], original+[copy.deepcopy(original[0])],
                   [original[0], copy.deepcopy(original[0])]]
        for field, replacement in (('source_index', 272), ('length', 8192),
                                   ('mode', 'newrisk')):
            changed = copy.deepcopy(original)
            changed[1][field] = replacement
            invalid.append(changed)
        for rows in invalid:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                pooled_document(rows)

    def test_statistical_units_are_three_pooled_documents_not_six_repeats(self):
        paired = []
        for doc, speed_factor in zip(DOCS, (1.1, .9, 1.2)):
            baseline = saved_cell(doc=doc)
            treatment = saved_cell(doc=doc, mode='newrisk')
            for row in treatment:
                row['outer_seconds'] /= speed_factor
            paired.append(100*(pooled_document(treatment)['tps']/
                               pooled_document(baseline)['tps']-1))
        result = paired_statistics(paired)
        self.assertEqual(result['n'], 3)
        self.assertAlmostEqual(result['mean'], 20/3)
        self.assertIsNotNone(result['ci95'])
        one_document = paired_statistics([pooled_document(saved_cell())['tps']])
        self.assertEqual(one_document['n'], 1)
        self.assertIsNone(one_document['ci95'])

    def test_bootstrap_is_deterministic_and_requires_frozen_10000_draws(self):
        result = paired_statistics([-3., 2., 7.])
        self.assertEqual(result, paired_statistics([-3., 2., 7.]))
        self.assertEqual(result, dict(n=3, mean=2., ci95=[-3., 7.]))
        self.assertEqual(paired_statistics([]), dict(n=0, mean=None, ci95=None))
        for values, kwargs in (([1., 2.], dict(draws=200)),
                               ([float('nan')], {}), ([float('inf')], {}),
                               ([[1., 2.]], {})):
            with self.subTest(values=values, kwargs=kwargs), self.assertRaises(ValueError):
                paired_statistics(values, **kwargs)

    def test_repeat0_fit_is_pooled_steps_not_median_of_context_medians(self):
        result, issues = calibration_summary(profiles())
        self.assertEqual(issues, [])
        self.assertEqual(result['fit'], dict(sample_count=6, median=1.))
        self.assertNotEqual(result['fit']['median'], (1.+9.)/2)
        self.assertEqual(set(result['validation']), {'4096', '8192'})
        self.assertTrue(all(v['passed'] for v in result['validation'].values()))

    def test_repeat1_is_not_fit_and_validation_uses_observed_denominator(self):
        values = profiles(fit_4k=(8.,)*5, fit_8k=(8.,),
                          validation_4k=(10.,), validation_8k=(10.,))
        result, issues = calibration_summary(values)
        self.assertEqual(issues, [])
        self.assertEqual(result['fit'], dict(sample_count=6, median=8.))
        for value in result['validation'].values():
            self.assertEqual(value['relative_error'], .2)
            self.assertTrue(value['passed'])
        self.assertGreater(abs(8.-10.)/8., .2)

    def test_more_than_twenty_percent_repeat1_error_fails_without_refitting(self):
        values = profiles(fit_4k=(8.,)*5, fit_8k=(8.,),
                          validation_4k=(11.,), validation_8k=(10.,))
        result, issues = calibration_summary(values)
        self.assertEqual(result['fit']['median'], 8.)
        self.assertFalse(result['validation']['4096']['passed'])
        self.assertTrue(result['validation']['8192']['passed'])
        self.assertTrue(any('4096' in v and 'exceeds20%' in v for v in issues))

    def test_capture_contamination_is_rejected_on_fit_and_validation(self):
        for repeat, captures, seconds in ((0, 1, .5), (1, 0, .5), (1, 1, 0.)):
            with self.subTest(repeat=repeat, captures=captures, seconds=seconds):
                values = profiles()
                values[repeat]['draft_runtime']['request'].update(
                    captures=captures, capture_host_seconds=seconds)
                _, issues = calibration_summary(values)
                self.assertTrue(any('includes capture cost' in v for v in issues))

    def test_incomplete_duplicate_or_invalid_per_step_profiles_cannot_pass(self):
        original = profiles()
        invalid = [original[:-1], original+[copy.deepcopy(original[0])]]
        for samples, proposed in (([], 0), ([0.], 1), ([-1.], 1),
                                  ([float('nan')], 1), ([1.], 2)):
            changed = copy.deepcopy(original)
            changed[0]['cost_samples']['student_step'] = samples
            changed[0]['proposed'] = proposed
            invalid.append(changed)
        for values in invalid:
            with self.subTest(values=values):
                _, issues = calibration_summary(values)
                self.assertTrue(issues)

    def test_empty_or_truncated_partial_has_five_figures_and_standalone_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = analyze(root)
            self.assertEqual(empty['status'], 'PARTIAL')
            self.assertTrue(empty['problems'])
            self.assertEqual(empty['usable_documents'], {str(n): [] for n in LENGTHS})
            (root/'runtime.jsonl').write_text('{"unfinished":')
            partial = analyze(root)
            self.assertEqual(partial['status'], 'PARTIAL')
            self.assertTrue(any('Malformed runtime.jsonl' in v for v in partial['problems']))
            output = root/'analysis'
            summary = write_report(root, output)
            self.assertEqual(summary['status'], 'PARTIAL')
            names = {'acceptance', 'tps', 'calibration_capture', 'actions_service', 'latency'}
            for name in names:
                self.assertTrue((output/(name+'.png')).read_bytes().startswith(b'\x89PNG'))
                self.assertIn('<svg', (output/(name+'.svg')).read_text())
            self.assertIn('PARTIAL', (output/'REPORT.md').read_text())
            receipt = json.loads((output/'ANALYSIS_COMPLETE.json').read_text())
            self.assertEqual(receipt['status'], 'PARTIAL')
            for name, digest in receipt['artifacts_sha256'].items():
                self.assertEqual(hashlib.sha256((output/name).read_bytes()).hexdigest(), digest)
            self.assertTrue((output/'analysis_source.py').is_file())
            self.assertTrue((output/'analysis_support.py').is_file())
            rerun = root/'standalone_rerun'
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            process = subprocess.run([sys.executable, str(output/'analysis_source.py'),
                str(root), '--output', str(rerun)], cwd=output, env=env,
                text=True, capture_output=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
            saved = json.loads((rerun/'summary.json').read_text())
            self.assertEqual(saved['status'], 'PARTIAL')
            self.assertEqual(saved['observed_counts'], summary['observed_counts'])
            self.assertEqual({p.stem for p in rerun.glob('*.png')}, names)
            self.assertEqual({p.stem for p in rerun.glob('*.svg')}, names)


class DraftCostFullSchemaTests(unittest.TestCase):
    def test_synthetic94_complete_pools_two_repeats_then_bootstraps_three_docs(self):
        with tempfile.TemporaryDirectory() as tmp, synthetic_evidence(tmp):
            result = analyze(tmp)
        self.assertEqual(result['status'], 'COMPLETE', result['problems'])
        self.assertEqual(result['observed_counts'], EXPECTED)
        self.assertEqual(sum(v for k, v in EXPECTED.items() if 'owners' not in k), 94)
        self.assertEqual(result['usable_documents'], {str(n): list(DOCS) for n in LENGTHS})
        self.assertEqual(result['calibration']['fit'], dict(sample_count=256, median=.01))
        for length in LENGTHS:
            cell = result['metrics'][str(length)]['oldrisk']
            self.assertEqual(cell['documents'], 3)
            self.assertEqual(set(cell['per_document']), {str(d) for d in DOCS})
            self.assertAlmostEqual(cell['mean_document_tps'],
                (256/12.+256/13.2+256/14.4)/3)
            self.assertAlmostEqual(cell['pooled_tps'], 768/(12.+13.2+14.4))
            self.assertNotAlmostEqual(cell['mean_document_tps'], cell['pooled_tps'])
            self.assertEqual(cell['acceptance_length'], 3.)
            self.assertEqual(cell['acceptance_length_ci95'], [3., 3.])
            self.assertEqual(cell['warm_captures'], 0)
            comparison = result['comparisons'][str(length)]['oldrisk']
            statistic = comparison['relative_percent']
            self.assertEqual(statistic['n'], 3)
            expected = [100*(1/factor-1) for factor in (.9, .8, .7)]
            self.assertAlmostEqual(statistic['mean'], sum(expected)/3)
            self.assertAlmostEqual(statistic['ci95'][0], min(expected))
            self.assertAlmostEqual(statistic['ci95'][1], max(expected))
            self.assertTrue(comparison['strict_speed_gate'])
            self.assertEqual(comparison['acceptance_delta']['mean'], 0.)

    def test_resealed_nonstudent_coefficient_change_is_still_partial(self):
        with tempfile.TemporaryDirectory() as tmp, synthetic_evidence(tmp):
            root = Path(tmp)
            new_cost = json.loads((root/'cost_new.json').read_text())
            new_cost['verify_beta'][0] += 1.
            save_json(root/'cost_new.json', new_cost)
            reseal_synthetic_cost(root)
            result = analyze(root)
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertTrue(any('fields other than student_step' in v for v in result['problems']))
        self.assertEqual(result['usable_documents'], {str(n): [] for n in LENGTHS})

    def test_missing_developer_pair_is_not_imputed_from_other_docs_or_repeats(self):
        with tempfile.TemporaryDirectory() as tmp, synthetic_evidence(tmp) as evidence:
            remaining = [r for r in evidence['streams']['runtime.jsonl']
                if not (r['mode'] == 'newrisk' and r['source_index'] == 288
                        and r['length'] == 8192 and r['repeat'] == 1)]
            save_rows(Path(tmp)/'runtime.jsonl', remaining)
            result = analyze(tmp)
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertEqual(result['observed_counts']['timing_requests'], 47)
        self.assertEqual(result['usable_documents']['4096'], list(DOCS))
        # Its incomplete persistent-owner ledger invalidates the paired 8K
        # context; a missing repeat must never become a full document sample.
        self.assertEqual(result['usable_documents']['8192'], [])
        self.assertEqual(result['metrics']['8192'], {})
        self.assertEqual(result['comparisons']['8192'], {})

    def test_synthetic_complete_path_renders_all_five_quality_system_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp, synthetic_evidence(tmp):
            output = Path(tmp)/'analysis'
            result = write_report(tmp, output)
            self.assertEqual(result['status'], 'COMPLETE', result['problems'])
            names = {'acceptance', 'tps', 'calibration_capture', 'actions_service', 'latency'}
            self.assertEqual({p.stem for p in output.glob('*.png')}, names)
            self.assertEqual({p.stem for p in output.glob('*.svg')}, names)
            for name in names:
                self.assertTrue((output/(name+'.png')).read_bytes().startswith(b'\x89PNG'))
                self.assertIn('<svg', (output/(name+'.svg')).read_text())
            text = (output/'REPORT.md').read_text()
            for phrase in ('COMPLETE', 'Only student_step changes', 'not held-out',
                           'calibration_profiles.jsonl', 'runtime.jsonl', 'analysis_support.py'):
                self.assertIn(phrase, text)
            receipt = json.loads((output/'ANALYSIS_COMPLETE.json').read_text())
            self.assertEqual(receipt['status'], 'COMPLETE')
            for name, wanted in receipt['artifacts_sha256'].items():
                self.assertEqual(file_sha(output/name), wanted)


if __name__ == '__main__':
    unittest.main()
