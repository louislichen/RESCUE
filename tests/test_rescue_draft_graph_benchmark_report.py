"""Synthetic CPU program tests, NOT measurements of models or GPUs."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rescue.draft_graph_benchmark_report import (
    ASSETS, DOCS, EXPECTED, LENGTHS, MODES, analyze, bootstrap, plot,
    quality_errors, row_errors, write_report,
)


def save_rows(path, rows):
    path.write_text(''.join(json.dumps(v)+'\n' for v in rows))


def load_rows(path):
    return [json.loads(v) for v in path.read_text().splitlines() if v]


def fixture(directory, graph_factors=None):
    """The full small registered matrix, with synthetic evidence only."""
    directory = Path(directory)
    reference = list(range(128))
    graph_keys = [[513, True, True], [514, True, True]]

    def digest(shape):
        return dict(dtype='torch.bfloat16', shape=shape, sha256='a'*64)

    def row(doc, length, mode, request_id):
        trace = [dict(round_index=i, prefix=length-1+4*i, cap=4, proposed=4,
                      accepted=3, terminal=i == 31, workset_tokens=512) for i in range(32)]
        return dict(request_id=request_id, source_index=doc, length=length, mode=mode,
            generated=reference, tokens=128, accepted=96, proposed=128, blocks=32,
            draft_forwards=128, trace=trace, prefill_seconds=.2, decode_seconds=7.79,
            cleanup_seconds=.01, total_seconds=8., total_tps=16., decode_tps=128/7.79,
            output_agrees=True, selector_refreshes=8, working_gathered_position_layers=1000,
            projected_positions=64, selection_similarity_elements=4096)

    def timed(doc, length, repeat, mode, owner_id, count, cold=False):
        value = row(doc, length, mode, f'{mode}:{length}:{doc}:{repeat}:timed')
        factor = {'baseline_eager': 1., 'resident': .97,
                  'cuda_graph': (graph_factors or {}).get(length, .75)}[mode]
        seconds = (10. if cold else [4., 6., 10.][repeat])*factor
        value.update(phase='cold_prime' if cold else 'warm', repeat=repeat, owner_id=owner_id,
            outer_seconds=seconds, outer_tps=128/seconds,
            prefill_seconds=.2, decode_seconds=seconds-.215, cleanup_seconds=.01,
            total_seconds=seconds-.005, total_tps=128/(seconds-.005), decode_tps=128/(seconds-.215),
            owner_signature=dict(native_t=[owner_id], native_s=[owner_id], working=[owner_id], runtime=owner_id))
        request = dict(captures=2 if cold and mode == 'cuda_graph' else 0,
            capture_host_seconds=.7 if cold and mode == 'cuda_graph' else 0.,
            replays=128 if mode == 'cuda_graph' else 0,
            resident_steps=128 if mode == 'resident' else 0)
        lifetime = dict(captures=2 if mode == 'cuda_graph' else 0,
            replays=128*count if mode == 'cuda_graph' else 0,
            resident_steps=128*count if mode == 'resident' else 0,
            entries=2 if mode != 'baseline_eager' else 0,
            graph_keys=graph_keys if mode == 'cuda_graph' else [])
        value.update(draft_runtime=dict(request=request, lifetime=lifetime), runtime_stats=lifetime)
        return value

    refs, references, quality = [], {}, []
    for length in LENGTHS:
        for doc in DOCS:
            refs.append(dict(request_id=f'ar:{doc}:{length}', source_index=doc, length=length,
                             mode='canonical_ar', generated=reference))
            references[f'{doc}:{length}'] = reference
            for mode in MODES:
                value = row(doc, length, mode, f'quality:{doc}:{length}:{mode}')
                observed = [dict(round_index=i, prefix=length-1+4*i,
                    selected=[0, 1], budget_features=[0.]*11+[.1],
                    draft=[4*i, 4*i+1, 4*i+2, 10000], checks=[4*i, 4*i+1, 4*i+2, 4*i+3, 0], accepted=3) for i in range(32)]
                n = length+128
                value.update(owner_id=f'quality-owner:{doc}:{length}:{mode}',
                    observed_rounds=observed, nonzero_injection=True,
                    queries=[dict(logits=True, query=digest([1, 12, 1, 128])) for _ in range(128)],
                    native_t=dict(lengths=[n]*28, layers=[dict(key=digest([1, 4, n, 128]), value=digest([1, 4, n, 128])) for _ in range(28)]),
                    native_s=dict(lengths=[n-1]*28, layers=[dict(key=digest([1, 2, n-1, 128]), value=digest([1, 2, n-1, 128])) for _ in range(28)]),
                    runtime_stats=dict(captures=2 if mode == 'cuda_graph' else 0,
                                       replays=128 if mode == 'cuda_graph' else 0,
                                       graph_keys=graph_keys if mode == 'cuda_graph' else []))
                quality.append(value)
    plan = [dict(source_index=d, length=l, repeat=r, mode=m)
            for l in LENGTHS for d in DOCS for r in range(3) for m in MODES]
    primes, runtime, owners, contexts = [], [], [], []
    for length in LENGTHS:
        for mode in MODES:
            owner_id = f'owner:{mode}:{length}'
            prime = timed(303, length, -1, mode, owner_id, 1, cold=True)
            prime.update(constructor_and_detach_seconds=.02, cold_complete_outer_seconds=.02+prime['outer_seconds'],
                         required_graph_keys=graph_keys if mode == 'cuda_graph' else [])
            primes.append(prime)
            group = []
            for count, item in enumerate([p for p in plan if (p['mode'], p['length']) == (mode, length)], start=2):
                group.append(timed(item['source_index'], length, item['repeat'], mode, owner_id, count))
            runtime.extend(group)
            warm_seconds = sum(v['outer_seconds'] for v in group)
            active = .02+prime['outer_seconds']+warm_seconds+.001
            owners.append(dict(owner_id=owner_id, mode=mode, length=length,
                constructor_and_detach_seconds=.02, cold_prime_outer_seconds=prime['outer_seconds'],
                prime_request_id=prime['request_id'], warm_request_ids=[r['request_id'] for r in group],
                warm_outer_seconds_sum=warm_seconds, prime_tokens=128, warm_tokens=18*128,
                close_seconds=.001, accounted_service_seconds=active, accounted_service_tps=19*128/active,
                owner_lifetime_wall_seconds=active*4, owner_signature=prime['owner_signature'],
                hook_restored=True, closed_runtime_stats=dict(entries=0)))
        owner_active = sum(r['accounted_service_seconds'] for r in owners if r['length'] == length)
        contexts.append(dict(length=length, context_wall_seconds=owner_active+2., tokens=3*19*128, mode_count=3))
    indexed = {(r['source_index'], r['length'], r['repeat'], r['mode']): r for r in runtime}
    runtime = [indexed[p['source_index'], p['length'], p['repeat'], p['mode']] for p in plan]
    protocol = dict(source_indices=list(DOCS), lengths=list(LENGTHS), repeats=3,
        max_new_tokens=128, modes=list(MODES), expected_counts=EXPECTED, timing_order=plan,
        config=dict(max_new_tokens=128, max_draft=12, attention_backend='sdpa', schedule='serial',
                    working_mode='packed', length_policy='risk', optimized_execution=True,
                    deferred_probes=True, injection_layers=[27], inject_tokens=64, injection_strength=.5,
                    profile=False, prefill_precision='bfloat16'),
        environment=dict(gpu_name='NVIDIA H100', total_memory=85_000_000_000,
                         target_dtype='torch.float32', draft_dtype='torch.bfloat16', prefix_dtype='torch.bfloat16', tf32=False),
        no_training=True, mapper_frozen=True, training_epochs=0, **ASSETS, source_sha256={},
        assets_sha256={'/synthetic/full_final.pt': ASSETS['mapper_sha256'],
                       '/synthetic/budget_selected.pt': ASSETS['policy_sha256'],
                       '/synthetic/cost.json': ASSETS['cost_sha256']})
    for role in ('baseline_eager', 'new'):
        path = directory/'source'/role/'rescue'/'synthetic.py'
        path.parent.mkdir(parents=True)
        path.write_text('# Only synthetic unit-test evidence\n')
        protocol['source_sha256'][role] = {'rescue/synthetic.py': hashlib.sha256(path.read_bytes()).hexdigest()}
    receipt = dict(passed=True, counts=EXPECTED, assets_unchanged=True, sources_unchanged=True)
    for name, value in (('protocol.json', protocol), ('references.json', references), ('COMPLETE.json', receipt)):
        (directory/name).write_text(json.dumps(value))
    for name, values in (('reference.jsonl', refs), ('correctness.jsonl', quality), ('prime.jsonl', primes),
                         ('runtime.jsonl', runtime), ('owners.jsonl', owners), ('contexts.jsonl', contexts)):
        save_rows(directory/name, values)
    return dict(protocol=protocol, canonical=refs, quality=quality, prime=primes, runtime=runtime, owners=owners)


class PersistentGraphReportTests(unittest.TestCase):
    def analyze(self, directory):
        return analyze(Path(directory), bootstrap_draws=200)

    def test_complete_matrix_and_document_unit_pooled_repeats(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp)
            result = self.analyze(tmp)
        self.assertEqual(result['status'], 'COMPLETE', result['problems'])
        self.assertTrue(result['strict_warm_speed_gate']['passed'])
        self.assertEqual(result['observed_counts'], EXPECTED)
        for length in LENGTHS:
            data = result['contexts'][str(length)]
            self.assertEqual(data['n_documents'], 6)
            self.assertAlmostEqual(data['modes']['baseline_eager']['tps']['mean'], 384/20.)
            self.assertNotAlmostEqual(data['modes']['baseline_eager']['tps']['mean'], 128/6.)
            self.assertAlmostEqual(data['comparisons']['cuda_graph_vs_baseline_eager']['relative_percent']['mean'], 100/3)
            self.assertEqual(data['comparisons']['cuda_graph_vs_baseline_eager']['acceptance_delta']['mean'], 0.)

    def test_missing_repeat_excludes_whole_document_all_modes_not_imputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = fixture(tmp)
            save_rows(Path(tmp)/'runtime.jsonl', data['runtime'][:-1])
            result = self.analyze(tmp)
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertFalse(result['strict_warm_speed_gate']['passed'])
        # Incomplete owner accounting conservatively excludes that context's
        # entire three-mode paired set, not just its last repeat.
        self.assertEqual(result['contexts']['8192']['n_documents'], 0)
        self.assertEqual(result['contexts']['4096']['n_documents'], 6)

    def test_complete_does_not_imply_both_context_speed_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp, graph_factors={8192: 1.05})
            result = self.analyze(tmp)
        self.assertEqual(result['status'], 'COMPLETE', result['problems'])
        self.assertEqual(result['strict_warm_speed_gate']['by_context'], {'4096': True, '8192': False})
        self.assertFalse(result['strict_warm_speed_gate']['passed'])

    def test_first_token_eos_can_have_no_controller_vector(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = fixture(tmp)
            row = copy.deepcopy(data['quality'][0])
            first = row['observed_rounds'][0]
            # A real first-token EOS returns before a second-token feature can
            # exist. Test this guard without fabricating an EOS benchmark.
            first.update(budget_features=None, draft=[0], checks=[0, 1], accepted=1)
            row['trace'][0].update(proposed=1, accepted=1)
            errors = quality_errors(row)
            self.assertNotIn('Missing/invalid frozen twelve-dimensional features', errors)

    def test_canonical_wrong_tokens_not_saved_boolean(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = fixture(tmp); data['runtime'][0]['generated'][0] = 999
            save_rows(Path(tmp)/'runtime.jsonl', data['runtime'])
            result = self.analyze(tmp)
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertTrue(any('Canonical output' in s for s in result['problems']))

    def test_saved_native_kv_and_query_proofs_must_match_not_just_flags(self):
        for field in ('native_t', 'native_s', 'queries'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                data = fixture(tmp)
                row = copy.deepcopy(data['quality'][2])
                if field == 'queries':
                    row[field][0]['query']['sha256'] = 'b'*64
                else:
                    row[field]['layers'][0]['key']['sha256'] = 'b'*64
                data['quality'][2] = row
                save_rows(Path(tmp)/'correctness.jsonl', data['quality'])
                result = self.analyze(tmp)
                self.assertEqual(result['status'], 'PARTIAL')
                self.assertTrue(any(field+' differs' in s for s in result['problems']))

    def test_nonzero_is_recomputed_from_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = fixture(tmp)
            for row in data['quality'][2]['observed_rounds']:
                row['budget_features'][-1] = 0
            save_rows(Path(tmp)/'correctness.jsonl', data['quality'])
            result = self.analyze(tmp)
        self.assertTrue(any('actual nonzero' in s for s in result['problems']))

    def test_no_warm_recapture_or_cold_graph_borrowing(self):
        for phase in ('warm', 'cold'):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as tmp:
                data = fixture(tmp)
                rows = data['runtime'] if phase == 'warm' else data['prime']
                row = next(r for r in rows if r['mode'] == 'cuda_graph')
                row['draft_runtime']['request']['captures'] = 1 if phase == 'warm' else 0
                save_rows(Path(tmp)/('runtime.jsonl' if phase == 'warm' else 'prime.jsonl'), rows)
                result = self.analyze(tmp)
                self.assertEqual(result['status'], 'PARTIAL')
                self.assertTrue(any('Warm Graph request recaptured' in s if phase == 'warm' else 'Cold owner did not pay' in s for s in result['problems']))

    def test_persistent_owner_pointer_changes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = fixture(tmp)
            data['runtime'][2] = copy.deepcopy(data['runtime'][2])
            data['runtime'][2]['owner_signature']['runtime'] = 'replacement-runtime'
            save_rows(Path(tmp)/'runtime.jsonl', data['runtime'])
            result = self.analyze(tmp)
        self.assertTrue(any('pointer signature changed' in s for s in result['problems']))

    def test_ar_uuid_must_be_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = fixture(tmp)
            data['canonical'][0]['request_id'] = data['runtime'][0]['request_id']
            save_rows(Path(tmp)/'reference.jsonl', data['canonical'])
            result = self.analyze(tmp)
        self.assertIn('Missing/reused request UUID across AR, quality, prime and timing', result['problems'])

    def test_source_hash_and_asset_receipt_are_mandatory(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp); path = Path(tmp)
            (path/'source/new/rescue/synthetic.py').write_text('# modified\n')
            receipt = json.loads((path/'COMPLETE.json').read_text()); receipt['assets_unchanged'] = False
            (path/'COMPLETE.json').write_text(json.dumps(receipt))
            result = self.analyze(path)
        self.assertTrue(any('Frozen source hash mismatch' in s for s in result['problems']))
        self.assertTrue(any('immutable assets/sources' in s for s in result['problems']))

    def test_all28_cache_layers_and_reconstructed_tokens_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = fixture(tmp)
            row = copy.deepcopy(data['quality'][0]); row['native_s']['layers'].pop()
            row['observed_rounds'][0]['checks'][3] = 500
            errors = quality_errors(row)
        self.assertIn('Missing all28 valid Native KV layers: native_s', errors)
        self.assertIn('Instrumented committed token sequence differs from canonical output', errors)

    def test_native_prefix_minus_one_and_remaining_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = fixture(tmp); row = copy.deepcopy(data['quality'][0])
            self.assertEqual(row_errors(row, list(range(128))), [])
            row['trace'][0]['prefix'] += 1
            self.assertIn('First native prefix must be exact input length minus one', row_errors(row, list(range(128))))
            row = copy.deepcopy(data['quality'][0]); row['trace'][0].update(cap=3, proposed=3)
            self.assertIn('Depth is not frozen 2/4 with remaining-output clipping', row_errors(row, list(range(128))))

    def test_single_document_no_fake_ci_and_ci_unit_is_document(self):
        self.assertIsNone(bootstrap([1.], draws=200)['ci95'])
        result = bootstrap([1., 2., 3., 4., 5., 6.], draws=10000)
        self.assertEqual(result['n'], 6)
        self.assertAlmostEqual(result['mean'], 3.5)
        self.assertEqual(result['ci95'], [2.1666666666666665, 4.833333333333333])

    def test_empty_and_malformed_partial_still_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); output = path/'analysis'; output.mkdir()
            (path/'runtime.jsonl').write_text('{bad json\n')
            result = self.analyze(path)
            self.assertEqual(result['status'], 'PARTIAL')
            names = plot(result, output)
            self.assertEqual(len(names), 6)
            self.assertTrue(all((output/(n+'.png')).read_bytes().startswith(b'\x89PNG') for n in names))

    def test_six_separate_quality_performance_mechanism_figures_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); fixture(path); result = self.analyze(path)
            output = path/'analysis'; output.mkdir()
            from matplotlib.axes import Axes
            original_bar = Axes.bar
            explicit_colors = []

            def record_bar(axis, *args, **kwargs):
                explicit_colors.append(kwargs.get('color'))
                return original_bar(axis, *args, **kwargs)

            with patch.object(Axes, 'bar', record_bar):
                names = plot(result, output)
            self.assertTrue(explicit_colors)
            self.assertTrue(all(v is not None for v in explicit_colors), 'Every feature must keep one color across modes, independent of the Matplotlib call cycle')
            write_report(path, output, result, names)
            self.assertEqual(set(names), {'acceptance', 'warm_tps', 'per_document_delta',
                'cold_and_accounted_service', 'stage_latency', 'runtime_mechanisms'})
            for name in names:
                self.assertTrue((output/(name+'.png')).read_bytes().startswith(b'\x89PNG'))
                self.assertIn('<svg', (output/(name+'.svg')).read_text())
            text = (output/'REPORT.md').read_text()
            for required in ('NOT held-out', 'not per-mode', 'non-contiguous', '12 causal features', 'runtime.jsonl'):
                self.assertIn(required.lower(), text.lower())


if __name__ == '__main__':
    unittest.main()
