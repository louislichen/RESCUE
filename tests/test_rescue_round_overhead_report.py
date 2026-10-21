"""Synthetic CPU fixtures test the analyzer; they are not model measurements."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from rescue.round_overhead_report import bootstrap, summarize, write_report


def fixture(directory):
    directory = Path(directory)
    ids, lengths = list(range(100, 106)), [4096, 8192]
    protocol = dict(source_indices=ids, lengths=lengths, repeats=3, max_new_tokens=128,
        max_draft=12, mapper_nonzero=True, mapper_frozen=True, request_buffer_policy='cold_request',
        environment=dict(gpu_name='NVIDIA H100 80GB HBM3'), source_sha256={},
        scope='SYNTHETIC UNIT TEST, NOT MODEL DATA')
    for mode in ('before', 'after'):
        source = directory/'source'/mode/'rescue'/'synthetic.py'
        source.parent.mkdir(parents=True)
        source.write_text('# synthetic unit-test source, never model code\n')
        protocol['source_sha256'][mode] = {'rescue/synthetic.py': hashlib.sha256(source.read_bytes()).hexdigest()}
    refs, rows, correctness = {}, [], []
    for doc in ids:
        for n in lengths:
            generated = list(range(12))
            refs[f'{doc}:{n}'] = generated
            trace = [dict(round_index=i, cap=4, proposed=4, accepted=2, prefix=n-1+3*i,
                          terminal=i == 3, workset_tokens=512) for i in range(4)]
            rounds = [dict(round_index=i, prefix=n-1+3*i, progress=3*i, remaining=128-3*i,
                           selected=[0, 1, 2], budget_features=[0.]*11+[.1],
                           draft=[10, 11, 13, 14], checks=[10, 11, 12, 15, 17], accepted=2)
                      for i in range(4)]
            correctness.append(dict(source_index=doc, length=n, passed=True, output_agrees=True,
                trajectory_equal=True, rounds_equal=True, nonzero_injection=dict(before=True, after=True),
                before=dict(generated=generated, rounds=copy.deepcopy(rounds), trace_signature=copy.deepcopy(trace)),
                after=dict(generated=generated, rounds=copy.deepcopy(rounds), trace_signature=copy.deepcopy(trace))))
            for repeat in range(3):
                for mode in ('before', 'after'):
                    duration = (10.+(doc-100)*.2+n/8192+repeat*.05)/(1.05 if mode == 'after' else 1.)
                    rows.append(dict(mode=mode, source_index=doc, length=n, repeat=repeat,
                        generated=generated, tokens=12, proposed=16, accepted=8, blocks=4,
                        trace=copy.deepcopy(trace), output_agrees=True,
                        prefill_seconds=duration*.1, decode_seconds=duration*.85,
                        cleanup_seconds=duration*.05, total_seconds=duration,
                        total_tps=12/duration, decode_tps=12/(duration*.85)))
    receipt = dict(passed=True, expected_timing_requests=72, timing_requests=72,
                   correctness_pairs=12, reference_requests=12, warmup_requests=4,
                   assets_unchanged=True, source_unchanged=True)
    save(directory, protocol, refs, rows, correctness, receipt)
    return protocol, refs, rows, correctness, receipt


def save(directory, protocol, refs, rows, correctness, receipt):
    directory = Path(directory)
    for name, value in (('protocol.json', protocol), ('references.json', refs), ('COMPLETE.json', receipt)):
        (directory/name).write_text(json.dumps(value))
    for name, values in (('runtime.jsonl', rows), ('correctness.jsonl', correctness)):
        (directory/name).write_text(''.join(json.dumps(row)+'\n' for row in values))


class RoundOverheadReportTests(unittest.TestCase):
    def test_complete_matrix_bootstraps_six_documents_not_36_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp)
            result = summarize(tmp, samples=500)
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertEqual(result['observed_timing_rows'], 72)
        self.assertTrue(all(result['gates'].values()))
        for n in ('4096', '8192'):
            context = result['contexts'][n]
            self.assertEqual(context['n_documents'], 6)
            comparison = context['tps_change_percent']
            self.assertEqual(comparison['n_documents'], 6)
            self.assertAlmostEqual(comparison['mean'], 5., places=10)
            for value in comparison['ci95']:
                self.assertAlmostEqual(value, 5., places=10)
            self.assertEqual(context['acceptance_length_delta']['mean'], 0.)

    def test_missing_repeat_excludes_entire_document_context_and_marks_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, refs, rows, correct, receipt = fixture(tmp)
            rows = [r for r in rows if not (r['mode'] == 'after' and r['source_index'] == 100 and
                                           r['length'] == 4096 and r['repeat'] == 2)]
            save(tmp, p, refs, rows, correct, receipt)
            result = summarize(tmp, samples=500)
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertEqual(result['contexts']['4096']['n_documents'], 5)
        self.assertEqual(result['contexts']['8192']['n_documents'], 6)
        self.assertEqual(len(result['missing_timing_rows']), 1)
        self.assertFalse(result['both_contexts_speed_gate'])
        self.assertFalse(any(result['speed_ci_lower_gt_zero'].values()))

    def test_completion_is_distinct_from_both_contexts_acceleration(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, refs, rows, correct, receipt = fixture(tmp)
            good = summarize(tmp, samples=500)
            self.assertTrue(good['both_contexts_speed_gate'])
            for row in rows:
                if row['mode'] == 'after' and row['length'] == 4096:
                    for field in ('prefill_seconds', 'decode_seconds', 'cleanup_seconds', 'total_seconds'):
                        row[field] *= 1.06
                    row['total_tps'] /= 1.06
                    row['decode_tps'] /= 1.06
            save(tmp, p, refs, rows, correct, receipt)
            result = summarize(tmp, samples=500)
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertEqual(result['speed_ci_lower_gt_zero'], {'4096': False, '8192': True})
        self.assertFalse(result['both_contexts_speed_gate'])

    def test_repeat_median_is_taken_before_document_paired_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, refs, rows, correct, receipt = fixture(tmp)
            for row in rows:
                if row['source_index'] == 100 and row['length'] == 4096 and row['repeat'] == 2:
                    row.update({key: value*100 for key, value in row.items() if key.endswith('_seconds')})
                    row['total_tps'] /= 100
                    row['decode_tps'] /= 100
            save(tmp, p, refs, rows, correct, receipt)
            result = summarize(tmp, samples=500)
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertAlmostEqual(result['contexts']['4096']['tps_change_percent']['mean'], 5., places=10)

    def test_duplicate_or_outside_protocol_row_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, refs, rows, correct, receipt = fixture(tmp)
            save(tmp, p, refs, rows+[copy.deepcopy(rows[0])], correct, receipt)
            with self.assertRaisesRegex(ValueError, 'Duplicate timing row'):
                summarize(tmp)
            rows[0]['length'] = 2048
            save(tmp, p, refs, rows, correct, receipt)
            with self.assertRaisesRegex(ValueError, 'outside protocol'):
                summarize(tmp)

    def test_canonical_output_not_just_boolean_flag_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, refs, rows, correct, receipt = fixture(tmp)
            rows[0]['generated'] = list(range(1, 13))
            save(tmp, p, refs, rows, correct, receipt)
            result = summarize(tmp, samples=500)
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertFalse(result['gates']['all_timing_outputs_and_accounting_valid'])
        self.assertEqual(result['contexts']['4096']['n_documents'], 5)

    def test_unchanged_output_with_changed_cap_fails_trajectory_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, refs, rows, correct, receipt = fixture(tmp)
            rows[0]['trace'][0]['cap'] = 5
            save(tmp, p, refs, rows, correct, receipt)
            result = summarize(tmp, samples=500)
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertFalse(result['gates']['cap_acceptance_trajectory_invariant'])
        self.assertEqual(result['contexts']['4096']['n_documents'], 5)

    def test_bad_seconds_or_acceptance_accounting_is_not_a_valid_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, refs, rows, correct, receipt = fixture(tmp)
            rows[0]['total_seconds'] = float('nan')
            rows[1]['accepted'] = 9
            save(tmp, p, refs, rows, correct, receipt)
            result = summarize(tmp, samples=500)
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertGreaterEqual(len(result['invalid_rows']), 2)
        self.assertEqual(result['contexts']['4096']['n_documents'], 5)

    def test_correctness_nonzero_and_completion_receipt_are_mandatory(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, refs, rows, correct, receipt = fixture(tmp)
            correct[0]['nonzero_injection']['after'] = False
            receipt['timing_requests'] = 71
            p['mapper_frozen'] = False
            save(tmp, p, refs, rows, correct, receipt)
            result = summarize(tmp, samples=500)
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertFalse(result['gates']['nonzero_correctness_exact'])
        self.assertFalse(result['gates']['completion_receipt'])
        self.assertFalse(result['gates']['frozen_mapper'])

    def test_correctness_evidence_is_recomputed_instead_of_trusting_true_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, refs, rows, correct, receipt = fixture(tmp)
            correct[0]['after']['rounds'][0]['selected'] = [0, 2, 3]
            for packet in correct[1]['after']['rounds']:
                packet['budget_features'][-1] = 0.
            save(tmp, p, refs, rows, correct, receipt)
            result = summarize(tmp, samples=500)
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertFalse(result['gates']['nonzero_correctness_exact'])
        self.assertEqual(len(result['correctness_failures']), 2)

    def test_saved_source_bytes_and_unchanged_receipt_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, refs, rows, correct, receipt = fixture(tmp)
            (Path(tmp)/'source'/'before'/'rescue'/'synthetic.py').write_text('# tampered synthetic fixture\n')
            receipt['source_unchanged'] = False
            save(tmp, p, refs, rows, correct, receipt)
            result = summarize(tmp, samples=500)
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertFalse(result['gates']['frozen_sources_hash_match'])
        self.assertFalse(result['gates']['assets_and_sources_unchanged'])
        self.assertEqual(len(result['source_failures']), 1)

    def test_prefill_excludes_current_root_and_prefixes_follow_committed_accepts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, refs, rows, correct, receipt = fixture(tmp)
            self.assertEqual(rows[0]['trace'][0]['prefix'], rows[0]['length']-1)
            rows[0]['trace'][1]['prefix'] += 1
            save(tmp, p, refs, rows, correct, receipt)
            result = summarize(tmp, samples=500)
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertFalse(result['gates']['all_timing_outputs_and_accounting_valid'])

    def test_no_receipt_or_no_rows_never_reports_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, refs, _, _, _ = fixture(tmp)
            save(tmp, p, refs, [], [], {})
            result = summarize(tmp, samples=500)
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertEqual(result['contexts']['4096']['n_documents'], 0)
        self.assertIsNone(result['contexts']['4096']['tps_change_percent'])

    def test_bootstrap_is_reproducible_and_uses_document_vector(self):
        data = [-4., 2., 3., 5., 9., -1.]
        a = bootstrap(data, samples=1000, seed=83)
        b = bootstrap(data, samples=1000, seed=83)
        self.assertEqual(a, b)
        self.assertEqual(a['n_documents'], 6)
        self.assertAlmostEqual(a['mean'], sum(data)/6)
        self.assertLessEqual(a['ci95'][0], a['mean'])
        self.assertGreaterEqual(a['ci95'][1], a['mean'])

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'matplotlib is required to render figures')
    def test_render_source_separate_figures_and_partial_notice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p, refs, rows, correct, receipt = fixture(root)
            save(root, p, refs, rows[:-1], correct, receipt)
            result = write_report(root, root/'analysis', samples=500)
            self.assertEqual(result['status'], 'PARTIAL')
            for stem in ('acceptance_length', 'whole_request_tps', 'stage_latency', 'per_document_tps_delta'):
                self.assertTrue((root/'analysis'/f'{stem}.png').stat().st_size > 1000)
                self.assertIn('PARTIAL', (root/'analysis'/f'{stem}.svg').read_text())
            report = (root/'analysis'/'REPORT.md').read_text()
            self.assertIn('PARTIAL', report)
            self.assertIn('not RTX 4090', report)
            self.assertIn('runtime.jsonl', report)
            self.assertEqual((root/'analysis'/'analysis_source.py').read_bytes(),
                             Path(__file__).parents[1].joinpath('rescue/round_overhead_report.py').read_bytes())
            with self.assertRaisesRegex(ValueError, 'fresh analysis directory'):
                write_report(root, root/'analysis', samples=500)


if __name__ == '__main__':
    unittest.main()
