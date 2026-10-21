"""Synthetic audit fixtures only; none of these numbers are model results."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from rescue.draft_graph_report import MODES, analyze, plot, row_errors


def save_rows(path, values):
    path.write_text(''.join(json.dumps(v)+'\n' for v in values))


def fixture(directory):
    directory = Path(directory)
    reference = list(range(32))
    trace = [dict(round_index=i, prefix=4096+4*i, cap=4, proposed=4,
                  accepted=3, terminal=i == 7, workset_tokens=512) for i in range(8)]
    base = dict(generated=reference, tokens=32, output_agrees=True, trace=trace,
                blocks=8, accepted=24, proposed=32, draft_forwards=8,
                total_seconds=2., decode_seconds=1.5, total_tps=16., decode_tps=32/1.5)
    quality = []
    for mode in MODES:
        quality.append(dict(copy.deepcopy(base), request_id='quality-'+mode, mode=mode,
            observed_rounds=[dict(budget_features=[0.]*11+[.1], selected=[1, 2])],
            nonzero_injection=True, queries=[dict(query='same') for _ in range(8)],
            native_t=dict(lengths=[4128], layers=['same']), native_s=dict(lengths=[4128], layers=['same']),
            runtime_stats=dict(captures=1 if mode == 'cuda_graph' else 0,
                               replays=8 if mode == 'cuda_graph' else 0)))
    runtime, sessions = [], []
    for repeat in range(2):
        for mode in MODES:
            request_ids = []
            for phase in ('cold', 'warm'):
                request_id = f'{mode}-{repeat}-{phase}'; request_ids.append(request_id)
                duration = 2.5 if phase == 'cold' else 2.
                runtime.append(dict(copy.deepcopy(base), request_id=request_id, mode=mode,
                    repeat=repeat, phase=phase, outer_seconds=duration, outer_tps=32/duration,
                    constructor_seconds=.5 if phase == 'cold' else 0., draft_runtime=dict(request=dict(
                        captures=1 if mode == 'cuda_graph' and phase == 'cold' else 0,
                        replays=8 if mode == 'cuda_graph' else 0,
                        capture_host_seconds=.1 if mode == 'cuda_graph' and phase == 'cold' else 0.))))
            sessions.append(dict(mode=mode, repeat=repeat, request_ids=request_ids, tokens=64,
                session_seconds=5., session_tps=12.8, active_sum_seconds=4.6, close_seconds=.1))
    protocol = dict(source_index=303, length=4096, max_new_tokens=32, repeats=2,
                    source_sha256={})
    for role in ('baseline_eager', 'new'):
        source = directory/'source'/role/'rescue'/'synthetic.py'
        source.parent.mkdir(parents=True)
        source.write_text('# synthetic fixture only\n')
        protocol['source_sha256'][role] = {'rescue/synthetic.py': hashlib.sha256(source.read_bytes()).hexdigest()}
    for name, value in (('protocol.json', protocol), ('reference.json', reference),
            ('COMPLETE.json', dict(passed=True, counts=dict(reference=1, correctness=3, timing=12, sessions=6),
                                  sources_unchanged=True, assets_unchanged=True))):
        (directory/name).write_text(json.dumps(value))
    save_rows(directory/'reference.jsonl', [dict(generated=reference, request_id='canonical')])
    for name, values in (('correctness.jsonl', quality), ('runtime.jsonl', runtime), ('sessions.jsonl', sessions)):
        save_rows(directory/name, values)
    return quality, runtime, sessions


class DraftGraphReportTests(unittest.TestCase):
    def test_complete_one_document_is_not_independent_replication_or_ci(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp); result = analyze(Path(tmp))
        self.assertEqual(result['status'], 'COMPLETE', result['problems'])
        self.assertEqual(result['usable_repeats'], [0, 1])
        self.assertEqual(result['documents'], 1)
        self.assertIsNone(result['confidence_intervals'])
        self.assertEqual(result['metrics']['cuda_graph']['accepted_per_round'], 3.)
        self.assertEqual(result['metrics']['cuda_graph']['session']['tps'], 12.8)

    def test_canonical_ar_needs_its_own_uuid(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp); path = Path(tmp)
            save_rows(path/'reference.jsonl', [dict(generated=list(range(32)))])
            result = analyze(path)
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertIn('Missing/reused request UUID', result['problems'])

    def test_zero_proposals_are_rejected_before_rate_division(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, runtime, _ = fixture(tmp)
            bad = runtime[0]
            for t in bad['trace']:
                t.update(proposed=0, accepted=0)
            bad.update(proposed=0, accepted=0)
            self.assertIn('Invalid bounded 2/4 trajectory', row_errors(bad, list(range(32))))
            save_rows(Path(tmp)/'runtime.jsonl', runtime)
            result = analyze(Path(tmp))
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertEqual(result['usable_repeats'], [1])

    def test_missing_request_excludes_whole_matched_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, runtime, _ = fixture(tmp)
            save_rows(Path(tmp)/'runtime.jsonl', runtime[:-1])
            result = analyze(Path(tmp))
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertEqual(result['usable_repeats'], [0])

    def test_four_separate_figures_render_png_and_svg(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp); path = Path(tmp); output = path/'analysis'; output.mkdir()
            names = plot(analyze(path), output)
            self.assertEqual(set(names), {'acceptance', 'cold_warm_tps', 'session_tps', 'latency'})
            for name in names:
                self.assertTrue((output/(name+'.png')).read_bytes().startswith(b'\x89PNG'))
                self.assertIn('<svg', (output/(name+'.svg')).read_text())


if __name__ == '__main__':
    unittest.main()
