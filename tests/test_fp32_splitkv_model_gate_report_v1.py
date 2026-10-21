"""CPU-only saved-data and explicitly synthetic metadata/report checks.

No model/operator execution. Actual archive validation becomes runnable only
after collection; the synthetic scope records are not CUDA observations.
"""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'exp/round_overhead/fp32_splitkv_model_gate_report_v1.py'
CURVE=ROOT/'exp/context_curve/out_h100_context_curve_v2_20261020'
ARCHIVE=ROOT/'exp/round_overhead/out_fp32_splitkv_model_20261021_v1'
spec=importlib.util.spec_from_file_location('model_gate_portable_report_test',SOURCE)
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


def synthetic_scope(enabled=True,closed=True):
    """SYNTHETIC CPU metadata only: never actual candidate/model calls."""
    records=[]
    if enabled:
        for action,q,n in (('original',2047,2047),('candidate',5,2052),('candidate',3,2054)):
            records.extend(dict(action=action,query_tokens=q,key_tokens=n,layer=i,outcome='returned',
                fallback_reason='query_length_out_of_scope' if action=='original' else None) for i in range(28))
    return dict(state='closed' if closed else ('active' if enabled else 'disabled'),enabled=enabled,cpu_metadata_only=True,
        tensor_readbacks=0,tensor_allocations_by_adapter=0,token_gate='NOT_RUN',performance_gate='NOT_RUN',
        calls=len(records),candidate_calls=56 if enabled else 0,original_calls=28 if enabled else 0,records=records)


def old_quality(mode):
    name='own_quality.jsonl' if mode=='rescue_graph' else 'baseline_quality.jsonl'
    row=next(r for r in report.jsonl(CURVE/'terminal/results'/name) if r['mode']==mode)
    # Explicit synthetic schema extension of saved quality data, not a new run.
    row.update(all_timing_fields_are_instrumented=True,performance_eligible=False,observer_enabled=True,profile_enabled=False)
    refs=report.read(CURVE/'terminal/results/references.json')
    return row,refs[f"{row['source_index']}:{row['length']}"]


class ModelGateReportV1Tests(unittest.TestCase):
    def test_matrix_is_exact48_unique_cells_and_not_performance(self):
        rows=report.expected();self.assertEqual(len(rows),48)
        self.assertEqual(len({tuple(r[k] for k in ('length','source_index','mode','variant')) for r in rows}),48)
        self.assertEqual([r['execution_index'] for r in rows],list(range(48)))
        self.assertEqual(rows[0]['length'],2048);self.assertEqual(rows[-1]['length'],16384)
        tree=ast.parse(SOURCE.read_text());imports=[a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        imports += [n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(name.split('.')[0] in ('torch','triton','rescue','exp','transformers') for name in imports))

    def test_synthetic_scope_covers_every_layer_and_restoration_is_mandatory(self):
        counts=report.scope_counts(synthetic_scope(), 'candidate')
        self.assertEqual(counts['candidate'],[2]*28);self.assertEqual(counts['original'],[1]*28)
        self.assertEqual(report.scope_counts(synthetic_scope(False),'original')['candidate'],[0]*28)
        changes=[lambda s:s.__setitem__('state','active'),lambda s:s['records'][28].__setitem__('layer',1),
            lambda s:s['records'][28].__setitem__('outcome','started'),lambda s:s.__setitem__('candidate_calls',55)]
        for change in changes:
            with self.subTest(change=change):
                value=synthetic_scope();change(value)
                with self.assertRaises(ValueError):report.scope_counts(value,'candidate')

    def test_saved_quality_accounting_validates_both_arms_without_runtime_imports(self):
        for mode in report.MODES:
            with self.subTest(mode=mode):
                row,ref=old_quality(mode);metric=report.quality_row(row,ref)
                self.assertEqual(metric['acceptance_length'],row['accepted']/row['blocks'])
                self.assertEqual(metric['acceptance_rate'],row['accepted']/row['proposed'])
                for change in (lambda r:r['generated'].__setitem__(0,-1),lambda r:r.__setitem__('accepted',0),
                    lambda r:r.__setitem__('performance_eligible',True),lambda r:r['owner_signature']['runtime'].__setitem__('bound_ptrs',[]),
                    lambda r:r['observed_rounds'][0].__setitem__('accepted',999)):
                    mutated=copy.deepcopy(row);change(mutated)
                    with self.assertRaises(ValueError):report.quality_row(mutated,ref)

    def test_diff_is_semantic_not_timer_equality(self):
        left,_=old_quality('rescue_graph');right=copy.deepcopy(left)
        right['trace'][0]['verify_seconds']=123.0
        self.assertTrue(report.difference(left,right)['trajectory_equal'])
        right['draft_forwards']+=1;right['queries'][0]['query']['sha256']='0'*64
        changes=report.difference(left,right)
        self.assertEqual(changes['work_fields']['draft_forwards'],dict(original=left['draft_forwards'],candidate=right['draft_forwards']))
        self.assertFalse(changes['queries_equal']);self.assertTrue(changes['native_t_equal'])

    def test_old_seal_proof_authenticates_without_new_corpus_or_models(self):
        proof=report.historical_proof(CURVE)
        self.assertEqual(len(proof['protocol']['source_sha256']),167)
        self.assertEqual(len(proof['history']),576)
        self.assertEqual(proof['pins']['FINAL_ARTIFACTS_MANIFEST.json'],report.PRIOR_MANIFEST)

    def test_synthetic_empty_partial_render_is_not_a_success_or_tps_plot(self):
        summary=dict(model_token_scope_complete=False,synthetic_program_test_only=True,valid_quality_rows=0,
            valid_rows=[],paired_changes=[],scope_layers=[],lifecycle=[],differences=[])
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_MODEL_GATE_FIGURES_') as t:
            report.figures(summary,Path(t))
            self.assertEqual(len(list(Path(t).glob('*.png'))),5)
            svg=(Path(t)/'acceptance_quality.svg').read_text()
            self.assertIn('SYNTHETIC CPU SCHEMA FIXTURE',svg);self.assertIn('PARTIAL',svg)
            self.assertIn('No valid completed quality rows',svg);self.assertNotIn('COMPLETE:',svg)
            self.assertFalse(any('tps' in p.name.lower() for p in Path(t).iterdir()))
            self.assertIn('NOT confidence interval',(Path(t)/'paired_quality_changes.svg').read_text())

    @unittest.skipUnless((ARCHIVE/'results/TERMINAL.json').is_file(),'Actual model-gate terminal archive not collected yet; CPU preparation only')
    def test_actual_saved_archive_and_portable_cli_reproduce_summary_bytes(self):
        summary=report.analyze(ARCHIVE,CURVE)
        if report.read(ARCHIVE/'results/TERMINAL.json').get('passed') is True:
            self.assertEqual(summary['issues'],[])
            self.assertEqual(summary['status'],'COMPLETE_FROZEN_TARGET_TOKEN_GATE_ONLY')
            self.assertEqual(summary['valid_quality_rows'],48);self.assertEqual(summary['owners_closed'],48)
            self.assertEqual(len(summary['differences']),24)
        else:
            self.assertEqual(summary['status'],'PARTIAL_TARGET_TOKEN_GATE')
            self.assertFalse(summary['model_token_scope_complete'])
        self.assertFalse(summary['goal_complete'])
        self.assertEqual(summary['tps'],'NOT_MEASURED');self.assertEqual(summary['latency'],'NOT_MEASURED')
        with tempfile.TemporaryDirectory(prefix='MODEL_GATE_PORTABLE_') as t:
            output=Path(t).resolve()/'first';second=Path(t).resolve()/'rerun';report.write_report(summary,ARCHIVE,CURVE,output)
            run=subprocess.run([sys.executable,str(output/'analysis_entry.py'),str(ARCHIVE),'--proof-root',str(output/'proof'),
                '--output',str(second)],cwd=t,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertEqual((output/'summary.json').read_bytes(),(second/'summary.json').read_bytes())


if __name__=='__main__':unittest.main()
