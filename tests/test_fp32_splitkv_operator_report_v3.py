"""CPU-only saved-data validation, mutations and standalone report reproduction.

Synthetic schema fixtures below reuse archived rows with an explicit label;
they are not independent CUDA observations or operator execution.
"""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'exp/round_overhead/fp32_splitkv_operator_report_v3.py'
ARCHIVE=ROOT/'exp/round_overhead/out_fp32_splitkv_operator_20261021_v3'
spec=importlib.util.spec_from_file_location('splitkv_report_v3_tested',SOURCE)
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')


def update_ledgers(root):
    terminal=json.loads((root/'results/TERMINAL.json').read_bytes())
    terminal['cases_sha256']=report.sha(root/'results/CASES.jsonl');dump(root/'results/TERMINAL.json',terminal)
    exitrow=json.loads((root/'PIPELINE_EXIT.json').read_bytes())
    exitrow['terminal_sha256']=report.sha(root/'results/TERMINAL.json');dump(root/'PIPELINE_EXIT.json',exitrow)
    receipt=json.loads((root/'COLLECTION_RECEIPT.json').read_bytes())
    receipt['collected_file_sha256']={n:report.sha(root/n) for n in receipt['collected_file_sha256']}
    for key in ('remote_read_only_barrier_before','remote_read_only_barrier_after'):
        receipt[key]['file_sha256']={n:report.sha(root/n) for n in receipt[key]['file_sha256']}
    dump(root/'COLLECTION_RECEIPT.json',receipt)


def fixture(root):
    """SYNTHETIC CPU schema fixture from saved data, not a CUDA rerun."""
    root=root.resolve()
    receipt=json.loads((ARCHIVE/'COLLECTION_RECEIPT.json').read_bytes())
    for name in (*receipt['collected_file_sha256'],'COLLECTION_RECEIPT.json'):
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ARCHIVE/name,target)
    p=json.loads((root/'PREREGISTRATION.json').read_bytes());p['synthetic_program_test_only']=True
    dump(root/'PREREGISTRATION.json',p);update_ledgers(root)
    return root


def mutate_rows(root,mutator):
    path=root/'results/CASES.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
    mutator(rows);path.write_text(''.join(json.dumps(r)+'\n' for r in rows));update_ledgers(root)


class OperatorReportV3Tests(unittest.TestCase):
    def test_actual_saved_v3_is_complete_only_in_candidate_analytic_scope(self):
        s=report.analyze(ARCHIVE)
        self.assertEqual(s['evidence_issues'],[])
        self.assertEqual(s['status'],'COMPLETE_CANDIDATE_ANALYTIC_SMOKE_ONLY')
        self.assertEqual((s['contract_version'],s['fixture_version'],s['observed_cases']),(3,2,112))
        self.assertEqual((len(s['random']),len(s['known']),len(s['not_run_case_ids'])),(96,16,0))
        self.assertEqual((s['candidate_known_true'],s['reference_known_true'],s['reference_known_false']),(16,12,4))
        self.assertEqual((s['jit'][-1]['_stage_one'],s['jit'][-1]['_stage_two']),(12,1))
        self.assertTrue(s['candidate_analytic_smoke_complete']);self.assertFalse(s['numerical_pass'])
        self.assertFalse(s['goal_complete']);self.assertEqual(s['goal_status'],'ACTIVE')
        self.assertEqual(s['reference_equivalence'],'NOT_ESTABLISHED')
        for key in ('model','tokens','acceptance','latency','tps'):self.assertEqual(s[key],'NOT_MEASURED')
        self.assertIsNone(s['confidence_intervals'])
        self.assertEqual(s['collection_verification']['verified_local_file_count'],17)

    def test_synthetic_label_and_production_pins_reject_mutated_fixture(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_V3_SCHEMA_') as t:
            root=fixture(Path(t));self.assertTrue(report.analyze(root)['evidence_issues'])
            with patch.object(report,'RAW_PINS',{}):s=report.analyze(root)
            self.assertEqual(s['evidence_issues'],[]);self.assertTrue(s['synthetic_program_test_only'])
            self.assertTrue(s['fixture_v2_cases_and_inputs_ast_unchanged'])

    def test_candidate_failure_random_promotion_mutation_order_jit_are_rejected(self):
        changes=[lambda r:r[99].__setitem__('candidate_known_bitwise',False),
            lambda r:r[0].__setitem__('random_equivalence_gate','PASS'),
            lambda r:r[0]['inputs_after'][0].__setitem__('version',999),
            lambda r:r[1].__setitem__('case_id','operator_002'),
            lambda r:r[2]['compiled_cache'][0].__setitem__('compiled_entries',0),
            lambda r:r[99].__setitem__('reference_analytic_gate','MANDATORY'),
            lambda r:r.pop()]
        for change in changes:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as t:
                root=fixture(Path(t));mutate_rows(root,change)
                with patch.object(report,'RAW_PINS',{}):s=report.analyze(root)
                self.assertEqual(s['status'],'PARTIAL_EVIDENCE_ERROR');self.assertEqual(s['observed_cases'],0)

    def test_terminal_counts_exit_binding_collection_and_preserved_history_are_checked(self):
        changes=[('results/TERMINAL.json','completed_cases',111),
            ('results/TERMINAL.json','numerical_pass',True),('PIPELINE_EXIT.json','exit_code',1),
            ('PREREGISTRATION.json','contract_version',2),('PREREGISTRATION.json','prior_failures',[])]
        for name,key,value in changes:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as t:
                root=fixture(Path(t));data=json.loads((root/name).read_bytes());data[key]=value;dump(root/name,data);update_ledgers(root)
                with patch.object(report,'RAW_PINS',{}):s=report.analyze(root)
                self.assertTrue(s['evidence_issues']);self.assertFalse(s['candidate_analytic_smoke_complete'])
        with tempfile.TemporaryDirectory() as t:
            root=fixture(Path(t));path=root/'COLLECTION_RECEIPT.json';data=json.loads(path.read_bytes())
            data['remote_read_only_barrier_after']['file_sha256']['results/CASES.jsonl']='0'*64;dump(path,data)
            with patch.object(report,'RAW_PINS',{}):s=report.analyze(root)
            self.assertTrue(any('barrier' in e for e in s['evidence_issues']))

    def test_no_runtime_imports_portable_cli_bytes_and_three_honest_figures(self):
        tree=ast.parse(SOURCE.read_text());imports=[a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        imports += [n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(x.split('.')[0] in ('torch','triton','rescue','exp','transformers') for x in imports))
        with tempfile.TemporaryDirectory(prefix='V3_REPORT_PORTABLE_') as t:
            output=Path(t)/'first';second=Path(t)/'rerun'
            report.write_report(report.analyze(ARCHIVE),ARCHIVE,output)
            with self.assertRaises(ValueError):report.write_report(report.analyze(ARCHIVE),ARCHIVE,output)
            run=subprocess.run([sys.executable,str(output/'analysis_entry.py'),str(ARCHIVE),'--output',str(second)],cwd=t,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertEqual((output/'summary.json').read_bytes(),(second/'summary.json').read_bytes())
            self.assertEqual(len(list(output.glob('*.png'))),3);self.assertEqual(len(list(output.glob('*.svg'))),3)
            svg=(output/'known_output_matrix.svg').read_text()
            self.assertIn('COMPLETE v3',svg);self.assertIn('MANDATORY',svg);self.assertIn('DIAGNOSTIC',svg)
            self.assertIn('16512',svg);self.assertNotIn('PARTIAL',svg)
            text=(output/'REPORT.md').read_text();self.assertIn('not relabeled as passes',text)
            self.assertIn('TPS and latency are NOT_MEASURED',text)

    def test_old_reporter_sources_are_preserved(self):
        for file,pin in [('fp32_splitkv_operator_report.py','116e2de8ae458fa86b5e3db369d81431e180790dbde6580ae6df4a78c012fb43'),
            ('fp32_splitkv_operator_report_v2.py','46e10177b4dd0b452425637b7ba71f5f446306bb441b6e620e315b2f73062f3f')]:
            self.assertEqual(report.sha(ROOT/'exp/round_overhead'/file),pin)


if __name__=='__main__':unittest.main()
