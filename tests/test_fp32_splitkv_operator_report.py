"""CPU report checks: synthetic schema mutations and saved-archive CLI only.

Synthetic tests bypass ONLY external raw SHA pins; source, layouts, descriptor
matrix, counters and lifecycle checks execute. No operator/runtime is imported.
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
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'exp/round_overhead/fp32_splitkv_operator_report.py'
ARCHIVE=ROOT/'exp/round_overhead/out_fp32_splitkv_operator_20261021_v1'
spec=importlib.util.spec_from_file_location('splitkv_report_tested',SOURCE)
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
def update_collection(root):
    names=set(report.SOURCES)|{'results/PLAN.json','results/CASES.jsonl','results/TERMINAL.json','PIPELINE_EXIT.json',
        'PREREGISTRATION.json','SOURCE_SHA256SUMS','LAUNCH_STARTED.json','operator.log'}
    barrier={n:report.sha(root/n) for n in ('PIPELINE_EXIT.json','results/TERMINAL.json','results/CASES.jsonl')}
    dump(root/'COLLECTION_RECEIPT.json',dict(status='TERMINAL_FAILURE_COLLECTED_READ_ONLY',scp_exit_codes=[0,0,0],operator_cases_recorded=100,operator_cases_planned=112,
        collected_file_sha256={n:report.sha(root/n) for n in names},remote_read_only_barrier_before=dict(command_exit=0,file_sha256=barrier),
        remote_read_only_barrier_after=dict(command_exit=0,file_sha256=barrier),process_observations_before_collection=dict(
            tmux_has_session_exit=1,ps_pid_and_direct_children_exit=1,nvidia_compute_apps_exit=0,nvidia_compute_apps_stdout='')))


def fixture(root):
    root=root.resolve()
    for name in report.SOURCES:
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ARCHIVE/name).read_bytes())
    # Actual source bytes are proof templates only, never imported or run.
    p=json.loads((ARCHIVE/'PREREGISTRATION.json').read_bytes());p['synthetic_program_test_only']=True;dump(root/'PREREGISTRATION.json',p)
    plan=json.loads((ARCHIVE/'results/PLAN.json').read_bytes());dump(root/'results/PLAN.json',plan)
    (root/'SOURCE_SHA256SUMS').write_bytes((ARCHIVE/'SOURCE_SHA256SUMS').read_bytes())
    (root/'LAUNCH_STARTED.json').write_text('SYNTHETIC CPU fixture\n');(root/'operator.log').write_text('SYNTHETIC CPU fixture\n')
    rows=[]
    for i,c in enumerate(report.expected_cases()[:100]):
        q,n=c['q'],c['n'];full=c['mask'].endswith('full')
        shapes=([4,7,q,128],[4,7,n,128],[4,7,n,128],None if c['mask']=='none' else ([4,7,q,n] if full else [1,1,q,n]))
        snapshots=[]
        for j,shape in enumerate(shapes):
            if shape is None:snapshots.append(None);continue
            stride=[896,128,3584,1] if j==0 else ([n*128,0,128,1] if j in (1,2) else [shape[1]*q*n,q*n,n,1])
            snapshots.append(dict(shape=shape,stride=stride,dtype='torch.bool' if j==3 and c['mask'].startswith('bool') else 'torch.float32',
                device='cuda:0',pointer=10000+i*100+j,version=0,storage_offset=0,is_conj=False,is_neg=False,backing_valid_bytes=128,backing_sha256=f'{i+1:064x}'))
        known=c['exact'];failed=i==99;equal=known and not failed
        row=dict(c,finite=True,output_contract=True,inputs_before=snapshots,inputs_after=copy.deepcopy(snapshots),
            output_logical_bytes_sha256=dict(candidate=f'{i+1:064x}',reference=f'{i+1 if equal else i+1000:064x}'),
            errors_diagnostic_only=dict(max_abs=0. if equal else (4.76837158203125e-7 if failed else 1e-7),rms=0. if equal else 1e-8,
                max_relative_floor1e6=0. if equal else .001,candidate_reference_bitwise=equal),candidate_known_bitwise=True if known else None,
            reference_known_bitwise=not failed if known else None,mandatory_checks_passed=not failed,
            status='FAIL' if failed else ('EXACT_SIMPLE_CASE_PASSED' if known else 'DIAGNOSTIC_COMPLETE'),random_equivalence_gate='NOT_DEFINED',
            compiled_cache=[dict(kernel=name,device=0,compiled_entries=count,key_sha256s=[f'{k+1:064x}' for k in range(count)])
                for name,count in (('_stage_one',min(9,i+1)),('_stage_two',1))],
            runtime_specials=dict(N=n,non_specialized=list(report.DYNAMIC),q_strides=snapshots[0]['stride'],k_strides=snapshots[1]['stride'],
                v_strides=snapshots[2]['stride'],mask_strides=None if snapshots[3] is None else snapshots[3]['stride']),**dict.fromkeys(report.GATES,'NOT_RUN'))
        rows.append(row)
    (root/'results/CASES.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    terminal=json.loads((ARCHIVE/'results/TERMINAL.json').read_bytes());dump(root/'results/TERMINAL.json',terminal)
    exitrow=json.loads((ARCHIVE/'PIPELINE_EXIT.json').read_bytes());exitrow['terminal_sha256']=report.sha(root/'results/TERMINAL.json');dump(root/'PIPELINE_EXIT.json',exitrow)
    update_collection(root);return root


class OperatorReportTests(unittest.TestCase):
    def test_synthetic100_prefix_keeps96_diagnostic_and12_known_not_run(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_OPERATOR_REPORT_') as t:
            root=fixture(Path(t))
            with patch.object(report,'RAW_PINS',{}):s=report.analyze(root)
            self.assertEqual(s['evidence_issues'],[]);self.assertEqual(s['status'],'PARTIAL_OPERATOR_FAIL')
            self.assertEqual(len(s['random']),96);self.assertEqual(len(s['known']),4);self.assertEqual(len(s['not_run_case_ids']),12)
            self.assertEqual((s['candidate_known_true'],s['reference_known_true'],s['reference_known_false']),(4,3,1))
            self.assertFalse(s['numerical_pass']);self.assertEqual(s['random_numerical_pass'],'NOT_DEFINED');self.assertEqual(s['tps'],'NOT_MEASURED')

    def test_order_counter_input_mutation_jit_and_gate_promotions_rejected(self):
        mutations=[lambda rows:rows[1].__setitem__('case_id','operator_002'),
            lambda rows:rows[0]['inputs_after'][0].__setitem__('version',1),
            lambda rows:rows[2]['compiled_cache'][0].__setitem__('compiled_entries',0),
            lambda rows:rows[0].__setitem__('random_equivalence_gate','PASS'),
            lambda rows:rows[99].__setitem__('reference_known_bitwise',True)]
        for change in mutations:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as t:
                root=fixture(Path(t));path=root/'results/CASES.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()];change(rows)
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows));update_collection(root)
                with patch.object(report,'RAW_PINS',{}):s=report.analyze(root)
                self.assertEqual(s['status'],'PARTIAL_EVIDENCE_ERROR');self.assertEqual(s['observed_cases'],0)
        with tempfile.TemporaryDirectory() as t:
            root=fixture(Path(t));path=root/'results/TERMINAL.json';value=json.loads(path.read_bytes());value['completed_cases']=99;dump(path,value)
            exitrow=json.loads((root/'PIPELINE_EXIT.json').read_bytes());exitrow['terminal_sha256']=report.sha(path);dump(root/'PIPELINE_EXIT.json',exitrow);update_collection(root)
            with patch.object(report,'RAW_PINS',{}):s=report.analyze(root)
            self.assertTrue(any('counter mismatch' in x for x in s['evidence_issues']))

    def test_actual_saved_source_and_raw_pins_verify_then_tamper_rejects(self):
        real=report.analyze(ARCHIVE);self.assertEqual(real['evidence_issues'],[]);self.assertEqual(real['observed_cases'],100)
        self.assertEqual(real['jit'][-1]['_stage_one'],9);self.assertEqual(real['jit'][-1]['_stage_two'],1)
        with tempfile.TemporaryDirectory() as t:
            root=fixture(Path(t));s=report.analyze(root)
            self.assertFalse(s['verified_saved_evidence']);self.assertTrue(any('hash mismatch' in x for x in s['evidence_issues']))

    def test_portable_saved_cli_reproduces_actual_summary_and_three_figures(self):
        tree=ast.parse(SOURCE.read_text());imports=[a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        imports += [n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(x.split('.')[0] in ('torch','triton','rescue','exp','transformers') for x in imports))
        with tempfile.TemporaryDirectory(prefix='OPERATOR_REPORT_CPU_') as t:
            output=Path(t)/'first';second=Path(t)/'saved_cli';s=report.analyze(ARCHIVE);report.write_report(s,ARCHIVE,output)
            self.assertEqual(len(list(output.glob('*.png'))),3);self.assertEqual(len(list(output.glob('*.svg'))),3)
            with self.assertRaises(ValueError):report.write_report(s,ARCHIVE,output)
            run=subprocess.run([sys.executable,str(output/'analysis_entry.py'),str(ARCHIVE),'--output',str(second)],cwd=t,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr);self.assertEqual(json.loads((output/'summary.json').read_bytes()),json.loads((second/'summary.json').read_bytes()))
            self.assertIn('PARTIAL',(output/'known_output_matrix.svg').read_text());self.assertIn('NOT_RUN',(output/'known_output_matrix.svg').read_text())


if __name__=='__main__':unittest.main()
