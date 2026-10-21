"""CPU-only saved-archive and explicitly synthetic statistical contracts.

Synthetic rows are scalar mock descriptors, never GPU/model observations.
Actual tests inspect saved immutable timing observations; no model/GPU calls.
"""
import ast
import copy
import importlib.util
import math
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'exp/round_overhead/fp32_splitkv_timing_report_v1.py'
spec=importlib.util.spec_from_file_location('portable_timing_report_test',SOURCE)
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
ARCHIVE=ROOT/'exp/round_overhead/out_fp32_splitkv_timing_20261021_v1'
CURVE=ROOT/'exp/context_curve/out_h100_context_curve_v2_20261020'
MODEL=ROOT/'exp/round_overhead/out_fp32_splitkv_model_20261021_v1'


def synthetic_rows(lengths=None):
    """SYNTHETIC scalar mocks, not measured model requests."""
    rows=[];speeds=dict(original_rescue=100.,candidate_rescue=200.,original_retrieval=110.,candidate_retrieval=180.)
    for n in lengths or report.LENGTHS:
        for arm in report.ARM_IDS:
            for i,d in enumerate(report.DOCS):
                for r in (0,1):
                    seconds=(1+.01*i+.02*r)*128/speeds[arm]
                    rows.append(dict(length=n,arm_id=arm,source_index=d,repeat=r,phase='warm',performance_eligible=True,
                        tokens=128,outer_seconds=seconds,accepted=80,blocks=49,proposed=160,draft_forwards=180,
                        prefill_seconds=.1*seconds,decode_seconds=.89*seconds,cleanup_seconds=.01*seconds))
    return rows


class TimingReportV1StatisticalTests(unittest.TestCase):
    def test_pooled_ratio_and_shared_document_cluster_seed_contract(self):
        rows=synthetic_rows((2048,));result=report.paired_statistics(rows);cell=result['contexts']['2048']
        self.assertEqual((result['seed'],result['draws'],result['document_order']),(20261031,10000,list(report.DOCS)))
        for arm in report.ARM_IDS:
            selected=[r for r in rows if r['arm_id']==arm]
            point=sum(r['tokens'] for r in selected)/sum(r['outer_seconds'] for r in selected)
            self.assertAlmostEqual(cell['arms'][arm]['tps']['point'],point)
            self.assertNotAlmostEqual(point,sum(r['tokens']/r['outer_seconds'] for r in selected)/32,places=4)
        primary=cell['contrasts']['primary_candidate_rescue_vs_candidate_retrieval']
        self.assertAlmostEqual(primary['point_percent'],100*(200/180-1))
        for value in primary['ci95_percent']:self.assertAlmostEqual(value,primary['point_percent'])
        self.assertTrue(cell['threshold_statistics_pass']);self.assertFalse(result['all_six_statistical_gates_pass'])
        self.assertEqual(result['missing_contexts'],list(report.LENGTHS[1:]))

    def test_both_repeats_clustered_order_invariant_and_capture_not_subtracted(self):
        rows=synthetic_rows((2048,));result=report.paired_statistics(rows)
        reverse=report.paired_statistics(list(reversed(rows)))
        self.assertEqual(result,reverse)
        changed=copy.deepcopy(rows)
        for row in changed:row['draft_runtime']={'request':{'capture_host_seconds':row['outer_seconds']*.9}}
        self.assertEqual(result,report.paired_statistics(changed))
        for cell in result['contexts'].values():
            self.assertAlmostEqual(cell['arms']['candidate_rescue']['acceptance_length']['point'],80/49)
            self.assertAlmostEqual(cell['arms']['candidate_rescue']['acceptance_rate']['point'],.5)

    def test_full_six_gate_and_descriptive_interaction_are_not_redefined(self):
        rows=synthetic_rows();result=report.paired_statistics(rows)
        self.assertTrue(result['all_six_statistical_gates_pass']);self.assertEqual(len(result['contexts']),6)
        interaction=result['contexts']['2048']['descriptive_interaction']
        self.assertAlmostEqual(interaction['ratio_of_execution_gains_percent'],100*((200/100)/(180/110)-1))
        self.assertIsNone(interaction['confidence_interval']);self.assertTrue(interaction['not_an_extra_success_gate'])
        for row in rows:
            if row['length']==16384 and row['arm_id']=='candidate_rescue':row['outer_seconds']*=1.5
        failed=report.paired_statistics(rows)
        self.assertFalse(failed['all_six_statistical_gates_pass'])
        self.assertFalse(failed['contexts']['16384']['threshold_statistics_pass'])

    def test_preregistered_ci_gate_is_positive_not_lower_above10_percent(self):
        rows=synthetic_rows((2048,))
        for row in rows:
            if row['arm_id']=='candidate_rescue':
                factor=.9 if report.DOCS.index(row['source_index'])<8 else 1.1
                for key in ('outer_seconds','prefill_seconds','decode_seconds','cleanup_seconds'):row[key]*=factor
        cell=report.paired_statistics(rows)['contexts']['2048']
        contrast=cell['contrasts']['primary_candidate_rescue_vs_candidate_retrieval']
        self.assertGreater(contrast['point_percent'],10)
        self.assertGreater(contrast['ci95_percent'][0],0);self.assertLess(contrast['ci95_percent'][0],10)
        self.assertTrue(cell['threshold_statistics_pass'])

    def test_missing_duplicate_cold_foreign_and_nonfinite_rows_not_promoted(self):
        rows=synthetic_rows((2048,))
        self.assertEqual(report.paired_statistics(rows[:-1])['contexts'],{})
        bad_cases=[rows+[copy.deepcopy(rows[0])]]
        for key,value in (('phase','cold_prime'),('outer_seconds',0),('outer_seconds',math.nan),('repeat',2),('source_index',999)):
            changed=copy.deepcopy(rows);changed[0][key]=value;bad_cases.append(changed)
        for changed in bad_cases:
            with self.subTest(value=changed[0]):
                with self.assertRaises(ValueError):report.paired_statistics(changed)

    def test_single_file_has_no_model_runtime_imports_or_ordinary_sd_arm(self):
        tree=ast.parse(SOURCE.read_text());imports=[a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        imports += [n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(name.split('.')[0] in ('torch','triton','rescue','exp','transformers') for name in imports))
        self.assertEqual(len(report.SOURCES),15);self.assertEqual(len(report.owner_order()),24)
        self.assertFalse(any('native' in name for name in report.ARM_IDS))

    def test_synthetic_graph_lifetime_preserves_late_captures_and_private_kv(self):
        signature={role:dict(identity=i*10+1,key_ptr=i*10+2,value_ptr=i*10+3,capacity=4096) for i,role in enumerate(('native_t','native_s','working'),1)}
        signature['runtime']={'bound_ptrs':[signature[role][k] for role in ('native_s','working') for k in ('key_ptr','value_ptr')]}
        delta=dict(captures=1,replays=2,resident_steps=0,capture_host_seconds=.1,shadow_updates=2)
        cold=dict(draft_forwards=2,owner_signature=signature,runtime_stats=dict(mode='cuda_graph',max_graph_entries=128,
            entries=1,graph_keys=[[513,True,True]],**delta),draft_runtime=dict(mode='cuda_graph',request=delta,lifetime=delta))
        previous=report.graph_step(cold)
        warm=copy.deepcopy(cold);update=dict(captures=1,replays=3,resident_steps=0,capture_host_seconds=.05,shadow_updates=3)
        total={key:delta[key]+update[key] for key in report.RUNTIME};warm['draft_forwards']=3
        warm['runtime_stats'].update(total,entries=2,graph_keys=[[513,True,True],[514,True,True]])
        warm['draft_runtime'].update(request=update,lifetime=total)
        observed=report.graph_step(warm,previous)
        self.assertAlmostEqual(observed['totals']['capture_host_seconds'],.15)
        for mutation in (lambda r:r['runtime_stats'].__setitem__('entries',1),
            lambda r:r['draft_runtime']['request'].__setitem__('capture_host_seconds',0),
            lambda r:r['owner_signature']['runtime'].__setitem__('bound_ptrs',[])):
            broken=copy.deepcopy(warm);mutation(broken)
            with self.assertRaises(ValueError):report.graph_step(broken,previous)

    def test_synthetic_nine_figure_contract_no_tps_claim_from_partial_empty_data(self):
        empty=dict(complete_raw_evidence=False,synthetic_program_test_only=True,statistics=None,owner_diagnostics=[])
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_TIMING_REPORT_EMPTY_') as t:
            report.figures(empty,Path(t))
            self.assertEqual(len(list(Path(t).glob('*.png'))),9);self.assertEqual(len(list(Path(t).glob('*.svg'))),9)
            svg=(Path(t)/'relative_tps_gain.svg').read_text()
            self.assertIn('SYNTHETIC CPU MOCK',svg);self.assertIn('PARTIAL',svg)
            self.assertIn('No complete balanced timing context',svg)
            self.assertFalse(any('native_sd' in p.name for p in Path(t).iterdir()))

    def test_synthetic_full_figures_keep_quality_and_actual_performance_separate(self):
        stats=report.paired_statistics(synthetic_rows());owners=[]
        for n,arm in report.owner_order():
            warm=stats['contexts'][str(n)]['arms'][arm]['totals']['outer_seconds'];candidate=arm.startswith('candidate')
            owners.append(dict(length=n,arm_id=arm,constructor_seconds=.02,cold_seconds=.4,close_seconds=.01,
                warm_seconds=warm,finite_service_tps=4224/(warm+.43),warm_captures=2,warm_capture_host_seconds=.002,
                graph_replays=2000,graph_entries=8,target_layer_calls={'candidate':[1024 if candidate else 0]*28,'original':[33 if candidate else 0]*28},
                scope_qn_distribution=[dict(action='candidate',query_tokens=q,key_tokens=n+offset,calls=512*28) for q,offset in ((2,12),(5,31))] if candidate else []))
        summary=dict(complete_raw_evidence=True,synthetic_program_test_only=True,statistics=stats,owner_diagnostics=owners)
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_TIMING_REPORT_FULL_') as t:
            report.figures(summary,Path(t))
            for name in report.FIGURES:self.assertTrue((Path(t)/(name+'.png')).is_file())
            self.assertIn('SYNTHETIC CPU MOCK',(Path(t)/'warm_tps.svg').read_text())
            self.assertIn('Actual input context length',(Path(t)/'acceptance_length.svg').read_text())
            self.assertIn('pointwise95%',(Path(t)/'relative_tps_gain.svg').read_text())


@unittest.skipUnless((ARCHIVE/'COLLECTION_RECEIPT.json').is_file(),'Actual collected archive is not installed')
class TimingReportV1SavedArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.actual=report.analyze(ARCHIVE,CURVE,MODEL)

    def test_actual_792_complete_canonical_and_gate_not_silently_relaxed(self):
        actual=self.actual
        self.assertEqual(actual['status'],'COMPLETE_MATCHED_DEVELOPMENT_TIMING')
        self.assertEqual(actual['issues'],[]);self.assertEqual(actual['warnings'],[])
        self.assertEqual(actual['counts'],dict(prime_requests=24,timing_requests=768,owners=24))
        self.assertTrue(actual['all_outputs_canonical']);self.assertFalse(actual['scientific_gate_pass'])
        self.assertFalse(actual['goal_complete']);self.assertEqual(len(actual['owner_diagnostics']),24)
        self.assertEqual(len(actual['work_differences']),384)
        self.assertTrue(all(not r['work_fields'] and r['trajectory_equal'] for r in actual['work_differences']))
        self.assertTrue(actual['collection']['saved_external_barriers_checked'])
        for n in report.LENGTHS:
            cell=actual['statistics']['contexts'][str(n)]
            self.assertFalse(cell['primary_point_exceeds10_percent'])
            for arm in report.ARM_IDS:
                rows=[r for r in actual['valid_warm_rows'] if (r['length'],r['arm_id'])==(n,arm)]
                self.assertEqual(len(rows),32)
                self.assertAlmostEqual(cell['arms'][arm]['tps']['point'],sum(r['tokens'] for r in rows)/sum(r['outer_seconds'] for r in rows),places=12)
            for method in ('rescue','retrieval'):
                self.assertEqual(cell['arms']['candidate_'+method]['acceptance_length'],cell['arms']['original_'+method]['acceptance_length'])
                self.assertEqual(cell['arms']['candidate_'+method]['acceptance_rate'],cell['arms']['original_'+method]['acceptance_rate'])
        self.assertLess(actual['statistics']['contexts']['16384']['contrasts']['primary_candidate_rescue_vs_candidate_retrieval']['ci95_percent'][1],0)

    def test_actual_saved_single_file_portable_cli_reproduces_summary(self):
        with tempfile.TemporaryDirectory(prefix='ACTUAL_TIMING_PORTABLE_CPU_') as t:
            base=Path(t).resolve();saved=base/'saved_analysis.py';shutil.copyfile(SOURCE,saved)
            # Copy only sealed proof files, never a model, repository module or weights.
            for source,names,sub in ((CURVE,report.PROOF_FILES,'curve'),(MODEL,tuple(report.MODEL_PINS),'model')):
                for name in names:
                    target=base/sub/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,target)
            output=base/'report';env=dict(os.environ,MPLCONFIGDIR=str(base/'mpl'),PYTHONPATH='')
            run=subprocess.run([sys.executable,'-I',str(saved),str(ARCHIVE),'--curve-proof-root',str(base/'curve'),
                '--model-proof-root',str(base/'model'),'--output',str(output)],cwd=base,env=env,text=True,capture_output=True)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertEqual((output/'summary.json').read_bytes(),(json.dumps(self.actual,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
            self.assertEqual(report.sha(saved),report.sha(output/'analysis_entry.py'))
            self.assertEqual(len(list(output.glob('*.png'))),9);self.assertEqual(len(list(output.glob('*.svg'))),9)
            self.assertIn('not fresh confirmation',(output/'REPORT.md').read_text())

    def test_actual_finite_legend_outside_axes_and_replay_padding(self):
        import matplotlib.figure
        observed={}
        def inspect(fig,path,*args,**kwargs):
            name=Path(path).stem
            if name=='finite_owner_costs':
                self.assertIsNone(fig.axes[0].get_legend())
                self.assertEqual(len(fig.legends),1)
                renderer=fig.canvas.get_renderer()
                legend=fig.legends[0].get_window_extent(renderer)
                self.assertGreater(legend.y0,max(ax.get_window_extent(renderer).y1 for ax in fig.axes))
                observed[name]=True
            elif name=='graph_mechanisms':
                ax=fig.axes[2];highest=max(r['graph_replays'] for r in self.actual['owner_diagnostics'])
                self.assertGreaterEqual(ax.get_ylim()[1],highest*1.03)
                observed[name]=True
        with tempfile.TemporaryDirectory(prefix='ACTUAL_DISPLAY_QA_CPU_') as t:
            with mock.patch.object(matplotlib.figure.Figure,'savefig',inspect):report.figures(self.actual,Path(t))
        self.assertEqual(observed,dict(finite_owner_costs=True,graph_mechanisms=True))


if __name__=='__main__':unittest.main()
