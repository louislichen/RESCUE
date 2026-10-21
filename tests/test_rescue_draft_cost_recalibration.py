"""Sealing/statistical gates and tiny real fixed-M2 execution, not speed data."""
import ast
import copy
from dataclasses import replace
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import weakref
from unittest.mock import patch

import torch

from rescue import draft_cost_recalibration as runner
from rescue.efficient_attention import install_efficient_attention
from rescue.engine import RescueEngine
from rescue.packing import pack_projections
from rescue.round_overhead_run import sha256, write
from test_rescue_draft_runtime import components, config
from test_rescue_system import COST


def profiles():
    # Pooling individual samples yields2, but the two context-medians yield2.25.
    values={(4096,0):[1.,1.,3.],(8192,0):[3.5],(4096,1):[2.],(8192,1):[2.1]}
    return [dict(mode='oldrisk',source_index=246,length=n,repeat=r,proposed=len(v),
        draft_forwards=len(v)+1,cost_samples={'student_step':v},draft_runtime={'request':dict(
            captures=0,capture_host_seconds=0.,replays=len(v)+1)}) for (n,r),v in values.items()]


class RecalibrationHelpers(unittest.TestCase):
    def test_fit_pools_only_repeat0_steps_and_preserves_every_other_field(self):
        old=dict(student_step=9.,verify_beta=[.1,.2],fixed_round=2.,cache_update=3.,
                 metadata={'legacy':True},unknown_key='keep')
        original=copy.deepcopy(old)
        new,fit,validation=runner.fit_student_cost(old,profiles())
        self.assertEqual(fit,dict(sample_count=4,median=2.))
        self.assertEqual(new,dict(old,student_step=2.));self.assertEqual(old,original)
        self.assertTrue(all(v['passed'] for v in validation.values()))
        rows=profiles();rows[-1]['cost_samples']['student_step']=[2.4]
        second,_,_=runner.fit_student_cost(old,rows)
        self.assertEqual(second['student_step'],2.)  # validation never fits

    def test_validation_and_all_warm_capture_gates_fail_closed(self):
        for corruption in ('validation','capture','capture_seconds','samples','missing','duplicate','replays'):
            rows=profiles()
            if corruption=='validation':rows[-1]['cost_samples']['student_step']=[3.]
            if corruption=='capture':rows[0]['draft_runtime']['request']['captures']=1
            if corruption=='capture_seconds':rows[0]['draft_runtime']['request']['capture_host_seconds']=.1
            if corruption=='samples':rows[0]['cost_samples']['student_step'][0]=float('nan')
            if corruption=='missing':rows.pop()
            if corruption=='duplicate':rows.append(copy.deepcopy(rows[0]))
            if corruption=='replays':rows[0]['draft_runtime']['request']['replays']=0
            with self.subTest(corruption=corruption),self.assertRaises(RuntimeError):
                runner.fit_student_cost({'student_step':9.},rows)

    def test_seal_is_exclusive_preserves_raw_old_bytes_and_detects_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);old=output/'external_cost.json';raw=b'{"student_step":9.0,"legacy":7}\n'
            old.write_bytes(raw)
            fitted,fit,validation=runner.fit_student_cost(json.loads(raw),profiles())
            digest=runner.seal_cost(output,old,fitted,fit,validation,{'seed':412,'margin':.05})
            self.assertEqual((output/'cost_old.json').read_bytes(),raw)
            receipt=runner.validate_seal(output,digest)
            self.assertEqual(receipt['seal_execution_index'],8)
            self.assertEqual(receipt['policy_metadata']['margin'],.05)
            self.assertEqual(set(json.loads((output/'seal_manifest.json').read_text())['file_sha256']),
                {'cost_old.json','cost_new.json','fit_receipt.json','COST_SEALED.json'})
            with self.assertRaisesRegex(RuntimeError,'already sealed'):
                runner.seal_cost(output,old,fitted,fit,validation,{})
            (output/'cost_new.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'changed'):runner.validate_seal(output,digest)

    def test_development_cannot_be_hashed_or_opened_before_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);development=output/'not_opened.jsonl'
            with patch.object(runner,'sha256',side_effect=AssertionError('development hash touched')) as digest:
                with self.assertRaisesRegex(RuntimeError,'sealed before'):
                    runner.read_development(development,output,'unsealed')
                digest.assert_not_called()

    def test_locked_documents_train_role_and_exact_natural_context(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(runner,'LENGTHS',(8,16)):
            output=Path(tmp);cal=output/'cal.jsonl';dev=output/'dev.jsonl';old=output/'old.json'
            def row(index):return dict(source_index=index,split='train',document_sha256=f'{index:064x}',
                prefix_ids=[1],suffix_ids=[2],document_ids=list(range(3,19)))
            cal.write_text(json.dumps(row(246))+'\n')
            self.assertEqual(runner.read_calibration(cal,sha256(cal),row(246)['document_sha256'])['split'],'train')
            old.write_text('{"student_step":9}')
            digest=runner.seal_cost(output,old,{'student_step':2.},{},{},{})
            dev.write_text(''.join(json.dumps(row(i))+'\n' for i in runner.DOCUMENTS))
            docs,actual=runner.read_development(dev,output,digest,sha256(dev))
            self.assertEqual(set(docs),set(runner.DOCUMENTS));self.assertEqual(actual,sha256(dev))
            rows=[row(i) for i in runner.DOCUMENTS];rows[0]['split']='test'
            dev.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(RuntimeError,'original train'):
                runner.read_development(dev,output,digest,sha256(dev))

    def test_frozen_arm_contract_and_48_adjacent_randomized_warm_requests(self):
        cfg=config();plan=runner.schedule()
        self.assertEqual(len(plan),48);self.assertEqual(plan,runner.schedule())
        self.assertNotEqual(plan,runner.schedule(5))
        self.assertEqual(len({tuple(r.values()) for r in plan}),48)
        for offset in range(0,len(plan),4):
            group=plan[offset:offset+4]
            self.assertEqual({r['mode'] for r in group},set(runner.ARMS))
            self.assertEqual(len({(r['source_index'],r['length'],r['repeat']) for r in group}),1)
        for arm in runner.ARMS:
            actual=runner.arm_config(cfg,arm)
            self.assertFalse(actual.profile)
            self.assertEqual(actual.length_policy,'risk' if arm in ('oldrisk','newrisk') else 'fixed')
            if arm=='retrieval32_k4_graph':
                self.assertEqual((actual.inject_tokens,actual.injection_strength),(0,0.))
            else:self.assertEqual(actual.inject_tokens,cfg.inject_tokens)
        self.assertEqual(sum(v for k,v in runner.EXPECTED.items() if 'owner' not in k),94)

    def test_called_global_recorder_is_not_shadowed(self):
        tree=ast.parse(inspect.getsource(runner.run))
        assigned={n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
        self.assertFalse(assigned & {'record','context','schedule','progress','write','append'})

    def test_python_source_binding_rejects_old_namespace_even_if_files_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bindings={}
            def imported(name):
                path=root/Path(*name.split('.')).with_suffix('.py')
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# frozen source\n')
                bindings[name]=SimpleNamespace(__file__=str(path))
                return bindings[name]
            with patch.object(runner.importlib,'import_module',side_effect=imported):
                receipt=runner.check_source_bindings(root)
            self.assertEqual(len(receipt),8)
            bindings['exp.selective_kv.bridge'].__file__=str(root/'other_repo/bridge.py')
            with patch.object(runner.importlib,'import_module',side_effect=bindings.__getitem__):
                with self.assertRaisesRegex(RuntimeError,'escaped frozen root'):
                    runner.check_source_bindings(root)

    def test_quality_observer_cycle_is_broken_without_waiting_for_garbage_collection(self):
        class Engine:pass
        engine=Engine();observer=runner.QualityObserver();observer.engine=engine
        engine.round_observer=observer;reference=weakref.ref(engine)
        runner.detach_quality_observer(engine)
        self.assertIsNone(observer.engine)
        del engine
        self.assertIsNone(reference())

    def test_bfloat16_rounding_alone_is_not_nonzero_injection_and_absolute_positions_match(self):
        native=torch.tensor([.1,.2,.3,.4]).reshape(1,1,4,1)
        indices=torch.tensor([0]);positions=torch.tensor([2,0])
        working=native[:,:,positions].to(torch.bfloat16).clone()
        observer=runner.QualityObserver()
        observer.engine=SimpleNamespace(working=SimpleNamespace(values=[working],positions=positions),
            native_s=SimpleNamespace(values=[native]),injector=SimpleNamespace(active={0:(0,indices,None,
                native[0,:,torch.tensor([2])].transpose(0,1).flatten(1))},summaries=[]))
        packet=dict(round_index=0,prefix=4,progress=0,remaining=1,selected=positions,
            draft=[1],checks=[2],accepted=0,budget_features=None)
        observer(packet)
        self.assertFalse(observer.nonzero_injection)
        self.assertEqual(observer.rounds[-1]['injection_evidence'][0]['different_elements'],0)
        working[0,0,0,0]+=.01;observer(packet)
        self.assertTrue(observer.nonzero_injection)
        self.assertEqual(observer.rounds[-1]['injection_evidence'][0]['positions'],[2])


class RealFixedMapperQuality(unittest.TestCase):
    @torch.no_grad()
    def test_real_nonzero_mapper_does_not_enable_any_m3_summary_feature_or_controller(self):
        teacher,student,mapper=components()
        for model in (teacher,student):install_efficient_attention(model);pack_projections(model)
        observer=runner.QualityObserver()
        cfg=replace(config('resident',limit=9),length_policy='fixed',fixed_k=4)
        prompt=list(range(3,22));frozen={k:v.clone() for k,v in mapper.state_dict().items()}
        with RescueEngine(teacher,student,cfg,mapper,None,COST,round_observer=observer) as engine:
            observer.engine=engine
            canonical=engine.reference(prompt)['generated']
            with patch('rescue.engine.compact_attention',side_effect=AssertionError('M3 attention enabled')),\
                 patch('rescue.engine.step_features',side_effect=AssertionError('M3 features enabled')),\
                 runner.record_draft_steps() as steps:
                row=engine.generate(prompt)
            self.assertEqual(row['generated'],canonical)
            self.assertTrue(observer.nonzero_injection)
            self.assertEqual(len(steps),row['draft_forwards'])
            self.assertTrue(all(not q['summary_requested'] and q['summary_count']==0 for q in steps))
            self.assertTrue(all(r['budget_features'] is None and r['summary_count']==0 for r in observer.rounds))
            self.assertTrue(all(e['native_matches_memo'] for r in observer.rounds for e in r['injection_evidence']))
            self.assertTrue(any(e['different_elements']>0 for r in observer.rounds for e in r['injection_evidence']))
        for name,value in mapper.state_dict().items():self.assertTrue(torch.equal(value,frozen[name]))


if __name__=='__main__':unittest.main()
