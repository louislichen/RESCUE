from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

import torch

from rescue.architecture_training import ArchitectureCollector,fit_budget,fit_mapper
from rescue.config import SystemConfig
from rescue.engine import RescueEngine
from rescue.policy import RiskModel
from rescue.query_mapper import mapper_from_payload
from rescue.residual_mapper import NativeResidualMapper
from rescue.rollout_training import replay
from rescue.step_budget import StepBudget,compatible_classes,outcome_tables
from test_rescue_system import models,COST


class ArchitectureTests(unittest.TestCase):
    def test_native_residual_identity_init_and_nonzero_cached_replay(self):
        teacher,student,old=models()
        mapper=NativeResidualMapper(layers=[1],teacher_layers=[2],source_dim=16,value_dim=8,
            query_dim=24,rank=4,residual_rank=3,gate_hidden=4)
        source,native=torch.randn(3,16),torch.randn(3,8)
        torch.testing.assert_close(mapper.correction(0,source,native,torch.randn(24),torch.randn(3,2))[0],native)
        with torch.no_grad():
            mapper.decoders[0].weight.normal_(0,.1)
            mapper.residuals[0][-1].weight.normal_(0,.1)
        loaded=mapper_from_payload(dict(config=mapper.config,state_dict=mapper.state_dict()),'cpu')
        for memo in (False,True):
            cfg=SystemConfig(budget=8,recent=2,selector='joint',length_policy='fixed',max_new_tokens=15,
                inject_tokens=2,injection_layers=(1,),memoized_gate_affine=memo)
            collector=ArchitectureCollector(student,max_records=5)
            with RescueEngine(teacher,student,cfg,loaded,draft_observer=collector,round_observer=collector.round) as engine:
                collector.engine=engine
                ref=engine.reference([2,4,1,5,3,8,7,6,2,4,9])['generated']
                row=engine.generate([2,4,1,5,3,8,7,6,2,4,9])
            self.assertEqual(row['generated'],ref)
            record=collector.records[0]
            torch.testing.assert_close(replay(student,loaded,record)[0],record['student_logits'][0],rtol=1e-5,atol=1e-6)
            self.assertTrue(collector.budget_frames)
            self.assertEqual(len(collector.budget_frames[0]['x']),12)

    def test_censored_short_budget_does_not_fabricate_tail_labels(self):
        self.assertEqual(compatible_classes([0,0],2).nonzero().flatten().tolist(),[2,3,4,7,8])
        self.assertEqual(compatible_classes([0,1],4).nonzero().flatten().tolist(),[1])
        self.assertEqual(compatible_classes([0,2],2).nonzero().flatten().tolist(),[6])
        self.assertEqual(compatible_classes([0,0,0,0],4).nonzero().flatten().tolist(),[4])
        with self.assertRaises(ValueError):compatible_classes([1,0],2)

    def test_step_outcome_tables_respect_eos_bonus_and_output_limits(self):
        for remaining in (1,2,3,4,9):
            committed,seconds=outcome_tables(1000,remaining,COST)
            for cls,accepted in enumerate([0,1,2,3,4,1,2,3,4]):
                for col,k in enumerate((min(2,remaining),min(4,remaining))):
                    eos=cls>=5
                    n=min(accepted,k);end=eos and accepted<=k
                    output=min(remaining,n+int(not end))
                    catchup=n==k and not end and output<remaining and not(eos and accepted==k+1)
                    b=COST['verify_beta'];nf=1000/2048
                    expected=b[0]+b[1]*nf+b[2]*k+b[3]*k*nf+(k+catchup)*COST['student_step']+COST['cache_update']
                    self.assertEqual(int(committed[cls,col]),output)
                    self.assertAlmostEqual(float(seconds[cls,col]),expected,places=7)

    def test_step_budget_changes_caps_and_keeps_target_across_requests(self):
        teacher,student,mapper=models();policy=StepBudget(prior_weight=0,margin=0)
        with torch.no_grad():
            for p in policy.parameters():p.zero_()
            policy.net[-1].bias[0]=20
        cfg=SystemConfig(budget=8,recent=2,selector='joint',inject_tokens=2,max_new_tokens=17,
            injection_layers=(1,))
        with RescueEngine(teacher,student,cfg,mapper,policy,COST) as engine:
            ids=[2,4,1,5,3,8,7,6,2,4,9]
            ref=engine.reference(ids)['generated'];a=engine.generate(ids);b=engine.generate(ids)
        self.assertEqual(a['generated'],ref);self.assertEqual(b['generated'],ref)
        self.assertEqual(a['injection_condition_steps'],0)
        self.assertTrue(all(t['cap']<=2 for t in a['trace']))
        self.assertEqual([(t['cap'],t['accepted']) for t in a['trace']],[(t['cap'],t['accepted']) for t in b['trace']])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'policy.pt';torch.save(policy.payload(),path)
            loaded=RiskModel.load(path,'cpu')
            self.assertEqual(policy.decide(torch.zeros(12),COST,1000,10)[0],loaded.decide(torch.zeros(12),COST,1000,10)[0])

    def test_censored_budget_fit_is_finite_and_base_defaults_are_eligible(self):
        frames=[dict(x=[float(i%3)]*12,labels=[0,0],proposed=2,remaining=12,prefix=1000+i) for i in range(4)]
        frames+= [dict(x=[float(i%3)]*12,labels=[0,1],proposed=4,remaining=12,prefix=1100+i) for i in range(4)]
        payloads=[dict(source_index=i,budget_frames=frames) for i in (1,2)]
        with tempfile.TemporaryDirectory() as tmp:
            model,result=fit_budget(payloads,COST,Path(tmp),'test',epochs=3)
            self.assertEqual(result['fit_censored'],4)
            self.assertTrue(torch.isfinite(model(torch.zeros(12))).all())

    def test_prefix_mapper_training_keeps_base_frozen_and_remaps_layer(self):
        teacher,student,old=models()
        cfg=SystemConfig(budget=8,recent=2,selector='joint',length_policy='fixed',max_new_tokens=13,
            inject_tokens=2,injection_layers=(1,))
        collector=ArchitectureCollector(student,max_records=4)
        with RescueEngine(teacher,student,cfg,old,draft_observer=collector,round_observer=collector.round) as engine:
            collector.engine=engine;engine.generate([2,4,1,5,3,8,7,6,2,4,9])
        mapper=NativeResidualMapper(layers=[1],teacher_layers=[2],source_dim=16,value_dim=8,
            query_dim=24,rank=4,residual_rank=3,gate_hidden=4)
        payloads=[dict(source_index=i,length=11,phase='test',records=collector.records) for i in (1,2)]
        with tempfile.TemporaryDirectory() as tmp:
            model,result=fit_mapper(student,mapper,payloads,Path(tmp),'test',epochs=1)
            self.assertEqual(result['fit_queries'],len(collector.records))
            self.assertTrue(all(p.grad is None for p in student.parameters()))
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))


if __name__=='__main__':unittest.main()
