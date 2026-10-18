from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from rescue.config import SystemConfig
from rescue.engine import RescueEngine
from rescue.objectives import accounting, distribution_quality, horizon_outcome, margin_guard_loss
from rescue.policy import RiskModel, HazardMixture
from test_rescue_system import models


class ObjectiveTests(unittest.TestCase):
    def test_rejected_first_token_and_tail_are_separate_from_catchup_work(self):
        row=dict(tokens=5,proposed=7,accepted=3,draft_forwards=8,total_seconds=2.,decode_seconds=1.5,
            trace=[dict(proposed=4,accepted=1,workset_tokens=8),dict(proposed=3,accepted=2,workset_tokens=8)])
        v=accounting(row)
        self.assertEqual(v['wasted_proposals'],4)
        self.assertEqual(v['avoidable_rejected_tail'],2)
        self.assertEqual(v['draft_forwards_per_committed'],1.6)
        self.assertEqual(v['verify_rounds_per_100_committed'],40.)

    def test_horizon_counterfactual_observes_rejection_censoring(self):
        frame=dict(proposed=12,features=[0]*4+[1.],labels=[0,0,1])
        self.assertEqual(horizon_outcome(frame,2),dict(accepted=2,committed=3,waste=0,avoidable_tail=0,catchup=True))
        self.assertEqual(horizon_outcome(frame,8)['avoidable_tail'],5)
        frame['labels']=[0]
        with self.assertRaises(ValueError): horizon_outcome(frame,4)

    def test_quality_and_margin_preservation_detect_native_harm(self):
        native=torch.tensor([3.,1.,0.])
        harmed=torch.tensor([0.,2.,1.],requires_grad=True)
        target=torch.tensor([4.,0.,0.])
        self.assertFalse(distribution_quality(harmed,target)['top1_matches_target'])
        bad=margin_guard_loss(harmed,target,torch.tensor(0.),native)
        good=margin_guard_loss(native,target,torch.tensor(0.),native)
        self.assertGreater(float(bad),float(good))
        bad.backward()
        self.assertLess(float(harmed.grad[0]),0)

    @torch.no_grad()
    def test_prior_mixture_serialization_and_featureless_fast_path(self):
        base=RiskModel(group='attention_injected').eval()
        prior=torch.tensor([[.7,.29,.01]]*12)
        model=HazardMixture(base,prior,1.)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'risk.pt'
            torch.save(dict(group=base.group,horizon=12,state_dict=base.state_dict(),temperature=1.,
                prior_hazards=prior.tolist(),prior_weight=1.),p)
            loaded=RiskModel.load(p,'cpu')
            torch.testing.assert_close(loaded.continuation(torch.rand(36)),model.continuation(torch.rand(36)))
        teacher,student,mapper=models()
        cfg=SystemConfig(budget=8,recent=2,selector='joint',max_new_tokens=12,inject_tokens=2)
        with patch('rescue.scheduler.features.prepare',side_effect=AssertionError('unused feature work')):
            with RescueEngine(teacher,student,cfg,mapper,model) as engine:
                ref=engine.reference([2,4,1,5,3,8,7,6,2,4,9])['generated']
                row=engine.generate([2,4,1,5,3,8,7,6,2,4,9])
        self.assertEqual(row['generated'],ref)
        self.assertEqual(row['injection_condition_steps'],0)

    @torch.no_grad()
    def test_early_publication_and_profile_keep_decisions_and_trajectories(self):
        teacher,student,mapper=models()
        risk=RiskModel(group='attention_injected').eval()
        cfg=SystemConfig(budget=8,recent=2,selector='joint',max_new_tokens=13,inject_tokens=2,
            injection_layers=(1,),schedule='async')
        ids=[2,4,1,5,3,8,7,6,2,4,9]
        with RescueEngine(teacher,student,cfg,mapper,risk) as engine:
            original=engine.generate(ids)
        vectors=[]
        def capture(module,args): vectors.append(args[0].clone())
        hook=risk.register_forward_pre_hook(capture)
        try:
            with RescueEngine(teacher,student,replace(cfg,early_length_publish=True,profile=True),mapper,risk) as engine:
                new=engine.generate(ids)
        finally: hook.remove()
        self.assertEqual(original['generated'],new['generated'])
        self.assertEqual([(r['cap'],r['accepted']) for r in original['trace']],
            [(r['cap'],r['accepted']) for r in new['trace']])
        self.assertEqual(len(vectors),new['blocks'])
        self.assertEqual(new['timeline']['stages']['length_deadline']['calls'],new['blocks'])


if __name__=='__main__': unittest.main()
