import copy
from dataclasses import replace
import unittest
from unittest.mock import patch
import torch

from rescue.config import SystemConfig
from rescue.engine import RescueEngine
from rescue.policy import RiskModel,FeedbackRisk,choose
from rescue.query_mapper import QueryValueMapper,mapper_from_payload
from rescue.rollout_training import RolloutCollector,replay
from rescue.selection import ContextSelector,LazyReadState,mandatory
from test_rescue_system import models


class ModuleOptimizationTests(unittest.TestCase):
    def test_attention_anchors_and_modular_floor_keep_budget_and_salient_tokens(self):
        cfg=SystemConfig(budget=16,recent=4,selector='hierarchical',coverage_rows=8,
            coverage_candidates=32,coarse_chunk_size=8,coverage_anchor_fraction=.75,
            coverage_attention_weight=.25)
        values=torch.ones(1,64,8)
        weights=torch.arange(1,65).float()[None]
        state=LazyReadState(weights,values)
        selected=ContextSelector(cfg).select(state,state,0)
        fixed=mandatory(64,4,'cpu')
        free=torch.arange(1,60)
        expected=torch.cat((fixed,free[-(16-len(fixed)):])).sort().values
        self.assertEqual(selected.tolist(),expected.tolist())
        self.assertEqual(len(selected.unique()),16)

    def test_query_direction_initializes_from_base_and_changes_with_query(self):
        _,_,base=models()
        model=QueryValueMapper(**base.config)
        result=model.load_state_dict(base.state_dict(),strict=False)
        self.assertFalse(result.unexpected_keys)
        source=torch.randn(3,base.config['source_dim']);native=torch.randn(3,base.config['value_dim'])
        q=torch.randn(base.config['query_dim']);features=torch.randn(3,2)
        a=base.correction(1,source,native,q,features,alpha=.5)
        b=model.correction(1,source,native,q,features,alpha=.5)
        torch.testing.assert_close(a[0],b[0],rtol=1e-5,atol=1e-6)
        with torch.no_grad(): model.residuals[1][-1].weight.normal_(0,.1)
        d1=model.correction(1,source,native,q,features)[2]
        d2=model.correction(1,source,native,-q,features)[2]
        self.assertGreater(float((d1-d2).abs().sum()),0)
        loaded=mapper_from_payload(dict(config=model.config,state_dict=model.state_dict()),'cpu')
        torch.testing.assert_close(loaded.correction(1,source,native,q,features)[0],model.correction(1,source,native,q,features)[0])

    def test_query_mapper_cached_runtime_matches_replay_with_real_direction(self):
        teacher,student,old=models();mapper=QueryValueMapper(**old.config)
        mapper.load_state_dict(old.state_dict(),strict=False)
        with torch.no_grad(): mapper.residuals[1][-1].weight.normal_(0,.05)
        cfg=SystemConfig(budget=8,recent=2,selector='joint',length_policy='fixed',fixed_k=4,
            max_new_tokens=13,injection_layers=(1,),inject_tokens=2)
        for memo in (False,True):
            collector=RolloutCollector(student,max_records=4)
            with RescueEngine(teacher,student,replace(cfg,memoized_gate_affine=memo),mapper,
                draft_observer=collector,round_observer=collector.round) as engine:
                collector.engine=engine
                row=engine.generate([2,4,1,5,3,8,7,6,2,4,9])
            self.assertTrue(collector.records)
            record=collector.records[0]
            logits,_=replay(student,mapper,record)
            torch.testing.assert_close(logits,record['student_logits'][0],rtol=1e-5,atol=1e-6)

    def test_key_conditioned_mapper_cached_runtime_and_replay_agree(self):
        teacher,student,old=models()
        mapper=QueryValueMapper(**old.config,attention_features=student.config.num_key_value_heads)
        mapper.load_state_dict(old.state_dict(),strict=False)
        with torch.no_grad():
            mapper.residuals[1][-1].weight.normal_(0,.05)
            mapper.attention_gates[1].weight.normal_(0,.1)
        cfg=SystemConfig(budget=8,recent=2,selector='joint',length_policy='fixed',fixed_k=4,
            max_new_tokens=13,injection_layers=(1,),inject_tokens=2)
        collector=RolloutCollector(student,max_records=4)
        with RescueEngine(teacher,student,cfg,mapper,draft_observer=collector,round_observer=collector.round) as engine:
            collector.engine=engine
            row=engine.generate([2,4,1,5,3,8,7,6,2,4,9])
        record=collector.records[0]
        logits,_=replay(student,mapper,record)
        torch.testing.assert_close(logits,record['student_logits'][0],rtol=1e-5,atol=1e-6)
        loaded=mapper_from_payload(dict(config=mapper.config,state_dict=mapper.state_dict()),'cpu')
        torch.testing.assert_close(replay(student,loaded,record)[0],logits)

    def test_feedback_is_conditional_censored_and_request_local(self):
        base=RiskModel(group='attention_injected_light')
        prior=[[.8,.19,.01]]*12
        model=FeedbackRisk(base,prior,strength=8,use_attention=False)
        x=torch.zeros(36);original=model.continuation(x)
        model.observe([0,1])
        self.assertEqual(float(model.counts[2:].sum()),0)
        survival=model.continuation(x)
        self.assertGreater(float(survival[0]),float(original[0]))
        self.assertLess(float(survival[1]),float(original[1]))
        with self.assertRaises(ValueError): model.observe([1,0])
        model.reset();torch.testing.assert_close(model.continuation(x),original)

    def test_light_policy_skips_reads_and_feedback_repeated_requests_reset(self):
        teacher,student,mapper=models()
        base=RiskModel(group='attention_injected_light').eval()
        risk=FeedbackRisk(base,[[.7,.29,.01]]*12,strength=4)
        cfg=SystemConfig(budget=8,recent=2,selector='joint',inject_tokens=2,max_new_tokens=13,
            injection_layers=(1,))
        ids=[2,4,1,5,3,8,7,6,2,4,9]
        with patch.object(LazyReadState,'error',side_effect=AssertionError('read errors are unused')):
            with RescueEngine(teacher,student,cfg,mapper,risk) as engine:
                ref=engine.reference(ids)['generated']
                a,b=engine.generate(ids),engine.generate(ids)
        self.assertEqual(a['generated'],ref);self.assertEqual(b['generated'],ref)
        self.assertEqual([(t['cap'],t['accepted']) for t in a['trace']],[(t['cap'],t['accepted']) for t in b['trace']])


if __name__=='__main__': unittest.main()
