"""CPU tests for cache identity, causal rewards, and exploration coverage."""
import unittest
from unittest.mock import patch
import numpy as np
import run_csgb_controller as runner
from csgb import LinUCB, features


class Attention:
    def __init__(self, array): self.array=np.asarray(array)
    def detach(self): return self
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self.array


class Model:
    def __init__(self):
        self.selected_chunks=[(0,0,2)]
        self.attn_scores_final=None
    def draft(self, ids, nodes, threshold, max_depth):
        return (np.zeros((1,nodes)),None,None,None)
    def update_working_cache_retrieval(self,*args,**kwargs):
        self.attn_scores_final=None
        return []


class ControllerTests(unittest.TestCase):
    def test_custom_budget_exploration_and_fallback(self):
        m=LinUCB(budgets=(8,16,24,32),fallback=8,min_exploration=4)
        self.assertEqual(m.choose(None),8)
        x=np.ones(4)
        for _ in range(16):
            b=m.choose(x,True)
            self.assertIn(b,(8,16,24,32))
            m.update(x,b,1,1,1)
        self.assertEqual(m.counts.tolist(),[4,4,4,4])
        for budgets,fallback in [((8,8),8),((0,8),8),((8,16),64)]:
            with self.assertRaises(ValueError):LinUCB(budgets=budgets,fallback=fallback)

    def test_cache_reuse_and_invalidations(self):
        model=Model(); h=runner.Harness(model)
        h.reset('csgb',LinUCB())
        current=np.array([.1,.2,.3,.4])
        h.anchor=np.array([.4,.3,.2,.1])
        model.attn_scores_final=Attention(current)
        with patch.object(runner.torch.cuda,'synchronize'), patch.object(runner,'features',wraps=features) as f:
            model.draft(np.zeros((1,5)),64,.7,10)
            np.testing.assert_allclose(h.last_x,features(current,[0,1],h.anchor))
            model.attn_scores_final=None
            model.draft(np.zeros((1,7)),64,.7,10)
            self.assertEqual(f.call_count,1)
            model.selected_chunks=[(0,0,2),(1,4,9)]
            model.draft(np.zeros((1,9)),64,.7,10)
            self.assertEqual(f.call_count,1)
            model.selected_chunks=[(0,0,3)]
            model.draft(np.zeros((1,11)),64,.7,10)
            self.assertEqual(f.call_count,2)
            np.testing.assert_allclose(h.last_x,features(current,[0,1,2],h.anchor))
            h.anchor=np.array([.3,.3,.2,.2])
            model.draft(np.zeros((1,13)),64,.7,10)
            self.assertEqual(f.call_count,3)

    def test_refresh_anchor_captured_before_engine_reset(self):
        model=Model(); h=runner.Harness(model)
        h.reset('csgb',LinUCB())
        model.attn_scores_final=Attention([.2,.8])
        model.update_working_cache_retrieval(do_retrieval=True)
        self.assertIsNone(model.attn_scores_final)
        np.testing.assert_array_equal(h.anchor,[.2,.8])

    def test_reward_action_alignment_and_clean_timing(self):
        model=Model(); h=runner.Harness(model)
        h.reset('history',LinUCB(width=3),explore=True,rho=10.)
        with patch.object(runner.torch.cuda,'synchronize'):
            model.draft(np.zeros((1,5)),64,.7,10)
            model.draft(np.zeros((1,8)),64,.7,10)
            model.draft(np.zeros((1,10)),64,.7,10)
        self.assertEqual([r['committed'] for r in h.rows],[3,2])
        self.assertEqual(h.rows[0]['budget'],64)
        self.assertIsNone(h.rows[0]['x'])
        self.assertEqual(sum(h.learner.counts),1)
        for r in h.rows:
            self.assertAlmostEqual(r['reward'],r['committed']-10*r['seconds'])
        h.reset('fixed32',clean=True)
        with patch.object(runner.torch.cuda,'synchronize') as sync:
            model.draft(np.zeros((1,5)),64,.7,10)
            model.draft(np.zeros((1,8)),64,.7,10)
            sync.assert_not_called()
        self.assertIsNone(h.rows[0]['seconds'])
        self.assertIsNone(h.rows[0]['reward'])

    def test_balanced_exploration_and_freeze(self):
        m=LinUCB(min_exploration=32,epsilon=.1,alpha=2)
        x=np.array([1,.8,.04,.032])
        for _ in range(96):
            b=m.choose(x,True); m.update(x,b,1,1,1)
        self.assertEqual(m.counts.tolist(),[32,32,32])
        before=m.a.copy()
        m.choose(x,False)
        np.testing.assert_array_equal(before,m.a)


if __name__=='__main__': unittest.main()
