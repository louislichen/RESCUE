import time
import types
import unittest
from unittest.mock import patch
import numpy as np
from run_csgb_controller import Harness


class TerminalAccountingTests(unittest.TestCase):
    def harness(self,clean=False,explore=False):
        h=Harness.__new__(Harness)
        h.clean=clean;h.explore=explore;h.rho=2.;h.last_accept=0;h.ema=0.;h.rows=[]
        h.pending={'prefix':11,'start':time.perf_counter()-.1,'x':[1.,.8,.1,.08], 'budget':8}
        h.learner=types.SimpleNamespace(update=lambda *args: None)
        return h

    def test_terminal_closes_last_round_without_next_decision(self):
        h=self.harness();ids=types.SimpleNamespace(shape=(1,13))
        with patch('run_csgb_controller.torch.cuda.synchronize'):
            h.finalize_terminal(ids)
            self.assertEqual(h.rows[0]['committed'],3)
            self.assertEqual(h.last_accept,2)
            self.assertTrue(h.rows[0]['terminal_commit'])
            self.assertIsNone(h.pending)
            with self.assertRaises(RuntimeError): h.finalize_terminal(ids)

    def test_clean_mode_does_not_fabricate_round_latency(self):
        h=self.harness(clean=True);ids=types.SimpleNamespace(shape=(1,11))
        h.finalize_terminal(ids)
        self.assertEqual(h.rows[0]['committed'],1)
        self.assertIsNone(h.rows[0]['seconds']);self.assertIsNone(h.rows[0]['reward'])

    def test_training_terminal_updates_chosen_arm_once(self):
        h=self.harness(explore=True);calls=[];h.learner.update=lambda *a:calls.append(a)
        with patch('run_csgb_controller.torch.cuda.synchronize'):
            h.finalize_terminal(types.SimpleNamespace(shape=(1,13)))
        self.assertEqual(len(calls),1);self.assertEqual(calls[0][1],8);self.assertEqual(calls[0][2],3)
        np.testing.assert_array_equal(calls[0][0],[1.,.8,.1,.08])
