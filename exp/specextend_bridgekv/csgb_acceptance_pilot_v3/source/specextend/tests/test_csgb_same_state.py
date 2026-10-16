import time
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch
from csgb_same_state import fingerprint,InterventionHarness,AuditStop,Schedule,release_request_caches


class StateTests(unittest.TestCase):
    def test_release_only_request_caches(self):
        weight=torch.ones(2)
        model=types.SimpleNamespace(weight=weight,past_key_values=torch.ones(2),
            draft_model=types.SimpleNamespace(model=types.SimpleNamespace(past_key_position_ids=torch.ones(2))))
        with patch('csgb_same_state.torch.cuda.empty_cache') as clear:
            release_request_caches(model)
            clear.assert_called_once()
        self.assertIsNone(model.past_key_values)
        self.assertIsNone(model.draft_model.model.past_key_position_ids)
        self.assertIs(model.weight,weight)

    def test_active_cache_fingerprint(self):
        data=torch.zeros(1,1,4,2)
        kv=types.SimpleNamespace(data=data,current_length=torch.tensor(2))
        model=types.SimpleNamespace(total_seq_len=2,draft_stable_kv=[(torch.zeros(1,1,2,2),)*2],
            full_draft_kv=[(data.clone(),data.clone())],past_key_values=[(kv,kv)],
            draft_model=types.SimpleNamespace(model=types.SimpleNamespace(past_key_position_ids=torch.tensor([[0,1]]))),
            attn_scores_final=None)
        h=types.SimpleNamespace(anchor=np.array([.2,.8]),latest_attention=None)
        ids=torch.tensor([[1,2,3]])
        base=fingerprint(model,ids,h)
        data[:,:,3,:]=9
        self.assertEqual(base,fingerprint(model,ids,h))
        data[:,:,1,:]=1
        self.assertNotEqual(base,fingerprint(model,ids,h))
        data[:,:,1,:]=0
        self.assertEqual(base,fingerprint(model,ids,h))
        h.anchor=np.array([.8,.2])
        self.assertNotEqual(base,fingerprint(model,ids,h))
        model.selected_chunks=[(1,23)]
        first=fingerprint(model,ids,h)
        model.selected_chunks=[(12,3)]
        self.assertNotEqual(first,fingerprint(model,ids,h))

    def test_stop_before_next_draft_and_accounting(self):
        h=InterventionHarness.__new__(InterventionHarness)
        h.round_index=6;h.intervention_round=5;h.context=10;h.before_hash='hash'
        h.pending={'start':time.perf_counter()-.01,'prefix':12,'budget':32}
        ids=torch.arange(15).reshape(1,15)
        with patch('csgb_same_state.torch.cuda.synchronize'):
            with self.assertRaises(AuditStop):h.draft(None,ids,8,.7,10)
        self.assertEqual(h.observation['committed'],3)
        self.assertEqual(h.observation['committed_output_ids'],[10,11,12,13])
        self.assertGreater(h.observation['seconds'],0)

    def test_only_one_intervention(self):
        h=types.SimpleNamespace(round_index=0,intervention_round=5,intervention_budget=32)
        s=Schedule(h)
        for i in range(6):
            h.round_index=i
            self.assertEqual(s.choose(None),32 if i==5 else 8)


if __name__=='__main__':unittest.main()
