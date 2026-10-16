import types
import unittest
from classic.attention_observation import wants_attention


class ObservationTests(unittest.TestCase):
    def test_default_observes_only_refresh_rounds(self):
        m=types.SimpleNamespace(use_retrieval_cache=True,retrieval_policy='attention')
        self.assertFalse(wants_attention(m,False))
        self.assertTrue(wants_attention(m,True))

    def test_fresh_observation_does_not_mutate_refresh_schedule(self):
        m=types.SimpleNamespace(use_retrieval_cache=True,retrieval_policy='attention',
                                csgb_attention_every_round=True,retrieval_condition=False)
        self.assertTrue(wants_attention(m,m.retrieval_condition))
        self.assertFalse(m.retrieval_condition)

    def test_other_policies_and_disabled_cache_are_unchanged(self):
        for enabled,policy in ((False,'attention'),(True,'random')):
            m=types.SimpleNamespace(use_retrieval_cache=enabled,retrieval_policy=policy,
                                    csgb_attention_every_round=True)
            self.assertFalse(wants_attention(m,True))
