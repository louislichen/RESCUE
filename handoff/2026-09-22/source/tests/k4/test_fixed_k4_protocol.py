from __future__ import annotations

import unittest

from fixed_k4_protocol import fixed_k4_decode, greedy_reference


class FakeLM:
    def __init__(self, sequence, overrides=None):
        self.sequence = list(sequence)
        self.overrides = dict(overrides or {})

    def _next(self, context):
        key = tuple(context)
        if key in self.overrides:
            return self.overrides[key]
        i = len(context)
        return self.sequence[i] if i < len(self.sequence) else 99

    def prefill(self, prompt_ids):
        context = list(prompt_ids)
        return self._next(context), context

    def step(self, token, cache):
        context = list(cache) + [int(token)]
        return self._next(context), context

    def crop_cache(self, cache, length):
        return list(cache)[:length]

    def clone_cache(self, cache):
        return list(cache)

    def verify(self, proposal_ids, cache):
        context = list(cache)
        predictions = []
        for token in proposal_ids:
            predictions.append(self._next(context))
            context.append(int(token))
        predictions.append(self._next(context))
        return predictions, context


class FixedK4Tests(unittest.TestCase):
    def run_case(self, draft_seq, target_seq, prompt=(0,), max_new=8, eos=98, overrides=None):
        draft = FakeLM(draft_seq)
        target = FakeLM(target_seq, overrides=overrides)
        result = fixed_k4_decode(prompt, draft, target, eos, k=4, max_new_tokens=max_new)
        reference = greedy_reference(prompt, FakeLM(target_seq, overrides=overrides), eos, max_new)
        self.assertEqual(result.token_ids, reference)
        self.assertEqual(result.k, 4)
        return result

    def test_all_accept_and_bonus(self):
        result = self.run_case([99, 1, 2, 3, 4, 5, 6, 98], [99, 1, 2, 3, 4, 5, 6, 98])
        self.assertGreaterEqual(result.counters.full_accept_blocks, 1)
        self.assertGreater(result.counters.bonus_tokens, 0)

    def test_first_token_reject(self):
        result = self.run_case([99, 7, 2, 3, 4, 5, 6], [99, 1, 2, 3, 4, 5, 6])
        self.assertGreater(result.counters.reject_blocks, 0)
        self.assertGreater(result.counters.correction_tokens, 0)
        self.assertTrue(result.counters.rollback_events)

    def test_middle_reject(self):
        result = self.run_case([99, 1, 9, 4, 5, 6], [99, 1, 2, 3, 4, 5, 6])
        self.assertGreater(result.counters.accepted_draft_tokens, 0)
        self.assertGreater(result.counters.reject_blocks, 0)
        self.assertTrue(result.counters.rollback_events[0]["accepted_before_reject"] >= 1)

    def test_eos(self):
        result = self.run_case([99, 1, 98, 3], [99, 1, 98, 3], max_new=8)
        self.assertEqual(result.token_ids, [1])
        self.assertTrue(result.counters.eos_terminated)

    def test_short_proposal_at_generation_boundary(self):
        result = self.run_case([99, 1, 2, 3], [99, 1, 2, 3], max_new=3)
        self.assertEqual(result.token_ids, [1, 2, 3])
        self.assertEqual(result.counters.proposed_draft_tokens, 3)

    def test_length_boundary_one(self):
        result = self.run_case([99, 1], [99, 1], max_new=1)
        self.assertEqual(result.token_ids, [1])

    def test_empty_proposal_after_verified_block_counts_as_block(self):
        result = self.run_case(
            [99, 1, 2, 3, 4, 5, 98],
            [99, 1, 2, 3, 4, 5, 6, 7],
            max_new=6,
        )
        self.assertEqual(result.token_ids, [1, 2, 3, 4, 5, 6])
        self.assertEqual(result.counters.target_verify_calls, 1)
        self.assertEqual(result.counters.blocks, 2)

    def test_multiround_rollback(self):
        result = self.run_case([99, 1, 2, 8, 4, 5, 6, 7], [99, 1, 2, 3, 4, 5, 6, 7], max_new=7)
        self.assertEqual(result.token_ids, [1, 2, 3, 4, 5, 6, 7])
        self.assertGreaterEqual(len(result.counters.rollback_events), 1)

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            fixed_k4_decode((0,), FakeLM([1]), FakeLM([1]), 98, k=3)


if __name__ == "__main__":
    unittest.main()
