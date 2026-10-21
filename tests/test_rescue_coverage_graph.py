"""Literal selection-equivalence checks; toy data are not throughput results."""
import unittest

import torch

from rescue.coverage_graph import CoverageGraphRuntime, greedy_coverage, _greedy


def inputs(pool_size=80, device='cpu', ties=False):
    generator = torch.Generator().manual_seed(813+pool_size)
    sims = [torch.rand(rows, pool_size, generator=generator).to(device) for rows in (7, 5)]
    covers = [torch.rand(rows, generator=generator).to(device) for rows in (7, 5)]
    weights = [torch.rand(rows, generator=generator).to(device) for rows in (7, 5)]
    weights = [w/w.sum() for w in weights]
    if ties:
        sims = [torch.ones_like(t) for t in sims]
        covers = [torch.zeros_like(t) for t in covers]
    pool = torch.arange(100, 100+pool_size, device=device)
    joint_pool = torch.rand(pool_size, generator=generator).to(device)
    denominator = joint_pool.sum().clamp_min(1e-12)
    return sims, covers, weights, pool, joint_pool, denominator


def literal(similarities, initial, objectives, pool, slots, batch,
            student_weight=1., attention_weight=0., joint_pool=None, denominator=None):
    """Original loop, including the original weighted-division association."""
    coverage = list(initial)
    chosen, used = [], torch.zeros(len(pool), device=pool.device, dtype=torch.bool)
    for offset in range(0, slots, batch):
        gains = sum(scale*((sim-cover[:, None]).clamp_min(0)*weight[:, None]).sum(0)
                    for scale, sim, cover, weight in
                    zip((1., student_weight), similarities, coverage, objectives))
        if attention_weight:
            gains = gains+attention_weight*joint_pool/denominator
        gains.masked_fill_(used, -torch.inf)
        take = gains.topk(min(batch, slots-offset)).indices
        used[take] = True
        for index, sim in enumerate(similarities):
            coverage[index] = torch.maximum(coverage[index], sim[:, take].amax(-1))
        chosen.append(pool[take])
    return torch.cat(chosen)


def snapshot(values):
    return [v.detach().clone() for v in values]


class EagerCoverageTests(unittest.TestCase):
    def test_reference_preserves_literal_28_round_arithmetic_and_attention_term(self):
        sims, covers, weights, pool, joint, denominator = inputs(512)
        for weight in (0., .17):
            bonus = weight*joint/denominator if weight else None
            expected = literal(sims, covers, weights, pool, 447, 16, .7,
                               weight, joint, denominator)
            actual = greedy_coverage(sims, covers, weights, pool, 447, 16, .7, bonus)
            self.assertTrue(torch.equal(actual, expected))
            graph_safe = _greedy(sims, covers, weights, pool, 447, 16, .7, bonus, graph_safe=True)
            self.assertTrue(torch.equal(graph_safe, expected))
            self.assertEqual(actual.numel(), 447)
            self.assertEqual(actual.unique().numel(), 447)

    def test_ties_and_incomplete_final_batch_keep_exact_topk_picks(self):
        sims, covers, weights, pool, joint, denominator = inputs(ties=True)
        for slots, batch in ((1, 16), (23, 8), (80, 16)):
            expected = literal(sims, covers, weights, pool, slots, batch)
            actual = greedy_coverage(sims, covers, weights, pool, slots, batch)
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(_greedy(sims, covers, weights, pool, slots,
                batch, 1., None, graph_safe=True), expected))

    def test_reused_owner_copies_all_state_preserves_inputs_and_old_output(self):
        sims, covers, weights, pool, joint, denominator = inputs()
        bonus = .2*joint/denominator
        tensors = [*sims, *covers, *weights, pool, bonus]
        before = snapshot(tensors)
        runtime = CoverageGraphRuntime(mode='eager')
        first = runtime.select(sims, covers, weights, pool, 23, 8, .3, bonus)
        retained = first.clone()
        self.assertTrue(all(torch.equal(a, b) for a,b in zip(tensors, before)))
        self.assertNotEqual(first.data_ptr(), next(iter(runtime._entries.values())).output.data_ptr())
        sims[0].zero_(); covers[1].fill_(.99); weights[0].mul_(.13)
        pool.add_(600); bonus.copy_(torch.arange(len(pool), dtype=bonus.dtype))
        second = runtime.select(sims, covers, weights, pool, 23, 8, .3, bonus)
        expected = greedy_coverage(sims, covers, weights, pool, 23, 8, .3, bonus)
        self.assertTrue(torch.equal(second, expected))
        self.assertTrue(torch.equal(first, retained))
        self.assertEqual(runtime.stats()['entries'], 1)
        self.assertEqual(runtime.stats()['eager_calls'], 2)
        self.assertEqual(runtime.stats()['input_copies'], 2)
        runtime.close()
        self.assertEqual(runtime.stats()['entries'], 0)
        with self.assertRaisesRegex(ValueError, 'closed'):
            runtime.select(sims, covers, weights, pool, 23, 8, .3, bonus)

    def test_static_choices_and_actual_shapes_are_keys_and_limits_fail_explicitly(self):
        sims, covers, weights, pool, joint, denominator = inputs()
        runtime = CoverageGraphRuntime(mode='eager', max_entries=3)
        runtime.select(sims, covers, weights, pool, 23, 8)
        runtime.select(sims, covers, weights, pool, 23, 8, .25)
        runtime.select(sims, covers, weights, pool, 23, 8, .25, torch.zeros_like(joint))
        self.assertEqual(runtime.stats()['entries'], 3)
        with self.assertRaisesRegex(ValueError, 'entry limit'):
            runtime.select(sims, covers, weights, pool, 24, 8)
        self.assertEqual(runtime.stats()['eager_calls'], 3)
        for limit in (0, -1, 65, True, 2.):
            with self.assertRaises(ValueError):
                CoverageGraphRuntime(mode='eager', max_entries=limit)

    def test_cpu_graph_and_unsupported_shapes_fail_without_fallback(self):
        sims, covers, weights, pool, _, _ = inputs()
        runtime = CoverageGraphRuntime()
        with self.assertRaisesRegex(ValueError, 'CUDA inputs'):
            runtime.select(sims, covers, weights, pool, 23, 8)
        self.assertEqual(runtime.stats()['entries'], 0)
        self.assertEqual(runtime.stats()['eager_calls'], 0)
        bad_cases = [dict(slots=0), dict(slots=len(pool)+1), dict(batch=0),
                     dict(student_weight=float('inf')), dict(attention_bonus=torch.ones(1)),
                     dict(pool=pool.float()), dict(similarities=[sims[0][:, ::2], sims[1]])]
        original = dict(similarities=sims, coverage=covers, objective=weights, pool=pool,
                        slots=23, batch=8)
        for change in bad_cases:
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                greedy_coverage(**dict(original, **change))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable; no graph equivalence/speed claim from CPU')
class CudaCoverageTests(unittest.TestCase):
    @torch.no_grad()
    def test_33_actual_pool_shapes_dynamic_inputs_and_nonzero_bonus_match_exactly(self):
        runtime = CoverageGraphRuntime()
        try:
            for size in range(64, 97):
                sims, covers, weights, pool, joint, denominator = inputs(size, 'cuda')
                for step in range(2):
                    if step:
                        sims[0].mul_(.37); covers[1].mul_(.13); weights[1].mul_(.73)
                        joint.copy_(joint.flip(0)+.001)
                        pool.add_(800)
                    bonus = .13*joint/denominator
                    values = [*sims, *covers, *weights, pool, bonus]
                    before = snapshot(values)
                    expected = literal(sims, covers, weights, pool, 23, 8, .7, .13, joint, denominator)
                    result = runtime.select(sims, covers, weights, pool, 23, 8, .7, bonus)
                    self.assertTrue(torch.equal(result, expected), (size, step))
                    self.assertTrue(all(torch.equal(a, b) for a,b in zip(values, before)))
            stats = runtime.stats()
            self.assertEqual(stats['entries'], 33)
            self.assertEqual(stats['captures'], 33)
            self.assertEqual(stats['replays'], 66)
            self.assertGreater(stats['capture_host_seconds'], 0.)
        finally:
            runtime.close()

    @torch.no_grad()
    def test_ties_private_owners_and_retained_selections_are_not_overwritten(self):
        left, right = CoverageGraphRuntime(), CoverageGraphRuntime()
        sims, covers, weights, pool, joint, denominator = inputs(80, 'cuda', ties=True)
        try:
            expected = literal(sims, covers, weights, pool, 47, 16)
            result = left.select(sims, covers, weights, pool, 47, 16)
            saved = result.clone()
            self.assertTrue(torch.equal(result, expected))
            other_pool = pool+900
            other = right.select(sims, covers, weights, other_pool, 47, 16)
            self.assertTrue(torch.equal(other, expected+900))
            self.assertNotEqual(next(iter(left._entries.values())).pool.data_ptr(),
                                next(iter(right._entries.values())).pool.data_ptr())
            joint.fill_(1.); bonus = .17*joint/denominator
            wanted = literal(sims, covers, weights, pool, 47, 16, 1., .17, joint, denominator)
            actual = left.select(sims, covers, weights, pool, 47, 16, attention_bonus=bonus)
            self.assertTrue(torch.equal(actual, wanted))
            self.assertTrue(torch.equal(result, saved))
            sims[0].zero_(); covers[0].fill_(5.)
            left.select(sims, covers, weights, pool+2000, 47, 16)
            self.assertTrue(torch.equal(result, saved))
        finally:
            left.close(); right.close()


if __name__ == '__main__':
    unittest.main()
