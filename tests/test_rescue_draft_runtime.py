"""Toy state/trajectory checks, not a throughput or pretrained-model experiment."""
import copy
from contextlib import ExitStack
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from rescue.attention import DraftAttention
from rescue.config import SystemConfig
from rescue.engine import RescueEngine
from rescue.query_mapper import QueryValueMapper
from rescue.step_budget import StepBudget
from test_rescue_system import COST, models


def snapshot(native, working):
    return dict(native_lengths=tuple(native.lengths),
        native=[(k.detach().clone(), v.detach().clone()) for k, v in (native[i] for i in range(len(native)))],
        working_lengths=tuple(working.lengths),
        working=[(working.keys[i][:, :, :n].detach().clone(), working.values[i][:, :, :n].detach().clone())
                 for i, n in enumerate(working.lengths)],
        positions=None if working.positions is None else working.positions.detach().clone())


def assert_tree_close(test, actual, expected):
    if torch.is_tensor(expected):
        test.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    elif isinstance(expected, dict):
        test.assertEqual(set(actual), set(expected))
        for key in expected:
            assert_tree_close(test, actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        test.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            assert_tree_close(test, a, b)
    else:
        test.assertEqual(actual, expected)


def components(device='cpu', constant_tokens=False):
    teacher, student, _ = models()
    torch.manual_seed(715)
    mapper = QueryValueMapper(layers=[1], teacher_layers=[2], source_dim=16,
        value_dim=8, query_dim=24, rank=4, residual_rank=3).eval()
    with torch.no_grad():
        mapper.residuals[0][-1].weight.normal_(0., .1)
        if constant_tokens:
            teacher.lm_head.weight.zero_()
            student.lm_head.weight.zero_()
    return tuple(m.to(device).eval().requires_grad_(False) for m in (teacher, student, mapper))


def policy_for(outcome, device='cpu'):
    policy = StepBudget(prior_weight=0., margin=.05).eval()
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        policy.net[-1].bias[outcome] = 20.
    return policy.to(device).requires_grad_(False)


def config(mode='eager', limit=17):
    return SystemConfig(budget=8, recent=2, selector='hierarchical', coverage_rows=4,
        coverage_candidates=12, coarse_chunk_size=4, coverage_batch=2,
        injection_layers=(1,), inject_tokens=2, length_policy='risk', max_new_tokens=limit,
        attention_backend='sdpa', working_mode='packed', optimized_execution=True,
        deferred_probes=True, schedule='serial', draft_execution=mode)


class RuntimePairMixin:
    device = 'cpu'
    candidate_mode = 'resident'
    draft_dtype = torch.float32

    @torch.no_grad()
    def pair(self, limit=17, outcome=4, eos=None, constant_tokens=False, prompts=None, poison=False):
        teacher, student, mapper = components(self.device, constant_tokens)
        student.to(dtype=self.draft_dtype)
        frozen = {name: tensor.clone() for name, tensor in mapper.state_dict().items()}
        prompts = prompts or [list(range(3, 22)), list(range(5, 19))]
        records = []
        step_original = DraftAttention.step
        bind_original = __import__('rescue.cache', fromlist=['WorkingCache']).WorkingCache.bind
        for mode in ('eager', self.candidate_mode):
            policy = policy_for(outcome, self.device)
            mode_runs, steps, rounds, raw_queries, raw_summaries = [], [], [], [], []

            def observed_step(attention, token, native, logits=True):
                scores = step_original(attention, token, native, logits=logits)
                query = attention.last_query
                summaries = attention.injector.summaries if attention.injector is not None else []
                raw_queries.append((query, None if query is None else query.detach().clone()))
                raw_summaries.extend((v, v.detach().clone()) for v in summaries)
                steps.append(dict(logits=logits, state=snapshot(native, attention.working),
                    query=None if query is None else query.detach().clone(),
                    summaries=[v.detach().clone() for v in summaries],
                    argmax=None if scores is None else int(scores[0].argmax())))
                return scores

            def observe(packet):
                rounds.append(dict(selected=packet['selected'].detach().clone(),
                    features=None if packet['budget_features'] is None else packet['budget_features'].detach().clone(),
                    draft=list(packet['draft']), checks=list(packet['checks']), accepted=packet['accepted'],
                    prefix=packet['prefix'], state=snapshot(engine.native_s, engine.working)))

            def poisoned_bind(working, native, positions):
                bind_original(working, native, positions)
                if poison:
                    # Invalid tail storage must never enter a one-query read.
                    working.key_storage[:, :, :, len(positions):].fill_(float('nan'))
                    working.value_storage[:, :, :, len(positions):].fill_(float('nan'))
                    for i, n in enumerate(native.lengths):
                        native.keys[i][:, :, n:].fill_(float('nan'))
                        native.values[i][:, :, n:].fill_(float('nan'))

            with ExitStack() as stack:
                stack.enter_context(patch.object(DraftAttention, 'step', new=observed_step))
                if poison:
                    stack.enter_context(patch('rescue.cache.WorkingCache.bind', new=poisoned_bind))
                engine = stack.enter_context(RescueEngine(teacher, student, config(mode, limit), mapper,
                                                        policy, COST, round_observer=observe))
                for prompt in prompts:
                    step_start, round_start = len(steps), len(rounds)
                    reference = engine.reference(prompt, eos)['generated']
                    result = engine.generate(prompt, eos)
                    self.assertEqual(result['generated'], reference)
                    self.assertLessEqual(result['tokens'], limit)
                    if eos is not None and eos in reference:
                        self.assertEqual(reference[-1], eos)
                        self.assertNotIn(eos, reference[:-1])
                    mode_runs.append(dict(result=result, steps=steps[step_start:], rounds=rounds[round_start:]))
                if mode != 'eager':
                    stats = engine.draft_runtime.stats()
                    if mode == 'cuda_graph':
                        self.assertGreater(stats['captures'], 0)
                        self.assertGreater(stats['replays'], 0)
                    else:
                        self.assertGreater(stats['resident_steps'], 0)
                        self.assertEqual(stats['captures'], 0)
                        self.assertEqual(stats['replays'], 0)
            # Retaining a reference to one token's Q/summary must be safe even
            # after subsequent graph replays and another request reset.
            for value, saved in raw_queries+raw_summaries:
                if saved is not None:
                    torch.testing.assert_close(value, saved, rtol=0., atol=0.)
            records.append(mode_runs)
        for before, after in zip(*records):
            b, a = before['result'], after['result']
            for key in ('generated', 'tokens', 'accepted', 'proposed', 'blocks', 'draft_forwards',
                        'selector_refreshes', 'projected_positions'):
                self.assertEqual(a[key], b[key], key)
            fields = ('prefix', 'cap', 'proposed', 'accepted', 'terminal', 'workset_tokens')
            self.assertEqual([tuple(t[k] for k in fields) for t in a['trace']],
                             [tuple(t[k] for k in fields) for t in b['trace']])
            assert_tree_close(self, after['steps'], before['steps'])
            assert_tree_close(self, after['rounds'], before['rounds'])
        for name, value in mapper.state_dict().items():
            self.assertTrue(torch.equal(value, frozen[name]), name)
        return records

    @torch.no_grad()
    def rollback_pair(self, accepted):
        """Direct cache transition, independent of random model acceptance.

        All forwards are genuine toy-model operations. The accepted count only
        selects the explicit crop/rebind transition being tested.
        """
        from rescue.draft_graph import ResidentDraftRuntime
        teacher, student, mapper = components(self.device)
        student.to(dtype=self.draft_dtype)
        outputs = []
        for mode in ('eager', self.candidate_mode):
            stages, queries = [], []
            with RescueEngine(teacher, student, config('eager'), mapper, policy_for(4, self.device), COST) as engine:
                engine._buffers(96)
                attention = DraftAttention(student, engine.working, engine.injector,
                    backend='sdpa', optimized=True, deferred=True).install()
                runtime = None
                try:
                    engine.observation.mode = 'last'
                    engine._prefill(teacher, list(range(3, 21)), engine.native_t)
                    engine._prefill(student, list(range(3, 21)), engine.native_s)
                    attention.prefill = False
                    prior = engine.observation.rows.mean(0)
                    initial = engine.native_s.get_seq_length()
                    selected = torch.tensor([0, 2, 4, 6, 8, 10, 16, 17], device=self.device)
                    engine.working.bind(engine.native_s, selected)
                    engine.injector.bind(engine.native_t, engine.native_s, selected, prior, 2,
                                         working=engine.working)
                    if mode != 'eager':
                        runtime = ResidentDraftRuntime(student, engine.native_s, engine.working,
                                                       engine.injector, mode=mode)

                    def step(token, logits=True, summary=True):
                        if runtime is None:
                            attention.record_injection_summary = summary
                            scores = attention.step(token, engine.native_s, logits=logits)
                            query, moments = attention.last_query, engine.injector.summaries
                        else:
                            scores, query, moments = runtime.step(token, logits=logits, summary=summary)
                        queries.append((query, query.clone()))
                        stages.append(dict(state=snapshot(engine.native_s, engine.working),
                            query=query.clone(), summaries=[m.clone() for m in moments],
                            scores=None if scores is None else scores.clone()))
                        return token if scores is None else int(scores.argmax())

                    token = 21
                    for _ in range(4):
                        token = step(token)
                    if accepted == 4:
                        token = step(token, logits=False, summary=False)
                    engine.native_s.crop(initial+accepted+1)
                    engine.working.invalidate(); engine.injector.clear()
                    # Same physical capacity but different absolute token set.
                    selected = torch.tensor([0, 1, 3, 5, 7, 9, 15, 17], device=self.device)
                    engine.working.bind(engine.native_s, selected)
                    engine.injector.bind(engine.native_t, engine.native_s, selected, prior, 2,
                                         working=engine.working)
                    step(7)
                    self.assertEqual(engine.native_s.lengths,
                                     [initial+accepted+2]*len(engine.native_s))
                    if runtime is not None and mode == 'cuda_graph':
                        self.assertGreater(runtime.stats()['replays'], 0)
                finally:
                    if runtime is not None:
                        runtime.close()
                    attention.restore()
            for value, saved in queries:
                torch.testing.assert_close(value, saved, rtol=0., atol=0.)
            outputs.append(stages)
        assert_tree_close(self, outputs[1], outputs[0])


class DraftRuntimeConfigTests(unittest.TestCase):
    def test_default_and_modes_reject_unsupported_static_combinations(self):
        self.assertEqual(SystemConfig().draft_execution, 'eager')
        for mode in ('eager', 'resident', 'cuda_graph'):
            self.assertEqual(config(mode).draft_execution, mode)
        for mode in ('graph', 'resident_gpu', '', None, True):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                config(mode)
        changes = [dict(optimized_execution=False), dict(deferred_probes=False),
                   dict(working_mode='layerwise'), dict(working_mode='incremental'),
                   dict(attention_backend='eager'), dict(schedule='async'),
                   dict(length_endpoint=('127.0.0.1', 1234)), dict(early_length_publish=True)]
        for mode in ('resident', 'cuda_graph'):
            for changed in changes:
                with self.subTest(mode=mode, changed=changed), self.assertRaises(ValueError):
                    replace(config(mode), **changed)

    def test_runtime_observer_and_read_conditioning_reject_before_installing_hooks(self):
        for unsupported in ('draft_observer', 'attention_features'):
            with self.subTest(unsupported=unsupported):
                teacher, student, mapper = components()
                original = teacher.model.layers[-1].self_attn.forward
                kwargs = {}
                if unsupported == 'draft_observer':
                    kwargs['draft_observer'] = lambda packet: None
                else:
                    mapper.attention_features = 1
                with self.assertRaises(ValueError):
                    RescueEngine(teacher, student, config('resident'), mapper, policy_for(4), COST, **kwargs)
                self.assertEqual(teacher.model.layers[-1].self_attn.forward, original)

    def test_graph_on_cpu_rejects_before_installing_hooks(self):
        teacher, student, mapper = components()
        original = teacher.model.layers[-1].self_attn.forward
        with self.assertRaises(ValueError):
            RescueEngine(teacher, student, config('cuda_graph'), mapper, policy_for(4), COST)
        self.assertEqual(teacher.model.layers[-1].self_attn.forward, original)

    def test_failing_runtime_close_still_restores_target_hook_and_closes_engine(self):
        teacher, student, mapper = components()
        original = teacher.model.layers[-1].self_attn.forward
        engine = RescueEngine(teacher, student, config('resident'), mapper, policy_for(4), COST)
        engine._buffers(64)
        self.assertIsNotNone(engine.draft_runtime)
        self.assertNotEqual(teacher.model.layers[-1].self_attn.forward, original)
        with patch.object(engine.draft_runtime, 'close', side_effect=RuntimeError('synthetic close failure')) as close:
            with self.assertRaisesRegex(RuntimeError, 'synthetic close failure'):
                engine.close()
            self.assertTrue(engine.closed)
            self.assertEqual(teacher.model.layers[-1].self_attn.forward, original)
            self.assertFalse(engine.observation.originals)
            # Once closed, another close must not retry the failed runtime.
            engine.close()
            close.assert_called_once_with()


class ResidentDraftRuntimeTests(RuntimePairMixin, unittest.TestCase):
    def test_nonzero_mapper_cap_two_and_four_keep_all_layer_states_and_query_snapshots(self):
        for outcome in (0, 4):
            with self.subTest(outcome=outcome):
                records = self.pair(outcome=outcome)
                caps = [t['cap'] for t in records[1][0]['result']['trace']]
                self.assertIn(2 if outcome == 0 else 4, caps)
                # The full request includes at least one genuine re-selection.
                self.assertGreater(records[1][0]['result']['selector_refreshes'], 1)

    def test_remaining_one_through_six_preserves_tail_and_bonus_boundaries(self):
        for remaining in range(1, 7):
            for outcome in (0, 4):
                with self.subTest(remaining=remaining, outcome=outcome):
                    self.pair(limit=remaining, outcome=outcome, prompts=[list(range(3, 22))])

    def test_target_eos_and_draft_eos_do_not_generate_an_extra_token(self):
        baseline = self.pair(limit=6, prompts=[list(range(3, 22))])[0][0]
        eos_ids = {baseline['result']['generated'][0], baseline['result']['generated'][min(2, baseline['result']['tokens']-1)],
                   baseline['steps'][0]['argmax']}
        for eos in eos_ids:
            with self.subTest(eos=eos):
                self.pair(limit=6, eos=eos, prompts=[list(range(3, 22))])

    def test_all_accept_catchup_skips_logits_but_preserves_all_layer_kv(self):
        records = self.pair(limit=7, constant_tokens=True, prompts=[list(range(3, 22))])
        self.assertTrue(any(not step['logits'] for step in records[0][0]['steps']))
        self.assertTrue(any(not step['logits'] for step in records[1][0]['steps']))

    def test_nonresident_padding_is_ignored_after_workset_changes(self):
        self.pair(limit=17, poison=True)

    def test_explicit_accept_zero_through_four_rollback_rebinds_every_layer(self):
        for accepted in range(5):
            with self.subTest(accepted=accepted):
                self.rollback_pair(accepted)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable; GPU graph validation is not claimed')
class CudaGraphDraftRuntimeTests(RuntimePairMixin, unittest.TestCase):
    device = 'cuda'
    candidate_mode = 'cuda_graph'

    def test_real_capture_replay_cap_boundaries_workset_changes_and_request_reset(self):
        for outcome in (0, 4):
            with self.subTest(outcome=outcome):
                self.pair(limit=17, outcome=outcome)

    def test_graph_eos_and_short_output_boundaries(self):
        for remaining in range(1, 7):
            with self.subTest(remaining=remaining):
                self.pair(limit=remaining, prompts=[list(range(3, 22))])
        reference = self.pair(limit=6, prompts=[list(range(3, 22))])[0][0]['result']['generated']
        self.pair(limit=6, eos=reference[0], prompts=[list(range(3, 22))])

    def test_graph_all_accept_catchup_and_padding_poison(self):
        self.pair(limit=7, constant_tokens=True, prompts=[list(range(3, 22))], poison=True)
        self.pair(limit=17, poison=True)

    def test_graph_explicit_rollback_zero_through_four_preserves_every_layer(self):
        for accepted in range(5):
            with self.subTest(accepted=accepted):
                self.rollback_pair(accepted)

    def test_graph_fp32_target_bfloat16_draft_with_nonzero_float_mapper(self):
        self.draft_dtype = torch.bfloat16
        self.pair(limit=17, prompts=[list(range(3, 22))])


if __name__ == '__main__':
    unittest.main()
