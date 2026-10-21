"""Runner safety/ownership regressions; toy outputs are not performance data."""
import ast
import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import torch

from rescue.attention import DraftAttention
from rescue.draft_graph_benchmark import (EXPECTED, HookOwners, check_output,
    owner_signature, record, run, schedule)
from rescue.draft_graph_smoke import native_digest, record_queries
from rescue.efficient_attention import install_efficient_attention
from rescue.engine import RescueEngine
from rescue.packing import pack_projections
from rescue.round_overhead_run import CorrectnessObserver, check_pair
from test_rescue_draft_runtime import components, config, policy_for
from test_rescue_system import COST


class DraftBenchmarkHelpers(unittest.TestCase):
    def test_registered_108_warm_schedule_is_seeded_adjacent_crossover(self):
        plan = schedule()
        self.assertEqual(plan, schedule())
        self.assertNotEqual(plan, schedule(7))
        self.assertEqual(len(plan), EXPECTED['timing_requests'])
        identities = {(r['mode'], r['source_index'], r['length'], r['repeat']) for r in plan}
        self.assertEqual(len(identities), 108)
        for offset in range(0, len(plan), 3):
            group = plan[offset:offset+3]
            self.assertEqual({r['mode'] for r in group}, {'baseline_eager', 'resident', 'cuda_graph'})
            self.assertEqual(len({(r['source_index'], r['length'], r['repeat']) for r in group}), 1)

    def test_recorder_flushes_raw_before_duplicate_uuid_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp); counts = dict(timing_requests=0); seen = set()
            record(output, 'raw.jsonl', dict(request_id='one'), counts, 'timing_requests', seen)
            with self.assertRaisesRegex(RuntimeError, 'UUID'):
                record(output, 'raw.jsonl', dict(request_id='one'), counts, 'timing_requests', seen)
            self.assertEqual(len((output/'raw.jsonl').read_text().splitlines()), 2)
            self.assertEqual(counts['timing_requests'], 2)

    def test_max128_checker_honors_eos_and_rejects_empty_proposals(self):
        row = dict(generated=[7, 9], output_agrees=True, tokens=2, blocks=1, accepted=1,
            proposed=2, trace=[dict(round_index=0, terminal=True, proposed=2, accepted=1, cap=2)])
        check_output(row, [7, 9], eos=9)
        with self.assertRaisesRegex(RuntimeError, 'Short output'):
            check_output(row, [7, 9])
        row['proposed'] = 0; row['accepted'] = 0
        row['trace'][0].update(proposed=0, accepted=0)
        with self.assertRaisesRegex(RuntimeError, 'proposal trajectory'):
            check_output(row, [7, 9], eos=9)

    def test_runner_does_not_locally_shadow_called_global_helpers(self):
        tree = ast.parse(inspect.getsource(run))
        function = tree.body[0]
        assigned = {n.id for n in ast.walk(function) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        # The historical recorder/owner-record naming collision would raise
        # before any real request. Protect the audited recording helpers.
        self.assertFalse(assigned & {'record', 'sources', 'context', 'schedule', 'progress', 'write', 'append'})


class PersistentHandoffMixin:
    device = 'cpu'
    modes = ('eager', 'resident', 'resident')

    @torch.no_grad()
    def run_interleaved(self):
        teacher, student, mapper = components(self.device)
        if self.device == 'cuda':
            student.to(torch.bfloat16)
        install_efficient_attention(teacher); install_efficient_attention(student)
        pack_projections(teacher); pack_projections(student)
        owners = HookOwners(teacher, student)
        engines, observers = [], []
        original_mapper = {k: v.clone() for k,v in mapper.state_dict().items()}
        pointers = {}
        prompts = [list(range(3, 22)), list(range(4, 23))]
        try:
            for mode in self.modes:
                observer = CorrectnessObserver()
                engine = RescueEngine(teacher, student, config(mode), mapper,
                    policy_for(4, self.device), COST, round_observer=observer)
                owners.add(engine); engines.append(engine); observers.append(observer)
            owners.idle()
            for repeat, order in enumerate(((2, 0, 1), (1, 2, 0))):
                prompt = prompts[repeat]
                expected_observer = CorrectnessObserver()
                control = RescueEngine(teacher, student, config('eager'), mapper,
                    policy_for(4, self.device), COST, round_observer=expected_observer)
                owners.add(control)
                try:
                    with record_queries(DraftAttention) as queries:
                        expected = owners.generate(control, prompt, None)
                    expected_state = dict(queries=copy.deepcopy(queries), rounds=expected_observer.rounds,
                        native_t=native_digest(control.native_t), native_s=native_digest(control.native_s))
                finally:
                    owners.close(control)
                for index in order:
                    engine, observer = engines[index], observers[index]
                    offset = len(observer.rounds)
                    with record_queries(DraftAttention) as queries:
                        actual = owners.generate(engine, prompt, None)
                    check_pair(expected, actual)
                    self.assertEqual(queries, expected_state['queries'])
                    self.assertEqual(observer.rounds[offset:], expected_state['rounds'])
                    self.assertEqual(native_digest(engine.native_t), expected_state['native_t'])
                    self.assertEqual(native_digest(engine.native_s), expected_state['native_s'])
                    self.assertTrue(observer.nonzero_injection)
                    signature = owner_signature(engine)
                    if index in pointers:
                        self.assertEqual(signature, pointers[index])
                    pointers[index] = signature
                    if self.modes[index] == 'cuda_graph':
                        stats = actual['draft_runtime']['request']
                        self.assertGreater(stats['replays'], 0)
                        if repeat:
                            self.assertEqual(stats['captures'], 0)
                        else:
                            self.assertGreater(stats['captures'], 0)
                    owners.idle()
            self.assertEqual(len({v['native_s']['key_ptr'] for v in pointers.values()}), 3)
            for key, value in mapper.state_dict().items():
                self.assertTrue(torch.equal(value, original_mapper[key]), key)
        finally:
            for engine in list(owners.engines):
                owners.close(engine)
            owners.idle()


class PersistentHandoffCPU(PersistentHandoffMixin, unittest.TestCase):
    def test_three_persistent_owners_crossover_exact_and_private(self):
        self.run_interleaved()

    def test_nested_constructors_and_unmanaged_hook_replacements_fail_closed(self):
        teacher, student, mapper = components()
        install_efficient_attention(teacher); install_efficient_attention(student)
        owners = HookOwners(teacher, student)
        first = RescueEngine(teacher, student, config(), mapper, policy_for(4), COST)
        second = RescueEngine(teacher, student, config(), mapper, policy_for(4), COST)
        try:
            with self.assertRaisesRegex(RuntimeError, 'nested or replaced'):
                owners.add(second)
        finally:
            second.close(); first.close()
        owners.idle()
        original = owners.modules[0].forward
        owners.modules[0].forward = lambda *args, **kwargs: None
        try:
            with self.assertRaisesRegex(RuntimeError, 'original efficient callable'):
                owners.idle()
        finally:
            owners.modules[0].forward = original


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable; no GPU claim from CPU tests')
class PersistentHandoffCUDA(PersistentHandoffMixin, unittest.TestCase):
    device = 'cuda'
    modes = ('eager', 'resident', 'cuda_graph')

    def test_real_cuda_graph_owner_survives_shared_model_crossover(self):
        self.run_interleaved()


if __name__ == '__main__':
    unittest.main()
