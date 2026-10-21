"""CPU fake792-call/lifecycle contracts only; no CUDA/model/timing evidence."""
import ast
from contextlib import contextmanager, redirect_stdout
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from exp.round_overhead import fp32_splitkv_model_gate_v1 as model
from exp.round_overhead import fp32_splitkv_timing_v1 as gate
from tests.test_fp32_splitkv_model_gate_v1 import lifecycle, row_for, scope_summary


def state():
    sequence = [dict(source_index=document, repeat=repeat) for repeat in (0, 1) for document in gate.DOCS]
    references = {f'{document}:{n}': [42]*128 for n in gate.LENGTHS for document in gate.DOCS}
    historic = {(n, document, arm['mode'], repeat): row_for(arm, references[f'{document}:{n}'], dict(length=n))
        for n in gate.LENGTHS for document in gate.DOCS for arm in model.arms().values() for repeat in (0, 1)}
    return dict(sequence=sequence, references=references, inputs={(document, n): [1, 2] for n in gate.LENGTHS for document in gate.DOCS},
        historic=historic, seen=set(), counts=dict.fromkeys(gate.EXPECTED, 0), attempted_calls=0, returned_calls=0, validated_calls=0)


class FakeAdapter:
    def __init__(self, mismatch=False, repeat_drift=False):
        self.live = None
        self.mismatch, self.repeat_drift = mismatch, repeat_drift
        self.made, self.closed, self.compared, self.actual_cuda = [], [], [], True
        self.unchanged = False

    @contextmanager
    def owner(self, descriptor):
        assert self.live is None
        handle = NS(constructor_seconds=.1, close_receipt={}, scope_summary={}, calls=0, captures=0,
                    totals=dict.fromkeys(gate.RUNTIME_KEYS, 0))
        self.live = handle
        self.made.append(descriptor)
        try:
            yield handle
        finally:
            handle.close_receipt = dict(engine_closed=True, hook_restored=True, scope_and_close_seconds=.2)
            handle.scope_summary = scope_summary(descriptor['variant'])
            self.closed.append(handle)
            self.live = None

    def generate(self, handle, descriptor, ids, reference, item):
        assert handle is self.live
        handle.calls += 1
        row = row_for(descriptor['arm'], reference, item)
        if descriptor['variant'] == 'candidate':
            row['accepted'] = row['trace'][0]['accepted'] = 3
            if self.mismatch:
                row['generated'][0] = 43
            if self.repeat_drift and item.get('repeat') == 1:
                row['accepted'] = row['trace'][0]['accepted'] = 4
        for key in ('queries', 'observed_rounds', 'native_t', 'native_s', 'nonzero_injection'):
            row.pop(key)
        delta = dict(captures=int(handle.calls in (1, 3)), replays=1, shadow_updates=1,
                     resident_steps=0, capture_host_seconds=.25 if handle.calls in (1, 3) else 0.)
        handle.totals = {key: handle.totals[key]+delta[key] for key in gate.RUNTIME_KEYS}
        keys = [[index] for index in range(handle.totals['captures'])]
        row.update(draft_forwards=1, owner_signature=dict(owner=id(handle)),
            runtime_stats=dict(mode='cuda_graph', max_graph_entries=128, entries=len(keys), graph_keys=keys, **handle.totals),
            draft_runtime=dict(request=delta, lifetime=dict(handle.totals)))
        elapsed = (1. if descriptor['variant'] == 'original' else .8)+delta['capture_host_seconds']
        return row, elapsed

    def validate(self, row, reference, descriptor, cold):
        assert row['generated'] == reference
        assert not any(key in row for key in ('observed_rounds', 'queries', 'native_t', 'native_s'))

    def compare_work(self, left, right):
        assert left['generated'] == right['generated']
        if any(left[key] != right[key] for key in model.WORK):
            raise ValueError('work drift')
        self.compared.append(right)

    def work_difference(self, left, right):
        difference = model.work_difference(left, right)
        return {key: difference[key] for key in ('work_fields', 'trajectory_equal')}

    def assert_unchanged(self):
        assert self.live is None
        self.unchanged = True


class TimingCpuTests(unittest.TestCase):
    def test_default_plan_no_load_no_output_no_success(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'not_created'
            with patch.object(gate, 'run', side_effect=AssertionError('run')) as run, \
                 patch.object(gate, 'load_contract', side_effect=AssertionError('load')) as load, \
                 redirect_stdout(io.StringIO()) as captured:
                self.assertEqual(gate.main(['--output', str(output)]), 0)
            run.assert_not_called(); load.assert_not_called()
            self.assertFalse(output.exists())
        plan = json.loads(captured.getvalue())
        self.assertEqual(plan['planned_generation_calls'], 792)
        self.assertEqual(plan['planned_private_owners'], 24)
        self.assertFalse(plan['passed']); self.assertFalse(plan['actual_cuda'])
        self.assertEqual(plan['performance_gate'], 'NOT_RUN')

    def test_fresh_default_has_no_heavy_runtime_import(self):
        script = """import builtins,sys
original=builtins.__import__
def guard(name,*a,**kw):
 if name.split('.')[0] in ('torch','triton','rescue'): raise AssertionError(name)
 return original(name,*a,**kw)
builtins.__import__=guard
from exp.round_overhead import fp32_splitkv_timing_v1 as gate
assert gate.main([])==0
assert not any(n.split('.')[0] in ('torch','triton','rescue') for n in sys.modules)
"""
        result = subprocess.run([sys.executable, '-B', '-c', script], cwd=gate.ROOT, text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'], 'PLAN_ONLY')

    def test_full792_shared_sequence_24owners_capture_costs_and_candidate_differences(self):
        prepared, adapter = state(), FakeAdapter()
        refs = copy.deepcopy(prepared['references'])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = gate.execute_matrix(adapter, prepared, output)
            warm, primes, owners, differences = (model.rows(output/name) for name in
                ('RUNTIME.jsonl', 'PRIME.jsonl', 'OWNERS.jsonl', 'WORK_DIFFERENCES.jsonl'))
            self.assertEqual((len(warm), len(primes), len(owners), len(differences)), (768, 24, 24, 384))
            self.assertTrue(all(not item['trajectory_equal'] for item in differences))
            self.assertTrue(all(item['fullstate_equality_not_measured'] for item in differences))
            self.assertEqual({(r['length'], r['arm_id'], r['source_index'], r['repeat']) for r in warm},
                {(n, arm, document, repeat) for n in gate.LENGTHS for arm in gate.ARM_IDS for document in gate.DOCS for repeat in (0, 1)})
            for owner in owners:
                rows = [r for r in warm if r['owner_id'] == owner['owner_id']]
                self.assertEqual([(r['source_index'], r['repeat']) for r in rows],
                                 [(r['source_index'], r['repeat']) for r in prepared['sequence']])
                self.assertAlmostEqual(owner['warm_outer_seconds_sum'], sum(r['outer_seconds'] for r in rows))
                self.assertAlmostEqual(owner['warm_tps'], sum(r['tokens'] for r in rows)/sum(r['outer_seconds'] for r in rows))
                self.assertGreater(owner['accounted_service_seconds'], owner['warm_outer_seconds_sum'])
                self.assertTrue(any(r['draft_runtime']['request']['captures'] > 0 for r in rows))
                self.assertTrue(all(not r['observer_enabled'] and not r['profile_enabled'] for r in rows))
        self.assertEqual(prepared['references'], refs)
        self.assertEqual(len(adapter.made), len(adapter.closed))
        self.assertEqual(len(adapter.closed), 24)
        self.assertTrue(adapter.unchanged)
        self.assertEqual(result['logical_requests'], 792)
        self.assertEqual(result['performance_gate'], 'ANALYSIS_REQUIRED')
        self.assertFalse(result['goal_complete'])

    def test_one_token_mismatch_preserved_failfast_close(self):
        prepared, adapter = state(), FakeAdapter(mismatch=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(ValueError, 'canonical Token mismatch'):
                gate.execute_matrix(adapter, prepared, output)
            self.assertEqual(len(model.rows(output/'PRIME.jsonl')), 2)
            self.assertEqual(len(model.rows(output/'RUNTIME.jsonl')), 32)
            self.assertEqual(len(model.rows(output/'OWNERS.jsonl')), 2)
        self.assertEqual(prepared['returned_calls'], 34)
        self.assertEqual(prepared['validated_calls'], 33)
        self.assertEqual(len(adapter.closed), 2)
        self.assertIsNone(adapter.live)

    def test_candidate_repeat_work_drift_rejected_without_comparing_to_old_work(self):
        prepared, adapter = state(), FakeAdapter(repeat_drift=True)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'work drift'):
                gate.execute_matrix(adapter, prepared, Path(directory))
        self.assertEqual(len(adapter.closed), 2)
        self.assertIsNone(adapter.live)

    def test_model48pass_and_external_terminal_barrier_are_mandatory(self):
        terminal = dict(status='TARGET_TOKEN_GATE_PASSED', passed=True, actual_cuda=True, completed_calls=48,
            attempted_calls=48, returned_calls=48, validated_calls=48, private_owners_closed=48,
            frozen_canonical_token_exact=True, token_gate='FROZEN_CANONICAL_TOKEN_EXACT_PASSED',
            performance_gate='NOT_RUN', performance_measured=False, runner_sha256=gate.MODEL_HELPER_SHA)
        gate.validate_model_terminal(terminal)
        for key, value in (('passed', False), ('status', 'PLAN_ONLY'), ('completed_calls', 47), ('actual_cuda', False)):
            changed = dict(terminal, **{key: value})
            with self.assertRaises(ValueError): gate.validate_model_terminal(changed)
        barrier = dict(schema=1, status='TERMINAL_EXTERNAL_BARRIER_PASSED', passed=True, model_root=str(gate.MODEL_ROOT),
            model_terminal_sha256='a'*64, model_process_absent=True, model_tmux_absent=True, model_children_absent=True,
            gpu_compute_idle=True, gpu_uuid=model.GPU_UUID)
        gate.validate_terminal_barrier(barrier, 'a'*64, model.GPU_UUID)
        for key in ('model_process_absent', 'model_tmux_absent', 'model_children_absent', 'gpu_compute_idle'):
            with self.assertRaises(ValueError):
                gate.validate_terminal_barrier(dict(barrier, **{key: False}), 'a'*64, model.GPU_UUID)

    def test_no_old_prepare_run_generate_quality_or_observer_invocation(self):
        tree = ast.parse(Path(gate.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                text = ast.unparse(node.func)
                self.assertNotIn(text, ('contract.prepare', 'contract.run', 'contract.execute_matrix', 'self.shared.generate', 'self.shared.assert_unchanged'))
                if text == 'self.base.generate':
                    self.assertIsInstance(node.args[-1], ast.Constant)
                    self.assertIs(node.args[-1].value, False)

    def test_scope_once_entire_persistent_owner_guards_restore_and_no_observer(self):
        for variant in ('original', 'candidate'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                shared, engine, group, teacher, student = lifecycle(Path(directory))
                base = shared.base
                original = tuple(layer.self_attn.forward for layer in teacher.model.layers)
                old_list, old_values = group.originals, tuple(group.originals)
                def make(arm, quality):
                    self.assertIs(quality, False); group.idle(); return engine
                def close(owner):
                    self.assertIs(owner, engine); group.idle(); engine.closed=True; group.engines.remove(engine)
                    return dict(engine_closed=True, hook_restored=True)
                base.make, base.close = make, close
                def generate(owner, arm, ids, reference, item, quality):
                    self.assertIs(quality, False); group.events.append('base.before'); group.idle()
                    result = group.generate(owner, ids, 0); result.update(item)
                    return result, .75
                base.generate = generate
                runtime = object.__new__(gate.RuntimeAdapter)
                runtime.base, runtime.external, runtime.contract = base, shared.external, model
                runtime.args, runtime.actual_cuda, runtime.owner_failures = shared.args, False, []
                descriptor = gate.arms()[variant+'_rescue']
                with runtime.owner(descriptor) as handle:
                    scoped = tuple(layer.self_attn.forward for layer in teacher.model.layers)
                    for index in range(2):
                        row, elapsed = runtime.generate(handle, descriptor, [1, 2], [42]*128, dict(execution_index=index))
                        self.assertEqual(elapsed, .75)
                        self.assertTrue(all(layer.self_attn.forward is previous for layer, previous in zip(teacher.model.layers, scoped)))
                    self.assertEqual(group.events.count('base.before'), 2)
                self.assertTrue(all(layer.self_attn.forward is previous for layer, previous in zip(teacher.model.layers, original)))
                self.assertIs(group.originals, old_list); self.assertEqual(tuple(group.originals), old_values)
                self.assertTrue(engine.closed); self.assertEqual(group.engines, [])
                self.assertEqual(handle.close_receipt['engine_closed'], True)
                self.assertEqual(runtime.owner_failures, [])

    def test_scope_guard_failure_still_closes_owner_and_preserves_primary_error(self):
        with tempfile.TemporaryDirectory() as directory:
            shared, engine, group, teacher, student = lifecycle(Path(directory))
            closed = []
            shared.base.make = lambda arm, quality: engine
            def close(owner):
                group.idle(); closed.append(owner); engine.closed=True; group.engines.remove(engine)
                return dict(engine_closed=True, hook_restored=True)
            shared.base.close = close
            runtime = object.__new__(gate.RuntimeAdapter)
            runtime.base, runtime.external = shared.base, shared.external
            runtime.contract = NS(hook_snapshot=model.hook_snapshot, target_hook_lease=model.target_hook_lease,
                                  check_scope=lambda *args: (_ for _ in ()).throw(ValueError('scope guard fail')))
            runtime.args, runtime.actual_cuda, runtime.owner_failures = shared.args, False, []
            with self.assertRaisesRegex(RuntimeError, 'primary generate failed'):
                with runtime.owner(gate.arms()['candidate_rescue']):
                    raise RuntimeError('primary generate failed')
            self.assertEqual(closed, [engine])
            self.assertEqual(runtime.owner_failures[0]['primary_error']['error'], 'primary generate failed')
            self.assertEqual(runtime.owner_failures[0]['cleanup_errors'][0]['stage'], 'scope_or_hook_guard')


if __name__ == '__main__':
    unittest.main()
