"""CPU fake orchestration/lifecycle proofs, NOT actual CUDA/model Token proof."""
from contextlib import nullcontext, redirect_stdout
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
import uuid

from exp.round_overhead import fp32_splitkv_model_gate_v1 as gate


def scope_summary(variant):
    enabled = variant == 'candidate'
    records = [dict(layer=layer, query_tokens=2, key_tokens=2048, action='candidate', outcome='returned')
               for layer in range(28)] if enabled else []
    return dict(state='closed', enabled=enabled, calls=len(records), candidate_calls=len(records),
                original_calls=0, records=records)


def row_for(arm, reference, item):
    row = dict(item, generated=list(reference), request_id=uuid.uuid4().hex,
               nonzero_injection=arm['family'] == 'rescue', observed_rounds=[dict(draft=[1], checks=[1, 2])],
               queries=[dict(query='q')], native_t=dict(sha='teacher'), native_s=dict(sha='student'))
    row.update(dict.fromkeys(gate.WORK, 1))
    row.update(tokens=len(reference), accepted=2, proposed=4, blocks=1,
        trace=[dict(round_index=0, prefix=item['length']-1, cap=4, proposed=4, accepted=2,
                    terminal=True, workset_tokens=512, instrumented_seconds=1.)])
    return row


def prepared():
    refs = {f'{document}:{length}': [42]*128 for length in gate.LENGTHS for document in gate.DOCS}
    historic = {}
    for length in gate.LENGTHS:
        for document in gate.DOCS:
            for mode, arm in gate.arms().items():
                historic[length, document, mode] = row_for(arm, refs[f'{document}:{length}'],
                    dict(length=length, source_index=document, mode=mode))
    return dict(inputs={(document, length): [1, 2] for length in gate.LENGTHS for document in gate.DOCS},
                references=refs, historic=historic, seen=set())


class MatrixAdapter:
    def __init__(self, source, mismatch=False):
        self.source, self.mismatch = source, mismatch
        self.live, self.made, self.closed, self.compared = None, [], [], []
        self.actual_cuda = True  # Synthetic metadata, never an execution receipt.
        self.unchanged = False

    def make(self, arm):
        assert self.live is None
        self.live = object()
        self.made.append(self.live)
        return self.live

    def generate(self, owner, arm, ids, reference, item, variant):
        assert owner is self.live
        row = row_for(arm, reference, item)
        if variant == 'candidate':
            row['accepted'] = 3
            row['trace'][0]['accepted'] = 3
            row['observed_rounds'][0]['draft'] = [2]
            if self.mismatch:
                row['generated'][0] += 1
        self.source['returned_calls'] = self.source.get('returned_calls', 0)+1
        return row, scope_summary(variant)

    def validate(self, row, reference, arm):
        assert row['generated'] == reference
        assert row['nonzero_injection'] is (arm['family'] == 'rescue')

    def compare_historical(self, old, new):
        assert new['variant'] == 'original'
        assert all(old[key] == new[key] for key in gate.WORK)
        self.compared.append(new)

    def close(self, owner):
        assert owner is self.live
        self.closed.append(owner)
        self.live = None
        return dict(engine_closed=True, hook_restored=True)

    def assert_unchanged(self):
        assert self.live is None
        self.unchanged = True


class Observation:
    def __init__(self, module, fail_install=False):
        self.module, self.originals, self.fail_install = module, [], fail_install

    def install(self):
        original = self.module.forward
        self.originals.append((self.module, original))
        self.module.forward = object()
        if self.fail_install:
            raise RuntimeError('partial observation install')

    def restore(self):
        for module, original in self.originals:
            module.forward = original
        self.originals.clear()


class FakeGroup:
    """Mirror frozen HookOwners idle/install-before-try/generate/finally order."""
    def __init__(self, teacher, student, engine):
        self.modules = [teacher.model.layers[-1].self_attn]+[layer.self_attn for layer in student.model.layers]
        self.originals = [module.forward for module in self.modules]
        self.engines, self.events = [engine], []

    def idle(self):
        self.events.append('idle')
        gate.require(all(module.forward is original for module, original in zip(self.modules, self.originals)), 'idle identity')
        gate.require(all(not owner.observation.originals for owner in self.engines), 'idle active observation')

    def generate(self, engine, ids, eos):
        self.idle()
        gate.require(engine in self.engines and not engine.closed, 'owner registration')
        self.events.append('install')
        engine.observation.install()
        try:
            self.events.append('engine.generate')
            return engine.generate(ids, eos)
        finally:
            self.events.append('restore')
            engine.observation.restore()
            self.idle()


def lifecycle(output, *, candidate_failure=False, install_failure=False, scope_exit_failure=False):
    teacher = NS(model=NS(layers=[NS(self_attn=NS(forward=object())) for _ in range(28)]))
    student = NS(model=NS(layers=[NS(self_attn=NS(forward=object())) for _ in range(3)]))
    engine = NS(closed=False, observation=Observation(teacher.model.layers[-1].self_attn, install_failure))
    group = FakeGroup(teacher, student, engine)
    source = prepared()
    class Scope:
        def __init__(self, target, *, enabled, cache_dir):
            self.target, self.enabled, self.state = target, enabled, 'new'
            self.original = [layer.self_attn.forward for layer in target.model.layers]
        def __enter__(self):
            self.state = 'active' if self.enabled else 'disabled'
            if self.enabled:
                for layer in self.target.model.layers:
                    layer.self_attn.forward = object()
            return self.target
        def __exit__(self, *args):
            for layer, original in zip(self.target.model.layers, self.original):
                layer.self_attn.forward = original
            self.state = 'closed'
            if scope_exit_failure:
                raise RuntimeError('scope exit after restoration')
        def stats(self):
            return dict(scope_summary('candidate' if self.enabled else 'original'), state=self.state)
    row = row_for(gate.arms()[gate.MODES[0]], [42]*128, dict(length=2048, source_index=271))
    def generate(ids, eos):
        if candidate_failure:
            raise RuntimeError('candidate operation failed')
        return copy.deepcopy(row)
    engine.generate = generate
    def base_generate(owner, arm, ids, reference, item, quality):
        assert quality is True
        group.events.append('base.before')
        group.idle()  # The real base.before() guard must pass while scoped.
        result = group.generate(owner, ids, 0)
        result.update(item)
        return result, 999.  # Instrumented clock must not be published as TPS.
    base = NS(teacher=teacher, student=student, group=group, generate=base_generate,
              r=NS(torch=NS(no_grad=nullcontext, cuda=NS(synchronize=lambda _: None))))
    runtime = object.__new__(gate.RuntimeAdapter)
    runtime.base, runtime.external = base, NS(TargetAttentionScope=Scope)
    runtime.args, runtime.prepared, runtime.actual_cuda = NS(cache_dir=Path('/not_used'), output=output), source, False
    return runtime, engine, group, teacher, student


class ModelGateCpuTests(unittest.TestCase):
    def test_default_plan_has_no_heavy_import_output_or_success(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'never_created'
            with patch.object(gate, 'run', side_effect=AssertionError('execution')) as run, \
                 patch.object(gate.importlib, 'import_module', side_effect=AssertionError('heavy import')), \
                 redirect_stdout(io.StringIO()) as captured:
                self.assertEqual(gate.main(['--output', str(output)]), 0)
            run.assert_not_called()
            self.assertFalse(output.exists())
        plan = json.loads(captured.getvalue())
        self.assertEqual(plan['status'], 'PLAN_ONLY')
        self.assertFalse(plan['passed']); self.assertFalse(plan['actual_cuda'])
        self.assertEqual(plan['planned_generation_calls'], 48)
        self.assertEqual(plan['planned_private_owners'], 48)
        self.assertEqual(plan['performance_gate'], 'NOT_RUN')

    def test_fresh_subprocess_plan_does_not_load_torch_triton_or_rescue(self):
        script = """import builtins,sys
original=builtins.__import__
def guard(name,*args,**kwargs):
 if name.split('.')[0] in ('torch','triton','rescue'): raise AssertionError(name)
 return original(name,*args,**kwargs)
builtins.__import__=guard
from exp.round_overhead import fp32_splitkv_model_gate_v1 as gate
assert gate.main([])==0
assert not any(name.split('.')[0] in ('torch','triton','rescue') for name in sys.modules)
"""
        result = subprocess.run([sys.executable, '-B', '-c', script], cwd=gate.ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['new_model_calls'], 0)

    def test_48_private_owners_frozen_refs_and_different_candidate_work_allowed(self):
        data = prepared()
        before = copy.deepcopy(data['references'])
        adapter = MatrixAdapter(data)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = gate.execute_matrix(adapter, data, output)
            saved = gate.rows(output/'QUALITY.jsonl')
            self.assertEqual(len(saved), 48)
            self.assertEqual(len(list(output.glob('OWNER_*.json'))), 48)
            self.assertEqual(len(list(output.glob('DIFF_*.json'))), 24)
            self.assertTrue(all(not gate.read(path)['trajectory_equal'] for path in output.glob('DIFF_*.json')))
            self.assertEqual({(r['length'], r['source_index'], r['mode'], r['variant']) for r in saved},
                {(n, d, m, v) for n in gate.LENGTHS for d in gate.DOCS for m in gate.MODES for v in gate.VARIANTS})
            self.assertTrue(all(r['performance_eligible'] is False and r['observer_enabled'] is True for r in saved))
        self.assertEqual(data['references'], before)
        self.assertEqual(len(adapter.made), len(adapter.closed))
        self.assertEqual(len(adapter.compared), 24)
        self.assertTrue(adapter.unchanged)
        self.assertEqual(result['attempted_calls'], result['returned_calls'])
        self.assertEqual(result['validated_calls'], result['private_owners_closed'])
        self.assertEqual(result['completed_calls'], 48)
        self.assertEqual(result['status'], 'TARGET_TOKEN_GATE_PASSED')
        self.assertEqual(result['performance_gate'], 'NOT_RUN')
        self.assertFalse(result['performance_measured']); self.assertFalse(result['goal_complete'])

    def test_one_token_mismatch_saved_then_failfast_and_owner_closed(self):
        data = prepared(); adapter = MatrixAdapter(data, mismatch=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(ValueError, 'canonical Token mismatch'):
                gate.execute_matrix(adapter, data, output)
            saved = gate.rows(output/'QUALITY.jsonl')
            self.assertEqual(len(saved), 2)
            self.assertNotEqual(saved[1]['generated'], data['references']['271:2048'])
            self.assertEqual(len(list(output.glob('OWNER_*.json'))), 2)
        self.assertEqual(data['attempted_calls'], 2)
        self.assertEqual(data['returned_calls'], 2)
        self.assertEqual(data['validated_calls'], 1)
        self.assertEqual(data['closed_owners'], 2)
        self.assertEqual(len(adapter.made), 2)
        self.assertEqual(len(adapter.closed), 2)
        self.assertIsNone(adapter.live)

    def test_all28_layers_required_and_original_scope_must_be_identity(self):
        gate.check_scope(scope_summary('original'), 'original')
        gate.check_scope(scope_summary('candidate'), 'candidate')
        for mutate in (lambda row: row.update(candidate_calls=0), lambda row: row.update(state='active'),
                       lambda row: row['records'][0].update(layer=1),
                       lambda row: row['records'][0].update(outcome='error')):
            summary = scope_summary('candidate'); mutate(summary)
            with self.assertRaises(ValueError):
                gate.check_scope(summary, 'candidate')
        with self.assertRaises(ValueError):
            gate.check_scope(scope_summary('candidate'), 'original')

    def test_scope_group_lease_keeps_before_generate_guards_and_restores_exact_identity(self):
        for variant in gate.VARIANTS:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                runtime, engine, group, teacher, student = lifecycle(Path(directory))
                original = tuple(layer.self_attn.forward for layer in teacher.model.layers)
                group_list, group_values = group.originals, tuple(group.originals)
                row, summary = runtime.generate(engine, gate.arms()[gate.MODES[0]], [1, 2], [42]*128,
                    dict(execution_index=0, length=2048, source_index=271), variant)
                self.assertTrue(all(layer.self_attn.forward is previous for layer, previous in zip(teacher.model.layers, original)))
                self.assertIs(group.originals, group_list)
                self.assertEqual(tuple(group.originals), group_values)
                self.assertEqual(engine.observation.originals, [])
                group.idle()
                self.assertIn('base.before', group.events)
                self.assertIn('engine.generate', group.events)
                gate.check_scope(summary, variant)
                self.assertEqual(runtime.prepared['returned_calls'], 1)
                self.assertEqual(gate.read(Path(directory)/'GENERATION_RETURNED_000.json')['row']['generated'], [42]*128)
                self.assertNotIn('outer_tps', row)
                self.assertTrue(row['all_timing_fields_are_instrumented'])

    def test_candidate_and_partial_install_failure_restore_scope_and_group(self):
        for option in ('candidate_failure', 'install_failure'):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as directory:
                runtime, engine, group, teacher, student = lifecycle(Path(directory), **{option: True})
                original = tuple(layer.self_attn.forward for layer in teacher.model.layers)
                old_list, old_values = group.originals, tuple(group.originals)
                with self.assertRaises(RuntimeError):
                    runtime.generate(engine, gate.arms()[gate.MODES[0]], [1, 2], [42]*128,
                        dict(execution_index=0, length=2048, source_index=271), 'candidate')
                self.assertTrue(all(layer.self_attn.forward is previous for layer, previous in zip(teacher.model.layers, original)))
                self.assertIs(group.originals, old_list)
                self.assertEqual(tuple(group.originals), old_values)
                self.assertEqual(engine.observation.originals, [])
                group.idle()
                self.assertFalse((Path(directory)/'GENERATION_RETURNED_000.json').exists())

    def test_returned_raw_survives_scope_exit_error_without_faking_completed_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, engine, group, teacher, student = lifecycle(Path(directory), scope_exit_failure=True)
            original = tuple(layer.self_attn.forward for layer in teacher.model.layers)
            with self.assertRaisesRegex(RuntimeError, 'scope exit'):
                runtime.generate(engine, gate.arms()[gate.MODES[0]], [1, 2], [42]*128,
                    dict(execution_index=0, length=2048, source_index=271), 'candidate')
            saved = gate.read(Path(directory)/'GENERATION_RETURNED_000.json')
            self.assertEqual(saved['row']['generated'], [42]*128)
            self.assertEqual(saved['target_scope_at_return']['state'], 'active')
            self.assertEqual(runtime.prepared['returned_calls'], 1)
            self.assertTrue(all(layer.self_attn.forward is previous for layer, previous in zip(teacher.model.layers, original)))
            group.idle()

    def test_original_work_diagnostics_ignore_instrumented_clocks_but_not_actions(self):
        arm = gate.arms()[gate.MODES[0]]
        left = row_for(arm, [42], dict(length=2048))
        right = copy.deepcopy(left)
        right['trace'][0]['instrumented_seconds'] = 999.
        self.assertTrue(gate.work_difference(left, right)['trajectory_equal'])
        right['trace'][0]['cap'] = 2
        self.assertFalse(gate.work_difference(left, right)['trajectory_equal'])

    def test_gate_receipt_rejects_partial_plan_wrong_source_or_nonactual(self):
        receipt = dict(status='CANDIDATE_ANALYTIC_SMOKE_PASSED', passed=True, actual_cuda=True,
            candidate_analytic_smoke_passed=True, passed_scope='CANDIDATE_ANALYTIC_ONLY', completed_cases=112,
            attempted_cases=112, synchronized_cases=112, reference_returned_cases=112, candidate_returned_cases=112,
            numerical_pass=False, quality_gate='NOT_RUN', performance_gate='NOT_RUN', model_gate='NOT_RUN', token_gate='NOT_RUN',
            gpu_uuid=gate.GPU_UUID, torch_version='2.7.1+cu128', torch_cuda_version='12.8', triton_version='3.3.1', tf32=False,
            source_pins=dict(candidate_sha256=gate.CANDIDATE_SHA, gate_sha256=gate.OPERATOR_GATE_SHA, contract_version=3, fixture_version=2))
        gate.validate_operator_gate(receipt, 'a'*64)
        for key, value in (('completed_cases', 100), ('actual_cuda', False), ('status', 'PLAN_ONLY'), ('numerical_pass', True)):
            changed = copy.deepcopy(receipt); changed[key] = value
            with self.assertRaises(ValueError):
                gate.validate_operator_gate(changed, 'a'*64)
        changed = copy.deepcopy(receipt); changed['source_pins']['gate_sha256'] = 'b'*64
        with self.assertRaises(ValueError):
            gate.validate_operator_gate(changed, 'a'*64)


if __name__ == '__main__':
    unittest.main()
