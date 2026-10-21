"""CPU-only plan/schema/safety tests. Never calls execute or imports Torch."""
import ast
from contextlib import redirect_stderr, redirect_stdout
import io
import inspect
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from exp.round_overhead import fp32_splitkv_cuda_gate_v1 as gate


class SplitKvGatePlanTests(unittest.TestCase):
    def test_default_plan_only_no_heavy_import_execution_or_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'must_not_exist'
            printed = io.StringIO()
            with patch.object(gate.importlib, 'import_module', side_effect=AssertionError('heavy import')), \
                 patch.object(gate, 'execute', side_effect=AssertionError('actual execution')), \
                 redirect_stdout(printed):
                code = gate.main(['--output', str(output), '--cache-dir', '/not/present'])
            payload = json.loads(printed.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(payload['status'], 'PLAN_ONLY')
            self.assertFalse(payload['passed']); self.assertFalse(payload['actual_cuda'])
            self.assertEqual(payload['new_cuda_calls'], 0)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_fixed_case_metadata_covers_required_contexts_queries_and_boundaries(self):
        cases = gate.cases()
        self.assertEqual(len(cases), 112)
        self.assertEqual(len({row['case_id'] for row in cases}), 112)
        self.assertEqual(sum(row['exact'] for row in cases), 16)
        self.assertEqual(sum(not row['exact'] for row in cases), 96)
        self.assertEqual({row['q'] for row in cases}, {2, 3, 4, 5})
        self.assertTrue({2, 129, 1025, 2048, 4096, 8192, 16384, 16512} <= {row['n'] for row in cases})
        self.assertEqual({row['n'] for row in cases if row['kind'] == 'random'}, {2048, 4096, 6144, 8192, 12288, 16384})
        self.assertEqual({row['allowed_key'] for row in cases if row['kind'] == 'one_key'}, {0, 127, 128, 1024})
        self.assertTrue(all(row['query_layout'] == 'noncontiguous_folded' and row['kv_head_stride'] == 0 for row in cases))
        self.assertEqual({row['mask'] for row in cases}, {'none', 'bool_broadcast', 'additive_broadcast', 'bool_full', 'additive_full'})
        self.assertEqual((gate.plan()['torch_version'], gate.plan()['torch_cuda_version'], gate.plan()['triton_version']),
                         ('2.7.1+cu128', '12.8', '3.3.1'))

    def test_source_pins_match_existing_candidate_and_cpu_proof(self):
        values = gate.source_pins()
        self.assertEqual(values['candidate_sha256'], gate.CANDIDATE_SHA)
        self.assertEqual(values['candidate_test_sha256'], gate.CANDIDATE_TEST_SHA)
        with patch.object(gate, 'sha', return_value='0'*64):
            with self.assertRaisesRegex(ValueError, 'candidate source changed'):
                gate.source_pins()

    def test_exclusive_json_refuses_existing_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'receipt.json'
            gate.write_json(path, {'status': 'original'})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                gate.write_json(path, {'status': 'replacement'})
            self.assertEqual(path.read_bytes(), before)

    def test_fresh_output_empty_cache_and_existing_env_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            cache, output = parent/'cache', parent/'new_result'
            cache.mkdir(mode=0o700)
            env = {'TRITON_CACHE_DIR': str(cache), 'CUDA_CACHE_DISABLE': '1'}
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(gate.preflight_paths(output, cache), (output, cache))
                self.assertFalse(output.exists())
                with patch.dict(os.environ, {'TRITON_CACHE_MANAGER': ''}):
                    with self.assertRaises(ValueError):
                        gate.preflight_paths(output, cache)
                output.mkdir(mode=0o700)
                with self.assertRaisesRegex(ValueError, 'Fresh nonexistent'):
                    gate.preflight_paths(output, cache)
                self.assertTrue(output.is_dir())

    def test_nonempty_cache_and_missing_output_parent_refuse_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            cache = parent/'cache'; cache.mkdir(mode=0o700)
            gate.write_json(cache/'old.json', {'keep': True})
            with patch.dict(os.environ, {'TRITON_CACHE_DIR': str(cache), 'CUDA_CACHE_DISABLE': '1'}, clear=True):
                with self.assertRaisesRegex(ValueError, 'Fresh empty'):
                    gate.preflight_paths(parent/'results', cache)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    gate.preflight_paths(parent/'missing'/'results', cache)
            self.assertEqual(json.loads((cache/'old.json').read_text()), {'keep': True})
            self.assertFalse((parent/'results').exists())

    def test_operator_status_never_equates_random_errors_to_numerical_model_token_or_speed_pass(self):
        for case in gate.cases():
            good = gate.case_status(case, finite=True, unchanged=True,
                candidate_known=True if case['exact'] else None, reference_known=True if case['exact'] else None)
            self.assertTrue(good['mandatory_checks_passed'])
            self.assertEqual(good['status'], 'EXACT_SIMPLE_CASE_PASSED' if case['exact'] else 'DIAGNOSTIC_COMPLETE')
            for name in ('quality_gate', 'performance_gate', 'model_gate', 'token_gate'):
                self.assertEqual(good[name], 'NOT_RUN')
            for kwargs in (dict(finite=False, unchanged=True), dict(finite=True, unchanged=False)):
                self.assertEqual(gate.case_status(case, **kwargs)['status'], 'FAIL')
            self.assertEqual(gate.case_status(case, finite=True, unchanged=True, output_contract=False)['status'], 'FAIL')
        exact = next(row for row in gate.cases() if row['exact'])
        self.assertEqual(gate.case_status(exact, finite=True, unchanged=True,
            candidate_known=True, reference_known=False)['status'], 'FAIL')
        self.assertEqual(gate.case_status(exact, finite=True, unchanged=True,
            candidate_known=False, reference_known=True)['status'], 'FAIL')

    def test_no_execute_requires_explicit_flag_and_preflight_refusal_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'never_created'
            with patch.object(gate, 'fresh_process', return_value=True), \
                 patch.object(gate, 'source_pins', side_effect=ValueError('injected pin failure')) as pins, \
                 patch.object(gate, 'execute', side_effect=AssertionError('must not execute')) as execute, \
                 redirect_stderr(io.StringIO()):
                self.assertEqual(gate.main(['--execute-cuda', '--output', str(path)]), 2)
                pins.assert_called_once_with(); execute.assert_not_called()
            self.assertFalse(path.exists())

    def test_preloaded_runtime_refusal_precedes_source_check_or_execution(self):
        with patch.object(gate, 'fresh_process', return_value=False), \
             patch.object(gate, 'source_pins') as pins, patch.object(gate, 'execute') as execute, \
             redirect_stderr(io.StringIO()):
            self.assertEqual(gate.main(['--execute-cuda']), 2)
            pins.assert_not_called(); execute.assert_not_called()

    def test_kernel_cache_inventory_uses_existing_entries_without_binding_or_compilation(self):
        kernel = types.SimpleNamespace(__name__='_stage_one', device_caches={0: ({'key1': object()}, None, None, None)})
        payload = gate._compiled_cache(types.SimpleNamespace(_kernels=(kernel,)))
        self.assertEqual(payload[0]['compiled_entries'], 1)
        self.assertEqual(payload[0]['key_sha256s'], [gate.hashlib.sha256(b'key1').hexdigest()])
        with self.assertRaises(ValueError):
            gate._compiled_cache(types.SimpleNamespace(_kernels=(types.SimpleNamespace(device_caches={0: ()}),)))

    def test_reference_fixed_backend_source_and_default_module_import_have_no_heavy_import(self):
        source = Path(gate.__file__).read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(not alias.name.startswith(('torch', 'triton', 'rescue', 'subprocess')) for alias in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or '').startswith(('torch', 'triton', 'rescue', 'subprocess')))
        self.assertIn('attention.sdpa_kernel(backends=[attention.SDPBackend.EFFICIENT_ATTENTION])', source)
        self.assertNotIn('assert_close', source)
        self.assertNotIn('allclose', {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)})
        self.assertNotIn('time.perf_counter', source)
        self.assertNotIn('engine.generate', source)
        self.assertNotIn('actual_cuda=True', source)
        body = ast.parse(inspect.getsource(gate.execute))
        true_writes = [node.lineno for node in ast.walk(body) if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant) and node.value.value is True
            and any(isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
                    and target.slice.value == 'actual_cuda' for target in node.targets)]
        syncs = [node.lineno for node in ast.walk(body) if isinstance(node, ast.Call)
                 and ast.unparse(node.func) == 'torch.cuda.synchronize']
        self.assertEqual(len(true_writes), 1); self.assertEqual(len(syncs), 1)
        self.assertGreater(true_writes[0], syncs[0])
        self.assertIn("receipt['reference_returned_cases'] += 1", source)
        self.assertIn("receipt['candidate_returned_cases'] += 1", source)

    def test_fixed_uuid_parser_rejects_noncanonical_alias(self):
        self.assertEqual(gate._canonical_uuid(gate.GPU_UUID), gate.GPU_UUID)
        self.assertEqual(gate._canonical_uuid(gate.GPU_UUID[4:]), gate.GPU_UUID)
        for text in ('GPU-bogus', gate.GPU_UUID[4:].replace('-', ''), '{'+gate.GPU_UUID[4:]+'}'):
            with self.assertRaises(ValueError):
                gate._canonical_uuid(text)


if __name__ == '__main__':
    unittest.main()
