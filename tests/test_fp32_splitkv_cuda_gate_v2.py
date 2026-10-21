"""CPU fixture-delta/AST checks only. No CUDA/Triton compile or model calls."""
import ast
from contextlib import redirect_stdout
import hashlib
import inspect
import io
import json
from pathlib import Path
import textwrap
import unittest
from unittest.mock import patch

import torch

from exp.round_overhead import fp32_splitkv_cuda_gate_v1 as v1
from exp.round_overhead import fp32_splitkv_cuda_gate_v2 as v2


def tree(module):
    return ast.parse(Path(module.__file__).read_text())


def definitions(module, cls):
    return {node.name: node for node in tree(module).body if isinstance(node, cls)}


def dump(node):
    return ast.dump(node, include_attributes=False)


def cpu_inputs(module):
    # Test-only private source copy: change five device literals to CPU. This
    # never modifies either gate's globals/source or executes its CUDA path.
    source = textwrap.dedent(inspect.getsource(module._inputs))
    if source.count("'cuda:0'") != 5:
        raise ValueError('Pinned CPU-only fixture-copy device preimage changed')
    source = source.replace("'cuda:0'", "'cpu'")
    namespace = dict(module._inputs.__globals__)
    exec(compile(source, f'{module.__file__}#CPU_TEST_ONLY', 'exec'), namespace)
    return namespace['_inputs']


def assert_bytes(test, left, right):
    if left is None or right is None:
        test.assertIs(left, right)
        return
    test.assertEqual(left.dtype, right.dtype)
    test.assertEqual(left.shape, right.shape)
    test.assertEqual(left.stride(), right.stride())
    test.assertTrue(torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                                right.contiguous().reshape(-1).view(torch.uint8)))


class SplitKvGateV2FixtureTests(unittest.TestCase):
    def test_old_gate_and_cpu_proof_sha_unchanged_and_bound_into_v2(self):
        self.assertEqual(v2.sha(v2.V1_GATE), v2.V1_GATE_SHA)
        self.assertEqual(v2.sha(v2.V1_GATE_TEST), v2.V1_GATE_TEST_SHA)
        self.assertEqual(v2.V1_GATE_SHA, '62957e77c8d26b61dd89fc7692189ff0f46d61f3d77b9f6131b996eae26de98d')
        self.assertEqual(v2.V1_GATE_TEST_SHA, 'ae1f6665852f99f947c95ade0c7f766186fb80615b1260911a5f8b27aada78ef')
        pins = v2.source_pins()
        self.assertEqual(pins['fixture_version'], 2)
        self.assertEqual(pins['v1_gate_sha256'], v2.V1_GATE_SHA)
        self.assertEqual(pins['v1_gate_test_sha256'], v2.V1_GATE_TEST_SHA)
        self.assertEqual(pins['candidate_sha256'], v1.source_pins()['candidate_sha256'])
        self.assertEqual(pins['candidate_test_sha256'], v1.source_pins()['candidate_test_sha256'])

    def test_only_one_key_zero_qk_insertion_changes_fixture_ast(self):
        old, new = definitions(v1, ast.FunctionDef)['_inputs'], definitions(v2, ast.FunctionDef)['_inputs']
        insertion = ast.parse("if case['kind'] == 'one_key':\n    query_base.zero_()\n    key_base.zero_()\n").body[0]
        matches = [index for index, node in enumerate(new.body) if dump(node) == dump(insertion)]
        self.assertEqual(len(matches), 1)
        index = matches[0]
        self.assertIsInstance(new.body[index-1], ast.Assign)
        self.assertEqual(ast.unparse(new.body[index-1].targets[0]), '(key_base, value_base)')
        self.assertEqual(ast.unparse(new.body[index+1].targets[0]), 'expected')
        del new.body[index]
        self.assertEqual(dump(old), dump(new))

    def test_execute_main_status_cases_and_all_other_functions_ast_identical(self):
        before, after = definitions(v1, ast.FunctionDef), definitions(v2, ast.FunctionDef)
        self.assertEqual(before.keys(), after.keys())
        approved = {'_inputs', 'source_pins', 'plan'}
        for name in before.keys()-approved:
            with self.subTest(function=name):
                self.assertEqual(dump(before[name]), dump(after[name]))
        self.assertEqual(v1.cases(), v2.cases())
        self.assertEqual(len(v2.cases()), 112)
        self.assertEqual(sum(case['exact'] for case in v2.cases()), 16)
        self.assertEqual({case['allowed_key'] for case in v2.cases() if case['kind'] == 'one_key'}, {0, 127, 128, 1024})

    def test_imports_and_original_constants_ast_unchanged_only_version_pins_added(self):
        before, after = tree(v1), tree(v2)
        imports = lambda value: [dump(node) for node in value.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertEqual(imports(before), imports(after))
        assignments = lambda value: {ast.unparse(node.targets[0]): dump(node) for node in value.body if isinstance(node, ast.Assign)}
        old, new = assignments(before), assignments(after)
        self.assertEqual(set(new)-set(old), {'V1_GATE', 'V1_GATE_TEST', 'V1_GATE_SHA', 'V1_GATE_TEST_SHA', 'FIXTURE_VERSION'})
        for name in old:
            self.assertEqual(old[name], new[name])

    def test_plan_version_only_metadata_delta_not_a_pass_or_gpu_result(self):
        before, after = v1.plan(), v2.plan()
        extras = {'fixture_version', 'v1_gate_sha256', 'v1_gate_test_sha256', 'fixture_only_change', 'prior_v1_failure'}
        self.assertEqual(set(after)-set(before), extras)
        self.assertEqual({key: value for key, value in after.items() if key not in extras}, before)
        self.assertEqual(after['fixture_version'], 2)
        self.assertEqual(after['status'], 'PLAN_ONLY')
        self.assertFalse(after['passed']); self.assertFalse(after['actual_cuda'])
        with patch.object(v2.importlib, 'import_module', side_effect=AssertionError('heavy import')), \
             patch.object(v2, 'execute', side_effect=AssertionError('CUDA execution')), redirect_stdout(io.StringIO()) as captured:
            self.assertEqual(v2.main([]), 0)
        self.assertEqual(json.loads(captured.getvalue())['fixture_version'], 2)

    def test_known_value_checks_stay_strict_for_both_candidate_and_reference(self):
        for case in v2.cases():
            if not case['exact']:
                continue
            for candidate_known, reference_known in ((True, False), (False, True), (False, False)):
                result = v2.case_status(case, finite=True, unchanged=True,
                    candidate_known=candidate_known, reference_known=reference_known)
                self.assertEqual(result['status'], 'FAIL')
            self.assertEqual(v2.case_status(case, finite=True, unchanged=True,
                candidate_known=True, reference_known=True)['status'], 'EXACT_SIMPLE_CASE_PASSED')

    def test_cpu_value_delta_qk_only_all_four_one_key_cases_v_mask_expected_and_rng_unchanged(self):
        old, new = cpu_inputs(v1), cpu_inputs(v2)
        hashes = {path: v2.sha(path) for path in (v2.V1_GATE, v2.V1_GATE_TEST, v2.CANDIDATE, v2.CANDIDATE_TEST)}
        globals_before = dict(v1._inputs.__globals__)
        cpu_rng_before = torch.random.get_rng_state().clone()
        initialized = torch.cuda.is_initialized()
        for case in (row for row in v2.cases() if row['kind'] == 'one_key'):
            with self.subTest(q=case['q'], key=case['allowed_key']):
                previous, actual = old(torch, case), new(torch, case)
                for old_tensor, new_tensor in zip(previous[:2], actual[:2]):
                    self.assertTrue(bool(torch.count_nonzero(old_tensor)))
                    self.assertEqual(int(torch.count_nonzero(new_tensor)), 0)
                    self.assertEqual(old_tensor.shape, new_tensor.shape)
                    self.assertEqual(old_tensor.stride(), new_tensor.stride())
                for left, right in zip(previous[2:], actual[2:]):
                    assert_bytes(self, left, right)
                self.assertEqual(int(torch.count_nonzero(actual[3].isfinite() if actual[3].dtype == torch.float32 else actual[3])), case['q'])
        self.assertEqual(dict(v1._inputs.__globals__), globals_before)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), cpu_rng_before))
        self.assertEqual(torch.cuda.is_initialized(), initialized)
        self.assertEqual({path: v2.sha(path) for path in hashes}, hashes)

    def test_representative_other_families_cpu_byte_identical(self):
        old, new = cpu_inputs(v1), cpu_inputs(v2)
        sample = [case for case in v2.cases() if
                  case['kind'] == 'random' and case['q'] == 2 and case['n'] == 2048 or
                  case['kind'] == 'zero_scores' and case['q'] == 3 or
                  case['kind'] == 'constant_v' and case['q'] == 2 or
                  case['kind'] == 'all_masked' and case['q'] in (2, 3)]
        self.assertEqual(len(sample), 8)
        for case in sample:
            with self.subTest(case=case['case_id']):
                for left, right in zip(old(torch, case), new(torch, case)):
                    assert_bytes(self, left, right)


if __name__ == '__main__':
    unittest.main()
