"""CPU-only v3 contract/AST proof; no CUDA execution, compilation or model calls."""
import ast
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from exp.round_overhead import fp32_splitkv_cuda_gate_v2 as v2
from exp.round_overhead import fp32_splitkv_cuda_gate_v3 as v3


def tree(module):
    return ast.parse(Path(module.__file__).read_text())


def definitions(module):
    return {node.name: node for node in tree(module).body if isinstance(node, ast.FunctionDef)}


def dump(node):
    return ast.dump(node, include_attributes=False)


class NormalizeScopedLabels(ast.NodeTransformer):
    """Only receipt labels differ in execute/main, never operator operations."""
    def visit_Call(self, node):
        node = self.generic_visit(node)
        node.keywords = [keyword for keyword in node.keywords
                         if keyword.arg not in ('passed_scope', 'reference_equivalence_gate')]
        for keyword in node.keywords:
            if keyword.arg == 'candidate_analytic_smoke_passed':
                keyword.arg = 'operator_smoke_passed'
        return node

    def visit_Constant(self, node):
        if node.value == 'CANDIDATE_ANALYTIC_SMOKE_PASSED':
            node.value = 'OPERATOR_SMOKE_PASSED'
        elif node.value == 'candidate_analytic_smoke_passed':
            node.value = 'operator_smoke_passed'
        return node


class SplitKvGateV3ContractTests(unittest.TestCase):
    def test_frozen_candidate_v1_v2_sources_and_tests_are_bound(self):
        pins = v3.source_pins()
        self.assertEqual(pins['candidate_sha256'], v3.CANDIDATE_SHA)
        self.assertEqual(pins['candidate_test_sha256'], v3.CANDIDATE_TEST_SHA)
        for version in (1, 2):
            self.assertEqual(pins[f'v{version}_gate_sha256'], getattr(v3, f'V{version}_GATE_SHA'))
            self.assertEqual(pins[f'v{version}_gate_test_sha256'], getattr(v3, f'V{version}_GATE_TEST_SHA'))
        self.assertEqual(pins['fixture_version'], 2)
        self.assertEqual(pins['contract_version'], 3)
        self.assertEqual(pins['v2_gate_sha256'], '1d4b340786851eb41c80489fb6c3b775101ba8125e97d50e81903d4536a52549')
        self.assertEqual(pins['v2_gate_test_sha256'], '525ad7ade08a3fff6c8b593c88d8d32bc4d7f3399de35025f48146b4f1ff02b5')

    def test_all112_fixtures_and_input_arithmetic_ast_identical_to_v2(self):
        self.assertEqual(v2.cases(), v3.cases())
        self.assertEqual(len(v3.cases()), 112)
        self.assertEqual(sum(case['exact'] for case in v3.cases()), 16)
        before, after = definitions(v2), definitions(v3)
        self.assertEqual(dump(before['_inputs']), dump(after['_inputs']))
        self.assertEqual(dump(before['cases']), dump(after['cases']))

    def test_execution_arithmetic_unchanged_only_explicit_receipt_labels(self):
        before, after = definitions(v2), definitions(v3)
        self.assertEqual(before.keys(), after.keys())
        for name in ('execute', 'main'):
            normalized = NormalizeScopedLabels().visit(copy.deepcopy(after[name]))
            self.assertEqual(dump(before[name]), dump(normalized), name)
        allowed = {'source_pins', 'plan', 'case_status', 'execute', 'main'}
        for name in before.keys()-allowed:
            self.assertEqual(dump(before[name]), dump(after[name]), name)

    def test_imports_and_previous_constants_unchanged(self):
        imports = lambda module: [dump(node) for node in tree(module).body
                                  if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertEqual(imports(v2), imports(v3))
        assignments = lambda module: {ast.unparse(node.targets[0]): dump(node)
                                     for node in tree(module).body if isinstance(node, ast.Assign)}
        before, after = assignments(v2), assignments(v3)
        self.assertEqual(set(after)-set(before),
                         {'V2_GATE', 'V2_GATE_TEST', 'V2_GATE_SHA', 'V2_GATE_TEST_SHA', 'CONTRACT_VERSION'})
        for name in before:
            self.assertEqual(before[name], after[name], name)

    def test_reference_false_or_missing_is_diagnostic_candidate_true_is_strict(self):
        for case in (row for row in v3.cases() if row['exact']):
            for reference_known in (False, None, True):
                with self.subTest(case=case['case_id'], reference=reference_known):
                    result = v3.case_status(case, finite=True, unchanged=True,
                                           candidate_known=True, reference_known=reference_known)
                    self.assertEqual(result['status'], 'CANDIDATE_ANALYTIC_CASE_PASSED')
                    self.assertTrue(result['mandatory_checks_passed'])
                    self.assertEqual(result['reference_analytic_gate'], 'DIAGNOSTIC_ONLY')
                    self.assertEqual(result['reference_equivalence_gate'], 'NOT_ESTABLISHED_DIAGNOSTIC_ONLY')
            for candidate_known in (False, None, 1):
                result = v3.case_status(case, finite=True, unchanged=True,
                                       candidate_known=candidate_known, reference_known=True)
                self.assertEqual(result['status'], 'FAIL')
                self.assertFalse(result['mandatory_checks_passed'])

    def test_nonfinite_mutation_or_output_contract_failure_remain_mandatory(self):
        for exact in (False, True):
            case = next(row for row in v3.cases() if row['exact'] is exact)
            for arguments in (dict(finite=False, unchanged=True), dict(finite=True, unchanged=False),
                              dict(finite=True, unchanged=True, output_contract=False)):
                result = v3.case_status(case, candidate_known=True, reference_known=False, **arguments)
                self.assertEqual(result['status'], 'FAIL')
                self.assertFalse(result['mandatory_checks_passed'])

    def test_random_difference_never_becomes_numerical_reference_token_or_speed_pass(self):
        for case in (row for row in v3.cases() if not row['exact']):
            result = v3.case_status(case, finite=True, unchanged=True,
                                   candidate_known=False, reference_known=False)
            self.assertEqual(result['status'], 'DIAGNOSTIC_COMPLETE')
            self.assertEqual(result['random_equivalence_gate'], 'NOT_DEFINED')
            self.assertEqual(result['reference_equivalence_gate'], 'NOT_ESTABLISHED_DIAGNOSTIC_ONLY')
            for name in ('quality_gate', 'performance_gate', 'model_gate', 'token_gate'):
                self.assertEqual(result[name], 'NOT_RUN')

    def test_plan_is_explicit_contract_correction_not_a_success_or_new_fixture(self):
        plan = v3.plan()
        self.assertEqual(plan['status'], 'PLAN_ONLY')
        for key in ('passed', 'actual_cuda', 'candidate_analytic_smoke_passed', 'numerical_pass'):
            self.assertIs(plan[key], False)
        self.assertEqual(plan['new_cuda_calls'], 0)
        self.assertEqual(plan['fixture_version'], 2)
        self.assertEqual(plan['contract_version'], 3)
        self.assertEqual(plan['cases'], v2.plan()['cases'])
        self.assertEqual(plan['random_tolerance'], None)
        self.assertIn('terminalFAIL', plan['prior_failures'])
        self.assertIn('falsified', plan['zero_qk_hypothesis'])
        self.assertEqual(plan['passed_scope'], 'CANDIDATE_ANALYTIC_ONLY')
        for name in ('quality_gate', 'performance_gate', 'model_gate', 'token_gate'):
            self.assertEqual(plan[name], 'NOT_RUN')

    def test_default_plan_does_not_import_execute_pin_preflight_or_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'never_created'
            with patch.object(v3.importlib, 'import_module', side_effect=AssertionError('heavy import')), \
                 patch.object(v3, 'execute', side_effect=AssertionError('execution')) as execute, \
                 patch.object(v3, 'source_pins', side_effect=AssertionError('pins')) as pins, \
                 patch.object(v3, 'preflight_paths', side_effect=AssertionError('preflight')) as preflight, \
                 redirect_stdout(io.StringIO()) as captured:
                self.assertEqual(v3.main(['--output', str(output)]), 0)
            self.assertFalse(output.exists())
            execute.assert_not_called(); pins.assert_not_called(); preflight.assert_not_called()
            self.assertEqual(json.loads(captured.getvalue())['status'], 'PLAN_ONLY')

    def test_actual_fresh_process_default_does_not_load_torch_or_triton(self):
        script = """import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'triton'):
        raise AssertionError('Forbidden heavy import '+name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from exp.round_overhead import fp32_splitkv_cuda_gate_v3 as gate
assert gate.main([]) == 0
assert not any(name.split('.')[0] in ('torch', 'triton') for name in sys.modules)
"""
        process = subprocess.run([sys.executable, '-B', '-c', script], cwd=v3.ROOT,
                                 capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)['status'], 'PLAN_ONLY')

    def test_execute_cli_refuses_changed_pins_without_creating_output_or_importing_heavy(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'never_created'
            with patch.object(v3, 'fresh_process', return_value=True), \
                 patch.object(v3, 'source_pins', side_effect=ValueError('changed source')) as pins, \
                 patch.object(v3.importlib, 'import_module', side_effect=AssertionError('heavy import')), \
                 patch.object(v3, 'execute', side_effect=AssertionError('execution')) as execute, \
                 patch.object(sys, 'stderr', new_callable=io.StringIO) as captured:
                self.assertEqual(v3.main(['--execute-cuda', '--output', str(output)]), 2)
            pins.assert_called_once_with(); execute.assert_not_called()
            self.assertIn('changed source', captured.getvalue())
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
