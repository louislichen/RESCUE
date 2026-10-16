"""Terminal-Draft regression tests, without importing CUDA model packages.

Pure helpers and AST compatibility checks run on a standard-library machine.
CPU tensor integration tests run when PyTorch is present; no GPU is required.
"""

import ast
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from classic.terminal_commit import (
    candidate_prefix_through_eos,
    committed_generation_should_stop,
    finish_terminal_commit,
)

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]


def load_function(path, name, namespace, class_name=None):
    """Compile the real implementation without importing its CUDA imports."""
    parsed = ast.parse(path.read_text())
    nodes = parsed.body
    if class_name is not None:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef)
                     and n.name == class_name).body
    function = next(n for n in nodes if isinstance(n, ast.FunctionDef)
                    and n.name == name)
    function.decorator_list = []
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class TerminalPureTests(unittest.TestCase):
    def test_exact_hit_overshoot_one_and_not_yet_terminal(self):
        self.assertTrue(committed_generation_should_stop([4, 5], 2, 99))
        self.assertTrue(committed_generation_should_stop([4, 5, 6], 2, 99))
        self.assertTrue(committed_generation_should_stop([4], 1, 99))
        self.assertFalse(committed_generation_should_stop([4], 2, 99))
        with self.assertRaises(ValueError):
            committed_generation_should_stop([], 0, 99)

    def test_eos_trimming_keeps_first_eos_and_not_its_suffix(self):
        self.assertEqual(candidate_prefix_through_eos([4, 99, 6, 99], 99), 2)
        self.assertEqual(candidate_prefix_through_eos([99, 6], 99), 1)
        self.assertEqual(candidate_prefix_through_eos([4, 5], None), 2)
        self.assertTrue(committed_generation_should_stop([4, 99], 8, 99))
        self.assertFalse(committed_generation_should_stop([4, 5], 8, 99))

    def test_terminal_callback_exactly_once_and_no_next_action(self):
        callback = Mock()
        model = SimpleNamespace(csgb_terminal_commit_callback=callback)
        ids = object()
        predicate = Mock(return_value=True)
        self.assertTrue(finish_terminal_commit(model, ids, predicate))
        predicate.assert_called_once_with(ids)
        callback.assert_called_once_with(ids)

    def test_no_callback_or_predicate_on_compatibility_path(self):
        callback = Mock()
        model = SimpleNamespace(csgb_terminal_commit_callback=callback)
        self.assertFalse(finish_terminal_commit(model, object(), None))
        self.assertFalse(finish_terminal_commit(model, object(), lambda ids: False))
        callback.assert_not_called()

    def test_verifier_appends_optional_arguments_and_stops_before_draft(self):
        tree = ast.parse((ROOT / "classic/utils_classic.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "verify")
        self.assertEqual([arg.arg for arg in fn.args.args][-2:],
                         ["stop_after_commit", "terminal_eos_token_id"])
        self.assertTrue(all(isinstance(value, ast.Constant) and value.value is None
                            for value in fn.args.defaults))
        finish = next(n for n in ast.walk(fn) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Name)
                      and n.func.id == "finish_terminal_commit")
        draft = next(n for n in ast.walk(fn) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == "draft")
        compact = next(n for n in ast.walk(fn) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Attribute) and n.func.attr == "fill_")
        self.assertLess(compact.lineno, finish.lineno)
        self.assertLess(finish.lineno, draft.lineno)
        self.assertEqual(ast.unparse(finish.args[1]), "input_ids")


@unittest.skipIf(torch is None, "CPU integration requires PyTorch")
class TerminalVerifyCPUTests(unittest.TestCase):
    @staticmethod
    def verify_case(tokens=(10, 11, 12), stop=True, eos=None):
        namespace = dict(torch=torch,
                         candidate_prefix_through_eos=candidate_prefix_through_eos,
                         finish_terminal_commit=finish_terminal_commit)
        verify = load_function(ROOT / "classic/utils_classic.py", "verify", namespace)
        current_length = torch.tensor([5])
        data = torch.arange(5, dtype=torch.float32).reshape(1, 1, 5, 1)
        events = []
        draft = Mock(return_value=(torch.tensor([[88]]), torch.tensor([5]),
                                   torch.ones((1, 1)), torch.tensor([0])))

        def callback(ids):
            events.append(("terminal", ids.tolist(), current_length.item()))

        model = SimpleNamespace(
            use_retrieval_cache=False,
            csgb_terminal_commit_callback=Mock(side_effect=callback),
            draft=draft,
            update_dynamic_cache_budget=Mock(side_effect=lambda *a: events.append(("budget", a[0]))),
        )
        logits = torch.full((1, 3, 100), -100.0)
        for row, next_id in enumerate((tokens[1], tokens[2], 77)):
            logits[0, row, next_id] = 1.0
        predicate = None if stop is None else Mock(return_value=stop)
        result = verify(
            torch.tensor([[1, 2]]), logits, torch.tensor([tokens]),
            torch.tensor([2, 3, 4]), torch.ones((3, 3)), [data],
            current_length, torch.tensor([0, 0]), model, 2, .7, 10, None,
            stop_after_commit=predicate, terminal_eos_token_id=eos,
        )
        return result, model, current_length, events, predicate

    def test_stopped_verify_compacts_and_does_not_draft(self):
        result, model, length, events, predicate = self.verify_case()
        self.assertEqual(result[0].tolist(), [[1, 2, 10, 11, 12]])
        self.assertEqual(result[1:3], ([10, 11, 12], 2))
        self.assertEqual(result[3:], (None, None, None, None))
        self.assertEqual(length.item(), 5)
        self.assertEqual(events, [("budget", 2), ("terminal", [[1, 2, 10, 11, 12]], 5)])
        model.draft.assert_not_called()
        model.csgb_terminal_commit_callback.assert_called_once()
        self.assertEqual(predicate.call_args.args[0].tolist(), [[1, 2, 10, 11, 12]])

    def test_nonterminal_verify_calls_draft_once(self):
        result, model, _, _, _ = self.verify_case(stop=False)
        model.draft.assert_called_once()
        self.assertEqual(model.draft.call_args.args[0].tolist(), [[1, 2, 10, 11, 12, 77]])
        self.assertEqual(result[3].tolist(), [[77, 88]])
        model.csgb_terminal_commit_callback.assert_not_called()

    def test_default_verifier_path_unchanged(self):
        result, model, _, _, _ = self.verify_case(stop=None)
        model.draft.assert_called_once()
        model.csgb_terminal_commit_callback.assert_not_called()
        self.assertEqual(result[1:3], ([10, 11, 12], 2))

    def test_eos_path_is_trimmed_before_kv_and_acceptance_accounting(self):
        result, model, length, events, _ = self.verify_case(tokens=(10, 99, 12), eos=99)
        self.assertEqual(result[0].tolist(), [[1, 2, 10, 99]])
        self.assertEqual(result[1:3], ([10, 99], 1))
        self.assertEqual(length.item(), 4)
        self.assertEqual(events[-1], ("terminal", [[1, 2, 10, 99]], 4))
        model.draft.assert_not_called()

    def test_eos_root_is_one_committed_token(self):
        result, _, length, _, _ = self.verify_case(tokens=(99, 11, 12), eos=99)
        self.assertEqual(result[0].tolist(), [[1, 2, 99]])
        self.assertEqual(result[1:3], ([99], 0))
        self.assertEqual(length.item(), 3)


@unittest.skipIf(torch is None, "CPU integration requires PyTorch")
class TerminalGenerateCPUTests(unittest.TestCase):
    def run_generation(self, rounds, maximum, enabled=True, eos=99):
        remaining = list(rounds)
        calls = []
        callbacks = []

        def initial_cache(*args):
            cache = torch.zeros((1, 1, 5, 1))
            return [(cache, cache)], [], torch.tensor([5])

        def fake_verify(ids, *args, **kwargs):
            candidate = remaining.pop(0)
            calls.append(kwargs)
            candidate = candidate[:candidate_prefix_through_eos(
                candidate, kwargs.get("terminal_eos_token_id"))]
            ids = torch.cat((ids, torch.tensor([candidate])), dim=1)
            stopped = finish_terminal_commit(model, ids, kwargs.get("stop_after_commit"))
            next_tree = (None, None, None, None) if stopped else (
                torch.tensor([[200]]), torch.tensor([5]), torch.ones((1, 1)), torch.tensor([0]))
            return (ids, candidate, len(candidate) - 1, *next_tree)

        namespace = dict(
            torch=torch, reset_bridge_runtime=lambda model: None,
            committed_generation_should_stop=committed_generation_should_stop,
            initialize_past_key_values=initial_cache,
            tree_decoding=lambda *args: (None, None, None), verify=fake_verify,
            print_newly_accepted_tokens=lambda *args, **kwargs: [], datetime=datetime,
        )
        generate = load_function(ROOT / "classic/model_classic.py", "spgenerate", namespace, "SPModel")

        class Model:
            def __call__(self, **kwargs):
                return (torch.tensor([[101]]), torch.tensor([6]), torch.ones((1, 1)),
                        torch.tensor([[100]]), torch.tensor([0]), None)

        model = Model()
        model.base_model = object()
        model.tokenizer = SimpleNamespace(eos_token_id=eos)
        model.csgb_skip_terminal_draft = enabled
        model.csgb_terminal_commit_callback = lambda ids: callbacks.append(ids.tolist())
        result = generate(model, torch.tensor([[1, 2]]), max_new_tokens=maximum, return_output_ids=True)
        return result, model, calls, callbacks, remaining

    def test_exact_hit_records_terminal_round_before_exit(self):
        result, model, calls, callbacks, remaining = self.run_generation([[10, 11], [12, 13], [14]], 4)
        self.assertEqual(model.timestep, 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(remaining, [[14]])
        self.assertEqual(result["raw_accept_length_list"], [1, 1])
        self.assertEqual(result["output_token_ids"], [10, 11, 12, 13])
        self.assertEqual(callbacks, [[[1, 2, 10, 11, 12, 13]]])

    def test_overshoot_preserves_raw_and_scored_accounting(self):
        result, model, _, callbacks, _ = self.run_generation([[10, 11, 12], [13, 14, 15]], 4)
        self.assertEqual(model.timestep, 2)
        self.assertEqual(result["raw_total_generated"], 6)
        self.assertEqual(result["total_generated"], 4)
        self.assertEqual(result["generation_overshoot"], 2)
        self.assertEqual(result["raw_accept_length_list"], [2, 2])
        self.assertEqual(result["accept_length_list"], [2, 0])
        self.assertEqual(callbacks, [[[1, 2, 10, 11, 12, 13, 14, 15]]])

    def test_maximum_one_does_not_enter_another_round(self):
        result, model, calls, callbacks, remaining = self.run_generation([[10], [11]], 1)
        self.assertEqual(model.timestep, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["output_token_ids"], [10])
        self.assertEqual(callbacks, [[[1, 2, 10]]])
        self.assertEqual(remaining, [[11]])

    def test_eos_excludes_later_accepted_tokens(self):
        result, model, _, callbacks, _ = self.run_generation([[10, 99, 12]], 8)
        self.assertEqual(model.timestep, 1)
        self.assertEqual(result["output_token_ids"], [10, 99])
        self.assertEqual(result["raw_total_generated"], 2)
        self.assertEqual(result["raw_accept_length_list"], [1])
        self.assertEqual(callbacks, [[[1, 2, 10, 99]]])

    def test_disabled_passes_no_new_kwargs_or_callback(self):
        result, _, calls, callbacks, _ = self.run_generation([[10, 11]], 2, enabled=False)
        self.assertEqual(calls, [{}])
        self.assertEqual(callbacks, [])
        self.assertEqual(result["total_generated"], 2)


if __name__ == "__main__":
    unittest.main()
