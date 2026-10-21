"""CPU/toy tests for the no-training paired execution runner."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import uuid

import torch

from rescue import round_overhead_run as run


ROOT = Path(__file__).resolve().parents[1]


def toy_row():
    return dict(generated=[7, 8], tokens=2, accepted=1, proposed=2, blocks=1,
        draft_forwards=2, selector_refreshes=1,
        trace=[dict(round_index=0, prefix=4095, cap=2, proposed=2, accepted=1,
                    terminal=True, workset_tokens=512)])


class RoundOverheadRunTests(unittest.TestCase):
    def test_fixed_balanced_adjacent_pair_schedule_has_72_unique_requests(self):
        plan = run.timing_schedule(619)
        self.assertEqual(plan, run.timing_schedule(619))
        self.assertNotEqual(plan, run.timing_schedule(620))
        self.assertEqual(len(plan), 72)
        keys = {(r['source_index'], r['length'], r['repeat'], r['mode']) for r in plan}
        expected = {(i, n, repeat, mode) for i in run.SOURCE_INDICES for n in run.LENGTHS
                    for repeat in range(3) for mode in ('before', 'after')}
        self.assertEqual(keys, expected)
        first_before = 0
        for pair in range(36):
            a, b = plan[2*pair:2*pair+2]
            self.assertEqual(a['pair_index'], pair)
            self.assertEqual(b['pair_index'], pair)
            self.assertEqual({a['mode'], b['mode']}, {'before', 'after'})
            self.assertEqual({k: v for k, v in a.items() if k != 'mode'},
                             {k: v for k, v in b.items() if k != 'mode'})
            first_before += a['mode'] == 'before'
        self.assertEqual(first_before, 18)

    def test_natural_nested_context_and_document_registration_fail_closed(self):
        rows = [dict(source_index=i, split='confirmation', document_sha256=f'{i:064x}',
                     prefix_ids=[1, 2], suffix_ids=[3], document_ids=list(range(8192)))
                for i in run.SOURCE_INDICES]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'data.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            documents = run.read_documents([path])
            self.assertEqual(set(documents), set(run.SOURCE_INDICES))
            short, long = [run.context(rows[0], n) for n in run.LENGTHS]
            self.assertEqual((len(short), len(long)), (4096, 8192))
            self.assertEqual(short[:-1], long[:4095])
            self.assertEqual(short[-1:], long[-1:])
            for changed in (rows[:-1], rows+[rows[0]],
                            [dict(rows[0], source_index=999)]+rows[1:],
                            [dict(rows[0], split='test')]+rows[1:]):
                path.write_text(''.join(json.dumps(r)+'\n' for r in changed))
                with self.assertRaises(RuntimeError):
                    run.read_documents([path])
        with self.assertRaises(RuntimeError):
            run.context(dict(prefix_ids=[1], suffix_ids=[2], document_ids=[3]), 4096)

    def test_canonical_and_pair_gates_reject_quality_and_trajectory_changes(self):
        row = toy_row(); run.check_reference(row, [7, 8])
        self.assertTrue(row['output_agrees'])
        run.check_pair(row, copy.deepcopy(row))
        for changed in (dict(row, generated=[7, 9]), dict(row, proposed=3), dict(row, blocks=2)):
            with self.assertRaises(RuntimeError):
                run.check_reference(changed, [7, 8])
        for key, value in (('cap', 4), ('workset_tokens', 1024), ('prefix', 4096)):
            changed = copy.deepcopy(row); changed['trace'][0][key] = value
            with self.assertRaises(RuntimeError):
                run.check_pair(row, changed)
        changed = dict(row, draft_forwards=3)
        with self.assertRaises(RuntimeError):
            run.check_pair(row, changed)

    def test_correctness_observer_captures_exact_causal_fields_and_actual_nonzero_summary(self):
        observer = run.CorrectnessObserver()
        packet = dict(round_index=0, prefix=4095, progress=0, remaining=128,
                      selected=torch.tensor([0, 4, 7]), budget_features=torch.zeros(12),
                      draft=[7, 9], checks=[7, 8, 2], accepted=1)
        observer(packet)
        self.assertFalse(observer.nonzero_injection)
        packet['budget_features'][-1] = .125
        observer(packet)
        self.assertTrue(observer.nonzero_injection)
        self.assertEqual(observer.rounds[-1]['selected'], [0, 4, 7])
        self.assertEqual(observer.rounds[-1]['budget_features'][-1], .125)
        packet['budget_features'][0] = float('nan')
        with self.assertRaises(RuntimeError):
            observer(packet)

    def test_alias_package_uses_isolated_relative_modules(self):
        name = '_rescue_runner_toy_'+uuid.uuid4().hex
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp)/'rescue'; package.mkdir()
            (package/'__init__.py').write_text('marker="before"\n')
            for part in ('engine', 'config', 'query_mapper', 'step_budget', 'features', 'selection'):
                (package/f'{part}.py').write_text('from . import marker\nvalue=marker\n')
            try:
                modules = run.alias_package(Path(tmp), name)
                for part, module in modules.items():
                    self.assertEqual(module.value, 'before')
                    self.assertEqual(Path(module.__file__).resolve(), (package/f'{part}.py').resolve())
                    self.assertTrue(module.__name__.startswith(name+'.'))
                self.assertEqual(run.alias_package(Path(tmp), name), modules)
                with tempfile.TemporaryDirectory() as other:
                    alternate = Path(other)/'rescue'; alternate.mkdir()
                    (alternate/'__init__.py').write_text('')
                    with self.assertRaises(RuntimeError):
                        run.alias_package(Path(other), name)
            finally:
                for key in list(sys.modules):
                    if key == name or key.startswith(name+'.'):
                        del sys.modules[key]

    def test_pinned_sources_allow_only_the_registered_treatment_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            before, after = Path(tmp)/'before', Path(tmp)/'after'
            names = ['rescue/'+name for name in run.BASELINE_PINS]
            names += ['rescue/attention.py', *run.SHARED_EXP_FILES]
            for name in names:
                original = subprocess.check_output(['git', 'show', run.BASELINE_COMMIT+':'+name], cwd=ROOT)
                for root in (before, after):
                    path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(original)
            for name in run.BASELINE_PINS:
                with (after/'rescue'/name).open('a') as handle:
                    handle.write('\n# execution treatment\n')
            inventories = run.validate_sources(before, after)
            self.assertEqual(len(inventories['before']), len(names))
            with (after/'rescue/attention.py').open('a') as handle:
                handle.write('\n# unauthorized change\n')
            with self.assertRaisesRegex(RuntimeError, 'Unexpected non-treatment'):
                run.validate_sources(before, after)

    def test_writes_are_exclusive_and_partial_jsonl_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'protocol.json'
            run.write(path, {'frozen': True})
            with self.assertRaises(FileExistsError):
                run.write(path, {'frozen': False})
            self.assertEqual(json.loads(path.read_text()), {'frozen': True})
            partial = Path(tmp)/'runtime.jsonl'
            run.append(partial, {'mode': 'before'}); run.append(partial, {'mode': 'after'})
            self.assertEqual([json.loads(l) for l in partial.read_text().splitlines()],
                             [{'mode': 'before'}, {'mode': 'after'}])

    def test_existing_output_is_not_reused_or_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'output'; output.mkdir()
            marker = output/'partial.jsonl'; marker.write_text('saved\n')
            with self.assertRaisesRegex(RuntimeError, 'Fresh output required'):
                run.run_experiment(SimpleNamespace(output=output))
            self.assertEqual(marker.read_text(), 'saved\n')
            self.assertFalse((output/'FAILURE.json').exists())

    def test_output_cannot_be_created_inside_baseline_or_model_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            protected = Path(tmp)/'protected'; protected.mkdir()
            for name in ('baseline_root', 'teacher', 'student'):
                output = protected/'new_output'
                args = SimpleNamespace(output=output, **{name: protected})
                with self.assertRaisesRegex(RuntimeError, 'inside baseline or base-model'):
                    run.run_experiment(args)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
