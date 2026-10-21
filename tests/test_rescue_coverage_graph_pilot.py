"""CPU protocol/helper tests; synthetic records are not experiment results."""
import ast
import copy
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rescue.config import SystemConfig
from rescue.coverage_graph_pilot import (EXPECTED, LENGTHS, MODES,
    coverage_snapshot, finish_owner_record, private_signature, profile_only,
    run, schedule, source_inventory)
from rescue.draft_graph_benchmark import record
from rescue.specextend_match_pilot import check_quality


def quality_record():
    trace = dict(round_index=0, prefix=4095, cap=2, proposed=2, accepted=1,
                 terminal=True, workset_tokens=512)
    return dict(generated=[3, 4], tokens=2, accepted=1, proposed=2, blocks=1,
        draft_forwards=2, selector_refreshes=1, trace=[trace],
        observed_rounds=[dict(selected=[1, 3], budget_features=[0.]*11+[.5],
                              draft=[3, 5], checks=[3, 4], accepted=1)],
        queries=[dict(sha256='query')], native_t=dict(sha256='target'),
        native_s=dict(sha256='draft'))


class CoveragePilotHelpers(unittest.TestCase):
    def test_registered22_request_matrix_and_reproducible_interleaving(self):
        self.assertEqual(sum(v for k, v in EXPECTED.items() if k != 'owners'), 22)
        self.assertEqual(EXPECTED['owners'], 4)
        self.assertEqual(EXPECTED['profile_requests'], 4)
        self.assertEqual(MODES, ('draft_graph', 'draft_coverage_graph'))
        self.assertEqual(LENGTHS, (4096, 8192))
        plan = schedule()
        self.assertEqual(plan, schedule())
        self.assertNotEqual(plan, schedule(1))
        self.assertEqual(len(plan), 8)
        self.assertEqual(len({(p['mode'], p['length'], p['repeat']) for p in plan}), 8)
        for offset in range(0, 8, 2):
            pair = plan[offset:offset+2]
            self.assertEqual({p['mode'] for p in pair}, set(MODES))
            self.assertEqual(len({(p['length'], p['repeat']) for p in pair}), 1)
            self.assertTrue(all(p['source_index'] == 303 for p in pair))

    def test_profile_switch_changes_only_profile_and_restores_on_failure(self):
        original = SystemConfig(selector='hierarchical', schedule='serial',
            draft_execution='cuda_graph', coverage_execution='cuda_graph')
        engine = SimpleNamespace(config=original)
        for fail in (False, True):
            try:
                with profile_only(engine):
                    self.assertTrue(engine.config.profile)
                    changed = asdict(engine.config); changed['profile'] = False
                    self.assertEqual(changed, asdict(original))
                    if fail:
                        raise ValueError('synthetic profile failure')
            except ValueError:
                self.assertTrue(fail)
            self.assertIs(engine.config, original)
        engine.config = SystemConfig(profile=True)
        with self.assertRaisesRegex(RuntimeError, 'profile=False'):
            with profile_only(engine):
                pass

    def test_accounted_service_excludes_profile_and_other_owner_lifetime(self):
        source = dict(owner_start=10., constructor_and_detach_seconds=1.,
            cold_prime_outer_seconds=2., warm_outer_seconds_sum=4.,
            prime_tokens=128, warm_tokens=256, profile_tokens=128,
            profile_outer_seconds_sum=999., profile_request_ids=['profile'])
        saved = copy.deepcopy(source)
        owner = finish_owner_record(source, .5, 2010.)
        self.assertEqual(source, saved)
        self.assertEqual(owner['accounted_service_seconds'], 7.5)
        self.assertEqual(owner['accounted_service_tokens'], 384)
        self.assertEqual(owner['accounted_service_tps'], 384/7.5)
        self.assertEqual(owner['owner_lifetime_wall_seconds'], 2000.)
        self.assertEqual(owner['profile_outer_seconds_sum'], 999.)
        self.assertTrue(owner['profile_excluded_from_service'])
        self.assertNotIn('owner_start', owner)
        self.assertNotIn('session_tps', owner)

    def test_coverage_snapshot_keeps_actual_shapes_and_private_identity(self):
        eager = coverage_snapshot(SimpleNamespace(coverage_runtime=None))
        self.assertEqual(eager['mode'], 'eager')
        self.assertEqual(eager['captures'], 0)
        runtime = SimpleNamespace(max_entries=64,
            _entries={(((64, 87), (87, 1), 'torch.float32', 'cuda:0'), 28, 16, 1.): object()},
            stats=lambda: dict(mode='cuda_graph', entries=1, captures=1,
                replays=3, capture_host_seconds=.2, eager_calls=0, input_copies=3))
        engine = SimpleNamespace(coverage_runtime=runtime)
        snapshot = coverage_snapshot(engine)
        self.assertEqual(snapshot['owner_identity'], id(runtime))
        self.assertEqual(snapshot['shape_keys'], list(runtime._entries))
        self.assertEqual(json.loads(json.dumps(snapshot))['entries'], 1)
        with patch('rescue.coverage_graph_pilot.owner_signature', return_value={'native': 123}):
            self.assertEqual(private_signature(engine),
                dict(native=123, coverage_runtime_identity=id(runtime)))

    def test_source_inventory_freezes_all_rescue_and_both_legacy_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ['rescue/a.py', 'rescue/b.py',
                'exp/attention_horizon/selection.py', 'exp/selective_kv/bridge.py',
                'tests/test_rescue_coverage_graph_pilot.py']
            for name in names:
                path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# synthetic source\n')
            inventory = source_inventory(root)
            self.assertEqual(set(inventory), set(names))
            self.assertTrue(all(len(value) == 64 for value in inventory.values()))
            (root/names[-1]).unlink()
            with self.assertRaisesRegex(RuntimeError, 'Missing frozen'):
                source_inventory(root)

    def test_quality_gate_rejects_token_selection_feature_query_and_all_kv_drift(self):
        base = quality_record(); check_quality(base, copy.deepcopy(base))
        mutations = [
            lambda row: row['generated'].__setitem__(0, 99),
            lambda row: row['trace'][0].__setitem__('cap', 4),
            lambda row: row['observed_rounds'][0]['selected'].__setitem__(0, 2),
            lambda row: row['observed_rounds'][0]['budget_features'].__setitem__(-1, 0.),
            lambda row: row['queries'][0].__setitem__('sha256', 'wrong'),
            lambda row: row['native_t'].__setitem__('sha256', 'wrong'),
            lambda row: row['native_s'].__setitem__('sha256', 'wrong')]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                row = copy.deepcopy(base); mutate(row)
                with self.assertRaises(RuntimeError):
                    check_quality(base, row)

    def test_failed_uuid_gate_preserves_raw_evidence_not_a_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp); counts = dict(timing_requests=0); seen = set()
            record(output, 'runtime.jsonl', {'request_id': 'one'}, counts, 'timing_requests', seen)
            with self.assertRaisesRegex(RuntimeError, 'duplicate'):
                record(output, 'runtime.jsonl', {'request_id': 'one'}, counts, 'timing_requests', seen)
            self.assertEqual(len((output/'runtime.jsonl').read_text().splitlines()), 2)
            self.assertFalse((output/'COMPLETE.json').exists())

    def test_runner_is_fresh_only_profiles_after_all_timing_and_no_silent_shape_gate(self):
        source = inspect.getsource(run); tree = ast.parse(source)
        assigned = {n.id for n in ast.walk(tree)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        self.assertFalse(assigned & {'record', 'context', 'source_inventory', 'schedule', 'write', 'append'})
        self.assertIn('require(not output.exists()', source)
        self.assertIn("counts['timing_requests']==8", source)
        self.assertLess(source.index("counts['timing_requests']==8"), source.index('with profile_only(engine)'))
        self.assertLess(source.index('with profile_only(engine)'), source.index('for key,engine in active.items()'))
        self.assertIn("phase='profile'", source)
        self.assertIn('performance_eligible=False', source)
        self.assertIn('profile_outer_seconds=end-start', source)
        self.assertNotIn("['captures']==0", source)
        self.assertNotIn("['captures'] == 0", source)
        self.assertIn("exact_selected_features_cap_draft_checks_native_queries=True", source)


if __name__ == '__main__':
    unittest.main()
