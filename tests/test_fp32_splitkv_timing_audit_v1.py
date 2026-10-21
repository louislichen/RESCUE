"""Synthetic CPU statistics checks, not new performance/model measurements."""
import copy
import importlib.util
from pathlib import Path
import unittest

source = Path(__file__).resolve().parent/'audit.py'
if not source.is_file():
    source = Path(__file__).resolve().parents[1]/'exp/round_overhead/fp32_splitkv_timing_audit_v1.py'
spec = importlib.util.spec_from_file_location('_independent_saved_timing_audit', source)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def synthetic():
    speeds = dict(original_rescue=90., candidate_rescue=120., original_retrieval=100., candidate_retrieval=100.)
    return [dict(length=n, arm_id=arm, source_index=d, repeat=repeat, tokens=128,
                 outer_seconds=128/speeds[arm], accepted=64, proposed=96, blocks=32)
            for n in audit.LENGTHS for arm in audit.ARMS for d in audit.DOCS for repeat in (0, 1)]


class StatisticsTests(unittest.TestCase):
    def test_identical_documents_preserve_pooled_ratios_and_registered_gate(self):
        rows = synthetic(); before = copy.deepcopy(rows)
        result = audit.statistics(rows)
        self.assertEqual(rows, before)
        self.assertEqual(result['bootstrap_draws'], 10000)
        self.assertEqual(result['bootstrap_seed'], 20261031)
        self.assertTrue(result['all_six_primary_numeric_gates'])
        self.assertEqual(result['document_order'], list(audit.DOCS))
        for point in result['lengths']:
            primary = point['contrasts'][0]
            self.assertAlmostEqual(primary['relative_tps_percent'], 20.)
            self.assertTrue(all(abs(v-20.) < 1e-10 for v in primary['paired95_ci_percent']))
            self.assertEqual(point['pooled']['candidate_rescue']['accepted_per_round'], 2.)
            self.assertAlmostEqual(point['pooled']['candidate_rescue']['acceptance_rate'], 2/3)

    def test_missing_repeat_refused_not_silently_unpaired_or_imputed(self):
        rows = synthetic(); rows.pop()
        with self.assertRaisesRegex(ValueError, 'paired2 repeats'):
            audit.statistics(rows)

    def test_equal_primary_speed_cannot_be_goal_gate_pass(self):
        rows = synthetic()
        for row in rows:
            if row['arm_id'] == 'candidate_rescue':
                row['outer_seconds'] = 128/100.
        result = audit.statistics(rows)
        self.assertFalse(result['all_six_primary_numeric_gates'])
        self.assertTrue(all(abs(p['contrasts'][0]['relative_tps_percent']) < 1e-10 for p in result['lengths']))


if __name__ == '__main__':
    unittest.main()
