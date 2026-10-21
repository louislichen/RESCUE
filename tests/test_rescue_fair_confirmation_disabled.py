"""The superseded short-horizon draft must never start a scientific run."""
from pathlib import Path
import subprocess
import unittest


class UnmatchedHorizonLaunchGuard(unittest.TestCase):
    def test_candidate_stops_before_remote_roots_or_confirmation_data_access(self):
        source = Path(__file__).resolve().parents[1]
        script = source / 'exp/draft_graph/run_fair_confirmation_20261020_v1.sh'
        result = subprocess.run(['bash', str(script)], cwd=source,
            capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 64)
        self.assertEqual(result.stdout, '')
        self.assertIn('DISABLED: pending horizon-matched baseline selection', result.stderr)


if __name__ == '__main__':
    unittest.main()
