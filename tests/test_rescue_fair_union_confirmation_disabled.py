"""The unexecuted union launcher cannot touch any experiment/data path."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from rescue.fair_union_confirmation_data import protocol_template

ROOT=Path(__file__).resolve().parents[1]


class DisabledUnionLauncherTests(unittest.TestCase):
    def test_launcher_always_exits64_without_mutation(self):
        script=ROOT/'exp/draft_graph/run_fair_union_confirmation_20261020_v1.sh'
        with tempfile.TemporaryDirectory() as tmp:
            for args in ([],['--validate-only'],['--launch']):
                result=subprocess.run(['bash',str(script),*args],cwd=tmp,capture_output=True,text=True)
                self.assertEqual(result.returncode,64)
                self.assertIn('DISABLED',result.stderr)
                self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_candidate_template_exactly_matches_pre_access_factory(self):
        path=ROOT/'exp/draft_graph/fair_union_confirmation_20261020_preregistration.json'
        value=json.loads(path.read_text())
        self.assertEqual(value,protocol_template())
        self.assertTrue(value['is_template'])
        self.assertEqual(value['status'],'TEMPLATE_NOT_EXECUTED')


if __name__=='__main__':unittest.main()
