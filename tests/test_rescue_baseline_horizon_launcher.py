"""CPU-only launcher dry-validation tests with explicitly synthetic evidence.

Never invoke the normal launcher branch.  No SSH, real GPU utility, model, or
scientific dataset is used; the tiny local validator stub tests shell ordering,
not the scientific validity of a stage1 receipt.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT / 'exp/draft_graph/run_baseline_horizon_20261020_v1.sh'
REMOTE_ROOT = '/mnt/8T/cl/rescue_experiment_cache/baseline_horizon_20261020_v1'
REMOTE_PYTHON = '/mnt/8T/cl/rescue_experiment_cache/specextend_match_20261020_v1/venv/bin/python'
GPU_UUID = 'GPU-9db4f7b5-1447-9a2a-5c1b-80379d909fd9'


def environment(root):
    run = root / 'run'; source = run / 'after'; package = source / 'rescue'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    stub = '''from pathlib import Path
def validate_baseline_freeze(results, analysis, verified, manifest):
    base = results.parent
    (base / 'SYNTHETIC_VALIDATOR_CALLED').write_text('shell-order test, not scientific proof')
    if not (base / 'SYNTHETIC_READY').exists():
        raise RuntimeError('SYNTHETIC stage1 gate rejected')
    return {'selection_freeze_sha256': 'a' * 64}
'''
    (package / 'fair_confirmation_data.py').write_text(stub)
    evidence = run / 'stage1_evidence'; evidence.mkdir()
    (evidence / 'SYNTHETIC_READY').write_text('synthetic fixture only')
    inventory = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in package.glob('*.py')}
    (run / 'LAUNCH_MANIFEST.json').write_text(json.dumps(dict(source_sha256=inventory)))
    fakebin = root / 'fakebin'; fakebin.mkdir()
    # Recording benign substitutes ensure the validate-only path does not lock,
    # run tests, query jobs, or invoke tmux. UUID inspection is read-only too.
    gpu = fakebin / 'nvidia-smi'
    gpu.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> ' + str(root / 'gpu_calls') + '\n'
        'case "$1" in --query-gpu=uuid) printf "%s\\n" "' + GPU_UUID + '";; *) exit 99;; esac\n')
    gpu.chmod(0o755)
    for name in ('tmux', 'flock'):
        tool = fakebin / name
        tool.write_text('#!/bin/sh\nprintf "%s\\n" "unexpected ' + name + '" >&2\nexit 99\n')
        tool.chmod(0o755)
    source_text = LAUNCHER.read_text().replace(REMOTE_ROOT, str(run)).replace(REMOTE_PYTHON, sys.executable)
    script = root / 'synthetic_validate_launcher.sh'; script.write_text(source_text)
    env = dict(os.environ, PATH=str(fakebin) + os.pathsep + os.environ['PATH'])
    return run, script, env


def validate(script, env, args=('--validate-only',)):
    # Only the literal read-only flag or invalid CLI syntax is allowed here.
    assert tuple(args) in (('--validate-only',), ('--unknown',), ('--validate-only', '--extra'))
    return subprocess.run(['bash', str(script), *args], env=env, capture_output=True,
                          text=True, timeout=20)


class HorizonLauncherTests(unittest.TestCase):
    def test_real_launcher_parses_and_excludes_server_matplotlib_report_tests(self):
        parsed = subprocess.run(['bash', '-n', str(LAUNCHER)], capture_output=True, text=True, timeout=10)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        source = LAUNCHER.read_text()
        matches = re.findall(r'for test_name in (.*?); do', source)
        self.assertEqual(len(matches), 1)
        tests = matches[0].split()
        self.assertIn('test_rescue_baseline_horizon_data.py', tests)
        self.assertIn('test_rescue_baseline_horizon.py', tests)
        self.assertTrue(all(name.startswith('test_rescue_') and name.endswith('.py') for name in tests))
        self.assertFalse(any('report' in name or 'plot' in name or 'launcher' in name for name in tests))
        for flag in ('--train-file', '--opt-confirmation-file', '--arch-confirmation-file', '--prime-file',
                     '--stage1-results', '--stage1-analysis', '--stage1-verified', '--stage1-manifest', '--output'):
            self.assertIn(flag, source)
        self.assertIn('test ! -e "$run_root/RUN_STARTED.json"', source)
        self.assertIn('test ! -e "$run_root/results"', source)

    def test_validate_only_calls_source_and_stage1_gates_without_mutating_run_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run, script, env = environment(root)
            result = validate(script, env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('FROZEN_SOURCE_VERIFIED', result.stdout)
            self.assertIn('STAGE1_COMPLETE_538_76_VERIFIED', result.stdout)
            self.assertIn('VALIDATED_ONLY', result.stdout)
            for name in ('RUN_STARTED.json', 'PIPELINE_EXIT.json', 'pipeline.lock', 'results',
                         'gpu_tests.log', 'benchmark.log'):
                self.assertFalse((run / name).exists(), name)
            self.assertEqual((root / 'gpu_calls').read_text().splitlines(),
                             ['--query-gpu=uuid --format=csv,noheader'])

    def test_changed_source_fails_before_stage1_or_gpu_and_creates_no_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run, script, env = environment(root)
            (run / 'after/rescue/__init__.py').write_text('# changed synthetic bytes\n')
            result = validate(script, env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Staged source changed', result.stderr)
            self.assertFalse((run / 'stage1_evidence/SYNTHETIC_VALIDATOR_CALLED').exists())
            self.assertFalse((root / 'gpu_calls').exists())
            self.assertFalse((run / 'pipeline.lock').exists())

    def test_stage1_gate_failure_prevents_gpu_and_run_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run, script, env = environment(root)
            (run / 'stage1_evidence/SYNTHETIC_READY').unlink()
            result = validate(script, env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('stage1 gate rejected', result.stderr)
            self.assertFalse((root / 'gpu_calls').exists())
            self.assertFalse((run / 'RUN_STARTED.json').exists())

    def test_dirty_root_and_unknown_cli_fail_before_preflight(self):
        for state in ('marker', 'results', 'unknown_cli', 'extra_cli'):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); run, script, env = environment(root)
                if state == 'marker': (run / 'RUN_STARTED.json').write_text('synthetic prior run')
                if state == 'results': (run / 'results').mkdir()
                args = ('--unknown',) if state == 'unknown_cli' else (
                    ('--validate-only', '--extra') if state == 'extra_cli' else ('--validate-only',))
                result = validate(script, env, args)
                self.assertNotEqual(result.returncode, 0)
                if state.endswith('cli'): self.assertEqual(result.returncode, 64)
                self.assertFalse((run / 'stage1_evidence/SYNTHETIC_VALIDATOR_CALLED').exists())
                self.assertFalse((root / 'gpu_calls').exists())


if __name__ == '__main__':
    unittest.main()
