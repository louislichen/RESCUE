"""Read-only horizon snapshot/terminal export; writes only a stdout archive.

Live snapshots preserve complete JSONL prefixes and explicitly are PARTIAL.
Terminal mode additionally requires exit0, all2540 requests and no live handle.
No SSH, GPU work, original corpus access or remote filesystem writes occur here.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile


ROOT = Path('/mnt/8T/cl/rescue_experiment_cache/baseline_horizon_20261020_v1')
SESSION = 'rescue_baseline_horizon_v1'
COUNTS = dict(reference_requests=32, prime_requests=76, timing_requests=2432, owners=76)
TARGET = Path('/mnt/8T/cl/models/Qwen2-7B-Instruct')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def handles(root):
    session = subprocess.run(['tmux', 'has-session', '-t', SESSION], capture_output=True)
    processes = subprocess.run(['ps', '-eo', 'pid,args'], capture_output=True,
                               check=True, text=True).stdout.splitlines()
    active = [line.strip() for line in processes if 'python' in line and
              '-m rescue.baseline_horizon ' in line and str(root/'results') in line]
    return dict(tmux_present=session.returncode == 0, actual_python_processes=active)


def terminal(root):
    done = json.loads((root/'results/COMPLETE.json').read_text())
    exit_receipt = json.loads((root/'PIPELINE_EXIT.json').read_text())
    require(type(exit_receipt.get('exit_code')) is int and exit_receipt['exit_code'] == 0,
            'Pipeline has not exited successfully')
    require(done.get('passed') is True and done.get('counts') == COUNTS and
            done.get('logical_requests') == 2540 and done.get('owner_closed') == 76,
            'Missing full2540/76 execution receipt')
    require(not any((root/p).exists() for p in ('FAILURE.json', 'results/FAILURE.json')),
            'Failure marker present')
    state = handles(root)
    require(not state['tmux_present'] and not state['actual_python_processes'],
            'Scientific tmux/Python handle still exists')
    return state


def inventory(root):
    result = [p for name in ('after', 'results') for p in (root/name).rglob('*') if p.is_file()]
    names = ('LAUNCH_MANIFEST.json', 'DEPENDENCY_RECEIPT.json', 'RUN_STARTED.json',
             'PIPELINE_EXIT.json', 'FAILURE.json', 'pipeline.log', 'gpu_tests.log', 'benchmark.log')
    result.extend(root/name for name in names if (root/name).is_file())
    require(all(not p.is_symlink() and p.resolve().is_relative_to(root.resolve()) for p in result),
            'Unsafe artifact path')
    return sorted(set(result))


def export(root, *, complete=False):
    require(root.is_dir() and not root.is_symlink(), 'Missing exact experiment root')
    started = datetime.now(timezone.utc).isoformat()
    before = terminal(root) if complete else handles(root)
    original = inventory(root)
    payloads, cuts = {}, {}
    for path in original:
        name = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        if name.endswith('.jsonl'):
            cut = len(raw) if raw.endswith(b'\n') else raw.rfind(b'\n')+1
            require(not complete or cut == len(raw), 'Terminal JSONL incomplete tail: '+name)
            cuts[name] = len(raw)-cut
            raw = raw[:cut]
            for line in raw.splitlines():
                require(bool(line.strip()), 'Blank JSONL row: '+name)
                json.loads(line)
        elif name.endswith('.json'):
            json.loads(raw)
        payloads[name] = raw
    protocol = json.loads(payloads['results/protocol.json'])
    teacher = TARGET
    require(protocol.get('args', {}).get('teacher') == str(teacher), 'Wrong frozen Target root')
    external = {}
    for filename in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
        path = teacher/filename
        raw = path.read_bytes()
        require(protocol['assets_sha256'].get(str(path)) == hashlib.sha256(raw).hexdigest(),
                'Frozen tokenizer metadata changed')
        name = 'asset_metadata/Qwen2-7B-Instruct/'+filename
        payloads[name] = raw
        external[name] = path
    after = terminal(root) if complete else handles(root)
    if complete:
        require(original == inventory(root), 'Terminal artifact set changed during collection')
    for path in original:
        name = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        if not complete and (name.endswith('.jsonl') or name.endswith('.log')):
            require(raw.startswith(payloads[name]), 'Live artifact rewrote its collected prefix: '+name)
        else:
            require(raw == payloads[name], 'Artifact changed during collection: '+name)
    require(all(p.read_bytes() == payloads[n] for n, p in external.items()),
            'Tokenizer metadata changed during collection')
    files = {name: dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
             for name, raw in payloads.items()}
    for name, path in external.items():
        files[name]['read_only_external_source'] = str(path)
    counts = {key: len(payloads.get('results/'+filename, b'').splitlines()) for key, filename in (
        ('reference_requests', 'reference.jsonl'), ('prime_requests', 'prime.jsonl'),
        ('timing_requests', 'runtime.jsonl'), ('owners', 'owners.jsonl'))}
    require(not complete or counts == COUNTS, 'Terminal raw counts differ from COMPLETE')
    receipt = dict(status='TERMINAL_COLLECTED_NOT_YET_INDEPENDENTLY_AUDITED' if complete else 'PARTIAL_READ_ONLY_PREFIX_SNAPSHOT',
        read_only=True, root=str(root), started_at_utc=started,
        finished_at_utc=datetime.now(timezone.utc).isoformat(), counts=counts, expected_counts=COUNTS,
        process_and_tmux_absent=complete, pipeline_exit_code=0 if complete else None,
        handles_before=before, handles_after=after, files=files,
        complete_file_set_rechecked=complete, all_file_bytes_rechecked=complete,
        collected_live_prefixes_rechecked=not complete, dropped_incomplete_jsonl_tail_bytes=cuts,
        snapshot_is_atomic=complete, goal_complete=False,
        note='No original corpus/weights access. A live multi-file snapshot is not atomic or final; no authoritative selection or completion from collection alone.')
    receipt_name = 'TERMINAL_COLLECTION.json' if complete else 'SNAPSHOT_COLLECTION.json'
    payloads[receipt_name] = (json.dumps(receipt, indent=2)+'\n').encode()
    return payloads, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--terminal', action='store_true')
    args = parser.parse_args()
    payloads, _ = export(ROOT, complete=args.terminal)
    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
        for name, raw in payloads.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(raw), 0o444
            archive.addfile(info, io.BytesIO(raw))


if __name__ == '__main__':
    main()
