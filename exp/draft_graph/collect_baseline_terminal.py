"""Read-only terminal baseline archive to stdout; no remote filesystem writes."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile


ROOT = Path('/mnt/8T/cl/rescue_experiment_cache/baseline_selection_20261020_v1')
SESSION = 'rescue_baseline_selection_v1'
COUNTS = dict(reference_requests=6, prime_requests=76, timing_requests=456, owners=76)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def terminal():
    require(ROOT.is_dir(), 'Missing exact experiment root')
    done = json.loads((ROOT / 'results/COMPLETE.json').read_text())
    exit_receipt = json.loads((ROOT / 'PIPELINE_EXIT.json').read_text())
    require(type(exit_receipt.get('exit_code')) is int and exit_receipt['exit_code'] == 0,
            'Pipeline has not exited successfully')
    require(done.get('passed') is True and done.get('counts') == COUNTS and
            done.get('logical_requests') == 538 and done.get('owner_closed') == 76,
            'Missing full538/76 execution receipt')
    require(not (ROOT / 'results/FAILURE.json').exists(), 'Failure marker present')
    session = subprocess.run(['tmux', 'has-session', '-t', SESSION], capture_output=True)
    require(session.returncode != 0, 'Scientific tmux session still exists')
    processes = subprocess.run(['ps', '-eo', 'pid,args'], capture_output=True,
                               check=True, text=True).stdout.splitlines()
    active = [line for line in processes if 'python' in line and
              ('-m rescue.baseline_selection' in line) and str(ROOT / 'results') in line]
    require(not active, 'Scientific Python process still exists')
    return done, exit_receipt


def paths():
    result = [p for name in ('after', 'results') for p in (ROOT / name).rglob('*') if p.is_file()]
    result.extend(ROOT / n for n in ('LAUNCH_MANIFEST.json', 'DEPENDENCY_RECEIPT.json',
        'RUN_STARTED.json', 'PIPELINE_EXIT.json', 'pipeline.log', 'gpu_tests.log', 'benchmark.log'))
    require(all(p.is_file() and not p.is_symlink() for p in result), 'Missing/unsafe terminal artifact')
    return sorted(set(result))


started = datetime.now(timezone.utc).isoformat()
terminal()
original = paths()
payloads = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in original}
protocol = json.loads(payloads['results/protocol.json'])
asset_sources = {}
teacher = Path('/mnt/8T/cl/models/Qwen2-7B-Instruct')
require(protocol.get('args', {}).get('teacher') == str(teacher), 'Unexpected frozen Target asset root')
for filename in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
    path = teacher / filename
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    require(protocol['assets_sha256'].get(str(path)) == digest, 'Tokenizer asset changed from frozen protocol')
    name = 'asset_metadata/Qwen2-7B-Instruct/' + filename
    payloads[name] = payload
    asset_sources[name] = path
for name, payload in payloads.items():
    if name.endswith('.jsonl'):
        require(payload.endswith(b'\n'), 'Terminal JSONL incomplete tail: ' + name)
        for line in payload.splitlines():
            if line.strip():
                json.loads(line)
    elif name.endswith('.json'):
        json.loads(payload)
terminal()
require(original == paths(), 'Terminal file set changed during collection')
require(all(p.read_bytes() == payloads[p.relative_to(ROOT).as_posix()] for p in original),
        'Terminal artifacts changed during collection')
require(all(path.read_bytes() == payloads[name] for name, path in asset_sources.items()),
        'Tokenizer metadata changed during collection')
inventory = {name: dict(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
             for name, payload in payloads.items()}
for name, path in asset_sources.items():
    inventory[name]['read_only_external_source'] = str(path)
receipt = dict(status='TERMINAL_COLLECTED_NOT_YET_INDEPENDENTLY_AUDITED', read_only=True,
    root=str(ROOT), started_at_utc=started,
    finished_at_utc=datetime.now(timezone.utc).isoformat(),
    process_and_tmux_absent=True, pipeline_exit_code=0,
    complete_file_set_rechecked=True, all_file_bytes_rechecked=True,
    counts=COUNTS, files=inventory,
    supplemental_tokenizer_metadata='Only three frozen JSON files, no weights or new experiment data; permits independent offline EOS verification',
    note='No final scientific verification from collection alone; strict analysis, figures and independent raw audit remain required')
payloads['TERMINAL_COLLECTION.json'] = (json.dumps(receipt, indent=2) + '\n').encode()
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
    for name, payload in payloads.items():
        info = tarfile.TarInfo(name)
        info.size, info.mode = len(payload), 0o444
        archive.addfile(info, io.BytesIO(payload))
