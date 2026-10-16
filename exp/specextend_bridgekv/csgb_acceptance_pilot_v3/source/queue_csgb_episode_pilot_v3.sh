#!/usr/bin/env bash
set -euo pipefail
root=/storage/dzj/spec_repro/CSGB_episode_pilot_20261016_v3
py=/storage/dzj/spec_repro/venv/bin/python
gpu_uuid=GPU-0dab56d0-2236-4c0e-fc96-fe28081dba9b
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$root/specextend:$root/specextend/tests:$root"
cd "$root"
exec 9>"$root/queue.lock"
flock -n 9
for output in smoke smoke_analysis pilot analysis preflight_never_created; do
  test ! -e "$root/$output"
  test ! -L "$root/$output"
done
sha256sum -c stage.sha256
# BEGIN TERMINAL HARDWARE RECOVERY GUARD
assert_terminal_recovery_safe() {
  "$py" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess

def validate_snapshot(s):
    expected={'schema':1,'kind':'csgb_episode_hardware_failure_v1',
        'old_root':'/storage/dzj/spec_repro/CSGB_episode_pilot_20261016_v1',
        'unused_root':'/storage/dzj/spec_repro/CSGB_episode_pilot_20261016_v2',
        'pipeline_log_sha256':'0098cbc032fad5923b22a3efd2027a24899d5cf5e4b55232055598609808070f',
        'terminal_error':'Physical GPU contention/identity failure; no process will be interrupted',
        'predecessor_stage_sha256':'2c8bcee2bee4dbfdb30ef3c4c360906e7629fcf0172cd3ab91f1b08bf5471c89',
        'predecessor_protocol_sha256':'2a7d1bfde8127d9eac28d88b3805accf8f2914e5d8ffc3f3d5c39c515f24aba1',
        'registered_pids_absent':[1302918,1310503,1341898,1337237],
        'old_tmux_session':'csgb_episode_pilot_v1','old_tmux_absent':True,'old_outputs_absent':True,
        'unused_never_started':True,'unused_outputs_absent':True,'unused_queue_lock_absent':True,
        'unused_cancellation_receipt_absent':True,'unused_tmux_absent':True,'hardware_failure_only':True,
        'algorithm_results_present':False,'no_signals_sent':True}
    if any(s.get(k)!=v or type(s.get(k)) is not type(v) for k,v in expected.items()):
        raise RuntimeError('Failure snapshot does not prove this exact hardware-only recovery')
    for key in ('unused_stage_sha256','unused_protocol_sha256'):
        value=s.get(key)
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise RuntimeError('Missing unused-v2 immutable source proof')
    log=Path(s['pipeline_log'])
    if not log.is_absolute() or log.parent!=Path(s['old_root']):
        raise RuntimeError('Failure log must be the registered predecessor-root log')

def verify_predecessors(proc_root,old_root,unused_root,s,*,query=None):
    validate_snapshot(s)
    proc_root,old_root,unused_root=Path(proc_root),Path(old_root),Path(unused_root)
    if (proc_root.is_symlink() or not proc_root.is_dir()
            or not (proc_root/'self/status').read_text().startswith('Name:')):
        raise RuntimeError('Unknown Linux procfs observation; no launch')
    for _ in range(2):
        with os.scandir(proc_root) as entries:visible={p.name for p in entries if p.name.isdigit()}
        if str(os.getpid()) not in visible or any(str(p) in visible for p in s['registered_pids_absent']):
            raise RuntimeError('Old process live or incomplete procfs observation; no launch')
    for folder in (old_root,unused_root):
        if folder.is_symlink() or folder.resolve()!=folder or not folder.is_dir():raise RuntimeError('Missing/aliased predecessor root')
        for name in ('smoke','smoke_analysis','pilot','analysis','preflight_never_created'):
            if os.path.lexists(folder/name):raise RuntimeError('Predecessor has algorithm/partial output; no reuse')
    for name in ('queue.lock','pipeline.log','cancellation_receipt.json','cancellation_receipt.json.pending'):
        if os.path.lexists(unused_root/name):raise RuntimeError('Unused-v2 launch/cancellation state exists')
    if list(unused_root.glob('*.pending')):raise RuntimeError('Partial unused-v2 state exists')
    query=subprocess.run if query is None else query
    result=query(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True,timeout=15)
    if result.returncode!=0 or result.stderr!='' or not result.stdout.strip():raise RuntimeError('Unknown tmux observation; no launch')
    sessions=result.stdout.splitlines()
    if (any(not n or n.strip()!=n for n in sessions) or len(set(sessions))!=len(sessions)
            or {'csgb_episode_pilot_v1','csgb_episode_pilot_v2'}.intersection(sessions)):
        raise RuntimeError('Old tmux live or malformed session snapshot; no launch')

new=Path('/storage/dzj/spec_repro/CSGB_episode_pilot_20261016_v3')
plan_bytes=(new/'hardware_recovery.json').read_bytes();plan=json.loads(plan_bytes)
snapshot_bytes=(new/'terminal_failure_snapshot.json').read_bytes();snapshot=json.loads(snapshot_bytes)
protocol=json.loads((new/'pilot_protocol.json').read_text())
validate_snapshot(snapshot)
if (protocol.get('hardware_recovery_sha256')!=hashlib.sha256(plan_bytes).hexdigest()
        or plan.get('terminal_failure_snapshot_sha256')!=hashlib.sha256(snapshot_bytes).hexdigest()
        or plan.get('all_statistical_fields_unchanged') is not True or plan.get('new_root')!=str(new)
        or protocol.get('hardware')!={'physical_gpu_index':0,'expected_gpu_uuid':'GPU-0dab56d0-2236-4c0e-fc96-fe28081dba9b'}):
    raise RuntimeError('Recovery protocol/source event binding differs')
old,unused=Path(snapshot['old_root']),Path(snapshot['unused_root'])
verify_predecessors('/proc',old,unused,snapshot)
for path,wanted in ((old/'stage.sha256',snapshot['predecessor_stage_sha256']),
                    (old/'pilot_protocol.json',snapshot['predecessor_protocol_sha256']),
                    (unused/'stage.sha256',snapshot['unused_stage_sha256']),
                    (unused/'pilot_protocol.json',snapshot['unused_protocol_sha256']),
                    (Path(snapshot['pipeline_log']),snapshot['pipeline_log_sha256']),
                    (new/'predecessor_terminal.log',snapshot['pipeline_log_sha256'])):
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=wanted:
        raise RuntimeError('Terminal failure/source log proof changed; no launch')
if snapshot['terminal_error'] not in (new/'predecessor_terminal.log').read_text():
    raise RuntimeError('Bound terminal hardware failure is absent')
print(json.dumps({'terminal_hardware_recovery_checked':True,'predecessors_absent':True,
                  'old_outputs_absent':True,'unused_v2_not_launched':True,'guard_signals_sent':False}),flush=True)
PY
}
# END TERMINAL HARDWARE RECOVERY GUARD
assert_terminal_recovery_safe
common=(--source "$root/train24.jsonl" --scaler "$root/scaler.json"
  --protocol "$root/pilot_protocol.json" --seed 101
  --base /storage/dzj/spec_repro/models/vicuna-7b-v1.5-16k
  --draft /storage/dzj/spec_repro/models/vicuna-68m
  --checkpoint-receipt "$root/checkpoint_observation.json"
  --expected-gpu-uuid "$gpu_uuid" --physical-gpu-index 0)
"$py" "$root/specextend/csgb_episode_runner.py" "${common[@]}" \
  --validate-only --out "$root/preflight_never_created"
test ! -e "$root/preflight_never_created"
"$py" "$root/csgb_episode_gpu_guard.py" --index 0 --uuid "$gpu_uuid"
assert_terminal_recovery_safe
printf '%s\n' 'TECHNICAL_SMOKE_START: disposable doc0; no efficacy inference'
"$py" -u "$root/specextend/csgb_episode_runner.py" "${common[@]}" --smoke --out "$root/smoke"
"$py" "$root/analyze_csgb_episode_pilot.py" --raw "$root/smoke" --out "$root/smoke_analysis" --smoke
"$py" -c 'import json,sys; p=json.load(open(sys.argv[1])); sys.exit(0 if p.get("smoke_passed") is True else "Smoke gate failed; pilot not authorized")' "$root/smoke_analysis/summary.json"
sha256sum -c stage.sha256
"$py" "$root/csgb_episode_gpu_guard.py" --index 0 --uuid "$gpu_uuid"
assert_terminal_recovery_safe
printf '%s\n' 'PILOT_START: fit0..15 then frozen internal monitor16..23 at actual4K only'
"$py" -u "$root/specextend/csgb_episode_runner.py" "${common[@]}" --out "$root/pilot"
"$py" "$root/analyze_csgb_episode_pilot.py" --raw "$root/pilot" --out "$root/analysis"
sha256sum -c stage.sha256
printf '%s\n' 'PILOT_ANALYSIS_COMPLETE: inspect gates/raw/figures; no development/confirmation launch; NOT goal achieved'
