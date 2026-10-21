"""One exclusive, read-only terminal collection; never launches a GPU job."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = '/mnt/8T/cl/rescue_experiment_cache/fp32_splitkv_timing_20261021_v1'
DEST = Path(__file__).resolve().parent/'out_fp32_splitkv_timing_20261021_v1'
PYTHON = '/mnt/8T/cl/rescue_experiment_cache/specextend_match_20261020_v1/venv/bin/python'
HOST = 'wanghl@sz.acdawn.cn'
OPTS = ['-i', '/Users/zirui/.ssh/id_ed25519', '-o', 'BatchMode=yes',
        '-o', 'IdentitiesOnly=yes', '-o', 'ConnectTimeout=8']
FILES = ('LAUNCH_STARTED.json', 'PIPELINE_EXIT.json', 'PREREGISTRATION.json',
         'SOURCE_SHA256SUMS', 'launch.sh', 'timing.log')
DIRS = ('results', 'capsule', 'inputs')
REMOTE = r'''
from datetime import datetime, timezone
import hashlib,json,subprocess
from pathlib import Path
r=Path('/mnt/8T/cl/rescue_experiment_cache/fp32_splitkv_timing_20261021_v1')
def observe(argv):
 p=subprocess.run(argv,text=True,capture_output=True,check=False)
 return dict(argv=argv,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
tmux=observe(['tmux','has-session','-t','rescue_fp32_splitkv_timing_v1'])
proc=observe(['ps','-p','334007,334017,334580','-o','pid,ppid,comm'])
children=observe(['ps','--ppid','334007,334017,334580','-o','pid,ppid,comm'])
gpu=observe(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'])
identity=observe(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'])
assert tmux['exit_code']==proc['exit_code']==children['exit_code']==1
assert gpu['exit_code']==0 and not gpu['stdout'].strip()
assert identity['exit_code']==0 and identity['stdout'].strip()=='GPU-9db4f7b5-1447-9a2a-5c1b-80379d909fd9'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
e=json.loads((r/'PIPELINE_EXIT.json').read_bytes());t=json.loads((r/'results/TERMINAL.json').read_bytes())
assert sha(r/'results/TERMINAL.json')=='e56ffb5f51c7d399cdce4b930c6a09a9f0265fffbc5cc93540f296efb3d64b1f'
assert e['exit_code']==0 and e['terminal_sha256']==sha(r/'results/TERMINAL.json')
assert t['passed'] and t['attempted_calls']==t['returned_calls']==t['validated_calls']==792
assert t['counts']==dict(owners=24,prime_requests=24,timing_requests=768)
assert not r.is_symlink()
paths=[r/n for n in ('LAUNCH_STARTED.json','PIPELINE_EXIT.json','PREREGISTRATION.json','SOURCE_SHA256SUMS','launch.sh','timing.log')]
for d in ('results','capsule','inputs'):
 assert not (r/d).is_symlink()
 for p in (r/d).rglob('*'):
  assert not p.is_symlink()
  if p.is_file():
   assert p.suffix not in ('.pt','.bin','.safetensors','.so','.cubin','.ptx','.pyc')
   paths.append(p)
barrier=dict(status='TERMINAL_EXTERNAL_BARRIER_PASSED',passed=True,observed_at_utc=datetime.now(timezone.utc).isoformat(),
 tmux_observation=tmux,process_observation=proc,children_observation=children,gpu_observation=gpu,identity_observation=identity,
 tmux_absent=True,model_process_absent=True,model_children_absent=True,gpu_compute_idle=True,
 root=str(r),terminal_sha256=sha(r/'results/TERMINAL.json'),scope='Read-only actual subprocess observations; point-in-time, not signed attestation.')
print(json.dumps(dict(file_sha256={str(p.relative_to(r)):sha(p) for p in sorted(paths)},
 file_size_bytes={str(p.relative_to(r)):p.stat().st_size for p in sorted(paths)},external_terminal_barrier=barrier),sort_keys=True))
'''


def inventory():
    result = subprocess.run(['ssh', '-p', '7981', *OPTS, HOST, PYTHON+' -B -'],
                            input=REMOTE, text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def write(name, value):
    with (DEST/name).open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write('\n')


def main():
    assert not DEST.exists(), 'Never reuse/overwrite a partial collection root'
    before = inventory()
    DEST.mkdir()
    write('COLLECTION_BEFORE.json', before)
    print('Terminal barrier passed; collecting', len(before['file_sha256']),
          'files /', sum(before['file_size_bytes'].values()), 'bytes', flush=True)
    codes = []
    for names, recursive in ((FILES, False), (DIRS, True)):
        argv = ['scp', '-C', '-P', '7981', *OPTS]
        if recursive:
            argv.append('-r')
        argv += [HOST+':'+ROOT+'/'+n for n in names]+[str(DEST)]
        code = subprocess.run(argv, check=False).returncode
        codes.append(code)
        assert code == 0, 'Preserve partial output; no automatic retry'
    after = inventory()
    write('COLLECTION_AFTER.json', after)
    assert before['file_sha256'] == after['file_sha256']
    assert before['file_size_bytes'] == after['file_size_bytes']
    for name, digest in before['file_sha256'].items():
        p = DEST/name
        assert not p.is_symlink() and p.is_file() and p.stat().st_size == before['file_size_bytes'][name]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, name
    actual = {p.relative_to(DEST).as_posix() for p in DEST.rglob('*') if p.is_file()}
    assert actual == set(before['file_sha256'])|{'COLLECTION_BEFORE.json','COLLECTION_AFTER.json'}
    write('TERMINAL_EXTERNAL_BARRIER.json', after['external_terminal_barrier'])
    write('COLLECTION_RECEIPT.json', dict(status='READ_ONLY_TERMINAL_COLLECTION_VERIFIED',
          collected_at_utc=datetime.now(timezone.utc).isoformat(), remote_root=ROOT,
          collected_files=len(before['file_sha256']), scp_exit_codes=codes,
          file_sha256=before['file_sha256'], file_size_bytes=before['file_size_bytes'],
          remote_read_only_barrier_before=before, remote_read_only_barrier_after=after,
          local_matches_both_remote_inventories=True, weights_transferred=False,
          compiled_cache_transferred=False, remote_writes=False, automatic_retry=False,
          collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print('Collection READY:', DEST, flush=True)


if __name__ == '__main__':
    main()
