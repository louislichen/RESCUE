"""Read-only old-development export; stdout tar is the only output.

Terminal collection is NOT scientific audit or a speed/goal claim. It requires
the exact once-only launcher to exit successfully, 130 saved generation calls,
closed owners, and two stable file/process barriers. Partial snapshots retain
only newline-complete JSONL prefixes and can never authorize completion.
Original model, corpus, interpreter and credential paths are never followed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import tarfile

ROOT=Path('/mnt/8T/cl/rescue_experiment_cache/frozen_devcheck_20261020_v1')
SESSION='rescue_frozen_devcheck_v1'
EXPECTED=dict(own_quality_requests=64,quality_owners=64,prime_requests=2,timing_requests=64,owners=2)
COUNTS=dict(own_quality_requests='own_quality.jsonl',quality_owners='quality_owners.jsonl',
    prime_requests='prime.jsonl',timing_requests='runtime.jsonl',owners='owners.jsonl')
ROOT_FILES={'LAUNCH_MANIFEST.json','STAGE_RECEIPT.json','LAUNCH_ATTEMPT.json','RUN_STARTED.json',
    'DEPENDENCY_PREFLIGHT.json','GPU_PREFLIGHT.json','PIPELINE_EXIT.json','FAILURE.json',
    'pipeline.log','dependency_preflight.log','gpu_preflight.log','benchmark.log'}
DIRECTORIES={'capsule','tools','tests','results'}
SUFFIXES={'.py','.json','.jsonl','.md','.png','.svg','.sh','.log','.txt'}
SPECIAL={'terminal.tar.gz','evidence_snapshot.tar.gz','collection.stderr'}


def need(ok,message):
    if not ok:raise RuntimeError(message)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def relative(name):
    need(isinstance(name,str) and name and '\\' not in name and '\x00' not in name,'Invalid exported path')
    path=PurePosixPath(name)
    need(path.as_posix()==name and not path.is_absolute() and '..' not in path.parts and
        not {'.git','.ssh','.env','__pycache__'}.intersection(path.parts) and
        (path.suffix.lower() in SUFFIXES or path.name in SPECIAL),'Unregistered artifact/weight/corpus path: '+name)
    return Path(*path.parts)


def safe(root,name):
    path=root/relative(name)
    need(not any(p.is_symlink() for p in reversed((path,*path.parents))),'Symlink export is forbidden')
    need(path.is_file() and path.resolve().is_relative_to(root.resolve()),'Missing/escaped artifact: '+name)
    return path


def inventory(root):
    need(root.is_dir() and not any(p.is_symlink() for p in reversed((root,*root.parents))),'Unsafe or missing fixed root')
    result={}
    for path in root.rglob('*'):
        need(not path.is_symlink(),'Symlink must not be followed during collection')
        name=path.relative_to(root).as_posix();parts=Path(name).parts
        if path.is_dir():
            need(parts[0] in DIRECTORIES and not {'.git','.ssh','.env','__pycache__'}.intersection(parts),'Unknown artifact directory')
            continue
        need(path.is_file() and (name in ROOT_FILES or parts[0] in DIRECTORIES),'Unknown root artifact: '+name)
        result[name]=safe(root,name)
    return result


def handles(root):
    probe=subprocess.run(['tmux','has-session','-t','='+SESSION],capture_output=True)
    need(probe.returncode in (0,1),'Cannot inspect exact tmux session')
    values=subprocess.run(['ps','-eo','pid=,args='],capture_output=True,text=True,check=True).stdout.splitlines()
    science=[];execution=[]
    for value in values:
        try:pid,args=value.strip().split(None,1);tokens=shlex.split(args)
        except ValueError:continue
        if int(pid)==os.getpid():continue
        if str(root/'tools/frozen_devcheck.py') in tokens and '--output' in tokens:
            index=tokens.index('--output')
            if index+1<len(tokens) and tokens[index+1]==str(root/'results'):science.append(value.strip())
        if str(root/'tools/frozen_devcheck_stage.py') in tokens and 'run' in tokens:execution.append(value.strip())
    return dict(tmux_present=probe.returncode==0,actual_python_processes=science,actual_execution_processes=execution)


def idle(value):return not value['tmux_present'] and not value['actual_python_processes'] and not value['actual_execution_processes']


def object_value(payloads,name):
    need(name in payloads,'Missing saved receipt: '+name)
    value=json.loads(payloads[name]);need(isinstance(value,dict),'Expected JSON object: '+name);return value


def jsonl(raw,terminal=False):
    if terminal:need(not raw or raw.endswith(b'\n'),'Incomplete terminal JSONL')
    if raw and not raw.endswith(b'\n'):raw=raw[:raw.rfind(b'\n')+1]
    need(all(v.strip() for v in raw.splitlines()),'Blank JSONL record')
    values=[json.loads(v) for v in raw.splitlines()]
    need(all(isinstance(v,dict) for v in values),'Non-object JSONL record')
    return raw,values


def source_chain(payloads,terminal=False):
    manifest=object_value(payloads,'LAUNCH_MANIFEST.json')
    need(manifest.get('status')=='FROZEN_DEVCHECK_STAGED_NOT_LAUNCHED' and manifest.get('remote_root')==str(ROOT) and
        manifest.get('session')==SESSION and manifest.get('launch_authorized') is False and manifest.get('receiver_launches') is False and
        manifest.get('expected_counts')==EXPECTED and manifest.get('new_generation_requests')==130 and
        manifest.get('no_fresh_data') is manifest.get('no_training') is manifest.get('no_search') is True,'Not the fixed old-development deployment')
    pins=manifest.get('file_sha256',{})
    need(isinstance(pins,dict) and pins and set(pins).issubset(payloads),'Incomplete staged source/evidence inventory')
    for name,digest in pins.items():
        relative(name);need(sha(payloads[name])==digest,'Staged source/evidence SHA changed: '+name)
    tools=manifest.get('tool_sha256',{})
    need(tools and all(pins.get(n)==s for n,s in tools.items()),'Tool inventory not pinned')
    need(tools.get('tools/collect_frozen_devcheck.py')==sha(Path(__file__).read_bytes()),'Collector source differs from staged source')
    cap=object_value(payloads,'capsule/LAUNCH_MANIFEST.json')
    need(sha(payloads['capsule/LAUNCH_MANIFEST.json'])==manifest.get('capsule_manifest_sha256') and
        cap.get('status')=='STAGED_DISABLED_NOT_LAUNCHED' and cap.get('launcher_disabled') is True and cap.get('launch_authorized') is False and
        cap.get('union_freeze')==manifest.get('union_freeze'),'Inherited disabled capsule/proof differs')
    receipt=object_value(payloads,'STAGE_RECEIPT.json')
    need(receipt.get('status')==manifest['status'] and receipt.get('launch_authorized') is False and
        receipt.get('launch_manifest_sha256')==sha(payloads['LAUNCH_MANIFEST.json']),'Stage receive receipt differs')
    if 'results/protocol.json' in payloads:
        protocol=object_value(payloads,'results/protocol.json')
        need(protocol.get('fresh_confirmation') is False and protocol.get('no_new_confirmation_access') is True and protocol.get('no_training') is True and
            protocol.get('no_search') is True and protocol.get('goal_complete') is False and protocol.get('expected_counts')==EXPECTED and
            protocol.get('expected_requests')==130 and protocol.get('union_freeze')==manifest.get('union_freeze'),'Scientific protocol/truth differs from staged proof')
        need(protocol.get('source_sha256')==manifest.get('scientific_source_sha256'),'Scientific source differs from inherited runtime')
        for name,digest in protocol['source_sha256'].items():
            need(sha(payloads['results/source/'+name])==digest==sha(payloads['capsule/after/'+name]),'Saved scientific source SHA differs')
        need(protocol.get('entry_source_sha256')==sha(payloads['tools/frozen_devcheck.py'])==sha(payloads['results/FROZEN_DEVCHECK_SOURCE.py']),'Actual wrapper source not frozen')
    if terminal:
        attempt=object_value(payloads,'LAUNCH_ATTEMPT.json');started=object_value(payloads,'RUN_STARTED.json');finish=object_value(payloads,'PIPELINE_EXIT.json')
        links=dict(launch_manifest_sha256=sha(payloads['LAUNCH_MANIFEST.json']),launch_attempt_sha256=sha(payloads['LAUNCH_ATTEMPT.json']))
        need(attempt.get('status')=='ONCE_ONLY_LAUNCH_ATTEMPT' and attempt.get('session')==SESSION and attempt.get('authorized_explicitly') is True and
            attempt.get('launch_manifest_sha256')==links['launch_manifest_sha256'],'No exact explicit once-only launch')
        need(started.get('status')=='RUN_STARTED' and started.get('session')==SESSION and all(started.get(k)==v for k,v in links.items()),'Launch start links differ')
        need(type(finish.get('exit_code')) is int and finish['exit_code']==0 and all(finish.get(k)==v for k,v in links.items()) and
            finish.get('results_complete_sha256')==sha(payloads['results/COMPLETE.json']) and finish.get('analysis_required_locally') is True and
            finish.get('goal_complete') is False,'Pipeline not cleanly finished and linked to saved COMPLETE')
        dependency=object_value(payloads,'DEPENDENCY_PREFLIGHT.json');gpu=object_value(payloads,'GPU_PREFLIGHT.json')
        need(dependency.get('passed') is True and dependency.get('no_installation') is True and all(dependency.get(k)==v for k,v in links.items()),'Dependency preflight incomplete')
        need(gpu.get('passed') is True and gpu.get('tests_run')==8 and gpu.get('skipped')==0 and len(gpu.get('actual_gpu_cases',[]))==8 and
            gpu.get('gpu_uuid')=='GPU-9db4f7b5-1447-9a2a-5c1b-80379d909fd9' and all(gpu.get(k)==v for k,v in links.items()),'Real CUDA gate incomplete')
    return manifest


def terminal_counts(payloads):
    need(not any(n=='FAILURE.json' or n=='results/FAILURE.json' for n in payloads),'Failure evidence cannot be terminal-success collected')
    counts={name:len(jsonl(payloads.get('results/'+filename,b''),True)[1]) for name,filename in COUNTS.items()}
    need(counts==EXPECTED,'Incomplete130 generation/owner matrix')
    complete=object_value(payloads,'results/COMPLETE.json');quality=object_value(payloads,'results/QUALITY_PASSED.json')
    need(complete.get('passed') is True and complete.get('counts')==complete.get('expected_counts')==EXPECTED and complete.get('logical_requests')==130 and
        all(complete.get(k) is True for k in ('sources_unchanged','assets_unchanged','dependencies_unchanged','quality_gate_passed','all_canonical','owners_closed')) and
        complete.get('fresh_confirmation') is False and complete.get('goal_complete') is False and complete.get('analysis_required_locally') is True,'Terminal COMPLETE flags differ')
    need(quality.get('passed') is quality.get('own_all32_pairs') is quality.get('actual_same_bfloat16_nonzero_mapper') is quality.get('quality_owners_closed') is True,'Quality gate incomplete')
    generation=[r for n in ('own_quality.jsonl','prime.jsonl','runtime.jsonl') for r in jsonl(payloads['results/'+n],True)[1]]
    need(len({r.get('request_id') for r in generation})==130 and all(isinstance(r.get('request_id'),str) and len(r['request_id'])==32 for r in generation) and
        sorted(r.get('execution_index') for r in generation)==list(range(130)),'Generation identities/order missing or repeated')
    qowners=jsonl(payloads['results/quality_owners.jsonl'],True)[1];owners=jsonl(payloads['results/owners.jsonl'],True)[1]
    need(len({r['owner_id'] for r in qowners+owners})==66 and
        all(r.get('engine_closed') is r.get('hook_restored') is True and r.get('closed_runtime_stats',{}).get('entries')==0 for r in qowners+owners),'Owner not private/closed/restored')
    diagnostics=jsonl(payloads.get('results/gpu_diagnostics.jsonl',b''),True)[1]
    need(len(diagnostics)==4 and {(r.get('owner_id'),r.get('phase')) for r in diagnostics}==
        {(r['owner_id'],p) for r in owners for p in ('owner_start','owner_end')},'Must retain four baseline-cadence timing-owner telemetry records')
    return counts


def snapshot(terminal=False):
    root=ROOT;before=handles(root)
    if terminal:need(idle(before),'Study still has live Python/execution/tmux handles')
    paths=inventory(root);original={n:p.read_bytes() for n,p in paths.items()};source_chain(original,terminal)
    payloads=dict(original);counts={}
    for name in payloads:
        if name.endswith('.jsonl'):payloads[name],_=jsonl(payloads[name],terminal)
    for name,filename in COUNTS.items():counts[name]=len(jsonl(payloads.get('results/'+filename,b''),terminal)[1])
    if terminal:counts=terminal_counts(payloads)
    after=handles(root)
    if terminal:
        need(idle(after),'Study handles appeared while collecting')
        latest=inventory(root);need(set(latest)==set(paths) and all(p.read_bytes()==original[n] for n,p in latest.items()),'Terminal bytes changed during collection')
    collector=Path(__file__).read_bytes();payloads['COLLECTOR_SOURCE.py']=collector
    receipt=dict(schema=1,status='FROZEN_DEVCHECK_TERMINAL_COLLECTED_NOT_YET_AUDITED' if terminal else 'PARTIAL_FROZEN_DEVCHECK_SNAPSHOT',
        fixed_root=str(root),session=SESSION,terminal=terminal,counts=counts,expected_counts=EXPECTED,
        logical_requests=sum(counts[n] for n in ('own_quality_requests','prime_requests','timing_requests')),expected_requests=130,
        source_before={n:dict(bytes=len(v),sha256=sha(v)) for n,v in original.items()},
        files={n:dict(bytes=len(v),sha256=sha(v)) for n,v in payloads.items()},handles_before=before,handles_after=after,
        collector_sha256=sha(collector),analysis_required_locally=True,independent_raw_audit_passed=False,
        fresh_confirmation=False,goal_complete=False,collected_at_utc=datetime.now(timezone.utc).isoformat())
    filename='TERMINAL_COLLECTION.json' if terminal else 'SNAPSHOT_COLLECTION.json'
    payloads[filename]=(json.dumps(receipt,indent=2,allow_nan=False)+'\n').encode()
    evidence=dict(schema=1,status=receipt['status'],file_sha256={n:sha(v) for n,v in payloads.items()},goal_complete=False)
    payloads['EVIDENCE_MANIFEST.json']=(json.dumps(evidence,indent=2,allow_nan=False)+'\n').encode()
    return payloads,receipt


def archive(payloads,output):
    with tarfile.open(fileobj=output,mode='w|gz',format=tarfile.USTAR_FORMAT) as handle:
        for name,raw in sorted(payloads.items()):
            relative(name);entry=tarfile.TarInfo(name);entry.size=len(raw);entry.mode=0o444
            handle.addfile(entry,io.BytesIO(raw))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--terminal',action='store_true');args=parser.parse_args()
    payloads,_=snapshot(args.terminal);archive(payloads,sys.stdout.buffer)


if __name__=='__main__':main()
