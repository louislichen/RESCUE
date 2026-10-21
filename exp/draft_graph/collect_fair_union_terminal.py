"""Read-only UNION snapshots/terminal export; the only output is stdout tar.

Never reads candidate Parquet, model weights, credentials, original execution
source paths or an external plan. Terminal export has two handle/file barriers;
live exports retain complete JSONL prefixes and cannot authorize completion.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

ROOT=Path('/mnt/8T/cl/rescue_experiment_cache/fair_union_confirmation_20261020_v2')
CAPSULE=Path('/mnt/8T/cl/rescue_experiment_cache/fair_union_confirmation_20261020_v1')
PYTHON='/mnt/8T/cl/rescue_experiment_cache/specextend_match_20261020_v1/venv/bin/python'
PYTHON_CHAIN=dict(status='PINNED_PYTHON_CHAIN_VERIFIED',links=[dict(path=PYTHON,readlink='/home/wanghl/miniconda3/envs/rescue/bin/python'),
    dict(path='/home/wanghl/miniconda3/envs/rescue/bin/python',readlink='python3.10')],
    real_binary_path='/home/wanghl/miniconda3/envs/rescue/bin/python3.10',
    real_binary_sha256='8119915f96beb2ae0797f1b92434b15f4a0f4a55900297fe3c713c3f1e5a5a3d')
GPU='GPU-9db4f7b5-1447-9a2a-5c1b-80379d909fd9'
SESSION='rescue_fair_union_confirmation_v2'
ROOT_FILES=('EXECUTION_PLAN.json','EXECUTION_MANIFEST.json','RUN_STARTED.json','PIPELINE_EXIT.json',
    'FINAL_EXECUTION_VERIFIED.json','DEPENDENCY_PREFLIGHT.json','GPU_PREFLIGHT.json','FAILURE.json',
    'CAPSULE_LAUNCH_MANIFEST.json','frozen_protocol.json','pipeline.log','dependency_preflight.log','gpu_preflight.log','benchmark.log')
SOURCE_FILES=('execution_source/tools/fair_union_execution.py',
    'execution_source/tools/run_fair_union_confirmation_20261020_v2.sh',
    'execution_source/helpers/rescue/__init__.py',
    'execution_source/helpers/rescue/fair_union_confirmation_data.py',
    'execution_source/helpers/rescue/fair_confirmation_data.py',
    'execution_source/helpers/rescue/baseline_horizon_data.py',
    'execution_source/helpers/exp/draft_graph/receive_fair_union_stage.py')
TOOL_FILES=('tools/fair_union_execution.py','tools/run_fair_union_confirmation_20261020_v2.sh')
PLAN_KEYS={'schema','status','execution_authorized','execution_root','capsule_root','python','python_sha256',
    'expected_gpu_uuid','session','capsule_transport','capsule_manifest_sha256','tools_sha256','dependencies',
    'assets','old_metadata','raw_source','raw_source_manifest'}
DEPENDENCIES=('python','torch','cuda','transformers','pyarrow','flash_attn')
ASSET_ROLES=('teacher','student','mapper','policy','cost','prime_file')
OLD_METADATA=('exp/long_system/data/train.jsonl','exp/long_system/data/development.jsonl','exp/long_system/data/test.jsonl',
    *('exp/module_'+part+'/data_v1_20261018/'+name+'.jsonl' for part in ('opt','arch') for name in ('fit','tune','warmup','confirmation')))
GPU_CASES=(
    'test_rescue_draft_runtime.CudaGraphDraftRuntimeTests.test_real_capture_replay_cap_boundaries_workset_changes_and_request_reset',
    'test_rescue_draft_runtime.CudaGraphDraftRuntimeTests.test_graph_eos_and_short_output_boundaries',
    'test_rescue_draft_runtime.CudaGraphDraftRuntimeTests.test_graph_all_accept_catchup_and_padding_poison',
    'test_rescue_draft_runtime.CudaGraphDraftRuntimeTests.test_graph_explicit_rollback_zero_through_four_preserves_every_layer',
    'test_rescue_draft_runtime.CudaGraphDraftRuntimeTests.test_graph_fp32_target_bfloat16_draft_with_nonzero_float_mapper',
    'test_rescue_baseline_runtime.GraphBaselineTests.test_graph_variable_length_geometry_full_kv_and_queries_match_eager',
    'test_rescue_baseline_runtime.GraphBaselineTests.test_graph_fixed_k12_eos_and_all_accept_catchup',
    'test_rescue_baseline_runtime.GraphBaselineTests.test_graph_retrieval_shared_model_owner_crossover')
SUFFIXES={'.py','.sh','.json','.jsonl','.md','.log','.txt','.png','.svg'}
SPECIAL={'evidence_snapshot.tar.gz','terminal.tar.gz','collection.stderr'}
FORBIDDEN={'.git','.ssh','.env'}
COUNT_FILES=dict(reference_requests='reference.jsonl',own_quality_requests='own_quality.jsonl',
    baseline_quality_requests='baseline_quality.jsonl',matching_quality_requests='matching_quality.jsonl',
    prime_requests='prime.jsonl',timing_requests='runtime.jsonl',owners='owners.jsonl')


def require(ok,message):
    if not ok:raise RuntimeError(message)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def valid_sha(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def absolute(value):
    return isinstance(value,str) and bool(value) and '\x00' not in value and '\\' not in value and Path(value).is_absolute() and '..' not in Path(value).parts


def json_object(raw,label):
    value=json.loads(raw);require(isinstance(value,dict),'Expected object: '+label);return value


def safe(root,name):
    require(isinstance(name,str) and name and not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe saved path')
    path=root/name
    require(path.resolve().is_relative_to(root.resolve()) and not any(p.is_symlink() for p in (path,*path.parents) if p.is_relative_to(root)),
        'Symlink/escaped saved path: '+name)
    require(path.is_file(),'Missing regular artifact: '+name);return path


def matrix(protocol):
    choices=protocol.get('frozen_choices',{});require(set(choices)=={'4096','8192'},'Both frozen union contexts required')
    require(all(isinstance(v,dict) and 4<=len(v)<=8 for v in choices.values()),'Four to eight unique contenders/context required')
    m=sum(len(v) for v in choices.values());extra=[]
    for n in (4096,8192):
        arms=list(choices[str(n)].values())
        require(all(isinstance(a,dict) and mode==a.get('mode') and a.get('family') in ('retrieval','full_tree') for mode,a in choices[str(n)].items()),'Invalid union mode/arm')
        seen={(a['top_chunks'],a['fixed_k']) for a in arms if a['family']=='retrieval' and a.get('backend')=='eager'}
        for arm in arms:
            if arm['family']!='retrieval' or arm.get('backend')=='eager':continue
            key=arm['top_chunks'],arm['fixed_k']
            if key not in seen:extra.append(dict(length=n,top_chunks=key[0],fixed_k=key[1]));seen.add(key)
    r=len(extra);require(8<=m<=16 and 0<=r<=8,'Invalid union multiplicity')
    expected=dict(reference_requests=32,own_quality_requests=64,baseline_quality_requests=m,matching_quality_requests=r,
        prime_requests=m+2,timing_requests=32*(m+2),owners=m+2)
    total=162+34*m+r
    require(protocol.get('comparisons')==m and protocol.get('expected_counts')==expected and protocol.get('expected_requests')==total and
        protocol.get('extra_eager_reference_geometries')==extra,'Frozen M/R/count plan differs')
    proof=protocol.get('union_freeze',{})
    require(proof.get('status')=='UNION_VERIFIED' and proof.get('choices')==choices and proof.get('M')==m and proof.get('R')==r and
        proof.get('expected_counts')==expected and proof.get('scientific_requests')==total,'Executed union is not bound to the complete proof')
    return m,r,expected,total


def handles(root):
    try:session=subprocess.run(['tmux','has-session','-t','='+SESSION],capture_output=True)
    except OSError as error:raise RuntimeError('Cannot inspect tmux handles') from error
    require(session.returncode in (0,1),'tmux handle check failed')
    lines=subprocess.run(['ps','-eo','pid=,args='],capture_output=True,text=True,check=True).stdout.splitlines()
    science,execution=[],[]
    for line in lines:
        try:
            pid_text,args=line.strip().split(None,1);pid=int(pid_text);tokens=shlex.split(args)
        except (ValueError,TypeError):continue
        if pid==os.getpid():continue
        if '-m' in tokens:
            pos=tokens.index('-m')
            if pos+1<len(tokens) and tokens[pos+1]=='rescue.fair_union_confirmation' and '--output' in tokens:
                j=tokens.index('--output')
                if j+1<len(tokens) and tokens[j+1]==str(root/'results'):science.append(line.strip())
        if str(root/'tools/fair_union_execution.py') in tokens or str(root/'tools/run_fair_union_confirmation_20261020_v2.sh') in tokens:
            execution.append(line.strip())
    return dict(tmux_present=session.returncode==0,actual_python_processes=science,actual_execution_processes=execution)


def inventory(root):
    result=[]
    for name in ('results','data_gate'):
        directory=root/name
        if not directory.exists():continue
        require(directory.is_dir() and not directory.is_symlink(),'Unsafe artifact directory: '+name)
        for path in directory.rglob('*'):
            require(not path.is_symlink(),'Symlink artifact must not be dereferenced')
            if '__pycache__' in path.parts:continue
            require(not FORBIDDEN.intersection(path.relative_to(root).parts),'Forbidden credential/repository directory')
            if path.is_file():
                require(path.suffix.lower() in SUFFIXES or path.name in SPECIAL,'Unregistered binary/weight/corpus artifact: '+str(path))
                safe(root,path.relative_to(root).as_posix());result.append(path)
    for name in (*ROOT_FILES,*SOURCE_FILES,*TOOL_FILES):
        path=root/name
        if path.exists():result.append(safe(root,name))
    return sorted(set(result))


def execution_chain(payloads,root,protocol,terminal=False):
    def read(name):return json_object(payloads[name],name)
    manifest=read('EXECUTION_MANIFEST.json');plan=read('EXECUTION_PLAN.json');started=read('RUN_STARTED.json') if 'RUN_STARTED.json' in payloads else {}
    require(manifest.get('plan_path')==str(root/'EXECUTION_PLAN.json') and manifest.get('plan_sha256')==sha(payloads['EXECUTION_PLAN.json']),
        'Saved original plan bytes/SHA differ')
    require(set(plan)==PLAN_KEYS and plan.get('schema')==1 and plan.get('status')=='FROZEN_EXECUTION_PLAN_NOT_EXECUTED' and
        plan.get('execution_authorized') is True,'Execution plan was not explicitly authorized/closed-schema frozen')
    require(plan.get('execution_root')==str(root) and plan.get('capsule_root')==str(CAPSULE) and plan.get('python')==PYTHON and
        plan.get('expected_gpu_uuid')==GPU and plan.get('session')==SESSION,'Execution plan changes fixed roots/runtime/GPU/session')
    require(isinstance(plan.get('dependencies'),dict) and set(plan['dependencies'])==set(DEPENDENCIES) and
        all(isinstance(v,str) and v for v in plan['dependencies'].values()) and isinstance(plan.get('assets'),dict) and
        set(plan['assets'])==set(ASSET_ROLES) and all(absolute(v) for v in plan['assets'].values()),'Execution plan retains placeholder dependency/asset roles')
    old=plan.get('old_metadata',{})
    require(isinstance(old,dict) and set(old)==set(OLD_METADATA) and all(isinstance(v,dict) and set(v)=={'path','sha256'} and
        absolute(v['path']) and v['path'].endswith('/'+n) and valid_sha(v['sha256']) for n,v in old.items()),'All eleven explicit old metadata pins required')
    raw=plan.get('raw_source',{});inherited=plan.get('raw_source_manifest',{})
    require(set(raw)=={'path','expected_sha256'} and absolute(raw.get('path')) and Path(raw['path']).name=='validation-00000-of-00001.parquet' and
        valid_sha(raw.get('expected_sha256')) and set(inherited)=={'path','sha256'} and absolute(inherited.get('path')) and
        inherited['path'].endswith('/exp/module_arch/data_v1_20261018/manifest.json') and valid_sha(inherited.get('sha256')),
        'Candidate provenance/inherited manifest pins required; original paths are never followed')
    require(manifest.get('schema')==1 and manifest.get('status')=='EXECUTION_FROZEN_BEFORE_DATA_ACCESS' and
        manifest.get('execution_root')==str(root) and manifest.get('capsule_root')==str(CAPSULE),'Execution manifest status/roots differ')
    identity=manifest.get('python',{})
    require(identity.get('python_chain')==PYTHON_CHAIN and identity.get('python')==PYTHON and
        identity.get('python_sha256')==plan.get('python_sha256')==PYTHON_CHAIN['real_binary_sha256'] and
        identity.get('version')==plan.get('dependencies',{}).get('python'),'Fixed pinned interpreter-chain receipt differs; original links are never followed')
    sources=manifest.get('source_sha256',{})
    require(set(sources)==set(SOURCE_FILES),'Exact seven saved execution/helper sources required')
    provenance={n:str(root/n.removeprefix('execution_source/')) if n.startswith('execution_source/tools/') else
        str(CAPSULE/'after'/n.removeprefix('execution_source/helpers/')) for n in SOURCE_FILES}
    require(manifest.get('source_path_map')==provenance,'Execution source provenance is not the exact fixed allowlist')
    for name,digest in sources.items():
        require((not terminal and name not in payloads) or (name in payloads and sha(payloads[name])==digest),'Saved execution/helper source changed: '+name)
    tools=manifest.get('tools_sha256',{})
    require(set(tools)==set(TOOL_FILES) and tools==plan.get('tools_sha256'),'Exact plan-pinned two execution tools required')
    for name,digest in tools.items():require(name in payloads and sha(payloads[name])==digest==sources.get('execution_source/'+name),'Actual/saved execution tool changed: '+name)
    capsule=read('CAPSULE_LAUNCH_MANIFEST.json') if 'CAPSULE_LAUNCH_MANIFEST.json' in payloads else {}
    if terminal or capsule:
        require(manifest.get('capsule_saved_manifest')=='CAPSULE_LAUNCH_MANIFEST.json' and
            sha(payloads['CAPSULE_LAUNCH_MANIFEST.json'])==manifest.get('capsule_manifest_sha256')==plan.get('capsule_manifest_sha256'),
            'Saved capsule manifest bytes/plan pins differ')
        require(capsule.get('schema')==1 and capsule.get('status')=='STAGED_DISABLED_NOT_LAUNCHED' and capsule.get('remote_root')==str(CAPSULE) and
            capsule.get('launcher_disabled') is True and capsule.get('launch_authorized') is False and capsule.get('union_freeze')==manifest.get('union_freeze') and
            capsule.get('expected_counts')==manifest.get('expected_counts') and capsule.get('expected_requests')==manifest.get('expected_requests'),
            'Disabled capsule/proof/count chain differs')
        if protocol:require(capsule.get('scientific_source_sha256')==protocol.get('source_sha256'),'Scientific output source differs from the disabled capsule freeze')
        for name in SOURCE_FILES:
            if name.startswith('execution_source/helpers/'):
                require(sources[name]==capsule.get('source_sha256',{}).get(name.removeprefix('execution_source/helpers/')),'Helper is not pinned by saved capsule')
    transport=plan.get('capsule_transport',{})
    require(set(transport)=={'path','sha256'} and absolute(transport.get('path')) and valid_sha(transport.get('sha256')) and
        manifest.get('capsule_transport_path')==transport['path'] and manifest.get('capsule_transport_sha256')==transport.get('sha256'),
        'External transport provenance/pin differs; original path is never followed')
    require(manifest.get('session')==SESSION,'Wrong execution session')
    if terminal or started:
        require(started.get('session')==SESSION and started.get('status')=='RUN_STARTED' and
            started.get('execution_manifest_sha256')==sha(payloads['EXECUTION_MANIFEST.json']) and started.get('plan_sha256')==manifest['plan_sha256'],
            'RUN_STARTED is not bound to exact plan/execution source')
    require(manifest.get('analysis_required_locally') is True and manifest.get('goal_complete') is False,'Execution incorrectly claims analysis/goal complete')
    if protocol:
        m,r,expected,total=matrix(protocol)
        require(manifest.get('union_freeze')==protocol['union_freeze'] and manifest.get('expected_counts')==expected and manifest.get('expected_requests')==total,
            'Execution manifest differs from scientific union/counts')
    if terminal:
        finish=read('PIPELINE_EXIT.json');require(type(finish.get('exit_code')) is int and finish['exit_code']==0,'Pipeline did not exit integer0')
        require(finish.get('execution_manifest_sha256')==sha(payloads['EXECUTION_MANIFEST.json']) and finish.get('plan_sha256')==manifest['plan_sha256'] and
            finish.get('results_complete_sha256')==sha(payloads['results/COMPLETE.json']) and finish.get('analysis_required_locally') is True and
            finish.get('goal_complete') is False,'Pipeline terminal receipt has broken source/plan/result links')
        require(finish.get('session')==SESSION and finish.get('phase')=='EXECUTION_FINISHED_ANALYSIS_REQUIRED','Unexpected terminal phase/session')
        links=dict(execution_manifest_sha256=sha(payloads['EXECUTION_MANIFEST.json']),plan_sha256=manifest['plan_sha256'],
            capsule_manifest_sha256=manifest['capsule_manifest_sha256'])
        dependency=read('DEPENDENCY_PREFLIGHT.json');gpu=read('GPU_PREFLIGHT.json');final=read('FINAL_EXECUTION_VERIFIED.json')
        require(dependency.get('status')=='DEPENDENCY_PREFLIGHT_PASSED' and dependency.get('passed') is True and dependency.get('no_installation') is True and
            all(dependency.get(k)==v for k,v in links.items()) and set(plan.get('dependencies',{}))==set(DEPENDENCIES) and
            {k:dependency.get('versions',{}).get(k) for k in DEPENDENCIES}==plan['dependencies'] and
            dependency.get('versions',{}).get('flash_receipt')==protocol.get('flash_attention'),'Linked dependency/actual Flash preflight differs')
        require(gpu.get('status')=='GPU_PREFLIGHT_PASSED' and gpu.get('passed') is True and type(gpu.get('tests_run')) is int and gpu['tests_run']==8 and
            type(gpu.get('skipped')) is int and gpu['skipped']==0 and gpu.get('actual_gpu_cases')==list(GPU_CASES) and gpu.get('physical_gpu_uuid')==GPU and
            all(gpu.get(k)==v for k,v in links.items()),'Linked eight real CUDA preflight cases/UUID differ')
        require(final.get('status')=='EXECUTION_FINISHED_ANALYSIS_REQUIRED' and final.get('passed') is True and
            final.get('execution_manifest_sha256')==links['execution_manifest_sha256'] and final.get('plan_sha256')==links['plan_sha256'] and
            final.get('counts')==protocol['expected_counts'] and final.get('logical_requests')==protocol['expected_requests'] and
            final.get('results_complete_sha256')==sha(payloads['results/COMPLETE.json']) and
            final.get('dependency_preflight_sha256')==sha(payloads['DEPENDENCY_PREFLIGHT.json']) and final.get('gpu_preflight_sha256')==sha(payloads['GPU_PREFLIGHT.json']) and
            all(final.get(k) is True for k in ('execution_sources_unchanged','capsule_unchanged','analysis_required_locally')) and
            final.get('goal_complete') is False,'Final execution receipt/closed preflight links differ')
        for name,key in (('frozen_protocol.json','final_protocol_sha256'),('data_gate/GATE_SOURCE_FREEZE.json','source_freeze_sha256'),
            ('data_gate/data/DATASET_MANIFEST.json','dataset_manifest_sha256'),('data_gate/data/confirmation.jsonl','dataset_sha256')):
            require(name in payloads and sha(payloads[name])==protocol.get(key),'Execution/data preparation snapshot differs from scientific output: '+name)
    return manifest


def terminal_check(root):
    for name in ('FAILURE.json','results/FAILURE.json','data_gate/GATE_FAILURE.json','data_gate/data/DATA_PREP_FAILURE.json'):
        require(not (root/name).exists(),'Failure marker exists: '+name)
    state=handles(root)
    require(not state['tmux_present'] and not state['actual_python_processes'] and not state['actual_execution_processes'],
        'Scientific/control process or tmux handle still exists')
    return state


def export(root=ROOT,*,complete=False):
    root=Path(root)
    require(root.resolve()==ROOT.resolve() and root.is_dir() and not root.is_symlink(),'Only the exact isolated v2 root may be exported')
    started=datetime.now(timezone.utc).isoformat();before=terminal_check(root) if complete else handles(root)
    original=inventory(root);payloads,cuts={},{}
    for path in original:
        name=path.relative_to(root).as_posix();raw=path.read_bytes()
        if path.suffix=='.jsonl':
            cut=len(raw) if not raw or raw.endswith(b'\n') else raw.rfind(b'\n')+1
            require(not complete or cut==len(raw),'Terminal JSONL has an incomplete tail: '+name)
            cuts[name]=len(raw)-cut;raw=raw[:cut]
            for line in raw.splitlines():require(bool(line.strip()) and isinstance(json.loads(line),dict),'Malformed/blank JSONL record: '+name)
        elif path.suffix=='.json':json_object(raw,name)
        payloads[name]=raw
    protocol=json_object(payloads['results/protocol.json'],'protocol') if 'results/protocol.json' in payloads else {}
    manifest=execution_chain(payloads,root,protocol,complete)
    if protocol:m,r,expected,total=matrix(protocol)
    else:
        proof=manifest.get('union_freeze',{});m,r=proof.get('M'),proof.get('R');expected=manifest.get('expected_counts');total=manifest.get('expected_requests')
        require(type(m) is int and 8<=m<=16 and type(r) is int and 0<=r<=8 and total==162+34*m+r,'Pre-science M/R plan invalid')
    counts={key:len(payloads.get('results/'+name,b'').splitlines()) for key,name in COUNT_FILES.items()}
    logical=sum(v for k,v in counts.items() if k!='owners')
    if complete:
        done=json_object(payloads['results/COMPLETE.json'],'COMPLETE')
        require(counts==expected and logical==total and done.get('passed') is True and done.get('counts')==expected and done.get('expected_counts')==expected and
            done.get('logical_requests')==total and done.get('comparisons')==m and done.get('matching_geometry_count')==r,'Incomplete terminal request/owner matrix')
        require(all(done.get(k) is True for k in ('sources_unchanged','assets_unchanged','dependencies_unchanged','quality_gate_passed','all_canonical',
            'owners_closed','analysis_required_locally','data_unchanged','dual_study_freeze_unchanged')) and done.get('goal_complete') is False,'Terminal scientific gate flags differ')
        require(len(payloads.get('results/quality_owners.jsonl',b'').splitlines())==64+m+r,'Quality owner ledger incomplete')
        requests=[json.loads(line) for key,name in COUNT_FILES.items() if key!='owners' for line in payloads.get('results/'+name,b'').splitlines()]
        require(len({row.get('request_id') for row in requests})==total and all(isinstance(row.get('request_id'),str) and row['request_id'] for row in requests) and
            all(type(row.get('execution_index')) is int for row in requests) and sorted(row['execution_index'] for row in requests)==list(range(total)),
            'Terminal UUID/execution ledger incomplete')
        owners=[json.loads(v) for v in payloads['results/owners.jsonl'].splitlines()]
        require(all(o.get('engine_closed') is True and o.get('hook_restored') is True and o.get('maximum_live_owners')==1 and
            o.get('closed_runtime_stats',{}).get('entries')==0 for o in owners),'Terminal private owner not closed')
        quality=[json.loads(v) for v in payloads['results/quality_owners.jsonl'].splitlines()]
        require(all(o.get('engine_closed') is True and o.get('hook_restored') is True and o.get('closed_runtime_stats',{}).get('entries')==0 for o in quality),
            'Terminal quality owner not closed')
        tree_count=sum(a.get('family')=='full_tree' for v in protocol['frozen_choices'].values() for a in v.values())
        require(len(payloads.get('results/tree_raw.jsonl',b'').splitlines())==34*tree_count,'Terminal raw full-tree ledger incomplete')
        require('FINAL_EXECUTION_VERIFIED.json' in payloads and 'DEPENDENCY_PREFLIGHT.json' in payloads and 'GPU_PREFLIGHT.json' in payloads and
            'frozen_protocol.json' in payloads,'Terminal preflight/final execution/frozen protocol evidence missing')
    after=terminal_check(root) if complete else handles(root)
    if complete:require(original==inventory(root),'Terminal artifact set changed between barriers')
    for path in original:
        name=path.relative_to(root).as_posix();raw=path.read_bytes()
        if not complete and path.suffix in ('.jsonl','.log'):require(raw.startswith(payloads[name]),'Live artifact rewrote its collected prefix: '+name)
        else:require(raw==payloads[name],'Artifact bytes changed between barriers: '+name)
    own_source=Path(__file__).read_bytes()
    payloads['COLLECTOR_SOURCE.py']=own_source
    files={name:dict(bytes=len(raw),sha256=sha(raw)) for name,raw in payloads.items()}
    receipt=dict(schema=1,status='UNION_TERMINAL_COLLECTED_NOT_YET_INDEPENDENTLY_AUDITED' if complete else 'PARTIAL_READ_ONLY_PREFIX_SNAPSHOT',
        read_only=True,root=str(root),session=SESSION,started_at_utc=started,finished_at_utc=datetime.now(timezone.utc).isoformat(),
        M=m,R=r,counts=counts,expected_counts=expected,logical_requests=logical,expected_requests=total,
        handles_before=before,handles_after=after,process_and_tmux_absent=complete,pipeline_exit_code=0 if complete else None,
        complete_file_set_rechecked=complete,all_file_bytes_rechecked=complete,collected_live_prefixes_rechecked=not complete,
        dropped_incomplete_jsonl_tail_bytes=cuts,snapshot_is_atomic=complete,files=files,goal_complete=False,collector_source_sha256=sha(own_source),
        note='Read-only saved artifact export; no original execution paths/model weights/corpus access. Live snapshots are not atomic or final.')
    payloads['TERMINAL_COLLECTION.json' if complete else 'SNAPSHOT_COLLECTION.json']=(json.dumps(receipt,indent=2)+'\n').encode()
    payloads['EVIDENCE_MANIFEST.json']=(json.dumps(dict(schema=1,status='TERMINAL_RAW_EXPORT' if complete else 'PARTIAL_PREFIX_EXPORT',goal_complete=False,
        file_sha256={name:sha(raw) for name,raw in payloads.items()}),indent=2)+'\n').encode()
    require(Path(__file__).read_bytes()==own_source,'Collector source changed during export')
    return payloads,receipt


def write_archive(payloads,stream):
    with tarfile.open(fileobj=stream,mode='w|gz',format=tarfile.USTAR_FORMAT) as archive:
        for name,raw in sorted(payloads.items()):
            info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o444;archive.addfile(info,io.BytesIO(raw))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--terminal',action='store_true')
    args=parser.parse_args();payloads,_=export(ROOT,complete=args.terminal);write_archive(payloads,sys.stdout.buffer)


if __name__=='__main__':main()
