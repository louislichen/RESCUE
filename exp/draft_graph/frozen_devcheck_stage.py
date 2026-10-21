"""One-shot old-development staging/validation; receiving NEVER launches.

Only a SHA-pinned, fully verified disabled UNION capsule supplies scientific
source/evidence. No fresh-data argument exists. Build/receive never open model
assets; explicit --authorize-run is required for the separate run operation.
No SSH, installer, queue, retry, cleanup or experiment-resume path exists.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

ROOT=Path('/mnt/8T/cl/rescue_experiment_cache/frozen_devcheck_20261020_v1')
CAPSULE=Path('/mnt/8T/cl/rescue_experiment_cache/fair_union_confirmation_20261020_v1')
SESSION='rescue_frozen_devcheck_v1'
PYTHON=Path('/mnt/8T/cl/rescue_experiment_cache/specextend_match_20261020_v1/venv/bin/python')
GPU='GPU-9db4f7b5-1447-9a2a-5c1b-80379d909fd9'
STATUS='FROZEN_DEVCHECK_STAGED_NOT_LAUNCHED'
MANIFEST='LAUNCH_MANIFEST.json'
PREREG='exp/draft_graph/frozen_devcheck_20261020_preregistration.json'
EXPECTED=dict(own_quality_requests=64,quality_owners=64,prime_requests=2,timing_requests=64,owners=2)
ASSETS=dict(teacher='/mnt/8T/cl/models/Qwen2-7B-Instruct',student='/mnt/8T/cl/models/Qwen2-1.5B-Instruct',
    mapper='/home/wanghl/cl/RESCUE/exp/module_arch/out_v1_20261018/full_final.pt',
    policy='/home/wanghl/cl/RESCUE/exp/module_arch/out_v1_20261018/budget_selected.pt',
    cost='/home/wanghl/cl/RESCUE/exp/module_arch/out_v1_20261018/cost.json',
    prime_file='/home/wanghl/cl/RESCUE/exp/module_opt/data_v1_20261018/tune.jsonl')
ASSET_SHA=dict(mapper='d08889e796ca2527e23241ff34a072ef640e2fdf3fe9a2f9ba5439975af04d9f',
    policy='3c333a470119c236afcc1308c18592842a535bee0b8f83c4fb4da359ff17eff5',
    cost='d92c843a84676c0cd6adaf9364bbf7e8c771d56548f1016b5720a3e04f371fa1')
GATE_SHA={'rescue/__init__.py':'89eba56e38c2689930a5bf1ba4e0221ada1da541dc16e4a759156fa9dd7fe11a',
    'rescue/fair_union_confirmation_data.py':'9ef8b39565303661e865f7a97a6d9bdcbad7838c41fbb7c11cc0c5864654446e',
    'rescue/fair_confirmation_data.py':'0224aa0e03ec3c3867a43dbb1a05142f1886f22872b4a004447e4d20fa6ebba4',
    'rescue/baseline_horizon_data.py':'1b4699a9427e6584c5476f3e2ee0ccf3bdab20b75ca5fee3ed23eb107a04d4cd'}
TOOLS={
    'tools/frozen_devcheck_stage.py':'exp/draft_graph/frozen_devcheck_stage.py',
    'tools/collect_frozen_devcheck.py':'exp/draft_graph/collect_frozen_devcheck.py',
    'tools/frozen_devcheck.py':'exp/draft_graph/frozen_devcheck.py',
    'tools/report_frozen_devcheck.py':'exp/draft_graph/report_frozen_devcheck.py',
    'tools/fair_union_execution_support.py':'exp/draft_graph/fair_union_execution.py',
    'tools/preregistration.json':PREREG,
    'tests/test_rescue_frozen_devcheck_tools.py':'tests/test_rescue_frozen_devcheck_tools.py',
    'tests/test_rescue_frozen_devcheck.py':'tests/test_rescue_frozen_devcheck.py',
    'tests/test_rescue_frozen_devcheck_report.py':'tests/test_rescue_frozen_devcheck_report.py'}
REPORT_SUPPORT={'tools/analysis_support.py':'rescue/baseline_selection_report.py',
    'tools/analysis_horizon_support.py':'rescue/baseline_horizon_report.py',
    'tools/analysis_fair_support.py':'rescue/fair_confirmation_report.py',
    'tools/analysis_union_support.py':'rescue/fair_union_confirmation_report.py'}
SUFFIXES={'.py','.json','.jsonl','.md','.png','.svg','.sh','.log','.txt'}
SPECIAL={'terminal.tar.gz','evidence_snapshot.tar.gz','collection.stderr'}


def need(ok,message):
    if not ok:raise RuntimeError(message)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda:handle.read(8*1024*1024),b''):digest.update(part)
    return digest.hexdigest()


def valid_sha(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def relative(name):
    need(isinstance(name,str) and '\\' not in name and '\x00' not in name,'Non-POSIX artifact path')
    p=PurePosixPath(name)
    need(p.parts and p.as_posix()==name and not p.is_absolute() and '..' not in p.parts and
        not {'.git','.ssh','.env','__pycache__'}.intersection(p.parts) and
        (p.suffix.lower() in SUFFIXES or p.name in SPECIAL),'Unsafe/non-artifact capsule member: '+name)
    return Path(*p.parts)


def no_links(path):
    path=Path(path)
    need(not any(p.is_symlink() for p in reversed((path,*path.parents))),'Symlink in explicit artifact path')


def file(root,name):
    path=root/relative(name);no_links(path)
    need(path.resolve().is_relative_to(root.resolve()) and path.is_file(),'Missing/escaped regular artifact: '+name)
    return path


def read(path):return json.loads(Path(path).read_bytes())


def write(path,value):
    with Path(path).open('x') as h:json.dump(value,h,indent=2,allow_nan=False);h.write('\n');h.flush();os.fsync(h.fileno())


def load_module(path,name):
    previous=sys.dont_write_bytecode;sys.dont_write_bytecode=True
    try:
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    finally:sys.dont_write_bytecode=previous


def gate_helpers(capsule,manifest):
    """Only authenticated stdlib source is imported; no rescue model modules."""
    for name,expected in GATE_SHA.items():
        need(manifest['source_sha256'].get(name)==expected and sha(file(capsule,'after/'+name).read_bytes())==expected,'Known frozen gate source changed')
    receiver_name='exp/draft_graph/receive_fair_union_stage.py'
    need(sha(file(capsule,'after/'+receiver_name).read_bytes())==manifest['source_sha256'][receiver_name],'Receiver helper changed')
    receiver=load_module(capsule/'after'/receiver_name,'_devcheck_capsule_receiver_'+sha(str(capsule).encode())[:16])
    alias='_devcheck_gate_'+sha(str(capsule).encode())[:16]
    spec=importlib.util.spec_from_file_location(alias,capsule/'after/rescue/__init__.py',submodule_search_locations=[str(capsule/'after/rescue')])
    package=importlib.util.module_from_spec(spec);sys.modules[alias]=package
    previous=sys.dont_write_bytecode;sys.dont_write_bytecode=True
    try:
        spec.loader.exec_module(package);data=importlib.import_module(alias+'.fair_union_confirmation_data')
    finally:sys.dont_write_bytecode=previous
    return data,receiver


def capsule_state(capsule,expected):
    capsule=Path(capsule);no_links(capsule);capsule=capsule.resolve();need(capsule.is_dir() and valid_sha(expected),'Explicit pinned complete capsule required')
    raw=file(capsule,MANIFEST).read_bytes();need(sha(raw)==expected,'Disabled capsule manifest SHA changed')
    manifest=json.loads(raw)
    need(manifest.get('status')=='STAGED_DISABLED_NOT_LAUNCHED' and manifest.get('launcher_disabled') is True and manifest.get('launch_authorized') is False,
         'Only a verified still-disabled UNION capsule may supply runtime')
    paths=list(capsule.rglob('*'));need(not any(p.is_symlink() for p in paths),'Capsule symlink is forbidden')
    contents={str(p.relative_to(capsule)):p.read_bytes() for p in paths if p.is_file()}
    inventory=manifest['file_sha256'];need(set(contents)==set(inventory)|{MANIFEST},'Capsule has added/missing files')
    for name,digest in inventory.items():relative(name);need(valid_sha(digest) and sha(contents[name])==digest,'Capsule artifact SHA changed')
    data,receiver=gate_helpers(capsule,manifest);receiver.validate_manifest(manifest,contents)
    arguments={k:capsule/v for k,v in manifest['proof_arguments'].items()}
    proof=data.validate_union_freeze(**arguments)
    later=read(arguments['stage2_results']/'protocol.json');old=read(arguments['stage1_results']/'protocol.json')
    need(proof==manifest['union_freeze'] and data._sources(capsule/'after',later['source_sha256'])==manifest['scientific_source_sha256'],
         'Real complete dual proof or preserved scientific runtime changed')
    return dict(manifest=manifest,proof=proof,arguments=arguments,stage1=old,stage2=later,contents=contents)


def asset_contract(state):
    a,b=state['stage1'],state['stage2'];pins={}
    for role in ('teacher','student'):
        need(a['args'][role]==b['args'][role]==ASSETS[role],'Model root differs from sealed studies')
        subset={n:s for n,s in a['assets_sha256'].items() if Path(n).is_relative_to(Path(ASSETS[role]))}
        need(subset and all(b['assets_sha256'].get(n)==s for n,s in subset.items()),'Frozen model inventory is missing/changed')
        pins.update(subset)
    prime=ASSETS['prime_file'];need(valid_sha(a['assets_sha256'].get(prime)) and a['assets_sha256'][prime]==b['assets_sha256'].get(prime),'Old271 prime not frozen by both studies')
    pins[prime]=a['assets_sha256'][prime]
    pins.update({ASSETS[k]:v for k,v in ASSET_SHA.items()})
    return pins


def validate_manifest(manifest,contents):
    need(manifest.get('status')==STATUS and manifest.get('remote_root')==str(ROOT) and manifest.get('session')==SESSION and
        manifest.get('launch_authorized') is False and manifest.get('receiver_launches') is False and
        manifest.get('expected_counts')==EXPECTED and manifest.get('new_generation_requests')==130 and
        manifest.get('no_fresh_data') is manifest.get('no_training') is manifest.get('no_search') is True,'Wrong olddev-only staged protocol')
    pins=manifest['file_sha256'];need(set(contents)==set(pins)|{MANIFEST},'Unknown/missing staged member')
    for name,digest in pins.items():relative(name);need(valid_sha(digest) and sha(contents[name])==digest,'Staged inventory SHA changed')
    expected_tools=set(TOOLS)|set(REPORT_SUPPORT)
    need(set(manifest['tool_sha256'])==expected_tools and all(pins.get(n)==s for n,s in manifest['tool_sha256'].items()),'Unregistered/missing external tool/support')
    need(all(n.startswith('capsule/') or n in expected_tools for n in pins),'Unknown top-level staged payload')
    cap={n.removeprefix('capsule/'):v for n,v in contents.items() if n.startswith('capsule/')}
    cm=json.loads(cap[MANIFEST]);need(sha(cap[MANIFEST])==manifest['capsule_manifest_sha256'] and cm['union_freeze']==manifest['union_freeze'], 'Inherited proof manifest differs')
    need(set(cap)==set(cm['file_sha256'])|{MANIFEST} and all(sha(cap[n])==s for n,s in cm['file_sha256'].items()),'Inherited capsule bytes differ')
    need(manifest['assets']==ASSETS and all(manifest['asset_sha256'].get(ASSETS[k])==s for k,s in ASSET_SHA.items()),'Changed fixed asset roles')
    prereg=json.loads(contents['tools/preregistration.json'])
    need(prereg['status']=='TEMPLATE_NOT_EXECUTED' and prereg['remote_root']==str(ROOT) and prereg['session']==SESSION and
        prereg['execution']['total_new_generation_requests']==130 and prereg['truth']['no_new_confirmation_access'] is True and
        all(prereg['frozen_own_method'][k+'_sha256']==s for k,s in ASSET_SHA.items()),'Preregistered frozen130 protocol changed')
    for destination,original in REPORT_SUPPORT.items():need(sha(contents[destination])==cm['source_sha256'][original],'Report support differs from inherited runtime')


def build(repo,capsule,capsule_sha256,output):
    repo,capsule,output=Path(repo),Path(capsule),Path(output)
    for path in (repo,capsule,output):no_links(path)
    repo,capsule,output=repo.resolve(),capsule.resolve(),output.resolve()
    need(not output.exists() and all(not (output==p or output.is_relative_to(p) or p.is_relative_to(output)) for p in (repo,capsule)), 'Fresh disjoint local output only')
    state=capsule_state(capsule,capsule_sha256)  # Real dual proof before FIRST mkdir/copy.
    assets=asset_contract(state);contents={'capsule/'+n:v for n,v in state['contents'].items()}
    for destination,original in TOOLS.items():contents[destination]=file(repo,original).read_bytes()
    for destination,original in REPORT_SUPPORT.items():contents[destination]=state['contents']['after/'+original]
    manifest=dict(schema=1,status=STATUS,remote_root=str(ROOT),session=SESSION,launch_authorized=False,receiver_launches=False,
        capsule_manifest_sha256=capsule_sha256,union_freeze=state['proof'],expected_counts=EXPECTED,new_generation_requests=130,
        reused_reference_requests=34,generated_reference_requests=0,assets=ASSETS,asset_sha256=assets,
        scientific_source_sha256=state['manifest']['scientific_source_sha256'],
        tool_sha256={n:sha(contents[n]) for n in (*TOOLS,*REPORT_SUPPORT)},file_sha256={n:sha(v) for n,v in contents.items()},
        no_fresh_data=True,no_training=True,no_search=True,no_model_weights_transferred=True,no_goal_completion=True,
        created_at_utc=datetime.now(timezone.utc).isoformat())
    contents[MANIFEST]=(json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode();validate_manifest(manifest,contents)
    output.mkdir();payload=output/'payload';payload.mkdir()
    for name,raw in contents.items():
        path=payload/relative(name);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as h:h.write(raw)
        path.chmod(0o444)
    portable=capsule_state(payload/'capsule',capsule_sha256);need(portable['proof']==state['proof'],'Portable dual proof differs')
    transport=output/'transport.tar.gz'
    with tarfile.open(transport,'w:gz',format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(contents):archive.add(payload/name,arcname=name,recursive=False)
    receipt=dict(status=STATUS,transport_sha256=sha(transport.read_bytes()),launch_manifest_sha256=sha(contents[MANIFEST]),
        remote_root=str(ROOT),files=len(contents),launch_authorized=False,new_generation_requests=130)
    write(output/'TRANSPORT_RECEIPT.json',receipt);return receipt


def unpack(payload):
    contents={};cursor=0
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
        members=archive.getmembers();need(not archive.pax_headers,'PAX metadata forbidden')
        for entry in members:
            relative(entry.name);need(entry.isfile() and not entry.pax_headers and entry.name not in contents and entry.offset==cursor and entry.offset_data==cursor+512,'Link/duplicate/hidden/renaming tar header')
            archive.fileobj.seek(entry.offset+156);need(archive.fileobj.read(1) in (b'0',b'\0'),'Extended/sparse tar header')
            with archive.extractfile(entry) as h:contents[entry.name]=h.read()
            need(len(contents[entry.name])==entry.size,'Truncated artifact')
            archive.fileobj.seek(entry.offset_data+entry.size);need(not any(archive.fileobj.read((-entry.size)%512)),'Nonzero tar padding')
            cursor=entry.offset_data+((entry.size+511)//512)*512
        archive.fileobj.seek(cursor);tail=archive.fileobj.read();need(len(tail)>=1024 and not any(tail),'Hidden trailing tar bytes')
    for name in contents:
        parts=PurePosixPath(name).parts;need(not any(PurePosixPath(*parts[:i]).as_posix() in contents for i in range(1,len(parts))),'File/ancestor collision')
    need(MANIFEST in contents,'Missing staged manifest');validate_manifest(json.loads(contents[MANIFEST]),contents);return contents


def receive(expected,stream=None):
    need(valid_sha(expected),'Invalid expected transport SHA')
    raw=(sys.stdin.buffer if stream is None else stream).read();need(sha(raw)==expected,'Transport SHA mismatch')
    no_links(ROOT);need(not ROOT.exists(),'Never overwrite/clean a partial or existing devcheck root')
    contents=unpack(raw);manifest=json.loads(contents[MANIFEST])
    # The original already-received disabled capsule is a read-only prerequisite.
    # Real dual-proof validation must succeed BEFORE creating this new root.
    original=capsule_state(CAPSULE,manifest['capsule_manifest_sha256'])
    inherited={n.removeprefix('capsule/'):v for n,v in contents.items() if n.startswith('capsule/')}
    need(original['proof']==manifest['union_freeze'] and original['contents']==inherited,
        'Incoming capsule differs from the already verified disabled original')
    ROOT.mkdir()
    for name,value in contents.items():
        target=ROOT/relative(name);target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as h:h.write(value)
        target.chmod(0o444)
    need(all(file(ROOT,n).read_bytes()==v for n,v in contents.items()),'Received bytes changed; preserve partial root')
    state=capsule_state(ROOT/'capsule',json.loads(contents[MANIFEST])['capsule_manifest_sha256'])
    need(state['proof']==json.loads(contents[MANIFEST])['union_freeze'],'Received real dual proof differs')
    receipt=dict(status=STATUS,transport_sha256=expected,launch_manifest_sha256=sha(contents[MANIFEST]),files=len(contents),launch_authorized=False)
    write(ROOT/'STAGE_RECEIPT.json',receipt);return receipt


def stage_state(fresh=True):
    no_links(ROOT);need(ROOT.is_dir(),'Missing separately staged isolated root')
    manifest=read(file(ROOT,MANIFEST));paths=list(ROOT.rglob('*'));need(not any(p.is_symlink() for p in paths),'Symlink in staged root')
    if fresh:need({str(p.relative_to(ROOT)) for p in paths if p.is_file()}==set(manifest['file_sha256'])|{MANIFEST,'STAGE_RECEIPT.json'},'Partial/executed root; no resume')
    contents={n:file(ROOT,n).read_bytes() for n in (*manifest['file_sha256'],MANIFEST)};validate_manifest(manifest,contents)
    receipt=read(file(ROOT,'STAGE_RECEIPT.json'));need(receipt['status']==STATUS and receipt['launch_manifest_sha256']==sha(contents[MANIFEST]),'No authentic receive receipt')
    state=capsule_state(ROOT/'capsule',manifest['capsule_manifest_sha256']);need(state['proof']==manifest['union_freeze'] and asset_contract(state)==manifest['asset_sha256'],'Inherited dual proof/assets changed')
    return dict(manifest=manifest,capsule=state,manifest_sha256=sha(contents[MANIFEST]))


def environment():
    value={k:os.environ[k] for k in ('PATH','LANG','LC_ALL','TZ') if k in os.environ}
    value.update(CUDA_VISIBLE_DEVICES=GPU,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',
        PYTHONNOUSERSITE='1',PYTHONHASHSEED='20261027',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    return value


def support(state):
    path=ROOT/'tools/fair_union_execution_support.py'
    need(sha(path.read_bytes())==state['manifest']['tool_sha256']['tools/fair_union_execution_support.py'],'Execution support changed')
    return load_module(path,'_devcheck_fixed_probe_support')


def validate():
    """Read-only checks only. No directories, receipts, logs, CUDA or imports of report."""
    state=stage_state();helper=support(state);identity=helper.python_chain(check_binary=True)
    need(Path(sys.executable).absolute()==PYTHON,'Use the pinned isolated interpreter')
    check_assets(state)
    state.update(identity=identity,gpu=helper.probe_gpu());return state


def check_assets(state):
    for original,expected in state['manifest']['asset_sha256'].items():
        path=Path(original);no_links(path);need(path.is_file() and file_sha(path)==expected,'Frozen existing asset changed')


def logged(command,name):
    with (ROOT/name).open('x') as h:
        result=subprocess.run(command,cwd=ROOT/'capsule/after',env=environment(),stdout=h,stderr=subprocess.STDOUT)
    need(result.returncode==0,'Command failed; preserve isolated root and inspect '+name)


def run(authorize=False):
    need(authorize is True,'Run requires explicit --authorize-run; receiving/validation never launches')
    state=validate();helper=support(state)
    need(bool(os.environ.get('TMUX')) and subprocess.check_output(['tmux','display-message','-p','#S'],text=True).strip()==SESSION,'Use the exact persistent tmux session')
    lock=ROOT/'LAUNCH_ATTEMPT.json'
    write(lock,dict(status='ONCE_ONLY_LAUNCH_ATTEMPT',launch_manifest_sha256=state['manifest_sha256'],pid=os.getpid(),session=SESSION,authorized_explicitly=True))
    links=dict(launch_manifest_sha256=state['manifest_sha256'],launch_attempt_sha256=sha(lock.read_bytes()))
    try:
        write(ROOT/'RUN_STARTED.json',dict(status='RUN_STARTED',session=SESSION,pid=os.getpid(),**links))
        code="""import json,platform,torch,transformers,pyarrow,flash_attn
from rescue.baseline_selection import flash_receipt
print('PREFLIGHT_RECEIPT='+json.dumps(dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,transformers=transformers.__version__,pyarrow=pyarrow.__version__,flash_attn=flash_attn.__version__,flash_receipt=flash_receipt())),flush=True)
"""
        logged([str(PYTHON),'-c',code],'dependency_preflight.log')
        receipts=[json.loads(v.split('PREFLIGHT_RECEIPT=',1)[1]) for v in (ROOT/'dependency_preflight.log').read_text().splitlines() if v.startswith('PREFLIGHT_RECEIPT=')]
        need(len(receipts)==1,'Missing successful dependency receipt');value=receipts[0];old=state['capsule']['stage1'];later=state['capsule']['stage2']
        need(value['python']=='3.10.20' and value['torch']==old['environment']['torch']==later['environment']['torch'] and value['cuda']==old['environment']['cuda']==later['environment']['cuda'] and
            value['transformers']=='4.40.2' and value['pyarrow']=='24.0.0' and value['flash_attn']=='2.8.3' and
            value['flash_receipt']==old['flash_attention']==later['flash_attention'],'Installed frozen dependencies changed')
        write(ROOT/'DEPENDENCY_PREFLIGHT.json',dict(passed=True,no_installation=True,versions=value,**links))
        gpu_code="""import json,sys,unittest,torch
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'tests'))
assert torch.cuda.is_available() and torch.cuda.device_count()==1
names=json.loads(sys.argv[1]);r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
assert r.testsRun==len(names) and r.wasSuccessful() and not r.skipped
print('GPU_PREFLIGHT_PASSED',flush=True)
"""
        helper.probe_gpu();logged([str(PYTHON),'-c',gpu_code,json.dumps(helper.GPU_CASES)],'gpu_preflight.log')
        write(ROOT/'GPU_PREFLIGHT.json',dict(passed=True,tests_run=len(helper.GPU_CASES),skipped=0,actual_gpu_cases=list(helper.GPU_CASES),gpu_uuid=GPU,**links))
        current=stage_state(fresh=False);need(current['manifest_sha256']==state['manifest_sha256'],'Staged bytes changed before scientific launch');helper.probe_gpu()
        args={k:ROOT/'capsule'/v for k,v in state['capsule']['manifest']['proof_arguments'].items()}
        args.update(source_root=ROOT/'capsule/after',**{k:Path(v) for k,v in ASSETS.items()},output=ROOT/'results')
        need(not args['output'].exists(),'Never precreate/resume scientific result root')
        command=[str(PYTHON),str(ROOT/'tools/frozen_devcheck.py')]
        for name,path in args.items():command.extend(('--'+name.replace('_','-'),str(path)))
        logged(command,'benchmark.log');complete=read(ROOT/'results/COMPLETE.json')
        need(complete.get('passed') is True and complete.get('counts')==EXPECTED and complete.get('logical_requests')==130 and complete.get('goal_complete') is False,'Incomplete130 result; no automatic restart')
        stage_state(fresh=False);check_assets(state);helper.python_chain(check_binary=True)
        write(ROOT/'PIPELINE_EXIT.json',dict(exit_code=0,results_complete_sha256=sha((ROOT/'results/COMPLETE.json').read_bytes()),analysis_required_locally=True,goal_complete=False,**links))
        return dict(status='DEVCHECK_FINISHED_ANALYSIS_REQUIRED',counts=EXPECTED,logical_requests=130,goal_complete=False)
    except BaseException as error:
        if not (ROOT/'PIPELINE_EXIT.json').exists():write(ROOT/'PIPELINE_EXIT.json',dict(exit_code=1,error_type=type(error).__name__,error=str(error),goal_complete=False,**links))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    b=sub.add_parser('build')
    for name in ('repo','capsule','output'):b.add_argument('--'+name,type=Path,required=True)
    b.add_argument('--capsule-sha256',required=True)
    r=sub.add_parser('receive');r.add_argument('--sha256',required=True)
    sub.add_parser('validate-only');r=sub.add_parser('run');r.add_argument('--authorize-run',action='store_true')
    args=vars(parser.parse_args());mode=args.pop('mode')
    if mode=='build':value=build(**args)
    elif mode=='receive':value=receive(args['sha256'])
    elif mode=='validate-only':s=validate();value=dict(status='VALIDATED_NOT_LAUNCHED',launch_manifest_sha256=s['manifest_sha256'])
    else:value=run(args['authorize_run'])
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
