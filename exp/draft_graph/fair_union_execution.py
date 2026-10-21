"""One-shot v2 execution using a read-only, disabled v1 scientific capsule.

Stdlib-only orchestration. Validation never hashes/opens candidate Parquet.
This file and its launcher have their own manifest; they are not represented
as part of rescue.fair_union_confirmation_data._sources(). No SSH, installs,
VPN operations, automatic retries, dataset replacements or cleanup exist.
"""
import sys
sys.dont_write_bytecode = True

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import tarfile


ROOT = Path('/mnt/8T/cl/rescue_experiment_cache/fair_union_confirmation_20261020_v2')
CAPSULE = Path('/mnt/8T/cl/rescue_experiment_cache/fair_union_confirmation_20261020_v1')
PYTHON = Path('/mnt/8T/cl/rescue_experiment_cache/specextend_match_20261020_v1/venv/bin/python')
PYTHON_ENV = Path('/home/wanghl/miniconda3/envs/rescue')
PYTHON_LINK = PYTHON_ENV/'bin/python'
PYTHON_REAL = PYTHON_ENV/'bin/python3.10'
PYTHON_SHA = '8119915f96beb2ae0797f1b92434b15f4a0f4a55900297fe3c713c3f1e5a5a3d'
GPU = 'GPU-9db4f7b5-1447-9a2a-5c1b-80379d909fd9'
SESSION = 'rescue_fair_union_confirmation_v2'
ENTRY = 'fair_union_execution.py'
LAUNCHER = 'run_fair_union_confirmation_20261020_v2.sh'
HELPERS = ('rescue/__init__.py', 'rescue/fair_union_confirmation_data.py',
    'rescue/fair_confirmation_data.py', 'rescue/baseline_horizon_data.py',
    'exp/draft_graph/receive_fair_union_stage.py')
METADATA = ('exp/long_system/data/train.jsonl', 'exp/long_system/data/development.jsonl',
    'exp/long_system/data/test.jsonl', 'exp/module_opt/data_v1_20261018/fit.jsonl',
    'exp/module_opt/data_v1_20261018/tune.jsonl', 'exp/module_opt/data_v1_20261018/warmup.jsonl',
    'exp/module_opt/data_v1_20261018/confirmation.jsonl', 'exp/module_arch/data_v1_20261018/fit.jsonl',
    'exp/module_arch/data_v1_20261018/tune.jsonl', 'exp/module_arch/data_v1_20261018/warmup.jsonl',
    'exp/module_arch/data_v1_20261018/confirmation.jsonl')
GPU_CASES = (
    'test_rescue_draft_runtime.CudaGraphDraftRuntimeTests.test_real_capture_replay_cap_boundaries_workset_changes_and_request_reset',
    'test_rescue_draft_runtime.CudaGraphDraftRuntimeTests.test_graph_eos_and_short_output_boundaries',
    'test_rescue_draft_runtime.CudaGraphDraftRuntimeTests.test_graph_all_accept_catchup_and_padding_poison',
    'test_rescue_draft_runtime.CudaGraphDraftRuntimeTests.test_graph_explicit_rollback_zero_through_four_preserves_every_layer',
    'test_rescue_draft_runtime.CudaGraphDraftRuntimeTests.test_graph_fp32_target_bfloat16_draft_with_nonzero_float_mapper',
    'test_rescue_baseline_runtime.GraphBaselineTests.test_graph_variable_length_geometry_full_kv_and_queries_match_eager',
    'test_rescue_baseline_runtime.GraphBaselineTests.test_graph_fixed_k12_eos_and_all_accept_catchup',
    'test_rescue_baseline_runtime.GraphBaselineTests.test_graph_retrieval_shared_model_owner_crossover')
DEPENDENCIES = ('python', 'torch', 'cuda', 'transformers', 'pyarrow', 'flash_attn')
ASSET_ROLES = ('teacher', 'student', 'mapper', 'policy', 'cost', 'prime_file')
PLAN_KEYS = {'schema', 'status', 'execution_authorized', 'execution_root', 'capsule_root', 'python',
    'python_sha256', 'expected_gpu_uuid', 'session', 'capsule_transport', 'capsule_manifest_sha256',
    'tools_sha256', 'dependencies', 'assets', 'old_metadata', 'raw_source', 'raw_source_manifest'}


def require(ok, message):
    if not ok: raise RuntimeError(message)


def utc(): return datetime.now(timezone.utc).isoformat()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''): result.update(block)
    return result.hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def read(path):
    value = json.loads(Path(path).read_text()); require(isinstance(value, dict), 'Expected JSON object'); return value


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False); handle.write('\n'); handle.flush(); os.fsync(handle.fileno())


def absolute(value):
    require(isinstance(value, str) and Path(value).is_absolute() and '\x00' not in value, 'Explicit absolute path required')
    return Path(value)


def canonical(value):
    """Lexical only: in particular never resolve/stat the candidate path."""
    path=absolute(value)
    require('\\' not in value and not value.startswith('//') and '..' not in path.parts and
        PurePosixPath(value).as_posix()==value,'Non-canonical path/alias in explicit plan')
    return path


def pre_data_boundaries(plan,plan_path):
    """Reject role aliases before any planned file is hashed or probed."""
    raw=canonical(plan['raw_source']['path'])
    require(raw.name=='validation-00000-of-00001.parquet','Explicit original candidate filename required')
    require(set(plan['assets'])==set(ASSET_ROLES) and set(plan['old_metadata'])==set(METADATA),
        'Explicit assets and all eleven metadata paths required')
    roles=[canonical(str(plan_path)),canonical(str(ROOT)),canonical(str(CAPSULE)),canonical(str(PYTHON)),canonical(str(PYTHON.parent.parent)),
        canonical(str(PYTHON_ENV)),canonical(str(PYTHON_LINK)),canonical(str(PYTHON_REAL)),
        canonical(str(ROOT/'tools'/ENTRY)),canonical(str(ROOT/'tools'/LAUNCHER)),canonical(str(CAPSULE/'LAUNCH_MANIFEST.json')),
        canonical(plan['capsule_transport']['path']),canonical(plan['raw_source_manifest']['path'])]
    roles.extend(canonical(value) for value in plan['assets'].values())
    old_roots=set()
    for name,entry in plan['old_metadata'].items():
        require(isinstance(entry,dict) and set(entry)=={'path','sha256'},'Incomplete old metadata pin')
        path=canonical(entry['path']);relative=Path(name)
        require(path.parts[-len(relative.parts):]==relative.parts,'Wrong exact old cohort relative path')
        old_roots.add(Path(*path.parts[:-len(relative.parts)]));roles.append(path)
    require(len(old_roots)==1,'All eleven metadata files must share one exact old repository root')
    old_root=old_roots.pop();roles.append(old_root)
    require(canonical(plan['raw_source_manifest']['path'])==old_root/'exp/module_arch/data_v1_20261018/manifest.json',
        'Inherited raw manifest must use the same exact old repository root')
    require(all(not (raw==path or raw.is_relative_to(path) or path.is_relative_to(raw)) for path in roles),
        'Candidate path aliases/overlaps a readable execution role')
    # Only non-candidate roles are inspected. Reject leaf OR ancestor symlinks
    # without resolving them or following a link to candidate bytes.
    for path in roles:
        # Parent-first checks never traverse an unapproved ancestor link.
        require(not any(p.is_symlink() for p in reversed((path,*path.parents)) if p not in (PYTHON,PYTHON_LINK)),
            'Symlink in a readable execution role')
    python_chain(check_binary=False)


def python_chain(*,check_binary):
    """Only the two independently inspected interpreter leaf links may exist."""
    for path in (PYTHON,PYTHON_LINK,PYTHON_REAL):
        require(not any(p.is_symlink() for p in reversed(path.parents)),'Interpreter parent symlink is forbidden')
    require(PYTHON.is_symlink() and os.readlink(PYTHON)==str(PYTHON_LINK),'Fixed isolated interpreter first link changed')
    require(PYTHON_LINK.is_symlink() and os.readlink(PYTHON_LINK)=='python3.10','Fixed isolated interpreter second link changed')
    require(not PYTHON_REAL.is_symlink() and PYTHON_REAL.is_file(),'Fixed real interpreter is not a regular file')
    if check_binary:require(sha(PYTHON_REAL)==PYTHON_SHA,'Fixed real interpreter binary SHA changed')
    return dict(status='PINNED_PYTHON_CHAIN_VERIFIED',links=[dict(path=str(PYTHON),readlink=str(PYTHON_LINK)),
        dict(path=str(PYTHON_LINK),readlink='python3.10')],real_binary_path=str(PYTHON_REAL),real_binary_sha256=PYTHON_SHA)


def plan_template():
    """Unusable placeholders only; accurate external paths/pins are mandatory."""
    return dict(schema=1, status='TEMPLATE_NOT_EXECUTED', execution_authorized=False,
        execution_root=str(ROOT), capsule_root=str(CAPSULE), python=str(PYTHON), python_sha256=None,
        expected_gpu_uuid=GPU, session=SESSION, capsule_transport=dict(path=None, sha256=None),
        capsule_manifest_sha256=None, tools_sha256={'tools/'+ENTRY:None, 'tools/'+LAUNCHER:None},
        dependencies={k:None for k in DEPENDENCIES}, assets={k:None for k in ASSET_ROLES},
        old_metadata={n:dict(path=None, sha256=None) for n in METADATA},
        raw_source=dict(path=None, expected_sha256=None), raw_source_manifest=dict(path=None, sha256=None))


def environment():
    # Explicit allowlist: credentials/tokens are neither copied into child
    # environments nor printed/stored in execution receipts.
    result = {k:os.environ[k] for k in ('PATH', 'LANG', 'LC_ALL', 'TZ') if k in os.environ}
    result.update(CUDA_VISIBLE_DEVICES=GPU, PYTHONHASHSEED='20261026', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    return result


def probe_identity(plan):
    require(Path(sys.executable).absolute() == PYTHON and PYTHON.is_file() and os.access(PYTHON, os.X_OK), 'Wrong fixed isolated interpreter')
    chain=python_chain(check_binary=True)
    require(plan['python_sha256']==PYTHON_SHA,'Plan changed the pinned real interpreter SHA')
    value = subprocess.check_output([str(PYTHON), '-c', 'import platform;print(platform.python_version())'], env=environment(), text=True).strip()
    require(value == plan['dependencies']['python'], 'Python version differs from explicit plan')
    return dict(python=str(PYTHON), python_sha256=plan['python_sha256'], version=value,python_chain=chain)


def probe_gpu():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total', '--format=csv,noheader,nounits'], text=True).strip()
    rows = [r.split(',') for r in raw.splitlines() if r.strip()]
    require(len(rows) == 1 and len(rows[0]) == 4 and rows[0][1].strip() == GPU and 'H100' in rows[0][2] and
        int(rows[0][3].strip()) >= 80_000, 'Require the exact single H10080GB UUID')
    jobs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True).strip()
    require(not jobs, 'GPU busy; no process was interrupted')
    return dict(nvidia_smi_gpu=raw, nvidia_smi_compute=jobs, gpu_uuid=GPU, idle=True)


def probe_session():
    require(bool(os.environ.get('TMUX')), 'Formal execution must use the fixed persistent tmux session')
    name=subprocess.check_output(['tmux','display-message','-p','#S'],text=True).strip()
    require(name==SESSION,'Wrong persistent tmux session');return name


def own_sources(plan):
    tools = {'tools/'+ENTRY:ROOT/'tools'/ENTRY, 'tools/'+LAUNCHER:ROOT/'tools'/LAUNCHER}
    require(Path(__file__).resolve() == tools['tools/'+ENTRY], 'Entry must execute from the fixed v2/tools path')
    require(set(plan['tools_sha256']) == set(tools), 'Exactly the two own execution sources must be pinned')
    for name, path in tools.items():
        require(path.is_file() and not path.is_symlink() and valid_sha(plan['tools_sha256'][name]) and sha(path) == plan['tools_sha256'][name],
            'Own execution source changed')
    return tools


def load_helpers(manifest):
    """Only five explicitly capsule-SHA-pinned stdlib helper files are used."""
    for name in HELPERS:
        path = CAPSULE/'after'/name
        require(manifest['source_sha256'].get(name) == sha(path), 'Pinned stdlib helper changed')
    path = CAPSULE/'after/exp/draft_graph/receive_fair_union_stage.py'
    spec = importlib.util.spec_from_file_location('frozen_union_receiver_helper', path)
    receiver = importlib.util.module_from_spec(spec); spec.loader.exec_module(receiver)
    require(receiver.ROOT == CAPSULE, 'Receiver helper has a different fixed capsule root')
    sys.path.insert(0, str(CAPSULE/'after'))
    data = importlib.import_module('rescue.fair_union_confirmation_data')
    for name in ('rescue', 'rescue.fair_union_confirmation_data', 'rescue.fair_confirmation_data', 'rescue.baseline_horizon_data'):
        module = importlib.import_module(name)
        relative = 'rescue/__init__.py' if name == 'rescue' else name.replace('.', '/')+'.py'
        require(Path(module.__file__).resolve() == CAPSULE/'after'/relative, 'Foreign stdlib gate namespace')
    return data, receiver


def verify_capsule(plan):
    manifest_path = CAPSULE/'LAUNCH_MANIFEST.json'
    require(not any(p.is_symlink() for p in reversed((CAPSULE,*CAPSULE.parents))), 'Symlink in capsule path')
    require(CAPSULE.is_dir(), 'Missing pinned v1 capsule')
    require(not any(p.is_symlink() for p in CAPSULE.rglob('*')), 'Symlink in capsule before manifest/helper access')
    require(sha(manifest_path) == plan['capsule_manifest_sha256'], 'Missing/changed pinned v1 capsule')
    manifest = read(manifest_path)
    require(manifest.get('status') == 'STAGED_DISABLED_NOT_LAUNCHED' and manifest.get('launch_authorized') is False and manifest.get('launcher_disabled') is True,
        'Only an untouched disabled v1 capsule may supply scientific source')
    data, receiver = load_helpers(manifest)
    transport = absolute(plan['capsule_transport']['path'])
    require(not transport.is_relative_to(CAPSULE) and not transport.is_relative_to(ROOT) and transport.is_file() and not transport.is_symlink() and
        sha(transport) == plan['capsule_transport']['sha256'], 'Missing/changed explicit independent capsule transport')
    contents, cursor = {}, 0
    with tarfile.open(transport, 'r:gz') as archive:
        members = archive.getmembers(); require(not archive.pax_headers, 'Unexpected transport PAX metadata')
        for item in members:
            receiver.relative(item.name)
            require(item.isfile() and not item.pax_headers and item.name not in contents and item.offset == cursor and item.offset_data == cursor+512,
                'Unexpected transport alias/link/hidden header')
            archive.fileobj.seek(item.offset+156); require(archive.fileobj.read(1) in (b'0', b'\x00'), 'Extended transport header')
            with archive.extractfile(item) as handle: contents[item.name] = handle.read()
            require(len(contents[item.name]) == item.size, 'Truncated transport member')
            archive.fileobj.seek(item.offset_data+item.size)
            require(not any(archive.fileobj.read((-item.size) % 512)), 'Uninventoried transport padding')
            cursor = item.offset_data+((item.size+511)//512)*512
        archive.fileobj.seek(cursor); trailing = archive.fileobj.read()
        require(len(trailing) >= 1024 and not any(trailing), 'Uninventoried trailing transport data')
    require(hashlib.sha256(contents['LAUNCH_MANIFEST.json']).hexdigest() == plan['capsule_manifest_sha256'], 'Transport/capsule manifest mismatch')
    receiver.validate_manifest(manifest, contents)
    paths = list(CAPSULE.rglob('*')); require(not any(p.is_symlink() for p in paths), 'Symlink in capsule')
    files = {p.relative_to(CAPSULE).as_posix():p for p in paths if p.is_file()}
    require(set(files) == set(contents), 'Disabled capsule has added/missing artifacts')
    for name, path in files.items():
        require(sha(path) == hashlib.sha256(contents[name]).hexdigest(), 'Capsule byte differs from original pinned transport')
    arguments = {k:CAPSULE/v for k,v in manifest['proof_arguments'].items()}
    proof = data.validate_union_freeze(**arguments)
    require(proof == manifest['union_freeze'] and data._sources(CAPSULE/'after', read(arguments['stage2_results']/'protocol.json')['source_sha256']) == manifest['scientific_source_sha256'],
        'Complete dual proof or frozen stage2 scientific runtime changed')
    return dict(data=data, manifest=manifest, proof=proof, arguments=arguments,
        stage1=read(arguments['stage1_results']/'protocol.json'), stage2=read(arguments['stage2_results']/'protocol.json'))


def model_files(root):
    root = Path(root); suffixes = {'.json', '.model', '.safetensors', '.bin', '.txt'}
    paths = list(root.rglob('*'))
    require(not any(p.is_symlink() for p in paths), 'Symlink in frozen model inventory')
    files = sorted(p for p in paths if p.is_file() and p.suffix in suffixes and not any(v.startswith('.') for v in p.relative_to(root).parts))
    require((root/'config.json').is_file() and any(p.suffix in ('.safetensors', '.bin') for p in files), 'Frozen local model/config missing')
    return files


def verify_inputs(plan, state):
    data, old, later = state['data'], state['stage1'], state['stage2']
    require(set(plan['assets']) == set(ASSET_ROLES) and set(plan['old_metadata']) == set(METADATA), 'Explicit assets and all eleven metadata paths required')
    assets = {k:absolute(v) for k,v in plan['assets'].items()}; pins = {}
    for role in ('teacher', 'student'):
        require(str(assets[role]) == old['args'][role] == later['args'][role], 'Model root differs from completed studies')
        for path in model_files(assets[role]):
            expected = old['assets_sha256'].get(str(path)); require(valid_sha(expected) and expected == later['assets_sha256'].get(str(path)) and sha(path) == expected,
                'Frozen model/tokenizer file changed'); pins[str(path)] = expected
    for role in ('mapper', 'policy', 'cost'):
        expected = data.OWN_METHOD[role+'_sha256']; require(sha(assets[role]) == expected, 'Frozen Mapper/policy/old-cost asset changed'); pins[str(assets[role])] = expected
    metadata = []
    for relative, entry in plan['old_metadata'].items():
        require(isinstance(entry, dict) and set(entry) == {'path', 'sha256'}, 'Incomplete old metadata pin')
        path = absolute(entry['path']); require(path.as_posix().endswith('/'+relative) and valid_sha(entry['sha256']) and sha(path) == entry['sha256'],
            'Old metadata identity changed'); metadata.append(path); pins[str(path)] = entry['sha256']
    data.old.exclusion_receipt(metadata)
    prime = plan['old_metadata']['exp/module_opt/data_v1_20261018/tune.jsonl']
    require(str(assets['prime_file']) == prime['path'] and old['assets_sha256'].get(prime['path']) == later['assets_sha256'].get(prime['path']) == prime['sha256'],
        'Old271 prime differs from both verified studies')
    raw_manifest = plan['raw_source_manifest']; require(set(raw_manifest) == {'path', 'sha256'}, 'Explicit inherited raw manifest pin required')
    path = absolute(raw_manifest['path']); require(path.as_posix().endswith('/exp/module_arch/data_v1_20261018/manifest.json') and
        valid_sha(raw_manifest['sha256']) and sha(path) == raw_manifest['sha256'], 'Inherited raw manifest changed')
    raw = plan['raw_source']; require(set(raw) == {'path', 'expected_sha256'} and absolute(raw['path']).name == 'validation-00000-of-00001.parquet' and
        valid_sha(raw['expected_sha256']) and read(path).get('source_sha256') == raw['expected_sha256'], 'Explicit candidate path/inherited SHA required')
    require(not any(Path(raw['path']).is_relative_to(p) for p in (ROOT,CAPSULE,*assets.values())),
        'Candidate path overlaps execution/capsule/frozen assets')
    # No exists/stat/resolve/hash/open of raw['path'] occurs in validation.
    pins[str(path)] = raw_manifest['sha256']
    return assets, metadata, pins


def validate(plan_path, *, fresh=True):
    plan_path = Path(plan_path); require(plan_path.is_absolute() and plan_path == ROOT/'EXECUTION_PLAN.json', 'Plan must be fixed ROOT/EXECUTION_PLAN.json')
    require(not any(p.is_symlink() for p in reversed((plan_path,*plan_path.parents))), 'Symlink in execution plan path')
    plan = read(plan_path)
    require(set(plan) == PLAN_KEYS and plan['schema'] == 1 and plan['status'] == 'FROZEN_EXECUTION_PLAN_NOT_EXECUTED' and plan['execution_authorized'] is True,
        'Missing/unknown/unapproved execution plan fields')
    require(plan['execution_root'] == str(ROOT) and plan['capsule_root'] == str(CAPSULE) and plan['python'] == str(PYTHON) and
        plan['session'] == SESSION and plan['expected_gpu_uuid'] == GPU, 'Fixed root/runtime/GPU/session changed')
    require(isinstance(plan['dependencies'],dict) and set(plan['dependencies']) == set(DEPENDENCIES) and all(isinstance(v,str) and v for v in plan['dependencies'].values()) and
        valid_sha(plan['python_sha256']) and valid_sha(plan['capsule_manifest_sha256']) and set(plan['capsule_transport']) == {'path','sha256'} and
        valid_sha(plan['capsule_transport']['sha256']), 'Missing explicit runtime/transport dependency pins')
    pre_data_boundaries(plan,plan_path)
    require(ROOT.is_dir() and not ROOT.is_symlink(), 'Stage only the new v2/tools and explicit plan first')
    if fresh:
        require({p.name for p in ROOT.iterdir()} == {'tools','EXECUTION_PLAN.json'} and
            {p.name for p in (ROOT/'tools').iterdir()} == {ENTRY,LAUNCHER}, 'Existing/partial execution state; never resume or overwrite')
    tools = own_sources(plan); identity = probe_identity(plan); gpu = probe_gpu(); state = verify_capsule(plan)
    require(gpu['nvidia_smi_gpu'] == state['stage1']['environment']['nvidia_smi_gpu'] == state['stage2']['environment']['nvidia_smi_gpu'], 'GPU geometry differs from both studies')
    require(plan['dependencies']['torch'] == state['stage2']['environment']['torch'] and plan['dependencies']['cuda'] == state['stage2']['environment']['cuda'] and
        plan['dependencies']['flash_attn'] == '2.8.3', 'Dependency plan differs from completed study')
    assets, metadata, pins = verify_inputs(plan,state)
    source_paths = {'execution_source/tools/'+Path(n).name:p for n,p in tools.items()}
    source_paths.update({'execution_source/helpers/'+n:CAPSULE/'after'/n for n in HELPERS})
    source_sha = {n:sha(p) for n,p in source_paths.items()}
    return dict(state,plan=plan,plan_path=plan_path,plan_sha256=sha(plan_path),tools=tools,identity=identity,gpu=gpu,
        assets=assets,metadata=metadata,input_pins=pins,source_paths=source_paths,source_sha256=source_sha)


def unchanged(state):
    require(sha(state['plan_path']) == state['plan_sha256'], 'Execution plan changed')
    for name,path in state['source_paths'].items(): require(sha(path) == state['source_sha256'][name], 'Execution source/helper changed')
    for path,digest in state['input_pins'].items(): require(sha(path) == digest, 'Frozen input asset/metadata changed')
    require(python_chain(check_binary=True)==state['identity']['python_chain'],'Fixed interpreter chain/binary changed')
    require(verify_capsule(state['plan'])['proof'] == state['proof'], 'Disabled capsule/dual proof changed')


def logged(command, name):
    with (ROOT/name).open('x') as handle:
        result = subprocess.run(command, cwd=CAPSULE/'after', env=environment(), stdout=handle, stderr=subprocess.STDOUT)
    require(result.returncode == 0, 'Preflight/generation command failed; inspect '+name)


def marker(log):
    values = [json.loads(line.split('PREFLIGHT_RECEIPT=',1)[1]) for line in (ROOT/log).read_text().splitlines() if line.startswith('PREFLIGHT_RECEIPT=')]
    require(len(values) == 1, 'Exactly one successful dependency/GPU preflight receipt required'); return values[0]


def receipt_links(state):
    return dict(execution_manifest_sha256=state['execution_manifest_sha256'],plan_sha256=state['plan_sha256'],
        capsule_manifest_sha256=state['plan']['capsule_manifest_sha256'])


def verify_preflights(state):
    dependency=read(ROOT/'DEPENDENCY_PREFLIGHT.json');gpu=read(ROOT/'GPU_PREFLIGHT.json')
    links=receipt_links(state)
    require(dependency.get('status')=='DEPENDENCY_PREFLIGHT_PASSED' and dependency.get('passed') is True and
        dependency.get('no_installation') is True and all(dependency.get(k)==v for k,v in links.items()) and
        {k:dependency.get('versions',{}).get(k) for k in DEPENDENCIES}==state['plan']['dependencies'] and
        dependency.get('versions',{}).get('flash_receipt')==state['stage1']['flash_attention']==state['stage2']['flash_attention'],
        'Complete linked dependency preflight required')
    require(gpu.get('status')=='GPU_PREFLIGHT_PASSED' and gpu.get('passed') is True and gpu.get('tests_run')==len(GPU_CASES) and
        gpu.get('skipped')==0 and gpu.get('actual_gpu_cases')==list(GPU_CASES) and gpu.get('physical_gpu_uuid')==GPU and
        all(gpu.get(k)==v for k,v in links.items()),'Complete linked real CUDA preflight required')
    return {name:sha(ROOT/name) for name in ('DEPENDENCY_PREFLIGHT.json','GPU_PREFLIGHT.json')}


def dependency_preflight(state):
    code = """import json,platform,torch,transformers,pyarrow,flash_attn
from rescue.baseline_selection import flash_receipt
print('PREFLIGHT_RECEIPT='+json.dumps(dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,transformers=transformers.__version__,pyarrow=pyarrow.__version__,flash_attn=flash_attn.__version__,flash_receipt=flash_receipt())),flush=True)
"""
    logged([str(PYTHON),'-c',code],'dependency_preflight.log'); value=marker('dependency_preflight.log')
    require({k:value[k] for k in DEPENDENCIES} == state['plan']['dependencies'] and value['flash_receipt'] == state['stage1']['flash_attention'] == state['stage2']['flash_attention'],
        'Installed isolated dependencies/actual Flash extension differ')
    write(ROOT/'DEPENDENCY_PREFLIGHT.json',dict(status='DEPENDENCY_PREFLIGHT_PASSED',passed=True,versions=value,no_installation=True,**receipt_links(state))); return value


def gpu_preflight(state):
    code = """import json,sys,unittest,torch
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'tests'))
assert torch.cuda.is_available() and torch.cuda.device_count()==1
p=torch.cuda.get_device_properties(0)
assert 'H100' in p.name and p.total_memory>=80000000000
names=json.loads(sys.argv[1]);suite=unittest.defaultTestLoader.loadTestsFromNames(names)
r=unittest.TextTestRunner(verbosity=2).run(suite)
assert r.testsRun==len(names) and r.wasSuccessful() and not r.skipped
print('PREFLIGHT_RECEIPT='+json.dumps(dict(passed=True,tests_run=r.testsRun,skipped=0,actual_gpu_cases=names)),flush=True)
"""
    probe_gpu(); logged([str(PYTHON),'-c',code,json.dumps(GPU_CASES)],'gpu_preflight.log'); value=marker('gpu_preflight.log')
    require(value == dict(passed=True,tests_run=len(GPU_CASES),skipped=0,actual_gpu_cases=list(GPU_CASES)), 'Real CUDA Graph preflight cases missing/skipped')
    write(ROOT/'GPU_PREFLIGHT.json',dict(value,status='GPU_PREFLIGHT_PASSED',physical_gpu_uuid=GPU,**receipt_links(state))); return value


def scientific(state):
    command=[str(PYTHON),'-m','rescue.fair_union_confirmation']
    for role,path in state['assets'].items(): command += ['--'+role.replace('_','-'),str(path)]
    command += ['--freeze-receipt',str(ROOT/'data_gate/GATE_SOURCE_FREEZE.json'),
        '--dataset-manifest',str(ROOT/'data_gate/data/DATASET_MANIFEST.json'),'--data-file',str(ROOT/'data_gate/data/confirmation.jsonl'),
        '--output',str(ROOT/'results')]
    probe_gpu(); logged(command,'benchmark.log')


def run(plan_path, *, validate_only=False):
    state=validate(plan_path)
    if validate_only:
        return dict(status='VALIDATED_ONLY',no_writes=True,no_candidate_content_accessed=True,expected_counts=state['proof']['expected_counts'],expected_requests=state['proof']['scientific_requests'])
    probe_session()
    lock=(ROOT/'pipeline.lock').open('x'); fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    phase='execution_manifest';manifest_sha=None;code=1
    def stop(signum,frame): raise SystemExit(128+signum)
    previous={n:signal.signal(n,stop) for n in (signal.SIGINT,signal.SIGTERM)}
    try:
        # A fresh lock is the first mutation. Repeat critical source/GPU checks
        # before starting any private preflight or opening candidate data.
        unchanged(state); probe_gpu()
        manifest=dict(schema=1,status='EXECUTION_FROZEN_BEFORE_DATA_ACCESS',created_at_utc=utc(),session=SESSION,
            execution_root=str(ROOT),capsule_root=str(CAPSULE),plan_path=str(state['plan_path']),plan_sha256=state['plan_sha256'],
            tools_sha256=state['plan']['tools_sha256'],source_sha256=state['source_sha256'],source_path_map={k:str(v) for k,v in state['source_paths'].items()},
            capsule_manifest_sha256=state['plan']['capsule_manifest_sha256'],capsule_transport_sha256=state['plan']['capsule_transport']['sha256'],
            capsule_transport_path=state['plan']['capsule_transport']['path'],capsule_saved_manifest='CAPSULE_LAUNCH_MANIFEST.json',union_freeze=state['proof'],expected_counts=state['proof']['expected_counts'],
            expected_requests=state['proof']['scientific_requests'],python=state['identity'],gpu=state['gpu'],gpu_preflight_cases=list(GPU_CASES),
            analysis_required_locally=True,goal_complete=False,scope='Independent v2 execution-source chain; not covered by scientific data._sources')
        write(ROOT/'EXECUTION_MANIFEST.json',manifest);manifest_sha=sha(ROOT/'EXECUTION_MANIFEST.json')
        state['execution_manifest_sha256']=manifest_sha
        write(ROOT/'RUN_STARTED.json',dict(status='RUN_STARTED',utc=utc(),orchestrator_pid=os.getpid(),session=SESSION,
            execution_manifest_sha256=manifest_sha,plan_sha256=state['plan_sha256'],goal_complete=False))
        started_sha=sha(ROOT/'RUN_STARTED.json')
        phase='execution_source_snapshot'
        with (ROOT/'CAPSULE_LAUNCH_MANIFEST.json').open('xb') as handle:handle.write((CAPSULE/'LAUNCH_MANIFEST.json').read_bytes())
        require(sha(ROOT/'CAPSULE_LAUNCH_MANIFEST.json')==state['plan']['capsule_manifest_sha256'],'Saved capsule manifest changed')
        for name,path in state['source_paths'].items():
            destination=ROOT/name;destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('xb') as handle: handle.write(path.read_bytes())
            require(sha(destination)==state['source_sha256'][name],'Execution-source snapshot changed')
            destination.chmod(0o444)
        phase='dependency_preflight';dependency_preflight(state)
        phase='gpu_preflight';gpu_preflight(state);preflight_pins=verify_preflights(state);unchanged(state)
        require(sha(ROOT/'EXECUTION_MANIFEST.json')==manifest_sha and sha(ROOT/'RUN_STARTED.json')==started_sha,'Execution freeze/start receipt changed')
        phase='formal_protocol'
        data=state['data'];template=CAPSULE/'after/exp/draft_graph/fair_union_confirmation_20261020_preregistration.json'
        data.write_json(ROOT/'frozen_protocol.json',data.make_final_protocol(template,state['proof']))
        phase='source_freeze'
        data.freeze_gate(ROOT/'data_gate',metadata_paths=state['metadata'],raw_source_manifest=Path(state['plan']['raw_source_manifest']['path']),
            raw_source=Path(state['plan']['raw_source']['path']),tokenizer_root=state['assets']['teacher'],source_root=CAPSULE/'after',
            final_protocol=ROOT/'frozen_protocol.json',**state['arguments'])
        phase='candidate_preparation';data.prepare_confirmation(ROOT/'data_gate/GATE_SOURCE_FREEZE.json')
        data.verify_dataset_manifest(ROOT/'data_gate/data')
        phase='scientific_generation';scientific(state)
        phase='terminal_verification';unchanged(state);data.validate_source_freeze(ROOT/'data_gate/GATE_SOURCE_FREEZE.json');data.verify_dataset_manifest(ROOT/'data_gate/data')
        require(sha(ROOT/'EXECUTION_MANIFEST.json')==manifest_sha and sha(ROOT/'RUN_STARTED.json')==started_sha,'Execution freeze/start receipt changed')
        require(verify_preflights(state)==preflight_pins and sha(ROOT/'CAPSULE_LAUNCH_MANIFEST.json')==state['plan']['capsule_manifest_sha256'],
            'Preflight/capsule-manifest receipt changed')
        complete=read(ROOT/'results/COMPLETE.json')
        require(complete.get('passed') is True and complete.get('counts') == state['proof']['expected_counts'] and
            complete.get('logical_requests') == state['proof']['scientific_requests'] and complete.get('goal_complete') is False and all(complete.get(k) is True for k in
            ('sources_unchanged','assets_unchanged','dependencies_unchanged','quality_gate_passed','all_canonical','owners_closed','data_unchanged','dual_study_freeze_unchanged')),
            'Scientific run has no complete exact immutable execution matrix')
        for name,digest in state['source_sha256'].items():require(sha(ROOT/name)==digest,'Saved execution source changed')
        write(ROOT/'FINAL_EXECUTION_VERIFIED.json',dict(status='EXECUTION_FINISHED_ANALYSIS_REQUIRED',passed=True,utc=utc(),
            execution_manifest_sha256=manifest_sha,plan_sha256=state['plan_sha256'],counts=complete['counts'],
            logical_requests=complete['logical_requests'],results_complete_sha256=sha(ROOT/'results/COMPLETE.json'),
            dependency_preflight_sha256=preflight_pins['DEPENDENCY_PREFLIGHT.json'],gpu_preflight_sha256=preflight_pins['GPU_PREFLIGHT.json'],
            execution_sources_unchanged=True,capsule_unchanged=True,analysis_required_locally=True,goal_complete=False))
        phase='EXECUTION_FINISHED_ANALYSIS_REQUIRED';code=0
        return dict(status=phase,analysis_required_locally=True,goal_complete=False)
    except BaseException as error:
        code=int(error.code) if isinstance(error,SystemExit) and isinstance(error.code,int) else 130 if isinstance(error,KeyboardInterrupt) else 1
        write(ROOT/'FAILURE.json',dict(status='FAILED_PRESERVE_PARTIAL',utc=utc(),phase=phase,error_type=type(error).__name__,error=str(error),
            execution_manifest_sha256=manifest_sha,plan_sha256=state['plan_sha256'],no_automatic_retry=True,goal_complete=False))
        raise
    finally:
        value=dict(exit_code=code,utc=utc(),phase=phase,orchestrator_pid=os.getpid(),session=SESSION,
            execution_manifest_sha256=manifest_sha,plan_sha256=state['plan_sha256'],analysis_required_locally=True,goal_complete=False)
        if (ROOT/'results/COMPLETE.json').is_file(): value['results_complete_sha256']=sha(ROOT/'results/COMPLETE.json')
        write(ROOT/'PIPELINE_EXIT.json',value)
        for n,handler in previous.items():signal.signal(n,handler)
        lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path)
    parser.add_argument('--validate-only',action='store_true');parser.add_argument('--print-plan-template',action='store_true')
    args=parser.parse_args()
    require(not args.print_plan_template or (args.plan is None and not args.validate_only),'Template mode has no execution arguments')
    if args.print_plan_template: result=plan_template()
    else:
        require(args.plan is not None,'Explicit frozen plan required');result=run(args.plan,validate_only=args.validate_only)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
