"""Bounded one-shot transport/launch/read-only collection for the ordinary-SD comparison.

Only the new registered root is writable. No SSH, VPN, installation, cleanup,
resume, corpus download or automatic retry is implemented here.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

sys.dont_write_bytecode = True
ROOT = Path('/mnt/8T/cl/rescue_experiment_cache/native_sd_comparison_20261020_v1')
OLD = Path('/mnt/8T/cl/rescue_experiment_cache/frozen_devcheck_20261020_v1')
SOURCE = OLD/'capsule/after'
PYTHON = Path('/mnt/8T/cl/rescue_experiment_cache/specextend_match_20261020_v1/venv/bin/python')
SESSION = 'rescue_native_sd_v1'
PRIOR_MANIFEST = '70e46d52c3f3f57b1efa8d1633b665f345d54d7656529c9b09e2d25e185bf41e'
PRIOR_VERIFIED = 'c5bf8390ec5349c6d7ce29a688d3b9b13d95b6fd11f006e46a7522b9be42dffc'
CURVE = Path('/mnt/8T/cl/rescue_experiment_cache/context_curve_20261020_v2')
EXECUTION_HELPER_SHA = '921be20adf54bfe115394a2138f21ac47e12af59bbb34943eb791ee20f4ee58c'
TOOLS = {
    'tools/stage.py': 'exp/native_sd_comparison/stage.py',
    'tools/run_native_sd.py': 'exp/native_sd_comparison/run_native_sd.py',
    'tools/report.py': 'exp/native_sd_comparison/report.py',
    'tools/preregistration.json': 'exp/native_sd_comparison/preregistration_20261020_v1.json',
    'tests/test_native_sd_runner.py': 'tests/test_native_sd_runner.py',
    'tests/test_native_sd_quality.py': 'tests/test_native_sd_quality.py',
    'tests/test_native_sd_report.py': 'tests/test_native_sd_report.py',
    'tests/test_native_sd_stage.py': 'tests/test_native_sd_stage.py',
}
SUPPORT = {'analysis_fair_support.py': 'rescue/fair_confirmation_report.py'}
ASSETS = dict(teacher='/mnt/8T/cl/models/Qwen2-7B-Instruct', student='/mnt/8T/cl/models/Qwen2-1.5B-Instruct',
    mapper='/home/wanghl/cl/RESCUE/exp/module_arch/out_v1_20261018/full_final.pt',
    policy='/home/wanghl/cl/RESCUE/exp/module_arch/out_v1_20261018/budget_selected.pt',
    cost='/home/wanghl/cl/RESCUE/exp/module_arch/out_v1_20261018/cost.json')


def need(ok, message):
    if not ok: raise RuntimeError(message)


def digest(raw): return hashlib.sha256(raw).hexdigest()
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(8*1024*1024), b''): h.update(data)
    return h.hexdigest()
def read(path): return json.loads(Path(path).read_bytes())
def utc(): return datetime.now(timezone.utc).isoformat()
def encoded(value): return (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()
def write(path, value):
    with Path(path).open('xb') as handle: handle.write(encoded(value));handle.flush();os.fsync(handle.fileno())
def safe(name):
    p = PurePosixPath(name)
    need(isinstance(name, str) and p.parts and p.as_posix() == name and not p.is_absolute() and
         '..' not in p.parts and '\\' not in name and '\x00' not in name and
         not {'.ssh', '.git', '__pycache__', '.env'}.intersection(p.parts), 'Unsafe artifact name')
    return Path(*p.parts)
def no_links(path):
    need(not any(p.is_symlink() for p in (Path(path), *Path(path).parents)), 'Symlink artifact path')
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def inventory(contents):
    return {n: digest(v) for n, v in contents.items()}


def check_payload(contents):
    m = json.loads(contents['LAUNCH_MANIFEST.json'])
    expected = set(TOOLS)|{'tools/'+n for n in SUPPORT}|{
        'inputs/prior_local_seal/FINAL_ARTIFACTS_MANIFEST.json', 'inputs/prior_local_seal/VERIFIED_COMPLETE.json'}
    need(m.get('status') == 'STAGED_NOT_LAUNCHED' and m.get('root') == str(ROOT) and m.get('session') == SESSION and
         m.get('new_generation_requests') == 690 and m.get('launches_on_receive') is False and
         m.get('no_training') is m.get('no_new_confirmation_access') is True, 'Wrong one-shot scope')
    need(set(contents) == expected|{'LAUNCH_MANIFEST.json'} and set(m['file_sha256']) == expected and
         all(digest(contents[n]) == h for n, h in m['file_sha256'].items()), 'Stage inventory changed')
    need(digest(contents['inputs/prior_local_seal/FINAL_ARTIFACTS_MANIFEST.json']) == PRIOR_MANIFEST and
         digest(contents['inputs/prior_local_seal/VERIFIED_COMPLETE.json']) == PRIOR_VERIFIED, 'Wrong prior complete seal')
    return m


def archive_bytes(contents):
    target = io.BytesIO()
    with tarfile.open(fileobj=target, mode='w:gz', format=tarfile.USTAR_FORMAT) as archive:
        for name, raw in sorted(contents.items()):
            safe(name);info = tarfile.TarInfo(name);info.size = len(raw);info.mode = 0o444
            archive.addfile(info, io.BytesIO(raw))
    return target.getvalue()


def unpack(raw):
    contents = {};cursor = 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
        need(not archive.pax_headers, 'PAX header forbidden')
        for member in archive:
            safe(member.name)
            need(member.isfile() and not member.pax_headers and member.name not in contents and
                 member.offset == cursor and member.offset_data == cursor+512, 'Nonregular/duplicate/hidden tar entry')
            archive.fileobj.seek(member.offset+156)
            need(archive.fileobj.read(1) in (b'0', b'\0'), 'Nonregular physical tar header')
            contents[member.name] = archive.extractfile(member).read()
            need(len(contents[member.name]) == member.size, 'Truncated tar entry')
            archive.fileobj.seek(member.offset_data+member.size)
            need(not any(archive.fileobj.read((-member.size)%512)), 'Nonzero tar padding')
            cursor = member.offset_data+((member.size+511)//512)*512
        archive.fileobj.seek(cursor);tail = archive.fileobj.read()
        need(len(tail) >= 1024 and not any(tail), 'Hidden trailing tar bytes')
    for name in contents:
        p = PurePosixPath(name)
        need(not any(PurePosixPath(*p.parts[:i]).as_posix() in contents for i in range(1, len(p.parts))), 'Ancestor collision')
    return contents


def materialize(root, contents):
    no_links(root);need(not root.exists(), 'Fresh root only; preserve any partial root')
    root.mkdir()
    for name, raw in contents.items():
        path = root/safe(name);path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:handle.write(raw)
        path.chmod(0o444)


def build(repo, prior, output):
    for p in (repo, prior, output):no_links(p)
    need(not output.exists() and all(not (output.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(output.resolve()))
         for p in (repo, prior)), 'Fresh disjoint package directory required')
    contents = {target:(repo/source).read_bytes() for target, source in TOOLS.items()}
    for n in ('FINAL_ARTIFACTS_MANIFEST.json', 'VERIFIED_COMPLETE.json'):
        contents['inputs/prior_local_seal/'+n] = (prior/n).read_bytes()
    prior_meta = read(prior/'FINAL_ARTIFACTS_MANIFEST.json')
    for name, relative in SUPPORT.items():
        origin = prior/'terminal/results/source'/relative
        key = origin.relative_to(prior).as_posix()
        need(sha(origin) == prior_meta['file_sha256'][key], 'Saved analysis source changed')
        contents['tools/'+name] = origin.read_bytes()
    m = dict(status='STAGED_NOT_LAUNCHED', root=str(ROOT), session=SESSION, created_at_utc=utc(),
        file_sha256=inventory(contents), new_generation_requests=690, launches_on_receive=False,
        no_training=True, no_new_confirmation_access=True, no_model_weights_transferred=True)
    contents['LAUNCH_MANIFEST.json'] = encoded(m);check_payload(contents)
    output.mkdir();materialize(output/'payload', contents)
    raw = archive_bytes(contents)
    with (output/'transport.tar.gz').open('xb') as handle:handle.write(raw)
    receipt = dict(status='PACKAGED_NOT_LAUNCHED', transport_sha256=digest(raw),
        manifest_sha256=digest(contents['LAUNCH_MANIFEST.json']), files=len(contents), root=str(ROOT))
    write(output/'TRANSPORT_RECEIPT.json', receipt);return receipt


def receive(expected):
    raw = sys.stdin.buffer.read();need(digest(raw) == expected, 'Transport SHA mismatch')
    contents = unpack(raw);check_payload(contents);materialize(ROOT, contents)
    return dict(status='RECEIVED_NOT_LAUNCHED', files=len(contents), root=str(ROOT))


def stage_state(fresh=True):
    no_links(ROOT)
    no_links(ROOT/'LAUNCH_MANIFEST.json')
    m = read(ROOT/'LAUNCH_MANIFEST.json')
    for name in m['file_sha256']:no_links(ROOT/safe(name))
    contents = {n:(ROOT/safe(n)).read_bytes() for n in m['file_sha256']}
    contents['LAUNCH_MANIFEST.json'] = (ROOT/'LAUNCH_MANIFEST.json').read_bytes()
    check_payload(contents)
    for name in contents:no_links(ROOT/name)
    if fresh:
        need(not any((ROOT/n).exists() for n in ('LAUNCH_ATTEMPT.json', 'RUN_STARTED.json', 'PIPELINE_EXIT.json', 'results')),
             'Already attempted or partial; never resume/restart')
        need({p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()} == set(contents),
             'Unknown partial artifact in fresh stage')
    return m


def arguments():
    return dict(curve_root=CURVE, old_root=OLD, source_root=SOURCE, prior_seal_root=ROOT/'inputs/prior_local_seal',
        prior_manifest_sha256=PRIOR_MANIFEST, prior_verified_sha256=PRIOR_VERIFIED,
        preregistration=ROOT/'tools/preregistration.json', output=ROOT/'results', **{k:Path(v) for k,v in ASSETS.items()})


def validate():
    m = stage_state()
    need(Path(sys.executable).absolute() == PYTHON, 'Pinned isolated Python required')
    support = OLD/'tools/fair_union_execution_support.py'
    no_links(support);need(sha(support) == EXECUTION_HELPER_SHA, 'Frozen execution support changed')
    helper = load(support, '_curve_execution_support');identity = helper.python_chain(check_binary=True)
    runner = load(ROOT/'tools/run_native_sd.py', '_curve_runner')
    prepared = runner.prepare(argparse.Namespace(**arguments()))  # Read-only, no CUDA or mkdir.
    return m, helper, identity, prepared


def environment():
    # Do not forward credentials or unrelated Python/CUDA injection variables.
    env = {k:os.environ[k] for k in ('PATH', 'LANG', 'LC_ALL', 'TZ') if k in os.environ}
    env.update(PYTHONPATH=str(SOURCE), PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
        PYTHONHASHSEED='20261028', CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return env


def logged(command, name):
    with (ROOT/name).open('x') as handle:
        result = subprocess.run(command, cwd=SOURCE, env=environment(), stdout=handle, stderr=subprocess.STDOUT)
    need(result.returncode == 0, 'Execution failed; preserve root and inspect '+name)


def run(authorize):
    need(authorize, 'Explicit --authorize-run required')
    m, helper, identity, prepared = validate()
    need(os.environ.get('TMUX') and subprocess.check_output(['tmux','display-message','-p','#S'], text=True).strip() == SESSION,
         'Exact persistent tmux session required')
    manifest_sha = sha(ROOT/'LAUNCH_MANIFEST.json')
    write(ROOT/'LAUNCH_ATTEMPT.json', dict(at_utc=utc(), pid=os.getpid(), session=SESSION, manifest_sha256=manifest_sha))
    try:
        write(ROOT/'RUN_STARTED.json', dict(at_utc=utc(), pid=os.getpid(), session=SESSION, python=identity, manifest_sha256=manifest_sha))
        code = """import json,platform,torch,transformers,pyarrow,flash_attn
from rescue.baseline_selection import flash_receipt
print('RECEIPT='+json.dumps(dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,transformers=transformers.__version__,pyarrow=pyarrow.__version__,flash_attn=flash_attn.__version__,flash_receipt=flash_receipt())),flush=True)
"""
        logged([str(PYTHON), '-c', code], 'dependency_preflight.log')
        values = [json.loads(v.split('RECEIPT=',1)[1]) for v in (ROOT/'dependency_preflight.log').read_text().splitlines() if v.startswith('RECEIPT=')]
        need(len(values) == 1, 'Missing dependency receipt');v = values[0];old = prepared['old']
        need(v['python'] == '3.10.20' and v['torch'] == old['environment']['torch'] and v['cuda'] == old['environment']['cuda'] and
            v['transformers'] == '4.40.2' and v['pyarrow'] == '24.0.0' and v['flash_attn'] == '2.8.3' and
            v['flash_receipt'] == old['flash_attention'], 'Dependencies changed; no installation allowed')
        write(ROOT/'DEPENDENCY_PREFLIGHT.json', dict(passed=True, values=v, no_installation=True))
        gpu_code = """import json,sys,unittest,torch
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'tests'))
assert torch.cuda.is_available() and torch.cuda.device_count()==1
names=json.loads(sys.argv[1]);r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
assert r.testsRun==len(names) and r.wasSuccessful() and not r.skipped
print('GPU_PREFLIGHT_PASSED',flush=True)
"""
        helper.probe_gpu();logged([str(PYTHON), '-c', gpu_code, json.dumps(helper.GPU_CASES)], 'gpu_preflight.log')
        write(ROOT/'GPU_PREFLIGHT.json', dict(passed=True, tests_run=len(helper.GPU_CASES), skipped=0, cases=list(helper.GPU_CASES)))
        need(digest(encoded(stage_state(fresh=False))) == digest(encoded(m)), 'Stage changed before generation')
        helper.probe_gpu()
        command = [str(PYTHON), str(ROOT/'tools/run_native_sd.py')]
        for k,v in arguments().items():command.extend(('--'+k.replace('_','-'), str(v)))
        logged(command, 'benchmark.log')
        done = read(ROOT/'results/COMPLETE.json')
        need(done.get('passed') is True and done.get('logical_requests') == 690 and done.get('goal_complete') is False,
             'Incomplete690 matrix; never retry')
        stage_state(fresh=False);helper.python_chain(check_binary=True)
        write(ROOT/'PIPELINE_EXIT.json', dict(at_utc=utc(), exit_code=0, complete_sha256=sha(ROOT/'results/COMPLETE.json'), analysis_required_locally=True))
        return dict(status='FINISHED_ANALYSIS_REQUIRED', goal_complete=False)
    except BaseException as error:
        if not (ROOT/'PIPELINE_EXIT.json').exists():write(ROOT/'PIPELINE_EXIT.json', dict(at_utc=utc(), exit_code=1, error_type=type(error).__name__, error=str(error)))
        raise


def terminal():
    status = subprocess.run(['tmux','has-session','-t','='+SESSION], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    need(status.returncode != 0, 'Experiment tmux still live')
    processes = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():continue
        try:raw = (entry/'cmdline').read_bytes()
        except OSError:continue
        args = raw.split(b'\0')
        if any(v.decode(errors='replace') in (str(ROOT/'tools/stage.py'), str(ROOT/'tools/run_native_sd.py')) for v in args):processes.append(entry.name)
    need(not processes, 'Actual experiment process still live')
    need((ROOT/'PIPELINE_EXIT.json').is_file(), 'No authoritative pipeline exit')
    return dict(tmux_present=False, execution_processes=[], exit=read(ROOT/'PIPELINE_EXIT.json'))


def collect():
    stage_state(fresh=False);before = terminal()
    paths = sorted(p for p in ROOT.rglob('*') if p.is_file())
    need(not any(p.is_symlink() for p in ROOT.rglob('*')), 'Export symlink forbidden')
    contents = {p.relative_to(ROOT).as_posix():p.read_bytes() for p in paths}
    for p in paths:need(sha(p) == digest(contents[p.relative_to(ROOT).as_posix()]), 'Raw changed during terminal collection')
    after = terminal()
    need(sorted(p for p in ROOT.rglob('*') if p.is_file()) == paths, 'Artifact set changed during collection')
    for p in paths:
        no_links(p);need(sha(p) == digest(contents[p.relative_to(ROOT).as_posix()]), 'Raw changed after terminal barrier')
    raw_pins = {n.removeprefix('results/'):digest(v) for n,v in contents.items() if n.startswith('results/')}
    need(raw_pins, 'No scientific raw results')
    contents['results/EVIDENCE_MANIFEST.json'] = encoded(dict(status='TERMINAL_READ_ONLY_EXPORT', file_sha256=raw_pins))
    contents['COLLECTION.json'] = encoded(dict(at_utc=utc(), handles_before=before, handles_after=after,
        original_file_sha256=inventory(contents), remote_writes=False))
    sys.stdout.buffer.write(archive_bytes(contents))


def main():
    p = argparse.ArgumentParser(description=__doc__);sub = p.add_subparsers(dest='mode', required=True)
    b = sub.add_parser('build')
    for n in ('repo','prior','output'):b.add_argument('--'+n, type=Path, required=True)
    r = sub.add_parser('receive');r.add_argument('--sha256', required=True)
    sub.add_parser('validate-only');r = sub.add_parser('run');r.add_argument('--authorize-run', action='store_true')
    sub.add_parser('collect');a = p.parse_args()
    if a.mode == 'build':value = build(a.repo, a.prior, a.output)
    elif a.mode == 'receive':value = receive(a.sha256)
    elif a.mode == 'validate-only':m,_,_,_ = validate();value = dict(status='VALIDATED_NOT_LAUNCHED', manifest_sha256=sha(ROOT/'LAUNCH_MANIFEST.json'))
    elif a.mode == 'collect':collect();return
    else:value = run(a.authorize_run)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':main()
