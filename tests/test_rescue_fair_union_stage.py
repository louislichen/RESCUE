"""Synthetic capsule-only tests: no SSH, scientific data, models or launches."""
import copy
from contextlib import redirect_stdout
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from rescue import fair_union_confirmation_data as data


PROJECT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


builder = load(PROJECT/'exp/draft_graph/build_fair_union_stage.py', 'union_stage_builder_test')
receiver = load(PROJECT/'exp/draft_graph/receive_fair_union_stage.py', 'union_stage_receiver_test')
fixtures = load(PROJECT/'tests/test_rescue_fair_union_confirmation_data.py', 'union_stage_fixture_test')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def fixture(root, maximum=False):
    """Two complete fabricated study chains; never opens a real corpus file."""
    env = fixtures.fixture(root, same_winners=not maximum)
    r2, a2, repo = env['r2'], env['a2'], env['union_source']
    (env['stage1']/'evidence_snapshot.tar.gz').write_bytes(b'SYNTHETIC OPAQUE COLLECTION ARCHIVE; NEVER EXTRACTED')
    fixtures.seal(env['stage1'], env['analysis'], False)
    p2 = data.read_json(r2/'protocol.json')
    p2['stage1_freeze'] = data.old.validate_baseline_freeze(env['args']['stage1_results'], env['args']['stage1_analysis'],
        env['args']['stage1_verified'], env['args']['stage1_manifest'])
    (env['stage2']/'terminal.tar.gz').write_bytes(b'SYNTHETIC OPAQUE TERMINAL ARCHIVE; NEVER EXTRACTED')
    (env['stage2']/'collection.stderr').write_text('SYNTHETIC COLLECTION DIAGNOSTIC ONLY\n')
    for name in builder.CORE_SOURCE:
        source = r2/'source'/name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text('# SYNTHETIC SEALED STAGE2 SOURCE ONLY\n')
        p2['source_sha256'][name] = data.sha256(source)
    if maximum:
        winners = {'retrieval32_k1_eager', 'retrieval16_k4_resident', 'retrieval32_k12_graph', 'tree32_n64'}
        raw = [json.loads(v) for v in (r2/'runtime.jsonl').read_text().splitlines()]
        for row in raw: row['outer_seconds'] = 1. if row['mode'] in winners else 2.
        fixtures.append(r2/'runtime.jsonl', raw)
    write(r2/'protocol.json', p2)
    raw = [json.loads(v) for v in (r2/'runtime.jsonl').read_text().splitlines()]
    computed = data._recompute(raw, p2, data.read_json(env['results']/'FROZEN_SELECTION.json'))
    frozen = data.read_json(r2/'FROZEN_SELECTION.json')
    frozen.update(source_sha256=p2['source_sha256'], protocol_sha256=data.sha256(r2/'protocol.json'), per_context=computed)
    write(r2/'FROZEN_SELECTION.json', frozen)
    complete = data.read_json(r2/'COMPLETE.json'); complete['frozen_selection_sha256'] = data.sha256(r2/'FROZEN_SELECTION.json')
    write(r2/'COMPLETE.json', complete)
    write(a2/'summary.json', dict(status='COMPLETE', problems=[], observed_counts=data.STAGE2_COUNTS, authoritative_union=True, stage2=computed))
    write(a2/'VERIFIED_UNION.json', dict(status='VERIFIED_FROZEN_UNION', authoritative=True, per_context=computed,
        future_contrasts=sum(len(c['union_modes']) for c in computed.values())))
    fixtures.seal(env['stage2'], a2, True)
    env['proof'] = data.validate_union_freeze(**env['args'])
    for name in set(data.REQUIRED_NEW_SOURCE) | set(builder.EXTRA_SOURCE):
        path = repo/name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == builder.PIPELINE: path.write_bytes(builder.DISABLED_LAUNCHER)
        elif name == data.REQUIRED_NEW_SOURCE[-2]: write(path, data.protocol_template())
        else: path.write_text('# SYNTHETIC NEW SOURCE ONLY\n')
    # Untracked data/credentials/logs and even poisoned current archived source
    # must never be swept into the source capsule.
    (repo/'unlisted-secret.json').write_text('SYNTHETIC MUST NOT BE COPIED\n')
    (repo/'benchmark.log').write_text('SYNTHETIC MUST NOT BE COPIED\n')
    for name in p2['source_sha256']:
        path = repo/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# POISONED CURRENT SOURCE MUST NOT REPLACE SEALED RUNTIME\n')
    return env


def unpack(raw):
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
        return {m.name: archive.extractfile(m).read() for m in archive.getmembers()}


def archive(contents, *, extras=(), format=tarfile.USTAR_FORMAT):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz', format=format) as target:
        for name, value in list(contents.items())+list(extras):
            if isinstance(value, tarfile.TarInfo):
                target.addfile(value)
            else:
                info = tarfile.TarInfo(name); info.size = len(value); info.mode = 0o444
                target.addfile(info, io.BytesIO(value))
    return stream.getvalue()


def resign(contents):
    manifest = json.loads(contents[builder.MANIFEST])
    manifest['file_sha256'] = {n: hashlib.sha256(v).hexdigest() for n, v in contents.items() if n != builder.MANIFEST}
    contents[builder.MANIFEST] = (json.dumps(manifest)+'\n').encode()


class CapsuleTests(unittest.TestCase):
    def capsule(self, root, maximum=False):
        env = fixture(root/'synthetic', maximum)
        output = root/'capsule'
        with patch.object(builder, 'load_data', return_value=data):
            receipt = builder.build(env['union_source'], output, **env['args'])
        return env, output, receipt

    def receive(self, raw, root):
        with patch.object(receiver, 'ROOT', root), redirect_stdout(io.StringIO()):
            contents = unpack(raw)
            manifest = json.loads(contents[builder.MANIFEST]); manifest['remote_root'] = str(root)
            contents[builder.MANIFEST] = (json.dumps(manifest)+'\n').encode()
            raw = archive(contents)
            return receiver.receive(hashlib.sha256(raw).hexdigest(), io.BytesIO(raw))

    def test_min_max_real_dual_gate_portable_copy_exact_counts_and_sealed_source(self):
        self.assertEqual(receiver.NEW_SOURCE,set(data.REQUIRED_NEW_SOURCE)|set(builder.EXTRA_SOURCE))
        for maximum, m, r in ((False, 8, 4), (True, 16, 8)):
            with self.subTest(maximum=maximum), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); env, output, receipt = self.capsule(root, maximum)
                manifest = data.read_json(output/'payload'/builder.MANIFEST)
                self.assertEqual((receipt['M'], receipt['R'], receipt['expected_requests']), (m, r, 162+34*m+r))
                self.assertEqual(receipt['status'], builder.STATUS); self.assertFalse(receipt['launch_authorized'])
                self.assertTrue(receipt['launcher_disabled'])
                self.assertEqual((output/'payload/stage1_evidence/evidence_snapshot.tar.gz').read_bytes(),
                    b'SYNTHETIC OPAQUE COLLECTION ARCHIVE; NEVER EXTRACTED')
                self.assertTrue((output/'payload/stage2_evidence/terminal.tar.gz').is_file())
                self.assertTrue((output/'payload/stage2_evidence/collection.stderr').is_file())
                self.assertEqual(data.validate_union_freeze(**{k: output/'payload'/v for k,v in manifest['proof_arguments'].items()}), env['proof'])
                for name, digest in data.read_json(env['r2']/'protocol.json')['source_sha256'].items():
                    self.assertEqual(data.sha256(output/'payload/after'/name), digest)
                self.assertFalse((output/'payload/after/unlisted-secret.json').exists())
                self.assertFalse((output/'payload/after/benchmark.log').exists())
                self.assertEqual((output/'payload/after'/builder.PIPELINE).read_bytes(), builder.DISABLED_LAUNCHER)
                self.assertEqual(set(unpack((output/'transport.tar.gz').read_bytes())), set(manifest['file_sha256'])|{builder.MANIFEST})
                self.assertEqual(len([n for n in manifest['scientific_source_sha256'] if n.startswith(data.old.TREE_PREFIX)]),27)
                remote = root/'received'
                got = self.receive((output/'transport.tar.gz').read_bytes(),remote)
                self.assertEqual(got['status'],builder.STATUS); self.assertTrue((remote/'stage2_evidence/VERIFIED_COMPLETE.json').is_file())
                with self.assertRaisesRegex(RuntimeError,'Never overwrite'):
                    self.receive((output/'transport.tar.gz').read_bytes(),remote)

    def test_failed_actual_gate_makes_no_capsule_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);env=fixture(root/'synthetic');output=root/'absent'
            bad=data.read_json(env['args']['stage2_verified']);bad['rendered_figures_inspected']=False
            write(env['args']['stage2_verified'],bad)
            with patch.object(builder,'load_data',return_value=data),self.assertRaises(RuntimeError):
                builder.build(env['union_source'],output,**env['args'])
            self.assertFalse(output.exists())

    def test_mutating_launcher_or_missing_source_blocks_before_mkdir(self):
        for kind in ('launcher','core','newfile','unknown_archive'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);env=fixture(root/'synthetic');output=root/'absent'
                if kind=='launcher':(env['union_source']/builder.PIPELINE).write_text('#!/bin/bash\nmkdir forbidden\n')
                if kind=='core':(env['r2']/'source'/builder.CORE_SOURCE[0]).write_text('changed\n')
                if kind=='newfile':(env['union_source']/'rescue/fair_union_confirmation.py').unlink()
                if kind=='unknown_archive':
                    (env['stage2']/'unregistered.tar.gz').write_bytes(b'SYNTHETIC UNREGISTERED ARCHIVE')
                    fixtures.seal(env['stage2'], env['a2'], True)
                with patch.object(builder,'load_data',return_value=data),self.assertRaises(RuntimeError):
                    builder.build(env['union_source'],output,**env['args'])
                self.assertFalse(output.exists())

    def test_receiver_rejects_every_mutation_before_creating_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);env,output,_=self.capsule(root);original=unpack((output/'transport.tar.gz').read_bytes())
            for kind in ('unknown','listed_new_source','listed_evidence','evidence_byte','resigned_evidence','source_byte','source_inventory','count','forged_M','launcher','missing_proof','alias','absolute','parent','ancestor','duplicate','symlink','pax','global_pax','gnu_rename','trailing_tar','weight'):
                with self.subTest(kind=kind):
                    contents=copy.deepcopy(original);remote=root/('rejected_'+kind)
                    value=json.loads(contents[builder.MANIFEST]);value['remote_root']=str(remote)
                    contents[builder.MANIFEST]=(json.dumps(value)+'\n').encode();extras=[];fmt=tarfile.USTAR_FORMAT
                    if kind=='unknown':contents['unknown.json']=b'{}\n'
                    if kind=='listed_new_source':
                        contents['after/unregistered.json']=b'{}\n'
                        value['source_sha256']['unregistered.json']=hashlib.sha256(contents['after/unregistered.json']).hexdigest()
                        contents[builder.MANIFEST]=json.dumps(value).encode();resign(contents)
                    if kind=='listed_evidence':contents['stage2_evidence/unlisted.log']=b'SYNTHETIC UNLISTED LOG\n';resign(contents)
                    if kind=='evidence_byte':contents['stage2_evidence/results/runtime.jsonl']+=b'changed\n'
                    if kind=='resigned_evidence':contents['stage2_evidence/results/runtime.jsonl']+=b'changed\n';resign(contents)
                    if kind=='source_byte':contents['after/rescue/engine.py']+=b'changed\n'
                    if kind=='source_inventory':
                        value['source_sha256'].pop('rescue/engine.py');contents[builder.MANIFEST]=json.dumps(value).encode()
                    if kind=='count':value['expected_requests']+=1;contents[builder.MANIFEST]=json.dumps(value).encode()
                    if kind=='forged_M':
                        value['M']=value['union_freeze']['M']=9
                        value['expected_counts']['baseline_quality_requests']=9;value['expected_counts'].update(prime_requests=11,timing_requests=352,owners=11)
                        value['union_freeze']['expected_counts']=value['expected_counts']
                        value['expected_requests']=value['union_freeze']['scientific_requests']=162+34*9+value['R']
                        contents[builder.MANIFEST]=json.dumps(value).encode()
                    if kind=='launcher':
                        contents['after/'+builder.PIPELINE]=b'#!/bin/bash\nmkdir forbidden\n';resign(contents)
                    if kind=='missing_proof':contents.pop('stage2_evidence/VERIFIED_COMPLETE.json');resign(contents)
                    if kind=='alias':extras=[('after//alias.py',b'# bad\n')]
                    if kind=='absolute':extras=[('/after/bad.py',b'# bad\n')]
                    if kind=='parent':extras=[('after/../bad.py',b'# bad\n')]
                    if kind=='ancestor':extras=[('after/rescue',b'# collision\n')]
                    if kind=='duplicate':extras=[('after/rescue/engine.py',b'# duplicate\n')]
                    if kind=='symlink':
                        item=tarfile.TarInfo('after/symlink.py');item.type=tarfile.SYMTYPE;item.linkname='/etc/passwd';extras=[(item.name,item)]
                    if kind=='pax':
                        fmt=tarfile.PAX_FORMAT
                        stream=io.BytesIO()
                        with tarfile.open(fileobj=stream,mode='w:gz',format=fmt) as target:
                            for name,raw in contents.items():
                                item=tarfile.TarInfo(name);item.size=len(raw);item.pax_headers={'comment':'SYNTHETIC PAX'};target.addfile(item,io.BytesIO(raw))
                        raw=stream.getvalue()
                    elif kind=='global_pax':
                        stream=io.BytesIO()
                        with tarfile.open(fileobj=stream,mode='w:gz',format=tarfile.PAX_FORMAT,pax_headers={'comment':'SYNTHETIC GLOBAL PAX'}) as target:
                            for name,raw in contents.items():
                                item=tarfile.TarInfo(name);item.size=len(raw);target.addfile(item,io.BytesIO(raw))
                        raw=stream.getvalue()
                    elif kind=='gnu_rename':
                        extras=[('after/'+('long_'*30)+'.py',b'# GNU long-name header\n')];raw=archive(contents,extras=extras,format=tarfile.GNU_FORMAT)
                    elif kind=='trailing_tar':raw=archive(contents)+archive({'after/hidden.py':b'# SYNTHETIC HIDDEN CONCATENATED TAR\n'})
                    elif kind=='weight':contents['after/model.safetensors']=b'SYNTHETIC WEIGHT MUST NOT BE COPIED';resign(contents);raw=archive(contents)
                    else:raw=archive(contents,extras=extras)
                    with patch.object(receiver,'ROOT',remote),redirect_stdout(io.StringIO()),self.assertRaises((RuntimeError,KeyError)):
                        receiver.receive(hashlib.sha256(raw).hexdigest(),io.BytesIO(raw))
                    self.assertFalse(remote.exists())

    def test_wrong_transport_sha_and_root_boundary_no_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'never-created'
            with patch.object(receiver,'ROOT',root),self.assertRaisesRegex(RuntimeError,'Transport SHA'):
                receiver.receive('0'*64,io.BytesIO(b'SYNTHETIC WRONG TRANSPORT'))
            self.assertFalse(root.exists())
            for name in ('after/./file.py','after//file.py','after/../outside.py','after\\file.py','stage2_evidence/.ssh/key.json'):
                with self.subTest(name=name),self.assertRaises(RuntimeError):receiver.relative(name)

    def test_swallowed_global_headers_rejected_before_mkdir(self):
        """Tarfile may hide empty/global headers without exposing pax_headers."""
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,output,_=self.capsule(root)
            original=unpack((output/'transport.tar.gz').read_bytes())
            for size in (0,512):
                with self.subTest(hidden_metadata_bytes=size):
                    contents=copy.deepcopy(original);remote=root/('rejected_hidden_global_'+str(size))
                    value=json.loads(contents[builder.MANIFEST]);value['remote_root']=str(remote)
                    contents[builder.MANIFEST]=(json.dumps(value)+'\n').encode()
                    info=tarfile.TarInfo('ignored-global-header');info.type=tarfile.XGLTYPE;info.size=size
                    hidden=b'SYNTHETIC NONZERO METADATA WITHOUT PAX KEY/VALUE RECORDS'
                    # Tarfile stops parsing at the initial NUL and can ignore
                    # nonzero metadata bytes later in this same global block.
                    body=(b'\x00'+hidden+b'\x00'*(size-len(hidden)-1)) if size else b''
                    raw=gzip.compress(info.tobuf(format=tarfile.USTAR_FORMAT)+body+gzip.decompress(archive(contents)))
                    # These are the exact formerly hidden cases: no exposed
                    # PAX dictionary, but the first regular header skips bytes.
                    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as parsed:
                        members=parsed.getmembers()
                        self.assertFalse(parsed.pax_headers)
                        self.assertTrue(all(not m.pax_headers for m in members))
                        self.assertEqual(members[0].offset,512+size)
                    with patch.object(receiver,'ROOT',remote),redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'Non-contiguous/hidden'):
                        receiver.receive(hashlib.sha256(raw).hexdigest(),io.BytesIO(raw))
                    self.assertFalse(remote.exists())


if __name__=='__main__':unittest.main()
