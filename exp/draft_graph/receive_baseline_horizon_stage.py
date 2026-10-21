"""Receive one SHA-pinned capsule into one fresh experiment root; no launch."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile


ROOT = Path('/mnt/8T/cl/rescue_experiment_cache/baseline_horizon_20261020_v1')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def receive(expected):
    require(len(expected) == 64 and all(c in '0123456789abcdef' for c in expected), 'Invalid transport SHA')
    payload = sys.stdin.buffer.read()
    require(hashlib.sha256(payload).hexdigest() == expected, 'Transport SHA mismatch; no root was created')
    require(not ROOT.exists(), 'Never overwrite or clean a partial/existing isolated root')
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
        members = archive.getmembers()
        names = set()
        by_name = {}
        for entry in members:
            path = PurePosixPath(entry.name)
            require(entry.isfile() and path.parts and path.as_posix() == entry.name and
                    not path.is_absolute() and '..' not in path.parts and
                    entry.name not in names and (ROOT/entry.name).resolve().is_relative_to(ROOT.resolve()),
                    'Unsafe/duplicate archive member; no root was created')
            names.add(entry.name)
            by_name[entry.name] = entry
        for name in names:
            path = PurePosixPath(name)
            require(not any(PurePosixPath(*path.parts[:i]).as_posix() in names
                            for i in range(1, len(path.parts))),
                    'Archive file-ancestor collision; no root was created')
        require('LAUNCH_MANIFEST.json' in names and any(n.startswith('after/') for n in names) and
                'stage1_evidence/VERIFIED_COMPLETE.json' in names, 'Incomplete capsule')
        with archive.extractfile(by_name['LAUNCH_MANIFEST.json']) as handle:
            manifest = json.load(handle)
        require(manifest.get('remote_root') == str(ROOT) and manifest.get('expected_requests') == 2540 and
                manifest.get('status') == 'STAGED_NOT_LAUNCHED' and
                isinstance(manifest.get('source_sha256'), dict) and manifest['source_sha256'],
                'Unexpected staged protocol; no root was created')
        require({name for name in names if name.startswith('after/')} ==
                {'after/'+name for name in manifest['source_sha256']},
                'Unlisted/missing staged source; no root was created')
        for name, expected_sha in manifest['source_sha256'].items():
            require(isinstance(name, str), 'Malformed source path')
            path = PurePosixPath(name)
            require(path.parts and path.as_posix() == name and not path.is_absolute() and '..' not in path.parts and
                    isinstance(expected_sha, str) and len(expected_sha) == 64 and
                    all(c in '0123456789abcdef' for c in expected_sha) and 'after/'+name in by_name,
                    'Unsafe or missing manifest source; no root was created')
            with archive.extractfile(by_name['after/'+name]) as handle:
                require(hashlib.sha256(handle.read()).hexdigest() == expected_sha,
                        'Source SHA mismatch; no root was created')
        ROOT.mkdir()  # exclusive; a race or partial root is fatal, never cleaned
        for entry in members:
            destination = ROOT/entry.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(entry) as source, destination.open('xb') as target:
                target.write(source.read())
            destination.chmod(0o444)
    manifest = json.loads((ROOT/'LAUNCH_MANIFEST.json').read_text())
    require(manifest.get('remote_root') == str(ROOT) and manifest.get('expected_requests') == 2540 and
            manifest.get('status') == 'STAGED_NOT_LAUNCHED', 'Unexpected staged protocol; preserve root')
    for name, expected_sha in manifest['source_sha256'].items():
        require(hashlib.sha256((ROOT/'after'/name).read_bytes()).hexdigest() == expected_sha,
                'Staged source SHA differs; preserve partial root')
    print(json.dumps(dict(status='STAGED_NOT_LAUNCHED', root=str(ROOT),
        transport_sha256=expected, source_files=len(manifest['source_sha256']),
        launch_manifest_sha256=hashlib.sha256((ROOT/'LAUNCH_MANIFEST.json').read_bytes()).hexdigest())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sha256', required=True)
    receive(parser.parse_args().sha256)
