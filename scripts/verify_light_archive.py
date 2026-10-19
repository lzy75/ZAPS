"""校验轻量 tar.gz 清单，提取到新的目录，保留原始字节。"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def verify(archive, destination):
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(set(names)) != len(names) or sum(m.size for m in members) > 100 * 1024**2:
            raise ValueError('Duplicate paths or excessive archive size')
        for member in members:
            p = PurePosixPath(member.name)
            if not member.isfile() or p.is_absolute() or '..' in p.parts or '\\' in member.name or ':' in member.name:
                raise ValueError('Unsafe archive path: ' + member.name)
        manifest = json.load(tar.extractfile('MANIFEST.json'))
        expected = [f['path'] for f in manifest['files']]
        if len(set(expected)) != len(expected) or set(names) != set(expected) | {'MANIFEST.json'}:
            raise ValueError('Manifest inventory mismatch')
        for entry in manifest['files']:
            data = tar.extractfile(entry['path']).read()
            if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
                raise ValueError('SHA256 mismatch: ' + entry['path'])
        Path(destination).mkdir(parents=True, exist_ok=False)
        tar.extractall(destination, filter='data')
    print('Verified files:', len(expected))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive')
    p.add_argument('destination')
    args = p.parse_args()
    verify(args.archive, args.destination)
