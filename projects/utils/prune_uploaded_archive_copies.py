"""Reclaim local export copies by checking their bytes against live results refs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


ARCHIVE_PATH = Path('projects/server_records')


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])


def local_files(repo, bundle, algorithm):
    """Inspect the manifest and compute Git's exact blob IDs without writing objects."""
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if (manifest.get('status') != 'complete' or
            manifest.get('experiment') != 'refinement_confirmation_archive'):
        raise ValueError('Unsupported archive manifest')
    entries = {item['archive_path']: item for item in manifest['files']}
    if not entries or len(entries) != len(manifest['files']):
        raise ValueError('Empty or duplicated manifest entries')
    expected = entries.keys() | {'manifest.json', '.gitattributes'}
    files = {}
    total = 0
    for path in bundle.rglob('*'):
        if path.is_symlink() or path.is_mount():
            raise ValueError('Archive contains a symlink or mount')
        if path.is_dir():
            continue
        short = path.relative_to(bundle).as_posix()
        if short not in expected or not path.is_file():
            raise ValueError('Unexpected archive entry ' + short)
        stat = path.stat()
        blob = hashlib.new(algorithm, f'blob {stat.st_size}\0'.encode())
        digest = hashlib.sha256()
        with path.open('rb') as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b''):
                blob.update(block)
                digest.update(block)
        if short in entries:
            item = entries[short]
            if stat.st_size != item['bytes'] or digest.hexdigest() != item['sha256']:
                raise ValueError('Archive copy differs from manifest: ' + short)
        if stat != path.stat():
            # atime may change when the file is read. Compare only identity/content metadata.
            current = path.stat()
            fields = ('st_dev', 'st_ino', 'st_mtime_ns', 'st_ctime_ns', 'st_size')
            if any(getattr(stat,key) != getattr(current,key) for key in fields):
                raise ValueError('Archive changed during verification')
        files[path.relative_to(repo).as_posix()] = blob.hexdigest()
        if stat.st_nlink == 1:
            total += stat.st_blocks * 512
    if {Path(key).relative_to(bundle.relative_to(repo)).as_posix() for key in files} != expected:
        raise ValueError('Missing files in archive')
    return files, total


def remote_refs(repo):
    return [(line.split()[0], line.split()[1]) for line in
            git(repo, 'ls-remote', '--heads', 'origin', 'refs/heads/results/*').decode().splitlines()]


def has_commit(repo, commit):
    result = subprocess.run(['git', '-C', str(repo), 'cat-file', '-e', commit + '^{commit}'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def matching_bundles(repo, candidates, scratch=None, fetch_missing=False):
    """Verify known tips first. Any missing reference objects are written to scratch."""
    verified = {}
    verifier = scratch or repo
    refs = remote_refs(repo) if candidates else []
    known, missing = [], []
    for commit, ref in refs:
        (known if has_commit(verifier,commit) else missing).append((commit,ref))
    if fetch_missing and missing:
        remote = git(repo,'remote','get-url','origin').decode().strip()
        git(verifier,'config','--local','remote.archive-verification.url',remote)
    if missing:
        print(f'{len(missing)} uploaded results tips have no local commit metadata',flush=True)
    for commit, ref in known + (missing if fetch_missing else []):
        pending = [bundle for bundle in candidates if bundle not in verified]
        if not pending:
            break
        if not has_commit(verifier,commit):
            fetched = subprocess.run(['git','-C',str(verifier),'fetch','--quiet','--no-tags',
                                      '--depth=1','--filter=blob:none','archive-verification',ref],capture_output=True)
            if fetched.returncode or not has_commit(verifier,commit):
                print(f'SKIP {ref}: could not read the uploaded tip in scratch',flush=True)
                if fetched.returncode:
                    print(fetched.stderr.decode(errors='replace').strip(),flush=True)
                continue
        try:
            raw = git(verifier, 'ls-tree', '-r', '-z', commit, '--',
                      *(bundle.relative_to(repo).as_posix() for bundle in pending))
        except subprocess.CalledProcessError:
            continue
        blobs = {}
        for row in raw.split(b'\0'):
            if not row:
                continue
            meta, path = row.split(b'\t', 1)
            mode, kind, oid = meta.split()
            if kind == b'blob' and mode in (b'100644', b'100755'):
                blobs[os.fsdecode(path)] = oid.decode()
        for bundle in pending:
            expected, size = candidates[bundle]
            if all(blobs.get(path) == oid for path, oid in expected.items()):
                verified[bundle] = commit, ref, size
    return verified


def prune(repo, limit, apply=False, scratch=None, fetch_missing=False):
    repo = Path(git(repo,'rev-parse','--show-toplevel').decode().strip()).resolve()
    root = repo / ARCHIVE_PATH
    if root.resolve() != root or not root.is_dir():
        raise ValueError('Expected real projects/server_records directory')
    algorithm = git(repo,'rev-parse','--show-object-format').decode().strip()
    tracked = set(os.fsdecode(path) for path in git(repo,'ls-files','-z','--',str(ARCHIVE_PATH)).split(b'\0'))
    candidates = {}
    for bundle in sorted(root.glob('refinement_confirmation_*')):
        if bundle.is_symlink() or not bundle.is_dir() or bundle.is_mount():
            continue
        if any(path.startswith(bundle.relative_to(repo).as_posix() + '/') for path in tracked):
            continue
        try:
            candidates[bundle] = local_files(repo,bundle,algorithm)
        except (ValueError, KeyError, OSError) as error:
            print(f'SKIP {bundle.name}: {error}', flush=True)
    if scratch:
        scratch = Path(scratch).resolve()
        if scratch.is_relative_to(repo) or scratch.stat().st_dev == root.stat().st_dev:
            raise ValueError('Verification scratch must be outside the repository on another filesystem')
    verified = matching_bundles(repo,candidates,scratch,fetch_missing)
    chosen = sorted(verified, key=lambda path: (-verified[path][2],path.name))[:limit]
    latest = set(remote_refs(repo)) if apply and chosen else set()
    reclaimed = 0
    may_delete = 'DELETE' if apply else 'PREVIEW'
    for bundle in chosen:
        commit, ref, size = verified[bundle]
        if apply:
            if (commit,ref) not in latest or bundle.is_symlink() or local_files(repo,bundle,algorithm) != candidates[bundle]:
                print(f'SKIP {bundle}: archive or remote ref changed',flush=True)
                continue
            if git(repo,'ls-files','-z','--',bundle.relative_to(repo).as_posix()):
                print(f'SKIP {bundle}: now tracked',flush=True)
                continue
            shutil.rmtree(bundle)
        reclaimed += size
        print(f'{may_delete} {size / 1024**2:9.2f} MiB  {bundle}  {ref} {commit}',flush=True)
    print(f'{may_delete} total {reclaimed / 1024**2:.2f} MiB, verified {len(verified)}/{len(candidates)} bundles',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--scratch', type=Path, help='Shared no-checkout clone on a separate filesystem')
    parser.add_argument('--fetch-missing', action='store_true', help='Read missing live results tips into scratch')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    if args.fetch_missing and not args.scratch:
        parser.error('--fetch-missing requires --scratch on a separate filesystem')
    prune(args.repo, args.limit, args.apply,args.scratch,args.fetch_missing)


if __name__ == '__main__':
    main()
