"""A42 archive-only recovery: verified raw Git blobs, data-disk index, zero UNet calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS))
from utils import diag_uncertainty_feedback as original


class ArchiveGitError(RuntimeError):pass


def git(repo,args,env=None,data=None,allow_failure=False):
    command=['git','-c','gc.auto=0','-c','commit.gpgsign=false',*args]
    result=subprocess.run(command,cwd=repo,env=env,input=data,capture_output=True)
    if result.returncode and not allow_failure:
        raise ArchiveGitError('Git command '+repr(args)+' failed ('+str(result.returncode)+'):\n'+
                              result.stderr.decode(errors='replace')[-20000:])
    return result


def verify_bundle(bundle):
    """Validate every archived file; source binary exclusions remain inventory, not copies."""
    manifest=json.loads((bundle/'manifest.json').read_text())
    expected={'manifest.json','.gitattributes'}
    for item in manifest['files']:
        name=item['path'];path=bundle/name
        if not path.resolve().is_relative_to(bundle) or path.is_symlink():raise ValueError('Unsafe archive member')
        if path.stat().st_size!=item['bytes'] or original.source.digest(path)!=item['sha256']:
            raise ValueError('Archive member differs: '+name)
        expected.add(name)
    found=set()
    for path in bundle.rglob('*'):
        if path.is_symlink():raise ValueError('Symlink in archive')
        if path.is_file():found.add(str(path.relative_to(bundle)))
    if found!=expected:raise ValueError('Unlisted or missing archive members')
    return manifest


def commit_bytes(bundle,branch,repo,work_dir,object_store=None):
    """Avoid git-add/clean filters; leave checkout and real index unchanged; update ref by CAS."""
    bundle,repo,work_dir=Path(bundle).resolve(),Path(repo).resolve(),Path(work_dir).resolve()
    if not branch.startswith('results/') or not bundle.is_relative_to(repo/'projects/server_records'):
        raise ValueError('Only results refs and server_records bundles allowed')
    git(repo,['check-ref-format','--branch',branch]);verify_bundle(bundle)
    ref='refs/heads/'+branch
    checked=git(repo,['symbolic-ref','-q','HEAD'],allow_failure=True)
    if checked.returncode==0 and checked.stdout.decode().strip()==ref:raise ValueError('Refuse updating the checked-out branch')
    prior=git(repo,['rev-parse','--verify',ref],allow_failure=True)
    old_sha=prior.stdout.decode().strip() if prior.returncode==0 else None
    base=old_sha or git(repo,['rev-parse','HEAD']).stdout.decode().strip()
    work_dir.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ)
    if object_store is not None:env['GIT_OBJECT_DIRECTORY']=str(Path(object_store).resolve())
    with tempfile.TemporaryDirectory(prefix='index_',dir=work_dir) as temp:
        env['GIT_INDEX_FILE']=str(Path(temp)/'index')
        git(repo,['read-tree',base],env)
        entries=[];fingerprints={}
        for path in sorted(bundle.rglob('*')):
            if not path.is_file():continue
            payload=path.read_bytes();name=str(path.relative_to(repo));fingerprints[name]=hashlib.sha256(payload).hexdigest()
            oid=git(repo,['hash-object','--no-filters','-w','--stdin'],env,payload).stdout.strip()
            entries.append(b'100644 '+oid+b'\t'+os.fsencode(name)+b'\0')
        git(repo,['update-index','-z','--index-info'],env,b''.join(entries))
        tree=git(repo,['write-tree'],env).stdout.decode().strip()
        for name,digest in fingerprints.items():
            blob=git(repo,['cat-file','blob',tree+':'+name],env).stdout
            if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Git blob differs: '+name)
        if git(repo,['rev-parse',base+'^{tree}'],env).stdout.decode().strip()==tree:
            return base
        message=('Recover A42 archive with byte-preserving Git plumbing\n'
                 'Recovery implementation SHA256: '+original.source.digest(__file__)+'\n').encode()
        commit=git(repo,['commit-tree',tree,'-p',base],env,message).stdout.decode().strip()
        git(repo,['update-ref',ref,commit,old_sha or '0'*40],env)
    return commit


def disk_full(error):
    text=str(error).lower()
    return any(s in text for s in ('no space left on device','disk full','空间不足'))


def external_objects(repo,store):
    """Register a persistent object alternate only after a confirmed ENOSPC failure."""
    store=Path(store).resolve();store.mkdir(parents=True,exist_ok=True,mode=0o700)
    objects=Path(git(repo,['rev-parse','--git-path','objects']).stdout.decode().strip())
    if not objects.is_absolute():objects=Path(repo)/objects
    objects=objects.resolve();info=objects/'info';info.mkdir(exist_ok=True)
    alternatives=info/'alternates'
    if alternatives.is_symlink():raise ValueError('Refuse rewriting a symlinked alternates file')
    old=alternatives.read_bytes() if alternatives.exists() else b''
    line=os.fsencode(str(store))
    if line not in old.splitlines():
        # Preserve all existing alternate paths; never move/delete existing Git objects.
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(prefix='alternates_a42_',dir=info,delete=False) as stream:
                temporary=Path(stream.name)
                stream.write(old+(b'\n' if old and not old.endswith(b'\n') else b'')+line+b'\n')
            current=alternatives.read_bytes() if alternatives.exists() else b''
            if current!=old:raise ValueError('Alternates changed concurrently; no overwrite')
            temporary.replace(alternatives)
        finally:
            if temporary is not None and temporary.exists():temporary.unlink()
    return store,objects


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=original.ROOT)
    args=parser.parse_args();root=Path(args.root).resolve();p=original.source.check_plan(root);original.verify(p)
    items=[original.done(root,p,c) for c in p['cases']]
    if len(items)!=4 or any(item is None for item in items):raise ValueError('A42 incomplete; no experiment will be started')
    summary=original.source.load_signed(root/'summary.json')
    if summary['completed_cases']!=4:raise ValueError('Need complete A42 summary')
    print('A42 Summary:',json.dumps(summary,ensure_ascii=False,allow_nan=False),flush=True)
    print('ARCHIVE_RECOVERY:4 complete cases verified; NEW UNet NFE=0. No model loaded, no rerender.',flush=True)
    repo=PROJECTS.parent;work=root.parent/'git_recovery_work';store=root.parent/'git_objects'
    def publish(bundle,branch,repo=repo,message=None):
        manifest=verify_bundle(Path(bundle).resolve())
        from utils.archive_refinement_confirmation import inventory
        fingerprint=hashlib.sha256(json.dumps(inventory(root),sort_keys=True).encode()).hexdigest()
        if manifest['source']!=str(root) or manifest['fingerprint']!=fingerprint:
            raise ValueError('Archive does not match the current complete source')
        try:return commit_bytes(bundle,branch,repo,work)
        except ArchiveGitError as error:
            print(str(error),flush=True)
            if not disk_full(error):raise
            external,objects=external_objects(repo,store)
            env_alternates=os.environ.get('GIT_ALTERNATE_OBJECT_DIRECTORIES')
            # The external writer needs access to the original parent commit. Scope to this operation.
            combined=str(objects)+(os.pathsep+env_alternates if env_alternates else '')
            with patch.dict(os.environ,{'GIT_ALTERNATE_OBJECT_DIRECTORIES':combined}):
                commit=commit_bytes(bundle,branch,repo,work,external)
            print('ENOSPC_RECOVERED: new result Git objects stored at',external,
                  '; registered as an alternate. Keep this directory for future Git access.',flush=True)
            return commit
    # Runtime archive transport substitution only; SHA-bound experiment sources remain unchanged.
    from utils import archive_state_indicator_stage
    with patch.object(archive_state_indicator_stage,'commit_results',publish):bundle,commit=original.archive(root)
    print('Archive:',bundle,'Results commit:',commit,'Branch:',original.BRANCH,flush=True)
    print('ARCHIVE_RECOVERY_PASS: byte-identical archive committed; checkout/index untouched; NEW UNet NFE=0.',flush=True)


if __name__=='__main__':main()
