"""Relocate only the frozen A2a plan directory; preserve the old path as a symlink.

No Torch/GPU. Copy -> verify every byte -> switch path -> remove verified duplicate.
The incomplete tensor and temporary JSON are copied as evidence, never repaired here.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def inventory(root):
    root=Path(root);files={};directories=[]
    for p in sorted(root.rglob('*')):
        if p.is_symlink():raise ValueError('Refusing nested symlink: '+str(p))
        rel=str(p.relative_to(root))
        if p.is_dir():directories.append(rel)
        elif p.is_file():files[rel]=dict(bytes=p.stat().st_size,sha256=digest(p))
        else:raise ValueError('Refusing special file: '+str(p))
    return dict(files=files,directories=directories)


def migrate(source,destination,require_other_filesystem=True,reserve_bytes=2*1024**3):
    source=Path(os.path.abspath(source));destination=Path(destination).absolute()
    if source.is_symlink():
        if source.resolve()!=destination.resolve():raise ValueError('Source already points to another destination')
        print('Already migrated:',source,'->',source.resolve());return
    if not source.is_dir():raise ValueError('Missing original A2a directory')
    plan=json.loads((source/'plan.json').read_text())
    if plan.get('experiment')!='guidance_replication_plan' or plan.get('status')!='prepared':
        raise ValueError('Only a prepared A2a plan directory may be migrated')
    if destination.exists() or destination.is_symlink():raise ValueError('Destination already exists; no overwrite')
    if destination.is_relative_to(source) or source.is_relative_to(destination):raise ValueError('Nested destination/source')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if require_other_filesystem and source.stat().st_dev==destination.parent.stat().st_dev:
        raise ValueError('Destination must be on a different filesystem to free root space')
    print('Hashing original files; UNet NFE=0...',flush=True)
    before=inventory(source);needed=sum(f['bytes'] for f in before['files'].values())
    if shutil.disk_usage(destination.parent).free<needed+reserve_bytes:
        raise OSError('Destination lacks copy size plus 2 GiB reserve')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    staging=destination.with_name(destination.name+'.copying_'+stamp)
    backup=source.with_name(source.name+'.verified_root_copy_'+stamp)
    if backup.exists():raise ValueError('Backup path collision')
    print('Copying to:',staging,flush=True)
    shutil.copytree(source,staging)
    if inventory(staging)!=before or inventory(source)!=before:
        raise RuntimeError('Copy mismatch or original changed; original retained, no switch/deletion')
    marker=destination.with_name(destination.name+'.migration_'+stamp+'.json')
    record=dict(status='copy_verified',source=str(source),destination=str(destination),root_duplicate_backup=str(backup),
                bytes=needed,inventory=before,plan_sha256=before['files']['plan.json']['sha256'])
    marker.write_text(json.dumps(record,indent=2)+'\n')
    staging.rename(destination)
    source.rename(backup)
    try:
        source.symlink_to(destination,target_is_directory=True)
        if inventory(source)!=before or inventory(backup)!=before:
            raise RuntimeError('Alias/old-copy mismatch; refusing old-copy deletion')
    except BaseException:
        if source.is_symlink():source.unlink()
        if not source.exists():backup.rename(source)
        raise
    # Only delete the duplicate proven identical to the verified destination.
    # The destination and the historical source path both retain every file.
    shutil.rmtree(backup)
    record['status']='complete';marker.write_text(json.dumps(record,indent=2)+'\n')
    print('MIGRATED:',source,'->',destination,flush=True)
    print('Verified files:',len(before['files']),'relocated bytes:',needed,'manifest:',marker,flush=True)
    print('Original plan/case bytes unchanged; root duplicate removed; UNet NFE=0.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan-dir',required=True);p.add_argument('--destination',required=True)
    args=p.parse_args();migrate(args.plan_dir,args.destination)


if __name__=='__main__':main()
