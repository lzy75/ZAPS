"""Byte-preserving A3-A6 lightweight archive, binary SHA inventory, temporary Git index."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from utils.archive_state_indicator_stage import ALLOWED,commit_results,sha256,PROJECTS

BRANCH='results/refinement-confirmation-20261020'


def inventory(root):
    files={}
    for path in sorted(Path(root).rglob('*')):
        if path.is_symlink():raise ValueError('Symlink inside stage source: '+str(path))
        if path.is_file():files[str(path.relative_to(root))]=dict(bytes=path.stat().st_size,sha256=sha256(path))
    return files


def export_and_commit(roots,projects=PROJECTS):
    projects=Path(projects).resolve();roots={k:Path(v).resolve() for k,v in roots.items()}
    parent=projects/'server_records';parent.mkdir(exist_ok=True)
    if any(parent.is_relative_to(root) or root.is_relative_to(parent) for root in roots.values()):raise ValueError('Archive and sources overlap')
    before={k:inventory(root) for k,root in roots.items()}
    fingerprint=hashlib.sha256(json.dumps(dict(roots={k:str(v) for k,v in roots.items()},files=before),sort_keys=True).encode()).hexdigest()
    bundle=parent/('refinement_confirmation_'+fingerprint[:16])
    if not bundle.exists():
        manifest=dict(experiment='refinement_confirmation_archive',status='complete',sources={k:str(v) for k,v in roots.items()},
            source_fingerprint=fingerprint,files=[],excluded=[],new_method_nfe=0,new_diagnostic_nfe=0,
            scope='Complete lightweight stage records. Binary SHA inventory only; tensors remain on server, not a binary backup.')
        with tempfile.TemporaryDirectory(prefix='.refinement_export_',dir=parent) as tmp:
            work=Path(tmp)/'bundle';work.mkdir();(work/'.gitattributes').write_text('* -text\n')
            for label,items in before.items():
                for name,meta in items.items():
                    relative=str(Path(label)/name);item=dict(archive_path=relative,source=str(roots[label]/name),**meta)
                    if Path(name).suffix.lower() not in ALLOWED|{'.sha256'}:
                        manifest['excluded'].append(item);continue
                    target=work/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(roots[label]/name,target)
                    if sha256(target)!=meta['sha256']:raise ValueError('Archive copy differs')
                    manifest['files'].append(item)
            if any(inventory(root)!=before[k] for k,root in roots.items()):raise ValueError('Source changed during export')
            (work/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');work.rename(bundle)
    manifest=json.loads((bundle/'manifest.json').read_text())
    if manifest['source_fingerprint']!=fingerprint or any(sha256(bundle/f['archive_path'])!=f['sha256'] for f in manifest['files']):raise ValueError('Existing stage archive differs')
    # Reuse the previous results commit if it already contains exactly this archive.
    relative=str((bundle/'manifest.json').relative_to(projects.parent))
    probe=subprocess.run(['git','show',BRANCH+':'+relative],cwd=projects.parent,capture_output=True)
    if probe.returncode==0 and hashlib.sha256(probe.stdout).hexdigest()==sha256(bundle/'manifest.json'):
        commit=subprocess.check_output(['git','rev-parse',BRANCH],cwd=projects.parent,text=True).strip()
    else:commit=commit_results(bundle,BRANCH,repo=projects.parent,message='Archive A3-A6 fixed refinement stage records\n')
    return bundle,commit
