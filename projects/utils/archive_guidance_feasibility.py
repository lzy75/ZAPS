"""Read-only audit and one lightweight archive for A2a/A2b0/A2c/A2d. UNet NFE=0."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS))
from utils import diag_direction_refresh as d
from utils.archive_state_indicator_stage import ALLOWED, commit_results, sha256


def audit(args):
    root, output, plan, _ = d.bind(SimpleNamespace(source_plan_dir=args.source_plan_dir, a2c_dir=args.a2c_dir,
                                                  run_dir=args.a2d_dir, inspect=True))
    records = d.completed(output, plan)
    summary = d.stored_summary(output)
    if len(records) != 6 or summary != json.loads((output/'summary.json').read_text()):
        raise ValueError('Need six complete A2d cases and an exact saved summary; do not rerun GPU')
    for record in records.values():
        for p in record['points']:
            for comparison, a, b in (('F_C','F','C'), ('M_C','M','C'), ('F_M','F','M')):
                if d.effect(p['branches'][a], p['branches'][b], p['floors']) != p['effects'][comparison]:
                    raise ValueError('A2d endpoint arithmetic/classification differs')
            g=p['geometry']
            if any(g[k] != v for k,v in d.geometry(g['a0'],g['ah'],g['chord'],g['repeat_chord']).items()):
                raise ValueError('A2d geometry readout differs')
            if d.direction_candidate(p['effects'],p['shift_resolved']) != p['joint_direction_candidate']:
                raise ValueError('A2d joint decision differs')
    b0=Path(args.a2b0_dir).resolve()
    record=json.loads((b0/'run.json').read_text())
    if (record['status'] != 'complete' or Path(record['parent']).resolve() != root or
            record['method_nfe'] != 0 or record['diagnostic_nfe'] != 0 or len(record['rows']) != 30 or
            not record['gates'] or not all(g['passed'] for g in record['gates'].values()) or
            any(sha256(path) != expected for path,expected in record['source_sha256'].items())):
        raise ValueError('A2b0 incomplete, incompatible or source changed; retain original')
    return dict(A2a=root,A2b0=b0,A2c=Path(args.a2c_dir).resolve(),A2d=output),summary


def export(roots, summary, projects=PROJECTS):
    projects=Path(projects).resolve(); parent=projects/'server_records'
    if any(parent.is_relative_to(root) or root.is_relative_to(parent) for root in roots.values()):
        raise ValueError('Archive destination must be outside every source')
    before={label:d.archive.inventory(root) for label,root in roots.items()}
    sidecars=d.archive.extras(roots['A2a'])
    sidecar_inventory={str(p):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in sidecars}
    fingerprint=hashlib.sha256(json.dumps(dict(roots={k:str(v) for k,v in roots.items()},files=before,
                                             sidecars=sidecar_inventory),sort_keys=True).encode()).hexdigest()
    parent.mkdir(exist_ok=True)
    destination=parent/('guidance_feasibility_'+fingerprint[:16])
    if destination.exists():
        manifest=json.loads((destination/'manifest.json').read_text())
        if manifest['source_fingerprint'] != fingerprint:
            raise ValueError('Existing archive identity differs; do not overwrite')
        for f in manifest['files']:
            if sha256(destination/f['archive_path']) != f['sha256']:
                raise ValueError('Existing archived bytes changed')
        if manifest['a2d_summary'] != summary:
            raise ValueError('Archive summary differs')
        return destination,manifest
    manifest=dict(experiment='guidance_feasibility_archive',status='complete',new_method_nfe=0,new_diagnostic_nfe=0,
                  source_fingerprint=fingerprint,sources={k:str(v) for k,v in roots.items()},a2d_summary=summary,
                  files=[],excluded=[],scope='Complete lightweight A2a/A2b0/A2c/A2d files and binary SHA inventory; tensors remain on server, not a binary backup.')
    with tempfile.TemporaryDirectory(prefix='.feasibility_',dir=parent) as folder:
        work=Path(folder)
        (work/'.gitattributes').write_text('* -text\n')
        for label,root in roots.items():
            for name,meta in before[label].items():
                relative=str(Path(label)/name)
                item=dict(archive_path=relative,source=str(root/name),**meta)
                if Path(name).suffix.lower() not in ALLOWED|{'.sha256'}:
                    manifest['excluded'].append(item); continue
                target=work/relative;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(root/name,target)
                if sha256(target) != meta['sha256']:raise ValueError('Copy hash mismatch: '+relative)
                manifest['files'].append(item)
        for p in sidecars:
            relative=str(Path('A2a_sidecars')/p.name);target=work/relative;target.parent.mkdir(exist_ok=True)
            shutil.copyfile(p,target)
            if sha256(target) != sidecar_inventory[str(p)]['sha256']:raise ValueError('Sidecar copy differs')
            manifest['files'].append(dict(archive_path=relative,source=str(p),**sidecar_inventory[str(p)]))
        if any(d.archive.inventory(root) != before[label] for label,root in roots.items()) or any(sha256(p)!=sidecar_inventory[str(p)]['sha256'] for p in sidecars):
            raise ValueError('Source changed during archive; no packet published')
        (work/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        work.rename(destination)
    return destination,manifest


def archive_commit(bundle, branch, repo=PROJECTS.parent):
    # Repeating a failed push must not create a duplicate archive commit.
    bundle=Path(bundle).resolve();repo=Path(repo).resolve()
    ref='refs/heads/'+branch
    exists=subprocess.run(['git','rev-parse','--verify',ref],cwd=repo,text=True,capture_output=True)
    if exists.returncode == 0:
        same=True
        for p in sorted(bundle.rglob('*')):
            if not p.is_file():continue
            stored=subprocess.run(['git','show',ref+':'+str(p.relative_to(repo))],cwd=repo,capture_output=True)
            if stored.returncode or hashlib.sha256(stored.stdout).hexdigest()!=sha256(p):same=False;break
        if same:return exists.stdout.strip()
    return commit_results(bundle,branch,repo=repo,message='Archive guidance feasibility A2a-A2d with complete provenance\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-plan-dir',default='results/diag_guidance_replication/A2a_20261018')
    parser.add_argument('--a2b0-dir',default='/data0/lzy/ZAPS-results/diag_cached_guidance_state_path/20261018_195853_125059')
    parser.add_argument('--a2c-dir',default='/data0/lzy/ZAPS-results/diag_one_step_guidance_scale/A2c_20261019')
    parser.add_argument('--a2d-dir',default='/data0/lzy/ZAPS-results/diag_direction_refresh/A2d_20261019')
    parser.add_argument('--results-branch',default='results/guidance-feasibility-20261019')
    args=parser.parse_args()
    if not args.results_branch.startswith('results/'):raise ValueError('Only results/* archive branches allowed')
    roots,summary=audit(args)
    print('AUDIT PASS: six A2a/A2c/A2d cases, complete A2b0; all 90 A2d endpoint comparisons recomputed. UNet NFE=0.',flush=True)
    bundle,manifest=export(roots,summary)
    commit=archive_commit(bundle,args.results_branch)
    print('Archive:',bundle,'\nCopied files:',len(manifest['files']),'Excluded files:',len(manifest['excluded']),
          '\nResults commit:',commit,'\nWorking branch and index unchanged.\nPush command: git push origin '+args.results_branch,
          '\nUNet NFE=0; binary tensors remain on the server.',flush=True)


if __name__=='__main__':main()
