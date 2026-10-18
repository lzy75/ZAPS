"""Export one completed A1b provenance chain, zero UNet/model/tensor loading.

Copy result JSON/CSV/cards/plots; inventory binary tensors without uploading them.
Optionally commit the bundle to a results/* ref using a temporary Git index;
the working branch, HEAD, user's index and original results are never switched.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
from zoneinfo import ZoneInfo

PROJECTS = Path(__file__).resolve().parents[1]
REPO = PROJECTS.parent
KINDS = ('frozen_time_guidance_local_probe', 'cached_guidance_geometry',
         'passive_fixed_parent_state_trajectory', 'fixed30_state_measurement_domain_paired')
ALLOWED = {'.json', '.csv', '.md', '.png', '.svg', '.txt', '.log'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def source_chain(probe_dir, projects=PROJECTS):
    root = (Path(projects)/'results').resolve()
    path = (Path(probe_dir)/'run.json').resolve()
    chain = []
    for kind in KINDS:
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Source must be an existing run.json below projects/results')
        record = json.loads(path.read_text(encoding='utf-8'))
        if record.get('experiment') != kind or record.get('status') != 'complete':
            raise ValueError('Incomplete/wrong provenance layer: '+str(path))
        if len(chain) < 3 and (not record.get('gates') or
                not all(g.get('passed') is True for g in record['gates'].values())):
            raise ValueError('Need complete gate-passing diagnostic layers')
        chain.append((path, record))
        if len(chain) < 4:
            next_path = Path(record['source']).resolve()
            if not next_path.is_relative_to(root):
                raise ValueError('Provenance source escapes projects/results')
            if sha256(next_path) != record['source_sha256']:
                raise ValueError('Source fingerprint mismatch: '+str(next_path))
            path = next_path
    if (chain[0][1].get('diagnostic_nfe') != 30 or chain[0][1].get('method_nfe') != 0 or
            len(chain[0][1].get('rows', [])) != 5):
        raise ValueError('Need the complete five-snapshot/30-forward A1b run')
    for name, expected in chain[0][1].get('identity_files', {}).items():
        original = Path(name).resolve()
        if original.is_relative_to(root) and (not original.is_file() or sha256(original) != expected):
            raise ValueError('Original result/tensor fingerprint changed: '+str(original))
    trace_path = chain[2][0]
    # A0b is a sibling diagnostic, not an ancestor of A1a; include all completed
    # records bound to this exact A0, without selecting by any quality metric.
    for sibling in sorted(root.glob('diag_cached_direction/*/run.json')):
        record = json.loads(sibling.read_text(encoding='utf-8'))
        if (record.get('experiment') == 'cached_conditional_mean_direction' and record.get('status') == 'complete' and
                Path(record.get('source', '')).resolve() == trace_path and
                record.get('source_sha256') == sha256(trace_path) and record.get('gates') and
                all(g.get('passed') is True for g in record['gates'].values())):
            chain.append((sibling.resolve(), record))
    return chain


def export_bundle(chain, projects=PROJECTS):
    projects = Path(projects).resolve()
    parent = projects/'server_records'; parent.mkdir(exist_ok=True)
    name = '20261018_state_indicators_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    manifest = dict(experiment='state_indicator_stage_archive', status='complete',
                    archived_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                    new_unet_nfe=0, files=[], excluded=[], provenance=[],
                    scope='Text/cards/plots copied verbatim. Binary tensors and weights are NOT backed up by this bundle.')
    total = 0
    with tempfile.TemporaryDirectory(prefix='.indicator_export_', dir=parent) as temporary:
        work = Path(temporary)/'bundle'; work.mkdir()
        for number, (path, record) in enumerate(chain):
            layer = f'{number:02d}_{record["experiment"]}'
            manifest['provenance'].append(dict(source=str(path), source_sha256=sha256(path),
                                                bundled_run=f'{layer}/run.json',
                                                git_revision=record.get('git_revision')))
            for src in sorted(path.parent.rglob('*')):
                if src.is_symlink(): raise ValueError('Refusing symlink in result archive: '+str(src))
                if not src.is_file(): continue
                rel = Path(layer)/src.relative_to(path.parent)
                size = src.stat().st_size
                item = dict(source=str(src), relative_path=str(rel), bytes=size)
                if src.suffix.lower() not in ALLOWED:
                    item['sha256'] = sha256(src) if size <= 64*1024*1024 else None
                    item['reason'] = 'binary or unrecognized extension; retained only on server'
                    manifest['excluded'].append(item); continue
                total += size
                if size > 20*1024*1024 or total > 100*1024*1024:
                    raise ValueError('Archive text/plot size limit exceeded; no Git ref modified')
                dest = work/rel; dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest); item['sha256'] = sha256(src)
                if sha256(dest) != item['sha256']: raise RuntimeError('Copy hash mismatch')
                manifest['files'].append(item)
        manifest['copied_bytes'] = total
        (work/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        final = parent/name
        work.rename(final)
    return final, manifest


def commit_results(bundle, branch, repo=REPO):
    repo = Path(repo).resolve(); bundle = Path(bundle).resolve()
    if not branch.startswith('results/') or not bundle.is_relative_to(repo/'projects/server_records'):
        raise ValueError('Only a results/* ref and projects/server_records bundle are permitted')
    subprocess.run(['git', 'check-ref-format', '--branch', branch], cwd=repo, check=True, capture_output=True)
    ref = 'refs/heads/'+branch
    old = subprocess.run(['git', 'rev-parse', '--verify', ref], cwd=repo, capture_output=True, text=True)
    old_sha = old.stdout.strip() if old.returncode == 0 else None
    base = old_sha or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    with tempfile.TemporaryDirectory(prefix='indicator_git_index_') as folder:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(folder)/'index'))
        subprocess.run(['git', 'read-tree', base], cwd=repo, env=env, check=True)
        subprocess.run(['git', 'add', '-f', '--', str(bundle.relative_to(repo))], cwd=repo, env=env, check=True)
        tree = subprocess.check_output(['git', 'write-tree'], cwd=repo, env=env, text=True).strip()
        commit = subprocess.check_output(['git', 'commit-tree', tree, '-p', base], cwd=repo, env=env,
            input='Archive state indicator A0-A1b stage records\n', text=True).strip()
    # Compare-and-swap; never overwrite a concurrently changed ref.
    subprocess.run(['git', 'update-ref', ref, commit, old_sha or '0'*40], cwd=repo, check=True)
    return commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-dir', required=True, help='Exact completed A1b Records directory')
    parser.add_argument('--inspect', action='store_true', help='Validate provenance only; no copy or Git mutation')
    parser.add_argument('--results-branch', help='Optional local results/* commit; does not push or switch branches')
    args = parser.parse_args()
    chain = source_chain(args.probe_dir)
    print('Provenance:', json.dumps([str(p) for p, _ in chain]), flush=True)
    if args.inspect: return
    bundle, manifest = export_bundle(chain)
    print('Archive:', bundle, '\nCopied files:', len(manifest['files']),
          '\nExcluded binaries:', len(manifest['excluded']), '\nManifest SHA256:', sha256(bundle/'manifest.json'), flush=True)
    if args.results_branch:
        commit = commit_results(bundle, args.results_branch)
        print('Results commit:', commit, '\nWorking branch and index unchanged.', flush=True)
        print('Push command: git push origin '+shlex.quote(args.results_branch), flush=True)
    print('UNet NFE=0; binary tensors remain at the original server paths.', flush=True)


if __name__ == '__main__': main()
