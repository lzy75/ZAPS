"""Archive the complete A2a cohort, including interrupted/recovered provenance.

Zero model loads or UNet calls. Verify the frozen plan and six active records;
copy text/plots once, hash every excluded binary, derive matched-t tables.
Optionally create a results/* commit without changing HEAD or the user's index.
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shlex
import shutil
import stat
import statistics
import sys
import tempfile
from zoneinfo import ZoneInfo

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
from utils import recover_guidance_replication_storage as recovery
from utils.archive_state_indicator_stage import ALLOWED, commit_results, sha256


def audit_source(root):
    root = Path(root).resolve()
    plan = json.loads((root/'plan.json').read_text(encoding='utf-8'))
    recovery.validate_plan(plan)
    registry = recovery.registry_for(root)
    active = recovery.load_active(root, plan, registry)
    if len(active) != 6 or any(record is None for record in active.values()):
        raise ValueError('Archive requires all six active cases complete; do not rerun GPU to archive')
    summary = recovery.stage_summary(root, plan, registry, active)
    if summary['completed_cases'] != 6:
        raise ValueError('Need six complete active cases, including the registered recovery')
    saved = json.loads((root/'recovery_summary.json').read_text(encoding='utf-8'))
    if saved != summary:
        raise ValueError('Saved recovery_summary differs from recomputed active records; inspect before archiving')
    return summary, active


def inventory(root):
    """Root symlink may be the verified migration alias; reject nested links."""
    result = {}
    for path in sorted(Path(root).rglob('*')):
        if path.is_symlink():
            raise ValueError('Nested symlink in result directory: '+str(path))
        mode = path.stat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError('Non-regular result file: '+str(path))
        result[str(path.relative_to(root))] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    return result


def extras(root):
    paths = list(root.parent.glob(root.name+'.migration_*.json'))
    paths += [root.parent/name for name in ('A2a_repair_20261018.log', 'A2a_remaining_after_repair_20261018.log')
              if (root.parent/name).exists()]
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError('Invalid migration/log sidecar: '+str(path))
    return sorted(paths)


def phase_readouts(active):
    result = {}
    keys = ('guidance_cosine', 'guidance_variation3_per_u', 'guidance_backtrack3',
            'guidance_rms', 'residual_noise_ratio', 'raw_x0_clip_fraction')
    for case, record in active.items():
        result[case] = {}
        for label, predicate in (('high', lambda t: t >= 667),
                                 ('middle', lambda t: 334 <= t < 667), ('low', lambda t: t < 334)):
            rows = [r for r in record['rows'] if predicate(r['t'])]
            result[case][label] = dict(n=len(rows))
            for key in keys:
                values = [r[key] for r in rows if r.get(key) is not None]
                result[case][label][key] = dict(n=len(values), median=statistics.median(values) if values else None)
    return result


def write_derived(work, summary, active):
    destination = work/'derived'
    destination.mkdir()
    def save(name, value):
        (destination/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    save('recomputed_recovery_summary.json', summary)
    save('probes.json', {key: record['probes'] for key, record in active.items()})
    save('trajectory_phases.json', phase_readouts(active))
    tables = {'same_t_descriptive_associations.csv': summary['same_t_descriptive_associations'],
              'trajectories.csv': [dict(case=key, image_id=record['case']['image_id'],
                                        sampling_seed=record['case']['sampling_seed'], **row)
                                   for key, record in active.items() for row in record['rows']]}
    for name, rows in tables.items():
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (destination/name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def export_bundle(root, summary, active, projects=PROJECTS):
    root = Path(root).resolve()
    parent = Path(projects).resolve()/'server_records'
    if parent.is_relative_to(root) or root.is_relative_to(parent):
        raise ValueError('Archive output must be outside the source cohort')
    before = inventory(root)
    sidecars = {str(p): dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in extras(root)}
    parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(experiment='guidance_replication_stage_archive', status='complete',
                    source=str(root), plan_sha256=sha256(root/'plan.json'), new_unet_nfe=0,
                    scientific_summary='derived/recomputed_recovery_summary.json',
                    files=[], excluded=[], sidecar_inventory=sidecars,
                    scope='Text/cards/plots, original interrupted metadata and registered recovery retained. '
                          'All binary tensors remain on the server; SHA256 inventory is not a binary backup.',
                    statistical_unit='Three image clusters, two sampling seeds each; descriptive only')
    total = 0
    with tempfile.TemporaryDirectory(prefix='.a2a_export_', dir=parent) as temporary:
        work = Path(temporary)/'bundle'
        work.mkdir()
        sources = [(root/relative, Path('source')/relative, fingerprint) for relative, fingerprint in before.items()]
        sources += [(Path(path), Path('sidecars')/Path(path).name, fingerprint) for path, fingerprint in sidecars.items()]
        for source, relative, fingerprint in sources:
            item = dict(source=str(source), relative_path=str(relative), **fingerprint)
            if source.suffix.lower() not in ALLOWED | {'.tmp'}:
                item['reason'] = 'Binary or unrecognized extension; retained on server, full SHA256 recorded'
                manifest['excluded'].append(item)
                continue
            total += item['bytes']
            if item['bytes'] > 20*1024**2 or total > 100*1024**2:
                raise ValueError('Archive payload exceeds text/plot size limit; no ref changed')
            target = work/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if sha256(target) != item['sha256']:
                raise ValueError('Archive copy differs: '+str(source))
            manifest['files'].append(item)
        write_derived(work, summary, active)
        for path in sorted((work/'derived').iterdir()):
            manifest['files'].append(dict(relative_path=str(path.relative_to(work)), bytes=path.stat().st_size,
                                          sha256=sha256(path), source='Derived from six validated active records'))
        (work/'.gitattributes').write_bytes(b'* -text\n')
        manifest['git_attributes'] = dict(relative_path='.gitattributes', sha256=sha256(work/'.gitattributes'))
        if inventory(root) != before or {str(p): dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in extras(root)} != sidecars:
            raise ValueError('Source changed during export; no archive or Git ref published')
        manifest['copied_bytes'] = sum(item['bytes'] for item in manifest['files'])
        (work/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        final = parent/('20261018_guidance_replication_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f'))
        work.rename(final)
    return final, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-dir', required=True)
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--results-branch')
    args = parser.parse_args()
    if args.inspect and args.results_branch:
        parser.error('--inspect does not create a Git ref')
    root = Path(args.plan_dir).resolve()
    summary, active = audit_source(root)
    print('AUDIT PASS:', summary['classification'], 'completed=', summary['completed_cases'],
          'method/diagnostic=', summary['method_nfe'], summary['diagnostic_nfe'], flush=True)
    if args.inspect:
        print('Same-t associations:', json.dumps(summary['same_t_descriptive_associations']), flush=True)
        print('Trajectory phases:', json.dumps(phase_readouts(active)), flush=True)
        print('UNet NFE=0; source records unchanged.', flush=True)
        return
    bundle, manifest = export_bundle(root, summary, active)
    print('Archive:', bundle, '\nManifest SHA256:', sha256(bundle/'manifest.json'),
          '\nCopied files:', len(manifest['files']), 'excluded binaries:', len(manifest['excluded']), flush=True)
    if args.results_branch:
        commit = commit_results(bundle, args.results_branch, message='Archive complete A2a guidance cohort and recovery provenance\n')
        print('Results commit:', commit, '\nWorking branch and index unchanged.', flush=True)
        print('Push after network is available: git push origin '+shlex.quote(args.results_branch), flush=True)
    print('UNet NFE=0. Binary tensors remain on server. No push performed.', flush=True)


if __name__ == '__main__':
    main()
