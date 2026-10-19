"""只读核验 A2a/A2b0/A2c，将轻量归档以 tar.gz 写入 stdout。"""
import ast
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

REPO = Path('/home/lzy/ZAPS')
ROOT = Path('/data0/lzy/ZAPS-results')
A = ROOT / 'A2a_20261018'
B = ROOT / 'diag_cached_guidance_state_path/20261018_195853_125059'
C = ROOT / 'diag_one_step_guidance_scale/A2c_20261019'
OLD = REPO / 'projects/server_records/20261018_guidance_replication_20261018_192016_089274'
ALLOWED = {'.json', '.csv', '.md', '.sha256', '.log', '.txt', '.tmp'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()


def main():
    index_before = digest(REPO / '.git/index')
    head = git('rev-parse', 'HEAD')
    files, checks, payload, excluded = {}, {}, {}, []

    def check(path, expected):
        path = Path(path)
        require(path.is_file(), 'Missing source: ' + str(path))
        if str(path) not in files:
            files[str(path)] = digest(path)
        actual = files[str(path)]
        require(actual == expected, 'SHA256 mismatch: ' + str(path))
        checks[str(path)] = actual

    def add(path, name):
        path = Path(path)
        require(not path.is_symlink() and path.is_file(), 'Invalid file: ' + str(path))
        require(path.stat().st_size <= 20 * 1024**2, 'Oversized text: ' + str(path))
        data = path.read_bytes()
        require(name not in payload, 'Duplicate archive path: ' + name)
        payload[name] = data
        files[str(path)] = hashlib.sha256(data).hexdigest()

    manifest = read(OLD / 'manifest.json')
    for item in manifest['files']:
        check(OLD / item['relative_path'], item['sha256'])
        if item['source'].startswith('/'):
            check(item['source'], item['sha256'])
    for item in manifest['excluded']:
        check(item['source'], item['sha256'])
    check(OLD / '.gitattributes', manifest['git_attributes']['sha256'])
    for path in sorted(OLD.rglob('*')):
        if path.is_file():
            add(path, 'A2a_archive/' + path.relative_to(OLD).as_posix())
    for stage, root in [('A2a', A), ('A2b0', B), ('A2c', C)]:
        for path in sorted(root.rglob('*')):
            require(not path.is_symlink(), 'Nested symlink: ' + str(path))
            if not path.is_file():
                continue
            if path.suffix.lower() in ALLOWED:
                add(path, stage + '/' + path.relative_to(root).as_posix())
            else:
                excluded.append(dict(stage=stage, path=str(path), bytes=path.stat().st_size,
                                     sha256=digest(path)))
    plan_a, plan_c = read(A / 'plan.json'), read(C / 'plan.json')
    require(plan_c['times'] == [903, 641, 319, 54, 14] and plan_c['gammas'] == [1., .5, 0.], 'A2c frozen intervention grid')
    record_b = read(B / 'run.json')
    for mapping in (plan_a['identity_files'], plan_c['identity_files'], record_b['source_sha256']):
        for path, expected in mapping.items():
            check(path, expected)
    registry = read(A / 'recovery_registry.json')
    check(A / 'plan.json', registry['plan_sha256'])
    active = {}
    for case in plan_c['source_cases']:
        override = registry['active'].get(case)
        directory = A / (override['directory'] if override else case)
        record = read(directory / 'run.json')
        require(record['status'] == 'complete', 'Incomplete active A2a: ' + case)
        require(record['case']['key'] == case, 'A2a case identity: ' + case)
        require(record['case'] == next(c for c in plan_a['cases'] if c['key'] == case), 'A2a planned case differs')
        require(record['plan_sha256'] == digest(A / 'plan.json'), 'A2a plan identity')
        require(all(g['passed'] for g in record['gates'].values()), 'A2a production gates')
        check(record['case']['image'], record['case']['image_sha256'])
        for name, expected in record['artifacts_sha256'].items():
            check(directory / name, expected)
        require(len(record['rows']) == 30 and len(record['probes']) == 5, 'A2a rows/probes: ' + case)
        require(Path(plan_c['source_cases'][case]).resolve() == directory.resolve(), 'Recovery source: ' + case)
        if override:
            check(directory / 'run.json', override['run_sha256'])
        active[case] = record
    summary_a = read(A / 'recovery_summary.json')
    require(summary_a == read(OLD / 'derived/recomputed_recovery_summary.json'), 'A2a archived summary differs')
    require((summary_a['completed_cases'], summary_a['method_nfe'], summary_a['diagnostic_nfe']) == (6, 1800, 840), 'A2a count/cost')
    require(record_b['status'] == 'complete' and len(record_b['rows']) == 30, 'A2b0 status/rows')
    require(record_b['method_nfe'] == record_b['diagnostic_nfe'] == 0, 'A2b0 cost')
    require(all(g['passed'] for g in record_b['gates'].values()), 'A2b0 gates')

    # 只提取纯标量函数，避免导入诊断入口或模型依赖。
    source = REPO / 'projects/utils/diag_one_step_guidance_scale.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    names = {'effect', 'summarize', 'completed'}
    pure = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    require(len(pure.body) == 3, 'Frozen scalar function contract changed')
    scope = dict(math=math, statistics=statistics, json=json, digest=digest,
                 TIMES=(903, 641, 319, 54, 14), METRICS=('psnr', 'ssim', 'lpips'))
    exec(compile(pure, str(source), 'exec'), scope)
    records = scope['completed'](C, plan_c)
    require(len(records) == 6, 'A2c incomplete cohort')
    for record in records.values():
        parent = active[record['case']['key']]
        require(record['case'] == parent['case'] and record['config'] == parent['config'], 'A2c parent case/config differs')
        for point in record['points']:
            for gamma in ('0.5', '0.0'):
                result = scope['effect'](point['branches'][gamma], point['branches']['1.0'], point['floors'])
                require(result == point['effects'][gamma], 'A2c effect mismatch')
    recomputed = scope['summarize'](list(records.values()))
    require(recomputed == read(C / 'summary.json'), 'A2c full summary mismatch')
    require(recomputed['classification'] == 'LOCAL_HALF_DOSE_NOT_SUPPORTED', 'A2c classification')
    payload['audit/recomputed_A2c_summary.json'] = (json.dumps(recomputed, indent=2) + '\n').encode()
    for path in sorted(set(checks) | {str(source)}):
        p = Path(path)
        if p.is_relative_to(REPO) and p.suffix.lower() in {'.py', '.md', '.json', '.txt'}:
            name = 'frozen_sources/' + p.relative_to(REPO).as_posix()
            if name not in payload:
                add(p, name)
    for name in ('EXPERIMENT_STANDARD_20261013.md', 'A2A_ARCHIVE_AND_NEXT_20261018.md'):
        add(REPO / 'projects' / name, 'context/' + name)
    for path, expected in files.items():
        require(digest(path) == expected, 'Source changed while collecting: ' + path)
    require(digest(REPO / '.git/index') == index_before and git('rev-parse', 'HEAD') == head, 'Server index/HEAD changed')
    remote = git('ls-remote', 'origin', 'refs/heads/results/guidance-replication-20261018', 'refs/heads/master')
    require('8f58702867f6227dabad87d3fecff0fc9eb9a08e\trefs/heads/results/guidance-replication-20261018' in remote, 'A2a remote ref mismatch')
    report = dict(status='passed', timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  server_head=head, server_branch=git('branch', '--show-current'), server_index_sha256=index_before,
                  remote_refs=remote.splitlines(), verified_source_hashes=checks,
                  A2a=dict(active_cases=sorted(active), recovery=registry, method_nfe=1800, diagnostic_nfe=840,
                           original_archive_files=len(manifest['files']), original_excluded_binaries=len(manifest['excluded'])),
                  A2b0=dict(points=30, method_nfe=0, diagnostic_nfe=0, gates=record_b['gates']),
                  A2c=dict(cases=6, points=30, effects_recomputed=60, summary=recomputed),
                  new_method_nfe=0, new_diagnostic_nfe=0,
                  excluded_binaries=excluded, limits='Binary files hashed only; retained on server. No model imports, unit tests, training or diagnostic runs.')
    payload['audit/verification.json'] = (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode()
    entries = [dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) for name, data in sorted(payload.items())]
    payload['MANIFEST.json'] = (json.dumps(dict(files=entries, source='server via SSH', new_unet_nfe=0), indent=2) + '\n').encode()
    require(sum(map(len, payload.values())) < 100 * 1024**2, 'Archive too large')
    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as tar:
        for name, data in sorted(payload.items()):
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            entry.mode = 0o644
            tar.addfile(entry, io.BytesIO(data))


if __name__ == '__main__':
    main()
