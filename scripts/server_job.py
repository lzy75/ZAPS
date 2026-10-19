"""服务器任务包装器；默认只检查，显式 submit 才启动新任务。"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import tarfile

REPO = Path('/home/lzy/ZAPS')
ROOT = Path('/data0/lzy/ZAPS-jobs')
CODE = ROOT.parent / 'ZAPS-code.git'
PYTHON = '/home/lzy/anaconda3/bin/python'
FROZEN = {'diag_guidance_replication.py', 'recover_guidance_replication_storage.py',
          'diag_cached_guidance_state_path.py', 'diag_one_step_guidance_scale.py',
          'diag_guidance_local_probe.py', 'diag_cached_guidance_geometry.py'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def state(job, status, **extra):
    save(job / 'status.json', dict(status=status, updated_utc=datetime.now(timezone.utc).isoformat(), **extra))


def job_path(run_id):
    require(bool(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id)), 'Invalid run ID')
    return ROOT / run_id


def relative(value, prefix):
    p = PurePosixPath(value)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in value and str(p).startswith(prefix), 'Invalid project path')
    return p.as_posix()


def git(*args):
    return subprocess.check_output(['git', '-C', str(CODE if CODE.exists() else REPO), *args])


def sync(commit):
    import fcntl
    require(bool(re.fullmatch('[0-9a-f]{40}', commit)), 'Full commit required')
    with (ROOT.parent / '.zaps-code-sync.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not CODE.exists():
            subprocess.run(['git', 'clone', '--bare', '--shared', str(REPO), str(CODE)], check=True)
            origin = subprocess.check_output(['git', '-C', str(REPO), 'remote', 'get-url', 'origin'], text=True).strip()
            subprocess.run(['git', '-C', str(CODE), 'remote', 'set-url', 'origin', origin], check=True)
            proxy = subprocess.run(['git', '-C', str(REPO), 'config', '--get', 'remote.origin.proxy'], capture_output=True, text=True)
            if proxy.returncode == 0:
                subprocess.run(['git', '-C', str(CODE), 'config', 'remote.origin.proxy', proxy.stdout.strip()], check=True)
        subprocess.run(['git', '-C', str(CODE), 'fetch', '--no-tags', '--no-write-fetch-head', 'origin', commit],
                       check=True, env=dict(os.environ, GIT_TERMINAL_PROMPT='0'), timeout=90)
    actual = git('rev-parse', commit + '^{commit}').decode().strip()
    require(actual == commit, 'Fetched commit mismatch')
    return dict(commit=actual, code_store=str(CODE), server_worktree_unchanged=True)


def validate(spec):
    job_path(spec['run_id'])
    require(bool(re.fullmatch('[0-9a-f]{40}', spec['commit'])), 'Full commit required')
    require(type(spec['gpu']) is int and spec['gpu'] >= 0, 'Physical GPU index required')
    for key in ('method_nfe', 'diagnostic_nfe', 'timeout_seconds'):
        require(type(spec[key]) is int and spec[key] >= 0, 'Invalid budget: ' + key)
    require(spec['timeout_seconds'] > 0, 'Positive timeout required')
    require(spec['nfe_contract'] == 'zaps-job-env-v1', 'Entrypoint must enforce the NFE environment contract')
    entry = relative(spec['entrypoint'], 'utils/')
    card = relative(spec['card'], 'experiment_cards/')
    require(entry.endswith('.py') and PurePosixPath(entry).name not in FROZEN, 'Frozen diagnostic entrypoint is blocked')
    require(not re.match(r'A2(?:a|b0|c)[_.]', PurePosixPath(card).name, re.I), 'Completed A2 card is blocked')
    require(isinstance(spec['arguments'], list) and all(isinstance(a, str) and '\x00' not in a for a in spec['arguments']), 'Arguments must be strings')
    resolved = dict(spec)
    for key, path in [('entrypoint_sha256', entry), ('card_sha256', card)]:
        resolved[key] = sha(git('show', spec['commit'] + ':projects/' + path))
    resolved['entrypoint'], resolved['card'] = entry, card
    return resolved


def gpu_idle(gpu):
    # 兼顾包装器外的任务，不自动抢占或排队重试。
    line = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    memory, utilization = map(int, line.strip().split(','))
    require(memory < 256 and utilization == 0, 'GPU busy; no task started')


def submit(spec):
    spec = validate(spec)
    gpu_idle(spec['gpu'])
    ROOT.mkdir(parents=True, exist_ok=True)
    require(shutil.disk_usage(ROOT).free > 1024**3, '/data0 requires at least 1 GiB free')
    job = job_path(spec['run_id'])
    job.mkdir()  # 原子占用 ID，断线后相同 ID 绝不重复启动。
    save(job / 'spec.json', spec)
    source = globals().get('CONTROLLER_SOURCE')
    if source is None:
        source = Path(__file__).read_text(encoding='utf-8')
    (job / 'controller.py').write_text(source, encoding='utf-8')
    state(job, 'submitted', controller_sha256=sha(source.encode()))
    with (job / 'supervisor.log').open('ab') as log:
        child = subprocess.Popen([PYTHON, '-B', str(job / 'controller.py'), 'worker', '--run-id', spec['run_id']],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, close_fds=True)
    (job / 'supervisor.pid').write_text(str(child.pid), encoding='ascii')
    return dict(run_id=spec['run_id'], supervisor_pid=child.pid, job_dir=str(job))


def worker(job):
    import fcntl
    spec = json.loads((job / 'spec.json').read_text(encoding='utf-8'))
    child = None
    with (ROOT / '.single-task.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            require(validate(spec) == spec, 'Commit/card/source identity changed')
            gpu_idle(spec['gpu'])
            state(job, 'preparing', supervisor_pid=os.getpid())
            checkout = job / 'checkout'
            checkout.mkdir()
            with tarfile.open(fileobj=io.BytesIO(git('archive', '--format=tar', spec['commit']))) as tar:
                tar.extractall(checkout, filter='data')
            # 大工件沿用服务器路径，任务代码与原工作区隔离。
            for name in ('datasets', 'dps-checkpoint', 'guided_diffusion', 'DPS', 'projects/modules/models'):
                destination, original = checkout / name, REPO / name
                if not destination.exists() and original.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.symlink_to(original, target_is_directory=True)
            results = job / 'results'
            results.mkdir()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(spec['gpu']), PYTHONDONTWRITEBYTECODE='1',
                       ZAPS_JOB_DIR=str(job), ZAPS_RESULTS_DIR=str(results),
                       ZAPS_COMMIT=spec['commit'], ZAPS_CARD_SHA256=spec['card_sha256'],
                       ZAPS_MAX_METHOD_NFE=str(spec['method_nfe']), ZAPS_MAX_DIAGNOSTIC_NFE=str(spec['diagnostic_nfe']))
            env['PYTHONPATH'] = str(checkout / 'projects') + os.pathsep + str(checkout) + os.pathsep + env.get('PYTHONPATH', '')
            command = [PYTHON, '-B', spec['entrypoint'], *spec['arguments']]
            with (job / 'stdout.log').open('ab') as log:
                child = subprocess.Popen(command, cwd=checkout / 'projects', env=env,
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
                state(job, 'running', supervisor_pid=os.getpid(), child_pid=child.pid, command=command)
                code = child.wait(timeout=spec['timeout_seconds'])
            require(code == 0, 'Entrypoint exit code: ' + str(code))
            budget = json.loads((results / 'budget.json').read_text(encoding='utf-8'))
            for key in ('method_nfe', 'diagnostic_nfe'):
                require(type(budget[key]) is int and 0 <= budget[key] <= spec[key], 'Actual NFE exceeds budget: ' + key)
            state(job, 'complete', supervisor_pid=os.getpid(), actual_budget=budget)
        except Exception as error:
            if child is not None:
                # 超时或验证失败时回收整个子进程组，包括派生进程。
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
            state(job, 'failed', supervisor_pid=os.getpid(), reason=str(error))


def status(job):
    if not job.exists():
        return dict(status='absent')
    p = job / 'status.json'
    if not p.exists():
        return dict(status='incomplete_submission', retry_allowed=False)
    value = json.loads(p.read_text(encoding='utf-8'))
    if value['status'] in ('submitted', 'preparing', 'running'):
        pid_file = job / 'supervisor.pid'
        pid = value.get('supervisor_pid') or (pid_file.read_text().strip() if pid_file.exists() else None)
        if pid is None:
            return dict(value, retry_allowed=False)
        command = Path('/proc') / str(pid) / 'cmdline'
        if not command.exists() or str(job / 'controller.py').encode() not in command.read_bytes():
            value = dict(value, status='orphaned', retry_allowed=False)
    return value


def collect(job):
    require(status(job)['status'] in ('complete', 'failed', 'orphaned'), 'Collect requires terminal state')
    payload, excluded, total = {}, [], 0
    for path in sorted(job.rglob('*')):
        if 'checkout' in path.relative_to(job).parts:
            continue
        require(not path.is_symlink(), 'Result symlinks are not allowed')
        if not path.is_file():
            continue
        name = path.relative_to(job).as_posix()
        if path.suffix.lower() not in {'.json', '.csv', '.md', '.txt', '.log', '.sha256', '.py'}:
            excluded.append(dict(path=name, bytes=path.stat().st_size, sha256=file_sha(path)))
            continue
        require(path.stat().st_size <= 20 * 1024**2, 'File exceeds light archive limit: ' + name)
        data = path.read_bytes()
        total += len(data)
        require(total <= 100 * 1024**2, 'Archive exceeds 100 MiB')
        payload[name] = data
    manifest = dict(files=[dict(path=k, bytes=len(v), sha256=sha(v)) for k, v in payload.items()],
                    excluded=excluded, run_id=job.name, binary_backup=False)
    payload['MANIFEST.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as tar:
        for name, data in payload.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['sync', 'check', 'submit', 'worker', 'status', 'logs', 'collect'])
    parser.add_argument('--commit')
    parser.add_argument('--spec-base64')
    parser.add_argument('--run-id')
    args = parser.parse_args()
    if args.action == 'sync':
        print(json.dumps(sync(args.commit)))
        return
    if args.action in ('check', 'submit'):
        spec = json.loads(base64.b64decode(args.spec_base64).decode('utf-8-sig'))
        print(json.dumps(validate(spec) if args.action == 'check' else submit(spec), ensure_ascii=False))
        return
    job = job_path(args.run_id)
    if args.action == 'worker':
        worker(job)
    elif args.action == 'status':
        print(json.dumps(status(job), ensure_ascii=False))
    elif args.action == 'logs':
        p = job / 'stdout.log'
        if not p.exists():
            p = job / 'supervisor.log'
        if p.exists():
            with p.open('rb') as stream:
                stream.seek(max(0, p.stat().st_size - 32768))
                print('\n'.join(stream.read().decode('utf-8', errors='replace').splitlines()[-80:]))
    else:
        collect(job)


if __name__ == '__main__':
    main()
