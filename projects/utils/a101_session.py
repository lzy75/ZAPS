"""A101 resumable two-GPU queue with final summary and upload."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

PROJECTS = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(PROJECTS))
from utils import diag_equal_nfe_feedback as driver
from utils import historical_ffhq_gain as files
from utils.a54_session import exclusive, busy
from utils.a65_session import allocation


def state_dir(root): return root.parent / (root.name + '_session')
def invoke(*args): subprocess.run([sys.executable, '-u', *args], cwd=PROJECTS, check=True)
def worker_status(root, shard):
    path = root / 'workers' / str(shard) / 'status.json'
    return files.read(path)['status'] if path.exists() else 'not_started'


def workers(root, state, assigned):
    queued = [shard for shard in (0, 1) if worker_status(root, shard) != 'complete']
    files.require(all(worker_status(root, shard) != 'failed' for shard in queued),
                  'Prior A101 worker failure retained')
    running = {}; failed = False
    while queued or running:
        occupied = {row['uuid'] for _, row, _ in running.values()}
        for shard in list(queued):
            row = assigned[str(shard)]
            if row['uuid'] in occupied: continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=row['uuid'], OMP_NUM_THREADS='2',
                       MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1',
                       MKL_THREADING_LAYER='GNU')
            log = (state / f'shard{shard}.log').open('ab', buffering=0)
            command = [sys.executable, '-u', 'utils/diag_equal_nfe_feedback.py', '--worker',
                       '--root', str(root), '--shard', str(shard), '--device-uuid', row['uuid']]
            process = subprocess.Popen(command, cwd=PROJECTS, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
            running[shard] = (process, row, log); queued.remove(shard); occupied.add(row['uuid'])
            print('A101 WORKER START:', shard, 'GPU=', row['index'], flush=True)
        for shard, (process, row, log) in list(running.items()):
            if process.poll() is not None:
                log.close(); del running[shard]; failed |= process.returncode != 0
                print('A101 WORKER END:', shard, 'exit=', process.returncode, flush=True)
        completed = sum(files.read(path)['status'] == 'complete'
                        for path in (root / 'cases').glob('*/*/run.json')) if root.exists() else 0
        print('A101 QUEUE:', {shard: worker_status(root, shard) for shard in (0, 1)},
              'completed=', completed, '/360', flush=True)
        if running: time.sleep(30)
    if failed: raise RuntimeError('A101 worker failed; records retained, no automatic retry')
    return all(worker_status(root, shard) == 'complete' for shard in (0, 1))


def coordinator(args):
    root = Path(args.root).resolve(); state = state_dir(root)
    with exclusive(state / 'coordinator.lock'):
        record = dict(status='running', pid=os.getpid(), root=str(root)); files.write(state / 'session.json', record)
        try:
            invoke('-m', 'unittest', '-v', 'utils.test_equal_nfe_feedback',
                   'utils.test_equal_nfe_feedback_runtime')
            invoke('utils/diag_equal_nfe_feedback.py', '--prepare', '--root', str(root),
                   '--ffhq-dir', args.ffhq_dir, '--a71-root', args.a71_root)
            assigned = allocation(root, args.gpus, args.visible)
            record['status'] = 'resource_limited' if assigned is None else (
                'complete' if workers(root, state, assigned) else 'resource_limited')
        except BaseException as error:
            record.update(status='failed', error=repr(error), traceback=traceback.format_exc())
            print('A101 SESSION FAILED:', repr(error), flush=True)
        finally:
            if (root / 'plan.json').exists():
                logs = root / 'logs'; logs.mkdir(exist_ok=True)
                for path in state.glob('shard*.log'): shutil.copyfile(path, logs / path.name)
                try:
                    invoke('utils/diag_equal_nfe_feedback.py', '--summarize', '--root', str(root))
                    if record['status'] == 'complete':
                        invoke('utils/diag_equal_nfe_feedback.py', '--report', '--root', str(root))
                    invoke('utils/diag_equal_nfe_feedback.py', '--archive', '--root', str(root))
                except subprocess.CalledProcessError as error:
                    record['summary_or_upload_error'] = repr(error)
                    print('A101 records retained; use --archive-only after inspection.', flush=True)
            record['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            files.write(state / 'session.json', record)


def follow(state, process):
    offsets = {}
    while True:
        for path in [state / 'coordinator.log', *sorted(state.glob('shard*.log'))]:
            if path.exists():
                with path.open(errors='replace') as stream:
                    stream.seek(offsets.get(str(path), 0)); value = stream.read(); offsets[str(path)] = stream.tell()
                if value: print(value, end='', flush=True)
        if not busy(state / 'coordinator.lock') and (process is None or process.poll() is not None): break
        time.sleep(2)
    path = state / 'session.json'
    if path.exists():
        row = files.read(path); print('A101 SESSION:', row, flush=True)
        if row['status'] == 'failed': raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', default=driver.ROOT)
    parser.add_argument('--a71-root', default=driver.PARENT)
    parser.add_argument('--ffhq-dir', default='/home/lzy/FFHQ/00000/00000')
    parser.add_argument('--gpus', default='0,1'); parser.add_argument('--visible', default=os.environ.get('CUDA_VISIBLE_DEVICES'))
    for flag in ('coordinator', 'status', 'archive-only'): parser.add_argument('--' + flag, action='store_true')
    args = parser.parse_args(); root = Path(args.root).resolve(); state = state_dir(root); state.mkdir(parents=True, exist_ok=True)
    if args.coordinator: coordinator(args); return
    if args.status:
        completed = sum(files.read(path)['status'] == 'complete'
                        for path in (root / 'cases').glob('*/*/run.json')) if root.exists() else 0
        print(dict(active=busy(state / 'coordinator.lock'),
                   session=files.read(state / 'session.json') if (state / 'session.json').exists() else None,
                   workers={shard: worker_status(root, shard) for shard in (0, 1)},
                   completed_runs=completed, planned_runs=360), flush=True); return
    if args.archive_only:
        files.require(not busy(state / 'coordinator.lock'), 'Wait for active A101 session')
        invoke('utils/diag_equal_nfe_feedback.py', '--summarize', '--root', str(root))
        completed = sum(files.read(path)['status'] == 'complete'
                        for path in (root / 'cases').glob('*/*/run.json')) if root.exists() else 0
        if completed == 360: invoke('utils/diag_equal_nfe_feedback.py', '--report', '--root', str(root))
        invoke('utils/diag_equal_nfe_feedback.py', '--archive', '--root', str(root)); return
    with exclusive(state / 'client.lock'):
        process = None
        if not busy(state / 'coordinator.lock'):
            if (state / 'session.json').exists():
                previous = files.read(state / 'session.json')
                files.require(previous['status'] != 'failed', 'Prior A101 failure retained; use --status')
                if previous['status'] == 'complete': print('A101 already complete.'); return
            command = [sys.executable, '-u', str(Path(__file__)), '--coordinator', '--root', str(root),
                       '--a71-root', args.a71_root, '--ffhq-dir', args.ffhq_dir, '--gpus', args.gpus]
            if args.visible is not None: command += ['--visible', args.visible]
            with (state / 'coordinator.log').open('ab', buffering=0) as log:
                process = subprocess.Popen(command, cwd=PROJECTS, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True, close_fds=True)
            print('A101 detached queue started; following to final summary.', flush=True)
        else: print('A101 active queue attached; no duplicate.', flush=True)
        follow(state, process)


if __name__ == '__main__': main()
