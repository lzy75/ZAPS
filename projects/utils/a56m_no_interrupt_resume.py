"""Pin the existing A56M1 native continuation without interrupting ANY process.

Resource adapter only. Original source, scientific plan, worker and gates remain
unchanged. The legacy idle-queue signal handoff is replaced with a refusal.
"""
import argparse
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
from utils import a56m_imagenet_session as legacy
from utils.a54_session import busy, digest, exclusive, read, write

BASE = legacy.run.OUT.parent/'A56M1_no_interrupt_20261027'
CARD = PROJECTS/'experiment_cards/A56M1_no_interrupt_resume_20261027.md'


def refuse_active_queue(plan):
    """Never invoke the legacy signal handoff, including for an idle waiter."""
    folder = legacy.run.queue.BASE
    if folder.exists() and busy(folder/'session.lock'):
        raise RuntimeError('Existing A56 queue is active; no signal sent and no second worker started. Inspect its --status.')


def pinned_environment(gpu, original=None):
    if not gpu or ',' in gpu or any(c.isspace() for c in gpu):
        raise ValueError('Select exactly one GPU index or UUID')
    env = dict(os.environ if original is None else original)
    # CUDA visibility is the complete allocation; old per-slot preferences must
    # not accidentally expand it or point outside this explicit allocation.
    for key in ('ZAPS_A56_GPU0', 'ZAPS_A56_GPU1'):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    return env


def coordinator():
    BASE.mkdir(parents=True, exist_ok=True)
    with exclusive(BASE/'session.lock'):
        state = dict(pid=os.getpid(), status='running', gpu=os.environ['CUDA_VISIBLE_DEVICES'], no_process_interruption=True)
        write(BASE/'session.json', state)
        try:
            own = [Path(__file__), CARD, PROJECTS/'utils/test_a56m_no_interrupt_resume.py']
            receipt = dict(kind='RESOURCE_PIN_ONLY_NO_INTERRUPTION', gpu=state['gpu'],
                sources={str(p): digest(p) for p in own}, source_plan_sha256=digest(legacy.run.ROOT/'plan.json'),
                limits='No new scientific arm or extra NN budget; original source and numerical gates unchanged. No process signals.')
            write(BASE/'dispatch.json', receipt)
            original_snapshot = legacy.stable_snapshot
            def snapshot(plan):
                original_snapshot(plan)
                out = legacy.run.OUT/'no_interrupt_adapter'; out.mkdir(exist_ok=True)
                shutil.copyfile(BASE/'dispatch.json', out/'dispatch.json')
                for p in own:
                    shutil.copyfile(p, out/p.name)
            with patch.object(legacy.run, 'stop_idle_queue', refuse_active_queue), patch.object(legacy, 'stable_snapshot', snapshot):
                code = legacy.coordinator()
            state.update(status='complete' if code == 0 else 'stopped', exit_code=code)
        except BaseException as error:
            state.update(status='stopped', exit_code=1, error=repr(error))
            print('A56M1 NO-INTERRUPT STOP:', repr(error), '; no process signaled.', flush=True)
        finally:
            write(BASE/'session.json', state)
        return state['exit_code']


def follow(log):
    offset = 0
    if log.exists():
        data = log.read_bytes()
        print(data[-14000:].decode(errors='replace'), end='', flush=True); offset = len(data)
    while True:
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(offset); data = stream.read(); offset = stream.tell()
            if data: print(data.decode(errors='replace'), end='', flush=True)
        if not busy(BASE/'session.lock'):
            # Coordinator flushes its final log before releasing this lock.
            if log.exists():
                with log.open('rb') as stream:
                    stream.seek(offset); data = stream.read()
                if data: print(data.decode(errors='replace'), end='', flush=True)
            return read(BASE/'session.json').get('exit_code', 1)
        time.sleep(2)


def client(gpu, status=False):
    BASE.mkdir(parents=True, exist_ok=True)
    if status:
        print(dict(adapter_active=busy(BASE/'session.lock'),
            adapter=read(BASE/'session.json') if (BASE/'session.json').exists() else None), flush=True)
        return legacy.client(True)
    if busy(BASE/'session.lock'):
        return follow(BASE/'session.log')
    if busy(legacy.BASE/'session.lock'):
        print('A56M1 already active; following it without starting or stopping any task.', flush=True)
        return legacy.client(False)
    with exclusive(BASE/'client.lock'):
        if not busy(BASE/'session.lock'):
            old = read(legacy.BASE/'session.json') if (legacy.BASE/'session.json').exists() else {}
            if old.get('status') in ('failed', 'complete'):
                code = legacy.client(True)
                return code if old['status'] == 'complete' else 1
            # Check before dispatch AND again in the legacy coordinator.
            refuse_active_queue(None)
            with (BASE/'session.log').open('ab', buffering=0) as output:
                proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--coordinator'],
                    cwd=PROJECTS, env=pinned_environment(gpu), stdin=subprocess.DEVNULL,
                    stdout=output, stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
            for _ in range(100):
                if busy(BASE/'session.lock') or proc.poll() is not None: break
                time.sleep(.05)
    print('A56M1 pinned native resume: disconnect-safe; same command reattaches; no process termination.', flush=True)
    return follow(BASE/'session.log')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', help='One explicitly allocated physical index or GPU UUID')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--coordinator', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.coordinator: return coordinator()
    if not args.status and not args.gpu: parser.error('--gpu is required for an explicit resume')
    return client(args.gpu, args.status)


if __name__ == '__main__': sys.exit(main())
