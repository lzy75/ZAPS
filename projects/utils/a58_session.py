"""A58 detached execution with a foreground log follower; reconnect with same command."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import os
import subprocess
import sys
import time

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
from utils.a54_session import exclusive, busy, read, write, progress

BASE = Path('/data0/lzy/ZAPS-results/diag_blur_pcg_admission')
ROOT = BASE/'A58_20261027'


def coordinator():
    BASE.mkdir(parents=True, exist_ok=True)
    with exclusive(BASE/'session.lock'):
        state = dict(pid=os.getpid(), status='running_detached', started_utc=datetime.now(timezone.utc).isoformat())
        write(BASE/'session.json', state)
        try:
            print('A58 detached job; disconnecting the SSH log follower does not stop computation.', flush=True)
            result = subprocess.run(['bash', str(PROJECTS/'run_a58_20261027.sh')], cwd=PROJECTS, stdin=subprocess.DEVNULL)
            if result.returncode:
                raise RuntimeError(f'Launcher exited {result.returncode}; records retained; no automatic failed-case retry')
            state.update(status='complete', exit_code=0)
        except BaseException as error:
            state.update(status='failed', exit_code=1, error=repr(error))
            print('A58 SESSION FAILED:', repr(error), flush=True)
        finally:
            state['finished_utc'] = datetime.now(timezone.utc).isoformat()
            write(BASE/'session.json', state)
        return state['exit_code']


def client(status_only=False):
    BASE.mkdir(parents=True, exist_ok=True)
    log, state_file = BASE/'session.log', BASE/'session.json'
    if status_only:
        print(dict(active=busy(BASE/'session.lock'), session=read(state_file) if state_file.exists() else None,
                   progress=progress(ROOT)))
        if log.exists():
            print('\n'.join(log.read_text(errors='replace').splitlines()[-30:]))
        return 0
    with exclusive(BASE/'client.lock'):
        active = busy(BASE/'session.lock')
        prior = read(state_file) if state_file.exists() else {}
        if not active and prior.get('status') == 'complete':
            return client(status_only=True)
        if not active and prior.get('status') == 'failed':
            print('A58 previous launch failed; use --status. No automatic repeated retry.')
            return 1
        if not active:
            with log.open('ab', buffering=0) as output:
                worker = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--coordinator'],
                    cwd=PROJECTS, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                    start_new_session=True, close_fds=True)
            for _ in range(100):
                if busy(BASE/'session.lock') or worker.poll() is not None:
                    break
                time.sleep(.05)
        print('A58 log follower; reconnect with the same command. Final summary and lightweight records are archived automatically.', flush=True)
    offset = 0
    if log.exists():
        data = log.read_bytes(); print(data[-16000:].decode(errors='replace'), end='', flush=True); offset = len(data)
    while True:
        with log.open('rb') as stream:
            stream.seek(offset); data = stream.read(); offset = stream.tell()
        if data:
            print(data.decode(errors='replace'), end='', flush=True)
        if not busy(BASE/'session.lock'):
            with log.open('rb') as stream:
                stream.seek(offset); data = stream.read()
            if data:
                print(data.decode(errors='replace'), end='', flush=True)
            state = read(state_file) if state_file.exists() else {}
            return int(state.get('exit_code', 1))
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', action='store_true'); parser.add_argument('--coordinator', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    return coordinator() if args.coordinator else client(args.status)


if __name__ == '__main__':
    sys.exit(main())
