"""Disconnect-safe A54 coordinator. Historical plan/worker/launcher stay unchanged.

The foreground client only follows a detached coordinator's log. Reconnecting
and issuing the same command attaches to it. No automatic scientific-failure
retry: one stale, interrupted control attempt per case may be preserved/replayed.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECTS = Path(__file__).resolve().parents[1]
BASE = Path('/data0/lzy/ZAPS-results/diag_norm_gain_value')
ROOT = BASE / 'A54_20261026'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


@contextmanager
def exclusive(path):
    """Never infer liveness from a PID, run.json status, or an old lock filename."""
    with Path(path).open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def busy(path):
    try:
        with exclusive(path):
            return False
    except BlockingIOError:
        return True


def recover_stale(root):
    """Caller holds launcher AND both shard locks; preserve bytes before replay."""
    root = Path(root)
    ledger_path = root / 'interruption_recovery.json'
    if not root.exists():
        return None
    plan = read(root / 'plan.json')
    if plan['experiment'] != 'A54_current_norm_gain':
        raise ValueError('Wrong experiment; no recovery mutation')
    plan_sha = digest(root / 'plan.json')
    ledger = read(ledger_path) if ledger_path.exists() else dict(
        schema_version=1, experiment=plan['experiment'], plan_sha256=plan_sha, entries=[],
        limits='Interrupted recorded attempts are additional to the original A54 summary. '
               'Counters record attempted calls, not verified completed forwards; abrupt-loss cost can be incomplete. '
               'Only stale running/KeyboardInterrupt attempts replayed once, from original frozen initial state. '
               'No metric-based selection; completed cases and original plan/source bytes unchanged.')
    if ledger['plan_sha256'] != plan_sha:
        raise ValueError('Recovery ledger/plan mismatch')
    # Finish a previously journaled rename, without deleting either copy.
    for entry in ledger['entries']:
        old, saved = root / entry['original'], root / entry['preserved']
        if entry['state'] == 'preserved':
            if not saved.exists() or digest(saved / 'run.json') != entry['run_sha256']:
                raise ValueError('Preserved interrupted accounting changed')
            continue
        if old.exists() and saved.exists():
            raise ValueError('Ambiguous recovery copies; retained for inspection')
        if not saved.exists():
            if not old.exists():
                raise ValueError('Journaled interrupted bytes missing')
            if digest(old / 'run.json') != entry['run_sha256']:
                raise ValueError('Journaled attempt changed; no move')
            saved.parent.mkdir(parents=True, exist_ok=True)
            old.rename(saved)
        entry['state'] = 'preserved'
    ledger['recorded_interrupted_forwards'] = sum(e['recorded_attempted_forwards'] for e in ledger['entries'])
    ledger['recorded_interrupted_backwards'] = sum(e['recorded_backward_attempts'] for e in ledger['entries'])
    additions = []
    for case in plan['cases']:
        key = case['key']
        if Path(key).name != key:
            raise ValueError('Unsafe case key')
        attempts = sorted((root / key).glob('control_*'))
        if len(attempts) > 1:
            raise ValueError('Multiple active attempts; no output selection')
        for folder in attempts:
            record = folder / 'run.json'
            if not record.is_file():
                raise ValueError('Attempt lacks readable accounting; retained, not replayed')
            row = read(record)
            if row['case'] != case or row['plan_sha256'] != plan_sha:
                raise ValueError('Attempt identity differs')
            if row['status'] == 'complete':
                # Full numerical/cached-artifact validation remains in A54.item.
                continue
            interrupted = row['status'] == 'running' or (
                row['status'] == 'failed' and row.get('error', '').startswith('KeyboardInterrupt('))
            if not interrupted:
                raise ValueError('Scientific/software failure is not auto-retried: ' + str(record))
            if any(entry['case'] == key for entry in ledger['entries']):
                raise ValueError('One interruption replay already used for ' + key)
            nfe, back = row.get('nfe', 0), row.get('backward_attempts', 0)
            if not isinstance(nfe, int) or not isinstance(back, int) or not 0 <= back <= nfe <= 90:
                raise ValueError('Invalid partial-attempt counters')
            sha = digest(record)
            saved = Path('interrupted_attempts') / key / (folder.name + '_' + sha[:12])
            if (root / saved).exists():
                raise ValueError('Preservation destination exists')
            inventory = {str(f.relative_to(folder)): dict(bytes=f.stat().st_size, sha256=digest(f))
                         for f in sorted(folder.rglob('*')) if f.is_file()}
            additions.append(dict(case=key, original=str(folder.relative_to(root)), preserved=str(saved),
                run_sha256=sha, recorded_attempted_forwards=nfe, recorded_backward_attempts=back,
                prior_status=row['status'], inventory=inventory, state='journaled'))
    if additions:
        ledger['entries'].extend(additions)
        ledger['recorded_interrupted_forwards'] = sum(e['recorded_attempted_forwards'] for e in ledger['entries'])
        ledger['recorded_interrupted_backwards'] = sum(e['recorded_backward_attempts'] for e in ledger['entries'])
        # Write-ahead journal: a crash during moves can be recovered on next entry.
        write(ledger_path, ledger)
        for entry in additions:
            old, saved = root / entry['original'], root / entry['preserved']
            saved.parent.mkdir(parents=True, exist_ok=True)
            old.rename(saved)
            entry['state'] = 'preserved'
        write(ledger_path, ledger)
    elif ledger_path.exists():
        write(ledger_path, ledger)
    print('A54 recovery: preserved interrupted=', len(ledger['entries']),
          '; completed cases reused; original frozen plan unchanged.', flush=True)
    return ledger


def progress(root):
    rows = []
    for path in Path(root).glob('*/control_*/run.json'):
        try:
            rows.append(read(path))
        except (OSError, ValueError):
            continue
    return dict(completed=sum(r.get('status') == 'complete' for r in rows), planned=10,
                active_recorded_forward_attempts=sum(r.get('nfe', 0) for r in rows))


def coordinator():
    BASE.mkdir(parents=True, exist_ok=True)
    with exclusive(BASE / 'session.lock'):
        state_path = BASE / 'session.json'
        state = dict(pid=os.getpid(), status='waiting_for_previous_launcher',
                     started_utc=datetime.now(timezone.utc).isoformat())
        write(state_path, state)
        try:
            # An old frontend may be dead while a shard is still exiting. Never
            # move a running worker's directory, even if launcher.lock is free.
            while True:
                try:
                    with exclusive(BASE / 'launcher.lock'):
                        from contextlib import ExitStack
                        with ExitStack() as stack:
                            if ROOT.exists():
                                for shard in (0, 1):
                                    stack.enter_context(exclusive(ROOT / f'control_shard{shard}.lock'))
                            ledger = recover_stale(ROOT)
                    break
                except BlockingIOError:
                    print('A54 existing launcher/shard still active; waiting, no duplicate.', progress(ROOT), flush=True)
                    time.sleep(30)
            state['status'] = 'running_detached'
            write(state_path, state)
            print('A54 detached coordinator: SSH follower can disconnect; GPU job remains in this session.', flush=True)
            result = subprocess.run(['bash', str(PROJECTS / 'run_a54_20261026.sh')],
                                    cwd=PROJECTS, stdin=subprocess.DEVNULL)
            if result.returncode:
                raise RuntimeError(f'A54 launcher exited {result.returncode}; inspect retained log, no automatic failure retry')
            summary = read(ROOT / 'summary.json')
            waste_f = (ledger or {}).get('recorded_interrupted_forwards', 0)
            waste_b = (ledger or {}).get('recorded_interrupted_backwards', 0)
            print('A54 execution accounting: successful new forwards=', summary['diagnostic_nfe'],
                  '; interrupted recorded attempts=', waste_f,
                  '; combined recorded forward attempts=', summary['diagnostic_nfe'] + waste_f,
                  '; combined backward attempts=', summary['backward_attempts'] + waste_b,
                  '; interrupted costs remain in interruption_recovery.json and archive.', flush=True)
            state.update(status='complete', exit_code=0)
        except BaseException as error:
            state.update(status='failed', error=repr(error), exit_code=1)
            print('A54 SESSION FAILED:', repr(error), flush=True)
        finally:
            state['finished_utc'] = datetime.now(timezone.utc).isoformat()
            write(state_path, state)
        return state['exit_code']


def client(status_only=False):
    BASE.mkdir(parents=True, exist_ok=True)
    log = BASE / 'session.log'
    state_path = BASE / 'session.json'
    if status_only:
        print(json.dumps(dict(session=read(state_path) if state_path.exists() else None,
                              active=busy(BASE / 'session.lock'), progress=progress(ROOT)), ensure_ascii=False))
        if log.exists():
            print('\n'.join(log.read_text(errors='replace').splitlines()[-25:]))
        return 0
    with exclusive(BASE / 'client.lock'):
        active = busy(BASE / 'session.lock')
        prior = read(state_path) if state_path.exists() else {}
        if not active and prior.get('status') == 'complete':
            return client(status_only=True)
        if not active and prior.get('status') == 'failed':
            print('A54 prior detached run failed. Inspect --status; no automatic repeated retry.')
            return 1
        if not active:
            with log.open('ab', buffering=0) as output:
                process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--coordinator'],
                    cwd=PROJECTS, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                    start_new_session=True, close_fds=True)
            for _ in range(100):
                if busy(BASE / 'session.lock') or process.poll() is not None:
                    break
                time.sleep(.05)
        print('A54 log follower; disconnecting this client does not stop the detached job. '
              'Reconnect with the same command; --status gives a snapshot.', flush=True)
    offset = 0
    if log.exists():
        data = log.read_bytes()
        print(data[-16000:].decode(errors='replace'), end='', flush=True)
        offset = len(data)
    while True:
        with log.open('rb') as stream:
            stream.seek(offset)
            data = stream.read()
            offset = stream.tell()
        if data:
            print(data.decode(errors='replace'), end='', flush=True)
        if not busy(BASE / 'session.lock'):
            with log.open('rb') as stream:
                stream.seek(offset)
                final = stream.read()
            if final:
                print(final.decode(errors='replace'), end='', flush=True)
            state = read(state_path) if state_path.exists() else {}
            return int(state.get('exit_code', 1))
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--coordinator', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    return coordinator() if args.coordinator else client(args.status)


if __name__ == '__main__':
    sys.exit(main())
