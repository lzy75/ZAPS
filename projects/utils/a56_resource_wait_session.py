"""Resume the frozen A56S2 plan after its zero-call GPU admission failure only."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
from utils import a56_selection_r2 as frozen
from utils.a54_session import busy, digest, exclusive, read, write

PRIOR = frozen.BASE
ROOT = frozen.ROOT
BASE = PRIOR/'resource_wait_20261027'
EVIDENCE = ROOT/'resource_resume_20261027'
MIN_FREE_MIB = 12288
RESOURCE_ERROR = 'A56 ImageNet graph requires an allocated GPU with at least12 GiB free; no run started'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def select_devices(rows, visible=None, first=None, second=None):
    """None means wait; invalid allocation/explicit choices remain errors."""
    aliases = {str(token): r for r in rows for token in (r['index'], r['uuid'])}
    allowed = [str(r['index']) for r in rows] if visible is None else [v.strip() for v in visible.split(',')]
    if not allowed or any(v not in aliases for v in allowed):
        raise ValueError('CUDA_VISIBLE_DEVICES is empty or unrecognized; no allocation override')
    tokens = {}
    for token in allowed:
        tokens.setdefault(aliases[token]['uuid'], token)
    for token in (first, second):
        if token is not None and (token not in aliases or aliases[token]['uuid'] not in tokens):
            raise ValueError('Explicit GPU outside CUDA_VISIBLE_DEVICES: '+str(token))
    good = sorted((u for u in tokens if int(aliases[u]['free']) >= MIN_FREE_MIB),
                  key=lambda u: -int(aliases[u]['free']))
    if not good or any(t is not None and int(aliases[t]['free']) < MIN_FREE_MIB for t in (first, second)):
        return None
    one = aliases[first]['uuid'] if first is not None else None
    two = aliases[second]['uuid'] if second is not None else None
    if one is None:
        one = next((u for u in good if u != two), good[0])
    if two is None:
        two = next((u for u in good if u != one), one)
    return [tokens[one], tokens[two]]


def inventory():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    result = []
    for line in raw.splitlines():
        index, uuid, free = (x.strip() for x in line.split(','))
        result.append(dict(index=index, uuid=uuid, free=int(free)))
    return result


def no_attempts():
    if list(ROOT.rglob('run.json')) or list(ROOT.rglob('attempt_*')):
        raise ValueError('A56 case attempt already exists; no zero-call resource retry')


def source_files():
    return [Path(__file__), PROJECTS/'utils/test_a56_resource_wait.py',
            PROJECTS/'experiment_cards/A56_resource_wait_20261027.md']


def admit_frozen():
    """Caller holds prior session/client locks; preserve exact plan and failed bytes."""
    state = read(PRIOR/'session.json')
    text = (PRIOR/'session.log').read_text(errors='replace')
    if state.get('status') != 'failed' or RESOURCE_ERROR not in text:
        raise ValueError('Not the reported A56S2 pre-worker resource failure; no automatic retry')
    no_attempts()
    frozen.proof()
    plan = frozen.original.check(ROOT)
    if plan.get('implementation_revision') != 'A56_SELECTION_R2' or len(plan['cases']) != 20:
        raise ValueError('Need the existing signed A56S2 twenty-case frozen plan')
    EVIDENCE.mkdir(exist_ok=True)
    manifest = EVIDENCE/'resume_manifest.json'
    own = {str(p): digest(p) for p in source_files()}
    if manifest.exists():
        m = read(manifest)
        if m['plan_sha256'] != digest(ROOT/'plan.json') or m['resume_sources'] != own:
            raise ValueError('Frozen resource recovery binding changed')
        if any(digest(p) != sha for p, sha in m['snapshots'].items()):
            raise ValueError('Preserved resource failure bytes changed')
        return m
    target = EVIDENCE/'source_snapshot'
    target.mkdir()
    snapshots = {}
    for pattern in ('*.log', 'session.json', 'devices.txt'):
        for p in PRIOR.glob(pattern):
            if p.is_file():
                copy = target/p.name
                shutil.copyfile(p, copy)
                snapshots[str(copy)] = digest(copy)
    for p in source_files():
        copy = target/p.name
        shutil.copyfile(p, copy)
        snapshots[str(copy)] = digest(copy)
    m = dict(created_utc=stamp(), plan_sha256=digest(ROOT/'plan.json'),
        resume_sources=own, snapshots=snapshots, failed_UNet_calls=0, minimum_free_MiB=MIN_FREE_MIB,
        limits='Resource wait only; original plan/source/scientific gates unchanged. No attempt retry after any case starts.')
    write(manifest, m)
    print('A56 RESOURCE RESUME PASS: frozen20 cases unchanged; original failed logs saved; no model calls.', flush=True)
    return m


def progress():
    from utils.a56_selection_r2_session import progress as original_progress
    return original_progress(ROOT)


def coordinator():
    BASE.mkdir(parents=True, exist_ok=True)
    with exclusive(BASE/'session.lock'):
        state = dict(pid=os.getpid(), status='checking_frozen_resource_failure', started_utc=stamp())
        write(BASE/'session.json', state)
        try:
            with ExitStack() as locks:
                for path in (PRIOR/'client.lock', PRIOR/'session.lock',
                             Path(frozen.OLD_ROOT).parent/'client.lock'):
                    locks.enter_context(exclusive(path))
                # The old launcher cannot be active while validating absence of attempts.
                with exclusive(Path(frozen.OLD_ROOT).parent/'launcher.lock'):
                    manifest = admit_frozen()
                cycle = 0
                while True:
                    rows = inventory()
                    gpus = select_devices(rows, os.environ.get('CUDA_VISIBLE_DEVICES'),
                        os.environ.get('ZAPS_A56_GPU0'), os.environ.get('ZAPS_A56_GPU1'))
                    if gpus is None:
                        state.update(status='waiting_for_gpu', minimum_free_MiB=MIN_FREE_MIB, inventory=rows,
                                     checked_utc=stamp())
                        write(BASE/'session.json', state)
                        print('A56 waiting for >=12GiB free within allocation; no UNet calls; freeMiB=',
                              {r['index']: r['free'] for r in rows}, flush=True)
                        time.sleep(30)
                        continue
                    no_attempts()
                    if digest(ROOT/'plan.json') != manifest['plan_sha256']:
                        raise ValueError('Frozen plan changed while waiting')
                    state.update(status='running_frozen_launcher', selected_gpus=gpus, admitted_utc=stamp())
                    write(BASE/'session.json', state)
                    cycle += 1
                    event = dict(utc=stamp(), selected_gpus=gpus, inventory=rows, plan_sha256=manifest['plan_sha256'])
                    write(EVIDENCE/f'admission_{cycle:03d}.json', event)
                    print('A56 admitted GPUs:', gpus, '; same IDs mean serial shards; original plan reused.', flush=True)
                    env = dict(os.environ, ZAPS_A56_GPU0=gpus[0], ZAPS_A56_GPU1=gpus[1])
                    log = PRIOR/'launcher_20261026.log'
                    prior_log_stamp = log.stat().st_mtime_ns if log.exists() else None
                    result = subprocess.run(['bash', str(PROJECTS/'run_a56_selection_r2_20261027.sh')],
                        cwd=PROJECTS, stdin=subprocess.DEVNULL, env=env)
                    if result.returncode == 0:
                        break
                    # Only a fresh, recognizable resource race before ANY attempt can wait again.
                    latest = log.read_text(errors='replace') if log.exists() else ''
                    no_attempts()
                    fresh_log = log.exists() and log.stat().st_mtime_ns != prior_log_stamp
                    resource_race = fresh_log and any(message in '\n'.join(latest.splitlines()[-20:]) for message in (
                        RESOURCE_ERROR, 'A56 explicit GPU requires at least12 GiB free:',
                        'No allocated GPU has 6144 MiB free (', 'Explicit GPU has less than 6 GiB free:'))
                    if not resource_race:
                        raise RuntimeError(f'Frozen launcher exited {result.returncode}; failure retained, no retry')
                    shutil.copyfile(log, EVIDENCE/f'resource_race_{cycle:03d}.log')
                    print('A56 pre-worker GPU availability changed; zero attempts; returning to resource wait.', flush=True)
                    time.sleep(30)
                if digest(ROOT/'plan.json') != manifest['plan_sha256']:
                    raise ValueError('Frozen plan changed during launch')
                state.update(status='complete', exit_code=0)
        except BaseException as error:
            state.update(status='failed', exit_code=1, error=repr(error))
            print('A56 RESOURCE SESSION FAILED:', repr(error), flush=True)
        finally:
            state['finished_utc'] = stamp()
            write(BASE/'session.json', state)
        return state['exit_code']


def client(status_only=False):
    BASE.mkdir(parents=True, exist_ok=True)
    log, state_file = BASE/'session.log', BASE/'session.json'
    if status_only:
        print(dict(active=busy(BASE/'session.lock'), session=read(state_file) if state_file.exists() else None,
                   progress=progress()))
        if log.exists():
            print('\n'.join(log.read_text(errors='replace').splitlines()[-30:]))
        return 0
    with exclusive(BASE/'client.lock'):
        active = busy(BASE/'session.lock')
        prior = read(state_file) if state_file.exists() else {}
        if not active and prior.get('status') == 'complete':
            return client(True)
        if not active and prior.get('status') == 'failed':
            print('A56 resource resume failed; use --status. Scientific/software failures are not auto-retried.')
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
        print('A56 resource wait log follower; SSH disconnect does not stop job; reconnect with same command.', flush=True)
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
        if not busy(BASE/'session.lock'):
            with log.open('rb') as stream:
                stream.seek(offset)
                data = stream.read()
            if data:
                print(data.decode(errors='replace'), end='', flush=True)
            state = read(state_file) if state_file.exists() else {}
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
