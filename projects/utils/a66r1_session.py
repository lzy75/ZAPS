"""A66R1: preserve frozen sources, inject transparent counter, resume unfinished runs."""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

PROJECTS = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(PROJECTS))
from utils import a66_session as session
from utils import diag_current_strong_anchor as driver
from utils import counted_adjoint_repair as repair
from utils import current_strong_anchor as rules
from utils import historical_ffhq_gain as design
from utils.a54_session import busy, exclusive

ENTRY = Path(__file__).resolve()


def run(*args):
    if args and args[0] == 'utils/diag_current_strong_anchor.py':
        cmd = [sys.executable, '-u', str(ENTRY), '--repair-driver', *args[1:]]
    else: cmd = [sys.executable, '-u', *args]
    subprocess.run(cmd, cwd=PROJECTS, check=True)


def phase(root, state, args):
    repair.manifest_check(root)
    p = rules.read(root/'plan.json')
    devices = {r['uuid']: r for r in design.available_gpus(session.gpu_inventory(), args.gpus, args.visible)}
    if any(u not in devices for u in p['required_devices'].values()):
        print('A66R1 pinned GPU outside allocation; no model run.', flush=True); return False
    queued = [i for i in (0, 1) if session.status(root, i) != 'complete']; running = {}; failed = False
    rules.require(not any(session.status(root, i) == 'failed' for i in queued), 'Paid failure retained; no automatic retry')
    while queued or running:
        occupied = {u for _, u, _ in running.values()}
        for shard in list(queued):
            uuid = p['required_devices'][str(shard)]
            if uuid in occupied: continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
            log = (state/f'shard{shard}.log').open('ab', buffering=0)
            process = subprocess.Popen([sys.executable, '-u', str(ENTRY), '--repair-driver', '--worker',
                '--root', str(root), '--shard', str(shard), '--device-uuid', uuid], cwd=PROJECTS,
                env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            running[shard] = (process, uuid, log); queued.remove(shard); occupied.add(uuid)
            print('A66R1 WORKER START:', shard, 'GPU', devices[uuid]['index'], flush=True)
        for shard, (process, uuid, log) in list(running.items()):
            if process.poll() is not None:
                log.close(); del running[shard]; failed |= process.returncode != 0
                print('A66R1 WORKER END:', shard, 'exit', process.returncode, flush=True)
        if failed:
            # A shared interface error need not be repeated on another unstarted shard.
            queued.clear()
        print('A66R1 QUEUE STATUS:', {i: session.status(root, i) for i in (0, 1)}, flush=True)
        if running: time.sleep(30)
    if failed: raise RuntimeError('A66R1 worker failed; no automatic retry')
    return all(session.status(root, i) == 'complete' for i in (0, 1))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--repair-driver', action='store_true')
    known, rest = parser.parse_known_args(); sys.argv = [str(ENTRY), *rest]
    with repair.installed():
        if known.repair_driver:
            original = driver.summarize
            def summarize(root):
                value = repair.augment_summary(root, original(root))
                print('A66R1 COST AUDIT:', value['new_cost_including_archived_failures'], flush=True)
                return value
            with patch.object(driver, 'summarize', summarize): driver.main()
            return
        options = argparse.ArgumentParser(add_help=False)
        options.add_argument('--root', default=rules.ROOT)
        options.add_argument('--status', action='store_true'); options.add_argument('--archive-only', action='store_true')
        options.add_argument('--coordinator', action='store_true')
        args, _ = options.parse_known_args(rest)
        root = Path(args.root).resolve(); state = session.state_dir(root); state.mkdir(parents=True, exist_ok=True)
        if not (args.status or args.archive_only or args.coordinator):
            with exclusive(state/'repair_r1.lock'):
                if not busy(state/'coordinator.lock'):
                    rules.require(not any(busy(f) for f in (root/'workers').glob('*/worker.lock')), 'Live worker; use --status')
                    driver.check(root)
                    def completed():
                        p = rules.read(root/'plan.json')
                        for file in (root/'cases').glob('*/*/repeat_*/run.json'):
                            r = rules.read(file)
                            if r['status'] == 'complete':
                                c = next(c for c in p['cases'] if c['key'] == r['case']['key'])
                                driver.executor.done(root, c, r['arm'], r['repeat'])
                    repair.prepare_recovery(root, state, validate_completed=completed)
        with ExitStack() as stack:
            stack.enter_context(patch.object(session, 'run', run))
            stack.enter_context(patch.object(session, 'phase', phase))
            stack.enter_context(patch.object(session, '__file__', str(ENTRY)))
            session.main()


if __name__ == '__main__': main()
