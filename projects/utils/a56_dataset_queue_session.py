"""Four disjoint frozen A56 groups, one process per GPU, dataset-specific admission."""
import argparse
from contextlib import ExitStack, contextmanager
import gc
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
from utils import a56_resource_wait_session as old
from utils import a56_selection_r2 as frozen
from utils.a54_session import busy, digest, exclusive, read, write

ROOT, PRIOR = frozen.ROOT, frozen.BASE
BASE = PRIOR/'dataset_queue_20261027'
EVIDENCE = ROOT/'dataset_queue_execution'
MIN_FREE = dict(ffhq=6144, imagenet=12288)
TESTS = ('utils.test_a56_dataset_queue', 'utils.test_a56_selection_r2',
    'utils.test_frozen_direction_transfer', 'utils.test_current_direction_control',
    'utils.test_linear_adjoint_autograd_guard', 'utils.test_current_score_checkpoint_r2')


def groups(plan):
    result = []
    for dataset in frozen.original.rules.DATASETS:
        for shard in (0, 1):
            cases = [c for i, c in enumerate(plan['cases']) if i % 2 == shard and c['dataset'] == dataset]
            result.append(dict(key=f'{dataset}_{shard}', dataset=dataset, shard=shard,
                               case_keys=[c['key'] for c in cases], minimum_free_MiB=MIN_FREE[dataset]))
    keys = [key for g in result for key in g['case_keys']]
    if len(keys) != len(set(keys)) or set(keys) != {c['key'] for c in plan['cases']} or len(keys) != 20:
        raise ValueError('Queue must partition the exact frozen twenty cases once')
    return result


def allocation(rows, visible=None, first=None, second=None):
    aliases = {str(token): r for r in rows for token in (r['index'], r['uuid'])}
    allowed = list(aliases[r['uuid']]['index'] for r in rows) if visible is None else [v.strip() for v in visible.split(',')]
    if not allowed or any(t not in aliases for t in allowed):
        raise ValueError('Empty/unrecognized CUDA_VISIBLE_DEVICES; allocation unchanged')
    tokens = {}
    for t in allowed:
        tokens.setdefault(aliases[t]['uuid'], t)
    explicit = [t for t in (first, second) if t is not None]
    for t in explicit:
        if t not in aliases or aliases[t]['uuid'] not in tokens:
            raise ValueError('Explicit GPU outside CUDA_VISIBLE_DEVICES: '+str(t))
    permitted = {aliases[t]['uuid'] for t in explicit} if explicit else set(tokens)
    return [dict(aliases[u], token=tokens[u]) for u in tokens if u in permitted]


def assignments(pending, rows, occupied=(), slots=2):
    available = [r for r in rows if r['uuid'] not in occupied]
    result = []
    # Give the large-memory groups a suitable card first, leaving smaller cards to FFHQ.
    for job in sorted(pending, key=lambda g: (-g['minimum_free_MiB'], g['key'])):
        fits = [r for r in available if r['free'] >= job['minimum_free_MiB']]
        if not fits or len(result) >= slots:
            continue
        gpu = min(fits, key=lambda r: r['free'])
        result.append((job, gpu))
        available = [r for r in available if r['uuid'] != gpu['uuid']]
    return result


def require_no_attempts(case_keys=None):
    folders = [ROOT/k for k in case_keys] if case_keys is not None else [ROOT]
    if any(list(p.rglob('run.json')) or list(p.rglob('attempt_*')) for p in folders):
        raise ValueError('A case attempt exists; no zero-call queue handoff/retry')


def waiter_identity(state, command, owner, uid, pgid):
    expected = str(PROJECTS/'utils/a56_resource_wait_session.py')
    return bool(state.get('status') == 'waiting_for_gpu' and isinstance(state.get('pid'), int)
        and state['pid'] > 1 and owner == uid and pgid == state['pid']
        and expected in command and '--coordinator' in command)


def handoff_waiter():
    """Caller holds public launcher lock: no old child can reach a model call."""
    old.BASE.mkdir(parents=True, exist_ok=True)
    if not busy(old.BASE/'session.lock'):
        return
    state = read(old.BASE/'session.json')
    require_no_attempts()
    pid = state.get('pid')
    if not isinstance(pid, int):
        raise ValueError('Old coordinator PID is invalid; no process signaled')
    proc = Path('/proc')/str(pid)
    command = [v.decode() for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
    if not waiter_identity(state, command, proc.stat().st_uid, os.geteuid(), os.getpgid(pid)):
        raise ValueError('Old job is not the verified zero-call resource waiter; no process signaled')
    saved = EVIDENCE/'old_waiter_snapshot'; saved.mkdir(parents=True, exist_ok=True)
    for name in ('session.json', 'session.log'):
        shutil.copyfile(old.BASE/name, saved/name)
    os.killpg(pid, signal.SIGTERM)
    for _ in range(100):
        if not busy(old.BASE/'session.lock'):
            print('A56 prior zero-call resource waiter stopped; frozen plan retained.', flush=True)
            return
        time.sleep(.05)
    raise ValueError('Old waiter has not released its lock; no new worker started')


def own_sources():
    paths = [Path(__file__), PROJECTS/'utils/test_a56_dataset_queue.py',
             PROJECTS/'experiment_cards/A56_dataset_queue_20261027.md']
    return {str(p): digest(p) for p in paths}


def manifest():
    path = EVIDENCE/'manifest.json'
    m = read(path)
    if m['plan_sha256'] != digest(ROOT/'plan.json') or m['queue_sources'] != own_sources():
        raise ValueError('Queue source/plan binding changed')
    if any(digest(p) != sha for p, sha in m['snapshots'].items()):
        raise ValueError('Queue execution evidence changed')
    return m


@contextmanager
def execution():
    from unittest.mock import patch
    m = manifest()
    pin = dict(kind='DATASET_QUEUE_ONLY', manifest_sha256=digest(EVIDENCE/'manifest.json'), plan_sha256=m['plan_sha256'])
    def tagged(writer):
        def save(path, value):
            if Path(path).name in ('run.json', 'summary.json', 'evaluation.json'):
                value = dict(value, queue_execution=pin)
            return writer(path, value)
        return save
    with frozen.bindings():
        d = frozen.original
        with patch.object(d, 'signed', tagged(d.signed)), patch.object(d.a40.source, 'signed', tagged(d.a40.source.signed)):
            yield


def worker(key):
    # This check precedes importing/loading the real model and creating any case attempt.
    m = manifest()
    job = next(g for g in m['groups'] if g['key'] == key)
    rows = allocation(old.inventory(), os.environ.get('CUDA_VISIBLE_DEVICES'))
    if len(rows) != 1 or rows[0]['free'] < job['minimum_free_MiB']:
        require_no_attempts(job['case_keys'])
        print('A56_QUEUE_RESOURCE_WAIT: no case attempt, threshold=', job['minimum_free_MiB'], flush=True)
        return 75
    import torch
    from modules.main_single import load_diffusion_model
    d = frozen.original
    with exclusive(EVIDENCE/(key+'.lock')), execution():
        p = d.check(ROOT)
        actual = dict(torch=str(torch.__version__), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
            gpu=torch.cuda.get_device_name('cuda:0'), NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),
            TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        d.require(actual == p['runtime'] and d.parent.P.precision.precision_state(torch) == p['native_policy'] and
                  d.a40.model_sources() == p['model_library_sources'], 'Runtime/model libraries changed')
        cases = [c for c in p['cases'] if c['key'] in job['case_keys']]
        # No retry of incomplete/failed attempts, even if another group completed.
        for c in cases:
            for kind, validator in (('BASELINE', lambda c:d.baseline_done(ROOT,p,c,'BASELINE')), ('SAMPLING', lambda c:d.finished(ROOT,p,c))):
                if list((ROOT/c['key']/kind).glob('attempt_*')) and validator(c) is None:
                    raise ValueError('Incomplete attempt retained; no retry: '+c['key']+'/'+kind)
        torch.set_num_threads(2)
        memory = []
        with d.parent.scan.preserved_rng():
            model = load_diffusion_model(job['dataset'], 'cuda:0'); model.model.eval()
            d.require(d.source.digest(model.ckpt_path) == p['checkpoints'][job['dataset']]['sha256'] and model.num_steps == 1000 and
                all(v.dtype == torch.float32 for v in model.model.parameters()), 'Checkpoint/dtype differs')
            for c in cases:
                if d.finished(ROOT,p,c):
                    continue
                operator, mask = d.make_operator(c, 'cuda:0'); gate = d.a40.operator_gate(operator,'cuda:0')
                torch.cuda.reset_peak_memory_stats('cuda:0')
                with d.baseline_bindings():
                    d.a40.training(ROOT,p,c,model,'cuda:0',operator,gate)
                peak = int(torch.cuda.max_memory_allocated('cuda:0'))
                d.sample(ROOT,p,c,model,'cuda:0',operator,mask)
                _,record = d.finished(ROOT,p,c)
                memory.append(dict(case=c['key'], baseline_peak_allocated_bytes=peak, sampling_arm_peak_allocated_bytes=record['arm_peak_memory_bytes']))
                write(EVIDENCE/(key+'_memory.json'), dict(rows=memory, scope='Allocated peak; not free-memory guarantee or reserved/external processes'))
                gc.collect(); torch.cuda.empty_cache()
            del model
        gc.collect(); torch.cuda.empty_cache()
    return 0


def coordinator():
    BASE.mkdir(parents=True, exist_ok=True)
    with exclusive(BASE/'session.lock'):
        state = dict(pid=os.getpid(), status='preparing_queue', started_utc=old.stamp())
        write(BASE/'session.json', state)
        running = {}
        try:
            with ExitStack() as locks:
                common = Path(frozen.OLD_ROOT).parent
                locks.enter_context(exclusive(common/'launcher.lock'))
                handoff_waiter()
                for base in (PRIOR, common, common/'operator_repair_20261027', old.BASE):
                    if base.exists():
                        for name in ('client.lock','session.lock'):
                            locks.enter_context(exclusive(base/name))
                for shard in (0,1):
                    locks.enter_context(exclusive(ROOT/f'worker_shard{shard}.lock'))
                require_no_attempts()
                old.admit_frozen()
                p = frozen.original.check(ROOT)
                pending = groups(p)
                EVIDENCE.mkdir(exist_ok=True)
                snap = EVIDENCE/'source_snapshot'; snap.mkdir(exist_ok=True)
                saved = {}
                for path in own_sources():
                    target=snap/Path(path).name; shutil.copyfile(path,target); saved[str(target)]=digest(target)
                for path in (EVIDENCE/'old_waiter_snapshot').glob('*'):
                    saved[str(path)] = digest(path)
                write(EVIDENCE/'manifest.json', dict(created_utc=old.stamp(), plan_sha256=digest(ROOT/'plan.json'),
                    queue_sources=own_sources(), snapshots=saved, groups=pending,
                    scope='Same frozen twenty cases, one process/GPU, max2GPUs, original algorithm/gates; FFHQ6GiB/IN12GiB admission only'))
                manifest()
                env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
                subprocess.run([sys.executable,'-m','unittest',*TESTS,'-v'],cwd=PROJECTS,env=env,check=True)
                finished=[]; failures=[]; cancelled=[]; sequence=0; last_report=0.
                starts={};records=[]
                while pending or running:
                    for key, (proc, log, job, gpu) in list(running.items()):
                        code = proc.poll()
                        if code is None:
                            continue
                        log.close(); del running[key]
                        records.append(dict(group=key,exit_code=code,started_utc=starts[key][1],finished_utc=old.stamp(),
                            elapsed_seconds=time.monotonic()-starts[key][0],gpu_uuid=gpu['uuid']))
                        if code == 0:
                            finished.append(key); print('A56 QUEUE GROUP COMPLETE:',key,flush=True)
                        elif code == 75:
                            require_no_attempts(job['case_keys']); pending.append(job)
                        else:
                            failures.append(key)
                            print('A56 QUEUE GROUP FAILED, no retry:',key,flush=True)
                            print('\n'.join((BASE/(key+'.log')).read_text(errors='replace').splitlines()[-60:]),flush=True)
                    if failures:
                        cancelled.extend(g['key'] for g in pending);pending=[]
                    rows = allocation(old.inventory(),os.environ.get('CUDA_VISIBLE_DEVICES'),os.environ.get('ZAPS_A56_GPU0'),os.environ.get('ZAPS_A56_GPU1'))
                    for job,gpu in assignments(pending,rows,[item[3]['uuid'] for item in running.values()],2-len(running)):
                        sequence+=1
                        write(EVIDENCE/f'admission_{sequence:03d}.json',dict(utc=old.stamp(),job=job['key'],gpu=gpu))
                        log=(BASE/(job['key']+'.log')).open('ab',buffering=0)
                        worker_env=dict(env,CUDA_VISIBLE_DEVICES=gpu['token'])
                        proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',job['key']],
                            cwd=PROJECTS,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=worker_env,close_fds=True)
                        running[job['key']]=(proc,log,job,gpu);pending.remove(job)
                        starts[job['key']]=(time.monotonic(),old.stamp())
                        print('A56 QUEUE START:',job['key'],'GPU=',gpu['token'],'cases=',len(job['case_keys']),flush=True)
                    state.update(status='running_or_waiting_for_memory',completed_groups=finished,failed_groups=failures,
                        running_groups=list(running),pending_groups=[g['key'] for g in pending],cancelled_groups=cancelled,inventory=rows)
                    write(BASE/'session.json',state)
                    write(EVIDENCE/'schedule_summary.json',dict(started_utc=state['started_utc'],updated_utc=old.stamp(),
                        completed_groups=finished,failed_groups=failures,cancelled_groups=cancelled,group_launches=records,
                        minimum_free_MiB=MIN_FREE,scope='Queue timings include load/writes/shared load; no speedup guarantee; original NN accounting remains in run records'))
                    if time.monotonic()-last_report>=30:
                        print('A56 QUEUE PROGRESS:',old.progress(),'; pending=',state['pending_groups'],'; running=',list(running),flush=True)
                        last_report=time.monotonic()
                    if pending or running:
                        time.sleep(5 if running else 30)
                if failures:
                    raise RuntimeError('Groups failed, original attempts retained, no automatic retry: '+str(failures))
                state['status']='archiving';write(BASE/'session.json',state)
                with execution():
                    before=sys.argv
                    try:
                        sys.argv=[frozen.original.__file__,'--archive']
                        frozen.original.main()
                    finally:sys.argv=before
                try:
                    subprocess.run(['git','push','origin',frozen.BRANCH],cwd=PROJECTS,env=dict(env,GIT_TERMINAL_PROMPT='0'),timeout=45,check=True)
                    print('RESULTS_PUSH_PASS: A56 dataset queue results uploaded.',flush=True)
                except (subprocess.SubprocessError,OSError) as error:
                    print('RESULTS_PUSH_PENDING: archive retained, no GPU rerun;',repr(error),flush=True)
                state.update(status='complete',exit_code=0)
        except BaseException as error:
            for proc, log, _, _ in running.values():
                if proc.poll() is None:proc.terminate()
            for proc, log, _, _ in running.values():
                proc.wait();log.close()
            state.update(status='failed',exit_code=1,error=repr(error))
            print('A56 DATASET QUEUE FAILED:',repr(error),'; attempts retained; no automatic retry.',flush=True)
        finally:
            state['finished_utc']=old.stamp();write(BASE/'session.json',state)
        return state['exit_code']


def client(status_only=False):
    BASE.mkdir(parents=True, exist_ok=True)
    log,state_file=BASE/'session.log',BASE/'session.json'
    if not busy(BASE/'session.lock'):
        if old.BASE.exists() and busy(old.BASE/'session.lock'):
            state=read(old.BASE/'session.json')
            if state.get('status')!='waiting_for_gpu':
                print('Existing A56 resource job already checking/running; following it without interruption.',flush=True)
                return old.client(status_only)
        elif PRIOR.exists() and busy(PRIOR/'session.lock'):
            from utils import a56_selection_r2_session as earlier
            print('Existing A56 selection launcher active; following it without interruption.',flush=True)
            return earlier.client(status_only)
    if status_only:
        print(dict(active=busy(BASE/'session.lock'),session=read(state_file) if state_file.exists() else None,progress=old.progress()))
        if log.exists():print('\n'.join(log.read_text(errors='replace').splitlines()[-30:]))
        return 0
    with exclusive(BASE/'client.lock'):
        active=busy(BASE/'session.lock');prior=read(state_file) if state_file.exists() else {}
        if not active and prior.get('status')=='complete':return client(True)
        if not active and prior.get('status')=='failed':
            print('A56 queue previously failed; use --status; no automatic attempt retry.');return 1
        if not active:
            with log.open('ab',buffering=0) as out:
                proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--coordinator'],cwd=PROJECTS,
                    stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            for _ in range(100):
                if busy(BASE/'session.lock') or proc.poll() is not None:break
                time.sleep(.05)
    print('A56 four-group queue; detached; reconnect with this same command.',flush=True)
    offset=0
    if log.exists():
        data=log.read_bytes();print(data[-16000:].decode(errors='replace'),end='',flush=True);offset=len(data)
    while True:
        with log.open('rb') as f:f.seek(offset);data=f.read();offset=f.tell()
        if data:print(data.decode(errors='replace'),end='',flush=True)
        if not busy(BASE/'session.lock'):
            with log.open('rb') as f:f.seek(offset);data=f.read()
            if data:print(data.decode(errors='replace'),end='',flush=True)
            return int(read(state_file).get('exit_code',1)) if state_file.exists() else 1
        time.sleep(2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status',action='store_true');parser.add_argument('--coordinator',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--worker',choices=('ffhq_0','ffhq_1','imagenet_0','imagenet_1'),help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:return worker(args.worker)
    return coordinator() if args.coordinator else client(args.status)


if __name__=='__main__':sys.exit(main())
