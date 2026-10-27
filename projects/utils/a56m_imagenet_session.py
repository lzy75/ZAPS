"""Detached bounded ImageNet memory qualification and native frozen-case continuation."""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS))
from utils import a56m_imagenet_runtime as run
from utils.a54_session import busy,exclusive,read,write

BASE=run.OUT.parent/'A56M1_session_20261027'
TESTS=('utils.test_a56m_imagenet_runtime','utils.test_a56m_torch_plumbing','utils.test_current_direction_control',
       'utils.test_linear_adjoint_autograd_guard','utils.test_current_score_checkpoint_r2')


def choose_gpu(rows,visible=None,first=None,second=None):
    allowed=run.queue.allocation(rows,visible,first,second)
    good=[r for r in allowed if r['free']>=run.PROBE_MIN_FREE_MIB]
    return max(good,key=lambda r:r['free']) if good else None


def stable_snapshot(p):
    snap=run.OUT/'source_snapshot';snap.mkdir(exist_ok=True)
    for path in run.own_sources():shutil.copyfile(path,snap/Path(path).name)
    for name in ('plan.json','plan.sha256','early_FFHQ_evaluation.json'):
        if (run.ROOT/name).exists():shutil.copyfile(run.ROOT/name,snap/('A56_'+name))
    for c in run.selected_cases(p):
        for path in (run.ROOT/c['key']).glob('*/attempt_*/run.json'):
            shutil.copyfile(path,snap/(c['key']+'_'+path.parent.parent.name+'_run.json'))


def coordinator():
    BASE.mkdir(parents=True,exist_ok=True);run.OUT.mkdir(parents=True,exist_ok=True)
    with exclusive(BASE/'session.lock'):
        state=dict(pid=os.getpid(),status='preparing',started_utc=run.queue.old.stamp())
        write(BASE/'session.json',state)
        try:
            d=run.frozen.original;p=d.check(run.ROOT)
            for c in p['cases']:
                if c['dataset']=='ffhq' and d.finished(run.ROOT,p,c) is None:raise ValueError('Need all12 FFHQ cases complete')
            run.stop_idle_queue(p)
            common=Path(run.frozen.OLD_ROOT).parent
            with ExitStack() as locks:
                locks.enter_context(exclusive(common/'launcher.lock'))
                for folder in (run.queue.BASE,run.queue.old.BASE,run.frozen.BASE,common,common/'operator_repair_20261027'):
                    if folder.exists():
                        for name in ('client.lock','session.lock'):locks.enter_context(exclusive(folder/name))
                for shard in (0,1):locks.enter_context(exclusive(run.ROOT/f'worker_shard{shard}.lock'))
                run.prepare(p)
                # Refuse any old scientific failure; completed original records may be reused.
                for c in run.selected_cases(p):
                    for kind,validator in (('BASELINE',lambda:d.baseline_done(run.ROOT,p,c,'BASELINE')),('SAMPLING',lambda:d.finished(run.ROOT,p,c))):
                        if list((run.ROOT/c['key']/kind).glob('attempt_*')) and validator() is None:raise ValueError('Original failed/partial attempt retained; no retry: '+c['key']+'/'+kind)
                env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                subprocess.run([sys.executable,'-m','unittest',*TESTS,'-v'],cwd=PROJECTS,env=env,check=True)
                rows=run.queue.old.inventory()
                gpu=choose_gpu(rows,os.environ.get('CUDA_VISIBLE_DEVICES'),os.environ.get('ZAPS_A56_GPU0'),os.environ.get('ZAPS_A56_GPU1'))
                write(run.OUT/'admission.json',dict(utc=run.queue.old.stamp(),inventory=rows,selected=gpu,minimum_free_MiB=run.PROBE_MIN_FREE_MIB))
                if gpu is None:
                    value=dict(experiment='A56M1_imagenet_native_qualification',status='resource_limited',classification='NO_ALLOCATED_GPU_WITH8GiB_FOR_ONE_BOUNDED_PROBE',new_profile_nfe=0,new_profile_input_VJP=0,original_ImageNet_recorded_cost=run.cost_snapshot(p))
                    run.record(run.OUT/'summary.json',value);print('A56M1 Summary:',value,flush=True)
                else:
                    state.update(status='running_profile_then_frozen_cases',gpu=gpu,maximum_new_forwards=3924);write(BASE/'session.json',state)
                    print('A56M1 admitted GPU',gpu['token'],'freeMiB=',gpu['free'],'; profile<=72F/32inputB, then original8 cases if qualified.',flush=True)
                    proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker'],cwd=PROJECTS,
                        env=dict(env,CUDA_VISIBLE_DEVICES=gpu['token']),stdin=subprocess.DEVNULL,close_fds=True)
                    last_report=time.monotonic()
                    while proc.poll() is None:
                        time.sleep(1)
                        if time.monotonic()-last_report>=30:
                            snapshot=progress();print('A56M1 PROGRESS:',dict(profile=snapshot['profile'],source_A56=snapshot['source_A56']),flush=True)
                            last_report=time.monotonic()
                    if proc.returncode:raise RuntimeError('Worker exited '+str(proc.returncode)+'; partial records retained')
                value=run.frozen.original.source.load_signed(run.OUT/'summary.json')
                stable_snapshot(p)
                from unittest.mock import patch
                from utils import archive_refinement_confirmation as archive
                with patch.object(archive,'BRANCH',run.BRANCH):bundle,commit=archive.export_and_commit(dict(RESOURCE_QUALIFICATION=run.OUT))
                print('Archive:',bundle,'Results commit:',commit,'Branch:',run.BRANCH,flush=True)
                refs=[run.BRANCH]
                if value.get('completed_ImageNet_cases')==8:refs.append(run.frozen.BRANCH)
                try:
                    subprocess.run(['git','push','origin',*refs],cwd=PROJECTS,env=dict(env,GIT_TERMINAL_PROMPT='0'),timeout=45,check=True)
                    print('RESULTS_PUSH_PASS: A56M1 qualification and available ImageNet results uploaded.',flush=True)
                except (subprocess.SubprocessError,OSError) as error:
                    print('RESULTS_PUSH_PENDING:',repr(error),'; no GPU rerun; git push origin '+' '.join(refs),flush=True)
                status=value.get('status')
                state.update(status='waiting_for_resources' if status=='resource_limited' else ('failed' if status=='failed' else 'complete'),
                    exit_code=1 if status=='failed' else 0,result_classification=value['classification'])
        except BaseException as error:
            state.update(status='failed',exit_code=1,error=repr(error))
            print('A56M1 SESSION FAILED:',repr(error),'; records retained, no automatic failed-attempt retry.',flush=True)
        finally:
            state['finished_utc']=run.queue.old.stamp();write(BASE/'session.json',state)
        return state['exit_code']


def progress():
    value=read(run.OUT/'summary.json') if (run.OUT/'summary.json').exists() else None
    prof=read(run.OUT/'profile.json') if (run.OUT/'profile.json').exists() else None
    return dict(summary=value,profile={k:prof.get(k) for k in ('status','phase','nfe','input_backward_attempts','gates_passed','error')} if prof else None,
                source_A56=run.queue.old.progress())


def client(status_only=False):
    BASE.mkdir(parents=True,exist_ok=True)
    if not busy(BASE/'session.lock') and run.queue.BASE.exists() and busy(run.queue.BASE/'session.lock'):
        prior=read(run.queue.BASE/'session.json')
        if prior.get('running_groups') or prior.get('status')=='archiving':
            print('Original ImageNet queue already executing/archiving; following it without interruption.',flush=True)
            return run.queue.client(status_only)
    log,state_file=BASE/'session.log',BASE/'session.json'
    if status_only:
        print(dict(active=busy(BASE/'session.lock'),session=read(state_file) if state_file.exists() else None,progress=progress()))
        if log.exists():print('\n'.join(log.read_text(errors='replace').splitlines()[-25:]))
        return 0
    with exclusive(BASE/'client.lock'):
        active=busy(BASE/'session.lock');state=read(state_file) if state_file.exists() else {}
        if not active and state.get('status')=='complete':return client(True)
        if not active and state.get('status')=='failed':
            print('A56M1 previous attempt failed; use --status. No automatic repeated probe or failed-case retry.');return 1
        if not active:
            with log.open('ab',buffering=0) as output:
                proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--coordinator'],cwd=PROJECTS,
                    stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            for _ in range(100):
                if busy(BASE/'session.lock') or proc.poll() is not None:break
                time.sleep(.05)
    print('A56M1 detached; same command reconnects. Finite resource probe; no indefinite memory wait.',flush=True)
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
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:run.worker();return 0
    return coordinator() if args.coordinator else client(args.status)


if __name__=='__main__':sys.exit(main())
