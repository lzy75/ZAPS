"""Detached serial-on-one-explicit-GPU A62; never interrupt existing processes."""
import argparse
from pathlib import Path
import os
import subprocess
import sys
import time
from unittest.mock import patch

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS))
from utils import diag_blur_fft_quality as run
from utils import a56m_no_interrupt_resume as resource
from utils.a54_session import busy,exclusive,read,write

BASE=run.ROOT.parent/'session_20261027'
TESTS=('utils.test_zero_boundary_fft','utils.test_blur_fft_quality','utils.test_norm_gain_control',
       'utils.test_current_score_checkpoint_r2')


def progress():
    rows=[]; unreadable=[]
    for f in run.ROOT.glob('*/control_*/run.json') if run.ROOT.exists() else []:
        try:rows.append(read(f))
        except (OSError,ValueError):unreadable.append(str(f))
    return dict(completed_cases=sum(r['status']=='complete' for r in rows),planned_cases=10,
        recorded_forwards=sum(r.get('nfe',0) for r in rows),forward_cap=1200,
        input_VJP_attempts=sum(r.get('backward_attempts',0) for r in rows),
        unreadable_attempts=unreadable,cost_is_lower_bound=bool(unreadable),
        active_fft=read(run.ROOT/'active_fft_solve.json') if (run.ROOT/'active_fft_solve.json').exists() else None)


def coordinator():
    BASE.mkdir(parents=True,exist_ok=True)
    with exclusive(BASE/'session.lock'):
        state=dict(pid=os.getpid(),status='running',gpu=os.environ['CUDA_VISIBLE_DEVICES'],no_process_interruption=True)
        write(BASE/'session.json',state)
        try:
            run.prepare()
            subprocess.run([sys.executable,'-m','unittest',*TESTS,'-v'],cwd=PROJECTS,check=True)
            rows=resource.legacy.run.queue.old.inventory()
            selected=resource.legacy.run.queue.allocation(rows,os.environ['CUDA_VISIBLE_DEVICES'])
            if len(selected)!=1 or selected[0]['free']<6144:
                state.update(status='waiting_for_resources',exit_code=0)
                print('A62 RESOURCE LIMITED: selected GPU needs >=6GiB free; no new case; same command explicitly resumes.',flush=True)
            else:
                write(BASE/'admission.json',dict(selected=selected,no_process_interruption=True))
                proc=subprocess.Popen([sys.executable,'-u',str(Path(run.__file__).resolve()),'--worker'],cwd=PROJECTS,
                    stdin=subprocess.DEVNULL,close_fds=True)
                last=time.monotonic()
                while proc.poll() is None:
                    time.sleep(1)
                    if time.monotonic()-last>=30:
                        print('A62 PROGRESS:',progress(),flush=True); last=time.monotonic()
                if proc.returncode==75:
                    state.update(status='waiting_for_resources',exit_code=0)
                elif proc.returncode:raise RuntimeError('A62 worker failed; partial costs retained, no automatic retry')
                else:
                    subprocess.run([sys.executable,'-u',str(Path(run.__file__).resolve())],cwd=PROJECTS,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
                    state.update(status='complete',exit_code=0)
        except BaseException as error:
            state.update(status='failed',exit_code=1,error=repr(error))
            if run.ROOT.exists():
                run.old.signed(run.ROOT/'failure_accounting.json',dict(status='failed',error=repr(error),
                    **progress(),limits='No final quality selection; attempted calls retained; abrupt interruption can undercount cost.'))
            print('A62 SESSION FAILED:',repr(error),'; no automatic failed-case retry.',flush=True)
        if state['status']=='waiting_for_resources':
            run.old.signed(run.ROOT/'resource_accounting.json',dict(status=state['status'],**progress(),
                limits='Completed cases retained; no attempt starts below memory gate; explicit resource resume allowed, not a scientific-failure retry.'))
        try:
            if (run.ROOT/'plan.json').exists():
                from utils import archive_refinement_confirmation as archive
                with patch.object(archive,'BRANCH',run.BRANCH):bundle,commit=archive.export_and_commit(dict(BLUR_FFT_QUALITY=run.ROOT))
                print('Archive:',bundle,'Results commit:',commit,'Branch:',run.BRANCH,flush=True)
                try:
                    subprocess.run(['git','push','origin',run.BRANCH],cwd=PROJECTS,env=dict(os.environ,GIT_TERMINAL_PROMPT='0'),timeout=45,check=True)
                    print('RESULTS_PUSH_PASS: A62 quality or failure accounting uploaded.',flush=True)
                except (OSError,subprocess.SubprocessError) as error:
                    print('RESULTS_PUSH_PENDING:',repr(error),'; no GPU rerun; git push origin '+run.BRANCH,flush=True)
        except BaseException as error:
            state.update(status='failed',exit_code=1,archive_error=repr(error)); print('A62 archive error:',repr(error),flush=True)
        finally:write(BASE/'session.json',state)
        return state['exit_code']


def client(gpu,status=False):
    BASE.mkdir(parents=True,exist_ok=True)
    if status:
        print(dict(active=busy(BASE/'session.lock'),session=read(BASE/'session.json') if (BASE/'session.json').exists() else None,
            progress=progress()),flush=True)
        if (BASE/'session.log').exists():print('\n'.join((BASE/'session.log').read_text(errors='replace').splitlines()[-18:]))
        return 0
    with exclusive(BASE/'client.lock'):
        if not busy(BASE/'session.lock'):
            old=read(BASE/'session.json') if (BASE/'session.json').exists() else {}
            if old.get('status') in ('failed','complete'):
                print('A62 prior status:',old['status'],'; inspect --status; no automatic retry.',flush=True)
                if old['status']=='complete':print('A62 Summary:',read(run.ROOT/'summary.json'),flush=True)
                return int(old.get('exit_code',1))
            with (BASE/'session.log').open('ab',buffering=0) as log:
                proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--coordinator'],cwd=PROJECTS,
                    env=resource.pinned_environment(gpu),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True,close_fds=True)
            for _ in range(100):
                if busy(BASE/'session.lock') or proc.poll() is not None:break
                time.sleep(.05)
    print('A62:10 old blur cases,4 arms,1200F/<=600 inputB; CPU solves+one GPU; same command reconnects.',flush=True)
    with patch.object(resource,'BASE',BASE):return resource.follow(BASE/'session.log')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu');parser.add_argument('--status',action='store_true');parser.add_argument('--coordinator',action='store_true')
    args=parser.parse_args()
    if not args.status and not args.coordinator and not args.gpu:parser.error('--gpu is required')
    sys.exit(coordinator() if args.coordinator else client(args.gpu,args.status))
