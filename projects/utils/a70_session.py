"""A70 direct worker CLI, one detached queue, explicit device UUID arguments."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS))
from utils import diag_noise_signal_shrink as driver
from utils import historical_ffhq_gain as files
from utils.a54_session import exclusive,busy
from utils.a65_session import allocation


def state_dir(root):return root.parent/(root.name+'_session')
def invoke(*args):subprocess.run([sys.executable,'-u',*args],cwd=PROJECTS,check=True)
def worker_status(root,shard,phase):
    f=root/'workers'/str(shard)/(phase+'.json')
    return files.read(f)['status'] if f.exists() else 'not_started'


def worker_command(root,shard,uuid,phase):
    return [sys.executable,'-u','utils/diag_noise_signal_shrink.py','--worker','--root',str(root),
            '--shard',str(shard),'--device-uuid',uuid,'--phase',phase]


def workers(root,state,assigned,phase):
    queued=[s for s in (0,1) if worker_status(root,s,phase)!='complete'];running={};failed=False
    files.require(all(worker_status(root,s,phase)!='failed' for s in queued),'Prior worker failure retained')
    while queued or running:
        occupied={row['uuid'] for _,row,_ in running.values()}
        for shard in list(queued):
            row=assigned[str(shard)]
            if row['uuid'] in occupied:continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=row['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                     OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            log=(state/f'shard{shard}_{phase}.log').open('ab',buffering=0)
            p=subprocess.Popen(worker_command(root,shard,row['uuid'],phase),cwd=PROJECTS,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            running[shard]=(p,row,log);queued.remove(shard);occupied.add(row['uuid'])
            print('A70 WORKER START:',shard,'GPU=',row['index'],flush=True)
        for shard,(p,row,log) in list(running.items()):
            if p.poll() is not None:
                log.close();del running[shard];failed|=p.returncode!=0
                print('A70 WORKER END:',shard,'exit=',p.returncode,flush=True)
        print('A70 QUEUE:',{s:worker_status(root,s,phase) for s in (0,1)},flush=True)
        if running:time.sleep(30)
    if failed:raise RuntimeError('A70 worker failed; logs retained, no automatic retry')
    return all(worker_status(root,s,phase)=='complete' for s in (0,1))


def coordinator(args):
    root=Path(args.root).resolve();state=state_dir(root)
    with exclusive(state/'coordinator.lock'):
        record=dict(status='running',pid=os.getpid(),root=str(root));files.write(state/'session.json',record)
        try:
            invoke('-m','unittest','-v','utils.test_a70_cli','utils.test_noise_signal_shrink',
                   'utils.test_noise_signal_shrink_runtime')
            invoke('utils/diag_noise_signal_shrink.py','--prepare','--root',str(root),'--ffhq-dir',args.ffhq_dir,'--a69-root',args.a69_root)
            for f in (root/'cases').glob('*/*/run.json'):
                files.require(files.read(f)['status']=='complete','Unfinished attempt retained: '+str(f))
            assigned=allocation(root,args.gpus,args.visible)
            if assigned is None:record['status']='resource_limited'
            else:
                stages=(('calibration','--freeze-constants'),('feedback','--freeze-mismatches'),('controls',None))
                for phase,freeze in stages:
                    if not workers(root,state,assigned,phase):record['status']='resource_limited';break
                    if freeze:invoke('utils/diag_noise_signal_shrink.py',freeze,'--root',str(root))
                else:record['status']='complete'
        except BaseException as error:
            record.update(status='failed',error=repr(error),traceback=traceback.format_exc())
            print('A70 SESSION FAILED:',repr(error),flush=True)
        finally:
            if (root/'plan.json').exists():
                logs=root/'logs';logs.mkdir(exist_ok=True)
                for f in state.glob('shard*.log'):shutil.copyfile(f,logs/f.name)
                try:
                    invoke('utils/diag_noise_signal_shrink.py','--summarize','--root',str(root))
                    invoke('utils/diag_noise_signal_shrink.py','--archive','--root',str(root))
                except subprocess.CalledProcessError as error:
                    record['summary_or_upload_error']=repr(error)
                    print('A70 local records retained; --archive-only retries upload.',flush=True)
            record['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
            files.write(state/'session.json',record)


def follow(state,process):
    offsets={}
    while True:
        for f in [state/'coordinator.log',*sorted(state.glob('shard*.log'))]:
            if f.exists():
                with f.open(errors='replace') as stream:
                    stream.seek(offsets.get(str(f),0));value=stream.read();offsets[str(f)]=stream.tell()
                if value:print(value,end='',flush=True)
        if not busy(state/'coordinator.lock') and (process is None or process.poll() is not None):break
        time.sleep(2)
    f=state/'session.json'
    if f.exists():
        row=files.read(f);print('A70 SESSION:',row,flush=True)
        if row['status']=='failed':raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=driver.ROOT)
    p.add_argument('--ffhq-dir',default='/home/lzy/FFHQ/00000/00000');p.add_argument('--a69-root',default=driver.PARENT);p.add_argument('--gpus',default='0,1')
    p.add_argument('--visible',default=os.environ.get('CUDA_VISIBLE_DEVICES'))
    for flag in ('coordinator','status','archive-only'):p.add_argument('--'+flag,action='store_true')
    a=p.parse_args();root=Path(a.root).resolve();state=state_dir(root);state.mkdir(parents=True,exist_ok=True)
    if a.coordinator:coordinator(a);return
    if a.status:
        print(dict(active=busy(state/'coordinator.lock'),session=files.read(state/'session.json')
                   if (state/'session.json').exists() else None),flush=True);return
    if a.archive_only:
        files.require(not busy(state/'coordinator.lock'),'Wait for running session')
        invoke('utils/diag_noise_signal_shrink.py','--archive','--root',str(root));return
    with exclusive(state/'client.lock'):
        process=None
        if not busy(state/'coordinator.lock'):
            for f in (root/'workers').glob('*/worker.lock'):files.require(not busy(f),'Live orphan worker; inspect --status')
            if (state/'session.json').exists():
                old=files.read(state/'session.json');files.require(old['status']!='failed','Prior failure retained; use --status')
                if old['status']=='complete':print('A70 already complete:',root/'comparison.md',flush=True);return
            command=[sys.executable,'-u',str(Path(__file__)),'--coordinator','--root',str(root),
                     '--ffhq-dir',a.ffhq_dir,'--gpus',a.gpus,'--a69-root',a.a69_root]
            if a.visible is not None:command+=['--visible',a.visible]
            with (state/'coordinator.log').open('ab',buffering=0) as log:
                process=subprocess.Popen(command,cwd=PROJECTS,stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            print('A70 detached queue started; following to final summary.',flush=True)
        else:print('A70 active queue attached; no duplicate.',flush=True)
        follow(state,process)


if __name__=='__main__':main()
