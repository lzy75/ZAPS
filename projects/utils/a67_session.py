"""A67 detached queue with foreground log following; no automatic failed replay."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS))
from utils import historical_ffhq_gain as files
from utils import diag_noise_feasible_zaps as driver
from utils.a54_session import exclusive,busy
from utils.a65_session import allocation


def invoke(*arguments):subprocess.run([sys.executable,'-u',*arguments],cwd=PROJECTS,check=True)
def state_dir(root):return root.parent/(root.name+'_session')
def status(root,shard,phase):
    f=root/'workers'/str(shard)/(phase+'.json')
    return files.read(f)['status'] if f.exists() else 'not_started'


def phase(root,state,assigned,name):
    queued=[i for i in (0,1) if status(root,i,name)!='complete'];running={};failed=False
    files.require(all(status(root,i,name)!='failed' for i in queued),'Failed worker preserved; no automatic retry')
    while queued or running:
        occupied={row['uuid'] for _,row,_ in running.values()}
        for shard in list(queued):
            row=assigned[str(shard)]
            if row['uuid'] in occupied:continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=row['uuid'],OMP_NUM_THREADS='2',
                     MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            log=(state/f'shard{shard}_{name}.log').open('ab',buffering=0)
            proc=subprocess.Popen([sys.executable,'-u','utils/diag_noise_feasible_zaps.py','--worker',
                '--root',str(root),'--shard',str(shard),'--phase',name,'--device-uuid',row['uuid']],
                cwd=PROJECTS,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            running[shard]=(proc,row,log);queued.remove(shard);occupied.add(row['uuid'])
            print('A67 WORKER START:',name,shard,'GPU=',row['index'],flush=True)
        for shard,(proc,row,log) in list(running.items()):
            if proc.poll() is not None:
                log.close();del running[shard];failed|=proc.returncode!=0
                print('A67 WORKER END:',name,shard,'exit=',proc.returncode,flush=True)
        print('A67 QUEUE:',name,{i:status(root,i,name) for i in (0,1)},flush=True)
        if running:time.sleep(30)
    if failed:raise RuntimeError('A67 worker failed; original records retained')
    return all(status(root,i,name)=='complete' for i in (0,1))


def coordinator(args):
    root=Path(args.root).resolve();state=state_dir(root)
    with exclusive(state/'coordinator.lock'):
        record=dict(status='running',pid=os.getpid(),root=str(root));files.write(state/'session.json',record)
        try:
            invoke('-m','unittest','-v','utils.test_noise_feasible_multiplier','utils.test_noise_feasible_runtime')
            invoke('utils/diag_noise_feasible_zaps.py','--prepare','--root',str(root),'--ffhq-dir',args.ffhq_dir)
            for f in (root/'cases').glob('*/*/run.json'):
                files.require(files.read(f)['status']=='complete','Prior model attempt retained: '+str(f))
            assigned=allocation(root,args.gpus,args.visible)
            if assigned is None:record['status']='resource_limited'
            elif not phase(root,state,assigned,'feedback'):record['status']='resource_limited'
            elif not phase(root,state,assigned,'time'):record['status']='resource_limited'
            else:record['status']='complete'
        except BaseException as error:
            record.update(status='failed',error=repr(error),traceback=traceback.format_exc())
            print('A67 SESSION FAILED:',repr(error),flush=True)
        finally:
            if (root/'plan.json').exists():
                logs=root/'logs';logs.mkdir(exist_ok=True)
                for f in state.glob('shard*.log'):shutil.copyfile(f,logs/f.name)
                try:
                    invoke('utils/diag_noise_feasible_zaps.py','--summarize','--root',str(root))
                    invoke('utils/diag_noise_feasible_zaps.py','--archive','--root',str(root))
                except subprocess.CalledProcessError as error:
                    record['summary_or_upload_error']=repr(error)
                    print('A67 local records retained; --archive-only retries upload without GPU.',flush=True)
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
        r=files.read(f);print('A67 SESSION:',r,flush=True)
        if r['status']=='failed':raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=driver.ROOT)
    p.add_argument('--ffhq-dir',default='/home/lzy/FFHQ/00000/00000');p.add_argument('--gpus',default='0,1')
    p.add_argument('--visible',default=os.environ.get('CUDA_VISIBLE_DEVICES'))
    p.add_argument('--coordinator',action='store_true');p.add_argument('--status',action='store_true')
    p.add_argument('--archive-only',action='store_true');a=p.parse_args()
    root=Path(a.root).resolve();state=state_dir(root);state.mkdir(parents=True,exist_ok=True)
    if a.coordinator:coordinator(a);return
    if a.status:
        print(dict(active=busy(state/'coordinator.lock'),session=files.read(state/'session.json')
                   if (state/'session.json').exists() else None),flush=True);return
    if a.archive_only:
        files.require(not busy(state/'coordinator.lock'),'Wait for active session')
        invoke('utils/diag_noise_feasible_zaps.py','--archive','--root',str(root));return
    with exclusive(state/'client.lock'):
        proc=None
        if not busy(state/'coordinator.lock'):
            for f in (root/'workers').glob('*/worker.lock'):files.require(not busy(f),'Live orphan worker; inspect --status')
            if (state/'session.json').exists():
                old=files.read(state/'session.json')
                files.require(old['status']!='failed','Failed attempt preserved. Use --status; no automatic retry')
                if old['status']=='complete':print('A67 already complete:',root/'comparison.md',flush=True);return
            command=[sys.executable,'-u',str(Path(__file__)),'--coordinator','--root',str(root),
                     '--ffhq-dir',a.ffhq_dir,'--gpus',a.gpus]
            if a.visible is not None:command+=['--visible',a.visible]
            with (state/'coordinator.log').open('ab',buffering=0) as log:
                proc=subprocess.Popen(command,cwd=PROJECTS,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                      start_new_session=True,close_fds=True)
            print('A67 detached queue started; following until summary. SSH disconnect does not kill workers.',flush=True)
        else:print('A67 active queue attached; no duplicate.',flush=True)
        follow(state,proc)


if __name__=='__main__':main()
