"""A88 per-dataset detached DPS queue; no automatic failed-case retries."""
import argparse
import os
os.environ['MKL_THREADING_LAYER']='GNU'
os.environ.pop('MKL_SERVICE_FORCE_INTEL',None)
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS))
driver=None
LABEL='DPS follow-up'
from utils import historical_ffhq_gain as files
from utils.a54_session import exclusive,busy



def state_dir(root):return root.parent/(root.name+'_session')
def invoke(*args):subprocess.run([sys.executable,'-u',*args],cwd=PROJECTS,check=True)
def worker_status(root,shard):
    f=root/'workers'/str(shard)/'status.json'
    return files.read(f)['status'] if f.exists() else 'not_started'


def worker_command(root,shard,uuid):
    return [sys.executable,'-u',str(Path(driver.__file__).relative_to(PROJECTS)),'--worker','--root',str(root),
            '--shard',str(shard),'--device-uuid',uuid]


def workers(root,state,assigned):
    queued=[s for s in (0,1) if worker_status(root,s)!='complete'];running={};failed=False
    files.require(all(worker_status(root,s)!='failed' for s in queued),'Prior worker failure retained')
    while queued or running:
        occupied={row['uuid'] for _,row,_ in running.values()}
        for shard in list(queued):
            row=assigned[str(shard)]
            if row['uuid'] in occupied:continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=row['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                     OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            log=(state/f'shard{shard}.log').open('ab',buffering=0)
            p=subprocess.Popen(worker_command(root,shard,row['uuid']),cwd=PROJECTS,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            running[shard]=(p,row,log);queued.remove(shard);occupied.add(row['uuid'])
            print(LABEL+' WORKER START:',shard,'GPU=',row['index'],flush=True)
        for shard,(p,row,log) in list(running.items()):
            if p.poll() is not None:
                log.close();del running[shard];failed|=p.returncode!=0
                print(LABEL+' WORKER END:',shard,'exit=',p.returncode,flush=True)
        print(LABEL+' QUEUE:',{s:worker_status(root,s) for s in (0,1)},flush=True)
        if running:time.sleep(30)
    if failed:raise RuntimeError(LABEL+' worker failed; logs retained, no automatic retry')
    return all(worker_status(root,s)=='complete' for s in (0,1))


def allocation(root,args):
    from utils.diag_paper_clock_alignment import gpu_inventory
    from utils import dps_missing_tasks as rules
    path=root/'allocation.json'
    rows=files.available_gpus(gpu_inventory(),args.gpus,args.visible)
    if path.exists():
        assigned=files.read(path)
        files.require(all(r['uuid'] in {v['uuid'] for v in rows} for r in assigned.values()),
                      'Pinned device outside requested allocation')
        return assigned
    admitted=sorted([r for r in rows if rules.admission(args.dataset,r)['admitted']],key=lambda r:-r['free_bytes'])
    if not admitted:
        print('A88 resource pause BEFORE model load:',args.dataset,flush=True);return None
    assigned={'0':admitted[0],'1':admitted[1] if len(admitted)>1 else admitted[0]}
    files.write(path,assigned);return assigned


def coordinator(args):
    root=Path(args.root).resolve();state=state_dir(root)
    with exclusive(state/'coordinator.lock'):
        record=dict(status='running',pid=os.getpid(),root=str(root));files.write(state/'session.json',record)
        try:
            invoke('-m','unittest','-v','utils.test_dps_missing_tasks',
                   'utils.test_dps_missing_tasks_runtime','utils.test_current_score_checkpoint_r2')
            invoke(str(Path(driver.__file__).relative_to(PROJECTS)),'--prepare','--root',str(root),
                   '--dataset',args.dataset,'--a75-root',args.a75_root,
                   '--ffhq-dir',args.ffhq_dir,'--imagenet-dir',args.imagenet_dir)
            for f in (root/'cases').glob('*/*/run.json'):
                files.require(files.read(f)['status']=='complete','Unfinished attempt retained: '+str(f))
            assigned=allocation(root,args)
            if assigned is None:record['status']='resource_limited'
            else:record['status']='complete' if workers(root,state,assigned) else 'resource_limited'
        except BaseException as error:
            record.update(status='failed',error=repr(error),traceback=traceback.format_exc())
            print(LABEL+' SESSION FAILED:',repr(error),flush=True)
        finally:
            if (root/'plan.json').exists():
                logs=root/'logs';logs.mkdir(exist_ok=True)
                for f in state.glob('shard*.log'):shutil.copyfile(f,logs/f.name)
                try:
                    invoke(str(Path(driver.__file__).relative_to(PROJECTS)),'--summarize','--root',str(root))
                    invoke(str(Path(driver.__file__).relative_to(PROJECTS)),'--archive','--root',str(root))
                except subprocess.CalledProcessError as error:
                    record['summary_or_upload_error']=repr(error)
                    print(LABEL+' local records retained; --archive-only retries upload.',flush=True)
            if record['status']=='resource_limited':print(LABEL+' RESOURCE PAUSE: reissue SAME command after allocated GPU memory is available.',flush=True)
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
        row=files.read(f);print(LABEL+' SESSION:',row,flush=True)
        if row['status']=='failed':raise SystemExit(1)


def main():
    global driver,LABEL
    from utils import diag_dps_missing_tasks
    driver=diag_dps_missing_tasks
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,choices=('ffhq','imagenet'))
    p.add_argument('--root');p.add_argument('--a75-root',default=driver.PARENT)
    p.add_argument('--ffhq-dir',default='/home/lzy/FFHQ/00000/00000')
    p.add_argument('--imagenet-dir',default='/home/lzy/imagenet/256x256');p.add_argument('--gpus',default='0,1')
    p.add_argument('--visible',default=os.environ.get('CUDA_VISIBLE_DEVICES'))
    for flag in ('coordinator','status','archive-only'):p.add_argument('--'+flag,action='store_true')
    a=p.parse_args();LABEL='A88 '+a.dataset
    if a.root is None:a.root=driver.ROOT.format(dataset=a.dataset)
    root=Path(a.root).resolve();state=state_dir(root);state.mkdir(parents=True,exist_ok=True)
    if a.coordinator:coordinator(a);return
    if a.status:
        print(dict(active=busy(state/'coordinator.lock'),session=files.read(state/'session.json')
                   if (state/'session.json').exists() else None),flush=True);return
    if a.archive_only:
        files.require(not busy(state/'coordinator.lock'),'Wait for running session')
        invoke(str(Path(driver.__file__).relative_to(PROJECTS)),'--summarize','--root',str(root))
        invoke(str(Path(driver.__file__).relative_to(PROJECTS)),'--archive','--root',str(root));return
    with exclusive(state/'client.lock'):
        process=None
        if not busy(state/'coordinator.lock'):
            for f in (root/'workers').glob('*/worker.lock'):files.require(not busy(f),'Live orphan worker; inspect --status')
            if (state/'session.json').exists():
                old=files.read(state/'session.json');files.require(old['status']!='failed','Prior failure retained; use --status')
                if old['status']=='complete':print(LABEL+' already complete:',root/'comparison.md',flush=True);return
            command=[sys.executable,'-u',str(Path(__file__)),'--coordinator','--dataset',a.dataset,'--root',str(root),
                     '--a75-root',a.a75_root,'--ffhq-dir',a.ffhq_dir,'--imagenet-dir',a.imagenet_dir,'--gpus',a.gpus]
            if a.visible is not None:command+=['--visible',a.visible]
            with (state/'coordinator.log').open('ab',buffering=0) as log:
                process=subprocess.Popen(command,cwd=PROJECTS,stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            print(LABEL+' detached queue started; following to final summary.',flush=True)
        else:print(LABEL+' active queue attached; no duplicate.',flush=True)
        follow(state,process)


if __name__=='__main__':main()
