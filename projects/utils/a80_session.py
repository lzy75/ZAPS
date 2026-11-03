"""A80 direct worker CLI, one detached queue, explicit device UUID arguments."""
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
from utils.diag_paper_clock_alignment import gpu_inventory
from utils.diffpir_multiplier_transfer import MIN_FREE

def allocation(root,requested,visible):
    rows=files.available_gpus(gpu_inventory(),requested,visible)
    file=root/'allocation.json'
    if file.exists():
        assigned=files.read(file)
        files.require(all(v['uuid'] in {r['uuid'] for r in rows} for v in assigned.values()),'Pinned GPU outside current allocation')
        return assigned
    eligible=sorted((r for r in rows if r['free_bytes']>=MIN_FREE),key=lambda r:-r['free_bytes'])
    if not eligible:return None
    assigned={'1':eligible[0],'0':eligible[1] if len(eligible)>1 else eligible[0]}
    files.write(file,assigned);return assigned


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


def coordinator(args):
    root=Path(args.root).resolve();state=state_dir(root)
    with exclusive(state/'coordinator.lock'):
        record=dict(status='running',pid=os.getpid(),root=str(root));files.write(state/'session.json',record)
        try:
            invoke('-m','unittest','-v','utils.test_diffpir_host_aware',
                   'utils.test_diffpir_numeric_runtime','utils.test_diffpir_host_aware_runtime')
            invoke(str(Path(driver.__file__).relative_to(PROJECTS)),'--prepare','--root',str(root),'--a79r1-root',args.a79r1_root)
            for f in (root/'cases').glob('*/*/run.json'):
                files.require(files.read(f)['status']=='complete','Unfinished attempt retained: '+str(f))
            if args.experiment=='A80':
                from utils.diag_paper_clock_alignment import gpu_inventory
                from utils.diffpir_multiplier_transfer import MIN_FREE
                visible=files.available_gpus(gpu_inventory(),args.gpus,args.visible)
                if not any(r['free_bytes']>=MIN_FREE for r in visible):
                    print('A80 resource pause: no allocated GPU has6GiB free; no model load.',flush=True)
                    assigned=None
                else:assigned=allocation(root,args.gpus,args.visible)
            else:assigned=allocation(root,args.gpus,args.visible)
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
    selected=argparse.ArgumentParser(add_help=False);selected.add_argument('--experiment',required=True,choices=('A80',))
    pick,_=selected.parse_known_args();LABEL=pick.experiment
    import importlib
    driver=importlib.import_module('utils.diag_diffpir_host_aware')
    p=argparse.ArgumentParser(description=__doc__,parents=[selected]);p.add_argument('--root',default=driver.ROOT)
    p.add_argument('--a79r1-root',default=driver.PARENT);p.add_argument('--gpus',default='0,1')
    p.add_argument('--visible',default=os.environ.get('CUDA_VISIBLE_DEVICES'))
    for flag in ('coordinator','status','archive-only'):p.add_argument('--'+flag,action='store_true')
    a=p.parse_args();root=Path(a.root).resolve();state=state_dir(root);state.mkdir(parents=True,exist_ok=True)
    if a.coordinator:coordinator(a);return
    if a.status:
        print(dict(active=busy(state/'coordinator.lock'),session=files.read(state/'session.json')
                   if (state/'session.json').exists() else None),flush=True);return
    if a.archive_only:
        files.require(not busy(state/'coordinator.lock'),'Wait for running session')
        invoke(str(Path(driver.__file__).relative_to(PROJECTS)),'--report','--root',str(root))
        invoke(str(Path(driver.__file__).relative_to(PROJECTS)),'--archive','--root',str(root));return
    with exclusive(state/'client.lock'):
        process=None
        if not busy(state/'coordinator.lock'):
            for f in (root/'workers').glob('*/worker.lock'):files.require(not busy(f),'Live orphan worker; inspect --status')
            if (state/'session.json').exists():
                old=files.read(state/'session.json');files.require(old['status']!='failed','Prior failure retained; use --status')
                if old['status']=='complete':print(LABEL+' already complete:',root/'comparison.md',flush=True);return
            command=[sys.executable,'-u',str(Path(__file__)),'--coordinator','--experiment',a.experiment,'--root',str(root),
                     '--a79r1-root',a.a79r1_root,'--gpus',a.gpus]
            if a.visible is not None:command+=['--visible',a.visible]
            with (state/'coordinator.log').open('ab',buffering=0) as log:
                process=subprocess.Popen(command,cwd=PROJECTS,stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            print(LABEL+' detached queue started; following to final summary.',flush=True)
        else:print(LABEL+' active queue attached; no duplicate.',flush=True)
        follow(state,process)


if __name__=='__main__':main()
