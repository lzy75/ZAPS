"""Disconnect-safe A82-A84 serial-host/two-task queue with retained failures."""
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
from utils import native_ray_transfer as rules
from utils import historical_ffhq_gain as files
from utils.a54_session import exclusive,busy
from utils.diag_paper_clock_alignment import gpu_inventory


def state_dir(root):return root.parent/(root.name+'_session')


def invoke(experiment,root,action,*extra):
    subprocess.run([sys.executable,'-u','utils/diag_native_ray_transfer.py','--experiment',experiment,
        '--root',str(root),'--'+action,*extra],cwd=PROJECTS,check=True)


def status(root,shard):
    f=root/'workers'/str(shard)/'status.json'
    return files.read(f)['status'] if f.exists() else 'not_started'


def allocation(root,args):
    rows=files.available_gpus(gpu_inventory(),args.gpus,args.visible)
    saved=root/'allocation.json'
    if saved.exists():
        assigned=files.read(saved)
        files.require(all(v['uuid'] in {r['uuid'] for r in rows} for v in assigned.values()),'Pinned GPU outside current allocation')
        return assigned
    rows=sorted((r for r in rows if r['free_bytes']>=rules.MIN_FREE),key=lambda r:-r['free_bytes'])
    if not rows:return None
    assigned={'0':rows[0],'1':rows[1] if len(rows)>1 else rows[0]};files.write(saved,assigned);return assigned


def worker_command(experiment,root,shard,uuid):
    return [sys.executable,'-u','utils/diag_native_ray_transfer.py','--experiment',experiment,
        '--worker','--root',str(root),'--shard',str(shard),'--device-uuid',uuid]


def workers(experiment,root,state,assigned):
    queued=[s for s in (0,1) if status(root,s)!='complete'];running={};failed=False
    files.require(all(status(root,s)!='failed' for s in queued),'Prior worker failure retained; no automatic retry')
    while queued or running:
        occupied={row['uuid'] for _,row,_ in running.values()}
        for shard in list(queued):
            row=assigned[str(shard)]
            if row['uuid'] in occupied:continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=row['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                     OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',MKL_THREADING_LAYER='GNU')
            log=(state/f'shard{shard}.log').open('ab',buffering=0)
            process=subprocess.Popen(worker_command(experiment,root,shard,row['uuid']),cwd=PROJECTS,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            running[shard]=(process,row,log);queued.remove(shard);occupied.add(row['uuid'])
            print(experiment+' WORKER START:',shard,'GPU=',row['index'],flush=True)
        for shard,(process,row,log) in list(running.items()):
            if process.poll() is not None:
                log.close();del running[shard];failed|=process.returncode!=0
                print(experiment+' WORKER END:',shard,'exit=',process.returncode,flush=True)
        print(experiment+' QUEUE:',{s:status(root,s) for s in (0,1)},flush=True)
        if running:time.sleep(30)
    if failed:raise RuntimeError(experiment+' worker failed; records retained, no automatic retry')
    return all(status(root,s)=='complete' for s in (0,1))


def coordinator(args):
    root=Path(args.root);state=state_dir(root)
    with exclusive(state/'coordinator.lock'):
        record=dict(status='running',pid=os.getpid(),root=str(root),experiment=args.experiment)
        files.write(state/'session.json',record)
        try:
            subprocess.run([sys.executable,'-u','-m','unittest','-v','utils.test_native_ray_transfer'],cwd=PROJECTS,check=True)
            invoke(args.experiment,root,'prepare','--a71-root',args.a71_root)
            assigned=allocation(root,args)
            if assigned is None:
                record['status']='resource_limited';print(args.experiment+' needs an allocated GPU>=6GiB free; no NN started.',flush=True)
            else:
                # Real Torch numerical gates are REQUIRED, not skipped when Torch/CUDA absent.
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=assigned['0']['uuid'],RAY_EXPERIMENT=args.experiment)
                tests=['utils.test_native_ray_transfer_runtime']
                if args.experiment=='A84':tests+=['utils.test_current_score_checkpoint_r2']
                subprocess.run([sys.executable,'-u','-m','unittest','-v',*tests],cwd=PROJECTS,env=env,check=True)
                record['status']='complete' if workers(args.experiment,root,state,assigned) else 'resource_limited'
        except BaseException as error:
            record.update(status='failed',error=repr(error),traceback=traceback.format_exc())
            print(args.experiment+' SESSION FAILED:',repr(error),flush=True)
        finally:
            if (root/'plan.json').exists():
                logs=root/'logs';logs.mkdir(exist_ok=True)
                for f in state.glob('*.log'):shutil.copyfile(f,logs/f.name)
                try:
                    invoke(args.experiment,root,'summarize');invoke(args.experiment,root,'archive')
                except subprocess.CalledProcessError as error:
                    record['summary_or_upload_error']=repr(error)
                    print(args.experiment+' local records retained; --archive-only retries upload.',flush=True)
            record['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());files.write(state/'session.json',record)


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
    if (state/'session.json').exists():
        row=files.read(state/'session.json');print(row,flush=True)
        if row['status']=='failed':raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=rules.HOSTS,required=True);parser.add_argument('--root')
    parser.add_argument('--a71-root',default='/data0/lzy/ZAPS-results/diag_shrink_multitask/A71_20261029')
    parser.add_argument('--gpus',default='0,1');parser.add_argument('--visible',default=os.environ.get('CUDA_VISIBLE_DEVICES'))
    for flag in ('coordinator','status','archive-only'):parser.add_argument('--'+flag,action='store_true')
    args=parser.parse_args();args.root=args.root or rules.root_for(args.experiment)
    root=Path(args.root).resolve();state=state_dir(root);state.mkdir(parents=True,exist_ok=True)
    if args.coordinator:coordinator(args);return
    if args.status:
        print(dict(active=busy(state/'coordinator.lock'),session=files.read(state/'session.json') if (state/'session.json').exists() else None));return
    if args.archive_only:
        files.require(not busy(state/'coordinator.lock'),'Wait for active coordinator')
        invoke(args.experiment,root,'summarize');invoke(args.experiment,root,'archive');return
    with exclusive(state/'client.lock'):
        process=None
        if not busy(state/'coordinator.lock'):
            for f in (root/'workers').glob('*/worker.lock'):files.require(not busy(f),'Live orphan worker; inspect status')
            if (state/'session.json').exists():
                old=files.read(state/'session.json');files.require(old['status']!='failed','Prior failure retained; use --status')
                if old['status']=='complete':print(args.experiment+' already complete:',root/'comparison.md');return
            command=[sys.executable,'-u',str(Path(__file__)),'--coordinator','--experiment',args.experiment,
                '--root',str(root),'--a71-root',args.a71_root,'--gpus',args.gpus]
            if args.visible is not None:command+=['--visible',args.visible]
            with (state/'coordinator.log').open('ab',buffering=0) as log:
                process=subprocess.Popen(command,cwd=PROJECTS,stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            print(args.experiment+' detached queue started; following to final summary.',flush=True)
        follow(state,process)


if __name__=='__main__':main()
