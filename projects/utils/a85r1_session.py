"""Disconnect-safe A85 noisy paper-PiGDM four-task queue with retained failures."""
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
from utils import pigdm_noisy_r1 as rules
from utils import historical_ffhq_gain as files
from utils.a54_session import exclusive,busy
from utils.diag_paper_clock_alignment import gpu_inventory


def state_dir(root):return root.parent/(root.name+'_session')


def invoke(experiment,root,action,*extra):
    subprocess.run([sys.executable,'-u','utils/diag_pigdm_noisy_r1.py','--dataset',experiment,
        '--root',str(root),'--'+action,*extra],cwd=PROJECTS,check=True)


def status(root,shard,phase):
    f=root/'workers'/(str(shard)+'_'+phase)/'status.json'
    return files.read(f)['status'] if f.exists() else 'not_started'


def allocation(root,args):
    rows=files.available_gpus(gpu_inventory(),args.gpus,args.visible)
    saved=root/'allocation.json'
    if saved.exists():
        assigned=files.read(saved)
        files.require(all(v['uuid'] in {r['uuid'] for r in rows} for v in assigned.values()),'Pinned GPU outside current allocation')
        return assigned
    rows=sorted((r for r in rows if r['free_bytes']>=rules.min_free(args.dataset)),key=lambda r:-r['free_bytes'])
    if not rows:return None
    assigned={'0':rows[0],'1':rows[1] if len(rows)>1 else rows[0]};files.write(saved,assigned);return assigned


def worker_command(experiment,root,shard,uuid,phase):
    return [sys.executable,'-u','utils/diag_pigdm_noisy_r1.py','--dataset',experiment,
        '--worker','--root',str(root),'--shard',str(shard),'--device-uuid',uuid,'--phase',phase]


def workers(experiment,root,state,assigned,phase):
    queued=[s for s in (0,1) if status(root,s,phase)!='complete'];running={};failed=False
    files.require(all(status(root,s,phase)!='failed' for s in queued),'Prior worker failure retained; no automatic retry')
    while queued or running:
        occupied={row['uuid'] for _,row,_ in running.values()}
        for shard in list(queued):
            row=assigned[str(shard)]
            if row['uuid'] in occupied:continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=row['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                     OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',MKL_THREADING_LAYER='GNU')
            log=(state/f'shard{shard}_{phase}.log').open('ab',buffering=0)
            process=subprocess.Popen(worker_command(experiment,root,shard,row['uuid'],phase),cwd=PROJECTS,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            running[shard]=(process,row,log);queued.remove(shard);occupied.add(row['uuid'])
            print(experiment+' WORKER START:',shard,'GPU=',row['index'],flush=True)
        for shard,(process,row,log) in list(running.items()):
            if process.poll() is not None:
                log.close();del running[shard];failed|=process.returncode!=0
                print(experiment+' WORKER END:',shard,'exit=',process.returncode,flush=True)
        print(experiment+' QUEUE:',{s:status(root,s,phase) for s in (0,1)},flush=True)
        if running:time.sleep(30)
    if failed:raise RuntimeError(experiment+' worker failed; records retained, no automatic retry')
    return all(status(root,s,phase)=='complete' for s in (0,1))


def coordinator(args):
    root=Path(args.root);state=state_dir(root)
    with exclusive(state/'coordinator.lock'):
        record=dict(status='running',pid=os.getpid(),root=str(root),experiment=args.dataset)
        files.write(state/'session.json',record)
        try:
            subprocess.run([sys.executable,'-u','-m','unittest','-v','utils.test_pigdm_noisy_r1'],cwd=PROJECTS,check=True)
            invoke(args.dataset,root,'prepare','--a71-root',args.a71_root)
            assigned=allocation(root,args)
            if assigned is None:
                record['status']='resource_limited';print(args.dataset+' needs an allocated GPU with sufficient free VRAM; no NN started.',flush=True)
            else:
                # Real Torch numerical gates are REQUIRED, not skipped when Torch/CUDA absent.
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=assigned['0']['uuid'],PIGDM_DATASET=args.dataset)
                tests=['utils.test_pigdm_noisy_r1_runtime','utils.test_current_score_checkpoint_r2']
                subprocess.run([sys.executable,'-u','-m','unittest','-v',*tests],cwd=PROJECTS,env=env,check=True)
                if not workers(args.dataset,root,state,assigned,'baseline'):record['status']='resource_limited'
                else:
                    invoke(args.dataset,root,'qualify-baseline')
                    record['status']='complete' if workers(args.dataset,root,state,assigned,'comparison') else 'resource_limited'
        except BaseException as error:
            record.update(status='failed',error=repr(error),traceback=traceback.format_exc())
            print(args.dataset+' SESSION FAILED:',repr(error),flush=True)
        finally:
            if (root/'plan.json').exists():
                logs=root/'logs';logs.mkdir(exist_ok=True)
                for f in state.glob('*.log'):shutil.copyfile(f,logs/f.name)
                try:
                    invoke(args.dataset,root,'summarize');invoke(args.dataset,root,'archive')
                except subprocess.CalledProcessError as error:
                    record['summary_or_upload_error']=repr(error)
                    print(args.dataset+' local records retained; --archive-only retries upload.',flush=True)
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
    parser.add_argument('--dataset',choices=('ffhq','imagenet'),required=True);parser.add_argument('--root')
    parser.add_argument('--a71-root',default='/data0/lzy/ZAPS-results/diag_shrink_multitask/A71_20261029')
    parser.add_argument('--gpus',default='0,1');parser.add_argument('--visible',default=os.environ.get('CUDA_VISIBLE_DEVICES'))
    for flag in ('coordinator','status','archive-only'):parser.add_argument('--'+flag,action='store_true')
    args=parser.parse_args();args.root=args.root or rules.root_for(args.dataset)
    root=Path(args.root).resolve();state=state_dir(root);state.mkdir(parents=True,exist_ok=True)
    if args.coordinator:coordinator(args);return
    if args.status:
        print(dict(active=busy(state/'coordinator.lock'),session=files.read(state/'session.json') if (state/'session.json').exists() else None));return
    if args.archive_only:
        files.require(not busy(state/'coordinator.lock'),'Wait for active coordinator')
        invoke(args.dataset,root,'summarize');invoke(args.dataset,root,'archive');return
    with exclusive(state/'client.lock'):
        process=None
        if not busy(state/'coordinator.lock'):
            for f in (root/'workers').glob('*/worker.lock'):files.require(not busy(f),'Live orphan worker; inspect status')
            if (state/'session.json').exists():
                old=files.read(state/'session.json');files.require(old['status']!='failed','Prior failure retained; use --status')
                if old['status']=='complete':print('A85 '+args.dataset+' already complete:',root/'comparison.md');return
            command=[sys.executable,'-u',str(Path(__file__)),'--coordinator','--dataset',args.dataset,
                '--root',str(root),'--a71-root',args.a71_root,'--gpus',args.gpus]
            if args.visible is not None:command+=['--visible',args.visible]
            with (state/'coordinator.log').open('ab',buffering=0) as log:
                process=subprocess.Popen(command,cwd=PROJECTS,stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            print(args.dataset+' detached queue started; following to final summary.',flush=True)
        follow(state,process)


if __name__=='__main__':main()
