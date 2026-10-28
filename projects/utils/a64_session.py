"""Detached A64 coordinator: fixed allocations, baseline gates, no failed-attempt retries."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS))
from utils import paper_clock_alignment as rules
from utils.a54_session import exclusive,busy


def state_directory(root): return root.parent/(root.name+'_session')


def run(command,env=None):
    subprocess.run([sys.executable,'-u',*command],cwd=PROJECTS,env=env,check=True)


def dataset_status(root,dataset):
    file=root/'workers'/dataset/'status.json'
    return rules.read(file).get('status') if file.exists() else 'not_started'


def choose_assignment(root,rows,pending):
    """Preserve UUID on resume. Reserve larger available GPU for ImageNet first."""
    chosen={}; used=set()
    for dataset in sorted(pending,key=lambda d:-rules.MIN_FREE[d]):
        identity=root/'workers'/dataset/'device.json'
        old=rules.read(identity)['uuid'] if identity.exists() else None
        eligible=[r for r in rows if r['free_bytes']>=rules.MIN_FREE[dataset] and (old is None or r['uuid']==old)]
        eligible.sort(key=lambda r:(r['uuid'] in used,-r['free_bytes']))
        if eligible:
            chosen[dataset]=eligible[0]; used.add(eligible[0]['uuid'])
    return chosen


def coordinator(args):
    from utils.diag_paper_clock_alignment import gpu_inventory
    root=Path(args.root).resolve(); state=state_directory(root)
    with exclusive(state/'coordinator.lock'):
        record=dict(status='running',pid=os.getpid(),root=str(root),requested=args.gpus,
                    inherited_allocation=args.visible,start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        rules.write(state/'session.json',record)
        try:
            # Torch is a required import in the numerical test module: missing Torch is never skipped.
            run(['-m','unittest','-v','utils.test_paper_clock_alignment',
                 'utils.test_paper_clock_runtime','utils.test_current_score_checkpoint_r2'])
            run(['utils/diag_paper_clock_alignment.py','--root',str(root),'--prepare',
                 '--ffhq-dir',args.ffhq_dir,'--imagenet-dir',args.imagenet_dir])
            for file in (root/'cases').glob('*/*/repeat_*/run.json'):
                rules.require(rules.read(file)['status']=='complete',
                              'Prior started/failed attempt retained, no automatic retry: '+str(file))
            pending=[d for d in rules.DATASETS if dataset_status(root,d) not in ('complete','baseline_review_required')]
            rules.require(not any(dataset_status(root,d)=='failed' for d in pending),
                          'Failed worker/attempt preserved; inspect logs, no automatic rerun')
            rows=rules.available_gpus(gpu_inventory(),args.gpus,args.visible)
            assigned=choose_assignment(root,rows,pending)
            record['assignments']=assigned;rules.write(state/'session.json',record)
            # Two distinct UUIDs run concurrently; same UUID queues datasets serially.
            queued=list(assigned); running={}; workers_failed=[]
            while queued or running:
                occupied={row['uuid'] for _,row,_ in running.values()}
                for dataset in list(queued):
                    row=assigned[dataset]
                    if row['uuid'] in occupied: continue
                    environment=dict(os.environ,CUDA_VISIBLE_DEVICES=row['uuid'],OMP_NUM_THREADS='2',
                                     MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
                    log=(state/(dataset+'.log')).open('ab',buffering=0)
                    process=subprocess.Popen([sys.executable,'-u','utils/diag_paper_clock_alignment.py',
                        '--root',str(root),'--worker','--dataset',dataset,'--device-uuid',row['uuid']],
                        cwd=PROJECTS,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    running[dataset]=(process,row,log);queued.remove(dataset);occupied.add(row['uuid'])
                    print('A64 WORKER START:',dataset,'GPU',row['index'],row['uuid'],flush=True)
                for dataset,(process,row,log) in list(running.items()):
                    if process.poll() is not None:
                        log.close(); del running[dataset]
                        if process.returncode: workers_failed.append(dataset)
                        print('A64 WORKER END:',dataset,'exit=',process.returncode,
                              'status=',dataset_status(root,dataset),flush=True)
                progress={}
                for dataset in rules.DATASETS:
                    file=root/(dataset+'_progress.json')
                    progress[dataset]=rules.read(file) if file.exists() else dict(status=dataset_status(root,dataset))
                print('A64 QUEUE PROGRESS:',progress,flush=True)
                rules.write(state/'progress.json',progress)
                if running: time.sleep(30)
            # Logs copied after worker handles are closed; archive sources will be stable.
            logs=root/'logs';logs.mkdir(exist_ok=True)
            for dataset in rules.DATASETS:
                file=state/(dataset+'.log')
                if file.exists(): shutil.copyfile(file,logs/(dataset+'.log'))
            run(['utils/diag_paper_clock_alignment.py','--root',str(root),'--summarize'])
            unfinished=[d for d in rules.DATASETS if dataset_status(root,d) not in ('complete','baseline_review_required')]
            record['dataset_status']={d:dataset_status(root,d) for d in rules.DATASETS}
            record['status']='failed' if workers_failed else 'resource_limited' if unfinished else 'complete'
            rules.write(state/'session.json',record)
            # Incomplete results are still archived, preserving all costs and baseline readouts.
            try:
                run(['utils/diag_paper_clock_alignment.py','--root',str(root),'--archive'])
            except subprocess.CalledProcessError:
                record['upload']='failed';rules.write(state/'session.json',record)
                print('A64 upload failed; local records retained. Use --archive-only; do not rerun GPU.',flush=True)
            if workers_failed: raise RuntimeError('Worker failed; see saved dataset logs, no automatic retry')
            if unfinished:
                print('A64 RESOURCE PAUSE:',unfinished,
                      'No processes terminated. Reissue SAME command after allocated memory becomes available.',flush=True)
        except BaseException as error:
            record.update(status='failed',error=repr(error),traceback=traceback.format_exc())
            rules.write(state/'session.json',record);raise
        finally:
            record['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
            rules.write(state/'session.json',record)


def follow(root,state,process=None):
    positions={}
    while True:
        for name in ('coordinator.log','ffhq.log','imagenet.log'):
            file=state/name
            if file.exists():
                with file.open('r',errors='replace') as stream:
                    stream.seek(positions.get(name,0)); data=stream.read();positions[name]=stream.tell()
                if data: print(data,end='',flush=True)
        active=busy(state/'coordinator.lock')
        if not active and (process is None or process.poll() is not None): break
        time.sleep(2)
    file=state/'session.json'
    if file.exists():
        r=rules.read(file);print('A64 SESSION:',r,flush=True)
        if r['status']=='failed': raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=rules.ROOT);parser.add_argument('--gpus',default='0,1')
    parser.add_argument('--ffhq-dir',default='/home/lzy/FFHQ/00000/00000')
    parser.add_argument('--imagenet-dir',default='/home/lzy/imagenet/256x256')
    parser.add_argument('--visible',default=os.environ.get('CUDA_VISIBLE_DEVICES'))
    parser.add_argument('--coordinator',action='store_true');parser.add_argument('--status',action='store_true')
    parser.add_argument('--archive-only',action='store_true')
    args=parser.parse_args();root=Path(args.root).resolve();state=state_directory(root)
    state.mkdir(parents=True,exist_ok=True)
    if args.coordinator: coordinator(args);return
    if args.status:
        print(dict(active=busy(state/'coordinator.lock'),session=rules.read(state/'session.json')
                   if (state/'session.json').exists() else None),flush=True);return
    if args.archive_only:
        rules.require(not busy(state/'coordinator.lock'),'Active session; wait before archive')
        run(['utils/diag_paper_clock_alignment.py','--root',str(root),'--archive']);return
    # Client lock prevents simultaneous detached spawn; never trusts a stale PID alone.
    with exclusive(state/'client.lock'):
        process=None
        if busy(state/'coordinator.lock'):
            print('A64 existing coordinator attached; no duplicate run.',flush=True)
        else:
            for dataset in rules.DATASETS:
                lock=root/'workers'/dataset/'worker.lock'
                rules.require(not lock.exists() or not busy(lock),
                              'Live worker remains after coordinator loss; use --status, do not start a duplicate')
            if (state/'session.json').exists():
                prior=rules.read(state/'session.json')
                rules.require(prior['status']!='failed','Previous session failed. Use --status; no automatic failed retry')
                if prior['status']=='complete':
                    print('A64 already complete. Summary:',root/'summary.json',flush=True);return
            command=[sys.executable,'-u',str(Path(__file__)),'--coordinator','--root',str(root),
                     '--gpus',args.gpus,'--ffhq-dir',args.ffhq_dir,'--imagenet-dir',args.imagenet_dir]
            if args.visible is not None:command+=['--visible',args.visible]
            with (state/'coordinator.log').open('ab',buffering=0) as log:
                process=subprocess.Popen(command,cwd=PROJECTS,start_new_session=True,stdin=subprocess.DEVNULL,
                                         stdout=log,stderr=subprocess.STDOUT,close_fds=True)
            print('A64 detached session started; this client follows through final summary. '
                  'Disconnecting leaves the experiment running.',flush=True)
        follow(root,state,process)


if __name__=='__main__':main()
