"""A66 disconnect-safe continuation; no historical-quality gate and no source mutation."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS))
from utils import current_strong_anchor as rules
from utils import historical_ffhq_gain as design
from utils.a54_session import exclusive,busy
from utils.a65_session import no_model_attempts
from utils.diag_paper_clock_alignment import gpu_inventory


def state_dir(root):return root.parent/(root.name+'_session')


def run(*args):subprocess.run([sys.executable,'-u',*args],cwd=PROJECTS,check=True)


def status(root,shard):
    path=root/'workers'/('shard'+str(shard))/'status.json'
    return rules.read(path)['status'] if path.exists() else 'not_started'


def phase(root,state,args):
    p=rules.read(root/'plan.json')
    rows=design.available_gpus(gpu_inventory(),args.gpus,args.visible)
    devices={r['uuid']:r for r in rows}
    if any(u not in devices for u in p['required_devices'].values()):
        print('A66 inherited timing GPU outside current allocation; no attempt started.',flush=True);return False
    queued=[i for i in (0,1) if status(root,i)!='complete'];running={};failed=False
    rules.require(not any(status(root,i)=='failed' for i in queued),'Failed worker retained; no automatic retry')
    while queued or running:
        occupied={u for _,u,_ in running.values()}
        for shard in list(queued):
            uuid=p['required_devices'][str(shard)]
            if uuid in occupied:continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                     OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            log=(state/f'shard{shard}.log').open('ab',buffering=0)
            process=subprocess.Popen([sys.executable,'-u','utils/diag_current_strong_anchor.py','--worker',
                '--root',str(root),'--shard',str(shard),'--device-uuid',uuid],cwd=PROJECTS,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            running[shard]=(process,uuid,log);queued.remove(shard);occupied.add(uuid)
            print('A66 WORKER START:',shard,'GPU',devices[uuid]['index'],uuid,flush=True)
        for shard,(process,uuid,log) in list(running.items()):
            if process.poll() is not None:
                log.close();del running[shard];failed|=process.returncode!=0
                print('A66 WORKER END:',shard,'exit',process.returncode,'status',status(root,shard),flush=True)
        print('A66 QUEUE STATUS:',{i:status(root,i) for i in (0,1)},flush=True)
        if running:time.sleep(30)
    if failed:raise RuntimeError('A66 worker failed; records retained, no automatic retry')
    return all(status(root,i)=='complete' for i in (0,1))


def coordinator(args):
    root=Path(args.root).resolve();state=state_dir(root)
    with exclusive(state/'coordinator.lock'):
        record=dict(status='running',pid=os.getpid(),root=str(root),parent=args.parent)
        rules.write(state/'session.json',record)
        try:
            run('-m','unittest','-v','utils.test_current_strong_anchor','utils.test_historical_ffhq_runtime',
                'utils.test_current_score_checkpoint_r2')
            run('utils/diag_current_strong_anchor.py','--prepare','--root',str(root),'--parent',args.parent)
            for path in (root/'cases').glob('*/*/repeat_*/run.json'):
                rules.require(rules.read(path)['status']=='complete','Started/failed attempt retained: '+str(path))
            record['status']='complete' if phase(root,state,args) else 'resource_limited'
        except BaseException as error:
            record.update(status='failed',error=repr(error),traceback=traceback.format_exc())
            print('A66 SESSION ERROR:',repr(error),flush=True)
        finally:
            if (root/'plan.json').exists():
                logs=root/'logs';logs.mkdir(exist_ok=True)
                for file in state.glob('shard*.log'):shutil.copyfile(file,logs/file.name)
                try:
                    run('utils/diag_current_strong_anchor.py','--summarize','--root',str(root))
                    run('utils/diag_current_strong_anchor.py','--archive','--root',str(root))
                except subprocess.CalledProcessError as error:
                    record['summary_or_upload_error']=repr(error)
                    print('A66 results retained; --archive-only retries upload without GPU work.',flush=True)
            record['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
            rules.write(state/'session.json',record)
            if record['status']=='resource_limited':
                print('A66 RESOURCE PAUSE: reissue SAME command when allocated memory is available.',flush=True)


def follow(state,process):
    positions={}
    while True:
        for file in [state/'coordinator.log',*sorted(state.glob('shard*.log'))]:
            if file.exists():
                with file.open(errors='replace') as stream:
                    stream.seek(positions.get(str(file),0));text=stream.read();positions[str(file)]=stream.tell()
                if text:print(text,end='',flush=True)
        if not busy(state/'coordinator.lock') and (process is None or process.poll() is not None):break
        time.sleep(2)
    if (state/'session.json').exists():
        value=rules.read(state/'session.json');print('A66 SESSION:',value,flush=True)
        if value['status']=='failed':raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=rules.ROOT);parser.add_argument('--parent',default=design.ROOT)
    parser.add_argument('--gpus',default='0,1');parser.add_argument('--visible',default=os.environ.get('CUDA_VISIBLE_DEVICES'))
    parser.add_argument('--coordinator',action='store_true');parser.add_argument('--status',action='store_true')
    parser.add_argument('--archive-only',action='store_true')
    args=parser.parse_args();root=Path(args.root).resolve();state=state_dir(root);state.mkdir(parents=True,exist_ok=True)
    if args.coordinator:coordinator(args);return
    if args.status:
        print(dict(active=busy(state/'coordinator.lock'),session=rules.read(state/'session.json')
                   if (state/'session.json').exists() else None),flush=True);return
    if args.archive_only:
        rules.require(not busy(state/'coordinator.lock'),'Wait for active session before archive')
        run('utils/diag_current_strong_anchor.py','--archive','--root',str(root));return
    with exclusive(state/'client.lock'):
        process=None
        if busy(state/'coordinator.lock'):print('A66 existing session attached; no duplicate.',flush=True)
        else:
            for file in (root/'workers').glob('*/worker.lock'):rules.require(not busy(file),'Orphan live worker; use --status')
            if (state/'session.json').exists():
                old=rules.read(state/'session.json')
                if old['status']=='failed':
                    rules.require(no_model_attempts(root),'Failed paid attempt retained; use --status')
                    folder=state/'zero_model_preflight_failures';folder.mkdir(exist_ok=True)
                    rules.write(folder/(str(len(list(folder.glob('*.json'))))+'.json'),old)
                if old['status']=='complete':print('A66 already complete:',root/'comparison.md',flush=True);return
            command=[sys.executable,'-u',str(Path(__file__)),'--coordinator','--root',str(root),
                     '--parent',args.parent,'--gpus',args.gpus]
            if args.visible is not None:command+=['--visible',args.visible]
            with (state/'coordinator.log').open('ab',buffering=0) as log:
                process=subprocess.Popen(command,cwd=PROJECTS,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                         start_new_session=True,close_fds=True)
            print('A66 detached continuation started; following logs through final summary. Disconnect-safe.',flush=True)
        follow(state,process)


if __name__=='__main__':main()
