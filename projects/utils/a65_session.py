"""Disconnect-safe two-phase A65 queue; single-GPU serial or two-GPU parallel."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS))
from utils import historical_ffhq_gain as rules
from utils.a54_session import exclusive,busy
from utils.diag_paper_clock_alignment import gpu_inventory


def state_dir(root):return root.parent/(root.name+'_session')


def no_model_attempts(root):
    if any((root/'cases').glob('*/*/repeat_*/run.json')) or (root/'null_diagnostic').exists():return False
    for file in (root/'workers').glob('*/load_*.json'):
        if rules.read(file).get('warmup',{}).get('forward_attempts',0)>0:return False
    return True


def run(*arguments):
    subprocess.run([sys.executable,'-u',*arguments],cwd=PROJECTS,check=True)


def status(root,shard,phase):
    file=root/'workers'/('shard'+str(shard))/('status_'+phase+'.json')
    return rules.read(file).get('status') if file.exists() else 'not_started'


def allocation(root,requested,visible):
    file=root/'allocation.json'
    rows=rules.available_gpus(gpu_inventory(),requested,visible)
    if file.exists():
        assigned=rules.read(file)
        rules.require(all(v['uuid'] in {r['uuid'] for r in rows} for v in assigned.values()),
                      'Pinned timing device is outside current allocation')
        return assigned
    rows=sorted((r for r in rows if r['free_bytes']>=6*rules.GIB),key=lambda r:-r['free_bytes'])
    if not rows:return None
    assigned={'0':rows[0],'1':rows[1] if len(rows)>1 else rows[0]}
    rules.write(file,assigned);return assigned


def phase(root,state,assigned,name):
    queued=[i for i in (0,1) if status(root,i,name)!='complete'];running={};failed=False
    rules.require(not any(status(root,i,name)=='failed' for i in queued),'Failed phase preserved; no automatic retry')
    while queued or running:
        occupied={row['uuid'] for _,row,_ in running.values()}
        for shard in list(queued):
            row=assigned[str(shard)]
            if row['uuid'] in occupied:continue
            environment=dict(os.environ,CUDA_VISIBLE_DEVICES=row['uuid'],OMP_NUM_THREADS='2',
                             MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            log=(state/f'shard{shard}_{name}.log').open('ab',buffering=0)
            process=subprocess.Popen([sys.executable,'-u','utils/diag_historical_ffhq_gain.py','--worker',
                '--root',str(root),'--shard',str(shard),'--phase',name,'--device-uuid',row['uuid']],
                cwd=PROJECTS,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            running[shard]=(process,row,log);queued.remove(shard);occupied.add(row['uuid'])
            print('A65 WORKER START:',name,'shard',shard,'GPU',row['index'],flush=True)
        for shard,(process,row,log) in list(running.items()):
            if process.poll() is not None:
                log.close();del running[shard]
                failed|=process.returncode!=0
                print('A65 WORKER END:',name,shard,'exit=',process.returncode,'status=',status(root,shard,name),flush=True)
        print('A65 QUEUE STATUS:',name,{i:status(root,i,name) for i in (0,1)},flush=True)
        if running:time.sleep(30)
    if failed:raise RuntimeError('A65 worker failed; retained records, no automatic retry')
    return all(status(root,i,name)=='complete' for i in (0,1))


def coordinator(args):
    root=Path(args.root).resolve();state=state_dir(root)
    with exclusive(state/'coordinator.lock'):
        record=dict(status='running',pid=os.getpid(),root=str(root),gpus=args.gpus,visible=args.visible)
        rules.write(state/'session.json',record)
        try:
            run('-m','unittest','-v','utils.test_historical_ffhq_gain','utils.test_historical_ffhq_runtime',
                'utils.test_current_score_checkpoint_r2')
            run('utils/diag_historical_ffhq_gain.py','--prepare','--root',str(root),'--ffhq-dir',args.ffhq_dir)
            for file in (root/'cases').glob('*/*/repeat_*/run.json'):
                rules.require(rules.read(file)['status']=='complete','Started/failed attempt retained: '+str(file))
            assigned=allocation(root,args.gpus,args.visible)
            if assigned is None:
                record['status']='resource_limited';print('A65 no allocated GPU has6GiB free; no attempt started.',flush=True)
            elif not phase(root,state,assigned,'baseline'):
                record['status']='resource_limited'
            else:
                run('utils/diag_historical_ffhq_gain.py','--qualify','--root',str(root))
                if not rules.read(root/'replay_gate.json')['passed']:
                    record.update(status='complete',outcome='historical_baseline_review_required')
                elif phase(root,state,assigned,'comparison'):
                    record.update(status='complete',outcome='all_comparisons_completed')
                else:record['status']='resource_limited'
        except BaseException as error:
            record.update(status='failed',error=repr(error),traceback=traceback.format_exc())
            print('A65 SESSION ERROR:',repr(error),flush=True)
        finally:
            # Worker logs are closed before copying/archiving; no mutable source during export.
            if (root/'plan.json').exists():
                logs=root/'logs';logs.mkdir(exist_ok=True)
                for file in state.glob('shard*.log'):shutil.copyfile(file,logs/file.name)
                try:
                    run('utils/diag_historical_ffhq_gain.py','--summarize','--root',str(root))
                    run('utils/diag_historical_ffhq_gain.py','--archive','--root',str(root))
                except subprocess.CalledProcessError as error:
                    record['summary_or_upload_error']=repr(error)
                    print('A65 local records retained. Use --archive-only for upload retry; no GPU rerun.',flush=True)
            record['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
            rules.write(state/'session.json',record)
            if record['status']=='resource_limited':
                print('A65 RESOURCE PAUSE: reissue SAME command after memory is available. No other process stopped.',flush=True)


def follow(state,process):
    offsets={}
    while True:
        for file in [state/'coordinator.log',*sorted(state.glob('shard*.log'))]:
            if file.exists():
                with file.open(errors='replace') as stream:
                    stream.seek(offsets.get(str(file),0));value=stream.read();offsets[str(file)]=stream.tell()
                if value:print(value,end='',flush=True)
        if not busy(state/'coordinator.lock') and (process is None or process.poll() is not None):break
        time.sleep(2)
    file=state/'session.json'
    if file.exists():
        result=rules.read(file);print('A65 SESSION:',result,flush=True)
        if result['status']=='failed':raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=rules.ROOT);parser.add_argument('--gpus',default='0,1')
    parser.add_argument('--ffhq-dir',default='/home/lzy/FFHQ/00000/00000')
    parser.add_argument('--visible',default=os.environ.get('CUDA_VISIBLE_DEVICES'))
    parser.add_argument('--coordinator',action='store_true');parser.add_argument('--status',action='store_true')
    parser.add_argument('--archive-only',action='store_true')
    args=parser.parse_args();root=Path(args.root).resolve();state=state_dir(root);state.mkdir(parents=True,exist_ok=True)
    if args.coordinator:coordinator(args);return
    if args.status:
        print(dict(active=busy(state/'coordinator.lock'),session=rules.read(state/'session.json')
                   if (state/'session.json').exists() else None),flush=True);return
    if args.archive_only:
        rules.require(not busy(state/'coordinator.lock'),'Active coordinator; wait before archive')
        run('utils/diag_historical_ffhq_gain.py','--archive','--root',str(root));return
    with exclusive(state/'client.lock'):
        process=None
        if busy(state/'coordinator.lock'):print('A65 existing session attached; no duplicate.',flush=True)
        else:
            for file in (root/'workers').glob('*/worker.lock'):
                rules.require(not busy(file),'Orphan live worker; use --status, no duplicate')
            if (state/'session.json').exists():
                old=rules.read(state/'session.json')
                if old['status']=='failed':
                    rules.require(no_model_attempts(root),
                                  'Failed model/diagnostic attempt retained. Use --status; no automatic retry')
                    preserved=state/'zero_model_preflight_failures';preserved.mkdir(exist_ok=True)
                    rules.write(preserved/(str(len(list(preserved.glob('*.json'))))+'.json'),old)
                    print('A65 prior failure had no pretrained-model attempt; this explicit invocation reruns preflight only.',flush=True)
                if old['status']=='complete':print('A65 already complete:',root/'comparison.md',flush=True);return
            command=[sys.executable,'-u',str(Path(__file__)),'--coordinator','--root',str(root),
                     '--gpus',args.gpus,'--ffhq-dir',args.ffhq_dir]
            if args.visible is not None:command+=['--visible',args.visible]
            with (state/'coordinator.log').open('ab',buffering=0) as log:
                process=subprocess.Popen(command,cwd=PROJECTS,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                         start_new_session=True,close_fds=True)
            print('A65 detached queue started; this client follows through final summary. Disconnect-safe.',flush=True)
        follow(state,process)


if __name__=='__main__':main()
