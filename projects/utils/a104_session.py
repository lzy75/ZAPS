"""A104 detached queue."""
import argparse,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS))
from utils import diag_actual_response_budget_control as driver,actual_response_budget_control as rules
from utils import historical_ffhq_gain as files
from utils.a54_session import exclusive,busy
from utils.a65_session import allocation
def state_dir(root):return root.parent/(root.name+'_session')
def invoke(*a):subprocess.run([sys.executable,'-u',*a],cwd=PROJECTS,check=True)
def shard_status(root,s):
 p=root/'workers'/str(s)/'status.json';return files.read(p)['status'] if p.exists() else 'not_started'
def workers(root,state,assigned):
 queued=[s for s in (0,1) if shard_status(root,s)!='complete'];running={};failed=False
 files.require(all(shard_status(root,s)!='failed' for s in queued),'Prior A104 failure retained')
 while queued or running:
  occupied={row['uuid'] for _,row,_ in running.values()}
  for s in list(queued):
   row=assigned[str(s)]
   if row['uuid'] in occupied:continue
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=row['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',MKL_THREADING_LAYER='GNU')
   log=(state/f'shard{s}.log').open('ab',buffering=0);proc=subprocess.Popen([sys.executable,'-u','utils/diag_actual_response_budget_control.py','--worker','--root',str(root),'--shard',str(s),'--device-uuid',row['uuid']],cwd=PROJECTS,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   running[s]=proc,row,log;queued.remove(s);occupied.add(row['uuid']);print('A104 START:',s,'GPU',row['index'],flush=True)
  for s,(proc,row,log) in list(running.items()):
   if proc.poll() is not None:log.close();del running[s];failed|=proc.returncode!=0;print('A104 END:',s,proc.returncode,flush=True)
  complete=sum(files.read(x)['status']=='complete' for x in (root/'cases').glob('*/run.json')) if root.exists() else 0
  print('A104 QUEUE:',{s:shard_status(root,s) for s in (0,1)},complete,'/12',flush=True)
  if running:time.sleep(30)
 if failed:raise RuntimeError('A104 worker failed; records retained')
 return all(shard_status(root,s)=='complete' for s in (0,1))
def coordinate(a):
 root=Path(a.root).resolve();state=state_dir(root)
 with exclusive(state/'coordinator.lock'):
  record=dict(status='running',pid=os.getpid(),root=str(root));files.write(state/'session.json',record)
  try:
   invoke('-m','unittest','-v','utils.test_actual_response_budget_control','utils.test_equal_budget_ray_diagonal','utils.test_equal_budget_ray_diagonal_runtime','utils.test_full_unroll_response_runtime')
   invoke('utils/diag_actual_response_budget_control.py','--prepare','--root',str(root),'--a100-root',a.a100_root)
   assigned=allocation(root,a.gpus,a.visible);record['status']='resource_limited' if assigned is None else ('complete' if workers(root,state,assigned) else 'resource_limited')
  except BaseException as e:record.update(status='failed',error=repr(e),traceback=traceback.format_exc());print('A104 FAILED:',repr(e),flush=True)
  finally:
   if (root/'plan.json').exists():
    logs=root/'logs';logs.mkdir(exist_ok=True)
    for p in state.glob('shard*.log'):shutil.copyfile(p,logs/p.name)
    try:invoke('utils/diag_actual_response_budget_control.py','--summarize','--root',str(root));invoke('utils/diag_actual_response_budget_control.py','--archive','--root',str(root))
    except subprocess.CalledProcessError as e:record['summary_or_upload_error']=repr(e)
   record['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());files.write(state/'session.json',record)
def follow(state,child):
 offsets={}
 while True:
  for p in [state/'coordinator.log',*sorted(state.glob('shard*.log'))]:
   if p.exists():
    with p.open(errors='replace') as f:f.seek(offsets.get(str(p),0));v=f.read();offsets[str(p)]=f.tell()
    if v:print(v,end='',flush=True)
  if not busy(state/'coordinator.lock') and (child is None or child.poll() is not None):break
  time.sleep(2)
 if (state/'session.json').exists():
  row=files.read(state/'session.json');print('A104 SESSION:',row,flush=True)
  if row['status']=='failed':raise SystemExit(1)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default=rules.ROOT);p.add_argument('--a100-root',default=rules.PARENT);p.add_argument('--gpus',default='0,1');p.add_argument('--visible',default=os.environ.get('CUDA_VISIBLE_DEVICES'))
 for flag in ('coordinator','status','archive-only'):p.add_argument('--'+flag,action='store_true')
 a=p.parse_args();root=Path(a.root).resolve();state=state_dir(root);state.mkdir(parents=True,exist_ok=True)
 if a.coordinator:coordinate(a);return
 complete=sum(files.read(x)['status']=='complete' for x in (root/'cases').glob('*/run.json')) if root.exists() else 0
 if a.status:print(dict(active=busy(state/'coordinator.lock'),session=files.read(state/'session.json') if (state/'session.json').exists() else None,workers={s:shard_status(root,s) for s in (0,1)},completed_cases=complete,planned_cases=12));return
 if a.archive_only:files.require(not busy(state/'coordinator.lock'),'Active');invoke('utils/diag_actual_response_budget_control.py','--summarize','--root',str(root));invoke('utils/diag_actual_response_budget_control.py','--archive','--root',str(root));return
 with exclusive(state/'client.lock'):
  child=None
  if not busy(state/'coordinator.lock'):
   if (state/'session.json').exists():
    prior=files.read(state/'session.json');files.require(prior['status']!='failed','Prior A104 failure retained')
    if prior['status']=='complete':print('A104 already complete.');return
   cmd=[sys.executable,'-u',str(Path(__file__)),'--coordinator','--root',str(root),'--a100-root',a.a100_root,'--gpus',a.gpus]
   if a.visible is not None:cmd+=['--visible',a.visible]
   with (state/'coordinator.log').open('ab',buffering=0) as log:child=subprocess.Popen(cmd,cwd=PROJECTS,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
   print('A104 detached queue started; following.',flush=True)
  follow(state,child)
if __name__=='__main__':main()
