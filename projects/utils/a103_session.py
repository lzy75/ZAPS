"""A103 detached queue."""
import argparse,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P))
from utils import diag_shared_prefix_feedback_multinoise as d,shared_prefix_feedback_multinoise as r
from utils import historical_ffhq_gain as f
from utils.a54_session import exclusive,busy
from utils.a65_session import allocation
def sd(x):return x.parent/(x.name+'_session')
def inv(*a):subprocess.run([sys.executable,'-u',*a],cwd=P,check=True)
def ws(root,s):
 p=root/'workers'/str(s)/'status.json';return f.read(p)['status'] if p.exists() else 'not_started'
def workers(root,state,assigned):
 q=[s for s in (0,1) if ws(root,s)!='complete'];run={};fail=False
 while q or run:
  used={v[1]['uuid'] for v in run.values()}
  for s in list(q):
   row=assigned[str(s)]
   if row['uuid'] in used:continue
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=row['uuid'],MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1');log=(state/f'shard{s}.log').open('ab',buffering=0)
   p=subprocess.Popen([sys.executable,'-u','utils/diag_shared_prefix_feedback_multinoise.py','--worker','--root',str(root),'--shard',str(s),'--device-uuid',row['uuid']],cwd=P,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);run[s]=p,row,log;q.remove(s)
  for s,(p,row,log) in list(run.items()):
   if p.poll() is not None:log.close();del run[s];fail|=p.returncode!=0
  n=sum(f.read(x)['status']=='complete' for x in (root/'cases').glob('*/run.json')) if root.exists() else 0;print('A103 QUEUE:',{s:ws(root,s) for s in (0,1)},n,'/80',flush=True)
  if run:time.sleep(30)
 if fail:raise RuntimeError('A103 worker failed; records retained')
 return all(ws(root,s)=='complete' for s in (0,1))
def coord(a):
 root=Path(a.root);state=sd(root)
 with exclusive(state/'coordinator.lock'):
  rec=dict(status='running',pid=os.getpid(),root=str(root));f.write(state/'session.json',rec)
  try:
   inv('-m','unittest','-v','utils.test_shared_prefix_feedback','utils.test_shared_prefix_feedback_runtime','utils.test_shared_prefix_feedback_multinoise','utils.test_equal_nfe_feedback_runtime','utils.test_equal_nfe_feedback_r1','utils.test_equal_nfe_feedback_r3')
   inv('utils/diag_shared_prefix_feedback_multinoise.py','--prepare','--root',str(root),'--ffhq-dir',a.ffhq_dir,'--a71-root',a.a71_root);al=allocation(root,a.gpus,a.visible);rec['status']='resource_limited' if al is None else ('complete' if workers(root,state,al) else 'resource_limited')
  except BaseException as e:rec.update(status='failed',error=repr(e),traceback=traceback.format_exc());print('A103 FAILED:',repr(e))
  finally:
   if (root/'plan.json').exists():
    logs=root/'logs';logs.mkdir(exist_ok=True)
    for x in state.glob('shard*.log'):shutil.copyfile(x,logs/x.name)
    try:inv('utils/diag_shared_prefix_feedback_multinoise.py','--summarize','--root',str(root));
    except subprocess.CalledProcessError as e:rec['summary_error']=repr(e)
    if rec['status']=='complete':inv('utils/diag_shared_prefix_feedback_multinoise.py','--report','--root',str(root))
    inv('utils/diag_shared_prefix_feedback_multinoise.py','--archive','--root',str(root))
   rec['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());f.write(state/'session.json',rec)
def follow(state,child):
 off={}
 while True:
  for p in [state/'coordinator.log',*sorted(state.glob('shard*.log'))]:
   if p.exists():
    with p.open(errors='replace') as x:x.seek(off.get(str(p),0));v=x.read();off[str(p)]=x.tell()
    if v:print(v,end='',flush=True)
  if not busy(state/'coordinator.lock') and (child is None or child.poll() is not None):break
  time.sleep(2)
 if (state/'session.json').exists():
  row=f.read(state/'session.json');print('A103 SESSION:',row);raise SystemExit(1 if row['status']=='failed' else 0)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default=r.ROOT);p.add_argument('--a71-root',default=d.base.PARENT);p.add_argument('--ffhq-dir',default='/home/lzy/FFHQ/00000/00000');p.add_argument('--gpus',default='0,1');p.add_argument('--visible',default=os.environ.get('CUDA_VISIBLE_DEVICES'))
 for x in ('coordinator','status'):p.add_argument('--'+x,action='store_true')
 a=p.parse_args();root=Path(a.root);state=sd(root);state.mkdir(parents=True,exist_ok=True)
 if a.coordinator:coord(a);return
 if a.status:print(dict(active=busy(state/'coordinator.lock'),session=f.read(state/'session.json') if (state/'session.json').exists() else None,completed=sum(f.read(x)['status']=='complete' for x in (root/'cases').glob('*/run.json')) if root.exists() else 0,planned=80));return
 with exclusive(state/'client.lock'):
  child=None
  if not busy(state/'coordinator.lock'):
   cmd=[sys.executable,'-u',str(Path(__file__)),'--coordinator','--root',str(root),'--a71-root',a.a71_root,'--ffhq-dir',a.ffhq_dir,'--gpus',a.gpus]
   with (state/'coordinator.log').open('ab',buffering=0) as log:child=subprocess.Popen(cmd,cwd=P,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  follow(state,child)
if __name__=='__main__':main()
