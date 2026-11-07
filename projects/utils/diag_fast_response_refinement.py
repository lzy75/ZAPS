"""A98: zero-NN SR audit, then cheap same-target30-position refinement after A97."""
import os
os.environ['MKL_THREADING_LAYER']='GNU';os.environ.pop('MKL_SERVICE_FORCE_INTEL',None)
import argparse,contextlib,csv,shutil,statistics,subprocess,sys,time,traceback
from pathlib import Path
from unittest.mock import patch
PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import historical_ffhq_gain as files,diag_full_unroll_response as old,fast_response_refinement as rules
from utils.a54_session import exclusive,busy
ROOT='/data0/lzy/ZAPS-results/diag_fast_response/A98_20261107'
PARENT=old.ROOT
BRANCH='results/zaps-fast-response-refinement-20261107'
REFERENCES=('BASE','LEGACY_SCALAR','REAL_SCALAR','REAL_DIAG')
NAMES=('utils/fast_response_refinement.py','utils/diag_fast_response_refinement.py','utils/test_fast_response_refinement.py',
 'utils/test_fast_response_refinement_runtime.py','run_a98_20261107.sh','experiment_cards/A98_fast_response_refinement_20261107.md')


def check(root):
 p=files.read(root/'plan.json');files.require(p['experiment']=='A98_fast_response','Wrong A98 plan')
 for n,h in p['sources'].items():files.require(files.digest(PROJECTS/n)==h,'Source changed: '+n)
 for n,h in p['inputs'].items():files.require(files.digest(n)==h,'Input changed: '+n)
 return p


def source_rows(parent,task=None):
 rows=[]
 for path in sorted((parent/'cases').glob('*/*/run.json')):
  r=files.read(path)
  if r['status']!='complete' or r['arm'] not in REFERENCES or (task and r['case']['task']!=task):continue
  rows.append((path,r))
 return rows


def audit_sr(parent,state):
 """CPU only. Completed immutable outputs; NEVER load the diffusion model."""
 destination=state/'sr_audit.json'
 if destination.exists():
  saved=files.read(destination)
  for n,h in saved['inputs'].items():files.require(files.digest(n)==h,'Audited complete source changed')
  print('A98 SR AUDIT reused:',destination,flush=True);return saved
 p=old.check(parent)
 import torch
 from modules.main_single import load_image_as_tensor
 from modules.wavelet import OrthogonalDWT2D
 from utils.diag_shrink_multitask import make_operator
 torch.set_num_threads(2);dwt=OrthogonalDWT2D('db4',3);rows=[];bound={str(parent/'plan.json'):files.digest(parent/'plan.json')}
 def tensor(path):return torch.load(path,map_location='cpu',weights_only=True)
 start=time.perf_counter()
 for path,r in source_rows(parent,'super_resolution'):
  c=r['case'];output=path.parent/'output.pt';files.require(files.digest(output)==r['artifacts']['output.pt'],'A97 output changed')
  bound[str(path)]=files.digest(path);bound[str(output)]=files.digest(output)
  x=tensor(output);gt=load_image_as_tensor(c['image']).cpu();bound[c['image']]=files.digest(c['image'])
  op=make_operator(Path(p['a71_parent']),c,'cpu');y=tensor(c['observation'])['y'];bound[c['observation']]=files.digest(c['observation'])
  with torch.no_grad():
   error=x.double()-gt.double();coeff=dwt.analysis(error);h,w=coeff.shape[-2:];coarse=coeff[...,:h//4,:w//4].square().sum()/error.numel()
   total=coeff.square().sum()/error.numel();raw=error.square().mean();files.require(abs(float(total-raw))<=1e-5*max(float(raw),1e-12),'Wavelet decomposition failed')
   observed_true=float((op.H(x)-op.H(gt)).double().square().mean());observed_y=float((op.H(x)-y).double().square().mean())
  steps=[]
  for meta in sorted((path.parent/'steps').glob('step_*.json')):
   item=files.read(meta);bound[str(meta)]=files.digest(meta);d=item['detail'];s=d.get('scalar',{});a=d.get('diagonal',{})
   steps.append(dict(step=item['step'],t=d['t'],next_t=d['next_t'],omega=d.get('omega'),scalar_norm=d.get('scalar_norm'),
    diagonal_norm=d.get('diagonal_norm'),cosine=d.get('diagonal_cosine_to_native'),negative_fraction=d.get('negative_gain_fraction'),
    scalar_value=s.get('final_value'),diagonal_value=a.get('final_value'),diagonal_stop=a.get('status')))
  row=dict(case=c['key'],arm=r['arm'],metrics=r['metrics'],raw_mse=float(raw),clamped_mse=float((x.clamp(-1,1).double()-gt.double()).square().mean()),
   clamp_fraction=float((x.abs()>1).float().mean()),coarse_LL2_mse=float(coarse),fine_mse=float(total-coarse),
   clean_observation_mse=observed_true,noisy_observation_mse=observed_y,steps=steps);rows.append(row)
  print('A98 SR AUDIT:',c['image_id'],r['arm'],'raw/coarse/fine=',*[round(row[k],7) for k in ('raw_mse','coarse_LL2_mse','fine_mse')],'Htrue=',round(observed_true,7),flush=True)
 files.require(len(rows)==12,'Need completed3 SR images x4 core arms for zero-NN audit')
 result=dict(status='complete',inputs=bound,rows=rows,UNet=0,VJP=0,JVP=0,offline_H_actions=3*len(rows),offline_dwt_analysis=len(rows),seconds=time.perf_counter()-start,
  scope='Completed A97 final outputs and recorded local objectives only. LL2/fine are orthogonal frequency errors, NOT exact H-nullspace. No intermediate noisy state is called a clean prediction. GT only offline; no method selection.')
 files.write(destination,result);return result


def prepare(root,parent,audit):
 if (root/'plan.json').exists():return check(root)
 files.require(not root.exists() or not any(root.iterdir()),'Unplanned nonempty root retained')
 pp=old.check(parent);s=files.read(parent/'summary.json');files.require(s['completed_runs']==s['planned_runs']==72,'Completed A97 required; do not compete with its workers')
 bound=dict(pp['inputs']);bound[str(parent/'plan.json')]=files.digest(parent/'plan.json');bound[str(parent/'summary.json')]=files.digest(parent/'summary.json');bound.update(audit['inputs'])
 cases=[]
 for c0 in pp['cases']:
  c=dict(c0);c['a97_references']={}
  for arm in REFERENCES:
   path=parent/'cases'/c['key']/arm/'run.json';r=files.read(path);files.require(r['status']=='complete','Incomplete A97 control')
   for n,h in r['artifacts'].items():files.require(files.digest(path.parent/n)==h,'Reference artifact changed')
   c['a97_references'][arm]=str(path);bound[str(path)]=files.digest(path)
   for n in ('output.pt','final_rng.pt'):bound[str(path.parent/n)]=r['artifacts'][n]
  cases.append(c)
 root.mkdir(parents=True)
 p=dict(experiment='A98_fast_response',parent=str(parent),a71_parent=pp['a71_parent'],cases=cases,checkpoint=pp['checkpoint'],grid=pp['grid'],arms=list(rules.ARMS),budget=rules.budget(),
  sources={**pp['sources'],**{n:files.digest(PROJECTS/n) for n in NAMES}},inputs=bound,
  protocol=dict(scope='Original3 FFHQ x4tasks sigma_model.05, same cached nine-unroll prefix/learned zetaD, original30points and29 innovations, IEEE, last_opt. Four NEW full30 suffix policies,48runs; no new image/seed/noise or optimizer. Completed A97 controls reused, costs remain paid.',
   target='EXACT SAME A97 target at each OWN state: F_s=PH(clamped clean_s); base=F_s(b),r=Py-base,omega=(1-Msigma²/||r||²)+,target=base+omega*r. Terminal F_-1=PH(identity), no negative-time NN. Known-noise heuristic, not calibrated whole-image risk.',
   scalar='SCALAR_K0 uses the frozen-epsilon NEXT-state response Hq/sqrt(alpha_s) and closed scalar projection. Differs from legacy A71 current-state reference; both controls retained. No expensive real-scalar nonlinear optimization as initializer.',
   solver='DIAG_K1/DIAG_K3 start same closed-form scalar, jointly update physical wavelet coefficients1/3 times. g=active*J^T residual; eta=||g||²/||Jg||² from exact local response. Candidate p-eta*g, at most2 true value checks eta/eta2; positive actual gain/agreement>.1. No innerCG/LM loop/damping/0..2 bound/new regularizer. Budget is not convergence. Signed coefficients allowed.',
   structure='COARSE_K3 uses same3 updates and objective, but only top-left quarter in each axis (LL2 within originaldb4/level3 packing) may change from scalar initializer. Fine coefficients keep scalar values. Same structure on ALL4 tasks, not GT-dependent task selection. LL2 is motivated by4x spatial scale, not asserted exact physical rowspace/nullspace.',
   costs='Per suffix forward upper bounds59/175/407/407; total12576F,2436VJP+2436JVP maximum plus actual2F/1VJP/1JVP probe each model load. Terminal operator derivatives separate. Equivalent fullmethod forwards270+actualsuffix; cached learning NOT free. Same-session suffix/checkpoint timing, no independent300-clock or inheritedA97 runtime subtraction.',
   pairing='Reuse A97 nativeparts/full30 endpoint/time/RNG gates, saved per-position checkpoints and resource-only boundary resume. Actual checkpoint source fingerprints plus forwardAD primal/adjoint probe mandatory. Frozen weights, no optimizer. Final RNG matches original A92; outputs/metrics use original protocol.',
   diagnosis='CPU-only SR audit first: all3 complete core arms final rawMSE/LL2/fine energy, H(output)-H(GT) vs H(output)-y, recorded norms/cosines/negativefraction/objectives/stops. No new network replay and no claim of unique causal attribution. Frequency bands not H-nullspace.',
   decision='Primary fastfull vs same-target SCALAR_K0, per task meanPSNR>=+.1dB,>=2/3wins,SSIMnondecrease,LPIPSnonincrease. SR recovery from prior matrix regression requires >=2/3PSNRwins vs cached A97 REAL_DIAG; usable against ZAPS separately requires meanPSNR/SSIMnondecrease,LPIPSnonincrease. Report all configs; no GT-selected final policy or auto additional experiments.',
   resources='Read completed immutable A97 files only; audit CPU while A97 runs, then wait for A97 lock to clear and72complete. No task termination. Up to2 explicitly allocated GPUs>=8GiB, pinned static cases. No automatic retry for failed/interrupted attempts; no weight/precision downloads. No changes to A97 or historical SHA-bound sources.'))
 files.write(root/'plan.json',p);files.write(root/'sr_audit.json',audit);shutil.copyfile(PROJECTS/'experiment_cards/A98_fast_response_refinement_20261107.md',root/'experiment_card.md');return p


class LabelStream:
 def __init__(self,stream):self.stream=stream
 def write(self,value):return self.stream.write(value.replace('A97','A98'))
 def flush(self):self.stream.flush()


def worker(root,index):
 # Reuse immutable, already exercised step checkpoints, model admission and RNG.
 with patch.multiple(old,check=check,rules=rules),contextlib.redirect_stdout(LabelStream(sys.stdout)):
  old.worker(root,index)


def summarize(root):
 p=check(root);new=[files.read(f) for f in (root/'cases').glob('*/*/run.json')];done=[r for r in new if r['status']=='complete'];refs=[]
 for c in p['cases']:
  for path in c['a97_references'].values():refs.append(files.read(path))
 groups=[];comparisons=[]
 for task in ('super_resolution','gaussian_deblur','inpainting','motion_deblur'):
  for arm in (*REFERENCES,*rules.ARMS):
   rows=[r for r in [*refs,*done] if r['case']['task']==task and r['arm']==arm]
   if rows:groups.append(dict(task=task,arm=arm,images=len(rows),quality={k:statistics.mean(r['metrics'][k] for r in rows) for k in ('psnr','ssim','lpips')},
    mean_suffix_seconds=statistics.mean(r['suffix_seconds'] for r in rows),mean_suffix_forwards=statistics.mean(r['counts']['forward_attempts'] for r in rows),reused=arm in REFERENCES))
  for arm in rules.ARMS:
   selected=[r for r in done if r['case']['task']==task and r['arm']==arm]
   for reference in (*REFERENCES,'SCALAR_K0'):
    if reference==arm:continue
    by={r['case']['key']:r for r in [*refs,*done] if r['case']['task']==task and r['arm']==reference};pairs=[(by[r['case']['key']],r) for r in selected if r['case']['key'] in by]
    if pairs:
     delta={k:statistics.mean(b['metrics'][k]-a['metrics'][k] for a,b in pairs) for k in ('psnr','ssim','lpips')};wins=sum(b['metrics']['psnr']>a['metrics']['psnr'] for a,b in pairs)
     comparisons.append(dict(task=task,arm=arm,reference=reference,images=len(pairs),delta=delta,wins=wins,positive=len(pairs)==3 and wins>=2 and delta['psnr']>=.1 and delta['ssim']>=0 and delta['lpips']<=0))
 s=dict(experiment=p['experiment'],completed_runs=len(done),planned_runs=48,groups=groups,comparisons=comparisons,
  actual_new_cost={k:sum(r.get('counts',{}).get(k,0) for r in new) for k in ('forward_attempts','forwards','input_vjp_attempts','input_jvp_attempts','optimizer_backward_attempts','terminal_operator_vjp_attempts','terminal_operator_jvp_attempts')},
  cost_is_lower_bound=any(r['status'] in ('running','failed') for r in new),protocol=p['protocol'],probe_loads=[files.read(f) for f in (root/'workers').glob('*/loads/*.json')])
 files.write(root/'summary.json',s)
 with (root/'comparison.csv').open('w',newline='') as stream:
  writer=csv.writer(stream);writer.writerow(['task','arm','images','PSNR','SSIM','LPIPS','suffix_seconds','suffix_forwards','reused'])
  for r in groups:writer.writerow([r['task'],r['arm'],r['images'],*[r['quality'][k] for k in ('psnr','ssim','lpips')],r['mean_suffix_seconds'],r['mean_suffix_forwards'],r['reused']])
 print('A98 STATUS:',len(done),'/48',flush=True)
 for r in groups:print('A98 GROUP:',r['task'],r['arm'],r['images'],r['quality'],'seconds=',round(r['mean_suffix_seconds'],3),'F=',r['mean_suffix_forwards'],flush=True)
 return s


def archive(root):
 from utils import archive_refinement_confirmation as arch
 arch.BRANCH=BRANCH;bundle,sha=arch.export_and_commit({'A98':root},projects=PROJECTS)
 print('Archive:',bundle,'Results commit:',sha,'Branch:',BRANCH,flush=True)
 subprocess.run(['git','push','origin',BRANCH],cwd=PROJECTS.parent,check=True);print('RESULTS_PUSH_PASS: A98 uploaded.',flush=True)


def coordinate(args,root,parent,state,record):
 children=[];streams=[]
 with exclusive(state/'worker.lock'):
  status=dict(status='running',pid=os.getpid(),root=str(root));files.write(record,status)
  try:
   with exclusive(state/'audit.lock'):audit=audit_sr(parent,state)
   parent_state=parent.parent/(parent.name+'_session')
   while busy(parent_state/'worker.lock') or any(busy(f) for f in (parent/'workers').glob('*/worker.lock')):
    status.update(status='waiting_for_A97',reason='CPU audit done; preserve A97 GPU queue');files.write(record,status)
    print('A98 WAIT: A97 still active; no A98 model/GPU calls.',flush=True);time.sleep(30)
   status.update(status='running');status.pop('reason',None);files.write(record,status)
   if not (parent/'summary.json').exists() or files.read(parent/'summary.json')['completed_runs']!=72:raise old.ResourcePause('A97 stopped before72complete; retain its queue, resume A97 first. No A98 GPU attempts.')
   prepare(root,parent,audit)
   from utils.diag_paper_clock_alignment import gpu_inventory
   choices=files.available_gpus(gpu_inventory(),args.gpus,os.environ.get('CUDA_VISIBLE_DEVICES'))
   choices=sorted([d for d in choices if d['free_bytes']>=8*files.GIB],key=lambda d:-d['free_bytes'])
   allocation=root/'allocation.json'
   if allocation.exists():
    devices=files.read(allocation)['devices']
    if not all(any(c['uuid']==d['uuid'] for c in choices) for d in devices):raise old.ResourcePause('Pinned GPU(s) lack8GiB margin')
   else:
    if not choices:raise old.ResourcePause('No allocated GPU>=8GiB')
    devices=choices[:2];files.write(allocation,dict(devices=devices))
   command=[sys.executable,'-m','unittest','-v','utils.test_fast_response_refinement','utils.test_fast_response_refinement_runtime','utils.test_full_unroll_response_runtime']
   result=subprocess.run(command,cwd=PROJECTS,capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':devices[0]['uuid']})
   print(result.stdout+result.stderr,end='',flush=True);files.write(root/'preflight_tests.json',dict(command=command,returncode=result.returncode,output=result.stdout+result.stderr));result.check_returncode()
   for index in range(len(devices)):
    folder=root/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'session.json').exists():files.require(files.read(folder/'session.json')['status'] in ('complete','resource_limited'),'Failed/interrupted worker retained')
    logfile=folder/'worker.log';stream=logfile.open('ab',buffering=0);streams.append(stream);offset=logfile.stat().st_size
    child=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--root',str(root),'--worker-index',str(index)],cwd=PROJECTS,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
    children.append(dict(child=child,log=logfile,offset=offset,index=index))
   while children:
    alive=False
    for item in children:
     with item['log'].open('rb') as stream:stream.seek(item['offset']);chunk=stream.read();item['offset']=stream.tell()
     if chunk:print(chunk.decode(errors='replace'),end='',flush=True)
     if item['child'].poll() is None:alive=True
    if not alive:break
    time.sleep(1)
   statuses=[files.read(root/'workers'/str(i)/'session.json')['status'] for i in range(len(devices))]
   if any(c['child'].returncode for c in children) or 'failed' in statuses:raise RuntimeError('A98 worker failed; attempts retained, no automatic retry')
   status['status']='resource_limited' if 'resource_limited' in statuses else 'complete'
   if status['status']=='complete':files.require(sum(files.read(f)['status']=='complete' for f in (root/'cases').glob('*/*/run.json'))==48,'Incomplete48 runs')
  except old.ResourcePause as e:status.update(status='resource_limited',reason=str(e));print('A98 RESOURCE PAUSE:',e,flush=True)
  except BaseException as e:status.update(status='failed',error=repr(e),traceback=traceback.format_exc());print('A98 FAILED:',status,flush=True)
  finally:
   for item in children:item['child'].wait()
   for stream in streams:stream.close()
   files.write(record,status)
   if (root/'plan.json').exists():
    shutil.copyfile(record,root/'session.json')
    try:summarize(root);archive(root)
    except Exception as e:print('A98 archive incomplete; use --archive-only:',repr(e),flush=True)


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=ROOT);parser.add_argument('--a97-root',default=PARENT);parser.add_argument('--gpus',default='0,1');parser.add_argument('--worker-index',type=int)
 for flag in ('execute','status','archive-only','audit-only'):parser.add_argument('--'+flag,action='store_true')
 args=parser.parse_args();root=Path(args.root).resolve();parent=Path(args.a97_root).resolve();state=root.parent/(root.name+'_session');state.mkdir(parents=True,exist_ok=True);record=state/'session.json'
 if args.worker_index is not None:
  folder=root/'workers'/str(args.worker_index);folder.mkdir(parents=True,exist_ok=True)
  with exclusive(folder/'worker.lock'):worker(root,args.worker_index)
  if files.read(folder/'session.json')['status']=='failed':raise SystemExit(1)
  return
 if args.status:
  print(dict(active=busy(state/'worker.lock'),session=files.read(record) if record.exists() else None,workers=[files.read(f) for f in (root/'workers').glob('*/session.json')]))
  for path in sorted((root/'cases').glob('*/*/run.json')):
   r=files.read(path);print(r['case']['key'],r['arm'],r['status'],str(r['completed_steps'])+'/30','F=',r['counts'].get('forwards',0))
  return
 if args.audit_only:
  with exclusive(state/'audit.lock'):audit_sr(parent,state)
  return
 if args.archive_only:files.require(not busy(state/'worker.lock'),'Worker active');summarize(root);archive(root);return
 if args.execute:coordinate(args,root,parent,state,record);return
 with exclusive(state/'client.lock'):
  child=None
  if not busy(state/'worker.lock'):
   if record.exists():
    prior=files.read(record);files.require(prior['status'] in ('complete','resource_limited'),'Prior failure/interruption retained; no automatic retry')
    if prior['status']=='complete':print('A98 already complete; --status / --archive-only');return
   command=[sys.executable,'-u',str(Path(__file__)),'--execute','--root',str(root),'--a97-root',str(parent),'--gpus',args.gpus]
   with (state/'worker.log').open('ab',buffering=0) as log:child=subprocess.Popen(command,cwd=PROJECTS,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  offset=0
  while True:
   if (state/'worker.log').exists():
    with (state/'worker.log').open('rb') as stream:stream.seek(offset);chunk=stream.read();offset=stream.tell()
    if chunk:print(chunk.decode(errors='replace'),end='',flush=True)
   if not busy(state/'worker.lock') and (child is None or child.poll() is not None):break
   time.sleep(1)
  if record.exists():
   status=files.read(record);print('A98 SESSION:',status,flush=True)
   if status['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
