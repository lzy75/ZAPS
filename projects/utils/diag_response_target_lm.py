"""A95 fixed-state response-target experiment; native files and old results unchanged."""
import os
os.environ['MKL_THREADING_LAYER']='GNU';os.environ.pop('MKL_SERVICE_FORCE_INTEL',None)
import argparse,csv,subprocess,sys,time,traceback,statistics,shutil
from pathlib import Path
PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import historical_ffhq_gain as files,response_target_lm as rules
from utils import diag_joint_diagonal_response as parent_driver
from utils.a54_session import exclusive,busy
ROOT='/data0/lzy/ZAPS-results/diag_response_target/A95_20261106'
PARENT=parent_driver.ROOT
BRANCH='results/zaps-response-target-lm-20261106'
NAMES=('utils/response_target_lm.py','utils/response_target_runtime.py','utils/diag_response_target_lm.py',
 'utils/test_response_target_lm.py','utils/test_response_target_runtime.py','run_a95_20261106.sh',
 'experiment_cards/A95_response_target_lm_20261106.md','../DPS/guided_diffusion/unet.py')
class ResourcePause(RuntimeError):pass


def check(root):
 p=files.read(root/'plan.json');files.require(p['experiment']=='A95_response_target_lm','Wrong A95 plan')
 for n,h in p['sources'].items():files.require(files.digest(PROJECTS/n)==h,'Source changed: '+n)
 for n,h in p['inputs'].items():files.require(files.digest(n)==h,'Input changed: '+n)
 return p


def prepare(root,parent):
 if (root/'plan.json').exists():return check(root)
 files.require(not root.exists() or not any(root.iterdir()),'Unplanned nonempty root retained')
 pp=parent_driver.check(parent);s=files.read(parent/'summary.json')
 files.require(s['completed_runs']==s['planned_runs']==96 and s['unresolved_runs']==0,'Complete A94 required')
 cases=[dict(c) for c in pp['cases'] if c['sigma']==.05]
 files.require(len(cases)==12 and {c['image_id'] for c in cases}=={'00226','00443','00963'},'Only original12 normal-noise cases')
 bound={str(parent/n):files.digest(parent/n) for n in ('plan.json','summary.json')}
 a71=Path(pp['a71_parent'])
 for path in (a71/'plan.json',a71/'motion_kernel.npy'):bound[str(path)]=files.digest(path)
 for c in cases:
  for name in ('bundle','entry','observation','image'):bound[c[name]]=files.digest(c[name])
  c['references']={k:v for k,v in c['references'].items() if k in ('NATIVE_100','PHYS_DIRECT_100')}
  for f in c['references'].values():
   record=Path(f);bound[str(record)]=files.digest(record);row=files.read(record)
   files.require(row['status']=='complete','Reference not complete')
   path=record.parent/'final_rng.pt';files.require(files.digest(path)==row['artifacts']['final_rng.pt'],'Reference RNG changed');bound[str(path)]=files.digest(path)
  if c['task']=='inpainting':
   path=a71/'observations'/c['key']/'mask.pt';bound[str(path)]=files.digest(path)
 root.mkdir(parents=True)
 p=dict(experiment='A95_response_target_lm',parent=str(parent),a71_parent=str(a71),cases=cases,checkpoint=pp['checkpoint'],grid=pp['grid'],
  sources={**pp['sources'],**{n:files.digest(PROJECTS/n) for n in NAMES}},inputs=bound,arms=list(rules.ARMS),budget=rules.budget(),
  solver=dict(max_outer=rules.MAX_OUTER,max_trials=rules.MAX_TRIALS,cg_max=rules.CG_MAX,cg_true_rtol=rules.CG_RTOL,
   projected_gradient_rtol=rules.GRAD_RTOL,projected_gradient_atol=rules.GRAD_ATOL,initial_damping=1.,accept_ratio=.1),
  protocol=dict(scope='Original3 FFHQ x4 tasks, sigma_model=.05 only, one seed. Same A92 intervention index15 t333->309; original14 updates afterwards. No high-noise, image, window or grid expansion.',
   objective='base=PHm_s(z_s); r=Py-base; omega=(1-M*sigma^2/||r||^2)+. All real-response arms minimize .5||PHm_s(z_s+Delta)-(base+omega*r)||^2. Freeze omega in this solve. No gain penalty, dimension-dependent scalar penalty, coefficientwise target or image-GT feedback.',
   arms='LEGACY_SCALAR recomputes originalA71 at CURRENT A92 state. NEXT_PROXY_SCALAR changes base/time to cached next uncorrected clean and uses frozen-eps Hq/sqrt(alpha_s). REAL_SCALAR beta>=0 on q/||q||, same target. REAL_DIAG solves physical p=(Wq)*gamma jointly, signed full coefficients, support=(Wq)!=0.',
   attribution='LEGACY vs NEXT_PROXY tests base/time proxy convention; NEXT_PROXY vs REAL_SCALAR tests real nonlinear response; REAL_SCALAR vs REAL_DIAG tests shape freedom under the SAME target. OriginalZAPS and physical-direct single-step controls are SHA-bound A92 outputs, not newly sampled.',
   scalar_initialization='REAL_SCALAR starts NEXT_PROXY physical length. REAL_DIAG starts the real scalar solution as a feasible point only; no norm bound/attraction penalty or GT choice. Both costs retained separately; diagonal deployment would also pay for scalar initialization.',
   geometry='Optimization coordinates are physical length for scalar, orthogonal db4 coefficient p for diagonal. Exact-zero native coefficients remain0. All nonzero coefficients admit signed gains; no subband sharing or gamma clipping. Gamma ratios are diagnostic only, never computed to generate the update.',
   solver='Matrix-free inexact Gauss-Newton/LM. True AD JVP and original-checkpoint VJP, full cross terms J^T J. Damping applies to numerical step, not final objective; trust agreement adapts it. Same projected stationarity rule, not fixed2 updates. Outer/trial budgets/stall remain explicitly unqualified, with last accepted iterate retained for diagnostic reconstruction; no policy fallback or retries.',
   derivative='Original custom checkpoint lacks forward AD. Only within JVP temporarily call the exact same checkpointed functions directly, restoring globals afterwards. Native values/VJP/sampling retain original implementation. Actual model primal parity, JVP/VJP adjoint identity and memory probe are REQUIRED; no finite-difference response probes.',
   replay='One shared16F original replay per case recovers CURRENT clean for originalA71 formula, verifies q/uncond relativeL2<=1e-6 and exact CPU/CUDA RNG against A92. Old scalar gammas from different trajectories are never transplanted. Prefix/inputs/source/model hashes retained.',
   noise='Same A71 noise-energy heuristic, observedRGB only for masks. Not calibrated SURE/unbiased full-image risk; no new uncertainty estimator in this experiment. Current and next prediction baselines deliberately separated by NEXT_PROXY.',
   limits='Small reused panel, fixed state only. Full signed coefficient diagonal is nearly an arbitrary vector representation; a matrix itself is not a novel method. Local objective/stationarity does not guarantee best PSNR/global minimum or posterior sampling.',
   qualification='Primary REAL_DIAG vs REAL_SCALAR needs both solvers numerically qualified, >=+.1dB meanPSNR,>=2of3 wins,SSIM nondecrease,LPIPS nonincrease. Also compare legacy scalar, native, direct, target gain and offline signal/noise gain. All finite budget-limited images retained but never called converged or used to certify matrix superiority.',
   accounting='Count every JVP/VJP NN forward, rejected trust trial, true-CG-residual product, shared16F replay and14F continuation. Warmup actual checkpoint2F/1JVP/1VJP separate. Acquisition of parent caches was previously paid. Numerical caps are recorded, no timing-based ranking or complete300-clock/speedup claim.',
   resources='AllocatedGPU>=8GiB, actual forwardAD/VJP peak+1GiB before work. No process termination/precision or resolution changes. Same final RNG as A92; no optimizer or NN weight update; data-GT read only after final action. No automatic continuation to full30.'))
 files.write(root/'plan.json',p);shutil.copyfile(PROJECTS/'experiment_cards/A95_response_target_lm_20261106.md',root/'experiment_card.md')
 print('A95 FROZEN:12 cases x4 new arms; sigma=.05; one fixed correction; max newF=',p['budget']['max_new_F'],flush=True)
 return p


def execute(root,p):
 import torch
 from modules.main_single import load_diffusion_model,load_image_as_tensor
 from modules.dataset_loader import tensor_to_image
 from utils import response_target_runtime as runtime,paper_clock_runtime as clock,historical_ffhq_runtime as arithmetic
 from utils.a67_model_freeze import frozen_network
 from utils.diag_shrink_multitask import make_operator
 from utils.diag_noise_feasible_zaps import metrics
 from utils.terminal_mask_counterfactual import region_errors
 load=lambda f:torch.load(f,map_location='cpu',weights_only=True)
 files.require(files.digest(p['checkpoint']['path'])==p['checkpoint']['sha256'],'Wrong checkpoint');torch.set_num_threads(2)
 start=time.perf_counter();model=load_diffusion_model('ffhq','cuda:0');model.model.eval()
 weights=load(p['checkpoint']['path']);model.model.load_state_dict(weights,strict=True);del weights
 torch.cuda.synchronize();folder=root/'loads';folder.mkdir(exist_ok=True);path=folder/(str(len(list(folder.glob('*.json')))).zfill(3)+'.json')
 probe=dict(model_load_seconds=time.perf_counter()-start,probe_F=0,probe_JVP=0,probe_VJP=0,
  environment=dict(python=sys.version,torch=str(torch.__version__),cuda=torch.version.cuda,device=torch.cuda.get_device_name()))
 with frozen_network(model.model),arithmetic.precision('LEGACY_IEEE'):
  files.write(path,probe);torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  try:probe['gate']=runtime.checkpoint_probe(model,p['grid'][16],probe)
  finally:
   torch.cuda.synchronize();probe.update(probe_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated());files.write(path,probe)
  peak=probe['peak_allocated_bytes']
  def admission():
   if int(torch.cuda.mem_get_info()[0])+torch.cuda.memory_reserved()-torch.cuda.memory_allocated()<peak+files.GIB:raise ResourcePause('AD peak+1GiB lacks margin before next unstarted item')
  for c in p['cases']:
   case_root=root/'cases'/c['key'];shared=case_root/'prefix.json'
   if not shared.exists():
    admission();case_root.mkdir(parents=True,exist_ok=True);record=dict(status='running',case=c);files.write(shared,record);count={}
    try:
     with clock.counters(model.model) as count:
      bundle=load(c['bundle']);entry=load(c['entry']);y=load(c['observation'])['y'].to(model.device);op=make_operator(Path(p['a71_parent']),c,model.device,count)
      clean,gate=runtime.shared_prefix(model,op,y,entry,bundle,c['task'],count)
     files.require(count['forwards']==16 and count['calls']==p['grid'][:16] and count['input_vjp_attempts']==0,'Shared prefix budget mismatch')
     torch.save(clean,case_root/'current_clean.pt');record.update(status='complete',counts=count,gate=gate,artifacts={'current_clean.pt':files.digest(case_root/'current_clean.pt')});files.write(shared,record)
    except BaseException as error:
     record.update(status='failed',counts=count,error=repr(error),traceback=traceback.format_exc());files.write(shared,record);raise
   else:
    prior=files.read(shared);files.require(prior['status']=='complete','Prior prefix failure retained')
    files.require(files.digest(case_root/'current_clean.pt')==prior['artifacts']['current_clean.pt'],'New shared clean changed')
   for arm in rules.ARMS:
    folder=case_root/arm;record=folder/'run.json'
    if record.exists():
     prior=files.read(record);files.require(prior['status']=='complete','Prior failed/interrupted arm retained')
     for name,sha in prior['artifacts'].items():files.require(files.digest(folder/name)==sha,'Output changed')
     continue
    admission();folder.mkdir(parents=True);row=dict(status='running',case=c,arm=arm,counts={});files.write(record,row);count={}
    try:
     print('A95 START:',c['key'],arm,flush=True)
     def progress(snapshot):
      files.write(record,{**row,'counts':snapshot})
      print('A95 PROGRESS:',c['key'],arm,'F=',snapshot['forwards'],'JVP=',snapshot.get('input_jvp_attempts',0),'VJP=',snapshot['input_vjp_attempts'],flush=True)
     with clock.counters(model.model,tick=progress) as count:
      def operation():
       bundle=load(c['bundle']);entry=load(c['entry']);y=load(c['observation'])['y'].to(model.device);op=make_operator(Path(p['a71_parent']),c,model.device,count)
       seed=files.read(case_root/'REAL_SCALAR/run.json')['detail']['solver'] if arm=='REAL_DIAG' else None
       return runtime.run(model,op,y,entry,bundle,load(case_root/'current_clean.pt'),c['task'],arm,c['sigma'],count,seed)
      (output,first,rng,correction,detail),seconds=clock.clocks(torch.cuda.synchronize,operation)
     sf=sum(count.get(k,0) for k in ('solver_value_attempts','solver_vjp_forward_attempts','solver_jvp_forward_attempts'))
     files.require(sf<=rules.budget()['max_solver_forward'] and count['forwards']==count['forward_attempts']==sf+14,'NN count mismatch')
     files.require(count['calls']==[p['grid'][16]]*sf+p['grid'][16:],'Grid mismatch')
     files.require(count['input_vjp_attempts']==count.get('input_vjp_completed',0)<=rules.budget()['max_solver_vjp'] and count.get('input_jvp_attempts',0)==count.get('input_jvp_completed',0)<=rules.budget()['max_solver_jvp'],'Derivative count mismatch')
     files.require(count['optimizer_backward_attempts']==0,'Unexpected optimizer backward')
     reference=load(Path(c['references']['NATIVE_100']).parent/'final_rng.pt');files.require(runtime.rng_equal(rng,reference),'Future RNG mismatch')
     row.update(counts=count,suffix_seconds=seconds,detail=detail);files.write(record,row)
     start=time.perf_counter();gt=load_image_as_tensor(c['image']).cpu();m=metrics(output,gt)
     bundle=load(c['bundle']);op=make_operator(Path(p['a71_parent']),c,model.device);y=load(c['observation'])['y'].to(model.device)
     audit=runtime.old.offline_readout(op,y,gt.to(model.device),bundle['reference_clean'].to(model.device),first.to(model.device),correction.to(model.device),bundle['physical'].to(model.device))
     if c['task']=='inpainting':
      mask=load(Path(p['a71_parent'])/'observations'/c['key']/'mask.pt');m.update(region_errors(output,gt,mask));audit['next_clean_region_errors']=region_errors(first,gt,mask)
     row['offline_seconds']=time.perf_counter()-start
     for name,value in (('output.pt',output),('correction.pt',correction),('final_rng.pt',rng)):torch.save(value,folder/name)
     tensor_to_image(output.squeeze(0),denormalize=True).save(folder/'output.png');files.write(folder/'solver.json',detail)
     row.update(status='complete',metrics=m,audit=audit,artifacts={n:files.digest(folder/n) for n in ('output.pt','correction.pt','final_rng.pt','output.png','solver.json')});files.write(record,row)
     print('A95 COMPLETE:',c['key'],arm,detail['solver']['status'],'F=',count['forwards'],'JVP=',count.get('input_jvp_attempts',0),'VJP=',count['input_vjp_attempts'],flush=True)
    except BaseException as error:
     row.update(status='failed',counts=count,error=repr(error),traceback=traceback.format_exc());files.write(record,row);raise


def summarize(root):
 p=check(root);rows=[files.read(f) for f in (root/'cases').glob('*/*/run.json')];done=[r for r in rows if r['status']=='complete'];groups=[];comparisons=[]
 for task in ('super_resolution','gaussian_deblur','inpainting','motion_deblur'):
  for arm in rules.ARMS:
   rs=[r for r in done if r['case']['task']==task and r['arm']==arm]
   if not rs:continue
   groups.append(dict(task=task,arm=arm,images=len(rs),quality={k:statistics.mean(r['metrics'][k] for r in rs) for k in ('psnr','ssim','lpips')},
    numerically_converged=sum(r['detail']['solver'].get('converged') is True for r in rs),solver_qualified_images=sum(numeric_qualified(r) for r in rs),mean_suffix_seconds=statistics.mean(r['suffix_seconds'] for r in rs),
    mean_target_objective=statistics.mean(r['detail']['actual_target_objective'] for r in rs),mean_omega=statistics.mean(r['detail']['target']['omega'] for r in rs)))
   for reference in ('NATIVE_100','PHYS_DIRECT_100')+tuple(a for a in rules.ARMS if a!=arm):
    pairs=[]
    for r in rs:
     if reference in c_refs(r):other=files.read(r['case']['references'][reference])
     else:
      f=root/'cases'/r['case']['key']/reference/'run.json'
      if not f.exists() or files.read(f)['status']!='complete':continue
      other=files.read(f)
     pairs.append((other,r))
    if not pairs:continue
    delta={k:statistics.mean(b['metrics'][k]-a['metrics'][k] for a,b in pairs) for k in ('psnr','ssim','lpips')}
    qualified=all(numeric_qualified(a) and numeric_qualified(b) for a,b in pairs);wins=sum(b['metrics']['psnr']>a['metrics']['psnr'] for a,b in pairs)
    comparisons.append(dict(task=task,arm=arm,reference=reference,cases=len(pairs),delta=delta,images_PSNR_positive=wins,numerically_qualified=qualified,
     positive_development_candidate=len(pairs)==3 and qualified and delta['psnr']>=.1 and wins>=2 and delta['ssim']>=0 and delta['lpips']<=0))
 prefixes=[files.read(f) for f in (root/'cases').glob('*/prefix.json')];allrows=rows+prefixes
 cost={k:sum(r.get('counts',{}).get(k,0) for r in allrows) for k in ('forward_attempts','forwards','input_vjp_attempts','input_jvp_attempts','optimizer_backward_attempts','base_H_actions','base_adjoint_actions','guidance_mapping_actions','module_dwt_analysis','module_dwt_synthesis')}
 s=dict(experiment=p['experiment'],classification='NORMAL_NOISE_FIXED_STATE_RESPONSE_TARGET_DIAGNOSIS',completed_runs=len(done),planned_runs=48,completed_prefixes=sum(r['status']=='complete' for r in prefixes),groups=groups,comparisons=comparisons,actual_new_cost=cost,
  cost_is_lower_bound=any(r['status']!='complete' for r in allrows),loads=[files.read(f) for f in (root/'loads').glob('*.json')],protocol=p['protocol'])
 files.write(root/'summary.json',s)
 if done:
  values=[dict(task=r['case']['task'],image_id=r['case']['image_id'],arm=r['arm'],**r['metrics'],solver_status=r['detail']['solver']['status'],suffix_seconds=r['suffix_seconds']) for r in done];keys=list(dict.fromkeys(k for r in values for k in r))
  with (root/'comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(values)
 print('A95 STATUS:',dict(completed=len(done),planned=48,cost=cost),flush=True)
 for row in comparisons:
  if row['arm']=='REAL_DIAG' and row['reference'] in ('REAL_SCALAR','NATIVE_100'):
   print('A95 DIAG_MINUS:',row['task'],row['reference'],{k:round(v,5) for k,v in row['delta'].items()},'numerically_qualified=',row['numerically_qualified'],flush=True)
 print('Full summary:',root/'summary.json',flush=True);return s


def c_refs(row):return row['case'].get('references',{})
def numeric_qualified(row):
 solver=row.get('detail',{}).get('solver',{})
 return not solver or solver.get('converged') is True or solver.get('status')=='CLOSED_FORM_PROXY'


def archive(root):
 from utils import archive_refinement_confirmation as arch
 arch.BRANCH=BRANCH;bundle,sha=arch.export_and_commit({'A95':root},projects=PROJECTS)
 print('Archive:',bundle,'Results commit:',sha,'Branch:',BRANCH,flush=True)
 subprocess.run(['git','push','origin',BRANCH],cwd=PROJECTS.parent,check=True);print('RESULTS_PUSH_PASS: A95 uploaded.',flush=True)


def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',default=ROOT);a.add_argument('--a94-root',default=PARENT);a.add_argument('--gpus',default='0,1')
 for flag in ('execute','status','archive-only'):a.add_argument('--'+flag,action='store_true')
 args=a.parse_args();root=Path(args.root).resolve();parent=Path(args.a94_root).resolve();state=root.parent/(root.name+'_session');state.mkdir(parents=True,exist_ok=True);record=state/'session.json'
 if args.status:print(dict(active=busy(state/'worker.lock'),session=files.read(record) if record.exists() else None));return
 if args.archive_only:files.require(not busy(state/'worker.lock'),'Worker active');summarize(root);archive(root);return
 if args.execute:
  with exclusive(state/'worker.lock'):
   status=dict(status='running',pid=os.getpid(),root=str(root));files.write(record,status)
   try:
    p=prepare(root,parent)
    from utils.diag_paper_clock_alignment import gpu_inventory
    choices=files.available_gpus(gpu_inventory(),args.gpus,os.environ.get('CUDA_VISIBLE_DEVICES'));choices=sorted([r for r in choices if r['free_bytes']>=8*files.GIB],key=lambda r:-r['free_bytes'])
    if not choices:raise ResourcePause('No allocated GPU>=8GiB')
    if (root/'allocation.json').exists():
     allocated=files.read(root/'allocation.json')
     if not any(r['uuid']==allocated['uuid'] for r in choices):raise ResourcePause('Pinned GPU lacks headroom')
    else:allocated=choices[0];files.write(root/'allocation.json',allocated)
    os.environ['CUDA_VISIBLE_DEVICES']=allocated['uuid']
    command=[sys.executable,'-m','unittest','-v','utils.test_response_target_lm','utils.test_response_target_runtime','utils.test_current_score_checkpoint_r2']
    result=subprocess.run(command,cwd=PROJECTS,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);print(result.stdout,end='',flush=True)
    files.write(root/'preflight_tests.json',dict(command=command,returncode=result.returncode,output=result.stdout));result.check_returncode()
    execute(root,p);status['status']='complete'
   except ResourcePause as error:status.update(status='resource_limited',reason=str(error));print('A95 RESOURCE PAUSE:',error,flush=True)
   except BaseException as error:status.update(status='failed',error=repr(error),traceback=traceback.format_exc());print('A95 FAILED:',repr(error),flush=True)
   finally:
    files.write(record,status)
    if (root/'plan.json').exists():
     shutil.copyfile(record,root/'session.json')
     try:summarize(root);archive(root)
     except Exception as error:print('A95 archive incomplete; --archive-only:',repr(error),flush=True)
  return
 with exclusive(state/'client.lock'):
  child=None
  if not busy(state/'worker.lock'):
   if record.exists():
    prior=files.read(record);files.require(prior['status']!='failed','Prior failure retained; no automatic retry')
    if prior['status']=='complete':print('A95 already complete');return
   command=[sys.executable,'-u',str(Path(__file__)),'--execute','--root',str(root),'--a94-root',str(parent),'--gpus',args.gpus]
   with (state/'worker.log').open('ab',buffering=0) as log:child=subprocess.Popen(command,cwd=PROJECTS,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  offset=0
  while True:
   if (state/'worker.log').exists():
    with (state/'worker.log').open(errors='replace') as f:f.seek(offset);text=f.read();offset=f.tell()
    if text:print(text,end='',flush=True)
   if not busy(state/'worker.lock') and (child is None or child.poll() is not None):break
   time.sleep(1)
  if record.exists():
   status=files.read(record);print('A95 SESSION:',status,flush=True)
   if status['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
