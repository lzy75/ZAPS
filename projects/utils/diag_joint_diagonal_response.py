"""A94 independent full-coefficient signed joint nonlinear response experiment."""
import os
os.environ['MKL_THREADING_LAYER']='GNU';os.environ.pop('MKL_SERVICE_FORCE_INTEL',None)
import argparse,csv,json,subprocess,sys,time,traceback,statistics,shutil
from pathlib import Path
PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import historical_ffhq_gain as files,joint_diagonal_response as rules
from utils import diag_overall_direction_audit as a92
from utils.a54_session import exclusive,busy
ROOT='/data0/lzy/ZAPS-results/diag_joint_diagonal/A94_20261106'
PARENT=a92.ROOT
BRANCH='results/zaps-joint-diagonal-response-20261106'
NAMES=('utils/joint_diagonal_response.py','utils/diag_joint_diagonal_response.py',
 'utils/test_joint_diagonal_response.py','utils/test_joint_diagonal_response_runtime.py',
 'run_a94_20261106.sh','experiment_cards/A94_joint_diagonal_response_20261106.md')
class ResourcePause(RuntimeError):pass


def check(root):
 p=files.read(root/'plan.json');files.require(p['experiment']=='A94_joint_diagonal','Wrong A94 plan')
 for n,h in p['sources'].items():files.require(files.digest(PROJECTS/n)==h,'Source changed: '+n)
 for n,h in p['inputs'].items():files.require(files.digest(n)==h,'Input changed: '+n)
 return p


def prepare(root,parent):
 if (root/'plan.json').exists():return check(root)
 files.require(not root.exists() or not any(root.iterdir()),'Unplanned nonempty root retained')
 pp=a92.check(parent);s=files.read(parent/'summary.json')
 files.require(s['completed_runs']==s['planned_runs']==216 and s['unresolved_runs']==0,'Complete A92 required')
 bound=dict(pp['inputs']);bound.update({str(parent/n):files.digest(parent/n) for n in ('plan.json','summary.json')});cases=[]
 for c0 in pp['cases']:
  c=dict(c0);folder=parent/'cases'/c['key'];record=folder/'prefix.json';r=files.read(record)
  files.require(r['status']=='complete' and r['plan_sha256']==files.digest(parent/'plan.json'),'A92 prefix identity failed')
  bound[str(record)]=files.digest(record);c['bundle']=str(folder/'state.pt')
  files.require(files.digest(c['bundle'])==r['artifacts']['state.pt'],'Cached gradient/state changed');bound[c['bundle']]=r['artifacts']['state.pt']
  c['references']={}
  for arm in ('NATIVE_100','DIAG_ALIGN_100','PHYS_DIRECT_100'):
   f=folder/arm/'run.json';row=files.read(f);files.require(row['status']=='complete','Incomplete A92 control')
   bound[str(f)]=files.digest(f);c['references'][arm]=str(f)
   for name in ('output.pt','final_rng.pt'):
    file=f.parent/name;files.require(files.digest(file)==row['artifacts'][name],'A92 control tensor changed');bound[str(file)]=row['artifacts'][name]
  cases.append(c)
 files.require(len(cases)==24,'Fixed24 cases only');root.mkdir(parents=True)
 p=dict(experiment='A94_joint_diagonal',parent=str(parent),a71_parent=pp['parent'],cases=cases,checkpoint=pp['checkpoint'],grid=pp['intervention']['fixed_grid'],
   sources={**pp['sources'],**{n:files.digest(PROJECTS/n) for n in NAMES}},inputs=bound,arms=list(rules.ARMS),
   budget=dict(arms=96,max_new_diagnostic_forwards=2592,max_new_input_vjp=96,optimizer_backward=0,
      per_arm_max_solver_forward=13,per_arm_continuation_forward=14,reused_initial_gradient=24,warmup_F_VJP_per_load=1),
   protocol=dict(scope='Same OLD3 FFHQ x4tasks x sigma.05/.10. Single intervention at A92 index15 only. A92 prefix/initial true gradient reused; native14-point continuation. No new image/seed/grid/frequency or window expansion.',
    objective='gamma=all db4/level3 coefficients, Delta=W^T[(Wq)*gamma]. Min .5*||observed(y-H(clamped rawD_s(z+Delta)))||^2/sigma_y^2 + .5*precision*sum(gamma^2). Real next network response, not target-gradient matching, g_j/q_j division, diagonal Gram approximation or independent coefficient solve.',
    arms='SIGNED_JOINT precision1; SCALAR_TIED constrains all gains identical under the SAME full-field objective/prior; NONNEG_JOINT precision1; SIGNED_DATA_ONLY precision0. precision1 means gamma~N(0,1) working prior, an explicit model choice, not parameter-free/calibrated posterior. Tied scalar isolates coefficient freedom, nonnegative isolates signs, zero-prior isolates regularization.',
    solver='All coefficients updated jointly by inputVJP through H/next denoiser and exact DWT chain. gamma starts0, not1/scalar reference. Two accepted proximal-gradient updates; at most6 one-sided actual-objective backtracks/update. First eta=1/max(1,maxabs(data gradient)); halve on failed majorization, reuse accepted eta. No GT tuning, no0..2 cap, no native/scalar correction norm budget.',
    stability='Backtracking requires smooth-data majorization and nonincrease of complete objective within64 float32 eps relative scalar tolerance. The sigma likelihood is Gaussian fidelity, NOT subtraction of noise energy/SURE/full-image unbiased risk. Fixed2 accepted updates means budget-limited diagnostic, not optimal/converged.',
    failure='No accepted step within budget => unresolved with attempted costs and solver trace, no fallback image. Nonfinite/RNG/source/count mismatch => failed stop, no automatic retry. Exact zero proximal move => stationary at working precision, explicitly reported.',
    accounting='At most12 trialF+1 extra gradientF/VJP+14 continuationF=27F/1VJP per arm. 96arms =>2592F/96VJP max, warmup1F/1VJP per load separate. The first current gradient is SHA-bound A92 and not a new call. Internal checkpoint recompute included in walltime, not additional top-level denoiser calls.',
    measurement='Original model[-1,1] sigma and original H. Mask stats/likelihood observedRGB only. Other given-y, kernels and original q mapping unchanged; no pretrained model updates or new zeta/D learning.',
    pairing='Same initial uncond/current innovation and A92 future CPU/CUDA RNG. Solver never draws randomness. Continuation grid exactly original14 calls; solver calls all at next_t309. Bounds/hash/real Torch checkpoint gates mandatory. No large tensor downloads/Mac GPU claims.',
    decisions='Per task/noise signed joint must compare to native, A92 grouped diagonal and A92 physical direct at same intervention. Positive: meanPSNR>=+.1dB,>=2/3 wins,SSIM nondecrease,LPIPS nonincrease. Report objective reduction/negative fraction/gain tails/actual correction length/noise-fitting decomposition and final regional risks. No arm chosen by GT.',
    historical='A71 scalar summary read from bound completed summary (fixes missing per-run metrics issue); comparison means descriptive only, not same control scope or matched full runtime.',
    timing='Solver+14F suffix includes cached state/y/entry loads/operator construction/outputCPU sync; solver time separated inside. Metrics/noise-decomposition/image writes/model loading/warmup separately. Not complete300 reconstruction timing or acceleration.',
    limits='Original3 reused development images, one seed. A high-dimensional signed diagonal can represent almost arbitrary vectors, so success does not itself establish novelty or calibrated risk. New nonlinear objective and prior require this experiment; no guarantee all tasks positive.'))
 files.write(root/'plan.json',p)
 shutil.copyfile(PROJECTS/'experiment_cards/A94_joint_diagonal_response_20261106.md',root/'experiment_card.md')
 return p


def execute(root,p):
 import torch
 from modules.main_single import load_diffusion_model,load_image_as_tensor
 from modules.dataset_loader import tensor_to_image
 from utils import paper_clock_runtime as clock,historical_ffhq_runtime as arithmetic
 from utils.a67_model_freeze import frozen_network
 from utils.current_score_runtime import raw_mean
 from utils.current_score_runtime_r2 import input_vjp_parameters
 from utils.diag_shrink_multitask import make_operator
 from utils.diag_noise_feasible_zaps import metrics
 from utils.terminal_mask_counterfactual import region_errors
 load=lambda f:torch.load(f,map_location='cpu',weights_only=True)
 files.require(files.digest(p['checkpoint']['path'])==p['checkpoint']['sha256'],'Wrong checkpoint');torch.set_num_threads(2)
 start=time.perf_counter();model=load_diffusion_model('ffhq','cuda:0');model.model.eval()
 weights=load(p['checkpoint']['path']);model.model.load_state_dict(weights,strict=True);del weights
 torch.cuda.synchronize();loads=root/'loads';loads.mkdir(exist_ok=True)
 lf=loads/(str(len(list(loads.glob('*.json')))).zfill(3)+'.json');lr=dict(model_load_seconds=time.perf_counter()-start,warmup_forward_attempts=0,warmup_input_vjp_attempts=0,
  environment=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),device=torch.cuda.get_device_name(),
   allocation=files.read(root/'allocation.json')))
 with frozen_network(model.model),arithmetic.precision('LEGACY_IEEE'):
  torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  with input_vjp_parameters(model.model),torch.enable_grad():
   lr['warmup_forward_attempts']=1;files.write(lf,lr)
   inp,raw=raw_mean(model,torch.zeros(1,3,256,256,device=model.device),p['grid'][16],graph=True)
   lr['warmup_input_vjp_attempts']=1;files.write(lf,lr)
   vjp=torch.autograd.grad(raw.square().mean(),inp)[0];files.require(bool(torch.isfinite(vjp).all()),'Checkpoint probe failed')
  del inp,raw,vjp;torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()
  lr.update(warmup_seconds=time.perf_counter()-start,peak_allocated_bytes=peak);files.write(lf,lr)
  for c in p['cases']:
   for arm in rules.ARMS:
    folder=root/'cases'/c['key']/arm;record=folder/'run.json'
    if record.exists():
     saved=files.read(record);files.require(saved['status'] in ('complete','unresolved'),'Failed/interrupted arm retained')
     for n,h in saved.get('artifacts',{}).items():files.require(files.digest(folder/n)==h,'Artifact changed')
     continue
    free=int(torch.cuda.mem_get_info()[0]);reusable=torch.cuda.memory_reserved()-torch.cuda.memory_allocated()
    if free+reusable<peak+files.GIB:raise ResourcePause('VJP peak+1GiB lacks margin before attempt')
    folder.mkdir(parents=True);row=dict(status='running',case=c,arm=arm,counts={});files.write(record,row);count={}
    try:
     with clock.counters(model.model) as count:
      def operation():
       bundle=load(c['bundle']);entry=load(c['entry']);y=load(c['observation'])['y'].to(model.device)
       op=make_operator(Path(p['a71_parent']),c,model.device,count)
       return rules.run(model,op,y,entry,bundle,c['task'],arm,c['sigma'],count)
      (output,first,rng,detail,gamma,correction),seconds=clock.clocks(torch.cuda.synchronize,operation)
     sf=count.get('solver_forward_attempts',0)
     files.require(sf<=13 and count['forwards']==count['forward_attempts']==sf+14,'NN budget mismatch')
     files.require(count['calls']==[p['grid'][16]]*sf+p['grid'][16:],'Solver/continuation grid mismatch')
     files.require(count['input_vjp_attempts']==count.get('input_vjp_completed',0)<=1 and count['optimizer_backward_attempts']==0,'Backward count mismatch')
     reference=load(Path(c['references']['NATIVE_100']).parent/'final_rng.pt')
     files.require(torch.equal(rng['cpu_rng'],reference['cpu_rng']) and len(rng['cuda_rng'])==len(reference['cuda_rng']) and all(torch.equal(a,b) for a,b in zip(rng['cuda_rng'],reference['cuda_rng'])),'Future RNG mismatch')
     row.update(counts=count,suffix_seconds=seconds,solver=detail['solver'],solver_seconds=detail['solver_seconds'],continuation_seconds=detail['continuation_seconds']);files.write(record,row)
     start=time.perf_counter();gt=load_image_as_tensor(c['image']).cpu();m=metrics(output,gt)
     bundle=load(c['bundle']);op=make_operator(Path(p['a71_parent']),c,model.device);y=load(c['observation'])['y'].to(model.device)
     audit=rules.old.offline_readout(op,y,gt.to(model.device),bundle['reference_clean'].to(model.device),first.to(model.device),correction.to(model.device),bundle['physical'].to(model.device))
     actual_data=.5*audit['residual_energy_after']/(c['sigma']**2);predicted_data=detail['solver']['final_data']
     files.require(abs(actual_data-predicted_data)<=64*torch.finfo(torch.float32).eps*max(1.,abs(actual_data),abs(predicted_data)),
       'Actual first continuation clean prediction differs from optimized objective')
     audit['solver_vs_continuation_data_error']=abs(actual_data-predicted_data)
     if c['task']=='inpainting':
      mask=load(Path(p['a71_parent'])/'observations'/c['key']/'mask.pt');m.update(region_errors(output,gt,mask));audit['next_clean_region_errors']=region_errors(first,gt,mask)
     row['offline_seconds']=time.perf_counter()-start;start=time.perf_counter()
     torch.save(output,folder/'output.pt');torch.save(gamma,folder/'gamma.pt');torch.save(correction,folder/'correction.pt');torch.save(rng,folder/'final_rng.pt')
     tensor_to_image(output.squeeze(0),denormalize=True).save(folder/'output.png');files.write(folder/'solver.json',detail);files.write(folder/'audit.json',audit)
     row.update(status='complete',metrics=m,audit=audit,artifact_seconds=time.perf_counter()-start,
       artifacts={n:files.digest(folder/n) for n in ('output.pt','gamma.pt','correction.pt','final_rng.pt','output.png','solver.json','audit.json')})
     files.write(record,row);print('A94 COMPLETE:',c['key'],arm,'F=',count['forwards'],'VJP=',count['input_vjp_attempts'],'PSNR=',m['psnr'],flush=True)
    except rules.UnresolvedSolve as error:
     row.update(status='unresolved',reason=str(error),trace=error.trace,counts=count);files.write(record,row);print('A94 UNRESOLVED:',c['key'],arm,str(error),flush=True)
    except BaseException as error:
     row.update(status='failed',error=repr(error),traceback=traceback.format_exc(),counts=count);files.write(record,row);raise


def summarize(root):
 p=check(root);rows=[files.read(f) for f in (root/'cases').glob('*/*/run.json')];done=[r for r in rows if r['status']=='complete'];groups=[];comparisons=[]
 for task in ('super_resolution','gaussian_deblur','inpainting','motion_deblur'):
  for sigma in (.05,.10):
   for arm in rules.ARMS:
    rs=[r for r in done if r['arm']==arm and r['case']['task']==task and r['case']['sigma']==sigma]
    if not rs:continue
    groups.append(dict(task=task,sigma=sigma,arm=arm,images=len(rs),quality={k:statistics.mean(r['metrics'][k] for r in rs) for k in ('psnr','ssim','lpips')},mean_suffix_seconds=statistics.mean(r['suffix_seconds'] for r in rs),mean_solver_seconds=statistics.mean(r['solver_seconds'] for r in rs),mean_negative_fraction=statistics.mean(r['solver']['negative_fraction'] for r in rs),mean_correction_norm_ratio=statistics.mean(r['solver']['correction_norm']/max(r['solver']['native_norm'],1e-30) for r in rs),mean_clean_gain=statistics.mean(r['audit']['clean_measurement_gain'] for r in rs),mean_fitted_noise_gain=statistics.mean(r['audit']['fitted_noise_gain'] for r in rs)))
    for reference in ('NATIVE_100','DIAG_ALIGN_100','PHYS_DIRECT_100'):
     pairs=[(files.read(r['case']['references'][reference]),r) for r in rs]
     delta={k:statistics.mean(b['metrics'][k]-a['metrics'][k] for a,b in pairs) for k in ('psnr','ssim','lpips')};wins=sum(b['metrics']['psnr']>a['metrics']['psnr'] for a,b in pairs)
     comparisons.append(dict(task=task,sigma=sigma,arm=arm,reference=reference,cases=len(rs),delta=delta,images_PSNR_positive=wins,positive_development_candidate=len(rs)==3 and delta['psnr']>=.1 and wins>=2 and delta['ssim']>=0 and delta['lpips']<=0))
    if arm=='SIGNED_JOINT':
     for reference in ('SCALAR_TIED','NONNEG_JOINT','SIGNED_DATA_ONLY'):
      other={r['case']['key']:r for r in done if r['arm']==reference};pairs=[(other[r['case']['key']],r) for r in rs if r['case']['key'] in other]
      if pairs:
       delta={k:statistics.mean(b['metrics'][k]-a['metrics'][k] for a,b in pairs) for k in ('psnr','ssim','lpips')};wins=sum(b['metrics']['psnr']>a['metrics']['psnr'] for a,b in pairs)
       comparisons.append(dict(task=task,sigma=sigma,arm=arm,reference=reference,cases=len(pairs),delta=delta,images_PSNR_positive=wins,
        positive_development_candidate=len(pairs)==3 and delta['psnr']>=.1 and wins>=2 and delta['ssim']>=0 and delta['lpips']<=0))
 historical=files.read(Path(p['a71_parent'])/'summary.json')['groups']
 scalar=[r for r in historical if r['arm']=='SHRINK' and r['sigma'] in (.05,.10)]
 s=dict(experiment=p['experiment'],classification='NONLINEAR_JOINT_DIAGONAL_SINGLE_INTERVENTION_DIAGNOSIS',completed_runs=len(done),planned_runs=96,
   unresolved_runs=sum(r['status']=='unresolved' for r in rows),groups=groups,comparisons=comparisons,historical_A71_scalar=scalar,
   new_diagnostic_forward_attempts=sum(r.get('counts',{}).get('forward_attempts',0) for r in rows),new_input_vjp_attempts=sum(r.get('counts',{}).get('input_vjp_attempts',0) for r in rows),optimizer_backward_attempts=sum(r.get('counts',{}).get('optimizer_backward_attempts',0) for r in rows),
   operator_actions={k:sum(r.get('counts',{}).get(k,0) for r in rows) for k in ('base_H_actions','base_adjoint_actions','solver_dwt_analysis','solver_dwt_synthesis')},
   cost_is_lower_bound=any(r['status'] not in ('complete','unresolved') for r in rows),loads=[files.read(f) for f in (root/'loads').glob('*.json')],protocol=p['protocol'])
 files.write(root/'summary.json',s)
 if done:
  values=[dict(task=r['case']['task'],sigma=r['case']['sigma'],image_id=r['case']['image_id'],arm=r['arm'],**r['metrics'],suffix_seconds=r['suffix_seconds'],solver_seconds=r['solver_seconds']) for r in done];keys=list(dict.fromkeys(k for r in values for k in r))
  with (root/'comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(values)
 print('A94 STATUS:',{k:s[k] for k in ('completed_runs','planned_runs','unresolved_runs','new_diagnostic_forward_attempts','new_input_vjp_attempts')},flush=True)
 for r in comparisons:
  if r['arm']=='SIGNED_JOINT' and r['reference']=='PHYS_DIRECT_100':print('A94 JOINT_MINUS_DIRECT:',r['task'],r['sigma'],{k:round(v,5) for k,v in r['delta'].items()},'wins',r['images_PSNR_positive'],flush=True)
 print('Full summary:',root/'summary.json',flush=True);return s


def archive(root):
 from utils import archive_refinement_confirmation as arch
 arch.BRANCH=BRANCH;bundle,sha=arch.export_and_commit({'A94':root},projects=PROJECTS)
 print('Archive:',bundle,'Results commit:',sha,'Branch:',BRANCH,flush=True)
 subprocess.run(['git','push','origin',BRANCH],cwd=PROJECTS.parent,check=True);print('RESULTS_PUSH_PASS: A94 uploaded.',flush=True)


def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',default=ROOT);a.add_argument('--a92-root',default=PARENT);a.add_argument('--gpus',default='0,1')
 for flag in ('execute','status','archive-only'):a.add_argument('--'+flag,action='store_true')
 args=a.parse_args();root=Path(args.root).resolve();parent=Path(args.a92_root).resolve();state=root.parent/(root.name+'_session');state.mkdir(parents=True,exist_ok=True);record=state/'session.json'
 if args.status:print(dict(active=busy(state/'worker.lock'),session=files.read(record) if record.exists() else None));return
 if args.archive_only:files.require(not busy(state/'worker.lock'),'Worker active');summarize(root);archive(root);return
 if args.execute:
  with exclusive(state/'worker.lock'):
   status=dict(status='running',pid=os.getpid(),root=str(root));files.write(record,status)
   try:
    p=prepare(root,parent)
    from utils.diag_paper_clock_alignment import gpu_inventory
    choices=files.available_gpus(gpu_inventory(),args.gpus,os.environ.get('CUDA_VISIBLE_DEVICES'));choices=sorted([r for r in choices if r['free_bytes']>=6*files.GIB],key=lambda r:-r['free_bytes'])
    if not choices:raise ResourcePause('No allocated GPU>=6GiB')
    if (root/'allocation.json').exists():
     allocated=files.read(root/'allocation.json')
     if not any(r['uuid']==allocated['uuid'] for r in choices):raise ResourcePause('Pinned GPU lacks headroom')
    else:allocated=choices[0];files.write(root/'allocation.json',allocated)
    os.environ['CUDA_VISIBLE_DEVICES']=allocated['uuid']
    command=[sys.executable,'-m','unittest','-v','utils.test_joint_diagonal_response','utils.test_joint_diagonal_response_runtime','utils.test_current_score_checkpoint_r2']
    result=subprocess.run(command,cwd=PROJECTS,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    print(result.stdout,end='',flush=True);files.write(root/'preflight_tests.json',dict(command=command,returncode=result.returncode,output=result.stdout));result.check_returncode()
    execute(root,p);status['status']='complete'
   except ResourcePause as error:status.update(status='resource_limited',reason=str(error));print('A94 RESOURCE PAUSE:',error,flush=True)
   except BaseException as error:status.update(status='failed',error=repr(error),traceback=traceback.format_exc());print('A94 FAILED:',repr(error),flush=True)
   finally:
    files.write(record,status)
    if (root/'plan.json').exists():
     shutil.copyfile(record,root/'session.json')
     try:summarize(root);archive(root)
     except Exception as error:print('A94 archive incomplete; --archive-only:',repr(error),flush=True)
  return
 with exclusive(state/'client.lock'):
  child=None
  if not busy(state/'worker.lock'):
   if record.exists():
    prior=files.read(record);files.require(prior['status']!='failed','Prior failure retained; no automatic retry')
    if prior['status']=='complete':print('A94 already complete');return
   command=[sys.executable,'-u',str(Path(__file__)),'--execute','--root',str(root),'--a92-root',str(parent),'--gpus',args.gpus]
   with (state/'worker.log').open('ab',buffering=0) as log:child=subprocess.Popen(command,cwd=PROJECTS,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  offset=0
  while True:
   if (state/'worker.log').exists():
    with (state/'worker.log').open(errors='replace') as f:f.seek(offset);text=f.read();offset=f.tell()
    if text:print(text,end='',flush=True)
   if not busy(state/'worker.lock') and (child is None or child.poll() is not None):break
   time.sleep(1)
  if record.exists():
   status=files.read(record);print('A94 SESSION:',status,flush=True)
   if status['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
