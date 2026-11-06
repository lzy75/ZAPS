"""A96 direction/length factorial plus fixed-stage SR error diagnosis, zero new solves."""
import os
os.environ['MKL_THREADING_LAYER']='GNU';os.environ.pop('MKL_SERVICE_FORCE_INTEL',None)
import argparse,subprocess,sys,time,traceback,statistics,shutil,csv
from pathlib import Path
PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import historical_ffhq_gain as files,direction_length_attribution as rules
from utils import diag_response_target_lm as parent_driver
from utils.a54_session import exclusive,busy
ROOT='/data0/lzy/ZAPS-results/diag_direction_length/A96_20261106'
PARENT='/data0/lzy/ZAPS-results/diag_response_target/A95R2_20261106'
BRANCH='results/zaps-direction-length-attribution-20261106'
NAMES=('utils/direction_length_attribution.py','utils/diag_direction_length_attribution.py',
 'utils/test_direction_length_attribution.py','utils/test_direction_length_attribution_runtime.py',
 'run_a96_20261106.sh','experiment_cards/A96_direction_length_attribution_20261106.md')
class ResourcePause(RuntimeError):pass


def check(root):
 p=files.read(root/'plan.json');files.require(p['experiment']=='A96_direction_length','Wrong A96 plan')
 for n,h in p['sources'].items():files.require(files.digest(PROJECTS/n)==h,'Source changed: '+n)
 for n,h in p['inputs'].items():files.require(files.digest(n)==h,'Input changed: '+n)
 return p


def prepare(root,parent):
 if (root/'plan.json').exists():return check(root)
 files.require(not root.exists() or not any(root.iterdir()),'Unplanned nonempty root retained')
 pp=parent_driver.check(parent);s=files.read(parent/'summary.json')
 files.require(pp.get('revision')=='A95R2' and s['completed_runs']==s['planned_runs']==48,'Completed A95R2 required')
 bound=dict(pp['inputs']);bound.update({str(parent/n):files.digest(parent/n) for n in ('plan.json','summary.json','actual_checkpoint_sources.json')})
 for path,sha in files.read(parent/'actual_checkpoint_sources.json').items():
  files.require(files.digest(path)==sha,'Actual loaded model code changed');bound[path]=sha
 cases=[]
 for c0 in pp['cases']:
  c=dict(c0);c['a95']={}
  for arm in ('REAL_SCALAR','REAL_DIAG'):
   record=parent/'cases'/c['key']/arm/'run.json';row=files.read(record);files.require(row['status']=='complete','Incomplete A95 reference')
   bound[str(record)]=files.digest(record);c['a95'][arm]=str(record)
   for name in ('correction.pt','output.pt','final_rng.pt'):
    path=record.parent/name;files.require(files.digest(path)==row['artifacts'][name],'Parent tensor changed');bound[str(path)]=row['artifacts'][name]
  cases.append(c)
 files.require(len(cases)==12 and all(c['sigma']==.05 for c in cases),'Only same12 normal-noise cases')
 root.mkdir(parents=True)
 p=dict(experiment='A96_direction_length',parent=str(parent),a71_parent=pp['a71_parent'],cases=cases,checkpoint=pp['checkpoint'],grid=pp['grid'],
  sources={**pp['sources'],**{n:files.digest(PROJECTS/n) for n in NAMES}},inputs=bound,arms=list(rules.ARMS),
  budget=dict(new_forwards=840,arms=60,forwards_per_arm=14,new_input_vjp=0,new_input_jvp=0,new_optimizer_backward=0,warmup_F_per_load=1),
  protocol=dict(scope='Same3 reused FFHQ x4 tasks sigma=.05, one A92 correction at t333->309. Reuse A95R2 finite20-iteration vectors, not newly optimized vectors. No full30 intervention, new image/seed or high noise.',
   arms='SCALAR_REPLAY S; DIAG_REPLAY D; DIAG_DIR_SCALAR_LEN ||S|| D/||D||; SCALAR_DIR_DIAG_LEN ||D|| q_native/||q_native||; ZERO=0. Scalar ray defined by original q even if scalar length0. Undefined nonzero-length zero-diagonal direction stops, no fallback.',
   validity='S must lie on nonnegative native ray. Both original replays must match A95 output relativeL2<=1e-6 and exact final RNG before quality interpretation. All5 share A92 state/current innovation/future draws and native14 continuation; no LM/JVP/VJP/optimizer or new scalar decisions.',
   factorial='Direction contrasts at fixed scalar and diagonal lengths; length contrasts on each direction; interaction=Y_DD-Y_DS-Y_SD+Y_SS. PSNR/SSIM/LPIPS all reported; lower LPIPS is better. Zero action separates natural continuation from intervention. No GT best-arm selector or imposed online length cap.',
   sr_diagnosis='GT read only AFTER actions/continuation. At clean stages global16(s309),24(t119),29(t0), and final output, log raw/clamped MSE, clamp fraction, exact db4 error energy contributions LL3/details and observed clean measurement MSE. First-clean observed gain splits into true signal gain and fitted-noise gain. Frequency detail is NOT mathematical H-nullspace.',
   hypotheses='SR worse only at long lengths suggests amplitude contribution; worse at matched short length suggests direction contribution. Lower measured-domain error with worse detail/full error suggests objective/whole-image mismatch. First-clean gain followed by later regression suggests continuation interaction. These paired diagnostics distinguish contributions, not unique universal causes.',
   qualification='All same-state interventions, fixed factorial, no residual-based policy rescue. Parent DIAG was budget-limited: attribute that observed finite vector only, do not certify converged optimal diagonal. Per-task directional gain requires +.1dB,>=2/3 wins,SSIM nondecrease,LPIPS nonincrease at matched length; report mixed results and interaction.',
   accounting='60 x14=840 new diagnostic UNet forwards;1 warmupF/load separate;0 new solver/JVP/VJP/optimizer. Previous A95 optimization acquisition remains paid. Metrics/wavelet/H-error accounting separate; no complete300 runtime or acceleration claim.',
   limits='Single-position causal attribution is fair against single-position scalar. Full method comparison to prior last30 scalar requires a later same-scope last30 experiment after this diagnosis. Small reused panel; no generalization or novelty theorem.',
   resources='Allocated GPU>=6GiB; no model downloads/process termination/precision/resolution changes. Original checkpoint strict reload and frozen weights, original IEEE. Interrupted/failed records retained, no automatic retries. No next-phase auto launch.'))
 files.write(root/'plan.json',p);shutil.copyfile(PROJECTS/'experiment_cards/A96_direction_length_attribution_20261106.md',root/'experiment_card.md')
 return p


def execute(root,p):
 import torch
 from modules.main_single import load_diffusion_model,load_image_as_tensor
 from modules.dataset_loader import tensor_to_image
 from modules.wavelet import OrthogonalDWT2D
 from utils import overall_direction_audit as old,paper_clock_runtime as clock,historical_ffhq_runtime as arithmetic
 from utils.a67_model_freeze import frozen_network
 from utils.diag_shrink_multitask import make_operator
 from utils.diag_noise_feasible_zaps import metrics
 from utils.shrink_multitask_runtime import informative
 from utils.response_target_runtime import target,rng_equal
 load=lambda f:torch.load(f,map_location='cpu',weights_only=True)
 files.require(files.digest(p['checkpoint']['path'])==p['checkpoint']['sha256'],'Wrong checkpoint');torch.set_num_threads(2)
 start=time.perf_counter();model=load_diffusion_model('ffhq','cuda:0');model.model.eval();weights=load(p['checkpoint']['path']);model.model.load_state_dict(weights,strict=True);del weights
 loads=root/'loads';loads.mkdir(exist_ok=True);lf=loads/(str(len(list(loads.glob('*.json')))).zfill(3)+'.json')
 rec=dict(model_load_seconds=time.perf_counter()-start,warmup_F_attempts=0,torch=str(torch.__version__),device=torch.cuda.get_device_name())
 with frozen_network(model.model),arithmetic.precision('LEGACY_IEEE'):
  torch.cuda.reset_peak_memory_stats();rec['warmup_F_attempts']=1;files.write(lf,rec)
  with torch.no_grad():model._predict_eps(torch.zeros(1,3,256,256,device=model.device),torch.tensor([p['grid'][16]],device=model.device))
  torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated();rec['peak_allocated_bytes']=peak;files.write(lf,rec)
  dwt=OrthogonalDWT2D('db4',3)
  for c in p['cases']:
   for arm in rules.ARMS:
    folder=root/'cases'/c['key']/arm;record=folder/'run.json'
    if record.exists():
     row=files.read(record);files.require(row['status']=='complete','Prior failure/interruption retained')
     for name,sha in row['artifacts'].items():files.require(files.digest(folder/name)==sha,'Artifact changed')
     continue
    if int(torch.cuda.mem_get_info()[0])+torch.cuda.memory_reserved()-torch.cuda.memory_allocated()<peak+files.GIB:raise ResourcePause('Peak+1GiB lacks margin')
    folder.mkdir(parents=True);row=dict(status='running',case=c,arm=arm);files.write(record,row);count={}
    try:
     with clock.counters(model.model) as count:
      def operation():
       bundle=load(c['bundle']);entry=load(c['entry']);y=load(c['observation'])['y'].to(model.device);op=make_operator(Path(p['a71_parent']),c,model.device,count)
       native=bundle['state']['q'].to(model.device)
       scalar=load(Path(c['a95']['REAL_SCALAR']).parent/'correction.pt').to(model.device);diagonal=load(Path(c['a95']['REAL_DIAG']).parent/'correction.pt').to(model.device)
       nn=rules.length(native);sn=rules.length(scalar)
       coefficient=float((scalar.double()*native.double()).sum())/max(nn*nn,1e-30)
       files.require(nn>0 and coefficient>=-1e-7 and rules.length(scalar-coefficient*native)<=1e-5*max(sn,1e-12),'Parent scalar is not on original nonnegative ray')
       delta=rules.correction(native,scalar,diagonal,arm);z=old.make_z(model,op,entry,c['task'])
       expected_norm={'SCALAR_REPLAY':sn,'DIAG_REPLAY':rules.length(diagonal),'DIAG_DIR_SCALAR_LEN':sn,'SCALAR_DIR_DIAG_LEN':rules.length(diagonal),'ZERO':0.}[arm]
       files.require(abs(rules.length(delta)-expected_norm)<=3e-5*max(expected_norm,1e-20),'Matched-length gate failed')
       output,snapshots,rng=rules.continue_run(model,z,y,bundle['state'],delta)
       return output,snapshots,rng,delta.cpu(),dict(native_norm=nn,scalar_norm=sn,diagonal_norm=rules.length(diagonal),correction_norm=rules.length(delta),scalar_ray_coefficient=coefficient)
      (output,snapshots,rng,delta,geometry),seconds=clock.clocks(torch.cuda.synchronize,operation)
     files.require(count['forwards']==count['forward_attempts']==14 and count['calls']==p['grid'][16:],'Continuation count/grid mismatch')
     files.require(count['input_vjp_attempts']==count['optimizer_backward_attempts']==0,'Unexpected backward')
     reference=load(Path(c['references']['NATIVE_100']).parent/'final_rng.pt');files.require(rng_equal(rng,reference),'Future RNG mismatch')
     replay=None
     if arm in ('SCALAR_REPLAY','DIAG_REPLAY'):
      key='REAL_SCALAR' if arm=='SCALAR_REPLAY' else 'REAL_DIAG';source=Path(c['a95'][key]).parent
      expected=load(source/'output.pt');error=rules.length(output-expected)/max(rules.length(expected),1e-30)
      files.require(error<=1e-6 and rng_equal(rng,load(source/'final_rng.pt')),'A95 replay fidelity failed');replay=dict(passed=True,relative_L2=error)
     row.update(counts=count,suffix_seconds=seconds,geometry=geometry,replay=replay);files.write(record,row)
     # All GT reads and risk decomposition occur only after action and all14 calls.
     start=time.perf_counter();gt=load_image_as_tensor(c['image']).cpu();m=metrics(output,gt);bundle=load(c['bundle']);y=load(c['observation'])['y'].to(model.device);op=make_operator(Path(p['a71_parent']),c,model.device)
     first=snapshots['16']['clean'];audit=old.offline_readout(op,y,gt.to(model.device),bundle['reference_clean'].to(model.device),first.to(model.device),delta.to(model.device),bundle['physical'].to(model.device))
     destination,omega=target(model,op,y,bundle,c['sigma'])
     with torch.no_grad():
      predicted=informative(op.H(first.to(model.device)),op);audit['response_target_objective']=.5*float((predicted.double()-destination.double()).square().sum())
      truth=op.H(gt.to(model.device));stages={}
      for k,item in {**snapshots,'output':dict(clean=output,t=-1)}.items():
       image=item['clean'];r=rules.wavelet_risk(image,gt,dwt)
       err=informative(op.H(image.to(model.device))-truth,op).double();r.update(t=item['t'],clean_measurement_mse=float(err.square().mean()))
       stages[k]=r
     row['offline_seconds']=time.perf_counter()-start
     torch.save(output,folder/'output.pt');torch.save(snapshots,folder/'snapshots.pt');torch.save(rng,folder/'final_rng.pt');torch.save(delta,folder/'correction.pt')
     tensor_to_image(output.squeeze(0),denormalize=True).save(folder/'output.png');files.write(folder/'diagnosis.json',dict(stages=stages,audit=audit,omega=omega))
     row.update(status='complete',metrics=m,audit=audit,stages=stages,offline_extra_H_actions=7,
      artifacts={n:files.digest(folder/n) for n in ('output.pt','snapshots.pt','final_rng.pt','correction.pt','output.png','diagnosis.json')});files.write(record,row)
     print('A96 COMPLETE:',c['key'],arm,'PSNR=',round(m['psnr'],5),flush=True)
    except BaseException as error:
     row.update(status='failed',counts=count,error=repr(error),traceback=traceback.format_exc());files.write(record,row);raise


def summarize(root):
 p=check(root);rows=[files.read(f) for f in (root/'cases').glob('*/*/run.json')];done=[r for r in rows if r['status']=='complete'];groups=[];contrasts=[]
 for task in ('super_resolution','gaussian_deblur','inpainting','motion_deblur'):
  rs=[r for r in done if r['case']['task']==task]
  for arm in rules.ARMS:
   selected=[r for r in rs if r['arm']==arm]
   if selected:groups.append(dict(task=task,arm=arm,images=len(selected),quality={k:statistics.mean(r['metrics'][k] for r in selected) for k in ('psnr','ssim','lpips')},
    mean_target_objective=statistics.mean(r['audit']['response_target_objective'] for r in selected),mean_clean_gain=statistics.mean(r['audit']['clean_measurement_gain'] for r in selected),mean_noise_gain=statistics.mean(r['audit']['fitted_noise_gain'] for r in selected),
    stages={stage:{key:statistics.mean(r['stages'][stage][key] for r in selected) for key in ('raw_mse','clamped_mse','coarse_LL3_mse','detail_mse','clean_measurement_mse','clamp_fraction')} for stage in ('16','24','29','output')}))
  paired=[]
  for image in sorted({r['case']['key'] for r in rs}):
   values={r['arm']:r for r in rs if r['case']['key']==image}
   if all(a in values for a in rules.ARMS):paired.append(values)
  if paired:
   contrasts.append(dict(task=task,images=len(paired),factorial={metric:{key:statistics.mean(rules.factorial({a:v[a]['metrics'][metric] for a in rules.ARMS})[key] for v in paired) for key in rules.factorial({a:0. for a in rules.ARMS})} for metric in ('psnr','ssim','lpips')}))
 references=[]
 for task in ('super_resolution','gaussian_deblur','inpainting','motion_deblur'):
  cases=[c for c in p.get('cases',[]) if c['task']==task]
  for reference in ('NATIVE_100','PHYS_DIRECT_100'):
   values=[files.read(c['references'][reference])['metrics'] for c in cases]
   if values:references.append(dict(task=task,arm=reference,images=len(values),quality={k:statistics.mean(v[k] for v in values) for k in ('psnr','ssim','lpips')},scope='Existing A92 same-position control, no new NN'))
 s=dict(experiment=p['experiment'],classification='CACHED_FINITE_VECTOR_DIRECTION_LENGTH_FACTORIAL',completed_runs=len(done),planned_runs=60,groups=groups,contrasts=contrasts,reference_groups=references,
  actual_new_cost={k:sum(r.get('counts',{}).get(k,0) for r in rows) for k in ('forward_attempts','forwards','input_vjp_attempts','optimizer_backward_attempts')},
  cost_is_lower_bound=any(r['status']!='complete' for r in rows),loads=[files.read(f) for f in (root/'loads').glob('*.json')],protocol=p['protocol'])
 files.write(root/'summary.json',s)
 if done:
  with (root/'comparison.csv').open('w',newline='') as f:
   keys=['task','image','arm','psnr','ssim','lpips'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
   for r in done:w.writerow(dict(task=r['case']['task'],image=r['case']['image_id'],arm=r['arm'],**{k:r['metrics'][k] for k in keys[3:]}))
 print('A96 STATUS:',len(done),'/60; new cost=',s['actual_new_cost'],flush=True)
 for r in contrasts:print('A96 FACTORIAL:',r['task'],{k:round(v,5) for k,v in r['factorial']['psnr'].items()},flush=True)
 print('Full summary:',root/'summary.json',flush=True);return s


def archive(root):
 from utils import archive_refinement_confirmation as arch
 arch.BRANCH=BRANCH;bundle,sha=arch.export_and_commit({'A96':root},projects=PROJECTS)
 print('Archive:',bundle,'Results commit:',sha,'Branch:',BRANCH,flush=True)
 subprocess.run(['git','push','origin',BRANCH],cwd=PROJECTS.parent,check=True);print('RESULTS_PUSH_PASS: A96 uploaded.',flush=True)


def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',default=ROOT);a.add_argument('--a95-root',default=PARENT);a.add_argument('--gpus',default='0,1')
 for flag in ('execute','status','archive-only'):a.add_argument('--'+flag,action='store_true')
 args=a.parse_args();root=Path(args.root).resolve();parent=Path(args.a95_root).resolve();state=root.parent/(root.name+'_session');state.mkdir(parents=True,exist_ok=True);record=state/'session.json'
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
    command=[sys.executable,'-m','unittest','-v','utils.test_direction_length_attribution','utils.test_direction_length_attribution_runtime']
    result=subprocess.run(command,cwd=PROJECTS,capture_output=True,text=True);print(result.stdout+result.stderr,end='',flush=True)
    files.write(root/'preflight_tests.json',dict(command=command,returncode=result.returncode,output=result.stdout+result.stderr));result.check_returncode()
    execute(root,p);status['status']='complete'
   except ResourcePause as error:status.update(status='resource_limited',reason=str(error));print('A96 RESOURCE PAUSE:',error,flush=True)
   except BaseException as error:status.update(status='failed',error=repr(error),traceback=traceback.format_exc());print('A96 FAILED:',repr(error),flush=True)
   finally:
    files.write(record,status)
    if (root/'plan.json').exists():
     shutil.copyfile(record,root/'session.json')
     try:summarize(root);archive(root)
     except Exception as error:print('A96 archive incomplete; --archive-only:',repr(error),flush=True)
  return
 with exclusive(state/'client.lock'):
  child=None
  if not busy(state/'worker.lock'):
   if record.exists():
    prior=files.read(record);files.require(prior['status']!='failed','Prior failure retained; no automatic retry')
    if prior['status']=='complete':print('A96 already complete');return
   command=[sys.executable,'-u',str(Path(__file__)),'--execute','--root',str(root),'--a95-root',str(parent),'--gpus',args.gpus]
   with (state/'worker.log').open('ab',buffering=0) as log:child=subprocess.Popen(command,cwd=PROJECTS,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  offset=0
  while True:
   if (state/'worker.log').exists():
    with (state/'worker.log').open(errors='replace') as f:f.seek(offset);text=f.read();offset=f.tell()
    if text:print(text,end='',flush=True)
   if not busy(state/'worker.lock') and (child is None or child.poll() is not None):break
   time.sleep(1)
  if record.exists():
   status=files.read(record);print('A96 SESSION:',status,flush=True)
   if status['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
