"""A92 whole-vector direction/length and noise-fitting counterfactual diagnosis."""
import os
os.environ['MKL_THREADING_LAYER']='GNU';os.environ.pop('MKL_SERVICE_FORCE_INTEL',None)
import argparse,csv,json,subprocess,sys,time,traceback,statistics,shutil
from pathlib import Path
PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import historical_ffhq_gain as files
from utils import overall_direction_audit as rules
from utils.diag_directional_reliability import resolve_artifact
from utils.a54_session import exclusive,busy
ROOT='/data0/lzy/ZAPS-results/diag_overall_direction/A92_20261105'
PARENT='/data0/lzy/ZAPS-results/diag_shrink_multitask/A71_20261029'
BRANCH='results/zaps-overall-direction-length-audit-20261105'
SOURCE_NAMES=('utils/overall_direction_audit.py','utils/diag_overall_direction_audit.py',
 'utils/test_overall_direction_audit.py','utils/test_overall_direction_audit_runtime.py',
 'experiment_cards/A92_overall_direction_length_20261105.md','run_a92_20261105.sh',
 'utils/diag_directional_reliability.py','utils/directional_reliability.py',
 'utils/shrink_multitask.py','utils/shrink_multitask_runtime.py','utils/diag_shrink_multitask.py',
 'utils/current_score_runtime.py','utils/current_score_runtime_r2.py','utils/current_score_geometry.py',
 'utils/a67_model_freeze.py','utils/paper_clock_runtime.py','modules/main_single.py',
 'modules/diffusion_model.py','modules/zaps_algorithm.py','modules/degradations.py','modules/wavelet.py',
 'utils/metrics.py','utils/terminal_mask_counterfactual.py','utils/diag_noise_feasible_zaps.py',
 'utils/historical_ffhq_runtime.py','utils/diag_ffhq_regression.py','../DPS/util/resizer.py',
 'utils/historical_ffhq_gain.py','utils/paper_clock_alignment.py',
 'utils/diag_clean_response_precision.py','utils/test_mask_stage_response_runtime.py',
 'utils/test_current_score_checkpoint_r2.py','utils/archive_refinement_confirmation.py',
 'utils/archive_state_indicator_stage.py','../DPS/guided_diffusion/nn.py')
class ResourcePause(RuntimeError):pass


def check(root):
 p=files.read(root/'plan.json');files.require(p['experiment']=='A92_overall_direction','Wrong plan')
 for n,h in p['sources'].items():files.require(files.digest(PROJECTS/n)==h,'Bound source changed: '+n)
 for n,h in p['inputs'].items():files.require(files.digest(n)==h,'Parent artifact changed: '+n)
 return p


def prepare(root,parent):
 if (root/'plan.json').exists():return check(root)
 files.require(not root.exists() or not any(root.iterdir()),'Nonempty unplanned root retained')
 from utils.diag_shrink_multitask import check as parent_check
 pp=parent_check(parent);summary=files.read(parent/'summary.json')
 files.require(summary['completed_runs']==summary['planned_runs']==72,'Completed A71 required')
 cases=[dict(c) for c in pp['cases'] if c['sigma'] in (.05,.10)]
 files.require(len(cases)==24 and {c['image_id'] for c in cases}=={'00226','00443','00963'},'Same fixed24 cases required')
 bound={str(parent/'plan.json'):files.digest(parent/'plan.json'),str(parent/'summary.json'):files.digest(parent/'summary.json')}
 for c in cases:
  record=parent/'cases'/c['key']/'BASE/run.json'
  c['entry']=resolve_artifact(record,'entry.pt',bound)
  c['reference']=resolve_artifact(record,'paired.pt',bound)
  c['observation']=str(parent/'observations'/c['key']/'y.pt')
  bound[c['observation']]=files.digest(c['observation']);bound[c['image']]=files.digest(c['image'])
  if c['task']=='inpainting':
   f=parent/'observations'/c['key']/'mask.pt';bound[str(f)]=files.digest(f)
 f=parent/'motion_kernel.npy';bound[str(f)]=files.digest(f)
 grid=files.grid()[::-1]
 root.mkdir(parents=True)
 p=dict(experiment='A92_overall_direction',parent=str(parent),cases=cases,checkpoint=pp['checkpoint'],
  sources={n:files.digest(PROJECTS/n) for n in SOURCE_NAMES},inputs=bound,arms=list(rules.ARMS),
  intervention=dict(zero_based_index=15,t=grid[15],next_t=grid[16],fixed_grid=grid),
  budget=dict(cases=24,prefix_forwards_per_case=16,physical_reference_forwards_per_case=1,
   physical_reference_input_vjp_per_case=1,continuation_arms=9,continuation_forwards_per_arm=14,
   total_diagnostic_forwards=3432,total_input_vjp=24,optimizer_backward=0,warmup_F_VJP_per_load=1),
  protocol=dict(scope='Same3 old FFHQ images x4tasks x sigma.05/.10. Same learnedA71 nine-unroll prefix. One fixed intervention in last unroll; native continuation afterwards. Not a new online algorithm or independent30-call reconstruction.',
   reference='Negative gradient wrt next noisy state of .5||y-H(clamp(raw denoiser mean_s))||^2, at uncorrected DDPM next state. Real next-network inputVJP; original network weights never updated. Not unconditional score, next-time numerical step control, JAPS proximal target, or SURE.',
   directions='NATIVE q; DIAG_ALIGN projection of physical gradient onto cone{P_b q} of original db4/level3 ten orthogonal components, then normalize whole vector; PHYS_DIRECT unrestricted physical gradient normalized. Reference/projection depend only on y/state, never GT.',
   lengths='Each direction independently tested at .5/1/1.5 times original native q norm. Fixed paired diagnostic ladder, NOT selected by GT/loss, no new scalar policy, no global shrink norm anchor inside a proposed online method. No upper2 gamma rule.',
   cone='Coefficients max(0,<P_b q,g>/||P_b q||^2). This maximizes whole-vector cosine in the feasible cone; all directions matched in length before comparisons. Zero cone/reference marked unresolved, no fallback/substitution.',
   noise='GT read only after each pre-specified continuation for offline exact identity: physical loss gain = clean-measurement gain + <actual measurement noise, response>. Noise remains unknown to controller. No noise-separation/SURE or unbiased full-image risk claim.',
   pairing='BASE_NATIVE100 replay must match original BASE paired tensor relativeL2<=1e-6. Same given y/mask/init/current DDPM innovation; future CPU/CUDA RNG equal across arms. Exactly14 native continuation forwards. GT only final/next-clean/region metrics.',
   timing='Shared16F prefix plus1F/1VJP reference are separate. Arm clock includes its14 native continuation calls andCPU sync; offline H/metrics/image writes separately. Not complete300 walltime or production timing; model load/warmup separate.',
   decision='Primary orientation test DIAG_ALIGN100 vsNATIVE100 at equal lengths: per task/noise meanPSNR>=+.1dB,>=2/3 wins,SSIM nondecrease,LPIPS nonincrease. DIRECT100 diagnoses cone restriction. Length ladder descriptive only; no per-case best arm/GT selector. Mixed/negative retained.',
   resources='AllocatedGPU>=6GiB before load; real checkpoint VJP peak+1GiB checked after warmup and before unstarted work. No precision/resolution change, process termination or automatic failed-attempt retry.',
   limits='One fixed intervention, one seed and reused panel. Local physical descent need not imply final quality; local full/observed/missing GT errors are diagnostic only. Not matrix novelty, universal improvement, noise-unbiased estimate or inference speedup.'))
 files.write(root/'plan.json',p);return p


def archive(root):
 from utils import archive_refinement_confirmation as arch
 arch.BRANCH=BRANCH;bundle,sha=arch.export_and_commit({'A92':root},projects=PROJECTS)
 print('Archive:',bundle,'Results commit:',sha,'Branch:',BRANCH,flush=True)
 subprocess.run(['git','push','origin',BRANCH],cwd=PROJECTS.parent,check=True)
 print('RESULTS_PUSH_PASS: A92 uploaded.',flush=True)


def execute(root,parent):
 import torch
 from modules.main_single import load_diffusion_model,load_image_as_tensor
 from modules.dataset_loader import tensor_to_image
 from utils import paper_clock_runtime as clock,historical_ffhq_runtime as precision
 from utils.a67_model_freeze import frozen_network
 from utils.current_score_runtime_r2 import input_vjp_parameters
 from utils.current_score_runtime import raw_mean
 from utils.diag_shrink_multitask import make_operator
 from utils.diag_noise_feasible_zaps import metrics
 from utils.terminal_mask_counterfactual import region_errors
 p=check(root);torch.set_num_threads(2)
 files.require(files.digest(p['checkpoint']['path'])==p['checkpoint']['sha256'],'Checkpoint changed')
 load=lambda f:torch.load(f,map_location='cpu',weights_only=True)
 start=time.perf_counter();model=load_diffusion_model('ffhq','cuda:0');model.model.eval()
 weights=load(p['checkpoint']['path']);model.model.load_state_dict(weights,strict=True);del weights
 torch.cuda.synchronize();loads=root/'loads';loads.mkdir(exist_ok=True)
 loadfile=loads/(str(len(list(loads.glob('*.json')))).zfill(3)+'.json')
 loading=dict(model_load_seconds=time.perf_counter()-start,warmup_forward_attempts=0,warmup_input_vjp_attempts=0)
 with frozen_network(model.model),precision.precision('LEGACY_IEEE'):
  torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  with input_vjp_parameters(model.model),torch.enable_grad():
   loading['warmup_forward_attempts']=1;files.write(loadfile,loading)
   inp,raw=raw_mean(model,torch.zeros(1,3,256,256,device=model.device),p['intervention']['next_t'],graph=True)
   loading['warmup_input_vjp_attempts']=1;files.write(loadfile,loading)
   grad=torch.autograd.grad(raw.square().mean(),inp)[0]
   files.require(bool(torch.isfinite(grad).all()),'Checkpoint probe gradient invalid')
  del grad,raw,inp;torch.cuda.synchronize()
  peak=torch.cuda.max_memory_allocated();loading.update(warmup_seconds=time.perf_counter()-start,peak_allocated_bytes=peak)
  files.write(loadfile,loading)
  def admission():
   free=int(torch.cuda.mem_get_info()[0]);reusable=torch.cuda.memory_reserved()-torch.cuda.memory_allocated()
   if free+reusable<peak+files.GIB:raise ResourcePause('Measured checkpoint VJP peak+1GiB lacks margin; no new attempt')
  for c in p['cases']:
   admission();casefolder=root/'cases'/c['key'];prefixrecord=casefolder/'prefix.json'
   entry=load(c['entry']);y=load(c['observation'])['y'].to(model.device)
   if prefixrecord.exists():
    row=files.read(prefixrecord);files.require(row['status']=='complete','Failed/interrupted prefix retained')
    files.require(row['plan_sha256']==files.digest(root/'plan.json'),'Prefix plan mismatch')
    for n,h in row['artifacts'].items():files.require(files.digest(casefolder/n)==h,'Prefix artifact changed')
    bundle=load(casefolder/'state.pt')
   else:
    casefolder.mkdir(parents=True,exist_ok=True);row=dict(status='running',case=c,counts={},plan_sha256=files.digest(root/'plan.json'))
    files.write(prefixrecord,row)
    count={};probe={}
    try:
     with clock.counters(model.model) as count:
      op=make_operator(parent,c,model.device,count)
      state,seconds=clock.clocks(torch.cuda.synchronize,lambda:rules.prefix(model,op,y,entry,c['task']))
     files.require(count['forwards']==count['forward_attempts']==16 and count['calls']==p['intervention']['fixed_grid'][:16],'Prefix grid/count mismatch')
     files.require(count['input_vjp_attempts']==count['optimizer_backward_attempts']==0,'Prefix backward forbidden')
     row.update(counts=count,prefix_seconds=seconds);files.write(prefixrecord,row)
     with clock.counters(model.model) as probe:
      op=make_operator(parent,c,model.device,probe)
      (physical,clean),probe_seconds=clock.clocks(torch.cuda.synchronize,lambda:rules.physical_reference(model,op,y,state,probe))
     files.require(probe['forwards']==probe['forward_attempts']==probe['input_vjp_attempts']==1 and probe['optimizer_backward_attempts']==0,'Reference cost mismatch')
     z=rules.make_z(model,op,entry,c['task'])
     units,detail=rules.directions(state['q'].to(model.device),physical,z.dwt)
     bundle=dict(state=state,physical=physical.cpu(),reference_clean=clean.cpu(),units={k:None if v is None else v.cpu() for k,v in units.items()},detail=detail)
     torch.save(bundle,casefolder/'state.pt')
     files.write(casefolder/'directions.json',detail)
     row.update(status='complete',probe_counts=probe,reference_seconds=probe_seconds,
       artifacts={n:files.digest(casefolder/n) for n in ('state.pt','directions.json')})
     files.write(prefixrecord,row);print('A92 STATE READY:',c['key'],'prefixF=16 referenceF/VJP=1/1',flush=True)
    except BaseException as error:
     row.update(status='failed',error=repr(error),traceback=traceback.format_exc(),counts=locals().get('count',{}),probe_counts=locals().get('probe',{}));files.write(prefixrecord,row);raise
   state=bundle['state'];detail=bundle['detail']
   for arm in rules.ARMS:
    folder=casefolder/arm;record=folder/'run.json'
    if record.exists():
     old=files.read(record);files.require(old['status'] in ('complete','unresolved'),'Failed/interrupted arm retained')
     for n,h in old.get('artifacts',{}).items():files.require(files.digest(folder/n)==h,'Arm artifact changed')
     continue
    admission();kind,scale=rules.specification(arm);folder.mkdir()
    if bundle['units'][kind] is None:
     files.write(record,dict(status='unresolved',case=c,arm=arm,reason='ZERO_CONE_OR_REFERENCE_NO_SUBSTITUTION',counts={}));continue
    row=dict(status='running',case=c,arm=arm,counts={});files.write(record,row)
    count={}
    try:
     correction=(state['q'].to(model.device)*scale if kind=='NATIVE' else bundle['units'][kind].to(model.device)*(detail['native_norm']*scale))
     matched=float(correction.double().norm())/(detail['native_norm']*scale)
     files.require(abs(matched-1)<3e-5,'Whole-vector length mismatch')
     row['applied_band_gains']=([scale]*10 if kind=='NATIVE' else
       [float(v)*detail['native_norm']*scale/detail['directions']['DIAG_ALIGN']['original_norm'] for v in detail['band_projection_coefficients']] if kind=='DIAG_ALIGN' else None)
     with clock.counters(model.model) as count:
      op=make_operator(parent,c,model.device,count);z=rules.make_z(model,op,entry,c['task'])
      (output,nextclean,rng),seconds=clock.clocks(torch.cuda.synchronize,lambda:rules.continue_run(model,z,y,state,correction))
     row['counts']=count
     files.require(count['forwards']==count['forward_attempts']==14 and count['calls']==p['intervention']['fixed_grid'][16:],'Continuation grid/count changed')
     files.require(count['input_vjp_attempts']==count['optimizer_backward_attempts']==0,'Continuation backward forbidden')
     if arm=='NATIVE_100':
      reference=load(c['reference']);error=float((output.double()-reference.double()).norm()/reference.double().norm().clamp(min=1e-30))
      files.require(error<=1e-6,'Native output replay failed');row['native_replay_relative_error']=error
     else:
      other=load(casefolder/'NATIVE_100/final_rng.pt')
      files.require(torch.equal(rng['cpu_rng'],other['cpu_rng']) and len(rng['cuda_rng'])==len(other['cuda_rng']) and all(torch.equal(a,b) for a,b in zip(rng['cuda_rng'],other['cuda_rng'])),'Future innovation RNG mismatch')
     files.write(record,row)
     start=time.perf_counter();gt=load_image_as_tensor(c['image']).cpu()
     op=make_operator(parent,c,model.device)
     audit=rules.offline_readout(op,y,gt.to(model.device),bundle['reference_clean'].to(model.device),nextclean.to(model.device),correction,bundle['physical'].to(model.device))
     audit['residual_to_noise_ratio']=audit['residual_energy_after']/(audit['active_measurements']*c['sigma']**2)
     if c['task']=='inpainting':
      active=op._mask_cache.bool().expand_as(correction)
      audit['correction_observed_energy']=float(correction.double()[active].square().sum())
      audit['correction_missing_energy']=float(correction.double()[~active].square().sum())
     m=metrics(output,gt)
     if c['task']=='inpainting':
      mask=load(parent/'observations'/c['key']/'mask.pt');m.update(region_errors(output,gt,mask))
      audit['next_clean_region_errors']=region_errors(nextclean,gt,mask)
     row['offline_metrics_seconds']=time.perf_counter()-start
     start=time.perf_counter()
     torch.save(output,folder/'output.pt');torch.save(nextclean,folder/'next_clean.pt');torch.save(rng,folder/'final_rng.pt')
     tensor_to_image(output.squeeze(0),denormalize=True).save(folder/'output.png')
     files.write(folder/'audit.json',audit)
     row.update(status='complete',counts=count,metrics=m,continuation_seconds=seconds,artifact_seconds=time.perf_counter()-start,
      length_ratio_to_native=scale,matched_length_ratio=matched,cosine_to_physical=detail['directions'][kind]['cosine_to_physical'],
      audit=audit,artifacts={n:files.digest(folder/n) for n in ('output.pt','next_clean.pt','final_rng.pt','output.png','audit.json')})
     files.write(record,row);print('A92 COMPLETE:',c['key'],arm,'F=14 PSNR=',m['psnr'],flush=True)
    except BaseException as error:
     row.update(status='failed',error=repr(error),traceback=traceback.format_exc(),counts=locals().get('count',{}));files.write(record,row);raise
 summarize(root)


def summarize(root):
 p=check(root);rows=[files.read(f) for f in (root/'cases').glob('*/*/run.json')]
 prefixes=[files.read(f) for f in (root/'cases').glob('*/prefix.json')];done=[r for r in rows if r['status']=='complete']
 groups=[];comparisons=[]
 for task in ('super_resolution','gaussian_deblur','inpainting','motion_deblur'):
  for sigma in (.05,.10):
   byarm={arm:{r['case']['key']:r for r in done if r['arm']==arm and r['case']['task']==task and r['case']['sigma']==sigma} for arm in rules.ARMS}
   for arm,rs in byarm.items():
    if not rs:continue
    vals=list(rs.values());mean=lambda f:statistics.mean(f(r) for r in vals)
    groups.append(dict(task=task,sigma=sigma,arm=arm,images=len(vals),quality={k:mean(lambda r:r['metrics'][k]) for k in ('psnr','ssim','lpips')},mean_continuation_seconds=mean(lambda r:r['continuation_seconds']),
     mean_physical_gain=mean(lambda r:r['audit']['physical_gain']),mean_clean_measurement_gain=mean(lambda r:r['audit']['clean_measurement_gain']),mean_fitted_noise_gain=mean(lambda r:r['audit']['fitted_noise_gain']),
     physical_improved_but_clean_measurement_worse=sum(r['audit']['physical_gain']>0 and r['audit']['clean_measurement_gain']<0 for r in vals),
     physical_improved_but_final_PSNR_worse_than_native=sum(r['audit']['physical_gain']>0 and r['case']['key'] in byarm['NATIVE_100'] and r['metrics']['psnr']<byarm['NATIVE_100'][r['case']['key']]['metrics']['psnr'] for r in vals)))
   for arm in rules.ARMS:
    if arm=='NATIVE_100':continue
    matched='NATIVE_'+arm.rsplit('_',1)[1]
    references=['NATIVE_100']+([matched] if matched!='NATIVE_100' and not arm.startswith('NATIVE_') else [])
    for reference in references:
     pairs=[(byarm[reference][key],r) for key,r in byarm[arm].items() if key in byarm[reference]]
     if not pairs:continue
     delta={k:statistics.mean(b['metrics'][k]-a['metrics'][k] for a,b in pairs) for k in ('psnr','ssim','lpips')}
     audit_delta={k:statistics.mean(b['audit'][k]-a['audit'][k] for a,b in pairs) for k in ('physical_gain','clean_measurement_gain','fitted_noise_gain')}
     wins=sum(b['metrics']['psnr']>a['metrics']['psnr'] for a,b in pairs)
     comparisons.append(dict(task=task,sigma=sigma,arm=arm,reference=reference,cases=len(pairs),quality_delta=delta,offline_gain_delta=audit_delta,images_PSNR_positive=wins,
      equal_length_orientation_test=not arm.startswith('NATIVE_') and reference==matched,
      primary_orientation_test=arm=='DIAG_ALIGN_100' and reference=='NATIVE_100',
      positive_development_candidate=len(pairs)==3 and delta['psnr']>=.1 and wins>=2 and delta['ssim']>=0 and delta['lpips']<=0))
 total=lambda key:sum(r.get('counts',{}).get(key,0) for r in rows+prefixes)+sum(r.get('probe_counts',{}).get(key,0) for r in prefixes)
 s=dict(experiment=p['experiment'],classification='SINGLE_INTERVENTION_DIRECTION_LENGTH_AND_NOISE_DIAGNOSIS',completed_runs=len(done),planned_runs=216,
  unresolved_runs=sum(r['status']=='unresolved' for r in rows),completed_prefixes=sum(r['status']=='complete' for r in prefixes),groups=groups,comparisons=comparisons,
  recorded_diagnostic_forwards=total('forward_attempts'),input_vjp_attempts=total('input_vjp_attempts'),optimizer_backward_attempts=total('optimizer_backward_attempts'),
  offline_H_actions=sum(r['audit'].get('offline_H_actions',0) for r in done),
  cost_is_lower_bound=any(r['status'] not in ('complete','unresolved') for r in rows+prefixes),
  loads=[files.read(f) for f in (root/'loads').glob('*.json')],protocol=p['protocol'])
 files.write(root/'summary.json',s)
 if done:
  values=[dict(task=r['case']['task'],sigma=r['case']['sigma'],image_id=r['case']['image_id'],arm=r['arm'],**r['metrics'],continuation_seconds=r['continuation_seconds'],**{k:r['audit'][k] for k in ('physical_gain','clean_measurement_gain','fitted_noise_gain')}) for r in done]
  keys=list(dict.fromkeys(k for r in values for k in r))
  with (root/'comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(values)
 print('A92 Summary:',json.dumps(s,ensure_ascii=False),flush=True)


def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',default=ROOT);a.add_argument('--a71-root',default=PARENT);a.add_argument('--gpus',default='0,1')
 for flag in ('execute','status','archive-only'):a.add_argument('--'+flag,action='store_true')
 args=a.parse_args();root=Path(args.root).resolve();parent=Path(args.a71_root).resolve();state=root.parent/(root.name+'_session');state.mkdir(parents=True,exist_ok=True);record=state/'session.json'
 if args.status:print(dict(active=busy(state/'worker.lock'),session=files.read(record) if record.exists() else None));return
 if args.archive_only:files.require(not busy(state/'worker.lock'),'Active worker');summarize(root);archive(root);return
 if args.execute:
  with exclusive(state/'worker.lock'):
   status=dict(status='running',pid=os.getpid(),root=str(root));files.write(record,status)
   try:
    prepare(root,parent)
    from utils.diag_paper_clock_alignment import gpu_inventory
    candidates=files.available_gpus(gpu_inventory(),args.gpus,os.environ.get('CUDA_VISIBLE_DEVICES'))
    candidates=sorted([r for r in candidates if r['free_bytes']>=6*files.GIB],key=lambda r:-r['free_bytes'])
    if not candidates:raise ResourcePause('No allocated GPU>=6GiB free')
    if (root/'allocation.json').exists():
     assigned=files.read(root/'allocation.json')
     if not any(r['uuid']==assigned['uuid'] for r in candidates):raise ResourcePause('Pinned GPU lacks margin')
    else:assigned=candidates[0];files.write(root/'allocation.json',assigned)
    os.environ['CUDA_VISIBLE_DEVICES']=assigned['uuid']
    command=[sys.executable,'-m','unittest','-v','utils.test_overall_direction_audit','utils.test_overall_direction_audit_runtime','utils.test_current_score_checkpoint_r2']
    test=subprocess.run(command,cwd=PROJECTS,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    print(test.stdout,end='',flush=True);files.write(root/'preflight_tests.json',dict(command=command,returncode=test.returncode,output=test.stdout));test.check_returncode()
    execute(root,parent);status['status']='complete'
   except ResourcePause as error:status.update(status='resource_limited',reason=str(error));print('A92 RESOURCE PAUSE:',error,flush=True)
   except BaseException as error:status.update(status='failed',error=repr(error),traceback=traceback.format_exc());print('A92 FAILED:',repr(error),flush=True)
   finally:
    files.write(record,status)
    if (root/'plan.json').exists():
     shutil.copyfile(record,root/'session.json')
     try:summarize(root);archive(root)
     except Exception as error:print('A92 archive incomplete; --archive-only:',repr(error),flush=True)
  return
 with exclusive(state/'client.lock'):
  process=None
  if not busy(state/'worker.lock'):
   if record.exists():
    old=files.read(record);files.require(old['status']!='failed','Prior failure retained; no automatic retry')
    if old['status']=='complete':print('A92 already complete:',root/'summary.json');return
   with (state/'worker.log').open('ab',buffering=0) as log:process=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--execute','--root',str(root),'--a71-root',str(parent),'--gpus',args.gpus],cwd=PROJECTS,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  offset=0
  while True:
   if (state/'worker.log').exists():
    with (state/'worker.log').open(errors='replace') as f:f.seek(offset);v=f.read();offset=f.tell()
    if v:print(v,end='',flush=True)
   if not busy(state/'worker.lock') and (process is None or process.poll() is not None):break
   time.sleep(1)
  if record.exists():
   row=files.read(record);print('A92 SESSION:',row,flush=True)
   if row['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
