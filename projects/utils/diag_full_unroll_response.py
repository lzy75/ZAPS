"""A97 online30 suffix comparison, GPU case shards and safe step-boundary resume."""
import os
os.environ['MKL_THREADING_LAYER']='GNU';os.environ.pop('MKL_SERVICE_FORCE_INTEL',None)
import argparse,subprocess,sys,time,traceback,statistics,shutil,csv
from pathlib import Path
PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import historical_ffhq_gain as files,full_unroll_response as rules
from utils import diag_direction_length_attribution as parent_driver
from utils.a54_session import exclusive,busy
ROOT='/data0/lzy/ZAPS-results/diag_full_unroll_response/A97_20261106'
PARENT=parent_driver.ROOT
BRANCH='results/zaps-full-last-unroll-response-20261106'
NAMES=('utils/full_unroll_response.py','utils/diag_full_unroll_response.py','utils/test_full_unroll_response.py',
 'utils/test_full_unroll_response_runtime.py','run_a97_20261106.sh','experiment_cards/A97_full_unroll_response_20261106.md')
class ResourcePause(RuntimeError):pass


def check(root):
 p=files.read(root/'plan.json');files.require(p['experiment']=='A97_full_unroll','Wrong A97 plan')
 for n,h in p['sources'].items():files.require(files.digest(PROJECTS/n)==h,'Source changed: '+n)
 for n,h in p['inputs'].items():files.require(files.digest(n)==h,'Input changed: '+n)
 return p


def prepare(root,parent):
 if (root/'plan.json').exists():return check(root)
 files.require(not root.exists() or not any(root.iterdir()),'Unplanned nonempty root retained')
 pp=parent_driver.check(parent);s=files.read(parent/'summary.json')
 files.require(s['completed_runs']==s['planned_runs']==60,'Completed A96 required')
 from utils.diag_directional_reliability import resolve_artifact
 bound=dict(pp['inputs']);bound.update({str(parent/n):files.digest(parent/n) for n in ('plan.json','summary.json')});cases=[]
 for c0 in pp['cases']:
  c=dict(c0);c['full_references']={}
  for arm,source in (('BASE','BASE'),('LEGACY_SCALAR','SHRINK')):
   record=Path(pp['a71_parent'])/'cases'/c['key']/source/'run.json'
   c['full_references'][arm]=resolve_artifact(record,'paired.pt',bound)
  cases.append(c)
 files.require(len(cases)==12 and all(c['sigma']==.05 for c in cases),'Same12 cases only')
 root.mkdir(parents=True)
 p=dict(experiment='A97_full_unroll',parent=str(parent),a71_parent=pp['a71_parent'],cases=cases,checkpoint=pp['checkpoint'],grid=pp['grid'],arms=list(rules.ARMS),
  sources={**pp['sources'],**{n:files.digest(PROJECTS/n) for n in NAMES}},inputs=bound,budget=rules.budget(),
  protocol=dict(scope='Same3 FFHQ x4tasks sigma_model=.05, fixed original30 grid. Identical completed nine-unroll learned entry; online controller at ALL30 positions in final unroll. No model/seed/noise/grid/LM-target tuning. No additional final sampling.',
   arms='BASE gamma1; LEGACY_SCALAR originalA71 current-state formula; REAL_SCALAR actual-response length; REAL_DIAG joint signed coefficient solve; DIAG_DIR_SCALAR_LEN and SCALAR_DIR_DIAG_LEN crossed vectors. Each arm recomputes BOTH candidate vectors on ITS OWN evolving state wherever needed. Never transplant A95/A96 step vectors or another arm trajectory.',
   objective='Unchanged A95 real-response target base+omega*(Py-base), omega=(1-M*sigma^2/||r||^2)+, physical coefficient p, full cross-response LM, MAX_OUTER20/MAX_TRIALS4/CG12/inexactRTOL.1/gradRTOL1e-3+ATOL1e-6. Diagonal initialized with same-state scalar only, no norm cap or gamma penalty. Nonconverged last accepted iterate explicitly retained as finite-budget algorithm.',
   endpoint='s>=0: actual clamped raw next denoiser. At final0->-1: m_-1(x)=x, actual returned clean-coordinate state, NOT alpha[-1]/network t=-1 or extra terminal denoising. Terminal H JVP/VJP counted as operator derivatives separately from NN derivatives.',
   degeneracy='q=0 implies only zero diagonal/native-ray correction, recorded ZERO_NATIVE_FEASIBLE_SET. Crossed zero-length vectors follow A96 semantics; undefined nonzero requested diagonal direction stops, no replacement policy.',
   pairing='Every arm starts original entry RNG and consumes original29 innovations only. Solver never draws sampling RNG. Per-step native call=t, all lookahead calls=s>=0; no negative-time NN. BASE/LEGACY outputs must match completed A71 paired tensors relativeL2<=1e-6; every tail RNG matches A92. No GT before all30 actions complete.',
   costs='Only last30 reconstructed here; 270 native forwards/nine optimization rounds inherited, not free method cost. Equivalent full reconstruction forward count=270+this arm actual NN calls. All response bases, JVP/VJP forwards, rejected trials and solver costs recorded; terminal operator derivatives separate. Timing is suffix + controller/checkpoint/bookkeeping, not independent full300 wall-clock.',
   checkpoint='Each successful position gets immutable step PT+SHA metadata. Resource pause ONLY between positions, with all finished steps verified on resume and same pinned GPU. Failed/interrupted in-progress position is retained, no automatic retry. All arrays/full solver traces stay server, lightweight step summaries and final PNG/JSON uploaded.',
   decision='Observe all six trajectories without GT arm/iteration selection. Primary joint vs original scalar and same-target scalar per task: >=+.1dB,>=2of3wins,SSIM nondecrease,LPIPS nonincrease. Cross arms report joint-vs-cross differences; evolving-state cross trajectories are POLICY ablations, not A96 fixed-state factorial effects. Record solver budget-hit rates and failures.',
   resources='Up to2 allocated GPUs>=8GiB, static case shards, one strict model per worker. Actual root/DPS checkpoint source fingerprint + real primal/adjoint probe required; fixed IEEE. No process termination, precision/resolution change, model download or old source edits. Whole queue detached-safe, live progress, explicit resource-limited/failed status.'))
 files.write(root/'plan.json',p);shutil.copyfile(PROJECTS/'experiment_cards/A97_full_unroll_response_20261106.md',root/'experiment_card.md')
 return p


def small_counts(count):return {k:v for k,v in count.items() if k not in ('calls','_rng')}


def arm_run(root,p,c,arm,model,peak,uuid):
 import torch
 from modules.main_single import load_image_as_tensor
 from modules.dataset_loader import tensor_to_image
 from utils import overall_direction_audit as native,paper_clock_runtime as clock
 from utils.response_target_runtime import rng_equal
 from utils.diag_shrink_multitask import make_operator
 from utils.diag_noise_feasible_zaps import metrics
 from utils.terminal_mask_counterfactual import region_errors
 load=lambda f:torch.load(f,map_location='cpu',weights_only=True)
 folder=root/'cases'/c['key']/arm;record=folder/'run.json'
 if record.exists():
  prior=files.read(record)
  if prior['status']=='complete':
   for name,sha in prior['artifacts'].items():files.require(files.digest(folder/name)==sha,'Output changed')
   return
  files.require(prior['status']=='resource_limited' and prior['device_uuid']==uuid,'Failed/interrupted arm or changed resume GPU retained')
 folder.mkdir(parents=True,exist_ok=True);steps=folder/'steps';steps.mkdir(exist_ok=True)
 done=sorted(steps.glob('step_*.json'));saved=None;trace=[]
 for k,path in enumerate(done):
  item=files.read(path);files.require(item['step']==k and item['status']=='complete','Nonconsecutive/incomplete checkpoint')
  tensor=path.with_suffix('.pt');files.require(files.digest(tensor)==item['sha256'],'Step checkpoint changed');trace.append(item['detail'])
  saved=tensor
 for attempt in steps.glob('attempt_*.json'):files.require(files.read(attempt)['status']=='complete','Interrupted/failed position retained; no automatic replay')
 start=time.perf_counter()
 entry=load(c['entry']);y=load(c['observation'])['y'].to(model.device);op=make_operator(Path(p['a71_parent']),c,model.device)
 z=native.make_z(model,op,entry,c['task']);state=load(saved) if saved else None
 if state:native.restore_rng(state,model.device);x=state['x'].to(model.device)
 else:native.restore_rng(entry,model.device);x=entry['initial'].to(model.device)
 row=dict(status='running',case=c,arm=arm,device_uuid=uuid,completed_steps=len(done),counts=small_counts(state['counts']) if state else {},suffix_seconds=state['elapsed_seconds'] if state else 0.)
 files.write(record,row);count={}
 def tick(value):
  row['counts']=small_counts(value);files.write(record,row)
  print('A97 PROGRESS:',c['key'],arm,'step',row['completed_steps'],'F',value['forwards'],'VJP',value['input_vjp_attempts'],'JVP',value.get('input_jvp_attempts',0),flush=True)
 try:
  with clock.counters(model.model,tick=tick) as count:
   if state:count.update(state['counts'])
   clock.count_operator(op,count)
   for k in range(len(done),30):
    if int(torch.cuda.mem_get_info()[0])+torch.cuda.memory_reserved()-torch.cuda.memory_allocated()<peak+files.GIB:raise ResourcePause('AD peak+1GiB lacks margin at step boundary')
    attempt=steps/f'attempt_{k:03d}.json';files.write(attempt,dict(status='running',step=k,counts=small_counts(count)))
    x,rng,detail,solver_traces,delta=rules.step(model,z,op,y,x,k,arm,count)
    elapsed=row['suffix_seconds']+time.perf_counter()-start
    checkpoint=steps/f'step_{k:03d}.pt';temporary=checkpoint.with_suffix('.tmp')
    torch.save(dict(x=x.detach().cpu(),delta=delta.cpu(),**rng,counts=dict(count),solver_traces=solver_traces,elapsed_seconds=elapsed),temporary);temporary.replace(checkpoint)
    files.write(checkpoint.with_suffix('.json'),dict(status='complete',step=k,sha256=files.digest(checkpoint),detail=detail,counts=small_counts(count)))
    files.write(attempt,dict(status='complete',step=k));trace.append(detail)
    row.update(completed_steps=k+1,counts=small_counts(count));files.write(record,row)
    print('A97 STEP:',c['key'],arm,k+1,'/30',detail.get('scalar',{}).get('status',detail.get('status')),detail.get('diagonal',{}).get('status',''),flush=True)
  torch.cuda.synchronize();output=x.detach().cpu();rng=native.rng_state(model.device)
  files.require(count['optimizer_backward_attempts']==0,'Unexpected optimizer backward')
  expected_rng=load(Path(c['references']['NATIVE_100']).parent/'final_rng.pt');files.require(rng_equal(rng,expected_rng),'Sampling RNG pairing failed')
  gate=None
  if arm in c['full_references']:
   expected=load(c['full_references'][arm]);error=float((output.double()-expected.double()).norm()/expected.double().norm().clamp(min=1e-30))
   files.require(error<=1e-6 and count['forwards']==30,'Full30 baseline/scalar replay mismatch');gate=dict(relative_L2=error,passed=True)
  row.update(suffix_seconds=row['suffix_seconds']+time.perf_counter()-start,counts=small_counts(count),replay_gate=gate)
  # Read GT only after every action, terminal update, and RNG/replay gate.
  gt=load_image_as_tensor(c['image']).cpu();m=metrics(output,gt)
  if c['task']=='inpainting':m.update(region_errors(output,gt,load(Path(p['a71_parent'])/'observations'/c['key']/'mask.pt')))
  torch.save(output,folder/'output.pt');torch.save(rng,folder/'final_rng.pt');files.write(folder/'trajectory.json',trace)
  tensor_to_image(output.squeeze(0),denormalize=True).save(folder/'output.png')
  row.update(status='complete',metrics=m,equivalent_full_method_forward=270+count['forward_attempts'],
   solver_stops={name:{status:sum(t.get(name,{}).get('status')==status for t in trace) for status in ('GRADIENT_TOLERANCE','OUTER_BUDGET','TRIAL_BUDGET')} for name in ('scalar','diagonal')},
   artifacts={n:files.digest(folder/n) for n in ('output.pt','final_rng.pt','trajectory.json','output.png')});files.write(record,row)
  print('A97 COMPLETE:',c['key'],arm,m,flush=True)
 except ResourcePause as error:
  row.update(status='resource_limited',reason=str(error),counts=small_counts(count));files.write(record,row);raise
 except BaseException as error:
  row.update(status='failed',error=repr(error),traceback=traceback.format_exc(),counts=small_counts(count));files.write(record,row);raise


def worker(root,index):
 allocation=files.read(root/'allocation.json')['devices'];uuid=allocation[index]['uuid'];os.environ['CUDA_VISIBLE_DEVICES']=uuid
 path=root/'workers'/str(index);path.mkdir(parents=True,exist_ok=True);status=dict(status='running',pid=os.getpid(),uuid=uuid);files.write(path/'session.json',status)
 try:
  p=check(root)
  import torch
  from unittest.mock import patch
  from modules.main_single import load_diffusion_model
  from utils import historical_ffhq_runtime as arithmetic,response_target_runtime as rt,a95_checkpoint_scope_r1 as repair
  from utils.a67_model_freeze import frozen_network
  from utils.diag_response_target_lm_r2 import loaded_sources
  torch.set_num_threads(2);model=load_diffusion_model('ffhq','cuda:0');model.model.eval()
  files.require(files.digest(p['checkpoint']['path'])==p['checkpoint']['sha256'],'Wrong checkpoint')
  weights=torch.load(p['checkpoint']['path'],map_location='cpu',weights_only=True);model.model.load_state_dict(weights,strict=True);del weights
  load_dir=path/'loads';load_dir.mkdir(exist_ok=True);lf=load_dir/f'{len(list(load_dir.glob("*.json"))):03d}.json'
  probe=dict(probe_F=0,probe_JVP=0,probe_VJP=0,actual_sources=loaded_sources(model),torch=str(torch.__version__),uuid=uuid)
  for filename,sha in probe['actual_sources'].items():files.require(p['inputs'].get(filename)==sha,'Actual loaded NN source differs from qualified A96 source')
  with frozen_network(model.model),arithmetic.precision('LEGACY_IEEE'),patch.object(rt,'forward_ad_checkpoint_scope',repair.forward_ad_checkpoint_scope):
   torch.cuda.reset_peak_memory_stats();files.write(lf,probe)
   try:probe['gate']=rt.checkpoint_probe(model,p['grid'][16],probe)
   finally:torch.cuda.synchronize();probe['peak_allocated_bytes']=torch.cuda.max_memory_allocated();files.write(lf,probe)
   for c in p['cases'][index::len(allocation)]:
    for arm in rules.ARMS:arm_run(root,p,c,arm,model,probe['peak_allocated_bytes'],uuid)
  status['status']='complete'
 except ResourcePause as error:status.update(status='resource_limited',reason=str(error))
 except BaseException as error:status.update(status='failed',error=repr(error),traceback=traceback.format_exc());print('A97 WORKER FAILED:',status,flush=True)
 finally:files.write(path/'session.json',status)


def summarize(root):
 p=check(root);rows=[files.read(f) for f in (root/'cases').glob('*/*/run.json')];done=[r for r in rows if r['status']=='complete'];groups=[];comparisons=[]
 for task in ('super_resolution','gaussian_deblur','inpainting','motion_deblur'):
  rs=[r for r in done if r['case']['task']==task]
  for arm in rules.ARMS:
   selected=[r for r in rs if r['arm']==arm]
   if selected:groups.append(dict(task=task,arm=arm,images=len(selected),quality={k:statistics.mean(r['metrics'][k] for r in selected) for k in ('psnr','ssim','lpips')},mean_suffix_seconds=statistics.mean(r['suffix_seconds'] for r in selected),
    mean_equivalent_full_NN=statistics.mean(r['equivalent_full_method_forward'] for r in selected),solver_stops=[r['solver_stops'] for r in selected]))
   if arm in ('BASE',):continue
   for reference in ('BASE','LEGACY_SCALAR','REAL_SCALAR','REAL_DIAG'):
    if reference==arm:continue
    by={r['case']['key']:r for r in rs if r['arm']==reference};pairs=[(by[r['case']['key']],r) for r in selected if r['case']['key'] in by]
    if pairs:
     delta={k:statistics.mean(b['metrics'][k]-a['metrics'][k] for a,b in pairs) for k in ('psnr','ssim','lpips')};wins=sum(b['metrics']['psnr']>a['metrics']['psnr'] for a,b in pairs)
     comparisons.append(dict(task=task,arm=arm,reference=reference,images=len(pairs),delta=delta,wins=wins,positive_development_candidate=len(pairs)==3 and wins>=2 and delta['psnr']>=.1 and delta['ssim']>=0 and delta['lpips']<=0))
 s=dict(experiment=p['experiment'],completed_runs=len(done),planned_runs=72,groups=groups,comparisons=comparisons,
  actual_new_cost={k:sum(r.get('counts',{}).get(k,0) for r in rows) for k in ('forward_attempts','forwards','input_vjp_attempts','input_jvp_attempts','optimizer_backward_attempts','terminal_operator_jvp_attempts','terminal_operator_vjp_attempts')},
  cost_is_lower_bound=any(r['status'] in ('failed','running') for r in rows),workers=[files.read(f) for f in (root/'workers').glob('*/session.json')],probe_loads=[files.read(f) for f in (root/'workers').glob('*/loads/*.json')],protocol=p['protocol'])
 files.write(root/'summary.json',s);print('A97 STATUS:',len(done),'/72',flush=True)
 if done:
  with (root/'comparison.csv').open('w',newline='') as stream:
   columns=['task','image','arm','psnr','ssim','lpips','suffix_seconds','equivalent_full_method_forward']
   writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
   for r in done:writer.writerow(dict(task=r['case']['task'],image=r['case']['image_id'],arm=r['arm'],**{k:r['metrics'][k] for k in ('psnr','ssim','lpips')},suffix_seconds=r['suffix_seconds'],equivalent_full_method_forward=r['equivalent_full_method_forward']))
 for r in groups:print('A97 GROUP:',r['task'],r['arm'],'n=',r['images'],r['quality'],'suffix_seconds=',round(r['mean_suffix_seconds'],3),'mean_full_NN=',r['mean_equivalent_full_NN'],flush=True)
 for r in comparisons:
  if r['arm']=='REAL_DIAG' and r['reference'] in ('BASE','LEGACY_SCALAR','REAL_SCALAR'):print('A97 JOINT_MINUS:',r['task'],r['reference'],r['delta'],flush=True)
 return s


def archive(root):
 from utils import archive_refinement_confirmation as arch
 arch.BRANCH=BRANCH;bundle,sha=arch.export_and_commit({'A97':root},projects=PROJECTS)
 print('Archive:',bundle,'Results commit:',sha,'Branch:',BRANCH,flush=True)
 subprocess.run(['git','push','origin',BRANCH],cwd=PROJECTS.parent,check=True);print('RESULTS_PUSH_PASS: A97 uploaded.',flush=True)


def coordinate(args,root,parent,state,record):
 with exclusive(state/'worker.lock'):
  status=dict(status='running',pid=os.getpid(),root=str(root));files.write(record,status)
  children=[];streams=[]
  try:
   prepare(root,parent)
   from utils.diag_paper_clock_alignment import gpu_inventory
   choices=files.available_gpus(gpu_inventory(),args.gpus,os.environ.get('CUDA_VISIBLE_DEVICES'))
   choices=sorted([r for r in choices if r['free_bytes']>=8*files.GIB],key=lambda r:-r['free_bytes'])
   allocation=root/'allocation.json'
   if allocation.exists():
    devices=files.read(allocation)['devices']
    if not all(any(c['uuid']==d['uuid'] for c in choices) for d in devices):raise ResourcePause('Pinned GPU(s) lack8GiB margin; resume only with same allocation')
   else:
    if not choices:raise ResourcePause('No allocated GPU>=8GiB; no model calls')
    devices=choices[:2];files.write(allocation,dict(devices=devices))
   command=[sys.executable,'-m','unittest','-v','utils.test_full_unroll_response','utils.test_full_unroll_response_runtime']
   result=subprocess.run(command,cwd=PROJECTS,capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':devices[0]['uuid']})
   print(result.stdout+result.stderr,end='',flush=True)
   files.write(root/'preflight_tests.json',dict(command=command,returncode=result.returncode,output=result.stdout+result.stderr));result.check_returncode()
   for index in range(len(devices)):
    folder=root/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True)
    prior=folder/'session.json'
    if prior.exists():files.require(files.read(prior)['status'] in ('resource_limited','complete'),'Failed/interrupted worker retained; no automatic retry')
    log=folder/'worker.log';stream=log.open('ab',buffering=0);streams.append(stream)
    child=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--root',str(root),'--worker-index',str(index)],cwd=PROJECTS,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
    children.append(dict(child=child,log=log,offset=log.stat().st_size,index=index))
   while children:
    alive=False
    for item in children:
     with item['log'].open('rb') as stream:stream.seek(item['offset']);chunk=stream.read();item['offset']=stream.tell()
     if chunk:print(chunk.decode(errors='replace'),end='',flush=True)
     if item['child'].poll() is None:alive=True
    if not alive:break
    time.sleep(1)
   statuses=[files.read(root/'workers'/str(i)/'session.json')['status'] for i in range(len(devices))]
   if any(c['child'].returncode for c in children) or any(s=='failed' for s in statuses):raise RuntimeError('A97 worker failed; logs/attempts retained, no automatic retry')
   status['status']='resource_limited' if 'resource_limited' in statuses else 'complete'
   if status['status']=='complete':files.require(sum(files.read(f)['status']=='complete' for f in (root/'cases').glob('*/*/run.json'))==72,'Worker completion without all72 runs')
  except ResourcePause as error:status.update(status='resource_limited',reason=str(error));print('A97 RESOURCE PAUSE:',error,flush=True)
  except BaseException as error:status.update(status='failed',error=repr(error),traceback=traceback.format_exc());print('A97 FAILED:',status,flush=True)
  finally:
   # Never export a moving directory or abandon a live shard on coordinator errors.
   for item in children:item['child'].wait()
   for stream in streams:stream.close()
   files.write(record,status)
   if (root/'plan.json').exists():
    shutil.copyfile(record,root/'session.json')
    try:summarize(root);archive(root)
    except Exception as error:print('A97 archive incomplete; use --archive-only:',repr(error),flush=True)


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=ROOT);parser.add_argument('--a96-root',default=PARENT);parser.add_argument('--gpus',default='0,1')
 parser.add_argument('--worker-index',type=int)
 for flag in ('execute','status','archive-only'):parser.add_argument('--'+flag,action='store_true')
 args=parser.parse_args();root=Path(args.root).resolve();parent=Path(args.a96_root).resolve();state=root.parent/(root.name+'_session');state.mkdir(parents=True,exist_ok=True);record=state/'session.json'
 if args.worker_index is not None:
  folder=root/'workers'/str(args.worker_index);folder.mkdir(parents=True,exist_ok=True)
  with exclusive(folder/'worker.lock'):worker(root,args.worker_index)
  if files.read(folder/'session.json')['status']=='failed':raise SystemExit(1)
  return
 if args.status:
  print(dict(active=busy(state/'worker.lock'),session=files.read(record) if record.exists() else None,workers=[files.read(f) for f in (root/'workers').glob('*/session.json')]))
  for path in sorted((root/'cases').glob('*/*/run.json')):
   row=files.read(path);print(row['case']['key'],row['arm'],row['status'],str(row['completed_steps'])+'/30',row['counts'])
  return
 if args.archive_only:files.require(not busy(state/'worker.lock'),'Worker active');summarize(root);archive(root);return
 if args.execute:coordinate(args,root,parent,state,record);return
 with exclusive(state/'client.lock'):
  child=None
  if not busy(state/'worker.lock'):
   if record.exists():
    prior=files.read(record);files.require(prior['status'] in ('complete','resource_limited'),'Prior failure/interruption retained; no automatic retry')
    if prior['status']=='complete':print('A97 already complete; use --status or --archive-only');return
   command=[sys.executable,'-u',str(Path(__file__)),'--execute','--root',str(root),'--a96-root',str(parent),'--gpus',args.gpus]
   with (state/'worker.log').open('ab',buffering=0) as log:child=subprocess.Popen(command,cwd=PROJECTS,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  offset=0
  while True:
   if (state/'worker.log').exists():
    with (state/'worker.log').open('rb') as log:log.seek(offset);chunk=log.read();offset=log.tell()
    if chunk:print(chunk.decode(errors='replace'),end='',flush=True)
   if not busy(state/'worker.lock') and (child is None or child.poll() is not None):break
   time.sleep(1)
  if record.exists():
   status=files.read(record);print('A97 SESSION:',status,flush=True)
   if status['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
