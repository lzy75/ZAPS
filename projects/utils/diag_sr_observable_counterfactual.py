"""A99 CPU-only exact-H final SR counterfactual. Runs while A97/A98 continue."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['MKL_THREADING_LAYER']='GNU';os.environ.pop('MKL_SERVICE_FORCE_INTEL',None)
import argparse,csv,shutil,subprocess,sys,time,traceback,statistics
from pathlib import Path
PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import historical_ffhq_gain as files,sr_observable_counterfactual as rules,diag_full_unroll_response as parent_driver
from utils.a54_session import exclusive,busy
ROOT='/data0/lzy/ZAPS-results/diag_sr_observable_counterfactual/A99_20261107'
PARENT=parent_driver.ROOT
BRANCH='results/zaps-sr-observable-counterfactual-20261107'
NAMES=('utils/sr_observable_counterfactual.py','utils/diag_sr_observable_counterfactual.py','utils/test_sr_observable_counterfactual.py',
 'utils/test_sr_observable_counterfactual_runtime.py','run_a99_20261107.sh','experiment_cards/A99_sr_observable_counterfactual_20261107.md')


def check(root):
 p=files.read(root/'plan.json');files.require(p['experiment']=='A99_sr_observable_counterfactual','Wrong A99 plan')
 for n,h in p['sources'].items():files.require(files.digest(PROJECTS/n)==h,'Source changed: '+n)
 for n,h in p['inputs'].items():files.require(files.digest(n)==h,'Input changed: '+n)
 return p


def prepare(root,parent):
 if (root/'plan.json').exists():return check(root)
 files.require(not root.exists() or not any(root.iterdir()),'Unplanned nonempty root retained')
 pp=parent_driver.check(parent);bound=dict(pp['inputs']);bound[str(parent/'plan.json')]=files.digest(parent/'plan.json');cases=[]
 for path in (PROJECTS/'modules/degradations.py',PROJECTS.parent/'DPS/util/resizer.py',PROJECTS/'utils/metrics.py',PROJECTS/'modules/wavelet.py'):
  bound[str(path)]=files.digest(path)
 # The live A97 summary/session is intentionally NOT a frozen input.
 for c0 in pp['cases']:
  if c0['task']!='super_resolution':continue
  c=dict(c0);c['outputs']={}
  for arm in ('REAL_SCALAR','REAL_DIAG'):
   path=parent/'cases'/c['key']/arm/'run.json';r=files.read(path);files.require(r['status']=='complete','Need complete same3 SR scalar/matrix outputs')
   output=path.parent/'output.pt';files.require(files.digest(output)==r['artifacts']['output.pt'],'A97 tensor changed')
   c['outputs'][arm]=dict(path=str(output),record=str(path));bound[str(path)]=files.digest(path);bound[str(output)]=r['artifacts']['output.pt']
  cases.append(c)
 files.require(len(cases)==3 and all(c['sigma']==.05 for c in cases),'Only original3 SR cases')
 root.mkdir(parents=True)
 p=dict(experiment='A99_sr_observable_counterfactual',parent=str(parent),cases=cases,a71_parent=pp['a71_parent'],
  sources={**pp['sources'],**{n:files.digest(PROJECTS/n) for n in NAMES}},inputs=bound,arms=list(rules.ARMS),
  budget=dict(diffusion_UNet=0,input_VJP=0,input_JVP=0,optimizer_backward=0,LPIPS_evaluation_calls=15,axes_SVD=2,offline_wavelet_analyses=3,offline_wavelet_syntheses=3),
  protocol=dict(scope='Original3 FFHQ SR4 sigma_model.05, completed A97 REAL_SCALAR and REAL_DIAG final outputs ONLY. No new reconstruction, trajectory, image, seed, model load, UNet or guidance solver. Can run CPU while other experiments continue; no waiting for all72.',
   operator='Extract actual DPS Resizer separable64x256 axes from frozen native forward taps, accumulating mirrored repeated indices. Float64 SVD yields exact-H orthogonal row/null projectors after full-row-rank, native-forward and orthogonality gates. NEVER use historical bicubic upsample as H adjoint/pseudoinverse. No CG, ridge or new noise threshold.',
   counterfactual='S=scalar final,D=matrix final,v=D-S,vR=HdaggerHv,vN=v-vR. Evaluate S,D,S+vR,S+vN and same-W LL2-only difference. ALL arms retained; no output/GT selection. H(S+vR)=H(D),H(S+vN)=H(S) in RAW model coordinates.',
   diagnosis='GT used only after ALL outputs are constructed. Full rawMSE=row-error+null-error exact identity. Record row/null excess of D vsS, row-only restoration, clampedPSNR/SSIM/LPIPS, raw and clamped-H errors. LL2 is a frequency-space comparator, never called the physical rowspace.',
   decision='Terminal null-risk contribution supported if3images,net aggregate rawMSE excess>0,null excess>0 and>50%of net excess, row-only rawMSE improves over D in>=2of3. Ratio can exceed1 if row component improves. Report each image/all metrics even if negative. No claim this proves the optimization cause or that a full30-step restricted policy works; that is A98.',
   fidelity='Native separable forward relativeL2<=1e-10 float64; native float32<=3e-6; row/null invariance<=1e-10 relative to original difference, orthogonal energy decomposition<=1e-10relative. Only numerical gates stop. Cached reference quality replay tolerance PSNR1e-4/SSIM1e-5/LPIPS1e-5.',
   resources='CPU2threads,CUDA hidden,pre-existing LPIPSalex weights only; no diffusion/model downloads or process termination. Parent completed records and bytes untouched. Report15LPIPS calls separately from zero diffusionNN. New directory/branch, no automatic failed retries, detached-safe logs and archive.'))
 files.write(root/'plan.json',p);shutil.copyfile(PROJECTS/'experiment_cards/A99_sr_observable_counterfactual_20261107.md',root/'experiment_card.md');return p


def execute(root,p):
 import numpy as np,torch
 from modules.main_single import load_image_as_tensor
 from modules.dataset_loader import tensor_to_image
 from modules.wavelet import OrthogonalDWT2D
 from utils.historical_ffhq_runtime import make_operator
 from utils.diag_noise_feasible_zaps import metrics
 torch.set_num_threads(2)
 files.require((Path(torch.hub.get_dir())/'checkpoints'/'alexnet-owt-7be5be79.pth').is_file(),'Existing LPIPS alexnet checkpoint required; no download')
 op=make_operator('cpu');ay,ax=rules.native_matrices(op.base.down_sample,(1,3,256,256));projector=rules.Projector(ay,ax)
 start=time.perf_counter();generator=np.random.default_rng(20261107);probe=generator.standard_normal((1,3,256,256))
 native=op.H(torch.from_numpy(probe)).numpy();dense=projector.H(probe)
 error=rules.relative(dense,native);files.require(error<=1e-10,'Native separableH double gate failed')
 error32=rules.relative(op.H(torch.from_numpy(probe).float()).numpy(),dense);files.require(error32<=3e-6,'Native float32H mismatch')
 row=projector.row(probe);null=probe-row
 files.require(rules.relative(projector.row(row),row)<=1e-10 and np.linalg.norm(projector.H(null))<=1e-10*max(np.linalg.norm(projector.H(probe)),1e-30),'Projector qualification failed')
 row_dimension=3*ay.shape[0]*ax.shape[0]
 files.write(root/'operator_gate.json',dict(passed=True,native_double_error=error,native_float_error=error32,axes=projector.info,
  row_dimension=row_dimension,null_dimension=3*256*256-row_dimension,seconds=time.perf_counter()-start,new_UNet=0))
 dwt=OrthogonalDWT2D('db4',3)
 load=lambda f:torch.load(f,map_location='cpu',weights_only=True)
 for c in p['cases']:
  folder=root/'cases'/c['key'];record=folder/'run.json'
  if record.exists():
   r=files.read(record);files.require(r['status']=='complete','Prior failed/interrupted case retained')
   for n,h in r['artifacts'].items():files.require(files.digest(folder/n)==h,'Output changed')
   continue
  folder.mkdir(parents=True);r=dict(status='running',case=c,LPIPS_evaluation_attempts=0);files.write(record,r)
  try:
   s=load(c['outputs']['REAL_SCALAR']['path']).double().numpy();d=load(c['outputs']['REAL_DIAG']['path']).double().numpy()
   outputs,visible,invisible=rules.counterfactuals(s,d,projector);difference=d-s;hd=projector.H(difference)
   denominator=max(float(np.linalg.norm(hd)),float(np.linalg.norm(projector.H(s))),1e-30)
   invariant_row=float(np.linalg.norm(projector.H(outputs['ROW_ONLY_DIFFERENCE'])-projector.H(d)))/denominator
   invariant_null=float(np.linalg.norm(projector.H(outputs['NULL_ONLY_DIFFERENCE'])-projector.H(s)))/denominator
   files.require(max(invariant_row,invariant_null)<=1e-10,'Counterfactual raw observation invariance failed')
   coeff=dwt.analysis(torch.from_numpy(difference));h,w=coeff.shape[-2:];coarse=torch.zeros_like(coeff);coarse[...,:h//4,:w//4]=coeff[...,:h//4,:w//4]
   outputs['LL2_ONLY_DIFFERENCE']=s+dwt.synthesis(coarse).numpy()
   # GT is loaded only after all five image constructions and numerical gates.
   gt=load_image_as_tensor(c['image']).double().numpy();y=load(c['observation'])['y'].double().numpy();truth_H=projector.H(gt)
   rows=[]
   for arm in rules.ARMS:
    image=outputs[arm];r['LPIPS_evaluation_attempts']+=1;files.write(record,r)
    m=metrics(torch.from_numpy(image).float(),torch.from_numpy(gt).float());risk=rules.risks(image,gt,projector)
    risk.update(raw_observation_true_mse=float(np.mean((projector.H(image)-truth_H)**2)),raw_observation_y_mse=float(np.mean((projector.H(image)-y)**2)),
     clamped_observation_true_mse=float(np.mean((projector.H(np.clip(image,-1,1))-truth_H)**2)))
    if arm in ('SCALAR_FINAL','DIAG_FINAL'):
     key='REAL_SCALAR' if arm=='SCALAR_FINAL' else 'REAL_DIAG';expected=files.read(c['outputs'][key]['record'])['metrics']
     files.require(all(abs(m[k]-expected[k])<=tol for k,tol in (('psnr',1e-4),('ssim',1e-5),('lpips',1e-5))),'Cached metric replay mismatch')
    tensor_to_image(torch.from_numpy(image).float().squeeze(0),denormalize=True).save(folder/(arm+'.png'))
    rows.append(dict(arm=arm,metrics=m,risk=risk))
    print('A99 CASE:',c['image_id'],arm,m,'row/nullMSE=',risk['row_mse'],risk['null_mse'],flush=True)
   by={v['arm']:v for v in rows};scalar=by['SCALAR_FINAL']['risk'];diag=by['DIAG_FINAL']['risk'];row_only=by['ROW_ONLY_DIFFERENCE']['risk']
   excess=dict(raw_excess=diag['raw_mse']-scalar['raw_mse'],row_excess=diag['row_mse']-scalar['row_mse'],null_excess=diag['null_mse']-scalar['null_mse'],
    row_only_raw_improvement_over_diag=diag['raw_mse']-row_only['raw_mse'])
   files.require(abs(excess['raw_excess']-excess['row_excess']-excess['null_excess'])<=1e-10*max(diag['raw_mse'],scalar['raw_mse'],1e-12),'Excess-error identity failed')
   r.update(status='complete',rows=rows,excess=excess,gates=dict(passed=True,row_only_observation_error=invariant_row,null_only_observation_error=invariant_null),
    difference_energy=dict(total=float(np.sum(difference**2)),row=float(np.sum(visible**2)),null=float(np.sum(invisible**2))),
    artifacts={arm+'.png':files.digest(folder/(arm+'.png')) for arm in rules.ARMS});files.write(record,r)
  except BaseException as e:r.update(status='failed',error=repr(e),traceback=traceback.format_exc());files.write(record,r);raise


def summarize(root):
 p=check(root);records=[files.read(f) for f in (root/'cases').glob('*/run.json')];done=[r for r in records if r['status']=='complete'];groups=[]
 for arm in rules.ARMS:
  rows=[next(v for v in r['rows'] if v['arm']==arm) for r in done]
  if rows:groups.append(dict(arm=arm,images=len(rows),metrics={k:statistics.mean(r['metrics'][k] for r in rows) for k in ('psnr','ssim','lpips')},
   risk={k:statistics.mean(r['risk'][k] for r in rows) for k in rows[0]['risk']}))
 s=dict(experiment=p['experiment'],classification='TERMINAL_EXACT_H_COUNTERFACTUAL_ONLY',completed_cases=len(done),planned_cases=3,groups=groups,
  attribution=rules.decision([r['excess'] for r in done]),diffusion_UNet=0,input_VJP=0,input_JVP=0,optimizer_backward=0,
  LPIPS_evaluation_attempts=sum(r['LPIPS_evaluation_attempts'] for r in records),cost_is_lower_bound=any(r['status']!='complete' for r in records),protocol=p['protocol'])
 files.write(root/'summary.json',s)
 with (root/'comparison.csv').open('w',newline='') as stream:
  writer=csv.writer(stream);writer.writerow(['arm','images','PSNR','SSIM','LPIPS','rawMSE','rowMSE','nullMSE'])
  for r in groups:writer.writerow([r['arm'],r['images'],*[r['metrics'][k] for k in ('psnr','ssim','lpips')],*[r['risk'][k] for k in ('raw_mse','row_mse','null_mse')]])
 print('A99 COMPLETED:',len(done),'/3; UNet=0; LPIPS=',s['LPIPS_evaluation_attempts'],flush=True)
 for r in groups:print('A99 GROUP:',r['arm'],r['metrics'],flush=True)
 print('A99 ATTRIBUTION:',s['attribution'],flush=True);return s


def archive(root):
 from utils import archive_refinement_confirmation as arch
 arch.BRANCH=BRANCH;bundle,sha=arch.export_and_commit({'A99':root},projects=PROJECTS)
 print('Archive:',bundle,'Results commit:',sha,'Branch:',BRANCH,flush=True)
 subprocess.run(['git','push','origin',BRANCH],cwd=PROJECTS.parent,check=True);print('RESULTS_PUSH_PASS: A99 uploaded.',flush=True)


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=ROOT);parser.add_argument('--a97-root',default=PARENT)
 for flag in ('execute','status','archive-only'):parser.add_argument('--'+flag,action='store_true')
 args=parser.parse_args();root=Path(args.root).resolve();parent=Path(args.a97_root).resolve();state=root.parent/(root.name+'_session');state.mkdir(parents=True,exist_ok=True);record=state/'session.json'
 if args.status:print(dict(active=busy(state/'worker.lock'),session=files.read(record) if record.exists() else None));return
 if args.archive_only:files.require(not busy(state/'worker.lock'),'Worker active');summarize(root);archive(root);return
 if args.execute:
  with exclusive(state/'worker.lock'):
   status=dict(status='running',pid=os.getpid(),root=str(root));files.write(record,status)
   try:
    p=prepare(root,parent)
    command=[sys.executable,'-m','unittest','-v','utils.test_sr_observable_counterfactual','utils.test_sr_observable_counterfactual_runtime']
    result=subprocess.run(command,cwd=PROJECTS,capture_output=True,text=True);print(result.stdout+result.stderr,end='',flush=True)
    files.write(root/'preflight_tests.json',dict(command=command,returncode=result.returncode,output=result.stdout+result.stderr));result.check_returncode()
    execute(root,p);status['status']='complete'
   except BaseException as e:status.update(status='failed',error=repr(e),traceback=traceback.format_exc());print('A99 FAILED:',status,flush=True)
   finally:
    files.write(record,status)
    if (root/'plan.json').exists():
     shutil.copyfile(record,root/'session.json')
     try:summarize(root);archive(root)
     except Exception as e:print('A99 archive incomplete; --archive-only:',repr(e),flush=True)
  return
 # Multiple terminals may follow the same detached job; none mutate run state.
 with exclusive(state/'launch.lock'):
  child=None
  if not busy(state/'worker.lock'):
   if record.exists():
    previous=files.read(record);files.require(previous['status']=='complete','Prior failure/interruption retained; no automatic retry')
    print('A99 already complete; --status / --archive-only');return
   command=[sys.executable,'-u',str(Path(__file__)),'--execute','--root',str(root),'--a97-root',str(parent)]
   with (state/'worker.log').open('ab',buffering=0) as stream:child=subprocess.Popen(command,cwd=PROJECTS,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
   for _ in range(100):
    if busy(state/'worker.lock') or child.poll() is not None:break
    time.sleep(.05)
 offset=0
 while True:
  if (state/'worker.log').exists():
   with (state/'worker.log').open('rb') as stream:stream.seek(offset);chunk=stream.read();offset=stream.tell()
   if chunk:print(chunk.decode(errors='replace'),end='',flush=True)
  if not busy(state/'worker.lock') and (child is None or child.poll() is not None):break
  time.sleep(.5)
 if record.exists():
  status=files.read(record);print('A99 SESSION:',status,flush=True)
  if status['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
