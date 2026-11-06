"""A93 immutable A92 reuse: zero-NN capacity audit, then three-update feedback."""
import os
os.environ['MKL_THREADING_LAYER']='GNU';os.environ.pop('MKL_SERVICE_FORCE_INTEL',None)
import argparse,subprocess,sys,time,traceback,csv,statistics,shutil,json
from pathlib import Path
PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import historical_ffhq_gain as files
from utils import direction_window as rules
from utils import diag_overall_direction_audit as parent_driver
from utils.a54_session import exclusive,busy
ROOT='/data0/lzy/ZAPS-results/diag_direction_window/A93_20261106'
PARENT=parent_driver.ROOT
BRANCH='results/zaps-direction-capacity-window-20261106'
NAMES=('utils/direction_window.py','utils/diag_direction_window.py','utils/test_direction_window.py',
 'utils/test_direction_window_runtime.py','run_a93_20261106.sh','experiment_cards/A93_direction_capacity_window_20261106.md')
class ResourcePause(RuntimeError):pass


def check(root):
    p=files.read(root/'plan.json');files.require(p['experiment']=='A93_direction_capacity_window','Wrong A93 plan')
    for name,sha in p['sources'].items():files.require(files.digest(PROJECTS/name)==sha,'Source changed: '+name)
    for name,sha in p['inputs'].items():files.require(files.digest(name)==sha,'Bound input changed: '+name)
    return p


def prepare(root,parent):
    if (root/'plan.json').exists():return check(root)
    files.require(not root.exists() or not any(root.iterdir()),'Unplanned nonempty root retained')
    pp=parent_driver.check(parent);s=files.read(parent/'summary.json')
    files.require(s['completed_runs']==s['planned_runs']==216 and s['unresolved_runs']==0,'Complete A92 required')
    bound=dict(pp['inputs']);bound.update({str(parent/n):files.digest(parent/n) for n in ('plan.json','summary.json')})
    cases=[]
    for c0 in pp['cases']:
        c=dict(c0);folder=parent/'cases'/c['key'];prefix=folder/'prefix.json';r=files.read(prefix)
        files.require(r['status']=='complete' and r['plan_sha256']==files.digest(parent/'plan.json'),'A92 prefix not qualified')
        bound[str(prefix)]=files.digest(prefix);c['a92_bundle']=str(folder/'state.pt')
        files.require(files.digest(c['a92_bundle'])==r['artifacts']['state.pt'],'A92 state changed')
        bound[c['a92_bundle']]=r['artifacts']['state.pt'];c['references']={}
        for arm in ('NATIVE_100','DIAG_ALIGN_100','PHYS_DIRECT_100'):
            path=folder/arm/'run.json';row=files.read(path);files.require(row['status']=='complete','A92 reference incomplete')
            bound[str(path)]=files.digest(path);c['references'][arm]=str(path)
            for name in ('output.pt','next_clean.pt','final_rng.pt'):
                f=path.parent/name;files.require(files.digest(f)==row['artifacts'][name],'A92 reference artifact changed')
                bound[str(f)]=row['artifacts'][name]
        scalar=Path(pp['parent'])/'cases'/c['key']/'SHRINK/run.json'
        if scalar.exists():
            sr=files.read(scalar);files.require(sr['status']=='complete','Historical scalar incomplete')
            bound[str(scalar)]=files.digest(scalar);c['scalar_reference']=str(scalar)
        cases.append(c)
    files.require(len(cases)==24,'Original fixed24 cases only');root.mkdir(parents=True)
    p=dict(experiment='A93_direction_capacity_window',parent=str(parent),a71_parent=pp['parent'],cases=cases,
      checkpoint=pp['checkpoint'],inputs=bound,sources={**pp['sources'],**{n:files.digest(PROJECTS/n) for n in NAMES}},
      grid=pp['intervention']['fixed_grid'],arms=list(rules.ARMS),
      budget=dict(new_forwards=768,new_input_vjp=96,new_optimizer_backward=0,
        forwards_per_arm=16,input_vjp_per_arm=2,new_arms=48,
        geometry_UNet=0,geometry_input_vjp=0,geometry_DWT_analysis=48,reused_A92_F=3432,
        warmup_F_VJP_per_model_load=1),
      protocol=dict(scope='Same3 OLD FFHQ x4tasks x sigma.05/.10, original A71 learned state and A92 fixed intervention. No new image/seed/noise/frequency search.',
       geometry='ZeroNN on A92 q/g: grouped10 nonnegative vs grouped10 signed vs coefficientwise nonnegative vs coefficientwise signed. Formal unbounded representation capacities only, not reconstruction policies. Report sign-opposed and near-zero-source energy and formal gain tails.',
       window='Same current physical-gradient reference and original db4 grouped10 direction as A92. Fixed length1 times CURRENT native q norm. Apply at last-unroll update indices15,16,17 then native. Refresh current q and actual next-denoiser gradient in each arm; direction lag forbidden. No0.5/1.5 scan, adaptive time grid, noise/SURE controller or curvature estimation.',
       reuse='First A92 reference/direction/q/RNG reused with SHA. A92 original/one-step results used as controls, no new baseline sampling. First clean output must match corresponding A92 one-step next_clean tensor relativeL2<=1e-6. FinalCPU/CUDA RNG matches A92 baseline.',
       costs='Per new arm14 native continuationF+2 lookaheadF+2 inputVJP=16F/2VJP, twoarms24cases=768F/96VJP. Original first reference and prefix not charged as NEW. Model load/warmup, geometry CPU/DWT, continuation sampling/extra probes, metrics/writes separately logged. No complete300-wallclock claim.',
       decision='Each task/noise compare three-update to BOTH same-direction one-update and native ZAPS: PSNRmean>=+.1dB,>=2of3wins,SSIM nondecrease,LPIPS nonincrease. Include same-panel A71 scalar metrics as historical descriptive comparator only. No best-arm or label-based selection.',
       resources='AllocatedGPU>=6GiB and actual checkpoint inputVJP peak+1GiB before attempts; no process kill/precision change. Geometry can finish beforeGPU admission. Unresolved zero directions retain charged partial attempts, no policy fallback.',
       failures='CPU/Torch required tests; missingTorch fails. Exact old source/input hashes; any numerical/replay/RNG/count failure retains record and stops, no automatic failed retry. Resource pause only before attempts.',
       boundary='Mechanism diagnosis only. Larger cosine not proof of quality, coefficient diagonal formal bounds may require huge gains, signed coefficient freedom can almost reproduce any reference when q nonzero. No innovation/generalization/uncertainty guarantee. GT final/region metrics only.'))
    files.write(root/'plan.json',p);return p


def geometry_phase(root,p):
    import torch
    from modules.wavelet import OrthogonalDWT2D
    dwt=OrthogonalDWT2D('db4',3)
    for c in p['cases']:
        path=root/'geometry'/(c['key']+'.json')
        if path.exists():files.require(files.read(path)['status']=='complete','Incomplete geometry retained');continue
        row=dict(status='running',case=c,UNet=0,input_vjp=0);files.write(path,row)
        try:
            start=time.perf_counter();bundle=torch.load(c['a92_bundle'],map_location='cpu',weights_only=True)
            row.update(rules.geometry(bundle,dwt),seconds=time.perf_counter()-start,status='complete');files.write(path,row)
        except BaseException as error:
            row.update(status='failed',error=repr(error),traceback=traceback.format_exc());files.write(path,row);raise
    print('A93 GEOMETRY COMPLETE:24 cached states;0 new NN/VJP.',flush=True)


def execute(root,p):
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
    load=lambda f:torch.load(f,map_location='cpu',weights_only=True)
    files.require(files.digest(p['checkpoint']['path'])==p['checkpoint']['sha256'],'Wrong checkpoint')
    torch.set_num_threads(2);start=time.perf_counter();model=load_diffusion_model('ffhq','cuda:0');model.model.eval()
    weights=load(p['checkpoint']['path']);model.model.load_state_dict(weights,strict=True);del weights
    torch.cuda.synchronize();loads=root/'loads';loads.mkdir(exist_ok=True)
    loading=dict(model_load_seconds=time.perf_counter()-start,warmup_forward_attempts=0,warmup_input_vjp_attempts=0)
    loadfile=loads/(str(len(list(loads.glob('*.json')))).zfill(3)+'.json')
    with frozen_network(model.model),precision.precision('LEGACY_IEEE'):
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        with input_vjp_parameters(model.model),torch.enable_grad():
            loading['warmup_forward_attempts']=1;files.write(loadfile,loading)
            inp,raw=raw_mean(model,torch.zeros(1,3,256,256,device=model.device),p['grid'][16],graph=True)
            loading['warmup_input_vjp_attempts']=1;files.write(loadfile,loading)
            grad=torch.autograd.grad(raw.square().mean(),inp)[0]
            files.require(bool(torch.isfinite(grad).all()),'Warmup VJP nonfinite')
        del inp,raw,grad;torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()
        loading.update(warmup_seconds=time.perf_counter()-start,peak_allocated_bytes=peak);files.write(loadfile,loading)
        for c in p['cases']:
            for arm in rules.ARMS:
                folder=root/'cases'/c['key']/arm;record=folder/'run.json'
                if record.exists():
                    prior=files.read(record);files.require(prior['status'] in ('complete','unresolved'),'Prior failed attempt retained')
                    for n,h in prior.get('artifacts',{}).items():files.require(files.digest(folder/n)==h,'Output artifact changed')
                    continue
                free=int(torch.cuda.mem_get_info()[0]);reusable=torch.cuda.memory_reserved()-torch.cuda.memory_allocated()
                if free+reusable<peak+files.GIB:raise ResourcePause('Checkpoint VJP peak+1GiB lacks margin before next arm')
                folder.mkdir(parents=True);row=dict(status='running',case=c,arm=arm,counts={});files.write(record,row);count={}
                try:
                    with clock.counters(model.model) as count:
                        def operation():
                            bundle=load(c['a92_bundle']);entry=load(c['entry']);y=load(c['observation'])['y'].to(model.device)
                            op=make_operator(Path(p['a71_parent']),c,model.device,count)
                            return rules.run(model,op,y,entry,bundle,c['task'],arm,count)
                        (output,first,rng,trace),seconds=clock.clocks(torch.cuda.synchronize,operation)
                    files.require(count['forwards']==count['forward_attempts']==16 and count['calls']==rules.expected_calls(p['grid']),'Window call budget/grid changed')
                    files.require(count['input_vjp_attempts']==count.get('input_vjp_completed')==2 and count['optimizer_backward_attempts']==0,'Window backward count mismatch')
                    refpath=Path(c['references'][rules.REFERENCES[arm]])
                    expected=load(refpath.parent/'next_clean.pt')
                    error=float((first.double()-expected.double()).norm()/expected.double().norm().clamp(min=1e-30))
                    files.require(error<=1e-6,'Shared first intervention changed')
                    expected_rng=load(Path(c['references']['NATIVE_100']).parent/'final_rng.pt')
                    files.require(torch.equal(rng['cpu_rng'],expected_rng['cpu_rng']) and len(rng['cuda_rng'])==len(expected_rng['cuda_rng']) and all(torch.equal(a,b) for a,b in zip(rng['cuda_rng'],expected_rng['cuda_rng'])),'Future innovation RNG changed')
                    row.update(counts=count,suffix_seconds=seconds,first_clean_relative_error=error);files.write(record,row)
                    start=time.perf_counter();gt=load_image_as_tensor(c['image']).cpu();m=metrics(output,gt)
                    if c['task']=='inpainting':m.update(region_errors(output,gt,load(Path(p['a71_parent'])/'observations'/c['key']/'mask.pt')))
                    row['metrics_seconds']=time.perf_counter()-start;start=time.perf_counter()
                    torch.save(output,folder/'output.pt');torch.save(rng,folder/'final_rng.pt')
                    tensor_to_image(output.squeeze(0),denormalize=True).save(folder/'output.png');files.write(folder/'trace.json',trace)
                    row.update(status='complete',metrics=m,artifact_seconds=time.perf_counter()-start,
                       artifacts={n:files.digest(folder/n) for n in ('output.pt','final_rng.pt','output.png','trace.json')})
                    files.write(record,row);print('A93 COMPLETE:',c['key'],arm,'F=16 VJP=2 PSNR=',m['psnr'],flush=True)
                except rules.UnresolvedDirection as error:
                    row.update(status='unresolved',reason=str(error),counts=count);files.write(record,row)
                except BaseException as error:
                    row.update(status='failed',error=repr(error),traceback=traceback.format_exc(),counts=count);files.write(record,row);raise


def summarize(root,verbose=False):
    p=check(root);geometry=[files.read(f) for f in (root/'geometry').glob('*.json')]
    rows=[files.read(f) for f in (root/'cases').glob('*/*/run.json')];done=[r for r in rows if r['status']=='complete']
    groups=[];comparisons=[];capacity=[]
    for task in ('super_resolution','gaussian_deblur','inpainting','motion_deblur'):
        for sigma in (.05,.10):
            gs=[r for r in geometry if r['status']=='complete' and r['case']['task']==task and r['case']['sigma']==sigma]
            if gs:capacity.append(dict(task=task,sigma=sigma,images=len(gs),**{k:statistics.mean(r[k] for r in gs) for k in ('cosine_band_nonnegative','cosine_band_signed','cosine_coefficient_nonnegative','cosine_coefficient_signed','reference_energy_opposite_sign_fraction','reference_energy_tiny_source_fraction')}))
            for arm in rules.ARMS:
                rs=[r for r in done if r['arm']==arm and r['case']['task']==task and r['case']['sigma']==sigma]
                if not rs:continue
                groups.append(dict(task=task,sigma=sigma,arm=arm,images=len(rs),quality={k:statistics.mean(r['metrics'][k] for r in rs) for k in ('psnr','ssim','lpips')},mean_suffix_seconds=statistics.mean(r['suffix_seconds'] for r in rs)))
                for baseline in ('NATIVE_100',rules.REFERENCES[arm]):
                    paired=[(files.read(r['case']['references'][baseline]),r) for r in rs]
                    delta={k:statistics.mean(b['metrics'][k]-a['metrics'][k] for a,b in paired) for k in ('psnr','ssim','lpips')}
                    wins=sum(b['metrics']['psnr']>a['metrics']['psnr'] for a,b in paired)
                    comparisons.append(dict(task=task,sigma=sigma,arm=arm,reference=baseline,cases=len(rs),delta=delta,images_PSNR_positive=wins,positive_development_candidate=len(rs)==3 and delta['psnr']>=.1 and wins>=2 and delta['ssim']>=0 and delta['lpips']<=0))
    historical=[]
    for c in p['cases']:
        path=Path(c['scalar_reference']) if c.get('scalar_reference') else None
        if path is not None:
            r=files.read(path)
            if r['status']=='complete' and 'metrics' in r:historical.append(dict(case=c['key'],metrics=r['metrics'],scope='Historical full last30 scalar, descriptive; not matched window cost'))
    s=dict(experiment=p['experiment'],classification='CACHED_CAPACITY_AND_THREE_UPDATE_WINDOW_DIAGNOSIS',
       geometry_complete=sum(r['status']=='complete' for r in geometry),completed_runs=len(done),planned_runs=48,
       unresolved_runs=sum(r['status']=='unresolved' for r in rows),capacity=capacity,groups=groups,comparisons=comparisons,
       new_diagnostic_forward_attempts=sum(r.get('counts',{}).get('forward_attempts',0) for r in rows),
       new_input_vjp_attempts=sum(r.get('counts',{}).get('input_vjp_attempts',0) for r in rows),
       optimizer_backward_attempts=sum(r.get('counts',{}).get('optimizer_backward_attempts',0) for r in rows),
       geometry_new_NN=0,geometry_new_VJP=0,geometry_seconds=sum(r.get('seconds',0) for r in geometry),
       cost_is_lower_bound=any(r['status'] not in ('complete','unresolved') for r in rows+geometry),
       loads=[files.read(f) for f in (root/'loads').glob('*.json')],historical_A71_scalar=historical,protocol=p['protocol'])
    files.write(root/'summary.json',s)
    if done:
        table=[dict(task=r['case']['task'],sigma=r['case']['sigma'],image_id=r['case']['image_id'],arm=r['arm'],**r['metrics'],suffix_seconds=r['suffix_seconds']) for r in done]
        keys=list(dict.fromkeys(k for r in table for k in r))
        with (root/'comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(table)
    print('A93 STATUS:',dict(geometry=s['geometry_complete'],complete=len(done),planned=48,unresolved=s['unresolved_runs'],newF=s['new_diagnostic_forward_attempts'],newVJP=s['new_input_vjp_attempts']),flush=True)
    for r in comparisons:
        if r['reference']==rules.REFERENCES[r['arm']]:print('A93 THREE_MINUS_ONE:',r['task'],r['sigma'],r['arm'],{k:round(v,5) for k,v in r['delta'].items()},'wins',r['images_PSNR_positive'],flush=True)
    print('Full machine-readable summary:',root/'summary.json',flush=True)
    return s


def archive(root):
    from utils import archive_refinement_confirmation as arch
    arch.BRANCH=BRANCH;bundle,sha=arch.export_and_commit({'A93':root},projects=PROJECTS)
    print('Archive:',bundle,'Results commit:',sha,'Branch:',BRANCH,flush=True)
    subprocess.run(['git','push','origin',BRANCH],cwd=PROJECTS.parent,check=True);print('RESULTS_PUSH_PASS: A93 uploaded.',flush=True)


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',default=ROOT);a.add_argument('--a92-root',default=PARENT);a.add_argument('--gpus',default='0,1')
    for flag in ('execute','status','archive-only','geometry-only'):a.add_argument('--'+flag,action='store_true')
    args=a.parse_args();root=Path(args.root).resolve();parent=Path(args.a92_root).resolve();state=root.parent/(root.name+'_session');state.mkdir(parents=True,exist_ok=True);record=state/'session.json'
    if args.status:
        print(dict(active=busy(state/'worker.lock'),session=files.read(record) if record.exists() else None))
        if (root/'summary.json').exists():
            saved=files.read(root/'summary.json');print({k:saved[k] for k in ('geometry_complete','completed_runs','planned_runs','unresolved_runs','new_diagnostic_forward_attempts','new_input_vjp_attempts')})
        return
    if args.archive_only:files.require(not busy(state/'worker.lock'),'Active worker');summarize(root);archive(root);return
    if args.execute:
        with exclusive(state/'worker.lock'):
            status=dict(status='running',pid=os.getpid(),root=str(root));files.write(record,status)
            try:
                p=prepare(root,parent)
                command=[sys.executable,'-m','unittest','-v','utils.test_direction_window','utils.test_direction_window_runtime','utils.test_current_score_checkpoint_r2']
                result=subprocess.run(command,cwd=PROJECTS,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                print(result.stdout,end='',flush=True);files.write(root/'preflight_tests.json',dict(command=command,returncode=result.returncode,output=result.stdout));result.check_returncode()
                geometry_phase(root,p)
                if args.geometry_only:status['status']='geometry_complete'
                else:
                    from utils.diag_paper_clock_alignment import gpu_inventory
                    choices=files.available_gpus(gpu_inventory(),args.gpus,os.environ.get('CUDA_VISIBLE_DEVICES'))
                    choices=sorted([r for r in choices if r['free_bytes']>=6*files.GIB],key=lambda r:-r['free_bytes'])
                    if not choices:raise ResourcePause('No allocated GPU>=6GiB; zeroNN geometry already saved')
                    if (root/'allocation.json').exists():
                        allocated=files.read(root/'allocation.json')
                        if not any(r['uuid']==allocated['uuid'] for r in choices):raise ResourcePause('Pinned GPU lacks margin')
                    else:allocated=choices[0];files.write(root/'allocation.json',allocated)
                    os.environ['CUDA_VISIBLE_DEVICES']=allocated['uuid'];execute(root,p);status['status']='complete'
            except ResourcePause as error:status.update(status='resource_limited',reason=str(error));print('A93 RESOURCE PAUSE:',error,flush=True)
            except BaseException as error:status.update(status='failed',error=repr(error),traceback=traceback.format_exc());print('A93 FAILED:',repr(error),flush=True)
            finally:
                files.write(record,status)
                if (root/'plan.json').exists():
                    shutil.copyfile(record,root/'session.json')
                    try:summarize(root);archive(root)
                    except Exception as error:print('A93 archive incomplete; --archive-only:',repr(error),flush=True)
        return
    with exclusive(state/'client.lock'):
        child=None
        if not busy(state/'worker.lock'):
            if record.exists():
                previous=files.read(record);files.require(previous['status']!='failed','Prior failure retained; no automatic retry')
                if previous['status']=='complete' or (args.geometry_only and previous['status']=='geometry_complete'):print('Already complete');return
            command=[sys.executable,'-u',str(Path(__file__)),'--execute','--root',str(root),'--a92-root',str(parent),'--gpus',args.gpus]
            if args.geometry_only:command.append('--geometry-only')
            with (state/'worker.log').open('ab',buffering=0) as log:child=subprocess.Popen(command,cwd=PROJECTS,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        offset=0
        while True:
            if (state/'worker.log').exists():
                with (state/'worker.log').open(errors='replace') as f:f.seek(offset);text=f.read();offset=f.tell()
                if text:print(text,end='',flush=True)
            if not busy(state/'worker.lock') and (child is None or child.poll() is not None):break
            time.sleep(1)
        if record.exists():
            row=files.read(record);print('A93 SESSION:',row,flush=True)
            if row['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
