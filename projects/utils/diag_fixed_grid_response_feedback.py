"""A32 fixed-grid online response feedback: five 300-NFE arms, 1800 cached-prefix diagnostic calls."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np

PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS))
from utils import diag_state_action_response as source
from utils import fixed_grid_response_feedback as rules
P=source.parent
ROOT='/data0/lzy/ZAPS-results/diag_fixed_grid_response_feedback/A32_20261021'
CARD=PROJECTS/'experiment_cards/A32_fixed_grid_response_feedback_20261021.md'


def prepare(args):
    root=Path(args.root).resolve();old=Path(args.a22_dir).resolve();p=source.load_signed(old/'plan.json')
    source.load_signed(old/'summary.json')
    if root.is_relative_to(old) or old.is_relative_to(root):raise ValueError('Output overlaps A22')
    if len(p['cases'])!=12 or {s['case']['key'] for s in p['cases']}!={f'{i:05d}_s{s}' for i in range(20,26) for s in (1000,2000)}:raise ValueError('Need all twelve original A22 cases')
    identities=dict(p['identity_files'])
    for s in p['cases']:
        P.verify_case_sources(p,s)
        if P.load_record(old,s) is None:raise ValueError('Incomplete A22')
        for name in ('run.json','run.sha256','outputs.pt'):
            path=old/s['case']['key']/name;identities[str(path)]=source.digest(path)
    for name in ('plan.json','plan.sha256','summary.json','summary.sha256'):
        path=old/name;identities[str(path)]=source.digest(path)
    own=[Path(__file__),Path(rules.__file__),Path(source.__file__),Path(source.fields.__file__),
         Path(source.fields.policy.__file__),Path(P.coupled.__file__),Path(P.terminal.__file__),
         PROJECTS/'modules/wavelet.py',PROJECTS/'modules/zaps_algorithm.py',PROJECTS/'utils/metrics.py',
         CARD,PROJECTS/'run_a32_20261021.sh',PROJECTS/'utils/test_fixed_grid_response_feedback.py']
    workers=dict(p['worker_identity_files'])
    for path in own:workers[str(path)]=source.digest(path)
    for path,sha in {**identities,**workers}.items():
        if source.digest(path)!=sha:raise ValueError('Frozen source/worker differs: '+path)
    plan=dict(experiment='A32_fixed_grid_response_feedback',source_A22=str(old),cases=p['cases'],
        config=p['config'],grid=p['grid'],checkpoint_sha256=p['checkpoint_sha256'],runtime=p['runtime'],
        native_policy=p['native_policy'],identity_files=identities,worker_identity_files=workers,
        arms=list(rules.ARMS),gammas=list(rules.GAMMAS),minimum_predicted_improvement=.01,
        planned_method_nfe=0,planned_diagnostic_nfe=1800,deployment_nfe_per_arm=300,
        sensor_extra_unet=0,prefix_reused_nfe=270,GT_online=False,
        scope='Six reused development images, final unroll only; unmasked same-t wavelet response SURROGATE, not true next-state Jacobian or curvature.')
    if root.exists():
        if source.load_signed(root/'plan.json')!=plan:raise ValueError('Existing plan differs; preserve originals')
    else:
        root.parent.mkdir(parents=True,exist_ok=True);P.coupled.archive.recovery.require_space(root.parent)
        root.mkdir();source.signed(root/'plan.json',plan);(root/'experiment_card.md').write_text(CARD.read_text())
    print('A32 PREPARE PASS: fixed 30 points; 12 cases x5 arms x30 calls=1800 diagnostic. All scientific arms=300; sensor extra UNet=0.',flush=True)


def setup(plan,spec,model,device):
    import torch
    from modules.zaps_algorithm import ZAPS
    from modules.degradations import get_operator
    from utils.diag_ffhq_regression import TransposeMode
    from utils.diag_imagenet_gaussian_unit_domain import unit_operator
    src=Path(spec['source']);state=torch.load(src/'pre_last_state.pt',map_location='cpu',weights_only=True)
    cache=torch.load(src/'trajectory.pt',map_location='cpu',weights_only=True)
    if state['tau'].tolist()!=plan['grid'] or any(cache[n].shape!=(30,3,256,256) or not torch.isfinite(cache[n]).all() for n in ('latent_inputs','clean','guidance')):raise ValueError('Cache/grid differs')
    operator=unit_operator(TransposeMode(get_operator('super_resolution',device=device,scale_factor=4,noise_sigma=0.),'legacy_bicubic').to(device)).to(device)
    z=ZAPS(model,operator,img_size=256,**plan['config']);z.tau=state['tau'].to(device)
    with torch.no_grad():z.zeta.copy_(state['zeta'].to(device));z.D.copy_(state['D'].to(device))
    y=torch.load(spec['measurement']['path'],map_location='cpu',weights_only=True)['y'].to(device)
    positions=source.fields.policy.Positions(source.fields.policy.noise_u(z.dm.alphas_cumprod.cpu().numpy()),list(reversed(plan['grid'])))
    return z,y,state,cache,positions


def completed(root,plan,spec):
    complete=[]
    for path in sorted((root/spec['case']['key']).glob('attempt_*/run.json')):
        r=source.read(path)
        if r['status']!='complete':continue
        r=source.load_signed(path)
        if (r['plan_sha256']!=source.digest(root/'plan.json') or r['case']!=spec['case'] or r['diagnostic_nfe']!=150
                or r['method_nfe']!=0 or set(r['metrics'])!=set(rules.ARMS)|{'F'} or not all(g['passed'] for g in r['gates'].values())
                or source.digest(path.parent/'outputs.pt')!=r['outputs_sha256']):raise ValueError('Completed record differs')
        expected=list(reversed(plan['grid']))
        for arm in rules.ARMS:
            if [c['t'] for c in r['calls'] if c['arm']==arm]!=expected:raise ValueError('Arm call schedule differs')
        complete.append((path,r))
    if len(complete)>1:raise ValueError('Multiple complete attempts; no output selection')
    return complete[0] if complete else None


def pictures(folder,case,outputs,gt,metrics):
    from PIL import Image,ImageDraw
    from utils.metrics import _to_uint8_numpy
    from utils.diag_cached_guidance_final_value import font
    names=('GT','C','F','S','STAGE','M'); arrays={'GT':_to_uint8_numpy(gt)}
    arrays.update({a:_to_uint8_numpy(outputs[a]) for a in names if a!='GT'})
    sheet=Image.new('RGB',(6*276+20,680),'white');d=ImageDraw.Draw(sheet)
    d.text((20,8),case+' | Fixed grid; every method 300 NFE; full image + same ROI',fill='black',font=font(19))
    roi=(64,48,192,176)
    for j,a in enumerate(names):
        x=20+j*276;raw=Image.fromarray(arrays[a]);full=raw.copy();mark=ImageDraw.Draw(full);mark.rectangle(roi,outline='#f8bd46',width=2)
        d.text((x,43),a+(' (t319 x2)' if a=='F' else ''),fill='black',font=font(17))
        if a!='GT':
            m=metrics[a];d.text((x,68),f'P {m["psnr"]:.3f} S {m["ssim"]:.4f}',fill='black',font=font(14))
            d.text((x,88),f'LPIPS {m["lpips"]:.4f}',fill='black',font=font(14))
        sheet.paste(full,(x,115));sheet.paste(raw.crop(roi).resize((256,256),Image.Resampling.NEAREST),(x,390))
    d.text((20,653),'C: ZAPS local reference | S: response feedback | STAGE: time-only | M: direction removed | ROI fixed across all cases',fill='black',font=font(15))
    sheet.save(folder/'comparison.png')
    return dict(roi=list(roi),display='clamp[-1,1] -> uint8 truncate; zoom nearest; no enhancement',
        image_sha256=source.digest(folder/'comparison.png'),source_pixels_sha256={a:__import__('hashlib').sha256(v.tobytes()).hexdigest() for a,v in arrays.items()})


def run_case(root,plan,spec,model,device,profiles):
    import torch
    from unittest.mock import patch
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state,same_rng
    from utils.diag_zaps_trace_audit import relative_error
    from modules.main_single import load_image_as_tensor
    from utils.metrics import compute_all_metrics
    if completed(root,plan,spec):return
    folder=root/spec['case']['key'];folder.mkdir(exist_ok=True)
    attempt=folder/f'attempt_{len(list(folder.glob("attempt_*"))):03d}';attempt.mkdir()
    path=attempt/'run.json';started=time.perf_counter();context={}
    r=dict(status='running',case=spec['case'],plan_sha256=source.digest(root/'plan.json'),method_nfe=0,diagnostic_nfe=0,calls=[],gates={})
    def save():r['seconds']=time.perf_counter()-started;source.WRITE(path,r)
    def count(_module,inputs):
        actual=int(inputs[1].flatten()[0]);r['diagnostic_nfe']+=1;r['calls'].append(dict(**context,t=actual));save()
        if actual!=context['expected_t'] or r['diagnostic_nfe']>150:raise RuntimeError('Call time/budget failed')
    save();hook=model.model.register_forward_pre_hook(count)
    try:
        P.verify_case_sources(plan,spec);z,y,state,cache,positions=setup(plan,spec,model,device)
        original=torch.load(Path(spec['source'])/'last_opt.pt',map_location='cpu',weights_only=True)
        native=source.read(Path(spec['source'])/'run.json')
        noises,replay=P.coupled.replay_noises(state,cache,dict(rows=native['rows']),z,y,device,original)
        reference=rules.stage_profile(profiles,spec['case']['image_id']);outputs={};traces={};parity=[];external=rng_state(device)
        with torch.no_grad(),model_rng_guard(model.model,device),patch('torch.randn_like',side_effect=RuntimeError('Unexpected sampling noise')):
            for arm in rules.ARMS:
                x=state['init_noise'].to(device);trace=[];sensor_seconds=0.;arm_start=time.perf_counter()
                for k,t in enumerate(positions.nominal):
                    context.update(arm=arm,step=k,expected_t=t);f=source.fields.field(z,y,x,t,positions)
                    if arm=='C':
                        error=max(relative_error(x.cpu(),cache['latent_inputs'][k:k+1]),relative_error(f['clean'].cpu(),cache['clean'][k:k+1]),relative_error(f['q'].cpu(),cache['guidance'][k:k+1]))
                        if not math.isfinite(error) or error>1e-6:raise RuntimeError('Core replay differs')
                        parity.append(error)
                    if arm in ('S','M','I'):
                        now=time.perf_counter();readout=rules.sensor(z,y,f['clean'],f['q'],t,29-k,identity=arm=='I',magnitude_only=arm=='M');sensor_seconds+=time.perf_counter()-now
                        gamma=readout['gamma']
                    else:gamma=1. if arm=='C' else reference[k];readout={}
                    trace.append(dict(step=k,t=t,gamma=gamma,R=f['R'],clip=f['clip_fraction'],**{n:v for n,v in readout.items() if n not in ('R','gamma')}))
                    base=P.terminal.terminal_output(z,x,f) if k==29 else P.coupled.posterior(x,f['clean'],t,positions.nominal[k+1],z.dm.alphas_cumprod,noises[k])+f['q']
                    x=base if gamma==1. else base+(gamma-1.)*f['q']
                    if not torch.isfinite(x).all():raise RuntimeError('Nonfinite output')
                outputs[arm]=x.cpu();traces[arm]=dict(rows=trace,seconds=time.perf_counter()-arm_start,sensor_wall_seconds=sensor_seconds)
        if r['diagnostic_nfe']!=150 or not same_rng(external,rng_state(device)) or P.precision.precision_state(torch)!=plan['native_policy']:raise RuntimeError('Budget/RNG/precision differs')
        final_error=relative_error(outputs['C'],original)
        if not math.isfinite(final_error) or final_error>1e-6:raise RuntimeError('Core final parity differs')
        if not all(torch.equal(a.cpu(),state[n]) for a,n in ((z.tau,'tau'),(z.zeta,'zeta'),(z.D,'D'))):raise RuntimeError('Parameters/grid mutated')
        # Source fixed-F and original C are hash-bound; no GT choice of output.
        a22=Path(plan['source_A22']);p22=P.load_record(a22,spec)
        saved=torch.load(a22/spec['case']['key']/'outputs.pt',map_location='cpu',weights_only=True)['outputs']
        outputs['F']=saved['T319_G2'];tmp=attempt/'outputs.pt.tmp';torch.save(outputs,tmp);tmp.replace(attempt/'outputs.pt')
        r.update(status='sampled',outputs_sha256=source.digest(attempt/'outputs.pt'),traces=traces);save()
        gt=load_image_as_tensor(spec['case']['image']).cpu()
        metrics={a:dict(loss=float((y-z.A.H(v.to(device))).double().square().mean()/(2*.05**2)),**compute_all_metrics(v,gt,lpips_net='alex')) for a,v in outputs.items()}
        for a in ('C','F'):
            target=p22['branches']['C' if a=='C' else 'T319_G2']
            if any(abs(metrics[a][m]-target[m])>tol for m,tol in (('psnr',.01),('ssim',.001),('lpips',.001))):raise RuntimeError('Saved reference metric protocol differs')
        visual=pictures(attempt,spec['case']['key'],outputs,gt,metrics)
        P.verify_case_sources(plan,spec)
        r.update(status='complete',metrics=metrics,visual=visual,gates=dict(core=dict(passed=True,max_step_error=max(parity),final_error=final_error),
            sampling=dict(passed=True,noise=replay),sensor_no_extra_unet=dict(passed=True),GT_after_outputs=dict(passed=True),parameters_sources=dict(passed=True)))
        save();source.signed(path,r);print('A32 CASE:',spec['case']['key'],'S-C PSNR=',metrics['S']['psnr']-metrics['C']['psnr'],'calls=150; image=',attempt/'comparison.png',flush=True)
    except BaseException as error:r.update(status='failed',error=repr(error));save();raise
    finally:hook.remove()


def worker(args):
    import fcntl
    import torch
    from utils.diag_state_weight_scan import preserved_rng
    root=Path(args.root).resolve();plan=source.check_plan(root)
    with (root/f'shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        specs=[s for i,s in enumerate(sorted(plan['cases'],key=lambda x:x['case']['key'])) if (i//2)%2==args.shard]
        if all(completed(root,plan,s) for s in specs):print('A32 shard complete; new UNet=0.');return
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(args.device),
            NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        if runtime!=plan['runtime'] or P.precision.precision_state(torch)!=plan['native_policy']:raise ValueError('Source runtime/precision differs; no experiment started')
        torch.set_num_threads(2)
        with preserved_rng(),torch.no_grad():
            model=P.coupled.archive.recovery.load_model(plan,args.device);profiles={}
            # Zero-UNet reference profiles, no GT labels. Both seeds of held image excluded.
            for spec in plan['cases']:
                P.verify_case_sources(plan,spec);z,y,state,cache,pos=setup(plan,spec,model,args.device)
                profiles[spec['case']['key']]=dict(image_id=spec['case']['image_id'],S=[rules.sensor(z,y,cache['clean'][k:k+1].to(args.device),
                    cache['guidance'][k:k+1].to(args.device),t,29-k) for k,t in enumerate(pos.nominal)])
            source.signed(root/f'profiles_shard{args.shard}.json',profiles)
            for spec in specs:run_case(root,plan,spec,model,args.device,profiles)
        source.check_plan(root)


def summarize(args):
    root=Path(args.root).resolve();plan=source.check_plan(root);records=[];cost=0;partials=0
    for s in plan['cases']:
        item=completed(root,plan,s)
        if item:records.append(item[1])
        for path in (root/s['case']['key']).glob('attempt_*/run.json'):
            attempt=source.read(path);cost+=attempt['diagnostic_nfe'];partials+=attempt['status']!='complete'
    comparisons={}
    for a in ('S','M','I','STAGE','F'):
        comparisons[a]={}
        for b in ('C','F','STAGE','M'):
            if not records:continue
            delta={m:float(np.mean([r['metrics'][a][m]-r['metrics'][b][m] for r in records])) for m in ('psnr','ssim','lpips','loss')}
            per_image={i:float(np.mean([r['metrics'][a]['psnr']-r['metrics'][b]['psnr'] for r in records if r['case']['image_id']==i])) for i in sorted({r['case']['image_id'] for r in records})}
            comparisons[a][b]=dict(mean_delta=delta,per_image_psnr=per_image,quality_guard=delta['ssim']>=-.002 and delta['lpips']<=.005)
    supported=len(records)==12 and all(comparisons['S'][b]['mean_delta']['psnr']>1e-5 and comparisons['S'][b]['quality_guard'] for b in ('C','F','STAGE'))
    histogram={a:{str(g):sum(v['gamma']==g for r in records for v in r['traces'][a]['rows']) for g in rules.GAMMAS} for a in rules.ARMS}
    result=dict(experiment=plan['experiment'],completed_cases=len(records),planned_cases=12,image_clusters=len({r['case']['image_id'] for r in records}),
        method_nfe=0,diagnostic_nfe=cost,planned_diagnostic_nfe=1800,partial_attempts=partials,cost_is_lower_bound_if_abruptly_interrupted=bool(partials),
        deployment_nfe_per_arm=300,sensor_extra_unet=0,classification=('DEVELOPMENT_FEEDBACK_CANDIDATE_ONLY' if supported else 'NO_CONFIRMED_RESPONSE_FEEDBACK_GAIN') if len(records)==12 else 'INCOMPLETE_PILOT',
        comparisons=comparisons,gamma_histogram=histogram,
        limits=plan['scope']+' No independent confirmation, exact curvature, author parity, or multi-epoch adaptation claim. H/DWT overhead is additional wall time; saved first270 calls reused.')
    source.signed(root/'summary.json',result);print('A32 Summary:',json.dumps(result,ensure_ascii=False),flush=True);print('Records:',root)
    if args.archive:
        if len(records)!=12:raise RuntimeError('Incomplete A32; completed cases retained, rerun launcher to resume')
        from utils.archive_refinement_confirmation import export_and_commit
        bundle,commit=export_and_commit({'A32':str(root)});print('Archive:',bundle,'Results commit:',commit,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=ROOT)
    p.add_argument('--a22-dir',default=P.ROOT);p.add_argument('--prepare',action='store_true');p.add_argument('--shard',type=int,choices=(0,1));p.add_argument('--device',default='cuda:0');p.add_argument('--archive',action='store_true')
    args=p.parse_args()
    if args.prepare:prepare(args)
    elif args.shard is not None:worker(args)
    else:summarize(args)


if __name__=='__main__':main()
