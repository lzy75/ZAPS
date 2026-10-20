"""A23 first fixed-count position-feedback screen: six cases x four 30-call arms."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import numpy as np

PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS))
from utils import diag_equal_nfe_path_opportunity as parent
from utils import fixed_count_state_positions as policy
from utils.diag_state_trajectory import digest
WRITE=parent.WRITE
ROOT='/data0/lzy/ZAPS-results/diag_fixed_count_state_positions/A23_20261020'
CARD=PROJECTS/'experiment_cards/A23_A27_fixed_count_state_positions_20261020.md'
CALLS=120


def prepare(args):
    source=Path(args.a22_dir).resolve();p=parent.parent.old.read(source/'plan.json');root=Path(args.root).resolve()
    if digest(source/'plan.json')!=(source/'plan.sha256').read_text().strip() or digest(source/'summary.json')!=(source/'summary.sha256').read_text().strip():raise ValueError('A22 fingerprints differ')
    a22=[]
    for s in p['cases']:
        r=parent.load_record(source,s)
        if r is None:raise ValueError('Need twelve completed A22 cases')
        a22.append(r)
    if parent.summarize(a22)!=parent.parent.old.read(source/'summary.json'):raise ValueError('A22 summary differs')
    identities=dict(p['identity_files'])
    for path,sha in identities.items():
        if digest(path)!=sha:raise ValueError('Frozen source/code changed: '+path)
    def bind(path):identities[str(path)]=digest(path)
    for path in (source/'plan.json',source/'plan.sha256',source/'summary.json',source/'summary.sha256'):bind(path)
    cases=[s for s in p['cases'] if s['cohort']=='development']
    if {(s['case']['image_id'],s['case']['sampling_seed']) for s in cases}!={(f'{i:05d}',seed) for i in range(20,23) for seed in (1000,2000)}:raise ValueError('Fixed development cohort differs')
    records=[parent.parent.old.read(Path(s['source'])/'run.json') for s in cases]
    refs={s['case']['key']:policy.reference(records,exclude_image=s['case']['image_id']) for s in cases}
    for s in cases:
        path=source/s['case']['key']/'run.json';bind(path);bind(path.with_suffix('.sha256'))
    own=[Path(__file__),Path(policy.__file__),CARD,PROJECTS/'run_a23_20261020.sh',PROJECTS/'utils/test_diag_fixed_count_state_positions.py',PROJECTS/'modules/zaps_algorithm.py',PROJECTS/'modules/wavelet.py']
    for path in own:bind(path)
    if root.is_relative_to(source) or source.is_relative_to(root) or any(root.is_relative_to(Path(s['source']).resolve()) for s in cases):raise ValueError('Run overlaps source')
    plan=dict(experiment='A23_fixed_count_state_positions',cases=cases,config=p['config'],grid=p['grid'],checkpoint_sha256=p['checkpoint_sha256'],
        identity_files=identities,worker_identity_files={str(x):identities[str(x)] for x in own},runtime=p['runtime'],native_policy=p['native_policy'],references=refs,
        arms=list(policy.ARMS),rho=policy.RHO,planned_method_nfe=0,planned_diagnostic_nfe=6*CALLS,deployment_nfe_per_case=300,
        source_A22=str(source),reference_GT_used=False,noise_pairing='Original 29 draws paired by evaluation slot; common random numbers, not a common Brownian path.')
    if root.exists():
        if parent.parent.old.read(root/'plan.json')!=plan:raise ValueError('Frozen A23 differs; refuse overwrite')
    else:
        root.mkdir(parents=True,exist_ok=False);WRITE(root/'plan.json',plan);(root/'plan.sha256').write_text(digest(root/'plan.json')+'\n');(root/'experiment_card.md').write_text(CARD.read_text())
    print('A23 PREPARE PASS: six cases; C/R/D/RC each 30 calls, all scientific arms=300 NFE. New diagnostic=720.',flush=True)


def summary(records):
    complete=[r for r in records if r['status']=='complete'];cost=sum(r['diagnostic_nfe'] for r in records);out=dict(completed_cases=len(complete),planned_cases=6,image_clusters=len({r['case']['image_id'] for r in complete}),method_nfe=0,diagnostic_nfe=cost,
        deployment_nfe_per_case=300,classification='INCOMPLETE_PILOT',cumulative_queue_cost=dict(method_nfe=7200,diagnostic_nfe=3608+cost,total=10808+cost),
        limits='Fixed-count last-unroll scheduling only, original first 270 calls reused. Leave-one-development-image-out unlabeled references; no trained controller, step-size error estimate, NFE saving, independent confirmation or full-ten-epoch adaptive claim. Arm times include diagnostic instrumentation, not end-to-end acceleration evidence.',
        cost_is_lower_bound_if_abruptly_interrupted=any(r['status']=='running' for r in records))
    if len(complete)!=6:return out
    effects={arm:parent.endpoint(complete,[arm]*6) for arm in policy.ARMS}
    def increment(arm):
        return dict(mean_psnr=statistics.mean(r['branches']['RC']['psnr']-r['branches'][arm]['psnr'] for r in complete),
            images_better=sum(statistics.mean(r['branches']['RC']['psnr']-r['branches'][arm]['psnr'] for r in complete if r['case']['image_id']==image)>.00001 for image in sorted({r['case']['image_id'] for r in complete})))
    inc={a:increment(a) for a in ('R','D')};e=effects['RC'];joint=(e['quality_guard'] and e['mean_delta']['psnr']>=.05 and inc['R']['mean_psnr']>=.01 and inc['R']['images_better']>=2)
    classification='INITIAL_POSITION_FEEDBACK_CANDIDATE_ONLY' if joint else 'CURRENT_POSITION_FEEDBACK_NOT_SUPPORTED'
    out.update(classification=classification,relative_to_C=effects,RC_increment=inc,
        relative_to_A22_fixed_T319_G2={k:statistics.mean(r['branches']['RC'][k]-r['a22_fixed'][k] for r in complete) for k in parent.METRICS},
        changes={a:dict(mean_moved_nodes=statistics.mean(sum(x['t']!=x['nominal_t'] for x in r['traces'][a]) for r in complete),mean_abs_displacement_t=statistics.mean(statistics.mean(abs(x['t']-x['nominal_t']) for x in r['traces'][a]) for r in complete)) for a in policy.ARMS},
        rows=[dict(case=r['case']['key'],branches=r['branches'],gates_passed=all(g['passed'] for g in r['gates'].values())) for r in complete])
    return out


def completed(root,spec):
    path=root/spec['case']['key']/'run.json'
    if not path.exists():
        if path.parent.exists():raise ValueError('Interrupted A23 retained')
        return None
    r=parent.parent.old.read(path)
    if (r['status']!='complete' or digest(path)!=(path.parent/'run.sha256').read_text().strip() or r['case']!=spec['case'] or r['plan_sha256']!=digest(root/'plan.json') or r['diagnostic_nfe']!=CALLS or r['method_nfe']!=0 or len(r['calls'])!=CALLS or
        set(r['branches'])!=set(policy.ARMS) or digest(path.parent/'outputs.pt')!=r['outputs_sha256'] or not r['gates'] or not all(g['passed'] for g in r['gates'].values())):raise ValueError('Invalid or incomplete A23 retained')
    for a in policy.ARMS:
        times=[x['t'] for x in r['traces'][a]]
        if len(times)!=30 or times[0]!=999 or times[-1]!=0 or any(t<=s for t,s in zip(times,times[1:])) or sum(c['arm']==a for c in r['calls'])!=30:raise ValueError('Saved actual budget/endpoint differs')
    return r


def field(z,y,x,t,positions):
    import torch
    tb=torch.full((x.shape[0],),t,device=x.device,dtype=torch.long);eps=z.dm._predict_eps(x,tb);ab=z.dm.alphas_cumprod[t]
    raw=(x-(1-ab).sqrt()*eps.detach())/ab.sqrt().clamp(min=1e-8);clean=raw.clamp(-1,1)
    zeta,D,weights=policy.parameters(z,t,positions);res=y-z.A.H(clean);v=z.A.transpose(res);hv=z.dwt.synthesis(D*z.dwt.analysis(v))
    q=zeta*((v+(1-ab)*hv)/ab.sqrt().clamp(min=1e-8))
    if not all(torch.isfinite(a).all() for a in (clean,q)):raise RuntimeError('Nonfinite field')
    return dict(clean=clean,q=q,R=float(res.double().square().mean().sqrt()/.05),clip_fraction=float((raw.abs()>1).float().mean()),weights=weights)


def run_case(root,plan,spec,model,device):
    import torch
    from unittest.mock import patch
    from modules.zaps_algorithm import ZAPS
    from modules.degradations import get_operator
    from modules.main_single import load_image_as_tensor
    from utils.diag_ffhq_regression import TransposeMode
    from utils.diag_imagenet_gaussian_unit_domain import unit_operator
    from utils.diag_cached_guidance_geometry import vector_readouts
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state,same_rng
    from utils.diag_zaps_trace_audit import relative_error
    from utils.metrics import compute_all_metrics
    folder=root/spec['case']['key'];folder.mkdir(exist_ok=False);started=time.perf_counter();context={}
    r=dict(status='running',case=spec['case'],plan_sha256=digest(root/'plan.json'),method_nfe=0,diagnostic_nfe=0,calls=[],gates={})
    def save():r['seconds']=time.perf_counter()-started;WRITE(folder/'run.json',r)
    def count(_module,inputs):
        r['diagnostic_nfe']+=1;r['calls'].append(dict(**context,t=int(inputs[1].flatten()[0])));save()
        if r['diagnostic_nfe']>CALLS or r['calls'][-1]['t']!=context['expected_t']:raise RuntimeError('Call/time contract failed')
    save();hook=model.model.register_forward_pre_hook(count)
    try:
        parent.verify_case_sources(plan,spec);src=Path(spec['source']);old=parent.parent.old.read(src/'run.json');external=rng_state(device)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))),torch.no_grad():
            state=torch.load(src/'pre_last_state.pt',map_location='cpu',weights_only=True);cache=torch.load(src/'trajectory.pt',map_location='cpu',weights_only=True);original=torch.load(src/'last_opt.pt',map_location='cpu',weights_only=True)
            if state['tau'].tolist()!=plan['grid'] or len(old['rows'])!=30:raise ValueError('Original grid differs')
            operator=unit_operator(TransposeMode(get_operator('super_resolution',device=device,scale_factor=4,noise_sigma=0.),'legacy_bicubic').to(device)).to(device)
            z=ZAPS(model,operator,img_size=256,**plan['config']);z.tau=state['tau'].to(device);z.zeta.copy_(state['zeta'].to(device));z.D.copy_(state['D'].to(device))
            positions=policy.Positions(policy.noise_u(z.dm.alphas_cumprod.cpu().numpy()),list(reversed(plan['grid'])))
            if any(abs(positions.u[row['t']]-row['noise_u'])>1e-8 for row in old['rows']):raise RuntimeError('Noise coordinate convention differs')
            y=torch.load(spec['measurement']['path'],map_location='cpu',weights_only=True)['y'].to(device)
            noises,noise_gate=parent.coupled.replay_noises(state,cache,dict(rows=old['rows']),z,y,device,original)
            outputs={};traces={};parity=[];ref=plan['references'][spec['case']['key']]
            if ref['GT_used'] or spec['case']['image_id'] in ref['reference_images']:raise ValueError('Reference leaks current development image or GT')
            with model_rng_guard(model.model,device),patch('torch.randn_like',side_effect=RuntimeError('Unaccounted draw')):
                for arm in policy.ARMS:
                    x=state['init_noise'].to(device);t=999;qhistory=[];rows=[];arm_start=time.perf_counter()
                    for k in range(30):
                        context.update(arm=arm,step=k,expected_t=t);f=field(z,y,x,t,positions);u=float(positions.u[t])
                        geo=vector_readouts(f['q'],qhistory[-1] if qhistory else None,qhistory[-2] if len(qhistory)>1 else None,u,rows[-1]['u'] if rows else None,rows[-2]['u'] if len(rows)>1 else None)
                        signals=policy.scores(f['R'],geo['variation3_per_u'],u,ref);E=policy.combined(arm,signals)
                        s=positions.choose(k+1,E) if k<29 else -1
                        rows.append(dict(step=k,t=t,next_t=s,u=u,nominal_t=positions.nominal[k],R=f['R'],qRMS=geo['rms'],cosine=geo['cosine'],turn=geo['turn'],Omega=geo['variation3_per_u'],clip_fraction=f['clip_fraction'],E=E,**signals,parameter_weights=f['weights']))
                        if arm=='C':
                            errors={name:relative_error(a,b) for name,a,b in (('latent',x.cpu(),cache['latent_inputs'][k:k+1]),('clean',f['clean'].cpu(),cache['clean'][k:k+1]),('q',f['q'].cpu(),cache['guidance'][k:k+1]))}
                            if s!=old['rows'][k]['t_prev'] or any(not math.isfinite(e) or e>1e-6 for e in errors.values()):raise RuntimeError('Original-grid null parity failed')
                            parity.append(dict(step=k,**errors))
                        qhistory.append(f['q']);qhistory=qhistory[-2:]
                        x=(parent.terminal.terminal_output(z,x,f) if s<0 else parent.coupled.posterior(x,f['clean'],t,s,z.dm.alphas_cumprod,noises[k])+f['q']);t=s
                    outputs[arm]=x.cpu();traces[arm]=rows
                    r.setdefault('arm_seconds',{})[arm]=time.perf_counter()-arm_start
                    if not torch.isfinite(outputs[arm]).all():raise RuntimeError('Nonfinite endpoint')
            final_error=relative_error(original,outputs['C'])
            if not math.isfinite(final_error) or final_error>1e-6 or r['diagnostic_nfe']!=CALLS:raise RuntimeError('Baseline/budget parity failed')
            if not(torch.equal(z.tau.cpu(),state['tau']) and torch.equal(z.zeta.cpu(),state['zeta']) and torch.equal(z.D.cpu(),state['D'])):raise RuntimeError('Learned parameters changed')
            tmp=folder/'outputs.pt.tmp';torch.save(outputs,tmp);tmp.replace(folder/'outputs.pt');r['outputs_sha256']=digest(folder/'outputs.pt')
            gt=load_image_as_tensor(spec['case']['image']).cpu()
            def metrics(x):return dict(loss=float((y-operator.H(x.to(device))).double().square().mean()/(2*.05**2)),**compute_all_metrics(x,gt,lpips_net='alex'))
            branches={a:metrics(x) for a,x in outputs.items()};orig_metrics=metrics(original);source_metrics=old.get('final_metrics',old.get('baseline'))
            if not all(math.isfinite(v) for b in (*branches.values(),orig_metrics) for v in b.values()):raise RuntimeError('Nonfinite quality/physics metrics')
            drift={k:abs(branches['C'][k]-source_metrics[k]) for k in ('psnr','ssim','lpips')}
            if any(drift[k]>v for k,v in (('psnr',.01),('ssim',.001),('lpips',.001))) or abs(orig_metrics['loss']-branches['C']['loss'])>1e-7+2e-5*abs(orig_metrics['loss']):raise RuntimeError('Metric/physics parity failed')
            floors={k:max(drift.get(k,0.),abs(orig_metrics[k]-branches['C'][k])) for k in parent.METRICS}
            a22=parent.parent.old.read(Path(plan['source_A22'])/spec['case']['key']/'run.json')
            r.update(branches=branches,traces=traces,floors=floors,a22_fixed=a22['branches']['T319_G2'])
        if not same_rng(external,rng_state(device)) or parent.precision.precision_state(torch)!=plan['native_policy']:raise RuntimeError('External RNG/precision changed')
        parent.verify_case_sources(plan,spec)
        for path,sha in plan['worker_identity_files'].items():
            if digest(path)!=sha:raise ValueError('Worker changed during run')
        r['gates']=dict(original_noise=noise_gate,original_grid=dict(passed=True,states=parity,final_error=final_error),GT_free_causal_control=dict(passed=True),parameters_RNG_budget=dict(passed=True),sources_unchanged=dict(passed=True))
        r['status']='complete';save();(folder/'run.sha256').write_text(digest(folder/'run.json')+'\n');completed(root,spec)
        print('A23 CASE:',spec['case']['key'],'diagnostic=',r['diagnostic_nfe'],flush=True)
    except BaseException as error:r.update(status='failed',error=repr(error));save();raise
    finally:hook.remove()


def worker(args,plan,root):
    import fcntl,torch
    with (root/f'shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        todo=[s for i,s in enumerate(sorted(plan['cases'],key=lambda x:x['case']['key'])) if i%2==args.shard and completed(root,s) is None]
        if not todo:return
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(args.device),NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        if runtime!=plan['runtime'] or parent.precision.precision_state(torch)!=plan['native_policy']:raise ValueError('Native runtime differs; no UNet calls')
        torch.set_num_threads(2);parent.coupled.archive.recovery.require_space(root)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            model=parent.coupled.archive.recovery.load_model(plan,args.device)
            for spec in todo:run_case(root,plan,spec,model,args.device)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=ROOT);p.add_argument('--a22-dir',default=parent.ROOT);p.add_argument('--device',default='cuda:0')
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--prepare',action='store_true');mode.add_argument('--shard',type=int,choices=(0,1));mode.add_argument('--archive',action='store_true');mode.add_argument('--summary',action='store_true');args=p.parse_args();root=Path(args.root).resolve()
    if args.prepare:prepare(args);return
    plan=parent.parent.old.read(root/'plan.json')
    if digest(root/'plan.json')!=(root/'plan.sha256').read_text().strip():raise ValueError('Plan differs')
    for path,sha in plan['worker_identity_files'].items():
        if digest(path)!=sha:raise ValueError('Worker/analysis code differs')
    if args.shard is not None:worker(args,plan,root);return
    if args.summary:
        if digest(root/'summary.json')!=(root/'summary.sha256').read_text().strip():raise ValueError('Summary differs')
        print('A23 Summary:',json.dumps(parent.parent.old.read(root/'summary.json')),flush=True);print('Records:',root,flush=True);return
    records=[]
    for spec in plan['cases']:
        path=root/spec['case']['key']/'run.json'
        if path.exists():
            r=parent.parent.old.read(path)
            if r['status']=='complete':completed(root,spec)
            records.append(r)
    out=summary(records);WRITE(root/'summary.json',out);(root/'summary.sha256').write_text(digest(root/'summary.json')+'\n')
    from utils.archive_refinement_confirmation import export_and_commit
    bundle,commit=export_and_commit({'A23':str(root)});print('Archive:',bundle,'Results commit:',commit,flush=True)


if __name__=='__main__':main()
