"""A24: amplitude-matched residual, shared clock and slot-parameter diagnostic; 720 calls."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
import numpy as np

PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS))
from utils import diag_fixed_count_state_positions as source
from utils import fixed_count_state_positions as policy
from utils.diag_state_trajectory import digest
WRITE=source.WRITE;old=source.parent.parent.old;METRICS=source.parent.METRICS
ARMS=('RC_REPLAY','R_MATCHED','FIXED_GRID','SLOT_REPLAY');CALLS=120
ROOT='/data0/lzy/ZAPS-results/diag_state_feedback_diagnosis/A24_20261020'
CARD=PROJECTS/'experiment_cards/A24_state_feedback_diagnosis_20261020.md'


def matched(state):
    return state['e_R'] if state['e_C'] is None else .5*state['e_R']


def shared_grid(records,positions):
    if len(records)!=6:raise ValueError('Need six fixed A23 cases')
    times=[999]
    for j in range(1,29):
        target=statistics.median(float(positions.u[r['traces']['RC'][j]['t']]) for r in records)
        options=positions.cells[j];times.append(int(options[np.argmin(abs(positions.u[options]-target))]))
    times.append(0)
    if any(t<=s for t,s in zip(times,times[1:])):raise RuntimeError('Shared clock order failed')
    return times


def prepare(args):
    import torch
    from modules.diffusion_model import _precompute_diffusion_params
    a23=Path(args.a23_dir).resolve();p=old.read(a23/'plan.json');root=Path(args.root).resolve()
    if digest(a23/'plan.json')!=(a23/'plan.sha256').read_text().strip() or digest(a23/'summary.json')!=(a23/'summary.sha256').read_text().strip():raise ValueError('A23 fingerprints differ')
    active=[source.completed(a23,s) for s in p['cases']]
    if any(r is None for r in active) or len(active)!=6 or source.summary(active)!=old.read(a23/'summary.json'):raise ValueError('Need six audited completed A23 cases')
    if old.read(a23/'summary.json')['classification']!='CURRENT_POSITION_FEEDBACK_NOT_SUPPORTED':raise ValueError('Need declared A23 negative-context diagnosis')
    identities=dict(p['identity_files'])
    for path,sha in identities.items():
        if digest(path)!=sha:raise ValueError('Bound source/code changed: '+path)
    def bind(path):identities[str(path)]=digest(path)
    for name in ('plan.json','plan.sha256','summary.json','summary.sha256'):bind(a23/name)
    for spec in p['cases']:
        for name in ('run.json','run.sha256','outputs.pt'):bind(a23/spec['case']['key']/name)
    alpha=_precompute_diffusion_params()['alphas_cumprod'];positions=policy.Positions(policy.noise_u(alpha.cpu().numpy()),list(reversed(p['grid'])))
    fixed=shared_grid(active,positions)
    own=[Path(__file__),CARD,PROJECTS/'run_a24_20261020.sh',PROJECTS/'utils/test_diag_state_feedback_diagnosis.py']
    for path in own:bind(path)
    if root.is_relative_to(a23) or a23.is_relative_to(root) or any(root.is_relative_to(Path(s['source']).resolve()) for s in p['cases']):raise ValueError('Run overlaps historical source')
    plan=dict(experiment='A24_state_feedback_diagnosis',cases=p['cases'],config=p['config'],grid=p['grid'],checkpoint_sha256=p['checkpoint_sha256'],identity_files=identities,
        worker_identity_files={**p['worker_identity_files'],**{str(x):identities[str(x)] for x in own}},runtime=p['runtime'],native_policy=p['native_policy'],references=p['references'],
        source_A23=str(a23),fixed_grid=fixed,arms=list(ARMS),planned_method_nfe=0,planned_diagnostic_nfe=720,
        deployable_nfe_per_case=dict(original=300,source_RC=300,R_MATCHED=300,FIXED_GRID=300),
        SLOT_REPLAY='Diagnostic clock conditioned on this image prior RC run; online collect-clock-plus-replay would cost 330 NFE, not a free 300-NFE method.',
        source_summary_classification=old.read(a23/'summary.json')['classification'],round_in_user_limit='2/5; original held-out A24 plan replaced by authorized failure diagnosis')
    if root.exists():
        if old.read(root/'plan.json')!=plan:raise ValueError('Frozen A24 differs; refuse overwrite')
    else:
        root.mkdir(parents=True,exist_ok=False);WRITE(root/'plan.json',plan);(root/'plan.sha256').write_text(digest(root/'plan.json')+'\n');(root/'experiment_card.md').write_text(CARD.read_text())
    print('A24 PREPARE PASS: four 30-call replays x six cases; 720 diagnostic NFE. SLOT_REPLAY is diagnostic only.',flush=True)


def pair(records,a,b):
    delta={k:statistics.mean(r['branches'][a][k]-r['branches'][b][k] for r in records) for k in METRICS}
    images=sorted({r['case']['image_id'] for r in records});per={i:statistics.mean(r['branches'][a]['psnr']-r['branches'][b]['psnr'] for r in records if r['case']['image_id']==i) for i in images}
    return dict(mean_delta=delta,per_image_psnr_delta=per,images_psnr_better=sum(v>1e-5 for v in per.values()),
        psnr_mechanism_hint=delta['psnr']>=.01 and sum(v>1e-5 for v in per.values())>=2)


def summarize(records):
    complete=[r for r in records if r['status']=='complete'];cost=sum(r['diagnostic_nfe'] for r in records)
    out=dict(completed_cases=len(complete),planned_cases=6,image_clusters=len({r['case']['image_id'] for r in complete}),method_nfe=0,diagnostic_nfe=cost,classification='INCOMPLETE_PILOT',
        round_in_user_limit='2/5',deployable_nfe_per_case=dict(original=300,source_RC=300,R_MATCHED=300,FIXED_GRID=300),
        SLOT_REPLAY='Offline conditional diagnostic, not a deployable 300-NFE policy; collecting its image-specific RC clock online then replaying costs 330 NFE.',
        cumulative_queue_cost=dict(method_nfe=7200,diagnostic_nfe=4328+cost,total=11528+cost),cost_is_lower_bound_if_abruptly_interrupted=any(r['status']=='running' for r in records),
        limits='Retrospective three-development-image diagnosis, two dependent seeds each. No new controller selection, threshold/rho search, independent confirmation or state-gain claim; instrumented times are not acceleration evidence.')
    if len(complete)!=6:return out
    effects={a:source.parent.endpoint(complete,[a]*6) for a in ('C','RC','R_MATCHED','FIXED_GRID','SLOT_REPLAY')}
    comparisons=dict(direction_after_residual_amplitude_match=pair(complete,'RC','R_MATCHED'),online_feedback_vs_shared_fixed_clock=pair(complete,'RC','FIXED_GRID'),
        parameter_lookup_on_identical_clock=pair(complete,'SLOT_REPLAY','RC'))
    out.update(classification='MECHANISM_DIAGNOSIS_ONLY',relative_to_C=effects,comparisons=comparisons,
        source_RC_net_benefit_supported=effects['RC']['quality_guard'] and effects['RC']['mean_delta']['psnr']>=.05,
        source_RC_parameter_lookup=dict(mean_abs_relative_zeta_change=statistics.mean(x['zeta_relative_difference'] for r in complete for x in r['source_parameter_changes']),
            max_abs_relative_zeta_change=max(x['zeta_relative_difference'] for r in complete for x in r['source_parameter_changes']),mean_D_relative_difference=statistics.mean(x['D_relative_difference'] for r in complete for x in r['source_parameter_changes'])),
        rows=[dict(case=r['case']['key'],branches=r['branches'],gates_passed=all(g['passed'] for g in r['gates'].values())) for r in complete])
    return out


def completed(root,spec):
    path=root/spec['case']['key']/'run.json'
    if not path.exists():
        if path.parent.exists():raise ValueError('Interrupted A24 retained')
        return None
    r=old.read(path)
    if (r['status']!='complete' or digest(path)!=(path.parent/'run.sha256').read_text().strip() or r['plan_sha256']!=digest(root/'plan.json') or r['case']!=spec['case'] or r['method_nfe']!=0 or r['diagnostic_nfe']!=CALLS or len(r['calls'])!=CALLS or
        digest(path.parent/'outputs.pt')!=r['outputs_sha256'] or set(r['branches'])!=set(ARMS)|{'C','RC'} or set(r['traces'])!=set(ARMS) or not r['gates'] or not all(g['passed'] for g in r['gates'].values())):raise ValueError('Invalid or interrupted A24 retained')
    for a in ARMS:
        times=[x['t'] for x in r['traces'][a]]
        if len(times)!=30 or times[0]!=999 or times[-1]!=0 or any(t<=s for t,s in zip(times,times[1:])) or sum(c['arm']==a for c in r['calls'])!=30:raise ValueError('Saved clock/budget invalid')
    return r


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
        if r['diagnostic_nfe']>CALLS or r['calls'][-1]['t']!=context['expected_t']:raise RuntimeError('Score budget/time mismatch')
    def check_sources():
        source.parent.verify_case_sources(plan,spec)
        for name in ('run.json','run.sha256','outputs.pt'):
            path=Path(plan['source_A23'])/spec['case']['key']/name
            if digest(path)!=plan['identity_files'][str(path)]:raise ValueError('A23 case changed')
    save();hook=model.model.register_forward_pre_hook(count)
    try:
        check_sources();src=Path(spec['source']);native=old.read(src/'run.json');a23=source.completed(Path(plan['source_A23']),spec)
        external=rng_state(device)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))),torch.no_grad():
            state=torch.load(src/'pre_last_state.pt',map_location='cpu',weights_only=True);cache=torch.load(src/'trajectory.pt',map_location='cpu',weights_only=True);original=torch.load(src/'last_opt.pt',map_location='cpu',weights_only=True)
            saved=torch.load(Path(plan['source_A23'])/spec['case']['key']/'outputs.pt',map_location='cpu',weights_only=True)
            operator=unit_operator(TransposeMode(get_operator('super_resolution',device=device,scale_factor=4,noise_sigma=0.),'legacy_bicubic').to(device)).to(device)
            z=ZAPS(model,operator,img_size=256,**plan['config']);z.tau=state['tau'].to(device);z.zeta.copy_(state['zeta'].to(device));z.D.copy_(state['D'].to(device))
            positions=policy.Positions(policy.noise_u(z.dm.alphas_cumprod.cpu().numpy()),list(reversed(plan['grid'])))
            if state['tau'].tolist()!=plan['grid'] or any(abs(positions.u[x['t']]-x['noise_u'])>1e-8 for x in native['rows']):raise ValueError('Source grid/noise coordinates differ')
            y=torch.load(spec['measurement']['path'],map_location='cpu',weights_only=True)['y'].to(device)
            noises,noise_gate=source.parent.coupled.replay_noises(state,cache,dict(rows=native['rows']),z,y,device,original)
            changes=[]
            for k,row in enumerate(a23['traces']['RC']):
                zz,dd,ww=policy.parameters(z,row['t'],positions);zs=z.zeta[29-k]
                changes.append(dict(step=k,t=row['t'],zeta_interpolated=float(zz),zeta_slot=float(zs),zeta_relative_difference=float((zz-zs).abs()/torch.maximum(zz.abs(),zs.abs()).clamp(min=1e-30)),D_relative_difference=relative_error(dd,z.D[29-k]),**ww))
            ref=plan['references'][spec['case']['key']];outputs={};traces={};repeat_checks=[];clock=[x['t'] for x in a23['traces']['RC']]
            with model_rng_guard(model.model,device),patch('torch.randn_like',side_effect=RuntimeError('Unaccounted sampling noise')):
                for arm in ARMS:
                    t=999;x=state['init_noise'].to(device);history=[];rows=[]
                    for k in range(30):
                        context.update(arm=arm,step=k,expected_t=t)
                        lookup=SimpleNamespace(weights=lambda ignored,index=29-k:(index,index,0.)) if arm=='SLOT_REPLAY' else positions
                        f=source.field(z,y,x,t,lookup);u=float(positions.u[t])
                        geo=vector_readouts(f['q'],history[-1] if history else None,history[-2] if len(history)>1 else None,u,rows[-1]['u'] if rows else None,rows[-2]['u'] if len(rows)>1 else None)
                        signals=policy.scores(f['R'],geo['variation3_per_u'],u,ref);E=matched(signals) if arm=='R_MATCHED' else None
                        if k==29:s=-1
                        elif arm=='R_MATCHED':s=positions.choose(k+1,E)
                        elif arm=='FIXED_GRID':s=plan['fixed_grid'][k+1]
                        else:s=clock[k+1]
                        row=dict(step=k,t=t,next_t=s,u=u,R=f['R'],qRMS=geo['rms'],Omega=geo['variation3_per_u'],clip_fraction=f['clip_fraction'],E=E,**signals,parameter_weights=f['weights']);rows.append(row)
                        if arm=='RC_REPLAY':
                            previous=a23['traces']['RC'][k]
                            if t!=previous['t'] or s!=previous['next_t'] or f['weights']!=previous['parameter_weights']:raise RuntimeError('Source RC clock/weights differ')
                            for key in ('R','qRMS','clip_fraction'):
                                if not math.isclose(row[key],previous[key],rel_tol=1e-6,abs_tol=1e-7):raise RuntimeError('Source RC state differs: '+key)
                            repeat_checks.append(dict(step=k,R_error=abs(row['R']-previous['R']),qRMS_error=abs(row['qRMS']-previous['qRMS'])))
                        history.append(f['q']);history=history[-2:]
                        x=(source.parent.terminal.terminal_output(z,x,f) if s<0 else source.parent.coupled.posterior(x,f['clean'],t,s,z.dm.alphas_cumprod,noises[k])+f['q']);t=s
                    outputs[arm]=x.cpu();traces[arm]=rows
                    if not torch.isfinite(outputs[arm]).all():raise RuntimeError('Nonfinite endpoint')
            error=relative_error(outputs['RC_REPLAY'],saved['RC'])
            if not math.isfinite(error) or error>1e-6 or r['diagnostic_nfe']!=CALLS:raise RuntimeError('RC endpoint/budget parity failed')
            if not(torch.equal(z.tau.cpu(),state['tau']) and torch.equal(z.zeta.cpu(),state['zeta']) and torch.equal(z.D.cpu(),state['D'])):raise RuntimeError('PRE parameters changed')
            tmp=folder/'outputs.pt.tmp';torch.save(outputs,tmp);tmp.replace(folder/'outputs.pt');r['outputs_sha256']=digest(folder/'outputs.pt')
            gt=load_image_as_tensor(spec['case']['image']).cpu()
            def metrics(x):return dict(loss=float((y-operator.H(x.to(device))).double().square().mean()/(2*.05**2)),**compute_all_metrics(x,gt,lpips_net='alex'))
            branches={a:metrics(x) for a,x in outputs.items()};branches['C']=metrics(saved['C']);branches['RC']=a23['branches']['RC']
            if not all(math.isfinite(v) for b in branches.values() for v in b.values()):raise RuntimeError('Nonfinite metrics')
            drift={k:max(abs(branches['C'][k]-a23['branches']['C'][k]),abs(branches['RC_REPLAY'][k]-branches['RC'][k])) for k in METRICS}
            if any(drift[k]>v for k,v in (('psnr',.01),('ssim',.001),('lpips',.001))) or drift['loss']>1e-7+2e-5*abs(branches['RC']['loss']):raise RuntimeError('Quality/physics repeat differs')
            r.update(branches=branches,traces=traces,source_parameter_changes=changes,floors={k:max(a23['floors'][k],drift[k]) for k in METRICS})
        if not same_rng(external,rng_state(device)) or source.parent.precision.precision_state(torch)!=plan['native_policy']:raise RuntimeError('RNG/precision changed')
        check_sources()
        for path,sha in plan['worker_identity_files'].items():
            if digest(path)!=sha:raise ValueError('Worker changed')
        r['gates']=dict(original_noise=noise_gate,RC_replay=dict(passed=True,relative_error=error,states=repeat_checks),parameters_RNG_budget=dict(passed=True),sources_unchanged=dict(passed=True),offline_clocks_GT_free=dict(passed=True,diagnostic_only=['SLOT_REPLAY']))
        r['status']='complete';save();(folder/'run.sha256').write_text(digest(folder/'run.json')+'\n');completed(root,spec)
        print('A24 CASE:',spec['case']['key'],'diagnostic=',r['diagnostic_nfe'],flush=True)
    except BaseException as e:r.update(status='failed',error=repr(e));save();raise
    finally:hook.remove()


def worker(args,plan,root):
    import fcntl,torch
    with (root/f'shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        todo=[s for i,s in enumerate(sorted(plan['cases'],key=lambda x:x['case']['key'])) if i%2==args.shard and completed(root,s) is None]
        if not todo:return
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(args.device),NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        if runtime!=plan['runtime'] or source.parent.precision.precision_state(torch)!=plan['native_policy']:raise ValueError('Native runtime differs; no calls')
        torch.set_num_threads(2);source.parent.coupled.archive.recovery.require_space(root)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            model=source.parent.coupled.archive.recovery.load_model(plan,args.device)
            for spec in todo:run_case(root,plan,spec,model,args.device)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=ROOT);p.add_argument('--a23-dir',default=source.ROOT);p.add_argument('--device',default='cuda:0')
    m=p.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--shard',type=int,choices=(0,1));m.add_argument('--archive',action='store_true');m.add_argument('--summary',action='store_true');args=p.parse_args();root=Path(args.root).resolve()
    if args.prepare:prepare(args);return
    plan=old.read(root/'plan.json')
    if digest(root/'plan.json')!=(root/'plan.sha256').read_text().strip():raise ValueError('Plan changed')
    for path,sha in plan['worker_identity_files'].items():
        if digest(path)!=sha:raise ValueError('Worker/analysis changed')
    if args.shard is not None:worker(args,plan,root);return
    if args.summary:
        if digest(root/'summary.json')!=(root/'summary.sha256').read_text().strip():raise ValueError('Summary changed')
        print('A24 Summary:',json.dumps(old.read(root/'summary.json')),flush=True);print('Records:',root,flush=True);return
    records=[]
    for spec in plan['cases']:
        path=root/spec['case']['key']/'run.json'
        if path.exists():
            r=old.read(path)
            if r['status']=='complete':completed(root,spec)
            records.append(r)
    out=summarize(records);WRITE(root/'summary.json',out);(root/'summary.sha256').write_text(digest(root/'summary.json')+'\n')
    from utils.archive_refinement_confirmation import export_and_commit
    bundle,commit=export_and_commit({'A24':str(root)});print('Archive:',bundle,'Results commit:',commit,flush=True)


if __name__=='__main__':main()
