"""A39 common C-state physical/internal risk diagnosis; fixed grid, 900 diagnostic calls."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS))
from utils import diag_frozen_response_transfer as parent
from utils import common_state_risk as rules

source,P,scan=parent.source,parent.P,parent.scan
ROOT='/data0/lzy/ZAPS-results/diag_common_state_risk/A39_20261025'
CARD=PROJECTS/'experiment_cards/A39_common_state_risk_20261025.md'
IMAGES=('00339','00447','00561','00609','00875','00952')


def references(p,case):
    old=Path(p['source_A36'])
    for name in ('plan.json','summary.json'):
        f=old/name
        if source.digest(f)!=p['identity_files'][str(f)]:raise ValueError('Parent metadata changed: '+str(f))
    previous=source.check_plan(old)
    items=[parent.done(old,previous,case,kind) for kind in ('BASELINE','FEEDBACK')]
    if any(item is None for item in items):raise ValueError('Incomplete source: '+case['key'])
    for path,r in items:
        if source.digest(path)!=p['identity_files'][str(path)]:raise ValueError('Source run changed')
    return items


def prepare(args):
    root,old=Path(args.root).resolve(),Path(args.a36_dir).resolve()
    if root.is_relative_to(old) or old.is_relative_to(root):raise ValueError('Output overlaps source')
    if root.exists():
        p=source.check_plan(root)
        if p['source_A36']!=str(old):raise ValueError('Frozen source differs')
        for case in p['cases']:references(p,case)
        print('A39 frozen plan reused; completed attempts retained.',flush=True);return
    previous=source.check_plan(old);summary=source.load_signed(old/'summary.json')
    if previous['experiment']!='A36_frozen_response_transfer' or summary['completed_cases']!=18:
        raise ValueError('Need all eighteen complete A36 cases')
    cases=[c for c in previous['cases'] if c['sigma'] in rules.SIGMAS]
    if (len(cases)!=12 or {c['image_id'] for c in cases}!=set(IMAGES)
            or {(c['image_id'],c['sigma'],c['sampling_seed']) for c in cases}!=
               {(i,s,1000) for i in IMAGES for s in rules.SIGMAS}):raise ValueError('Frozen cohort differs')
    if len(rules.expected_calls(previous['grid']))!=rules.CALLS:raise ValueError('Budget mismatch')
    identities={str(old/n):source.digest(old/n) for n in ('plan.json','summary.json')}
    for case in cases:
        for kind in ('BASELINE','FEEDBACK'):
            item=parent.done(old,previous,case,kind)
            if item is None:raise ValueError('Missing source case')
            path,r=item;identities[str(path)]=source.digest(path)
        identities[case['image']]=case['image_sha256']
    own=[Path(__file__),Path(rules.__file__),Path(rules.geometry.__file__),Path(parent.__file__),CARD,
         PROJECTS/'run_a39_20261025.sh',PROJECTS/'utils/test_common_state_risk.py',
         PROJECTS/'utils/diag_cached_guidance_geometry.py',PROJECTS/'utils/experiment_gpu_selection.py',
         PROJECTS/'utils/test_experiment_gpu_selection.py']
    workers={**previous['worker_identity_files'],**{str(f):source.digest(f) for f in own}}
    p={k:previous[k] for k in ('config','grid','runtime','native_policy','checkpoint_sha256')}
    p.update(experiment='A39_common_state_risk',source_A36=str(old),cases=cases,image_ids=list(IMAGES),
             points=[list(v) for v in rules.POINTS],gammas=list(rules.GAMMAS),features=rules.FEATURES,
             identity_files=identities,worker_identity_files=workers,planned_method_nfe=0,
             planned_diagnostic_nfe=12*rules.CALLS,calls_per_case=rules.CALLS,scientific_nfe_per_arm=300,
             source_cost={k:summary[k] for k in ('method_nfe','diagnostic_nfe','cost_is_lower_bound_if_abruptly_interrupted')},
             scope='Six REUSED development images, two observation noise levels, one seed. Original C common prefix; '
                   'one absolute gamma pulse at t641 OR t319, then original C to final output. '
                   'Final quality/risk diagnosis, no combined pulse path, regularization, online controller, '
                   'fresh confirmation, task/dataset transfer, step-size, NFE saving, convergence or author parity.')
    root.parent.mkdir(parents=True,exist_ok=True);P.coupled.archive.recovery.require_space(root.parent)
    root.mkdir();source.signed(root/'plan.json',p);(root/'experiment_card.md').write_text(CARD.read_text())
    print('A39 PREPARE PASS:12 cases x(30 C replay +3x(10+5) suffix calls)=900 diagnostic calls; method0.',flush=True)


def done(root,p,case):
    complete=[]
    arms={'C'}|{rules.arm(k,g) for k,_ in rules.POINTS for g in rules.GAMMAS}
    for path in sorted((root/case['key']).glob('attempt_*/run.json')):
        r=source.read(path)
        if r['status']!='complete':continue
        r=source.load_signed(path)
        if (r['case']!=case or r['plan_sha256']!=source.digest(root/'plan.json') or r['method_nfe']!=0
                or r['nfe']!=rules.CALLS or r['calls']!=rules.expected_calls(p['grid'])
                or [v['t'] for v in r['rows']]!=list(reversed(p['grid'])) or set(r['metrics'])!=arms
                or source.digest(path.parent/'outputs.pt')!=r['outputs_sha256']
                or not r['gates'] or not all(g['passed'] for g in r['gates'].values())):
            raise ValueError('Invalid completed A39 attempt: '+str(path))
        training,feedback=references(p,case)
        if (r['source_training_sha256']!=source.digest(training[0]) or
                r['source_feedback_sha256']!=source.digest(feedback[0])):raise ValueError('Source provenance differs')
        complete.append((path,r))
    if len(complete)>1:raise ValueError('Multiple complete outputs; no GT selection')
    return complete[0] if complete else None


def run_case(root,p,case,model,device,operator):
    import torch
    from unittest.mock import patch
    from modules.zaps_algorithm import ZAPS
    from modules.main_single import load_image_as_tensor
    from utils.diag_cached_guidance_geometry import physical_descent
    from utils.diag_zaps_sampler_parity import rng_state,same_rng
    from utils.diag_zaps_guidance_clipping import saved_cuda_selection,restore_sampling_rng
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_trace_audit import relative_error
    from utils.metrics import compute_all_metrics
    if done(root,p,case):return
    training,feedback=references(p,case)
    folder=root/case['key'];folder.mkdir(exist_ok=True)
    attempt=folder/f'attempt_{len(list(folder.glob("attempt_*"))):03d}';attempt.mkdir();path=attempt/'run.json'
    started=time.perf_counter()
    state=torch.load(training[0].parent/'pre_last_state.pt',map_location='cpu',weights_only=True)
    original=torch.load(training[0].parent/'last_opt.pt',map_location='cpu',weights_only=True)
    saved=torch.load(training[1]['measurement']['path'],map_location='cpu',weights_only=True)
    if saved['image_sha256']!=case['image_sha256'] or saved['seed']!=case['measurement_seed']:
        raise ValueError('Observation identity differs')
    y=saved['mean'].to(device)+case['sigma']*saved['standard_noise'].to(device)
    z=ZAPS(model,operator,img_size=256,**p['config']);z.tau=state['tau'].to(device)
    with torch.no_grad():z.zeta.copy_(state['zeta'].to(device));z.D.copy_(state['D'].to(device))
    if z.tau.tolist()!=p['grid']:raise ValueError('Snapshot grid differs')
    positions=source.fields.policy.Positions(source.fields.policy.noise_u(z.dm.alphas_cumprod.cpu().numpy()),list(reversed(p['grid'])))
    with scan.preserved_rng():
        mapping=saved_cuda_selection(state,device);restore_sampling_rng(state,device,mapping['saved_sampling_index'])
        noises=[torch.randn_like(state['init_noise'].to(device)) for _ in range(29)]
    r=dict(case=case,method_nfe=0,plan_sha256=source.digest(root/'plan.json'),
           source_training_sha256=source.digest(training[0]),source_feedback_sha256=source.digest(feedback[0]),
           gates={},rows=[],sensor_seconds=0.,arm_seconds={})
    external=rng_state(device);outputs={};anchors={};history=rules.History()
    with scan.counted(model,path,r,rules.CALLS):
        with torch.no_grad(),model_rng_guard(model.model,device),patch('torch.randn_like',side_effect=RuntimeError('Unexpected draw')):
            torch.cuda.synchronize(device);now=time.perf_counter();x=state['init_noise'].to(device)
            for k,t in enumerate(positions.nominal):
                f=source.fields.field(z,y,x,t,positions)
                torch.cuda.synchronize(device);sensor_started=time.perf_counter()
                readout=parent.rules.sensor(z,y,f['clean'],f['q'],t,case['sigma'])
                gradient,loss=physical_descent(z.A,f['clean'],y,sigma=case['sigma'])
                observed=history.append(float(positions.u[t]),f['clean'][0].cpu().numpy(),f['q'][0].cpu().numpy(),
                                        gradient[0].cpu().numpy(),readout,f['clip_fraction'])
                r['sensor_seconds']+=time.perf_counter()-sensor_started
                r['rows'].append(dict(step=k,t=t,u=float(positions.u[t]),state=observed,physical_loss=loss))
                if k in {v[0] for v in rules.POINTS}:
                    anchors[k]=(x.clone(),{name:f[name].clone() for name in ('clean','q')})
                x=rules.advance(z,x,f,positions,k,noises)
                if not torch.isfinite(x).all():raise RuntimeError('Nonfinite C output')
            torch.cuda.synchronize(device);r['arm_seconds']['C']=time.perf_counter()-now;outputs['C']=x.cpu()
            for k,_ in rules.POINTS:
                for gamma in rules.GAMMAS:
                    name=rules.arm(k,gamma);torch.cuda.synchronize(device);now=time.perf_counter()
                    center,f=anchors[k];x=rules.advance(z,center,f,positions,k,noises,gamma)
                    for j in range(k+1,30):
                        f=source.fields.field(z,y,x,positions.nominal[j],positions)
                        x=rules.advance(z,x,f,positions,j,noises)
                    if not torch.isfinite(x).all():raise RuntimeError('Nonfinite pulse output; no replacement')
                    torch.cuda.synchronize(device);r['arm_seconds'][name]=time.perf_counter()-now;outputs[name]=x.cpu()
        parity=relative_error(outputs['C'],original)
        if (not math.isfinite(parity) or parity>1e-6 or r['nfe']!=rules.CALLS
                or r['calls']!=rules.expected_calls(p['grid']) or not same_rng(external,rng_state(device))
                or P.precision.precision_state(torch)!=p['native_policy']
                or any(v.grad is not None for v in model.model.parameters())
                or not all(torch.equal(v.cpu(),state[n]) for v,n in ((z.tau,'tau'),(z.zeta,'zeta'),(z.D,'D')))):
            raise RuntimeError('C parity/budget/RNG/precision/parameters gate failed')
        parent.save_tensor(outputs,attempt/'outputs.pt')
        r.update(status='sampled',outputs_sha256=source.digest(attempt/'outputs.pt'));source.WRITE(path,r)
        # All candidate outputs and causal readouts exist before GT is loaded for evaluation.
        gt=load_image_as_tensor(case['image']).cpu()
        metrics={name:dict(loss=float((y-z.A.H(out.to(device))).double().square().mean()/(2*case['sigma']**2)),
                           **compute_all_metrics(out,gt,lpips_net='alex')) for name,out in outputs.items()}
        if any(not math.isfinite(v) for m in metrics.values() for v in m.values()):raise RuntimeError('Nonfinite metric')
        metric_error={m:abs(metrics['C'][m]-feedback[1]['metrics']['C'][m]) for m in ('psnr','ssim','lpips')}
        if any(metric_error[m]>tol for m,tol in (('psnr',.01),('ssim',.001),('lpips',.001))):
            raise RuntimeError('Historical C metric fingerprint differs')
        r.update(metrics=metrics,gates=dict(original_C=dict(passed=True,relative_error=parity,metric_error=metric_error),
                    paired_noise_fixed_budget=dict(passed=True),parameters_native_precision=dict(passed=True),
                    passive_clean_gradient_no_model_grads=dict(passed=True),GT_after_outputs=dict(passed=True)))
    r.update(status='complete',seconds=time.perf_counter()-started);source.signed(path,r)
    print('A39 CASE:',case['key'],'complete; diagnostic calls=',r['nfe'],flush=True)


def worker(args):
    import fcntl
    import torch
    from modules.degradations import get_operator
    from utils.diag_ffhq_regression import TransposeMode
    from utils.diag_imagenet_gaussian_unit_domain import unit_operator
    root=Path(args.root).resolve();p=source.check_plan(root)
    with (root/f'shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cases=[c for c in p['cases'] if p['image_ids'].index(c['image_id'])%2==args.shard]
        if all(done(root,p,c) for c in cases):return
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                     gpu=torch.cuda.get_device_name(args.device),NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),
                     TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        if runtime!=p['runtime'] or P.precision.precision_state(torch)!=p['native_policy']:raise ValueError('Native runtime differs')
        torch.set_num_threads(2);P.coupled.archive.recovery.require_space(root)
        with scan.preserved_rng():
            model=P.coupled.archive.recovery.load_model(p,args.device)
            operator=unit_operator(TransposeMode(get_operator('super_resolution',device=args.device,
                       scale_factor=4,noise_sigma=0.),'legacy_bicubic').to(args.device)).to(args.device)
            for case in cases:
                if source.digest(case['image'])!=case['image_sha256']:raise ValueError('Target image changed')
                run_case(root,p,case,model,args.device,operator)
            source.check_plan(root)


def pictures(path,r):
    import torch
    from PIL import Image,ImageDraw
    from modules.main_single import load_image_as_tensor
    from utils.metrics import _to_uint8_numpy
    from utils.diag_cached_guidance_final_value import font
    outputs=torch.load(path.parent/'outputs.pt',map_location='cpu',weights_only=True)
    gt=Image.fromarray(_to_uint8_numpy(load_image_as_tensor(r['case']['image'])))
    for k,t in rules.POINTS:
        names=('GT','C')+tuple(rules.arm(k,g) for g in rules.GAMMAS)
        sheet=Image.new('RGB',(20+276*len(names),680),'white');draw=ImageDraw.Draw(sheet)
        draw.text((20,10),r['case']['key']+f' | single pulse t{t}; all arms300 NFE',fill='black',font=font(18))
        for j,name in enumerate(names):
            raw=gt if name=='GT' else Image.fromarray(_to_uint8_numpy(outputs[name]));full=raw.copy();x=20+276*j
            ImageDraw.Draw(full).rectangle((64,48,192,176),outline='#f8bd46',width=2)
            draw.text((x,45),name,fill='black',font=font(17))
            if name!='GT':
                m=r['metrics'][name];draw.text((x,70),f'P {m["psnr"]:.3f} S {m["ssim"]:.4f}',fill='black',font=font(14))
                draw.text((x,90),f'LPIPS {m["lpips"]:.4f}',fill='black',font=font(14))
            sheet.paste(full,(x,115));sheet.paste(raw.crop((64,48,192,176)).resize((256,256),Image.Resampling.NEAREST),(x,390))
        sheet.save(path.parent/f'comparison_t{t}.png')


def summarize(args):
    root=Path(args.root).resolve();p=source.check_plan(root);items=[]
    for case in p['cases']:
        item=done(root,p,case)
        if item:items.append(item)
    attempts=[source.read(f) for f in root.glob('*/attempt_*/run.json')]
    r=dict(experiment=p['experiment'],classification='INCOMPLETE_PILOT',completed_cases=len(items),planned_cases=12,
           method_nfe=0,diagnostic_nfe=sum(v['nfe'] for v in attempts),planned_diagnostic_nfe=900,
           scientific_nfe_per_arm=300,partial_attempts=sum(v['status']!='complete' for v in attempts),
           cost_is_lower_bound_if_abruptly_interrupted=any(v['status'] in ('running','sampled') for v in attempts),
           source_cost=p['source_cost'],scope=p['scope'])
    if len(items)==12:r.update(rules.analyze([v for _,v in items]))
    source.signed(root/'summary.json',r)
    if args.archive:
        for path,record in items:pictures(path,record)
    compact={k:v for k,v in r.items() if k not in ('actions','offline_single_decision_emulators','pooled_prediction')}
    if 'pooled_prediction' in r:
        compact['pooled_prediction']={m:r['pooled_prediction'][m] for m in ('lpips','psnr')}
    if 'offline_single_decision_emulators' in r:
        compact['offline_single_decision_emulators']={n:r['offline_single_decision_emulators'][n]['cells']
                                                     for n in ('STAGE_NOISE','P','PI')}
    print('A39 Summary:',json.dumps(compact,ensure_ascii=False),flush=True);print('Records:',root,flush=True)
    if args.archive:
        from utils.archive_refinement_confirmation import export_and_commit
        bundle,commit=export_and_commit({'A39':str(root)});print('Archive:',bundle,'Results commit:',commit,flush=True)
    if len(items)!=12:raise SystemExit('A39 incomplete; inspect shard logs. Finished attempts retained.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=ROOT)
    p.add_argument('--a36-dir',default=parent.ROOT);p.add_argument('--prepare',action='store_true')
    p.add_argument('--shard',type=int,choices=(0,1));p.add_argument('--device',default='cuda:0');p.add_argument('--archive',action='store_true')
    args=p.parse_args()
    if args.prepare:prepare(args)
    elif args.shard is not None:worker(args)
    else:summarize(args)


if __name__=='__main__':main()
