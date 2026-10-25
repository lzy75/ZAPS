"""A37 cached A36 continuation: REG/BLIND/TIME_NOISE, 1440 new diagnostic NFE."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS))
from utils import diag_frozen_response_transfer as parent
from utils import noise_regularized_feedback as rules

source,P,scan = parent.source,parent.P,parent.scan
ROOT = '/data0/lzy/ZAPS-results/diag_noise_regularized_feedback/A37_20261025'
CARD = PROJECTS/'experiment_cards/A37_noise_regularized_feedback_20261025.md'


def source_records(old,p,case):
    training = parent.done(old,p,case,'BASELINE')
    feedback = parent.done(old,p,case,'FEEDBACK')
    if (training is None or feedback is None
            or source.digest(training[0]) != feedback[1]['training_sha256']):
        raise ValueError('Need complete compatible A36 training and feedback: '+case['key'])
    return training,feedback


def prepare(args):
    root,old = Path(args.root).resolve(),Path(args.a36_dir).resolve()
    if root.is_relative_to(old) or old.is_relative_to(root):raise ValueError('Output overlaps A36')
    if root.exists():
        p = source.check_plan(root)
        if p['source_A36'] != str(old):raise ValueError('Frozen parent differs')
        print('A37 frozen plan reused; completed cases retained.',flush=True);return
    p36 = source.check_plan(old);summary = source.load_signed(old/'summary.json')
    if p36['experiment'] != 'A36_frozen_response_transfer' or summary['completed_cases'] != 18:
        raise ValueError('Need all eighteen A36 cases')
    identities = dict(p36['identity_files']);refs = {};records = []
    for case in p36['cases']:
        training,feedback = source_records(old,p36,case);records.append(feedback[1])
        refs[case['key']] = dict(training=str(training[0]),feedback=str(feedback[0]))
        paths = [training[0],feedback[0],training[0].parent/'pre_last_state.pt',
                 training[0].parent/'last_opt.pt',feedback[0].parent/'outputs.pt',
                 Path(training[1]['measurement']['path'])]
        identities.update({str(path):source.digest(path) for path in paths})
    if parent.rules.summarize(records)['groups'] != summary['groups']:
        raise ValueError('A36 per-case metrics disagree with summary')
    for name in ('plan.json','summary.json'):identities[str(old/name)] = source.digest(old/name)
    a33 = Path(p36['source_A33']);p33 = source.check_plan(a33);calibration = []
    for spec in p33['cases']:
        item = parent.parent.task_done(a33,p33,spec,'I_cont2')
        if item is None:raise ValueError('Old calibration arm incomplete')
        calibration.append(item[1]);identities[str(item[0])] = source.digest(item[0])
    profiles = rules.reference_profiles(calibration,p36['grid'],parent.rules.SIGMAS)
    workers = dict(p36['worker_identity_files'])
    own = [Path(__file__),Path(rules.__file__),CARD,PROJECTS/'run_a37_20261025.sh',
           PROJECTS/'utils/test_noise_regularized_feedback.py',PROJECTS/'utils/experiment_gpu_selection.py']
    workers.update({str(f):source.digest(f) for f in own})
    p = dict(p36)
    p.update(experiment='A37_noise_regularized_feedback',source_A36=str(old),source_records=refs,
             identity_files=identities,worker_identity_files=workers,profiles=profiles,
             lambda_value=rules.LAMBDA,upper=rules.UPPER,blind_sigma=rules.SIGMA_REFERENCE,
             planned_method_nfe=0,planned_diagnostic_nfe=sum(rules.calls(c['sigma']) for c in p36['cases']),
             source_cost=dict(method_nfe=summary['method_nfe'],diagnostic_nfe=summary['diagnostic_nfe'],
                              cost_is_lower_bound=summary['cost_is_lower_bound_if_abruptly_interrupted']),
             scope='Post-A36 mechanism development on the same six images/three sigmas; no independent confirmation. '
                   'Frozen lambda1, upper2, original grid and first270; every scientific arm300 NFE. '
                   'TIME_NOISE uses old reference quadratics and known sigma, no target state or GT. '
                   'No true-curvature, posterior correctness, convergence or NFE-saving claim.')
    root.parent.mkdir(parents=True,exist_ok=True);P.coupled.archive.recovery.require_space(root.parent)
    root.mkdir();source.signed(root/'plan.json',p);(root/'experiment_card.md').write_text(CARD.read_text())
    print('A37 PREPARE PASS: lambda=1; 18 cases; 1440 new diagnostic calls, zero retraining; native BLIND reused from REG.',flush=True)


def done(root,p,case):
    complete = []
    for path in sorted((root/case['key']).glob('attempt_*/run.json')):
        r = source.read(path)
        if r['status'] != 'complete':continue
        r = source.load_signed(path)
        expected_alias = {'BLIND':'REG'} if case['sigma'] == rules.SIGMA_REFERENCE else {}
        if (r['case'] != case or r['plan_sha256'] != source.digest(root/'plan.json')
                or r['method_nfe'] != 0 or r['nfe'] != rules.calls(case['sigma'])
                or r['calls'] != list(reversed(p['grid']))*len(rules.active_arms(case['sigma']))
                or r['aliases'] != expected_alias or set(r['metrics']) != set(rules.ARMS)
                or set(r['traces']) != set(rules.ARMS)
                or any([v['t'] for v in r['traces'][a]] != list(reversed(p['grid'])) for a in rules.ARMS)
                or not r['gates'] or not all(g['passed'] for g in r['gates'].values())
                or source.digest(path.parent/'outputs.pt') != r['outputs_sha256']
                or r['source_feedback_sha256'] != p['identity_files'][p['source_records'][case['key']]['feedback']]):
            raise ValueError('Invalid complete A37 record: '+str(path))
        if expected_alias and (r['metrics']['BLIND'] != r['metrics']['REG'] or r['traces']['BLIND'] != r['traces']['REG']):
            raise ValueError('Native-noise alias differs')
        complete.append((path,r))
    if len(complete)>1:raise ValueError('Multiple complete outputs; no GT selection')
    return complete[0] if complete else None


def run_case(root,p,case,model,device,operator,p36):
    import torch
    from unittest.mock import patch
    from modules.zaps_algorithm import ZAPS
    from modules.main_single import load_image_as_tensor
    from utils.diag_zaps_sampler_parity import rng_state,same_rng
    from utils.diag_zaps_guidance_clipping import saved_cuda_selection,restore_sampling_rng
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.metrics import compute_all_metrics
    if done(root,p,case):return
    training,feedback = source_records(Path(p['source_A36']),p36,case)
    for path in (training[0],feedback[0],Path(training[1]['measurement']['path'])):
        if source.digest(path) != p['identity_files'][str(path)]:raise ValueError('Frozen A36 input changed')
    folder = root/case['key'];folder.mkdir(exist_ok=True)
    attempt = folder/f'attempt_{len(list(folder.glob("attempt_*"))):03d}';attempt.mkdir();path = attempt/'run.json'
    started = time.perf_counter()
    state = torch.load(training[0].parent/'pre_last_state.pt',map_location='cpu',weights_only=True)
    saved = torch.load(training[1]['measurement']['path'],map_location='cpu',weights_only=True)
    if saved['image_sha256'] != case['image_sha256'] or saved['seed'] != case['measurement_seed']:
        raise ValueError('Cached observation identity differs')
    y = saved['mean'].to(device)+case['sigma']*saved['standard_noise'].to(device)
    z = ZAPS(model,operator,img_size=256,**p['config']);z.tau = state['tau'].to(device)
    with torch.no_grad():z.zeta.copy_(state['zeta'].to(device));z.D.copy_(state['D'].to(device))
    if z.tau.tolist() != p['grid']:raise ValueError('Snapshot grid differs')
    positions = source.fields.policy.Positions(source.fields.policy.noise_u(z.dm.alphas_cumprod.cpu().numpy()),list(reversed(p['grid'])))
    with scan.preserved_rng():
        mapping = saved_cuda_selection(state,device);restore_sampling_rng(state,device,mapping['saved_sampling_index'])
        noises = [torch.randn_like(state['init_noise'].to(device)) for _ in range(29)]
    r = dict(case=case,method_nfe=0,plan_sha256=source.digest(root/'plan.json'),
             source_feedback_sha256=source.digest(feedback[0]),gates={},aliases={},sensor_seconds=0.)
    outputs,traces = {},{};external = rng_state(device);center_error = 0.
    with scan.counted(model,path,r,rules.calls(case['sigma'])):
        with torch.no_grad(),model_rng_guard(model.model,device),patch('torch.randn_like',side_effect=RuntimeError('Unexpected draw')):
            for arm in rules.active_arms(case['sigma']):
                x = state['init_noise'].to(device);trace = []
                for k,t in enumerate(positions.nominal):
                    f = source.fields.field(z,y,x,t,positions)
                    if arm == 'REG' and k == 0:
                        fingerprint = parent.rules.sensor(z,y,f['clean'],f['q'],t,case['sigma'])
                        old = feedback[1]['traces']['STATE'][0]
                        for name in ('gamma','R','qRMS','cosine','relative_response'):
                            value,target = fingerprint[name],old[name]
                            if value is None or target is None:
                                if value != target:raise RuntimeError('Unresolved center differs')
                            else:center_error = max(center_error,abs(value-target)/max(abs(target),1e-12))
                        if (center_error > 1e-6 or not math.isfinite(center_error)
                                or abs(f['clip_fraction']-old['clip']) > 2e-5):raise RuntimeError('A36 first-center scalar parity failed')
                    now = time.perf_counter()
                    if arm == 'TIME_NOISE':
                        reference = p['profiles']['values'][str(case['sigma'])][k]
                        readout = dict(gamma=reference['gamma'],reference_quadratic=reference)
                    else:
                        readout = rules.sensor(z,y,f['clean'],f['q'],t,case['sigma'],
                                               rules.SIGMA_REFERENCE if arm == 'BLIND' else None)
                    r['sensor_seconds'] += time.perf_counter()-now
                    gamma = readout['gamma']
                    trace.append(dict(step=k,t=t,gamma=gamma,R=f['R']*.05/case['sigma'],
                                      qRMS=float(f['q'].double().square().mean().sqrt()),clip=f['clip_fraction'],
                                      **{n:v for n,v in readout.items() if n not in ('gamma','R','qRMS')}))
                    base = P.terminal.terminal_output(z,x,f) if k == 29 else P.coupled.posterior(
                        x,f['clean'],t,positions.nominal[k+1],z.dm.alphas_cumprod,noises[k])+f['q']
                    x = base if gamma == 1. else base+(gamma-1.)*f['q']
                    if not torch.isfinite(x).all():raise RuntimeError('Nonfinite output; no fallback selection')
                outputs[arm] = x.cpu();traces[arm] = trace
        if case['sigma'] == rules.SIGMA_REFERENCE:
            outputs['BLIND'] = outputs['REG'];traces['BLIND'] = traces['REG'];r['aliases'] = {'BLIND':'REG'}
        if (r['nfe'] != rules.calls(case['sigma'])
                or r['calls'] != list(reversed(p['grid']))*len(rules.active_arms(case['sigma']))
                or not same_rng(external,rng_state(device)) or P.precision.precision_state(torch) != p['native_policy']
                or not all(torch.equal(v.cpu(),state[n]) for v,n in ((z.tau,'tau'),(z.zeta,'zeta'),(z.D,'D')))):
            raise RuntimeError('Budget/RNG/precision/parameters gate failed')
        parent.save_tensor(outputs,attempt/'outputs.pt')
        r.update(status='sampled',traces=traces,outputs_sha256=source.digest(attempt/'outputs.pt'));source.WRITE(path,r)
        gt = load_image_as_tensor(case['image']).cpu()
        metrics = {arm:dict(loss=float((y-z.A.H(out.to(device))).double().square().mean()/(2*case['sigma']**2)),
                            **compute_all_metrics(out,gt,lpips_net='alex')) for arm,out in outputs.items()
                   if arm != 'BLIND' or case['sigma'] != rules.SIGMA_REFERENCE}
        if r['aliases']:metrics['BLIND'] = dict(metrics['REG'])
        if any(not math.isfinite(v) for m in metrics.values() for v in m.values()):raise RuntimeError('Nonfinite metric')
        r.update(metrics=metrics,gates=dict(source_first_center=dict(passed=True,relative_error=center_error,scope='scalar fingerprint'),
                  paired_noise_budget=dict(passed=True),parameters_precision=dict(passed=True),GT_after_outputs=dict(passed=True),
                  native_alias=dict(passed=True,reused=bool(r['aliases']))))
    r.update(status='complete',seconds=time.perf_counter()-started);source.signed(path,r)
    print('A37 CASE:',case['key'],'complete; new calls=',r['nfe'],'REG-C PSNR=',metrics['REG']['psnr']-feedback[1]['metrics']['C']['psnr'],flush=True)


def worker(args):
    import fcntl
    import torch
    from modules.degradations import get_operator
    from utils.diag_ffhq_regression import TransposeMode
    from utils.diag_imagenet_gaussian_unit_domain import unit_operator
    root = Path(args.root).resolve();p = source.check_plan(root);p36 = source.check_plan(Path(p['source_A36']))
    with (root/f'shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        cases = [c for c in p['cases'] if p['image_ids'].index(c['image_id'])%2 == args.shard]
        if all(done(root,p,c) for c in cases):return
        runtime = dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                       gpu=torch.cuda.get_device_name(args.device),NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),
                       TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        if runtime != p['runtime'] or P.precision.precision_state(torch) != p['native_policy']:raise ValueError('Native runtime differs')
        torch.set_num_threads(2);P.coupled.archive.recovery.require_space(root)
        with scan.preserved_rng():
            model = P.coupled.archive.recovery.load_model(p,args.device)
            operator = unit_operator(TransposeMode(get_operator('super_resolution',device=args.device,scale_factor=4,noise_sigma=0.),'legacy_bicubic').to(args.device)).to(args.device)
            for case in cases:
                if source.digest(case['image']) != case['image_sha256']:raise ValueError('Target image changed')
                run_case(root,p,case,model,args.device,operator,p36)
            source.check_plan(root)


def interpretation(records):
    groups = []
    for sigma in parent.rules.SIGMAS:
        current = [r for r in records if r['case']['sigma'] == sigma]
        group = dict(sigma=sigma,completed_cases=len(current),planned_cases=6)
        if current:
            comparisons = {f'{a}_{b}':parent.rules.contrast(current,a,b) for a,b in
                           (('REG','C'),('REG','OLD'),('REG','TIME'),('REG','BLIND'),('REG','TIME_NOISE'),
                            ('BLIND','C'),('TIME_NOISE','C'))}
            group['comparisons'] = comparisons
            def baseline_candidate(name):
                c = comparisons[name]
                return len(current)==6 and c['mean_delta']['psnr']>=.05 and c['images_psnr_positive']>=4 and c['quality_guard']
            base = baseline_candidate('REG_C')
            def incremental(name):
                c = comparisons[name];d = c['mean_delta']
                perceptual_wins = sum(v['lpips'] < -1e-9 for v in c['per_image'].values())
                psnr = d['psnr']>=.02 and c['images_psnr_positive']>=4 and c['quality_guard']
                perceptual = d['lpips']<=-.005 and perceptual_wins>=4 and d['psnr']>=-.02 and d['ssim']>=-.002
                return base and (psnr or perceptual)
            group.update(REG_quality_candidate=base,TIME_NOISE_quality_candidate=baseline_candidate('TIME_NOISE_C'),
                         state_increment_candidate=incremental('REG_TIME_NOISE'),
                         noise_information_candidate=sigma!=rules.SIGMA_REFERENCE and incremental('REG_BLIND'))
            group['state_readouts'] = dict(
                mean_trust_weight=statistics.mean(v['trust_weight'] for r in current for v in r['traces']['REG']),
                mean_abs_gamma_vs_TIME_NOISE=statistics.mean(abs(a['gamma']-b['gamma']) for r in current
                                                            for a,b in zip(r['traces']['REG'],r['traces']['TIME_NOISE'])),
                terminal_clean_R_mean=statistics.mean(r['traces']['REG'][-1]['R'] for r in current),
                final_output_R_mean={a:statistics.mean(math.sqrt(2*r['metrics'][a]['loss']) for r in current)
                                     for a in ('C','OLD','REG','BLIND','TIME_NOISE')})
        groups.append(group)
    complete = all(g['completed_cases']==6 for g in groups)
    quality = complete and all(g['REG_quality_candidate'] for g in groups)
    return dict(classification=('NOISE_REGULARIZED_QUALITY_CANDIDATE_ONLY' if quality else
                                'MIXED_OR_NO_NOISE_REGULARIZATION_SUPPORT') if complete else 'INCOMPLETE_PILOT',
                completed_cases=len(records),planned_cases=18,groups=groups,quality_candidate=quality,
                state_increment_candidate_sigmas=[g['sigma'] for g in groups if g.get('state_increment_candidate')],
                noise_information_candidate_sigmas=[g['sigma'] for g in groups if g.get('noise_information_candidate')])


def pictures(path,r,reference_path,reference):
    import torch
    from PIL import Image,ImageDraw
    from modules.main_single import load_image_as_tensor
    from utils.metrics import _to_uint8_numpy
    from utils.diag_cached_guidance_final_value import font
    old = torch.load(reference_path.parent/'outputs.pt',map_location='cpu',weights_only=True)
    new = torch.load(path.parent/'outputs.pt',map_location='cpu',weights_only=True)
    outputs = dict(C=old['C'],TIME=old['TIME'],OLD=old['STATE'],**new)
    metrics = {'C':reference['metrics']['C'],'TIME':reference['metrics']['TIME'],'OLD':reference['metrics']['STATE'],**r['metrics']}
    names = ('GT','C','TIME','OLD','REG','BLIND','TIME_NOISE')
    sheet = Image.new('RGB',(20+276*len(names),680),'white');draw = ImageDraw.Draw(sheet)
    draw.text((20,10),r['case']['key']+' | noise-energy regularization; all arms300 NFE',fill='black',font=font(18))
    for j,arm in enumerate(names):
        raw = Image.fromarray(_to_uint8_numpy(load_image_as_tensor(r['case']['image']) if arm=='GT' else outputs[arm]))
        full = raw.copy();ImageDraw.Draw(full).rectangle((64,48,192,176),outline='#f8bd46',width=2);x = 20+276*j
        draw.text((x,45),arm,fill='black',font=font(17))
        if arm!='GT':
            m=metrics[arm];draw.text((x,70),f'P {m["psnr"]:.3f} S {m["ssim"]:.4f}',fill='black',font=font(14))
            draw.text((x,90),f'LPIPS {m["lpips"]:.4f}',fill='black',font=font(14))
        sheet.paste(full,(x,115));sheet.paste(raw.crop((64,48,192,176)).resize((256,256),Image.Resampling.NEAREST),(x,390))
    sheet.save(path.parent/'comparison.png')


def summarize(args):
    root=Path(args.root).resolve();p=source.check_plan(root);p36=source.check_plan(Path(p['source_A36']))
    combined=[];paths=[];refs=[]
    for case in p['cases']:
        item=done(root,p,case)
        if item is None:continue
        _,ref=source_records(Path(p['source_A36']),p36,case)
        if source.digest(ref[0]) != item[1]['source_feedback_sha256']:raise ValueError('Source outcome changed')
        metrics=dict(C=ref[1]['metrics']['C'],OLD=ref[1]['metrics']['STATE'],TIME=ref[1]['metrics']['TIME'],**item[1]['metrics'])
        combined.append(dict(item[1],metrics=metrics));paths.append(item[0]);refs.append(ref)
    result=interpretation(combined)
    attempts=[source.read(f) for f in root.glob('*/attempt_*/run.json')]
    result.update(experiment=p['experiment'],method_nfe=0,diagnostic_nfe=sum(r['nfe'] for r in attempts),
                  planned_diagnostic_nfe=p['planned_diagnostic_nfe'],deployment_nfe_per_arm=300,
                  partial_attempts=sum(r['status']!='complete' for r in attempts),source_A36_cost=p['source_cost'],
                  cost_is_lower_bound_if_abruptly_interrupted=any(r['status'] in ('running','sampled') for r in attempts),
                  limits=p['scope']+' Six image clusters; descriptive unadjusted bootstrap only. '
                         'Response/noise ratio is an engineering scale, not calibrated confidence.')
    source.signed(root/'summary.json',result)
    if args.archive:
        for path,r,ref in zip(paths,combined,refs):pictures(path,r,ref[0],ref[1])
    compact={k:v for k,v in result.items() if k!='groups'}
    compact['groups']=[{**{k:v for k,v in g.items() if k!='comparisons'},
                        'comparisons':{name:{k:v for k,v in c.items() if k!='per_image'} for name,c in g.get('comparisons',{}).items()}}
                       for g in result['groups']]
    print('A37 Summary:',json.dumps(compact,ensure_ascii=False),flush=True);print('Records:',root,flush=True)
    if args.archive:
        from utils.archive_refinement_confirmation import export_and_commit
        bundle,commit=export_and_commit({'A37':str(root)});print('Archive:',bundle,'Results commit:',commit,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=ROOT)
    p.add_argument('--a36-dir',default=parent.ROOT);p.add_argument('--prepare',action='store_true')
    p.add_argument('--shard',type=int,choices=(0,1));p.add_argument('--device',default='cuda:0');p.add_argument('--archive',action='store_true')
    args=p.parse_args()
    if args.prepare:prepare(args)
    elif args.shard is not None:worker(args)
    else:summarize(args)


if __name__=='__main__':main()
