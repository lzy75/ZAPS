"""A3: offline paired DDPM refinement, 180 diagnostic forwards, no training.

Two prespecified bridges per interval. Frozen-field endpoint equivalence is a
technical gate, never evidence that neural refinement improves reconstruction.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
from utils import archive_guidance_replication_stage as archive
from utils import diag_direction_refresh as history
from utils import diag_ddpm_refinement_contract as contract
from utils.diag_state_trajectory import digest

CARD = PROJECTS/'experiment_cards/A3_coupled_refinement_20261020.md'
TIMES = history.TIMES
WRITE = archive.recovery.write_json
METRICS = history.METRICS


def interval_spec(alpha, t, s):
    import numpy as np
    values = np.asarray(alpha, dtype=np.float64)
    if not 0 <= s < t-1 < len(values)-1:
        raise ValueError('Need an original interval with a strict interior and nonterminal s')
    if not np.isfinite(values).all() or not ((values > 0) & (values < 1)).all():
        raise ValueError('Invalid cumulative alphas')
    logs = .5*(np.log1p(-values)-np.log(values))
    candidates = np.arange(s+1, t)
    m = int(candidates[np.argmin(abs(logs[candidates]-.5*(logs[t]+logs[s])))])
    result = contract.interval(values[t], values[m], values[s])
    result.update(t=t, m=m, s=s)
    return result


def bridge_seed(case_index, step, bridge):
    if not (0 <= case_index < 6 and step in (8,19,24,27,28) and bridge in (0,1)):
        raise ValueError('Unplanned case, step or bridge')
    return 322010+100*case_index+2*step+bridge


def candidate(effect):
    d = effect['delta']
    practical = d['psnr'] >= .1 or d['ssim'] >= .001 or d['lpips'] <= -.001
    return bool(effect['joint_local_candidate'] and practical)


def summarize(records):
    complete = [r for r in records if r['status'] == 'complete']
    layers = []
    for t in TIMES:
        points = [(r['case'],p) for r in complete for p in r['points'] if p['source_t'] == t]
        images = {c['image_id'] for c,p in points}
        hits = [(c,p) for c,p in points if p['both_bridges_candidate']]
        consistent = sum(len([p for c,p in points if c['image_id'] == image]) == 2 and
                         len([p for c,p in hits if c['image_id'] == image]) == 2 for image in images)
        means = {arm: {metric: statistics.mean(p['effects'][arm]['delta'][metric] for c,p in points) if points else None
                       for metric in ('loss',*METRICS)} for arm in ('F0_C','F1_C')}
        layers.append(dict(source_t=t,n=len(points),image_clusters=len(images),
                           joint_local_points=len(hits),images_both_seeds_both_bridges=consistent,mean_delta=means,
                           bridge_candidates={arm:sum(p['bridge_candidates'][arm] for c,p in points) for arm in ('F0','F1')}))
    support = len(complete) == 6 and any(l['images_both_seeds_both_bridges'] >= 2 for l in layers)
    return dict(completed_cases=len(complete),planned_cases=6,
                method_nfe=sum(r['method_nfe'] for r in records),diagnostic_nfe=sum(r['diagnostic_nfe'] for r in records),
                classification=('LOCAL_COUPLED_REFINEMENT_CANDIDATE_ONLY' if support else 'LOCAL_COUPLED_REFINEMENT_NOT_SUPPORTED') if len(complete)==6 else 'INCOMPLETE_PILOT',
                layers=layers,limits='Three development image clusters; two bridges are not independent images. Next-time endpoints only; no final reconstruction, scheduling, cosine-control, generalization or GT selection claim.')


def bind(args):
    root, a2d, old, parents = history.bind(SimpleNamespace(source_plan_dir=args.source_plan_dir,
                                    a2c_dir=args.a2c_dir,run_dir=args.a2d_dir,inspect=True))
    records = history.completed(a2d, old)
    parent_config = json.loads((root/'plan.json').read_text())['config']
    if parent_config['sampler_mode'] != 'ddpm' or parent_config['eta'] != 1 or parent_config['use_learned_var']:
        raise ValueError('Bridge requires original DDPM fixed variance eta=1')
    if len(records) != 6 or history.stored_summary(a2d) != json.loads((a2d/'summary.json').read_text()):
        raise ValueError('Need six complete audited A2d cases and exact summary')
    paths = [a2d/'plan.json',a2d/'summary.json',CARD,Path(__file__),Path(history.__file__),Path(contract.__file__),
             PROJECTS/'run_a3_20261020.sh',PROJECTS/'utils/diag_cached_direction.py',
             PROJECTS/'utils/diag_zaps_guidance_clipping.py',PROJECTS/'utils/test_diag_coupled_refinement.py',
             PROJECTS/'utils/test_diag_ddpm_refinement_contract.py']
    for key in sorted(parents):
        paths.extend([a2d/key/'run.json',a2d/key/'run.sha256',a2d/key/'branches.pt',
                      Path(old['source_cases'][key])/'last_opt.pt'])
    plan = dict(experiment='coupled_refinement_plan',source=str(root),source_cases=old['source_cases'],
                a2c_source=old['a2c_source'],a2d_source=str(a2d),times=list(TIMES),
                identity_files={**old['identity_files'],**{str(p):digest(p) for p in paths}},
                bridge_seed_rule='322010+100*sorted_case_index+2*source_step+bridge_index',bridges=[0,1],
                practical_thresholds=dict(psnr=.1,ssim=.001,lpips=-.001),
                mapping='Original interval G_t, kappa=1/(1+A_ms), log-sigma nearest interior midpoint; endpoint-anchored affine increments',
                planned_method_nfe=0,planned_diagnostic_nfe=180,
                parent_cost=old['parent_cost'],preceding_a2c_cost=old['preceding_a2c_cost'],
                preceding_a2d_cost={k:history.summarize(list(records.values()))[k] for k in ('method_nfe','diagnostic_nfe')})
    directory = Path(args.run_dir).resolve()
    for source in (root,Path(old['a2c_source']),a2d):
        if directory.is_relative_to(source) or source.is_relative_to(directory):
            raise ValueError('New run must be separate from all historical sources')
    if directory.exists():
        if json.loads((directory/'plan.json').read_text()) != plan:
            raise ValueError('Frozen plan differs; do not overwrite or silently change mapping')
    elif not args.inspect:
        directory.mkdir(parents=True,exist_ok=False); WRITE(directory/'plan.json',plan)
    return root,directory,plan,parents


def completed(directory, plan):
    records = {}
    for key in sorted(plan['source_cases']):
        path = directory/key/'run.json'
        if not path.exists():
            if path.parent.exists(): raise ValueError('Interrupted case directory retained: '+key)
            continue
        r = json.loads(path.read_text())
        if (digest(path) != (path.parent/'run.sha256').read_text().strip() or
                r['case']['key'] != key or r['status'] != 'complete' or r['plan_sha256'] != digest(directory/'plan.json') or
                r['method_nfe'] != 0 or r['diagnostic_nfe'] != 30 or len(r['calls']) != 30 or len(r['points']) != 5 or
                tuple(p['source_t'] for p in r['points']) != TIMES or not r['gates'] or
                not all(g['passed'] for g in r['gates'].values()) or digest(path.parent/'branches.pt') != r['branch_archive']['sha256']):
            raise ValueError('Failed/interrupted/incompatible case retained: '+key)
        for p in r['points']:
            effects = {a+'_C':history.effect(p['branches'][a],p['branches']['C'],p['floors']) for a in ('F0','F1')}
            hits = {a:candidate(effects[a+'_C']) for a in ('F0','F1')}
            if effects != p['effects'] or hits != p['bridge_candidates'] or all(hits.values()) != p['both_bridges_candidate']:
                raise ValueError('Saved effect or H5 classification differs: '+key)
        records[key] = r
    return records


def frozen_guidance(z, y, clean, t, index):
    # The midpoint denoiser uses m, but both corrections use the original G_t.
    ab = z.dm.alphas_cumprod[t]
    v = z.A.transpose(y-z.A.H(clean))
    hv = z.dwt.synthesis(z.D[index]*z.dwt.analysis(v))
    return z.zeta[index]*((v+(1-ab)*hv)/ab.sqrt().clamp(min=1e-8))


def posterior(x, clean, t, s, alpha, noise):
    import torch
    from modules.zaps_algorithm import ddpm_posterior_step
    if noise.shape != x.shape or noise.dtype != x.dtype or noise.device != x.device or s < 0:
        raise ValueError('Explicit draw contract differs')
    with patch('torch.randn_like',return_value=noise) as draw:
        result = ddpm_posterior_step(x,clean,t,s,alpha,eta=1.,mode='ddpm')
    if draw.call_count != 1: raise RuntimeError('Expected exactly one supplied DDPM draw')
    return result


def refined_endpoint(coarse, clean0, q0, clean_mid, q_mid, spec):
    """Exact affine difference from the frozen-field endpoint, stably evaluated.

    In exact arithmetic the two substeps equal the coupled coarse endpoint
    plus B_ms*(clean_mid-clean0)+kappa*(G_t(clean_mid)-q0). Subtracting
    separately rounded posterior means loses this identity near alpha=1.
    Coarse is the original cached endpoint, not a new high-precision baseline.
    """
    if any(v.shape != coarse.shape for v in (clean0,q0,clean_mid,q_mid)):
        raise ValueError('Endpoint increment shapes differ')
    correction = spec['second']['b']*(clean_mid.double()-clean0.double())
    correction = correction+spec['local_guidance_factor']*(q_mid.double()-q0.double())
    return (coarse.double()+correction).to(coarse.dtype)


def frozen_bridge(x, clean, q, noise, auxiliary, spec, alpha):
    n1,n2 = contract.coupled_noises(noise,auxiliary,spec)
    t,m,s,dose = spec['t'],spec['m'],spec['s'],spec['local_guidance_factor']
    mid = posterior(x,clean,t,m,alpha,n1)+dose*q
    coarse = posterior(x,clean,t,s,alpha,noise)+q
    fine = refined_endpoint(coarse,clean,q,clean,q,spec)
    return mid,fine,n1,n2


def replay_noises(state, cache, parent, z, y, device, last_opt):
    import torch
    from utils.diag_cached_direction import cached_step
    from utils.diag_zaps_guidance_clipping import saved_cuda_selection,restore_sampling_rng
    from utils.diag_zaps_sampler_parity import rng_state,same_rng
    from utils.diag_zaps_trace_audit import relative_error
    mapping = saved_cuda_selection(state,device)
    noises,checks = {},[]
    original = torch.randn_like
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        restore_sampling_rng(state,device,mapping['saved_sampling_index'])
        for k,row in enumerate(parent['rows']):
            draws = []
            def capture(*args,**kwargs):
                value = original(*args,**kwargs); draws.append(value.detach().clone()); return value
            i,t,s = row['parameter_index'],row['t'],row['t_prev']
            with patch('torch.randn_like',side_effect=capture):
                replay = cached_step(cache['latent_inputs'][k:k+1].to(device),cache['clean'][k:k+1].to(device),
                                     y,z.A,z.dwt,z.zeta[i],z.D[i],t,s,z.dm.alphas_cumprod)
            expected = cache['latent_inputs'][k+1:k+2] if k<29 else last_opt
            err = relative_error(expected,replay['next'].cpu())
            qerr = relative_error(cache['guidance'][k:k+1],replay['correction'].cpu())
            if not math.isfinite(err+qerr) or max(err,qerr)>1e-6 or len(draws) != int(s>=0):
                raise RuntimeError('Original transition / raw-noise replay gate failed at step '+str(k))
            if draws: noises[k] = draws[0]
            checks.append(dict(step=k,next_error=err,q_error=qerr,draws=len(draws)))
        suffix = rng_state(device)
        restore_sampling_rng(state,device,mapping['saved_sampling_index'])
        for _ in range(29): original(state['init_noise'].to(device))
        if len(noises)!=29 or not same_rng(suffix,rng_state(device)):
            raise RuntimeError('Original 29-draw RNG suffix differs')
    return noises,dict(passed=True,mapping=mapping,checks=checks,noise_draws=29)


def run_case(parent_plan, parent, source, directory, plan, model, device):
    import torch
    from modules.zaps_algorithm import ZAPS
    from modules.degradations import get_operator
    from modules.main_single import load_image_as_tensor
    from utils.diag_ffhq_regression import TransposeMode
    from utils.diag_imagenet_gaussian_unit_domain import unit_operator
    from utils.diag_guidance_local_probe import evaluate_field
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state,same_rng
    from utils.diag_zaps_trace_audit import relative_error
    from utils.diag_cached_guidance_geometry import rms
    from utils.metrics import compute_all_metrics
    directory.mkdir(exist_ok=False)
    (directory/'experiment_card.md').write_text(CARD.read_text(),encoding='utf-8')
    record = dict(experiment='coupled_refinement_case',status='running',case=parent['case'],
                  plan_sha256=digest(directory.parent/'plan.json'),method_nfe=0,diagnostic_nfe=0,
                  calls=[],points=[],progress=[],gates={},config=parent['config'],
                  git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECTS,text=True).strip(),
                  runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda),experiment_card=CARD.read_text(),
                  outcome='Offline next-time clean endpoints; no training, rollout, online selection or GT branch choice')
    started = time.perf_counter()
    def save():
        record['seconds'] = time.perf_counter()-started; WRITE(directory/'run.json',record)
    save()
    context = {}
    def count(_module, inputs):
        record['diagnostic_nfe'] += 1
        record['calls'].append(dict(context,t=int(inputs[1].flatten()[0]),cumulative=record['diagnostic_nfe']))
        if record['diagnostic_nfe']>30 or record['calls'][-1]['t']!=context['eval_t']:
            raise RuntimeError('UNet budget/time gate failed')
    hook = model.model.register_forward_pre_hook(count)
    try:
        before = rng_state(device)
        baseline_checks,null_checks = [],[]
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))),torch.no_grad():
            state = torch.load(source/'pre_last_state.pt',map_location='cpu',weights_only=True)
            cache = torch.load(source/'trajectory.pt',map_location='cpu',weights_only=True)
            last_opt = torch.load(source/'last_opt.pt',map_location='cpu',weights_only=True)
            if any(tuple(cache[k].shape)!=(30,3,256,256) or cache[k].dtype!=torch.float32 or not torch.isfinite(cache[k]).all()
                   for k in ('clean','guidance','latent_inputs')) or state['tau'].tolist()!=parent_plan['grid']:
                raise ValueError('Source tensor/grid contract differs')
            base = get_operator('super_resolution',device=device,scale_factor=4,noise_sigma=0.)
            operator = unit_operator(TransposeMode(base,'legacy_bicubic').to(device)).to(device)
            z = ZAPS(model,operator,img_size=256,**parent_plan['config'])
            z.tau = state['tau'].to(device)
            z.zeta.copy_(state['zeta'].to(device)); z.D.copy_(state['D'].to(device))
            y = torch.load(parent['measurement']['path'],map_location='cpu',weights_only=True)['y'].to(device)
            noises,replay_gate = replay_noises(state,cache,parent,z,y,device,last_opt)
            record['gates']['source_noise_replay'] = replay_gate; save()
            alpha = z.dm.alphas_cumprod
            case_index = sorted(plan['source_cases']).index(parent['case']['key'])
            pending,saved_inputs = [],[]
            with model_rng_guard(model.model,device):
                for probe in parent['probes']:
                    k,t = probe['step'],probe['t']
                    row = history.next_row(parent['rows'],probe)
                    s,index = row['t'],row['parameter_index']
                    current_index = parent['rows'][k]['parameter_index']
                    if current_index!=29-k: raise ValueError('Original parameter index differs')
                    spec = interval_spec(alpha.detach().cpu().numpy(),t,s)
                    expected_mid = {903:896,641:619,319:267,54:28,14:3}[t]
                    if spec['m']!=expected_mid: raise RuntimeError('Actual alpha midpoint differs from preregistration')
                    m,dose = spec['m'],spec['local_guidance_factor']
                    x = cache['latent_inputs'][k:k+1].to(device)
                    clean0 = cache['clean'][k:k+1].to(device)
                    q0 = cache['guidance'][k:k+1].to(device)
                    x_next = cache['latent_inputs'][k+1:k+2].to(device)
                    if not math.isclose(rms(q0),probe['q_rms'],rel_tol=1e-6,abs_tol=1e-12):
                        raise ValueError('Original correction RMS differs')
                    if relative_error(q0,frozen_guidance(z,y,clean0,t,current_index))>1e-6:
                        raise RuntimeError('Frozen G_t does not reproduce original q')
                    # Check both null bridges before interpreting any neural branch at this point.
                    constructions = []
                    artifact = dict(source_step=k,spec=spec,Z=noises[k].cpu())
                    for bridge in (0,1):
                        seed = bridge_seed(case_index,k,bridge)
                        generator = torch.Generator(device='cpu').manual_seed(seed)
                        auxiliary = torch.randn(x.shape,generator=generator,device='cpu',dtype=torch.float32).to(device)
                        mid,null,n1,n2 = frozen_bridge(x,clean0,q0,noises[k],auxiliary,spec,alpha)
                        error = relative_error(x_next,null)
                        absolute = float((x_next.double()-null.double()).abs().max())
                        raw_null = posterior(mid,clean0,m,s,alpha,n2)+dose*q0
                        null_check = dict(step=k,bridge=bridge,seed=seed,relative_error=error,max_abs=absolute,
                                          raw_float32_substep_relative_defect=relative_error(x_next,raw_null),
                                          raw_float32_substep_max_abs=float((x_next.double()-raw_null.double()).abs().max()))
                        if not math.isfinite(error+absolute) or error>1e-6 or absolute>1e-5:
                            raise RuntimeError('Frozen-field coupled endpoint gate failed: '+json.dumps(null_check))
                        null_checks.append(null_check)
                        constructions.append((bridge,seed,mid,n2))
                        artifact['aux'+str(bridge)] = auxiliary.cpu()
                        artifact['mid'+str(bridge)] = mid.cpu()
                    centers = []
                    context.clear();context.update(source_step=k,source_t=t,midpoint_t=m,next_t=s,eval_t=s,parameter_index=index,arm='C')
                    for repeat in (0,1):
                        context['repeat'] = repeat
                        centers.append(evaluate_field(z,y,x_next.clone(),s,index))
                    check = dict(step=k,clean_error=relative_error(cache['clean'][k+1:k+2],centers[0]['clean'].cpu()),
                                 q_error=relative_error(cache['guidance'][k+1:k+2],centers[0]['q'].cpu()),
                                 repeat_error=relative_error(centers[0]['clean'],centers[1]['clean']),
                                 loss_error=abs(centers[0]['loss']-row['measurement_loss']))
                    if (not all(math.isfinite(v) for v in check.values()) or max(check['clean_error'],check['q_error'])>1e-6 or
                            check['repeat_error']>1e-7 or check['loss_error']>1e-7+2e-5*abs(row['measurement_loss'])):
                        raise RuntimeError('Baseline field parity failed: '+json.dumps(check))
                    baseline_checks.append(check)
                    branches = dict(C=dict(clean=centers[0]['clean'].cpu(),loss=centers[0]['loss'],clip=centers[0]['clip_fraction']))
                    bridges = []
                    for bridge,seed,mid,n2 in constructions:
                        arm = 'F'+str(bridge)
                        context.update(eval_t=m,parameter_index=current_index,guidance_t=t,guidance_index=current_index,arm=arm+'_mid',repeat=None,bridge=bridge)
                        middle = evaluate_field(z,y,mid,m,current_index)
                        q_mid_frozen = frozen_guidance(z,y,middle['clean'],t,current_index)
                        fine = refined_endpoint(x_next,clean0,q0,middle['clean'],q_mid_frozen,spec)
                        if not torch.isfinite(q_mid_frozen).all() or not torch.isfinite(fine).all():
                            raise RuntimeError('Non-finite coupled branch')
                        context.update(eval_t=s,parameter_index=index,guidance_t=s,guidance_index=index,arm=arm)
                        field = evaluate_field(z,y,fine,s,index)
                        branches[arm] = dict(clean=field['clean'].cpu(),loss=field['loss'],clip=field['clip_fraction'])
                        artifact[arm] = fine.cpu()
                        bridges.append(dict(bridge=bridge,seed=seed,midpoint_clip=middle['clip_fraction'],
                                            midpoint_loss=middle['loss'],frozen_G_t_rms=rms(q_mid_frozen),
                                            native_G_m_rms=rms(middle['q'])))
                    saved_inputs.append(artifact)
                    # Persist actual raw draws and branch inputs after every point, including interrupted attempts.
                    temporary = directory/'branches.pt.tmp'
                    torch.save(dict(points=saved_inputs),temporary);temporary.replace(directory/'branches.pt')
                    record['branch_archive'] = dict(path=str(directory/'branches.pt'),sha256=digest(directory/'branches.pt'),
                                                    scope='CPU float32 raw coarse Z, auxiliary draws, midpoints and F0/F1 next inputs; five original intervals')
                    pending.append((probe,row,spec,branches,centers[1]['clean'].cpu(),
                                    abs(centers[1]['loss']-centers[0]['loss']),bridges))
                    record['progress'].append(dict(source_step=k,source_t=t,midpoint_t=m,next_t=s,
                                                    diagnostic_nfe=record['diagnostic_nfe'],null_gate_passed=True))
                    save(); print('PROGRESS:',parent['case']['key'],'t=',t,'m=',m,'calls=',record['diagnostic_nfe'],flush=True)
            if not (torch.equal(z.zeta.cpu(),state['zeta']) and torch.equal(z.D.cpu(),state['D']) and torch.equal(z.tau.cpu(),state['tau'])):
                raise RuntimeError('Parameters/grid changed')
            # All network calls and branch states are fixed before GT is loaded for offline evaluation.
            gt = load_image_as_tensor(parent['case']['image']).cpu()
            for probe,row,spec,branches,repeated_clean,repeated_loss,bridges in pending:
                for branch in branches.values():
                    branch.update(compute_all_metrics(branch.pop('clean'),gt,lpips_net='alex'))
                baseline = branches['C']
                repeat_metrics = compute_all_metrics(repeated_clean,gt,lpips_net='alex')
                drifts = {key:abs(baseline[key]-row[key]) for key in METRICS}
                if any(drifts[key]>limit for key,limit in (('psnr',.01),('ssim',.001),('lpips',.001))):
                    raise RuntimeError('Baseline metric convention/parity changed')
                floors = {key:max(drifts[key],abs(repeat_metrics[key]-baseline[key])) for key in METRICS}
                floors['loss'] = max(repeated_loss,abs(baseline['loss']-row['measurement_loss']))
                effects = {a+'_C':history.effect(branches[a],baseline,floors) for a in ('F0','F1')}
                hits = {a:candidate(effects[a+'_C']) for a in ('F0','F1')}
                point = dict(source_step=probe['step'],source_t=probe['t'],next_t=row['t'],spec=spec,
                             branches=branches,floors=floors,effects=effects,bridges=bridges,
                             bridge_candidates=hits,both_bridges_candidate=all(hits.values()),
                             predictors={key:parent['rows'][probe['step']][key] for key in
                                         ('noise_u','guidance_rms','guidance_cosine','guidance_variation3_per_u',
                                          'residual_noise_ratio','raw_x0_clip_fraction','delta_u')})
                record['points'].append(point); save()
                print('POINT:',parent['case']['key'],json.dumps(point),flush=True)
        if not same_rng(before,rng_state(device)): raise RuntimeError('External RNG changed')
        if record['diagnostic_nfe']!=30 or len(record['points'])!=5: raise RuntimeError('Expected thirty calls/five points')
        if any(digest(p)!=sha for p,sha in plan['identity_files'].items()): raise RuntimeError('Frozen source/implementation changed')
        record['gates'].update(baseline=dict(passed=True,checks=baseline_checks),
                              frozen_field_coupling=dict(passed=True,checks=null_checks),
                              parameters_grid_rng=dict(passed=True),identity_and_budget=dict(passed=True,method_nfe=0,diagnostic_nfe=30))
        record['status']='complete';save()
        (directory/'run.sha256').write_text(digest(directory/'run.json')+'\n')
        print('CASE:',parent['case']['key'],'Gates:',json.dumps(record['gates']),flush=True)
        return record
    except BaseException as error:
        record['status']='failed';record['error']=repr(error);save();raise
    finally:
        hook.remove()


def stored_summary(directory):
    attempts = [json.loads(p.read_text()) for p in sorted(directory.glob('*/run.json'))]
    result = summarize(attempts)
    result['cost_is_lower_bound_if_abruptly_interrupted'] = any(r['status'] == 'running' for r in attempts)
    return result


def execute(args):
    root, directory, plan, parents = bind(args)
    print('Source AUDIT PASS: six A2a, A2c and A2d cases; planned NFE=0 method +180 diagnostic.', flush=True)
    if args.inspect:
        print('A3 INSPECT PASS: six cases x five points x six calls. Offline paired intervals only; original production grid unchanged.', flush=True)
        return
    records = completed(directory, plan)
    keys = sorted(parents)  # The predefined 00020_s1000 technical smoke runs first.
    todo = [key for key in keys if key not in records]
    try:
        if todo:
            archive.recovery.require_space(directory)
            parent_plan = json.loads((root/'plan.json').read_text())
            import torch
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                model = archive.recovery.load_model(parent_plan, args.device)
                for key in todo:
                    records[key] = run_case(parent_plan, parents[key], Path(plan['source_cases'][key]), directory/key, plan, model, args.device)
    finally:
        summary = stored_summary(directory)
        WRITE(directory/'summary.json', summary)
        print('A3 Summary:', json.dumps(summary), '\nRecords:', directory, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-plan-dir', default='results/diag_guidance_replication/A2a_20261018')
    parser.add_argument('--a2c-dir', default='/data0/lzy/ZAPS-results/diag_one_step_guidance_scale/A2c_20261019')
    parser.add_argument('--a2d-dir', default='/data0/lzy/ZAPS-results/diag_direction_refresh/A2d_20261019')
    parser.add_argument('--run-dir', default='/data0/lzy/ZAPS-results/diag_coupled_refinement/A3_20261020')
    parser.add_argument('--device', default='cuda')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--inspect', action='store_true')
    group.add_argument('--run', action='store_true')
    group.add_argument('--summary-only', action='store_true')
    args = parser.parse_args()
    directory = Path(args.run_dir).resolve()
    if args.summary_only:
        if not (directory/'plan.json').is_file():
            raise ValueError('No stored A3 plan; no experiment started')
        print('A3 Summary:', json.dumps(stored_summary(directory)), '\nRecords:', directory)
    elif args.inspect:
        execute(args)
    else:
        # Per-run advisory lock survives SSH disconnection with the process.
        # Concurrent submits cannot enter bind/create or overwrite a running case.
        import fcntl
        directory.parent.mkdir(parents=True, exist_ok=True)
        with (directory.parent/(directory.name+'.lock')).open('a+') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('A3 is already running; inspect the existing log instead of resubmitting') from None
            execute(args)


if __name__ == '__main__':
    main()
