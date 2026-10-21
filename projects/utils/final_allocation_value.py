"""A30 frozen analysis of one extra call's FINAL value; no online allocation policy."""
import math
import numpy as np
from utils import state_action_response as old

STEPS = old.STEPS
CALLS_PER_CASE = sum(30-k for k in STEPS)
FEATURES = dict(STAGE=(), PHYSICAL=old.FEATURES['PHYSICAL'],
                RISK=old.FEATURES['PHYSICAL']+('mean_error',),
                PHYS_RESPONSE=old.FEATURES['PHYSICAL']+('mean_guidance_gain',),
                DUAL=old.FEATURES['PHYSICAL']+('mean_error', 'mean_guidance_gain'))


def analyze(records, original_points):
    if len(records) != 72:
        raise ValueError('Need all 72 complete final counterfactuals')
    rows = []
    for r in records:
        p = next(p for p in original_points if (p['image_id'], p['seed'], p['step']) ==
                 (r['case']['image_id'], r['case']['sampling_seed'], r['step']))
        row = old.derive(p)
        # This experiment refines the ORIGINAL interval, not A28's 1.5x interval.
        response = next(f for f in p['refinements'] if f['factor'] == 1.)
        row['features']['mean_error'] = math.log1p(response['mean_error'])
        row['labels'] = dict(gain_psnr=r['metrics']['psnr']-r['baseline']['psnr'],
                             gain_loss=r['baseline']['loss']-r['metrics']['loss'])
        rows.append(row)
    images = sorted({r['image_id'] for r in rows})
    if len(images) != 6 or {(r['image_id'], r['seed'], r['step']) for r in rows} != {
            (f'{i:05d}', s, k) for i in range(20, 26) for s in (1000, 2000) for k in STEPS}:
        raise ValueError('Original six-image/two-seed/whole-stage identity differs')
    # Per-image averaging prevents treating two seeds as independent images.
    gains = {i: {k: float(np.mean([r['labels']['gain_psnr'] for r in rows if r['image_id'] == i and r['step'] == k]))
                 for k in STEPS} for i in images}
    fixed_means = {k: float(np.mean([gains[i][k] for i in images])) for k in STEPS}
    best = min(STEPS, key=lambda k: (-fixed_means[k], k))
    oracle = float(np.mean([max(gains[i].values()) for i in images]))
    headroom = oracle-fixed_means[best]
    loio_fixed = []
    for image in images:
        chosen = min(STEPS, key=lambda k: (-float(np.mean([gains[i][k] for i in images if i != image])), k))
        loio_fixed.append(dict(image_id=image, chosen_step=chosen, psnr_delta_vs_C=gains[image][chosen]))
    targets = {}
    for target in ('gain_psnr', 'gain_loss'):
        fits = {n: old.cross_validate(rows, fs, target) for n, fs in FEATURES.items()}
        comparisons = {}
        for name in ('RISK', 'PHYS_RESPONSE', 'DUAL'):
            comparisons[name] = {}
            for baseline in ('STAGE', 'PHYSICAL'):
                a, b = fits[name], fits[baseline]
                reduction = 1-a['mae']/b['mae'] if b['mae'] > 1e-10 else None
                count = sum(a['per_image_mae'][i] < b['per_image_mae'][i]-1e-10 for i in images)
                comparisons[name][baseline] = dict(relative_mae_reduction=reduction, images_better=count,
                    threshold_passed=reduction is not None and reduction >= .10 and count >= 4)
        targets[target] = dict(models=fits, comparisons=comparisons)
    support = {name: all(targets['gain_psnr']['comparisons'][name][b]['threshold_passed'] for b in ('STAGE', 'PHYSICAL'))
               for name in ('RISK', 'PHYS_RESPONSE', 'DUAL')}
    layers = []
    for k in STEPS:
        group = [r for r in records if r['step'] == k]
        delta = {m: float(np.mean([r['metrics'][m]-r['baseline'][m] for r in group])) for m in ('psnr', 'ssim', 'lpips', 'loss')}
        layers.append(dict(step=k, source_t=group[0]['t'], midpoint=group[0]['m'], mean_delta_vs_C=delta,
                           images_positive=sum(gains[i][k] > 1e-9 for i in images),
                           quality_guard_vs_C=delta['ssim'] >= -.002 and delta['lpips'] <= .005))
        if all('fixed_F_reference' in r for r in group):
            layers[-1]['mean_delta_vs_fixed_F_300_reference'] = {
                m: float(np.mean([r['metrics'][m]-r['fixed_F_reference'][m] for r in group])) for m in ('psnr', 'ssim', 'lpips', 'loss')}
    classification = ('NO_MATERIAL_FINAL_ALLOCATION_HEADROOM' if headroom < .01 else
                      'FINAL_ALLOCATION_INFORMATION_CANDIDATE_ONLY' if any(support.values()) else
                      'STATE_FINAL_ALLOCATION_VALUE_NOT_SUPPORTED')
    return dict(classification=classification, image_clusters=6, points=72, layers=layers, targets=targets,
                predictive_support=support, best_fixed_step_in_sample=best,
                same_301_call_oracle_headroom_psnr=headroom, loio_calibrated_fixed=loio_fixed,
                loio_fixed_mean_psnr_delta_vs_C=float(np.mean([r['psnr_delta_vs_C'] for r in loio_fixed])),
                limits='All six candidate trajectories=301 denoiser calls; C=300 is reference only. Oracle uses GT and all candidate locations, not causal selection. Cached A28 sensor acquisition excluded from trajectory count, so no deployed 301-NFE state-policy claim. Six reused development images; fixed ridge1; no p-values or independent confirmation.')
