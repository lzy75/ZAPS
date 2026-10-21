"""A28 frozen NumPy analysis: within-stage state information, no online policy."""
import math
import numpy as np

STEPS = (2, 10, 19, 24, 26, 27)
SPANS = (.5, 1., 1.5)
GAMMAS = (.5, 1., 2.)
ETAS = (.01, .02)
CALLS_PER_POINT = 3 + 8 + 3 * 9
TARGETS = ('large_step_error', 'guidance_loss_gain', 'guidance_psnr_gain', 'large_refinement_psnr_gain')
FEATURES = {
    'PHYSICAL': ('log_R', 'log_q', 'clip', 'log_move'),
    'HISTORY': ('log_R', 'log_q', 'clip', 'log_move', 'history_turn'),
    'LOCAL': ('log_R', 'log_q', 'clip', 'log_move', 'q_dir', 'q_mag', 'travel_dir', 'travel_mag'),
    'RESPONSE': ('log_R', 'log_q', 'clip', 'log_move', 'mean_error', 'mean_guidance_gain'),
}


def next_times(u, t, nominal_s):
    """Whole remaining interval; no original Voronoi-cell position restriction."""
    u = np.asarray(u, dtype=float)
    if not 0 <= nominal_s < t - 1 < len(u) - 1 or not np.all(np.diff(u) < 0):
        raise ValueError('Invalid interval/noise coordinates')
    options = np.arange(t - 1)  # s <= t-2, so a strict interior exists
    result = []
    for factor in SPANS:
        target = min(1., u[t] + factor * (u[nominal_s] - u[t]))
        s = int(options[np.argmin(abs(u[options] - target))])
        result.append(dict(factor=factor, s=s, h=float(u[s]-u[t]), target_u=float(target)))
    if result[1]['s'] != nominal_s or len({r['s'] for r in result}) != 3:
        raise ValueError('Span candidates collide or original anchor differs; do not silently change design')
    return result


def scale_agreement(values):
    a = np.asarray(values, dtype=float)
    if len(a) != 2 or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError('Need two finite nonnegative responses')
    return float(abs(a[0]-a[1])/max(float(np.mean(a)), 1e-12))


def derive(point):
    """All labels are paired at the same endpoint noise, never across unequal s."""
    large = next(a for a in point['actions'] if a['factor'] == 1.5 and a['gamma'] == 1.)
    base = next(a for a in point['actions'] if a['factor'] == 1. and a['gamma'] == 1.)
    strong = next(a for a in point['actions'] if a['factor'] == 1. and a['gamma'] == 2.)
    refinement = next(r for r in point['refinements'] if r['factor'] == 1.5)
    probes = point['probes']
    feature = dict(log_R=math.log(max(point['R'], 1e-12)), log_q=math.log(max(point['q_rms'], 1e-12)),
                   clip=point['clip'], log_move=math.log(max(point['move_rms'], 1e-12)),
                   history_turn=point['history_turn'],
                   mean_error=math.log1p(refinement['mean_error']),
                   mean_guidance_gain=base['mean']['loss']-strong['mean']['loss'])
    for direction in ('q', 'travel'):
        p = probes[direction]
        for kind in ('dir', 'mag'):
            feature[direction+'_'+kind] = math.log1p(p['L_'+kind]*point['move_rms']) if p['usable'] else None
    labels = dict(large_step_error=math.log1p(refinement['stochastic_error']),
                  guidance_loss_gain=base['stochastic']['loss']-strong['stochastic']['loss'],
                  guidance_psnr_gain=strong['stochastic']['psnr']-base['stochastic']['psnr'],
                  large_refinement_psnr_gain=refinement['stochastic_fine']['psnr']-large['stochastic']['psnr'])
    return dict(image_id=point['image_id'], seed=point['seed'], step=point['step'], features=feature, labels=labels)


def cross_validate(rows, features, target):
    """Leave one entire image out; stage mean and every transform fit on training only.

    Regress residuals about training stage means. Stage-centering the features
    makes the incremental question explicitly within stage. No hyperparameter scan.
    """
    images = sorted({r['image_id'] for r in rows})
    predictions = []
    for image in images:
        train = [r for r in rows if r['image_id'] != image]
        test = [r for r in rows if r['image_id'] == image]
        stages = sorted({r['step'] for r in train})
        if len(images) < 3 or any(r['step'] not in stages for r in test):
            raise ValueError('Need at least three images and training data in every held-out stage')
        stage_y = {s: float(np.mean([r['labels'][target] for r in train if r['step'] == s])) for s in stages}
        def matrix(items):
            return np.asarray([[r['features'][f] for f in features] for r in items], dtype=float).reshape(len(items), len(features))
        x, xt = matrix(train), matrix(test)
        if not np.isfinite(x).all() or not np.isfinite(xt).all():
            raise ValueError('Unresolved feature; retain and classify before fitting')
        ys = np.asarray([r['labels'][target]-stage_y[r['step']] for r in train])
        residual = np.zeros(len(test))
        if features:
            stage_x = {s: x[[r['step'] == s for r in train]].mean(axis=0) for s in stages}
            x = x-np.asarray([stage_x[r['step']] for r in train])
            xt = xt-np.asarray([stage_x[r['step']] for r in test])
            sd = x.std(axis=0); sd[sd < 1e-12] = 1.
            x, xt = x/sd, xt/sd
            sy = max(float(ys.std()), 1e-12)
            beta = np.linalg.solve(x.T@x+np.eye(len(features)), x.T@(ys/sy))
            residual = xt@beta*sy
        for r, d in zip(test, residual):
            truth = r['labels'][target]; prediction = stage_y[r['step']]+float(d)
            predictions.append(dict(image_id=image, seed=r['seed'], step=r['step'], truth=truth,
                                    prediction=prediction, abs_error=abs(prediction-truth)))
    per_image = {i: float(np.mean([p['abs_error'] for p in predictions if p['image_id'] == i])) for i in images}
    return dict(mae=float(np.mean(list(per_image.values()))), per_image_mae=per_image, predictions=predictions)


def analyze(points):
    rows = [derive(p) for p in points]
    # All model comparisons use the SAME resolved states. Nothing is silently dropped.
    common = [r for r in rows if all(r['features'][f] is not None for fs in FEATURES.values() for f in fs)]
    result = dict(total_states=len(rows), resolved_states=len(common),
                  unresolved_states=[dict(image_id=r['image_id'], seed=r['seed'], step=r['step']) for r in rows if r not in common],
                  image_clusters=len({r['image_id'] for r in rows}), targets={})
    coverage = (len(rows) == 72 and len(common) >= 58 and len({r['image_id'] for r in common}) == 6
                and all(sum(r['step'] == s for r in common) >= 8 for s in STEPS))
    result['resolved_per_stage'] = {str(s): sum(r['step'] == s for r in common) for s in STEPS}
    # Even when a fixed-t probe is unresolved, the independently measured paid
    # mean-response readouts remain usable. Compare those on ALL states, with
    # their own matched stage/physical controls; do not hide high-noise failures.
    if len(rows) == 72 and len({r['image_id'] for r in rows}) == 6:
        paid = {}
        for target in TARGETS:
            models = {'STAGE': cross_validate(rows, (), target)}
            models.update({name: cross_validate(rows, FEATURES[name], target) for name in ('PHYSICAL', 'RESPONSE')})
            comparisons = {}
            for baseline in ('STAGE', 'PHYSICAL'):
                a, b = models['RESPONSE'], models[baseline]
                reduction = 1.-a['mae']/b['mae'] if b['mae'] > 1e-10 else None
                better = sum(a['per_image_mae'][i] < b['per_image_mae'][i]-1e-10 for i in a['per_image_mae'])
                comparisons[baseline] = dict(relative_mae_reduction=reduction, images_better=better,
                    threshold_passed=reduction is not None and reduction >= .10 and better >= 4)
            paid[target] = dict(models=models, comparisons=comparisons)
        result['paid_response_all_states'] = paid
        result['paid_response_supported'] = all(paid[t]['comparisons'][b]['threshold_passed']
             for t in ('large_step_error', 'guidance_loss_gain') for b in ('STAGE', 'PHYSICAL'))
    if not coverage:
        result['classification'] = ('PAID_RESPONSE_PROBE_CANDIDATE_ONLY' if result.get('paid_response_supported') else
                                    'INSUFFICIENT_RESOLVED_STATE_RESPONSE')
        return result
    for target in TARGETS:
        models = {'STAGE': cross_validate(common, (), target)}
        models.update({name: cross_validate(common, fs, target) for name, fs in FEATURES.items()})
        comparisons = {}
        for name in ('HISTORY', 'LOCAL', 'RESPONSE'):
            comparisons[name] = {}
            for baseline in ('STAGE', 'PHYSICAL'):
                a, b = models[name], models[baseline]
                reduction = 1.-a['mae']/b['mae'] if b['mae'] > 1e-10 else None
                better = sum(a['per_image_mae'][i] < b['per_image_mae'][i]-1e-10 for i in a['per_image_mae'])
                comparisons[name][baseline] = dict(relative_mae_reduction=reduction, images_better=better,
                    threshold_passed=reduction is not None and reduction >= .10 and better >= 4)
        result['targets'][target] = dict(models=models, comparisons=comparisons)
    def supported(name):
        return all(result['targets'][t]['comparisons'][name][b]['threshold_passed']
                   for t in ('large_step_error', 'guidance_loss_gain') for b in ('STAGE', 'PHYSICAL'))
    result.update(classification=('LOCAL_STATE_RESPONSE_CANDIDATE_ONLY' if supported('LOCAL') else
                   'PAID_RESPONSE_PROBE_CANDIDATE_ONLY' if result.get('paid_response_supported') else 'STATE_ACTION_INFORMATION_NOT_SUPPORTED'),
                  local_supported=supported('LOCAL'))
    return result
