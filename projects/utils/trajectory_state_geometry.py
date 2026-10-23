"""A35 causal path geometry on the improved I_cont2 trajectory; no GT input."""
import math
import numpy as np

POINTS = ((8, 903), (19, 641), (24, 319), (27, 54), (28, 14))
FACTORS = (.5, 1.5)
CALLS = 30+len(FACTORS)*sum(29-k for k, _ in POINTS)
WINDOW_U = .15
PHYSICAL = ('log_R', 'R_slope', 'R_oscillation', 'log_q', 'clip', 'log_prior',
            'log_guide', 'response_cos', 'log_b', 'log_speed', 'log_high_speed')
HISTORY = ('q_cos', 'log_q_omega3')
CURVE = ('log_curvature', 'backtrack', 'log_high_curvature', 'high_backtrack', 'drift_turn')
COUPLING = ('prior_guide_cos', 'high_prior_guide_cos', 'log_high_dose', 'guide_high_fraction')
FEATURES = {'STAGE': (), 'PHYSICAL': PHYSICAL, 'HISTORY': PHYSICAL+HISTORY,
            'CURVE': PHYSICAL+CURVE, 'COUPLED': PHYSICAL+CURVE+COUPLING}


def rms(x):
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)))))


def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    aa, bb = float(a@a), float(b@b)
    if min(aa, bb) <= 1e-24:
        return None
    return max(-1., min(1., float(a@b)/math.sqrt(aa)/math.sqrt(bb)))


def high_part(x):
    """Orthogonal complement of 4x4 block averages, NOT the nullspace of bicubic H."""
    x = np.asarray(x)
    c, h, w = x.shape
    if h % 4 or w % 4:
        raise ValueError('Need C,H,W with dimensions divisible by four')
    low = x.reshape(c, h//4, 4, w//4, 4).mean(axis=(2, 4))
    return x-low.repeat(4, axis=1).repeat(4, axis=2)


def path_geometry(frames, duration):
    if len(frames) < 3 or duration <= 0:
        return dict(curvature=None, backtrack=None, speed=None, resolved=False)
    edges = [np.asarray(b, dtype=np.float64)-np.asarray(a, dtype=np.float64) for a, b in zip(frames, frames[1:])]
    lengths = [rms(e) for e in edges]
    floor = 32*np.finfo(np.float32).eps*max(1., max(rms(v) for v in frames))
    total = sum(lengths)
    if min(lengths) <= floor:
        return dict(curvature=None, backtrack=None, speed=total/duration, resolved=False)
    turning = sum(math.acos(cosine(a, b)) for a, b in zip(edges, edges[1:]))
    chord = rms(np.asarray(frames[-1], dtype=np.float64)-np.asarray(frames[0], dtype=np.float64))
    return dict(curvature=turning/total, backtrack=max(0., min(1., 1.-chord/total)),
                speed=total/duration, resolved=True)


class History:
    def __init__(self):
        self.frames = []

    def append(self, u, clean, q, prior, gamma, readout, clip):
        if self.frames and u <= self.frames[-1]['u']:
            raise ValueError('Noise progress must increase strictly')
        clean, q, prior = [np.asarray(v, dtype=np.float32) for v in (clean, q, prior)]
        guide = gamma*q
        frame = dict(u=float(u), clean=clean, high=high_part(clean), q=q, prior=prior,
                     guide=guide, drift=prior+guide, R=readout['R'])
        self.frames.append(frame)
        k, j = len(self.frames)-1, len(self.frames)-1
        while j > 0 and (k-j < 2 or u-self.frames[j]['u'] < WINDOW_U):
            j -= 1
        window = self.frames[j:k+1]
        duration = u-window[0]['u']
        geometry = path_geometry([v['clean'] for v in window], duration)
        high_geometry = path_geometry([v['high'] for v in window], duration)
        if len(window) >= 2:
            times = np.asarray([v['u'] for v in window], dtype=float)
            log_r = np.log(np.maximum([v['R'] for v in window], 1e-12))
            centered = times-times.mean()
            slope = float(centered@(log_r-log_r.mean())/(centered@centered))
            travel = float(np.abs(np.diff(log_r)).sum())
            oscillation = 1.-abs(float(log_r[-1]-log_r[0]))/travel if travel > 1e-12 else 0.
        else:
            slope = oscillation = None
        q_cos = cosine(q, self.frames[-2]['q']) if k else None
        omega = None
        if k >= 2:
            old_cos = cosine(self.frames[-2]['q'], self.frames[-3]['q'])
            if q_cos is not None and old_cos is not None:
                omega = (math.sqrt(max(0., 2.-2.*q_cos))+math.sqrt(max(0., 2.-2.*old_cos)))/(u-self.frames[-3]['u'])
        drift_turn = None
        if k:
            c = cosine(frame['drift'], self.frames[-2]['drift'])
            if c is not None:
                drift_turn = math.acos(c)/(u-self.frames[-2]['u'])
        hp, hg = high_part(prior), high_part(guide)
        def log(v):
            return math.log(max(v, 1e-12)) if v is not None else None
        def log1(v):
            return math.log1p(v) if v is not None else None
        features = dict(log_R=log(readout['R']), R_slope=slope, R_oscillation=oscillation,
                        log_q=log(rms(q)), clip=float(clip), log_prior=log(rms(prior)),
                        log_guide=log(rms(guide)), response_cos=readout['cosine'],
                        log_b=log(readout['relative_response']), log_speed=log(geometry['speed']),
                        log_high_speed=log(high_geometry['speed']), q_cos=q_cos, log_q_omega3=log1(omega),
                        log_curvature=log1(geometry['curvature']), backtrack=geometry['backtrack'],
                        log_high_curvature=log1(high_geometry['curvature']), high_backtrack=high_geometry['backtrack'],
                        drift_turn=log1(drift_turn), prior_guide_cos=cosine(prior, guide),
                        high_prior_guide_cos=cosine(hp, hg),
                        log_high_dose=math.log1p(rms(hg)/max(rms(hp), 1e-12)),
                        guide_high_fraction=rms(hg)/max(rms(guide), 1e-12))
        if any(v is not None and not math.isfinite(v) for v in features.values()):
            raise ValueError('Nonfinite geometry feature')
        return dict(features=features, window_start=j, window_points=len(window), window_u=duration,
                    nominal_window_covered=duration >= WINDOW_U,
                    clean_geometry=geometry, high_geometry=high_geometry,
                    physical=dict(R=readout['R'], log_R_slope=slope, residual_oscillation=oscillation),
                    coupling=dict(prior_guide_cos=features['prior_guide_cos'],
                                  high_prior_guide_cos=features['high_prior_guide_cos']))


def cross_validate(rows, features, target):
    """Whole-image LOIO; training-only imputation, stage centering, scaling, ridge=1."""
    images = sorted({r['image_id'] for r in rows})
    predictions = []
    if len(images) < 3:
        raise ValueError('Need at least three image clusters')
    for held in images:
        train = [r for r in rows if r['image_id'] != held]
        test = [r for r in rows if r['image_id'] == held]
        stages = sorted({r['step'] for r in train})
        y_mean = {s: float(np.mean([r['labels'][target] for r in train if r['step'] == s])) for s in stages}
        residual = np.zeros(len(test))
        if features:
            impute = {}
            for s in stages:
                for f in features:
                    valid = [r['features'][f] for r in train if r['step'] == s and r['features'][f] is not None]
                    impute[(s, f)] = float(np.median(valid)) if valid else 0.
            def matrix(items):
                values = [[r['features'][f] if r['features'][f] is not None else impute[(r['step'], f)]
                           for f in features] for r in items]
                masks = [[float(r['features'][f] is None) for f in features] for r in items]
                return np.concatenate((np.asarray(values), np.asarray(masks)), axis=1)
            x, xt = matrix(train), matrix(test)
            stage_x = {s: x[[r['step'] == s for r in train]].mean(axis=0) for s in stages}
            x -= np.asarray([stage_x[r['step']] for r in train])
            xt -= np.asarray([stage_x[r['step']] for r in test])
            sd = x.std(axis=0); sd[sd < 1e-12] = 1.
            x, xt = x/sd, xt/sd
            y = np.asarray([r['labels'][target]-y_mean[r['step']] for r in train])
            sy = max(float(y.std()), 1e-12)
            beta = np.linalg.solve(x.T@x+np.eye(x.shape[1]), x.T@(y/sy))
            residual = xt@beta*sy
        for r, extra in zip(test, residual):
            prediction = y_mean[r['step']]+float(extra)
            predictions.append(dict(image_id=held, seed=r['seed'], step=r['step'],
                                    prediction=prediction, truth=r['labels'][target],
                                    abs_error=abs(prediction-r['labels'][target])))
    per_image = {i: float(np.mean([r['abs_error'] for r in predictions if r['image_id'] == i])) for i in images}
    return dict(mae=float(np.mean(list(per_image.values()))), per_image_mae=per_image, predictions=predictions)


def action_audit(rows_by_factor, predictions):
    """LOIO PSNR choice at EACH fixed position; never combine pulses or select a position."""
    indexed = {f: {(r['image_id'], r['seed'], r['step']): r for r in rows}
               for f, rows in rows_by_factor.items()}
    keys = sorted(indexed[FACTORS[0]])
    if any(set(items) != set(keys) for items in indexed.values()):
        raise ValueError('Paired pulse states differ')
    models = {}
    for model in FEATURES:
        estimates = {f: {(r['image_id'], r['seed'], r['step']): r['prediction']
                         for r in predictions[f][model]} for f in FACTORS}
        selected = []
        for key in keys:
            # Deterministic tie order: keep, weaken, strengthen. No held GT enters this choice.
            factor = max((1.,)+FACTORS, key=lambda f: 0. if f == 1. else estimates[f][key])
            labels = {m: 0. for m in ('psnr', 'ssim', 'lpips', 'loss')} if factor == 1. else indexed[factor][key]['labels']
            selected.append(dict(image_id=key[0], seed=key[1], step=key[2], factor=factor, delta=labels))
        images = sorted({r['image_id'] for r in selected})
        per_image = {i: {m: float(np.mean([r['delta'][m] for r in selected if r['image_id'] == i]))
                         for m in ('psnr', 'ssim', 'lpips', 'loss')} for i in images}
        delta = {m: float(np.mean([v[m] for v in per_image.values()])) for m in ('psnr', 'ssim', 'lpips', 'loss')}
        stages = {str(k): dict(source_t=t,
                              mean_delta={m: float(np.mean([r['delta'][m] for r in selected if r['step'] == k]))
                                          for m in ('psnr', 'ssim', 'lpips', 'loss')},
                              counts={str(f): sum(r['factor'] == f and r['step'] == k for r in selected)
                                      for f in (1.,)+FACTORS}) for k, t in POINTS}
        models[model] = dict(mean_delta=delta, per_image=per_image,
                             quality_guard=delta['ssim'] >= -.002 and delta['lpips'] <= .005,
                             counts={str(f): sum(r['factor'] == f for r in selected) for f in (1.,)+FACTORS},
                             per_stage=stages, choices=selected)
    comparisons = {}
    for name in ('HISTORY', 'CURVE', 'COUPLED'):
        comparisons[name] = {}
        for baseline in ('STAGE', 'PHYSICAL'):
            a, b = models[name], models[baseline]
            delta = {m: a['mean_delta'][m]-b['mean_delta'][m] for m in ('psnr', 'ssim', 'lpips', 'loss')}
            comparisons[name][baseline] = dict(mean_delta=delta,
                images_psnr_better=sum(a['per_image'][i]['psnr'] > b['per_image'][i]['psnr']+1e-9 for i in images),
                quality_guard=delta['ssim'] >= -.002 and delta['lpips'] <= .005)
    return dict(models=models, comparisons=comparisons,
                limits='Offline held-image PSNR-predicted choice among keep/x0.5/x1.5 at each fixed position. '
                       'Reported means average five DIFFERENT single-pulse experiments, not five pulses in one run; '
                       'no position selection, GT-based held-image action choice, or deployed multi-step controller.')


def analyze(records):
    results, rows_by_factor, predictions = {}, {}, {}
    for factor in FACTORS:
        rows = []
        for record in records:
            for k, t in POINTS:
                arm = f'T{t}_M{factor:g}'
                rows.append(dict(image_id=record['case']['image_id'], seed=record['case']['sampling_seed'], step=k,
                                 features=record['rows'][k]['state']['features'],
                                 labels={m: record['metrics'][arm][m]-record['metrics']['BASE'][m]
                                         for m in ('psnr', 'ssim', 'lpips', 'loss')}))
        rows_by_factor[factor] = rows
        targets = {}
        for target in ('psnr', 'ssim', 'lpips', 'loss'):
            models = {name: cross_validate(rows, features, target) for name, features in FEATURES.items()}
            if target == 'psnr':
                predictions[factor] = {n: v['predictions'] for n, v in models.items()}
            comparisons = {}
            for name in ('HISTORY', 'CURVE', 'COUPLED'):
                comparisons[name] = {}
                for baseline in ('STAGE', 'PHYSICAL'):
                    a, b = models[name], models[baseline]
                    reduction = 1.-a['mae']/b['mae'] if b['mae'] > 1e-9 else None
                    better = sum(a['per_image_mae'][i] < b['per_image_mae'][i]-1e-9 for i in a['per_image_mae'])
                    comparisons[name][baseline] = dict(relative_mae_reduction=reduction, images_better=better,
                        threshold_passed=reduction is not None and reduction >= .10 and better >= 4)
            targets[target] = dict(mae={n: v['mae'] for n, v in models.items()}, comparisons=comparisons,
                                   per_image_mae={n: v['per_image_mae'] for n, v in models.items()})
        results[str(factor)] = dict(points=len(rows), image_clusters=len({r['image_id'] for r in rows}), targets=targets)
    supported = [name for name in ('CURVE', 'COUPLED') if all(
        results[str(f)]['targets'][m]['comparisons'][name][b]['threshold_passed']
        for f in FACTORS for m in ('psnr', 'lpips') for b in ('STAGE', 'PHYSICAL'))]
    return dict(classification='GEOMETRY_INFORMATION_CANDIDATE_ONLY' if supported else 'NO_CONFIRMED_GEOMETRY_INCREMENT',
                supported_families=supported, actions=results,
                action_audit=action_audit(rows_by_factor, predictions),
                limits='Six reused development images; whole-image leave-out, stage-centered, ridge1, no tuning. '
                       'Prediction diagnosis only; no deployed controller, final adaptive gain, new-grid benefit or curvature theorem.')
