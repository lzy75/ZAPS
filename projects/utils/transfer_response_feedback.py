"""A36 frozen-policy transfer definitions and image-cluster summaries; no GT decisions."""
import math
import statistics
import numpy as np
from utils import continuous_response_feedback as feedback

SIGMAS = (.025, .05, .10)
ARMS = ('C', 'TIME', 'STATE')
IMAGES = 6
SEED = 1000
METRICS = ('psnr', 'ssim', 'lpips', 'loss')
METHOD_PER_CASE, DIAGNOSTIC_PER_CASE = 300, 90


def frozen_profile(records, grid):
    """One median table from the twelve OLD trajectories; targets never enter it."""
    expected = {(f'{i:05d}', s) for i in range(20, 26) for s in (1000, 2000)}
    if len(records) != 12 or {(r['case']['image_id'], r['case']['sampling_seed']) for r in records} != expected:
        raise ValueError('Need all twelve old calibration trajectories')
    if any([v['t'] for v in r['rows']] != list(reversed(grid)) for r in records):
        raise ValueError('Calibration grid differs')
    values = [statistics.median(r['rows'][k]['gamma'] for r in records) for k in range(30)]
    if any(not math.isfinite(v) or not 0 <= v <= 2 for v in values):
        raise ValueError('Invalid frozen multiplier')
    return values


def sensor(z, y, clean, q, t, sigma):
    if sigma not in SIGMAS:
        raise ValueError('Unregistered measurement noise')
    value = feedback.sensor(z, y, clean, q, t, 2.)
    # A33 hard-codes .05 only in its R READOUT, not in c/b or the decision.
    # Correct units for the new observation; do not add a noise-dependent weight rule.
    value['R'] *= .05/sigma
    return value


def contrast(records, arm, reference):
    images = sorted({r['case']['image_id'] for r in records})
    per_image = {i: {m: statistics.mean(r['metrics'][arm][m]-r['metrics'][reference][m]
                                      for r in records if r['case']['image_id'] == i) for m in METRICS}
                 for i in images}
    mean = {m: statistics.mean(v[m] for v in per_image.values()) for m in METRICS}
    values = np.asarray([[v[m] for m in METRICS] for v in per_image.values()], dtype=float)
    draws = np.random.default_rng(20261023).integers(0, len(images), size=(10000, len(images)))
    intervals = np.quantile(values[draws].mean(axis=1), [.025, .975], axis=0)
    return dict(mean_delta=mean, image_clusters=len(images), per_image=per_image,
                ci95_descriptive={m: intervals[:, j].tolist() for j, m in enumerate(METRICS)},
                images_psnr_positive=sum(v['psnr'] > 1e-9 for v in per_image.values()),
                min_image_psnr_delta=min(v['psnr'] for v in per_image.values()),
                max_image_lpips_delta=max(v['lpips'] for v in per_image.values()),
                quality_guard=mean['ssim'] >= -.002 and mean['lpips'] <= .005,
                image_lpips_risk_count=sum(v['lpips'] > .005 for v in per_image.values()))


def summarize(records):
    keys = [(r['case']['image_id'], r['case']['sigma'], r['case']['sampling_seed']) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate transfer case')
    groups = []
    for sigma in SIGMAS:
        current = [r for r in records if r['case']['sigma'] == sigma]
        group = dict(sigma=sigma, completed_cases=len(current), planned_cases=IMAGES)
        if current:
            results = {f'{a}_{b}': contrast(current, a, b) for a, b in
                       (('STATE', 'C'), ('TIME', 'C'), ('STATE', 'TIME'))}
            group['comparisons'] = results
            complete = len(current) == IMAGES and all(r['case']['sampling_seed'] == SEED for r in current)
            def passed(key, threshold):
                r = results[key]
                return complete and r['mean_delta']['psnr'] >= threshold and r['images_psnr_positive'] >= 4 and r['quality_guard']
            group.update(state_vs_C_candidate=passed('STATE_C', .05), time_vs_C_candidate=passed('TIME_C', .05),
                         state_vs_TIME_candidate=passed('STATE_TIME', .02))
            pairs = [(s, t) for r in current for s, t in zip(r['traces']['STATE'], r['traces']['TIME'])]
            group['weight_response'] = dict(
                mean_abs_gamma_vs_TIME=statistics.mean(abs(s['gamma']-t['gamma']) for s, t in pairs),
                state_cap_fraction=statistics.mean(float(s['cap_hit']) for s, _ in pairs),
                state_fallback_fraction=statistics.mean(float(s['fallback']) for s, _ in pairs),
                state_final_R_mean=statistics.mean(r['traces']['STATE'][-1]['R'] for r in current))
        groups.append(group)
    complete = all(g['completed_cases'] == IMAGES for g in groups)
    supported = complete and all(g['state_vs_C_candidate'] for g in groups)
    return dict(classification=('FROZEN_STATE_TRANSFER_CANDIDATE_ONLY' if supported else
                                'MIXED_OR_NO_FROZEN_STATE_TRANSFER_SUPPORT') if complete else 'INCOMPLETE_PILOT',
                completed_cases=len(records), planned_cases=IMAGES*len(SIGMAS), groups=groups,
                state_transfer_candidate=supported,
                time_transfer_candidate=complete and all(g['time_vs_C_candidate'] for g in groups),
                state_increment_candidate_sigmas=[g['sigma'] for g in groups if g.get('state_vs_TIME_candidate')],
                criteria='Engineering pilot: each condition STATE-C PSNR>=.05 dB, >=4/6 positive images, '
                         'mean SSIM>=-.002 and LPIPS<=+.005. Separate STATE-TIME candidate threshold .02 dB '
                         'with same image/quality requirements. No automatic scientific confirmation.',
                limits='Six images unused in audited accessible server records; one seed/image/condition. '
                       'Same images and standard noise across three sigmas: 18 cases are only six image clusters. '
                       '10,000 paired image-bootstrap percentile intervals are descriptive, unadjusted, approximate. '
                       'No per-image retuning or GT output selection. Last-unroll feedback only, native DDPM and '
                       'legacy transpose local reference. Positive C gain alone is not evidence of state-specific value.')
