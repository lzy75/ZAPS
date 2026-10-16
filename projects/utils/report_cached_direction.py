"""Read existing A0b run.json only: fair paired associations and component angles.

Zero torch/CUDA/model/tensor access. No candidate signal or control change.
New analyses are post-hoc diagnostics, not independent validation.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
from utils.diag_state_trajectory import digest, pearson, ranks


def valid(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def validate(record):
    if (record.get('experiment') != 'cached_conditional_mean_direction' or record.get('status') != 'complete' or
            record.get('method_nfe') != 0 or record.get('diagnostic_nfe') != 0 or record.get('model_loaded') is not False or
            not all(record.get('gates', {}).get(k, {}).get('passed') is True for k in
                    ('cached_steps', 'noise_draw_suffix', 'residual_and_clean_angles'))):
        raise ValueError('Need completed gated A0b cached-direction run')
    rows = record.get('rows', [])
    if (len(rows) != 30 or [r.get('step') for r in rows] != list(range(30)) or
            rows[0].get('t') != 999 or rows[-1].get('t') != 0 or
            any(a['t'] <= b['t'] or a['noise_u'] >= b['noise_u'] for a, b in zip(rows, rows[1:]))):
        raise ValueError('Need original complete descending time grid and increasing u')
    for row in rows:
        if not all(valid(row[k]) for k in ('noise_u', 'raw_x0_clip_fraction', 'psnr', 'lpips', 'mean_increment_rms')):
            raise ValueError('Non-finite readouts')
        for key in ('cosine_clean', 'cosine_latent', 'cosine_mean', 'prior_guidance_cosine'):
            if row[key] is not None and (not valid(row[key]) or not -1.000001 <= row[key] <= 1.000001):
                raise ValueError('Invalid direction cosine')


def select_run(explicit=None, root=PROJECTS):
    paths = [Path(explicit).resolve()/'run.json'] if explicit else sorted(
        Path(root).glob('results/diag_cached_direction/*/run.json'), reverse=True)
    errors = []
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding='utf-8')); validate(record)
            return path, record
        except (ValueError, KeyError, OSError) as error:
            errors.append(dict(path=str(path), error=str(error)))
    raise ValueError('No compatible completed report source: '+json.dumps(errors, ensure_ascii=False))


def stats(x, y, u):
    def detrend(values):
        mu, mv = statistics.mean(u), statistics.mean(values)
        den = sum((a-mu)**2 for a in u)
        b = sum((a-mu)*(v-mv) for a, v in zip(u, values))/den if den > 1e-15 else 0.
        return [v-mv-b*(a-mu) for a, v in zip(u, values)]
    if len(x) < 4:
        return dict(n=len(x), pearson=None, spearman=None, linear_u_partial=None)
    return dict(n=len(x), pearson=pearson(x, y), spearman=pearson(ranks(x), ranks(y)),
                linear_u_partial=pearson(detrend(x), detrend(y)))


def paired_report(rows):
    result = {}
    for scope in ('all_common', 'low_clip_common_exploratory'):
        indices = [k for k in range(len(rows)-1) if
                   all(valid(rows[k]['cosine_'+m]) for m in ('clean', 'latent', 'mean')) and
                   (scope == 'all_common' or rows[k]['raw_x0_clip_fraction'] <= .1)]
        associations = {}
        for mode in ('clean', 'mean'):
            for q in ('psnr', 'lpips'):
                for normalized in (False, True):
                    x = [rows[k]['cosine_'+mode] for k in indices]
                    y = [(rows[k+1][q]-rows[k][q]) /
                         (rows[k+1]['noise_u']-rows[k]['noise_u'] if normalized else 1.) for k in indices]
                    u = [rows[k]['noise_u'] for k in indices]
                    associations[mode+'_vs_next_'+q+('_per_u' if normalized else '_change')] = stats(x, y, u)
        result[scope] = dict(source_steps=indices, n=len(indices), associations=associations)
    return result


def components(rows):
    result = {}
    for name, group in (('high', [r for r in rows if r['t'] >= 667]),
                         ('middle', [r for r in rows if 334 <= r['t'] < 667]),
                         ('low', [r for r in rows if r['t'] < 334])):
        angles = [r['prior_guidance_cosine'] for r in group if valid(r['prior_guidance_cosine'])]
        result[name] = dict(points=len(group), valid_angles=len(angles),
                            prior_guidance_cosine_median=statistics.median(angles) if angles else None,
                            opposing_count=sum(c < 0 for c in angles), strongly_opposing_count=sum(c <= -.5 for c in angles))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', help='Completed A0b Records; omitted=latest completed compatible run')
    args = parser.parse_args(); path, record = select_run(args.run_dir)
    source = Path(record['source'])
    if digest(source) != record['source_sha256']:
        raise RuntimeError('Original passive-trace JSON changed')
    rows = record['rows']
    print('Report source:', path, '\nSHA256:', digest(path))
    print('Existing record read only; new method/diagnostic UNet NFE=0; no tensors/model loaded.')
    print(' k    t  clipFrac  priorGuideCos  meanStepRMS noise/mean angleDeg delta_u_back')
    def fmt(v): return '        NA' if v is None else f'{v:10.4f}'
    for k, r in enumerate(rows):
        c = r['cosine_mean']
        angle = math.degrees(math.acos(max(-1., min(1., c)))) if c is not None else None
        du = r['noise_u']-rows[k-1]['noise_u'] if k else None
        print(f"{k:2d} {r['t']:4d} {r['raw_x0_clip_fraction']:9.4f} {fmt(r['prior_guidance_cosine'])} "
              f"{r['mean_increment_rms']:12.5f} {fmt(r['noise_over_mean'])} {fmt(angle)} {fmt(du)}")
    print('Components:', json.dumps(components(rows), allow_nan=False))
    print('Common-pair associations:', json.dumps(paired_report(rows), allow_nan=False))
    print('Limits: post-hoc single-image analysis; shared pairs prevent missing-data mismatch. '
          'Per-u responses compare different finite intervals and remain grid-dependent. '
          'Opposing prior/guidance angles alone do not establish harmful overshoot or their norm ratio. '
          'No new controller, threshold tuning or reconstruction gain claim.')


if __name__ == '__main__': main()
