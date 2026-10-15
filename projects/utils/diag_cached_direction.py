"""Audit conditional-mean directions on cached FFHQ inputs, ZERO UNet NFE.

No model/checkpoint load, training, sampling-grid changes or new trajectory.
The actual posterior helper reconstructs each stored step using archived RNG.
The same helper with eta=0 only evaluates its mean at the STORED input.
"""
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
from utils.diag_state_trajectory import digest, describe_parent, cosine, association


def validate_trace(record):
    c = record.get('resolved_config', {})
    if (record.get('experiment') != 'passive_fixed_parent_state_trajectory' or
            record.get('status') != 'complete' or record.get('method_nfe') != 0 or
            record.get('diagnostic_nfe') != 90 or c.get('kind') != 'historical_development' or
            c.get('seed') != 1000 or Path(c.get('image', '')).stem != '00000' or
            c.get('transpose') != 'legacy_bicubic'):
        raise ValueError('Need the completed original FFHQ 00000 unit-null passive trace')
    if not all(record.get('gates', {}).get(k, {}).get('passed') is True for k in ('observer', 'archive')):
        raise ValueError('Parent replay gates missing/failed')
    rows, grid = record.get('rows', []), c.get('grid', [])
    if (len(rows) != 30 or len(grid) != 30 or grid != sorted(set(grid)) or grid[0] != 0 or grid[-1] != 999 or
            [r.get('step') for r in rows] != list(range(30)) or
            [r.get('t') for r in rows] != grid[::-1] or
            [r.get('parameter_index') for r in rows] != list(range(29, -1, -1)) or
            [r.get('t_prev') for r in rows] != grid[-2::-1] + [-1]):
        raise ValueError('Incomplete/changed time or parameter mapping')
    if any(not math.isfinite(r[k]) for r in rows for k in
           ('noise_u', 'residual_norm', 'raw_x0_clip_fraction', 'psnr', 'ssim', 'lpips')):
        raise ValueError('Non-finite archived readouts')
    if any(b['noise_u'] <= a['noise_u'] for a, b in zip(rows, rows[1:])):
        raise ValueError('Noise progress must increase strictly')


def select_trace(explicit=None, root=PROJECTS):
    paths = [Path(explicit).resolve()/'run.json'] if explicit else sorted(
        Path(root).glob('results/diag_state_trajectory/*/run.json'), reverse=True)
    rejected = []
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding='utf-8')); validate_trace(record)
            if not (path.parent/'trajectory.pt').is_file():
                raise ValueError('Raw cached trajectory missing')
            return path, record, rejected
        except (ValueError, KeyError, OSError) as error:
            rejected.append(dict(path=str(path), reason=str(error)))
    raise ValueError('No completed compatible cached trajectory: '+json.dumps(rejected, ensure_ascii=False))


def angular_readouts(c, du):
    """Unit-tangent chord and its change per u, NOT geometric curvature."""
    if c is None:
        return dict(tangent_turn=None, turn_per_u=None)
    if not math.isfinite(c) or c < -1.-1e-6 or c > 1.+1e-6 or not math.isfinite(du) or du <= 0:
        raise ValueError('Need a valid cosine and positive evaluation-point separation')
    turn = math.sqrt(max(0., 2.*(1.-max(-1., min(1., c)))))
    return dict(tangent_turn=turn, turn_per_u=turn/du)


def vector_turn(a, b, du):
    """Direct unit-tangent subtraction avoids cancellation in 1-cos near 1."""
    c = cosine(a, b)
    if c is None:
        return c, dict(tangent_turn=None, turn_per_u=None)
    if not math.isfinite(du) or du <= 0:
        raise ValueError('Need positive evaluation-point separation')
    a, b = a.detach().double().flatten(), b.detach().double().flatten()
    turn = float((a/a.norm()-b/b.norm()).norm())
    return c, dict(tangent_turn=turn, turn_per_u=turn/du)


def direction_summary(rows):
    scopes = {'all': rows, 'low_clip_exploratory': [r for r in rows if r['raw_x0_clip_fraction'] <= .1]}
    result = {}
    for name, subset in scopes.items():
        # Keep ORIGINAL adjacency when selecting valid source points. Never bridge gaps.
        selected = {r['step'] for r in subset}
        links = {}
        for direction in ('clean', 'latent', 'mean'):
            for metric in ('psnr', 'lpips', 'residual_noise_ratio'):
                masked = [dict(r, **{'cosine_'+direction: r['cosine_'+direction] if r['step'] in selected else None})
                          for r in rows]
                links[direction+'_vs_next_change_'+metric] = association(masked, 'cosine_'+direction, metric, True)
        valid = [r['cosine_mean'] for r in subset if r['cosine_mean'] is not None]
        result[name] = dict(n=len(subset), mean_cosine_valid=len(valid),
                            mean_cosine_median=statistics.median(valid) if valid else None,
                            mean_cosine_span=max(valid)-min(valid) if valid else None, associations=links)
    return dict(scopes=result, controller_suitability='NOT_ESTABLISHED',
                limits='Same explored single image/seed; associations are descriptive and autocorrelated. '
                       'low_clip<=.1 was motivated by the prior plot; exploratory, not independent validation. '
                       'Mean direction still depends on noisy input, clipping, learned guidance and finite grid intervals. '
                       'turn_per_u is angular change rate, not arc-length curvature or truncation error.')


def cached_step(x, clean, y, operator, dwt, zeta, diagonal, t, s, alpha):
    """Use original algebra/order and actual core helper; NEVER call the score."""
    from modules.zaps_algorithm import ddpm_posterior_step
    mean = ddpm_posterior_step(x, clean, t, s, alpha, eta=0., mode='ddpm')
    # This consumes exactly the one original draw, or none at terminal s=-1.
    sampled = ddpm_posterior_step(x, clean, t, s, alpha, eta=1., mode='ddpm')
    residual = y-operator.H(clean)
    vector = operator.transpose(residual)
    hv = dwt.synthesis(diagonal*dwt.analysis(vector))
    guided = (vector+(1.-alpha[t])*hv)/alpha[t].sqrt().clamp(min=1e-8)
    correction = zeta*guided
    return dict(next=sampled+correction, mean_increment=mean+correction-x,
                noise_increment=sampled-mean, prior_increment=mean-x,
                correction=correction, residual=residual)


def plot(directory, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    u = [r['noise_u'] for r in rows]
    def values(key): return [float('nan') if r[key] is None else r[key] for r in rows]
    for mode in ('clean', 'latent', 'mean'):
        axes[0, 0].plot(u, values('cosine_'+mode), '.-', label=mode)
        axes[0, 1].plot(u, values(mode+'_turn_per_u'), '.-', label=mode)
    axes[0, 0].set_ylim(-1.05, 1.05); axes[0, 0].set_ylabel('Adjacent direction cosine')
    axes[0, 1].set_ylabel('Unit-tangent change / delta u'); axes[0, 1].set_yscale('symlog', linthresh=.1)
    axes[1, 0].plot(u, values('noise_over_mean'), '.-'); axes[1, 0].set_ylabel('Injected noise norm / mean-step norm')
    axes[1, 1].plot(u, values('raw_x0_clip_fraction'), '.-', label='raw clean clipping')
    axes[1, 1].plot(u, values('prior_guidance_cosine'), '.-', label='prior/guidance cosine')
    for ax in axes.flat:
        ax.set_xlabel('Denoising progress u'); ax.grid(alpha=.25)
    axes[0, 0].legend(); axes[0, 1].legend(); axes[1, 1].legend()
    fig.suptitle('Cached fixed trajectory: conditional-mean direction diagnostic, zero UNet NFE')
    fig.tight_layout(); fig.savefig(directory/'directions.png', dpi=160)
    fig.savefig(directory/'directions.svg'); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-dir', help='Previous passive trace Records; auto-detect if omitted')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--inspect', action='store_true', help='Resolve cached source with zero tensor work')
    args = parser.parse_args()
    source, previous, rejected = select_trace(args.trace_dir)
    print('Cached trace:', source, '\nRejected:', rejected, flush=True)
    if args.inspect: return
    if not args.device.startswith('cuda'):
        parser.error('Production reconstruction of archived noise needs CUDA; CPU unit tests are separate')
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    directory = PROJECTS/'results/diag_cached_direction'/stamp; directory.mkdir(parents=True, exist_ok=False)
    card = (PROJECTS/'experiment_cards/A_cached_direction.md').read_text(encoding='utf-8')
    (directory/'experiment_card.md').write_text(card, encoding='utf-8')
    record = dict(experiment='cached_conditional_mean_direction', status='running', method_nfe=0, diagnostic_nfe=0,
                  optimizer_steps=0, model_loaded=False, arguments=vars(args), source=str(source),
                  historical_parent_method_nfe=300, historical_trace_diagnostic_nfe=90,
                  source_sha256=digest(source), trajectory_sha256=digest(source.parent/'trajectory.pt'),
                  git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECTS, text=True).strip(),
                  experiment_card=card, gates={}, rows=[], source_selection='latest completed compatible trace, no quality ranking')
    started = time.perf_counter()
    def save():
        record['seconds'] = time.perf_counter()-started
        (directory/'run.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    save(); print('Records:', directory, flush=True)
    try:
        import torch
        from modules.diffusion_model import _precompute_diffusion_params
        from modules.degradations import get_operator
        from modules.wavelet import OrthogonalDWT2D
        from utils.diag_ffhq_regression import TransposeMode
        from utils.diag_imagenet_gaussian_unit_domain import unit_operator
        from utils.diag_zaps_guidance_clipping import saved_cuda_selection, restore_sampling_rng
        from utils.diag_zaps_sampler_parity import rng_state, same_rng
        from utils.diag_zaps_trace_audit import relative_error
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        parent_path = Path(previous['source'])
        if digest(parent_path) != previous['source_sha256']:
            raise RuntimeError('Original parent run changed')
        parent = json.loads(parent_path.read_text(encoding='utf-8'))
        desc = describe_parent(parent, parent_path.parent)
        resolved = {k: str(v) if isinstance(v, Path) else v for k, v in desc.items() if k != 'sources'}
        if resolved != previous['resolved_config']:
            raise RuntimeError('Parent protocol/paths changed')
        if digest(PROJECTS/'utils/diag_state_trajectory.py') != previous['observer_sha256']:
            raise RuntimeError('Original trace implementation changed')
        for name, sha in desc['sources'].items():
            if digest(PROJECTS/name) != sha: raise RuntimeError('Parent source changed: '+name)
        for name in ('measurement', 'state', 'output'):
            if digest(desc[name]) != desc[name+'_sha']: raise RuntimeError('Parent tensor changed: '+name)
        state = torch.load(desc['state'], map_location='cpu', weights_only=True)
        cache = torch.load(source.parent/'trajectory.pt', map_location='cpu', weights_only=True)
        archived = torch.load(desc['output'], map_location='cpu', weights_only=True)
        if (cache['clean'].shape != (30, 3, 256, 256) or cache['latent_inputs'].shape != (30, 3, 256, 256) or
                cache['output'].shape != (1, 3, 256, 256) or
                not torch.equal(cache['latent_inputs'][:1], state['init_noise']) or
                relative_error(archived, cache['output']) > 1e-7 or state['tau'].tolist() != desc['grid']):
            raise RuntimeError('Cache shapes/initial state/output/parameter grid mismatch')
        if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in cache.values()):
            raise RuntimeError('Non-finite or non-fp32 cache')
        alpha = _precompute_diffusion_params(1000)['alphas_cumprod'].to(args.device)
        logs = .5*(torch.log1p(-alpha.double())-alpha.double().log())
        u = ((logs[-1]-logs)/(logs[-1]-logs[0])).cpu().tolist()
        if any(abs(row['noise_u']-u[row['t']]) > 1e-10 for row in previous['rows']):
            raise RuntimeError('Exact original alpha/noise coordinates differ')
        record['alpha_source'] = 'original source-hash-verified _precompute_diffusion_params(1000), float64 linear beta then float32 cumprod'
        record['runtime'] = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                                 gpu=torch.cuda.get_device_name(torch.device(args.device)),
                                 cudnn_benchmark=torch.backends.cudnn.benchmark,
                                 cudnn_deterministic=torch.backends.cudnn.deterministic,
                                 cudnn_tf32=torch.backends.cudnn.allow_tf32,
                                 matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
        base = get_operator('super_resolution', device=args.device, scale_factor=4, noise_sigma=0.)
        operator = unit_operator(TransposeMode(base, 'legacy_bicubic').to(args.device)).to(args.device)
        dwt = OrthogonalDWT2D(desc['config']['wave'], desc['config']['level']).to(args.device)
        y = torch.load(desc['measurement'], map_location=args.device, weights_only=True)
        zeta, diagonal = state['zeta'].to(args.device), state['D'].to(args.device)
        mapping = saved_cuda_selection(state, parent['arguments']['device']); record['rng_mapping'] = mapping
        device_index = torch.device(args.device).index
        device_index = torch.cuda.current_device() if device_index is None else device_index
        rows, errors, prev_mean = [], [], None
        def rms(v): return float(v.detach().double().square().mean().sqrt())
        with torch.random.fork_rng(devices=[device_index]), torch.no_grad():
            restore_sampling_rng(state, args.device, mapping['saved_sampling_index'])
            for k, old in enumerate(previous['rows']):
                x = cache['latent_inputs'][k:k+1].to(args.device); clean = cache['clean'][k:k+1].to(args.device)
                i, t, s = old['parameter_index'], old['t'], old['t_prev']
                step = cached_step(x, clean, y, operator, dwt, zeta[i], diagonal[i], t, s, alpha)
                expected = cache['latent_inputs'][k+1:k+2] if k < 29 else cache['output']
                error = relative_error(expected, step['next'].cpu()); errors.append(error)
                if not math.isfinite(error) or error > 1e-6:
                    raise RuntimeError(f'Cached step mismatch at k={k}: {error}')
                norm = float(step['residual'].double().norm())
                if abs(norm-old['residual_norm']) > 1e-6+2e-5*norm:
                    raise RuntimeError('Residual/source scalar mismatch')
                clean_delta = cache['clean'][k:k+1]-cache['clean'][k-1:k] if k else None
                old_delta = cache['clean'][k-1:k]-cache['clean'][k-2:k-1] if k >= 2 else None
                clean_cos = cosine(clean_delta, old_delta)
                if ((clean_cos is None) != (old['cosine_clean'] is None) or
                        clean_cos is not None and abs(clean_cos-old['cosine_clean']) > 2e-4):
                    raise RuntimeError('Cached clean cosine/source mismatch')
                eval_du = old['noise_u']-rows[-1]['noise_u'] if k else 1.
                mean_cos, mean_turn = vector_turn(step['mean_increment'], prev_mean, eval_du)
                row = dict(old, cosine_mean=mean_cos, mean_increment_rms=rms(step['mean_increment']),
                           noise_increment_rms=rms(step['noise_increment']),
                           prior_guidance_cosine=cosine(step['prior_increment'], step['correction']))
                row['noise_over_mean'] = row['noise_increment_rms']/row['mean_increment_rms'] if row['mean_increment_rms'] > 1e-8 else None
                for mode in ('clean', 'latent', 'mean'):
                    readout = mean_turn if mode == 'mean' else angular_readouts(row['cosine_'+mode], eval_du)
                    row.update({mode+'_'+key: value for key, value in readout.items()})
                prev_mean = step['mean_increment'].clone(); rows.append(row)
            actual_rng = rng_state(args.device)
            restore_sampling_rng(state, args.device, mapping['saved_sampling_index'])
            initial = state['init_noise'].to(args.device)
            for _ in range(29): torch.randn_like(initial)
            rng_equal = same_rng(actual_rng, rng_state(args.device))
        if not rng_equal: raise RuntimeError('Original 29-draw RNG suffix differs')
        record['gates'] = dict(cached_steps=dict(passed=True, max_relative_error=max(errors), hard_ceiling=1e-6),
                               noise_draw_suffix=dict(passed=rng_equal, expected_draws=29),
                               residual_and_clean_angles=dict(passed=True))
        record['rows'], record['summary'] = rows, direction_summary(rows)
        record['final_metrics_unchanged'] = previous['final_metrics']
        record['identity_sources'] = desc['sources']
        with (directory/'directions.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        record['status'] = 'complete'; save()
        try: plot(directory, rows); record['plot_status'] = 'complete'
        except ImportError: record['plot_status'] = 'matplotlib unavailable; CSV/JSON complete'
        save()
    except Exception as error:
        record['status'], record['error'] = 'failed', repr(error); save(); raise
    print('\n=== Cached conditional-mean direction / no new trajectory ===', flush=True)
    print(' k    t   cleanCos  latentCos   meanCos  meanTurn/u noise/mean', flush=True)
    def fmt(v): return '       NA' if v is None else f'{v:9.3f}'
    for r in record['rows']:
        print(f"{r['step']:2d} {r['t']:4d} {fmt(r['cosine_clean'])} {fmt(r['cosine_latent'])} "
              f"{fmt(r['cosine_mean'])} {fmt(r['mean_turn_per_u'])} {fmt(r['noise_over_mean'])}", flush=True)
    print('Gates:', json.dumps(record['gates']), flush=True)
    print('Summary:', json.dumps(record['summary']), flush=True)
    print('Method NFE=0; diagnostic UNet NFE=0; no checkpoint loaded; original output unchanged.', flush=True)
    print('Records:', directory, flush=True)


if __name__ == '__main__': main()
