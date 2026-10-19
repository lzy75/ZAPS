"""A1a: actual guidance and clean-space measurement descent on a fixed cache.

ZERO UNet calls, no checkpoint/image/GT load, no optimizer or new trajectory.
Reconstruct all original transitions and RNG before interpreting any readout.
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
from utils.diag_cached_direction import select_trace, cached_step
from utils.diag_state_trajectory import digest, describe_parent, cosine, association

NORM_FLOOR = 1e-8
CHORD_FLOOR = 1e-12


def trend_geometry(current, previous, endpoint, span):
    """Three unit directions: path length, net displacement, and backtracking."""
    values = (current, previous, endpoint, span)
    if not all(math.isfinite(v) for v in values) or min(values[:3]) < 0 or span <= 0:
        raise ValueError('Need finite nonnegative chords and positive u span')
    total = current + previous
    if endpoint > total + 1e-10:
        raise ValueError('Unit-direction chords violate triangle inequality')
    return dict(variation3_per_u=total/span,
                backtrack3=max(0., min(1., 1.-endpoint/total)) if total > CHORD_FLOOR else None,
                no_resolved_turn3=total <= CHORD_FLOOR)


def rms(vector):
    return float(vector.detach().double().square().mean().sqrt())


def unit_direction(vector):
    v = vector.detach().double().flatten()
    if not v.isfinite().all():
        raise ValueError('Non-finite direction tensor')
    norm = float(v.norm())
    return v/norm if norm > NORM_FLOOR else None


def vector_readouts(current, previous, older, u, u_previous=None, u_older=None):
    """Original neighbors only; no fabricated direction at a zero vector."""
    result = dict(valid=unit_direction(current) is not None, rms=rms(current), cosine=None,
                  turn=None, turn_per_u=None, log_rms_change_per_u=None,
                  variation3_per_u=None, backtrack3=None, no_resolved_turn3=None)
    n = unit_direction(current)
    p = unit_direction(previous) if previous is not None else None
    o = unit_direction(older) if older is not None else None
    if previous is not None and (u_previous is None or u <= u_previous):
        raise ValueError('Evaluation points must increase strictly')
    if n is not None and p is not None:
        span = u-u_previous
        result.update(cosine=cosine(current, previous), turn=float((n-p).norm()),
                      log_rms_change_per_u=abs(math.log(result['rms']/rms(previous)))/span)
        result['turn_per_u'] = result['turn']/span
        if o is not None:
            if u_older is None or u_previous <= u_older:
                raise ValueError('Need three strictly ordered original points')
            result.update(trend_geometry(result['turn'], float((p-o).norm()),
                                         float((n-o).norm()), u-u_older))
    return result


def physical_descent(operator, clean, y, sigma=.05):
    """Exact -grad_z ||A(z)-y||²/(2*m*sigma²), actual forward H only.

    This is NOT a latent-space gradient: no denoiser Jacobian is evaluated.
    The historical empirical transpose is intentionally never called here.
    """
    import torch
    with torch.enable_grad():
        z = clean.detach().clone().requires_grad_(True)
        loss = (operator.H(z)-y.detach()).double().square().mean()/(2*sigma*sigma)
        descent = -torch.autograd.grad(loss, z)[0]
    return descent.detach(), float(loss.detach())


def resolve_source(source, trace):
    """Read-only hash/protocol checks; also usable by --inspect without Torch."""
    parent_path = Path(trace['source'])
    if digest(parent_path) != trace['source_sha256']:
        raise RuntimeError('Original parent run changed')
    parent = json.loads(parent_path.read_text(encoding='utf-8'))
    desc = describe_parent(parent, parent_path.parent)
    resolved = {k: str(v) if isinstance(v, Path) else v for k, v in desc.items() if k != 'sources'}
    if resolved != trace['resolved_config']:
        raise RuntimeError('Original protocol or paths changed')
    if digest(PROJECTS/'utils/diag_state_trajectory.py') != trace['observer_sha256']:
        raise RuntimeError('Original observer source changed')
    files = {str(source): digest(source), str(source.parent/'trajectory.pt'): digest(source.parent/'trajectory.pt'),
             str(parent_path): trace['source_sha256']}
    for name, sha in desc['sources'].items():
        if digest(PROJECTS/name) != sha:
            raise RuntimeError('Historical source changed: '+name)
        files[str(PROJECTS/name)] = sha
    for name in ('measurement', 'state', 'output'):
        if digest(desc[name]) != desc[name+'_sha']:
            raise RuntimeError('Historical tensor changed: '+name)
        files[str(desc[name])] = desc[name+'_sha']
    return parent, desc, files


def verify_cache(cache, state, archived, desc):
    import torch
    from utils.diag_zaps_trace_audit import relative_error
    shapes = dict(clean=(30, 3, 256, 256), latent_inputs=(30, 3, 256, 256), output=(1, 3, 256, 256))
    if any(k not in cache or tuple(cache[k].shape) != shape for k, shape in shapes.items()):
        raise RuntimeError('Original cache tensor shapes changed')
    if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in cache.values()):
        raise RuntimeError('Cache must contain finite fp32 tensors')
    if (not torch.equal(cache['latent_inputs'][:1], state['init_noise']) or
            relative_error(archived, cache['output']) > 1e-7 or state['tau'].tolist() != desc['grid']):
        raise RuntimeError('Initial state/output/grid differs')
    if (tuple(state['zeta'].shape) != (30,) or tuple(state['D'].shape) != (30, 3, 256, 256) or
            not torch.isfinite(state['zeta']).all() or not torch.isfinite(state['D']).all() or
            not (state['zeta'] > 0).all()):
        raise RuntimeError('Invalid archived PRE parameters')


def summarize(rows, repeat_chord_floor):
    if len(rows) != 30 or [r['step'] for r in rows] != list(range(30)):
        raise ValueError('Need all 30 original points')
    for a, b in zip(rows, rows[1:]):
        if b['noise_u'] <= a['noise_u']:
            raise ValueError('Non-increasing noise coordinate')
    keys = ('cosine', 'turn_per_u', 'variation3_per_u', 'backtrack3', 'rms')
    phases = {}
    for name, pred in [('high', lambda t: t >= 667), ('middle', lambda t: 334 <= t < 667),
                       ('low', lambda t: t < 334)]:
        part = [r for r in rows if pred(r['t'])]
        phases[name] = dict(n=len(part))
        for mode in ('guidance', 'physical'):
            for key in keys:
                values = [r[mode+'_'+key] for r in part if r[mode+'_'+key] is not None]
                phases[name][mode+'_'+key] = dict(n=len(values), median=statistics.median(values) if values else None)
    # Keep original k->k+1 adjacency, and report the unequal response interval.
    paired = []
    for k, row in enumerate(rows):
        item = dict(row)
        for target in ('psnr', 'lpips', 'residual_noise_ratio'):
            item['next_'+target+'_rate'] = ((rows[k+1][target]-row[target]) /
                                            (rows[k+1]['noise_u']-row['noise_u'])) if k < 29 else None
        paired.append(item)
    links = {}
    for mode in ('guidance', 'physical'):
        for key in ('cosine', 'variation3_per_u', 'backtrack3'):
            for target in ('psnr', 'lpips', 'residual_noise_ratio'):
                links[mode+'_'+key+'_vs_next_'+target+'_per_u'] = association(
                    paired, mode+'_'+key, 'next_'+target+'_rate')
    valid = sum(r['guidance_valid'] for r in rows)
    threshold = 10*max(CHORD_FLOOR, repeat_chord_floor)
    resolved = sum(r['guidance_turn'] is not None and r['guidance_turn'] > threshold for r in rows)
    return dict(guidance_valid=valid, resolved_adjacent_turns=resolved,
                repeat_unit_chord_floor=repeat_chord_floor, resolved_turn_threshold=threshold,
                classification=('READABLE_GUIDANCE_DIRECTION_CHANGE' if valid >= 24 and resolved else
                                'CONSTANT_OR_INSUFFICIENT_RESOLVED_DIRECTIONS'),
                phases=phases, associations=links, controller_suitability='NOT_ESTABLISHED',
                local_input_sensitivity_measured=False,
                limits='One explored image/seed; original-neighbor three-direction window fixed in advance. '
                       'Phase bins describe training t, not matched u reference distributions. '
                       'q varies with x,t,zeta,D and historical randomness. Physical means clean-space '
                       'actual-H gradient, not latent gradient. No p-values, GT selection, causal or step-benefit claim.')


def plot(directory, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    u = [r['noise_u'] for r in rows]
    def values(key): return [r[key] if r[key] is not None else float('nan') for r in rows]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    for mode in ('guidance', 'physical'):
        axes[0, 0].plot(u, values(mode+'_cosine'), '.-', label=mode)
        axes[0, 1].plot(u, values(mode+'_variation3_per_u'), '.-', label=mode)
        axes[1, 0].plot(u, values(mode+'_backtrack3'), '.-', label=mode)
    axes[0, 0].plot(u, values('cosine_clean'), '--', label='clean increments')
    axes[0, 0].set_ylabel('Adjacent direction cosine'); axes[0, 0].set_ylim(-1.05, 1.05)
    axes[0, 1].set_ylabel('Three-direction variation / u')
    axes[1, 0].set_ylabel('Three-direction backtracking'); axes[1, 0].set_ylim(-.05, 1.05)
    for key, label in [('guidance_rms', 'actual q'), ('base_guidance_rms', 'q / zeta')]:
        axes[1, 1].plot(u, values(key), '.-', label=label)
    axes[1, 1].set_yscale('symlog', linthresh=1e-6); axes[1, 1].set_ylabel('Correction RMS (model units)')
    axes[2, 0].plot(u, values('residual_noise_ratio'), '.-', label='residual / sigma_y')
    ax_clip = axes[2, 0].twinx()
    ax_clip.plot(u, values('raw_x0_clip_fraction'), '--', color='tab:orange', label='clip fraction')
    ax_clip.set_ylabel('Raw clean clip fraction'); ax_clip.set_ylim(-.05, 1.05)
    axes[2, 0].set_ylabel('Noise-normalized residual')
    axes[2, 1].plot(u, values('psnr'), '.-', label='PSNR (offline)')
    ax_lpips = axes[2, 1].twinx()
    ax_lpips.plot(u, values('lpips'), '--', color='tab:orange'); ax_lpips.set_ylabel('LPIPS (offline)')
    axes[2, 1].set_ylabel('PSNR dB')
    for ax in axes.flat:
        ax.set_xlabel('Denoising progress u'); ax.grid(alpha=.25); ax.legend(loc='best')
    fig.suptitle('A1a: cached guidance geometry; fixed original trajectory; zero UNet NFE')
    fig.tight_layout(); fig.savefig(directory/'guidance_geometry.png', dpi=160)
    fig.savefig(directory/'guidance_geometry.svg'); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-dir', help='Exact completed A0 Records; default resolves protocol/date, never quality')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--inspect', action='store_true', help='Hash and source resolution only, without Torch')
    parser.add_argument('--plot-only', help='Plot a completed A1a directory from JSON; no Torch/cache replay')
    args = parser.parse_args()
    if args.plot_only:
        if args.inspect or args.trace_dir:
            parser.error('--plot-only cannot be combined with source selection or --inspect')
        directory = Path(args.plot_only).resolve()
        previous = json.loads((directory/'run.json').read_text(encoding='utf-8'))
        if (previous.get('experiment') != 'cached_guidance_geometry' or previous.get('status') != 'complete' or
                not previous.get('gates') or not all(g.get('passed') is True for g in previous['gates'].values())):
            raise ValueError('Plot only accepts a completed, gate-passing A1a run')
        summarize(previous['rows'], previous['summary']['repeat_unit_chord_floor'])
        plot(directory, previous['rows'])
        print('Plot:', directory/'guidance_geometry.png', '; UNet NFE=0', flush=True)
        return
    source, previous, rejected = select_trace(args.trace_dir)
    parent, desc, files = resolve_source(source, previous)
    print('Source:', source, '\nSource SHA256:', files[str(source)], '\nRejected:', rejected, flush=True)
    if args.inspect:
        print('Protocol:', json.dumps(previous['resolved_config'], ensure_ascii=False), flush=True)
        print('Identity files checked:', len(files), '; new UNet NFE=0', flush=True)
        return
    if not args.device.startswith('cuda'):
        parser.error('Production RNG reconstruction requires CUDA; synthetic CPU tests are separate')
    directory = PROJECTS/'results/diag_cached_guidance_geometry'/datetime.now(
        ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    directory.mkdir(parents=True, exist_ok=False)
    card = (PROJECTS/'experiment_cards/A_guidance_geometry.md').read_text(encoding='utf-8')
    (directory/'experiment_card.md').write_text(card, encoding='utf-8')
    implementation = ('utils/diag_cached_guidance_geometry.py', 'utils/diag_cached_direction.py')
    record = dict(experiment='cached_guidance_geometry', status='running', method_nfe=0, diagnostic_nfe=0,
                  model_loaded=False, optimizer_steps=0, arguments=vars(args), source=str(source),
                  source_sha256=files[str(source)], identity_files=files, resolved_config=previous['resolved_config'],
                  implementation_sha256={name: digest(PROJECTS/name) for name in implementation},
                  git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECTS, text=True).strip(),
                  historical_parent_method_nfe=300, historical_trace_diagnostic_nfe=90,
                  experiment_card=card, gates={}, rows=[], source_selection='protocol/date; no quality ranking')
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
        state = torch.load(desc['state'], map_location='cpu', weights_only=True)
        cache = torch.load(source.parent/'trajectory.pt', map_location='cpu', weights_only=True)
        archived = torch.load(desc['output'], map_location='cpu', weights_only=True)
        verify_cache(cache, state, archived, desc)
        alpha = _precompute_diffusion_params(1000)['alphas_cumprod'].to(args.device)
        logs = .5*(torch.log1p(-alpha.double())-alpha.double().log())
        u = ((logs[-1]-logs)/(logs[-1]-logs[0])).cpu().tolist()
        if any(abs(r['noise_u']-u[r['t']]) > 1e-10 for r in previous['rows']):
            raise RuntimeError('Exact original alpha/noise coordinate differs')
        base = get_operator('super_resolution', device=args.device, scale_factor=4, noise_sigma=0.)
        operator = unit_operator(TransposeMode(base, 'legacy_bicubic').to(args.device)).to(args.device)
        dwt = OrthogonalDWT2D(desc['config']['wave'], desc['config']['level']).to(args.device)
        y = torch.load(desc['measurement'], map_location=args.device, weights_only=True)
        zeta, diagonal = state['zeta'].to(args.device), state['D'].to(args.device)
        mapping = saved_cuda_selection(state, parent['arguments']['device'])
        record['rng_mapping'] = mapping
        device_index = torch.device(args.device).index
        device_index = torch.cuda.current_device() if device_index is None else device_index
        record['runtime'] = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                                 gpu=torch.cuda.get_device_name(device_index),
                                 cudnn_benchmark=torch.backends.cudnn.benchmark,
                                 cudnn_deterministic=torch.backends.cudnn.deterministic,
                                 cudnn_tf32=torch.backends.cudnn.allow_tf32,
                                 matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
        rows, errors, q_repeats, unit_repeats = [], [], [], []
        history = {'guidance': [], 'physical': []}
        with torch.random.fork_rng(devices=[device_index]), torch.no_grad():
            restore_sampling_rng(state, args.device, mapping['saved_sampling_index'])
            for k, old in enumerate(previous['rows']):
                x = cache['latent_inputs'][k:k+1].to(args.device)
                clean = cache['clean'][k:k+1].to(args.device)
                i, t, s = old['parameter_index'], old['t'], old['t_prev']
                # The nested fork repeats at the SAME RNG, leaving the outer draw count unchanged.
                with torch.random.fork_rng(devices=[device_index]):
                    repeated = cached_step(x, clean, y, operator, dwt, zeta[i], diagonal[i], t, s, alpha)
                step = cached_step(x, clean, y, operator, dwt, zeta[i], diagonal[i], t, s, alpha)
                expected = cache['latent_inputs'][k+1:k+2] if k < 29 else cache['output']
                error = relative_error(expected, step['next'].cpu()); errors.append(error)
                repeat_error = relative_error(repeated['correction'], step['correction']); q_repeats.append(repeat_error)
                if not all(math.isfinite(e) and e <= 1e-6 for e in (error, repeat_error)):
                    raise RuntimeError(f'Original/repeat cache mismatch at k={k}: {error}, {repeat_error}')
                n, nr = unit_direction(step['correction']), unit_direction(repeated['correction'])
                if (n is None) != (nr is None):
                    raise RuntimeError('Repeat changes whether a direction is defined')
                unit_repeats.append(float((n-nr).norm()) if n is not None and nr is not None else 0.)
                norm = float(step['residual'].double().norm())
                if abs(norm-old['residual_norm']) > 1e-6+2e-5*norm:
                    raise RuntimeError('Residual/source mismatch')
                delta = cache['clean'][k:k+1]-cache['clean'][k-1:k] if k else None
                prev_delta = cache['clean'][k-1:k]-cache['clean'][k-2:k-1] if k >= 2 else None
                c = cosine(delta, prev_delta)
                if ((c is None) != (old['cosine_clean'] is None) or
                        c is not None and abs(c-old['cosine_clean']) > 2e-4):
                    raise RuntimeError('Original clean cosine mismatch')
                physical, loss = physical_descent(operator, clean, y)
                expected_loss = norm*norm/(2*y.numel()*.05*.05)
                if not math.isclose(loss, expected_loss, rel_tol=2e-5, abs_tol=1e-7):
                    raise RuntimeError('Physical loss/residual normalization mismatch')
                row = dict(old, measurement_loss=loss, base_guidance_rms=rms(step['correction'])/float(zeta[i]),
                           actual_zeta=float(zeta[i]), alpha_bar=float(alpha[t]),
                           prior_increment_rms=rms(step['prior_increment']),
                           mean_increment_rms=rms(step['mean_increment']),
                           noise_increment_rms=rms(step['noise_increment']))
                for mode, vector in [('guidance', step['correction']), ('physical', physical)]:
                    h = history[mode]
                    readout = vector_readouts(vector.detach().cpu(), h[-1] if h else None, h[-2] if len(h) >= 2 else None,
                                             old['noise_u'], rows[-1]['noise_u'] if rows else None,
                                             rows[-2]['noise_u'] if len(rows) >= 2 else None)
                    row.update({mode+'_'+key: value for key, value in readout.items()})
                    h.append(vector.detach().cpu().clone())
                rows.append(row)
            actual_rng = rng_state(args.device)
            restore_sampling_rng(state, args.device, mapping['saved_sampling_index'])
            initial = state['init_noise'].to(args.device)
            for _ in range(29): torch.randn_like(initial)
            rng_equal = same_rng(actual_rng, rng_state(args.device))
        if not rng_equal: raise RuntimeError('Original 29-draw RNG suffix differs')
        if any(digest(path) != sha for path, sha in files.items()):
            raise RuntimeError('An original identity file changed during analysis')
        record['gates'] = dict(identity_files_unchanged=dict(passed=True),
            cached_steps=dict(passed=True, max_relative_error=max(errors), hard_ceiling=1e-6),
            correction_repeat=dict(passed=True, max_relative_error=max(q_repeats), unit_chord_floor=max(unit_repeats)),
            noise_draw_suffix=dict(passed=True, expected_draws=29), residual_clean_and_loss=dict(passed=True))
        record['rows'], record['summary'] = rows, summarize(rows, max(unit_repeats))
        record['final_metrics_unchanged'] = previous['final_metrics']
        record['physical_definition'] = 'negative actual-forward-H gradient in clean z space; no denoiser Jacobian'
        torch.save({mode: torch.cat(vectors, dim=0) for mode, vectors in history.items()}, directory/'geometry.pt')
        record['geometry_sha256'] = digest(directory/'geometry.pt')
        with (directory/'geometry.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        record['status'] = 'complete'; save()
        try: plot(directory, rows); record['plot_status'] = 'complete'
        except ImportError: record['plot_status'] = 'matplotlib unavailable; JSON/CSV/tensors complete'
        save()
    except Exception as error:
        record['status'], record['error'] = 'failed', repr(error); save(); raise
    print('\n=== A1a cached guidance geometry: no new trajectory ===', flush=True)
    print(' k    t  clipFrac     qCos    omega   Omega3       O3     qRMS  physCos', flush=True)
    def fmt(v): return '      NA' if v is None else f'{v:8.3f}'
    for r in rows:
        print(f"{r['step']:2d} {r['t']:4d} {r['raw_x0_clip_fraction']:9.3f} " + ' '.join(fmt(r[key]) for key in (
            'guidance_cosine', 'guidance_turn_per_u', 'guidance_variation3_per_u',
            'guidance_backtrack3', 'guidance_rms', 'physical_cosine')), flush=True)
    print('Gates:', json.dumps(record['gates']), flush=True)
    print('Summary:', json.dumps(record['summary']), flush=True)
    print('Final metrics unchanged:', record['final_metrics_unchanged'], flush=True)
    print('Method NFE=0; diagnostic UNet NFE=0; historical parent=300; local sensitivity NOT measured.', flush=True)
    print('Report source:', source, '\nSource SHA256:', record['source_sha256'], flush=True)
    print('Records:', directory, flush=True)


if __name__ == '__main__': main()
