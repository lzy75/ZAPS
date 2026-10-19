"""Passive single-image FFHQ SR trajectory audit. No training or schedule change.

Replay a completed parent's PRE-last-unroll snapshot, observation and RNG.
Three fixed-grid replays (90 diagnostic NFE); zero new method NFE.
Default source priority: known successful historical unit-null, then public-settings four-task run.
Selection uses protocol/date and predeclared image 00000, NEVER quality scores.
"""
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from unittest.mock import patch
from zoneinfo import ZoneInfo

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def describe_parent(parent, directory):
    """Normalize only the two known completed, no-state FFHQ SR protocols."""
    directory = Path(directory)
    if parent.get('status') != 'complete':
        raise ValueError('Parent must be complete')
    kind = parent.get('experiment')
    if kind == 'ffhq_four_task_public_settings_unit_measurements':
        item = parent['input_manifest'][0]
        key = '000_super_resolution'
        arm = parent['arms'][key]
        if (not arm.get('training_gate_passed') or
                not parent['gates'][key].get('passed') or
                parent['resolved_protocol']['state'] is not False):
            raise ValueError('Four-task parent gates/state invalid')
        folder = directory / 'image_000/super_resolution'
        result = next(r for r in parent['results']
                      if r['task'] == 'super_resolution' and r['image'] == item['path'])
        desc = dict(kind='public_settings', image=item['path'], seed=item['seed'],
                    image_sha=item['sha256'], config=arm['config'], grid=arm['grid'],
                    measurement=folder / 'measurement.pt', state=folder / 'pre_last_state.pt',
                    output=folder / 'last_opt.pt', measurement_sha=parent['observations'][key]['measurement_sha256'],
                    state_sha=arm['pre_snapshot_sha256'], output_sha=arm['output_sha256'],
                    checkpoint_sha=parent['checkpoint']['sha256'], sources=parent['source_sha256'],
                    transpose='exact', metrics={k: result[k] for k in ('psnr', 'ssim', 'lpips')})
    elif kind == 'fixed30_state_measurement_domain_paired' and parent.get('dataset') == 'ffhq':
        key = 'unit_null'
        arm = parent['arms'][key]
        grid = parent['nominal_timesteps_ascending']
        if (not parent['gates']['operator']['passed'] or
                not parent['gates']['unroll_parity']['unit']['passed'] or
                not arm['noise_pairing_passed'] or arm['changed_points_after_pilot'] != 0 or
                arm['grids'] != [list(reversed(grid))] * 10 or
                parent['scheduler_config'][key]['response_strength'] != 0):
            raise ValueError('Historical unit-null parent gates/grid invalid')
        result = next(r for r in parent['results'] if r['domain'] == 'unit' and r['policy'] == 'null')
        desc = dict(kind='historical_development', image=parent['image'], seed=parent['seed'],
                    image_sha=parent['identity']['image_sha256'], config=parent['config'], grid=grid,
                    measurement=directory / 'unit_measurement.pt', state=directory / 'unit_null_pre_last_state.pt',
                    output=directory / 'unit_null_last_opt.pt', measurement_sha=parent['measurement_sha256']['unit'],
                    state_sha=arm['pre_snapshot_sha256'], output_sha=arm['output_sha256'],
                    checkpoint_sha=parent['identity']['checkpoint_sha256'],
                    sources=parent['identity']['source_sha256'], transpose='legacy_bicubic',
                    metrics={k: result[k] for k in ('psnr', 'ssim', 'lpips')})
    else:
        raise ValueError('Unsupported parent: use completed FFHQ four-task or state-domain unit-null run')
    grid, c = desc['grid'], desc['config']
    if (Path(desc['image']).stem != '00000' or desc['seed'] != 1000 or
            len(grid) != 30 or grid != sorted(set(grid)) or grid[0] != 0 or grid[-1] != 999 or
            c['num_steps'] != 30 or c['num_epochs'] != 10 or c['eta'] != 1 or
            c['sampler_mode'] != 'ddpm' or c['use_learned_var'] or c['surrogate_score_jacobian']):
        raise ValueError('Card requires predeclared FFHQ 00000/seed1000/fixed30 DDPM')
    return desc


def select_parent(explicit=None, root=PROJECTS):
    if explicit:
        paths = [Path(explicit).resolve() / 'run.json']
    else:
        # Ordered protocol tiers, latest path within tier. No metric ranking.
        paths = sorted(Path(root).glob('results/diag_state_domain_paired/ffhq/state_domain_*/run.json'), reverse=True)
        paths += sorted(Path(root).glob('results/diag_ffhq_four_tasks_unit/*/run.json'), reverse=True)
    rejected = []
    for path in paths:
        try:
            parent = json.loads(path.read_text(encoding='utf-8'))
            desc = describe_parent(parent, path.parent)
            if not all(desc[k].is_file() for k in ('measurement', 'state', 'output')):
                raise ValueError('Raw observation/PRE snapshot/output missing; text-only archive is insufficient')
            return path, parent, desc, rejected
        except (ValueError, KeyError, OSError, StopIteration) as error:
            rejected.append(dict(path=str(path), reason=str(error)))
    raise ValueError('No compatible completed parent. Run --source-dir on the server with original tensors. '
                     + json.dumps(rejected, ensure_ascii=False))


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None
             and math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 4:
        return None
    xx, yy = zip(*pairs)
    mx, my = statistics.mean(xx), statistics.mean(yy)
    dx, dy = [x - mx for x in xx], [y - my for y in yy]
    denom = math.sqrt(sum(x*x for x in dx) * sum(y*y for y in dy))
    return sum(x*y for x, y in zip(dx, dy)) / denom if denom > 1e-15 else None


def ranks(values):
    ordered = sorted(range(len(values)), key=values.__getitem__)
    out = [0.] * len(values)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[ordered[j]] == values[ordered[i]]:
            j += 1
        for k in range(i, j):
            out[ordered[k]] = (i + j - 1) / 2
        i = j
    return out


def association(rows, xkey, ykey, next_change=False):
    pairs = []
    for k, row in enumerate(rows[:-1] if next_change else rows):
        x, y = row[xkey], row[ykey]
        if next_change:
            y = rows[k+1][ykey] - y
        if x is not None and y is not None and math.isfinite(x) and math.isfinite(y):
            pairs.append((x, y, row['noise_u']))
    if len(pairs) < 4:
        return dict(n=len(pairs), pearson=None, spearman=None, linear_u_partial=None)
    x, y, u = map(list, zip(*pairs))
    def detrend(v):
        mu, mv = statistics.mean(u), statistics.mean(v)
        den = sum((a-mu)**2 for a in u)
        slope = sum((a-mu)*(b-mv) for a, b in zip(u, v))/den if den > 1e-15 else 0.
        return [b - mv - slope*(a-mu) for a, b in zip(u, v)]
    return dict(n=len(pairs), pearson=pearson(x, y), spearman=pearson(ranks(x), ranks(y)),
                linear_u_partial=pearson(detrend(x), detrend(y)))


def summarize(rows):
    if len(rows) != 30 or [r['step'] for r in rows] != list(range(30)):
        raise ValueError('Need one complete ordered 30-step trajectory')
    phases = {}
    for name, predicate in (('high', lambda t: t >= 667), ('middle', lambda t: 334 <= t < 667),
                            ('low', lambda t: t < 334)):
        group = [r for r in rows if predicate(r['t'])]
        phases[name] = dict(n=len(group))
        for key in ('residual_noise_ratio', 'cosine_clean', 'cosine_latent', 'psnr', 'ssim', 'lpips'):
            values = [r[key] for r in group if r[key] is not None]
            phases[name][key + '_median'] = statistics.median(values) if values else None
    cos = [r['cosine_clean'] for r in rows if r['cosine_clean'] is not None]
    links = {f'{x}_vs_{"next_change_" if delta else ""}{y}': association(rows, x, y, delta)
             for x, y, delta in [('residual_noise_ratio', 'psnr', False),
                                 ('residual_noise_ratio', 'lpips', False),
                                 ('cosine_clean', 'psnr', True), ('cosine_clean', 'lpips', True),
                                 ('cosine_clean', 'residual_noise_ratio', True)]}
    return dict(phases=phases, associations=links,
                residual_final_over_initial=rows[-1]['residual_noise_ratio']/max(rows[0]['residual_noise_ratio'], 1e-12),
                residual_rise_count=sum(b['residual_norm'] > a['residual_norm'] for a, b in zip(rows, rows[1:])),
                clean_cosine_valid=len(cos), clean_cosine_span=max(cos)-min(cos) if cos else None,
                clean_cosine_readout=('DYNAMIC_RANGE_CANDIDATE_ONLY' if len(cos) >= 24 and max(cos)-min(cos) >= .05
                                      else 'FLAT_OR_INSUFFICIENT_VALID_DIRECTIONS'),
                lower_late_residual=phases['low']['residual_noise_ratio_median'] < phases['high']['residual_noise_ratio_median'],
                controller_suitability='NOT_ESTABLISHED_BY_SINGLE_PASSIVE_TRAJECTORY',
                limits='Descriptive, autocorrelated single-image data. Linear-u partial correlation only removes a linear trend. '
                       'No p-values/causal claims; no GT output selection; no required monotonic cosine or residual-to-zero target.')


@contextmanager
def capture_clean(zaps, captures):
    """Observe the real Tweedie helper; return the identical tensor object."""
    import torch
    original = zaps._tweedie_estimate
    def observe(x, eps, ab, index):
        clean = original(x, eps, ab, index)
        with torch.no_grad():
            raw = (x - (1-ab).sqrt()*eps.detach()) / ab.sqrt().clamp(min=1e-8)
            captures.append(dict(index=index, x=x.detach().cpu().clone(), clean=clean.detach().cpu().clone(),
                                 clip_fraction=float((raw.abs() > 1).float().mean())))
        return clean
    with patch.object(zaps, '_tweedie_estimate', side_effect=observe):
        yield


def cosine(a, b):
    if a is None or b is None:
        return None
    a, b = a.double().flatten(), b.double().flatten()
    # Undefined/near-zero directions are missing, never a synthetic cosine=0.
    if float(a.norm()) <= 1e-8 or float(b.norm()) <= 1e-8:
        return None
    return max(-1., min(1., float((a*b).sum() / (a.norm()*b.norm()))))


def make_rows(zaps, y, output, captures, indicators, gt):
    import torch
    from utils.metrics import compute_all_metrics
    ab = zaps.dm.alphas_cumprod.detach().cpu().double().tolist()
    logs = [.5 * (math.log1p(-a) - math.log(a)) for a in ab]
    rows, prev_clean_delta, prev_velocity, prev_latent_delta = [], None, None, None
    for k, (cap, core) in enumerate(zip(captures, indicators)):
        clean, x = cap['clean'], cap['x']
        t = core['t']; u = (logs[-1]-logs[t])/(logs[-1]-logs[0])
        next_x = captures[k+1]['x'] if k+1 < len(captures) else output.cpu()
        delta_x = next_x-x
        delta = clean-captures[k-1]['clean'] if k else None
        du = u-rows[-1]['noise_u'] if k else None
        velocity = delta.double()/du if k else None
        with torch.no_grad():
            residual = y-zaps.A.H(clean.to(zaps.device))
            norm = float(residual.double().norm())
        if abs(norm-core['residual_norm']) > 1e-6+2e-5*norm:
            raise RuntimeError('Observer/core residual disagrees')
        cclean, clatent = cosine(delta, prev_clean_delta), cosine(delta_x, prev_latent_delta)
        curvature = (float((velocity-prev_velocity).norm()/(velocity.norm()+prev_velocity.norm()).clamp(min=1e-30))
                     if velocity is not None and prev_velocity is not None
                     and float(velocity.norm()+prev_velocity.norm()) > 1e-12 else None)
        metrics = compute_all_metrics(clean, gt, lpips_net='alex')
        mse = float(((clean.double().clamp(-1, 1)-gt.double())/2).square().mean())
        row = dict(step=k, t=t, t_prev=int(zaps.tau[cap['index']-1]) if cap['index'] else -1,
                   noise_u=u, delta_u=du, parameter_index=cap['index'], zeta=float(zaps.zeta[cap['index']]),
                   residual_norm=norm, residual_rms=norm/math.sqrt(y.numel()),
                   residual_noise_ratio=norm/(.05*math.sqrt(y.numel())),
                   relative_residual_drop=(rows[-1]['residual_norm']-norm)/max(rows[-1]['residual_norm'], 1e-12) if k else None,
                   cosine_clean=cclean, cosine_latent=clatent, velocity_turn=curvature,
                   clean_delta_rms=float(delta.double().square().mean().sqrt()) if k else None,
                   clean_velocity_rms=float(velocity.square().mean().sqrt()) if k else None,
                   raw_x0_clip_fraction=cap['clip_fraction'], float_psnr=-10*math.log10(max(mse, 1e-30)), **metrics)
        # Cross-check the core's own historical angle diagnostics where norms are not tiny.
        for actual, key in ((cclean, 'cosine_sim_x0'), (clatent, 'cosine_sim')):
            if actual is not None and math.isfinite(core[key]) and abs(actual-core[key]) > 2e-4:
                raise RuntimeError('Observer/core cosine disagrees (possibly nearly zero updates)')
        rows.append(row)
        prev_clean_delta, prev_velocity, prev_latent_delta = delta, velocity, delta_x
    return rows


def plot_run(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    record = json.loads((Path(directory)/'run.json').read_text(encoding='utf-8'))
    if record['status'] != 'complete':
        raise ValueError('Plot only completed gated trajectories')
    rows = record['rows']; u = [r['noise_u'] for r in rows]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for key, label in (('cosine_clean', 'clean increments'), ('cosine_latent', 'noisy latent increments')):
        axes[0, 0].plot(u, [float('nan') if r[key] is None else r[key] for r in rows], '.-', label=label)
    axes[0, 0].set_ylim(-1.05, 1.05); axes[0, 0].set_ylabel('Adjacent direction cosine'); axes[0, 0].legend()
    axes[0, 1].semilogy(u, [r['residual_noise_ratio'] for r in rows], '.-')
    axes[0, 1].axhline(1, color='gray', linestyle='--', label='noise scale (not a stop threshold)')
    axes[0, 1].set_ylabel('Residual RMS / sigma_y'); axes[0, 1].legend(fontsize=8)
    for ax, key, label in ((axes[1, 0], 'psnr', 'PSNR / dB (offline GT)'),
                           (axes[1, 1], 'lpips', 'LPIPS (offline GT)'),
                           (axes[2, 0], 'velocity_turn', 'Clean velocity turn'),
                           (axes[2, 1], 'raw_x0_clip_fraction', 'Raw clean clip fraction')):
        ax.plot(u, [float('nan') if r[key] is None else r[key] for r in rows], '.-'); ax.set_ylabel(label)
    for ax in axes.flat:
        ax.grid(alpha=.25); ax.set_xlabel('Denoising progress u: t=999 -> t=0')
    fig.suptitle('FFHQ 00000 SR: original last optimization unroll, fixed parent grid')
    fig.tight_layout(); fig.savefig(Path(directory)/'trajectory.png', dpi=160)
    fig.savefig(Path(directory)/'trajectory.svg'); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', help='Original completed server run; omitted=protocol/date selection')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--inspect', action='store_true', help='Resolve source and print protocol, zero UNet')
    parser.add_argument('--plot-only', help='Plot an already completed trajectory directory; zero UNet')
    args = parser.parse_args()
    if args.plot_only:
        plot_run(Path(args.plot_only)); return
    source, parent, desc, rejected = select_parent(args.source_dir)
    print('Parent:', source, '\nProtocol:', desc['kind'], '/ unit measurements /', desc['transpose'], flush=True)
    print('Fixed config:', json.dumps(desc['config']), '\nFixed grid:', desc['grid'], flush=True)
    print('Cached final metrics:', desc['metrics'], '\nRejected sources:', rejected, flush=True)
    if args.inspect:
        return
    if not args.device.startswith('cuda'):
        parser.error('Production replay requires CUDA; CPU synthetic tests are separate')
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    directory = PROJECTS/'results/diag_state_trajectory'/stamp
    directory.mkdir(parents=True, exist_ok=False)
    card = (PROJECTS/'experiment_cards/A_state_trajectory.md').read_text(encoding='utf-8')
    (directory/'experiment_card.md').write_text(card, encoding='utf-8')
    record = dict(experiment='passive_fixed_parent_state_trajectory', status='running', source=str(source),
                  source_sha256=digest(source), source_git=parent['git_revision'], arguments=vars(args),
                  resolved_config={k: str(v) if isinstance(v, Path) else v for k, v in desc.items() if k != 'sources'},
                  method_nfe=0, diagnostic_nfe=0, planned_diagnostic_nfe=90, historical_method_nfe=300,
                  optimizer_steps=0, extra_final_sample=False, gates={}, rows=[], experiment_card=card,
                  git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECTS, text=True).strip(),
                  source_selection='protocol priority/date; image00000 seed1000 fixed, no quality ranking',
                  indicator_semantics='r/c_clean use pre-correction clipped Tweedie estimates; c_latent uses actual post-update increments. '
                    'First two clean cosines and first latent cosine undefined. u-normalization does not change cosine angles. '
                    'Velocity-turn is a proxy, not local truncation error. Final corrected output is a separate row in final_metrics.',
                  gt_usage='Only offline quality evaluation after replay gates; never control/stop/output selection')
    started = time.perf_counter()
    def save():
        record['elapsed_seconds'] = time.perf_counter()-started
        (directory/'run.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    save(); print('Records:', directory, flush=True)
    try:
        import torch
        from modules.main_single import load_diffusion_model, load_image_as_tensor
        from modules.degradations import get_operator
        from modules.zaps_algorithm import ZAPS
        from utils.diag_ffhq_regression import TransposeMode
        from utils.diag_imagenet_gaussian_unit_domain import unit_operator
        from utils.diag_zaps_guidance_clipping import saved_cuda_selection, restore_sampling_rng
        from utils.diag_zaps_sampler_parity import rng_state, same_rng
        from utils.diag_zaps_trace_audit import relative_error, classify_parity
        from utils.diag_zaps_first_direction import model_rng_guard
        from utils.metrics import compute_all_metrics
        from modules.dataset_loader import tensor_to_image
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        for key in ('measurement', 'state', 'output'):
            if digest(desc[key]) != desc[key+'_sha']:
                raise RuntimeError('Parent tensor hash changed: '+key)
        if digest(desc['image']) != desc['image_sha']:
            raise RuntimeError('Image hash changed')
        for name, expected in desc['sources'].items():
            if digest(PROJECTS/name) != expected:
                raise RuntimeError('Parent source changed: '+name)
        record['observer_sha256'] = digest(__file__)
        record['runtime'] = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                                 gpu=torch.cuda.get_device_name(torch.device(args.device)), python=sys.version,
                                 cudnn_benchmark=torch.backends.cudnn.benchmark,
                                 cudnn_deterministic=torch.backends.cudnn.deterministic,
                                 cudnn_tf32=torch.backends.cudnn.allow_tf32,
                                 matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
        state = torch.load(desc['state'], map_location='cpu', weights_only=True)
        if state['tau'].tolist() != desc['grid']:
            raise RuntimeError('PRE snapshot/grid mismatch')
        mapping = saved_cuda_selection(state, parent['arguments']['device'])
        y = torch.load(desc['measurement'], map_location=args.device, weights_only=True)
        archived = torch.load(desc['output'], map_location='cpu', weights_only=True)
        model = load_diffusion_model('ffhq', args.device); model.model.eval()
        if (digest(model.ckpt_path) != desc['checkpoint_sha'] or model.num_steps != 1000 or
                any(p.dtype != torch.float32 for p in model.model.parameters())):
            raise RuntimeError('Checkpoint identity mismatch')
        base = get_operator('super_resolution', device=args.device, scale_factor=4, noise_sigma=0.)
        operator = unit_operator(TransposeMode(base, desc['transpose']).to(args.device)).to(args.device)
        z = ZAPS(model, operator, img_size=256, **desc['config']); z.tau = state['tau'].to(args.device)
        with torch.no_grad():
            z.zeta.copy_(state['zeta'].to(args.device)); z.D.copy_(state['D'].to(args.device))
        initial = state['init_noise'].to(args.device)
        if (y.shape != (1, 3, 64, 64) or archived.shape != (1, 3, 256, 256) or
                initial.shape != archived.shape or y.dtype != torch.float32):
            raise RuntimeError('Expected original single-image fp32 FFHQ 4x SR shapes')
        if not all(torch.isfinite(v).all() for v in (y, archived, initial, z.zeta, z.D)):
            raise RuntimeError('Non-finite archived tensors')
        captures = []
        replay_results = []
        for which in ('observed', 'core_a', 'core_b'):
            restore_sampling_rng(state, args.device, mapping['saved_sampling_index'])
            visited = []
            def count(module, inputs):
                record['diagnostic_nfe'] += 1
                if record['diagnostic_nfe'] > 90:
                    raise RuntimeError('Diagnostic NFE limit exceeded')
                visited.append(int(inputs[1].flatten()[0]))
            hook = model.model.register_forward_pre_hook(count)
            try:
                with model_rng_guard(model.model, args.device) as guarded:
                    if which == 'observed':
                        with capture_clean(z, captures):
                            output, nfe, _ = z.sample(y, init_noise=initial)
                        indicators = list(z._indicator_log)
                    else:
                        output, nfe, _ = z.sample(y, init_noise=initial)
            finally:
                hook.remove()
            replay_results.append(dict(output=output.detach().cpu(), rng=rng_state(args.device),
                                       nfe=nfe, visited=visited, guarded=len(guarded)))
            save()
        observed, a, b = replay_results
        metadata = (record['diagnostic_nfe'] == 90 and len(captures) == 30 and
                    [c['index'] for c in captures] == list(range(29, -1, -1)) and
                    all(r['nfe'] == r['guarded'] == 30 and r['visited'] == list(reversed(desc['grid']))
                        for r in replay_results) and same_rng(observed['rng'], a['rng']) and same_rng(a['rng'], b['rng']) and
                    torch.equal(z.zeta.detach().cpu(), state['zeta']) and torch.equal(z.D.detach().cpu(), state['D']))
        floor = relative_error(a['output'], b['output'])
        record['gates']['observer'] = classify_parity(floor, relative_error(a['output'], observed['output']), metadata)
        record['gates']['archive'] = classify_parity(floor, relative_error(a['output'], archived), metadata)
        save()
        if not all(g['passed'] for g in record['gates'].values()):
            raise RuntimeError('Replay gate failed; do not interpret trajectory')
        # All online computation is over before any GT metric is evaluated.
        gt = load_image_as_tensor(desc['image']).cpu()
        rows = make_rows(z, y, observed['output'], captures, indicators, gt)
        final = compute_all_metrics(observed['output'], gt, lpips_net='alex')
        drift = {k: final[k]-desc['metrics'][k] for k in final}
        if any(abs(drift[k]) > limit for k, limit in (('psnr', .01), ('ssim', .001), ('lpips', .001))):
            raise RuntimeError('Archived metric convention drift: '+str(drift))
        record['rows'], record['summary'], record['final_metrics'], record['metric_drift'] = rows, summarize(rows), final, drift
        with torch.no_grad():
            final_residual = float((y-operator.H(observed['output'].to(args.device))).double().norm())
        record['final_residual_noise_ratio'] = final_residual/(.05*math.sqrt(y.numel()))
        torch.save(dict(clean=torch.cat([c['clean'] for c in captures]),
                        latent_inputs=torch.cat([c['x'] for c in captures]), output=observed['output']), directory/'trajectory.pt')
        with (directory/'trajectory.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        for k in (0, 5, 10, 15, 20, 25, 29):
            tensor_to_image(captures[k]['clean'].squeeze(0), denormalize=True).save(directory/f'clean_k{k:02d}_t{rows[k]["t"]:03d}.png')
        tensor_to_image(observed['output'].squeeze(0), denormalize=True).save(directory/'final.png')
        record['status'] = 'complete'; save()
        try:
            plot_run(directory); record['plot_status'] = 'complete'
        except ImportError:
            record['plot_status'] = 'matplotlib unavailable; CSV/tensors complete; use --plot-only after installing matplotlib'
        save()
    except Exception as error:
        record['status'], record['error'] = 'failed', repr(error); save(); raise
    print('\n=== Passive state trajectory: gated last optimization unroll ===', flush=True)
    print(' k    t    r/sigma  cleanCos noisyCos   PSNR  LPIPS', flush=True)
    def fmt(v): return '     NA' if v is None else f'{v:7.3f}'
    for r in record['rows']:
        print(f"{r['step']:2d} {r['t']:4d} {r['residual_noise_ratio']:10.3f} {fmt(r['cosine_clean'])} "
              f"{fmt(r['cosine_latent'])} {r['psnr']:6.2f} {r['lpips']:6.3f}", flush=True)
    print('Gates:', json.dumps(record['gates']), flush=True)
    print('Final metrics:', record['final_metrics'], flush=True)
    print('Summary:', json.dumps(record['summary'], ensure_ascii=False), flush=True)
    print('New method NFE=0; diagnostic NFE=90; historical parent method NFE=300', flush=True)
    print('Plot:', record['plot_status'], '\nRecords:', directory, flush=True)


if __name__ == '__main__':
    main()
