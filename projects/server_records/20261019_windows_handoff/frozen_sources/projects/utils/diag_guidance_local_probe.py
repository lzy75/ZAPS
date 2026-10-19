"""A1b: frozen-t guidance sensitivity on five archived FFHQ inputs.

30 diagnostic UNet forwards, zero optimization/new trajectory/method NFE.
No GT load or output selection. Both perturbation scales are always reported.
"""
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
from utils.diag_cached_direction import validate_trace
from utils.diag_cached_guidance_geometry import resolve_source, verify_cache, rms, unit_direction
from utils.diag_state_trajectory import digest

TARGETS = (.1, .3, .5, .7, .9)
ETAS = (.01, .02)
NFE_LIMIT = 30
STABILITY_RTOL = .20
INPUT_RTOL = .01


def select_snapshots(rows):
    if len(rows) != 30 or [r['step'] for r in rows] != list(range(30)):
        raise ValueError('Need all 30 original ordered rows')
    u = [r['noise_u'] for r in rows]
    if not all(math.isfinite(v) for v in u) or any(b <= a for a, b in zip(u, u[1:])):
        raise ValueError('Need strictly increasing finite original noise coordinates')
    chosen = [min(range(30), key=lambda k: (abs(u[k]-target), k)) for target in TARGETS]
    if len(set(chosen)) != 5:
        raise ValueError('Predeclared u targets map to duplicate points; do not substitute or rank by quality')
    return [dict(target_u=target, step=k, t=rows[k]['t'], noise_u=u[k],
                 parameter_index=rows[k]['parameter_index']) for target, k in zip(TARGETS, chosen)]


def validate_geometry(record):
    if (record.get('experiment') != 'cached_guidance_geometry' or record.get('status') != 'complete' or
            record.get('method_nfe') != 0 or record.get('diagnostic_nfe') != 0 or
            record.get('model_loaded') is not False or record.get('optimizer_steps') != 0):
        raise ValueError('Need the completed zero-UNet A1a record')
    gates = record.get('gates', {})
    required = ('identity_files_unchanged', 'cached_steps', 'correction_repeat',
                'noise_draw_suffix', 'residual_clean_and_loss')
    if not all(gates.get(k, {}).get('passed') is True for k in required):
        raise ValueError('Missing/failed A1a gates')
    if (record.get('summary', {}).get('classification') != 'READABLE_GUIDANCE_DIRECTION_CHANGE' or
            record['summary'].get('guidance_valid', 0) < 24):
        raise ValueError('A1a did not establish predeclared readability')
    select_snapshots(record['rows'])


def select_geometry(explicit=None, root=PROJECTS):
    paths = [Path(explicit).resolve()/'run.json'] if explicit else sorted(
        Path(root).glob('results/diag_cached_guidance_geometry/*/run.json'), reverse=True)
    rejected = []
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding='utf-8')); validate_geometry(record)
            if digest(path.parent/'geometry.pt') != record['geometry_sha256']:
                raise ValueError('A1a direction tensor hash changed')
            return path, record, rejected
        except (ValueError, OSError, KeyError) as error:
            rejected.append(dict(path=str(path), reason=str(error)))
    raise ValueError('No completed compatible A1a source: '+json.dumps(rejected, ensure_ascii=False))


def bind_source(path, geometry):
    trace_path = Path(geometry['source'])
    if digest(trace_path) != geometry['source_sha256']:
        raise RuntimeError('Original A0 record changed since A1a')
    trace = json.loads(trace_path.read_text(encoding='utf-8')); validate_trace(trace)
    parent, desc, files = resolve_source(trace_path, trace)
    if geometry['resolved_config'] != trace['resolved_config']:
        raise RuntimeError('A1a/A0 protocols differ')
    for old, new in zip(trace['rows'], geometry['rows']):
        if any(new.get(k) != v for k, v in old.items()):
            raise RuntimeError('A1a lost/changed original A0 rows')
    for name, sha in geometry['implementation_sha256'].items():
        if digest(PROJECTS/name) != sha:
            raise RuntimeError('A1a implementation changed: '+name)
        files[str(PROJECTS/name)] = sha
    for name, sha in geometry['identity_files'].items():
        if digest(name) != sha:
            raise RuntimeError('A1a historical source changed: '+name)
        files[name] = sha
    files.update({str(path): digest(path), str(path.parent/'geometry.pt'): geometry['geometry_sha256']})
    return trace_path, trace, parent, desc, files


def classify_snapshot(pairs, floors):
    """Distinguish direction resolution, finite-scale descent, and their limits."""
    if [p['eta'] for p in pairs] != list(ETAS):
        raise ValueError('Both predeclared scales, in order, are required')
    result = dict(direction_status='INCONCLUSIVE_SMALL_SIGNAL', loss_status='INCONCLUSIVE_SMALL_SIGNAL',
                  direction_relative_spread=None, loss_relative_spread=None, usable=False)
    if not all(p['input_resolved'] for p in pairs):
        result.update(direction_status='INCONCLUSIVE_INPUT_REPRESENTATION',
                      loss_status='INCONCLUSIVE_INPUT_REPRESENTATION'); return result
    def stable(key):
        values = [p[key] for p in pairs]
        if any(v is None or not math.isfinite(v) for v in values): return None
        denom = max(abs(v) for v in values)
        return abs(values[0]-values[1])/denom if denom else None
    if all(p['direction_chord'] is not None and
           p['direction_chord'] > 10*max(floors['direction_chord'], 1e-12) for p in pairs):
        spread = stable('s_dir'); result['direction_relative_spread'] = spread
        result['direction_status'] = ('RESOLVED_SCALE_CONSISTENT' if spread is not None and spread <= STABILITY_RTOL
                                      else 'INCONCLUSIVE_SCALE_DEPENDENT')
    if all(abs(p['loss_difference']) > 10*max(floors['loss_abs'], 1e-10) for p in pairs):
        spread = stable('phi'); result['loss_relative_spread'] = spread
        if (spread is not None and spread <= STABILITY_RTOL and pairs[0]['phi']*pairs[1]['phi'] > 0):
            result['loss_status'] = 'LOCAL_FINITE_SCALE_DESCENT' if pairs[0]['phi'] < 0 else 'LOCAL_FINITE_SCALE_ASCENT'
        else: result['loss_status'] = 'INCONCLUSIVE_SCALE_DEPENDENT'
    result['usable'] = (result['direction_status'] == 'RESOLVED_SCALE_CONSISTENT' and
                        result['loss_status'] in ('LOCAL_FINITE_SCALE_DESCENT', 'LOCAL_FINITE_SCALE_ASCENT'))
    return result


def evaluate_field(zaps, y, x, t, index):
    """One real epsilon forward; exact original Tweedie and likelihood algebra."""
    import torch
    with torch.no_grad():
        tb = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        eps = zaps.dm._predict_eps(x, tb)
        ab = zaps.dm.alphas_cumprod[t]
        clean = zaps._tweedie_estimate(x, eps, ab, index)
        raw = (x-(1-ab).sqrt()*eps.detach())/ab.sqrt().clamp(min=1e-8)
        residual = y-zaps.A.H(clean)
        v = zaps.A.transpose(residual)
        hv = zaps.dwt.synthesis(zaps.D[index]*zaps.dwt.analysis(v))
        q = zaps.zeta[index]*((v+(1-ab)*hv)/ab.sqrt().clamp(min=1e-8))
        loss = float(residual.double().square().mean()/(2*.05**2))
        if not all(torch.isfinite(a).all() for a in (eps, clean, raw, q)) or not math.isfinite(loss):
            raise RuntimeError('Non-finite snapshot field')
        clip_state = raw.sign()*(raw.abs() > 1).to(raw.dtype)
    return dict(clean=clean, q=q, loss=loss, clip_state=clip_state,
                clip_fraction=float((raw.abs() > 1).float().mean()))


def perturb_inputs(x, q, eta):
    """Use actual fp32 perturbations; disclose noncollinearity from rounding."""
    xp, xm = x+eta*q, x-eta*q
    direction = q.detach().double()
    den = float(direction.square().sum())
    if den <= 1e-16: raise ValueError('Original q direction is near zero')
    ap = float(((xp.double()-x.double())*direction).sum())/den
    am = float(((xm.double()-x.double())*direction).sum())/den
    span = ap-am
    if span <= 0:
        return xp, xm, dict(effective_span=span, input_orthogonal_relative_error=None,
                           input_span_relative_error=1., input_resolved=False)
    expected = direction*span
    orth = rms((xp.double()-xm.double())-expected)/max(rms(expected), 1e-30)
    relative = abs(span-2*eta)/(2*eta)
    return xp, xm, dict(effective_span=span, input_orthogonal_relative_error=orth,
                       input_span_relative_error=relative, input_resolved=orth <= INPUT_RTOL and relative <= INPUT_RTOL)


def pair_readout(plus, minus, input_record, q0, eta, center):
    p, m = unit_direction(plus['q']), unit_direction(minus['q'])
    chord = float((p-m).norm()) if p is not None and m is not None else None
    span = input_record['effective_span']
    difference = plus['loss']-minus['loss']
    return dict(eta=eta, **input_record, direction_chord=chord,
                s_dir=chord/span if chord is not None and span > 0 else None,
                l_path=rms(plus['q']-minus['q'])/(span*rms(q0)) if span > 0 else None,
                loss_difference=difference, phi=difference/span if span > 0 else None,
                plus_loss=plus['loss'], minus_loss=minus['loss'], plus_q_rms=rms(plus['q']), minus_q_rms=rms(minus['q']),
                plus_clip=plus['clip_fraction'], minus_clip=minus['clip_fraction'],
                plus_clip_state_changed=float((plus['clip_state'] != center['clip_state']).float().mean()),
                minus_clip_state_changed=float((minus['clip_state'] != center['clip_state']).float().mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry-dir', help='Completed A1a Records; omitted=compatible source/date selection')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--inspect', action='store_true', help='Bind source and five u targets, no model/Torch load')
    args = parser.parse_args()
    source, geometry, rejected = select_geometry(args.geometry_dir)
    trace_path, trace, parent, desc, files = bind_source(source, geometry)
    snapshots = select_snapshots(geometry['rows'])
    print('A1a source:', source, '\nSource SHA256:', files[str(source)], flush=True)
    print('Snapshots:', json.dumps(snapshots), '\nRejected:', rejected, flush=True)
    if args.inspect:
        print('Plan: 5*(2 center repeats + 4 probes) = 30 diagnostic UNet NFE; no sampling/GT.', flush=True)
        return
    if not args.device.startswith('cuda'): parser.error('Archived FFHQ numerical audit requires CUDA')
    directory = PROJECTS/'results/diag_guidance_local_probe'/datetime.now(
        ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    directory.mkdir(parents=True, exist_ok=False)
    card = (PROJECTS/'experiment_cards/A_guidance_local_probe.md').read_text(encoding='utf-8')
    (directory/'experiment_card.md').write_text(card, encoding='utf-8')
    record = dict(experiment='frozen_time_guidance_local_probe', status='running', source=str(source),
        source_sha256=files[str(source)], identity_files=files, source_trace=str(trace_path),
        resolved_config=geometry['resolved_config'], snapshots=snapshots, arguments=vars(args),
        diagnostic_nfe=0, planned_diagnostic_nfe=30, method_nfe=0, historical_parent_method_nfe=300,
        optimizer_steps=0, gt_loaded=False, new_trajectory=False, calls=[], rows=[], gates={}, experiment_card=card,
        git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECTS, text=True).strip(),
        implementation_sha256=digest(__file__), source_selection='protocol/date and fixed u targets, no quality ranking')
    started = time.perf_counter()
    def save():
        record['seconds'] = time.perf_counter()-started
        (directory/'run.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    save(); print('Records:', directory, flush=True)
    try:
        import torch
        from modules.main_single import load_diffusion_model
        from modules.degradations import get_operator
        from modules.zaps_algorithm import ZAPS
        from utils.diag_ffhq_regression import TransposeMode
        from utils.diag_imagenet_gaussian_unit_domain import unit_operator
        from utils.diag_zaps_first_direction import model_rng_guard
        from utils.diag_zaps_sampler_parity import rng_state, same_rng
        from utils.diag_zaps_trace_audit import relative_error
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        state = torch.load(desc['state'], map_location='cpu', weights_only=True)
        cache = torch.load(trace_path.parent/'trajectory.pt', map_location='cpu', weights_only=True)
        archived = torch.load(desc['output'], map_location='cpu', weights_only=True)
        vectors = torch.load(source.parent/'geometry.pt', map_location='cpu', weights_only=True)
        verify_cache(cache, state, archived, desc)
        if (set(vectors) != {'guidance', 'physical'} or any(tuple(v.shape) != (30, 3, 256, 256) or
                v.dtype != torch.float32 or not torch.isfinite(v).all() for v in vectors.values())):
            raise RuntimeError('Invalid A1a direction tensors')
        device_index = torch.device(args.device).index
        device_index = torch.cuda.current_device() if device_index is None else device_index
        before_rng = rng_state(args.device)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            model = load_diffusion_model('ffhq', args.device); model.model.eval()
            if (digest(model.ckpt_path) != desc['checkpoint_sha'] or model.num_steps != 1000 or
                    any(p.dtype != torch.float32 for p in model.model.parameters())):
                raise RuntimeError('Checkpoint/model precision identity mismatch')
            files[str(model.ckpt_path)] = desc['checkpoint_sha']
            base = get_operator('super_resolution', device=args.device, scale_factor=4, noise_sigma=0.)
            operator = unit_operator(TransposeMode(base, 'legacy_bicubic').to(args.device)).to(args.device)
            z = ZAPS(model, operator, img_size=256, **desc['config']); z.tau = state['tau'].to(args.device)
            with torch.no_grad():
                z.zeta.copy_(state['zeta'].to(args.device)); z.D.copy_(state['D'].to(args.device))
            alpha = model.alphas_cumprod.double()
            logs = .5*(torch.log1p(-alpha)-alpha.log()); u = ((logs[-1]-logs)/(logs[-1]-logs[0])).cpu().tolist()
            if any(abs(r['noise_u']-u[r['t']]) > 1e-10 for r in geometry['rows']):
                raise RuntimeError('Model alpha_bar/noise coordinates differ from parent')
            y = torch.load(desc['measurement'], map_location=args.device, weights_only=True)
            if tuple(y.shape) != (1, 3, 64, 64) or y.dtype != torch.float32 or not torch.isfinite(y).all():
                raise RuntimeError('Original unit measurement shape/dtype invalid')
            record['runtime'] = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(device_index), cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_tf32=torch.backends.cudnn.allow_tf32, matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
            current = {}
            def count(_module, inputs):
                if record['diagnostic_nfe'] >= NFE_LIMIT: raise RuntimeError('30-forward budget exceeded')
                record['diagnostic_nfe'] += 1
                if int(inputs[1].flatten()[0]) != current['t']: raise RuntimeError('Probe changed frozen t')
                record['calls'].append(dict(current, nfe=record['diagnostic_nfe']))
            hook = model.model.register_forward_pre_hook(count)
            fingerprints = []
            try:
                with model_rng_guard(model.model, args.device), torch.no_grad():
                    for snap in snapshots:
                        k, t, i = snap['step'], snap['t'], snap['parameter_index']
                        x = cache['latent_inputs'][k:k+1].to(args.device)
                        q0 = vectors['guidance'][k:k+1].to(args.device)
                        if unit_direction(q0) is None: raise RuntimeError('Predeclared snapshot has undefined q')
                        centers = []
                        for number in range(2):
                            current.update(step=k, t=t, parameter_index=i, kind='center', repeat=number, eta=0.)
                            c = evaluate_field(z, y, x, t, i); centers.append(c)
                            clean_error = relative_error(cache['clean'][k:k+1], c['clean'].cpu())
                            q_error = relative_error(q0, c['q'])
                            loss_error = abs(c['loss']-geometry['rows'][k]['measurement_loss'])
                            if not all(math.isfinite(v) and v <= 1e-6 for v in (clean_error, q_error)):
                                raise RuntimeError(f'Center does not match archived clean/q at k={k}: {clean_error}, {q_error}')
                            if loss_error > 1e-7+2e-5*abs(c['loss']):
                                raise RuntimeError('Center physical loss/archive mismatch')
                            fingerprints.append(dict(step=k, repeat=number, clean_relative_error=clean_error,
                                                     q_relative_error=q_error, loss_absolute_error=loss_error))
                            if abs(c['clip_fraction']-geometry['rows'][k]['raw_x0_clip_fraction']) > 1e-5:
                                raise RuntimeError('Center clipping differs from original')
                        n0, n1 = (unit_direction(c['q']) for c in centers)
                        if n0 is None or n1 is None:
                            raise RuntimeError('Center repeat has an undefined q direction')
                        floors = dict(direction_chord=float((n0-n1).norm()),
                                      q_rms=rms(centers[0]['q']-centers[1]['q']),
                                      loss_abs=abs(centers[0]['loss']-centers[1]['loss']))
                        pairs = []
                        for eta in ETAS:
                            xp, xm, inp = perturb_inputs(x, q0, eta)
                            current.update(kind='plus', eta=eta, repeat=None)
                            plus = evaluate_field(z, y, xp, t, i)
                            current.update(kind='minus')
                            minus = evaluate_field(z, y, xm, t, i)
                            pairs.append(pair_readout(plus, minus, inp, q0, eta, centers[0]))
                        row = dict(snap, q_rms=rms(q0), center_clip=centers[0]['clip_fraction'],
                                   center_loss=centers[0]['loss'], repeat_floors=floors, pairs=pairs,
                                   classification=classify_snapshot(pairs, floors))
                        record['rows'].append(row); save()
            finally: hook.remove()
            unchanged = (torch.equal(z.zeta.detach().cpu(), state['zeta']) and
                         torch.equal(z.D.detach().cpu(), state['D']) and torch.equal(z.tau.cpu(), state['tau']))
            if not unchanged or record['diagnostic_nfe'] != 30 or len(record['calls']) != 30:
                raise RuntimeError('Frozen parameters/grid or exact budget gate failed')
        if not same_rng(before_rng, rng_state(args.device)): raise RuntimeError('Global RNG changed after fork')
        if any(digest(path) != sha for path, sha in files.items()): raise RuntimeError('An original identity file changed')
        record['gates'] = dict(center_archive=dict(passed=True, fingerprints=fingerprints, relative_ceiling=1e-6),
            frozen_parameters_and_grid=dict(passed=True), no_sampling_and_rng_unchanged=dict(passed=True),
            identity_files_unchanged=dict(passed=True), exact_budget=dict(passed=True, unet_forwards=30))
        count_usable = sum(r['classification']['usable'] for r in record['rows'])
        record['summary'] = dict(usable_snapshots=count_usable,
            finite_scale_descents=sum(r['classification']['loss_status']=='LOCAL_FINITE_SCALE_DESCENT' for r in record['rows']),
            finite_scale_ascents=sum(r['classification']['loss_status']=='LOCAL_FINITE_SCALE_ASCENT' for r in record['rows']),
            classification='PROBE_READABLE_CANDIDATE' if count_usable >= 4 else 'INCONCLUSIVE_PROBE_READABILITY',
            controller_suitability='NOT_ESTABLISHED',
            limits='Five snapshots on one explored trajectory. Fixed-t directional finite-scale response, not full curvature, '
                   'a Lipschitz bound, infinitesimal derivative certification or step-count benefit. All scales disclosed; '
                   'small/unresolved signals and clipping changes retained. No GT use or new reconstructed output.')
        record['final_metrics_unchanged'] = geometry['final_metrics_unchanged']
        flat = [dict(step=r['step'], t=r['t'], noise_u=r['noise_u'], q_rms=r['q_rms'],
                     center_clip=r['center_clip'], **r['classification'], **p) for r in record['rows'] for p in r['pairs']]
        with (directory/'probes.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
        record['status'] = 'complete'; save()
    except Exception as error:
        record['status'], record['error'] = 'failed', repr(error); save(); raise
    print('\n=== A1b frozen-time guidance local probe / no new trajectory ===', flush=True)
    for r in record['rows']:
        print('Snapshot:', json.dumps({k:v for k,v in r.items() if k != 'pairs'}), flush=True)
        for p in r['pairs']: print('Pair:', json.dumps(p), flush=True)
    print('Gates:', json.dumps(record['gates']), flush=True)
    print('Summary:', json.dumps(record['summary']), flush=True)
    print('Final metrics unchanged:', record['final_metrics_unchanged'], flush=True)
    print('New method NFE=0; diagnostic UNet NFE=30; no optimizer, GT or sampling.', flush=True)
    print('Report source:', source, '\nSource SHA256:', record['source_sha256'], '\nRecords:', directory, flush=True)


if __name__ == '__main__': main()
