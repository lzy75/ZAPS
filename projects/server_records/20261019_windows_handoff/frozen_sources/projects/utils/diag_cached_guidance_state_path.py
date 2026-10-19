"""Exploratory CPU-only A2a path-speed/axis decomposition; no model or GT load.

Reuses the five fixed snapshots in all six verified active A2a cases. The state
path secant still mixes x, time, learned D and random history; it is not a
frozen-time derivative, curvature or a step-size/controller recommendation.
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
from utils import archive_guidance_replication_stage as archive
from utils.diag_state_trajectory import digest, pearson, ranks

CARD = PROJECTS/'experiment_cards/A2b0_cached_state_path_20261018.md'
TIMES = (903, 641, 319, 54, 14)


def path_decomposition(direction_arc, state_arc, u_span):
    if (not all(math.isfinite(v) for v in (direction_arc, state_arc, u_span)) or
            direction_arc < 0 or state_arc < 0 or u_span <= 0):
        raise ValueError('Need finite nonnegative arcs and a positive noise-coordinate span')
    omega = direction_arc/u_span
    speed = state_arc/u_span
    secant = direction_arc/state_arc if state_arc > 1e-12 else None
    return dict(direction_arc3=direction_arc, state_arc3_rms=state_arc, noise_span3=u_span,
                omega3_recomputed=omega, state_speed3=speed, direction_path_secant3=secant,
                factorization_error=abs(omega-secant*speed) if secant is not None else None,
                path_status='RESOLVED_FINITE_SECANT' if secant is not None else 'UNRESOLVED_ZERO_STATE_ARC')


def point_readout(q, x, rows, probe):
    from utils.diag_cached_guidance_geometry import rms, unit_direction
    k = probe['step']
    if k < 2 or rows[k]['t'] != probe['t']:
        raise ValueError('Need the original three adjacent states, no replacement snapshots')
    directions = [unit_direction(q[j]) for j in (k, k-1, k-2)]
    if any(n is None for n in directions):
        raise ValueError('Source promised valid directions but the cached tensor differs')
    n, previous, older = directions
    incoming = x[k].double()-x[k-1].double()
    prior = x[k-1].double()-x[k-2].double()
    result = path_decomposition(float((n-previous).norm()+(previous-older).norm()),
                                rms(incoming)+rms(prior), rows[k]['noise_u']-rows[k-2]['noise_u'])
    cached = rows[k]['guidance_variation3_per_u']
    if cached is None or not math.isclose(result['omega3_recomputed'], cached, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError('Cached Omega3 does not reconstruct under the original neighbors')
    amplitude = rms(q[k])
    if not math.isclose(amplitude, probe['q_rms'], rel_tol=1e-6, abs_tol=1e-12):
        raise ValueError('Cached guidance magnitude differs from the frozen-time probe')
    incoming_direction = unit_direction(incoming)
    alignment = float(incoming_direction.dot(n)) if incoming_direction is not None else None
    result.update(step=k, t=probe['t'], noise_u=rows[k]['noise_u'], q_rms=amplitude,
                  incoming_state_rms=rms(incoming), incoming_q_cosine=alignment,
                  incoming_q_abs_cosine=abs(alignment) if alignment is not None else None,
                  omega3_cached=cached, physical_residual_noise_ratio=rows[k]['residual_noise_ratio'],
                  clip=rows[k]['raw_x0_clip_fraction'])
    for scale, eta in enumerate((.01, .02)):
        pair = probe['pairs'][scale]
        if pair['eta'] != eta:
            raise ValueError('Frozen-time probe scales/order changed')
        suffix = '01' if scale == 0 else '02'
        result['s_dir_'+suffix] = pair['s_dir']
        result['kappa_dir_'+suffix] = pair['s_dir_per_state_rms']
    return result


def associations(rows):
    result = []
    for t in TIMES:
        part = [r for r in rows if r['t'] == t]
        if len(part) != 6 or len({r['case'] for r in part}) != 6:
            raise ValueError('Need all six fixed cases at each original snapshot')
        for predictor in ('omega3_cached', 'direction_path_secant3', 'state_speed3', 'incoming_q_abs_cosine'):
            for suffix, eta in (('01', .01), ('02', .02)):
                target = 'kappa_dir_'+suffix
                pairs = [r for r in part if r[predictor] is not None and r[target] is not None and
                         math.isfinite(r[predictor]) and math.isfinite(r[target])]
                x, y = [r[predictor] for r in pairs], [r[target] for r in pairs]
                result.append(dict(t=t, eta=eta, predictor=predictor, target='kappa_dir', n=len(pairs),
                                   image_clusters=len({r['image_id'] for r in pairs}),
                                   spearman=pearson(ranks(x), ranks(y))))
    return result


def hypothesis_readout(links):
    selected = [r for r in links if r['predictor'] == 'direction_path_secant3']
    values = [r['spearman'] for r in selected]
    if len(values) != 10 or any(v is None for v in values) or any(r['n'] != 6 or r['image_clusters'] != 3 for r in selected):
        return 'INCONCLUSIVE_ORDERING'
    return 'EXPLORATORY_POSITIVE_ORDERING_ONLY' if all(v > 0 for v in values) else 'UNIFORM_POSITIVE_PROXY_NOT_SUPPORTED'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-dir', required=True)
    parser.add_argument('--output-root', help='New analysis directories only; must be outside the A2a cohort')
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    root = Path(args.plan_dir).resolve()
    parent_summary, active = archive.audit_source(root)
    registry = json.loads((root/'recovery_registry.json').read_text(encoding='utf-8'))
    files = {str(p): digest(p) for p in (root/'plan.json', root/'recovery_registry.json', root/'recovery_summary.json',
                                        CARD, Path(__file__), Path(archive.__file__))}
    sources = {}
    for key, record in active.items():
        folder = root/registry['active'][key]['directory'] if key in registry['active'] else root/key
        sources[key] = folder
        for name in ('run.json', 'trajectory.pt'):
            files[str(folder/name)] = digest(folder/name)
        if tuple(p['t'] for p in record['probes']) != TIMES:
            raise ValueError('Original five snapshots differ')
    print('Parent AUDIT PASS: six active cases; historical method/diagnostic=',
          parent_summary['method_nfe'], parent_summary['diagnostic_nfe'], flush=True)
    if args.inspect:
        print('Plan: 6 cases x 5 original snapshots; CPU tensor readout only, no model or GT; new UNet NFE=0.', flush=True)
        return
    output_root = Path(args.output_root).resolve() if args.output_root else root.parent/'diag_cached_guidance_state_path'
    if output_root.is_relative_to(root):
        raise ValueError('New analysis output must be outside the frozen cohort')
    directory = output_root/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    directory.mkdir(parents=True, exist_ok=False)
    card = CARD.read_text(encoding='utf-8')
    (directory/'experiment_card.md').write_text(card, encoding='utf-8')
    record = dict(experiment='cached_guidance_state_path', experiment_id='A2b0', status='running', method_nfe=0, diagnostic_nfe=0,
                  parent=str(root), source_sha256=files, experiment_card=card, arguments=vars(args),
                  git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECTS, text=True).strip(),
                  git_tracked_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=PROJECTS, text=True)),
                  parent_actual_method_nfe=parent_summary['method_nfe'], parent_actual_diagnostic_nfe=parent_summary['diagnostic_nfe'],
                  source_final_metrics={key: r['final_metrics'] for key, r in active.items()},
                  resolved_config=dict(times=list(TIMES), etas=[.01, .02], state_arc_floor=1e-12,
                                       omega_relative_tolerance=1e-6, omega_absolute_tolerance=1e-8,
                                       parent_config=next(iter(active.values()))['config']),
                  gt_loaded=False, model_loaded=False, new_trajectory=False, statistical_unit='three image clusters',
                  rows=[], gates={}, limits='Post-hoc exploratory decomposition. Time/D/random-history mixing remains. '
                  'No input sensitivity, controller benefit, quality gain or independent validation claim.')
    started = time.perf_counter()
    def save():
        record['seconds'] = time.perf_counter()-started
        archive.recovery.write_json(directory/'run.json', record)
    save()
    print('Records:', directory, flush=True)
    try:
        import torch
        record['runtime'] = dict(python=sys.version, torch=str(torch.__version__), device='cpu')
        before_rng = torch.get_rng_state().clone()
        for key, source in active.items():
            cache = torch.load(sources[key]/'trajectory.pt', map_location='cpu', weights_only=True)
            q, x = cache['guidance'], cache['latent_inputs']
            if any(tuple(v.shape) != (30, 3, 256, 256) or v.dtype != torch.float32 or
                   not torch.isfinite(v).all() for v in (q, x)):
                raise ValueError('Unexpected cached state/direction tensor contract')
            for probe in source['probes']:
                row = point_readout(q, x, source['rows'], probe)
                row.update(case=key, image_id=source['case']['image_id'], sampling_seed=source['case']['sampling_seed'])
                record['rows'].append(row)
                print('POINT:', json.dumps(row), flush=True)
            del cache, q, x
        if not torch.equal(before_rng, torch.get_rng_state()):
            raise ValueError('Read-only CPU geometry changed the RNG')
        if any(digest(name) != sha for name, sha in files.items()):
            raise ValueError('Source changed during cached analysis')
        links = associations(record['rows'])
        baseline = {(r['t'], r['eta']): r['spearman'] for r in parent_summary['same_t_descriptive_associations']
                    if r['predictor'] == 'guidance_variation3_per_u' and r['target'] == 's_dir_per_state_rms'}
        if any(r['spearman'] != baseline[(r['t'], r['eta'])] for r in links if r['predictor'] == 'omega3_cached'):
            raise ValueError('Original matched-t association did not reproduce')
        record['gates'] = dict(source_unchanged=dict(passed=True), cpu_rng=dict(passed=True),
                               original_omega_associations=dict(passed=True),
                               factorization=dict(passed=all(r['factorization_error'] is None or
                                   r['factorization_error'] <= 1e-10*max(1., abs(r['omega3_recomputed'])) for r in record['rows'])))
        if not all(g['passed'] for g in record['gates'].values()):
            raise ValueError('State-path decomposition identity gate failed')
        record['summary'] = dict(points=len(record['rows']), matched_t_associations=links,
                                 hypothesis_readout=hypothesis_readout(links),
                                 unresolved_state_arcs=sum(r['direction_path_secant3'] is None for r in record['rows']))
        with (directory/'points.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(record['rows'][0]))
            writer.writeheader()
            writer.writerows(record['rows'])
        record['status'] = 'complete'
        save()
        print('Gates:', json.dumps(record['gates']), flush=True)
        print('Summary:', json.dumps(record['summary']), flush=True)
        print('New method/diagnostic NFE=0/0; CPU tensor analysis only; no trajectory or control change.', flush=True)
    except BaseException as error:
        record['status'] = 'failed'
        record['error'] = repr(error)
        save()
        raise


if __name__ == '__main__':
    main()
