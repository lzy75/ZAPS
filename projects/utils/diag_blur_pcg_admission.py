"""A58 original-target structural PCG admission, then conditionally four-arm quality."""
import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import shutil
from statistics import mean
import sys
import time

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS)); sys.path.insert(0, str(PROJECTS.parent))
from utils import diag_current_score_geometry_r2 as prior
from utils import diag_score_cg_metrology as metrology
from utils import diag_no_learning_replay as transport
from utils import blur_pcg as rules
from utils import angular_feedback_control as current

parent, source, engine = prior.parent, prior.source, transport.engine
signed = prior.signed
ROOT = '/data0/lzy/ZAPS-results/diag_blur_pcg_admission/A58_20261027'
CARD = PROJECTS/'experiment_cards/A58_blur_pcg_admission_20261027.md'
BRANCH = 'results/blur-pcg-admission-20261027'
ARMS = ('C_IEEE', 'OLD_U', 'U_PCG', 'V_PCG')
TASKS = ('gaussian_deblur', 'motion_deblur')


def require(value, message):
    if not value:
        raise ValueError(message)


def check(root):
    p = source.check_plan(root); require(p['experiment'] == 'A58_blur_pcg_admission', 'Wrong plan')
    parent.a42.verify(p); return p


def prepare(args):
    root, before_root = Path(args.root).resolve(), Path(args.a50r2_dir).resolve()
    require(not (root.is_relative_to(before_root) or before_root.is_relative_to(root)), 'Source/output overlap')
    if root.exists():
        p = check(root); require(p['source_A50r2'] == str(before_root), 'Frozen source differs')
        print('A58 frozen plan reused; prior outputs retained.', flush=True); return
    before = prior.check(before_root); spec = metrology.source_spec(before_root)
    cases = [c for c in before['cases'] if c['task'] in TASKS]
    require(len(cases) == 10, 'Need five original images x two blur tasks')
    identities = dict(before['identity_files']); references = {}
    for c in cases:
        found = prior.phase_done(before_root, before, c, 'generate')
        require(found is not None, 'Need complete source generate phase')
        f, r = found; references[c['key']] = str(f)
        identities[str(f)] = source.digest(f)
        for n, sha in r['artifacts'].items():
            identities[str(f.parent/n)] = sha
    own = [Path(__file__), Path(rules.__file__), CARD, PROJECTS/'run_a58_20261027.sh',
           PROJECTS/'utils/a58_session.py', PROJECTS/'utils/test_blur_pcg.py', Path(transport.__file__)]
    p = dict(before)
    p.update(experiment='A58_blur_pcg_admission', source_A50r2=str(before_root), cases=cases,
        metrology_spec=spec, source_generate_records=references, identity_files={**identities, **spec['identities']},
        worker_identity_files={**before['worker_identity_files'], **{str(f): source.digest(f) for f in own}},
        phases=['control'], arms=list(ARMS), planned_method_nfe=0, planned_diagnostic_nfe=1200,
        planned_backward_attempts_max=600, metrology_solves=24, metrology_seed=580027,
        limits='Five OLD FFHQ images x two blur tasks; one seed and known sigma=.05. '
               'Zero-boundary original coefficient matrix; promoted float64 arithmetic and periodic SPD preconditioner only. '
               '128 iterations and true-residual1e-5 unchanged; extra FFT/operator actions are charged separately. '
               'Synthetic RHS admission is not full trajectory admission. Unresolved trajectory steps retain original-q fallback. '
               'No unseen-image, exact posterior, curvature, sampling-grid or walltime-saving claim.')
    root.parent.mkdir(parents=True, exist_ok=True); parent.P.coupled.archive.recovery.require_space(root.parent)
    root.mkdir(); signed(root/'plan.json', p); shutil.copyfile(CARD, root/'experiment_card.md')
    print('A58 PREPARE PASS:24 zero-NN solves; only all8 PCG admissions enable <=1200 forwards/600 input VJPs.', flush=True)


def run_metrology(root, p, device):
    import torch
    from modules.degradations import GaussianBlurOperator
    from utils.diag_ffhq_four_tasks_unit import motion_operator
    from utils.current_score_geometry import conjugate_gradient
    path = root/'metrology.json'
    if path.exists():
        r = source.load_signed(path); require(r['status'] == 'complete', 'Interrupted metrology retained; no auto retry')
        require(r['plan_sha256'] == source.digest(root/'plan.json'), 'Metrology plan differs'); return r
    torch.set_num_threads(2)
    r = dict(status='running', plan_sha256=source.digest(root/'plan.json'), rows=[], method_nfe=0, diagnostic_nfe=0)
    signed(path, r)
    try:
        generator = torch.Generator(device='cpu').manual_seed(p['metrology_seed'])
        draw = torch.randn(p['metrology_spec']['shape'], generator=generator, dtype=torch.float32).to(device)
        kernel = torch.load(p['metrology_spec']['kernel'], map_location='cpu', weights_only=True).numpy()
        with parent.P.precision.ieee(torch), torch.no_grad():
            for task in TASKS:
                base = GaussianBlurOperator(61, 3., 0.).to(device) if task == 'gaussian_deblur' else motion_operator(kernel, device)
                double = copy.deepcopy(base).double()
                for state in p['metrology_spec']['eta'][task]:
                    eta = state['eta']; inverse = rules.periodic_inverse(double.kernel, draw.shape, eta)
                    for kind in ('WHITE', 'IN_OPERATOR_RANGE'):
                        rhs = draw if kind == 'WHITE' else .5*base.H(draw)
                        for variant in ('LEGACY32', 'CG64', 'PCG64'):
                            op = base if variant == 'LEGACY32' else double
                            b = rhs if variant == 'LEGACY32' else rhs.double()
                            action = lambda x, op=op: .25*op.H(op.transpose(x))+eta*x
                            torch.cuda.synchronize(device); started = time.perf_counter()
                            if variant == 'PCG64':
                                _, gate = rules.preconditioned_cg(action, b, inverse, rtol=1e-5, maxiter=128)
                            else:
                                _, gate = conjugate_gradient(action, b, rtol=1e-5, maxiter=128)
                            torch.cuda.synchronize(device)
                            r['rows'].append(dict(task=task, **state, rhs=kind, variant=variant, gate=gate,
                                                  seconds=time.perf_counter()-started))
                            signed(path, r)
                            print('A58 numerical:', len(r['rows']), '/24', task, state['t'], kind, variant,
                                  'passed=', gate['passed'], flush=True)
        r.update(status='complete', admitted=all(v['gate']['passed'] for v in r['rows'] if v['variant'] == 'PCG64'))
        require(len(r['rows']) == 24, 'Missing metrology solves')
    except BaseException as error:
        r.update(status='failed', error=repr(error)); signed(path, r); raise
    signed(path, r); return r


def admitted(root):
    r = source.load_signed(root/'metrology.json')
    require(r['status'] == 'complete' and len(r['rows']) == 24 and
            r['plan_sha256'] == source.digest(root/'plan.json'), 'Need complete matching numerical admission')
    return r['admitted']


def item(root, p, c, phase):
    files = sorted((root/c['key']).glob('control_*/run.json'))
    if not files:
        return None
    require(len(files) == 1, 'Multiple attempts; no selection')
    f = files[0]; r = source.read(f)
    if r['status'] != 'complete':
        return None
    r = source.load_signed(f)
    require(r['case'] == c and r['plan_sha256'] == source.digest(root/'plan.json') and
            r['metrology_sha256'] == source.digest(root/'metrology.json') and
            r['calls'] == list(reversed(p['grid']))*4 and r['nfe'] == 120 and
            r['backward_attempts'] == r['backward_completed'] <= 60 and
            all(g['passed'] for g in r['gates'].values()), 'Control contract differs')
    require(all(source.digest(f.parent/n) == sha for n, sha in r['artifacts'].items()), 'Outputs changed')
    return f, r, 'U_PCG'


def pcg_field(z, base64, y, x, t, s, positions, sigma, mode, backward):
    import torch
    base, algebra = current.base, current.algebra
    inp, raw = base.raw_mean(z.dm, x, t, graph=True)
    field = base.field_from_mean(z, y, raw, t, positions)
    with torch.no_grad():
        residual = y-z.A.H(raw.detach())
        row = dict(backward_calls=0, R_raw=base.informative_rms(residual)/sigma,
                   clip_fraction=float((raw.detach().abs()>1).float().mean()))
    v, gate = rules.proximal_vector(base64, residual, float(z.dm.alphas_cumprod[t]), sigma)
    row['cg'] = gate
    if v is None:
        row.update(resolved=False, reason='PCG_UNRESOLVED_ORIGINAL_Q_FALLBACK'); return field, 1., row
    coefficient = algebra.skip_coefficient(float(z.dm.alphas_cumprod[t]), float(z.dm.alphas_cumprod[s]) if s >= 0 else 1.)
    if mode == 'U_PCG':
        backward(); row['backward_calls'] = 1; u = algebra.input_vjp(raw, inp, v)
    else:
        u = None
    reading = base.gram_readout(field['q'], v, u, coefficient, float(x.detach().double().square().mean().sqrt()))
    row.update(reading, coefficient=coefficient)
    if not reading['resolved']:
        return field, 1., row
    if mode == 'U_PCG':
        target, extra = current.controlled_target(field['q'], v, u, coefficient, reading, 1.)
        row.update(extra)
    else:
        target = coefficient*v
        row.update(target_RMS=float(target.double().square().mean().sqrt()), current_Jacobian_used=False)
    return dict(field, q=target.detach()), 1., row


def sample(root, p, c, phase, model, device):
    import torch
    from unittest.mock import patch
    from utils.diag_zaps_sampler_parity import rng_state
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_trace_audit import relative_error
    require(admitted(root), 'Numerical admission failed: NN phase prohibited')
    if item(root, p, c, phase):
        return
    operator, mask, opgate, state, original, y, z, positions, noises = parent.setup(p, c, model, device)
    base64 = copy.deepcopy(operator.base).double()
    folder, r = prior.new_attempt(root, p, c, phase); path = folder/'run.json'
    r['metrology_sha256'] = source.digest(root/'metrology.json'); external = rng_state(device); outputs = {}
    with prior.counted(model, path, r, 120) as backward, current.base.input_vjp_parameters(model.model), \
         model_rng_guard(model.model, device), patch('torch.randn_like', side_effect=RuntimeError('Unexpected sampling RNG')):
        for mode in ARMS:
            x = state['init_noise'].to(device); r['traces'][mode] = []
            torch.cuda.synchronize(device); started = time.perf_counter()
            with parent.P.precision.ieee(torch):
                for k, t in enumerate(positions.nominal):
                    s = positions.nominal[k+1] if k < 29 else -1
                    if mode == 'C_IEEE':
                        _, raw = current.base.raw_mean(z.dm, x, t, graph=True)
                        field = current.base.field_from_mean(z, y, raw, t, positions); gamma = 1.; row = dict(backward_calls=0)
                    elif mode == 'OLD_U':
                        field, gamma, row = current.sensed_field(z, y, x, t, s, positions, c['sigma'], 1., None, backward)
                    else:
                        field, gamma, row = pcg_field(z, base64, y, x, t, s, positions, c['sigma'], mode, backward)
                    r['backward_completed'] += row['backward_calls']; r['traces'][mode].append(dict(step=k, t=t, readout=row))
                    with torch.no_grad():
                        x = parent.old.parent.rules.risk.advance(z, x, field, positions, k, noises, gamma)
                    require(bool(torch.isfinite(x).all()), 'Nonfinite control:'+mode)
            torch.cuda.synchronize(device); r['arm_seconds'][mode] = time.perf_counter()-started
            outputs[mode] = x.detach().cpu()
        parent.invariant(model, z, state, device, external, p)
        ref = Path(p['source_generate_records'][c['key']])
        cached = torch.load(ref.parent/'outputs.pt', map_location='cpu', weights_only=True)['C_IEEE']
        parity = relative_error(outputs['C_IEEE'], cached)
        require(parity <= 1e-6, 'Source C_IEEE replay parity failed')
        parent.old.parent.base.save_tensor(outputs, folder/'outputs.pt')
        signed(folder/'decisions_before_labels.json', dict(case=c, traces=r['traces'], GT_used=False))
        r['gates'] = dict(operator=opgate, calls_rng_weights=dict(passed=True), source_C_parity=dict(passed=True, relative_error=parity))
        r['artifacts'] = {n: source.digest(folder/n) for n in ('outputs.pt', 'decisions_before_labels.json')}
    r['status'] = 'complete'; signed(path, r)
    print('A58 control:', c['key'], 'F=', r['nfe'], 'B=', r['backward_attempts'], flush=True)


@contextmanager
def executor_bindings():
    from unittest.mock import patch
    with patch.object(engine, 'check', check), patch.object(engine, 'item', item), patch.object(engine, 'sample', sample):
        yield


def summarize(root, p):
    r = source.load_signed(root/'metrology.json'); require(r['status'] == 'complete', 'Metrology incomplete')
    groups = []; rows = []; values = []
    if r['admitted']:
        import torch
        from modules.main_single import load_image_as_tensor
        from utils.metrics import compute_all_metrics
        pairs = [(c, item(root, p, c, 'control')) for c in p['cases']]
        require(all(v for c, v in pairs), 'Need all outputs frozen before GT')
        with parent.scan.preserved_rng(), torch.no_grad():
            for c, (f, v, _) in pairs:
                outputs = torch.load(f.parent/'outputs.pt', map_location='cpu', weights_only=True)
                gt = load_image_as_tensor(c['image']).cpu()
                metrics = {n: compute_all_metrics(x, gt, lpips_net='alex') for n, x in outputs.items()}
                rows.append(dict(case=c, metrics=metrics, resolved={n:sum(q['readout'].get('resolved', False) for q in v['traces'][n]) for n in ARMS}))
                values.append(v)
        for task in TASKS:
            subset = [v for v in rows if v['case']['task'] == task]
            delta = {n:{m:mean(v['metrics']['U_PCG'][m]-v['metrics'][n][m] for v in subset)
                        for m in ('psnr','ssim','lpips')} for n in ('C_IEEE','OLD_U','V_PCG')}
            wins = {n:sum(v['metrics']['U_PCG']['psnr'] > v['metrics'][n]['psnr']+1e-9 for v in subset) for n in delta}
            q = delta['C_IEEE']; candidate = q['psnr'] >= .05 and q['ssim'] >= -.002 and q['lpips'] <= .005 and wins['C_IEEE'] >= 4
            groups.append(dict(task=task, U_PCG_minus=delta, images_positive=wins, quality_vs_C_development_candidate=bool(candidate)))
    result = dict(experiment=p['experiment'], classification='BLUR_PCG_DEVELOPMENT_QUALITY_ONLY' if r['admitted'] else 'NUMERICAL_ADMISSION_FAILED_NN_NOT_RUN',
        numerical_admitted=r['admitted'], completed_solves=24, completed_cases=len(rows), planned_cases=10,
        method_nfe=0, diagnostic_nfe=sum(v['nfe'] for v in values), input_backward_attempts=sum(v['backward_attempts'] for v in values),
        metrology_operator_calls=sum(v['gate']['operator_calls'] for v in r['rows']),
        metrology_preconditioner_calls=sum(v['gate'].get('preconditioner_calls',0) for v in r['rows']),
        metrology_spectral_setups=4,
        quality_solve_actions={arm:{name:sum(q['readout'].get('cg',{}).get(name,0)
             for value in values for q in value['traces'][arm])
             for name in ('linear_calls','adjoint_calls','preconditioner_calls','spectral_setup_ffts','fft_transforms')} for arm in ARMS},
        measured_quality_seconds_by_arm={arm:sum(v['arm_seconds'][arm] for v in values) for arm in ARMS},
        numerical_rows=r['rows'], groups=groups, per_image=rows, limits=p['limits'])
    signed(root/'summary.json', result); print('A58 Summary:', json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    print('Records:', root, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=ROOT); parser.add_argument('--a50r2-dir', default=prior.ROOT)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--metrology', action='store_true')
    parser.add_argument('--phase', choices=('control',), default='control'); parser.add_argument('--shard', type=int, choices=(0,1))
    parser.add_argument('--device', default='cuda:0'); parser.add_argument('--archive', action='store_true'); args = parser.parse_args()
    if args.prepare:
        prepare(args); return
    root = Path(args.root).resolve(); p = check(root)
    if args.metrology:
        r = run_metrology(root, p, args.device); print('A58 NUMERICAL ADMISSION:', r['admitted'], '; NN costs so far=0', flush=True); return
    if args.shard is not None:
        with executor_bindings():
            engine.worker(args)
        return
    summarize(root, p)
    if args.archive:
        from unittest.mock import patch
        with patch.object(engine.old.previous, 'BRANCH', BRANCH):
            engine.old.previous.archive(root)


if __name__ == '__main__':
    main()
