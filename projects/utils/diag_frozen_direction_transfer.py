"""A56 frozen current-direction rule: fresh FFHQ/ImageNet x SR/inpainting."""
import argparse
from contextlib import contextmanager, nullcontext
import gc
import json
import math
import os
from pathlib import Path
import shutil
from statistics import mean
import sys
import time

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS)); sys.path.insert(0, str(PROJECTS.parent))
from utils import diag_current_direction_value_r2 as previous
from utils import diag_multitask_state_transfer as a40
from utils import frozen_direction_transfer as rules
from utils import angular_feedback_control as current
from utils import norm_gain_control as time_rule
from utils import current_direction_control as direction
from utils.linear_adjoint_autograd_guard import operator_only_autograd

source, parent = previous.source, previous.parent
write, signed = previous.write, previous.signed
ROOT = '/data0/lzy/ZAPS-results/diag_frozen_direction_transfer/A56_20261026'
CARD = PROJECTS/'experiment_cards/A56_frozen_direction_transfer_20261026.md'
BRANCH = 'results/frozen-direction-transfer-20261026'


def require(test, message):
    if not test:
        raise ValueError(message)


def history_selection(data, root):
    """One pre-label scan shared by both datasets; paths and content hashes excluded."""
    paths = set()
    for base in (PROJECTS/'results', PROJECTS/'server_records', Path('/data0/lzy/ZAPS-results')):
        if base.exists():
            for pattern in ('run.json', 'plan.json'):
                paths.update(p.resolve() for p in base.rglob(pattern) if not p.resolve().is_relative_to(root))
    used_names = {d: set() for d in rules.DATASETS}; used_sha = set(); audit = []
    def visit(v):
        if isinstance(v, dict):
            if isinstance(v.get('image_sha256'), str):
                used_sha.add(v['image_sha256'])
            if isinstance(v.get('image'), str):
                image = Path(v['image']).resolve()
                for d, folder in data.items():
                    if image.parent == folder:
                        used_names[d].add(image.stem)
            for key, item in v.items():
                if key not in ('experiment_card', 'limits', 'scope'):
                    visit(item)
        elif isinstance(v, list):
            for item in v:
                visit(item)
    for path in sorted(paths):
        visit(source.read(path)); audit.append(dict(path=str(path), sha256=source.digest(path)))
    require(bool(audit), 'No accessible history; cannot identify unused images')
    selected = {}; pools = {}
    for d, folder in data.items():
        pool = [dict(image=str(f.resolve()), image_id=f.stem, image_sha256=source.digest(f))
                for f in rules.eligible_paths(folder) if f.stem not in used_names[d]]
        selected[d] = rules.select(pool, used_names[d], used_sha, rules.COUNTS[d], 20261026+rules.DATASETS.index(d))
        pools[d] = dict(directory=str(folder), eligible_files=len(pool), selected=selected[d])
        used_sha.update(c['image_sha256'] for c in selected[d])
    return selected, dict(history=audit, pools=pools,
        scope='Unused within accessible server run.json AND plan.json only, including failed/planned cases; '
              'content duplicates excluded; fixed sampling seed before any new outcomes, not globally unseen dataset guarantee.')


def prepare(args):
    root, prior = Path(args.root).resolve(), Path(args.a55r2_dir).resolve()
    require(not (root.is_relative_to(prior) or prior.is_relative_to(root)), 'Source/output overlap')
    if root.exists():
        p = check(root); print('A56 frozen plan reused; no re-selection.', flush=True); return
    before = previous.check(prior); report = source.load_signed(prior/'summary.json')
    require(report['completed_cases'] == report['planned_cases'] == 10 and
            report['classification'] == 'DEVELOPMENT_PULLBACK_DIRECTION_CANDIDATE', 'Need completed direction diagnosis')
    identities = {str(prior/name): source.digest(prior/name) for name in ('plan.json', 'summary.json')}
    curves = {}
    for c in before['cases']:
        found = previous.item(prior, before, c, 'control'); require(found is not None, 'Incomplete A55r2 source')
        f, r, _ = found; identities[str(f)] = source.digest(f)
        sf = Path(before['norm_sources'][c['key']]); sr = source.load_signed(sf)
        require(sr['status'] == 'complete' and sr['case'] == c, 'Old U source mismatch')
        curves[c['key']] = [row['readout']['v_RMS']/row['readout']['u_RMS'] for row in sr['traces']['NORM_ONLY']]
        identities[str(sf)] = source.digest(sf)
    profiles = rules.frozen_profiles(before['cases'], curves)
    require([list(reversed(before['grid']))[k] for k in rules.METROLOGY_STEPS] == [319, 14], 'Metrology source grid differs')
    p41_path = Path(before['source_A41'])/'plan.json'; p41 = source.load_signed(p41_path)
    p40_path = Path(p41['source_A40'])/'plan.json'; p40 = source.load_signed(p40_path)
    for f in (p41_path, p40_path):
        identities[str(f)] = source.digest(f)
    configs = dict(ffhq={t: dict(p41['configs']['ffhq'][t]) for t in rules.TASKS},
        imagenet=dict(super_resolution=dict(p40['configs']['imagenet']['super_resolution']),
                      inpainting=dict(p40['configs']['imagenet']['gaussian_deblur'], zeta_init=.1)))
    require(all(configs[d][t]['lr'] == (.01 if d == 'ffhq' else .001) for d in configs for t in rules.TASKS), 'Baseline LR changed')
    checkpoints = p40['checkpoints']
    for spec in checkpoints.values():
        require(source.digest(spec['path']) == spec['sha256'], 'Checkpoint missing/changed')
        identities[spec['path']] = spec['sha256']
    data = dict(ffhq=Path(args.ffhq_dir).resolve(), imagenet=Path(args.imagenet_dir).resolve())
    require(all(f.is_dir() for f in data.values()), 'Both datasets must exist before freezing')
    selected, audit = history_selection(data, root)
    cases = []
    for d in rules.DATASETS:
        for task in rules.TASKS:
            for i, image in enumerate(selected[d]):
                cases.append(dict(image, dataset=d, task=task, image_index=i, sigma=.05, sampling_seed=1000,
                    measurement_seed=(1000 if task == 'super_resolution' else 201000)+int(image['image_id']),
                    mask_seed=2001000+int(image['image_id']), key=f'{d}_{task}_{image["image_id"]}_s1000'))
                identities[image['image']] = image['image_sha256']
    metrology = [next(c['key'] for c in cases if c['dataset'] == d and c['task'] == 'super_resolution') for d in rules.DATASETS]
    own = [Path(__file__), Path(rules.__file__), CARD, PROJECTS/'run_a56_20261026.sh', PROJECTS/'utils/a56_session.py',
           PROJECTS/'utils/test_frozen_direction_transfer.py', Path(a40.__file__), PROJECTS/'modules/degradations.py',
           PROJECTS/'modules/diffusion_model.py', PROJECTS/'modules/zaps_algorithm.py',
           PROJECTS/'utils/diag_ffhq_four_tasks_unit.py', PROJECTS/'utils/metrics.py']
    p = dict(experiment='A56_frozen_direction_transfer', source_A55r2=str(prior), cases=cases, selected_images=selected,
        profiles=profiles, profiles_GT_used=False, configs=configs, checkpoints=checkpoints,
        grid=before['grid'], runtime=before['runtime'], native_policy=before['native_policy'],
        model_library_sources=before['model_library_sources'], metrology_cases=metrology,
        snapshot_cases=[c['key'] for c in cases if c['dataset'] == 'ffhq' and c['image_index'] < 2],
        identity_files=identities,
        worker_identity_files={**before['worker_identity_files'], **{str(f): source.digest(f) for f in own}},
        planned_method_nfe=6000, planned_diagnostic_nfe=3624, planned_total_forward_nfe=9624,
        planned_input_backward_max=1204, scientific_forward_nfe=300,
        baseline_optimizer_updates=200,
        limits='Six unused accessible-history FFHQ images and four ImageNet images x SR/IP, one seed, known sigma=.05. '
               'Frozen NORM_ONLY direction rule, no target-data gain fitting or output selection; TIME uses FFHQ OLD gains. '
               'Original first270/last30 grid unchanged. Separate matched native and IEEE local C; not author benchmark parity. '
               'ImageNet IP is a task extension with fixed zeta=.1/LR=.001; model and data both change. '
               'No four-FFHQ-task/trajectory-curvature/h/NFE-saving proof. V/Q remain current-state-dependent controls. '
               '300 is the inherited charged protocol, not proof U needs optimization: at fully resolved steps its target '
               'has no learned zeta/D dependence. A standalone no-learning replay is not yet confirmed. '
               '300 forwards is not equal backward/operator/time cost. Numerical failures retained.')
    root.parent.mkdir(parents=True, exist_ok=True); parent.P.coupled.archive.recovery.require_space(root.parent)
    root.mkdir(); signed(root/'selection_audit.json', audit)
    p['identity_files'][str(root/'selection_audit.json')] = source.digest(root/'selection_audit.json')
    signed(root/'plan.json', p); shutil.copyfile(CARD, root/'experiment_card.md')
    signed(root/'frozen_profiles.json', dict(profiles=profiles, GT_used=False, source=str(prior), scope='Old FFHQ only'))
    print('A56 PREPARE/FREEZE PASS:', {d: [c['image_id'] for c in images] for d, images in selected.items()},
          ';20 cases;6000 method +3624 diagnostic forwards;<=1204 input backwards. No tuning or grid changes.', flush=True)


def check(root):
    p = source.check_plan(root); require(p['experiment'] == 'A56_frozen_direction_transfer', 'Wrong plan')
    parent.a42.verify(p); return p


def strict_attempt(root, c, kind):
    folder = root/c['key']/kind; folder.mkdir(parents=True, exist_ok=True)
    require(not list(folder.glob('attempt_*')), 'Old failed/interrupted attempt retained; no automatic budget reset: '+str(folder))
    result = folder/'attempt_000'; result.mkdir(); return result


def baseline_done(root, p, c, kind):
    require(kind == 'BASELINE', 'Only baseline transport reused')
    files = sorted((root/c['key']/kind).glob('attempt_*/run.json'))
    if not files:
        return None
    require(len(files) == 1, 'Multiple training attempts; no selection')
    f = files[0]; r = source.read(f)
    if r['status'] != 'complete':
        return None
    r = source.load_signed(f)
    require(r['case'] == c and r['plan_sha256'] == source.digest(root/'plan.json') and
            r['nfe'] == 300 and r['calls'] == list(reversed(p['grid']))*10 and
            all(g['passed'] for g in r['gates'].values()), 'Baseline contract failed')
    require(all(source.digest(f.parent/n) == sha for n, sha in r['artifacts'].items()) and
            source.digest(r['measurement']['path']) == r['measurement']['sha256'], 'Baseline bytes changed')
    return f, r


@contextmanager
def baseline_bindings():
    from unittest.mock import patch
    with patch.object(a40, 'done', baseline_done), patch.object(a40, 'attempt', strict_attempt):
        yield


def make_operator(c, device):
    import torch
    from utils.diag_ffhq_four_tasks_unit import make_base, unit_operator
    shape_only = torch.empty(1, 3, 256, 256, device=device)
    raw = make_base(c['task'], device, shape_only, c['mask_seed'], None)
    return unit_operator(raw).to(device), raw._mask_cache if c['task'] == 'inpainting' else None


def finished(root, p, c):
    files = sorted((root/c['key']/'SAMPLING').glob('attempt_*/run.json'))
    if not files:
        return None
    require(len(files) == 1, 'Multiple sampling attempts')
    f = files[0]; r = source.read(f)
    if r['status'] != 'complete':
        return None
    r = source.load_signed(f); trained = baseline_done(root, p, c, 'BASELINE')
    require(trained is not None and r['training_sha256'] == source.digest(trained[0]) and
        r['case'] == c and r['plan_sha256'] == source.digest(root/'plan.json') and
        r['calls'] == rules.expected_calls(p['grid'], c['key'] in p['metrology_cases']) and
        r['nfe'] == len(r['calls']) and r['backward_attempts'] == r['backward_completed'] and
        set(r['traces']) == set(rules.ARMS) and all(g['passed'] for g in r['gates'].values()), 'Sampling gate failed')
    require(r['backward_attempts'] <= 60+(2 if c['key'] in p['metrology_cases'] else 0) and
            len(r['metrology']) == (2 if c['key'] in p['metrology_cases'] else 0) and
            all(len(rows) == 30 for rows in r['traces'].values()), 'Probe/backward/trace contract differs')
    require(all(source.digest(f.parent/n) == sha for n, sha in r['artifacts'].items()), 'Sampling artifact changed')
    return f, r


def sample(root, p, c, model, device, operator, mask):
    import torch
    from unittest.mock import patch
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state
    from utils.diag_zaps_trace_audit import relative_error
    if finished(root, p, c):
        return
    with baseline_bindings():
        trained, state, original, y, meta, z, positions, noises = a40.setup_sampling(root, p, c, model, device, operator)
    folder = strict_attempt(root, c, 'SAMPLING'); path = folder/'run.json'
    r = dict(case=c, phase='SAMPLING', plan_sha256=source.digest(root/'plan.json'),
             training_sha256=source.digest(trained[0]), gates={}, traces={}, backward_attempts=0, backward_completed=0,
             arm_seconds={}, arm_peak_memory_bytes={}, metrology=[])
    external = rng_state(device); outputs = {}; snapshots = {}; do_probe = c['key'] in p['metrology_cases']
    with previous.engine.old.previous.counted(model, path, r, len(rules.expected_calls(p['grid'], do_probe))) as backward, \
         current.base.input_vjp_parameters(model.model), model_rng_guard(model.model, device), \
         patch('torch.randn_like', side_effect=RuntimeError('Unexpected random draw')):
        for name in rules.ARMS:
            torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device); start = time.perf_counter()
            x = state['init_noise'].to(device); r['traces'][name] = []
            scope = nullcontext() if name == 'C_REPLAY' else parent.P.precision.ieee(torch)
            opguard = operator_only_autograd(r.setdefault('operator_autograd_accounting', {})) if name in direction.ARMS else nullcontext()
            with scope, opguard:
                for k, t in enumerate(positions.nominal):
                    s = positions.nominal[k+1] if k < 29 else -1
                    if name == 'C_REPLAY':
                        with torch.no_grad():
                            field = source.fields.field(z, y, x, t, positions)
                        gamma, row = 1., dict(backward_calls=0)
                    elif name == 'C_IEEE':
                        field, gamma, row = current.base.sensed_field(z, y, x, t, s, positions, c['sigma'], name, mask, backward)
                    elif name == 'U_CURRENT':
                        field, gamma, row = current.sensed_field(z, y, x, t, s, positions, c['sigma'], 1., mask, backward)
                    elif name == 'TIME_GAIN':
                        field, gamma, row = time_rule.sensed_field(z, y, x, t, s, positions, c['sigma'], p['profiles'][c['task']][k], mask, backward)
                    else:
                        field, gamma, row = direction.sensed_field(z, y, x, t, s, positions, c['sigma'], name, mask)
                    r['backward_completed'] += row.get('backward_calls', 0)
                    if name == 'C_IEEE' and do_probe and k in rules.METROLOGY_STEPS:
                        with torch.no_grad():
                            vector = operator.transpose(y-operator.H(field['clean']))
                        torch.cuda.synchronize(device); probe_started = time.perf_counter()
                        gate = current.base.vjp_metrology(model, x, t, vector, backward)
                        torch.cuda.synchronize(device); gate['seconds'] = time.perf_counter()-probe_started
                        r['backward_completed'] += gate['backward_calls']; r['metrology'].append(dict(step=k, t=t, **gate))
                        require(gate['passed'], 'Actual model VJP/grad-mode metrology failed')
                    if name == 'U_CURRENT' and c['key'] in p['snapshot_cases'] and t in (641, 319, 54):
                        snapshots[str(t)] = dict(x=x.detach().cpu(), clean=field['clean'].detach().cpu(),
                            target=field['q'].detach().cpu(), step=k, t=t, readout=dict(row))
                    entry = dict(step=k, t=t, next_t=s, readout=row)
                    if mask is not None:
                        energy = field['q'].detach().double().square(); active = mask.bool().expand_as(energy)
                        total = float(energy.sum())
                        entry['guidance_regions'] = dict(unobserved_energy_fraction=float(energy[~active].sum())/total if total else 0.)
                    r['traces'][name].append(entry)
                    with torch.no_grad():
                        x = parent.old.parent.rules.risk.advance(z, x, field, positions, k, noises, gamma)
                    require(bool(torch.isfinite(x).all()), 'Nonfinite trajectory:'+name)
                outputs[name] = x.detach().cpu()
            torch.cuda.synchronize(device); r['arm_seconds'][name] = time.perf_counter()-start
            r['arm_peak_memory_bytes'][name] = torch.cuda.max_memory_allocated(device)
        error = relative_error(outputs['C_REPLAY'], original)
        require(math.isfinite(error) and error <= 1e-6, 'Original native C replay differs')
        parent.invariant(model, z, state, device, external, p)
        require(r['calls'] == rules.expected_calls(p['grid'], do_probe) and r['backward_attempts'] == r['backward_completed'], 'Forward/backward accounting differs')
        a40.base.save_tensor(outputs, folder/'outputs.pt'); names = ['outputs.pt']
        signed(folder/'decisions_before_labels.json', dict(case=c, traces=r['traces'], GT_used=False, plan_sha256=r['plan_sha256']))
        names.append('decisions_before_labels.json')
        if mask is not None:
            a40.base.save_tensor(mask.detach().cpu().bool(), folder/'mask.pt'); names.append('mask.pt')
        if snapshots:
            a40.base.save_tensor(snapshots, folder/'geometry_states.pt'); names.append('geometry_states.pt')
        r['artifacts'] = {n: source.digest(folder/n) for n in names}
        r['gates'] = dict(native_C=dict(passed=True, relative_error=error),
                         calls_rng_parameters=dict(passed=True), no_GT_used=dict(passed=True),
                         actual_model_VJP=dict(passed=True, probes=len(r['metrology']), reused_if_no_probe='same pinned model/runtime'))
    r['status'] = 'complete'; signed(path, r)
    print('A56 SAMPLING:', c['key'], 'forwards=', r['nfe'], 'input_backwards=', r['backward_attempts'], flush=True)


def worker(args):
    import fcntl
    import torch
    from modules.main_single import load_diffusion_model
    root = Path(args.root).resolve(); p = check(root)
    with (root/f'worker_shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        actual = dict(torch=str(torch.__version__), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
            gpu=torch.cuda.get_device_name(args.device), NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),
            TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        require(actual == p['runtime'] and parent.P.precision.precision_state(torch) == p['native_policy'] and
                a40.model_sources() == p['model_library_sources'], 'Runtime/model libraries changed')
        torch.set_num_threads(2)
        cases = [c for i, c in enumerate(p['cases']) if i % 2 == args.shard]
        for dataset in rules.DATASETS:
            group = [c for c in cases if c['dataset'] == dataset]
            if all(finished(root, p, c) for c in group):
                continue
            with parent.scan.preserved_rng():
                model = load_diffusion_model(dataset, args.device); model.model.eval()
                require(source.digest(model.ckpt_path) == p['checkpoints'][dataset]['sha256'] and model.num_steps == 1000 and
                        all(v.dtype == torch.float32 for v in model.model.parameters()), 'Checkpoint/dtype differs')
                for c in group:
                    if finished(root, p, c):
                        continue
                    operator, mask = make_operator(c, args.device); gate = a40.operator_gate(operator, args.device)
                    with baseline_bindings():
                        a40.training(root, p, c, model, args.device, operator, gate)
                    sample(root, p, c, model, args.device, operator, mask)
                    gc.collect(); torch.cuda.empty_cache()
                del model
            gc.collect(); torch.cuda.empty_cache()


def summarize(root, p):
    import torch
    from modules.main_single import load_image_as_tensor
    from utils.metrics import compute_all_metrics
    items = [(c, finished(root, p, c)) for c in p['cases']]
    require(all(found for c, found in items), 'A56 incomplete: retain all attempts, no quality classification')
    inputs = {str(found[0]): source.digest(found[0]) for c, found in items}
    target = root/'evaluation.json'
    if target.exists():
        evaluation = source.load_signed(target); require(evaluation['inputs'] == inputs, 'Evaluation inputs differ')
    else:
        rows = []
        with parent.scan.preserved_rng(), torch.no_grad():
            for c, (f, r) in items:
                gt = load_image_as_tensor(c['image']).cpu()
                bank = torch.load(f.parent/'outputs.pt', map_location='cpu', weights_only=True)
                metrics = {n: compute_all_metrics(x, gt, lpips_net='alex') for n, x in bank.items()}
                require(all(math.isfinite(float(x)) for m in metrics.values() for x in m.values()), 'Nonfinite quality')
                rows.append(dict(case=c, metrics=metrics,
                    resolved={n: sum(row['readout'].get('resolved', False) for row in r['traces'][n])
                              for n in ('U_CURRENT', 'V_PHYSICAL', 'Q_ORIGINAL', 'TIME_GAIN')}))
        evaluation = dict(inputs=inputs, rows=rows, GT_used_only_after_all20_outputs_frozen=True)
        signed(target, evaluation)
    groups = []
    for dataset in rules.DATASETS:
        for task in rules.TASKS:
            rows = [r for r in evaluation['rows'] if r['case']['dataset'] == dataset and r['case']['task'] == task]
            effects = {n: {ref: {m: mean(r['metrics'][n][m]-r['metrics'][ref][m] for r in rows)
                                for m in ('psnr', 'ssim', 'lpips')}
                           for ref in ('C_REPLAY', 'C_IEEE', 'V_PHYSICAL', 'Q_ORIGINAL', 'TIME_GAIN')}
                       for n in ('U_CURRENT', 'V_PHYSICAL', 'Q_ORIGINAL', 'TIME_GAIN')}
            wins = {ref: sum(r['metrics']['U_CURRENT']['psnr'] > r['metrics'][ref]['psnr']+1e-9 for r in rows)
                    for ref in ('C_IEEE', 'V_PHYSICAL', 'Q_ORIGINAL', 'TIME_GAIN')}
            full = all(r['resolved'][n] == 30 for r in rows for n in r['resolved'])
            def supported(ref):
                d = effects['U_CURRENT'][ref]
                return bool(full and d['psnr'] >= .05 and d['ssim'] >= -.002 and d['lpips'] <= .005 and wins[ref] >= math.ceil(2*len(rows)/3))
            groups.append(dict(dataset=dataset, task=task, image_clusters=len(rows), fully_resolved=bool(full),
                U_minus=effects['U_CURRENT'], all_effects=effects, U_positive_images=wins,
                quality_transfer_candidate=supported('C_IEEE'),
                pullback_direction_transfer_candidate=bool(supported('V_PHYSICAL') and supported('Q_ORIGINAL')),
                per_image=rows))
    training = [source.read(f) for f in root.glob('*/BASELINE/attempt_*/run.json')]
    sampling = [r for c, (f, r) in items]
    result = dict(experiment=p['experiment'], classification='FROZEN_DIRECTION_TRANSFER_IN_'+str(sum(g['pullback_direction_transfer_candidate'] for g in groups))+'_OF_4_CONDITIONS',
        completed_cases=20, planned_cases=20, image_clusters=10, method_nfe=sum(r['nfe'] for r in training),
        diagnostic_nfe=sum(r['nfe'] for r in sampling), input_backward_attempts=sum(r['backward_attempts'] for r in sampling),
        input_backward_completed=sum(r['backward_completed'] for r in sampling), partial_attempts=sum(r['status'] != 'complete' for r in training+sampling),
        groups=groups, planned_total_forward_nfe=9624, scientific_forward_nfe=300,
        recorded_last_unroll_seconds={n: sum(r['arm_seconds'][n] for r in sampling) for n in rules.ARMS},
        metrology_seconds_in_C_IEEE=sum(g['seconds'] for r in sampling for g in r['metrology']),
        baseline_train_seconds=sum(r['sampling_optimization_seconds'] for r in training),
        operator_accounting={key: sum(row['readout'].get('cg', {}).get(key, 0) for r in sampling
            for trace in r['traces'].values() for row in trace) for key in ('linear_calls', 'adjoint_calls')},
        limits=p['limits']+' Ten image clusters, one seed; engineering thresholds, no broad generalization/p-value/efficiency claim. '
               'C_IEEE raw timing includes separately recorded metrology seconds at two cases. '
               'Operator accounting shown for CG only, not all native training/operator-gate calls; see full traces and timings.')
    signed(root/'summary.json', result)
    print('A56 Summary:', json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    print('Records:', root, flush=True)
    return evaluation


def render(root, p, evaluation):
    import torch
    import numpy as np
    from PIL import Image, ImageDraw
    from utils.diag_cached_guidance_final_value import font
    for row in evaluation['rows']:
        c = row['case']; f, r = finished(root, p, c)
        bank = torch.load(f.parent/'outputs.pt', map_location='cpu', weights_only=True)
        names = ('GT', 'C_IEEE', 'U_CURRENT', 'V_PHYSICAL', 'Q_ORIGINAL', 'TIME_GAIN')
        sheet = Image.new('RGB', (1668, 390), 'white'); draw = ImageDraw.Draw(sheet)
        draw.text((12, 8), c['key']+' | unused accessible-history image; FROZEN RULES; fixed original grid', fill='black', font=font(14))
        for j, name in enumerate(names):
            left = 12+276*j; draw.text((left, 40), name, fill='black', font=font(14))
            if name == 'GT':
                im = Image.open(c['image']).convert('RGB').resize((256, 256), Image.Resampling.LANCZOS)
            else:
                a = bank[name][0].numpy().transpose(1, 2, 0)
                im = Image.fromarray(((np.clip(a, -1, 1)+1)*127.5).astype(np.uint8))
                m = row['metrics'][name]; draw.text((left, 66), f'P {m["psnr"]:.3f} S {m["ssim"]:.4f} L {m["lpips"]:.4f}', fill='black', font=font(12))
            sheet.paste(im, (left, 96))
        sheet.save(root/c['key']/'transfer_comparison.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=ROOT); parser.add_argument('--a55r2-dir', default=previous.ROOT)
    parser.add_argument('--ffhq-dir', default='/home/lzy/FFHQ/00000/00000'); parser.add_argument('--imagenet-dir', default='/home/lzy/imagenet/256x256')
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--shard', type=int, choices=(0, 1))
    parser.add_argument('--device', default='cuda:0'); parser.add_argument('--archive', action='store_true'); args = parser.parse_args()
    if args.prepare:
        prepare(args); return
    if args.shard is not None:
        worker(args); return
    root = Path(args.root).resolve(); p = check(root); value = summarize(root, p)
    if args.archive:
        from unittest.mock import patch
        render(root, p, value)
        with patch.object(previous.engine.old.previous, 'BRANCH', BRANCH):
            previous.engine.old.previous.archive(root)


if __name__ == '__main__':
    main()
