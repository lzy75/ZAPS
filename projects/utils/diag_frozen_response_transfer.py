"""A36 frozen response formula vs one old-image time table; 5400 method +1620 diagnostic NFE."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
from utils import diag_continuous_response_feedback as parent
from utils import diag_guidance_replication as baseline
from utils import diag_state_weight_scan as scan
from utils import transfer_response_feedback as rules

source, P = parent.source, parent.P
ROOT = '/data0/lzy/ZAPS-results/diag_frozen_response_transfer/A36_20261023'
CARD = PROJECTS/'experiment_cards/A36_frozen_response_transfer_20261023.md'


def select_images(data, root):
    import numpy as np
    paths = set()
    for base in (PROJECTS/'results', PROJECTS/'server_records', Path('/data0/lzy/ZAPS-results')):
        if base.exists():
            paths.update(p.resolve() for p in base.rglob('run.json') if not p.resolve().is_relative_to(root))
    used_names, used_sha, audit = set(), set(), []
    def visit(value):
        if isinstance(value, dict):
            if isinstance(value.get('image_sha256'), str):used_sha.add(value['image_sha256'])
            image = value.get('image')
            if isinstance(image, str) and Path(image).resolve().parent == data:used_names.add(Path(image).stem)
            for name, item in value.items():
                if name not in ('experiment_card', 'limits', 'scope'):visit(item)
        elif isinstance(value, list):
            for item in value:visit(item)
    for path in sorted(paths):
        visit(source.read(path));audit.append(dict(path=str(path), sha256=source.digest(path)))
    if not audit:raise ValueError('No accessible historical records; cannot establish unused images')
    pool = {}
    for path in sorted(data.glob('*.png')):
        if path.stem.isdigit() and 32 <= int(path.stem) <= 999 and path.stem not in used_names:
            sha = source.digest(path)
            if sha not in used_sha:
                pool.setdefault(sha, dict(image=str(path.resolve()), image_id=path.stem, image_sha256=sha))
    pool = sorted(pool.values(), key=lambda c:c['image_id'])
    if len(pool) < rules.IMAGES:raise ValueError('Too few unused unique images; no replacements after outcomes')
    indices = np.random.default_rng(20261023).choice(len(pool), rules.IMAGES, replace=False)
    selected = sorted([pool[i] for i in indices], key=lambda c:c['image_id'])
    return selected, dict(history=audit, pool=pool, scope='Accessible server run.json, including failed attempts; image paths/SHA only.')


def prepare(args):
    root, old = Path(args.root).resolve(), Path(args.a33_dir).resolve()
    if root.is_relative_to(old) or old.is_relative_to(root):raise ValueError('New run overlaps A33')
    if root.exists():
        plan = source.check_plan(root)
        if (plan['source_A33'] != str(old) or
                (args.data_dir and Path(args.data_dir).resolve() != Path(plan['cases'][0]['image']).parent)):
            raise ValueError('Frozen source/data directory differs')
        print('A36 frozen plan reused; images:', plan['image_ids'], '; completed training/feedback reused.', flush=True)
        return
    previous = source.check_plan(old)
    if (previous['experiment'] != 'A33_continuous_response_feedback'
            or source.load_signed(old/'summary.json')['completed_cases'] != 12):
        raise ValueError('Need twelve completed A33 cases')
    records, identities = [], {}
    for spec in previous['cases']:
        item = parent.task_done(old, previous, spec, 'I_cont2')
        if item is None:raise ValueError('Incomplete calibration trajectory')
        path, r = item;records.append(r)
        identities[str(path)] = source.digest(path)
    for name in ('plan.json', 'summary.json'):
        identities[str(old/name)] = source.digest(old/name)
    profile = rules.frozen_profile(records, previous['grid'])
    data = Path(args.data_dir).resolve() if args.data_dir else Path(previous['cases'][0]['case']['image']).resolve().parent
    selected, audit = select_images(data, root)
    cases = [dict(c, key=f'{c["image_id"]}_n{int(sigma*1000):03d}_s{rules.SEED}', sigma=sigma,
                  sampling_seed=rules.SEED, measurement_seed=1000+int(c['image_id'])) for c in selected for sigma in rules.SIGMAS]
    identities.update({c['image']:c['image_sha256'] for c in selected})
    own = [Path(__file__), Path(rules.__file__), Path(scan.__file__), Path(baseline.__file__), CARD,
           PROJECTS/'run_a36_20261023.sh', PROJECTS/'utils/test_transfer_response_feedback.py',
           PROJECTS/'utils/diag_zaps_guidance_clipping.py', PROJECTS/'utils/diag_zaps_first_direction.py',
           PROJECTS/'utils/diag_zaps_sampler_parity.py', PROJECTS/'utils/metrics.py',
           PROJECTS/'utils/archive_refinement_confirmation.py', PROJECTS/'configs/config.py']
    workers = dict(previous['worker_identity_files'])
    workers.update({str(f):source.digest(f) for f in own})
    plan = dict(experiment='A36_frozen_response_transfer', source_A33=str(old), cases=cases,
                image_ids=[c['image_id'] for c in selected], selection_audit=audit,
                time_gamma=profile, profile_GT_used=False, config=previous['config'], grid=previous['grid'],
                checkpoint_sha256=previous['checkpoint_sha256'], native_policy=previous['native_policy'],
                runtime=previous['runtime'], identity_files=identities, worker_identity_files=workers,
                planned_method_nfe=5400, planned_diagnostic_nfe=1620, scientific_nfe_per_arm=300,
                scope='Six unused images within accessible history, three measurement sigmas, one seed. '
                      'Frozen I_cont2 and one old-image time table; original first270 shared, final30 changed. '
                      'Same standard observation noise across sigmas; native DDPM/legacy-transpose local reference. '
                      'No tuning on transfer outcomes, full-ten-epoch feedback, NFE saving or global generalization claim.')
    root.parent.mkdir(parents=True, exist_ok=True);P.coupled.archive.recovery.require_space(root.parent)
    root.mkdir();source.signed(root/'plan.json', plan);(root/'experiment_card.md').write_text(CARD.read_text())
    print('A36 PREPARE PASS: images=', plan['image_ids'], '; 18 cases; 5400 method +1620 diagnostic =7020 calls.', flush=True)


def done(root, plan, case, kind):
    complete = []
    for path in sorted((root/case['key']/kind).glob('attempt_*/run.json')):
        r = source.read(path)
        if r['status'] != 'complete':continue
        r = source.load_signed(path)
        limit = 300 if kind == 'BASELINE' else 90
        if (r['case'] != case or r['plan_sha256'] != source.digest(root/'plan.json') or r['nfe'] != limit
                or r['calls'] != list(reversed(plan['grid']))*(10 if kind == 'BASELINE' else 3)
                or not r['gates'] or not all(g['passed'] for g in r['gates'].values())
                or any(source.digest(path.parent/n) != sha for n, sha in r['artifacts'].items())
                or source.digest(r['measurement']['path']) != r['measurement']['sha256']):
            raise ValueError('Invalid complete A36 record: '+str(path))
        if kind == 'FEEDBACK' and (set(r['metrics']) != set(rules.ARMS) or set(r['traces']) != set(rules.ARMS)
                or any([v['t'] for v in r['traces'][a]] != list(reversed(plan['grid'])) for a in rules.ARMS)):
            raise ValueError('Completed feedback trace differs')
        complete.append((path, r))
    if len(complete) > 1:raise ValueError('Multiple complete attempts; no output selection')
    return complete[0] if complete else None


def attempt(root, case, kind):
    folder = root/case['key']/kind;folder.mkdir(parents=True, exist_ok=True)
    path = folder/f'attempt_{len(list(folder.glob("attempt_*"))):03d}';path.mkdir()
    return path


def save_tensor(value, path):
    import torch
    temporary = path.with_suffix(path.suffix+'.tmp');torch.save(value, temporary);temporary.replace(path)


def measurement(root, case, operator, device):
    import fcntl
    import torch
    from modules.main_single import load_image_as_tensor
    from utils.diag_learning_rate_ablation import set_seed
    folder = root/'measurements';folder.mkdir(exist_ok=True);path = folder/(case['image_id']+'.pt')
    with path.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            with scan.preserved_rng(), torch.no_grad():
                truth = load_image_as_tensor(case['image']).to(device)
                mean = operator.H(truth);set_seed(case['measurement_seed'])
                standard = torch.randn_like(mean)
                save_tensor(dict(mean=mean.cpu(), standard_noise=standard.cpu(), image_sha256=case['image_sha256'],
                                 seed=case['measurement_seed']), path)
        saved = torch.load(path, map_location='cpu', weights_only=True)
    if (saved['image_sha256'] != case['image_sha256'] or saved['seed'] != case['measurement_seed']
            or any(v.shape != (1,3,64,64) or v.dtype != torch.float32 or not torch.isfinite(v).all()
                   for v in (saved['mean'], saved['standard_noise']))):raise ValueError('Observation identity/contract differs')
    # Do the arithmetic on the sampling device, matching the historical .05 observation protocol.
    y = saved['mean'].to(device)+case['sigma']*saved['standard_noise'].to(device)
    return y, dict(path=str(path), sha256=source.digest(path), sigma=case['sigma'],
                   standard_noise_sha256=hashlib.sha256(saved['standard_noise'].numpy().tobytes()).hexdigest())


def training(root, plan, case, model, device, operator):
    import torch
    from utils.diag_learning_rate_ablation import set_seed
    from utils.diag_zaps_sampler_parity import rng_state, same_rng
    from utils.diag_zaps_first_direction import model_rng_guard
    item = done(root, plan, case, 'BASELINE')
    if item:return item
    folder = attempt(root, case, 'BASELINE');path = folder/'run.json';started = time.perf_counter()
    y, meta = measurement(root, case, operator, device)
    r = dict(kind='BASELINE', case=case, plan_sha256=source.digest(root/'plan.json'), measurement=meta, gates={})
    with scan.counted(model, path, r, 300), scan.preserved_rng(), model_rng_guard(model.model, device):
        z = baseline.snapshot_class()(model, operator, img_size=256, **plan['config'])
        z.tau = torch.tensor(plan['grid'], device=device)
        set_seed(case['sampling_seed']);initial = torch.randn(1,3,256,256,device=device)
        for _ in range(290):torch.randn_like(initial)
        expected = rng_state(device);set_seed(case['sampling_seed'])
        r['loss_history'] = z.optimize(y, verbose=False, x0_gt=None, scheduler=None)
        if (r['nfe'] != 300 or r['calls'] != list(reversed(plan['grid']))*10 or z._last_nfe != 300
                or z.audit_state is None or len(r['loss_history']) != 10
                or not all(math.isfinite(v) for v in r['loss_history'])
                or not torch.equal(z._last_opt_noise, initial) or not same_rng(expected, rng_state(device))
                or any(p.grad is not None for p in model.model.parameters())):raise RuntimeError('Original training gate failed')
        save_tensor(z.audit_state, folder/'pre_last_state.pt');save_tensor(z._last_opt_x0.detach().cpu(), folder/'last_opt.pt')
        r['artifacts'] = {n:source.digest(folder/n) for n in ('pre_last_state.pt','last_opt.pt')}
        r['gates'] = dict(training_budget_rng=dict(passed=True), GT_epoch_selection=dict(passed=True, GT_used=False))
    r.update(status='complete', seconds=time.perf_counter()-started);source.signed(path, r)
    print('A36 BASELINE:', case['key'], '300 calls complete.', flush=True)
    return path, r


def feedback_run(root, plan, case, model, device, operator, trained):
    import torch
    from unittest.mock import patch
    from modules.zaps_algorithm import ZAPS
    from modules.main_single import load_image_as_tensor
    from utils.diag_zaps_sampler_parity import rng_state, same_rng
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_guidance_clipping import saved_cuda_selection, restore_sampling_rng
    from utils.diag_zaps_trace_audit import relative_error
    from utils.metrics import compute_all_metrics
    if done(root, plan, case, 'FEEDBACK'):return
    folder = attempt(root, case, 'FEEDBACK');path = folder/'run.json';started = time.perf_counter()
    train_path, training_record = trained
    state = torch.load(train_path.parent/'pre_last_state.pt', map_location='cpu', weights_only=True)
    original = torch.load(train_path.parent/'last_opt.pt', map_location='cpu', weights_only=True)
    y, meta = measurement(root, case, operator, device)
    if meta != training_record['measurement']:raise ValueError('Training/feedback observations differ')
    z = ZAPS(model, operator, img_size=256, **plan['config']);z.tau = state['tau'].to(device)
    with torch.no_grad():z.zeta.copy_(state['zeta'].to(device));z.D.copy_(state['D'].to(device))
    if z.tau.tolist() != plan['grid']:raise ValueError('Snapshot grid differs')
    positions = source.fields.policy.Positions(source.fields.policy.noise_u(z.dm.alphas_cumprod.cpu().numpy()), list(reversed(plan['grid'])))
    with scan.preserved_rng():
        mapping = saved_cuda_selection(state, device)
        restore_sampling_rng(state, device, mapping['saved_sampling_index'])
        noises = [torch.randn_like(state['init_noise'].to(device)) for _ in range(29)]
    record = dict(kind='FEEDBACK', case=case, plan_sha256=source.digest(root/'plan.json'), measurement=meta,
                  training_record=str(train_path), training_sha256=source.digest(train_path), gates={})
    outputs, traces = {}, {};external = rng_state(device)
    with scan.counted(model, path, record, 90):
        with torch.no_grad(), model_rng_guard(model.model, device), patch('torch.randn_like', side_effect=RuntimeError('Unexpected random draw')):
            for arm in rules.ARMS:
                x = state['init_noise'].to(device);trace = []
                for k, t in enumerate(positions.nominal):
                    f = source.fields.field(z, y, x, t, positions)
                    readout = rules.sensor(z, y, f['clean'], f['q'], t, case['sigma']) if arm == 'STATE' else {}
                    gamma = readout['gamma'] if arm == 'STATE' else (plan['time_gamma'][k] if arm == 'TIME' else 1.)
                    trace.append(dict(step=k, t=t, gamma=gamma, R=f['R']*.05/case['sigma'], clip=f['clip_fraction'],
                                      qRMS=float(f['q'].double().square().mean().sqrt()),
                                      **{n:v for n,v in readout.items() if n not in ('gamma','R','qRMS')}))
                    base = P.terminal.terminal_output(z, x, f) if k == 29 else P.coupled.posterior(
                        x, f['clean'], t, positions.nominal[k+1], z.dm.alphas_cumprod, noises[k])+f['q']
                    x = base if gamma == 1. else base+(gamma-1.)*f['q']
                    if not torch.isfinite(x).all():raise RuntimeError('Nonfinite controlled state; no fallback output')
                outputs[arm] = x.cpu();traces[arm] = trace
        parity = relative_error(outputs['C'], original)
        if (not math.isfinite(parity) or parity > 1e-6 or record['nfe'] != 90
                or record['calls'] != list(reversed(plan['grid']))*3 or not same_rng(external, rng_state(device))
                or P.precision.precision_state(torch) != plan['native_policy']
                or not all(torch.equal(v.cpu(), state[n]) for v,n in ((z.tau,'tau'),(z.zeta,'zeta'),(z.D,'D')))):
            raise RuntimeError('C parity, paired noise, budget, precision or parameters gate failed')
        save_tensor(outputs, folder/'outputs.pt')
        record.update(status='sampled', traces=traces, artifacts={'outputs.pt':source.digest(folder/'outputs.pt')})
        source.WRITE(path, record)
        # Synthetic GT constructed the observations earlier; it never enters controller/epoch/output selection.
        gt = load_image_as_tensor(case['image']).cpu()
        metrics = {arm:dict(loss=float((y-z.A.H(out.to(device))).double().square().mean()/(2*case['sigma']**2)),
                            **compute_all_metrics(out, gt, lpips_net='alex')) for arm,out in outputs.items()}
        if any(not math.isfinite(v) for m in metrics.values() for v in m.values()):raise RuntimeError('Nonfinite metric')
        record.update(metrics=metrics, gates=dict(original_C_parity=dict(passed=True, relative_error=parity),
                      budget_rng_parameters=dict(passed=True), GT_after_outputs=dict(passed=True),
                      frozen_time_and_formula=dict(passed=True)))
    record.update(status='complete', seconds=time.perf_counter()-started);source.signed(path, record)
    print('A36 CASE:', case['key'], 'STATE-C PSNR=', metrics['STATE']['psnr']-metrics['C']['psnr'], '90 diagnostic calls.', flush=True)


def worker(args):
    import fcntl
    import torch
    from modules.degradations import get_operator
    from utils.diag_ffhq_regression import TransposeMode
    from utils.diag_imagenet_gaussian_unit_domain import unit_operator
    root = Path(args.root).resolve();plan = source.check_plan(root)
    with (root/f'shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cases = [c for c in plan['cases'] if plan['image_ids'].index(c['image_id'])%2 == args.shard]
        if all(done(root, plan, c, 'FEEDBACK') for c in cases):return
        runtime = dict(torch=str(torch.__version__), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                       gpu=torch.cuda.get_device_name(args.device), NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),
                       TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        if runtime != plan['runtime'] or P.precision.precision_state(torch) != plan['native_policy']:
            raise ValueError('Frozen runtime/precision differs')
        torch.set_num_threads(2);P.coupled.archive.recovery.require_space(root)
        with scan.preserved_rng():
            model = P.coupled.archive.recovery.load_model(plan, args.device)
            operator = unit_operator(TransposeMode(get_operator('super_resolution', device=args.device, scale_factor=4, noise_sigma=0.), 'legacy_bicubic').to(args.device)).to(args.device)
            for case in cases:
                if source.digest(case['image']) != case['image_sha256']:raise ValueError('Target image changed')
                trained = training(root, plan, case, model, args.device, operator)
                feedback_run(root, plan, case, model, args.device, operator, trained)
            source.check_plan(root)


def pictures(path, record):
    import torch
    from PIL import Image, ImageDraw
    from modules.main_single import load_image_as_tensor
    from utils.metrics import _to_uint8_numpy
    from utils.diag_cached_guidance_final_value import font
    outputs = torch.load(path.parent/'outputs.pt', map_location='cpu', weights_only=True)
    arrays = {'GT':_to_uint8_numpy(load_image_as_tensor(record['case']['image']))}
    arrays.update({a:_to_uint8_numpy(v) for a,v in outputs.items()})
    sheet = Image.new('RGB', (1120,680), 'white');draw = ImageDraw.Draw(sheet)
    draw.text((20,10), record['case']['key']+' | frozen controllers; each arm300 NFE', fill='black', font=font(18))
    for j, arm in enumerate(('GT',)+rules.ARMS):
        x = 20+276*j;raw = Image.fromarray(arrays[arm]);full = raw.copy()
        ImageDraw.Draw(full).rectangle((64,48,192,176), outline='#f8bd46', width=2)
        draw.text((x,45), arm, fill='black', font=font(18))
        if arm != 'GT':
            m = record['metrics'][arm]
            draw.text((x,70), f'P {m["psnr"]:.3f} S {m["ssim"]:.4f}', fill='black', font=font(14))
            draw.text((x,90), f'LPIPS {m["lpips"]:.4f}', fill='black', font=font(14))
        sheet.paste(full,(x,115));sheet.paste(raw.crop((64,48,192,176)).resize((256,256),Image.Resampling.NEAREST),(x,390))
    sheet.save(path.parent/'comparison.png')


def summarize(args):
    root = Path(args.root).resolve();plan = source.check_plan(root);records = [];paths = []
    for case in plan['cases']:
        item = done(root, plan, case, 'FEEDBACK')
        if item:
            # Feedback never licenses an unavailable/corrupt shared training source.
            baseline_item = done(root, plan, case, 'BASELINE')
            if baseline_item is None or source.digest(baseline_item[0]) != item[1]['training_sha256']:
                raise ValueError('Feedback training provenance differs')
            paths.append(item[0]);records.append(item[1])
    attempts = [source.read(p) for p in root.glob('*/*/attempt_*/run.json')]
    result = rules.summarize(records)
    result.update(experiment=plan['experiment'], method_nfe=sum(r['nfe'] for r in attempts if r['kind']=='BASELINE'),
                  diagnostic_nfe=sum(r['nfe'] for r in attempts if r['kind']=='FEEDBACK'),
                  planned_method_nfe=5400, planned_diagnostic_nfe=1620, deployment_nfe_per_arm=300,
                  partial_attempts=sum(r['status']!='complete' for r in attempts),
                  cost_is_lower_bound_if_abruptly_interrupted=any(r['status'] in ('running','sampled') for r in attempts),
                  image_ids=plan['image_ids'], frozen_time_gamma=plan['time_gamma'])
    source.signed(root/'summary.json', result)
    if args.archive:
        for path,record in zip(paths,records):pictures(path,record)
    compact = {k:v for k,v in result.items() if k!='groups'}
    compact['groups'] = [{**{k:v for k,v in g.items() if k!='comparisons'},
                          'comparisons':{name:{k:v for k,v in r.items() if k!='per_image'} for name,r in g.get('comparisons',{}).items()}}
                         for g in result['groups']]
    print('A36 Summary:', json.dumps(compact, ensure_ascii=False), flush=True);print('Records:', root, flush=True)
    if args.archive:
        from utils.archive_refinement_confirmation import export_and_commit
        bundle, commit = export_and_commit({'A36':str(root)})
        print('Archive:', bundle, 'Results commit:', commit, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=ROOT)
    p.add_argument('--a33-dir',default=parent.ROOT);p.add_argument('--data-dir')
    p.add_argument('--prepare',action='store_true');p.add_argument('--shard',type=int,choices=(0,1))
    p.add_argument('--device',default='cuda:0');p.add_argument('--archive',action='store_true');args=p.parse_args()
    if args.prepare:prepare(args)
    elif args.shard is not None:worker(args)
    else:summarize(args)


if __name__ == '__main__':main()
