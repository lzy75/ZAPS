"""A57: independent30-call U inference, no optimized zeta/D or training prefix."""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import shutil
import sys
import time

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS)); sys.path.insert(0, str(PROJECTS.parent))
from utils import diag_current_direction_value_r2 as previous
from utils import diag_frozen_direction_transfer as support
from utils import angular_feedback_control as current
from utils import no_learning_replay as rules

engine, parent, source = previous.engine, previous.parent, previous.source
write, signed = previous.write, previous.signed
ROOT = '/data0/lzy/ZAPS-results/diag_no_learning_replay/A57_20261027'
CARD = PROJECTS/'experiment_cards/A57_no_learning_replay_20261027.md'
BRANCH = 'results/no-learning-replay-20261027'


def require(test, message):
    if not test:
        raise ValueError(message)


def check(root):
    p = source.check_plan(root); require(p['experiment'] == 'A57_no_learning_replay', 'Wrong plan')
    parent.a42.verify(p); return p


def prepare(args):
    root, prior = Path(args.root).resolve(), Path(args.a55r2_dir).resolve()
    require(not (root.is_relative_to(prior) or prior.is_relative_to(root)), 'Source/output overlap')
    if root.exists():
        p = check(root); require(p['source_A55r2'] == str(prior), 'Frozen source differs')
        print('A57 frozen independent inference reused; completed work retained.', flush=True); return
    before = previous.check(prior); report = source.load_signed(prior/'summary.json')
    require(report['completed_cases'] == report['planned_cases'] == 10, 'Need completed A55r2')
    identities = dict(before['identity_files']); audits = {}
    for name in ('plan.json', 'summary.json', 'evaluation.json'):
        identities[str(prior/name)] = source.digest(prior/name)
    a41 = Path(before['source_A41']); p41 = source.load_signed(a41/'plan.json')
    identities[str(a41/'plan.json')] = source.digest(a41/'plan.json')
    for c in before['cases']:
        found = previous.item(prior, before, c, 'control'); require(found is not None, 'Missing direction source')
        trainroot, trainplan = parent.old.training_context(a41, p41, c)
        trained = parent.old.parent.done(trainroot, trainplan, c, 'BASELINE')
        require(trained is not None, 'Missing cached measurement/audit-only source')
        f, r = trained
        for name, sha in r['artifacts'].items():
            identities[str(f.parent/name)] = sha
        identities[str(f)] = source.digest(f)
        identities[r['measurement']['path']] = r['measurement']['sha256']
        audits[c['key']] = dict(measurement=r['measurement'], pre_audit_only=str(f.parent/'pre_last_state.pt'))
    own = [Path(__file__), Path(rules.__file__), CARD, PROJECTS/'run_a57_20261027.sh',
           PROJECTS/'utils/a57_session.py', PROJECTS/'utils/test_no_learning_replay.py', Path(support.__file__)]
    p = dict(before)
    p.update(experiment='A57_no_learning_replay', source_A55r2=str(prior), audit_sources=audits,
        untrained_configs={t: dict(p41['configs']['ffhq'][t], num_epochs=1, zeta_init=.1, d_init=.2)
                           for t in ('super_resolution', 'inpainting')},
        reference_evaluation=str(prior/'evaluation.json'), identity_files=identities,
        worker_identity_files={**before['worker_identity_files'], **{str(f): source.digest(f) for f in own}},
        phases=['control'], planned_method_nfe=300, planned_diagnostic_nfe=0,
        planned_backward_attempts_max=300, inference_forward_per_case=30, optimizer_updates=0,
        RNG=dict(initial_draws=1, skipped_innovations=261, sampling_innovations=29, reference_epochs=10),
        limits='Five OLD FFHQ development images x SR/IP, one seed. Independent seed/noise and fixed unlearned zeta=.1,D=.2. '
               'No optimize call or loading learned arrays into inference. Source PRE loaded only after output frozen for audit. '
               'A30-call equivalence test, not smaller sampling grid, new images, free VJP/operators or10x walltime proof. '
               'Only fully resolved paths can establish learned-parameter independence; fallback differences retained.')
    root.parent.mkdir(parents=True, exist_ok=True); parent.P.coupled.archive.recovery.require_space(root.parent)
    root.mkdir(); signed(root/'plan.json', p); shutil.copyfile(CARD, root/'experiment_card.md')
    print('A57 PREPARE PASS:10 independent no-learning30-call paths;300 NEW method forwards,<=300 input backwards;0 optimize updates.', flush=True)


def item(root, p, c, phase):
    require(phase == 'control', 'A57 has one inference phase')
    files = sorted((root/c['key']).glob('control_*/run.json'))
    if not files:
        return None
    require(len(files) == 1, 'Multiple attempts; no selection')
    f = files[0]; r = source.read(f)
    if r['status'] != 'complete':
        return None
    r = source.load_signed(f)
    require(r['case'] == c and r['plan_sha256'] == source.digest(root/'plan.json') and
            r['calls'] == list(reversed(p['grid'])) and r['nfe'] == 30 and
            r['backward_attempts'] == r['backward_completed'] <= 30 and
            all(g['passed'] for g in r['gates'].values()), 'Inference gate failed')
    require(all(source.digest(f.parent/n) == sha for n, sha in r['artifacts'].items()), 'Output bytes changed')
    return f, r, 'NO_LEARNING'


def sample(root, p, c, phase, model, device):
    import torch
    from unittest.mock import patch
    from modules.zaps_algorithm import ZAPS
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state, same_rng
    from utils.diag_zaps_guidance_clipping import saved_cuda_selection, restore_sampling_rng
    from utils.diag_zaps_trace_audit import relative_error
    if item(root, p, c, phase):
        return
    operator, mask = support.make_operator(c, device); opgate = support.a40.operator_gate(operator, device)
    meta = p['audit_sources'][c['key']]['measurement']
    saved = torch.load(meta['path'], map_location='cpu', weights_only=True)
    require(saved['image_sha256'] == c['image_sha256'] and saved['seed'] == c['measurement_seed'], 'Measurement identity differs')
    y = saved['mean'].to(device)+c['sigma']*saved['standard_noise'].to(device)
    z = ZAPS(model, operator, img_size=256, **p['untrained_configs'][c['task']])
    z.tau = torch.tensor(p['grid'], device=device)
    positions = source.fields.policy.Positions(source.fields.policy.noise_u(model.alphas_cumprod.cpu().numpy()), list(reversed(p['grid'])))
    initial_parameters = dict(zeta=z.zeta.detach().clone(), D=z.D.detach().clone(), tau=z.tau.detach().clone())
    require(bool((z.zeta == .1).all()) and bool((z.D == .2).all()), 'Unlearned constants differ')
    folder, r = engine.old.previous.new_attempt(root, p, c, phase); path = folder/'run.json'
    external = rng_state(device); r['traces'] = []; started = time.perf_counter()
    audit_draw_like = torch.randn_like
    with parent.scan.preserved_rng():
        initial, noises, before_noise = rules.noise_from_seed(c['sampling_seed'], device)
    with engine.old.previous.counted(model, path, r, 30) as backward, \
         current.base.input_vjp_parameters(model.model), model_rng_guard(model.model, device), \
         patch.object(z, 'optimize', side_effect=RuntimeError('A57 forbids optimization')), \
         patch('torch.randn_like', side_effect=RuntimeError('Unexpected sampling RNG draw')):
        x = initial
        with parent.P.precision.ieee(torch):
            for k, t in enumerate(positions.nominal):
                s = positions.nominal[k+1] if k < 29 else -1
                field, gamma, row = current.sensed_field(z, y, x, t, s, positions, c['sigma'], 1., mask, backward)
                r['backward_completed'] += row['backward_calls']; r['traces'].append(dict(step=k, t=t, readout=row))
                with torch.no_grad():
                    x = parent.old.parent.rules.risk.advance(z, x, field, positions, k, noises, gamma)
                require(bool(torch.isfinite(x).all()), 'Nonfinite independent trajectory')
        torch.cuda.synchronize(device); r['inference_seconds'] = time.perf_counter()-started
        require(same_rng(external, rng_state(device)) and all(v.grad is None for v in model.model.parameters()), 'RNG/weight grad changed')
        require(all(torch.equal(getattr(z, n), a) for n, a in initial_parameters.items()), 'Unlearned parameters changed')
        parent.old.parent.base.save_tensor(x.detach().cpu(), folder/'output.pt')
        signed(folder/'decisions_before_audit.json', dict(case=c, traces=r['traces'], GT_used=False,
            initialization='seed-only, fixed zeta=.1/D=.2; no training', plan_sha256=r['plan_sha256']))
        output_sha = source.digest(folder/'output.pt')
        # This is the first PRE deserialization; its learned arrays cannot have
        # influenced the already frozen inference output above.
        state = torch.load(p['audit_sources'][c['key']]['pre_audit_only'], map_location='cpu', weights_only=True)
        with parent.scan.preserved_rng():
            selection = saved_cuda_selection(state, device)
            restore_sampling_rng(state, device, selection['saved_sampling_index'])
            expected_noises = [audit_draw_like(initial) for _ in range(29)]
        initial_equal = torch.equal(initial.cpu(), state['init_noise'])
        bank_equal = all(torch.equal(a, b) for a, b in zip(noises, expected_noises))
        require(initial_equal and bank_equal, 'Independent random recipe does not pair with reference')
        f = Path(p['norm_sources'][c['key']])
        reference = torch.load(f.parent/'outputs.pt', map_location='cpu', weights_only=True)['NORM_ONLY']
        error = relative_error(x.detach().cpu(), reference)
        require(math.isfinite(error) and output_sha == source.digest(folder/'output.pt'), 'Output/audit failed')
        r['comparison'] = dict(relative_error=error, tolerance=1e-6,
            fully_resolved=all(row['readout']['resolved'] for row in r['traces']),
            equivalent=error <= 1e-6, initial_equal=bool(initial_equal), innovations_equal=bool(bank_equal),
            source_cpu_rng_equal=bool(torch.equal(before_noise['cpu'], state['cpu_rng'])),
            source_PRE_first_loaded_after_output_hash=output_sha)
        r['gates'] = dict(operator=opgate, independent_noise=dict(passed=True),
            no_optimizer_or_learned_parameters=dict(passed=True, optimizer_updates=0), calls_rng_weights=dict(passed=True))
        r['artifacts'] = {n: source.digest(folder/n) for n in ('output.pt', 'decisions_before_audit.json')}
    r['status'] = 'complete'; signed(path, r)
    print('A57:', c['key'], 'F=', r['nfe'], 'inputB=', r['backward_attempts'], 'equivalent=', r['comparison']['equivalent'], flush=True)


@contextmanager
def executor_bindings():
    from unittest.mock import patch
    with patch.object(engine, 'check', check), patch.object(engine, 'item', item), patch.object(engine, 'sample', sample):
        yield


def summarize(root, p):
    import torch
    from modules.main_single import load_image_as_tensor
    from utils.metrics import compute_all_metrics
    pairs = [(c, item(root, p, c, 'control')) for c in p['cases']]
    require(all(v for c, v in pairs), 'Need all independent outputs before labels')
    cached = {r['case']['key']: r for r in source.load_signed(Path(p['reference_evaluation']))['rows']}
    rows = []
    with parent.scan.preserved_rng(), torch.no_grad():
        for c, (f, r, _) in pairs:
            out = torch.load(f.parent/'output.pt', map_location='cpu', weights_only=True)
            metrics = compute_all_metrics(out, load_image_as_tensor(c['image']).cpu(), lpips_net='alex')
            rows.append(dict(case=c, metrics=metrics, comparison=r['comparison'],
                delta_vs_cached_U={m: metrics[m]-cached[c['key']]['metrics']['U_CURRENT'][m] for m in ('psnr','ssim','lpips')}))
    values = [r for c, (f, r, key) in pairs]
    passed = all(r['comparison']['equivalent'] and r['comparison']['fully_resolved'] for r in values)
    result = dict(experiment=p['experiment'], classification='NO_LEARNING_EQUIVALENCE_ON_DEVELOPMENT_COHORT' if passed else 'NO_LEARNING_EQUIVALENCE_NOT_CONFIRMED',
        completed_cases=10, planned_cases=10, image_clusters=5, method_nfe=sum(r['nfe'] for r in values), diagnostic_nfe=0,
        input_backward_attempts=sum(r['backward_attempts'] for r in values), input_backward_completed=sum(r['backward_completed'] for r in values),
        optimizer_updates=0, measured_inference_seconds=sum(r['inference_seconds'] for r in values),
        max_relative_error=max(r['comparison']['relative_error'] for r in values),
        seeded_noise_tensor_draws_per_case=291, standalone_forward_per_case=30, baseline_forward_per_case=300,
        per_image=rows, limits=p['limits']+' No10x walltime claim from forward count; no new independent-image confirmation.')
    signed(root/'summary.json', result); print('A57 Summary:', json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    print('Records:', root, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=ROOT); parser.add_argument('--a55r2-dir', default=previous.ROOT)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--phase', choices=('control',), default='control')
    parser.add_argument('--shard', type=int, choices=(0, 1)); parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--archive', action='store_true'); args = parser.parse_args()
    if args.prepare:
        prepare(args); return
    if args.shard is not None:
        with executor_bindings():
            engine.worker(args)
        return
    root = Path(args.root).resolve(); p = check(root); summarize(root, p)
    if args.archive:
        from unittest.mock import patch
        with patch.object(engine.old.previous, 'BRANCH', BRANCH):
            engine.old.previous.archive(root)


if __name__ == '__main__':
    main()
