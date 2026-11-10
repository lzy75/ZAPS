"""A101 strict 300-NFE FFHQ validation and qualitative report."""
import argparse
import csv
import gc
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS)); sys.path.insert(0, str(PROJECTS.parent))

from utils import equal_nfe_feedback as rules
from utils import equal_nfe_feedback_runtime as runtime
from utils import historical_ffhq_gain as files
from utils import historical_ffhq_runtime as old_runtime
from utils import paper_clock_runtime as clock
from utils import diag_noise_feasible_zaps as common
from utils.a67_model_freeze import frozen_network
from utils.diag_paper_clock_alignment import write, resource

ROOT = '/data0/lzy/ZAPS-results/diag_equal_nfe_feedback/A101_20261110'
PARENT = '/data0/lzy/ZAPS-results/diag_shrink_multitask/A71_20261029'
BRANCH = 'results/zaps-equal-nfe-feedback-20261110'
CARD = PROJECTS / 'experiment_cards/A101_equal_nfe_feedback_20261110.md'


def record_path(root, case, arm):
    return Path(root) / 'cases' / case['key'] / arm / 'run.json'


def check(root):
    root = Path(root); plan = files.read(root / 'plan.json')
    files.require(plan['experiment'] == 'A101_equal_nfe_feedback', 'Wrong A101 plan')
    for name, sha in plan['sources'].items():
        files.require(files.digest(PROJECTS / name) == sha, 'Source changed: ' + name)
    for name, sha in plan['inputs'].items():
        files.require(files.digest(name) == sha, 'Input changed: ' + name)
    files.require(files.digest(root / 'motion_kernel.npy') == plan['motion_kernel_sha256'],
                  'Motion kernel changed')
    return plan


def prepare(args):
    import numpy as np
    root = Path(args.root).resolve(); parent = Path(args.a71_root).resolve()
    if (root / 'plan.json').exists():
        check(root); print('A101 frozen plan reused.', flush=True); return
    files.require(not root.exists() or not any(root.iterdir()), 'Nonempty A101 root retained')
    from utils.diag_shrink_multitask import check as parent_check
    parent_plan = parent_check(parent); parent_summary = files.read(parent / 'summary.json')
    files.require(parent_summary['completed_runs'] == parent_summary['planned_runs'] == 72,
                  'Complete A71 identity anchor required')
    cases = rules.cases(args.ffhq_dir)
    for case in cases:
        files.require(Path(case['image']).is_file(), 'Missing fixed image: ' + case['image'])
        case['image_sha256'] = files.digest(case['image'])
    root.parent.mkdir(parents=True, exist_ok=True)
    files.require(shutil.disk_usage(root.parent).free >= 4 * files.GIB, 'Need at least4GiB result space')
    root.mkdir()
    from utils.diag_ffhq_four_tasks_unit import original_motion_kernel
    np.save(root / 'motion_kernel.npy', original_motion_kernel(0), allow_pickle=False)
    names = [
        'utils/equal_nfe_feedback.py', 'utils/equal_nfe_feedback_runtime.py',
        'utils/diag_equal_nfe_feedback.py', 'utils/a101_session.py',
        'utils/render_equal_nfe_feedback.py', 'utils/test_equal_nfe_feedback.py',
        'utils/test_equal_nfe_feedback_runtime.py', 'utils/noise_signal_shrink.py',
        'utils/shrink_multitask_runtime.py', 'utils/historical_ffhq_gain.py',
        'utils/historical_ffhq_runtime.py', 'utils/paper_clock_runtime.py',
        'utils/diag_ffhq_four_tasks_unit.py', 'utils/a67_model_freeze.py',
        'utils/metrics.py', 'modules/main_single.py', 'modules/degradations.py',
        'modules/zaps_algorithm.py', 'modules/wavelet.py', 'run_a101_20261110.sh',
        str(CARD.relative_to(PROJECTS)),
    ]
    inputs = {
        str(parent / 'plan.json'): files.digest(parent / 'plan.json'),
        str(parent / 'summary.json'): files.digest(parent / 'summary.json'),
    }
    checkpoint = parent_plan['checkpoint']
    files.require(files.digest(checkpoint['path']) == checkpoint['sha256'], 'Checkpoint changed')
    protocol = dict(
        baseline='9/17 strong local ZAPS implementation, not author-code parity or the paper 1000-image mean.',
        scope='Fixed FFHQ IDs00000-00009 x4 tasks x model-coordinate sigma .025/.05/.10 x3 arms. One seed/image; all cells retained.',
        arms='ZAPS_LOCAL_300 is unmodified local ZAPS. CLOSED_SCALAR_300 is the A71 noise-corrected native-q ray. CLOSED_DIAG_300 starts from that scalar and makes one exact steepest-descent update in the native db4 coefficient support for the same frozen linear observation target.',
        objective='omega=(1-Msigma^2/||r||^2)+; minimize .5||PH(delta/sqrt(alpha_t))-omega Pr||^2. Scalar is the nested ray solution; diagonal changes direction and length. This is a current-state frozen response proxy, not an actual next-denoiser response.',
        equal_nfe='Every complete reconstruction must have exactly300 denoiser forwards,10 optimizer backwards,0 input VJP,0 input JVP and the identical30x10 grid. H/Ht and db4 counts are separate.',
        pairing='Same image, y, operator, initial/final RNG and learned-prefix entry per arm. Independent from-seed complete runs; no cached learned policy, GT action, best epoch or output selection.',
        tasks='SR4 empirical historical bicubic guidance; Gaussian61/sigma3 zero boundary; frozen DPS motion61/intensity.5; random70% missing. Inpainting statistics use observed RGB coordinates.',
        decision='For each task/noise and candidate-vs-reference: mean PSNR>=+.1dB, >=7/10 PSNR wins, SSIM nondecrease, LPIPS nonincrease and paired gates. Report all failures/tradeoffs.',
        qualitative='Before outcomes, IDs00001/00004/00008 fixed for panels. Show GT, observation, all three outputs; no best/worst selection.',
        limits='Small ten-image local confirmation. No actual-response, posterior, author-reproduction, acceleration, universal-transfer or curvature claim.')
    plan = dict(
        experiment='A101_equal_nfe_feedback', parent=str(parent), cases=cases,
        checkpoint=checkpoint, grid=files.grid(), arms=list(rules.ARMS),
        configs={task: rules.configuration(task) for task in rules.TASKS},
        budget=dict(planned_runs=len(cases) * len(rules.ARMS),
                    method_forward_exact=len(cases) * len(rules.ARMS) * 300,
                    optimizer_backward_exact=len(cases) * len(rules.ARMS) * 10,
                    input_vjp=0, input_jvp=0, warmup_separate=True),
        motion_kernel_sha256=files.digest(root / 'motion_kernel.npy'),
        original_motion_generator_sha256=files.digest(PROJECTS.parent / 'DPS/motionblur/motionblur.py'),
        sources={name: files.digest(PROJECTS / name) for name in names},
        inputs=inputs, protocol=protocol,
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECTS.parent,
                                            text=True).strip())
    write(root / 'plan.json', plan); shutil.copyfile(CARD, root / 'experiment_card.md')
    print('A101 PREPARE/FREEZE: 10 images x4 tasks x3 noise x3 arms =360 complete runs; '
          'exact108000 denoiser forwards,3600 optimizer backwards,0 input VJP/JVP.', flush=True)


def make_operator(root, case, device, count=None):
    import numpy as np
    plan = files.read(Path(root) / 'plan.json')
    files.require(files.digest(Path(root) / 'motion_kernel.npy') == plan['motion_kernel_sha256'],
                  'Motion kernel changed')
    return runtime.make_operator(case, device, count=count,
                                 kernel=np.load(Path(root) / 'motion_kernel.npy', allow_pickle=False))


def observation(root, case, device):
    import torch
    from modules.main_single import load_image_as_tensor
    from modules.dataset_loader import tensor_to_image
    from utils.diag_learning_rate_ablation import set_seed
    folder = Path(root) / 'observations' / case['key']; meta = folder / 'observation.json'
    if meta.exists():
        row = files.read(meta); files.require(files.digest(row['path']) == row['sha256'],
                                             'Observation changed'); return row
    folder.mkdir(parents=True)
    with old_runtime.precision('LEGACY_IEEE'), torch.no_grad():
        gt = load_image_as_tensor(case['image']).to(device); op = make_operator(root, case, device)
        set_seed(case['measurement_seed']); mean = op.H(gt)
        y = mean + case['sigma'] * torch.randn_like(mean)
        torch.save(dict(y=y.cpu()), folder / 'y.pt')
        tensor_to_image(gt.cpu().squeeze(0), denormalize=True).save(folder / 'ground_truth.png')
        display = y.cpu(); mask = getattr(op, '_mask_cache', None)
        if mask is not None:
            display = display * mask.cpu() - (1 - mask.cpu()); torch.save(mask.cpu(), folder / 'mask.pt')
        image = tensor_to_image(display.squeeze(0), denormalize=True)
        if image.size != (256, 256):
            from PIL import Image
            image = image.resize((256, 256), Image.Resampling.BICUBIC)
        image.save(folder / 'observation.png')
    row = dict(path=str(folder / 'y.pt'), sha256=files.digest(folder / 'y.pt'),
               sigma=case['sigma'], domain='model-coordinate',
               display='SR enlarged; inpainting missing pixels black for display only')
    write(meta, row); return row


def validate_counts(counts):
    files.require(counts['forwards'] == counts['forward_attempts'] == 300,
                  'A101 denoiser NFE differs from300')
    files.require(counts['calls'] == files.grid()[::-1] * 10, 'A101 grid/calls differ')
    files.require(counts['optimizer_backward_attempts'] ==
                  counts['optimizer_backward_completed'] == 10, 'A101 optimizer count differs')
    files.require(counts.get('input_vjp_attempts', 0) == 0, 'A101 input VJP forbidden')
    files.require(counts.get('input_jvp_attempts', 0) == 0, 'A101 input JVP forbidden')


def region_errors(output, gt, operator):
    import torch
    error = (output.double() - gt.double()).square()
    row = dict(mse_full=float(error.mean()))
    mask = getattr(operator, '_mask_cache', None)
    if mask is not None:
        active = mask.bool().expand_as(error)
        row.update(mse_observed=float(error[active].mean()),
                   mse_missing=float(error[~active].mean()))
    return row


def measure(root, plan, case, arm, model, uuid):
    import torch
    from modules.main_single import load_image_as_tensor
    from modules.dataset_loader import tensor_to_image
    path = record_path(root, case, arm); folder = path.parent
    if path.exists():
        row = files.read(path); files.require(row['status'] == 'complete',
                                             'Prior unfinished A101 run retained: ' + str(path)); return
    files.require(resource(uuid, 'ffhq', torch.cuda.memory_reserved())['admitted'],
                  'Resource margin lost before A101 attempt')
    obs = observation(root, case, model.device); folder.mkdir(parents=True, exist_ok=False)
    row = dict(status='running', case=case, arm=arm,
               plan_sha256=files.digest(Path(root) / 'plan.json'), device_uuid=uuid,
               counts=None, artifacts={})
    write(path, row)
    def tick(value):
        row['counts'] = value; write(path, row)
        if value['forwards'] % 30 == 0:
            print('A101 PROGRESS:', case['key'], arm, 'F=', value['forwards'], '/300', flush=True)
    try:
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        operator = make_operator(root, case, model.device)
        with old_runtime.precision('LEGACY_IEEE'), clock.counters(model.model, tick) as counts:
            def operation():
                y = torch.load(obs['path'], map_location='cpu', weights_only=True)['y'].to(model.device)
                counted_operator = make_operator(root, case, model.device, counts)
                return runtime.execute(model, counted_operator, y, case, arm)
            (output, detail), seconds = clock.clocks(torch.cuda.synchronize, operation)
        row['counts'] = counts; validate_counts(counts)
        base_path = record_path(root, case, 'ZAPS_LOCAL_300')
        if arm == 'ZAPS_LOCAL_300':
            torch.save(detail['entry'], folder / 'entry.pt'); prefix = dict(passed=True, reference='self')
        else:
            files.require(base_path.exists() and files.read(base_path)['status'] == 'complete',
                          'ZAPS baseline must complete first')
            entry = torch.load(base_path.parent / 'entry.pt', map_location='cpu', weights_only=True)
            prefix = clock.snapshot_equal(entry, detail['entry'])
            files.require(prefix['passed'], 'Learned prefix/RNG differs from ZAPS baseline')
        gt = load_image_as_tensor(case['image']).cpu()
        metrics = common.metrics(output, gt); regions = region_errors(output, gt, operator)
        torch.save(output, folder / 'output.pt')
        tensor_to_image(output.squeeze(0), denormalize=True).save(folder / 'output.png')
        write(folder / 'trace.json', detail['trace'])
        artifacts = {name: files.digest(folder / name) for name in ('output.pt', 'output.png', 'trace.json')}
        if arm == 'ZAPS_LOCAL_300': artifacts['entry.pt'] = files.digest(folder / 'entry.pt')
        row.update(status='complete', metrics=metrics, region_errors=regions,
                   complete_reconstruction_seconds=seconds,
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   prefix_pairing=prefix, module_counts=detail.get('module_counts', {}),
                   loss_history=detail['loss'], counts=counts, artifacts=artifacts,
                   cost_is_lower_bound=False)
        write(path, row)
        print('A101 COMPLETE:', case['key'], arm, metrics, 'seconds=', round(seconds, 2), flush=True)
    except BaseException as error:
        if 'counts' in locals(): row['counts'] = counts
        row.update(status='failed', error=repr(error), traceback=traceback.format_exc(),
                   cost_is_lower_bound=True)
        write(path, row); raise


def worker(args):
    import torch
    from modules.main_single import load_diffusion_model
    from utils.a54_session import exclusive
    root = Path(args.root); plan = check(root); folder = root / 'workers' / str(args.shard)
    folder.mkdir(parents=True, exist_ok=True)
    with exclusive(folder / 'worker.lock'):
        if not resource(args.device_uuid, 'ffhq')['admitted']:
            write(folder / 'status.json', dict(status='resource_limited')); return
        files.require(files.digest(plan['checkpoint']['path']) == plan['checkpoint']['sha256'],
                      'Checkpoint changed')
        for case in plan['cases']:
            files.require(files.digest(case['image']) == case['image_sha256'], 'Image changed')
        start = time.perf_counter(); model = load_diffusion_model('ffhq', 'cuda:0'); model.model.eval()
        torch.cuda.synchronize(); load = dict(status='running', uuid=args.device_uuid,
                                             load_seconds=time.perf_counter() - start,
                                             warmup_forward_attempts=0)
        load_path = folder / f'load_{len(list(folder.glob("load_*.json"))):03d}.json'; write(load_path, load)
        try:
            with frozen_network(model.model):
                load['warmup_forward_attempts'] = 1; write(load_path, load)
                with torch.no_grad(), old_runtime.precision('LEGACY_IEEE'):
                    model._predict_eps(torch.zeros(1, 3, 256, 256, device=model.device),
                                       torch.tensor([999], device=model.device))
                torch.cuda.synchronize()
                for case_index, case in enumerate(plan['cases']):
                    if case['shard'] != args.shard: continue
                    observation(root, case, model.device)
                    order = ('ZAPS_LOCAL_300', 'CLOSED_SCALAR_300', 'CLOSED_DIAG_300')
                    if case_index % 2: order = ('ZAPS_LOCAL_300', 'CLOSED_DIAG_300', 'CLOSED_SCALAR_300')
                    for arm in order: measure(root, plan, case, arm, model, args.device_uuid)
            load['status'] = 'complete'; write(folder / 'status.json', dict(status='complete'))
        except BaseException as error:
            load.update(status='failed', error=repr(error), traceback=traceback.format_exc())
            write(folder / 'status.json', load); raise
        finally: write(load_path, load)


def summarize(root):
    root = Path(root); plan = check(root)
    rows = [files.read(path) for path in (root / 'cases').glob('*/*/run.json')]
    done = [row for row in rows if row['status'] == 'complete']; groups = []; comparisons = []
    for task_index, task in enumerate(rules.TASKS):
        for noise_index, sigma in enumerate(rules.SIGMAS):
            cell = [row for row in done if row['case']['task'] == task and row['case']['sigma'] == sigma]
            for arm in rules.ARMS:
                arm_rows = [row for row in cell if row['arm'] == arm]
                if arm_rows:
                    groups.append(dict(task=task, sigma=sigma, arm=arm, images=len(arm_rows),
                        quality={key: statistics.mean(row['metrics'][key] for row in arm_rows)
                                 for key in ('psnr', 'ssim', 'lpips')},
                        mean_full_seconds=statistics.mean(row['complete_reconstruction_seconds'] for row in arm_rows),
                        mean_peak_reserved_bytes=statistics.mean(row['peak_reserved_bytes'] for row in arm_rows),
                        mean_operator_counts={key: statistics.mean(row['counts'].get(key, 0) for row in arm_rows)
                                              for key in ('base_H_actions', 'base_adjoint_actions')}))
            by_arm = {arm: {row['case']['image_id']: row for row in cell if row['arm'] == arm}
                      for arm in rules.ARMS}
            for candidate, reference in (('CLOSED_SCALAR_300', 'ZAPS_LOCAL_300'),
                                         ('CLOSED_DIAG_300', 'ZAPS_LOCAL_300'),
                                         ('CLOSED_DIAG_300', 'CLOSED_SCALAR_300')):
                ids = sorted(set(by_arm[candidate]) & set(by_arm[reference]))
                if not ids: continue
                deltas = {key: [by_arm[candidate][image]['metrics'][key] -
                                by_arm[reference][image]['metrics'][key] for image in ids]
                          for key in ('psnr', 'ssim', 'lpips')}
                means = {key: statistics.mean(value) for key, value in deltas.items()}
                comparisons.append(dict(task=task, sigma=sigma, candidate=candidate,
                    reference=reference, images=len(ids), delta=means,
                    psnr_ci95=rules.bootstrap_mean_ci(deltas['psnr'],
                        20261110 + 100 * task_index + 10 * noise_index + len(comparisons)),
                    psnr_wins=sum(value > 0 for value in deltas['psnr']),
                    counts_equal=all(by_arm[candidate][image]['counts']['forwards'] ==
                                     by_arm[reference][image]['counts']['forwards'] == 300 for image in ids),
                    positive=len(ids) == 10 and means['psnr'] >= .1 and means['ssim'] >= 0 and
                             means['lpips'] <= 0 and sum(value > 0 for value in deltas['psnr']) >= 7))
    for row in done: validate_counts(row['counts'])
    loads = [files.read(path) for path in (root / 'workers').glob('*/load_*.json')]
    summary = dict(experiment=plan['experiment'],
        classification='STRICT_EQUAL_300_NFE_ZAPS_SCALAR_DIAGONAL',
        completed_runs=len(done), planned_runs=len(plan['cases']) * len(rules.ARMS),
        image_clusters=len({row['case']['image_id'] for row in done}), groups=groups,
        comparisons=comparisons,
        actual_new_cost=dict(method_forward_attempts=sum((row.get('counts') or {}).get('forward_attempts', 0) for row in rows),
            optimizer_backward_attempts=sum((row.get('counts') or {}).get('optimizer_backward_attempts', 0) for row in rows),
            input_vjp_attempts=sum((row.get('counts') or {}).get('input_vjp_attempts', 0) for row in rows),
            input_jvp_attempts=sum((row.get('counts') or {}).get('input_jvp_attempts', 0) for row in rows),
            warmup_forward_attempts=sum(row.get('warmup_forward_attempts', 0) for row in loads)),
        cost_is_lower_bound=len(done) < len(plan['cases']) * len(rules.ARMS) or
                            any(row['status'] != 'complete' for row in rows),
        protocol=plan['protocol'])
    write(root / 'summary.json', summary)
    with (root / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(
            ['task', 'sigma', 'arm', 'images', 'PSNR', 'SSIM', 'LPIPS', 'full_seconds',
             'peak_bytes', 'H_calls', 'Ht_calls'])
        for group in groups:
            writer.writerow([group['task'], group['sigma'], group['arm'], group['images'],
                *[group['quality'][key] for key in ('psnr', 'ssim', 'lpips')],
                group['mean_full_seconds'], group['mean_peak_reserved_bytes'],
                group['mean_operator_counts']['base_H_actions'],
                group['mean_operator_counts']['base_adjoint_actions']])
    print('A101 Summary:', json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def report(root):
    from utils.render_equal_nfe_feedback import render
    render(Path(root))


def archive(root):
    from utils import archive_refinement_confirmation as exporter
    exporter.BRANCH = BRANCH
    bundle, commit = exporter.export_and_commit({'A101': Path(root)}, projects=PROJECTS)
    print('Archive:', bundle, 'Results commit:', commit, 'Branch:', BRANCH, flush=True)
    subprocess.run(['git', 'push', 'origin', BRANCH], cwd=PROJECTS.parent, check=True)
    print('RESULTS_PUSH_PASS: A101 uploaded.', flush=True)


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('--root', default=ROOT); value.add_argument('--a71-root', default=PARENT)
    value.add_argument('--ffhq-dir', default='/home/lzy/FFHQ/00000/00000')
    value.add_argument('--shard', type=int, choices=(0, 1)); value.add_argument('--device-uuid')
    group = value.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'worker', 'summarize', 'report', 'archive'):
        group.add_argument('--' + name, action='store_true')
    return value


def main():
    parser_value = parser(); args = parser_value.parse_args()
    if args.worker and (args.shard is None or not args.device_uuid):
        parser_value.error('Worker needs shard and device UUID')
    if args.prepare: prepare(args)
    elif args.worker: worker(args)
    elif args.summarize: summarize(args.root)
    elif args.report: report(args.root)
    else: archive(args.root)


if __name__ == '__main__': main()
