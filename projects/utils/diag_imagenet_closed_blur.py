"""A114 tests the unmodified 300-NFE feedback controllers on ImageNet blur."""
import argparse
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
import traceback
from unittest.mock import patch

PROJECTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECTS), str(PROJECTS.parent)]

from utils import paper_clock_alignment as protocol
from utils.diag_sr_inpainting_followup import A103, METRICS
from utils.paper_clock_alignment import digest, read, require, write
from utils.shared_prefix_feedback import ARMS, IMAGES, ROOT as A102

ROOT = A103.with_name('A114_20261111')
BRANCH = 'results/imagenet-closed-blur-20261111'
CARD = PROJECTS/'experiment_cards/A114_imagenet_closed_blur.md'
MANIFEST = PROJECTS/'IMAGENET_BASELINE_MANIFEST_20261013.json'
DATA = Path('/home/lzy/imagenet/256x256')
CHECKPOINT = PROJECTS/'modules/models/256x256_diffusion_uncond.pt'
TASKS = ('gaussian_deblur', 'motion_deblur')


def config(task, size=256, level=3):
    require(task in TASKS and size == 256 and level == 3, 'A114 configuration differs')
    return dict(protocol.config(task), img_size=size, level=level)


def cases(directory):
    return [dict(dataset='imagenet', task=task, image_id=i, image=str(directory/(i+'.png')),
                 seed=1000+int(i), sigma=.05, measurement_seed=offset+int(i),
                 key=f'imagenet_{task}_{i}_s{1000+int(i)}')
            for i in IMAGES for task, offset in zip(TASKS, (401000, 601000))]


def prepare(root, directory, parent):
    reference = read(MANIFEST)['archived_reference']
    sha = reference['checkpoint_sha256_verified_in_prior_upload']
    require(sha == protocol.CHECKPOINTS['imagenet']['sha256'] and digest(CHECKPOINT) == sha,
            'ImageNet checkpoint differs from the local reference')
    require(protocol.grid() == reference['actual_timesteps_ascending'] and reference['lr'] == config(TASKS[0])['lr'],
            'ImageNet local reference configuration differs')
    frozen_kernel = parent/'motion_kernel.npy'
    parent_plan = read(parent/'plan.json')
    kernel_sha = digest(frozen_kernel)
    require(parent_plan['experiment'] == 'A102_shared_prefix_feedback'
            and kernel_sha == parent_plan['motion_kernel_sha256'], 'A102 motion kernel differs')
    panel = [dict(c, image_sha256=digest(c['image'])) for c in cases(directory)]
    require(len({c['image_sha256'] for c in panel}) == 10, 'ImageNet panel contains duplicate image bytes')
    frozen = dict(experiment='A114', cases=panel, configs={t:config(t) for t in TASKS}, arms=list(ARMS),
                  grid=protocol.grid(), checkpoint=dict(path=str(CHECKPOINT), sha256=sha),
                  motion_kernel_sha256=kernel_sha, reference_manifest_sha256=digest(MANIFEST),
                  a102_plan_sha256=digest(parent/'plan.json'), method_nfe=300, joint_forward_cap=7200,
                  source_sha256=digest(__file__), card_sha256=digest(CARD))
    plan_path = root/'plan.json'
    if plan_path.is_file():
        saved = read(plan_path)
        require(all(saved.get(k) == v for k,v in frozen.items()) and
                digest(root/'motion_kernel.npy') == kernel_sha, 'Existing A114 plan differs')
        return saved
    require(not root.exists() or root.is_dir() and not any(root.iterdir()), 'Unrecognized result directory')
    root.mkdir(parents=True, exist_ok=True)
    (root/'motion_kernel.npy').write_bytes(frozen_kernel.read_bytes())
    require(digest(root/'motion_kernel.npy') == kernel_sha, 'Motion kernel copy differs')
    (root/'experiment_card.md').write_bytes(CARD.read_bytes())
    frozen['code_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECTS, text=True).strip()
    write(plan_path, frozen)
    return frozen


def groups(rows):
    result = []
    for task in TASKS:
        cell = [r for r in rows if r['case']['task'] == task]
        if not cell:
            continue
        arms = {arm:dict(quality={k:mean(r['arms'][arm]['metrics'][k] for r in cell) for k in METRICS},
                         mean_seconds=mean(r['arms'][arm]['method_equivalent_seconds'] for r in cell), nfe=300)
                for arm in ARMS}
        comparisons = []
        for candidate, reference in ((ARMS[1], ARMS[0]), (ARMS[2], ARMS[0]), (ARMS[2], ARMS[1])):
            pairs = [(r['arms'][candidate]['metrics'], r['arms'][reference]['metrics']) for r in cell]
            delta = {k:mean(a[k]-b[k] for a,b in pairs) for k in METRICS}
            wins = sum(a['psnr'] > b['psnr'] for a,b in pairs)
            comparisons.append(dict(candidate=candidate, reference=reference, delta=delta, psnr_wins=wins,
                                    positive=len(cell) == 10 and wins >= 7 and delta['psnr'] >= .1
                                    and delta['ssim'] >= 0 and delta['lpips'] <= 0))
        result.append(dict(task=task, images=len(cell), arms=arms, comparisons=comparisons))
    return result


def saved_rows(root, plan, validate):
    rows = []
    for case in plan['cases']:
        folder = root/'cases'/case['key']
        if not folder.exists():
            continue
        row = read(folder/'run.json')
        if row['status'] != 'complete':
            continue
        require(row['case'] == case and row['plan_sha256'] == digest(root/'plan.json'), 'Saved case differs')
        validate(row['counts']['prefix'], 270, 9)
        for arm in ARMS:
            validate(row['arms'][arm]['counts'], 30, 1)
            for name, sha in row['arms'][arm]['artifacts'].items():
                require(digest(folder/name) == sha, f'Saved {arm} artifact differs')
        rows.append(row)
    return rows


def emit(root, plan, baseline, status, **detail):
    rows = saved_rows(root, plan, baseline.validate)
    counts = [c for row in rows for c in (row['counts']['prefix'], *(row['arms'][a]['counts'] for a in ARMS))]
    report = dict(experiment='A114', status=status, completed_cases=len(rows), planned_cases=20,
                  groups=groups(rows), recorded_completed_case_cost={k:sum(c.get(k, 0) for c in counts) for k in
                     ('forward_attempts', 'forwards', 'optimizer_backward_attempts', 'input_vjp_attempts',
                      'input_jvp_attempts')},
                  cost_is_lower_bound=status != 'complete', **detail)
    write(root/'summary.json', report)
    return report


def execute(root, directory, parent):
    import torch
    from modules.main_single import load_diffusion_model
    from utils import diag_shared_prefix_feedback as baseline
    from utils.a54_session import exclusive
    from utils.a67_model_freeze import frozen_network
    from utils.diag_paper_clock_alignment import gpu_inventory, resource

    require(not root.is_relative_to(parent.resolve()), 'Use an independent A114 result directory')
    root.parent.mkdir(parents=True, exist_ok=True)
    with exclusive(root.with_name(root.name+'.lock')):
        plan = prepare(root, directory, parent)
        rows = saved_rows(root, plan, baseline.validate)
        if len(rows) == 20:
            print('A114 SUMMARY:', json.dumps(emit(root, plan, baseline, 'complete'), ensure_ascii=False), flush=True)
            return
        done = {row['case']['key'] for row in rows}
        for case in plan['cases']:
            folder = root/'cases'/case['key']
            if folder.exists() and case['key'] not in done:
                row = read(folder/'run.json')
                raise RuntimeError(f'Previous A114 attempt retained at {folder}: {row.get("error", row["status"])}')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        require(visible and ',' not in visible and torch.cuda.is_available()
                and torch.cuda.device_count() == 1, 'Expose one allocated GPU with CUDA_VISIBLE_DEVICES')
        assigned = protocol.available_gpus(gpu_inventory(), visible, visible)[0]['uuid']
        admission = resource(assigned, 'imagenet')
        require(admission['admitted'] and torch.cuda.mem_get_info(0)[0] >= protocol.MIN_FREE['imagenet'],
                'Need 9 GiB free for the local FP32 ImageNet model')
        emit(root, plan, baseline, 'running', device_uuid=assigned)
        try:
            model = load_diffusion_model('imagenet', 'cuda:0')
            model.model.eval()

            def imagenet_resource(uuid, dataset, reserved=0):
                return resource(uuid, 'imagenet', reserved)

            with frozen_network(model.model), patch.object(baseline.rules, 'configuration', config), \
                    patch.object(baseline, 'resource', imagenet_resource):
                for case in plan['cases']:
                    if case['key'] not in done:
                        baseline.run_case(root, plan, case, model, assigned)
                        done.add(case['key'])
                        emit(root, plan, baseline, 'running', device_uuid=assigned)
                        print(f'A114 COMPLETE: {len(done)}/20 {case["key"]}', flush=True)
        except BaseException as error:
            emit(root, plan, baseline, 'failed', device_uuid=assigned, error=repr(error),
                 traceback=traceback.format_exc())
            raise
        print('A114 SUMMARY:', json.dumps(emit(root, plan, baseline, 'complete', device_uuid=assigned),
                                         ensure_ascii=False), flush=True)


def archive(root):
    from utils import archive_refinement_confirmation as exporter

    report = read(root/'summary.json')
    require(report['experiment'] == 'A114' and report['status'] == 'complete' and report['completed_cases'] == 20,
            'A114 must be complete for archive')
    exporter.BRANCH = BRANCH
    bundle, commit = exporter.export_and_commit({'A114': root}, projects=PROJECTS)
    print(f'Archive: {bundle} Results commit: {commit} Branch: {BRANCH}', flush=True)
    subprocess.run(['git', 'push', 'origin', BRANCH], cwd=PROJECTS.parent, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imagenet-dir', type=Path, default=DATA)
    parser.add_argument('--a102-root', type=Path, default=Path(A102))
    parser.add_argument('--root', type=Path, default=ROOT)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--status', action='store_true')
    modes.add_argument('--archive', action='store_true')
    args = parser.parse_args()
    if args.status:
        print(json.dumps(read(args.root/'summary.json'), ensure_ascii=False, indent=2))
    elif args.archive:
        archive(args.root)
    else:
        execute(args.root.resolve(), args.imagenet_dir.resolve(), args.a102_root.resolve())


if __name__ == '__main__':
    main()
