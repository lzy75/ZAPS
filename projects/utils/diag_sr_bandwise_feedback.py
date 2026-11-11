"""A115 compares scalar and diagonal updates for the same bandwise SR target."""
import argparse
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
import traceback
from types import SimpleNamespace
from unittest.mock import patch

PROJECTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECTS), str(PROJECTS.parent)]

from utils import paper_clock_alignment as protocol, sr_bandwise_feedback as band
from utils.diag_300nfe_feedback_confirmation import ObservableProjection, select_cases
from utils.diag_sr_inpainting_followup import A103, METRICS, source_case
from utils.diag_sr_remaining_signal import ROOT as A113, SCALAR, DIAG
from utils.paper_clock_alignment import digest, read, require, write

ROOT = A103.with_name('A115_20261111')
CARD = PROJECTS/'experiment_cards/A115_sr_bandwise_feedback.md'
BRANCH = 'results/zaps-sr-bandwise-feedback-20261111'
RAY, BAND_DIAG = 'BAND_RAY_SUFFIX', 'BAND_DIAG_SUFFIX'
ARMS = (SCALAR, RAY, BAND_DIAG)


def prepare(root, parent, previous):
    from utils.diag_shared_prefix_feedback_multinoise import check

    plan = check(parent)
    prior, summary = read(previous/'plan.json'), read(previous/'summary.json')
    cases = select_cases('sr', plan)
    require(prior['experiment'] == summary['experiment'] == 'A113' and summary['status'] == 'complete'
            and summary['completed_cases'] == len(cases) and prior['a103_plan_sha256'] == digest(parent/'plan.json')
            and prior['cases'] and [e['case'] for e in prior['cases']] == cases,
            'Completed matching A113 required')
    require(plan['grid'] == protocol.grid(), 'Time grid differs')
    frozen = dict(experiment='A115', entries=prior['cases'], arms=list(ARMS), checkpoint=plan['checkpoint'],
                  grid=plan['grid'], configs=plan['configs'], a103_plan_sha256=digest(parent/'plan.json'),
                  a113_summary_sha256=digest(previous/'summary.json'), method_nfe=300, joint_forward_cap=7200,
                  card_sha256=digest(CARD), source_sha256=digest(__file__), controller_sha256=digest(band.__file__))
    path = root/'plan.json'
    if path.is_file():
        require(read(path) == frozen, 'Existing A115 plan differs')
    else:
        require(not root.exists() or root.is_dir() and not any(root.iterdir()), 'Unrecognized A115 result directory')
        root.mkdir(parents=True, exist_ok=True)
        write(path, frozen)
    if not (root/'experiment_card.md').exists():
        (root/'experiment_card.md').write_bytes(CARD.read_bytes())
    return frozen


def groups(rows):
    result = []
    for sigma in sorted({r['case']['sigma'] for r in rows}):
        cell = [r for r in rows if r['case']['sigma'] == sigma]
        means = {arm:dict(quality={k:mean(r['arms'][arm]['metrics'][k] for r in cell) for k in METRICS},
                          clean_H_mse=mean(r['risk'][arm]['clean_H_mse'] for r in cell),
                          mean_seconds=mean(r['arms'][arm]['method_equivalent_seconds'] for r in cell), nfe=300)
                 for arm in ARMS}
        comparisons = []
        for candidate, reference in ((RAY, SCALAR), (BAND_DIAG, RAY), (BAND_DIAG, SCALAR)):
            pairs = [(r['arms'][candidate]['metrics'], r['arms'][reference]['metrics']) for r in cell]
            delta = {k:mean(a[k]-b[k] for a,b in pairs) for k in METRICS}
            errors = {k:mean(r['risk'][candidate][k]-r['risk'][reference][k] for r in cell)
                      for k in cell[0]['risk'][candidate]}
            wins = sum(a['psnr'] > b['psnr'] for a,b in pairs)
            comparisons.append(dict(candidate=candidate, reference=reference, delta=delta, error_delta=errors,
                                    psnr_wins=wins, positive=len(cell) == 10 and wins >= 7 and delta['psnr'] >= .1
                                    and delta['ssim'] >= 0 and delta['lpips'] <= 0 and errors['clean_H_mse'] < 0))
        activity = {arm:{key:sum(r['activity'][arm][key] for r in cell)
                         for key in ('steps_with_signal', 'strict_diagonal_descents')}
                    for arm in (RAY, BAND_DIAG)}
        result.append(dict(sigma=sigma, images=len(cell), arms=means, comparisons=comparisons, activity=activity))
    return result


def readout(root, plan, projection):
    import torch
    from modules.main_single import load_image_as_tensor

    rows = []
    for entry in plan['entries']:
        case = entry['case']
        folder = root/'cases'/case['key']
        record = folder/'run.json'
        if not record.is_file():
            continue
        row = read(record)
        if row['status'] != 'complete':
            continue
        sha = digest(record)
        require(row['case'] == case and row['plan_sha256'] == digest(root/'plan.json'), 'Saved A115 case differs')
        path = folder/'diagnostic.json'
        if path.is_file():
            extra = read(path)
            require(extra['run_sha256'] == sha, 'Saved A115 diagnostic differs')
        else:
            require(digest(entry['observation']) == entry['observation_sha256']
                    and digest(case['image']) == case['image_sha256'], 'Evaluation input changed')
            y = torch.load(entry['observation'], map_location='cpu', weights_only=True)['y']
            gt = load_image_as_tensor(case['image']).cpu()
            extra = dict(run_sha256=sha, risk={}, activity={})
            for arm in ARMS:
                output = folder/(arm+'.pt')
                require(digest(output) == row['arms'][arm]['artifacts'][output.name], 'A115 output changed')
                prediction = torch.load(output, map_location='cpu', weights_only=True)
                extra['risk'][arm] = projection.risk(prediction, gt, y)
                if arm != SCALAR:
                    trace = read(folder/(arm+'_trace.json'))
                    require(len(trace) == 30, 'A115 trace incomplete')
                    extra['activity'][arm] = dict(steps_with_signal=sum(any(p['weights'].values()) for p in trace),
                        strict_diagonal_descents=sum(p['proxy_at_selected'] is not None and
                        p['proxy_at_selected'] < p['proxy_at_ray']-1e-9 for p in trace))
            write(path, extra)
        rows.append(dict(row, risk=extra['risk'], activity=extra['activity']))
    return rows


def emit(root, rows, status, **details):
    counts = [c for row in rows for c in (row['counts']['prefix'], *(row['arms'][a]['counts'] for a in ARMS))]
    result = dict(experiment='A115', status=status, completed_cases=len(rows), planned_cases=20,
                  groups=groups(rows), recorded_completed_case_cost={key:sum(c.get(key, 0) for c in counts)
                    for key in ('forward_attempts', 'forwards', 'optimizer_backward_attempts',
                                'input_vjp_attempts', 'input_jvp_attempts')},
                  cost_is_lower_bound=status != 'complete', **details)
    write(root/'summary.json', result)
    return result


def execute(root, parent, previous):
    import torch
    from modules.main_single import load_diffusion_model
    from utils import diag_shared_prefix_feedback as baseline, equal_nfe_feedback_runtime_r3 as bridge
    from utils import shared_prefix_feedback_runtime_multinoise as multi
    from utils.a54_session import exclusive
    from utils.a67_model_freeze import frozen_network
    from utils.diag_paper_clock_alignment import gpu_inventory

    require(all(not root.is_relative_to(p.resolve()) for p in (parent, previous)), 'Use independent A115 results')
    root.parent.mkdir(parents=True, exist_ok=True)
    with exclusive(root.with_name(root.name+'.lock')):
        plan = prepare(root, parent, previous)
        entries = {e['case']['key']:e for e in plan['entries']}
        projection = ObservableProjection(multi.make_operator(next(iter(entries.values()))['case'], 'cpu'), 'cpu')
        rows = readout(root, plan, projection)
        if len(rows) == 20:
            print('A115 SUMMARY:', json.dumps(emit(root, rows, 'complete'), ensure_ascii=False), flush=True)
            return
        finished = {r['case']['key'] for r in rows}
        for key in entries.keys() - finished:
            require(not (root/'cases'/key).exists(), f'Previous incomplete case retained at {root}/cases/{key}')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        require(visible and ',' not in visible and torch.cuda.is_available() and torch.cuda.device_count() == 1,
                'Expose one allocated GPU')
        gpu = protocol.available_gpus(gpu_inventory(), visible, visible)[0]['uuid']
        require(torch.cuda.mem_get_info(0)[0] >= 6*1024**3, 'Need 6 GiB free')
        require(digest(PROJECTS/'modules/models/ffhq_10m.pt') == plan['checkpoint']['sha256'], 'Checkpoint differs')

        def observation(_, case, device):
            entry = entries[case['key']]
            folder, _, y_path = source_case(parent, case)
            require(digest(folder/'run.json') == entry['parent_record_sha256']
                    and digest(y_path) == entry['observation_sha256'], 'Frozen A113 observation differs')
            return dict(path=str(y_path), sha256=entry['observation_sha256'])

        def suffix(model, operator, y, case, arm, prefix):
            if arm == SCALAR:
                return multi.run_suffix(model, operator, y, case, arm, prefix)
            with patch.object(bridge, 'exact_diagonal_with_native_straight_through', band.controller(arm == BAND_DIAG)):
                return multi.run_suffix(model, operator, y, case, DIAG, prefix)

        emit(root, rows, 'running')
        try:
            model = load_diffusion_model('ffhq', 'cuda:0')
            model.model.eval()
            with frozen_network(model.model), patch.object(baseline, 'rules', SimpleNamespace(ARMS=ARMS)), \
                    patch.object(baseline, 'runtime', SimpleNamespace(train_prefix=multi.train_prefix, run_suffix=suffix)), \
                    patch.object(baseline, 'operator', lambda _, c, device, count: multi.make_operator(c, device, count)), \
                    patch.object(baseline.fixtures, 'observation', observation):
                for key, entry in entries.items():
                    if key in finished:
                        continue
                    baseline.run_case(root, plan, entry['case'], model, gpu)
                    rows = readout(root, plan, projection)
                    emit(root, rows, 'running')
                    print(f'A115 COMPLETE: {len(rows)}/20 {key}', flush=True)
        except BaseException as error:
            emit(root, rows, 'failed', error=repr(error), traceback=traceback.format_exc())
            raise
        require(len(rows) == 20, 'Incomplete A115 panel')
        print('A115 SUMMARY:', json.dumps(emit(root, rows, 'complete'), ensure_ascii=False), flush=True)


def archive(root):
    from utils import archive_refinement_confirmation as exporter

    report = read(root/'summary.json')
    require(report['experiment'] == 'A115' and report['status'] == 'complete', 'A115 must be complete')
    exporter.BRANCH = BRANCH
    bundle, commit = exporter.export_and_commit({'A115': root}, projects=PROJECTS)
    print(f'Archive: {bundle} Results commit: {commit} Branch: {BRANCH}', flush=True)
    subprocess.run(['git', 'push', 'origin', BRANCH], cwd=PROJECTS.parent, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=A103)
    parser.add_argument('--a113-root', type=Path, default=A113)
    parser.add_argument('--root', type=Path, default=ROOT)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--status', action='store_true')
    actions.add_argument('--archive', action='store_true')
    args = parser.parse_args()
    if args.status:
        print(json.dumps(read(args.root/'summary.json'), ensure_ascii=False, indent=2))
    elif args.archive:
        archive(args.root)
    else:
        execute(args.root.resolve(), args.parent, args.a113_root)


if __name__ == '__main__':
    main()
