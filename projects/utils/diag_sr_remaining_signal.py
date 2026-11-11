"""A113 tests noise subtraction on the residual left after scalar fitting."""
import argparse
import json
import math
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack, nullcontext
from unittest.mock import patch

PROJECTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECTS), str(PROJECTS.parent)]

from utils.diag_300nfe_feedback_confirmation import ARMS as ORIGINAL, ROOTS, ObservableProjection, select_cases
from utils.diag_sr_inpainting_followup import A103, METRICS, source_case
from utils.diag_sr_same_state_noise_attribution import ROOT as A112
from utils.paper_clock_alignment import digest, grid, read, require, write

ROOT = A103.with_name('A113_20261111')
CARD = PROJECTS/'experiment_cards/A113_sr_remaining_signal.md'
BRANCH = 'results/zaps-sr-remaining-signal-20261111'
SCALAR, DIAG = ORIGINAL[1:]
CORRECTED = 'REMAINING_SIGNAL_DIAG_SUFFIX'
ARMS = (SCALAR, DIAG, CORRECTED)
COST_KEYS = ('forward_attempts', 'forwards', 'optimizer_backward_attempts',
             'input_vjp_attempts', 'input_jvp_attempts')


def legacy_workers(root):
    """Detect the original unlocked Python entry on the Linux GPU host."""
    workers = []
    proc = Path('/proc')
    if not proc.is_dir():
        return workers
    for process in proc.iterdir():
        if not process.name.isdecimal() or int(process.name) == os.getpid():
            continue
        try:
            command = os.fsdecode((process/'cmdline').read_bytes()).rstrip('\0').split('\0')
            if not Path(command[0]).name.startswith('python'):
                continue
            is_script = len(command) > 1 and any(Path(a).name == Path(__file__).name for a in command[1:])
            is_module = '-m' in command and 'utils.diag_sr_remaining_signal' in command
            if not (is_script or is_module) or {'--status', '--archive'}.intersection(command):
                continue
            target = ROOT
            for i, argument in enumerate(command):
                if argument == '--root' and i+1 < len(command):
                    target = Path(command[i+1])
                elif argument.startswith('--root='):
                    target = Path(argument.partition('=')[2])
            if not target.is_absolute():
                target = (process/'cwd').resolve(strict=True)/target
            if target.resolve() == root.resolve():
                workers.append(int(process.name))
        except (OSError, RuntimeError):
            continue
    return workers


def interrupted_record(record):
    return record.get('status') == 'running' or record.get('error') == 'KeyboardInterrupt()'


def read_completed_cases(root, entries, descending):
    from utils.diag_shared_prefix_feedback import validate

    completed, interrupted = [], []
    for entry in entries:
        case = entry['case']
        directory = root/'cases'/case['key']
        if not directory.exists():
            continue
        record_path = directory/'run.json'
        if not record_path.is_file() and {p.name for p in directory.iterdir()} <= {'run.json.tmp'}:
            interrupted.append(directory)
            continue
        record = read(record_path)
        require(record['case'] == case, f'Case identity differs at {directory}')
        if interrupted_record(record):
            interrupted.append(directory)
            continue
        require(record['status'] == 'complete', f'Prior case failed at {directory}: {record.get("error")}')
        validate(record['counts']['prefix'], 270, 9)
        require(record['counts']['prefix']['calls'] == descending*9, 'Saved prefix grid differs')
        for arm in ARMS:
            count = record['counts'][arm]
            validate(count, 30, 1)
            require(count['calls'] == descending and (directory/(arm+'.pt')).is_file()
                    and arm in record['arms'], f'Incomplete saved output at {directory}/{arm}')
        require(len(record['updates']) == 30, 'Saved correction trace incomplete')
        completed.append(record)
    return completed, interrupted


def preserve_interrupted(root, paths):
    summary = root/'summary.json'
    if not paths and not (summary.is_file() and interrupted_record(read(summary))):
        return
    attempt = root/'interrupted'/str(time.time_ns())
    attempt.mkdir(parents=True)
    if summary.is_file():
        (attempt/'previous_summary.json').write_bytes(summary.read_bytes())
    for path in paths:
        path.rename(attempt/path.name)
    print(f'A113 retained interrupted case records at {attempt}', flush=True)


def print_summary(report):
    print('A113 SUMMARY:', json.dumps(report, ensure_ascii=False), flush=True)


def recorded_cost(rows):
    counts = [count for row in rows for count in row['counts'].values()]
    return {key:sum(count.get(key, 0) for count in counts) for key in COST_KEYS}


def run_or_reuse(parent, audit, root):
    from utils.a54_session import exclusive

    root = root.resolve()
    require(all(not root.is_relative_to(p.resolve()) for p in (parent, audit)),
            'Use an independent A113 result directory')
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = root.with_name(root.name+'.lock')
    # Prior revisions did not hold a lock. The updated entry uses an exclusive lock.
    with ExitStack() as scope:
        try:
            scope.enter_context(exclusive(lock))
        except BlockingIOError as exc:
            raise SystemExit(f'A113 is already running under {lock}; use --status to read progress') from exc
        while root.exists():
            summary_path = root/'summary.json'
            summary = read(summary_path) if summary_path.is_file() else {}
            if summary.get('status') == 'complete':
                require(summary.get('experiment') == 'A113'
                        and summary['completed_cases'] == summary['planned_cases'] == 20,
                        'Existing summary differs from a completed A113 run')
                plan = read(root/'plan.json')
                require(plan.get('experiment') == 'A113'
                        and Path(plan['parent']).resolve() == parent.resolve()
                        and Path(plan['audit']).resolve() == audit.resolve(),
                        'Completed A113 uses different input directories')
                print_summary(summary)
                return
            if summary.get('status') == 'failed' and not interrupted_record(summary):
                print_summary(summary)
                raise ValueError(f'Existing A113 recorded a failure: {summary.get("error")}')
            workers = legacy_workers(root)
            if not workers:
                break
            print(f'A113 waiting for existing worker {workers}; '
                  f'{summary.get("completed_cases", 0)}/20 complete', flush=True)
            time.sleep(15)
        execute(parent, audit, root)


def remaining_signal(scalar, size, sigma):
    require(size > 1 and math.isfinite(sigma) and sigma > 0, 'Invalid noise settings')
    omega = scalar['weight']
    if not scalar['resolved']:
        return dict(beta=0., residual_energy=None, noise_energy=None, remaining_weight=0.)
    dot = scalar['residual_response_dot']
    response = scalar['response_energy']
    fitted = dot > 0
    energy = max(0., scalar['residual_energy'] - (dot**2/response if fitted else 0.))
    noise = (size-int(fitted))*sigma**2
    weight = max(0., 1.-noise/energy) if energy > 0 else 0.
    beta = min(1., weight/omega) if omega > 0 else 0.
    return dict(beta=beta, residual_energy=energy, noise_energy=noise, remaining_weight=weight)


def corrected_update(original):
    def update(zaps, operator, clean, q, residual, alpha, sigma, counts):
        import torch

        full, detail, counts = original(zaps, operator, clean, q, residual, alpha, sigma, counts)
        scalar = detail['scalar']['gamma']*q
        audit = remaining_signal(detail['scalar'], residual.numel(), sigma)
        beta = audit['beta']
        with torch.no_grad():
            extra = full.detach()-scalar.detach()
            audit['increment_norm'] = float(extra.double().norm())
            audit['applied_increment_norm'] = beta*audit['increment_norm']
        detail['remaining_signal'] = audit
        if beta == 0:
            return scalar, detail, counts
        if beta == 1:
            return full, detail, counts
        return scalar + beta*extra, detail, counts
    return update


def summarize(rows):
    groups = []
    for sigma in sorted({r['case']['sigma'] for r in rows}):
        cases = [r for r in rows if r['case']['sigma'] == sigma]
        arms = {arm:dict(quality={k:mean(r['arms'][arm]['quality'][k] for r in cases) for k in METRICS},
                         mean_seconds=mean(r['arms'][arm]['seconds'] for r in cases), nfe=300)
                for arm in ARMS}
        comparisons = []
        for cand, ref in ((DIAG, SCALAR), (CORRECTED, DIAG), (CORRECTED, SCALAR)):
            pairs = [(r['arms'][cand], r['arms'][ref]) for r in cases]
            delta = {k:mean(a['quality'][k]-b['quality'][k] for a,b in pairs) for k in METRICS}
            errors = {k:mean(a['errors'][k]-b['errors'][k] for a,b in pairs) for k in pairs[0][0]['errors']}
            wins = sum(a['quality']['psnr'] > b['quality']['psnr'] for a,b in pairs)
            comparisons.append(dict(candidate=cand, reference=ref, delta=delta, error_delta=errors,
                psnr_wins=wins, positive=len(cases) == 10 and wins >= 7 and delta['psnr'] >= .1
                and delta['ssim'] >= 0 and delta['lpips'] <= 0 and errors['clean_H_mse'] < 0))
        points = [p for r in cases for p in r['updates']]
        stage = {}
        for label, start, stop in (('first10', 0, 10), ('middle10', 10, 20), ('last10', 20, 30)):
            stages = [p for p in points if start <= p['step'] < stop]
            stage[label] = dict(mean_beta=mean(p['beta'] for p in stages),
                                steps_with_increment=sum(p['applied_increment_norm'] > 1e-10 for p in stages))
        groups.append(dict(sigma=sigma, images=len(cases), arms=arms, comparisons=comparisons,
                           correction=dict(steps=len(points), beta_zero=sum(p['beta'] == 0 for p in points),
                                           beta_one=sum(p['beta'] == 1 for p in points), by_stage=stage)))
    return groups


def execute(parent, audit, root):
    import torch
    from modules.main_single import load_diffusion_model, load_image_as_tensor
    from utils import equal_nfe_feedback_runtime_r3 as bridge, historical_ffhq_runtime as precision
    from utils import paper_clock_runtime as clock
    from utils.a67_model_freeze import frozen_network
    from utils.diag_noise_feasible_zaps import metrics
    from utils.diag_shared_prefix_feedback import rng_equal, validate
    from utils.diag_shared_prefix_feedback_multinoise import check
    from utils.shared_prefix_feedback_runtime_multinoise import make_operator, run_suffix, train_prefix

    plan = check(parent)
    cases = select_cases('sr', plan)
    prior, prior_plan = read(audit/'summary.json'), read(audit/'plan.json')
    require(prior.get('experiment') == 'A112' and prior['status'] == 'complete'
            and prior['completed_cases'] == len(cases)
            and prior_plan['a103_plan_sha256'] == digest(parent/'plan.json')
            and [e['case'] for e in prior_plan['cases']] == cases, 'Completed matching A112 required')
    require(plan['grid'] == grid(), 'Grid differs')
    identity = dict(experiment='A113', cases=prior_plan['cases'], grid=plan['grid'],
          checkpoint=plan['checkpoint'], arms=list(ARMS),
          a103_plan_sha256=digest(parent/'plan.json'), a112_plan_sha256=digest(audit/'plan.json'),
          a112_summary_sha256=digest(audit/'summary.json'), joint_forward_cap=7200)
    plan_path = root/'plan.json'
    if plan_path.is_file():
        saved = read(plan_path)
        require(all(saved.get(k) == v for k, v in identity.items()), 'Existing A113 plan differs')
    else:
        require(not root.exists() or root.is_dir() and not any(root.iterdir()),
                'Existing result directory lacks an A113 plan and contains files')
    descending = plan['grid'][::-1]
    rows, interrupted = read_completed_cases(root, prior_plan['cases'], descending)
    finished_keys = {row['case']['key'] for row in rows}
    pending = [e for e in prior_plan['cases'] if e['case']['key'] not in finished_keys]
    if pending:
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expose one allocated GPU')
        require(torch.cuda.mem_get_info(0)[0] >= 6 * 1024**3, 'Need 6 GiB free')
        require(digest(PROJECTS/'modules/models/ffhq_10m.pt') == plan['checkpoint']['sha256'],
                'Checkpoint differs')
    root.mkdir(parents=True, exist_ok=True)
    if not plan_path.is_file():
        write(plan_path, dict(identity, parent=str(parent), audit=str(audit),
              code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECTS, text=True).strip(),
              source_sha256=digest(__file__), card_sha256=digest(CARD)))
    if not (root/'experiment_card.md').exists():
        require(digest(CARD) == read(plan_path)['card_sha256'], 'Saved experiment card hash differs')
        (root/'experiment_card.md').write_bytes(CARD.read_bytes())
    preserve_interrupted(root, interrupted)
    attempts = [path for path in (root/'interrupted').glob('*/*') if path.is_dir()]
    prior_cost = recorded_cost([read(path/'run.json') for path in attempts if (path/'run.json').is_file()])
    report = dict(experiment='A113', status='running', planned_cases=len(cases), completed_cases=len(rows),
                  groups=summarize(rows), retained_interrupted_cases=len(attempts), interrupted_recorded_cost=prior_cost)
    write(root/'summary.json', report)
    if not pending:
        report.update(status='complete', actual_cost=recorded_cost(rows), cost_is_lower_bound=bool(attempts))
        write(root/'summary.json', report)
        print_summary(report)
        return
    if rows:
        print(f'A113 reusing {len(rows)}/20 completed cases; running {len(pending)} pending', flush=True)
    active = None
    controller = corrected_update(bridge.exact_diagonal_with_native_straight_through)
    try:
        model = load_diffusion_model('ffhq', 'cuda:0')
        model.model.eval()
        with frozen_network(model.model), precision.precision('LEGACY_IEEE'):
            projection = ObservableProjection(make_operator(cases[0], model.device), model.device)
            for entry in pending:
                case = entry['case']
                dest = root/'cases'/case['key']
                active = dict(status='running', case=case, counts={}, arms={})
                write(dest/'run.json', active)
                folder, _, y_path = source_case(parent, case)
                require(digest(folder/'run.json') == entry['parent_record_sha256'] and
                        digest(y_path) == entry['observation_sha256'], 'A112 observation or case changed')
                y = torch.load(y_path, map_location='cpu', weights_only=True)['y'].to(model.device)
                with clock.counters(model.model) as count:
                    active['counts']['prefix'] = count
                    op = make_operator(case, model.device, count=count)
                    prefix, prefix_seconds = clock.clocks(torch.cuda.synchronize,
                        lambda: train_prefix(model, op, y, case))
                validate(count, 270, 9)
                require(count['calls'] == descending*9, 'Prefix grid differs')
                outputs, final_rng = {}, None
                for arm in ARMS:
                    scope = (patch.object(bridge, 'exact_diagonal_with_native_straight_through', controller)
                             if arm == CORRECTED else nullcontext())
                    with scope, clock.counters(model.model) as count:
                        active['counts'][arm] = count
                        op = make_operator(case, model.device, count=count)
                        (output, detail), seconds = clock.clocks(torch.cuda.synchronize,
                            lambda: run_suffix(model, op, y, case, DIAG if arm == CORRECTED else arm, prefix))
                    validate(count, 30, 1)
                    require(count['calls'] == descending, 'Suffix grid differs')
                    if final_rng is not None:
                        require(rng_equal(final_rng, detail['final_rng']), 'Suffix RNG differs')
                    final_rng = detail['final_rng']
                    require(bool(torch.isfinite(output).all()), 'Nonfinite output')
                    outputs[arm] = output
                    active['arms'][arm] = dict(seconds=prefix_seconds+seconds, suffix_seconds=seconds,
                                               module_counts=detail['module_counts'])
                    write(dest/(arm+'_trace.json'), detail['trace'])
                    if arm == CORRECTED:
                        require(len(detail['trace']) == 30, 'Incomplete correction trace')
                        active['updates'] = [dict(step=p['step'], **p['remaining_signal']) for p in detail['trace']]
                gt = load_image_as_tensor(case['image']).cpu()
                for arm, output in outputs.items():
                    active['arms'][arm].update(quality=metrics(output, gt), errors=projection.risk(output, gt, y))
                    torch.save(output, dest/(arm+'.pt'))
                active['status'] = 'complete'
                write(dest/'run.json', active)
                rows.append(active)
                active = None
                report.update(completed_cases=len(rows), groups=summarize(rows))
                write(root/'summary.json', report)
                print(f'A113 COMPLETE: {len(rows)}/{len(cases)} {case["key"]}', flush=True)
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        if active is not None:
            active.update(status='failed', error=repr(error))
            write(root/'cases'/active['case']['key']/'run.json', active)
        raise
    finally:
        attempted = rows + ([active] if active is not None else [])
        report['actual_cost'] = recorded_cost(attempted)
        report['cost_is_lower_bound'] = report['status'] != 'complete' or bool(attempts)
        write(root/'summary.json', report)
    print_summary(report)


def archive(root):
    from utils import archive_refinement_confirmation as exporter

    report = read(root/'summary.json')
    require(report.get('experiment') == 'A113' and report['status'] == 'complete',
            'A113 must be complete for archive')
    exporter.BRANCH = BRANCH
    bundle, commit = exporter.export_and_commit({'A113': root}, projects=PROJECTS)
    print(f'Archive: {bundle} Results commit: {commit} Branch: {BRANCH}', flush=True)
    subprocess.run(['git', 'push', 'origin', BRANCH], cwd=PROJECTS.parent, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=A103)
    parser.add_argument('--audit', type=Path, default=A112)
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
        run_or_reuse(args.parent, args.audit, args.root)


if __name__ == '__main__':
    main()
