"""A72 paired suffix ablation: original A71 prefix, no new optimization."""
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
from utils import historical_ffhq_gain as files
from utils import shrink_feedback_ablation as rules
from utils import shrink_feedback_ablation_runtime as runtime
from utils import diag_shrink_multitask as parent
from utils import diag_noise_feasible_zaps as common
from utils import historical_ffhq_runtime as precision
from utils import paper_clock_runtime as clock
from utils.a67_model_freeze import frozen_network
from utils.diag_paper_clock_alignment import resource

ROOT = '/data0/lzy/ZAPS-results/diag_shrink_feedback_ablation/A72_20261029'
PARENT = parent.ROOT
BRANCH = 'results/shrink-feedback-ablation-20261029'
CARD = PROJECTS/'experiment_cards/A72_shrink_feedback_ablation_20261029.md'


def check(root):
    p = files.read(Path(root)/'plan.json')
    files.require(p['experiment'] == 'A72_shrink_feedback_ablation', 'Wrong plan')
    for n, h in p['sources'].items():
        files.require(files.digest(PROJECTS/n) == h, 'Bound source changed: '+n)
    files.require(files.digest(Path(p['parent'])/'plan.json') == p['parent_plan_sha256'], 'Parent plan changed')
    return p


def physical_folder(root, c, arm, record):
    if record.get('reused'):
        source = Path(record['source_record'])
        files.require(files.digest(source) == record['source_record_sha256'], 'Reused source record changed')
        return source.parent
    return common.record_path(root, c, arm).parent


def prepare(args):
    root, source = Path(args.root).resolve(), Path(args.a71_root).resolve()
    if (root/'plan.json').exists():
        check(root); print('A72 frozen plan reused.', flush=True); return
    files.require(not root.exists() or not any(root.iterdir()), 'Nonempty unplanned directory retained')
    files.require(not root.is_relative_to(source) and not source.is_relative_to(root), 'Parent overlap')
    old = parent.check(source)
    summary = files.read(source/'summary.json')
    files.require(summary['completed_runs'] == summary['planned_runs'] == 72, 'Need complete A71')
    cases, curves, references, inputs, caches = old['cases'], {}, {}, {}, {}
    for c in cases:
        key = c['key']; references[key] = {}
        for arm in rules.REFERENCE:
            r = common.completed(source, c, arm)
            files.require(r is not None, 'Missing parent reference')
            record = common.record_path(source, c, arm)
            inputs[str(record)] = files.digest(record)
            references[key][arm] = dict(metrics=r['paired_metrics'], full_seconds=r['complete_reconstruction_seconds'],
                timing_origin=r['timing_origin'], source_record=str(record))
            folder = physical_folder(source, c, arm, r)
            if arm == 'BASE':
                entry = folder/'entry.pt'
            else:
                expected = folder/'paired.pt'
                trace = record.parent/'paired_trace.json'
                rows = files.read(trace)
                files.require([v['t'] for v in rows] == old['grid'][::-1], 'Donor grid differs')
                curves[key] = [float(v['gamma']) for v in rows]
                inputs[str(trace)] = files.digest(trace)
        obs = source/'observations'/key/'observation.json'
        y = Path(files.read(obs)['path'])
        for f in (entry, expected, obs, y, Path(c['image'])):
            inputs[str(f)] = files.digest(f)
        caches[key] = dict(entry=str(entry), expected=str(expected), observation=str(y))
    inputs[str(source/'motion_kernel.npy')] = files.digest(source/'motion_kernel.npy')
    calibration = rules.calibrate(cases, curves)
    names = ['utils/shrink_feedback_ablation.py', 'utils/shrink_feedback_ablation_runtime.py',
             'utils/diag_shrink_feedback_ablation.py', 'utils/a72_session.py',
             'utils/test_shrink_feedback_ablation.py', 'utils/test_shrink_feedback_ablation_runtime.py',
             'run_a72_20261029.sh', str(CARD.relative_to(PROJECTS)), *old['sources']]
    sources = {n: files.digest(PROJECTS/n) for n in dict.fromkeys(names)}
    root.parent.mkdir(parents=True, exist_ok=True)
    files.require(shutil.disk_usage(root.parent).free >= files.GIB, 'Need1GiB free')
    root.mkdir(exist_ok=True)
    p = dict(experiment='A72_shrink_feedback_ablation', parent=str(source), parent_plan_sha256=files.digest(source/'plan.json'),
             checkpoint=old['checkpoint'], grid=old['grid'], cases=cases, calibration=calibration,
             references=references, caches=caches, inputs=inputs, sources=sources,
             code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECTS.parent, text=True).strip(),
             budget=dict(diagnostic_forwards=len(cases)*len(rules.EXECUTED)*30, optimizer_backward=0, input_vjp=0,
                         warmup_separate=True, reused_complete_A71_runs=72),
             protocol=dict(quality='Same original A71 BASE pre-tenth-unroll parameters, initial state and RNG; cached270 prefix.',
                 timing='Suffix seconds only for new arms; no independent complete300 wall-clock or NFE-saving claim.',
                 calibration='Same-task/sigma other TWO images; GT excluded; fixed scalar median60, profile median per step, cyclic mismatch.',
                 scope='Same three development FFHQ images, four tasks, three sigma, one seed. No extra images/seeds/dataset or tuning.'))
    files.write(root/'plan.json', p); shutil.copyfile(CARD, root/'experiment_card.md')
    print('A72 PREPARE/FREEZE:36 cases x (4 ablations +1 equivalence check) x30 =5400 diagnostic forwards;'
          '0 optimization/input backwards; original A71 and270-prefix caches READ ONLY.', flush=True)


def completed(root, c, arm):
    path = Path(root)/'cases'/c['key']/arm/'run.json'
    if not path.exists():
        return None
    r = files.read(path)
    files.require(r['status'] == 'complete', 'Started/failed attempt retained: '+str(path))
    files.require(r['plan_sha256'] == files.digest(Path(root)/'plan.json'), 'Attempt plan changed')
    for n, h in r['artifacts'].items():
        files.require(files.digest(path.parent/n) == h, 'Output artifact changed')
    return r


def measure(root, p, c, arm, model, uuid):
    import torch
    from modules.main_single import load_image_as_tensor
    from modules.dataset_loader import tensor_to_image
    if completed(root, c, arm):
        return True
    if arm != rules.CHECK:
        files.require(completed(root, c, rules.CHECK) is not None, 'Need matching current-policy replay first')
    if not resource(uuid, 'ffhq', torch.cuda.memory_reserved())['admitted']:
        print('A72 resource pause BEFORE attempt:', c['key'], arm, flush=True)
        return False
    folder = root/'cases'/c['key']/arm
    folder.mkdir(parents=True, exist_ok=False)
    r = dict(status='running', case=c, arm=arm, plan_sha256=files.digest(root/'plan.json'), artifacts={}, counts=None)
    files.write(folder/'run.json', r)
    def tick(count):
        r['counts'] = count; files.write(folder/'run.json', r)
    try:
        cache = p['caches'][c['key']]
        entry = torch.load(cache['entry'], map_location='cpu', weights_only=True)
        y = torch.load(cache['observation'], map_location='cpu', weights_only=True)['y'].to(model.device)
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        with precision.precision('LEGACY_IEEE'), clock.counters(model.model, tick) as counts:
            op = parent.make_operator(Path(p['parent']), c, model.device, counts)
            (out, trace), seconds = clock.clocks(torch.cuda.synchronize, lambda: runtime.replay(
                model, op, y, c, entry, arm, p['calibration'][c['key']]))
        r['counts'] = counts
        files.require(counts['forward_attempts'] == counts['forwards'] == 30 and counts['calls'] == p['grid'][::-1], 'Suffix NFE/grid mismatch')
        files.require(counts['optimizer_backward_attempts'] == counts['input_vjp_attempts'] == 0, 'Unexpected backward')
        if arm == rules.CHECK:
            expected = torch.load(cache['expected'], map_location='cpu', weights_only=True)
            error = float((out.double()-expected.double()).norm()/expected.double().norm().clamp(min=1e-30))
            r['relative_error_vs_A71'] = error
            files.require(error <= 1e-6, 'A71 current-policy replay differs; do not interpret ablations')
        gt = load_image_as_tensor(c['image']).cpu()
        with torch.no_grad():
            measurement_op = parent.make_operator(Path(p['parent']), c, model.device)
            residual = runtime.parent.informative(y-measurement_op.H(out.to(model.device)), measurement_op)
            residual_mse = float(residual.double().square().mean())
        torch.save(out, folder/'output.pt')
        tensor_to_image(out.squeeze(0), denormalize=True).save(folder/'output.png')
        files.write(folder/'trace.json', trace)
        r.update(status='complete', metrics=common.metrics(out, gt), measurement_residual_mse=residual_mse,
                 evaluation_H_calls=1, suffix_seconds=seconds, full_seconds=None,
                 timing_origin='Cached-prefix suffix only; not full reconstruction walltime',
                 peak_reserved_bytes=torch.cuda.max_memory_reserved(), cost_is_lower_bound=False,
                 artifacts={n: files.digest(folder/n) for n in ('output.pt', 'output.png', 'trace.json')})
        files.write(folder/'run.json', r)
        print('A72 COMPLETE:', c['key'], arm, 'PSNR=', round(r['metrics']['psnr'], 4),
              'suffix_s=', round(seconds, 3), 'F=30', flush=True)
        return True
    except BaseException as e:
        if 'counts' in locals():
            r['counts'] = counts
        r.update(status='failed', error=repr(e), traceback=traceback.format_exc(), cost_is_lower_bound=True)
        files.write(folder/'run.json', r); raise


def worker(args):
    import torch
    from modules.main_single import load_diffusion_model
    from utils.a54_session import exclusive
    root = Path(args.root); p = check(root); folder = root/'workers'/str(args.shard)
    folder.mkdir(parents=True, exist_ok=True)
    with exclusive(folder/'worker.lock'):
        if not resource(args.device_uuid, 'ffhq')['admitted']:
            files.write(folder/'status.json', dict(status='resource_limited')); return
        for n, h in p['inputs'].items():
            files.require(files.digest(n) == h, 'Read-only parent input changed: '+n)
        files.require(files.digest(p['checkpoint']['path']) == p['checkpoint']['sha256'], 'Checkpoint changed')
        start = time.perf_counter(); model = load_diffusion_model('ffhq', 'cuda:0'); model.model.eval()
        torch.cuda.synchronize()
        load = dict(status='running', load_seconds=time.perf_counter()-start, warmup_forward_attempts=0, uuid=args.device_uuid)
        f = folder/f'load_{len(list(folder.glob("load_*.json"))):03d}.json'; files.write(f, load)
        try:
            with frozen_network(model.model):
                load['warmup_forward_attempts'] = 1; files.write(f, load)
                with torch.no_grad(), precision.precision('LEGACY_IEEE'):
                    model._predict_eps(torch.zeros(1, 3, 256, 256, device=model.device), torch.tensor([999], device=model.device))
                torch.cuda.synchronize()
                for index, c in enumerate(p['cases']):
                    if c['shard'] != args.shard:
                        continue
                    # Equivalence is mandatory first; rotate control order for suffix timing.
                    order = rules.ARMS[index % len(rules.ARMS):] + rules.ARMS[:index % len(rules.ARMS)]
                    for arm in (rules.CHECK, *order):
                        if not measure(root, p, c, arm, model, args.device_uuid):
                            load['status'] = 'resource_limited'
                            files.write(folder/'status.json', dict(status='resource_limited'))
                            return
            load['status'] = 'complete'; files.write(folder/'status.json', dict(status='complete'))
        except BaseException as e:
            load.update(status='failed', error=repr(e)); files.write(folder/'status.json', load); raise
        finally:
            files.write(f, load)


def summarize(root):
    root = Path(root); p = check(root)
    rows = [files.read(f) for f in (root/'cases').glob('*/*/run.json')]
    done = {(r['case']['key'], r['arm']): r for r in rows if r['status'] == 'complete'}
    groups, comparisons, table = [], [], []
    for task in parent.rule.TASKS:
        for sigma in parent.rule.SIGMAS:
            cases = [c for c in p['cases'] if c['task'] == task and c['sigma'] == sigma]
            for arm in (*rules.REFERENCE, *rules.ARMS):
                selected = []
                for c in cases:
                    if arm in rules.REFERENCE:
                        v = p['references'][c['key']][arm]
                        row = dict(case=c, arm=arm, metrics=v['metrics'], suffix_seconds=None,
                                   full_seconds=v['full_seconds'], timing_origin=v['timing_origin'])
                    else:
                        row = done.get((c['key'], arm))
                    if row:
                        selected.append(row)
                        table.append(dict(task=task, sigma=sigma, image_id=c['image_id'], arm=arm,
                            **{k: row['metrics'][k] for k in ('psnr', 'ssim', 'lpips')},
                            full_seconds=row['full_seconds'], suffix_seconds=row['suffix_seconds']))
                if len(selected) == 3:
                    groups.append(dict(task=task, sigma=sigma, arm=arm, cases=3,
                        quality={k: statistics.mean(v['metrics'][k] for v in selected) for k in ('psnr', 'ssim', 'lpips')},
                        mean_full_seconds=(statistics.mean(v['full_seconds'] for v in selected) if arm in rules.REFERENCE else None),
                        mean_suffix_seconds=(statistics.mean(v['suffix_seconds'] for v in selected) if arm not in rules.REFERENCE else None)))
            for arm in rules.ARMS:
                if not all((c['key'], arm) in done for c in cases):
                    continue
                deltas = [{k: p['references'][c['key']]['SHRINK']['metrics'][k] - done[c['key'], arm]['metrics'][k]
                           for k in ('psnr', 'ssim', 'lpips')} for c in cases]
                delta = {k: statistics.mean(v[k] for v in deltas) for k in ('psnr', 'ssim', 'lpips')}
                wins = sum(v['psnr'] > 0 for v in deltas)
                comparisons.append(dict(task=task, sigma=sigma, full_minus=arm, delta=delta,
                    images_PSNR_positive=wins, worst_PSNR=min(v['psnr'] for v in deltas),
                    engineering_support=delta['psnr'] >= .05 and wins >= 2 and delta['ssim'] >= -.002 and delta['lpips'] <= .005))
    loads = [files.read(f) for f in (root/'workers').glob('*/load_*.json')]
    counts = [r.get('counts') or {} for r in rows]
    result = dict(experiment=p['experiment'], classification='DEVELOPMENT_ONLINE_FEEDBACK_ABLATION',
        completed_replays=len(done), planned_replays=180, image_clusters=3, reused_complete_A71_runs=72,
        groups=groups, comparisons=comparisons,
        actual_new_cost=dict(diagnostic_forward_attempts=sum(v.get('forward_attempts', 0) for v in counts),
            optimizer_backward_attempts=sum(v.get('optimizer_backward_attempts', 0) for v in counts),
            input_vjp_attempts=sum(v.get('input_vjp_attempts', 0) for v in counts),
            warmup_forward_attempts=sum(v['warmup_forward_attempts'] for v in loads)),
        cost_is_lower_bound=any(r['status'] != 'complete' for r in rows), limits=p['protocol'])
    files.write(root/'summary.json', result)
    if table:
        with (root/'comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    lines = ['# A72：完整在线规则相对消融的配对差值', '',
             '三张固定开发图；正PSNR/SSIM、负LPIPS有利。新臂仅计末轮回放时间，完整300时钟不补算。', '',
             '|任务|sigma|对照|ΔPSNR|ΔSSIM|ΔLPIPS|正向图数|工程信号|', '|---|---:|---|---:|---:|---:|---:|---|']
    for r in comparisons:
        d = r['delta']
        lines.append(f"|{r['task']}|{r['sigma']}|{r['full_minus']}|{d['psnr']:+.4f}|{d['ssim']:+.5f}|{d['lpips']:+.5f}|{r['images_PSNR_positive']}/3|{r['engineering_support']}|")
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    brief = {k: v for k, v in result.items() if k != 'groups'}
    print('A72 Summary:', json.dumps(brief, ensure_ascii=False), flush=True)


def report(root):
    summarize(root)


def archive(root):
    from utils import archive_refinement_confirmation as exporter
    exporter.BRANCH = BRANCH
    bundle, commit = exporter.export_and_commit({'A72': Path(root)}, projects=PROJECTS)
    print('Archive:', bundle, 'Results commit:', commit, 'Branch:', BRANCH, flush=True)
    subprocess.run(['git', 'push', 'origin', BRANCH], cwd=PROJECTS.parent, check=True)
    print('RESULTS_PUSH_PASS: A72 ablations uploaded.', flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=ROOT); p.add_argument('--a71-root', default=PARENT)
    p.add_argument('--shard', type=int, choices=(0, 1)); p.add_argument('--device-uuid')
    p.add_argument('--check-worker-cli', action='store_true', help=argparse.SUPPRESS)
    group = p.add_mutually_exclusive_group(required=True)
    for n in ('prepare', 'worker', 'summarize', 'report', 'archive'):
        group.add_argument('--'+n, action='store_true')
    return p


def main():
    p = parser(); a = p.parse_args()
    if a.worker and (a.shard is None or not a.device_uuid):
        p.error('Worker requires explicit shard and device UUID')
    if a.check_worker_cli:
        files.require(a.worker, 'Need worker')
        print(json.dumps(dict(shard=a.shard, device_uuid=a.device_uuid, model_calls=0))); return
    if a.prepare:
        prepare(a)
    elif a.worker:
        worker(a)
    elif a.summarize or a.report:
        summarize(a.root)
    else:
        archive(a.root)


if __name__ == '__main__':
    main()
