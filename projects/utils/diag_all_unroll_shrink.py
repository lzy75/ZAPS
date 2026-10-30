"""A73 independent full300 all-unroll feedback and full-clock comparison."""
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
from utils import all_unroll_shrink as rules
from utils import all_unroll_shrink_runtime as runtime
from utils import diag_shrink_multitask as parent
from utils import diag_noise_feasible_zaps as common
from utils import historical_ffhq_runtime as precision
from utils import paper_clock_runtime as clock
from utils.a67_model_freeze import frozen_network
from utils.diag_paper_clock_alignment import resource

ROOT = '/data0/lzy/ZAPS-results/diag_all_unroll_shrink/A73_20261030'
PARENT = parent.ROOT
BRANCH = 'results/all-unroll-shrink-20261030'
CARD = PROJECTS/'experiment_cards/A73_all_unroll_shrink_20261030.md'


def check(root):
    p = files.read(Path(root)/'plan.json')
    files.require(p['experiment'] == 'A73_all_unroll_shrink', 'Wrong plan')
    for n, h in p['sources'].items():
        files.require(files.digest(PROJECTS/n) == h, 'Bound source changed: '+n)
    files.require(files.digest(Path(p['parent'])/'plan.json') == p['parent_plan_sha256'], 'Parent plan changed')
    return p


def prepare(args):
    root, source = Path(args.root).resolve(), Path(args.a71_root).resolve()
    if (root/'plan.json').exists():
        check(root); print('A73 frozen plan reused.', flush=True); return
    files.require(not root.exists() or not any(root.iterdir()), 'Nonempty unplanned directory retained')
    files.require(not root.is_relative_to(source) and not source.is_relative_to(root), 'Parent overlap')
    old = parent.check(source)
    summary = files.read(source/'summary.json')
    files.require(summary['completed_runs'] == summary['planned_runs'] == 72, 'Need completed A71')
    cases = rules.cases(old['cases']); inputs = {}; refs = {}; observations = {}
    for c in cases:
        obs = source/'observations'/c['key']/'observation.json'
        y = Path(files.read(obs)['path']); observations[c['key']] = str(y)
        for f in (obs, y, Path(c['image'])):
            inputs[str(f)] = files.digest(f)
        refs[c['key']] = {}
        for arm in ('BASE', 'SHRINK'):
            r = common.completed(source, c, arm)
            files.require(r is not None, 'Missing complete A71 reference')
            refs[c['key']][arm] = dict(metrics=r['paired_metrics'], full_seconds=r['complete_reconstruction_seconds'], timing_origin=r['timing_origin'])
    inputs[str(source/'motion_kernel.npy')] = files.digest(source/'motion_kernel.npy')
    names = ['utils/all_unroll_shrink.py', 'utils/all_unroll_shrink_runtime.py',
        'utils/diag_all_unroll_shrink.py', 'utils/a73_session.py', 'utils/test_all_unroll_shrink.py',
        'utils/test_all_unroll_shrink_runtime.py', 'run_a73_20261030.sh', str(CARD.relative_to(PROJECTS)), *old['sources']]
    root.parent.mkdir(parents=True, exist_ok=True)
    files.require(shutil.disk_usage(root.parent).free >= files.GIB, 'Need1GiB free')
    root.mkdir(exist_ok=True)
    p = dict(experiment='A73_all_unroll_shrink', parent=str(source), parent_plan_sha256=files.digest(source/'plan.json'),
        cases=cases, observations=observations, checkpoint=old['checkpoint'], grid=old['grid'], inputs=inputs,
        sources={n:files.digest(PROJECTS/n) for n in dict.fromkeys(names)}, references=refs,
        configs={t:parent.rule.configuration(t) for t in rules.TASKS}, arms=list(rules.ARMS),
        budget=dict(method_forwards=7200, optimizer_backwards=240, input_vjp=0, diagnostic_forwards=0, warmup_separate=True),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECTS.parent,text=True).strip(),
        protocol=dict(scope='Original3 A71 FFHQ images x4 tasks, sigma=.05, same seed/observations. No new image/seed/dataset.',
            policy='Unchanged A71 shrink formula, computed on all30 points in all10 unrolls. Gamma detached; original zeta/D graph retained.',
            learning='Independent from-seed training per arm, no learned cache. Control and optimization interact.',
            timing='Independent complete300 reconstruction, includes input loading/operator/init/optimization/CPU output sync; excludes model loading/warmup/metrics/disk writes.',
            output='Tenth-unroll last_opt; no extra sampling or best-epoch selection. Baseline all gamma1.'))
    files.write(root/'plan.json',p); shutil.copyfile(CARD,root/'experiment_card.md')
    print('A73 PREPARE/FREEZE:3 images x4 tasks x2 arms, sigma=.05;7200 method forwards,240 optimizer backwards,0 input VJP.',flush=True)


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
    if completed(root,c,arm):return True
    if not resource(uuid,'ffhq',torch.cuda.memory_reserved())['admitted']:
        print('A73 resource pause BEFORE attempt:',c['key'],arm,flush=True);return False
    folder=root/'cases'/c['key']/arm;folder.mkdir(parents=True,exist_ok=False)
    r=dict(status='running',case=c,arm=arm,plan_sha256=files.digest(root/'plan.json'),artifacts={},counts=None)
    files.write(folder/'run.json',r)
    def tick(count):
        r['counts']=count;files.write(folder/'run.json',r)
        print('A73 PROGRESS:',c['key'],arm,'F=',count['forwards'],'/300',flush=True)
    try:
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        with precision.precision('LEGACY_IEEE'),clock.counters(model.model,tick) as counts:
            def operation():
                y=torch.load(p['observations'][c['key']],map_location='cpu',weights_only=True)['y'].to(model.device)
                op=parent.make_operator(Path(p['parent']),c,model.device,counts)
                return runtime.execute(model,op,y,c,arm)
            (out,detail),seconds=clock.clocks(torch.cuda.synchronize,operation)
        r['counts']=counts
        files.require(counts['forward_attempts']==counts['forwards']==300 and counts['calls']==p['grid'][::-1]*10,'Full300 NFE/grid mismatch')
        files.require(counts['optimizer_backward_attempts']==counts['optimizer_backward_completed']==10 and counts['input_vjp_attempts']==0,'Backward count mismatch')
        if arm=='ALL_UNROLLS':files.require(len(detail['trace'])==300 and all(sum(v['unroll']==u for v in detail['trace'])==30 for u in range(1,11)),'Missing controlled unroll')
        gt=load_image_as_tensor(c['image']).cpu();metrics=common.metrics(out,gt)
        torch.save(out,folder/'output.pt');torch.save(dict(zeta=detail['zeta'],D=detail['D']),folder/'final_parameters.pt')
        tensor_to_image(out.squeeze(0),denormalize=True).save(folder/'output.png')
        files.write(folder/'trace.json',detail['trace']);files.write(folder/'loss.json',detail['losses'])
        r.update(status='complete',metrics=metrics,full_seconds=seconds,suffix_seconds=None,
            timing_origin='A73 fresh independent complete300 run',peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            cost_is_lower_bound=False,
            delta_vs_A71_base={k:metrics[k]-p['references'][c['key']]['BASE']['metrics'][k] for k in ('psnr','ssim','lpips')},
            artifacts={n:files.digest(folder/n) for n in ('output.pt','final_parameters.pt','output.png','trace.json','loss.json')})
        files.write(folder/'run.json',r)
        print('A73 COMPLETE:',c['key'],arm,'PSNR=',round(metrics['psnr'],4),'full_seconds=',round(seconds,3),'F=300',flush=True)
        return True
    except BaseException as e:
        if 'counts' in locals():r['counts']=counts
        r.update(status='failed',error=repr(e),traceback=traceback.format_exc(),cost_is_lower_bound=True)
        files.write(folder/'run.json',r);raise


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
                    # Alternate full-run order within each GPU shard.
                    shift = (index // 2) % len(rules.ARMS)
                    order = rules.ARMS[shift:] + rules.ARMS[:shift]
                    for arm in order:
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
    root=Path(root);p=check(root);rows=[files.read(f) for f in (root/'cases').glob('*/*/run.json')]
    done={(r['case']['key'],r['arm']):r for r in rows if r['status']=='complete'};groups=[];comparisons=[];table=[]
    for task in rules.TASKS:
        cases=[c for c in p['cases'] if c['task']==task]
        for arm in rules.ARMS:
            data=[done[c['key'],arm] for c in cases if (c['key'],arm) in done]
            for r in data:table.append(dict(task=task,image_id=r['case']['image_id'],arm=arm,**{k:r['metrics'][k] for k in ('psnr','ssim','lpips')},full_seconds=r['full_seconds']))
            if len(data)==3:groups.append(dict(task=task,arm=arm,cases=3,quality={k:statistics.mean(r['metrics'][k] for r in data) for k in ('psnr','ssim','lpips')},mean_full_seconds=statistics.mean(r['full_seconds'] for r in data),full_NFE=300))
        if all((c['key'],arm) in done for c in cases for arm in rules.ARMS):
            base=[done[c['key'],'BASE'] for c in cases];new=[done[c['key'],'ALL_UNROLLS'] for c in cases]
            delta={k:statistics.mean(n['metrics'][k]-b['metrics'][k] for b,n in zip(base,new)) for k in ('psnr','ssim','lpips')}
            comparisons.append(dict(task=task,sigma=.05,ALL_minus_BASE=delta,images_PSNR_positive=sum(n['metrics']['psnr']>b['metrics']['psnr'] for b,n in zip(base,new)),
                worst_PSNR=min(n['metrics']['psnr']-b['metrics']['psnr'] for b,n in zip(base,new)),
                full_clock_ratio=statistics.mean(n['full_seconds'] for n in new)/statistics.mean(b['full_seconds'] for b in base),
                ALL_minus_historical_last30={k:statistics.mean(n['metrics'][k]-p['references'][c['key']]['SHRINK']['metrics'][k] for c,n in zip(cases,new)) for k in ('psnr','ssim','lpips')}))
    counts=[r.get('counts') or {} for r in rows];loads=[files.read(f) for f in (root/'workers').glob('*/load_*.json')]
    result=dict(experiment=p['experiment'],classification='DEVELOPMENT_ALL_UNROLL_CONTROL_COMPARISON',completed_runs=len(done),planned_runs=24,image_clusters=3,groups=groups,comparisons=comparisons,
        actual_new_cost=dict(method_forward_attempts=sum(c.get('forward_attempts',0) for c in counts),optimizer_backward_attempts=sum(c.get('optimizer_backward_attempts',0) for c in counts),input_vjp_attempts=sum(c.get('input_vjp_attempts',0) for c in counts),warmup_forward_attempts=sum(v['warmup_forward_attempts'] for v in loads)),cost_is_lower_bound=any(r['status']!='complete' for r in rows),limits=p['protocol'])
    files.write(root/'summary.json',result)
    if table:
        with (root/'comparison.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    lines=['# A73：全程倍率 vs 原ZAPS完整运行','', '|任务|方法|PSNR|SSIM|LPIPS|完整秒数|NFE|','|---|---|---:|---:|---:|---:|---:|']
    for r in groups:
        q=r['quality'];lines.append(f"|{r['task']}|{r['arm']}|{q['psnr']:.4f}|{q['ssim']:.5f}|{q['lpips']:.5f}|{r['mean_full_seconds']:.3f}|300|")
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('A73 Summary:',json.dumps(result,ensure_ascii=False),flush=True)


def report(root):
    summarize(root)


def archive(root):
    from utils import archive_refinement_confirmation as exporter
    exporter.BRANCH = BRANCH
    bundle, commit = exporter.export_and_commit({'A73': Path(root)}, projects=PROJECTS)
    print('Archive:', bundle, 'Results commit:', commit, 'Branch:', BRANCH, flush=True)
    subprocess.run(['git', 'push', 'origin', BRANCH], cwd=PROJECTS.parent, check=True)
    print('RESULTS_PUSH_PASS: A73 all-unroll quality and full clocks uploaded.', flush=True)


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
