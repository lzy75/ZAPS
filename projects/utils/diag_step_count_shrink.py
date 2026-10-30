"""A74 fixed step-count comparison, last unroll only, full clocks; S30 reused."""
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
from utils import step_count_shrink as rules
from utils import step_count_shrink_runtime as runtime
from utils import diag_shrink_multitask as parent
from utils import diag_noise_feasible_zaps as common
from utils import historical_ffhq_runtime as precision
from utils import paper_clock_runtime as clock
from utils.a67_model_freeze import frozen_network
from utils.diag_paper_clock_alignment import resource

ROOT = '/data0/lzy/ZAPS-results/diag_step_count_shrink/A74_20261030'
PARENT = parent.ROOT
BRANCH = 'results/step-count-shrink-20261030'
CARD = PROJECTS/'experiment_cards/A74_step_count_shrink_20261030.md'


def check(root):
    p = files.read(Path(root)/'plan.json')
    files.require(p['experiment'] == 'A74_step_count_shrink', 'Wrong plan')
    for n, h in p['sources'].items():
        files.require(files.digest(PROJECTS/n) == h, 'Bound source changed: '+n)
    files.require(files.digest(Path(p['parent'])/'plan.json') == p['parent_plan_sha256'], 'Parent plan changed')
    return p


def prepare(args):
    root,source=Path(args.root).resolve(),Path(args.a71_root).resolve()
    if (root/'plan.json').exists():check(root);print('A74 frozen plan reused.',flush=True);return
    files.require(not root.exists() or not any(root.iterdir()),'Nonempty unplanned output retained')
    files.require(not root.is_relative_to(source) and not source.is_relative_to(root),'Parent overlap')
    old=parent.check(source);summary=files.read(source/'summary.json')
    files.require(summary['completed_runs']==summary['planned_runs']==72,'Need complete A71')
    cases=rules.cases(old['cases']);inputs={};observations={};reused=[]
    for c in cases:
        original=dict(c,key=c['parent_key'])
        obs=source/'observations'/c['parent_key']/'observation.json'
        y=Path(files.read(obs)['path']);observations[c['key']]=str(y)
        for f in (obs,y,Path(c['image'])):inputs[str(f)]=files.digest(f)
        if c['steps']==30:
            for arm in rules.ARMS:
                r=common.completed(source,original,arm);files.require(r is not None,'Missing completed S30 reference')
                record=common.record_path(source,original,arm);inputs[str(record)]=files.digest(record)
                reused.append(dict(status='complete',case=c,arm=arm,metrics=r['paired_metrics'],full_seconds=r['complete_reconstruction_seconds'],
                    reused=True,timing_origin=r['timing_origin'],source_record=str(record),logical_NFE=300))
    inputs[str(source/'motion_kernel.npy')]=files.digest(source/'motion_kernel.npy')
    names=['utils/step_count_shrink.py','utils/step_count_shrink_runtime.py','utils/diag_step_count_shrink.py','utils/all_unroll_shrink.py',
        'utils/a74_session.py','utils/test_step_count_shrink.py','utils/test_step_count_shrink_runtime.py',
        'run_a74_20261030.sh',str(CARD.relative_to(PROJECTS)),*old['sources']]
    root.parent.mkdir(parents=True,exist_ok=True);files.require(shutil.disk_usage(root.parent).free>=files.GIB,'Need1GiB free');root.mkdir(exist_ok=True)
    p=dict(experiment='A74_step_count_shrink',parent=str(source),parent_plan_sha256=files.digest(source/'plan.json'),
        cases=cases,observations=observations,checkpoint=old['checkpoint'],grids={str(n):rules.grid(n) for n in rules.COUNTS},
        inputs=inputs,sources={n:files.digest(PROJECTS/n) for n in dict.fromkeys(names)},reused=reused,
        configs={t:{str(n):rules.configuration(t,n) for n in rules.COUNTS} for t in rules.TASKS},arms=list(rules.ARMS),
        budget=dict(new_complete_runs=48,reused_complete_runs=24,method_forwards=6000,optimizer_backwards=480,input_vjp=0,warmup_separate=True),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECTS.parent,text=True).strip(),
        protocol=dict(scope='Same3 FFHQ images x4 tasks, sigma=.05, ten epochs; S10/15 fresh, S30 A71 reuse only.',
            grid='Low/mid/high sections fixed. Counts10:[5,3,2],15:[8,5,2],30:[15,10,5]; same original section-linear rounding.',
            policy='Unchanged A71 noise shrink, last unroll only. No input VJP or gamma gradient.',
            learning='Per-S independent from-seed zeta/D learning, same seeds per arm; no learned cache or parameter interpolation.',
            timing='S10/15 fresh complete10*S; model loading/warmup/metrics/writes excluded. S30 clocks inherited and explicitly tagged.',
            limits='Fixed epochs, variable totalNFE100/150/300; not equal-budget or adaptive step-size, no new image/dataset/seed confirmation.'))
    files.write(root/'plan.json',p);shutil.copyfile(CARD,root/'experiment_card.md')
    print('A74 PREPARE/FREEZE:NEW S10/15 x3 images x4 tasks x2 arms=48 complete runs,6000 forwards/480 optimizer backwards;S30 REUSE24,NO RERUN.',flush=True)


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
        print('A74 resource pause BEFORE attempt:',c['key'],arm,flush=True);return False
    folder=root/'cases'/c['key']/arm;folder.mkdir(parents=True,exist_ok=False)
    r=dict(status='running',case=c,arm=arm,plan_sha256=files.digest(root/'plan.json'),artifacts={},counts=None)
    files.write(folder/'run.json',r)
    def tick(count):
        r['counts']=count;files.write(folder/'run.json',r)
        print('A74 PROGRESS:',c['key'],arm,'F=',count['forwards'],'/',10*c['steps'],flush=True)
    try:
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        with precision.precision('LEGACY_IEEE'),clock.counters(model.model,tick) as counts:
            def operation():
                y=torch.load(p['observations'][c['key']],map_location='cpu',weights_only=True)['y'].to(model.device)
                op=parent.make_operator(Path(p['parent']),c,model.device,counts)
                return runtime.execute(model,op,y,c,arm)
            (out,detail),seconds=clock.clocks(torch.cuda.synchronize,operation)
        r['counts']=counts
        files.require(counts['forward_attempts']==counts['forwards']==10*c['steps'] and counts['calls']==p['grids'][str(c['steps'])][::-1]*10,'Full10*S NFE/grid mismatch')
        files.require(counts['optimizer_backward_attempts']==counts['optimizer_backward_completed']==10 and counts['input_vjp_attempts']==0,'Backward count mismatch')
        if arm=='SHRINK':files.require(len(detail['trace'])==c['steps'],'Only final unroll may be controlled')
        gt=load_image_as_tensor(c['image']).cpu();metrics=common.metrics(out,gt)
        torch.save(out,folder/'output.pt');torch.save(dict(zeta=detail['zeta'],D=detail['D']),folder/'final_parameters.pt')
        tensor_to_image(out.squeeze(0),denormalize=True).save(folder/'output.png')
        files.write(folder/'trace.json',detail['trace']);files.write(folder/'loss.json',detail['losses'])
        r.update(status='complete',metrics=metrics,full_seconds=seconds,suffix_seconds=None,
            timing_origin='A74 fresh independent complete10*S run',reused=False,logical_NFE=10*c['steps'],peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            cost_is_lower_bound=False,
            artifacts={n:files.digest(folder/n) for n in ('output.pt','final_parameters.pt','output.png','trace.json','loss.json')})
        files.write(folder/'run.json',r)
        print('A74 COMPLETE:',c['key'],arm,'PSNR=',round(metrics['psnr'],4),'full_seconds=',round(seconds,3),'F=',10*c['steps'],flush=True)
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
                    if c['steps'] == 30 or c['shard'] != args.shard:
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
    done={(r['case']['key'],r['arm']):r for r in [*rows,*p['reused']] if r['status']=='complete'}
    groups=[];comparisons=[];table=[]
    for task in rules.TASKS:
        for steps in rules.COUNTS:
            cases=[c for c in p['cases'] if c['task']==task and c['steps']==steps]
            for arm in rules.ARMS:
                data=[done[c['key'],arm] for c in cases if (c['key'],arm) in done]
                for r in data:table.append(dict(task=task,steps=steps,image_id=r['case']['image_id'],arm=arm,
                    **{k:r['metrics'][k] for k in ('psnr','ssim','lpips')},full_seconds=r['full_seconds'],NFE=10*steps,
                    reused=r.get('reused',False),timing_origin=r['timing_origin']))
                if len(data)==3:groups.append(dict(task=task,steps=steps,arm=arm,cases=3,
                    quality={k:statistics.mean(r['metrics'][k] for r in data) for k in ('psnr','ssim','lpips')},
                    mean_full_seconds=statistics.mean(r['full_seconds'] for r in data),full_NFE=10*steps,
                    reused=steps==30,timing_origin=sorted({r['timing_origin'] for r in data})))
            if all((c['key'],arm) in done for c in cases for arm in rules.ARMS):
                base=[done[c['key'],'BASE'] for c in cases];new=[done[c['key'],'SHRINK'] for c in cases]
                comparisons.append(dict(task=task,steps=steps,sigma=.05,
                    SHRINK_minus_BASE={k:statistics.mean(n['metrics'][k]-b['metrics'][k] for b,n in zip(base,new)) for k in ('psnr','ssim','lpips')},
                    images_PSNR_positive=sum(n['metrics']['psnr']>b['metrics']['psnr'] for b,n in zip(base,new)),
                    worst_PSNR=min(n['metrics']['psnr']-b['metrics']['psnr'] for b,n in zip(base,new)),
                    full_clock_ratio=statistics.mean(n['full_seconds'] for n in new)/statistics.mean(b['full_seconds'] for b in base),
                    full_NFE=10*steps,reused=steps==30))
    counts=[r.get('counts') or {} for r in rows];loads=[files.read(f) for f in (root/'workers').glob('*/load_*.json')]
    result=dict(experiment=p['experiment'],classification='DEVELOPMENT_FIXED_STEP_COUNT_SHRINK_COMPARISON',
        completed_runs=len(done),planned_runs=72,completed_new_runs=sum(r['status']=='complete' for r in rows),planned_new_runs=48,
        reused_complete_runs=24,image_clusters=3,groups=groups,comparisons=comparisons,
        actual_new_cost=dict(method_forward_attempts=sum(c.get('forward_attempts',0) for c in counts),optimizer_backward_attempts=sum(c.get('optimizer_backward_attempts',0) for c in counts),input_vjp_attempts=sum(c.get('input_vjp_attempts',0) for c in counts),warmup_forward_attempts=sum(v['warmup_forward_attempts'] for v in loads)),
        cost_is_lower_bound=any(r['status']!='complete' for r in rows),limits=p['protocol'])
    files.write(root/'summary.json',result)
    if table:
        with (root/'comparison.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    lines=['# A74：10/15/30步，固定10轮，末轮缩放','',
        'S30原样复用A71，计时非同场；S10/15为本轮完整独立测时。总NFE分别100/150/300。','',
        '|任务|每轮步数|方法|PSNR|SSIM|LPIPS|完整秒数|NFE|复用|','|---|---:|---|---:|---:|---:|---:|---:|---|']
    for r in groups:
        q=r['quality'];lines.append(f"|{r['task']}|{r['steps']}|{r['arm']}|{q['psnr']:.4f}|{q['ssim']:.5f}|{q['lpips']:.5f}|{r['mean_full_seconds']:.3f}|{r['full_NFE']}|{r['reused']}|")
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('A74 Summary:',json.dumps(result,ensure_ascii=False),flush=True)


def report(root):
    summarize(root)


def archive(root):
    from utils import archive_refinement_confirmation as exporter
    exporter.BRANCH = BRANCH
    bundle, commit = exporter.export_and_commit({'A74': Path(root)}, projects=PROJECTS)
    print('Archive:', bundle, 'Results commit:', commit, 'Branch:', BRANCH, flush=True)
    subprocess.run(['git', 'push', 'origin', BRANCH], cwd=PROJECTS.parent, check=True)
    print('RESULTS_PUSH_PASS: A74 S10/15 new results and S30 reused references uploaded.', flush=True)


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
