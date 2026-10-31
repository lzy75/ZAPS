"""A75 native1000 DPS and two fixed scalar-transfer policies, full clocks."""
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
from utils import dps_multiplier_transfer as rules
from utils import dps_multiplier_transfer_runtime as runtime
from utils import diag_shrink_multitask as parent
from utils import diag_noise_feasible_zaps as common
from utils import historical_ffhq_runtime as precision
from utils import paper_clock_runtime as clock
from utils.current_score_runtime_r2 import input_vjp_parameters
from utils.diag_paper_clock_alignment import resource

ROOT = '/data0/lzy/ZAPS-results/diag_dps_multiplier_transfer/A75_20261031'
PARENT = parent.ROOT
BRANCH = 'results/dps-multiplier-transfer-20261031'
CARD = PROJECTS/'experiment_cards/A75_dps_multiplier_transfer_20261031.md'


def check(root):
    p = files.read(Path(root)/'plan.json')
    files.require(p['experiment'] == 'A75_dps_multiplier_transfer', 'Wrong plan')
    for n, h in p['sources'].items():
        files.require(files.digest(PROJECTS/n) == h, 'Bound source changed: '+n)
    files.require(files.digest(Path(p['parent'])/'plan.json') == p['parent_plan_sha256'], 'Parent plan changed')
    return p


def prepare(args):
    import yaml
    root,source=Path(args.root).resolve(),Path(args.a71_root).resolve()
    if (root/'plan.json').exists():check(root);print('A75 frozen plan reused.',flush=True);return
    files.require(not root.exists() or not any(root.iterdir()),'Nonempty unplanned output retained')
    files.require(not root.is_relative_to(source) and not source.is_relative_to(root),'Parent overlap')
    old=parent.check(source);summary=files.read(source/'summary.json')
    files.require(summary['completed_runs']==summary['planned_runs']==72,'Need complete A71')
    dps_root=PROJECTS.parent/'DPS'
    config=yaml.safe_load((dps_root/'configs/diffusion_config.yaml').read_text())
    task_files=dict(super_resolution='super_resolution_config.yaml',motion_deblur='motion_deblur_config.yaml')
    tasks={t:yaml.load((dps_root/'configs'/n).read_text(),Loader=yaml.FullLoader) for t,n in task_files.items()}
    rules.validate(config,tasks)
    cases=rules.cases(old['cases']);inputs={};observations={}
    for c in cases:
        obs=source/'observations'/c['key']/'observation.json';y=Path(files.read(obs)['path'])
        observations[c['key']]=str(y)
        for f in (obs,y,Path(c['image'])):inputs[str(f)]=files.digest(f)
    inputs[str(source/'motion_kernel.npy')]=files.digest(source/'motion_kernel.npy')
    names=['utils/dps_multiplier_transfer.py','utils/dps_multiplier_transfer_runtime.py','utils/diag_dps_multiplier_transfer.py',
        'utils/a75_session.py','utils/test_dps_multiplier_transfer.py','utils/test_dps_multiplier_transfer_runtime.py',
        'utils/current_score_runtime_r2.py','utils/diag_zaps_sampler_parity.py','run_a75_20261031.sh',str(CARD.relative_to(PROJECTS)),
        '../DPS/guided_diffusion/gaussian_diffusion.py','../DPS/guided_diffusion/condition_methods.py',
        '../DPS/guided_diffusion/posterior_mean_variance.py','../DPS/util/img_utils.py',
        '../DPS/configs/diffusion_config.yaml',*[f'../DPS/configs/{n}' for n in task_files.values()],*old['sources']]
    root.parent.mkdir(parents=True,exist_ok=True);files.require(shutil.disk_usage(root.parent).free>=files.GIB,'Need1GiB free');root.mkdir(exist_ok=True)
    p=dict(experiment='A75_dps_multiplier_transfer',parent=str(source),parent_plan_sha256=files.digest(source/'plan.json'),
        cases=cases,observations=observations,checkpoint=old['checkpoint'],dps_root=str(dps_root),diffusion_config=config,task_configs=tasks,
        inputs=inputs,sources={n:files.digest(PROJECTS/n) for n in dict.fromkeys(names)},arms=list(rules.ARMS),
        budget=dict(method_forwards=18000,method_input_vjp=18000,optimizer_backwards=0,model_probe_per_load=dict(forwards=1,input_vjp=1)),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECTS.parent,text=True).strip(),
        protocol=dict(scope='Same3 FFHQ images, SR4/Motion, model-coordinate sigma=.05. No new measurement/model downloads.',
            baseline='Unmodified official vendored DPS loop, native1000 DDPM learned_range, PS scale=.3. Existing shared H/y, not paper benchmark reproduction.',
            insertion='Original gradient correction q=-.3*grad norm(y-H(clean)); scale q only, original noise and q_sample RNG unchanged.',
            policy='PROJECT uses unshrunk projection; SHRINK uses identical A71 noise correction. Same-time identity response proxy, no next-denoiser probes.',
            timing='Independent complete1000 loop with existing input VJP; excludes model load/probe/metrics/writes. No extra VJP for scalar module.',
            limits='Development transfer, not universal improvement/state necessity/posterior sampling guarantee. Positive original amplitude is canceled by projection.'))
    files.write(root/'plan.json',p);shutil.copyfile(CARD,root/'experiment_card.md')
    print('A75 PREPARE/FREEZE:3 images x2 tasks x3 arms x1000 =18000 forwards +18000 native input VJP; no optimizer; model probes separate.',flush=True)


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


def measure(root,p,c,arm,model,uuid):
    import torch
    from modules.main_single import load_image_as_tensor
    from modules.dataset_loader import tensor_to_image
    if completed(root,c,arm):return True
    if not resource(uuid,'ffhq',torch.cuda.memory_reserved())['admitted']:
        print('A75 resource pause BEFORE attempt:',c['key'],arm,flush=True);return False
    folder=root/'cases'/c['key']/arm;folder.mkdir(parents=True,exist_ok=False)
    r=dict(status='running',case=c,arm=arm,plan_sha256=files.digest(root/'plan.json'),artifacts={},counts=None)
    files.write(folder/'run.json',r)
    def progress(step,count):
        r['counts']=dict(count);files.write(folder/'run.json',r)
        print('A75 PROGRESS:',c['key'],arm,'F=',count['forwards'],'/1000 inputB=',count['input_vjp_attempts'],flush=True)
    try:
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        versions=[v._version for v in model.model.parameters()]
        with precision.precision('LEGACY_IEEE'),clock.counters(model.model) as count:
            def operation():
                y=torch.load(p['observations'][c['key']],map_location='cpu',weights_only=True)['y'].to(model.device)
                op=parent.make_operator(Path(p['parent']),c,model.device,count)
                return runtime.run(model,op,y,c,arm,p['diffusion_config'],p['dps_root'],count,progress=progress)
            (out,detail),seconds=clock.clocks(torch.cuda.synchronize,operation)
        r['counts']=count
        files.require(count['forward_attempts']==count['forwards']==1000 and count['calls']==list(range(999,-1,-1)),'Native DPS1000 grid/count differs')
        files.require(count['input_vjp_attempts']==1000 and count['optimizer_backward_attempts']==0,'Native input gradient budget differs')
        files.require(all(v.grad is None and v._version==ver for v,ver in zip(model.model.parameters(),versions)),'NN parameter changed or grad accumulated')
        gt=load_image_as_tensor(c['image']).cpu();metrics=common.metrics(out,gt)
        torch.save(out,folder/'output.pt');tensor_to_image(out.squeeze(0),denormalize=True).save(folder/'output.png')
        files.write(folder/'trace.json',detail['trace'])
        r.update(status='complete',metrics=metrics,full_seconds=seconds,
            timing_origin='A75 fresh native1000 DPS complete run',peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            cost_is_lower_bound=False,artifacts={n:files.digest(folder/n) for n in ('output.pt','output.png','trace.json')})
        files.write(folder/'run.json',r)
        print('A75 COMPLETE:',c['key'],arm,'PSNR=',round(metrics['psnr'],4),'full_seconds=',round(seconds,3),'F/B=1000/1000',flush=True)
        return True
    except BaseException as e:
        if 'count' in locals():r['counts']=count
        r.update(status='failed',error=repr(e),traceback=traceback.format_exc(),cost_is_lower_bound=True)
        files.write(folder/'run.json',r);raise


def worker(args):
    import torch
    import types
    from modules.main_single import load_diffusion_model
    from utils.a54_session import exclusive
    root=Path(args.root);p=check(root);folder=root/'workers'/str(args.shard);folder.mkdir(parents=True,exist_ok=True)
    with exclusive(folder/'worker.lock'):
        if not resource(args.device_uuid,'ffhq')['admitted']:
            files.write(folder/'status.json',dict(status='resource_limited'));return
        for n,h in p['inputs'].items():files.require(files.digest(n)==h,'Read-only input changed: '+n)
        files.require(files.digest(p['checkpoint']['path'])==p['checkpoint']['sha256'],'Checkpoint changed')
        start=time.perf_counter();model=load_diffusion_model('ffhq','cuda:0');model.model.eval();torch.cuda.synchronize()
        load=dict(status='running',load_seconds=time.perf_counter()-start,probe_counts={})
        f=folder/f'load_{len(list(folder.glob("load_*.json"))):03d}.json';files.write(f,load)
        try:
            with input_vjp_parameters(model.model):
                c=next(c for c in p['cases'] if c['shard']==args.shard)
                y=torch.load(p['observations'][c['key']],map_location='cpu',weights_only=True)['y'].to(model.device)
                dps,conditions,_=runtime.reference(p['dps_root']);sampler=dps.create_sampler(**p['diffusion_config'])
                op=parent.make_operator(Path(p['parent']),c,model.device)
                cond=conditions.get_conditioning_method('ps',operator=runtime.PureOperator(op),noiser=types.SimpleNamespace(__name__='gaussian'),scale=rules.SCALE)
                with precision.precision('LEGACY_IEEE'),clock.counters(model.model) as probe:
                    load['probe_counts']=probe;files.write(f,load)
                    x=torch.zeros(1,3,256,256,device=model.device,requires_grad=True)
                    o=sampler.p_sample(model.model,x,torch.tensor([999],device=model.device))
                    probe['input_vjp_attempts']+=1
                    grad,norm=cond.grad_and_value(x,o['pred_xstart'],y)
                    files.require(bool(torch.isfinite(grad).all() and torch.isfinite(norm)),'Real original DPS input gradient not finite')
                load['probe_counts']=probe;files.write(f,load)
                del x,o,grad,norm,y;torch.cuda.empty_cache()
                for index,c in enumerate(p['cases']):
                    if c['shard']!=args.shard:continue
                    shift=(index//2)%len(rules.ARMS);order=rules.ARMS[shift:]+rules.ARMS[:shift]
                    for arm in order:
                        if not measure(root,p,c,arm,model,args.device_uuid):
                            load['status']='resource_limited';files.write(folder/'status.json',dict(status='resource_limited'));return
            load['status']='complete';files.write(folder/'status.json',dict(status='complete'))
        except BaseException as e:
            load.update(status='failed',error=repr(e));files.write(folder/'status.json',load);raise
        finally:files.write(f,load)


def summarize(root):
    root=Path(root);p=check(root);rows=[files.read(f) for f in (root/'cases').glob('*/*/run.json')]
    done={(r['case']['key'],r['arm']):r for r in rows if r['status']=='complete'};groups=[];comparisons=[];table=[]
    for task in rules.TASKS:
        cases=[c for c in p['cases'] if c['task']==task]
        for arm in rules.ARMS:
            data=[done[c['key'],arm] for c in cases if (c['key'],arm) in done]
            for r in data:table.append(dict(task=task,image_id=r['case']['image_id'],arm=arm,**{k:r['metrics'][k] for k in ('psnr','ssim','lpips')},full_seconds=r['full_seconds'],NFE=1000,input_VJP=1000))
            if len(data)==3:groups.append(dict(task=task,arm=arm,cases=3,quality={k:statistics.mean(r['metrics'][k] for r in data) for k in ('psnr','ssim','lpips')},mean_full_seconds=statistics.mean(r['full_seconds'] for r in data),full_NFE=1000,input_VJP=1000))
        for arm in rules.ARMS[1:]:
            if all((c['key'],a) in done for c in cases for a in ('DPS',arm)):
                base=[done[c['key'],'DPS'] for c in cases];new=[done[c['key'],arm] for c in cases]
                comparisons.append(dict(task=task,arm=arm,minus_DPS={k:statistics.mean(n['metrics'][k]-b['metrics'][k] for b,n in zip(base,new)) for k in ('psnr','ssim','lpips')},
                    images_PSNR_positive=sum(n['metrics']['psnr']>b['metrics']['psnr'] for b,n in zip(base,new)),worst_PSNR=min(n['metrics']['psnr']-b['metrics']['psnr'] for b,n in zip(base,new)),
                    full_clock_ratio=statistics.mean(n['full_seconds'] for n in new)/statistics.mean(b['full_seconds'] for b in base)))
    counts=[r.get('counts') or {} for r in rows];probes=[files.read(f).get('probe_counts',{}) for f in (root/'workers').glob('*/load_*.json')]
    result=dict(experiment=p['experiment'],classification='DEVELOPMENT_DPS_MULTIPLIER_TRANSFER',completed_runs=len(done),planned_runs=18,image_clusters=3,groups=groups,comparisons=comparisons,
        actual_new_cost=dict(method_forward_attempts=sum(c.get('forward_attempts',0) for c in counts),input_vjp_attempts=sum(c.get('input_vjp_attempts',0) for c in counts),optimizer_backward_attempts=sum(c.get('optimizer_backward_attempts',0) for c in counts),probe_forward_attempts=sum(c.get('forward_attempts',0) for c in probes),probe_input_vjp_attempts=sum(c.get('input_vjp_attempts',0) for c in probes)),
        cost_is_lower_bound=any(r['status']!='complete' for r in rows),limits=p['protocol'])
    files.write(root/'summary.json',result)
    if table:
        with (root/'comparison.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    lines=['# A75：原DPS与倍率模块迁移','', '共享A71的H/y；原DPS1000采样与输入梯度，非原文均值复现。','',
        '|任务|方法|PSNR|SSIM|LPIPS|完整秒数|NFE|输入VJP|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in groups:
        q=r['quality'];lines.append(f"|{r['task']}|{r['arm']}|{q['psnr']:.4f}|{q['ssim']:.5f}|{q['lpips']:.5f}|{r['mean_full_seconds']:.3f}|1000|1000|")
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('A75 Summary:',json.dumps(result,ensure_ascii=False),flush=True)


def report(root):
    summarize(root)


def archive(root):
    from utils import archive_refinement_confirmation as exporter
    exporter.BRANCH = BRANCH
    bundle, commit = exporter.export_and_commit({'A75': Path(root)}, projects=PROJECTS)
    print('Archive:', bundle, 'Results commit:', commit, 'Branch:', BRANCH, flush=True)
    subprocess.run(['git', 'push', 'origin', BRANCH], cwd=PROJECTS.parent, check=True)
    print('RESULTS_PUSH_PASS: A75 DPS transfer quality and full clocks uploaded.', flush=True)


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
