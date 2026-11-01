"""A77 native1000 DPS and two fixed scalar-transfer policies, full clocks."""
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
from utils import dps_imagenet_transfer as rules
from utils import dps_multiplier_transfer_runtime as runtime
from utils import diag_shrink_multitask as parent
from utils import diag_noise_feasible_zaps as common
from utils import historical_ffhq_runtime as precision
from utils import paper_clock_runtime as clock
from utils.current_score_runtime_r2 import input_vjp_parameters
from utils.diag_paper_clock_alignment import gpu_inventory
MEASURED_PEAK=0
def resource(uuid,dataset,own_reserved=0):
    row=next(r for r in gpu_inventory() if r['uuid']==uuid)
    return rules.admission(row,own_reserved,MEASURED_PEAK)

ROOT = '/data0/lzy/ZAPS-results/diag_dps_imagenet_transfer/A77_20261101'
PARENT = '/data0/lzy/ZAPS-results/diag_dps_multiplier_transfer/A75_20261031'
BRANCH = 'results/dps-imagenet-transfer-20261101'
CARD = PROJECTS/'experiment_cards/A77_dps_imagenet_transfer_20261101.md'


def check(root):
    p = files.read(Path(root)/'plan.json')
    files.require(p['experiment'] == 'A77_dps_imagenet_transfer', 'Wrong plan')
    for n, h in p['sources'].items():
        files.require(files.digest(PROJECTS/n) == h, 'Bound source changed: '+n)
    files.require(files.digest(Path(p['parent'])/'plan.json') == p['parent_plan_sha256'], 'Parent plan changed')
    return p


def prepare(args):
    from utils import diag_dps_multiplier_transfer as previous
    root,source=Path(args.root).resolve(),Path(args.a75_root).resolve()
    if (root/'plan.json').exists():check(root);print('A77 frozen plan reused.',flush=True);return
    files.require(not root.exists() or not any(root.iterdir()),'Nonempty unplanned output retained')
    files.require(not root.is_relative_to(source) and not source.is_relative_to(root),'Parent overlap')
    old=previous.check(source)
    summary=files.read(source/'summary.json')
    files.require(summary['completed_runs']==summary['planned_runs']==18,'Need complete A75')
    root.parent.mkdir(parents=True,exist_ok=True)
    files.require(shutil.disk_usage(root.parent).free>=files.GIB,'Need1GiB free')
    cases=rules.cases(args.imagenet_dir);inputs={};observations={}
    for c in cases:
        files.require(Path(c['image']).is_file(),'Missing existing ImageNet image: '+c['image'])
        inputs[c['image']]=files.digest(c['image'])
        observations[c['key']]=str(root/'observations'/c['key']/'y.pt')
    kernel=Path(old['parent'])/'motion_kernel.npy';inputs[str(kernel)]=files.digest(kernel)
    physical_plan=Path(old['parent'])/'plan.json';inputs[str(physical_plan)]=files.digest(physical_plan)
    checkpoint=dict(path=str(PROJECTS/'modules/models/256x256_diffusion_uncond.pt'),sha256=rules.CHECKPOINT_SHA256)
    files.require(Path(checkpoint['path']).is_file(),'Existing ImageNet checkpoint missing; no download')
    files.require(files.digest(checkpoint['path'])==checkpoint['sha256'],'ImageNet checkpoint differs')
    inputs[str(PROJECTS.parent/'DPS/configs/imagenet_model_config.yaml')]=files.digest(PROJECTS.parent/'DPS/configs/imagenet_model_config.yaml')
    extra=dict(model_dataset='imagenet',observation_protocol='Fixed measurement seed; shared model-coordinate H(gt)+.05 noise; generated once before timing.')
    protocol=dict(scope='Three existing ImageNet images x SR4/Motion; ImageNet256 unconditional pretrained prior; two arms DPS/SHRINK.',
        baseline='Same original native1000 DDPM learned_range DPS scale=.3; same A75 multiplier formula without calibration, clipping2, or extra gradients.',
        prior='Different weights AND network architecture from FFHQ; model/data change together, not an isolated dataset effect.',
        resources='Allocated GPU>=10GiB free before loading; actual input-VJP probe peak+1GiB checked before each run. No precision/resolution changes; resource pause only before attempts.',
        timing='Complete1000 includes input load/operator/init/sampling/CPU output sync; model loading/probe/measurement creation/metrics/output writes separate; periodic progress bookkeeping included. Same NFE/VJP1000 per arm.',
        limits='Small fixed previously used panel, one seed. Native local DPS reference, not paper benchmark reproduction. Multiplier module adds no NN/VJP.')
    names=['utils/diag_dps_imagenet_transfer.py','utils/dps_imagenet_transfer.py','utils/dps_multiplier_transfer_runtime.py',
        'utils/dps_transfer_followup_session.py','utils/test_dps_transfer_followups.py',
        'utils/test_dps_transfer_followups_runtime.py','modules/diffusion_model.py',
        '../DPS/guided_diffusion/unet.py','../DPS/guided_diffusion/nn.py','run_a77_20261101.sh',str(CARD.relative_to(PROJECTS)),*old['sources']]
    frozen=dict(experiment='A77_dps_imagenet_transfer',parent=str(source),parent_plan_sha256=files.digest(source/'plan.json'),
        operator_root=old['parent'],cases=cases,observations=observations,checkpoint=checkpoint,
        dps_root=old['dps_root'],diffusion_config=old['diffusion_config'],task_configs=old['task_configs'],
        inputs=inputs,sources={n:files.digest(PROJECTS/n) for n in dict.fromkeys(names)},arms=list(rules.ARMS),
        budget=dict(method_forwards=12000,method_input_vjp=12000,optimizer_backwards=0,
                    model_probe_per_load=dict(forwards=1,input_vjp=1)),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECTS.parent,text=True).strip(),
        protocol=protocol,**extra)
    root.mkdir(exist_ok=True);files.write(root/'plan.json',frozen);shutil.copyfile(CARD,root/'experiment_card.md')
    print('A77 PREPARE/FREEZE: 12 new complete1000 runs;12000F+12000 native inputVJP; probes separate.',flush=True)



def verify_observation(root,p,c,device):
    import torch
    from modules.main_single import load_image_as_tensor
    from modules.dataset_loader import tensor_to_image
    from utils.diag_learning_rate_ablation import set_seed
    folder=root/'observations'/c['key'];meta=folder/'observation.json'
    if meta.exists():
        row=files.read(meta)
        files.require(row['plan_sha256']==files.digest(root/'plan.json') and files.digest(row['path'])==row['sha256'],'Observation identity changed')
        return
    files.require(not folder.exists() or not any(folder.iterdir()),'Partial observation retained; inspect before retry')
    folder.mkdir(parents=True,exist_ok=True)
    with precision.precision('LEGACY_IEEE'),torch.no_grad():
        gt=load_image_as_tensor(c['image']).to(device)
        op=parent.make_operator(Path(p['operator_root']),c,device)
        mean=op.H(gt);set_seed(c['measurement_seed']);y=mean+c['sigma']*torch.randn_like(mean)
        torch.save(dict(y=y.cpu()),folder/'y.pt')
        tensor_to_image(gt.cpu().squeeze(0),denormalize=True).save(folder/'ground_truth.png')
        display=tensor_to_image(y.cpu().squeeze(0),denormalize=True)
        if display.size!=(256,256):
            from PIL import Image
            display=display.resize((256,256),Image.Resampling.BICUBIC)
        display.save(folder/'observation.png')
    files.write(meta,dict(path=p['observations'][c['key']],sha256=files.digest(folder/'y.pt'),
                         plan_sha256=files.digest(root/'plan.json'),sigma=c['sigma'],measurement_seed=c['measurement_seed']))


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
    verify_observation(root,p,c,model.device)
    if not resource(uuid,'imagenet',torch.cuda.memory_reserved())['admitted']:
        print('A77 resource pause BEFORE attempt:',c['key'],arm,flush=True);return False
    folder=root/'cases'/c['key']/arm;folder.mkdir(parents=True,exist_ok=False)
    r=dict(status='running',case=c,arm=arm,plan_sha256=files.digest(root/'plan.json'),artifacts={},counts=None)
    files.write(folder/'run.json',r)
    def progress(step,count):
        r['counts']=dict(count);files.write(folder/'run.json',r)
        print('A77 PROGRESS:',c['key'],arm,'F=',count['forwards'],'/1000 inputB=',count['input_vjp_attempts'],flush=True)
    try:
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        versions=[v._version for v in model.model.parameters()]
        with precision.precision('LEGACY_IEEE'),clock.counters(model.model) as count:
            def operation():
                y=torch.load(p['observations'][c['key']],map_location='cpu',weights_only=True)['y'].to(model.device)
                op=parent.make_operator(Path(p['operator_root']),c,model.device,count)
                return runtime.run(model,op,y,c,arm,p['diffusion_config'],p['dps_root'],count,progress=progress)
            (out,detail),seconds=clock.clocks(torch.cuda.synchronize,operation)
        r['counts']=count
        files.require(count['forward_attempts']==count['forwards']==1000 and count['calls']==list(range(999,-1,-1)),'Native DPS1000 grid/count differs')
        files.require(count['input_vjp_attempts']==1000 and count['optimizer_backward_attempts']==0,'Native input gradient budget differs')
        files.require(all(v.grad is None and v._version==ver for v,ver in zip(model.model.parameters(),versions)),'NN parameter changed or grad accumulated')
        gt=load_image_as_tensor(c['image']).cpu();metrics=common.metrics(out,gt)
        with torch.no_grad():
            eval_op=parent.make_operator(Path(p['operator_root']),c,model.device)
            eval_y=torch.load(p['observations'][c['key']],map_location='cpu',weights_only=True)['y'].to(model.device)
            residual=(eval_y-eval_op.H(out.to(model.device))).double()
            metrics['measurement_mse']=float(residual.square().mean())
        del eval_op,eval_y,residual
        torch.save(out,folder/'output.pt');tensor_to_image(out.squeeze(0),denormalize=True).save(folder/'output.png')
        files.write(folder/'trace.json',detail['trace'])
        r.update(status='complete',metrics=metrics,final_controller_state=detail['trace'][-1],full_seconds=seconds,
            timing_origin='A77 fresh native1000 DPS complete run',peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            cost_is_lower_bound=False,artifacts={n:files.digest(folder/n) for n in ('output.pt','output.png','trace.json')})
        files.write(folder/'run.json',r)
        print('A77 COMPLETE:',c['key'],arm,'PSNR=',round(metrics['psnr'],4),'full_seconds=',round(seconds,3),'F/B=1000/1000',flush=True)
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
        if not resource(args.device_uuid,'imagenet')['admitted']:
            files.write(folder/'status.json',dict(status='resource_limited'));return
        for n,h in p['inputs'].items():files.require(files.digest(n)==h,'Read-only input changed: '+n)
        files.require(files.digest(p['checkpoint']['path'])==p['checkpoint']['sha256'],'Checkpoint changed')
        for c in p['cases']:
            if c['shard']==args.shard:verify_observation(root,p,c,'cuda:0')
        start=time.perf_counter();model=load_diffusion_model('imagenet','cuda:0');model.model.eval();torch.cuda.synchronize()
        import inspect
        network_source=Path(inspect.getfile(type(model.model))).resolve()
        load=dict(status='running',load_seconds=time.perf_counter()-start,probe_counts={},
            network_source=dict(path=str(network_source),sha256=files.digest(network_source)),
            model_config=type(model).MODEL_CONFIG)
        f=folder/f'load_{len(list(folder.glob("load_*.json"))):03d}.json';files.write(f,load)
        try:
            with input_vjp_parameters(model.model):
                c=next(c for c in p['cases'] if c['shard']==args.shard)
                y=torch.load(p['observations'][c['key']],map_location='cpu',weights_only=True)['y'].to(model.device)
                dps,conditions,_=runtime.reference(p['dps_root']);sampler=dps.create_sampler(**p['diffusion_config'])
                op=parent.make_operator(Path(p['operator_root']),c,model.device)
                cond=conditions.get_conditioning_method('ps',operator=runtime.PureOperator(op),noiser=types.SimpleNamespace(__name__='gaussian'),scale=rules.SCALE)
                torch.cuda.reset_peak_memory_stats()
                with precision.precision('LEGACY_IEEE'),clock.counters(model.model) as probe:
                    load['probe_counts']=probe;files.write(f,load)
                    x=torch.zeros(1,3,256,256,device=model.device,requires_grad=True)
                    o=sampler.p_sample(model.model,x,torch.tensor([999],device=model.device))
                    probe['input_vjp_attempts']+=1
                    grad,norm=cond.grad_and_value(x,o['pred_xstart'],y)
                    files.require(bool(torch.isfinite(grad).all() and torch.isfinite(norm)),'Real original DPS input gradient not finite')
                global MEASURED_PEAK
                MEASURED_PEAK=torch.cuda.max_memory_reserved()
                load.update(probe_counts=probe,probe_peak_reserved_bytes=MEASURED_PEAK,probe_required_bytes=max(rules.MIN_FREE,MEASURED_PEAK+files.GIB));files.write(f,load)
                del x,o,grad,norm,y;torch.cuda.empty_cache()
                for index,c in enumerate(p['cases']):
                    if c['shard']!=args.shard:continue
                    shift=(index//2)%len(rules.RUN_ARMS);order=rules.RUN_ARMS[shift:]+rules.RUN_ARMS[:shift]
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
    result=dict(experiment=p['experiment'],classification='DEVELOPMENT_DPS_IMAGENET_TRANSFER',completed_runs=len(done),planned_runs=len(p['cases'])*len(rules.ARMS),image_clusters=3,groups=groups,comparisons=comparisons,
        actual_new_cost=dict(method_forward_attempts=sum(c.get('forward_attempts',0) for c in counts),input_vjp_attempts=sum(c.get('input_vjp_attempts',0) for c in counts),optimizer_backward_attempts=sum(c.get('optimizer_backward_attempts',0) for c in counts),probe_forward_attempts=sum(c.get('forward_attempts',0) for c in probes),probe_input_vjp_attempts=sum(c.get('input_vjp_attempts',0) for c in probes)),
        cost_is_lower_bound=any(r['status']!='complete' for r in rows) or any(files.read(f).get('status')=='failed' for f in (root/'workers').glob('*/load_*.json')),limits=p['protocol'])
    files.write(root/'summary.json',result)
    if table:
        with (root/'comparison.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    lines=['# A77：dps_imagenet_transfer','',p['protocol']['scope'],'',
        '|任务|方法|PSNR|SSIM|LPIPS|完整秒数|NFE|输入VJP|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in groups:
        q=r['quality'];lines.append(f"|{r['task']}|{r['arm']}|{q['psnr']:.4f}|{q['ssim']:.5f}|{q['lpips']:.5f}|{r['mean_full_seconds']:.3f}|1000|1000|")
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('A77 Summary:',json.dumps(result,ensure_ascii=False),flush=True)


def report(root):
    summarize(root)


def archive(root):
    from utils import archive_refinement_confirmation as exporter
    exporter.BRANCH = BRANCH
    bundle, commit = exporter.export_and_commit({'A77': Path(root)}, projects=PROJECTS)
    print('Archive:', bundle, 'Results commit:', commit, 'Branch:', BRANCH, flush=True)
    subprocess.run(['git', 'push', 'origin', BRANCH], cwd=PROJECTS.parent, check=True)
    print('RESULTS_PUSH_PASS: A77 DPS transfer quality and full clocks uploaded.', flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=ROOT); p.add_argument('--a75-root', default=PARENT)
    p.add_argument('--imagenet-dir',default='/home/lzy/imagenet/256x256')
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
