"""A88 native DPS vs unchanged online shrink: Gaussian/mask on two priors."""
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
from utils import dps_missing_tasks as rules
from utils import dps_missing_tasks_runtime as runtime
from utils import diag_shrink_multitask as parent
from utils import diag_noise_feasible_zaps as common
from utils import historical_ffhq_runtime as precision
from utils import paper_clock_runtime as clock
from utils.current_score_runtime_r2 import input_vjp_parameters
from utils.diag_paper_clock_alignment import gpu_inventory
MEASURED_PEAK=0
MODEL_DATASET=None
def resource(uuid,dataset,own_reserved=0):
    row=next(r for r in gpu_inventory() if r['uuid']==uuid)
    return rules.admission(MODEL_DATASET,row,own_reserved,MEASURED_PEAK)

ROOT = '/data0/lzy/ZAPS-results/diag_dps_missing_tasks/A88_{dataset}_20261104'
PARENT = '/data0/lzy/ZAPS-results/diag_dps_multiplier_transfer/A75_20261031'
BRANCH = 'results/dps-missing-tasks-20261104'
CARD = PROJECTS/'experiment_cards/A88_dps_missing_tasks_20261104.md'


def check(root):
    p = files.read(Path(root)/'plan.json')
    files.require(p['experiment'] == 'A88_dps_missing_tasks', 'Wrong plan')
    global MODEL_DATASET
    MODEL_DATASET=p['model_dataset']
    for n, h in p['sources'].items():
        files.require(files.digest(PROJECTS/n) == h, 'Bound source changed: '+n)
    files.require(files.digest(Path(p['parent'])/'plan.json') == p['parent_plan_sha256'], 'Parent plan changed')
    return p


def prepare(args):
    import yaml
    from utils import diag_dps_multiplier_transfer as previous
    root,source=Path(args.root).resolve(),Path(args.a75_root).resolve()
    if (root/'plan.json').exists():
        frozen=check(root);files.require(frozen['model_dataset']==args.dataset,'Frozen dataset differs')
        print('A88 frozen plan reused.',flush=True);return
    files.require(not root.exists() or not any(root.iterdir()),'Nonempty unplanned output retained')
    files.require(not root.is_relative_to(source) and not source.is_relative_to(root),'Parent overlap')
    old=previous.check(source);summary=files.read(source/'summary.json')
    files.require(summary['completed_runs']==summary['planned_runs']==18,'Need complete A75')
    root.parent.mkdir(parents=True,exist_ok=True)
    files.require(shutil.disk_usage(root.parent).free>=2*files.GIB,'Need2GiB free')
    directory=args.ffhq_dir if args.dataset=='ffhq' else args.imagenet_dir
    cases=rules.cases(args.dataset,directory);inputs={};observations={}
    for c in cases:
        files.require(Path(c['image']).is_file(),'Missing existing image: '+c['image'])
        inputs[c['image']]=files.digest(c['image'])
        observations[c['key']]=str(root/'observations'/c['key']/'y.pt')
    for f in (Path(old['parent'])/'motion_kernel.npy',Path(old['parent'])/'plan.json'):
        inputs[str(f)]=files.digest(f)
    name,sha=rules.CHECKPOINTS[args.dataset]
    checkpoint=dict(path=str(PROJECTS/'modules/models'/name),sha256=sha)
    files.require(Path(checkpoint['path']).is_file(),'Existing checkpoint missing; no download')
    files.require(files.digest(checkpoint['path'])==sha,'Checkpoint differs')
    dps=Path(old['dps_root'])
    configs={t:yaml.load((dps/'configs'/n).read_text(),Loader=yaml.FullLoader) for t,n in
             dict(gaussian_deblur='gaussian_deblur_config.yaml',inpainting='inpainting_config.yaml').items()}
    config=yaml.safe_load((dps/'configs/diffusion_config.yaml').read_text());rules.validate(config,configs)
    for n in ('diffusion_config.yaml','gaussian_deblur_config.yaml','inpainting_config.yaml',('model_config.yaml' if args.dataset=='ffhq' else 'imagenet_model_config.yaml')):
        f=dps/'configs'/n;inputs[str(f)]=files.digest(f)
    names=['utils/diag_dps_missing_tasks.py','utils/dps_missing_tasks.py','utils/dps_missing_tasks_runtime.py',
        'utils/a88_session.py','utils/test_dps_missing_tasks.py','utils/test_dps_missing_tasks_runtime.py',
        'utils/test_current_score_checkpoint_r2.py','utils/a65_session.py','utils/a54_session.py',
        'modules/diffusion_model.py','../DPS/guided_diffusion/unet.py','../DPS/guided_diffusion/nn.py',
        'run_a88_20261104.sh',str(CARD.relative_to(PROJECTS)),*old['sources']]
    frozen=dict(experiment='A88_dps_missing_tasks',model_dataset=args.dataset,
        parent=str(source),parent_plan_sha256=files.digest(source/'plan.json'),
        operator_root=old['parent'],cases=cases,observations=observations,checkpoint=checkpoint,
        dps_root=str(dps),diffusion_config=config,task_configs=configs,
        inputs=inputs,sources={n:files.digest(PROJECTS/n) for n in dict.fromkeys(names)},arms=list(rules.ARMS),
        budget=dict(method_forwards=12000,method_input_vjp=12000,optimizer_backwards=0,
                    module_H_actions=18000,model_probe_per_load=dict(forwards=1,input_vjp=1)),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECTS.parent,text=True).strip(),
        protocol=dict(scope=args.dataset+': three existing images x Gaussian61/sigma3 zero boundary and random70% missing; DPS/SHRINK only. Completes missing A75/A77 task cells.',
            baseline='Original vendored native1000 DDPM learned_range DPS loop and PS condition; official task scales Gaussian=.3, inpainting=.5. Existing A71 physical operators, not author benchmark reproduction.',
            observation='NEW shared y per pair, model[-1,1] sigma=.05. Full-output additive Gaussian noise retained, including missing coordinates, as in official DPS sample_condition.py. Mask fixed from case seed; measurement is never replaced/clamped.',
            module='Unchanged A71 continuous noise_signal_shrink.select; observed RGB entries only for inpainting. Same-time Hq/sqrt(alpha) proxy. No beta, clipping2, calibration or next-denoiser probes.',
            pairing='Independent from seed; same observation/operator, initial/final CPU+CUDA RNG fingerprints and grid required. Alternating two-arm run order; no learned state.',
            cost='1000 forwards+1000 existing input VJP per run. Scalar module adds three H evaluations per step (residual plus two response evaluations); no extra NN/VJP/optimizer.',
            resources='FFHQ>=6GiB, ImageNet>=10GiB free before load; real checkpoint VJP peak+1GiB required before attempts. No precision/resolution change, no process termination.',
            timing='Full reconstruction includes given-y load/operator/init/native1000/outputCPU sync/bookkeeping; model load, probe, shared observation creation, metrics and artifact writes separately recorded. Same NFE/VJP per arm.',
            output='Original returned DPS next-state; no clean-output replacement or GT-based output selection. Save original GT/observation/BASE/SHRINK PNGs and fixed triptych per case.',
            decision='Per dataset/task: mean PSNR>=+.1dB, >=2/3 image wins, SSIM nondecrease, LPIPS nonincrease, paired noise/grid pass. Preserve mixed and negative results without tuning.',
            limits='Small reused panel, one seed, sigma=.05 only. No universal/state-necessity/posterior guarantee; positive scalar DPS amplitude may be canceled by projection. No rerun of completed SR/Motion.'))
    root.mkdir(exist_ok=True);files.write(root/'plan.json',frozen);shutil.copyfile(CARD,root/'experiment_card.md')
    print('A88 PREPARE/FREEZE:',args.dataset,'12 complete1000 runs;12000F+12000 native inputVJP; probes separate.',flush=True)



def verify_observation(root,p,c,device):
    import torch
    observation_start=time.perf_counter()
    from modules.main_single import load_image_as_tensor
    from modules.dataset_loader import tensor_to_image
    from utils.diag_learning_rate_ablation import set_seed
    folder=root/'observations'/c['key'];meta=folder/'observation.json'
    if meta.exists():
        row=files.read(meta)
        files.require(row['plan_sha256']==files.digest(root/'plan.json') and all(files.digest(folder/n)==h for n,h in row['artifacts'].items()),'Observation identity changed')
        return
    files.require(not folder.exists() or not any(folder.iterdir()),'Partial observation retained; inspect before retry')
    folder.mkdir(parents=True,exist_ok=True)
    with precision.precision('LEGACY_IEEE'),torch.no_grad():
        gt=load_image_as_tensor(c['image']).to(device)
        op=parent.make_operator(Path(p['operator_root']),c,device)
        mean=op.H(gt);set_seed(c['measurement_seed']);y=mean+c['sigma']*torch.randn_like(mean)
        torch.save(dict(y=y.cpu()),folder/'y.pt')
        tensor_to_image(gt.cpu().squeeze(0),denormalize=True).save(folder/'ground_truth.png')
        mask=getattr(op,'_mask_cache',None)
        if mask is not None:torch.save(mask.cpu(),folder/'mask.pt')
        shown=y if mask is None else y*mask-(1-mask)
        display=tensor_to_image(shown.cpu().squeeze(0),denormalize=True)
        if display.size!=(256,256):
            from PIL import Image
            display=display.resize((256,256),Image.Resampling.BICUBIC)
        display.save(folder/'observation.png')
    files.write(meta,dict(path=p['observations'][c['key']],sha256=files.digest(folder/'y.pt'),
                         plan_sha256=files.digest(root/'plan.json'),sigma=c['sigma'],measurement_seed=c['measurement_seed'],
                         creation_seconds=time.perf_counter()-observation_start,artifacts={n:files.digest(folder/n) for n in ('y.pt','ground_truth.png','observation.png')+
                                    (('mask.pt',) if mask is not None else ())}))


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
    if not resource(uuid,p['model_dataset'],torch.cuda.memory_reserved())['admitted']:
        print('A88 resource pause BEFORE attempt:',c['key'],arm,flush=True);return False
    folder=root/'cases'/c['key']/arm;folder.mkdir(parents=True,exist_ok=False)
    r=dict(status='running',case=c,arm=arm,plan_sha256=files.digest(root/'plan.json'),artifacts={},counts=None)
    files.write(folder/'run.json',r)
    def progress(step,count):
        r['counts']=dict(count);files.write(folder/'run.json',r)
        print('A88 PROGRESS:',c['key'],arm,'F=',count['forwards'],'/1000 inputB=',count['input_vjp_attempts'],flush=True)
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
        metric_start=time.perf_counter()
        gt=load_image_as_tensor(c['image']).cpu();metrics=common.metrics(out,gt)
        with torch.no_grad():
            eval_op=parent.make_operator(Path(p['operator_root']),c,model.device)
            eval_y=torch.load(p['observations'][c['key']],map_location='cpu',weights_only=True)['y'].to(model.device)
            residual=(eval_y-eval_op.H(out.to(model.device))).double()
            metrics['measurement_mse_full_output']=float(residual.square().mean())
            active=runtime.informative(residual,eval_op)
            metrics['measurement_mse_observed']=float(active.square().mean())
            mask=getattr(eval_op,'_mask_cache',None)
            if mask is not None:
                selected=mask.bool().expand_as(residual)
                error=(out.to(model.device).double()-gt.to(model.device).double()).square()
                metrics['raw_mse_observed']=float(error[selected].mean())
                metrics['raw_mse_missing']=float(error[~selected].mean())
                metrics['observed_RGB_entries']=int(selected.sum())
        del eval_op,eval_y,residual
        metrics_seconds=time.perf_counter()-metric_start;write_start=time.perf_counter()
        torch.save(out,folder/'output.pt');tensor_to_image(out.squeeze(0),denormalize=True).save(folder/'output.png')
        files.write(folder/'trace.json',detail['trace'])
        files.write(folder/'pairing.json',detail['pairing'])
        r['pairing']=detail['pairing']
        for other in rules.ARMS:
            if other==arm:continue
            paired=completed(root,c,other)
            if paired:files.require(paired['pairing']==r['pairing'],'Initial/final RNG/grid pairing failed')
        r.update(status='complete',metrics=metrics,final_controller_state=detail['trace'][-1],full_seconds=seconds,
            metrics_seconds=metrics_seconds,artifact_write_seconds=time.perf_counter()-write_start,
            timing_origin='A88 fresh native1000 DPS complete run',peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            cost_is_lower_bound=False,artifacts={n:files.digest(folder/n) for n in ('output.pt','output.png','trace.json','pairing.json')})
        files.write(folder/'run.json',r)
        print('A88 COMPLETE:',c['key'],arm,'PSNR=',round(metrics['psnr'],4),'full_seconds=',round(seconds,3),'F/B=1000/1000',flush=True)
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
        if not resource(args.device_uuid,p['model_dataset'])['admitted']:
            files.write(folder/'status.json',dict(status='resource_limited'));return
        for n,h in p['inputs'].items():files.require(files.digest(n)==h,'Read-only input changed: '+n)
        files.require(files.digest(p['checkpoint']['path'])==p['checkpoint']['sha256'],'Checkpoint changed')
        for c in p['cases']:
            if c['shard']==args.shard:verify_observation(root,p,c,'cuda:0')
        start=time.perf_counter();model=load_diffusion_model(p['model_dataset'],'cuda:0');model.model.eval();torch.cuda.synchronize()
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
                cond=conditions.get_conditioning_method('ps',operator=runtime.PureOperator(op),noiser=types.SimpleNamespace(__name__='gaussian'),scale=rules.SCALES[c['task']])
                torch.cuda.reset_peak_memory_stats()
                with precision.precision('LEGACY_IEEE'),clock.counters(model.model) as probe:
                    probe_start=time.perf_counter();load['probe_counts']=probe;files.write(f,load)
                    x=torch.zeros(1,3,256,256,device=model.device,requires_grad=True)
                    o=sampler.p_sample(model.model,x,torch.tensor([999],device=model.device))
                    probe['input_vjp_attempts']+=1
                    grad,norm=cond.grad_and_value(x,o['pred_xstart'],y)
                    files.require(bool(torch.isfinite(grad).all() and torch.isfinite(norm)),'Real original DPS input gradient not finite')
                global MEASURED_PEAK
                MEASURED_PEAK=torch.cuda.max_memory_reserved()
                load.update(probe_seconds=time.perf_counter()-probe_start,probe_counts=probe,probe_peak_reserved_bytes=MEASURED_PEAK,probe_required_bytes=resource(args.device_uuid,p['model_dataset'],torch.cuda.memory_reserved())['required_bytes']);files.write(f,load)
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
                    full_clock_ratio=statistics.mean(n['full_seconds'] for n in new)/statistics.mean(b['full_seconds'] for b in base),
                    paired_noise_grid_passed=all(b['pairing']==n['pairing'] for b,n in zip(base,new)),
                    positive_development_candidate=all(b['pairing']==n['pairing'] for b,n in zip(base,new)) and
                        statistics.mean(n['metrics']['psnr']-b['metrics']['psnr'] for b,n in zip(base,new))>=.1 and
                        sum(n['metrics']['psnr']>b['metrics']['psnr'] for b,n in zip(base,new))>=2 and
                        statistics.mean(n['metrics']['ssim']-b['metrics']['ssim'] for b,n in zip(base,new))>=0 and
                        statistics.mean(n['metrics']['lpips']-b['metrics']['lpips'] for b,n in zip(base,new))<=0))
    counts=[r.get('counts') or {} for r in rows];probes=[files.read(f).get('probe_counts',{}) for f in (root/'workers').glob('*/load_*.json')]
    result=dict(experiment=p['experiment'],classification='DEVELOPMENT_DPS_MISSING_TASKS_TRANSFER',completed_runs=len(done),planned_runs=len(p['cases'])*len(rules.ARMS),dataset=p['model_dataset'],image_clusters=3,groups=groups,comparisons=comparisons,
        actual_new_cost=dict(method_forward_attempts=sum(c.get('forward_attempts',0) for c in counts),input_vjp_attempts=sum(c.get('input_vjp_attempts',0) for c in counts),optimizer_backward_attempts=sum(c.get('optimizer_backward_attempts',0) for c in counts),
            module_H_actions=sum(c.get('module_H_actions',0) for c in counts),base_H_actions=sum(c.get('base_H_actions',0) for c in counts),probe_forward_attempts=sum(c.get('forward_attempts',0) for c in probes),probe_input_vjp_attempts=sum(c.get('input_vjp_attempts',0) for c in probes)),
        cost_is_lower_bound=any(r['status']!='complete' for r in rows) or any(files.read(f).get('status')=='failed' for f in (root/'workers').glob('*/load_*.json')),limits=p['protocol'],separate_clock_seconds=dict(
            model_load=sum(files.read(f).get('load_seconds',0) for f in (root/'workers').glob('*/load_*.json')),
            probe=sum(files.read(f).get('probe_seconds',0) for f in (root/'workers').glob('*/load_*.json')),
            observation_creation=sum(files.read(f).get('creation_seconds',0) for f in (root/'observations').glob('*/observation.json')),
            metrics=sum(r.get('metrics_seconds',0) for r in rows),artifact_writes=sum(r.get('artifact_write_seconds',0) for r in rows)))
    files.write(root/'summary.json',result)
    if table:
        with (root/'comparison.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    lines=['# A88：dps_missing_tasks','',p['protocol']['scope'],'',
        '|任务|方法|PSNR|SSIM|LPIPS|完整秒数|NFE|输入VJP|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in groups:
        q=r['quality'];lines.append(f"|{r['task']}|{r['arm']}|{q['psnr']:.4f}|{q['ssim']:.5f}|{q['lpips']:.5f}|{r['mean_full_seconds']:.3f}|1000|1000|")
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    render_panels(root,p,done)
    print('A88 Summary:',json.dumps(result,ensure_ascii=False),flush=True)


def render_panels(root,p,done):
    from PIL import Image,ImageDraw
    for c in p['cases']:
        if not all((c['key'],arm) in done for arm in rules.ARMS):continue
        sources=[('Ground truth',root/'observations'/c['key']/'ground_truth.png'),
                 ('DPS',root/'cases'/c['key']/'DPS'/'output.png'),
                 ('DPS + shrink',root/'cases'/c['key']/'SHRINK'/'output.png')]
        panel=Image.new('RGB',(3*256,280),'white');draw=ImageDraw.Draw(panel)
        for j,(label,f) in enumerate(sources):
            with Image.open(f) as im:panel.paste(im.convert('RGB'),(256*j,24))
            draw.text((256*j+8,5),label,fill='black')
        folder=root/'panels';folder.mkdir(exist_ok=True);panel.save(folder/(c['key']+'.png'))


def report(root):
    summarize(root)


def archive(root):
    from utils import archive_refinement_confirmation as exporter
    exporter.BRANCH = BRANCH+'-'+check(root)['model_dataset']
    bundle, commit = exporter.export_and_commit({'A88_'+check(root)['model_dataset']: Path(root)}, projects=PROJECTS)
    print('Archive:', bundle, 'Results commit:', commit, 'Branch:', exporter.BRANCH, flush=True)
    subprocess.run(['git', 'push', 'origin', exporter.BRANCH], cwd=PROJECTS.parent, check=True)
    print('RESULTS_PUSH_PASS: A88 DPS transfer quality and full clocks uploaded.', flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root');p.add_argument('--dataset',choices=('ffhq','imagenet'),default='ffhq')
    p.add_argument('--ffhq-dir',default='/home/lzy/FFHQ/00000/00000'); p.add_argument('--a75-root', default=PARENT)
    p.add_argument('--imagenet-dir',default='/home/lzy/imagenet/256x256')
    p.add_argument('--shard', type=int, choices=(0, 1)); p.add_argument('--device-uuid')
    p.add_argument('--check-worker-cli', action='store_true', help=argparse.SUPPRESS)
    group = p.add_mutually_exclusive_group(required=True)
    for n in ('prepare', 'worker', 'summarize', 'report', 'archive'):
        group.add_argument('--'+n, action='store_true')
    return p


def main():
    p = parser(); a = p.parse_args()
    if a.root is None:a.root=ROOT.format(dataset=a.dataset)
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
