"""A71 fixed3-image/four-task noise panel; cached SR reuse, no fixed-policy rerun."""
import argparse
import gc
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback
PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import shrink_multitask as rule
from utils import shrink_multitask_runtime as runtime
from utils import historical_ffhq_gain as files
from utils import historical_ffhq_runtime as old_runtime
from utils import paper_clock_runtime as clock
from utils import diag_noise_feasible_zaps as common
from utils.a67_model_freeze import frozen_network
from utils.diag_paper_clock_alignment import write,resource
ROOT='/data0/lzy/ZAPS-results/diag_shrink_multitask/A71_20261029'
PARENT='/data0/lzy/ZAPS-results/diag_noise_signal_shrink/A70_20261029'
BRANCH='results/shrink-multitask-20261029'
CARD=PROJECTS/'experiment_cards/A71_shrink_multitask_20261029.md'

def check(root):
    p=files.read(root/'plan.json');files.require(p['experiment']=='A71_shrink_multitask','Wrong plan')
    for n,h in p['sources'].items():files.require(files.digest(PROJECTS/n)==h,'Source changed: '+n)
    for n,h in p['parent_files'].items():files.require(files.digest(n)==h,'Parent changed: '+n)
    return p

def prepare(args):
    import numpy as np
    root=Path(args.root).resolve();parent=Path(args.a70_root).resolve()
    if (root/'plan.json').exists():check(root);print('A71 frozen plan reused.',flush=True);return
    files.require(not root.exists() or not any(root.iterdir()),'Nonempty output retained')
    files.require(not root.is_relative_to(parent) and not parent.is_relative_to(root),'Parent overlap')
    from utils.diag_noise_signal_shrink import check as parent_check
    before=parent_check(parent);summary=files.read(parent/'summary.json')
    files.require(summary['completed_runs']==summary['planned_runs']==56,'Need complete A70')
    cases=rule.cases(args.ffhq_dir)
    for c in cases:c['image_sha256']=files.digest(c['image'])
    parent_files={str(parent/'plan.json'):files.digest(parent/'plan.json'),str(parent/'summary.json'):files.digest(parent/'summary.json')}
    for c in cases:
        if not rule.reusable(c):continue
        pc=next(v for v in before['cases'] if v['key']==rule.parent_key(c))
        files.require(pc['image_sha256']==c['image_sha256'] and pc['seed']==c['seed'],'Reuse image/seed mismatch')
        for arm in rule.ARMS:
            r=common.completed(parent,pc,arm);files.require(r is not None,'Missing source arm')
            folder=common.record_path(parent,pc,arm).parent
            f=folder/'run.json';parent_files[str(f)]=files.digest(f)
            for n,h in r['artifacts'].items():parent_files[str(folder/n)]=h
        f=parent/'observations'/(pc['key']+'.json');obs=files.read(f)
        parent_files[str(f)]=files.digest(f);parent_files[obs['path']]=obs['sha256']
    root.parent.mkdir(parents=True,exist_ok=True);files.require(shutil.disk_usage(root.parent).free>=3*files.GIB,'Need3GiB free');root.mkdir(exist_ok=True)
    from utils.diag_ffhq_four_tasks_unit import original_motion_kernel
    np.save(root/'motion_kernel.npy',original_motion_kernel(0),allow_pickle=False)
    names=['utils/shrink_multitask.py','utils/shrink_multitask_runtime.py','utils/diag_shrink_multitask.py','utils/a71_session.py',
        'utils/render_shrink_panel.py','utils/test_shrink_multitask.py','utils/test_shrink_multitask_runtime.py','utils/test_a71_cli.py',
        'utils/noise_signal_shrink.py','utils/noise_signal_shrink_runtime.py','utils/noise_feasible_runtime.py',
        'utils/historical_ffhq_gain.py','utils/historical_ffhq_runtime.py','utils/paper_clock_runtime.py','utils/diag_ffhq_four_tasks_unit.py',
        'utils/a67_model_freeze.py','utils/metrics.py','modules/main_single.py','modules/degradations.py','modules/zaps_algorithm.py','modules/wavelet.py',
        'run_a71_20261029.sh',str(CARD.relative_to(PROJECTS))]
    p=dict(experiment='A71_shrink_multitask',parent=str(parent),parent_files=parent_files,cases=cases,
        configs={t:rule.configuration(t) for t in rule.TASKS},grid=files.grid(),checkpoint=before['checkpoint'],
        sources={n:files.digest(PROJECTS/n) for n in names},motion_kernel_sha256=files.digest(root/'motion_kernel.npy'),
        original_motion_generator_sha256=files.digest(PROJECTS.parent/'DPS/motionblur/motionblur.py'),arms=list(rule.ARMS),
        budget=dict(new_method_forwards=18000,new_diagnostic_forwards=1800,new_optimizer_backwards=600,input_vjp=0,
                    reused_complete_runs=12,warmup_separate=True),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECTS.parent,text=True).strip(),
        protocol=dict(grid='Original strong15/10/5,30 points x10 epochs; last unroll only',
            physics='Model[-1,1] H(x), sigma same units; SR empirical bicubic guidance, blur/mask exact adjoints',
            tasks='SR4; Gaussian61/sigma3 zero boundary; original DPS motion61/intensity.5 seed0 zero boundary; random70% missing',
            inpainting='Original full-output noisy observation/loss retained; policy statistics use observed RGB entries only',
            clocks='Independent complete300; model load/warmup/metrics/file writes excluded. Reused times tagged A70, not same-session timings.',
            legacy_fixed='Historical F(t319 x2) reused only in separate history report; current grid has no319, no mapping or fixed-arm rerun',
            scope='Three fixed already-seen A70 images, four FFHQ tasks, sigma .025/.05/.10, one seed/image. '
                  'No across-dataset/novel-theorem/curvature/step-size/NFE-saving claim. All task/noise cells retained.'))
    write(root/'plan.json',p);shutil.copyfile(CARD,root/'experiment_card.md')
    print('A71 PREPARE:',list(rule.IMAGES),'x4 tasks x3 noise x2 groups; reuse12, NEW60 complete runs;18000 method+1800 paired;0 input VJP.',flush=True)

def make_operator(root,c,device,count=None):
    import numpy as np
    p=files.read(root/'plan.json')
    files.require(files.digest(root/'motion_kernel.npy')==p['motion_kernel_sha256'],'Motion kernel changed')
    return runtime.make_operator(c,device,count,kernel=np.load(root/'motion_kernel.npy',allow_pickle=False))

def observation(root,p,c,device):
    import torch
    from modules.main_single import load_image_as_tensor
    from modules.dataset_loader import tensor_to_image
    from utils.diag_learning_rate_ablation import set_seed
    folder=root/'observations'/c['key'];f=folder/'observation.json'
    if f.exists():
        row=files.read(f);files.require(files.digest(row['path'])==row['sha256'],'Observation changed');return row
    folder.mkdir(parents=True,exist_ok=True)
    with old_runtime.precision('HISTORICAL_NATIVE'),torch.no_grad():
        gt=load_image_as_tensor(c['image']).to(device);op=make_operator(root,c,device)
        if rule.reusable(c):
            meta=files.read(Path(p['parent'])/'observations'/(rule.parent_key(c)+'.json'))
            files.require(files.digest(meta['path'])==meta['sha256'],'Cached observation differs')
            y=torch.load(meta['path'],map_location='cpu',weights_only=True)['y'].to(device)
        else:
            set_seed(c['measurement_seed']);mean=op.H(gt);y=mean+c['sigma']*torch.randn_like(mean)
        torch.save(dict(y=y.cpu()),folder/'y.pt')
        tensor_to_image(gt.cpu().squeeze(0),denormalize=True).save(folder/'ground_truth.png')
        display=y.cpu()
        mask=getattr(op,'_mask_cache',None)
        if mask is not None:
            display=display*mask.cpu()-(1-mask.cpu());torch.save(mask.cpu(),folder/'mask.pt')
        im=tensor_to_image(display.squeeze(0),denormalize=True)
        if im.size!=(256,256):
            from PIL import Image
            im=im.resize((256,256),Image.Resampling.BICUBIC)
        im.save(folder/'observation.png')
    row=dict(path=str(folder/'y.pt'),sha256=files.digest(folder/'y.pt'),sigma=c['sigma'],domain='model-coordinate',
        display='SR bicubic enlarged; inpainting missing pixels black for display only; no display changes to y')
    write(f,row);return row

def reuse(root,p,c,arm):
    if common.completed(root,c,arm):return
    parent=Path(p['parent']);pc=dict(c,key=rule.parent_key(c));old=common.completed(parent,pc,arm)
    files.require(old is not None,'Missing completed A70 source')
    source=common.record_path(parent,pc,arm).parent;folder=common.record_path(root,c,arm).parent;folder.mkdir(parents=True)
    artifacts={}
    for n in ('paired.png','paired_trace.json','independent_trace.json'):
        shutil.copyfile(source/n,folder/n);artifacts[n]=files.digest(folder/n)
    row=dict(old,case=c,plan_sha256=files.digest(root/'plan.json'),artifacts=artifacts,reused=True,
        source_record=str(source/'run.json'),source_record_sha256=files.digest(source/'run.json'),timing_origin='A70 cached complete run; no new same-session timing')
    write(folder/'run.json',row);print('A71 REUSE:',c['key'],arm,'NEW NN=0.',flush=True)

def validate(count,arm,full):
    files.require(count['forwards']==count['forward_attempts']==(300 if full else 30),'NN count differs')
    files.require(count['calls']==files.grid()[::-1]*(10 if full else 1),'Grid calls differ')
    files.require(count['optimizer_backward_attempts']==count['optimizer_backward_completed']==(10 if full else 0),'Optimizer count differs')
    files.require(count['input_vjp_attempts']==0,'Unexpected input VJP')

def measure(root,p,c,arm,model,uuid):
    import torch
    from modules.main_single import load_image_as_tensor
    from modules.dataset_loader import tensor_to_image
    if common.completed(root,c,arm):return
    files.require(resource(uuid,'ffhq',torch.cuda.memory_reserved())['admitted'],'Resource margin lost before attempt')
    obs=observation(root,p,c,model.device)
    folder=common.record_path(root,c,arm).parent;folder.mkdir(parents=True,exist_ok=False)
    r=dict(status='running',case=c,arm=arm,plan_sha256=files.digest(root/'plan.json'),artifacts={},
           device_uuid=uuid,full_counts=None,replay_counts=None)
    write(folder/'run.json',r)
    def tick(value):
        r['full_counts']=value;write(folder/'run.json',r)
        print('A71 PROGRESS:',c['key'],arm,'F=',value['forwards'],'cap=',300,flush=True)
    try:
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        with old_runtime.precision('LEGACY_IEEE'),clock.counters(model.model,tick) as counts:
            def operation():
                y=torch.load(obs['path'],map_location='cpu',weights_only=True)['y'].to(model.device)
                return runtime.execute(model,make_operator(root,c,model.device,counts),y,c,arm)
            (independent,detail),seconds=clock.clocks(torch.cuda.synchronize,operation)
        r['full_counts']=counts;validate(counts,arm,True)
        peak=torch.cuda.max_memory_reserved()
        if arm=='BASE':torch.save(detail['entry'],folder/'entry.pt');entry=detail['entry']
        else:
            files.require(common.completed(root,c,'BASE') is not None,'Need BASE first')
            entry=torch.load(common.record_path(root,c,'BASE').parent/'entry.pt',map_location='cpu',weights_only=True)
        r['independent_prefix_comparison']=clock.snapshot_equal(entry,detail['entry'])
        y=torch.load(obs['path'],map_location='cpu',weights_only=True)['y'].to(model.device)
        with old_runtime.precision('LEGACY_IEEE'),clock.counters(model.model) as replay_counts:
            paired,trace=runtime.replay(model,make_operator(root,c,model.device,replay_counts),y,c,entry,
                                        'NULL' if arm=='BASE' else arm)
        r['replay_counts']=replay_counts;validate(replay_counts,arm,False)
        if arm=='BASE':
            error=float((paired.double()-independent.double()).norm()/independent.double().norm().clamp(min=1e-30))
            r['null_relative_error']=error;files.require(error<=1e-6,'Core/null mismatch')
        gt=load_image_as_tensor(c['image']).cpu()
        r.update(paired_metrics=common.metrics(paired,gt),independent_metrics=common.metrics(independent,gt),
                 complete_reconstruction_seconds=seconds,peak_reserved_bytes=peak,
                 full_precision='IEEE',cost_is_lower_bound=False,reused=False,timing_origin='A71 independent complete run',
                 gamma_more_than_two=sum(v.get('gamma',1)>2 for v in trace),
                 gamma_zero=sum(v.get('gamma',1)==0 for v in trace))
        torch.save(paired,folder/'paired.pt');torch.save(independent,folder/'independent.pt')
        tensor_to_image(paired.squeeze(0),denormalize=True).save(folder/'paired.png')
        write(folder/'paired_trace.json',trace);write(folder/'independent_trace.json',detail['trace'])
        names=['paired.pt','independent.pt','paired.png','paired_trace.json','independent_trace.json']+(['entry.pt'] if arm=='BASE' else [])
        r.update(status='complete',artifacts={n:files.digest(folder/n) for n in names});write(folder/'run.json',r)
        print('A71 COMPLETE:',c['key'],arm,'paired PSNR=',round(r['paired_metrics']['psnr'],4),
              'full seconds=',round(seconds,2),'NN=',counts['forwards'],flush=True)
    except BaseException as error:
        if 'counts' in locals():r['full_counts']=counts
        if 'replay_counts' in locals():r['replay_counts']=replay_counts
        r.update(status='failed',error=repr(error),traceback=traceback.format_exc(),cost_is_lower_bound=True)
        write(folder/'run.json',r);raise


def worker(args):
    import torch
    from modules.main_single import load_diffusion_model
    from utils.a54_session import exclusive
    root=Path(args.root);p=check(root);folder=root/'workers'/str(args.shard);folder.mkdir(parents=True,exist_ok=True)
    with exclusive(folder/'worker.lock'):
        if not resource(args.device_uuid,'ffhq')['admitted']:write(folder/'status.json',dict(status='resource_limited'));return
        files.require(files.digest(p['checkpoint']['path'])==p['checkpoint']['sha256'],'Checkpoint changed')
        for c in p['cases']:files.require(files.digest(c['image'])==c['image_sha256'],'Image changed')
        start=time.perf_counter();model=load_diffusion_model('ffhq','cuda:0');model.model.eval();torch.cuda.synchronize()
        load=dict(status='running',load_seconds=time.perf_counter()-start,uuid=args.device_uuid,warmup_forward_attempts=0)
        f=folder/f'load_{len(list(folder.glob("load_*.json"))):03d}.json';write(f,load)
        try:
            with frozen_network(model.model):
                load['warmup_forward_attempts']=1;write(f,load)
                with torch.no_grad(),old_runtime.precision('LEGACY_IEEE'):
                    model._predict_eps(torch.zeros(1,3,256,256,device=model.device),torch.tensor([999],device=model.device))
                torch.cuda.synchronize()
                for c in p['cases']:
                    if c['shard']!=args.shard:continue
                    observation(root,p,c,model.device)
                    for arm in rule.ARMS:
                        if rule.reusable(c):reuse(root,p,c,arm)
                        else:measure(root,p,c,arm,model,args.device_uuid)
            load['status']='complete';write(folder/'status.json',dict(status='complete'))
        except BaseException as e:
            load.update(status='failed',error=repr(e));write(folder/'status.json',load);raise
        finally:write(f,load)

def summarize(root):
    root=Path(root);p=check(root);rows=[files.read(f) for f in (root/'cases').glob('*/*/run.json')]
    done=[r for r in rows if r['status']=='complete'];groups=[];comparisons=[]
    for task in rule.TASKS:
        for sigma in rule.SIGMAS:
            for arm in rule.ARMS:
                data=[r for r in done if r['case']['task']==task and r['case']['sigma']==sigma and r['arm']==arm]
                if not data:continue
                groups.append(dict(task=task,sigma=sigma,arm=arm,cases=len(data),
                    quality={k:statistics.mean(r['paired_metrics'][k] for r in data) for k in ('psnr','ssim','lpips')},
                    independent_quality={k:statistics.mean(r['independent_metrics'][k] for r in data) for k in ('psnr','ssim','lpips')},
                    mean_full_seconds=statistics.mean(r['complete_reconstruction_seconds'] for r in data),mean_full_NN=300,
                    reused_runs=sum(r.get('reused',False) for r in data),timing_origin=sorted({r['timing_origin'] for r in data})))
            base={r['case']['key']:r for r in done if r['arm']=='BASE'}
            pairs=[r for r in done if r['arm']=='SHRINK' and r['case']['task']==task and r['case']['sigma']==sigma and r['case']['key'] in base]
            if pairs:comparisons.append(dict(task=task,sigma=sigma,cases=len(pairs),
                delta={k:statistics.mean(r['paired_metrics'][k]-base[r['case']['key']]['paired_metrics'][k] for r in pairs) for k in ('psnr','ssim','lpips')},
                images_PSNR_positive=sum(r['paired_metrics']['psnr']>base[r['case']['key']]['paired_metrics']['psnr'] for r in pairs),
                worst_PSNR=min(r['paired_metrics']['psnr']-base[r['case']['key']]['paired_metrics']['psnr'] for r in pairs)))
    fresh=[r for r in rows if not r.get('reused',False)];loads=[files.read(f) for f in (root/'workers').glob('*/load_*.json')]
    result=dict(experiment=p['experiment'],classification='FIXED_PANEL_MULTITASK_SHRINK_COMPARISON',completed_runs=len(done),planned_runs=72,
        image_clusters=3,groups=groups,paired_comparisons=comparisons,reused_complete_runs=sum(r.get('reused',False) for r in done),
        actual_new_cost=dict(method_forward_attempts=sum((r.get('full_counts') or {}).get('forward_attempts',0) for r in fresh),
            diagnostic_forward_attempts=sum((r.get('replay_counts') or {}).get('forward_attempts',0) for r in fresh),
            optimizer_backward_attempts=sum((r.get('full_counts') or {}).get('optimizer_backward_attempts',0) for r in fresh),
            input_vjp_attempts=sum((r.get('full_counts') or {}).get('input_vjp_attempts',0) for r in fresh),
            warmup_forward_attempts=sum(r['warmup_forward_attempts'] for r in loads)),
        cost_is_lower_bound=any(r['status']!='complete' for r in fresh),limits=p['protocol']['scope'])
    write(root/'summary.json',result);print('A71 Summary:',json.dumps(result,ensure_ascii=False),flush=True)

def report(root):
    from utils.render_shrink_panel import render
    render(Path(root))

def archive(root):
    from utils import archive_refinement_confirmation as exporter
    exporter.BRANCH=BRANCH;bundle,commit=exporter.export_and_commit({'A71':Path(root)},projects=PROJECTS)
    print('Archive:',bundle,'Results commit:',commit,'Branch:',BRANCH,flush=True)
    subprocess.run(['git','push','origin',BRANCH],cwd=PROJECTS.parent,check=True);print('RESULTS_PUSH_PASS: A71 results and figures uploaded.',flush=True)

def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=ROOT);p.add_argument('--a70-root',default=PARENT)
    p.add_argument('--ffhq-dir',default='/home/lzy/FFHQ/00000/00000');p.add_argument('--shard',type=int,choices=(0,1));p.add_argument('--device-uuid')
    p.add_argument('--check-worker-cli',action='store_true',help=argparse.SUPPRESS)
    g=p.add_mutually_exclusive_group(required=True)
    for n in ('prepare','worker','summarize','report','archive'):g.add_argument('--'+n,action='store_true')
    return p

def main():
    p=parser();a=p.parse_args()
    if a.worker and (a.shard is None or not a.device_uuid):p.error('Worker needs shard and UUID')
    if a.check_worker_cli:
        files.require(a.worker,'Need worker');print(json.dumps(dict(shard=a.shard,device_uuid=a.device_uuid,model_calls=0)));return
    if a.prepare:prepare(a)
    elif a.worker:worker(a)
    elif a.summarize:summarize(a.root)
    elif a.report:report(a.root)
    else:archive(a.root)
if __name__=='__main__':main()
