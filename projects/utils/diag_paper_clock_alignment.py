"""A64 server driver: public-setting baseline first, complete reconstruction wall-clock."""
import argparse
from contextlib import nullcontext
import csv
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS)); sys.path.insert(0,str(PROJECTS.parent))
from utils import paper_clock_alignment as rules
from utils import paper_clock_runtime as runtime

CARD = PROJECTS/'experiment_cards/A64_paper_clock_alignment_20261028.md'
MANIFEST = PROJECTS/'reports/20261027_completed_transfer_received/full_manifest.json'
COHORT = PROJECTS/'reports/20261027_a56_ffhq_received/A56_plan.json'


def finite(value):
    if isinstance(value,dict): return {str(k):finite(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [finite(v) for v in value]
    if hasattr(value,'item'): return finite(value.item())
    if isinstance(value,float) and not math.isfinite(value): return None
    return value


def write(path,value): rules.write(path,finite(value))


def plan(root):
    p = rules.read(root/'plan.json')
    rules.require(p['experiment']==rules.EXPERIMENT and p['grid']==rules.grid(), 'Wrong frozen plan')
    for path,sha in p['source_sha256'].items():
        rules.require(rules.digest(path)==sha, 'Source identity changed: '+path)
    return p


def prepare(args):
    root = Path(args.root).resolve()
    if (root/'plan.json').exists():
        plan(root); print('A64 frozen plan reused; no re-selection or setting changes.',flush=True); return
    rules.require(not root.exists() or not any(root.iterdir()), 'Unplanned nonempty output; do not overwrite')
    cohort,manifest = rules.read(COHORT),rules.read(MANIFEST)
    cases = rules.cases(dict(ffhq=args.ffhq_dir,imagenet=args.imagenet_dir))
    for c in cases:
        references = [v for v in cohort['cases'] if v['dataset']==c['dataset'] and v['image_id']==c['image_id']]
        rules.require(references and all(v['image_sha256']==c['image_sha256'] for v in references),
                      'Frozen cohort hash mismatch')
        rules.require(rules.digest(c['image'])==c['image_sha256'], 'Image missing/changed: '+c['image'])
        if c['task'] in ('super_resolution','inpainting'):
            relative = f'measurements/{c["dataset"]}/{c["task"]}/{c["image_id"]}.pt'
            items = [v for v in manifest['excluded'] if v['path']==relative]
            rules.require(len(items)==1, 'Missing binary observation identity: '+relative)
            path = str(Path(manifest['source'])/relative)
            rules.require(rules.digest(path)==items[0]['sha256'], 'Observation fixture missing/changed: '+path)
            c['source_observation'] = dict(path=path,sha256=items[0]['sha256'])
    for spec in rules.CHECKPOINTS.values():
        rules.require(rules.digest(spec['path'])==spec['sha256'], 'Checkpoint identity mismatch')
    # Bind tracked Python dependencies, not untracked paused experiment drafts.
    names = subprocess.check_output(['git','ls-files','projects/modules','projects/utils','projects/configs',
                                    'DPS/motionblur','DPS/guided_diffusion'],cwd=PROJECTS.parent,text=True).splitlines()
    sources = [PROJECTS.parent/name for name in names if name.endswith('.py')]
    sources += [CARD,Path(__file__),Path(rules.__file__),Path(runtime.__file__),MANIFEST,COHORT,
                PROJECTS/'run_a64_20261028.sh',PROJECTS/'utils/a64_session.py']
    p = dict(experiment=rules.EXPERIMENT, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        cases=cases,grid=rules.grid(),configs={t:rules.config(t) for t in rules.TASKS},
        checkpoints=rules.CHECKPOINTS, repeats=rules.REPEATS,queue=rules.queue(),budget=rules.budget(),
        published_means=rules.PAPER,source_sha256={str(f):rules.digest(f) for f in sorted(set(sources))},
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECTS.parent,text=True).strip(),
        protocol=dict(measurement='H((x+1)/2), sigma_y=.05; exact derivative adjoint .5 H^T',
            model_domain=[-1,1],size=[256,256],precision='IEEE float32 for ALL arms',
            output='last optimization unroll before final Adam update; no post-update extra sample',
            independent_initialization=True, learned_cache_input=False,
            response='K||v||u/||u||, cosine=1; CG true residual 1e-5, cap128; explicit q fallback',
            fixed30_fallback='Fixed task initial zeta/D; different from old learned fallback',
            timing='Warm-model complete reconstruction: observation load, operator/parameters, RNG, all '
                   'optimization/CG/VJP, final CPU copy and CUDA sync. Metrics/disk writes excluded.',
            cold_start='Model load and warmup separately recorded; cold single-arm latency NOT measured',
            author_implementation_parity=False, benchmark_reproduced=False,
            gt_usage='Observation simulation and offline metrics/baseline severe-gap audit only; never action/output selection'))
    root.mkdir(parents=True,exist_ok=True); rules.write(root/'plan.json',p)
    shutil.copyfile(CARD,root/'experiment_card.md')
    print('A64 PREPARE PASS: 2 images/dataset x4 tasks; baseline audit first; '
          'max30240 method forwards + separately counted warmup; three complete timings.',flush=True)


def make_operator(c,device,kernel,mask=None):
    import torch
    from utils.diag_ffhq_four_tasks_unit import make_base,unit_operator
    base = make_base(c['task'],device,torch.empty(1,3,256,256,device=device),c['mask_seed'],kernel)
    if mask is not None: base._mask_cache=mask.to(device)
    return unit_operator(base).to(device),base


def fixture(root,c,device,kernel):
    """Given observation only; no learned parameters are serialized here."""
    import torch
    from modules.main_single import load_image_as_tensor
    from utils.diag_multitask_state_transfer import operator_gate
    path=root/'observations'/(c['key']+'.pt'); meta=path.with_suffix('.json')
    if meta.exists():
        item=rules.read(meta); rules.require(rules.digest(path)==item['sha256'],'Own observation changed'); return item
    op,base=make_operator(c,device,kernel)
    gate=operator_gate(op,device)
    with torch.no_grad():
        gt=load_image_as_tensor(c['image']).to(device)
        mean=op.H(gt).cpu()
        if 'source_observation' in c:
            old=torch.load(c['source_observation']['path'],map_location='cpu',weights_only=True)
            rules.require(old['image_sha256']==c['image_sha256'] and old['seed']==c['measurement_seed'],
                          'Source observation metadata mismatch')
            error=float((mean.double()-old['mean'].double()).norm()/old['mean'].double().norm().clamp(min=1e-30))
            rules.require(error<2e-5,'Observation/operator compatibility failed')
            mean,noise=old['mean'],old['standard_noise']
        else:
            error=0.
            generator=torch.Generator(device='cpu').manual_seed(c['measurement_seed'])
            noise=torch.randn(mean.shape,generator=generator,dtype=torch.float32)
        payload=dict(mean=mean,standard_noise=noise,image_sha256=c['image_sha256'],seed=c['measurement_seed'])
        if c['task']=='inpainting': payload['mask']=base._mask_cache.cpu()
    path.parent.mkdir(exist_ok=True); torch.save(payload,path)
    item=dict(path=str(path),sha256=rules.digest(path),operator_gate=gate,
              source_observation=c.get('source_observation'),relative_mean_error=error,case=c)
    write(meta,item); return item


def gpu_inventory():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free',
                                    '--format=csv,noheader,nounits'],text=True)
    return [dict(index=a.strip(),uuid=b.strip(),free_bytes=int(d.strip())*1024**2)
            for a,b,d in (line.split(',') for line in output.splitlines() if line.strip())]


def resource(device_uuid,dataset,own_reserved=0):
    row=next(r for r in gpu_inventory() if r['uuid']==device_uuid)
    try:
        text=subprocess.check_output(['nvidia-smi','--id='+device_uuid,
            '--query-gpu=utilization.gpu,utilization.memory,clocks.sm,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'],text=True).strip()
        load=dict(zip(('gpu_util_percent','memory_util_percent','sm_clock_MHz','temperature_C','power_W'),
                      [v.strip() for v in text.split(',')]))
    except subprocess.CalledProcessError as error:
        load=dict(unavailable=repr(error))
    return dict(**row,own_reserved_bytes=own_reserved,required_bytes=rules.MIN_FREE[dataset],
                admitted=row['free_bytes']+own_reserved>=rules.MIN_FREE[dataset],load_snapshot=load,
                timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))


def completed(root,c,arm,repeat):
    directory=root/'cases'/c['key']/arm/f'repeat_{repeat}'
    record=directory/'run.json'
    if not directory.exists(): return None
    rules.require(record.exists(), 'Unjournaled prior attempt retained: '+str(directory))
    r=rules.read(record)
    rules.require(r['status']=='complete', 'Prior started/failed attempt retained; no automatic retry: '+str(record))
    rules.require(r['plan_sha256']==rules.digest(root/'plan.json') and r['case']==c and
                  r['arm']==arm and r['repeat']==repeat, 'Completed identity differs')
    runtime.validate_counts(r['counts'],arm)
    for name,sha in r['artifacts'].items():
        rules.require(rules.digest(directory/name)==sha,'Completed artifact changed')
    return r


def measure(root,p,c,arm,repeat,model,device_uuid,kernel,versions):
    import torch
    from utils.current_score_runtime_r2 import input_vjp_parameters
    from utils.metrics import compute_all_metrics
    from modules.main_single import load_image_as_tensor
    from modules.dataset_loader import tensor_to_image
    existing=completed(root,c,arm,repeat)
    if existing: return existing
    admission=resource(device_uuid,c['dataset'],torch.cuda.memory_reserved())
    if not admission['admitted']: return None
    fixture_info=fixture(root,c,model.device,kernel)
    # Prevent duplicate/failed attempts from being silently rewritten.
    directory=root/'cases'/c['key']/arm/f'repeat_{repeat}'
    directory.mkdir(parents=True,exist_ok=False)
    r=dict(status='running',case=c,arm=arm,repeat=repeat,plan_sha256=rules.digest(root/'plan.json'),
           device_uuid=device_uuid,resource_admission=admission,observation=fixture_info,
           warm_complete_reconstruction_seconds=None,artifacts={},cost_lower_bound=True)
    write(directory/'run.json',r)
    count=None; beginning=time.perf_counter()
    def tick(value):
        r['counts']=value
        write(directory/'run.json',r)
        write(root/(c['dataset']+'_progress.json'),dict(case=c['key'],arm=arm,repeat=repeat,counts=value))
        print('A64 PROGRESS:',c['key'],arm,'repeat',repeat,'F=',value['forwards'],flush=True)
    try:
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        # Shared frozen-model input-VJP compatibility context; no weight update in any arm.
        with input_vjp_parameters(model.model), runtime.counters(model.model,tick) as count:
            def operation():
                data=torch.load(fixture_info['path'],map_location='cpu',weights_only=True)
                y=(data['mean']+rules.SIGMA*data['standard_noise']).to(model.device)
                mask=data.get('mask')
                operator,base=make_operator(c,model.device,kernel,mask)
                runtime.count_operator(base,count)
                return runtime.execute(model,operator,y,c['task'],arm,count,
                                       mask=None if mask is None else mask.to(model.device))
            (out,details),seconds=runtime.clocks(torch.cuda.synchronize,operation)
        runtime.validate_counts(count,arm)
        r['load_after_timing']=resource(device_uuid,c['dataset'],torch.cuda.memory_reserved())
        rules.require(all(math.isfinite(v) for v in details['losses']), 'Nonfinite training loss')
        rules.require(all(v._version==version and v.grad is None for v,version in versions),
                      'Frozen model weights changed or parameter gradients populated')
        if arm=='RESPONSE_300':
            baseline=torch.load(root/'cases'/c['key']/'PAPER_ZAPS'/'repeat_0'/'entry_snapshot.pt',
                                map_location='cpu',weights_only=True)
            match=runtime.snapshot_equal(baseline,details['snapshot'])
            rules.require(match['passed'],'Independent270 prefix does not match baseline final entry')
            r['independent_prefix_gate']=match
        elif arm=='RESPONSE_30':
            baseline=torch.load(root/'cases'/c['key']/'PAPER_ZAPS'/'repeat_0'/'entry_snapshot.pt',
                                map_location='cpu',weights_only=True)
            matching=runtime.snapshot_equal(baseline,details['snapshot'])
            rules.require(matching['rng_equal'] and matching['relative_errors']['initial']==0.,
                          'Independent30 initial/noise draw alignment failed')
            r['noise_pairing_gate']=dict(passed=True,learned_parameters_read_for_inference=False,
                audit_after_timing=True,learned_parameter_match_expected=False)
        if repeat:
            prior=torch.load(root/'cases'/c['key']/arm/'repeat_0'/'output.pt',map_location='cpu',weights_only=True)
            error=float((out.double()-prior.double()).norm()/prior.double().norm().clamp(min=1e-30))
            rules.require(error<=1e-6,'Repeated output changed; no best-of-repeat selection')
            r['repeat_relative_output_error']=error
        metric_start=time.perf_counter()
        gt=load_image_as_tensor(c['image'])
        metrics=compute_all_metrics(out,gt,lpips_net='alex')
        float_mse=float(((out.clamp(-1,1).double()-gt.double())*.5).square().mean())
        metrics['float_psnr']=-10*math.log10(max(float_mse,1e-30))
        rules.require(all(math.isfinite(float(v)) for v in metrics.values()),'Nonfinite final metrics')
        metrics_seconds=time.perf_counter()-metric_start
        torch.save(out,directory/'output.pt'); torch.save(details['snapshot'],directory/'entry_snapshot.pt')
        tensor_to_image(out.squeeze(0),denormalize=True).save(directory/'output.png')
        if not (root/'observations'/(c['key']+'_gt.png')).exists():
            tensor_to_image(gt.squeeze(0),denormalize=True).save(root/'observations'/(c['key']+'_gt.png'))
        write(directory/'trace.json',details['trace'])
        r.update(status='complete',counts=count,metrics=metrics,loss_history=details['losses'],
            warm_complete_reconstruction_seconds=seconds,metric_seconds_excluded=metrics_seconds,
            all_instrumented_attempt_seconds=time.perf_counter()-beginning,cost_lower_bound=False,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            response_resolved_steps=sum(bool(v.get('resolved')) for v in details['trace']),
            response_fallback_steps=sum(not bool(v.get('resolved')) for v in details['trace']),
            artifacts={name:rules.digest(directory/name) for name in ('output.pt','entry_snapshot.pt','output.png','trace.json')})
        write(directory/'run.json',r)
        print('A64 COMPLETE:',c['key'],arm,'repeat',repeat,'PSNR=',round(float(metrics['psnr']),4),
              'full_seconds=',round(seconds,3),'F=',count['forwards'],flush=True)
        return r
    except BaseException as error:
        r.update(status='failed',error=repr(error),traceback=traceback.format_exc(),counts=count,
                 elapsed_until_failure=time.perf_counter()-beginning,cost_lower_bound=True)
        write(directory/'run.json',r); raise


def worker(args):
    import torch
    from modules.main_single import load_diffusion_model
    from utils.current_score_runtime_r2 import input_vjp_parameters,raw_mean
    from utils.current_score_geometry import input_vjp
    from utils.diag_clean_response_precision import ieee,precision_state
    from utils.diag_ffhq_four_tasks_unit import original_motion_kernel
    from utils.a54_session import exclusive
    root=Path(args.root).resolve(); p=plan(root); dataset=args.dataset
    folder=root/'workers'/dataset; folder.mkdir(parents=True,exist_ok=True)
    with exclusive(folder/'worker.lock'):
        rules.require(torch.cuda.is_available(),'CUDA required, no CPU fallback')
        rules.require(rules.digest(p['checkpoints'][dataset]['path'])==p['checkpoints'][dataset]['sha256'],
                      'Checkpoint changed since prepare')
        for case in (c for c in p['cases'] if c['dataset']==dataset):
            rules.require(rules.digest(case['image'])==case['image_sha256'],'Image changed since prepare')
        identity=folder/'device.json'
        if identity.exists():
            rules.require(rules.read(identity)['uuid']==args.device_uuid,'Timing GPU UUID must remain fixed')
        admission=resource(args.device_uuid,dataset)
        if not admission['admitted']:
            write(folder/'status.json',dict(status='resource_limited',admission=admission)); return
        write(identity,dict(uuid=args.device_uuid,visible=os.environ.get('CUDA_VISIBLE_DEVICES')))
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        start=time.perf_counter(); model=load_diffusion_model(dataset,'cuda:0'); model.model.eval()
        torch.cuda.synchronize(); load_seconds=time.perf_counter()-start
        entry=dict(status='running',pid=os.getpid(),dataset=dataset,uuid=args.device_uuid,
                   model_load_seconds=load_seconds,torch=str(torch.__version__),cuda=torch.version.cuda,
                   gpu_name=torch.cuda.get_device_name(0),cudnn_benchmark=False,cudnn_deterministic=True)
        entry['packages']={}
        for name in ('numpy','scipy','PyWavelets','scikit-image','lpips','Pillow'):
            try: entry['packages'][name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: entry['packages'][name]='not_installed'
        number=len(list(folder.glob('load_*.json')))
        entry_file=folder/f'load_{number:03d}.json'; write(entry_file,entry)
        try:
            with ieee(torch):
                entry['precision']=precision_state(torch)
                warm=dict(status='running',forward_attempts=0,forwards=0,input_vjp_attempts=0)
                entry['warmup']=warm; write(entry_file,entry)
                torch.cuda.synchronize(); warm_start=time.perf_counter()
                with input_vjp_parameters(model.model),runtime.counters(model.model) as count:
                    warm['forward_attempts']=1; write(entry_file,entry)
                    x=torch.zeros(1,3,256,256,device=model.device)
                    inp,mean=raw_mean(model,x,999,graph=True)
                    warm['forwards']=1; warm['input_vjp_attempts']=1; write(entry_file,entry)
                    result=input_vjp(mean,inp,torch.ones_like(mean))
                    rules.require(bool(torch.isfinite(result).all()),'Warmup input VJP nonfinite')
                torch.cuda.synchronize(); warm.update(status='complete',seconds=time.perf_counter()-warm_start)
                entry['warmup_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
                del x,inp,mean,result; gc.collect(); torch.cuda.empty_cache(); write(entry_file,entry)
                versions=[(v,v._version) for v in model.model.parameters()]
                kernel=original_motion_kernel(0)
                kernel_file=folder/'motion_kernel.pt'
                if kernel_file.exists():
                    old=torch.load(kernel_file,map_location='cpu',weights_only=True)
                    rules.require(torch.equal(old,torch.from_numpy(kernel)),'Frozen motion kernel changed')
                else: torch.save(torch.from_numpy(kernel),kernel_file)
                entry['motion_kernel_sha256']=rules.digest(kernel_file)
                cases=[c for c in p['cases'] if c['dataset']==dataset]
                for c in cases:
                    r=measure(root,p,c,'PAPER_ZAPS',0,model,args.device_uuid,kernel,versions)
                    if r is None:
                        entry['status']='resource_limited'; return
                gate=rules.baseline_gate(dataset,[completed(root,c,'PAPER_ZAPS',0) for c in cases])
                write(root/(dataset+'_baseline_gate.json'),gate)
                print('A64 BASELINE GATE:',dataset,gate['classification'],flush=True)
                if not gate['proceed']:
                    entry['status']='baseline_review_required'; return
                for arm,repeat in rules.queue()[1:]:
                    for c in cases:
                        r=measure(root,p,c,arm,repeat,model,args.device_uuid,kernel,versions)
                        if r is None:
                            entry['status']='resource_limited'; return
                entry['status']='complete'
        except BaseException as error:
            entry.update(status='failed',error=repr(error),traceback=traceback.format_exc()); raise
        finally:
            entry['total_worker_seconds']=time.perf_counter()-start
            write(entry_file,entry); write(folder/'status.json',entry)


def summarize(root):
    root=Path(root); p=plan(root); records=[]; attempts=[]
    for file in sorted((root/'cases').glob('*/*/repeat_*/run.json')):
        r=rules.read(file); attempts.append(r)
        if r['status']=='complete': records.append(r)
    groups=[]
    for dataset in rules.DATASETS:
        for task in rules.TASKS:
            for arm in rules.ARMS:
                rows=[r for r in records if r['case']['dataset']==dataset and r['case']['task']==task and r['arm']==arm]
                quality=[r for r in rows if r['repeat']==0]
                full=len(rows)==2*rules.REPEATS
                group=dict(dataset=dataset,task=task,arm=arm,completed_runs=len(rows),planned_runs=6,
                           image_clusters=len(quality),paper_means=rules.PAPER[dataset].get(task))
                if quality:
                    group['mean_quality']={k:statistics.mean(float(r['metrics'][k]) for r in quality)
                                           for k in ('psnr','ssim','lpips','float_psnr')}
                if full:
                    by_image={image:rules.timing_summary([r['warm_complete_reconstruction_seconds']
                        for r in sorted(rows,key=lambda a:a['repeat']) if r['case']['image_id']==image])
                        for image in sorted({r['case']['image_id'] for r in rows})}
                    group['complete_reconstruction_seconds']=dict(per_image=by_image,
                        mean_of_image_medians=statistics.mean(v['median'] for v in by_image.values()))
                if quality and arm!='PAPER_ZAPS':
                    pairs=[]
                    for row in quality:
                        reference=next(r for r in records if r['case']==row['case'] and r['arm']=='PAPER_ZAPS' and r['repeat']==0)
                        pairs.append(dict(image_id=row['case']['image_id'],
                            **{k:float(row['metrics'][k])-float(reference['metrics'][k]) for k in ('psnr','ssim','lpips')}))
                    group['paired_delta_vs_public_setting_baseline']=dict(per_image=pairs,
                        mean={k:statistics.mean(r[k] for r in pairs) for k in ('psnr','ssim','lpips')},
                        psnr_positive_images=sum(r['psnr']>0 for r in pairs))
                if rows:
                    group['mean_fallback_steps']=statistics.mean(r['response_fallback_steps'] for r in rows)
                groups.append(group)
    loads=[rules.read(f) for f in sorted((root/'workers').glob('*/load_*.json'))]
    warmups=[r.get('warmup',{}) for r in loads]
    gates={d:rules.read(root/(d+'_baseline_gate.json')) if (root/(d+'_baseline_gate.json')).exists() else None
           for d in rules.DATASETS}
    statuses={d:rules.read(root/'workers'/d/'status.json').get('status') if
              (root/'workers'/d/'status.json').exists() else 'not_started' for d in rules.DATASETS}
    summary=dict(experiment=rules.EXPERIMENT,classification='BASELINE_AND_WALLCLOCK_AUDIT_ONLY',
        dataset_status=statuses,baseline_gates=gates,complete_runs=len(records),planned_max_runs=144,
        failed_or_started_runs=len(attempts)-len(records),groups=groups,
        costs=dict(method_forward_attempts=sum((r.get('counts') or {}).get('forward_attempts',0) for r in attempts),
                   method_forwards_completed=sum((r.get('counts') or {}).get('forwards',0) for r in attempts),
                   method_input_vjp_attempts=sum((r.get('counts') or {}).get('input_vjp_attempts',0) for r in attempts),
                   optimizer_backward_attempts=sum((r.get('counts') or {}).get('optimizer_backward_attempts',0) for r in attempts),
                   diagnostic_warmup_forward_attempts=sum(r.get('forward_attempts',0) for r in warmups),
                   diagnostic_warmup_input_vjp_attempts=sum(r.get('input_vjp_attempts',0) for r in warmups),
                   cost_is_lower_bound=any(r['status']!='complete' for r in attempts) or
                                       any(r.get('status')!='complete' for r in warmups)),
        model_loads=loads,limits='TWO reused images/dataset, one seed, public major settings/local undisclosed '
        'conventions; not author parity or paper1000-image reproduction. Warm complete reconstruction timing '
        'includes ALL required learning/solves/backwards/transfers; model load/warmup separately listed. '
        'Shared GPU/fixed small cohort do not establish production speed/generalization. Failed baseline '
        'gate prevents candidate runs for entire dataset; no GT online action or best-output selection.')
    write(root/'summary.json',summary)
    rows=[]
    for g in groups:
        row={k:g[k] for k in ('dataset','task','arm','completed_runs','image_clusters')}
        row.update(g.get('mean_quality',{}))
        row['warm_full_seconds_mean_of_image_medians']=g.get('complete_reconstruction_seconds',{}).get('mean_of_image_medians')
        rows.append(row)
    with (root/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader(); writer.writerows(rows)
    display_report(root,groups,statuses)
    print('A64 Summary:',json.dumps(finite(summary),ensure_ascii=False),flush=True)
    return summary


def display_report(root,groups,statuses):
    names=dict(PAPER_ZAPS='公开设置原采样（保留学习权重与小波参数）',
               RESPONSE_300='响应引导＋独立完整学习前缀',RESPONSE_30='独立30步响应引导（固定初值回退）')
    lines=['# 公开设置基线与完整重建计时对齐', '',
        '服务器实测结果。每数据集仅2张旧图、一个seed；不是原文1000图复现或作者实现认证。',
        '粗体/下划线仅标记**同两图、本地三版本均完成时**的最优/次优，不给跨样本原文行排名。',
        '时间为热模型完整重建：包含观测加载、算子/参数构造、所有学习/求解/反向、CPU输出和CUDA同步；',
        '模型加载/热身另见summary.json；冷启动单张总时间未测。GPU共享负载限制速度结论。', '']
    for dataset in rules.DATASETS:
        lines += [f'## {dataset}：{statuses[dataset]}', '']
        for task in rules.TASKS:
            selected=[g for g in groups if g['dataset']==dataset and g['task']==task]
            complete=all(g['image_clusters']==2 and g['completed_runs']==6 for g in selected)
            rankings={}
            if complete:
                for metric in ('psnr','ssim','lpips'):
                    ordered=sorted(selected,key=lambda g:g['mean_quality'][metric],reverse=metric!='lpips')
                    rankings[metric]={g['arm']:i for i,g in enumerate(ordered)}
            lines += [f'### {task}', '', '| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | 完整重建秒↓ | 前向/输入反向/参数反向 |',
                      '|---|---:|---:|---:|---:|---|']
            published=rules.PAPER[dataset].get(task)
            if published:
                lines.append(f'| 原文1000图均值（不同样本；A100时间不作同机比较） | {published["psnr"]:.4f} | '
                             f'{published["ssim"]:.4f} | {published["lpips"]:.4f} | 未提供匹配本机时间 | 300/未核验/10 |')
            else: lines.append('| 原文 | 未报告该任务均值 | — | — | — | — |')
            for group in selected:
                arm=group['arm']; quality=group.get('mean_quality'); values=[]
                for metric in ('psnr','ssim','lpips'):
                    value=f'{quality[metric]:.4f}' if quality else '未运行'
                    rank=rankings.get(metric,{}).get(arm)
                    if rank==0:value='**'+value+'**'
                    elif rank==1:value='<u>'+value+'</u>'
                    values.append(value)
                timing=group.get('complete_reconstruction_seconds')
                seconds=f'{timing["mean_of_image_medians"]:.3f}' if timing else '未完成3次计时'
                cost=f'{rules.NFE[arm]}/'+('0' if arm=='PAPER_ZAPS' else '≤30')+f'/{rules.UPDATES[arm]}'
                lines.append('| '+names[arm]+' | '+' | '.join(values)+f' | {seconds} | {cost} |')
            lines.append('')
    lines += ['## 解释限制', '',
        '通过严重失配警报只表示本小样本未触发工程保护线，不证明复现成功。',
        '原采样和完整前缀响应均300次前向，但反向、算子、求解代价不同，不能称等时间。',
        '独立30步响应从初值开始；若求解失败，使用固定初值回退，不继承学习后回退的收益。',
        '若基线触发审计停机，本表保留基线结果，方法收益结论继续暂停。', '']
    (root/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')


def archive(root):
    from utils import archive_refinement_confirmation as exporter
    exporter.BRANCH=rules.BRANCH
    bundle,commit=exporter.export_and_commit({'A64':Path(root)},projects=PROJECTS)
    print('Archive:',bundle,'Results commit:',commit,'Branch:',rules.BRANCH,flush=True)
    result=subprocess.run(['git','push','origin',rules.BRANCH],cwd=PROJECTS.parent)
    if result.returncode: raise RuntimeError('Results saved locally but push failed; no GPU rerun needed')
    print('RESULTS_PUSH_PASS: A64 public-setting baseline and complete wall-clock uploaded.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=rules.ROOT)
    parser.add_argument('--ffhq-dir',default='/home/lzy/FFHQ/00000/00000')
    parser.add_argument('--imagenet-dir',default='/home/lzy/imagenet/256x256')
    parser.add_argument('--dataset',choices=rules.DATASETS);parser.add_argument('--device-uuid')
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',action='store_true')
    parser.add_argument('--summarize',action='store_true');parser.add_argument('--archive',action='store_true')
    args=parser.parse_args()
    if args.prepare: prepare(args)
    elif args.worker: worker(args)
    elif args.summarize: summarize(args.root)
    elif args.archive: archive(args.root)
    else: parser.error('Choose --prepare/--worker/--summarize/--archive')


if __name__=='__main__': main()
