"""A65: reproduce the September17 strong FFHQ protocol before evaluating new directions."""
import argparse
import csv
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

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import historical_ffhq_gain as rules
from utils import historical_ffhq_runtime as engine
from utils import paper_clock_runtime as common
from utils.diag_paper_clock_alignment import write,resource,gpu_inventory,finite
CARD=PROJECTS/'experiment_cards/A65_historical_ffhq_method_gain_20261028.md'


def prepare(args):
    root=Path(args.root).resolve()
    if (root/'plan.json').exists():
        check(root);print('A65 frozen plan reused.',flush=True);return
    rules.require(not root.exists() or not any(root.iterdir()),'Unplanned nonempty root; do not overwrite')
    root.parent.mkdir(parents=True,exist_ok=True)
    rules.require(shutil.disk_usage(root.parent).free>=5*rules.GIB,'Need at least5GiB free on result filesystem')
    cases=rules.cases(args.ffhq_dir)
    for case in cases:case['image_sha256']=rules.digest(case['image'])
    checkpoint=dict(path=str(PROJECTS/'modules/models/ffhq_10m.pt'),
        sha256='81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa')
    rules.require(rules.digest(checkpoint['path'])==checkpoint['sha256'],'FFHQ checkpoint changed')
    names=subprocess.check_output(['git','ls-files','projects/modules','projects/utils','projects/configs',
                                   'DPS/guided_diffusion','DPS/util'],cwd=PROJECTS.parent,text=True).splitlines()
    sources=[PROJECTS.parent/n for n in names if n.endswith('.py')]
    sources += [CARD,rules.CSV_PATH,rules.AUDIT_PATH,PROJECTS/'run_a65_20261028.sh']
    p=dict(experiment=rules.EXPERIMENT,cases=cases,grid=rules.grid(),config=rules.config(),
        checkpoint=checkpoint,arms=list(rules.ARMS),queue=rules.queue(),budget=rules.budget(),
        reference_csv=str(rules.CSV_PATH),reference_csv_sha256=rules.digest(rules.CSV_PATH),
        source_sha256={str(f):rules.digest(f) for f in sources},
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECTS.parent,text=True).strip(),
        protocol=dict(domain='model [-1,1], sigma=.05',forward='DPS Resizer4',
            historical_mapping='bicubic upsample, empirical not exact adjoint',
            response_linear='H, not .5H; true H adjoint for covariance solve',
            trajectory='fixed 30-point 9/17 paper15/10/5, eta1 DDPM fixed-small',
            learned_weights='zeta and D learned in baseline/full-prefix; fixed initial only in standalone30',
            GT='synthetic observation and offline replay/metrics only; never online action or output selection',
            wallclock='Warm-model complete reconstruction from observation read/operator/parameter/RNG setup '
                      'through all learning/CG/VJP and CPU output sync; metrics and serialization excluded',
            primary='Independent30 vs BOTH native-history and IEEE-history. Other arms secondary, no winner selection',
            limitations='Ten reused development images, one seed; no unseen confirmation, author-code parity, '
                        'cross-task/data transfer, step-size or curvature-control proof'))
    root.mkdir(parents=True,exist_ok=True);write(root/'plan.json',p);shutil.copyfile(CARD,root/'experiment_card.md')
    print('A65 PREPARE/FREEZE PASS: original10 FFHQ images,5 arms,3 full timings; '
          'max36900 method forwards,<=1800 input VJP; baseline replay gate first.',flush=True)


def check(root):
    p=rules.read(root/'plan.json');rules.require(p['experiment']==rules.EXPERIMENT,'Wrong plan')
    for path,sha in p['source_sha256'].items():rules.require(rules.digest(path)==sha,'Source changed: '+path)
    return p


def observation(root,case,device):
    import torch
    import torch.nn.functional as F
    from modules.main_single import load_image_as_tensor
    from utils.diag_ffhq_regression import tensor_psnr,set_seed
    from utils.diag_multitask_state_transfer import operator_gate
    folder=root/'observations';folder.mkdir(exist_ok=True)
    path=folder/(case['key']+'.pt');meta=path.with_suffix('.json')
    if meta.exists():
        data=rules.read(meta);rules.require(rules.digest(path)==data['sha256'],'Given observation changed');return data
    with engine.precision('HISTORICAL_NATIVE'),torch.no_grad():
        op=engine.make_operator(device)
        # Test the REAL H/H^T separately, never pretend the historical upsampler is its adjoint.
        gate=operator_gate(op.base,device)
        gt=load_image_as_tensor(case['image']).to(device)
        set_seed(case['seed'])
        mean=op.H(gt);y=mean+torch.randn_like(mean)*.05
        seen=tensor_psnr(gt,F.interpolate(y,size=(256,256),mode='bicubic',align_corners=False))
        torch.save(dict(y=y.cpu(),image_sha256=case['image_sha256'],seed=case['seed']),path)
    data=dict(path=str(path),sha256=rules.digest(path),observed_psnr=seen,
        reference_observed_psnr=case['reference']['observed_psnr'],true_operator_gate=gate,
        domain='H(x_model)+.05*N(0,I); no unit-domain conversion')
    write(meta,data);return data


def done(root,case,arm,repeat):
    folder=root/'cases'/case['key']/arm/f'repeat_{repeat}'
    if not folder.exists():return None
    file=folder/'run.json';rules.require(file.exists(),'Unjournaled attempt preserved')
    r=rules.read(file);rules.require(r['status']=='complete','Prior failed/started attempt; no automatic retry: '+str(file))
    rules.require(r['plan_sha256']==rules.digest(root/'plan.json') and r['case']==case and
                  (r['arm'],r['repeat'])==(arm,repeat),'Completed attempt identity differs')
    engine.validate_counts(r['counts'],arm)
    for name,sha in r['artifacts'].items():rules.require(rules.digest(folder/name)==sha,'Result bytes changed')
    return r


def measure(root,case,arm,repeat,model,uuid,versions):
    import torch
    from modules.main_single import load_image_as_tensor
    from modules.dataset_loader import tensor_to_image
    from utils.current_score_runtime_r2 import input_vjp_parameters
    from utils.metrics import compute_all_metrics
    old=done(root,case,arm,repeat)
    if old:return old
    admission=resource(uuid,'ffhq',torch.cuda.memory_reserved())
    if not admission['admitted']:return None
    obs=observation(root,case,model.device)
    folder=root/'cases'/case['key']/arm/f'repeat_{repeat}';folder.mkdir(parents=True,exist_ok=False)
    r=dict(status='running',case=case,arm=arm,repeat=repeat,plan_sha256=rules.digest(root/'plan.json'),
           device_uuid=uuid,observation=obs,resource_before=admission,counts=None,artifacts={},cost_lower_bound=True)
    write(folder/'run.json',r);start=time.perf_counter();counts=None
    def tick(value):
        r['counts']=value;write(folder/'run.json',r)
        print('A65 PROGRESS:',case['image_id'],arm,'repeat',repeat,'F=',value['forwards'],flush=True)
    try:
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        with engine.precision(arm) as policy,input_vjp_parameters(model.model),common.counters(model.model,tick) as counts:
            def operation():
                payload=torch.load(obs['path'],map_location='cpu',weights_only=True)
                y=payload['y'].to(model.device)
                mode='exact' if arm=='EXACT_IEEE' else 'legacy_bicubic'
                operator=engine.make_operator(model.device,counts,mode=mode)
                return engine.execute(model,operator,y,case['seed'],arm,counts)
            (out,details),seconds=common.clocks(torch.cuda.synchronize,operation)
        engine.validate_counts(counts,arm)
        rules.require(all(math.isfinite(float(v)) for v in details['losses']),'Nonfinite loss')
        rules.require(all(v._version==version and v.grad is None for v,version in versions),'Frozen weights changed')
        if arm=='RESPONSE_300':
            reference=root/'cases'/case['key']/'LEGACY_IEEE'/'repeat_0'/'entry_snapshot.pt'
            gate=common.snapshot_equal(torch.load(reference,map_location='cpu',weights_only=True),details['snapshot'])
            r['independent_prefix_gate']=gate;rules.require(gate['passed'],'Independent270 prefix differs')
        elif arm=='RESPONSE_30':
            reference=root/'cases'/case['key']/'LEGACY_IEEE'/'repeat_0'/'entry_snapshot.pt'
            gate=common.snapshot_equal(torch.load(reference,map_location='cpu',weights_only=True),details['snapshot'])
            rules.require(gate['rng_equal'] and gate['relative_errors']['initial']==0.,'Independent noise pairing differs')
            r['noise_pairing_gate']=dict(passed=True,learned_cache_read_for_inference=False,audit_after_timer=True)
        if repeat:
            reference=torch.load(root/'cases'/case['key']/arm/'repeat_0'/'output.pt',map_location='cpu',weights_only=True)
            error=float((out.double()-reference.double()).norm()/reference.double().norm().clamp(min=1e-30))
            r['repeat_relative_error']=error;rules.require(error<=1e-6,'Repeat drift; no best-output selection')
        metric_start=time.perf_counter();gt=load_image_as_tensor(case['image'])
        metrics=compute_all_metrics(out,gt,lpips_net='alex')
        mse=float(((out.clamp(-1,1).double()-gt.double())*.5).square().mean())
        metrics['float_psnr']=-10*math.log10(max(mse,1e-30))
        rules.require(all(math.isfinite(float(v)) for v in metrics.values()),'Nonfinite metrics')
        metric_seconds=time.perf_counter()-metric_start
        torch.save(out,folder/'output.pt');torch.save(details['snapshot'],folder/'entry_snapshot.pt')
        tensor_to_image(out.squeeze(0),denormalize=True).save(folder/'output.png')
        if not (root/'observations'/(case['image_id']+'_gt.png')).exists():
            tensor_to_image(gt.squeeze(0),denormalize=True).save(root/'observations'/(case['image_id']+'_gt.png'))
        write(folder/'trace.json',details['trace'])
        r.update(status='complete',counts=counts,precision=policy,metrics=metrics,loss_history=details['losses'],
            complete_reconstruction_seconds=seconds,metric_seconds_excluded=metric_seconds,
            all_attempt_seconds=time.perf_counter()-start,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),resource_after=resource(uuid,'ffhq',torch.cuda.memory_reserved()),
            response_resolved_steps=sum(bool(v.get('resolved')) for v in details['trace']),
            fallback_steps=sum(not bool(v.get('resolved')) for v in details['trace']),cost_lower_bound=False,
            artifacts={name:rules.digest(folder/name) for name in ('output.pt','entry_snapshot.pt','trace.json','output.png')})
        write(folder/'run.json',r)
        print('A65 COMPLETE:',case['image_id'],arm,'repeat',repeat,'PSNR=',round(float(metrics['psnr']),4),
              'full_seconds=',round(seconds,3),flush=True)
        return r
    except BaseException as error:
        r.update(status='failed',counts=counts,error=repr(error),traceback=traceback.format_exc(),
                 elapsed_until_failure=time.perf_counter()-start,cost_lower_bound=True)
        write(folder/'run.json',r);raise


def null_gate(root,case,model):
    import torch
    from modules.zaps_algorithm import ZAPS
    from utils.current_score_runtime_r2 import input_vjp_parameters
    folder=root/'null_diagnostic';file=folder/'run.json'
    if folder.exists():
        old=rules.read(file);rules.require(old['status']=='complete' and old['passed'],'Failed null attempt retained');return
    folder.mkdir();r=dict(status='running',forwards=0,input_vjp_attempts=0,passed=False);write(file,r)
    counts=None
    try:
        base=root/'cases'/case['key']/'HISTORICAL_NATIVE'/'repeat_0'
        entry=torch.load(base/'entry_snapshot.pt',map_location='cpu',weights_only=True)
        obs=rules.read(root/'observations'/(case['key']+'.json'))
        y=torch.load(obs['path'],map_location=model.device,weights_only=True)['y']
        with engine.precision('HISTORICAL_NATIVE'),input_vjp_parameters(model.model),common.counters(model.model) as counts:
            z=ZAPS(model,engine.make_operator(model.device),**rules.config())
            with torch.no_grad():z.zeta.copy_(entry['zeta'].to(model.device));z.D.copy_(entry['D'].to(model.device))
            torch.set_rng_state(entry['cpu_rng']);torch.cuda.set_rng_state_all(entry['cuda_rng'])
            initial=entry['initial'].to(model.device);noises=[torch.randn_like(initial) for _ in range(29)]
            result,trace=common.response(z,y,initial,noises,None,counts,engine.fallback_sensor)
        expected=torch.load(base/'output.pt',map_location=model.device,weights_only=True)
        error=float((result.double()-expected.double()).norm()/expected.double().norm().clamp(min=1e-30))
        rules.require(counts['forwards']==30 and counts['input_vjp_attempts']==0 and error<=1e-6,'Null/core mismatch')
        r.update(status='complete',passed=True,forwards=30,relative_error=error,counts=counts)
    except BaseException as error:
        r.update(status='failed',error=repr(error),counts=counts);raise
    finally:write(file,r)


def worker(args):
    import torch
    from utils.a54_session import exclusive
    from modules.main_single import load_diffusion_model
    from utils.current_score_runtime_r2 import input_vjp_parameters,raw_mean
    from utils.current_score_geometry import input_vjp
    root=Path(args.root).resolve();p=check(root);name='shard'+str(args.shard)
    folder=root/'workers'/name;folder.mkdir(parents=True,exist_ok=True)
    with exclusive(folder/'worker.lock'):
        admission=resource(args.device_uuid,'ffhq')
        if not admission['admitted']:
            write(folder/('status_'+args.phase+'.json'),dict(status='resource_limited',admission=admission));return
        identity=folder/'device.json'
        if identity.exists():rules.require(rules.read(identity)['uuid']==args.device_uuid,'Timing GPU changed')
        write(identity,dict(uuid=args.device_uuid))
        rules.require(torch.cuda.is_available(),'CUDA required')
        rules.require(rules.digest(p['checkpoint']['path'])==p['checkpoint']['sha256'],'Checkpoint changed')
        cases=[c for i,c in enumerate(p['cases']) if i//5==args.shard]
        for c in cases:rules.require(rules.digest(c['image'])==c['image_sha256'],'Frozen image changed')
        started=time.perf_counter();model=load_diffusion_model('ffhq','cuda:0');model.model.eval();torch.cuda.synchronize()
        record=dict(status='running',phase=args.phase,shard=args.shard,uuid=args.device_uuid,
                    model_load_seconds=time.perf_counter()-started,torch=str(torch.__version__),cuda=torch.version.cuda,
                    gpu_name=torch.cuda.get_device_name(0),warmup=dict(status='running',forward_attempts=0,
                    forwards=0,input_vjp_attempts=0))
        number=len(list(folder.glob('load_*.json')));file=folder/f'load_{number:03d}.json';write(file,record)
        try:
            warm=record['warmup'];warm_start=time.perf_counter()
            with engine.precision('LEGACY_IEEE'),input_vjp_parameters(model.model):
                warm['forward_attempts']=1;write(file,record)
                inp,raw=raw_mean(model,torch.zeros(1,3,256,256,device=model.device),999,graph=True)
                warm.update(forwards=1,input_vjp_attempts=1);write(file,record)
                value=input_vjp(raw,inp,torch.ones_like(raw));rules.require(bool(torch.isfinite(value).all()),'Warmup nonfinite')
            torch.cuda.synchronize();warm.update(status='complete',seconds=time.perf_counter()-warm_start)
            del inp,raw,value;gc.collect();torch.cuda.empty_cache();write(file,record)
            versions=[(v,v._version) for v in model.model.parameters()]
            if args.phase=='baseline':queue=[('HISTORICAL_NATIVE',0)]
            else:
                rules.require(rules.read(root/'replay_gate.json')['passed'],'Historical gate not passed')
                queue=rules.queue()[1:]
            for arm,repeat in queue:
                for c in cases:
                    result=measure(root,c,arm,repeat,model,args.device_uuid,versions)
                    if result is None:record['status']='resource_limited';return
            if args.phase=='baseline' and args.shard==0:null_gate(root,cases[0],model)
            record['status']='complete'
        except BaseException as error:
            record.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
        finally:
            record['worker_seconds']=time.perf_counter()-started
            write(file,record);write(folder/('status_'+args.phase+'.json'),record)


def qualify(root):
    root=Path(root);p=check(root)
    rows=[done(root,c,'HISTORICAL_NATIVE',0) for c in p['cases']]
    rules.require(all(r is not None for r in rows),'Historical ten-case phase incomplete')
    null=rules.read(root/'null_diagnostic/run.json')
    rules.require(null['status']=='complete' and null['passed'],'Null numerical gate missing/failed')
    gate=rules.replay_gate(rows);write(root/'replay_gate.json',gate)
    print('A65 HISTORICAL REPLAY:',gate['classification'],flush=True)
    for item in gate['comparisons']:
        if not item['passed']:print('A65 REPLAY DIFFERENCE:',item,flush=True)
    return gate


def primary_inference(records):
    import numpy as np
    candidate={r['case']['image_id']:r for r in records if r['arm']=='RESPONSE_30' and r['repeat']==0}
    if len(candidate)!=10:return dict(classification='PRIMARY_COMPARISON_NOT_COMPLETE')
    comparisons={};generator=np.random.default_rng(20261028)
    indices=generator.integers(0,10,size=(100000,10))
    for arm in ('HISTORICAL_NATIVE','LEGACY_IEEE'):
        refs={r['case']['image_id']:r for r in records if r['arm']==arm and r['repeat']==0}
        if len(refs)!=10:return dict(classification='PRIMARY_COMPARISON_NOT_COMPLETE')
        rows=[dict(image_id=i,**{k:float(candidate[i]['metrics'][k])-float(refs[i]['metrics'][k])
                                 for k in ('psnr','ssim','lpips')}) for i in sorted(candidate)]
        delta=np.array([r['psnr'] for r in rows]);mean=float(delta.mean())
        bootstrap=delta[indices].mean(axis=1)
        lower=float(2*mean-np.quantile(bootstrap,1-.05/2))
        means={k:statistics.mean(r[k] for r in rows) for k in ('psnr','ssim','lpips')}
        positive=sum(r['psnr']>0 for r in rows)
        passed=bool(mean>0 and lower>0 and positive>=7 and means['ssim']>=-.002 and means['lpips']<=.005)
        comparisons[arm]=dict(mean_delta=means,psnr_positive_images=positive,
            psnr_basic_family_lower=lower,per_image=rows,passed=passed)
    active=sum(r['response_resolved_steps'] for r in candidate.values())
    passed=all(r['passed'] for r in comparisons.values()) and active>0
    return dict(classification='HISTORICAL_PROTOCOL_METHOD_GAIN_PILOT' if passed else
                'NO_CONFIRMED_PRIMARY_GAIN_ON_HISTORICAL_PROTOCOL',comparisons=comparisons,
                response_active_steps=active,family_size=2,resamples=100000,
                inference='One-sided Bonferroni alpha=.05/2 basic paired image bootstrap, ten reused image clusters; '
                          'approximate coverage, not a finite-sample or unseen-transfer guarantee')


def summarize(root):
    root=Path(root);p=check(root)
    attempts=[rules.read(f) for f in sorted((root/'cases').glob('*/*/repeat_*/run.json'))]
    records=[r for r in attempts if r['status']=='complete'];groups=[]
    for arm in rules.ARMS:
        rows=[r for r in records if r['arm']==arm]
        first=[r for r in rows if r['repeat']==0]
        group=dict(arm=arm,name=rules.NAMES[arm],complete_runs=len(rows),planned_runs=30,images=len(first),
                   forward_nfe=rules.NFE[arm],optimizer_backwards=rules.UPDATES[arm])
        if first:
            group['quality']={k:statistics.mean(float(r['metrics'][k]) for r in first)
                              for k in ('psnr','ssim','lpips','float_psnr')}
            group['mean_input_vjp']=statistics.mean(r['counts']['input_vjp_attempts'] for r in first)
            group['mean_fallback_steps']=statistics.mean(r['fallback_steps'] for r in first)
        group['times_by_image']={}
        for image in sorted({r['case']['image_id'] for r in rows}):
            values=[r['complete_reconstruction_seconds'] for r in sorted(rows,key=lambda r:r['repeat'])
                    if r['case']['image_id']==image]
            group['times_by_image'][image]=rules.summarize_times(values) if len(values)==3 else dict(repeats=values)
        if len(rows)==30:
            group['mean_of_image_median_seconds']=statistics.mean(v['median'] for v in group['times_by_image'].values())
        group['paired_vs']={}
        for ref in ('HISTORICAL_NATIVE','LEGACY_IEEE','EXACT_IEEE'):
            peers={r['case']['image_id']:r for r in records if r['arm']==ref and r['repeat']==0}
            paired=[dict(image_id=r['case']['image_id'],**{k:float(r['metrics'][k])-float(peers[r['case']['image_id']]['metrics'][k])
                    for k in ('psnr','ssim','lpips')}) for r in first if r['case']['image_id'] in peers]
            if paired:group['paired_vs'][ref]=dict(n=len(paired),per_image=paired,
                mean_delta={k:statistics.mean(r[k] for r in paired) for k in ('psnr','ssim','lpips')})
        groups.append(group)
    loads=[rules.read(f) for f in sorted((root/'workers').glob('*/load_*.json'))]
    null=rules.read(root/'null_diagnostic/run.json') if (root/'null_diagnostic/run.json').exists() else None
    summary=dict(experiment=rules.EXPERIMENT,classification='HISTORICAL_BASELINE_METHOD_GAIN_AUDIT',
        replay_gate=rules.read(root/'replay_gate.json') if (root/'replay_gate.json').exists() else None,
        published_reference=rules.PAPER,historical_ten_image_reference=rules.read(rules.AUDIT_PATH)['september17']['paper_15_10_5'],
        groups=groups,primary=primary_inference(records),completed_runs=len(records),planned_runs=150,
        costs=dict(method_forward_attempts=sum((r.get('counts') or {}).get('forward_attempts',0) for r in attempts),
            method_forwards_completed=sum((r.get('counts') or {}).get('forwards',0) for r in attempts),
            method_input_vjp_attempts=sum((r.get('counts') or {}).get('input_vjp_attempts',0) for r in attempts),
            optimizer_backward_attempts=sum((r.get('counts') or {}).get('optimizer_backward_attempts',0) for r in attempts),
            warmup_forward_attempts=sum(r.get('warmup',{}).get('forward_attempts',0) for r in loads),
            warmup_input_vjp_attempts=sum(r.get('warmup',{}).get('input_vjp_attempts',0) for r in loads),
            null_forward_attempts=(null.get('counts') or {}).get('forward_attempts',0) if null else 0,
            cost_is_lower_bound=any(r['status']!='complete' for r in attempts) or
                any(r.get('warmup',{}).get('status')!='complete' for r in loads) or
                bool(null and null['status']!='complete')),
        model_loads=loads,null_diagnostic=null,limits=p['protocol']['limitations'],
        timing=p['protocol']['wallclock'])
    write(root/'summary.json',summary);render_table(root,summary)
    print('A65 Summary:',json.dumps(finite(summary),ensure_ascii=False),flush=True)
    return summary


def render_table(root,summary):
    groups=summary['groups'];complete=all(g['complete_runs']==30 for g in groups)
    rank={}
    if complete:
        for metric in ('psnr','ssim','lpips'):
            order=sorted(groups,key=lambda r:r['quality'][metric],reverse=metric!='lpips')
            rank[metric]={r['arm']:i for i,r in enumerate(order)}
        rank['time']={r['arm']:i for i,r in enumerate(sorted(groups,key=lambda r:r['mean_of_image_median_seconds']))}
    lines=['# 原文、本地9月17日基线与新方法对照', '',
        '任务：FFHQ四倍超分辨率。当前实验使用原十图、seed1000—1009；固定15/10/5网格，lr=.01。',
        '原文为1000图均值，不能与本十图作配对因果差值；粗体/下划线只给本次全部完成的同图实测行排名。', '',
        '| 方法 | 图数 | PSNR↑ | SSIM↑ | LPIPS↓ | 完整重建秒/图↓ | 前向NFE | 输入反向 | 参数反向 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
        '| 原文报告 | 1000 | 26.6300 | .7680 | .1040 | 非同机，不排名 | 300 | 未核验 | 10 |',
        '| 9月17日历史记录 | 10 | 28.1905 | .8360 | .1265 | 旧优化段18.303，非本轮完整计时 | 300 | 0 | 10 |']
    csv_rows=[]
    for group in groups:
        arm=group['arm'];values=[]
        for metric in ('psnr','ssim','lpips'):
            v=f'{group["quality"][metric]:.4f}' if 'quality' in group else '未运行'
            i=rank.get(metric,{}).get(arm)
            values.append('**'+v+'**' if i==0 else '<u>'+v+'</u>' if i==1 else v)
        seconds=group.get('mean_of_image_median_seconds')
        value=f'{seconds:.3f}' if seconds is not None else '未完成三次'
        i=rank.get('time',{}).get(arm)
        if i==0:value='**'+value+'**'
        elif i==1:value='<u>'+value+'</u>'
        backward=f'{group["mean_input_vjp"]:.1f}均值' if 'mean_input_vjp' in group else '待测'
        lines.append('| '+group['name']+f' | {group["images"]} | '+' | '.join(values)+
                     f' | {value} | {group["forward_nfe"]} | {backward} | {group["optimizer_backwards"]} |')
        csv_rows.append(dict(method=group['name'],arm=arm,images=group['images'],complete_runs=group['complete_runs'],
            **group.get('quality',{}),complete_seconds=seconds,forward_nfe=group['forward_nfe'],
            mean_input_vjp=group.get('mean_input_vjp'),optimizer_backwards=group['optimizer_backwards']))
    lines += ['', '完整重建秒：三次独立完整运行的逐图中位数再跨图平均。计时包含观测读取、算子/参数构造、',
        '所有学习/CG/输入反向和CPU输出同步；模型加载/热身另列，GT指标/保存计时外。',
        '原文A100总体wall-clock为14.71秒，并非本机3090同任务逐图时间，不用于构造加速比。', '',
        '主要候选为独立30步，不在结果出来后换成另一个更好版本；完整300版本是同前向预算对照。',
        '历史经验上采样与真实伴随明确分开；若仅改伴随已经改善，不能把全部改善归给实时响应。',
        '当前响应的单位适配不是新调参：旧观测在模型坐标，因此使用H而不是.5H。',
        '判定：'+summary['primary']['classification'],
        '限度：'+summary['limits'], '']
    (root/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    with (root/'comparison.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(dict.fromkeys(k for row in csv_rows for k in row)))
        writer.writeheader();writer.writerows(csv_rows)


def archive(root):
    from utils import archive_refinement_confirmation as exporter
    exporter.BRANCH=rules.BRANCH
    bundle,commit=exporter.export_and_commit({'A65':Path(root)},projects=PROJECTS)
    print('Archive:',bundle,'Results commit:',commit,'Branch:',rules.BRANCH,flush=True)
    subprocess.run(['git','push','origin',rules.BRANCH],cwd=PROJECTS.parent,check=True)
    print('RESULTS_PUSH_PASS: A65 historical strong baseline, method quality and full timing uploaded.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=rules.ROOT)
    parser.add_argument('--ffhq-dir',default='/home/lzy/FFHQ/00000/00000')
    parser.add_argument('--shard',type=int,choices=(0,1));parser.add_argument('--phase',choices=('baseline','comparison'))
    parser.add_argument('--device-uuid')
    group=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','worker','qualify','summarize','archive'):group.add_argument('--'+name,action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare(args)
    elif args.worker:worker(args)
    elif args.qualify:qualify(args.root)
    elif args.summarize:summarize(args.root)
    elif args.archive:archive(args.root)


if __name__=='__main__':main()
