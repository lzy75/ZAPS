"""A67: strong ZAPS + A33/noise-feasible scalar controls. No old sources changed."""
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

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import noise_feasible_multiplier as policy
from utils import noise_feasible_runtime as runtime
from utils import historical_ffhq_gain as strong
from utils import historical_ffhq_runtime as old_runtime
from utils import paper_clock_runtime as clock
from utils.diag_paper_clock_alignment import write,resource

ROOT='/data0/lzy/ZAPS-results/diag_noise_feasible_zaps/A67_20261028'
BRANCH='results/strong-zaps-noise-feasible-20261028'
CARD=PROJECTS/'experiment_cards/A67_strong_zaps_noise_feasible_20261028.md'


def check(root):
    p=strong.read(root/'plan.json')
    strong.require(p['experiment']=='A67_strong_zaps_noise_feasible','Wrong plan')
    for relative,sha in p['sources'].items():
        strong.require(strong.digest(PROJECTS/relative)==sha,'Bound source changed: '+relative)
    return p


def prepare(args):
    root=Path(args.root).resolve()
    if (root/'plan.json').exists():check(root);print('A67 frozen plan reused.',flush=True);return
    strong.require(not root.exists() or not any(root.iterdir()),'Nonempty unplanned output; preserve it')
    cases=policy.cases(args.ffhq_dir)
    for c in cases:c['image_sha256']=strong.digest(c['image'])
    checkpoint=PROJECTS/'modules/models/ffhq_10m.pt'
    sha='81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa'
    strong.require(strong.digest(checkpoint)==sha,'FFHQ checkpoint differs')
    relative=['utils/noise_feasible_multiplier.py','utils/noise_feasible_runtime.py',
              'utils/diag_noise_feasible_zaps.py','utils/a67_session.py',
              'utils/test_noise_feasible_multiplier.py','utils/test_noise_feasible_runtime.py',
              'modules/zaps_algorithm.py','modules/wavelet.py','modules/degradations.py',
              'modules/main_single.py','utils/metrics.py','utils/historical_ffhq_gain.py',
              'utils/historical_ffhq_runtime.py','utils/paper_clock_runtime.py',
              'utils/continuous_response_feedback.py','utils/diag_ffhq_regression.py',
              'run_a67_20261028.sh',str(CARD.relative_to(PROJECTS))]
    root.parent.mkdir(parents=True,exist_ok=True)
    strong.require(shutil.disk_usage(root.parent).free>=3*strong.GIB,'Need3GiB result space')
    root.mkdir(exist_ok=True)
    p=dict(experiment='A67_strong_zaps_noise_feasible',cases=cases,config=strong.config(),grid=strong.grid(),
           checkpoint=dict(path=str(checkpoint),sha256=sha),arms=list(policy.ARMS),
           sources={r:strong.digest(PROJECTS/r) for r in relative},
           code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECTS.parent,text=True).strip(),
           budget=dict(method_forwards=18000,diagnostic_replay_forwards=1800,input_vjp=0,
                       optimizer_backwards=600,warmup_forwards_per_model_load=1),
           protocol=dict(domain='Historical model-coordinate H(x), no .5 factor; sigma in SAME units',
             baseline='9/17 strong local configuration, native precision, empirical bicubic guidance mapping',
             output='last_opt, tenth unroll only controlled; full300 forwards and10 optimizer backwards',
             primary_quality='Same saved BASE pre-tenth-unroll learned parameters and RNG; paid30 suffix replay per arm',
             deployment_clock='Separately measured independent full300 execution, no learned cache input; output also evaluated',
             time_profile='Same-sigma leave-one-IMAGE-out median of paired NEAREST_NOISE traces; no GT',
             limitations='Six OLD FFHQ SR development images, noise boundaries first3 only, one seed/image; '
                        'no independent confirmation, new theory, curve/step-size or NFE-saving claim. '
                        'Noise-energy target is a discrepancy heuristic, not calibrated confidence coverage. '
                        'State derivative uses A33 identity-response surrogate, not actual next-step network response.',
             measurements='Known synthetic noise; fixed seed reuses standardized noise across sigma; GT never in decisions'))
    write(root/'plan.json',p);shutil.copyfile(CARD,root/'experiment_card.md')
    print('A67 PREPARE:12 image/noise cases,5 arms;18000 full-pipeline forwards +1800 paired replay;0 input VJP.',flush=True)


def observation(root,c,device):
    import torch
    from modules.main_single import load_image_as_tensor
    from utils.diag_learning_rate_ablation import set_seed
    path=root/'observations'/(c['key']+'.pt');meta=path.with_suffix('.json')
    if meta.exists():
        r=strong.read(meta);strong.require(strong.digest(path)==r['sha256'],'Observation changed');return r
    path.parent.mkdir(exist_ok=True)
    with old_runtime.precision('HISTORICAL_NATIVE'),torch.no_grad():
        op=old_runtime.make_operator(device);gt=load_image_as_tensor(c['image']).to(device)
        set_seed(c['seed']);y=op.H(gt)+c['sigma']*torch.randn_like(op.H(gt))
        torch.save(dict(y=y.cpu()),path)
    r=dict(path=str(path),sha256=strong.digest(path),sigma=c['sigma'],domain='model-coordinate')
    write(meta,r);return r


def record_path(root,c,arm):return root/'cases'/c['key']/arm/'run.json'


def completed(root,c,arm):
    path=record_path(root,c,arm)
    if not path.exists():return None
    r=strong.read(path)
    strong.require(r['status']=='complete','Prior attempt retained; no automatic retry: '+str(path))
    strong.require(r['plan_sha256']==strong.digest(root/'plan.json'),'Plan identity differs')
    for name,sha in r['artifacts'].items():strong.require(strong.digest(path.parent/name)==sha,'Artifact changed')
    return r


def time_profile(root,p,c):
    file=root/'time_profiles'/(c['key']+'.json')
    peers=[other for other in p['cases'] if other['sigma']==c['sigma'] and other['image_id']!=c['image_id']]
    curves=[];sources={}
    for other in peers:
        r=completed(root,other,'NEAREST_NOISE');strong.require(r is not None,'Finish all NEAREST donors first')
        path=record_path(root,other,'NEAREST_NOISE').parent/'paired_trace.json'
        rows=strong.read(path);strong.require(len(rows)==30,'Incomplete donor trace')
        curves.append([row['gamma'] for row in rows]);sources[str(path.relative_to(root))]=strong.digest(path)
    strong.require(len(curves)>=2,'Need two other-image donors')
    data=dict(gamma=[statistics.median(v[k] for v in curves) for k in range(30)],
              sources=sources,excluded_image=c['image_id'],sigma=c['sigma'],GT_used=False)
    if file.exists():strong.require(strong.read(file)==data,'Frozen time profile changed')
    else:file.parent.mkdir(exist_ok=True);write(file,data)
    return data['gamma']


def metrics(out,gt):
    from utils.metrics import compute_all_metrics
    result=compute_all_metrics(out,gt,lpips_net='alex')
    strong.require(all(math.isfinite(float(v)) for v in result.values()),'Nonfinite metrics')
    return result


def measure(root,p,c,arm,model,uuid):
    import torch
    from modules.main_single import load_image_as_tensor
    from modules.dataset_loader import tensor_to_image
    if completed(root,c,arm):return
    admission=resource(uuid,'ffhq',torch.cuda.memory_reserved())
    if not admission['admitted']:raise MemoryError('Resource margin lost before attempt; no other process stopped')
    obs=observation(root,c,model.device)
    profile=time_profile(root,p,c) if arm=='TIME_NEAREST' else None
    folder=record_path(root,c,arm).parent;folder.mkdir(parents=True,exist_ok=False)
    r=dict(status='running',case=c,arm=arm,plan_sha256=strong.digest(root/'plan.json'),
           observation=obs,device_uuid=uuid,full_counts=None,replay_counts=None,artifacts={})
    write(folder/'run.json',r)
    def tick(count):
        r['full_counts']=count;write(folder/'run.json',r)
        print('A67 PROGRESS:',c['key'],arm,'F=',count['forwards'],'/300',flush=True)
    try:
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        with old_runtime.precision('HISTORICAL_NATIVE') as precision,clock.counters(model.model,tick) as count:
            def operation():
                y=torch.load(obs['path'],map_location='cpu',weights_only=True)['y'].to(model.device)
                op=old_runtime.make_operator(model.device,count)
                return runtime.execute(model,op,y,c['seed'],arm,c['sigma'],profile)
            (independent,detail),seconds=clock.clocks(torch.cuda.synchronize,operation)
        r['full_counts']=count
        strong.require(count['forwards']==count['forward_attempts']==300 and
                       count['optimizer_backward_completed']==count['optimizer_backward_attempts']==10 and
                       count['input_vjp_attempts']==0 and count['calls']==strong.grid()[::-1]*10,'Full budget/grid mismatch')
        strong.require(all(param.grad is None and not param.requires_grad for param in model.model.parameters()),'Model weights not frozen')
        peak=torch.cuda.max_memory_reserved()
        if arm=='BASE':
            torch.save(detail['entry'],folder/'entry.pt');entry=detail['entry']
        else:
            base=completed(root,c,'BASE');strong.require(base is not None,'Baseline must finish first')
            entry=torch.load(record_path(root,c,'BASE').parent/'entry.pt',map_location='cpu',weights_only=True)
        r['independent_prefix_comparison']=clock.snapshot_equal(entry,detail['entry'])
        y=torch.load(obs['path'],map_location='cpu',weights_only=True)['y'].to(model.device)
        # Replay removes prefix variability from the primary quality contrast.
        with old_runtime.precision('HISTORICAL_NATIVE'),clock.counters(model.model) as replay_count:
            paired,trace=runtime.replay(model,old_runtime.make_operator(model.device,replay_count),y,entry,
                                        'NULL' if arm=='BASE' else arm,c['sigma'],profile)
        r['replay_counts']=replay_count
        strong.require(replay_count['forwards']==replay_count['forward_attempts']==30 and
                       replay_count['optimizer_backward_attempts']==0 and replay_count['input_vjp_attempts']==0,'Replay budget mismatch')
        if arm=='BASE':
            error=float((paired.double()-independent.double()).norm()/independent.double().norm().clamp(min=1e-30))
            r['null_loop_relative_error']=error;strong.require(error<=1e-6,'Original/null final loop differs')
        # No metrics or GT were supplied to either controller or time-profile builder.
        gt=load_image_as_tensor(c['image']).cpu()
        r.update(independent_metrics=metrics(independent,gt),paired_metrics=metrics(paired,gt),
                 complete_reconstruction_seconds=seconds,peak_reserved_bytes=peak,precision=precision,
                 measured_prefix_seconds=detail['prefix_seconds'],
                 timing_scope='Warm-model instrumented independent full300, observation read/operator/init/optimization/output CPU sync; metrics/files excluded')
        torch.save(independent,folder/'independent.pt');torch.save(paired,folder/'paired.pt')
        tensor_to_image(paired.squeeze(0),denormalize=True).save(folder/'paired.png')
        write(folder/'independent_trace.json',detail['trace']);write(folder/'paired_trace.json',trace)
        names=['independent.pt','paired.pt','paired.png','independent_trace.json','paired_trace.json']+(['entry.pt'] if arm=='BASE' else [])
        r.update(status='complete',artifacts={n:strong.digest(folder/n) for n in names})
        write(folder/'run.json',r)
        print('A67 COMPLETE:',c['key'],arm,'paired_PSNR=',round(r['paired_metrics']['psnr'],4),
              'full_clock=',round(seconds,2),flush=True)
    except BaseException as error:
        if 'count' in locals():r['full_counts']=count
        if 'replay_count' in locals():r['replay_counts']=replay_count
        r.update(status='failed',error=repr(error),traceback=traceback.format_exc(),cost_is_lower_bound=True)
        write(folder/'run.json',r);raise


def worker(args):
    import torch
    from utils.a54_session import exclusive
    from modules.main_single import load_diffusion_model
    root=Path(args.root).resolve();p=check(root);folder=root/'workers'/str(args.shard);folder.mkdir(parents=True,exist_ok=True)
    with exclusive(folder/'worker.lock'):
        phase_status=folder/(args.phase+'.json')
        admission=resource(args.device_uuid,'ffhq')
        if not admission['admitted']:write(phase_status,dict(status='resource_limited',admission=admission));return
        strong.require(strong.digest(p['checkpoint']['path'])==p['checkpoint']['sha256'],'Checkpoint changed')
        for c in p['cases']:strong.require(strong.digest(c['image'])==c['image_sha256'],'Image changed')
        start=time.perf_counter();model=load_diffusion_model('ffhq','cuda:0');model.model.eval();torch.cuda.synchronize()
        load=dict(status='running',load_seconds=time.perf_counter()-start,warmup_forward_attempts=0,warmup_forwards=0,
                  uuid=args.device_uuid,torch=str(torch.__version__))
        file=folder/f'load_{len(list(folder.glob("load_*.json"))):03d}.json';write(file,load)
        try:
            load['warmup_forward_attempts']=1;write(file,load)
            with torch.no_grad(),old_runtime.precision('HISTORICAL_NATIVE'):
                model._predict_eps(torch.zeros(1,3,256,256,device=model.device),torch.tensor([999],device=model.device))
            torch.cuda.synchronize();load['warmup_forwards']=1;write(file,load)
            cases=[c for i,c in enumerate(p['cases']) if i%2==args.shard]
            arms=policy.ARMS[:-1] if args.phase=='feedback' else ('TIME_NEAREST',)
            for c in cases:
                for arm in arms:measure(root,p,c,arm,model,args.device_uuid)
            load['status']='complete';write(phase_status,dict(status='complete'))
        except BaseException as error:
            load.update(status='failed',error=repr(error));write(phase_status,load);raise
        finally:write(file,load)


def summarize(root):
    root=Path(root);p=check(root);records=[strong.read(f) for f in (root/'cases').glob('*/*/run.json')]
    complete=[r for r in records if r['status']=='complete'];groups=[]
    for sigma in (.025,.05,.1):
        baseline={r['case']['key']:r for r in complete if r['arm']=='BASE' and r['case']['sigma']==sigma}
        for arm in policy.ARMS:
            rows=[r for r in complete if r['arm']==arm and r['case']['sigma']==sigma]
            if not rows:continue
            group=dict(sigma=sigma,arm=arm,cases=len(rows),
                       paired_quality={k:statistics.mean(r['paired_metrics'][k] for r in rows) for k in ('psnr','ssim','lpips')},
                       independent_quality={k:statistics.mean(r['independent_metrics'][k] for r in rows) for k in ('psnr','ssim','lpips')},
                       mean_full_clock_seconds=statistics.mean(r['complete_reconstruction_seconds'] for r in rows))
            paired=[r for r in rows if r['case']['key'] in baseline]
            group['paired_delta_vs_BASE']={k:statistics.mean(r['paired_metrics'][k]-baseline[r['case']['key']]['paired_metrics'][k] for r in paired) for k in ('psnr','ssim','lpips')}
            group['images_PSNR_positive']=sum(r['paired_metrics']['psnr']>baseline[r['case']['key']]['paired_metrics']['psnr'] for r in paired)
            groups.append(group)
    loads=[strong.read(f) for f in (root/'workers').glob('*/load_*.json')]
    comparisons=[]
    for sigma in (.025,.05,.1):
        for reference in ('BASE','A33','FIRST_NOISE','TIME_NEAREST'):
            refs={r['case']['key']:r for r in complete if r['arm']==reference and r['case']['sigma']==sigma}
            rows=[r for r in complete if r['arm']=='NEAREST_NOISE' and r['case']['sigma']==sigma and r['case']['key'] in refs]
            if rows:
                comparisons.append(dict(sigma=sigma,comparison='NEAREST_NOISE_minus_'+reference,cases=len(rows),
                    mean_delta={k:statistics.mean(r['paired_metrics'][k]-refs[r['case']['key']]['paired_metrics'][k] for r in rows) for k in ('psnr','ssim','lpips')}))
    result=dict(experiment=p['experiment'],classification='DEVELOPMENT_STRONG_ZAPS_SCALAR_COMPARISON',
                completed_runs=len(complete),planned_runs=60,groups=groups,
                paired_comparisons=comparisons,
                actual_cost=dict(method_forward_attempts=sum((r.get('full_counts') or {}).get('forward_attempts',0) for r in records),
                    diagnostic_forward_attempts=sum((r.get('replay_counts') or {}).get('forward_attempts',0) for r in records),
                    optimizer_backward_attempts=sum((r.get('full_counts') or {}).get('optimizer_backward_attempts',0) for r in records),
                    input_vjp=0,warmup_forward_attempts=sum(r['warmup_forward_attempts'] for r in loads)),
                cost_is_lower_bound=any(r['status']!='complete' for r in records),limits=p['protocol']['limitations'])
    write(root/'summary.json',result)
    lines=['# A67 strong ZAPS: paired quality and independent full clock','',
           '|sigma|arm|n|PSNR|SSIM|LPIPS|delta PSNR|full seconds|','|---|---|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        m=g['paired_quality'];lines.append(f"|{g['sigma']}|{g['arm']}|{g['cases']}|{m['psnr']:.4f}|{m['ssim']:.5f}|{m['lpips']:.5f}|{g['paired_delta_vs_BASE']['psnr']:+.4f}|{g['mean_full_clock_seconds']:.3f}|")
    lines+=['','Quality uses the identical learned270-prefix replay. Clock is a separately measured complete independent300 execution; independent output quality is also in summary.json. One timing per case, not three-repeat median.','',result['limits']]
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('A67 Summary:',json.dumps(result,ensure_ascii=False),flush=True);return result


def archive(root):
    from utils import archive_refinement_confirmation as exporter
    exporter.BRANCH=BRANCH
    bundle,commit=exporter.export_and_commit({'A67':Path(root)},projects=PROJECTS)
    print('Archive:',bundle,'Results commit:',commit,'Branch:',BRANCH,flush=True)
    subprocess.run(['git','push','origin',BRANCH],cwd=PROJECTS.parent,check=True)
    print('RESULTS_PUSH_PASS: A67 strong ZAPS scalar comparison uploaded.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=ROOT)
    p.add_argument('--ffhq-dir',default='/home/lzy/FFHQ/00000/00000')
    p.add_argument('--shard',type=int,choices=(0,1));p.add_argument('--phase',choices=('feedback','time'))
    p.add_argument('--device-uuid');mode=p.add_mutually_exclusive_group(required=True)
    for name in ('prepare','worker','summarize','archive'):mode.add_argument('--'+name,action='store_true')
    a=p.parse_args()
    if a.prepare:prepare(a)
    elif a.worker:worker(a)
    elif a.summarize:summarize(a.root)
    else:archive(a.root)


if __name__=='__main__':main()
