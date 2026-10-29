"""A69 strong ZAPS, next-denoiser response and directional Gaussian risk."""
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
from utils import scalar_gamma_newton as rule
from utils import scalar_gamma_newton_runtime as runtime
from utils import historical_ffhq_gain as files
from utils import historical_ffhq_runtime as old_runtime
from utils import paper_clock_runtime as clock
from utils import diag_noise_feasible_zaps as common
from utils.a67_model_freeze import frozen_network
from utils.diag_paper_clock_alignment import write,resource

ROOT='/data0/lzy/ZAPS-results/diag_scalar_gamma_solver/A69_20261029'
BRANCH='results/scalar-gamma-solver-20261029'
CARD=PROJECTS/'experiment_cards/A69_scalar_gamma_solver_20261029.md'


def check(root):
    p=files.read(root/'plan.json');files.require(p['experiment']=='A69_scalar_gamma_solver','Wrong plan')
    for name,sha in p['sources'].items():files.require(files.digest(PROJECTS/name)==sha,'Source changed: '+name)
    return p


def prepare(args):
    root=Path(args.root).resolve()
    if (root/'plan.json').exists():check(root);print('A69 frozen plan reused.',flush=True);return
    files.require(not root.exists() or not any(root.iterdir()),'Nonempty unplanned output')
    cases=rule.cases(args.ffhq_dir)
    for c in cases:c['image_sha256']=files.digest(c['image'])
    checkpoint=PROJECTS/'modules/models/ffhq_10m.pt';sha='81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa'
    files.require(files.digest(checkpoint)==sha,'Checkpoint differs')
    sources=['utils/scalar_gamma_newton.py','utils/scalar_gamma_newton_runtime.py','utils/next_response_bayes.py','utils/next_response_bayes_runtime.py','utils/diag_scalar_gamma_solver.py',
             'utils/a69_session.py','utils/test_scalar_gamma_newton.py','utils/test_scalar_gamma_newton_runtime.py',
             'utils/test_a69_cli.py','utils/noise_feasible_runtime.py','utils/noise_feasible_multiplier.py',
             'utils/current_score_runtime_r2.py','utils/current_score_runtime.py',
             'utils/test_current_score_checkpoint_r2.py',
             'utils/a67_model_freeze.py','utils/diag_noise_feasible_zaps.py','utils/historical_ffhq_gain.py',
             'utils/historical_ffhq_runtime.py','utils/paper_clock_runtime.py','modules/zaps_algorithm.py',
             'modules/degradations.py','modules/wavelet.py','utils/metrics.py',
             'run_a69_20261029.sh',str(CARD.relative_to(PROJECTS))]
    root.parent.mkdir(parents=True,exist_ok=True)
    files.require(shutil.disk_usage(root.parent).free>=3*files.GIB,'Need3GiB result space')
    root.mkdir(exist_ok=True)
    p=dict(experiment='A69_scalar_gamma_solver',cases=cases,config=files.config(),grid=files.grid(),arms=list(rule.ARMS),
        checkpoint=dict(path=str(checkpoint),sha256=sha),sources={n:files.digest(PROJECTS/n) for n in sources},
        budget=dict(max_method=15016,max_paired_diagnostic=4216,optimizer_backwards=400,max_input_vjp=928,warmups_separate=True),
        protocol=dict(precision='IEEE float32 ALL groups; no native-vs-IEEE causal comparison',
          measurement='Historical model-coordinate H(x); sigma same units; empirical original guidance mapping retained',
          quality='Same BASE pre-final270 learned state/noise paired replay; independent outputs also reported',
          clock='Independent full pipeline incl optimization/probes/output CPU sync; metrics/files/model-load excluded',
          limitations='Four OLD FFHQ SR images, two noise levels, one seed/image; scalar local Gaussian approximation, '
            'same A68 REAL_ISO objective with beta1; no global-minimum or speedup claim. Input backwards and NN calls fully paid. '
            'Continuous remapping can cancel current positive zeta amplitude; learned path and D direction retained.'),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECTS.parent,text=True).strip())
    write(root/'plan.json',p);shutil.copyfile(CARD,root/'experiment_card.md')
    print('A69 PREPARE:8 cases x5 groups; <=15016 full +4216 paired forwards;<=928 input VJP; beta1, no gamma2 cap.',flush=True)


def validate(count,arm,full):
    cap=(rule.MAX_FULL if full else rule.MAX_REPLAY)[arm]
    files.require(count['forward_attempts']==count['forwards']<=cap,'Network budget differs')
    files.require(count['optimizer_backward_attempts']==count['optimizer_backward_completed']==(10 if full else 0),
                  'Optimization backward count differs')
    files.require(0<=count['input_vjp_attempts']<=(58 if arm=='AUTO_NEWTON' else 0),'Input VJP count differs')
    if full:files.require(count['calls'][:270]==files.grid()[::-1]*9,'Original prefix changed')
    if arm in ('BASE','FIRST_NOISE'):
        files.require(count['forwards']==(300 if full else 30),'Original scalar-arm calls differ')


def measure(root,p,c,arm,model,uuid):
    import torch
    from modules.main_single import load_image_as_tensor
    from modules.dataset_loader import tensor_to_image
    if common.completed(root,c,arm):return
    files.require(resource(uuid,'ffhq',torch.cuda.memory_reserved())['admitted'],'Resource margin lost before attempt')
    obs=common.observation(root,c,model.device)
    folder=common.record_path(root,c,arm).parent;folder.mkdir(parents=True,exist_ok=False)
    r=dict(status='running',case=c,arm=arm,plan_sha256=files.digest(root/'plan.json'),artifacts={},
           device_uuid=uuid,full_counts=None,replay_counts=None)
    write(folder/'run.json',r)
    def tick(value):
        r['full_counts']=value;write(folder/'run.json',r)
        print('A69 PROGRESS:',c['key'],arm,'F=',value['forwards'],'cap=',rule.MAX_FULL[arm],flush=True)
    try:
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        with old_runtime.precision('LEGACY_IEEE'),clock.counters(model.model,tick) as counts:
            def operation():
                y=torch.load(obs['path'],map_location='cpu',weights_only=True)['y'].to(model.device)
                return runtime.execute(model,old_runtime.make_operator(model.device,counts),y,c['seed'],arm,c['sigma'],counts)
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
            paired,trace=runtime.replay(model,old_runtime.make_operator(model.device,replay_counts),y,entry,
                                        'NULL' if arm=='BASE' else arm,c['sigma'],replay_counts)
        r['replay_counts']=replay_counts;validate(replay_counts,arm,False)
        if arm=='BASE':
            error=float((paired.double()-independent.double()).norm()/independent.double().norm().clamp(min=1e-30))
            r['null_relative_error']=error;files.require(error<=1e-6,'Core/null mismatch')
        gt=load_image_as_tensor(c['image']).cpu()
        r.update(paired_metrics=common.metrics(paired,gt),independent_metrics=common.metrics(independent,gt),
                 complete_reconstruction_seconds=seconds,peak_reserved_bytes=peak,
                 full_precision='IEEE',cost_is_lower_bound=False,
                 variance_state_steps=sum(v.get('uncertainty_source')=='DIRECTIONAL_STATE' for v in trace),
                 resolved_steps=sum(bool(v.get('resolved')) for v in trace),
                 gamma_more_than_two=sum(v.get('gamma',1)>2 for v in trace),
                 solver_stats=dict(stationarity_met_steps=sum(bool(v.get('stationarity_met')) for v in trace),
                     exact_gradient=arm=='AUTO_NEWTON',
                     final_gradient_unmeasured_steps=sum(not v.get('final_gradient_measured',True) for v in trace),
                     budget_stops=sum('BUDGET' in v.get('stop_reason','') for v in trace),
                     value_points=sum(v.get('value_points',0) for v in trace),
                     gradient_points=sum(v.get('gradient_points',0) for v in trace)))
        torch.save(paired,folder/'paired.pt');torch.save(independent,folder/'independent.pt')
        tensor_to_image(paired.squeeze(0),denormalize=True).save(folder/'paired.png')
        write(folder/'paired_trace.json',trace);write(folder/'independent_trace.json',detail['trace'])
        names=['paired.pt','independent.pt','paired.png','paired_trace.json','independent_trace.json']+(['entry.pt'] if arm=='BASE' else [])
        r.update(status='complete',artifacts={n:files.digest(folder/n) for n in names});write(folder/'run.json',r)
        print('A69 COMPLETE:',c['key'],arm,'paired PSNR=',round(r['paired_metrics']['psnr'],4),
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
        if not resource(args.device_uuid,'ffhq')['admitted']:
            write(folder/'status.json',dict(status='resource_limited'));return
        files.require(files.digest(p['checkpoint']['path'])==p['checkpoint']['sha256'],'Checkpoint changed')
        for c in p['cases']:files.require(files.digest(c['image'])==c['image_sha256'],'Image changed')
        start=time.perf_counter();model=load_diffusion_model('ffhq','cuda:0');model.model.eval();torch.cuda.synchronize()
        load=dict(status='running',load_seconds=time.perf_counter()-start,uuid=args.device_uuid,
                  warmup_forward_attempts=0,warmup_forwards=0,warmup_input_vjp_attempts=0,torch=str(torch.__version__))
        f=folder/f'load_{len(list(folder.glob("load_*.json"))):03d}.json';write(f,load)
        try:
            with frozen_network(model.model):
                load['warmup_forward_attempts']=1;write(f,load)
                from utils.current_score_runtime_r2 import input_vjp_parameters
                with torch.enable_grad(),old_runtime.precision('LEGACY_IEEE'),input_vjp_parameters(model.model):
                    gamma=torch.zeros((),device=model.device,requires_grad=True)
                    x=gamma*torch.ones(1,3,256,256,device=model.device)
                    eps=model._predict_eps(x,torch.tensor([999],device=model.device))
                    alpha=model.alphas_cumprod[999]
                    raw=(x-(1-alpha).sqrt()*eps)/alpha.sqrt()
                    load['warmup_forwards']=1;load['warmup_input_vjp_attempts']=1;write(f,load)
                    g=torch.autograd.grad(raw.double().square().mean(),gamma)[0]
                    files.require(bool(torch.isfinite(g)),'Real FFHQ scalar AD warmup failed')
                torch.cuda.synchronize();write(f,load)
                del gamma,x,eps,raw,g
                gc.collect();torch.cuda.empty_cache()
                for index,c in enumerate(p['cases']):
                    if int(c['image_id'])%2==args.shard:
                        for arm in rule.ARMS:measure(root,p,c,arm,model,args.device_uuid)
            load['status']='complete';write(folder/'status.json',dict(status='complete'))
        except BaseException as error:
            load.update(status='failed',error=repr(error));write(folder/'status.json',load);raise
        finally:write(f,load)


def summarize(root):
    root=Path(root);p=check(root);rows=[files.read(f) for f in (root/'cases').glob('*/*/run.json')]
    done=[r for r in rows if r['status']=='complete'];groups=[];contrasts=[]
    for sigma in (.05,.1):
        for arm in rule.ARMS:
            data=[r for r in done if r['case']['sigma']==sigma and r['arm']==arm]
            if data:
                groups.append(dict(sigma=sigma,arm=arm,cases=len(data),
                    quality={k:statistics.mean(r['paired_metrics'][k] for r in data) for k in ('psnr','ssim','lpips')},
                    independent_quality={k:statistics.mean(r['independent_metrics'][k] for r in data) for k in ('psnr','ssim','lpips')},
                    mean_full_seconds=statistics.mean(r['complete_reconstruction_seconds'] for r in data),
                    mean_full_NN=statistics.mean(r['full_counts']['forwards'] for r in data),
                    state_usable_steps=sum(r['variance_state_steps'] for r in data),
                    gamma_more_than_two=sum(r['gamma_more_than_two'] for r in data),
                    mean_full_input_backwards=statistics.mean(r['full_counts']['input_vjp_attempts'] for r in data),
                    solver_stats={k:sum(r['solver_stats'][k] for r in data) for k in
                        ('stationarity_met_steps','final_gradient_unmeasured_steps','budget_stops','value_points','gradient_points')}))
                for reference in ('BASE','FIRST_NOISE','REAL_ISO'):
                    peers={r['case']['key']:r for r in done if r['arm']==reference}
                    pairs=[r for r in data if r['case']['key'] in peers]
                    if pairs:
                        contrasts.append(dict(sigma=sigma,comparison=arm+' minus '+reference,cases=len(pairs),
                            delta={k:statistics.mean(r['paired_metrics'][k]-peers[r['case']['key']]['paired_metrics'][k] for r in pairs) for k in ('psnr','ssim','lpips')},
                            images_PSNR_positive=sum(r['paired_metrics']['psnr']>peers[r['case']['key']]['paired_metrics']['psnr'] for r in pairs)))
    loads=[files.read(f) for f in (root/'workers').glob('*/load_*.json')]
    result=dict(experiment=p['experiment'],classification='DEVELOPMENT_SCALAR_SOLVER_COMPARISON',
        completed_runs=len(done),planned_runs=40,groups=groups,paired_comparisons=contrasts,
        actual_cost=dict(method_forward_attempts=sum((r.get('full_counts') or {}).get('forward_attempts',0) for r in rows),
                        diagnostic_forward_attempts=sum((r.get('replay_counts') or {}).get('forward_attempts',0) for r in rows),
                        optimizer_backward_attempts=sum((r.get('full_counts') or {}).get('optimizer_backward_attempts',0) for r in rows),
                        method_input_vjp_attempts=sum((r.get('full_counts') or {}).get('input_vjp_attempts',0) for r in rows),
                        diagnostic_input_vjp_attempts=sum((r.get('replay_counts') or {}).get('input_vjp_attempts',0) for r in rows),
                        warmup_input_vjp_attempts=sum(r.get('warmup_input_vjp_attempts',0) for r in loads),
                        warmup_forward_attempts=sum(r['warmup_forward_attempts'] for r in loads)),
        cost_is_lower_bound=any(r['status']!='complete' for r in rows),limits=p['protocol']['limitations'])
    write(root/'summary.json',result)
    lines=['# A69 next-response scalar control','', '|sigma|method|n|PSNR|SSIM|LPIPS|full seconds|full NN|',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        m=g['quality'];lines.append(f"|{g['sigma']}|{g['arm']}|{g['cases']}|{m['psnr']:.4f}|{m['ssim']:.5f}|{m['lpips']:.5f}|{g['mean_full_seconds']:.3f}|{g['mean_full_NN']:.1f}|")
    lines+=['','Quality paired on the same270 prefix. Clock measured independently from initialization through all learning/probes/output CPU sync. Additional queries are paid.','',result['limits']]
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('A69 Summary:',json.dumps(result,ensure_ascii=False),flush=True)


def archive(root):
    from utils import archive_refinement_confirmation as exporter
    exporter.BRANCH=BRANCH
    bundle,commit=exporter.export_and_commit({'A69':Path(root)},projects=PROJECTS)
    print('Archive:',bundle,'Results commit:',commit,'Branch:',BRANCH,flush=True)
    subprocess.run(['git','push','origin',BRANCH],cwd=PROJECTS.parent,check=True)
    print('RESULTS_PUSH_PASS: A69 uploaded.',flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=ROOT)
    p.add_argument('--ffhq-dir',default='/home/lzy/FFHQ/00000/00000')
    p.add_argument('--shard',type=int,choices=(0,1));p.add_argument('--device-uuid')
    p.add_argument('--check-worker-cli',action='store_true',help=argparse.SUPPRESS)
    group=p.add_mutually_exclusive_group(required=True)
    for name in ('prepare','worker','summarize','archive'):group.add_argument('--'+name,action='store_true')
    return p


def main():
    p=parser();a=p.parse_args()
    if a.check_worker_cli:
        if not (a.worker and a.shard in (0,1) and a.device_uuid):p.error('Full worker arguments required')
        print(json.dumps(dict(shard=a.shard,device_uuid=a.device_uuid,model_calls=0)));return
    if a.prepare:prepare(a)
    elif a.worker:
        if a.shard is None or not a.device_uuid:p.error('worker requires --shard and --device-uuid')
        worker(a)
    elif a.summarize:summarize(a.root)
    else:archive(a.root)


if __name__=='__main__':main()
