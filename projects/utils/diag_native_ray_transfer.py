"""A82/A83/A84 paired official-host scalar transfer; no learned states or tuning."""
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
from contextlib import nullcontext

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import native_ray_transfer as rules
from utils import native_ray_transfer_runtime as runtime
from utils import historical_ffhq_gain as files
from utils import diag_shrink_multitask as parent
from utils import diag_noise_feasible_zaps as common
from utils import historical_ffhq_runtime as precision
from utils import paper_clock_runtime as clock
from utils.a67_model_freeze import frozen_network
from utils.diag_paper_clock_alignment import gpu_inventory

PARENT=parent.ROOT
CARD=PROJECTS/'experiment_cards/A82_A84_native_ray_transfer_20261104.md'
PEAK=0


def resource(uuid, own=0):
    row=next(r for r in gpu_inventory() if r['uuid']==uuid)
    required=max(rules.MIN_FREE,PEAK+files.GIB)
    return dict(**row,own_reserved_bytes=own,required_bytes=required,admitted=row['free_bytes']+own>=required)


def check(root):
    p=files.read(Path(root)/'plan.json')
    files.require(p['experiment'] in rules.HOSTS,'Unknown experiment plan')
    for n,h in p['sources'].items():files.require(files.digest(PROJECTS/n)==h,'Bound source changed: '+n)
    files.require(files.digest(Path(p['parent'])/'plan.json')==p['parent_plan_sha256'],'Parent identity changed')
    rules.verify(p['experiment']);return p


def prepare(args):
    root=Path(args.root).resolve();source=Path(args.a71_root).resolve()
    if (root/'plan.json').exists():
        p=check(root);files.require(p['experiment']==args.experiment,'Wrong host root');print(args.experiment+' frozen plan reused.',flush=True);return
    files.require(not root.exists() or not any(root.iterdir()),'Nonempty unplanned output retained')
    files.require(not root.is_relative_to(source) and not source.is_relative_to(root),'Parent overlap')
    old=parent.check(source);summary=files.read(source/'summary.json')
    files.require(summary['completed_runs']==summary['planned_runs']==72,'Need completed A71 image-panel manifest')
    cases=rules.cases(args.experiment,old['cases']);inputs={c['image']:files.digest(c['image']) for c in cases}
    path=PROJECTS/'modules/models'/rules.CHECKPOINT[0]
    files.require(path.is_file() and files.digest(path)==rules.CHECKPOINT[1],'Existing FFHQ checkpoint missing/different; no replacement download')
    inputs[str(path)]=rules.CHECKPOINT[1]
    names=['utils/native_ray_transfer.py','utils/native_ray_transfer_runtime.py','utils/diag_native_ray_transfer.py',
           'utils/a82_a84_session.py','utils/test_native_ray_transfer.py','utils/test_native_ray_transfer_runtime.py',
           'utils/noise_signal_shrink.py','utils/a67_model_freeze.py','utils/paper_clock_runtime.py',
           'utils/current_score_runtime_r2.py','utils/test_current_score_checkpoint_r2.py',
           'run_native_ray_transfer_20261104.sh',str(CARD.relative_to(PROJECTS))]
    names += [str(f.relative_to(PROJECTS)) for f in rules.vendor(args.experiment).rglob('*') if f.is_file() and '__pycache__' not in str(f)]
    # Keep historical image/source identities as evidence without modifying their files.
    names += list(old['sources'])
    root.parent.mkdir(parents=True,exist_ok=True)
    files.require(shutil.disk_usage(root.parent).free>=files.GIB,'Need1GiB output headroom');root.mkdir(exist_ok=True)
    host=rules.HOSTS[args.experiment]
    p=dict(experiment=args.experiment,host=host,parent=str(source),parent_plan_sha256=files.digest(source/'plan.json'),
        cases=cases,inputs=inputs,checkpoint=dict(path=str(path),sha256=rules.CHECKPOINT[1]),
        configs={t:rules.configuration(args.experiment,t) for t in rules.TASKS},arms=list(rules.ARMS),
        sources={n:files.digest(PROJECTS/n) for n in dict.fromkeys(names)},upstream=rules.verify(args.experiment),
        budget=dict(runs=12,method_forwards=12*host['nfe'],input_VJP=12*host['vjp'],
                    operator_gradients=12*host['operator_gradients'],module_H_actions=12*host['nfe'],
                    numerical_tests='Toy native-vs-hooked loops; no real UNet. Real weight probe1 forward/load separately counted.'),
        protocol=dict(scope='Three reused FFHQ images x SR4/Gaussian; paired BASE/SHRINK, one seed; not an author benchmark mean.',
            module='Unchanged A71 noise_signal_shrink.select; gamma>=0 continuous, no clipping2, beta, search, or policy fallback.',
            coordinates='Model[-1,1] sigma=.05. Each host owns its original H; NEW y shared within its two arms. Cross-host observations are not identical.',
            response='MPGD exact linear clean-space Hq. DMPS/PGDM same-time identity clean-response proxy Hq/sqrt(alpha_t); not a measured next-denoiser response.',
            baseline='Pinned author loop/conditioning/operator AST; only one explicit correction expression can be instrumented; forced gamma1 and unmodified native loop must match.',
            MPGD='Original x0ddim100,eta.5; scale5 for BOTH tasks from paper appendixD.1. Released SR100 script4 / CLI default10 differ and are not quality-selected. No autoencoder.',
            DMPS='Original DDPM1000 learned_range,scale1.75. Original bicubicSR and configuredGaussian61/intensity3 actually constructs31-tap separable native SVD kernel.',
            PIGDM='Author RED-diff PGDM, released default50 uniform t980..0,eta0,awdTrue,weight1; SR native average-pooling SVD, Gaussian native5-tap sigma25. Raw clean predictor, native pseudo-inverse initialization, native final xs. No noise-covariance modification.',
            cost='No extra UNet/input VJP from module; TWO extra H evaluations/candidate step. MPGD original clean/operator gradients are separate from UNet VJP.',
            timing='Full loop includes given-y load, native operator construction/SVD, RNG/init/sampling/outputCPU sync and progress bookkeeping. Excludes model load/probe, creation of shared y, metrics and artifact writes; record each separately.',
            boundaries='Original positive correction amplitude can be canceled by projection. Do not inherit manifold/posterior guarantees or infer state necessity from two-arm gain.',
            decision='Per host/task: mean PSNR>=+.1dB,>=2/3 PSNR wins,SSIM nondecrease,LPIPS nonincrease; require init/finalRNG/grid pairing. Mixed quality reported separately. No rescue tuning.'))
    files.write(root/'plan.json',p);shutil.copyfile(CARD,root/'experiment_card.md')
    print(args.experiment+' PREPARE/FREEZE:',p['budget'],flush=True)


def observation(root,p,c,device):
    import torch
    from modules.dataset_loader import tensor_to_image
    folder=root/'observations'/c['key'];meta=folder/'observation.json'
    if meta.exists():
        r=files.read(meta)
        files.require(r['plan_sha256']==files.digest(root/'plan.json'),'Given observation plan changed')
        for n,h in r['artifacts'].items():files.require(files.digest(folder/n)==h,'Given H/y changed')
        return r
    files.require(not folder.exists() or not any(folder.iterdir()),'Partial observation retained')
    folder.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    with precision.precision('LEGACY_IEEE'),torch.no_grad():
        gt=runtime.load_image(c['image'],device);runtime.seed(c['kernel_seed'])
        op=runtime.Operator(runtime.make_operator(p['experiment'],c['task'],device))
        runtime.seed(c['measurement_seed']);expected=op.H(gt);y=expected+c['sigma']*torch.randn_like(expected)
        torch.save(dict(y=y.cpu()),folder/'y.pt');torch.save(runtime.state(op),folder/'operator_state.pt')
        tensor_to_image(gt.cpu().squeeze(0),denormalize=True).save(folder/'ground_truth.png')
    r=dict(path=str(folder/'y.pt'),operator_state=str(folder/'operator_state.pt'),
        seconds=time.perf_counter()-started,plan_sha256=files.digest(root/'plan.json'),
        artifacts={n:files.digest(folder/n) for n in ('y.pt','operator_state.pt','ground_truth.png')})
    files.write(meta,r);return r


def fresh_operator(p,c,device,obs,count=None):
    import torch
    runtime.seed(c['kernel_seed']);op=runtime.Operator(runtime.make_operator(p['experiment'],c['task'],device),count)
    actual=runtime.state(op);expected=torch.load(obs['operator_state'],map_location='cpu',weights_only=True)
    files.require(set(actual)==set(expected) and all(torch.equal(actual[k],expected[k]) for k in actual),'Native operator state differs')
    return op


def completed(root,c,arm):
    f=root/'cases'/c['key']/arm/'run.json'
    if not f.exists():return None
    r=files.read(f);files.require(r['status']=='complete','Prior unfinished/failed attempt retained: '+str(f))
    files.require(r['plan_sha256']==files.digest(root/'plan.json'),'Result plan changed')
    for n,h in r['artifacts'].items():files.require(files.digest(f.parent/n)==h,'Result artifact changed')
    return r


def fingerprint(value):
    import hashlib
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def measure(root,p,c,arm,model,uuid):
    import torch
    from modules.dataset_loader import tensor_to_image
    if completed(root,c,arm):return True
    if not resource(uuid,torch.cuda.memory_reserved())['admitted']:return False
    obs=observation(root,p,c,model.device);folder=root/'cases'/c['key']/arm;folder.mkdir(parents=True,exist_ok=False)
    r=dict(status='running',case=c,arm=arm,counts={},plan_sha256=files.digest(root/'plan.json'),artifacts={},device_uuid=uuid)
    files.write(folder/'run.json',r)
    def tick(step,count):
        r['counts']=dict(count);files.write(folder/'run.json',r)
        print(p['experiment']+' PROGRESS:',c['key'],arm,step,'/',p['host']['nfe'],flush=True)
    try:
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        with precision.precision('LEGACY_IEEE'),runtime.counters(model.model) as count:
            def operation():
                y=torch.load(obs['path'],map_location='cpu',weights_only=True)['y'].to(model.device)
                op=fresh_operator(p,c,model.device,obs,count)
                return runtime.run(p['experiment'],model,op,y,c,arm,count,progress=tick)
            (out,detail),seconds=clock.clocks(torch.cuda.synchronize,operation)
        host=p['host'];files.require(count['forward_attempts']==count['forwards']==count['corrections']==host['nfe'],'Native forward/correction budget changed')
        files.require(count['input_vjp_attempts']==host['vjp'] and count['operator_gradient_attempts']==host['operator_gradients'],'Native derivative count changed')
        files.require(count['module_H_actions']==(2*host['nfe'] if arm=='SHRINK' else 0),'Extra response cost changed')
        gt=runtime.load_image(c['image']);metric_start=time.perf_counter();metrics=common.metrics(out,gt)
        with torch.no_grad():
            y=torch.load(obs['path'],map_location='cpu',weights_only=True)['y'].to(model.device)
            op=fresh_operator(p,c,model.device,obs)
            metrics['measurement_mse_model']=float((y-op.H(out.to(model.device))).double().square().mean())
        metrics_seconds=time.perf_counter()-metric_start
        torch.save(out,folder/'output.pt');tensor_to_image(out.squeeze(0),denormalize=True).save(folder/'output.png')
        files.write(folder/'trace.json',detail['trace'])
        pair=dict(initial=fingerprint(detail['initial']),entry_cpu=fingerprint(detail['entry_rng']['cpu']),
            entry_cuda=[fingerprint(v) for v in detail['entry_rng']['cuda']],final_cpu=fingerprint(detail['final_rng']['cpu']),
            final_cuda=[fingerprint(v) for v in detail['final_rng']['cuda']],grid=detail['grid'],calls=count['calls'])
        r.update(status='complete',counts=count,metrics=metrics,metrics_seconds=metrics_seconds,
            full_seconds=seconds,peak_reserved_bytes=torch.cuda.max_memory_reserved(),paired_fingerprints=pair,
            cost_is_lower_bound=False,artifacts={n:files.digest(folder/n) for n in ('output.pt','output.png','trace.json')})
        files.write(folder/'run.json',r)
        print(p['experiment']+' COMPLETE:',c['key'],arm,'PSNR=',round(metrics['psnr'],4),'seconds=',round(seconds,3),flush=True)
        return True
    except BaseException as error:
        r.update(status='failed',counts=locals().get('count',{}),error=repr(error),traceback=traceback.format_exc(),cost_is_lower_bound=True)
        files.write(folder/'run.json',r);raise


def worker(args):
    import torch
    from modules.main_single import load_diffusion_model
    from utils.a54_session import exclusive
    from utils.current_score_runtime_r2 import input_vjp_parameters
    global PEAK
    root=Path(args.root);p=check(root);folder=root/'workers'/str(args.shard);folder.mkdir(parents=True,exist_ok=True)
    with exclusive(folder/'worker.lock'):
        try:
            if not resource(args.device_uuid)['admitted']:files.write(folder/'status.json',dict(status='resource_limited'));return
            for path,h in p['inputs'].items():files.require(files.digest(path)==h,'Input identity changed: '+path)
            started=time.perf_counter();model=load_diffusion_model('ffhq','cuda:0');model.model.eval()
            # Strictly reject missing weights, random-weight fallbacks or architecture substitution.
            state=torch.load(p['checkpoint']['path'],map_location='cpu',weights_only=True)
            model.model.load_state_dict(state,strict=True);del state
            torch.cuda.synchronize();load=dict(status='running',load_seconds=time.perf_counter()-started,probe_counts={})
            f=folder/f'load_{len(list(folder.glob("load_*.json"))):03d}.json';files.write(f,load)
            with frozen_network(model.model), (input_vjp_parameters(model.model) if p['experiment']=='A84' else nullcontext()):
                with precision.precision('LEGACY_IEEE'),runtime.counters(model.model) as probe,\
                        (torch.enable_grad() if p['experiment']=='A84' else torch.no_grad()):
                    load['probe_counts']=probe;files.write(f,load)
                    x=torch.zeros(1,3,256,256,device=model.device,requires_grad=p['experiment']=='A84')
                    value=model.model(x,torch.tensor([999],device=model.device))
                    files.require(tuple(value.shape)==(1,6,256,256) and bool(torch.isfinite(value).all()),'Native epsilon/variance probe failed')
                    if p['experiment']=='A84':
                        probe['input_vjp_attempts']+=1
                        gradient=torch.autograd.grad(value[:,:3].square().mean(),x)[0]
                        files.require(bool(torch.isfinite(gradient).all()),'Real checkpoint input VJP failed')
                        del gradient
                load['probe_counts']=probe;files.write(f,load);del value,x;torch.cuda.empty_cache()
                for i,c in enumerate(v for v in p['cases'] if v['shard']==args.shard):
                    for arm in (rules.ARMS if i%2==0 else rules.ARMS[::-1]):
                        if not measure(root,p,c,arm,model,args.device_uuid):
                            load['status']='resource_limited';files.write(f,load);files.write(folder/'status.json',dict(status='resource_limited'));return
                        PEAK=max(PEAK,torch.cuda.max_memory_reserved())
            load['status']='complete';files.write(f,load);files.write(folder/'status.json',dict(status='complete'))
        except BaseException as error:
            if 'load' in locals():
                load.update(status='failed',error=repr(error));files.write(f,load)
            files.write(folder/'status.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc()));raise


def summarize(root):
    root=Path(root);p=check(root);rows=[files.read(f) for f in (root/'cases').glob('*/*/run.json')]
    done={(r['case']['key'],r['arm']):r for r in rows if r['status']=='complete'};groups=[];comparisons=[];table=[]
    for task in rules.TASKS:
        cases=[c for c in p['cases'] if c['task']==task]
        for arm in rules.ARMS:
            data=[done[c['key'],arm] for c in cases if (c['key'],arm) in done]
            for r in data:table.append(dict(task=task,image_id=r['case']['image_id'],arm=arm,**r['metrics'],full_seconds=r['full_seconds'],NFE=r['counts']['forwards']))
            if len(data)==len(cases):groups.append(dict(task=task,arm=arm,cases=len(data),quality={k:statistics.mean(r['metrics'][k] for r in data) for k in ('psnr','ssim','lpips')},mean_full_seconds=statistics.mean(r['full_seconds'] for r in data),NFE=p['host']['nfe']))
        if all((c['key'],a) in done for c in cases for a in rules.ARMS):
            base=[done[c['key'],'BASE'] for c in cases];new=[done[c['key'],'SHRINK'] for c in cases]
            delta={k:statistics.mean(n['metrics'][k]-b['metrics'][k] for b,n in zip(base,new)) for k in ('psnr','ssim','lpips')}
            paired=all(n['paired_fingerprints']==b['paired_fingerprints'] for b,n in zip(base,new))
            wins=sum(n['metrics']['psnr']>b['metrics']['psnr'] for b,n in zip(base,new))
            comparisons.append(dict(task=task,SHRINK_minus_BASE=delta,images_PSNR_positive=wins,
                worst_PSNR=min(n['metrics']['psnr']-b['metrics']['psnr'] for b,n in zip(base,new)),paired_noise_grid_passed=paired,
                full_clock_ratio=statistics.mean(n['full_seconds'] for n in new)/statistics.mean(b['full_seconds'] for b in base),
                positive_development_candidate=paired and delta['psnr']>=.1 and wins>=2 and delta['ssim']>=0 and delta['lpips']<=0))
    counts=[r.get('counts',{}) for r in rows];loads=[files.read(f) for f in (root/'workers').glob('*/load_*.json')]
    result=dict(experiment=p['experiment'],host=p['host']['host'],classification='DEVELOPMENT_NATIVE_RAY_TRANSFER',
        completed_runs=len(done),planned_runs=12,image_clusters=3,groups=groups,comparisons=comparisons,
        actual_new_cost={key:sum(c.get(key,0) for c in counts) for key in ('forward_attempts','forwards','input_vjp_attempts','operator_gradient_attempts','module_H_actions','base_H_actions')},
        probe_forwards=sum(v.get('probe_counts',{}).get('forward_attempts',0) for v in loads),
        probe_input_VJP=sum(v.get('probe_counts',{}).get('input_vjp_attempts',0) for v in loads),
        cost_is_lower_bound=any(r['status']!='complete' for r in rows),limits=p['protocol'])
    files.write(root/'summary.json',result)
    if table:
        with (root/'comparison.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    lines=['# '+p['experiment']+' '+p['host']['host'],'','|Task|Arm|PSNR|SSIM|LPIPS|Full seconds|NFE|','|---|---|---:|---:|---:|---:|---:|']
    for r in groups:
        q=r['quality'];lines.append(f"|{r['task']}|{r['arm']}|{q['psnr']:.4f}|{q['ssim']:.5f}|{q['lpips']:.5f}|{r['mean_full_seconds']:.3f}|{r['NFE']}|")
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(p['experiment']+' Summary:',json.dumps(result,ensure_ascii=False),flush=True)


def archive(root):
    from utils import archive_refinement_confirmation as exporter
    p=check(root);exporter.BRANCH=rules.branch(p['experiment'])
    bundle,commit=exporter.export_and_commit({p['experiment']:Path(root)},projects=PROJECTS)
    print('Archive:',bundle,'Results commit:',commit,'Branch:',exporter.BRANCH,flush=True)
    subprocess.run(['git','push','origin',exporter.BRANCH],cwd=PROJECTS.parent,check=True)
    print('RESULTS_PUSH_PASS:',p['experiment'],'native host transfer uploaded.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--experiment',choices=rules.HOSTS,required=True)
    parser.add_argument('--root');parser.add_argument('--a71-root',default=PARENT)
    parser.add_argument('--shard',type=int,choices=(0,1));parser.add_argument('--device-uuid')
    action=parser.add_mutually_exclusive_group(required=True)
    for flag in ('prepare','worker','summarize','archive'):action.add_argument('--'+flag,action='store_true')
    args=parser.parse_args();args.root=args.root or rules.root_for(args.experiment)
    if args.worker and (args.shard is None or not args.device_uuid):parser.error('Worker requires shard and allocated GPU UUID')
    if args.prepare:prepare(args)
    elif args.worker:worker(args)
    elif args.summarize:summarize(args.root)
    else:archive(args.root)


if __name__=='__main__':main()
