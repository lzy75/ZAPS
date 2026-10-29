"""A70 matched strong-ZAPS confirmation: discrepancy vs noise-corrected projection."""
import argparse
import gc
import hashlib
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
from utils import noise_signal_shrink as rule
from utils import noise_signal_shrink_runtime as runtime
from utils import historical_ffhq_gain as files
from utils import historical_ffhq_runtime as old_runtime
from utils import paper_clock_runtime as clock
from utils import diag_noise_feasible_zaps as common
from utils.a67_model_freeze import frozen_network
from utils.diag_paper_clock_alignment import write,resource

ROOT='/data0/lzy/ZAPS-results/diag_noise_signal_shrink/A70_20261029'
PARENT='/data0/lzy/ZAPS-results/diag_scalar_gamma_solver/A69_20261029'
BRANCH='results/noise-signal-shrink-20261029'
CARD=PROJECTS/'experiment_cards/A70_noise_signal_shrink_20261029.md'


def check(root):
    p=files.read(root/'plan.json');files.require(p['experiment']=='A70_noise_signal_shrink','Wrong plan')
    for name,sha in p['sources'].items():files.require(files.digest(PROJECTS/name)==sha,'Source changed: '+name)
    for name,sha in p['parent_files'].items():files.require(files.digest(name)==sha,'Parent changed: '+name)
    return p


def select_new(folder,root):
    from utils.a56_selection_r2 import collect_usage
    from utils.frozen_direction_transfer import select,eligible_paths
    data={'ffhq':str(Path(folder).resolve())};used=set();hashes=set();audit=hashlib.sha256();records=0
    paths=set()
    for base in (PROJECTS/'results',PROJECTS/'server_records',Path('/data0/lzy/ZAPS-results')):
        if base.exists():
            for pattern in ('run.json','plan.json'):
                paths.update(f.resolve() for f in base.rglob(pattern) if not f.resolve().is_relative_to(root))
    for f in sorted(paths):
        u=collect_usage(files.read(f),data);used.update(u['used_names']['ffhq']);hashes.update(u['used_hashes'])
        audit.update((str(f)+':'+files.digest(f)+'\n').encode());records+=1
    files.require(records>0,'No accessible history; cannot label images unused')
    pool=[dict(image=str(f.resolve()),image_id=f.stem,image_sha256=files.digest(f)) for f in eligible_paths(folder)]
    chosen=select(pool,used,hashes,rule.NEW_IMAGES,20261029)
    return chosen,dict(history_records=records,history_digest=audit.hexdigest(),used_names=len(used),
                       eligible_files=len(pool),selection_seed=20261029,
                       scope='Unused within accessible case/plan records, candidate inventories excluded; no model-training-held-out claim')


def prepare(args):
    root=Path(args.root).resolve();parent=Path(args.a69_root).resolve()
    if (root/'plan.json').exists():check(root);print('A70 frozen plan reused.',flush=True);return
    files.require(not root.exists() or not any(root.iterdir()),'Nonempty unplanned root retained')
    files.require(not root.is_relative_to(parent) and not parent.is_relative_to(root),'Parent/output overlap')
    from utils.diag_scalar_gamma_solver import check as parent_check
    before=parent_check(parent);summary=files.read(parent/'summary.json')
    files.require(summary['completed_runs']==summary['planned_runs']==40,'Need complete A69, no partial calibration')
    parent_files={str(parent/'plan.json'):files.digest(parent/'plan.json'),str(parent/'summary.json'):files.digest(parent/'summary.json')}
    for c in before['cases']:
        for arm in ('BASE','FIRST_NOISE'):
            files.require(common.completed(parent,c,arm) is not None,'Need complete parent source')
            for n in ('run.json','paired_trace.json')+ (('entry.pt',) if arm=='BASE' else ()):
                f=common.record_path(parent,c,arm).parent/n;parent_files[str(f)]=files.digest(f)
        meta=parent/'observations'/(c['key']+'.json');obs=files.read(meta)
        files.require(files.digest(obs['path'])==obs['sha256'],'Parent observation changed')
        for f in (meta,Path(obs['path'])):parent_files[str(f)]=files.digest(f)
    chosen,audit=select_new(args.ffhq_dir,root);cases=[]
    for j,c in enumerate(chosen):
        for sigma in rule.SIGMAS:
            seed=1000+int(c['image_id'])
            cases.append(dict(c,seed=seed,sigma=sigma,shard=j%2,key=f"{c['image_id']}_n{round(sigma*1000):03d}_s{seed}"))
    source_names=['utils/noise_signal_shrink.py','utils/noise_signal_shrink_runtime.py','utils/diag_noise_signal_shrink.py',
        'utils/a70_session.py','utils/test_noise_signal_shrink.py','utils/test_noise_signal_shrink_runtime.py','utils/test_a70_cli.py',
        'utils/noise_feasible_multiplier.py','utils/noise_feasible_runtime.py','utils/diag_noise_feasible_zaps.py',
        'utils/a67_model_freeze.py','utils/historical_ffhq_gain.py','utils/historical_ffhq_runtime.py','utils/paper_clock_runtime.py',
        'utils/a56_selection_r2.py','utils/frozen_direction_transfer.py','modules/zaps_algorithm.py','modules/degradations.py',
        'modules/wavelet.py','modules/main_single.py','utils/metrics.py','run_a70_20261029.sh',str(CARD.relative_to(PROJECTS))]
    checkpoint=before['checkpoint'];files.require(files.digest(checkpoint['path'])==checkpoint['sha256'],'Checkpoint differs')
    root.parent.mkdir(parents=True,exist_ok=True);files.require(shutil.disk_usage(root.parent).free>=3*files.GIB,'Need3GiB result space');root.mkdir()
    p=dict(experiment='A70_noise_signal_shrink',parent=str(parent),parent_files=parent_files,calibration_cases=before['cases'],
        cases=cases,selection=audit,mismatch_donors=rule.mismatches(cases),checkpoint=checkpoint,
        config=files.config(),grid=files.grid(),arms=list(rule.ARMS),sources={n:files.digest(PROJECTS/n) for n in source_names},
        budget=dict(method_forwards=16800,paired_diagnostic_forwards=1680,calibration_forwards=240,optimizer_backwards=560,input_vjp=0,warmup_separate=True),
        protocol=dict(quality='Same-image BASE270 prefix/noise paired suffix; independent complete outputs also evaluated',
            clock='Independent full300 learning+sampling+policy+output CPU sync; excludes model loading/warmup/metrics/files',
            baseline='9/17 strong local ZAPS, IEEE all arms, model-coordinate noise, original empirical bicubic guidance mapping',
            scope='Four unused-in-accessible-history FFHQ SR images, two noise levels, one seed/image. Fixed30, final unroll only. '
                  'No new optimizer/gradient/trajectory-curvature/step-size claim. Shrinkage Gaussian residual model is an engineering proxy, '
                  'not an exact posterior or SURE certificate. No gamma upper cap for SHRINK; FIRST retains historical0..2.',
            constants='Per-noise global median from OLD4-image traces only; never GT/confirmation labels',
            mismatch='Frozen cyclic other-image donor at same sigma/slot; actual confirmation feedback traces, no metrics read'),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECTS.parent,text=True).strip())
    write(root/'plan.json',p);shutil.copyfile(CARD,root/'experiment_card.md')
    print('A70 PREPARE/FREEZE:',[c['image_id'] for c in chosen],';8 cases x7 arms;16800 method+1680 paired+240 calibration forwards;0 input VJP.',flush=True)


def validate(count,arm,full):
    files.require(count['forward_attempts']==count['forwards']==(300 if full else 30),'Forward budget differs')
    files.require(count['optimizer_backward_attempts']==count['optimizer_backward_completed']==(10 if full else 0),'Optimizer budget differs')
    files.require(count['input_vjp_attempts']==0,'Unexpected input backward')
    files.require(count['calls']==files.grid()[::-1]*(10 if full else 1),'Original grid differs')


def calibrate(root,p,c,model):
    import torch
    folder=root/'calibration'/c['key'];f=folder/'run.json'
    if f.exists():
        row=files.read(f);files.require(row['status']=='complete','Previous calibration attempt retained')
        for n,sha in row['artifacts'].items():files.require(files.digest(folder/n)==sha,'Calibration artifact changed')
        return
    parent=Path(p['parent']);entry=torch.load(common.record_path(parent,c,'BASE').parent/'entry.pt',map_location='cpu',weights_only=True)
    obs=files.read(parent/'observations'/(c['key']+'.json'));y=torch.load(obs['path'],map_location='cpu',weights_only=True)['y'].to(model.device)
    folder.mkdir(parents=True);row=dict(status='running',case=c,counts=None,artifacts={});write(f,row)
    try:
        with old_runtime.precision('LEGACY_IEEE'),clock.counters(model.model) as counts:
            output,trace=runtime.replay(model,old_runtime.make_operator(model.device,counts),y,entry,'SHRINK',c['sigma'])
        validate(counts,'SHRINK',False);write(folder/'trace.json',trace)
        row.update(status='complete',counts=counts,artifacts={'trace.json':files.digest(folder/'trace.json')});write(f,row)
        print('A70 CALIBRATION:',c['key'],'30 forwards; labels excluded.',flush=True)
    except BaseException as e:
        if 'counts' in locals():row['counts']=counts
        row.update(status='failed',error=repr(e),traceback=traceback.format_exc());write(f,row);raise


def freeze_constants(root):
    root=Path(root);p=check(root);values={};sources={}
    for sigma in rule.SIGMAS:
        group=[c for c in p['calibration_cases'] if c['sigma']==sigma]
        for policy in ('FIRST','SHRINK'):
            curves=[]
            for c in group:
                f=(Path(p['parent'])/'cases'/c['key']/'FIRST_NOISE'/'paired_trace.json') if policy=='FIRST' else root/'calibration'/c['key']/'trace.json'
                if policy=='SHRINK':files.require(files.read(f.parent/'run.json')['status']=='complete','Calibration incomplete')
                curves.append([v['gamma'] for v in files.read(f)]);sources[str(f)]=files.digest(f)
            values[f'{policy}_{sigma}']=rule.constant(curves)
    document=dict(values=values,sources=sources,GT_used=False,aggregation='Global median over four OLD images x30 positions per sigma')
    f=root/'constants.json'
    if f.exists():files.require(files.read(f)==document,'Constants changed')
    else:write(f,document)
    print('A70 OLD-ONLY CONSTANTS FROZEN:',values,flush=True)


def freeze_mismatches(root):
    root=Path(root);p=check(root);curves={};sources={}
    for c in p['cases']:
        for policy,arm in (('FIRST','FIRST_NOISE'),('SHRINK','SHRINK')):
            files.require(common.completed(root,c,arm) is not None,'Finish all feedback cases first')
            f=common.record_path(root,c,arm).parent/'paired_trace.json'
            curve=[v['gamma'] for v in files.read(f)];files.require(len(curve)==30,'Incomplete donor curve')
            curves[policy+'_'+c['key']]=curve;sources[str(f)]=files.digest(f)
    document=dict(curves=curves,sources=sources,donors=p['mismatch_donors'],GT_used=False)
    f=root/'mismatches.json'
    if f.exists():files.require(files.read(f)==document,'Mismatch profiles changed')
    else:write(f,document)
    print('A70 cyclic mismatch frozen; no quality labels used.',flush=True)


def control_profile(root,p,c,arm):
    if arm in rule.FEEDBACK:return None
    policy=arm.split('_',1)[1]
    if arm.startswith('FIXED_'):
        data=files.read(root/'constants.json')
        return [data['values'][f"{policy}_{c['sigma']}"]]*30
    data=files.read(root/'mismatches.json');donor=p['mismatch_donors'][c['key']]
    files.require(donor!=c['key'],'Self donor forbidden')
    return data['curves'][policy+'_'+donor]

def measure(root,p,c,arm,model,uuid):
    import torch
    from modules.main_single import load_image_as_tensor
    from modules.dataset_loader import tensor_to_image
    if common.completed(root,c,arm):return
    files.require(resource(uuid,'ffhq',torch.cuda.memory_reserved())['admitted'],'Resource margin lost before attempt')
    obs=common.observation(root,c,model.device)
    profile=control_profile(root,p,c,arm)
    folder=common.record_path(root,c,arm).parent;folder.mkdir(parents=True,exist_ok=False)
    r=dict(status='running',case=c,arm=arm,plan_sha256=files.digest(root/'plan.json'),artifacts={},
           device_uuid=uuid,full_counts=None,replay_counts=None)
    write(folder/'run.json',r)
    def tick(value):
        r['full_counts']=value;write(folder/'run.json',r)
        print('A70 PROGRESS:',c['key'],arm,'F=',value['forwards'],'cap=',300,flush=True)
    try:
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        with old_runtime.precision('LEGACY_IEEE'),clock.counters(model.model,tick) as counts:
            def operation():
                y=torch.load(obs['path'],map_location='cpu',weights_only=True)['y'].to(model.device)
                return runtime.execute(model,old_runtime.make_operator(model.device,counts),y,c['seed'],arm,c['sigma'],profile)
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
                                        'NULL' if arm=='BASE' else arm,c['sigma'],profile)
        r['replay_counts']=replay_counts;validate(replay_counts,arm,False)
        if arm=='BASE':
            error=float((paired.double()-independent.double()).norm()/independent.double().norm().clamp(min=1e-30))
            r['null_relative_error']=error;files.require(error<=1e-6,'Core/null mismatch')
        gt=load_image_as_tensor(c['image']).cpu()
        r.update(paired_metrics=common.metrics(paired,gt),independent_metrics=common.metrics(independent,gt),
                 complete_reconstruction_seconds=seconds,peak_reserved_bytes=peak,
                 full_precision='IEEE',cost_is_lower_bound=False,
                 gamma_more_than_two=sum(v.get('gamma',1)>2 for v in trace),
                 gamma_zero=sum(v.get('gamma',1)==0 for v in trace))
        torch.save(paired,folder/'paired.pt');torch.save(independent,folder/'independent.pt')
        tensor_to_image(paired.squeeze(0),denormalize=True).save(folder/'paired.png')
        write(folder/'paired_trace.json',trace);write(folder/'independent_trace.json',detail['trace'])
        names=['paired.pt','independent.pt','paired.png','paired_trace.json','independent_trace.json']+(['entry.pt'] if arm=='BASE' else [])
        r.update(status='complete',artifacts={n:files.digest(folder/n) for n in names});write(folder/'run.json',r)
        print('A70 COMPLETE:',c['key'],arm,'paired PSNR=',round(r['paired_metrics']['psnr'],4),
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
        status=folder/(args.phase+'.json')
        if not resource(args.device_uuid,'ffhq')['admitted']:write(status,dict(status='resource_limited'));return
        files.require(files.digest(p['checkpoint']['path'])==p['checkpoint']['sha256'],'Checkpoint changed')
        for c in p['cases']:files.require(files.digest(c['image'])==c['image_sha256'],'Confirmation image changed')
        start=time.perf_counter();model=load_diffusion_model('ffhq','cuda:0');model.model.eval();torch.cuda.synchronize()
        load=dict(status='running',phase=args.phase,load_seconds=time.perf_counter()-start,uuid=args.device_uuid,warmup_forward_attempts=0)
        f=folder/f'load_{len(list(folder.glob("load_*.json"))):03d}.json';write(f,load)
        try:
            with frozen_network(model.model):
                load['warmup_forward_attempts']=1;write(f,load)
                with torch.no_grad(),old_runtime.precision('LEGACY_IEEE'):
                    model._predict_eps(torch.zeros(1,3,256,256,device=model.device),torch.tensor([999],device=model.device))
                torch.cuda.synchronize()
                if args.phase=='calibration':
                    for c in p['calibration_cases']:
                        if int(c['image_id'])%2==args.shard:calibrate(root,p,c,model)
                else:
                    for c in p['cases']:
                        if c['shard']!=args.shard:continue
                        arms=list(rule.FEEDBACK if args.phase=='feedback' else rule.CONTROLS)
                        if args.phase=='feedback' and int(c['image_id'])%2:arms=['BASE','SHRINK','FIRST_NOISE']
                        if args.phase=='controls':
                            shift=int(c['image_id'])%len(arms);arms=arms[shift:]+arms[:shift]
                        for arm in arms:measure(root,p,c,arm,model,args.device_uuid)
            load['status']='complete';write(status,dict(status='complete'))
        except BaseException as e:
            load.update(status='failed',error=repr(e));write(status,load);raise
        finally:write(f,load)


def summarize(root):
    root=Path(root);p=check(root);rows=[files.read(f) for f in (root/'cases').glob('*/*/run.json')]
    done=[r for r in rows if r['status']=='complete'];groups=[];contrasts=[]
    for sigma in rule.SIGMAS:
        for arm in rule.ARMS:
            data=[r for r in done if r['case']['sigma']==sigma and r['arm']==arm]
            if not data:continue
            groups.append(dict(sigma=sigma,arm=arm,cases=len(data),
                quality={k:statistics.mean(r['paired_metrics'][k] for r in data) for k in ('psnr','ssim','lpips')},
                independent_quality={k:statistics.mean(r['independent_metrics'][k] for r in data) for k in ('psnr','ssim','lpips')},
                mean_full_seconds=statistics.mean(r['complete_reconstruction_seconds'] for r in data),
                mean_full_NN=statistics.mean(r['full_counts']['forwards'] for r in data),
                gamma_more_than_two=sum(r['gamma_more_than_two'] for r in data),gamma_zero=sum(r['gamma_zero'] for r in data)))
            references=['BASE']
            if arm=='FIRST_NOISE':references+=['FIXED_FIRST','MISMATCH_FIRST']
            if arm=='SHRINK':references+=['FIRST_NOISE','FIXED_SHRINK','MISMATCH_SHRINK']
            for reference in references:
                if reference==arm:continue
                peers={r['case']['key']:r for r in done if r['arm']==reference}
                pairs=[r for r in data if r['case']['key'] in peers]
                if pairs:contrasts.append(dict(sigma=sigma,comparison=arm+' minus '+reference,cases=len(pairs),
                    delta={k:statistics.mean(r['paired_metrics'][k]-peers[r['case']['key']]['paired_metrics'][k] for r in pairs) for k in ('psnr','ssim','lpips')},
                    images_PSNR_positive=sum(r['paired_metrics']['psnr']>peers[r['case']['key']]['paired_metrics']['psnr'] for r in pairs),
                    worst_PSNR=min(r['paired_metrics']['psnr']-peers[r['case']['key']]['paired_metrics']['psnr'] for r in pairs)))
    calibrations=[files.read(f) for f in (root/'calibration').glob('*/run.json')]
    loads=[files.read(f) for f in (root/'workers').glob('*/load_*.json')]
    result=dict(experiment=p['experiment'],classification='FROZEN_RULE_MATCHING_CONFIRMATION',
        completed_runs=len(done),planned_runs=56,image_clusters=4,groups=groups,paired_comparisons=contrasts,
        constants=files.read(root/'constants.json')['values'] if (root/'constants.json').exists() else None,
        actual_cost=dict(method_forward_attempts=sum((r.get('full_counts') or {}).get('forward_attempts',0) for r in rows),
            diagnostic_forward_attempts=sum((r.get('replay_counts') or {}).get('forward_attempts',0) for r in rows),
            calibration_forward_attempts=sum((r.get('counts') or {}).get('forward_attempts',0) for r in calibrations),
            optimizer_backward_attempts=sum((r.get('full_counts') or {}).get('optimizer_backward_attempts',0) for r in rows),
            input_vjp_attempts=sum((r.get('full_counts') or {}).get('input_vjp_attempts',0)+(r.get('replay_counts') or {}).get('input_vjp_attempts',0) for r in rows),
            warmup_forward_attempts=sum(r['warmup_forward_attempts'] for r in loads)),
        cost_is_lower_bound=any(r['status']!='complete' for r in rows+calibrations),limits=p['protocol']['scope'])
    write(root/'summary.json',result)
    lines=['# A70 strong-ZAPS noise-aware scalar policies','', '|sigma|method|n|PSNR|SSIM|LPIPS|full seconds|NN|', '|---|---|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        m=g['quality'];lines.append(f"|{g['sigma']}|{g['arm']}|{g['cases']}|{m['psnr']:.4f}|{m['ssim']:.5f}|{m['lpips']:.5f}|{g['mean_full_seconds']:.3f}|{g['mean_full_NN']:.0f}|")
    lines+=['','Paired quality on identical BASE270 prefix; full clock is an independent300-forward execution. Model load/warmup/metrics/file writes excluded.', '',result['limits']]
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('A70 Summary:',json.dumps(result,ensure_ascii=False),flush=True)


def archive(root):
    from utils import archive_refinement_confirmation as exporter
    exporter.BRANCH=BRANCH
    bundle,commit=exporter.export_and_commit({'A70':Path(root)},projects=PROJECTS)
    print('Archive:',bundle,'Results commit:',commit,'Branch:',BRANCH,flush=True)
    subprocess.run(['git','push','origin',BRANCH],cwd=PROJECTS.parent,check=True)
    print('RESULTS_PUSH_PASS: A70 uploaded.',flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=ROOT);p.add_argument('--a69-root',default=PARENT)
    p.add_argument('--ffhq-dir',default='/home/lzy/FFHQ/00000/00000');p.add_argument('--shard',type=int,choices=(0,1))
    p.add_argument('--phase',choices=('calibration','feedback','controls'));p.add_argument('--device-uuid')
    p.add_argument('--check-worker-cli',action='store_true',help=argparse.SUPPRESS)
    group=p.add_mutually_exclusive_group(required=True)
    for n in ('prepare','worker','freeze-constants','freeze-mismatches','summarize','archive'):group.add_argument('--'+n,action='store_true')
    return p


def main():
    p=parser();a=p.parse_args()
    if a.worker and (a.shard is None or not a.phase or not a.device_uuid):p.error('Worker needs shard/phase/UUID')
    if a.check_worker_cli:
        files.require(a.worker,'Need worker');print(json.dumps(dict(shard=a.shard,phase=a.phase,device_uuid=a.device_uuid,model_calls=0)));return
    if a.prepare:prepare(a)
    elif a.worker:worker(a)
    elif a.freeze_constants:freeze_constants(Path(a.root))
    elif a.freeze_mismatches:freeze_mismatches(Path(a.root))
    elif a.summarize:summarize(a.root)
    else:archive(a.root)

if __name__=='__main__':main()
