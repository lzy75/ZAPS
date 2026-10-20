"""A17: frozen coefficient 2 on the other cached cohort; 12 diagnostic calls."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS))
from utils import diag_terminal_state_strength as old
from utils import diag_clean_response_precision as precision
from utils.diag_state_trajectory import digest

WRITE=old.WRITE
METRICS=old.METRICS
CARD=PROJECTS/'experiment_cards/A17_fixed_terminal_strength_replication_20261020.md'
ROOT='/data0/lzy/ZAPS-results/diag_fixed_terminal_replication/A17_20261020'


def source_case(record,case,config,grid):
    expected={k:v for k,v in case.items() if k!='cohort'}
    if (record['status']!='complete' or record['case']!=expected or not record['gates'] or
        not all(g['passed'] for g in record['gates'].values()) or record['config']!=config or record['grid']!=grid):
        raise ValueError('Need completed compatible source, including recovery: '+case['key'])
    return expected


def prepare(args):
    a16=Path(args.a16_dir).resolve();p16=old.read(a16/'plan.json')
    if digest(a16/'plan.json')!=(a16/'plan.sha256').read_text().strip():raise ValueError('A16 plan changed')
    prior=[old.load_record(a16,c) for c in p16['cases']]
    if any(r is None for r in prior) or old.summarize(prior)!=old.read(a16/'summary.json') or old.summarize(prior)['classification']!='FIXED_STRENGTH_CANDIDATE_ONLY':
        raise ValueError('Need completed A16 fixed-only context')
    a14=Path(args.a14_dir).resolve();p14=old.read(a14/'plan.json')
    if digest(a14/'plan.json')!=(a14/'plan.sha256').read_text().strip():raise ValueError('A14 plan changed')
    identities={};sources={};cases=[]
    def bind(path,sha=None):
        path=Path(path);value=digest(path)
        if sha is not None and value!=sha:raise ValueError('Source changed: '+str(path))
        identities[str(path)]=value
    for root in (a16,a14):
        for name in ('plan.json','plan.sha256','summary.json'):bind(root/name)
    for c in p16['cases']:
        bind(a16/c['key']/'run.json');bind(a16/c['key']/'run.sha256')
    for c in p14['cases']:
        if c['cohort']!='development':continue
        s=p14['case_sources'][c['key']];folder=Path(s['directory']);r=old.read(folder/'run.json')
        bind(folder/'run.json',s['record_sha256'])
        case=source_case(r,c,p16['config'],p16['grid']);cases.append(case)
        for name in ('pre_last_state.pt','last_opt.pt'):bind(folder/name,r['artifacts_sha256'][name])
        bind(r['measurement']['path'],r['measurement']['sha256']);bind(case['image'],case['image_sha256'])
        sources[c['key']]=dict(directory=str(folder),measurement=r['measurement'],final_metrics=r['final_metrics'])
    if len(cases)!=6 or {(c['image_id'],c['sampling_seed']) for c in cases}!={(i,s) for i in ('00020','00021','00022') for s in (1000,2000)}:
        raise ValueError('Frozen target cohort differs')
    for path,sha in p16['identity_files'].items():
        if Path(path).suffix=='.py' and Path(path).is_relative_to(PROJECTS):bind(path,sha)
    own=[Path(__file__),CARD,PROJECTS/'utils/test_diag_fixed_terminal_replication.py',PROJECTS/'run_a17_20261020.sh',PROJECTS/'utils/diag_terminal_state_strength.py']
    for path in own:bind(path)
    workers={str(path):identities[str(path)] for path in own}
    root=Path(args.root).resolve()
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in (a16,a14,*[Path(s['directory']).resolve() for s in sources.values()])):
        raise ValueError('New run overlaps sources')
    plan=dict(experiment='A17_fixed_terminal_replication',cases=cases,sources=sources,config=p16['config'],grid=p16['grid'],
        checkpoint_sha256=p16['checkpoint_sha256'],runtime=p16['runtime'],ieee_policy=p16['ieee_policy'],identity_files=identities,
        worker_identity_files=workers,prior_records=prior,source_A16=str(a16),coefficients=[1.,2.],user_mean_guards=old.GUARDS,
        planned_method_nfe=0,planned_diagnostic_nfe=12,evaluation_t=0,parameter_index=0)
    if root.exists():
        if old.read(root/'plan.json')!=plan:raise ValueError('Frozen A17 differs; refuse overwrite')
    else:
        root.mkdir(parents=True,exist_ok=False);WRITE(root/'plan.json',plan)
        (root/'plan.sha256').write_text(digest(root/'plan.json')+'\n');(root/'experiment_card.md').write_text(CARD.read_text())
    print('A17 PREPARE PASS: six cached target cases, active recovery respected; 12 diagnostic calls.',flush=True)


def contrast(records,a,b):
    complete=[r for r in records if r['status']=='complete'];images=sorted({r['case']['image_id'] for r in complete})
    delta={k:statistics.mean(r['branches'][a][k]-r['branches'][b][k] for r in complete) if complete else None for k in METRICS}
    floor={k:statistics.mean(old.limits(r['floors'])[k] for r in complete) if complete else None for k in METRICS}
    wins=sum(len(g)==2 and statistics.mean(r['branches'][a]['psnr']-r['branches'][b]['psnr']-old.limits(r['floors'])['psnr'] for r in g)>0
        for g in ([r for r in complete if r['case']['image_id']==image] for image in images))
    quality=bool(complete and delta['ssim']>=-.002 and delta['lpips']<=.005)
    return dict(mean_delta=delta,mean_numerical_limit=floor,images_psnr_better=wins,quality_guard=quality,
        passed=bool(complete and quality and delta['psnr']>floor['psnr'] and wins>=2 and (b!='C' or delta['loss']<=floor['loss'])))


def summarize(records,prior):
    complete=[r for r in records if r['status']=='complete']
    comparisons={a+'_'+b:contrast(records,a,b) for a,b in (('F1','C'),('F2','C'),('F2','F1'))}
    supported=len(complete)==6 and comparisons['F2_C']['passed'] and comparisons['F2_F1']['passed']
    combined=prior+complete
    return dict(completed_cases=len(complete),planned_cases=6,image_clusters=len({r['case']['image_id'] for r in complete}),
        method_nfe=sum(r['method_nfe'] for r in records),diagnostic_nfe=sum(r['diagnostic_nfe'] for r in records),
        scientific_extra_nfe_per_case=1,deployment_total_method_nfe_per_case=301,
        cumulative_queue_cost=dict(method_nfe=3600,diagnostic_nfe=2276+sum(r['diagnostic_nfe'] for r in records),total=5876+sum(r['diagnostic_nfe'] for r in records)),
        classification='INCOMPLETE_PILOT' if len(complete)!=6 else 'FIXED_F2_CROSS_COHORT_REPLICATION_SUPPORTED' if supported else 'FIXED_F2_CROSS_COHORT_REPLICATION_NOT_SUPPORTED',
        target_comparisons=comparisons,selection_comparisons={a+'_'+b:contrast(prior,a,b) for a,b in (('F1','C'),('F2','C'),('F2','F1'))},
        pooled_descriptive_comparisons={a+'_'+b:contrast(combined,a,b) for a,b in (('F1','C'),('F2','C'),('F2','F1'))},
        rows=[dict(case=r['case']['key'],branches=r['branches'],deltas_vs_C={a:{k:r['branches'][a][k]-r['branches']['C'][k] for k in METRICS} for a in ('F1','F2')},
                   gates_passed=all(g['passed'] for g in r['gates'].values())) for r in complete],
        target_per_case_lpips_over_mean_tolerance=[r['case']['key'] for r in complete if r['branches']['F2']['lpips']-r['branches']['C']['lpips']>.005],
        cost_is_lower_bound_if_abruptly_interrupted=any(r['status']=='running' for r in records),
        limits='Frozen F2 selected on A16; target images also historically observed. Cross-cohort check, not independent unseen validation or state benefit. Pooled means are descriptive and cannot rescue target-cohort failure.')


def load_record(root,case):
    path=root/case['key']/'run.json'
    if not path.exists():
        if path.parent.exists():raise ValueError('Interrupted case retained: '+case['key'])
        return None
    r=old.read(path)
    if (digest(path)!=(path.parent/'run.sha256').read_text().strip() or r['status']!='complete' or r['case']!=case or
        r['plan_sha256']!=digest(root/'plan.json') or r['method_nfe']!=0 or r['diagnostic_nfe']!=2 or len(r['calls'])!=2 or
        not r['gates'] or not all(g['passed'] for g in r['gates'].values()) or digest(path.parent/'outputs.pt')!=r['outputs_sha256']):
        raise ValueError('Invalid/interrupted A17 retained: '+case['key'])
    return r


def run_case(root,plan,case,model,device,policy,runtime):
    import torch
    from unittest.mock import patch
    from modules.zaps_algorithm import ZAPS
    from modules.degradations import get_operator
    from modules.main_single import load_image_as_tensor
    from utils.diag_ffhq_regression import TransposeMode
    from utils.diag_imagenet_gaussian_unit_domain import unit_operator
    from utils.diag_guidance_local_probe import evaluate_field
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state,same_rng
    from utils.diag_zaps_trace_audit import relative_error
    from utils.metrics import compute_all_metrics
    folder=root/case['key'];folder.mkdir(exist_ok=False);started=time.perf_counter()
    r=dict(status='running',case=case,plan_sha256=digest(root/'plan.json'),method_nfe=0,diagnostic_nfe=0,calls=[],gates={},policy=policy,runtime=runtime)
    def save():r['seconds']=time.perf_counter()-started;WRITE(folder/'run.json',r)
    def count(_module,inputs):
        r['diagnostic_nfe']+=1;r['calls'].append(dict(t=int(inputs[1].flatten()[0]),parameter_index=0));save()
        if r['diagnostic_nfe']>2 or r['calls'][-1]['t']!=0:raise RuntimeError('A17 call/time budget exceeded')
    save();hook=model.model.register_forward_pre_hook(count)
    try:
        src=plan['sources'][case['key']];source=Path(src['directory'])
        paths=[source/n for n in ('run.json','pre_last_state.pt','last_opt.pt')]+[Path(src['measurement']['path']),Path(case['image'])]
        for path in paths:
            if digest(path)!=plan['identity_files'][str(path)]:raise ValueError('Source changed: '+str(path))
        before=rng_state(device)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))),torch.no_grad():
            state=torch.load(source/'pre_last_state.pt',map_location='cpu',weights_only=True)
            c=torch.load(source/'last_opt.pt',map_location='cpu',weights_only=True).to(device)
            if c.shape!=(1,3,256,256) or c.dtype!=torch.float32 or not torch.isfinite(c).all() or state['tau'].tolist()!=plan['grid']:raise ValueError('Invalid original C/PRE')
            operator=unit_operator(TransposeMode(get_operator('super_resolution',device=device,scale_factor=4,noise_sigma=0.),'legacy_bicubic').to(device)).to(device)
            z=ZAPS(model,operator,img_size=256,**plan['config']);z.tau=state['tau'].to(device);z.zeta.copy_(state['zeta'].to(device));z.D.copy_(state['D'].to(device))
            y=torch.load(src['measurement']['path'],map_location='cpu',weights_only=True)['y'].to(device)
            with model_rng_guard(model.model,device),patch('torch.randn_like',side_effect=RuntimeError('Unexpected sampling draw')):
                fields,repeats=old.parent.polish_pair(z,y,c,evaluate_field)
            f=fields[0];outputs=dict(C=c.cpu(),F1=(f['clean']+f['q']).cpu(),F2=(f['clean']+2*f['q']).cpu())
            repeat_F2=(fields[1]['clean']+2*fields[1]['q']).cpu()
            checks=dict(clean_repeat=relative_error(f['clean'],fields[1]['clean']),q_repeat=relative_error(f['q'],fields[1]['q']),
                F1_repeat=relative_error(outputs['F1'],repeats[1]),F2_repeat=relative_error(outputs['F2'],repeat_F2),F1_core=relative_error(outputs['F1'],repeats[0]))
            if any(not math.isfinite(v) or v>1e-7 for v in checks.values()) or any(not torch.isfinite(x).all() for x in outputs.values()):raise RuntimeError('Repeat/core/finiteness gate failed')
            if not(torch.equal(z.tau.cpu(),state['tau']) and torch.equal(z.zeta.cpu(),state['zeta']) and torch.equal(z.D.cpu(),state['D'])):raise RuntimeError('Original parameters/grid changed')
            tmp=folder/'outputs.pt.tmp';torch.save(dict(outputs=outputs,repeat_F1=repeats[1],repeat_F2=repeat_F2,fields=dict(clean=f['clean'].cpu(),q=f['q'].cpu())),tmp)
            tmp.replace(folder/'outputs.pt');r['outputs_sha256']=digest(folder/'outputs.pt')
            gt=load_image_as_tensor(case['image']).cpu()
            def metrics(x):return dict(loss=float((y-operator.H(x.to(device))).double().square().mean()/(2*.05**2)),**compute_all_metrics(x,gt,lpips_net='alex'))
            branches={a:metrics(x) for a,x in outputs.items()};repeat=metrics(repeats[1]);repeat2=metrics(repeat_F2)
            drift={k:abs(branches['C'][k]-src['final_metrics'][k]) for k in ('psnr','ssim','lpips')}
            if any(drift[k]>v for k,v in (('psnr',.01),('ssim',.001),('lpips',.001))):raise RuntimeError('Historical C metric convention drift')
            floors={k:max(drift.get(k,0.),abs(repeat[k]-branches['F1'][k]),abs(repeat2[k]-branches['F2'][k])) for k in METRICS}
            residual=y-operator.H(f['clean']);action=operator.H(f['clean']+f['q'])-operator.H(f['clean'])
            predicted=float((residual.double()-2*action.double()).square().mean()/(2*.05**2));error=abs(predicted-branches['F2']['loss'])
            if error>1e-7+2e-5*abs(branches['F2']['loss']):raise RuntimeError('Affine loss gate failed')
            r.update(branches=branches,floors=floors)
        if not same_rng(before,rng_state(device)) or r['diagnostic_nfe']!=2 or precision.precision_state(torch)!=policy:raise RuntimeError('RNG/precision/budget gate failed')
        r['gates']=dict(repeat_core=dict(passed=True,errors=checks),historical_C=dict(passed=True,drift=drift),affine_loss=dict(passed=True,error=error),parameters_rng_budget=dict(passed=True))
        r['status']='complete';save();(folder/'run.sha256').write_text(digest(folder/'run.json')+'\n')
        print('A17 CASE:',case['key'],'F2-C PSNR=',branches['F2']['psnr']-branches['C']['psnr'],flush=True)
    except BaseException as error:r.update(status='failed',error=repr(error));save();raise
    finally:hook.remove()


def run(args):
    import fcntl,torch
    root=Path(args.root).resolve();plan=old.read(root/'plan.json')
    with (root/f'shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        todo=[c for i,c in enumerate(sorted(plan['cases'],key=lambda c:c['key'])) if i%2==args.shard and load_record(root,c) is None]
        if not todo:return
        old.parent.coupled.archive.recovery.require_space(root);torch.set_num_threads(2)
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(args.device),
            NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        if runtime!=plan['runtime']:raise ValueError('Runtime differs; stop before calls')
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))),precision.ieee(torch) as policy:
            if policy!=plan['ieee_policy']:raise ValueError('IEEE policy differs')
            model=old.parent.coupled.archive.recovery.load_model(plan,args.device)
            for c in todo:run_case(root,plan,c,model,args.device,policy,runtime)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=ROOT);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--a16-dir',default=old.ROOT);parser.add_argument('--a14-dir',default='/data0/lzy/ZAPS-results/diag_ieee_response_value/A14_20261020')
    action=parser.add_mutually_exclusive_group(required=True);action.add_argument('--prepare',action='store_true');action.add_argument('--shard',type=int,choices=(0,1))
    action.add_argument('--summary',action='store_true');action.add_argument('--archive',action='store_true');args=parser.parse_args();root=Path(args.root).resolve()
    if args.prepare:prepare(args);return
    plan=old.read(root/'plan.json')
    if digest(root/'plan.json')!=(root/'plan.sha256').read_text().strip():raise ValueError('Plan changed')
    for path,sha in plan['worker_identity_files'].items():
        if digest(path)!=sha:raise ValueError('Code/card changed')
    if args.shard is not None:run(args);return
    records=[]
    for c in plan['cases']:
        path=root/c['key']/'run.json'
        if path.exists():
            r=old.read(path)
            if r['status']=='complete':load_record(root,c)
            records.append(r)
    result=summarize(records,plan['prior_records']);WRITE(root/'summary.json',result);print('A17 Summary:',json.dumps(result),flush=True)
    if args.archive:
        from utils.archive_refinement_confirmation import export_and_commit
        bundle,commit=export_and_commit({'A17':str(root)});print('Archive:',bundle,'Results commit:',commit,flush=True)
    print('Records:',root,flush=True)


if __name__=='__main__':main()
