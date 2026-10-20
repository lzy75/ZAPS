"""A18: frozen F2 on six fresh images x two seeds; 3600 method +24 diagnostic NFE."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time

PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS))
from utils import diag_fixed_terminal_replication as parent
from utils import diag_guidance_replication as baseline
from utils import diag_clean_response_precision as precision
from utils import diag_fresh_refinement_confirmation as fresh
from utils.diag_state_trajectory import digest

WRITE=parent.WRITE
IDS=tuple(f'{i:05d}' for i in range(26,32));SEEDS=(1000,2000)
ROOT='/data0/lzy/ZAPS-results/diag_frozen_f2_confirmation/A18_20261020'
CARD=PROJECTS/'experiment_cards/A18_frozen_f2_fresh_images_20261020.md'
METRICS=parent.METRICS


def history_audit(paths,cases):
    ids={c['image_id'] for c in cases};hashes={c['image_sha256'] for c in cases};result=[]
    for path in sorted(set(Path(p).resolve() for p in paths)):
        value=parent.old.read(path)
        found=baseline.history_ids(value)|baseline.history_ids(str(path))|set(re.findall(r'(?<!\d)(\d{5})_s\d+',json.dumps(value)))
        if ids&found or hashes&fresh.all_hashes(value):
            raise ValueError('Fixed new-image cohort overlaps accessible history; no substitution: '+str(path))
        result.append(dict(path=str(path),sha256=digest(path)))
    if not result:raise ValueError('No accessible history to audit')
    return result


def shard_cases(cases,shard):
    # Both seeds of each image stay on the same worker; its observation is created once.
    images=sorted({c['image_id'] for c in cases})
    return [c for c in sorted(cases,key=lambda c:c['key']) if images.index(c['image_id'])%2==shard]


def prepare(args):
    source=Path(args.a17_dir).resolve();p17=parent.old.read(source/'plan.json')
    if digest(source/'plan.json')!=(source/'plan.sha256').read_text().strip():raise ValueError('A17 plan changed')
    records=[parent.load_record(source,c) for c in p17['cases']]
    result=parent.summarize(records,p17['prior_records']) if all(r is not None for r in records) else None
    if result is None or result!=parent.old.read(source/'summary.json') or result['classification']!='FIXED_F2_CROSS_COHORT_REPLICATION_SUPPORTED':
        raise ValueError('Need completed supported A17, no previous experiment started')
    directory=Path(args.root).resolve();data=Path(args.data_dir).resolve() if args.data_dir else Path(p17['cases'][0]['image']).parent
    cases=[dict(key=f'{i}_s{s}',image=str(data/(i+'.png')),image_id=i,sampling_seed=s,measurement_seed=1000+int(i),image_sha256=digest(data/(i+'.png'))) for i in IDS for s in SEEDS]
    if len({c['image_sha256'] for c in cases})!=len(IDS):raise ValueError('Duplicate candidate image bytes')
    identities={}
    def bind(path,sha=None):
        path=Path(path);value=digest(path)
        if sha is not None and value!=sha:raise ValueError('Source changed: '+str(path))
        identities[str(path)]=value
    for name in ('plan.json','plan.sha256','summary.json'):bind(source/name)
    for c in p17['cases']:
        for name in ('run.json','run.sha256'):bind(source/c['key']/name)
    for path,sha in p17['identity_files'].items():
        if Path(path).suffix=='.py' and Path(path).is_relative_to(PROJECTS):bind(path,sha)
    a13=Path(args.a13_dir).resolve()
    if directory.is_relative_to(source) or source.is_relative_to(directory) or directory.is_relative_to(a13):raise ValueError('New run overlaps source')
    bind(a13/'plan.json',(a13/'plan.sha256').read_text().strip())
    r13=precision.existing(a13,14)
    if r13 is None or r13['runtime']!=p17['runtime'] or r13['ieee_policy']!=p17['ieee_policy']:raise ValueError('Precision context differs')
    bind(a13/'t14/run.json');bind(a13/'t14/run.sha256')
    own=[Path(__file__),CARD,PROJECTS/'run_a18_20261020.sh',PROJECTS/'utils/test_diag_frozen_f2_confirmation.py',
         PROJECTS/'utils/diag_fixed_terminal_replication.py',PROJECTS/'utils/diag_guidance_replication.py',PROJECTS/'utils/diag_clean_response_precision.py']
    for path in own:bind(path)
    workers={str(p):identities[str(p)] for p in own}
    stable=dict(experiment='A18_frozen_f2_confirmation',cases=cases,config=p17['config'],grid=p17['grid'],checkpoint_sha256=p17['checkpoint_sha256'],
        runtime=p17['runtime'],native_policy=r13['native_policy'],ieee_policy=p17['ieee_policy'],identity_files=identities,worker_identity_files=workers,
        source_A17=str(source),coefficients=[1.,2.],user_mean_guards=parent.old.GUARDS,planned_method_nfe=3600,planned_diagnostic_nfe=24,
        original_final_mode='last_opt_pre_final_adam_update',scientific_extra_nfe_per_case=1)
    if directory.exists():
        plan=parent.old.read(directory/'plan.json')
        if any(plan.get(k)!=v for k,v in stable.items()):raise ValueError('Frozen A18 differs; refuse overwrite')
        if any(digest(v['path'])!=v['sha256'] for v in plan['history_audit']):raise ValueError('Bound history changed')
    else:
        roots=(PROJECTS/'results',PROJECTS/'server_records',Path('/data0/lzy/ZAPS-results'))
        paths=[p for root in roots if root.exists() for name in ('run.json','plan.json') for p in root.rglob(name) if not p.resolve().is_relative_to(directory)]
        plan=dict(stable,history_audit=history_audit(paths,cases),history_scope='Accessible local run/plan metadata only; IDs and hashes, fixed cohort, no replacement')
        directory.mkdir(parents=True,exist_ok=False);WRITE(directory/'plan.json',plan)
        (directory/'plan.sha256').write_text(digest(directory/'plan.json')+'\n');(directory/'experiment_card.md').write_text(CARD.read_text())
    print('A18 PREPARE PASS: 6 audited images x2 seeds; 3600 method +24 diagnostic calls; F1/F2 frozen.',flush=True)


def summarize(records):
    complete=[r for r in records if r['status']=='complete']
    comparisons={a+'_'+b:parent.contrast(records,a,b) for a,b in (('F1','C'),('F2','C'),('F2','F1'))}
    supported=len(complete)==12 and all(comparisons[k]['passed'] and comparisons[k]['images_psnr_better']>=4 for k in ('F2_C','F2_F1'))
    distributions={}
    for a in ('F1','F2'):
        distributions[a+'_C']={}
        for k in METRICS:
            values=[r['branches'][a][k]-r['branches']['C'][k] for r in complete]
            distributions[a+'_C'][k]=dict(mean=statistics.mean(values),median=statistics.median(values),min=min(values),max=max(values)) if values else None
    method=sum(r['method_nfe'] for r in records);diagnostic=sum(r['diagnostic_nfe'] for r in records)
    return dict(completed_cases=len(complete),planned_cases=12,image_clusters=len({r['case']['image_id'] for r in complete}),
        method_nfe=method,diagnostic_nfe=diagnostic,scientific_extra_nfe_per_case=1,deployment_method_nfe=dict(C=300,F1=301,F2=301),
        cumulative_queue_cost=dict(method_nfe=3600+method,diagnostic_nfe=2288+diagnostic,total=5888+method+diagnostic),
        classification='INCOMPLETE_PILOT' if len(complete)!=12 else 'FROZEN_F2_SUPPORTED_ON_FRESH_COHORT' if supported else 'FROZEN_F2_NOT_SUPPORTED_ON_FRESH_COHORT',
        comparisons=comparisons,distributions=distributions,
        rows=[dict(case=r['case']['key'],branches=r['branches'],seconds=r['seconds'],
            deltas_vs_C={a:{k:r['branches'][a][k]-r['branches']['C'][k] for k in METRICS} for a in ('F1','F2')},
            gates_passed=all(g['passed'] for g in r['gates'].values())) for r in complete],
        per_case_lpips_over_mean_tolerance=[r['case']['key'] for r in complete if r['branches']['F2']['lpips']-r['branches']['C']['lpips']>.005],
        cost_is_lower_bound_if_abruptly_interrupted=any(r['status']=='running' for r in records),
        limits='Six prespecified images unused in audited accessible records, two dependent seeds each. Frozen terminal correction only; no state-control, scheduling, cross-task or author-parity claim. Average tolerance is not a per-image guarantee.')


def load_record(root,case):
    path=root/case['key']/'run.json'
    if not path.exists():
        if path.parent.exists():raise ValueError('Interrupted case retained: '+case['key'])
        return None
    r=parent.old.read(path)
    if (digest(path)!=(path.parent/'run.sha256').read_text().strip() or r['status']!='complete' or r['case']!=case or r['plan_sha256']!=digest(root/'plan.json') or
        r['method_nfe']!=300 or r['diagnostic_nfe']!=2 or len(r['calls'])!=302 or not r['gates'] or not all(g['passed'] for g in r['gates'].values()) or
        any(digest(path.parent/n)!=sha for n,sha in r['artifacts_sha256'].items()) or digest(r['measurement']['path'])!=r['measurement']['sha256']):
        raise ValueError('Invalid/interrupted A18 retained: '+case['key'])
    return r


def training_contract(record,z,guarded,initial,expected_rng,actual_rng,same_rng,plan):
    if (record['method_nfe']!=300 or len(guarded)!=300 or z._last_nfe!=300 or len(record['loss_history'])!=10 or z.audit_calls!=10 or
        not all(math.isfinite(v) for v in record['loss_history']) or not initial.equal(z._last_opt_noise) or
        [v['t'] for v in record['calls']]!=plan['grid'][::-1]*10 or not same_rng(expected_rng,actual_rng) or z.audit_state is None or
        any(p.grad is not None for p in z.dm.model.parameters())):
        raise RuntimeError('Original training/NFE/time/RNG contract failed')


def run_case(root,plan,case,model,device):
    import torch
    from unittest.mock import patch
    from modules.zaps_algorithm import ZAPS
    from modules.main_single import load_image_as_tensor
    from modules.degradations import get_operator
    from utils.diag_ffhq_regression import TransposeMode
    from utils.diag_imagenet_gaussian_unit_domain import unit_operator
    from utils.diag_learning_rate_ablation import set_seed
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state,same_rng
    from utils.diag_zaps_trace_audit import relative_error
    from utils.diag_guidance_local_probe import evaluate_field
    from utils.metrics import compute_all_metrics
    folder=root/case['key'];folder.mkdir(exist_ok=False);started=time.perf_counter();phase=['method']
    r=dict(status='running',case=case,plan_sha256=digest(root/'plan.json'),method_nfe=0,diagnostic_nfe=0,calls=[],gates={},runtime=plan['runtime'],native_policy=plan['native_policy'],ieee_policy=plan['ieee_policy'])
    def save():r['seconds']=time.perf_counter()-started;WRITE(folder/'run.json',r)
    def count(_module,inputs):
        key=phase[0]+'_nfe';r[key]+=1;r['calls'].append(dict(phase=phase[0],t=int(inputs[1].flatten()[0])));save()
        if r[key]>(300 if phase[0]=='method' else 2) or (phase[0]=='diagnostic' and r['calls'][-1]['t']!=0):raise RuntimeError('Call/time budget exceeded')
    save();hook=model.model.register_forward_pre_hook(count)
    try:
        if digest(case['image'])!=case['image_sha256']:raise ValueError('Fixed image changed')
        before=rng_state(device)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            if precision.precision_state(torch)!=plan['native_policy']:raise ValueError('Training native policy differs')
            operator=unit_operator(TransposeMode(get_operator('super_resolution',device=device,scale_factor=4,noise_sigma=0.),'legacy_bicubic').to(device)).to(device)
            truth=load_image_as_tensor(case['image']).cpu();measurement=root/(case['image_id']+'_measurement.pt')
            if not measurement.exists():
                set_seed(case['measurement_seed'])
                with torch.no_grad():y=operator.H(truth.to(device))+.05*torch.randn(1,3,64,64,device=device)
                tmp=measurement.with_suffix('.pt.tmp');torch.save(dict(y=y.cpu(),image_sha256=case['image_sha256'],measurement_seed=case['measurement_seed']),tmp);tmp.replace(measurement)
            saved=torch.load(measurement,map_location='cpu',weights_only=True)
            if saved['image_sha256']!=case['image_sha256'] or saved['measurement_seed']!=case['measurement_seed']:raise ValueError('Shared observation identity differs')
            y=saved['y'].to(device)
            if y.shape!=(1,3,64,64) or y.dtype!=torch.float32 or not torch.isfinite(y).all():raise ValueError('Invalid observation')
            r['measurement']=dict(path=str(measurement),sha256=digest(measurement))
            z=baseline.snapshot_class()(model,operator,img_size=256,**plan['config']);z.tau=torch.tensor(plan['grid'],device=device)
            set_seed(case['sampling_seed']);initial=torch.randn(1,3,256,256,device=device)
            for _ in range(290):torch.randn_like(initial)
            expected=rng_state(device);set_seed(case['sampling_seed'])
            with model_rng_guard(model.model,device) as guarded:r['loss_history']=z.optimize(y,verbose=True,x0_gt=None,scheduler=None)
            training_contract(r,z,guarded,initial,expected,rng_state(device),same_rng,plan)
            state=z.audit_state;c=z._last_opt_x0.detach().cpu()
            if state['tau'].tolist()!=plan['grid'] or c.shape!=(1,3,256,256) or c.dtype!=torch.float32 or not torch.isfinite(c).all():raise ValueError('Invalid last_opt/PRE')
            torch.save(state,folder/'pre_last_state.pt');torch.save(c,folder/'last_opt.pt');del z
            z=ZAPS(model,operator,img_size=256,**plan['config']);z.tau=state['tau'].to(device)
            with torch.no_grad():z.zeta.copy_(state['zeta'].to(device));z.D.copy_(state['D'].to(device))
            phase[0]='diagnostic'
            with precision.ieee(torch) as policy,torch.no_grad():
                if policy!=plan['ieee_policy']:raise ValueError('Extra-score IEEE policy differs')
                with model_rng_guard(model.model,device),patch('torch.randn_like',side_effect=RuntimeError('Unexpected terminal draw')):
                    fields,repeats=parent.old.parent.polish_pair(z,y,c.to(device),evaluate_field)
                f=fields[0];repeat2=(fields[1]['clean']+2*fields[1]['q']).cpu()
                outputs=dict(C=c,F1=(f['clean']+f['q']).cpu(),F2=(f['clean']+2*f['q']).cpu())
                errors=dict(clean_repeat=relative_error(f['clean'],fields[1]['clean']),q_repeat=relative_error(f['q'],fields[1]['q']),
                    F1_repeat=relative_error(outputs['F1'],repeats[1]),F2_repeat=relative_error(outputs['F2'],repeat2),F1_core=relative_error(outputs['F1'],repeats[0]))
                if any(not math.isfinite(v) or v>1e-7 for v in errors.values()) or any(not torch.isfinite(v).all() for v in outputs.values()):raise RuntimeError('Extra-score repeat/core gate failed')
                if not(torch.equal(z.tau.cpu(),state['tau']) and torch.equal(z.zeta.cpu(),state['zeta']) and torch.equal(z.D.cpu(),state['D'])):raise RuntimeError('PRE parameters changed')
                tmp=folder/'outputs.pt.tmp';torch.save(dict(outputs=outputs,repeat_F1=repeats[1],repeat_F2=repeat2,fields=dict(clean=f['clean'].cpu(),q=f['q'].cpu())),tmp);tmp.replace(folder/'outputs.pt')
                # GT has generated y, but no quality metric or output selection occurs until outputs are fixed.
                def metrics(x):return dict(loss=float((y-operator.H(x.to(device))).double().square().mean()/(2*.05**2)),**compute_all_metrics(x,truth,lpips_net='alex'))
                branches={a:metrics(x) for a,x in outputs.items()};m1=metrics(repeats[1]);m2=metrics(repeat2)
                floors={k:max(abs(m1[k]-branches['F1'][k]),abs(m2[k]-branches['F2'][k])) for k in METRICS}
                residual=y-operator.H(f['clean']);action=operator.H(f['clean']+f['q'])-operator.H(f['clean'])
                predicted=float((residual.double()-2*action.double()).square().mean()/(2*.05**2));affine=abs(predicted-branches['F2']['loss'])
                if affine>1e-7+2e-5*abs(branches['F2']['loss']):raise RuntimeError('Affine loss gate failed')
                r.update(branches=branches,floors=floors)
            if precision.precision_state(torch)!=plan['native_policy']:raise RuntimeError('Precision policy not restored')
        if not same_rng(before,rng_state(device)) or r['method_nfe']!=300 or r['diagnostic_nfe']!=2:raise RuntimeError('External RNG/budget failed')
        if digest(case['image'])!=case['image_sha256'] or digest(measurement)!=r['measurement']['sha256']:raise ValueError('Source changed during case')
        r['gates']=dict(training=dict(passed=True,nfe=300,epochs=10,snapshot_calls=10),repeat_core=dict(passed=True,errors=errors),
            affine_loss=dict(passed=True,error=affine),parameters_rng_policy_budget=dict(passed=True))
        r['artifacts_sha256']={n:digest(folder/n) for n in ('pre_last_state.pt','last_opt.pt','outputs.pt')}
        r['status']='complete';save();(folder/'run.sha256').write_text(digest(folder/'run.json')+'\n')
        print('A18 CASE:',case['key'],'method/diagnostic=300/2; F2-C PSNR=',branches['F2']['psnr']-branches['C']['psnr'],flush=True)
    except BaseException as error:r.update(status='failed',error=repr(error));save();raise
    finally:hook.remove()


def run(args):
    import fcntl,torch
    root=Path(args.root).resolve();plan=parent.old.read(root/'plan.json')
    with (root/f'shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        todo=[c for c in shard_cases(plan['cases'],args.shard) if load_record(root,c) is None]
        if not todo:return
        parent.old.parent.coupled.archive.recovery.require_space(root);torch.set_num_threads(2)
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(args.device),
            NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        if runtime!=plan['runtime'] or precision.precision_state(torch)!=plan['native_policy']:raise ValueError('Training runtime/native policy differs; stop before calls')
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            model=parent.old.parent.coupled.archive.recovery.load_model(plan,args.device)
            for case in todo:run_case(root,plan,case,model,args.device)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=ROOT);p.add_argument('--device',default='cuda:0');p.add_argument('--data-dir')
    p.add_argument('--a17-dir',default=parent.ROOT);p.add_argument('--a13-dir',default=precision.ROOT)
    a=p.add_mutually_exclusive_group(required=True);a.add_argument('--prepare',action='store_true');a.add_argument('--shard',type=int,choices=(0,1))
    a.add_argument('--summary',action='store_true');a.add_argument('--archive',action='store_true');args=p.parse_args();root=Path(args.root).resolve()
    if args.prepare:prepare(args);return
    plan=parent.old.read(root/'plan.json')
    if digest(root/'plan.json')!=(root/'plan.sha256').read_text().strip():raise ValueError('Plan changed')
    for path,sha in plan['worker_identity_files'].items():
        if digest(path)!=sha:raise ValueError('Code/card changed')
    if args.shard is not None:run(args);return
    records=[]
    for case in plan['cases']:
        path=root/case['key']/'run.json'
        if path.exists():
            r=parent.old.read(path)
            if r['status']=='complete':load_record(root,case)
            records.append(r)
    result=summarize(records);WRITE(root/'summary.json',result);print('A18 Summary:',json.dumps(result),flush=True)
    if args.archive:
        from utils.archive_refinement_confirmation import export_and_commit
        bundle,commit=export_and_commit({'A18':str(root)});print('Archive:',bundle,'Results commit:',commit,flush=True)
    print('Records:',root,flush=True)


if __name__=='__main__':main()
