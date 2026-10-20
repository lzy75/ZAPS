"""A6 fixed-candidate fresh-image confirmation: 1800 method +432 diagnostic NFE."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS))
from utils import diag_fixed_position_continuation as prior
from utils import diag_guidance_replication as baseline
from utils import diag_coupled_refinement as coupled
from utils.diag_state_trajectory import digest
WRITE=coupled.WRITE
IDS=('00023','00024','00025')
SEEDS=(1000,2000)
CARD=PROJECTS/'experiment_cards/A6_fresh_image_confirmation_20261020.md'
METRICS=('psnr','ssim','lpips')


def all_hashes(value):
    if isinstance(value,dict):return set().union(*(all_hashes(v) for v in value.values()))
    if isinstance(value,list):return set().union(*(all_hashes(v) for v in value))
    return {value} if isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) else set()


def check_history(paths,cases):
    checked=[]
    for path in sorted(set(Path(p).resolve() for p in paths)):
        value=json.loads(path.read_text())
        ids=baseline.history_ids(value)|baseline.history_ids(str(path))
        ids.update(re.findall(r'(?<!\d)(\d{5})_s\d+',json.dumps(value)))
        if set(IDS)&ids or all_hashes(value)&{c['image_sha256'] for c in cases}:
            raise ValueError('Candidate already used in accessible history; fixed IDs retained: '+str(path))
        checked.append(dict(path=str(path),sha256=digest(path),image_ids=sorted(ids)))
    if not checked:raise ValueError('No readable history; cannot establish unused candidate images')
    return checked


def prior_sources(args):
    root,a5,old,parents,a3,a4=prior.bind(SimpleNamespace(source_plan_dir=args.source_plan_dir,a2c_dir=args.a2c_dir,
        a2d_dir=args.a2d_dir,a3_dir=args.a3_dir,a4_dir=args.a4_dir,run_dir=args.a5_dir,inspect=True))
    records=prior.completed(a5,old)
    if len(records)!=6 or prior.stored_summary(a5)!=json.loads((a5/'summary.json').read_text()):raise ValueError('Need complete SHA-compatible A5 source and summary')
    return root,a5,old,parents


def prepare(args):
    root,a5,old,parents=prior_sources(args)
    directory=Path(args.run_dir).resolve()
    data=Path(args.data_dir).resolve() if args.data_dir else Path(next(iter(parents.values()))['case']['image']).parent
    cases=[dict(key=f'{image}_s{seed}',image=str(data/(image+'.png')),image_id=image,sampling_seed=seed,
                measurement_seed=1000+int(image),image_sha256=digest(data/(image+'.png'))) for image in IDS for seed in SEEDS]
    if len({c['image_sha256'] for c in cases})!=3:raise ValueError('Candidate image bytes are duplicated')
    for source in (root,Path(args.a2c_dir).resolve(),Path(args.a2d_dir).resolve(),Path(old['a3_source']),Path(old['a4_source']),a5):
        if directory.is_relative_to(source) or source.is_relative_to(directory):raise ValueError('New run must be separate from all source runs')
    files=dict(old['identity_files'])
    paths=[a5/'plan.json',a5/'summary.json',CARD,Path(__file__),PROJECTS/'run_a6_20261020.sh',
           PROJECTS/'utils/test_diag_fresh_refinement_confirmation.py',PROJECTS/'utils/archive_refinement_confirmation.py']
    for key in parents:paths.extend(a5/key/name for name in ('run.json','run.sha256','final_outputs.pt'))
    files.update({str(p):digest(p) for p in paths})
    protocol=json.loads((root/'plan.json').read_text())
    stable=dict(experiment='fresh_refinement_confirmation_plan',cases=cases,config=protocol['config'],grid=protocol['grid'],
                checkpoint_sha256=protocol['checkpoint_sha256'],identity_files=files,
                sources=dict(A3=old['a3_source'],A4=old['a4_source'],A5=str(a5)),
                positions=[14,54],planned_method_nfe=1800,planned_diagnostic_nfe=432,experiment_card=CARD.read_text())
    if (directory/'plan.json').exists():
        plan=json.loads((directory/'plan.json').read_text())
        if any(plan.get(k)!=v for k,v in stable.items()):raise ValueError('Frozen A6 inputs/code/card changed; do not overwrite')
        if any(digest(item['path'])!=item['sha256'] for item in plan['history_audit']):raise ValueError('Predeclared historical evidence changed')
    else:
        if directory.exists():raise ValueError('Interrupted A6 prepare retained')
        roots=[PROJECTS/'results',PROJECTS/'server_records',Path('/data0/lzy/ZAPS-results')]
        paths=[p for source in roots if source.exists() for p in source.rglob('run.json') if not p.resolve().is_relative_to(directory)]
        paths.append(PROJECTS/'experiment_cards/evidence/STATE_INDICATOR_HISTORY_20261018_recomputed.json')
        plan=dict(stable,history_audit=check_history(paths,cases),history_scope='Accessible local server records only; fixed images retained on overlap')
        if not args.inspect:
            directory.mkdir(parents=True,exist_ok=False);WRITE(directory/'plan.json',plan)
    return directory,plan


def summarize(records):
    complete=[r for r in records if r['status']=='complete'];layers=[]
    for position in (14,54):
        images={r['case']['image_id'] for r in complete}
        count=sum(len([r for r in complete if r['case']['image_id']==image])==2 and
                  all(r['positions'][str(position)]['both_bridges_candidate'] for r in complete if r['case']['image_id']==image) for image in images)
        layers.append(dict(source_t=position,images_both_seeds_both_bridges=count,
            cases_both_bridges=sum(r['positions'][str(position)]['both_bridges_candidate'] for r in complete),
            classification=('FIXED_POSITION_CONFIRMED_IN_COHORT' if count>=2 else 'FIXED_POSITION_NOT_CONFIRMED') if len(complete)==6 else 'INCOMPLETE_PILOT',
            mean_delta={a+'_C':{k:statistics.mean(r['positions'][str(position)]['effects'][a+'_C']['delta'][k] for r in complete) if complete else None
                         for k in ('loss',*METRICS)} for a in ('F0','F1')}))
    return dict(completed_cases=len(complete),planned_cases=6,method_nfe=sum(r['method_nfe'] for r in records),
        diagnostic_nfe=sum(r['diagnostic_nfe'] for r in records),positions=layers,
        rows=[dict(case=r['case']['key'],baseline=r['baseline'],positions={t:dict(branches=p['branches'],bridge_candidates=p['bridge_candidates'],both_bridges_candidate=p['both_bridges_candidate']) for t,p in r['positions'].items()},
                   position_readout=r['position_comparison']['readout'],gates_passed=all(g['passed'] for g in r['gates'].values())) for r in complete],
        limits='Three prespecified images unused within audited accessible records; two seeds/two bridges are not independent image samples. Fixed positions only, no state control or broad generalization claim.',
        cost_is_lower_bound_if_abruptly_interrupted=any(r['status']=='running' for r in records))


def completed(directory,plan):
    records={}
    for case in plan['cases']:
        path=directory/case['key']/'run.json'
        if not path.exists():
            if path.parent.exists():raise ValueError('Interrupted case retained: '+case['key'])
            continue
        r=json.loads(path.read_text())
        if (digest(path)!=(path.parent/'run.sha256').read_text().strip() or r['status']!='complete' or r['case']!=case or
            r['plan_sha256']!=digest(directory/'plan.json') or r['method_nfe']!=300 or r['diagnostic_nfe']!=72 or len(r['calls'])!=372 or
            not r['gates'] or not all(g['passed'] for g in r['gates'].values()) or any(digest(path.parent/name)!=sha for name,sha in r['artifacts_sha256'].items())):
            raise ValueError('Failed/interrupted/incompatible A6 case retained: '+case['key'])
        for p in r['positions'].values():
            effects={a+'_C':coupled.history.effect(p['branches'][a],r['baseline'],r['floors']) for a in ('F0','F1')}
            hits={a:coupled.candidate(effects[a+'_C']) for a in ('F0','F1')}
            if effects!=p['effects'] or hits!=p['bridge_candidates'] or all(hits.values())!=p['both_bridges_candidate']:raise ValueError('Saved H8 outcome differs')
        if prior.position_comparison(r['positions']['54']['branches'],r['positions']['14']['branches'],r['floors'],r['floors'])!=r['position_comparison']:raise ValueError('Saved A6 position arithmetic differs')
        records[case['key']]=r
    return records


def run_case(plan,case,directory,model,device):
    import torch
    from modules.zaps_algorithm import ZAPS
    from modules.main_single import load_image_as_tensor
    from modules.degradations import get_operator
    from utils.diag_ffhq_regression import TransposeMode
    from utils.diag_imagenet_gaussian_unit_domain import unit_operator
    from utils.diag_learning_rate_ablation import set_seed
    from utils.diag_zaps_sampler_parity import rng_state,same_rng
    from utils.diag_zaps_guidance_clipping import restore_sampling_rng,saved_cuda_selection
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_trace_audit import relative_error
    from utils.diag_state_trajectory import capture_clean
    from utils.diag_guidance_local_probe import evaluate_field
    from utils.diag_cached_guidance_geometry import vector_readouts
    from utils.metrics import compute_all_metrics
    directory.mkdir(exist_ok=False)
    r=dict(experiment='fresh_refinement_confirmation_case',status='running',case=case,plan_sha256=digest(directory.parent/'plan.json'),
           method_nfe=0,diagnostic_nfe=0,calls=[],gates={},experiment_card=CARD.read_text())
    started=time.perf_counter();context={};phase=['method']
    def save():r['seconds']=time.perf_counter()-started;WRITE(directory/'run.json',r)
    def count(_module,inputs):
        key='method_nfe' if phase[0]=='method' else 'diagnostic_nfe';r[key]+=1
        t=int(inputs[1].flatten()[0]);r['calls'].append(dict(phase=phase[0],t=t,**context))
        if r[key]>(300 if key=='method_nfe' else 72) or ('eval_t' in context and t!=context['eval_t']):raise RuntimeError('A6 time/call budget failed')
    save();hook=model.model.register_forward_pre_hook(count)
    try:
        external=rng_state(device)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            base=get_operator('super_resolution',device=device,scale_factor=4,noise_sigma=0.)
            operator=unit_operator(TransposeMode(base,'legacy_bicubic').to(device)).to(device)
            truth=load_image_as_tensor(case['image']).cpu()
            measurement=directory.parent/(case['image_id']+'_measurement.pt')
            if not measurement.exists():
                set_seed(case['measurement_seed'])
                with torch.no_grad():y=operator.H(truth.to(device))+.05*torch.randn(1,3,64,64,device=device)
                tmp=measurement.with_suffix('.pt.tmp');torch.save(dict(y=y.cpu(),image_sha256=case['image_sha256'],measurement_seed=case['measurement_seed']),tmp);tmp.replace(measurement)
            saved=torch.load(measurement,map_location='cpu',weights_only=True)
            if saved['image_sha256']!=case['image_sha256'] or saved['measurement_seed']!=case['measurement_seed']:raise ValueError('Shared observation identity differs')
            y=saved['y'].to(device)
            if tuple(y.shape)!=(1,3,64,64) or y.dtype!=torch.float32 or not torch.isfinite(y).all():raise ValueError('Invalid observation')
            r['measurement']=dict(path=str(measurement),sha256=digest(measurement))
            z=baseline.snapshot_class()(model,operator,img_size=256,**plan['config']);z.tau=torch.tensor(plan['grid'],device=device)
            set_seed(case['sampling_seed']);initial=torch.randn(1,3,256,256,device=device)
            for _ in range(290):torch.randn_like(initial)
            expected_rng=rng_state(device);set_seed(case['sampling_seed'])
            with model_rng_guard(model.model,device) as guarded:r['loss_history']=z.optimize(y,verbose=True,x0_gt=None,scheduler=None)
            if (r['method_nfe']!=300 or len(guarded)!=300 or z._last_nfe!=300 or len(r['loss_history'])!=10 or
                not all(math.isfinite(v) for v in r['loss_history']) or not torch.equal(z._last_opt_noise,initial) or
                [c['t'] for c in r['calls']]!=plan['grid'][::-1]*10 or not same_rng(expected_rng,rng_state(device)) or
                z.audit_state is None or any(p.grad is not None for p in model.model.parameters())):raise RuntimeError('Original training contract failed')
            state=z.audit_state;original=z._last_opt_x0.detach().cpu()
            torch.save(state,directory/'pre_last_state.pt');torch.save(original,directory/'last_opt.pt');del z
            z=ZAPS(model,operator,img_size=256,**plan['config']);z.tau=state['tau'].to(device)
            with torch.no_grad():z.zeta.copy_(state['zeta'].to(device));z.D.copy_(state['D'].to(device))
            phase[0]='diagnostic';captures=[];mapping=saved_cuda_selection(state,device);replays=[]
            with torch.no_grad(),model_rng_guard(model.model,device):
                for mode in ('observed','plain'):
                    context.clear();context.update(kind=mode);restore_sampling_rng(state,device,mapping['saved_sampling_index'])
                    if mode=='observed':
                        with capture_clean(z,captures):output,nfe,_=z.sample(y,init_noise=state['init_noise'].to(device))
                    else:output,nfe,_=z.sample(y,init_noise=state['init_noise'].to(device))
                    replays.append((output.cpu(),nfe,rng_state(device)))
                checks=dict(observed_original=relative_error(original,replays[0][0]),plain_original=relative_error(original,replays[1][0]),
                            observed_plain=relative_error(replays[1][0],replays[0][0]))
                if (len(captures)!=30 or [c['index'] for c in captures]!=list(range(29,-1,-1)) or
                    any(v[1]!=30 for v in replays) or not same_rng(replays[0][2],replays[1][2]) or
                    max(checks.values())>1e-6 or checks['observed_plain']>1e-7):raise RuntimeError('Observed/plain/archive parity failed')
                cache=dict(clean=torch.cat([c['clean'] for c in captures]),latent_inputs=torch.cat([c['x'] for c in captures]))
                rows=[dict(step=k,parameter_index=29-k,t=t,t_prev=plan['grid'][::-1][k+1] if k<29 else -1) for k,t in enumerate(plan['grid'][::-1])]
                cache['guidance']=torch.cat([coupled.frozen_guidance(z,y,cache['clean'][k:k+1].to(device),row['t'],row['parameter_index']).cpu() for k,row in enumerate(rows)])
                noises,replay_gate=coupled.replay_noises(state,cache,dict(rows=rows),z,y,device,original)
                alpha=z.dm.alphas_cumprod;logs=.5*((1-alpha.double()).log()-alpha.double().log());qhistory=[]
                for k,row in enumerate(rows):
                    u=float((logs[-1]-logs[row['t']])/(logs[-1]-logs[0]));row['noise_u']=u
                    q=cache['guidance'][k:k+1]
                    geo=vector_readouts(q,qhistory[-1] if qhistory else None,qhistory[-2] if len(qhistory)>1 else None,u,rows[k-1]['noise_u'] if k else None,rows[k-2]['noise_u'] if k>1 else None)
                    row.update({'guidance_'+key:value for key,value in geo.items()});qhistory.append(q)
                    residual=y-operator.H(cache['clean'][k:k+1].to(device));row['residual_noise_ratio']=float(residual.double().square().mean().sqrt()/.05)
                torch.save(cache,directory/'trajectory.pt');r['rows']=rows
                outputs={};baseline_fields=[]
                context.clear();context.update(kind='C',eval_t=0,parameter_index=0)
                for repeat in (0,1):
                    context['repeat']=repeat;f=evaluate_field(z,y,cache['latent_inputs'][29:30].to(device),0,0)
                    baseline_fields.append((f,prior.terminal_output(z,cache['latent_inputs'][29:30].to(device),f).cpu()))
                outputs['C']=baseline_fields[0][1]
                baseline_check=dict(clean=relative_error(cache['clean'][29:30],baseline_fields[0][0]['clean'].cpu()),q=relative_error(cache['guidance'][29:30],baseline_fields[0][0]['q'].cpu()),
                                    final=relative_error(original,outputs['C']),repeat=relative_error(outputs['C'],baseline_fields[1][1]))
                if max(baseline_check.values())>1e-6 or baseline_check['repeat']>1e-7:raise RuntimeError('Terminal C parity failed')
                null_checks=[];branch_inputs=[];case_index=[c['key'] for c in plan['cases']].index(case['key'])
                for t,k in ((14,28),(54,27)):
                    index=29-k;s=rows[k]['t_prev'];spec=coupled.interval_spec(alpha.cpu().numpy(),t,s)
                    if (t,spec['m'],s) not in ((14,3,0),(54,28,14)):raise RuntimeError('Declared interval differs')
                    x=cache['latent_inputs'][k:k+1].to(device);clean=cache['clean'][k:k+1].to(device);q=cache['guidance'][k:k+1].to(device)
                    coarse=cache['latent_inputs'][k+1:k+2].to(device)
                    for bridge in (0,1):
                        seed=coupled.bridge_seed(case_index,k,bridge);generator=torch.Generator(device='cpu').manual_seed(seed)
                        auxiliary=torch.randn(x.shape,generator=generator,dtype=torch.float32,device='cpu').to(device)
                        mid,null,_,_=coupled.frozen_bridge(x,clean,q,noises[k],auxiliary,spec,alpha)
                        err=relative_error(coarse,null);absolute=float((coarse.double()-null.double()).abs().max())
                        if not math.isfinite(err+absolute) or err>1e-6 or absolute>1e-5:raise RuntimeError('Frozen-field gate failed')
                        null_checks.append(dict(t=t,bridge=bridge,relative_error=err,max_abs=absolute))
                        context.clear();context.update(kind=f'T{t}_F{bridge}_mid',eval_t=spec['m'],parameter_index=index)
                        middle=evaluate_field(z,y,mid,spec['m'],index)
                        qm=coupled.frozen_guidance(z,y,middle['clean'],t,index)
                        fine=coupled.refined_endpoint(coarse,clean,q,middle['clean'],qm,spec)
                        if t==54:
                            def before_eval(time_,index_):context.update(kind=f'T54_F{bridge}_suffix',eval_t=time_,parameter_index=index_)
                            output,_,_=prior.continue_suffix(z,y,fine,noises[28],evaluate_field,before_eval)
                        else:
                            context.update(kind=f'T14_F{bridge}_suffix',eval_t=0,parameter_index=0)
                            end=evaluate_field(z,y,fine,0,0);output=prior.terminal_output(z,fine,end)
                        if not torch.isfinite(output).all():raise RuntimeError('Nonfinite final candidate')
                        outputs[f'T{t}_F{bridge}']=output.cpu();branch_inputs.append(dict(t=t,bridge=bridge,seed=seed,Z=noises[k].cpu(),auxiliary=auxiliary.cpu(),mid=mid.cpu(),fine=fine.cpu()))
                if not(torch.equal(z.tau.cpu(),state['tau']) and torch.equal(z.zeta.cpu(),state['zeta']) and torch.equal(z.D.cpu(),state['D'])):raise RuntimeError('PRE parameters/grid changed')
                torch.save(dict(outputs=outputs,branch_inputs=branch_inputs,suffix_Z=noises[28].cpu()),directory/'final_outputs.pt')
                # Truth was used to generate y; offline metrics begin only after all candidates are fixed.
                def physical(output):return float((y-operator.H(output.to(device))).double().square().mean()/(2*.05**2))
                metrics={arm:dict(loss=physical(output),**compute_all_metrics(output,truth,lpips_net='alex')) for arm,output in outputs.items()}
                original_metrics=compute_all_metrics(original,truth,lpips_net='alex');repeat_metrics=compute_all_metrics(baseline_fields[1][1],truth,lpips_net='alex')
                floors={key:max(abs(metrics['C'][key]-original_metrics[key]),abs(metrics['C'][key]-repeat_metrics[key])) for key in METRICS}
                floors['loss']=max(abs(metrics['C']['loss']-physical(original)),abs(metrics['C']['loss']-physical(baseline_fields[1][1])))
                if any(floors[k]>v for k,v in (('psnr',.01),('ssim',.001),('lpips',.001))):raise RuntimeError('Final metric convention/parity gate failed')
                positions={}
                for t in (14,54):
                    branches={a:metrics[f'T{t}_{a}'] for a in ('F0','F1')}
                    effects={a+'_C':coupled.history.effect(branches[a],metrics['C'],floors) for a in ('F0','F1')}
                    hits={a:coupled.candidate(effects[a+'_C']) for a in ('F0','F1')}
                    positions[str(t)]=dict(branches=branches,effects=effects,bridge_candidates=hits,both_bridges_candidate=all(hits.values()))
                r.update(baseline=metrics['C'],floors=floors,positions=positions,position_comparison=prior.position_comparison(positions['54']['branches'],positions['14']['branches'],floors,floors))
            if r['method_nfe']!=300 or r['diagnostic_nfe']!=72:raise RuntimeError('Final NFE contract failed')
        if not same_rng(external,rng_state(device)):raise RuntimeError('External RNG changed')
        if digest(measurement)!=r['measurement']['sha256'] or any(digest(p)!=sha for p,sha in plan['identity_files'].items()) or digest(case['image'])!=case['image_sha256']:raise RuntimeError('Source identity changed')
        r['gates']=dict(training=dict(passed=True,nfe=300,optimizer_updates=10),baseline_replay=dict(passed=True,checks=checks),
            cached_noise=replay_gate,terminal_baseline=dict(passed=True,checks=baseline_check),frozen_field=dict(passed=True,checks=null_checks),
            parameters_grid_rng_identity_budget=dict(passed=True,method_nfe=300,diagnostic_nfe=72))
        r['artifacts_sha256']={name:digest(directory/name) for name in ('pre_last_state.pt','last_opt.pt','trajectory.pt','final_outputs.pt')}
        r['status']='complete';save();(directory/'run.sha256').write_text(digest(directory/'run.json')+'\n')
        print('A6 CASE:',case['key'],'method/diagnostic=300/72',flush=True);return r
    except BaseException as error:r['status']='failed';r['error']=repr(error);save();raise
    finally:hook.remove()


def stored_summary(directory):return summarize([json.loads(p.read_text()) for p in sorted(directory.glob('*/run.json'))])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-plan-dir',default='results/diag_guidance_replication/A2a_20261018')
    for label,folder in (('a2c','diag_one_step_guidance_scale/A2c_20261019'),('a2d','diag_direction_refresh/A2d_20261019'),('a3','diag_coupled_refinement/A3_20261020'),('a4','diag_terminal_continuation/A4_20261020'),('a5','diag_fixed_position_continuation/A5_20261020')):
        parser.add_argument('--'+label+'-dir',default='/data0/lzy/ZAPS-results/'+folder)
    parser.add_argument('--run-dir',default='/data0/lzy/ZAPS-results/diag_fresh_refinement_confirmation/A6_20261020');parser.add_argument('--data-dir');parser.add_argument('--device',default='cuda')
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run',action='store_true');group.add_argument('--inspect',action='store_true');group.add_argument('--summary-only',action='store_true');group.add_argument('--archive-only',action='store_true')
    args=parser.parse_args();directory=Path(args.run_dir).resolve()
    if args.summary_only:
        plan=json.loads((directory/'plan.json').read_text());completed(directory,plan);print('A6 Summary:',json.dumps(stored_summary(directory)),'\nRecords:',directory);return
    import fcntl
    directory.parent.mkdir(parents=True,exist_ok=True)
    with (directory.parent/(directory.name+'.lock')).open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A6 already running') from None
        directory,plan=prepare(args)
        if args.inspect:print('A6 INSPECT PASS: fixed fresh IDs 00023/24/25, method/diagnostic budget=1800/432');return
        records=completed(directory,plan)
        try:
            todo=[c for c in plan['cases'] if c['key'] not in records]
            if todo and not args.archive_only:
                import torch
                coupled.archive.recovery.require_space(directory)
                with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                    model=coupled.archive.recovery.load_model(plan,args.device)
                    for case in todo:records[case['key']]=run_case(plan,case,directory/case['key'],model,args.device)
        finally:
            summary=stored_summary(directory);WRITE(directory/'summary.json',summary);print('A6 Summary:',json.dumps(summary),'\nRecords:',directory,flush=True)
        if len(records)==6:
            from utils.archive_refinement_confirmation import export_and_commit
            bundle,commit=export_and_commit({**plan['sources'],'A6':str(directory)})
            print('Stage archive:',bundle,'\nResults commit:',commit,'\nResults branch: results/refinement-confirmation-20261020',flush=True)


if __name__=='__main__':main()
