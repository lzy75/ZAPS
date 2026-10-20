"""A8 conditional endpoint-component ablation, fixed t14, 60 diagnostic / zero method NFE."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS))
from utils import diag_fresh_refinement_confirmation as fresh
from utils import diag_coupled_refinement as coupled
from utils import diag_terminal_continuation as terminal
from utils.diag_state_trajectory import digest
WRITE=coupled.WRITE;METRICS=('psnr','ssim','lpips')
CARD=PROJECTS/'experiment_cards/A8_endpoint_components_20261020.md'


def components(coarse,clean,q,clean_mid,q_mid,spec):
    if any(v.shape!=coarse.shape for v in (clean,q,clean_mid,q_mid)):raise ValueError('Component tensor shapes differ')
    dp=spec['second']['b']*(clean_mid.double()-clean.double())
    dl=spec['local_guidance_factor']*(q_mid.double()-q.double())
    return dict(F=(coarse.double()+(dp+dl)).to(coarse.dtype),P=(coarse.double()+dp).to(coarse.dtype),L=(coarse.double()+dl).to(coarse.dtype)),dp,dl


def retention(component,full):
    if not coupled.candidate(component):return False
    for key,sign in (('psnr',1),('ssim',1),('lpips',-1),('loss',-1)):
        gf=sign*full['delta'][key];ga=sign*component['delta'][key];floor=max(full['numerical_limits'][key],component['numerical_limits'][key])
        if gf<=floor or ga<.8*gf-floor:return False
    return True


def summarize(records):
    complete=[r for r in records if r['status']=='complete'];images={r['case']['image_id'] for r in complete};arms={}
    for arm in ('F','P','L'):
        hits=sum(len([r for r in complete if r['case']['image_id']==image])==2 and all(all(r['joint_candidates'][arm].values()) for r in complete if r['case']['image_id']==image) for image in images)
        retained=None if arm=='F' else sum(len([r for r in complete if r['case']['image_id']==image])==2 and all(all(r['retention_80'][arm].values()) for r in complete if r['case']['image_id']==image) for image in images)
        arms[arm]=dict(images_both_seeds_both_bridges_joint=hits,images_retaining_80_percent_joint_gain=retained,
            mean_delta_vs_C={k:statistics.mean(r['effects'][f'{arm}{b}_C']['delta'][k] for r in complete for b in (0,1)) if complete else None for k in ('loss',*METRICS)},
            mean_delta_vs_F=None if arm=='F' else {k:statistics.mean(r['effects'][f'{arm}{b}_F{b}']['delta'][k] for r in complete for b in (0,1)) if complete else None for k in ('loss',*METRICS)})
    if len(complete)!=6:classification='INCOMPLETE_PILOT'
    else:
        p,l=(arms[a]['images_retaining_80_percent_joint_gain']>=2 for a in ('P','L'))
        classification='BOTH_COMPONENTS_SUFFICIENT_CANDIDATE' if p and l else 'PRIOR_COMPONENT_SUFFICIENT_CANDIDATE' if p else 'MEASUREMENT_COMPONENT_SUFFICIENT_CANDIDATE' if l else 'NO_SINGLE_COMPONENT_SUFFICIENCY_ESTABLISHED'
    return dict(completed_cases=len(complete),planned_cases=6,method_nfe=sum(r['method_nfe'] for r in records),diagnostic_nfe=sum(r['diagnostic_nfe'] for r in records),classification=classification,arms=arms,
        rows=[dict(case=r['case']['key'],baseline=r['branches']['C'],branches={a:{str(b):r['branches'][a+str(b)] for b in (0,1)} for a in ('F','P','L')},joint_candidates=r['joint_candidates'],retention_80=r['retention_80'],gates_passed=all(g['passed'] for g in r['gates'].values())) for r in complete],
        limits='Same midpoint conditional endpoint ablation on three reused A6 images. Quality effects need not add. No global causal attribution, state-control or generalization claim.',
        cost_is_lower_bound_if_abruptly_interrupted=any(r['status']=='running' for r in records))


def bind(args):
    source,old=fresh.prepare(SimpleNamespace(source_plan_dir=args.source_plan_dir,a2c_dir=args.a2c_dir,a2d_dir=args.a2d_dir,
        a3_dir=args.a3_dir,a4_dir=args.a4_dir,a5_dir=args.a5_dir,run_dir=args.a6_dir,data_dir=None,inspect=True))
    active=fresh.completed(source,old)
    if len(active)!=6 or fresh.stored_summary(source)!=json.loads((source/'summary.json').read_text()):raise ValueError('Need complete compatible A6 records; do not rerun')
    a7=Path(args.a7_dir).resolve();r7=json.loads((a7/'run.json').read_text())
    if digest(a7/'run.json')!=(a7/'run.sha256').read_text().strip() or r7['status']!='complete' or r7['method_nfe']!=0 or r7['diagnostic_nfe']!=0:
        raise ValueError('Need completed SHA-bound A7 record')
    if any(p['classification']!='DIRECTION_INCREMENT_NOT_SUPPORTED_IN_THIS_TRANSFER_TEST' for p in r7['positions']):raise ValueError('A7 outcome differs from declared A8 context')
    paths=[source/'plan.json',source/'summary.json',a7/'plan.json',a7/'run.json',a7/'run.sha256',CARD,Path(__file__),
           PROJECTS/'run_a8_20261020.sh',PROJECTS/'utils/test_diag_endpoint_components.py']
    for key,r in active.items():
        paths.extend([source/key/'run.json',source/key/'run.sha256',Path(r['measurement']['path'])])
        if digest(r['measurement']['path'])!=r['measurement']['sha256']:raise ValueError('A6 measurement changed')
        paths.extend(source/key/name for name in r['artifacts_sha256'])
    plan=dict(experiment='endpoint_component_plan',source=str(source),cases=old['cases'],config=old['config'],grid=old['grid'],checkpoint_sha256=old['checkpoint_sha256'],
        identity_files={**old['identity_files'],**{str(p):digest(p) for p in paths}},source_t=14,midpoint_t=3,next_t=0,source_index=1,terminal_index=0,
        arms=['C','F0','F1','P0','P1','L0','L1'],planned_method_nfe=0,planned_diagnostic_nfe=60,retention_fraction=.8)
    directory=Path(args.run_dir).resolve()
    if any(directory.is_relative_to(p) or p.is_relative_to(directory) for p in (source,a7)):raise ValueError('Analysis directory overlaps source')
    if directory.exists():
        if json.loads((directory/'plan.json').read_text())!=plan:raise ValueError('Frozen A8 plan differs; do not overwrite')
    else:directory.mkdir(parents=True,exist_ok=False);WRITE(directory/'plan.json',plan)
    return directory,plan,active


def completed(directory,plan):
    result={}
    for case in plan['cases']:
        path=directory/case['key']/'run.json'
        if not path.exists():
            if path.parent.exists():raise ValueError('Interrupted case retained')
            continue
        r=json.loads(path.read_text())
        if (digest(path)!=(path.parent/'run.sha256').read_text().strip() or r['status']!='complete' or r['case']!=case or r['plan_sha256']!=digest(directory/'plan.json') or
            r['method_nfe']!=0 or r['diagnostic_nfe']!=10 or len(r['calls'])!=10 or not r['gates'] or not all(g['passed'] for g in r['gates'].values()) or
            digest(path.parent/'outputs.pt')!=r['output_sha256']):raise ValueError('Existing failed/interrupted/incompatible A8 retained')
        for arm in ('F','P','L'):
            for b in (0,1):
                label=arm+str(b);e=coupled.history.effect(r['branches'][label],r['branches']['C'],r['floors'])
                if e!=r['effects'][label+'_C'] or coupled.candidate(e)!=r['joint_candidates'][arm][str(b)]:raise ValueError('Saved component outcome differs')
                if arm!='F':
                    if coupled.history.effect(r['branches'][label],r['branches']['F'+str(b)],r['floors'])!=r['effects'][label+'_F'+str(b)] or retention(e,r['effects']['F'+str(b)+'_C'])!=r['retention_80'][arm][str(b)]:raise ValueError('Saved component comparison differs')
        result[case['key']]=r
    return result


def run_case(plan,parent,directory,model,device):
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
    from utils.diag_cached_guidance_geometry import rms,unit_direction
    from utils.metrics import compute_all_metrics
    case=parent['case'];source=Path(plan['source'])/case['key'];directory.mkdir(exist_ok=False)
    (directory/'experiment_card.md').write_text(CARD.read_text())
    r=dict(experiment='endpoint_component_case',status='running',case=case,plan_sha256=digest(directory.parent/'plan.json'),method_nfe=0,diagnostic_nfe=0,calls=[],gates={},
           runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,device=str(device)),config=plan['config'],grid=plan['grid'])
    started=time.perf_counter();context={}
    def save():r['seconds']=time.perf_counter()-started;WRITE(directory/'run.json',r)
    def count(_module,inputs):
        r['diagnostic_nfe']+=1;t=int(inputs[1].flatten()[0]);r['calls'].append(dict(context,t=t))
        if r['diagnostic_nfe']>10 or t!=context['eval_t']:raise RuntimeError('A8 time/budget gate failed')
    save();hook=model.model.register_forward_pre_hook(count)
    try:
        external=rng_state(device)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))),torch.no_grad():
            state=torch.load(source/'pre_last_state.pt',map_location='cpu',weights_only=True)
            cache=torch.load(source/'trajectory.pt',map_location='cpu',weights_only=True)
            original=torch.load(source/'last_opt.pt',map_location='cpu',weights_only=True)
            archive=torch.load(source/'final_outputs.pt',map_location='cpu',weights_only=True)
            points=sorted([p for p in archive['branch_inputs'] if p['t']==14],key=lambda p:p['bridge'])
            if [p['bridge'] for p in points]!=[0,1] or state['tau'].tolist()!=plan['grid'] or int(state['tau'][1])!=14:raise ValueError('Original interval/bridges differ')
            base=get_operator('super_resolution',device=device,scale_factor=4,noise_sigma=0.)
            operator=unit_operator(TransposeMode(base,'legacy_bicubic').to(device)).to(device)
            z=ZAPS(model,operator,img_size=256,**plan['config']);z.tau=state['tau'].to(device)
            z.zeta.copy_(state['zeta'].to(device));z.D.copy_(state['D'].to(device))
            y=torch.load(parent['measurement']['path'],map_location='cpu',weights_only=True)['y'].to(device)
            coarse=cache['latent_inputs'][29:30].to(device);clean=cache['clean'][28:29].to(device);q=cache['guidance'][28:29].to(device)
            spec=coupled.interval_spec(z.dm.alphas_cumprod.cpu().numpy(),14,0)
            if spec['m']!=3:raise ValueError('Actual midpoint differs')
            outputs={};repeats=[];full_checks=[];geometries=[];inputs_saved=[]
            with model_rng_guard(model.model,device),patch('torch.randn_like',side_effect=RuntimeError('Unexpected new noise draw')):
                for repeat in (0,1):
                    context.clear();context.update(arm='C',repeat=repeat,eval_t=0,parameter_index=0)
                    f=evaluate_field(z,y,coarse,0,0);out=terminal.terminal_output(z,coarse,f).cpu();repeats.append((f,out))
                outputs['C']=repeats[0][1]
                for point in points:
                    b=point['bridge'];mid=point['mid'].to(device)
                    if tuple(mid.shape)!=(1,3,256,256) or mid.dtype!=torch.float32 or not torch.isfinite(mid).all():raise ValueError('Invalid cached midpoint')
                    context.clear();context.update(arm=f'mid{b}',eval_t=3,parameter_index=1)
                    middle=evaluate_field(z,y,mid,3,1);qm=coupled.frozen_guidance(z,y,middle['clean'],14,1)
                    arms,dp,dl=components(coarse,clean,q,middle['clean'],qm,spec)
                    inp=point['fine'].to(device);err=relative_error(inp,arms['F']);absolute=float((inp.double()-arms['F'].double()).abs().max())
                    if not math.isfinite(err+absolute) or err>1e-6 or absolute>1e-5:raise RuntimeError('Recomputed full A6 fine input differs')
                    np_,nl=unit_direction(dp),unit_direction(dl)
                    geometries.append(dict(bridge=b,prior_rms=rms(dp),measurement_rms=rms(dl),
                        component_cosine=None if np_ is None or nl is None else float((np_*nl).sum()),total_rms=rms(dp+dl)))
                    for arm in ('F','P','L'):
                        context.update(arm=arm+str(b),eval_t=0,parameter_index=0)
                        f=evaluate_field(z,y,arms[arm],0,0);out=terminal.terminal_output(z,arms[arm],f)
                        if not torch.isfinite(out).all():raise RuntimeError('Nonfinite ablation output')
                        outputs[arm+str(b)]=out.cpu()
                    saved=archive['outputs'][f'T14_F{b}'];err_out=relative_error(saved,outputs[f'F{b}']);abs_out=float((saved.double()-outputs[f'F{b}'].double()).abs().max())
                    if not math.isfinite(err_out+abs_out) or err_out>1e-6 or abs_out>1e-5:raise RuntimeError('Original full final A6 output differs')
                    full_checks.append(dict(bridge=b,input_error=err,input_max_abs=absolute,output_error=err_out,output_max_abs=abs_out))
                    inputs_saved.append(dict(bridge=b,**{a:v.cpu() for a,v in arms.items()},prior_delta=dp.cpu(),measurement_delta=dl.cpu()))
            baseline_check=dict(clean=relative_error(cache['clean'][29:30],repeats[0][0]['clean'].cpu()),q=relative_error(cache['guidance'][29:30],repeats[0][0]['q'].cpu()),
                final=relative_error(original,outputs['C']),repeat=relative_error(outputs['C'],repeats[1][1]))
            if not all(math.isfinite(v) for v in baseline_check.values()) or max(baseline_check.values())>1e-6 or baseline_check['repeat']>1e-7:raise RuntimeError('C terminal parity failed')
            if not(torch.equal(z.tau.cpu(),state['tau']) and torch.equal(z.zeta.cpu(),state['zeta']) and torch.equal(z.D.cpu(),state['D'])):raise RuntimeError('Frozen parameters changed')
            tmp=directory/'outputs.pt.tmp';torch.save(dict(outputs=outputs,inputs=inputs_saved),tmp);tmp.replace(directory/'outputs.pt');r['output_sha256']=digest(directory/'outputs.pt')
            # All arm states and neural calls are fixed before GT metrics.
            gt=load_image_as_tensor(case['image']).cpu()
            def physical(out):return float((y-operator.H(out.to(device))).double().square().mean()/(2*.05**2))
            branches={a:dict(loss=physical(out),**compute_all_metrics(out,gt,lpips_net='alex')) for a,out in outputs.items()}
            repeat_metrics=compute_all_metrics(repeats[1][1],gt,lpips_net='alex')
            references={'C':parent['baseline'],**{f'F{b}':parent['positions']['14']['branches'][f'F{b}'] for b in (0,1)}}
            drifts={a:{k:abs(branches[a][k]-ref[k]) for k in ('loss',*METRICS)} for a,ref in references.items()}
            for a,d in drifts.items():
                if any(d[k]>v for k,v in (('psnr',.01),('ssim',.001),('lpips',.001),('loss',1e-7+2e-5*abs(references[a]['loss'])))):raise RuntimeError('A6 original metric convention differs')
            floors={k:max(drifts[a][k] for a in references) for k in METRICS}
            for k in METRICS:floors[k]=max(floors[k],abs(repeat_metrics[k]-branches['C'][k]))
            floors['loss']=max(drifts[a]['loss'] for a in references);floors['loss']=max(floors['loss'],abs(physical(repeats[1][1])-branches['C']['loss']))
            effects={};hits={a:{} for a in ('F','P','L')};retained={a:{} for a in ('P','L')}
            for b in (0,1):
                for arm in ('F','P','L'):
                    label=arm+str(b);e=coupled.history.effect(branches[label],branches['C'],floors);effects[label+'_C']=e;hits[arm][str(b)]=coupled.candidate(e)
                for arm in ('P','L'):
                    label=arm+str(b);effects[label+'_F'+str(b)]=coupled.history.effect(branches[label],branches['F'+str(b)],floors)
                    retained[arm][str(b)]=retention(effects[label+'_C'],effects['F'+str(b)+'_C'])
            r.update(branches=branches,floors=floors,effects=effects,joint_candidates=hits,retention_80=retained,component_geometry=geometries)
        if not same_rng(external,rng_state(device)) or r['diagnostic_nfe']!=10:raise RuntimeError('External RNG/budget failed')
        if any(digest(p)!=sha for p,sha in plan['identity_files'].items()):raise RuntimeError('Source/implementation changed')
        r['gates']=dict(baseline=dict(passed=True,checks=baseline_check),full_A6_reproduction=dict(passed=True,checks=full_checks,metric_drifts=drifts),
                       parameters_grid_rng_no_draw=dict(passed=True),identity_budget=dict(passed=True,method_nfe=0,diagnostic_nfe=10))
        r['status']='complete';save();(directory/'run.sha256').write_text(digest(directory/'run.json')+'\n');return r
    except BaseException as error:r['status']='failed';r['error']=repr(error);save();raise
    finally:hook.remove()


def stored_summary(directory):return summarize([json.loads(p.read_text()) for p in sorted(directory.glob('*/run.json'))])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-plan-dir',default='results/diag_guidance_replication/A2a_20261018')
    for label,folder in (('a2c','diag_one_step_guidance_scale/A2c_20261019'),('a2d','diag_direction_refresh/A2d_20261019'),('a3','diag_coupled_refinement/A3_20261020'),('a4','diag_terminal_continuation/A4_20261020'),('a5','diag_fixed_position_continuation/A5_20261020'),('a6','diag_fresh_refinement_confirmation/A6_20261020'),('a7','diag_direction_value/A7_20261020')):
        parser.add_argument('--'+label+'-dir',default='/data0/lzy/ZAPS-results/'+folder)
    parser.add_argument('--run-dir',default='/data0/lzy/ZAPS-results/diag_endpoint_components/A8_20261020');parser.add_argument('--device',default='cuda');parser.add_argument('--summary-only',action='store_true')
    args=parser.parse_args();directory=Path(args.run_dir).resolve()
    if args.summary_only:
        plan=json.loads((directory/'plan.json').read_text());completed(directory,plan);print('A8 Summary:',json.dumps(stored_summary(directory)),'\nRecords:',directory);return
    import fcntl
    directory.parent.mkdir(parents=True,exist_ok=True)
    with (directory.parent/(directory.name+'.lock')).open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A8 already running') from None
        directory,plan,parents=bind(args);records=completed(directory,plan)
        try:
            todo=[case for case in plan['cases'] if case['key'] not in records]
            if todo:
                import torch
                coupled.archive.recovery.require_space(directory)
                with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                    model=coupled.archive.recovery.load_model(plan,args.device)
                    for case in todo:records[case['key']]=run_case(plan,parents[case['key']],directory/case['key'],model,args.device)
        finally:
            result=stored_summary(directory);WRITE(directory/'summary.json',result);print('A8 Summary:',json.dumps(result),'\nRecords:',directory,flush=True)
        from utils.archive_refinement_confirmation import export_and_commit
        bundle,commit=export_and_commit({'A8':str(directory)});print('Archive:',bundle,'\nResults commit:',commit,flush=True)


if __name__=='__main__':main()
