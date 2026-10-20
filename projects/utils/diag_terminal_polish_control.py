"""A9: matched extra-call terminal-polish control, 12 diagnostic / zero method NFE."""
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
from utils import diag_endpoint_components as parent
from utils import diag_coupled_refinement as coupled
from utils import diag_terminal_continuation as terminal
from utils.diag_state_trajectory import digest
WRITE=coupled.WRITE;METRICS=('psnr','ssim','lpips')
CARD=PROJECTS/'experiment_cards/A9_terminal_polish_control_20261020.md'


def compare(branches,floors):
    effects={'S_C':coupled.history.effect(branches['S'],branches['C'],floors)};readouts={}
    for arm in ('P','F'):
        for b in (0,1):
            name=arm+str(b)
            effects['S_'+name]=coupled.history.effect(branches['S'],branches[name],floors)
            effects[name+'_S']=coupled.history.effect(branches[name],branches['S'],floors)
        mid=all(effects[arm+str(b)+'_S']['joint_local_candidate'] for b in (0,1))
        end=all(effects['S_'+arm+str(b)]['joint_local_candidate'] for b in (0,1))
        if mid and end:raise ValueError('Contradictory dominance')
        readouts[arm]='MIDPOINT_DOMINATES_S_BOTH_BRIDGES' if mid else 'S_DOMINATES_BOTH_MIDPOINT_BRIDGES' if end else 'TRADEOFF_OR_UNRESOLVED'
    return effects,readouts


def summarize(records):
    complete=[r for r in records if r['status']=='complete'];images={r['case']['image_id'] for r in complete};comparisons={}
    for arm in ('P','F'):
        labels=('MIDPOINT_DOMINATES_S_BOTH_BRIDGES','S_DOMINATES_BOTH_MIDPOINT_BRIDGES','TRADEOFF_OR_UNRESOLVED')
        comparisons[arm]=dict(case_counts={label:sum(r['readouts'][arm]==label for r in complete) for label in labels},
            image_counts_both_seeds={label:sum(len([r for r in complete if r['case']['image_id']==i])==2 and all(r['readouts'][arm]==label for r in complete if r['case']['image_id']==i) for i in images) for label in labels})
    p=comparisons['P']['image_counts_both_seeds']
    classification='INCOMPLETE_PILOT' if len(complete)!=6 else 'MIDPOINT_PRIOR_MATCHED_COST_ADVANTAGE_CANDIDATE' if p['MIDPOINT_DOMINATES_S_BOTH_BRIDGES']>=2 else 'TERMINAL_POLISH_PREFERRED_IN_COHORT' if p['S_DOMINATES_BOTH_MIDPOINT_BRIDGES']>=2 else 'NO_ROBUST_MIDPOINT_ADVANTAGE_ESTABLISHED'
    return dict(completed_cases=len(complete),planned_cases=6,method_nfe=sum(r['method_nfe'] for r in records),diagnostic_nfe=sum(r['diagnostic_nfe'] for r in records),classification=classification,
        S_cases_joint_vs_C=sum(r['S_joint_vs_C'] for r in complete),comparisons=comparisons,
        mean_delta={key:{metric:statistics.mean(r['effects'][key]['delta'][metric] for r in complete) if complete else None for metric in ('loss',*METRICS)} for key in ('S_C','S_P0','S_P1','S_F0','S_F1')},
        rows=[dict(case=r['case']['key'],branches=r['branches'],readouts=r['readouts'],S_joint_vs_C=r['S_joint_vs_C'],gates_passed=all(g['passed'] for g in r['gates'].values())) for r in complete],
        limits='Three reused A6 image clusters. One extra scientific evaluation per candidate; repeats counted separately as diagnostics. Fixed candidates only, no state control or generalization claim.',
        cost_is_lower_bound_if_abruptly_interrupted=any(r['status']=='running' for r in records))


def bind(args):
    source=Path(args.a8_dir).resolve()
    if not(source/'plan.json').is_file():raise ValueError('Need completed A8; no previous experiment started')
    source,old,parents=parent.bind(SimpleNamespace(source_plan_dir=args.source_plan_dir,a2c_dir=args.a2c_dir,a2d_dir=args.a2d_dir,a3_dir=args.a3_dir,a4_dir=args.a4_dir,
        a5_dir=args.a5_dir,a6_dir=args.a6_dir,a7_dir=args.a7_dir,run_dir=str(source)))
    active=parent.completed(source,old)
    summary=parent.stored_summary(source)
    if len(active)!=6 or summary!=json.loads((source/'summary.json').read_text()) or summary['classification']!='PRIOR_COMPONENT_SUFFICIENT_CANDIDATE':raise ValueError('A8 audit/context differs')
    paths=[source/'plan.json',source/'summary.json',CARD,Path(__file__),PROJECTS/'run_a9_20261020.sh',PROJECTS/'utils/test_diag_terminal_polish_control.py']
    for key in active:paths.extend(source/key/name for name in ('run.json','run.sha256','outputs.pt'))
    plan=dict(experiment='terminal_polish_control_plan',source=str(source),a6_source=old['source'],cases=old['cases'],config=old['config'],grid=old['grid'],checkpoint_sha256=old['checkpoint_sha256'],
        identity_files={**old['identity_files'],**{str(p):digest(p) for p in paths}},evaluation_t=0,parameter_index=0,scientific_extra_evaluations=1,planned_method_nfe=0,planned_diagnostic_nfe=12)
    directory=Path(args.run_dir).resolve()
    if any(directory.is_relative_to(p) or p.is_relative_to(directory) for p in (source,Path(old['source']))):raise ValueError('New run overlaps source')
    if directory.exists():
        if json.loads((directory/'plan.json').read_text())!=plan:raise ValueError('Frozen A9 plan differs; do not overwrite')
    else:directory.mkdir(parents=True,exist_ok=False);WRITE(directory/'plan.json',plan)
    return directory,plan,active,parents


def completed(directory,plan):
    result={}
    for case in plan['cases']:
        path=directory/case['key']/'run.json'
        if not path.exists():
            if path.parent.exists():raise ValueError('Interrupted case retained')
            continue
        r=json.loads(path.read_text())
        if (digest(path)!=(path.parent/'run.sha256').read_text().strip() or r['status']!='complete' or r['case']!=case or r['plan_sha256']!=digest(directory/'plan.json') or
            r['method_nfe']!=0 or r['diagnostic_nfe']!=2 or len(r['calls'])!=2 or not r['gates'] or not all(g['passed'] for g in r['gates'].values()) or
            digest(path.parent/'outputs.pt')!=r['output_sha256']):raise ValueError('Failed/interrupted/incompatible A9 retained')
        effects,readouts=compare(r['branches'],r['floors'])
        if effects!=r['effects'] or readouts!=r['readouts'] or coupled.candidate(effects['S_C'])!=r['S_joint_vs_C']:raise ValueError('Saved A9 arithmetic differs')
        result[case['key']]=r
    return result


def polish_pair(z,y,c,evaluate):
    fields=[];outputs=[]
    for _ in (0,1):
        field=evaluate(z,y,c.clone(),0,0)
        fields.append(field);outputs.append(terminal.terminal_output(z,c,field).cpu())
    return fields,outputs


def run_case(plan,parent_record,a6_record,directory,model,device):
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
    case=parent_record['case'];source=Path(plan['a6_source'])/case['key'];directory.mkdir(exist_ok=False)
    (directory/'experiment_card.md').write_text(CARD.read_text())
    r=dict(experiment='terminal_polish_control_case',status='running',case=case,plan_sha256=digest(directory.parent/'plan.json'),method_nfe=0,diagnostic_nfe=0,calls=[],gates={})
    started=time.perf_counter()
    def save():r['seconds']=time.perf_counter()-started;WRITE(directory/'run.json',r)
    def count(_module,inputs):
        r['diagnostic_nfe']+=1;r['calls'].append(dict(repeat=r['diagnostic_nfe']-1,t=int(inputs[1].flatten()[0]),parameter_index=0))
        if r['diagnostic_nfe']>2 or r['calls'][-1]['t']!=0:raise RuntimeError('A9 call/time budget failed')
    save();hook=model.model.register_forward_pre_hook(count)
    try:
        before=rng_state(device)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))),torch.no_grad():
            state=torch.load(source/'pre_last_state.pt',map_location='cpu',weights_only=True)
            original=torch.load(source/'last_opt.pt',map_location='cpu',weights_only=True)
            prior_outputs=torch.load(Path(plan['source'])/case['key']/'outputs.pt',map_location='cpu',weights_only=True)['outputs']
            c=prior_outputs['C']
            if state['tau'].tolist()!=plan['grid'] or relative_error(original,c)>1e-6 or tuple(c.shape)!=(1,3,256,256) or c.dtype!=torch.float32 or not torch.isfinite(c).all():raise ValueError('Invalid original final C')
            base=get_operator('super_resolution',device=device,scale_factor=4,noise_sigma=0.)
            operator=unit_operator(TransposeMode(base,'legacy_bicubic').to(device)).to(device)
            z=ZAPS(model,operator,img_size=256,**plan['config']);z.tau=state['tau'].to(device);z.zeta.copy_(state['zeta'].to(device));z.D.copy_(state['D'].to(device))
            y=torch.load(a6_record['measurement']['path'],map_location='cpu',weights_only=True)['y'].to(device)
            fields=[];outputs=[]
            with model_rng_guard(model.model,device),patch('torch.randn_like',side_effect=RuntimeError('Unexpected terminal noise draw')):
                fields,outputs=polish_pair(z,y,c.to(device),evaluate_field)
            checks=dict(clean_repeat=relative_error(fields[0]['clean'],fields[1]['clean']),q_repeat=relative_error(fields[0]['q'],fields[1]['q']),output_repeat=relative_error(outputs[0],outputs[1]))
            if any(not math.isfinite(v) or v>1e-7 for v in checks.values()) or any(not torch.isfinite(o).all() for o in outputs):raise RuntimeError('Polish repeat/finiteness failed')
            if not(torch.equal(z.tau.cpu(),state['tau']) and torch.equal(z.zeta.cpu(),state['zeta']) and torch.equal(z.D.cpu(),state['D'])):raise RuntimeError('Original parameters changed')
            tmp=directory/'outputs.pt.tmp';torch.save(dict(C=c,S=outputs[0],S_repeat=outputs[1]),tmp);tmp.replace(directory/'outputs.pt');r['output_sha256']=digest(directory/'outputs.pt')
            gt=load_image_as_tensor(case['image']).cpu()
            def physical(output):return float((y-operator.H(output.to(device))).double().square().mean()/(2*.05**2))
            branches={a:parent_record['branches'][a] for a in ('F0','F1','P0','P1')}
            branches['C']=dict(loss=physical(c),**compute_all_metrics(c,gt,lpips_net='alex'))
            branches['S']=dict(loss=physical(outputs[0]),**compute_all_metrics(outputs[0],gt,lpips_net='alex'))
            repeat=dict(loss=physical(outputs[1]),**compute_all_metrics(outputs[1],gt,lpips_net='alex'))
            drift={k:abs(branches['C'][k]-parent_record['branches']['C'][k]) for k in ('loss',*METRICS)}
            if any(drift[k]>v for k,v in (('psnr',.01),('ssim',.001),('lpips',.001),('loss',1e-7+2e-5*abs(branches['C']['loss'])))):raise RuntimeError('Original C metric convention differs')
            floors={k:max(parent_record['floors'][k],drift[k],abs(repeat[k]-branches['S'][k])) for k in ('loss',*METRICS)}
            effects,readouts=compare(branches,floors);r.update(branches=branches,floors=floors,effects=effects,readouts=readouts,S_joint_vs_C=coupled.candidate(effects['S_C']))
        if not same_rng(before,rng_state(device)) or r['diagnostic_nfe']!=2 or any(digest(p)!=sha for p,sha in plan['identity_files'].items()):raise RuntimeError('RNG/identity/budget gate failed')
        r['gates']=dict(polish_repeat=dict(passed=True,checks=checks),baseline_C=dict(passed=True,metric_drifts=drift),
                       parameters_grid_rng_no_draw=dict(passed=True),identity_budget=dict(passed=True,method_nfe=0,diagnostic_nfe=2))
        r['status']='complete';save();(directory/'run.sha256').write_text(digest(directory/'run.json')+'\n');return r
    except BaseException as error:r['status']='failed';r['error']=repr(error);save();raise
    finally:hook.remove()


def stored_summary(directory):return summarize([json.loads(p.read_text()) for p in sorted(directory.glob('*/run.json'))])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-plan-dir',default='results/diag_guidance_replication/A2a_20261018')
    for label,folder in (('a2c','diag_one_step_guidance_scale/A2c_20261019'),('a2d','diag_direction_refresh/A2d_20261019'),('a3','diag_coupled_refinement/A3_20261020'),('a4','diag_terminal_continuation/A4_20261020'),('a5','diag_fixed_position_continuation/A5_20261020'),('a6','diag_fresh_refinement_confirmation/A6_20261020'),('a7','diag_direction_value/A7_20261020'),('a8','diag_endpoint_components/A8_20261020')):
        parser.add_argument('--'+label+'-dir',default='/data0/lzy/ZAPS-results/'+folder)
    parser.add_argument('--run-dir',default='/data0/lzy/ZAPS-results/diag_terminal_polish_control/A9_20261020');parser.add_argument('--device',default='cuda');parser.add_argument('--summary-only',action='store_true')
    args=parser.parse_args();directory=Path(args.run_dir).resolve()
    if args.summary_only:
        completed(directory,json.loads((directory/'plan.json').read_text()));print('A9 Summary:',json.dumps(stored_summary(directory)),'\nRecords:',directory);return
    import fcntl
    directory.parent.mkdir(parents=True,exist_ok=True)
    with (directory.parent/(directory.name+'.lock')).open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A9 already running') from None
        directory,plan,parents,a6=bind(args);records=completed(directory,plan)
        try:
            todo=[c for c in plan['cases'] if c['key'] not in records]
            if todo:
                import torch
                coupled.archive.recovery.require_space(directory)
                with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                    model=coupled.archive.recovery.load_model(plan,args.device)
                    for c in todo:records[c['key']]=run_case(plan,parents[c['key']],a6[c['key']],directory/c['key'],model,args.device)
        finally:
            summary=stored_summary(directory);WRITE(directory/'summary.json',summary);print('A9 Summary:',json.dumps(summary),'\nRecords:',directory,flush=True)
        from utils.archive_refinement_confirmation import export_and_commit
        bundle,commit=export_and_commit({'A9':str(directory)});print('Archive:',bundle,'\nResults commit:',commit,flush=True)


if __name__=='__main__':main()
