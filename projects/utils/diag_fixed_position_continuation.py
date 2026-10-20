"""A5: continue fixed t54 branches with original suffix noise; compare saved t14 final outputs."""
import argparse
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS))
from utils import diag_coupled_refinement as parent
from utils import diag_terminal_continuation as previous
from utils.diag_state_trajectory import digest
CARD=PROJECTS/'experiment_cards/A5_fixed_position_comparison_20261020.md'
WRITE=parent.WRITE
METRICS=parent.METRICS


def summarize(records):
    complete=[r for r in records if r['status']=='complete']
    images={r['case']['image_id'] for r in complete}
    both=sum(len([r for r in complete if r['case']['image_id']==image])==2 and
             all(r['both_bridges_candidate'] for r in complete if r['case']['image_id']==image) for image in images)
    supported=len(complete)==6 and both>=2
    return dict(completed_cases=len(complete),planned_cases=6,source_t=54,
        method_nfe=sum(r['method_nfe'] for r in records),diagnostic_nfe=sum(r['diagnostic_nfe'] for r in records),
        classification=('FINAL_T54_REFINEMENT_CANDIDATE_ONLY' if supported else 'FINAL_T54_REFINEMENT_NOT_SUPPORTED') if len(complete)==6 else 'INCOMPLETE_PILOT',
        cases_both_bridges=sum(r['both_bridges_candidate'] for r in complete),images_both_seeds_both_bridges=both,
        bridge_candidates={a:sum(r['bridge_candidates'][a] for r in complete) for a in ('F0','F1')},
        mean_delta={a+'_C':{k:statistics.mean(r['effects'][a+'_C']['delta'][k] for r in complete) if complete else None
                           for k in ('loss',*METRICS)} for a in ('F0','F1')},
        rows=[dict(case=r['case']['key'],branches=r['branches'],bridge_candidates=r['bridge_candidates'],
                   both_bridges_candidate=r['both_bridges_candidate'],position_comparison=dict(readout=r['position_comparison']['readout'],bridge_averaged_delta={k:statistics.mean(e['delta'][k] for e in r['position_comparison']['T54_minus_T14'].values()) for k in ('loss',*METRICS)}),gates_passed=all(g['passed'] for g in r['gates'].values())) for r in complete],
        position_counts={label:sum(r['position_comparison']['readout']==label for r in complete) for label in ('T54_DOMINATES_BOTH_T14_BRIDGES','T14_DOMINATES_BOTH_T54_BRIDGES','TRADEOFF_OR_UNRESOLVED')},
        position_images_both_seeds={label:sum(len([r for r in complete if r['case']['image_id']==image])==2 and all(r['position_comparison']['readout']==label for r in complete if r['case']['image_id']==image) for image in images) for label in ('T54_DOMINATES_BOTH_T14_BRIDGES','T14_DOMINATES_BOTH_T54_BRIDGES')},
        limits='Three reused development image clusters; fixed t54 vs saved fixed t14, all 2x2 bridge comparisons. No state-selection, cosine-control, generalization or GT output selection claim.',
        cost_is_lower_bound_if_abruptly_interrupted=any(r['status']=='running' for r in records))


def position_comparison(current,reference,current_floors,reference_floors):
    floors={k:max(current_floors[k],reference_floors[k]) for k in ('loss',*METRICS)}
    forward={a+'_'+b:parent.history.effect(current[a],reference[b],floors) for a in ('F0','F1') for b in ('F0','F1')}
    reverse={b+'_'+a:parent.history.effect(reference[b],current[a],floors) for a in ('F0','F1') for b in ('F0','F1')}
    win=all(e['joint_local_candidate'] for e in forward.values())
    lose=all(e['joint_local_candidate'] for e in reverse.values())
    if win and lose:raise ValueError('Contradictory Pareto position comparisons')
    return dict(readout='T54_DOMINATES_BOTH_T14_BRIDGES' if win else 'T14_DOMINATES_BOTH_T54_BRIDGES' if lose else 'TRADEOFF_OR_UNRESOLVED',
                floors=floors,T54_minus_T14=forward,T14_minus_T54=reverse)


def bind(args):
    root,a4,old,parents,active_a3=previous.bind(SimpleNamespace(source_plan_dir=args.source_plan_dir,a2c_dir=args.a2c_dir,
        a2d_dir=args.a2d_dir,a3_dir=args.a3_dir,run_dir=args.a4_dir,inspect=True))
    active=previous.completed(a4,old);result=previous.stored_summary(a4)
    if len(active)!=6 or result!=json.loads((a4/'summary.json').read_text()):raise ValueError('Need six complete audited A4 cases; do not rerun them')
    if result['classification']!='FINAL_FIXED_REFINEMENT_CANDIDATE_ONLY':raise ValueError('A4 H6 did not unlock the position comparison')
    paths=[a4/'plan.json',a4/'summary.json',Path(__file__),CARD,PROJECTS/'run_a5_20261020.sh',
           PROJECTS/'utils/test_diag_fixed_position_continuation.py']
    for key in active:paths.extend(a4/key/name for name in ('run.json','run.sha256','final_outputs.pt'))
    plan=dict(experiment='fixed_position_continuation_plan',source=str(root),source_cases=old['source_cases'],
        a3_source=old['a3_source'],a4_source=str(a4),identity_files={**old['identity_files'],**{str(p):digest(p) for p in paths}},
        source_step=27,source_t=54,suffix_times=[14,0],suffix_indices=[1,0],terminal_t=-1,
        original_suffix_noise_step=28,arms=['C','F0','F1'],planned_method_nfe=0,planned_diagnostic_nfe=36,
        parent_cost=old['parent_cost'],preceding_a2c_cost=old['preceding_a2c_cost'],preceding_a2d_cost=old['preceding_a2d_cost'],
        preceding_a3_cost=old['preceding_a3_cost'],preceding_a4_cost={k:result[k] for k in ('method_nfe','diagnostic_nfe')})
    directory=Path(args.run_dir).resolve()
    for source in (root,Path(args.a2c_dir).resolve(),Path(args.a2d_dir).resolve(),Path(old['a3_source']),a4):
        if directory.is_relative_to(source) or source.is_relative_to(directory):raise ValueError('New directory must be separate')
    if directory.exists():
        if json.loads((directory/'plan.json').read_text())!=plan:raise ValueError('Frozen A5 plan differs; do not overwrite')
    elif not args.inspect:
        directory.mkdir(parents=True,exist_ok=False);WRITE(directory/'plan.json',plan)
        evidence=directory/'parent_A4';evidence.mkdir()
        for source in [a4/'plan.json',a4/'summary.json',*[a4/k/n for k in sorted(active) for n in ('run.json','run.sha256')]]:
            target=evidence/source.relative_to(a4);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
            if digest(target)!=digest(source):raise ValueError('Parent evidence copy differs')
        WRITE(directory/'parent_audit.json',dict(passed=True,A3_cases=6,A4_cases=6,A4_comparisons=12,A4_summary=result,new_UNet_nfe=0))
    return root,directory,plan,parents,active_a3,active


def completed(directory,plan):
    records={}
    for key in sorted(plan['source_cases']):
        path=directory/key/'run.json'
        if not path.exists():
            if path.parent.exists():raise ValueError('Interrupted directory retained: '+key)
            continue
        r=json.loads(path.read_text())
        if (digest(path)!=(path.parent/'run.sha256').read_text().strip() or r['status']!='complete' or r['case']['key']!=key or
            r['plan_sha256']!=digest(directory/'plan.json') or r['method_nfe']!=0 or r['diagnostic_nfe']!=6 or len(r['calls'])!=6 or
            not r['gates'] or not all(g['passed'] for g in r['gates'].values()) or
            digest(path.parent/'final_outputs.pt')!=r['output_archive']['sha256']):raise ValueError('Incompatible/failed record retained: '+key)
        effects={a+'_C':parent.history.effect(r['branches'][a],r['branches']['C'],r['floors']) for a in ('F0','F1')}
        hits={a:parent.candidate(effects[a+'_C']) for a in ('F0','F1')}
        if effects!=r['effects'] or hits!=r['bridge_candidates'] or all(hits.values())!=r['both_bridges_candidate']:
            raise ValueError('Saved final decision differs: '+key)
        reference=json.loads((Path(plan['a4_source'])/key/'run.json').read_text())
        if position_comparison(r['branches'],reference['branches'],r['floors'],reference['floors'])!=r['position_comparison']:
            raise ValueError('Saved cross-position arithmetic differs: '+key)
        records[key]=r
    return records


def terminal_output(z,x,field):
    from modules.zaps_algorithm import ddpm_posterior_step
    # Actual frozen core: terminal posterior returns clean without a draw, then q is added.
    return ddpm_posterior_step(x,field['clean'],0,-1,z.dm.alphas_cumprod,eta=1.,mode='ddpm')+field['q']


def select_inputs(artifact):
    points=artifact['points']
    if len(points)!=5 or [p['source_step'] for p in points]!=[8,19,24,27,28]:raise ValueError('Original A3 point order differs')
    point=points[3]
    if (point['spec']['t'],point['spec']['m'],point['spec']['s'])!=(54,28,14):raise ValueError('Need fixed t54/28/14 interval')
    if (points[4]['spec']['t'],points[4]['spec']['s'])!=(14,0):raise ValueError('Need archived original suffix draw at t14')
    return {a:point[a] for a in ('F0','F1')}


def continue_suffix(z,y,x,noise,evaluate,before_eval):
    before_eval(14,1)
    first=evaluate(z,y,x,14,1)
    at_zero=parent.posterior(x,first['clean'],14,0,z.dm.alphas_cumprod,noise)+first['q']
    before_eval(0,0)
    final=evaluate(z,y,at_zero,0,0)
    return terminal_output(z,at_zero,final),first,final


def run_case(parent_plan,source_record,a3_record,a4_record,source,directory,plan,model,device):
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
    directory.mkdir(exist_ok=False);(directory/'experiment_card.md').write_text(CARD.read_text())
    record=dict(experiment='fixed_position_continuation_case',status='running',case=source_record['case'],
                plan_sha256=digest(directory.parent/'plan.json'),method_nfe=0,diagnostic_nfe=0,calls=[],gates={},
                experiment_card=CARD.read_text(),git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECTS,text=True).strip())
    started=time.perf_counter()
    def save():record['seconds']=time.perf_counter()-started;WRITE(directory/'run.json',record)
    save();context={}
    def count(_module,inputs):
        record['diagnostic_nfe']+=1;record['calls'].append(dict(context,t=int(inputs[1].flatten()[0])))
        if record['diagnostic_nfe']>6 or record['calls'][-1]['t']!=context['eval_t']:raise RuntimeError('Suffix budget/time gate failed')
    hook=model.model.register_forward_pre_hook(count)
    try:
        before=rng_state(device)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))),torch.no_grad():
            state=torch.load(source/'pre_last_state.pt',map_location='cpu',weights_only=True)
            cache=torch.load(source/'trajectory.pt',map_location='cpu',weights_only=True)
            original=torch.load(source/'last_opt.pt',map_location='cpu',weights_only=True)
            if state['tau'].tolist()!=parent_plan['grid'] or int(state['tau'][0])!=0:raise ValueError('Source grid differs')
            key=record['case']['key'];artifact=torch.load(Path(plan['a3_source'])/key/'branches.pt',map_location='cpu',weights_only=True)
            inputs=select_inputs(artifact);inputs['C']=cache['latent_inputs'][29:30]
            noise=artifact['points'][4]['Z'].to(device)
            if tuple(noise.shape)!=(1,3,256,256) or noise.dtype!=torch.float32 or not torch.isfinite(noise).all():raise ValueError('Invalid original raw suffix Z')
            if any(tuple(v.shape)!=(1,3,256,256) or v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in inputs.values()):
                raise ValueError('Branch input shape/dtype/finiteness differs')
            base=get_operator('super_resolution',device=device,scale_factor=4,noise_sigma=0.)
            operator=unit_operator(TransposeMode(base,'legacy_bicubic').to(device)).to(device)
            z=ZAPS(model,operator,img_size=256,**parent_plan['config']);z.tau=state['tau'].to(device)
            z.zeta.copy_(state['zeta'].to(device));z.D.copy_(state['D'].to(device))
            y=torch.load(source_record['measurement']['path'],map_location='cpu',weights_only=True)['y'].to(device)
            replay=parent.posterior(cache['latent_inputs'][28:29].to(device),cache['clean'][28:29].to(device),14,0,z.dm.alphas_cumprod,noise)+cache['guidance'][28:29].to(device)
            suffix_error=relative_error(cache['latent_inputs'][29:30],replay.cpu())
            if not math.isfinite(suffix_error) or suffix_error>1e-6:raise RuntimeError('Original t14 suffix noise replay differs')
            fields={};outputs={};repeated=None;endpoints={}
            with model_rng_guard(model.model,device),patch('torch.randn_like',side_effect=RuntimeError('Unexpected uncoupled noise draw')):
                for repeat in (0,1):
                    context.clear();context.update(arm='C',repeat=repeat,parameter_index=0,eval_t=0)
                    field=evaluate_field(z,y,inputs['C'].to(device),0,0)
                    output=terminal_output(z,inputs['C'].to(device),field)
                    if repeat==1:repeated=(field,output.cpu())
                    else:fields['C']=field;outputs['C']=output.cpu()
                for arm in ('F0','F1'):
                    context.clear();context.update(arm=arm,repeat=0,parameter_index=1,eval_t=14)
                    def before_eval(t,index):context.update(eval_t=t,parameter_index=index)
                    output,first,field=continue_suffix(z,y,inputs[arm].to(device),noise,evaluate_field,before_eval)
                    endpoints[arm]=first
                    if not torch.isfinite(output).all():raise RuntimeError('Nonfinite final output')
                    fields[arm]=field;outputs[arm]=output.cpu()
            check=dict(clean_error=relative_error(cache['clean'][29:30],fields['C']['clean'].cpu()),
                       q_error=relative_error(cache['guidance'][29:30],fields['C']['q'].cpu()),
                       final_error=relative_error(original,outputs['C']),repeat_error=relative_error(outputs['C'],repeated[1]))
            if any(not math.isfinite(v) for v in check.values()) or max(check.values())>1e-6 or check['repeat_error']>1e-7:
                raise RuntimeError('Original final output parity failed: '+json.dumps(check))
            a3_point=next(p for p in a3_record['points'] if p['source_t']==54)
            losses={}
            for arm,output in outputs.items():
                losses[arm]=float((y-operator.H(output.to(device))).double().square().mean()/(2*.05**2))
                if arm!='C':
                    reference=a3_point['branches'][arm]['loss']
                    if abs(endpoints[arm]['loss']-reference)>1e-7+2e-5*abs(reference):raise RuntimeError('A3 cached t14 clean loss differs')
            def physical(output):return float((y-operator.H(output.to(device))).double().square().mean()/(2*.05**2))
            original_loss,repeat_loss=physical(original),physical(repeated[1])
            if not all(math.isfinite(v) for v in (*losses.values(),original_loss,repeat_loss)):raise RuntimeError('Nonfinite final physical loss')
            if not(torch.equal(z.tau.cpu(),state['tau']) and torch.equal(z.zeta.cpu(),state['zeta']) and torch.equal(z.D.cpu(),state['D'])):
                raise RuntimeError('Frozen parameters changed')
            temporary=directory/'final_outputs.pt.tmp';torch.save(outputs,temporary);temporary.replace(directory/'final_outputs.pt')
            record['output_archive']=dict(path=str(directory/'final_outputs.pt'),sha256=digest(directory/'final_outputs.pt'))
            # Every final tensor and all six neural calls are fixed before loading GT.
            gt=load_image_as_tensor(source_record['case']['image']).cpu()
            branches={a:dict(loss=losses[a],**compute_all_metrics(output,gt,lpips_net='alex')) for a,output in outputs.items()}
            repeated_metrics=compute_all_metrics(repeated[1],gt,lpips_net='alex')
            drifts={k:abs(branches['C'][k]-source_record['final_metrics'][k]) for k in METRICS}
            if any(drifts[k]>v for k,v in (('psnr',.01),('ssim',.001),('lpips',.001))):raise RuntimeError('Original final metric convention differs')
            endpoint_drifts={}
            for arm,field in endpoints.items():
                metrics=compute_all_metrics(field['clean'].cpu(),gt,lpips_net='alex')
                delta={k:abs(metrics[k]-a3_point['branches'][arm][k]) for k in METRICS}
                if any(delta[k]>v for k,v in (('psnr',.01),('ssim',.001),('lpips',.001))):raise RuntimeError('A3 clean metric convention differs')
                endpoint_drifts[arm]=delta
            floors={k:max(drifts[k],abs(repeated_metrics[k]-branches['C'][k])) for k in METRICS}
            floors['loss']=max(abs(losses['C']-original_loss),abs(losses['C']-repeat_loss))
            effects={a+'_C':parent.history.effect(branches[a],branches['C'],floors) for a in ('F0','F1')}
            hits={a:parent.candidate(effects[a+'_C']) for a in ('F0','F1')}
            position=position_comparison(branches,a4_record['branches'],floors,a4_record['floors'])
            a4_drifts={k:abs(branches['C'][k]-a4_record['branches']['C'][k]) for k in ('loss',*METRICS)}
            if any(a4_drifts[k]>v for k,v in (('loss',1e-7+2e-5*abs(losses['C'])),('psnr',.01),('ssim',.001),('lpips',.001))):raise RuntimeError('A4 original C changed')
            record.update(branches=branches,floors=floors,effects=effects,bridge_candidates=hits,both_bridges_candidate=all(hits.values()),position_comparison=position)
        if not same_rng(before,rng_state(device)):raise RuntimeError('External RNG changed')
        if record['diagnostic_nfe']!=6 or any(digest(p)!=sha for p,sha in plan['identity_files'].items()):raise RuntimeError('Identity/budget gate failed')
        record['gates']=dict(final_baseline=dict(passed=True,checks=check,metric_drifts=drifts),
            cached_A3=dict(passed=True,endpoint_metric_drifts=endpoint_drifts),
            cached_A4=dict(passed=True,baseline_drifts=a4_drifts),original_suffix_noise=dict(passed=True,relative_error=suffix_error),terminal_no_draw=dict(passed=True),
            parameters_grid_rng=dict(passed=True),identity_and_budget=dict(passed=True,method_nfe=0,diagnostic_nfe=6))
        record['status']='complete';save();(directory/'run.sha256').write_text(digest(directory/'run.json')+'\n')
        print('A5 CASE:',key,json.dumps(dict(branches=branches,effects=effects,both_bridges_candidate=all(hits.values()),gates=record['gates'])),flush=True)
        return record
    except BaseException as error:
        record['status']='failed';record['error']=repr(error);save();raise
    finally:hook.remove()


def stored_summary(directory):
    return summarize([json.loads(p.read_text()) for p in sorted(directory.glob('*/run.json'))])


def execute(args):
    root,directory,plan,parents,a3,a4=bind(args)
    if args.inspect:
        print('A5 INSPECT PASS: full A3/A4 audited; fixed t54, six cases, planned method/diagnostic=0/36');return
    records=completed(directory,plan)
    todo=[key for key in sorted(parents) if key not in records]
    try:
        if todo:
            import torch
            parent.archive.recovery.require_space(directory)
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                model=parent.archive.recovery.load_model(json.loads((root/'plan.json').read_text()),args.device)
                for key in todo:
                    records[key]=run_case(json.loads((root/'plan.json').read_text()),parents[key],a3[key],a4[key],Path(plan['source_cases'][key]),directory/key,plan,model,args.device)
    finally:
        summary=stored_summary(directory);WRITE(directory/'summary.json',summary)
        print('A5 Summary:',json.dumps(summary),'\nRecords:',directory,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-plan-dir',default='results/diag_guidance_replication/A2a_20261018')
    parser.add_argument('--a2c-dir',default='/data0/lzy/ZAPS-results/diag_one_step_guidance_scale/A2c_20261019')
    parser.add_argument('--a2d-dir',default='/data0/lzy/ZAPS-results/diag_direction_refresh/A2d_20261019')
    parser.add_argument('--a3-dir',default='/data0/lzy/ZAPS-results/diag_coupled_refinement/A3_20261020')
    parser.add_argument('--a4-dir',default='/data0/lzy/ZAPS-results/diag_terminal_continuation/A4_20261020')
    parser.add_argument('--run-dir',default='/data0/lzy/ZAPS-results/diag_fixed_position_continuation/A5_20261020')
    parser.add_argument('--device',default='cuda')
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--inspect',action='store_true');group.add_argument('--run',action='store_true');group.add_argument('--summary-only',action='store_true')
    args=parser.parse_args();directory=Path(args.run_dir).resolve()
    if args.summary_only:
        plan=json.loads((directory/'plan.json').read_text());completed(directory,plan)
        print('A5 Summary:',json.dumps(stored_summary(directory)),'\nRecords:',directory);return
    if args.inspect:execute(args);return
    import fcntl
    directory.parent.mkdir(parents=True,exist_ok=True)
    with (directory.parent/(directory.name+'.lock')).open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A5 already running; do not resubmit') from None
        execute(args)


if __name__=='__main__':main()
