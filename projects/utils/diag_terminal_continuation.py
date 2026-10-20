"""A4: complete saved t14 A3 branches through the true terminal step; 24 diagnostic NFE."""
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
from utils.diag_state_trajectory import digest
CARD=PROJECTS/'experiment_cards/A4_terminal_continuation_20261020.md'
WRITE=parent.WRITE
METRICS=parent.METRICS


def summarize(records):
    complete=[r for r in records if r['status']=='complete']
    images={r['case']['image_id'] for r in complete}
    both=sum(len([r for r in complete if r['case']['image_id']==image])==2 and
             all(r['both_bridges_candidate'] for r in complete if r['case']['image_id']==image) for image in images)
    supported=len(complete)==6 and both>=2
    return dict(completed_cases=len(complete),planned_cases=6,source_t=14,
        method_nfe=sum(r['method_nfe'] for r in records),diagnostic_nfe=sum(r['diagnostic_nfe'] for r in records),
        classification=('FINAL_FIXED_REFINEMENT_CANDIDATE_ONLY' if supported else 'FINAL_FIXED_REFINEMENT_NOT_SUPPORTED') if len(complete)==6 else 'INCOMPLETE_PILOT',
        cases_both_bridges=sum(r['both_bridges_candidate'] for r in complete),images_both_seeds_both_bridges=both,
        bridge_candidates={a:sum(r['bridge_candidates'][a] for r in complete) for a in ('F0','F1')},
        mean_delta={a+'_C':{k:statistics.mean(r['effects'][a+'_C']['delta'][k] for r in complete) if complete else None
                           for k in ('loss',*METRICS)} for a in ('F0','F1')},
        rows=[dict(case=r['case']['key'],branches=r['branches'],bridge_candidates=r['bridge_candidates'],
                   both_bridges_candidate=r['both_bridges_candidate'],gates_passed=all(g['passed'] for g in r['gates'].values())) for r in complete],
        limits='Three reused development image clusters; two bridges are not independent images. Fixed t14 terminal continuation only; no state-selection, cosine-control or generalization claim.',
        cost_is_lower_bound_if_abruptly_interrupted=any(r['status']=='running' for r in records))


def bind(args):
    root,a3,old,parents=parent.bind(SimpleNamespace(source_plan_dir=args.source_plan_dir,a2c_dir=args.a2c_dir,
                                                  a2d_dir=args.a2d_dir,run_dir=args.a3_dir,inspect=True))
    active=parent.completed(a3,old)
    result=parent.stored_summary(a3)
    if len(active)!=6 or result!=json.loads((a3/'summary.json').read_text()):
        raise ValueError('Need six complete A3 cases with exact recomputed summary; do not rerun A3')
    if result['classification']!='LOCAL_COUPLED_REFINEMENT_CANDIDATE_ONLY':
        raise ValueError('A3 H5 did not unlock the fixed terminal continuation')
    paths=[a3/'plan.json',a3/'summary.json',Path(__file__),CARD,PROJECTS/'run_a4_20261020.sh',
           PROJECTS/'utils/test_diag_terminal_continuation.py']
    for key in active:paths.extend(a3/key/name for name in ('run.json','run.sha256','branches.pt'))
    plan=dict(experiment='terminal_continuation_plan',source=str(root),source_cases=old['source_cases'],a3_source=str(a3),
        identity_files={**old['identity_files'],**{str(p):digest(p) for p in paths}},source_step=28,source_t=14,
        final_evaluation_t=0,terminal_t=-1,parameter_index=0,arms=['C','F0','F1'],
        planned_method_nfe=0,planned_diagnostic_nfe=24,parent_cost=old['parent_cost'],
        preceding_a2c_cost=old['preceding_a2c_cost'],preceding_a2d_cost=old['preceding_a2d_cost'],
        preceding_a3_cost={k:result[k] for k in ('method_nfe','diagnostic_nfe')})
    directory=Path(args.run_dir).resolve()
    for source in (root,Path(old['a2c_source']),Path(old['a2d_source']),a3):
        if directory.is_relative_to(source) or source.is_relative_to(directory):raise ValueError('New directory must be separate')
    if directory.exists():
        if json.loads((directory/'plan.json').read_text())!=plan:raise ValueError('Frozen A4 plan differs; do not overwrite')
    elif not args.inspect:
        directory.mkdir(parents=True,exist_ok=False);WRITE(directory/'plan.json',plan)
        # Preserve the complete lightweight A3 evidence without modifying its source.
        archive=directory/'parent_A3';archive.mkdir()
        for source in [a3/'plan.json',a3/'summary.json',*[a3/k/n for k in sorted(active) for n in ('run.json','run.sha256')]]:
            target=archive/source.relative_to(a3);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
            if digest(target)!=digest(source):raise ValueError('Parent evidence copy differs')
        WRITE(directory/'parent_audit.json',dict(passed=True,cases=6,points=30,endpoint_comparisons=60,
              summary=result,scope='Complete A3 JSON and SHA/branch artifact checks; no new UNet calls'))
    return root,directory,plan,parents,active


def completed(directory,plan):
    records={}
    for key in sorted(plan['source_cases']):
        path=directory/key/'run.json'
        if not path.exists():
            if path.parent.exists():raise ValueError('Interrupted directory retained: '+key)
            continue
        r=json.loads(path.read_text())
        if (digest(path)!=(path.parent/'run.sha256').read_text().strip() or r['status']!='complete' or r['case']['key']!=key or
            r['plan_sha256']!=digest(directory/'plan.json') or r['method_nfe']!=0 or r['diagnostic_nfe']!=4 or len(r['calls'])!=4 or
            not r['gates'] or not all(g['passed'] for g in r['gates'].values()) or
            digest(path.parent/'final_outputs.pt')!=r['output_archive']['sha256']):raise ValueError('Incompatible/failed record retained: '+key)
        effects={a+'_C':parent.history.effect(r['branches'][a],r['branches']['C'],r['floors']) for a in ('F0','F1')}
        hits={a:parent.candidate(effects[a+'_C']) for a in ('F0','F1')}
        if effects!=r['effects'] or hits!=r['bridge_candidates'] or all(hits.values())!=r['both_bridges_candidate']:
            raise ValueError('Saved final decision differs: '+key)
        records[key]=r
    return records


def terminal_output(z,x,field):
    from modules.zaps_algorithm import ddpm_posterior_step
    # Actual frozen core: terminal posterior returns clean without a draw, then q is added.
    return ddpm_posterior_step(x,field['clean'],0,-1,z.dm.alphas_cumprod,eta=1.,mode='ddpm')+field['q']


def select_inputs(artifact):
    points=artifact['points']
    if len(points)!=5 or [p['source_step'] for p in points]!=[8,19,24,27,28]:raise ValueError('Original A3 point order differs')
    point=points[4]
    if (point['spec']['t'],point['spec']['m'],point['spec']['s'])!=(14,3,0):raise ValueError('Need fixed t14/3/0 interval')
    return {a:point[a] for a in ('F0','F1')}


def run_case(parent_plan,source_record,a3_record,source,directory,plan,model,device):
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
    record=dict(experiment='terminal_continuation_case',status='running',case=source_record['case'],
                plan_sha256=digest(directory.parent/'plan.json'),method_nfe=0,diagnostic_nfe=0,calls=[],gates={},
                experiment_card=CARD.read_text(),git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECTS,text=True).strip())
    started=time.perf_counter()
    def save():record['seconds']=time.perf_counter()-started;WRITE(directory/'run.json',record)
    save();context={}
    def count(_module,inputs):
        record['diagnostic_nfe']+=1;record['calls'].append(dict(context,t=int(inputs[1].flatten()[0])))
        if record['diagnostic_nfe']>4 or record['calls'][-1]['t']!=0:raise RuntimeError('Terminal budget/time gate failed')
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
            if any(tuple(v.shape)!=(1,3,256,256) or v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in inputs.values()):
                raise ValueError('Branch input shape/dtype/finiteness differs')
            base=get_operator('super_resolution',device=device,scale_factor=4,noise_sigma=0.)
            operator=unit_operator(TransposeMode(base,'legacy_bicubic').to(device)).to(device)
            z=ZAPS(model,operator,img_size=256,**parent_plan['config']);z.tau=state['tau'].to(device)
            z.zeta.copy_(state['zeta'].to(device));z.D.copy_(state['D'].to(device))
            y=torch.load(source_record['measurement']['path'],map_location='cpu',weights_only=True)['y'].to(device)
            fields={};outputs={};repeated=None
            with model_rng_guard(model.model,device),patch('torch.randn_like',side_effect=RuntimeError('Unexpected terminal noise draw')):
                for arm,repeat in (('C',0),('C',1),('F0',0),('F1',0)):
                    context.clear();context.update(arm=arm,repeat=repeat,parameter_index=0)
                    field=evaluate_field(z,y,inputs[arm].to(device),0,0)
                    output=terminal_output(z,inputs[arm].to(device),field)
                    if not torch.isfinite(output).all():raise RuntimeError('Nonfinite final output')
                    if arm=='C' and repeat==1:repeated=(field,output.cpu())
                    else:fields[arm]=field;outputs[arm]=output.cpu()
            check=dict(clean_error=relative_error(cache['clean'][29:30],fields['C']['clean'].cpu()),
                       q_error=relative_error(cache['guidance'][29:30],fields['C']['q'].cpu()),
                       final_error=relative_error(original,outputs['C']),repeat_error=relative_error(outputs['C'],repeated[1]))
            if any(not math.isfinite(v) for v in check.values()) or max(check.values())>1e-6 or check['repeat_error']>1e-7:
                raise RuntimeError('Original final output parity failed: '+json.dumps(check))
            a3_point=next(p for p in a3_record['points'] if p['source_t']==14)
            losses={}
            for arm,output in outputs.items():
                losses[arm]=float((y-operator.H(output.to(device))).double().square().mean()/(2*.05**2))
                reference=a3_point['branches'][arm]['loss']
                if abs(fields[arm]['loss']-reference)>1e-7+2e-5*abs(reference):raise RuntimeError('A3 cached endpoint clean loss differs')
            def physical(output):return float((y-operator.H(output.to(device))).double().square().mean()/(2*.05**2))
            original_loss,repeat_loss=physical(original),physical(repeated[1])
            if not all(math.isfinite(v) for v in (*losses.values(),original_loss,repeat_loss)):raise RuntimeError('Nonfinite final physical loss')
            if not(torch.equal(z.tau.cpu(),state['tau']) and torch.equal(z.zeta.cpu(),state['zeta']) and torch.equal(z.D.cpu(),state['D'])):
                raise RuntimeError('Frozen parameters changed')
            temporary=directory/'final_outputs.pt.tmp';torch.save(outputs,temporary);temporary.replace(directory/'final_outputs.pt')
            record['output_archive']=dict(path=str(directory/'final_outputs.pt'),sha256=digest(directory/'final_outputs.pt'))
            # Every final tensor and all four neural calls are fixed before loading GT.
            gt=load_image_as_tensor(source_record['case']['image']).cpu()
            branches={a:dict(loss=losses[a],**compute_all_metrics(output,gt,lpips_net='alex')) for a,output in outputs.items()}
            repeated_metrics=compute_all_metrics(repeated[1],gt,lpips_net='alex')
            drifts={k:abs(branches['C'][k]-source_record['final_metrics'][k]) for k in METRICS}
            if any(drifts[k]>v for k,v in (('psnr',.01),('ssim',.001),('lpips',.001))):raise RuntimeError('Original final metric convention differs')
            endpoint_drifts={}
            for arm,field in fields.items():
                metrics=compute_all_metrics(field['clean'].cpu(),gt,lpips_net='alex')
                delta={k:abs(metrics[k]-a3_point['branches'][arm][k]) for k in METRICS}
                if any(delta[k]>v for k,v in (('psnr',.01),('ssim',.001),('lpips',.001))):raise RuntimeError('A3 clean metric convention differs')
                endpoint_drifts[arm]=delta
            floors={k:max(drifts[k],abs(repeated_metrics[k]-branches['C'][k])) for k in METRICS}
            floors['loss']=max(abs(losses['C']-original_loss),abs(losses['C']-repeat_loss))
            effects={a+'_C':parent.history.effect(branches[a],branches['C'],floors) for a in ('F0','F1')}
            hits={a:parent.candidate(effects[a+'_C']) for a in ('F0','F1')}
            record.update(branches=branches,floors=floors,effects=effects,bridge_candidates=hits,both_bridges_candidate=all(hits.values()))
        if not same_rng(before,rng_state(device)):raise RuntimeError('External RNG changed')
        if record['diagnostic_nfe']!=4 or any(digest(p)!=sha for p,sha in plan['identity_files'].items()):raise RuntimeError('Identity/budget gate failed')
        record['gates']=dict(final_baseline=dict(passed=True,checks=check,metric_drifts=drifts),
            cached_A3=dict(passed=True,endpoint_metric_drifts=endpoint_drifts),terminal_no_draw=dict(passed=True),
            parameters_grid_rng=dict(passed=True),identity_and_budget=dict(passed=True,method_nfe=0,diagnostic_nfe=4))
        record['status']='complete';save();(directory/'run.sha256').write_text(digest(directory/'run.json')+'\n')
        print('A4 CASE:',key,json.dumps(dict(branches=branches,effects=effects,both_bridges_candidate=all(hits.values()),gates=record['gates'])),flush=True)
        return record
    except BaseException as error:
        record['status']='failed';record['error']=repr(error);save();raise
    finally:hook.remove()


def stored_summary(directory):
    return summarize([json.loads(p.read_text()) for p in sorted(directory.glob('*/run.json'))])


def execute(args):
    root,directory,plan,parents,a3=bind(args)
    if args.inspect:
        print('A4 INSPECT PASS: full A3 audited; fixed t14, six cases, planned method/diagnostic=0/24');return
    records=completed(directory,plan)
    todo=[key for key in sorted(parents) if key not in records]
    try:
        if todo:
            import torch
            parent.archive.recovery.require_space(directory)
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                model=parent.archive.recovery.load_model(json.loads((root/'plan.json').read_text()),args.device)
                for key in todo:
                    records[key]=run_case(json.loads((root/'plan.json').read_text()),parents[key],a3[key],Path(plan['source_cases'][key]),directory/key,plan,model,args.device)
    finally:
        summary=stored_summary(directory);WRITE(directory/'summary.json',summary)
        print('A4 Summary:',json.dumps(summary),'\nRecords:',directory,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-plan-dir',default='results/diag_guidance_replication/A2a_20261018')
    parser.add_argument('--a2c-dir',default='/data0/lzy/ZAPS-results/diag_one_step_guidance_scale/A2c_20261019')
    parser.add_argument('--a2d-dir',default='/data0/lzy/ZAPS-results/diag_direction_refresh/A2d_20261019')
    parser.add_argument('--a3-dir',default='/data0/lzy/ZAPS-results/diag_coupled_refinement/A3_20261020')
    parser.add_argument('--run-dir',default='/data0/lzy/ZAPS-results/diag_terminal_continuation/A4_20261020')
    parser.add_argument('--device',default='cuda')
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--inspect',action='store_true');group.add_argument('--run',action='store_true');group.add_argument('--summary-only',action='store_true')
    args=parser.parse_args();directory=Path(args.run_dir).resolve()
    if args.summary_only:
        plan=json.loads((directory/'plan.json').read_text());completed(directory,plan)
        print('A4 Summary:',json.dumps(stored_summary(directory)),'\nRecords:',directory);return
    if args.inspect:execute(args);return
    import fcntl
    directory.parent.mkdir(parents=True,exist_ok=True)
    with (directory.parent/(directory.name+'.lock')).open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A4 already running; do not resubmit') from None
        execute(args)


if __name__=='__main__':main()
