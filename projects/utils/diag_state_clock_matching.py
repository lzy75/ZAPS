"""A25: two fixed same-seed cross-image clock derangements; 360 diagnostic calls."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import numpy as np

PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS))
from utils import diag_state_feedback_diagnosis as parent
source=parent.source;policy=parent.policy;old=parent.old;WRITE=parent.WRITE;digest=parent.digest;METRICS=parent.METRICS
ARMS=('DONOR_PLUS','DONOR_MINUS');CALLS=60
ROOT='/data0/lzy/ZAPS-results/diag_state_clock_matching/A25_20261020'
CARD=PROJECTS/'experiment_cards/A25_state_clock_matching_20261020.md'


def clock_map(records):
    """No labels: use all other image clocks, preserve seed and clock multiset."""
    lookup={(r['case']['image_id'],r['case']['sampling_seed']):r for r in records}
    images=sorted({i for i,_ in lookup});seeds=sorted({s for _,s in lookup})
    if len(records)!=6 or len(lookup)!=6 or len(images)!=3 or seeds!=[1000,2000] or set(lookup)!={(i,s) for i in images for s in seeds}:raise ValueError('Need fixed three images x two seeds')
    result={}
    for r in records:
        i,s=r['case']['image_id'],r['case']['sampling_seed'];index=images.index(i);arms={}
        for arm,offset in zip(ARMS,(1,-1)):
            donor=lookup[(images[(index+offset)%3],s)]
            arms[arm]=dict(donor=donor['case']['key'],times=[row['t'] for row in donor['traces']['RC_REPLAY']])
        result[r['case']['key']]=arms
    return result


def validate_clock(times,positions):
    if len(times)!=30 or any(type(t) is not int for t in times) or times[0]!=999 or times[-1]!=0 or any(t<=s for t,s in zip(times,times[1:])):raise ValueError('Invalid clock endpoints/order')
    if any(times[j] not in positions.cells[j] for j in range(1,29)):raise ValueError('Clock left original position cells')


def prepare(args):
    from modules.diffusion_model import _precompute_diffusion_params
    a24=Path(args.a24_dir).resolve();p=old.read(a24/'plan.json');root=Path(args.root).resolve()
    for name in ('plan','summary'):
        if digest(a24/(name+'.json'))!=(a24/(name+'.sha256')).read_text().strip():raise ValueError('A24 fingerprint differs')
    records=[parent.completed(a24,s) for s in p['cases']];summary=old.read(a24/'summary.json')
    if any(r is None for r in records) or parent.summarize(records)!=summary:raise ValueError('Need six audited A24 cases')
    if summary['classification']!='MECHANISM_DIAGNOSIS_ONLY' or not summary['comparisons']['online_feedback_vs_shared_fixed_clock']['psnr_mechanism_hint']:raise ValueError('No preregistered A24 clock hint: do not start GPU')
    clocks=clock_map(records)
    positions=policy.Positions(policy.noise_u(_precompute_diffusion_params()['alphas_cumprod'].cpu().numpy()),list(reversed(p['grid'])))
    for arms in clocks.values():
        for arm in ARMS:validate_clock(arms[arm]['times'],positions)
    identities=dict(p['identity_files'])
    for path,sha in identities.items():
        if digest(path)!=sha:raise ValueError('Historical source/code changed: '+path)
    def bind(path):identities[str(path)]=digest(path)
    for name in ('plan.json','plan.sha256','summary.json','summary.sha256'):bind(a24/name)
    for s in p['cases']:
        for name in ('run.json','run.sha256','outputs.pt'):bind(a24/s['case']['key']/name)
    own=[Path(__file__),CARD,PROJECTS/'run_a25_20261020.sh',PROJECTS/'utils/test_diag_state_clock_matching.py']
    for path in own:bind(path)
    if root.is_relative_to(a24) or a24.is_relative_to(root) or any(root.is_relative_to(Path(s['source']).resolve()) for s in p['cases']):raise ValueError('Run overlaps history')
    plan=dict(experiment='A25_state_clock_matching',cases=p['cases'],config=p['config'],grid=p['grid'],checkpoint_sha256=p['checkpoint_sha256'],identity_files=identities,
        worker_identity_files={**p['worker_identity_files'],**{str(x):identities[str(x)] for x in own}},runtime=p['runtime'],native_policy=p['native_policy'],source_A24=str(a24),
        source_A23=p['source_A23'],clocks=clocks,arms=list(ARMS),planned_method_nfe=0,planned_diagnostic_nfe=360,round_in_user_limit='3/5',
        source_summary=summary,scope='Retrospective clock-matching diagnostic. No new online state policy; all donor clocks and source acquisitions retain historical cost.')
    if root.exists():
        if old.read(root/'plan.json')!=plan:raise ValueError('Frozen A25 differs; refuse overwrite')
    else:
        root.mkdir(parents=True,exist_ok=False);WRITE(root/'plan.json',plan);(root/'plan.sha256').write_text(digest(root/'plan.json')+'\n');(root/'experiment_card.md').write_text(CARD.read_text())
    print('A25 PREPARE PASS: two full same-seed derangements, six cases, 360 diagnostic calls; no new policy.',flush=True)


def summarize(records):
    rr=[r for r in records if r['status']=='complete'];cost=sum(r['diagnostic_nfe'] for r in records)
    out=dict(completed_cases=len(rr),planned_cases=6,image_clusters=len({r['case']['image_id'] for r in rr}),method_nfe=0,diagnostic_nfe=cost,round_in_user_limit='3/5',classification='INCOMPLETE_PILOT',
        cumulative_queue_cost=dict(method_nfe=7200,diagnostic_nfe=5048+cost,total=12248+cost),cost_is_lower_bound_if_abruptly_interrupted=any(r['status']=='running' for r in records),
        limits='Retrospective three-image diagnosis; donor clocks are fixed external inputs, not online state feedback. Two seeds/donors are dependent. DONOR_MEAN averages metrics, not pixels or best outputs. No generalization, curvature, acceleration or new state-policy claim.')
    if len(rr)!=6:return out
    effects={a:source.parent.endpoint(rr,[a]*6) for a in ('C','SELF','FIXED_GRID',*ARMS,'DONOR_MEAN')}
    pairs={a:parent.pair(rr,'SELF',a) for a in (*ARMS,'DONOR_MEAN')}
    hint=pairs['DONOR_MEAN']['psnr_mechanism_hint'] and all(pairs[a]['mean_delta']['psnr']>0 for a in ARMS)
    out.update(classification='CLOCK_MATCHING_HINT_WITHOUT_NET_GAIN' if hint else 'CURRENT_STATE_CLOCK_MATCHING_NOT_SUPPORTED',relative_to_C=effects,SELF_minus_donor=pairs,
        source_SELF_net_benefit_supported=effects['SELF']['quality_guard'] and effects['SELF']['mean_delta']['psnr']>=.05,
        clock_changes={a:dict(mean_different_nodes=statistics.mean(r['clock_changes'][a]['different_nodes'] for r in rr),mean_abs_delta_t=statistics.mean(r['clock_changes'][a]['mean_abs_delta_t'] for r in rr)) for a in ARMS},
        rows=[dict(case=r['case']['key'],branches=r['branches'],gates_passed=all(g['passed'] for g in r['gates'].values())) for r in rr])
    return out


def completed(root,spec):
    path=root/spec['case']['key']/'run.json'
    if not path.exists():
        if path.parent.exists():raise ValueError('Interrupted A25 retained')
        return None
    r=old.read(path);plan=old.read(root/'plan.json')
    if (r['status']!='complete' or digest(path)!=(path.parent/'run.sha256').read_text().strip() or r['case']!=spec['case'] or r['plan_sha256']!=digest(root/'plan.json') or r['diagnostic_nfe']!=CALLS or r['method_nfe']!=0 or len(r['calls'])!=CALLS or
        digest(path.parent/'outputs.pt')!=r['outputs_sha256'] or set(r['traces'])!=set(ARMS) or set(r['branches'])!={'C','SELF','FIXED_GRID','DONOR_MEAN',*ARMS} or not r['gates'] or not all(g['passed'] for g in r['gates'].values())):raise ValueError('Invalid/incomplete A25 retained')
    for a in ARMS:
        if [x['t'] for x in r['traces'][a]]!=plan['clocks'][spec['case']['key']][a]['times'] or sum(c['arm']==a for c in r['calls'])!=30:raise ValueError('Actual clock/cost differs')
    return r


def roll(z,y,initial,clock,positions,noises,before):
    """One forward per frozen time; no target-dependent choice of future locations."""
    import torch
    validate_clock(clock,positions)
    if len(noises)!=29:raise ValueError('Need exactly 29 original slot draws')
    x=initial;rows=[]
    for k,t in enumerate(clock):
        before(k,t);f=source.field(z,y,x,t,positions);s=clock[k+1] if k<29 else -1
        rows.append(dict(step=k,t=t,next_t=s,u=float(positions.u[t]),R=f['R'],qRMS=float(f['q'].double().square().mean().sqrt()),clip_fraction=f['clip_fraction'],parameter_weights=f['weights']))
        x=source.parent.terminal.terminal_output(z,x,f) if s<0 else source.parent.coupled.posterior(x,f['clean'],t,s,z.dm.alphas_cumprod,noises[k])+f['q']
        if not torch.isfinite(x).all():raise RuntimeError('Nonfinite replay state')
    return x,rows


def run_case(root,plan,spec,model,device):
    import torch
    from unittest.mock import patch
    from modules.zaps_algorithm import ZAPS
    from modules.degradations import get_operator
    from modules.main_single import load_image_as_tensor
    from utils.diag_ffhq_regression import TransposeMode
    from utils.diag_imagenet_gaussian_unit_domain import unit_operator
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state,same_rng
    from utils.metrics import compute_all_metrics
    folder=root/spec['case']['key'];folder.mkdir(exist_ok=False);start=time.perf_counter();context={}
    record=dict(status='running',case=spec['case'],plan_sha256=digest(root/'plan.json'),method_nfe=0,diagnostic_nfe=0,calls=[],gates={})
    def save():record['seconds']=time.perf_counter()-start;WRITE(folder/'run.json',record)
    def count(_module,inputs):
        record['diagnostic_nfe']+=1;record['calls'].append(dict(**context,t=int(inputs[1].flatten()[0])));save()
        if record['diagnostic_nfe']>CALLS or record['calls'][-1]['t']!=context['expected_t']:raise RuntimeError('Budget/time mismatch')
    def check():
        source.parent.verify_case_sources(plan,spec)
        a24=Path(plan['source_A24'])
        for s in plan['cases']:
            for name in ('run.json','run.sha256','outputs.pt'):
                path=a24/s['case']['key']/name
                if digest(path)!=plan['identity_files'][str(path)]:raise ValueError('Bound A24 target/donor changed')
        for path,sha in plan['worker_identity_files'].items():
            if digest(path)!=sha:raise ValueError('Bound worker changed')
    save();hook=model.model.register_forward_pre_hook(count)
    try:
        check();src=Path(spec['source']);native=old.read(src/'run.json');a24=parent.completed(Path(plan['source_A24']),spec);external=rng_state(device)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))),torch.no_grad():
            state=torch.load(src/'pre_last_state.pt',map_location='cpu',weights_only=True);cache=torch.load(src/'trajectory.pt',map_location='cpu',weights_only=True);original=torch.load(src/'last_opt.pt',map_location='cpu',weights_only=True)
            operator=unit_operator(TransposeMode(get_operator('super_resolution',device=device,scale_factor=4,noise_sigma=0.),'legacy_bicubic').to(device)).to(device)
            z=ZAPS(model,operator,img_size=256,**plan['config']);z.tau=state['tau'].to(device);z.zeta.copy_(state['zeta'].to(device));z.D.copy_(state['D'].to(device))
            positions=policy.Positions(policy.noise_u(z.dm.alphas_cumprod.cpu().numpy()),list(reversed(plan['grid'])))
            if state['tau'].tolist()!=plan['grid'] or any(abs(positions.u[x['t']]-x['noise_u'])>1e-8 for x in native['rows']):raise ValueError('Source noise/grid mismatch')
            y=torch.load(spec['measurement']['path'],map_location='cpu',weights_only=True)['y'].to(device)
            noises,noise_gate=source.parent.coupled.replay_noises(state,cache,dict(rows=native['rows']),z,y,device,original)
            outputs={};traces={};changes={};self_clock=[r['t'] for r in a24['traces']['RC_REPLAY']]
            with model_rng_guard(model.model,device),patch('torch.randn_like',side_effect=RuntimeError('Unaccounted draw')):
                for arm in ARMS:
                    clock=plan['clocks'][spec['case']['key']][arm]['times']
                    def before(k,t):context.update(arm=arm,step=k,expected_t=t)
                    output,rows=roll(z,y,state['init_noise'].to(device),clock,positions,noises,before)
                    outputs[arm]=output.cpu();traces[arm]=rows
                    changes[arm]=dict(donor=plan['clocks'][spec['case']['key']][arm]['donor'],different_nodes=sum(a!=b for a,b in zip(clock,self_clock)),mean_abs_delta_t=statistics.mean(abs(a-b) for a,b in zip(clock,self_clock)))
            if record['diagnostic_nfe']!=CALLS or not(torch.equal(z.tau.cpu(),state['tau']) and torch.equal(z.zeta.cpu(),state['zeta']) and torch.equal(z.D.cpu(),state['D'])):raise RuntimeError('Budget/parameters changed')
            tmp=folder/'outputs.pt.tmp';torch.save(outputs,tmp);tmp.replace(folder/'outputs.pt');record['outputs_sha256']=digest(folder/'outputs.pt')
            gt=load_image_as_tensor(spec['case']['image']).cpu()
            def metrics(x):return dict(loss=float((y-operator.H(x.to(device))).double().square().mean()/(2*.05**2)),**compute_all_metrics(x,gt,lpips_net='alex'))
            branches={a:metrics(x) for a,x in outputs.items()};branches.update(C=a24['branches']['C'],SELF=a24['branches']['RC'],FIXED_GRID=a24['branches']['FIXED_GRID'])
            branches['DONOR_MEAN']={m:statistics.mean(branches[a][m] for a in ARMS) for m in METRICS}
            if not all(math.isfinite(v) for b in branches.values() for v in b.values()):raise RuntimeError('Nonfinite metrics')
            record.update(branches=branches,traces=traces,clock_changes=changes,floors=a24['floors'])
        if not same_rng(external,rng_state(device)) or source.parent.precision.precision_state(torch)!=plan['native_policy']:raise RuntimeError('RNG/precision changed')
        check();record['gates']=dict(original_noise=noise_gate,parameters_RNG_budget=dict(passed=True),source_A24_parity_reused=dict(passed=True,scope='A24 SELF replay gate reused, not rerun'),clock_and_GT_free_mapping=dict(passed=True),sources_unchanged=dict(passed=True))
        record['status']='complete';save();(folder/'run.sha256').write_text(digest(folder/'run.json')+'\n');completed(root,spec)
        print('A25 CASE:',spec['case']['key'],'diagnostic=',record['diagnostic_nfe'],flush=True)
    except BaseException as e:record.update(status='failed',error=repr(e));save();raise
    finally:hook.remove()


def worker(args,plan,root):
    import fcntl,torch
    with (root/f'shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        todo=[s for i,s in enumerate(sorted(plan['cases'],key=lambda s:s['case']['key'])) if i%2==args.shard and completed(root,s) is None]
        if not todo:return
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(args.device),NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        if runtime!=plan['runtime'] or source.parent.precision.precision_state(torch)!=plan['native_policy']:raise ValueError('Native runtime differs; no calls')
        torch.set_num_threads(2);source.parent.coupled.archive.recovery.require_space(root)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            model=source.parent.coupled.archive.recovery.load_model(plan,args.device)
            for spec in todo:run_case(root,plan,spec,model,args.device)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=ROOT);p.add_argument('--a24-dir',default=parent.ROOT);p.add_argument('--device',default='cuda:0')
    m=p.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--shard',type=int,choices=(0,1));m.add_argument('--archive',action='store_true');m.add_argument('--summary',action='store_true');args=p.parse_args();root=Path(args.root).resolve()
    if args.prepare:prepare(args);return
    plan=old.read(root/'plan.json')
    if digest(root/'plan.json')!=(root/'plan.sha256').read_text().strip():raise ValueError('Plan changed')
    for path,sha in plan['worker_identity_files'].items():
        if digest(path)!=sha:raise ValueError('Bound analysis/worker changed')
    if args.shard is not None:worker(args,plan,root);return
    if args.summary:
        if digest(root/'summary.json')!=(root/'summary.sha256').read_text().strip():raise ValueError('Summary changed')
        print('A25 Summary:',json.dumps(old.read(root/'summary.json')),flush=True);print('Records:',root,flush=True);return
    records=[]
    for spec in plan['cases']:
        path=root/spec['case']['key']/'run.json'
        if path.exists():
            r=old.read(path)
            if r['status']=='complete':completed(root,spec)
            records.append(r)
    out=summarize(records);WRITE(root/'summary.json',out);(root/'summary.sha256').write_text(digest(root/'summary.json')+'\n')
    from utils.archive_refinement_confirmation import export_and_commit
    bundle,commit=export_and_commit({'A25':str(root)});print('Archive:',bundle,'Results commit:',commit,flush=True)


if __name__=='__main__':main()
