"""A11/A12: parallel fixed-time clean-field directional response; 120 diagnostic NFE."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
import numpy as np

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS))
from utils import diag_clean_roughness as previous
from utils import diag_coupled_refinement as coupled
from utils.diag_state_trajectory import digest
WRITE=previous.WRITE
ROOT='/data0/lzy/ZAPS-results/diag_clean_field_response'
STAGES={14:('A11',28,1),54:('A12',27,2)}
ETAS=(.01,.02)


def read(path):return json.loads(Path(path).read_text())
def rms(a):return float(np.sqrt(np.mean(np.asarray(a,dtype=np.float64)**2)))


def orthogonal(q,seed):
    q=np.asarray(q,dtype=np.float64)
    if not np.isfinite(q).all() or rms(q)<=1e-12:raise ValueError('Unresolved q; no replacement direction')
    a=np.random.default_rng(seed).standard_normal(q.shape)
    a-=float(np.sum(a*q)/np.sum(q*q))*q
    if rms(a)<=1e-12:raise ValueError('Unresolved orthogonal draw; no retry')
    a=(a*(rms(q)/rms(a))).astype(np.float32)
    cosine=abs(float(np.sum(a*q)/(np.linalg.norm(a.astype(np.float64))*np.linalg.norm(q))))
    error=abs(rms(a)/rms(q)-1)
    if cosine>1e-6 or error>1e-6:raise ValueError('Orthogonality/RMS contract failed')
    return a,dict(abs_cosine=cosine,rms_relative_error=error,seed=seed)


def cosine(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    den=float(np.linalg.norm(a)*np.linalg.norm(b))
    return max(-1.,min(1.,float(np.sum(a*b)/den))) if den>1e-30 else None


def collect(z,y,x,directions,t,index,evaluate,perturb):
    centers=[evaluate(z,y,x.clone(),t,index) for _ in range(2)]
    pairs=[]
    for name,direction in directions.items():
        for eta in ETAS:
            xp,xm,inp=perturb(x,direction,eta)
            pairs.append(dict(direction=name,eta=eta,input=inp,
                plus=evaluate(z,y,xp,t,index),minus=evaluate(z,y,xm,t,index)))
    return centers,pairs


def assess(payload):
    centers=payload['centers'];floor=rms(np.asarray(centers[0]['clean'],dtype=np.float64)-np.asarray(centers[1]['clean'],dtype=np.float64))
    directions={}
    for name,direction in payload['directions'].items():
        result=[];vectors=[]
        for pair in [p for p in payload['pairs'] if p['direction']==name]:
            diff=np.asarray(pair['plus']['clean'],dtype=np.float64)-np.asarray(pair['minus']['clean'],dtype=np.float64)
            span=pair['input']['effective_span'];resolved=bool(pair['input']['input_resolved'] and span>0 and rms(diff)>10*max(floor,1e-10))
            j=diff/span if span>0 else np.zeros_like(diff);vectors.append(j)
            result.append(dict(eta=pair['eta'],effective_span=span,input_resolved=pair['input']['input_resolved'],resolved=resolved,
                clean_difference_rms=rms(diff),gain=rms(j)/rms(direction) if span>0 else None,
                response_input_cosine=cosine(j,direction) if resolved else None,
                phi=(pair['plus']['loss']-pair['minus']['loss'])/span if span>0 else None,
                plus_clip=pair['plus']['clip_fraction'],minus_clip=pair['minus']['clip_fraction']))
        if [v['eta'] for v in result]!=list(ETAS):raise ValueError('Frozen two scales missing')
        scale=max(rms(j) for j in vectors);spread=rms(vectors[0]-vectors[1])/scale if scale>1e-30 else None
        c=cosine(*vectors)
        usable=all(v['resolved'] for v in result) and spread is not None and spread<=.2 and c is not None and c>0
        directions[name]=dict(pairs=result,relative_vector_spread=spread,cross_scale_cosine=c,usable=usable)
    usable=all(v['usable'] for v in directions.values())
    q,o=(directions[n]['pairs'] for n in ('q','orthogonal'))
    more_q=usable and all(a['gain']>=1.1*b['gain'] for a,b in zip(q,o))
    more_o=usable and all(b['gain']>=1.1*a['gain'] for a,b in zip(q,o))
    return dict(center_clean_repeat_rms=floor,directions=directions,usable=usable,
        q_over_orthogonal_gain=[a['gain']/b['gain'] if a['gain'] is not None and b['gain'] is not None and b['gain']>1e-30 else None for a,b in zip(q,o)],
        readout='GUIDANCE_MORE_SENSITIVE' if more_q else 'ORTHOGONAL_MORE_SENSITIVE' if more_o else 'NO_RESOLVED_DIFFERENCE' if usable else 'UNRESOLVED_OR_SCALE_DEPENDENT')


def directory(root,t):return Path(root).resolve()/f'{STAGES[t][0]}_t{t}_20261020'


def prepare(args):
    root10=Path(args.a10_dir).resolve();p10=read(root10/'plan.json');r10=read(root10/'run.json')
    if (digest(root10/'run.json')!=(root10/'run.sha256').read_text().strip() or r10['status']!='complete' or
        r10['method_nfe']!=0 or r10['diagnostic_nfe']!=0 or not all(r10['gates'].values()) or
        r10['plan_sha256']!=digest(root10/'plan.json') or r10['analysis']!=previous.analyze(r10['rows']) or
        r10['analysis']['classification']!='CLEAN_ROUGHNESS_INCREMENT_NOT_SUPPORTED' or previous.compact(r10)!=read(root10/'summary.json')):
        raise ValueError('Need verified complete A10 with declared context')
    a6=Path(args.a6_dir).resolve();parent=read(a6/'plan.json')
    if a6!=Path(read(Path(p10['source_a7'])/'plan.json')['source']).resolve():raise ValueError('A6 source differs from A10')
    files={**p10['identity_files']}
    for path,sha in files.items():
        if digest(path)!=sha:raise ValueError('Parent identity changed: '+path)
    sources={}
    for case in parent['cases']:
        source=a6/case['key'];r=read(source/'run.json')
        if (r['status']!='complete' or r['case']!=case or digest(source/'run.json')!=(source/'run.sha256').read_text().strip() or
            r['plan_sha256']!=digest(a6/'plan.json') or r['method_nfe']!=300 or r['diagnostic_nfe']!=72 or not r['gates'] or not all(g['passed'] for g in r['gates'].values())):
            raise ValueError('Need complete compatible A6 case: '+case['key'])
        artifacts={n:r['artifacts_sha256'][n] for n in ('pre_last_state.pt','trajectory.pt')}
        for n,sha in artifacts.items():
            if digest(source/n)!=sha:raise ValueError('Changed A6 tensor: '+n)
        if digest(r['measurement']['path'])!=r['measurement']['sha256']:raise ValueError('Observation changed')
        sources[case['key']]=dict(directory=str(source),record_sha256=digest(source/'run.json'),artifacts=artifacts,
            measurement=r['measurement'],baseline=r['baseline'],positions=r['positions'])
    expected={(i,s) for i in ('00023','00024','00025') for s in (1000,2000)}
    if len(sources)!=6 or {(c['image_id'],c['sampling_seed']) for c in parent['cases']}!=expected:raise ValueError('Fixed six cases differ')
    own=[Path(__file__),PROJECTS/'utils/test_diag_clean_field_response.py',PROJECTS/'run_a11_a12_20261020.sh',
         PROJECTS/'experiment_cards/A11_A12_clean_response_protocol_20261020.md',
         PROJECTS/'experiment_cards/A11_clean_response_t14_20261020.md',PROJECTS/'experiment_cards/A12_clean_response_t54_20261020.md']
    own_hashes={str(p):digest(p) for p in own}
    files.update(own_hashes);files.update({str(root10/n):digest(root10/n) for n in ('plan.json','run.json','run.sha256','summary.json')})
    files[str(a6/'plan.json')]=digest(a6/'plan.json')
    root=Path(args.root).resolve()
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in (a6,root10)):raise ValueError('New root overlaps parents')
    for t,(label,k,index) in STAGES.items():
        plan=dict(experiment=label+'_clean_field_response',source_t=t,source_step=k,parameter_index=index,config=parent['config'],grid=parent['grid'],
            checkpoint_sha256=parent['checkpoint_sha256'],cases=parent['cases'],case_sources=sources,identity_files=files,worker_identity_files=own_hashes,
            etas=list(ETAS),orthogonal_seed_rule='611020+100*sorted_case_index+source_t',planned_method_nfe=0,planned_diagnostic_nfe=60)
        target=directory(root,t)
        if target.exists():
            if read(target/'plan.json')!=plan:raise ValueError('Frozen stage plan differs; do not overwrite')
        else:
            target.mkdir(parents=True,exist_ok=False);WRITE(target/'plan.json',plan)
            (target/'plan.sha256').write_text(digest(target/'plan.json')+'\n')
            (target/'experiment_card.md').write_text(own[-2 if t==14 else -1].read_text())
            (target/'common_protocol.md').write_text(own[-3].read_text())
    print('A11/A12 PREPARE PASS: source audit once; six cases per stage; 0 method / 120 diagnostic NFE planned.',flush=True)


def completed(root,plan):
    active={}
    for case in plan['cases']:
        folder=root/case['key'];path=folder/'run.json'
        if not path.exists():
            if folder.exists():raise ValueError('Interrupted directory retained: '+str(folder))
            continue
        r=read(path)
        if (r['status']!='complete' or digest(path)!=(folder/'run.sha256').read_text().strip() or r['plan_sha256']!=digest(root/'plan.json') or
            r['case']!=case or r['diagnostic_nfe']!=10 or r['method_nfe']!=0 or len(r['calls'])!=10 or not r['gates'] or
            not all(g['passed'] for g in r['gates'].values()) or digest(folder/'fields.pt')!=r['fields_sha256']):
            raise ValueError('Failed/interrupted/incompatible case retained: '+case['key'])
        active[case['key']]=r
    return active


def run_case(root,plan,case,model,device):
    import torch
    from unittest.mock import patch
    from modules.zaps_algorithm import ZAPS
    from modules.degradations import get_operator
    from utils.diag_ffhq_regression import TransposeMode
    from utils.diag_imagenet_gaussian_unit_domain import unit_operator
    from utils.diag_guidance_local_probe import evaluate_field,perturb_inputs
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state,same_rng
    from utils.diag_zaps_trace_audit import relative_error
    target=root/case['key'];target.mkdir(exist_ok=False);start=time.perf_counter()
    r=dict(status='running',case=case,source_t=plan['source_t'],plan_sha256=digest(root/'plan.json'),method_nfe=0,diagnostic_nfe=0,calls=[],gates={})
    def save():r['seconds']=time.perf_counter()-start;WRITE(target/'run.json',r)
    def count(_module,inputs):
        r['diagnostic_nfe']+=1;r['calls'].append(dict(t=int(inputs[1].flatten()[0]),attempt=r['diagnostic_nfe']))
        save()
        if r['diagnostic_nfe']>10 or r['calls'][-1]['t']!=plan['source_t']:raise RuntimeError('Call/time budget failed')
    save();hook=model.model.register_forward_pre_hook(count)
    try:
        before=rng_state(device);source=plan['case_sources'][case['key']];folder=Path(source['directory'])
        if digest(folder/'run.json')!=source['record_sha256'] or any(digest(folder/n)!=sha for n,sha in source['artifacts'].items()):raise ValueError('Source changed')
        if digest(source['measurement']['path'])!=source['measurement']['sha256']:raise ValueError('Observation changed')
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))),torch.no_grad():
            state=torch.load(folder/'pre_last_state.pt',map_location='cpu',weights_only=True)
            cache=torch.load(folder/'trajectory.pt',map_location='cpu',weights_only=True)
            k,t,index=plan['source_step'],plan['source_t'],plan['parameter_index']
            if state['tau'].tolist()!=plan['grid'] or read(folder/'run.json')['rows'][k]['t']!=t:raise ValueError('Grid/time differs')
            for name in ('clean','guidance','latent_inputs'):
                a=cache[name]
                if a.shape!=(30,3,256,256) or a.dtype!=torch.float32 or not torch.isfinite(a[k]).all():raise ValueError('Invalid source tensor')
            x=cache['latent_inputs'][k:k+1].to(device);q=cache['guidance'][k:k+1].to(device)
            seed=611020+100*sorted(source_key for source_key in plan['case_sources']).index(case['key'])+t
            o,orth=orthogonal(q.cpu().numpy(),seed)
            directions={'q':q,'orthogonal':torch.from_numpy(o).to(device)}
            operator=unit_operator(TransposeMode(get_operator('super_resolution',device=device,scale_factor=4,noise_sigma=0.),'legacy_bicubic').to(device)).to(device)
            z=ZAPS(model,operator,img_size=256,**plan['config']);z.tau=state['tau'].to(device);z.zeta.copy_(state['zeta'].to(device));z.D.copy_(state['D'].to(device))
            y=torch.load(source['measurement']['path'],map_location='cpu',weights_only=True)['y'].to(device)
            def evaluate(z,y,input_,time_,index_):
                f=evaluate_field(z,y,input_,time_,index_)
                return {n:f[n].cpu() if n in ('clean','q') else f[n] for n in ('clean','q','loss','clip_fraction')}
            with model_rng_guard(model.model,device),patch('torch.randn_like',side_effect=RuntimeError('Unexpected sampling noise')):
                centers,pairs=collect(z,y,x,directions,t,index,evaluate,perturb_inputs)
            checks=dict(clean_cache=relative_error(cache['clean'][k:k+1],centers[0]['clean']),q_cache=relative_error(cache['guidance'][k:k+1],centers[0]['q']),
                clean_repeat=relative_error(centers[0]['clean'],centers[1]['clean']),q_repeat=relative_error(centers[0]['q'],centers[1]['q']))
            if max(checks['clean_cache'],checks['q_cache'])>1e-6 or max(checks['clean_repeat'],checks['q_repeat'])>1e-7 or not all(math.isfinite(v) for v in checks.values()):
                raise RuntimeError('Center cache/repeat parity failed')
            if not(torch.equal(z.tau.cpu(),state['tau']) and torch.equal(z.zeta.cpu(),state['zeta']) and torch.equal(z.D.cpu(),state['D'])):raise RuntimeError('Parameters changed')
            payload=dict(centers=centers,pairs=pairs,directions={n:d.cpu() for n,d in directions.items()})
            result=assess(payload)
            tmp=target/'fields.pt.tmp';torch.save(payload,tmp);tmp.replace(target/'fields.pt')
            r.update(response=result,orthogonal=orth,fields_sha256=digest(target/'fields.pt'),q_rms=rms(q.cpu().numpy()),
                center_clip=centers[0]['clip_fraction'],center_loss=centers[0]['loss'],source_noise_u=read(folder/'run.json')['rows'][k]['noise_u'],
                existing_final_gain={m:statistics.mean((source['positions'][str(t)]['branches'][arm][m]-source['baseline'][m])*(1 if m in ('psnr','ssim') else -1) for arm in ('F0','F1')) for m in ('loss','psnr','ssim','lpips')})
        if not same_rng(before,rng_state(device)) or r['diagnostic_nfe']!=10 or any(digest(p)!=sha for p,sha in plan['worker_identity_files'].items()):raise RuntimeError('RNG/identity/budget failed')
        r['gates']=dict(center=dict(passed=True,checks=checks),orthogonal=dict(passed=True,**orth),parameters_rng_no_sampling=dict(passed=True),budget=dict(passed=True,nfe=10))
        r['status']='complete';save();(target/'run.sha256').write_text(digest(target/'run.json')+'\n')
        print('CASE:',case['key'],'t=',t,'readout=',result['readout'],'q/orth=',result['q_over_orthogonal_gain'],flush=True)
    except BaseException as error:r.update(status='failed',error=repr(error));save();raise
    finally:hook.remove()


def summarize(plan,records):
    done=[r for r in records if r['status']=='complete']
    counts={}
    for label in ('GUIDANCE_MORE_SENSITIVE','ORTHOGONAL_MORE_SENSITIVE'):
        counts[label]=sum(len([r for r in done if r['case']['image_id']==i])==2 and all(r['response']['readout']==label for r in done if r['case']['image_id']==i) for i in ('00023','00024','00025'))
    classification='INCOMPLETE_PILOT' if len(done)!=6 else 'GUIDANCE_ALIGNED_RESPONSE_CANDIDATE' if counts['GUIDANCE_MORE_SENSITIVE']>=2 else 'ORTHOGONAL_RESPONSE_LARGER_CANDIDATE' if counts['ORTHOGONAL_MORE_SENSITIVE']>=2 else 'NO_COHERENT_DIRECTION_SPECIFICITY_ESTABLISHED'
    return dict(experiment=plan['experiment'],source_t=plan['source_t'],completed_cases=len(done),planned_cases=6,
        method_nfe=sum(r['method_nfe'] for r in records),diagnostic_nfe=sum(r['diagnostic_nfe'] for r in records),classification=classification,
        usable_cases=sum(r['response']['usable'] for r in done),image_counts_both_seeds=counts,
        rows=[dict(case=r['case']['key'],q_rms=r['q_rms'],center_clip=r['center_clip'],readout=r['response']['readout'],
            q_over_orthogonal_gain=r['response']['q_over_orthogonal_gain'],gains={n:[v['gain'] for v in d['pairs']] for n,d in r['response']['directions'].items()},
            scale_spread={n:d['relative_vector_spread'] for n,d in r['response']['directions'].items()},
            phi={n:[v['phi'] for v in d['pairs']] for n,d in r['response']['directions'].items()},existing_final_gain=r['existing_final_gain']) for r in done],
        cost_is_lower_bound_if_abruptly_interrupted=any(r['status']=='running' for r in records),
        limits='Three reused images, one orthogonal draw per case. Fixed-time finite-scale clean sensitivity only; not Hessian, step allocation benefit, new reconstruction or generalization.')


def summary(root):
    plan=read(root/'plan.json')
    if digest(root/'plan.json')!=(root/'plan.sha256').read_text().strip():raise ValueError('Plan SHA differs')
    records=[read(p) for p in sorted(root.glob('*/run.json'))]
    if len(records)==6 and all(r['status']=='complete' for r in records):completed(root,plan)
    return summarize(plan,records)


def run(args):
    import fcntl
    root=directory(args.root,args.stage)
    with (root.parent/(root.name+'.lock')).open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Stage already running') from None
        plan=read(root/'plan.json')
        if digest(root/'plan.json')!=(root/'plan.sha256').read_text().strip():raise ValueError('Plan SHA differs')
        active=completed(root,plan)
        try:
            todo=[c for c in plan['cases'] if c['key'] not in active]
            if todo:
                import torch
                torch.set_num_threads(2)
                for p,sha in plan['worker_identity_files'].items():
                    if digest(p)!=sha:raise ValueError('Worker code changed')
                coupled.archive.recovery.require_space(root)
                with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                    model=coupled.archive.recovery.load_model(plan,args.device)
                    for case in todo:run_case(root,plan,case,model,args.device)
        finally:
            out=summary(root);WRITE(root/'summary.json',out);print(STAGES[args.stage][0]+' Summary:',json.dumps(out),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=ROOT)
    parser.add_argument('--a10-dir',default='/data0/lzy/ZAPS-results/diag_clean_roughness/A10_20261020')
    parser.add_argument('--a6-dir',default='/data0/lzy/ZAPS-results/diag_fresh_refinement_confirmation/A6_20261020')
    parser.add_argument('--stage',type=int,choices=list(STAGES));parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--summary',action='store_true');parser.add_argument('--archive',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare(args)
    elif args.summary or args.archive:
        roots={}
        for t,(label,_,_) in STAGES.items():
            root=directory(args.root,t);out=summary(root)
            WRITE(root/'summary.json',out);print(label+' Summary:',json.dumps(out),flush=True);roots[label]=str(root)
        if args.archive:
            from utils.archive_refinement_confirmation import export_and_commit
            bundle,commit=export_and_commit(roots);print('Archive:',bundle,'Results commit:',commit,flush=True)
    elif args.stage is not None:run(args)
    else:parser.error('Choose --prepare, --stage, --summary or --archive')


if __name__=='__main__':main()
