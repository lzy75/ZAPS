"""A13 matched-input FP32 precision-policy metrology; at most 40 diagnostic NFE."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time
import numpy as np

PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS))
from utils import diag_clean_field_response as parent
from utils.diag_state_trajectory import digest
WRITE=parent.WRITE
ROOT='/data0/lzy/ZAPS-results/diag_clean_response_precision/A13_20261020'
CASE='00024_s1000'
NEW_FLAGS=('backends.fp32_precision','backends.cuda.matmul.fp32_precision','backends.cudnn.fp32_precision','backends.cudnn.conv.fp32_precision')


def get(obj,path):
    for key in path.split('.'):obj=getattr(obj,key)
    return obj


def put(obj,path,value):
    keys=path.split('.')
    for key in keys[:-1]:obj=getattr(obj,key)
    setattr(obj,keys[-1],value)


def precision_state(torch):
    try:flags={p:get(torch,p) for p in NEW_FLAGS}
    except AttributeError:
        return dict(api='legacy',matmul=bool(torch.backends.cuda.matmul.allow_tf32),cudnn=bool(torch.backends.cudnn.allow_tf32),
                    matmul_precision=torch.get_float32_matmul_precision())
    return dict(api='per_backend',flags=flags)


@contextmanager
def ieee(torch):
    saved=precision_state(torch)
    try:
        if saved['api']=='per_backend':
            for path in NEW_FLAGS:put(torch,path,'ieee')
        else:
            torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
        yield precision_state(torch)
    finally:
        if saved['api']=='per_backend':
            for path,value in saved['flags'].items():put(torch,path,value)
        else:
            torch.set_float32_matmul_precision(saved['matmul_precision']);torch.backends.cudnn.allow_tf32=saved['cudnn']
        if precision_state(torch)!=saved:raise RuntimeError('Precision policy not restored')


def vector_replay_error(current,source):
    errors=[]
    for a,b in zip(current['pairs'],source['pairs']):
        if (a['direction'],a['eta'])!=(b['direction'],b['eta']):raise ValueError('Different direction/scale order')
        def vector(p):return (np.asarray(p['plus']['clean'],dtype=np.float64)-np.asarray(p['minus']['clean'],dtype=np.float64))/p['input']['effective_span']
        ja,jb=vector(a),vector(b)
        errors.append(parent.rms(ja-jb)/max(parent.rms(ja),parent.rms(jb),1e-30))
    return max(errors)


def classify(native,full,reproduced,changed):
    if not changed:return 'NO_POLICY_CHANGE_ALREADY_IEEE'
    if not reproduced:return 'NATIVE_RESPONSE_NOT_REPRODUCED'
    if not full['usable']:return 'IEEE_FP32_NOT_SUFFICIENT'
    a=native['directions']['orthogonal']['relative_vector_spread'];b=full['directions']['orthogonal']['relative_vector_spread']
    if not native['usable'] and a is not None and b is not None and b<=.5*a:return 'IEEE_POLICY_IMPROVES_METROLOGY_THIS_CASE'
    return 'OTHER_METROLOGY_OUTCOME'


def prepare(args):
    sources={};identities={};plans={}
    for t in (14,54):
        folder=parent.directory(args.parent_root,t);plan=parent.read(folder/'plan.json')
        if digest(folder/'plan.json')!=(folder/'plan.sha256').read_text().strip():raise ValueError('Parent plan changed')
        active=parent.completed(folder,plan);summary=parent.summary(folder)
        if len(active)!=6 or summary!=parent.read(folder/'summary.json') or summary['usable_cases']!=0:raise ValueError('Need declared complete A11/A12 context')
        r=active[CASE];sources[str(t)]=dict(stage=str(folder),record_sha256=digest(folder/CASE/'run.json'),fields_sha256=r['fields_sha256'])
        identities.update(plan['identity_files']);plans[str(t)]=plan
        for name in ('plan.json','plan.sha256','summary.json'):
            identities[str(folder/name)]=digest(folder/name)
        for key in active:
            for name in ('run.json','run.sha256'):identities[str(folder/key/name)]=digest(folder/key/name)
    for path,sha in identities.items():
        if digest(path)!=sha:raise ValueError('Parent identity changed: '+path)
    own=[Path(__file__),PROJECTS/'utils/test_diag_clean_response_precision.py',PROJECTS/'run_a13_20261020.sh',PROJECTS/'experiment_cards/A13_clean_response_precision_20261020.md']
    own_sha={str(p):digest(p) for p in own};identities.update(own_sha)
    root=Path(args.root).resolve()
    if root.is_relative_to(Path(args.parent_root).resolve()) or Path(args.parent_root).resolve().is_relative_to(root):raise ValueError('New root overlaps sources')
    plan=dict(experiment='A13_clean_response_precision',case=CASE,sources=sources,parent_plans=plans,identity_files=identities,
              worker_identity_files=own_sha,planned_method_nfe=0,planned_diagnostic_nfe_ceiling=40,etas=list(parent.ETAS))
    if root.exists():
        if parent.read(root/'plan.json')!=plan:raise ValueError('Frozen A13 differs; do not overwrite')
    else:
        root.mkdir(parents=True,exist_ok=False);WRITE(root/'plan.json',plan);(root/'plan.sha256').write_text(digest(root/'plan.json')+'\n')
        (root/'experiment_card.md').write_text(own[-1].read_text())
    print('A13 PREPARE PASS: two fixed states, unchanged scales/directions; <=40 diagnostic NFE.',flush=True)


def existing(root,t):
    path=root/f't{t}'/'run.json'
    if not path.exists():
        if path.parent.exists():raise ValueError('Interrupted A13 task retained')
        return None
    r=parent.read(path)
    if (r['status']!='complete' or digest(path)!=(path.parent/'run.sha256').read_text().strip() or
        r['plan_sha256']!=digest(root/'plan.json') or r['source_t']!=t or r['method_nfe']!=0 or
        r['diagnostic_nfe'] not in (10,20) or len(r['calls'])!=r['diagnostic_nfe'] or
        not r['gates'] or not all(v['passed'] for v in r['gates'].values()) or digest(path.parent/'fields.pt')!=r['fields_sha256']):
        raise ValueError('Failed/interrupted/incompatible A13 retained')
    if classify(r['native'],r['ieee'],r['native_reproduced'],r['policy_changed'])!=r['classification']:raise ValueError('Classification differs')
    return r


def run(args):
    import fcntl,os
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
    root=Path(args.root).resolve();t=args.stage
    if digest(root/'plan.json')!=(root/'plan.sha256').read_text().strip():raise ValueError('Plan changed')
    plan=parent.read(root/'plan.json');p=plan['parent_plans'][str(t)];source=plan['sources'][str(t)]
    with (root/f't{t}.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Task already running') from None
        if existing(root,t) is not None:return
        target=root/f't{t}';target.mkdir(exist_ok=False);start=time.perf_counter()
        r=dict(status='running',source_t=t,case=CASE,plan_sha256=digest(root/'plan.json'),method_nfe=0,diagnostic_nfe=0,calls=[],gates={})
        def save():r['seconds']=time.perf_counter()-start;WRITE(target/'run.json',r)
        save();hook=None;arm='native'
        try:
            for path,sha in plan['worker_identity_files'].items():
                if digest(path)!=sha:raise ValueError('Code changed')
            src=Path(source['stage'])/CASE
            if digest(src/'run.json')!=source['record_sha256'] or digest(src/'fields.pt')!=source['fields_sha256']:raise ValueError('Source response changed')
            case_source=p['case_sources'][CASE];folder=Path(case_source['directory'])
            if any(digest(folder/n)!=sha for n,sha in case_source['artifacts'].items()) or digest(case_source['measurement']['path'])!=case_source['measurement']['sha256']:
                raise ValueError('Source tensors changed')
            parent.coupled.archive.recovery.require_space(root);torch.set_num_threads(2)
            r['runtime']=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(args.device),
                NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
            policy_before=precision_state(torch);before=rng_state(args.device)
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))),torch.no_grad():
                model=parent.coupled.archive.recovery.load_model(p,args.device)
                def count(_module,inputs):
                    r['diagnostic_nfe']+=1;r['calls'].append(dict(arm=arm,t=int(inputs[1].flatten()[0])));save()
                    if r['diagnostic_nfe']>20 or r['calls'][-1]['t']!=t:raise RuntimeError('Call/time budget failed')
                hook=model.model.register_forward_pre_hook(count)
                state=torch.load(folder/'pre_last_state.pt',map_location='cpu',weights_only=True)
                cache=torch.load(folder/'trajectory.pt',map_location='cpu',weights_only=True)
                saved=torch.load(src/'fields.pt',map_location='cpu',weights_only=True)
                k,index=p['source_step'],p['parameter_index'];x=cache['latent_inputs'][k:k+1].to(args.device)
                if state['tau'].tolist()!=p['grid'] or relative_error(saved['directions']['q'],cache['guidance'][k:k+1])>1e-7:raise ValueError('State/direction differs')
                directions={n:d.to(args.device) for n,d in saved['directions'].items()}
                operator=unit_operator(TransposeMode(get_operator('super_resolution',device=args.device,scale_factor=4,noise_sigma=0.),'legacy_bicubic').to(args.device)).to(args.device)
                z=ZAPS(model,operator,img_size=256,**p['config']);z.tau=state['tau'].to(args.device);z.zeta.copy_(state['zeta'].to(args.device));z.D.copy_(state['D'].to(args.device))
                y=torch.load(case_source['measurement']['path'],map_location='cpu',weights_only=True)['y'].to(args.device)
                def evaluate(z,y,input_,time_,index_):
                    f=evaluate_field(z,y,input_,time_,index_)
                    return {n:f[n].cpu() if n in ('clean','q') else f[n] for n in ('clean','q','loss','clip_fraction')}
                def probe():
                    centers,pairs=parent.collect(z,y,x,directions,t,index,evaluate,perturb_inputs)
                    if max(relative_error(centers[0][n],centers[1][n]) for n in ('clean','q'))>1e-7:raise RuntimeError('Repeat gate failed')
                    return dict(centers=centers,pairs=pairs,directions=saved['directions'])
                r['native_policy']=precision_state(torch)
                with model_rng_guard(model.model,args.device),patch('torch.randn_like',side_effect=RuntimeError('Unexpected sampling noise')):
                    native=probe()
                    drift=max(relative_error(native['centers'][0][n],saved['centers'][0][n]) for n in ('clean','q'))
                    if drift>1e-6:raise RuntimeError('Native center differs from source')
                    arm='ieee'
                    with ieee(torch) as changed:
                        r['ieee_policy']=changed;r['policy_changed']=changed!=r['native_policy']
                        full=probe() if r['policy_changed'] else native
                r['native']=parent.assess(native);r['ieee']=parent.assess(full)
                r['native_vector_replay_error']=vector_replay_error(native,saved);r['native_reproduced']=r['native_vector_replay_error']<=.05
                r['ieee_vs_native_center_relative_error']={n:relative_error(native['centers'][0][n],full['centers'][0][n]) for n in ('clean','q')}
                r['classification']=classify(r['native'],r['ieee'],r['native_reproduced'],r['policy_changed'])
                tmp=target/'fields.pt.tmp';torch.save(dict(native=native,ieee=full),tmp);tmp.replace(target/'fields.pt');r['fields_sha256']=digest(target/'fields.pt')
                if not(torch.equal(z.tau.cpu(),state['tau']) and torch.equal(z.zeta.cpu(),state['zeta']) and torch.equal(z.D.cpu(),state['D'])):raise RuntimeError('Parameters changed')
            expected=20 if r['policy_changed'] else 10
            if r['diagnostic_nfe']!=expected or not same_rng(before,rng_state(args.device)) or policy_before!=precision_state(torch):raise RuntimeError('Budget/RNG/policy restoration failed')
            if any(digest(path)!=sha for path,sha in plan['worker_identity_files'].items()):raise ValueError('Code changed during run')
            r['gates']=dict(source_center=dict(passed=True,max_error=drift),repeats=dict(passed=True),parameters_rng_precision=dict(passed=True),budget=dict(passed=True,nfe=expected))
            r['status']='complete';save();(target/'run.sha256').write_text(digest(target/'run.json')+'\n')
        except BaseException as error:r.update(status='failed',error=repr(error));save();raise
        finally:
            if hook is not None:hook.remove()


def report(root):
    if digest(root/'plan.json')!=(root/'plan.sha256').read_text().strip():raise ValueError('Plan changed')
    rows=[];all_records=[]
    for t in (14,54):
        path=root/f't{t}'/'run.json'
        if not path.exists():continue
        r=parent.read(path);all_records.append(r)
        if r['status']!='complete':continue
        existing(root,t)
        rows.append({**{k:r[k] for k in ('source_t','classification','diagnostic_nfe','native_policy','ieee_policy','policy_changed','native_reproduced','native_vector_replay_error','ieee_vs_native_center_relative_error')},
            'responses':{arm:dict(usable=r[arm]['usable'],q_over_orthogonal_gain=r[arm]['q_over_orthogonal_gain'],
                spread={n:v['relative_vector_spread'] for n,v in r[arm]['directions'].items()}) for arm in ('native','ieee')}})
    nfe=sum(r['diagnostic_nfe'] for r in all_records)
    return dict(completed_tasks=len(rows),planned_tasks=2,method_nfe=0,diagnostic_nfe=nfe,rows=rows,
        cumulative_queue_cost=dict(method_nfe=3600,diagnostic_nfe=2004+nfe,total=5604+nfe),
        cost_is_lower_bound_if_abruptly_interrupted=any(r['status']=='running' for r in all_records),
        limits='One retrospectively selected image/seed at two states; precision-policy metrology only, no reconstruction or state-control benefit.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=ROOT)
    parser.add_argument('--parent-root',default=parent.ROOT);parser.add_argument('--stage',type=int,choices=(14,54));parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--summary',action='store_true');parser.add_argument('--archive',action='store_true');args=parser.parse_args()
    root=Path(args.root).resolve()
    if args.prepare:prepare(args)
    elif args.summary or args.archive:
        out=report(root);WRITE(root/'summary.json',out);print('A13 Summary:',json.dumps(out),flush=True)
        if args.archive:
            from utils.archive_refinement_confirmation import export_and_commit
            bundle,commit=export_and_commit({'A13':str(root)});print('Archive:',bundle,'Results commit:',commit,flush=True)
    elif args.stage is not None:run(args)
    else:parser.error('Choose --prepare, --stage, --summary or --archive')


if __name__=='__main__':main()
