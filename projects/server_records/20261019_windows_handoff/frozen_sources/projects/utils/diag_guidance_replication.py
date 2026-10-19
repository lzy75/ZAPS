"""A2a fixed three-image/two-seed confirmation; independent, no core edits.

--prepare freezes a hash-bound plan without importing Torch or calling UNet.
--case runs the first smoke case; --remaining reuses completed cases exactly.
Each new case: 300 method + 90 replay + 30 frozen-time probe UNet calls.
"""
import argparse
from datetime import datetime
import csv
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from unittest.mock import patch
from zoneinfo import ZoneInfo

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
from utils.diag_state_trajectory import digest

IDS = ('00020', '00021', '00022')
SEEDS = (1000, 2000)
ANCHOR_SHA = '7e60a041cc53f0766a1b657389ead84820a038f7a70daef97088e5674260f2c6'
CARD = PROJECTS/'experiment_cards/A_guidance_replication_20261018.md'


def write_json(path, value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def history_ids(value):
    """Conservative filename audit, excluding copied cards and code text."""
    if isinstance(value, dict):
        result=set().union(*(history_ids(v) for k,v in value.items() if k not in ('experiment_card', 'limits', 'scope')))
        for key in ('image_id','image_ids','image_stem'):
            item=value.get(key,[]);items=item if isinstance(item,list) else [item]
            result.update(v for v in items if isinstance(v,str) and re.fullmatch(r'\d{5}',v))
        return result
    if isinstance(value, list):
        return set().union(*(history_ids(v) for v in value))
    if isinstance(value, str):
        return (set(re.findall(r'(?<!\d)(\d{5})\.(?:png|jpe?g)(?!\w)', value, flags=re.I)) |
                set(re.findall(r'_ffhq_(\d{5})(?=[/_.-]|$)',value,flags=re.I)))
    return set()


def cases_for(data_dir):
    return [dict(key=f'{image}_s{seed}', image=str((Path(data_dir)/(image+'.png')).resolve()),
                 image_id=image, measurement_seed=1000+int(image), sampling_seed=seed)
            for image in IDS for seed in SEEDS]


def audit_history(paths):
    checked=[];hits=[]
    for path in paths:
        value=json.loads(path.read_text(encoding='utf-8'))
        ids=sorted(history_ids(value)|history_ids(str(path)))
        checked.append(dict(path=str(path),sha256=digest(path),image_ids=ids))
        overlap=sorted(set(ids)&set(IDS))
        if overlap:hits.append(dict(path=str(path),image_ids=overlap))
    if not paths or hits:raise ValueError('Historical candidate overlap; stop, do not replace IDs: '+json.dumps(hits))
    return checked


def anchor_inputs(directory):
    from utils.diag_guidance_local_probe import select_geometry, bind_source
    path = Path(directory).resolve()/'run.json'
    if digest(path) != ANCHOR_SHA: raise ValueError('Wrong A1b anchor hash; do not substitute another run')
    record = json.loads(path.read_text())
    if record['status'] != 'complete' or not all(g['passed'] for g in record['gates'].values()):
        raise ValueError('Anchor is not completed/gate-passing')
    geometry_path, geometry, _ = select_geometry(Path(record['source']).parent)
    _, _, _, desc, files = bind_source(geometry_path, geometry)
    files[str(path)] = ANCHOR_SHA
    for name, expected in record['identity_files'].items():
        if digest(name) != expected: raise ValueError('Anchor identity changed: '+name)
        files[name] = expected
    return record, desc, files


def prepare(args):
    print('Checking fixed A1b provenance, source and tensor hashes; UNet NFE=0...',flush=True)
    anchor, desc, files = anchor_inputs(args.anchor_dir)
    roots = [PROJECTS/'results', PROJECTS/'server_records']
    paths = sorted(set(p for root in roots if root.exists() for p in root.rglob('run.json')
                       if 'diag_guidance_replication' not in p.parts))
    # This downloaded historical evidence also covers the old 00000-00019 batches.
    paths += [PROJECTS/'experiment_cards/evidence/STATE_INDICATOR_HISTORY_20261018_recomputed.json']
    checked=audit_history(paths)  # malformed history is not silently ignored
    cases = cases_for(args.data_dir)
    for case in cases:
        case['image_sha256'] = digest(case['image'])
    # Freeze every imported helper used by this entry, including the historical core identities.
    for name in ('utils/diag_guidance_replication.py','utils/diag_state_trajectory.py',
                 'utils/diag_guidance_local_probe.py','utils/diag_cached_guidance_geometry.py',
                 'utils/diag_cached_direction.py','utils/diag_zaps_trace_audit.py',
                 'utils/diag_zaps_first_direction.py','utils/diag_zaps_guidance_clipping.py',
                 'utils/diag_zaps_sampler_parity.py','utils/diag_learning_rate_ablation.py','configs/config.py'):
        files[str(PROJECTS/name)] = digest(PROJECTS/name)
    config = desc['config']
    if (desc['transpose'] != 'legacy_bicubic' or config['lr'] != .01 or config['eta'] != 1 or
            config['num_epochs'] != 10 or config['num_steps'] != 30):
        raise ValueError('Need the fixed historical development protocol')
    plan = dict(experiment='guidance_replication_plan', status='prepared',
                prepared_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECTS,text=True).strip(),
                tracked_worktree_status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=PROJECTS,text=True),
                anchor=str(Path(args.anchor_dir).resolve()/'run.json'), anchor_sha256=ANCHOR_SHA,
                config=config, grid=desc['grid'], checkpoint_sha256=desc['checkpoint_sha'],
                identity_files=files, cases=cases, history_audit=checked,
                history_scope='Accessible local run.json plus downloaded historical evidence only; no final independent-test claim',
                planned_method_nfe=1800, planned_diagnostic_nfe=720,
                experiment_card=CARD.read_text(encoding='utf-8'), experiment_card_sha256=digest(CARD))
    directory=Path(args.plan_dir).resolve()
    if not directory.is_relative_to(PROJECTS/'results/diag_guidance_replication'):
        raise ValueError('Plan must be below projects/results/diag_guidance_replication')
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory/'plan.json',plan)
    print('Prepared:',directory,'\nHistory records checked:',len(checked),'\nNo overlap found in accessible records.',flush=True)
    print('Cases:',json.dumps(cases,ensure_ascii=False),'\nPlan SHA256:',digest(directory/'plan.json'),flush=True)
    print('UNet NFE=0; next run --plan-dir <this directory> --case 00020_s1000',flush=True)


def validate_plan(plan):
    if plan.get('experiment') != 'guidance_replication_plan' or plan.get('status') != 'prepared':
        raise ValueError('Need a frozen prepared A2a plan')
    expected=[(i,s) for i in IDS for s in SEEDS]
    if [(c['image_id'],c['sampling_seed']) for c in plan['cases']] != expected:
        raise ValueError('Case order/sample membership changed')
    for c in plan['cases']:
        if c['key'] != f"{c['image_id']}_s{c['sampling_seed']}" or c['measurement_seed'] != 1000+int(c['image_id']):
            raise ValueError('Frozen image/measurement/sampling seed mismatch')
        if digest(c['image']) != c['image_sha256']: raise ValueError('Candidate image changed')
    for path, expected_sha in plan['identity_files'].items():
        if digest(path) != expected_sha: raise ValueError('Frozen source/tensor/checkpoint changed: '+path)
    if digest(CARD) != plan['experiment_card_sha256']: raise ValueError('Predeclared card changed')
    if plan['anchor_sha256'] != ANCHOR_SHA or digest(plan['anchor']) != ANCHOR_SHA:
        raise ValueError('Anchor changed')
    anchor=json.loads(Path(plan['anchor']).read_text())['resolved_config']
    if (plan['config']!=anchor['config'] or plan['grid']!=anchor['grid'] or
        plan['checkpoint_sha256']!=anchor['checkpoint_sha'] or
        plan['planned_method_nfe']!=1800 or plan['planned_diagnostic_nfe']!=720):
        raise ValueError('Frozen parent protocol/budget changed')


def reusable(directory, plan_sha, key):
    path=directory/'run.json'
    if not path.exists(): return None
    r=json.loads(path.read_text())
    if (r.get('status') != 'complete' or r.get('plan_sha256') != plan_sha or r.get('case',{}).get('key') != key or
            not r.get('gates') or not all(g.get('passed') is True for g in r['gates'].values()) or
            r.get('method_nfe') != 300 or r.get('diagnostic_nfe') != 120):
        raise ValueError('Existing case is incomplete/failed/incompatible; will not overwrite: '+str(path))
    for name, expected in r['artifacts_sha256'].items():
        if digest(directory/name) != expected: raise ValueError('Completed case artifact changed: '+name)
    if digest(r['measurement']['path']) != r['measurement']['sha256']:
        raise ValueError('Completed case shared observation changed')
    return r


def snapshot_class():
    import torch
    from modules.zaps_algorithm import ZAPS
    from utils.diag_zaps_sampler_parity import rng_state
    class SnapshotZAPS(ZAPS):
        audit_calls=0
        audit_state=None
        def _reverse_diffusion(self,y,*a,**kw):
            self.audit_calls+=1
            if self.audit_calls == self.num_epochs:
                r=rng_state(self.device)
                self.audit_state=dict(zeta=self.zeta.detach().cpu().clone(), D=self.D.detach().cpu().clone(),
                    tau=self.tau.detach().cpu().clone(),init_noise=kw['init_noise'].detach().cpu().clone(),
                    cpu_rng=r['cpu'],cuda_rng=r['cuda'],sampling_device=str(self.device))
            return super()._reverse_diffusion(y,*a,**kw)
    return SnapshotZAPS


def run_case(plan, case, directory, model, device, measurement_path):
    import torch
    from modules.zaps_algorithm import ZAPS
    from modules.main_single import load_image_as_tensor
    from modules.degradations import get_operator
    from modules.dataset_loader import tensor_to_image
    from utils.diag_ffhq_regression import TransposeMode
    from utils.diag_imagenet_gaussian_unit_domain import unit_operator
    from utils.diag_learning_rate_ablation import set_seed
    from utils.diag_zaps_sampler_parity import rng_state,same_rng
    from utils.diag_zaps_guidance_clipping import restore_sampling_rng,saved_cuda_selection
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_trace_audit import relative_error,classify_parity
    from utils.diag_state_trajectory import capture_clean,make_rows
    from utils.diag_cached_direction import cached_step
    from utils.diag_cached_guidance_geometry import vector_readouts,physical_descent,rms,unit_direction
    from utils.diag_guidance_local_probe import select_snapshots,evaluate_field,perturb_inputs,pair_readout,classify_snapshot,ETAS
    from utils.metrics import compute_all_metrics
    directory.mkdir(exist_ok=False)
    record=dict(experiment='guidance_replication_case',status='running',case=case,plan_sha256=digest(directory.parent/'plan.json'),
                method_nfe=0,diagnostic_nfe=0,gates={},rows=[],probes=[],calls=[],config=plan['config'],grid=plan['grid'],
                git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECTS,text=True).strip(),
                runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,python=sys.version,gpu=torch.cuda.get_device_name(torch.device(device)),
                    cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
                    cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
    started=time.perf_counter()
    def save():
        record['seconds']=time.perf_counter()-started;write_json(directory/'run.json',record)
    save()
    try:
        base=get_operator('super_resolution',device=device,scale_factor=4,noise_sigma=0.)
        legacy=TransposeMode(base,'legacy_bicubic').to(device)
        operator=unit_operator(legacy).to(device)
        gt=load_image_as_tensor(case['image']).cpu()
        # One measurement per image, shared across sampling seeds; never replace it silently.
        if not measurement_path.exists():
            set_seed(case['measurement_seed'])
            with torch.no_grad():
                truth=gt.to(device);noise=.05*torch.randn_like(operator.H(truth))
                y=operator.H(truth)+noise
                affine=operator.H(truth)-(.5*base.H(truth)+.5*base.H(torch.ones_like(truth)))
                scaling=operator.transpose(noise)-.5*legacy.transpose(noise)
                if max(float(affine.abs().max()),float(scaling.abs().max()))>2e-5:
                    raise RuntimeError('Unit affine/empirical T scaling gate failed')
            torch.save(dict(y=y.cpu(),image_sha256=case['image_sha256'],measurement_seed=case['measurement_seed']),measurement_path)
        measurement=torch.load(measurement_path,map_location='cpu',weights_only=True)
        if measurement['image_sha256']!=case['image_sha256'] or measurement['measurement_seed']!=case['measurement_seed']:
            raise RuntimeError('Shared observation belongs to another image/seed')
        y=measurement['y'].to(device)
        if y.shape!=(1,3,64,64) or y.dtype!=torch.float32 or not torch.isfinite(y).all():
            raise RuntimeError('Invalid shared observation')
        record['measurement']=dict(path=str(measurement_path),sha256=digest(measurement_path))
        z=snapshot_class()(model,operator,img_size=256,**plan['config']);z.tau=torch.tensor(plan['grid'],device=device)
        set_seed(case['sampling_seed']);expected_initial=torch.randn(1,3,256,256,device=device)
        for _ in range(290): torch.randn_like(expected_initial)
        expected_end=rng_state(device)
        phase=['method'];visited=[];context={}
        def count(_module,inputs):
            label=phase[0];key='method_nfe' if label=='method' else 'diagnostic_nfe'
            record[key]+=1
            if record[key]>(300 if key=='method_nfe' else 120): raise RuntimeError('Case NFE cap exceeded')
            t=int(inputs[1].flatten()[0]);record['calls'].append(dict(phase=label,t=t,cumulative=record[key],**context))
            if label=='probe' and t!=context['frozen_t']:raise RuntimeError('Probe changed frozen t')
            if label=='method': visited.append(t)
        hook=model.model.register_forward_pre_hook(count)
        try:
            set_seed(case['sampling_seed'])
            with model_rng_guard(model.model,device) as guarded:
                losses=z.optimize(y,verbose=True,x0_gt=None,scheduler=None)
            valid=(len(guarded)==300 and visited==list(reversed(plan['grid']))*10 and z._last_nfe==300 and
                   len(losses)==10 and all(math.isfinite(v) for v in losses) and z.audit_state is not None and
                   same_rng(expected_end,rng_state(device)) and torch.equal(z._last_opt_noise,expected_initial) and
                   all(p.grad is None for p in model.model.parameters()))
            if not valid: raise RuntimeError('Training grid/RNG/frozen model/300 NFE contract failed')
            record['gates']['training']=dict(passed=True,nfe=300,optimizer_updates=10)
            record['loss_history']=losses
            archived=z._last_opt_x0.detach().cpu();state=z.audit_state
            torch.save(state,directory/'pre_last_state.pt');torch.save(archived,directory/'last_opt.pt')
            del z
            z=ZAPS(model,operator,img_size=256,**plan['config']);z.tau=state['tau'].to(device)
            with torch.no_grad():z.zeta.copy_(state['zeta'].to(device));z.D.copy_(state['D'].to(device))
            mapping=saved_cuda_selection(state,device);captures=[];replays=[]
            phase[0]='replay'
            for which in ('observed','core_a','core_b'):
                context.update(kind=which)
                restore_sampling_rng(state,device,mapping['saved_sampling_index'])
                with model_rng_guard(model.model,device) as guarded:
                    if which=='observed':
                        with capture_clean(z,captures): output,nfe,_=z.sample(y,init_noise=state['init_noise'].to(device))
                        indicators=list(z._indicator_log)
                    else:output,nfe,_=z.sample(y,init_noise=state['init_noise'].to(device))
                replays.append(dict(output=output.cpu(),rng=rng_state(device),nfe=nfe,guarded=len(guarded)))
            observed,a,b=replays
            metadata=(record['diagnostic_nfe']==90 and len(captures)==30 and
                      [c['index'] for c in captures]==list(range(29,-1,-1)) and
                      all(v['nfe']==v['guarded']==30 for v in replays) and
                      same_rng(observed['rng'],a['rng']) and same_rng(a['rng'],b['rng']))
            floor=relative_error(a['output'],b['output'])
            record['gates']['observer']=classify_parity(floor,relative_error(a['output'],observed['output']),metadata)
            record['gates']['archive']=classify_parity(floor,relative_error(a['output'],archived),metadata)
            if not all(g['passed'] for g in record['gates'].values()):raise RuntimeError('Replay parity gate failed')
            rows=make_rows(z,y,observed['output'],captures,indicators,gt)
            # Reconstruct actual q without a score forward, including all 29 original noise draws.
            alpha=model.alphas_cumprod;history={'guidance':[],'physical':[]};cache_errors=[]
            index=torch.device(device).index
            index=torch.cuda.current_device() if index is None else index
            with torch.random.fork_rng(devices=[index]),torch.no_grad():
                restore_sampling_rng(state,device,mapping['saved_sampling_index'])
                for k,row in enumerate(rows):
                    i,t,s=row['parameter_index'],row['t'],row['t_prev']
                    step=cached_step(captures[k]['x'].to(device),captures[k]['clean'].to(device),y,operator,z.dwt,z.zeta[i],z.D[i],t,s,alpha)
                    expected=captures[k+1]['x'] if k<29 else archived
                    error=relative_error(expected,step['next'].cpu());cache_errors.append(error)
                    if not math.isfinite(error) or error>1e-6:raise RuntimeError('Cached transition mismatch')
                    physical,loss=physical_descent(operator,captures[k]['clean'].to(device),y)
                    row.update(measurement_loss=loss,base_guidance_rms=rms(step['correction'])/float(z.zeta[i]),
                        prior_increment_rms=rms(step['prior_increment']),mean_increment_rms=rms(step['mean_increment']),
                        noise_increment_rms=rms(step['noise_increment']))
                    for mode,vector in [('guidance',step['correction']),('physical',physical)]:
                        h=history[mode]
                        v=vector_readouts(vector,h[-1].to(device) if h else None,h[-2].to(device) if len(h)>=2 else None,
                            row['noise_u'],rows[k-1]['noise_u'] if k else None,rows[k-2]['noise_u'] if k>=2 else None)
                        row.update({mode+'_'+key:value for key,value in v.items()});h.append(vector.cpu().clone())
                actual_rng=rng_state(device)
                restore_sampling_rng(state,device,mapping['saved_sampling_index'])
                for _ in range(29):torch.randn_like(state['init_noise'].to(device))
                if not same_rng(actual_rng,rng_state(device)):raise RuntimeError('Cached 29-draw suffix mismatch')
            record['gates']['cached_geometry']=dict(passed=True,max_relative_error=max(cache_errors),noise_draws=29)
            record['rows']=rows
            # Five fixed targets and the unchanged A1b numerical classification.
            snaps=select_snapshots(rows);record['snapshots']=snaps
            phase[0]='probe';before_rng=rng_state(device);fingerprints=[]
            with torch.random.fork_rng(devices=[index]),model_rng_guard(model.model,device),torch.no_grad():
                for snap in snaps:
                    k,t,i=snap['step'],snap['t'],snap['parameter_index']
                    context.clear();context.update(step=k,frozen_t=t,parameter_index=i)
                    x=captures[k]['x'].to(device);q0=history['guidance'][k].to(device)
                    centers=[]
                    for repeat in range(2):
                        context.update(kind='center',repeat=repeat,eta=0.)
                        center=evaluate_field(z,y,x,t,i);centers.append(center)
                        ce=relative_error(captures[k]['clean'],center['clean'].cpu());qe=relative_error(q0,center['q'])
                        le=abs(center['loss']-rows[k]['measurement_loss'])
                        if (max(ce,qe)>1e-6 or not all(math.isfinite(v) for v in (ce,qe,le)) or
                            le>1e-7+2e-5*abs(center['loss']) or abs(center['clip_fraction']-rows[k]['raw_x0_clip_fraction'])>1e-5):
                            raise RuntimeError('Probe center differs from cached state')
                        fingerprints.append(dict(step=k,repeat=repeat,clean_error=ce,q_error=qe,loss_error=le))
                    n0,n1=[unit_direction(c['q']) for c in centers]
                    if n0 is None or n1 is None:
                        floors=dict(direction_chord=0.,q_rms=rms(centers[0]['q']-centers[1]['q']),loss_abs=abs(centers[0]['loss']-centers[1]['loss']))
                    else:floors=dict(direction_chord=float((n0-n1).norm()),q_rms=rms(centers[0]['q']-centers[1]['q']),loss_abs=abs(centers[0]['loss']-centers[1]['loss']))
                    pairs=[]
                    # q0=0 is retained as unresolved, never replaced with a new snapshot.
                    if unit_direction(q0) is None or n0 is None or n1 is None:
                        for eta in ETAS:
                            context.update(kind='plus_unresolved',repeat=None,eta=eta);plus=evaluate_field(z,y,x+eta*q0,t,i)
                            context.update(kind='minus_unresolved');minus=evaluate_field(z,y,x-eta*q0,t,i)
                            pairs.append(dict(eta=eta,effective_span=None,input_resolved=False,input_orthogonal_relative_error=None,
                                input_span_relative_error=None,direction_chord=None,s_dir=None,s_dir_per_state_rms=None,l_path=None,
                                phi=None,loss_difference=plus['loss']-minus['loss'],plus_loss=plus['loss'],minus_loss=minus['loss'],
                                plus_clip=plus['clip_fraction'],minus_clip=minus['clip_fraction'],unresolved_reason='SMALL_OR_REPEAT_DIRECTION_UNRESOLVED'))
                        item=dict(snap,q_rms=rms(q0),center_clip=centers[0]['clip_fraction'],center_loss=centers[0]['loss'],pairs=pairs,repeat_floors=floors,
                                  classification=classify_snapshot(pairs,floors))
                    else:
                        for eta in ETAS:
                            xp,xm,inp=perturb_inputs(x,q0,eta)
                            context.update(kind='plus',repeat=None,eta=eta);plus=evaluate_field(z,y,xp,t,i)
                            context.update(kind='minus');minus=evaluate_field(z,y,xm,t,i)
                            pair=pair_readout(plus,minus,inp,q0,eta,centers[0])
                            pair['s_dir_per_state_rms']=pair['s_dir']/rms(q0) if pair['s_dir'] is not None else None
                            pairs.append(pair)
                        item=dict(snap,q_rms=rms(q0),center_clip=centers[0]['clip_fraction'],center_loss=centers[0]['loss'],pairs=pairs,
                                  repeat_floors=floors,classification=classify_snapshot(pairs,floors))
                    record['probes'].append(item);save()
            frozen=(torch.equal(z.zeta.cpu(),state['zeta']) and torch.equal(z.D.cpu(),state['D']) and torch.equal(z.tau.cpu(),state['tau']))
            if not frozen or not same_rng(before_rng,rng_state(device)) or record['diagnostic_nfe']!=120:
                raise RuntimeError('Probe frozen parameter/RNG/120 diagnostic NFE gate failed')
            record['gates']['probe']=dict(passed=True,center_fingerprints=fingerprints,parameters_grid_frozen=True,rng_unchanged=True)
        finally:hook.remove()
        record['final_metrics']=compute_all_metrics(archived,gt,lpips_net='alex')
        record['guidance_valid']=sum(r['guidance_valid'] for r in rows)
        record['usable_snapshots']=sum(p['classification']['usable'] for p in record['probes'])
        record['finite_scale_descents']=sum(p['classification']['loss_status']=='LOCAL_FINITE_SCALE_DESCENT' for p in record['probes'])
        record['finite_scale_ascents']=sum(p['classification']['loss_status']=='LOCAL_FINITE_SCALE_ASCENT' for p in record['probes'])
        torch.save(dict(clean=torch.cat([c['clean'] for c in captures]),latent_inputs=torch.cat([c['x'] for c in captures]),
                        output=archived,guidance=torch.cat(history['guidance']),physical=torch.cat(history['physical'])),directory/'trajectory.pt')
        tensor_to_image(archived.squeeze(0),denormalize=True).save(directory/'final.png')
        with (directory/'geometry.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (directory/'experiment_card.md').write_text(plan['experiment_card'],encoding='utf-8')
        validate_plan(plan)
        if digest(measurement_path)!=record['measurement']['sha256']:raise RuntimeError('Shared observation changed')
        record['gates']['identities_and_budget']=dict(passed=True,method_nfe=300,diagnostic_nfe=120)
        record['artifacts_sha256']={p.name:digest(p) for p in directory.iterdir() if p.is_file() and p.name!='run.json'}
        record['status']='complete';save()
        print('CASE:',case['key'],'metrics:',record['final_metrics'],'valid_q:',record['guidance_valid'],
              'usable:',record['usable_snapshots'],'descents:',record['finite_scale_descents'],
              'ascents:',record['finite_scale_ascents'],'method/diagnostic NFE:',record['method_nfe'],record['diagnostic_nfe'],flush=True)
        return record
    except Exception as error:
        record['status']='failed';record['error']=repr(error);save();raise


def summarize_cases(records):
    from utils.diag_state_trajectory import pearson,ranks
    complete=[r for r in records if r['status']=='complete']
    supported=sum(r['guidance_valid']>=24 and r['usable_snapshots']>=4 for r in complete)
    links=[]
    for position in range(5):
        for target in ('s_dir','s_dir_per_state_rms'):
            for predictor in ('guidance_variation3_per_u','guidance_rms'):
                for scale in range(2):
                    values=[];ts=[]
                    for run in complete:
                        p=run['probes'][position];row=run['rows'][p['step']]
                        x,y=row[predictor],p['pairs'][scale][target]
                        if x is not None and y is not None and math.isfinite(x) and math.isfinite(y):
                            values.append((x,y));ts.append(p['t'])
                    if len(set(ts))>1:raise ValueError('Same-t descriptive association requires matched snapshots')
                    x,y=zip(*values) if values else ([],[])
                    links.append(dict(snapshot_position=position,t=ts[0] if ts else None,eta=(.01,.02)[scale],predictor=predictor,
                                      target=target,n=len(values),spearman=pearson(ranks(x),ranks(y))))
    return dict(completed_cases=len(complete),planned_cases=6,readable_cases=supported,
                classification=('PILOT_READABILITY_SUPPORTED' if supported>=5 else 'PILOT_READABILITY_NOT_SUPPORTED') if len(complete)==6 else 'INCOMPLETE_PILOT',
                method_nfe=sum(r['method_nfe'] for r in records),diagnostic_nfe=sum(r['diagnostic_nfe'] for r in records),
                rows=[dict(case=r['case']['key'],metrics=r.get('final_metrics'),status=r['status'],valid_q=r.get('guidance_valid'),
                           usable=r.get('usable_snapshots'),descents=r.get('finite_scale_descents'),ascents=r.get('finite_scale_ascents')) for r in records],
                same_t_descriptive_associations=links,
                limits='3 images x 2 seeds; dependent time points. No state intervention, control benefit or improved reconstruction claim. All ascents and unresolved points retained.')


def print_case(record):
    print('CASE:',record['case']['key'],'Gates:',json.dumps(record['gates']),flush=True)
    print(' k    t  qRMS     clip    S_dir(.01/.02)       per_state_RMS(.01/.02)   Phi(.01/.02)       usable',flush=True)
    def num(v):return 'NA' if v is None else f'{v:.6g}'
    for p in record['probes']:
        values=['/'.join(num(q[key]) for q in p['pairs']) for key in ('s_dir','s_dir_per_state_rms','phi')]
        print(p['step'],p['t'],num(p['q_rms']),num(p['center_clip']),*values,p['classification']['usable'],p['classification']['loss_status'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--case',choices=[c['key'] for c in cases_for('.')])
    mode.add_argument('--remaining',action='store_true')
    parser.add_argument('--plan-dir',required=True)
    parser.add_argument('--anchor-dir',default=str(PROJECTS/'results/diag_guidance_local_probe/20261018_162902_899938'))
    parser.add_argument('--data-dir',default='/home/lzy/FFHQ/00000/00000')
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    if args.prepare:prepare(args);return
    directory=Path(args.plan_dir).resolve();plan=json.loads((directory/'plan.json').read_text());validate_plan(plan)
    existing={c['key']:reusable(directory/c['key'],digest(directory/'plan.json'),c['key']) for c in plan['cases']}
    if args.remaining and existing['00020_s1000'] is None:
        raise ValueError('Run and inspect --case 00020_s1000 before --remaining')
    selected=[c for c in plan['cases'] if args.remaining or c['key']==args.case]
    todo=[c for c in selected if existing[c['key']] is None]
    if todo:
        if not args.device.startswith('cuda'):parser.error('Production cases require CUDA')
        import torch
        from modules.main_single import load_diffusion_model
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        model=load_diffusion_model('ffhq',args.device);model.model.eval()
        if (digest(model.ckpt_path)!=plan['checkpoint_sha256'] or model.num_steps!=1000 or
            any(p.dtype!=torch.float32 for p in model.model.parameters())):raise RuntimeError('Model identity/dtype mismatch')
        try:
            for c in todo:
                print('\nStarting:',c['key'],'fixed grid; 300 method + 120 diagnostic NFE',flush=True)
                existing[c['key']]=run_case(plan,c,directory/c['key'],model,args.device,directory/(c['image_id']+'_measurement.pt'))
        finally:
            records=[json.loads((directory/c['key']/'run.json').read_text()) for c in plan['cases'] if (directory/c['key']/'run.json').exists()]
            write_json(directory/'summary.json',summarize_cases(records))
    summary=summarize_cases([r for r in existing.values() if r is not None]);write_json(directory/'summary.json',summary)
    print('\nA2a Summary:',json.dumps({k:v for k,v in summary.items() if k!='same_t_descriptive_associations'},ensure_ascii=False),'\nRecords:',directory,flush=True)
    for r in existing.values():
        if r is not None:
            print_case(r)
    print('All fixed-t descriptive associations saved in summary.json; 3 image clusters, no p-values or control claims.',flush=True)


if __name__=='__main__':main()
