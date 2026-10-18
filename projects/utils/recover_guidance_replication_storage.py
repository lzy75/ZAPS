"""Independent A2a storage recovery. Frozen original entry/card remain unchanged.

Migrate first; --inspect is zero UNet. --repair-case repeats diagnostics only,
using the completed training component. --remaining reuses the registry and runs
only the two not-started original cases. Original interrupted records survive.
"""
import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS))
from utils import diag_guidance_replication as frozen
from utils.diag_state_trajectory import digest

RECOVERY_CARD=PROJECTS/'experiment_cards/A2a_storage_recovery_20261018.md'
write_json=frozen.write_json
validate_plan=frozen.validate_plan


def require_space(root):
    if shutil.disk_usage(root).free<2*1024**3:
        raise OSError('Need at least 2 GiB free on the plan filesystem; migrate before GPU work')
    if os.statvfs(root).f_favail<128:raise OSError('Insufficient free inodes')


def validate_source_metadata(source,plan,plan_sha):
    case=next(c for c in plan['cases'] if c['key']=='00021_s2000')
    if (source.get('case')!=case or source.get('plan_sha256')!=plan_sha or
        source.get('status') not in ('running','failed') or source.get('method_nfe')!=300 or
        source.get('diagnostic_nfe')!=120 or source.get('config')!=plan['config'] or source.get('grid')!=plan['grid']):
        raise ValueError('Need this exact interrupted 00021_s2000 case with completed training')
    for key in ('training','observer','archive','cached_geometry'):
        if source.get('gates',{}).get(key,{}).get('passed') is not True:
            raise ValueError('Missing original component gate: '+key)
    calls=source.get('calls',[])
    if ([c['t'] for c in calls if c['phase']=='method']!=plan['grid'][::-1]*10 or
        len([c for c in calls if c['phase']!='method'])!=120 or
        len(source.get('rows',[]))!=30 or len(source.get('probes',[]))!=5 or
        len(source.get('loss_history',[]))!=10 or not all(math.isfinite(v) for v in source['loss_history'])):
        raise ValueError('Incomplete original call/trajectory/training-component record')


def component_files(root,plan):
    source_dir=root/'00021_s2000';path=source_dir/'run.json'
    source=json.loads(path.read_text());validate_source_metadata(source,plan,digest(root/'plan.json'))
    observation=Path(source['measurement']['path'])
    if observation.resolve()!=(root/'00021_measurement.pt').resolve():raise ValueError('Unexpected observation source')
    if digest(observation)!=source['measurement']['sha256']:raise ValueError('Original observation hash changed')
    paths=[path,source_dir/'pre_last_state.pt',source_dir/'last_opt.pt',observation]
    for p in paths[1:]:
        with zipfile.ZipFile(p) as archive:
            bad=archive.testzip()
            if bad is not None:raise ValueError('Tensor ZIP CRC failed: '+str(p)+' / '+bad)
    return source,source_dir,{str(p):digest(p) for p in paths}


def validate_tensors(state,archived,y,plan):
    import torch
    shapes={'zeta':(30,),'D':(30,3,256,256),'init_noise':(1,3,256,256)}
    for name,shape in shapes.items():
        v=state[name]
        if tuple(v.shape)!=shape or v.dtype!=torch.float32 or not torch.isfinite(v).all():
            raise ValueError('Invalid PRE tensor: '+name)
    if (state['tau'].tolist()!=plan['grid'] or not (state['zeta']>0).all() or
        tuple(archived.shape)!=(1,3,256,256) or archived.dtype!=torch.float32 or not torch.isfinite(archived).all() or
        tuple(y.shape)!=(1,3,64,64) or y.dtype!=torch.float32 or not torch.isfinite(y).all()):
        raise ValueError('PRE/output/measurement shape, grid or finite check failed')
    if not isinstance(state.get('cuda_rng'),list) or not state['cuda_rng'] or 'cpu_rng' not in state:
        raise ValueError('Missing original sampling RNG')


def compare_source_rows(original,current):
    if len(original)!=30 or len(current)!=30:raise ValueError('Need original 30 rows')
    for a,b in zip(original,current):
        if any(not math.isfinite(r[k]) for r in (a,b) for k in ('noise_u','residual_norm','raw_x0_clip_fraction','guidance_rms')):
            raise ValueError('Non-finite trajectory scalar')
        if any(a[k]!=b[k] for k in ('step','t','t_prev','parameter_index')):
            raise ValueError('Recovered trajectory changed original grid/index')
        if abs(a['noise_u']-b['noise_u'])>1e-10:raise ValueError('Recovered noise coordinate differs')
        if abs(a['residual_norm']-b['residual_norm'])>1e-6+2e-5*abs(a['residual_norm']):
            raise ValueError('Recovered residual differs from saved scalar trajectory')
        if abs(a['raw_x0_clip_fraction']-b['raw_x0_clip_fraction'])>1e-5:
            raise ValueError('Recovered clipping differs')
        for k in ('cosine_clean','cosine_latent','guidance_cosine','physical_cosine'):
            if ((a[k] is None)!=(b[k] is None) or
                a[k] is not None and (not math.isfinite(a[k]) or not math.isfinite(b[k]) or abs(a[k]-b[k])>2e-4)):
                raise ValueError('Recovered adjacent direction differs: '+k)
        if not math.isclose(a['guidance_rms'],b['guidance_rms'],rel_tol=1e-6,abs_tol=1e-12):
            raise ValueError('Recovered q magnitude differs')


def registry_for(root):
    path=root/'recovery_registry.json'
    if path.exists():
        registry=json.loads(path.read_text())
        if (registry['plan_sha256']!=digest(root/'plan.json') or
            registry['implementation_sha256']!=digest(__file__) or registry['card_sha256']!=digest(RECOVERY_CARD)):
            raise ValueError('Recovery registry/source/card identity changed')
        return registry
    return dict(plan_sha256=digest(root/'plan.json'),implementation_sha256=digest(__file__),
                card_sha256=digest(RECOVERY_CARD),active={},attempts=[])


def reusable_recovery(root,entry):
    directory=(root/entry['directory']).resolve()
    if not directory.is_relative_to(root/'recovery_runs'):raise ValueError('Recovery path escapes plan')
    path=directory/'run.json'
    if digest(path)!=entry['run_sha256']:raise ValueError('Registered recovery record changed')
    r=json.loads(path.read_text())
    if (r['status']!='complete' or r['plan_sha256']!=digest(root/'plan.json') or
        r['recovery_implementation_sha256']!=digest(__file__) or r['recovery_card_sha256']!=digest(RECOVERY_CARD) or
        r['case']['key']!='00021_s2000' or r['method_nfe']!=0 or
        r['diagnostic_nfe']!=120 or r['reused_method_nfe']!=300 or not r['gates'] or
        not all(g['passed'] for g in r['gates'].values())):raise ValueError('Recovery is not complete/gate-passing')
    for name,h in r['artifacts_sha256'].items():
        if digest(directory/name)!=h:raise ValueError('Recovery artifact changed: '+name)
    for name,h in r['source_component_sha256'].items():
        if digest(name)!=h:raise ValueError('Interrupted source component changed: '+name)
    if digest(r['measurement']['path'])!=r['measurement']['sha256']:raise ValueError('Recovery measurement changed')
    return r


def load_active(root,plan,registry,allow_interrupted=False):
    result={}
    for case in plan['cases']:
        key=case['key']
        if key in registry['active']:result[key]=reusable_recovery(root,registry['active'][key])
        elif allow_interrupted and key=='00021_s2000':result[key]=None
        else:result[key]=frozen.reusable(root/key,digest(root/'plan.json'),key)
    return result


def stage_summary(root,plan,registry,active):
    # Scientific rows use one active diagnostic per case. Cost includes the
    # original interrupted case AND every registered recovery attempt.
    scientific=frozen.summarize_cases([r for r in active.values() if r is not None])
    original=[]
    for case in plan['cases']:
        path=root/case['key']/'run.json'
        if path.exists():original.append(json.loads(path.read_text()))
    attempts=[]
    for item in registry['attempts']:
        path=root/item['directory']/'run.json'
        record=json.loads(path.read_text()) if path.exists() else dict(method_nfe=0,diagnostic_nfe=0,status=item['status'])
        attempts.append(dict(directory=item['directory'],status=record['status'],method_nfe=record['method_nfe'],diagnostic_nfe=record['diagnostic_nfe']))
    scientific.update(method_nfe=sum(r['method_nfe'] for r in original)+sum(r['method_nfe'] for r in attempts),
        diagnostic_nfe=sum(r['diagnostic_nfe'] for r in original)+sum(r['diagnostic_nfe'] for r in attempts),
        nominal_planned_method_nfe=1800,nominal_planned_diagnostic_nfe=720,recovery_attempts=attempts,
        expected_total_with_one_successful_recovery=2640,
        cost_scope='All canonical case attempts including interrupted fourth + every recovery attempt; reused components counted once')
    return scientific


def recover_case(plan, source, source_dir, source_files, directory, model, device, measurement_path):
    import torch
    from modules.zaps_algorithm import ZAPS
    from modules.main_single import load_image_as_tensor
    from modules.degradations import get_operator
    from modules.dataset_loader import tensor_to_image
    from utils.diag_ffhq_regression import TransposeMode
    from utils.diag_imagenet_gaussian_unit_domain import unit_operator
    from utils.diag_zaps_sampler_parity import rng_state,same_rng
    from utils.diag_zaps_guidance_clipping import restore_sampling_rng,saved_cuda_selection
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_trace_audit import relative_error,classify_parity
    from utils.diag_state_trajectory import capture_clean,make_rows
    from utils.diag_cached_direction import cached_step
    from utils.diag_cached_guidance_geometry import vector_readouts,physical_descent,rms,unit_direction
    from utils.diag_guidance_local_probe import select_snapshots,evaluate_field,perturb_inputs,pair_readout,classify_snapshot,ETAS
    from utils.metrics import compute_all_metrics
    case=source['case']
    directory.mkdir(exist_ok=False)
    record=dict(experiment='guidance_replication_recovery',status='running',case=case,plan_sha256=digest(source_dir.parent/'plan.json'),
                method_nfe=0,diagnostic_nfe=0,gates={},rows=[],probes=[],calls=[],config=plan['config'],grid=plan['grid'],
                git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECTS,text=True).strip(),
                runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,python=sys.version,gpu=torch.cuda.get_device_name(torch.device(device)),
                    cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
                    cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
    record.update(source_run=str(source_dir/'run.json'),source_component_sha256=source_files,
        reused_method_nfe=300,prior_interrupted_diagnostic_nfe=source['diagnostic_nfe'],
        recovery_implementation_sha256=digest(__file__),recovery_card_sha256=digest(RECOVERY_CARD),
        original_experiment_card_sha256=plan['experiment_card_sha256'])
    started=time.perf_counter()
    def save():
        record['seconds']=time.perf_counter()-started;write_json(directory/'run.json',record)
    save()
    try:
        base=get_operator('super_resolution',device=device,scale_factor=4,noise_sigma=0.)
        operator=unit_operator(TransposeMode(base,'legacy_bicubic').to(device)).to(device)
        gt=load_image_as_tensor(case['image']).cpu()
        state=torch.load(source_dir/'pre_last_state.pt',map_location='cpu',weights_only=True)
        archived=torch.load(source_dir/'last_opt.pt',map_location='cpu',weights_only=True)
        y=torch.load(measurement_path,map_location='cpu',weights_only=True)['y'].to(device)
        validate_tensors(state,archived,y,plan)
        record['measurement']=dict(path=str(measurement_path),sha256=digest(measurement_path))
        record['gates']['training_component_reuse']=dict(passed=True,source_training_gate=source['gates']['training'],
            scope='Previously completed training component only; source full case was interrupted; no optimizer run')
        record['reused_loss_history']=source['loss_history']
        shutil.copyfile(source_dir/'pre_last_state.pt',directory/'pre_last_state.pt')
        shutil.copyfile(source_dir/'last_opt.pt',directory/'last_opt.pt')
        phase=['replay'];context={}
        def count(_module,inputs):
            record['diagnostic_nfe']+=1
            if record['diagnostic_nfe']>120:raise RuntimeError('Recovery 120 diagnostic NFE cap exceeded')
            t=int(inputs[1].flatten()[0])
            record['calls'].append(dict(phase=phase[0],t=t,cumulative=record['diagnostic_nfe'],**context))
            if phase[0]=='probe' and t!=context['frozen_t']:raise RuntimeError('Probe changed frozen t')
        hook=model.model.register_forward_pre_hook(count)
        try:
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
        (directory/'experiment_card.md').write_text(RECOVERY_CARD.read_text(encoding='utf-8'),encoding='utf-8')
        validate_plan(plan)
        if digest(measurement_path)!=record['measurement']['sha256']:raise RuntimeError('Shared observation changed')
        if any(digest(p)!=h for p,h in source_files.items()):raise RuntimeError('Interrupted source component changed')
        compare_source_rows(source['rows'],rows)
        record['gates']['identities_and_budget']=dict(passed=True,method_nfe=0,diagnostic_nfe=120,reused_method_nfe=300)
        record['artifacts_sha256']={p.name:digest(p) for p in directory.iterdir() if p.is_file() and p.name!='run.json'}
        record['status']='complete';save()
        print('CASE:',case['key'],'metrics:',record['final_metrics'],'valid_q:',record['guidance_valid'],
              'usable:',record['usable_snapshots'],'descents:',record['finite_scale_descents'],
              'ascents:',record['finite_scale_ascents'],'method/diagnostic NFE:',record['method_nfe'],record['diagnostic_nfe'],flush=True)
        return record
    except BaseException as error:
        record['status']='failed';record['error']=repr(error)
        try:save()
        except OSError as saving_error:print('EMERGENCY RECOVERY RECORD:',json.dumps(record),'SAVE ERROR:',repr(saving_error),flush=True)
        raise


def load_model(plan,device):
    import torch
    from modules.main_single import load_diffusion_model
    if not device.startswith('cuda') or not torch.cuda.is_available():raise RuntimeError('Recovery/remaining cases require CUDA')
    model=load_diffusion_model('ffhq',device);model.model.eval()
    if (digest(model.ckpt_path)!=plan['checkpoint_sha256'] or model.num_steps!=1000 or
        any(p.dtype!=torch.float32 for p in model.model.parameters())):raise ValueError('Model identity/dtype changed')
    return model


def print_summary(root,plan,registry,active):
    summary=stage_summary(root,plan,registry,active)
    try:write_json(root/'recovery_summary.json',summary)
    except OSError as error:print('SUMMARY WRITE ERROR:',repr(error),flush=True)
    print('A2a Recovery Summary:',json.dumps({k:v for k,v in summary.items() if k!='same_t_descriptive_associations'},ensure_ascii=False),flush=True)
    for r in active.values():
        if r is not None:frozen.print_case(r)
    print('Summary:',root/'recovery_summary.json',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--inspect',action='store_true')
    group.add_argument('--repair-case',choices=['00021_s2000'])
    group.add_argument('--remaining',action='store_true')
    parser.add_argument('--plan-dir',required=True);parser.add_argument('--device',default='cuda')
    args=parser.parse_args();root=Path(args.plan_dir).resolve()
    plan=json.loads((root/'plan.json').read_text());validate_plan(plan)
    registry=registry_for(root)
    active=load_active(root,plan,registry,allow_interrupted=not args.remaining)
    if args.inspect:
        source,source_dir,files=component_files(root,plan)
        import torch
        state=torch.load(source_dir/'pre_last_state.pt',map_location='cpu',weights_only=True)
        output=torch.load(source_dir/'last_opt.pt',map_location='cpu',weights_only=True)
        y=torch.load(source['measurement']['path'],map_location='cpu',weights_only=True)['y']
        validate_tensors(state,output,y,plan)
        print('INSPECT PASS: original plan/source/measurement hashes; completed-case artifact hashes; PRE/output ZIP CRC and CPU tensor contract.',flush=True)
        print('Active complete cases:',[k for k,v in active.items() if v is not None],flush=True)
        print('Recovery component fingerprints:',json.dumps(files),flush=True)
        print('Plan filesystem free GiB:',shutil.disk_usage(root).free/1024**3,'; UNet NFE=0',flush=True)
        return
    require_space(root)
    if args.repair_case:
        key=args.repair_case
        if key in registry['active']:
            active=load_active(root,plan,registry);print_summary(root,plan,registry,active)
            print('Registered recovery reused; new UNet NFE=0.',flush=True);return
        if registry['attempts']:raise ValueError('An earlier recovery attempt exists; inspect it, no automatic extra attempt')
        source,source_dir,files=component_files(root,plan)
        import torch
        state=torch.load(source_dir/'pre_last_state.pt',map_location='cpu',weights_only=True)
        output=torch.load(source_dir/'last_opt.pt',map_location='cpu',weights_only=True)
        y=torch.load(source['measurement']['path'],map_location='cpu',weights_only=True)['y']
        validate_tensors(state,output,y,plan);del state,output,y
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        directory=root/'recovery_runs'/(key+'_'+stamp);directory.parent.mkdir(exist_ok=True)
        attempt=dict(directory=str(directory.relative_to(root)),status='started')
        registry['attempts'].append(attempt);write_json(root/'recovery_registry.json',registry)
        try:
            model=load_model(plan,args.device)
            result=recover_case(plan,source,source_dir,files,directory,model,args.device,Path(source['measurement']['path']))
            registry['active'][key]=dict(directory=str(directory.relative_to(root)),run_sha256=digest(directory/'run.json'))
            attempt['status']='complete';active[key]=result
        except BaseException:
            attempt['status']='failed';raise
        finally:
            try:write_json(root/'recovery_registry.json',registry)
            except OSError as error:print('REGISTRY WRITE ERROR:',repr(error),flush=True)
            print_summary(root,plan,registry,active)
    else:
        if '00021_s2000' not in registry['active']:raise ValueError('Repair the interrupted fourth case before --remaining')
        todo=[c for c in plan['cases'] if active[c['key']] is None]
        try:
            model=load_model(plan,args.device) if todo else None
            for case in todo:
                require_space(root)
                print('Running original frozen case:',case['key'],'300 method +120 diagnostic NFE',flush=True)
                active[case['key']]=frozen.run_case(plan,case,root/case['key'],model,args.device,root/(case['image_id']+'_measurement.pt'))
        finally:print_summary(root,plan,registry,active)


if __name__=='__main__':main()
