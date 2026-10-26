"""A53 five-image development test of CURRENT cosine information, fixed grid."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from statistics import mean

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import diag_feedback_vector_constraint as old
from utils import angular_feedback_control as rules

parent,source=old.parent,old.source
write,signed=old.write,old.signed
ROOT='/data0/lzy/ZAPS-results/diag_angular_feedback_value/A53_20261026'
CARD=PROJECTS/'experiment_cards/A53_current_cosine_value_20261026.md'
BRANCH='results/ffhq-current-cosine-value-20261026'


def require(test,message):
    if not test:raise ValueError(message)


def check(root):
    p=source.check_plan(root);parent.a42.verify(p)
    require(p['experiment']=='A53_current_cosine_value','Wrong plan');return p


def prepare(args):
    root,prior=Path(args.root).resolve(),Path(args.a52_dir).resolve()
    require(not(root.is_relative_to(prior) or prior.is_relative_to(root)),'Source/output overlap')
    if root.exists():
        p=check(root);require(p['source_A52']==str(prior),'Frozen source differs')
        print('A53 frozen plan reused; completed work retained.',flush=True);return
    before=old.check(prior)
    require(source.digest(prior/'summary.json')=='e908e601135da7b6c3668b51a69a7053ed35b6e5ab0fc5f6a1a5665bcab45585',
            'Need complete audited A52 summary')
    a50=Path(before['source_A50r2']);p50=old.previous.check(a50)
    cases=[c for c in p50['cases'] if c['task'] in rules.TASKS]
    require(len(cases)==10 and {(c['task'],c['image_id']) for c in cases}==
            {(t,i) for t in rules.TASKS for i in rules.IMAGES},'Need frozen five-image/two-task cohort')
    identities=dict(before['identity_files']);reused={};refs={}
    for name in ('plan.json','summary.json'):identities[str(prior/name)]=source.digest(prior/name)
    for c in before['cases']:
        item=old.done(prior,before,c);require(item is not None,'Incomplete A52 source')
        f,r=item;reused[c['key']]=str(f);identities[str(f)]=source.digest(f)
        for n,sha in r['artifacts'].items():identities[str(f.parent/n)]=sha
    for c in cases:
        item=old.previous.phase_done(a50,p50,c,'generate');require(item is not None,'Missing A50 baseline')
        f,r=item;refs[c['key']]=str(f);identities[str(f)]=source.digest(f)
        for n,sha in r['artifacts'].items():identities[str(f.parent/n)]=sha
    own=[Path(__file__),Path(rules.__file__),CARD,PROJECTS/'run_a53_20261026.sh',
         PROJECTS/'utils/test_angular_feedback_control.py',PROJECTS/'utils/experiment_gpu_selection.py']
    p=dict(before)
    p.update(experiment='A53_current_cosine_value',source_A52=str(prior),cases=cases,
        reused_generation=reused,a50_baselines=refs,identity_files=identities,
        worker_identity_files={**before['worker_identity_files'],**{str(f):source.digest(f) for f in own}},
        phases=['generate','control'],arms=['STATE',*rules.CONTROLS],generate_arms=['STATE'],control_arms=list(rules.CONTROLS),
        planned_method_nfe=0,planned_diagnostic_nfe=1080,planned_backward_attempts_max=1080,
        planned_new_generate_cases=6,reused_A52_diagnostic_nfe=480,
        historical_training_nfe=3000,reused_prefix_forward_nfe=2700,
        limits='Five REUSED development image clusters x SR/inpainting, one seed, sigma=.05. '
               'LOO coefficient calibration, not independent confirmation. Every angular arm uses CURRENT '
               'u direction and v/u norms; only c changes. Local v-versus-J^T v cosine, not temporal curvature. '
               'Original first270 and last30 grid unchanged; scientific forwards300 is not equal walltime. '
               'No four-task, step/NFE-saving, new-theorem or state-free full-method claim.')
    root.parent.mkdir(parents=True,exist_ok=True);parent.P.coupled.archive.recovery.require_space(root.parent)
    root.mkdir();signed(root/'plan.json',p);shutil.copyfile(CARD,root/'experiment_card.md')
    snap=root/'source_snapshot';snap.mkdir()
    for n in ('plan.json','summary.json'):shutil.copyfile(prior/n,snap/('A52_'+n))
    for c in before['cases']:shutil.copyfile(reused[c['key']],snap/(c['key']+'_A52_run.json'))
    print('A53 PREPARE PASS:10 cases,4 STATE trajectories reused;6x30 generation +10x90 controls=1080 new forwards,<=1080 backwards.',flush=True)


def item(root,p,c,phase):
    if phase=='generate' and c['key'] in p['reused_generation']:
        f=Path(p['reused_generation'][c['key']]);r=source.load_signed(f)
        require(source.digest(f)==p['identity_files'][str(f)] and r['status']=='complete' and r['case']==c and
                all(v['passed'] for v in r['gates'].values()),'Invalid reused STATE source')
        require(all(source.digest(f.parent/n)==sha for n,sha in r['artifacts'].items()),'Reused artifact differs')
        return f,r,'FULL_ALIGN'
    paths=sorted((root/c['key']).glob(phase+'_*/run.json'))
    if not paths:return None
    require(len(paths)==1,'Multiple attempts; no selection')
    f=paths[0];r=source.read(f)
    if r['status']!='complete':return None
    r=source.load_signed(f);names=['STATE'] if phase=='generate' else list(rules.CONTROLS)
    require(r['case']==c and r['plan_sha256']==source.digest(root/'plan.json') and
            r['calls']==list(reversed(p['grid']))*len(names) and r['nfe']==30*len(names),'Paid call contract differs')
    require(set(r['traces'])==set(names) and all(len(v)==30 for v in r['traces'].values()),'Incomplete traces')
    require(r['backward_attempts']==r['backward_completed']<=r['nfe'] and
            r['gates'] and all(v['passed'] for v in r['gates'].values()),'Failed GPU gate')
    require(all(source.digest(f.parent/n)==sha for n,sha in r['artifacts'].items()),'Output changed')
    if phase=='control':require(r['allocation_sha256']==source.digest(root/'allocation.json'),'Allocation changed')
    return f,r,'STATE'


def freeze(root,p):
    curves={};sources={};errors={}
    for c in p['cases']:
        found=item(root,p,c,'generate');require(found is not None,'Need all10 STATE trajectories')
        f,r,key=found;sources[str(f)]=source.digest(f);values=[];maximum=0.
        for row in r['traces'][key]:
            require(row['readout'].get('resolved'),f"Unresolved STATE {c['key']} step{row['step']}; controls not started")
            z=row['readout'];cos=z['cosine_v_u']
            factor=z['v_RMS']/z['u_RMS']*cos
            error=abs(factor-z['alignment_scale'])/max(abs(z['alignment_scale']),1e-30)
            require(error<=2e-6,'A52 coefficient factorization differs');maximum=max(maximum,error)
            values.append(cos)
        curves[c['key']]=values;errors[c['key']]=maximum
    value=dict(plan_sha256=source.digest(root/'plan.json'),sources=sources,
               profiles=rules.allocate(p['cases'],curves),STATE_scalar_factorization_errors=errors,GT_used=False)
    path=root/'allocation.json'
    if path.exists():require(source.load_signed(path)==value,'Frozen coefficients differ')
    else:signed(path,value)
    print('A53 FREEZE PASS:10 STATE curves, held image excluded from TIME_COS; c-only cyclic shuffle; current direction/norms retained; no labels.',flush=True)


def sample(root,p,c,phase,model,device):
    import torch
    from unittest.mock import patch
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state
    if item(root,p,c,phase):return
    operator,mask,opgate,state,original,y,z,positions,noises=parent.setup(p,c,model,device)
    folder,r=old.previous.new_attempt(root,p,c,phase);path=folder/'run.json';external=rng_state(device)
    allocation=None
    if phase=='control':
        allocation=source.load_signed(root/'allocation.json')
        require(all(source.digest(Path(f))==sha for f,sha in allocation['sources'].items()),'Calibration source changed')
        r['allocation_sha256']=source.digest(root/'allocation.json')
    names=['STATE'] if phase=='generate' else list(rules.CONTROLS);outputs={}
    with old.previous.counted(model,path,r,30*len(names)) as before_backward, \
         old.previous.runtime.input_vjp_parameters(model.model),model_rng_guard(model.model,device), \
         patch('torch.randn_like',side_effect=RuntimeError('Unexpected random draw')):
        for name in names:
            torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device);started=time.perf_counter()
            x=state['init_noise'].to(device);r['traces'][name]=[]
            with parent.P.precision.ieee(torch):
                for k,t in enumerate(positions.nominal):
                    s=positions.nominal[k+1] if k<29 else -1
                    if name=='STATE':field,gamma,reading=rules.original.sensed_field(z,y,x,t,s,positions,c['sigma'],'FULL_ALIGN',mask,before_backward)
                    else:
                        cosine=allocation['profiles']['assignments'][c['key']][name][k]
                        field,gamma,reading=rules.sensed_field(z,y,x,t,s,positions,c['sigma'],cosine,mask,before_backward)
                    r['backward_completed']+=reading.get('backward_calls',0)
                    if reading.get('resolved'):
                        factor=reading['v_RMS']/reading['u_RMS']*reading['cosine_v_u']
                        error=abs(factor-reading['alignment_scale'])/max(abs(reading['alignment_scale']),1e-30)
                        require(error<=2e-6,'Scalar cosine factorization differs')
                        reading['scalar_factorization_error']=error
                    row=dict(step=k,t=t,next_t=s,readout=reading)
                    if mask is not None:
                        energy=field['q'].detach().double().square();known=mask.bool().expand_as(energy)
                        total=float(energy.sum())
                        row['guidance_regions']=dict(observed_RMS=float(energy[known].mean().sqrt()),
                            unobserved_RMS=float(energy[~known].mean().sqrt()),
                            unobserved_energy_fraction=float(energy[~known].sum())/total if total else 0.)
                    r['traces'][name].append(row)
                    with torch.no_grad():x=parent.old.parent.rules.risk.advance(z,x,field,positions,k,noises,gamma)
                    if not torch.isfinite(x).all():raise RuntimeError('Nonfinite trajectory:'+name)
                outputs[name]=x.detach().cpu()
            torch.cuda.synchronize(device);r['arm_seconds'][name]=time.perf_counter()-started
            r['arm_peak_memory_bytes'][name]=torch.cuda.max_memory_allocated(device)
        parent.invariant(model,z,state,device,external,p)
        require(r['calls']==list(reversed(p['grid']))*len(names) and r['backward_attempts']==r['backward_completed']<=r['nfe'],
                'Forward/backward contract failed')
        parent.old.parent.base.save_tensor(outputs,folder/'outputs.pt')
        signed(folder/'decisions_before_labels.json',dict(case=c,traces=r['traces'],GT_used=False,plan_sha256=r['plan_sha256']))
        artifacts=['outputs.pt','decisions_before_labels.json']
        if mask is not None:
            parent.old.parent.base.save_tensor(mask.detach().cpu().bool(),folder/'mask.pt');artifacts.append('mask.pt')
        r['artifacts']={n:source.digest(folder/n) for n in artifacts}
        r['gates']=dict(operator=opgate,calls_rng_parameters=dict(passed=True),no_current_GT_used=dict(passed=True),
            cosine_factorization=dict(passed=True,tolerance=2e-6,
                scope='Scalar identity at resolved steps; controls additionally check vector equivalence with A52.'))
    r['status']='complete';signed(path,r)
    print('A53 '+phase.upper()+':',c['key'],'forwards=',r['nfe'],'backwards=',r['backward_attempts'],flush=True)


def worker(args):
    import fcntl
    import torch
    from modules.main_single import load_diffusion_model
    root=Path(args.root).resolve();p=check(root)
    with (root/f'{args.phase}_shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cases=[c for i,c in enumerate(p['cases']) if i%2==args.shard]
        if all(item(root,p,c,args.phase) for c in cases):return
        actual=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
            gpu=torch.cuda.get_device_name(args.device),NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),
            TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        require(actual==p['runtime'] and parent.P.precision.precision_state(torch)==p['native_policy'],'Runtime/precision differs')
        require(parent.old.parent.model_sources()==p['model_library_sources'],'Model libraries differ')
        torch.set_num_threads(2)
        with parent.scan.preserved_rng():
            model=load_diffusion_model('ffhq',args.device);model.model.eval()
            require(source.digest(model.ckpt_path)==p['checkpoint']['sha256'] and model.num_steps==1000 and
                    all(v.dtype==torch.float32 for v in model.model.parameters()),'Checkpoint differs')
            for c in cases:
                sample(root,p,c,args.phase,model,args.device);gc.collect();torch.cuda.empty_cache()


def evaluate(root,p):
    import torch
    from modules.main_single import load_image_as_tensor
    from utils.metrics import compute_all_metrics
    pairs=[(c,item(root,p,c,'generate'),item(root,p,c,'control')) for c in p['cases']]
    require(all(a and b for c,a,b in pairs),'All10 STATE/control outputs required before GT')
    hashes={str(found[0]):source.digest(found[0]) for c,a,b in pairs for found in (a,b)}
    path=root/'evaluation.json'
    if path.exists():
        prior=source.load_signed(path);require(prior['inputs']==hashes,'Evaluation inputs changed');return prior
    rows=[]
    with parent.scan.preserved_rng(),torch.no_grad():
        for c,a,b in pairs:
            f,r,key=a;generated=torch.load(f.parent/'outputs.pt',map_location='cpu',weights_only=True)
            outputs=dict(STATE=generated[key]);outputs.update(torch.load(b[0].parent/'outputs.pt',map_location='cpu',weights_only=True))
            bank=torch.load(Path(p['a50_baselines'][c['key']]).parent/'outputs.pt',map_location='cpu',weights_only=True)
            outputs.update({n:bank[n] for n in ('C_NATIVE','C_IEEE')})
            gt=load_image_as_tensor(c['image']).cpu()
            metrics={name:compute_all_metrics(value,gt,lpips_net='alex') for name,value in outputs.items()}
            require(all(math.isfinite(float(v)) for m in metrics.values() for v in m.values()),'Nonfinite quality')
            rows.append(dict(case=c,metrics=metrics,resolved={'STATE':sum(x['readout'].get('resolved',False) for x in r['traces'][key]),
                **{n:sum(x['readout'].get('resolved',False) for x in b[1]['traces'][n]) for n in rules.CONTROLS}}))
    value=dict(inputs=hashes,allocation_sha256=source.digest(root/'allocation.json'),rows=rows,
        GT_used_only_after_all_frozen_outputs=True,limits=p['limits'])
    signed(path,value);return value


def summary(root,p):
    value=evaluate(root,p);attempts=[source.read(f) for f in root.glob('*/*_*/run.json')];groups=[]
    def guard(d):return d['ssim']>=-.002 and d['lpips']<=.005
    for task in rules.TASKS:
        rows=[r for r in value['rows'] if r['case']['task']==task]
        refs=('C_NATIVE','C_IEEE',*rules.CONTROLS)
        effects={ref:{m:mean(r['metrics']['STATE'][m]-r['metrics'][ref][m] for r in rows)
                          for m in ('psnr','ssim','lpips')} for ref in refs}
        positive={ref:sum(r['metrics']['STATE']['psnr']>r['metrics'][ref]['psnr']+1e-9 for r in rows) for ref in refs}
        covered=all(r['resolved'][n]>=24 for r in rows for n in ('STATE',*rules.CONTROLS))
        quality=covered and all(effects[b]['psnr']>=.05 and guard(effects[b]) and positive[b]>=4 for b in ('C_NATIVE','C_IEEE'))
        fully_resolved=all(r['resolved'][n]==30 for r in rows for n in ('STATE',*rules.CONTROLS))
        incremental=quality and fully_resolved and all(effects[b]['psnr']>=.02 and guard(effects[b]) and positive[b]>=4 for b in ('TIME_COS','SHUFFLE_COS'))
        groups.append(dict(task=task,STATE_minus=effects,positive_images=positive,numerical_coverage_passed=bool(covered),
            quality_candidate=bool(quality),fully_resolved_for_angle_test=bool(fully_resolved),current_angle_increment_candidate=bool(incremental),
            norm_attenuation_candidate=bool(covered and effects['NORM_ONLY']['psnr']>=.02 and guard(effects['NORM_ONLY']) and positive['NORM_ONLY']>=4),
            resolved_states={n:sum(r['resolved'][n] for r in rows) for n in ('STATE',*rules.CONTROLS)},
            per_image=[dict(image_id=r['case']['image_id'],metrics=r['metrics'],STATE_psnr_minus={ref:r['metrics']['STATE']['psnr']-r['metrics'][ref]['psnr'] for ref in refs}) for r in rows]))
    report=dict(experiment=p['experiment'],classification='DEVELOPMENT_CURRENT_ANGLE_INCREMENT_CANDIDATE' if any(g['current_angle_increment_candidate'] for g in groups)
        else 'GEOMETRY_QUALITY_WITHOUT_CONFIRMED_CURRENT_ANGLE_INCREMENT' if any(g['quality_candidate'] for g in groups) else 'NO_COHORT_QUALITY_CANDIDATE',
        completed_cases=len(value['rows']),planned_cases=10,image_clusters=5,reused_STATE_cases=4,
        method_nfe=0,diagnostic_nfe=sum(r['nfe'] for r in attempts),planned_diagnostic_nfe=1080,
        backward_attempts=sum(r['backward_attempts'] for r in attempts),backward_completed=sum(r['backward_completed'] for r in attempts),
        partial_attempts=sum(r['status']!='complete' for r in attempts),scientific_forward_nfe=300,
        reused_A52_diagnostic_nfe=480,new_arm_seconds={n:sum(r['arm_seconds'].get(n,0.) for r in attempts) for n in ('STATE',*rules.CONTROLS)},
        operator_accounting={key:sum(v['readout'].get('cg',{}).get(key,0) for r in attempts for trace in r['traces'].values() for v in trace)
                             for key in ('linear_calls','adjoint_calls')},
        groups=groups,limits=p['limits']+' STATE timings omit four cached trajectories; no controlled end-to-end speed claim. No p-values/confidence claim.')
    signed(root/'summary.json',report);print('A53 Summary:',json.dumps(report,ensure_ascii=False,allow_nan=False),flush=True)
    print('Records:',root,flush=True);return value


def render(root,p,value):
    import torch
    import numpy as np
    from PIL import Image,ImageDraw
    from utils.diag_cached_guidance_final_value import font
    for row in value['rows']:
        c=row['case'];a=item(root,p,c,'generate');b=item(root,p,c,'control')
        bank=torch.load(a[0].parent/'outputs.pt',map_location='cpu',weights_only=True)
        outputs=dict(STATE=bank[a[2]]);outputs.update(torch.load(b[0].parent/'outputs.pt',map_location='cpu',weights_only=True))
        baseline=torch.load(Path(p['a50_baselines'][c['key']]).parent/'outputs.pt',map_location='cpu',weights_only=True)
        outputs['C_IEEE']=baseline['C_IEEE'];sheet=Image.new('RGB',(1120,760),'white');draw=ImageDraw.Draw(sheet)
        draw.text((12,8),c['key']+' | CURRENT direction/norms; c-only ablations | reused development cohort',fill='black',font=font(14))
        for line,names in enumerate((('GT','C_IEEE','STATE','NORM_ONLY'),('GT','C_IEEE','TIME_COS','SHUFFLE_COS'))):
            top=36+358*line
            for j,name in enumerate(names):
                left=12+276*j;draw.text((left,top),name,fill='black',font=font(14))
                if name=='GT':im=Image.open(c['image']).convert('RGB').resize((256,256),Image.Resampling.LANCZOS)
                else:
                    arr=outputs[name][0].numpy().transpose(1,2,0);im=Image.fromarray(((np.clip(arr,-1,1)+1)*127.5).astype(np.uint8))
                    m=row['metrics'][name];draw.text((left,top+25),f'P {m["psnr"]:.3f} S {m["ssim"]:.4f} L {m["lpips"]:.4f}',fill='black',font=font(12))
                sheet.paste(im,(left,top+55))
        folder=root/c['key'];folder.mkdir(exist_ok=True);sheet.save(folder/'angular_comparison.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=ROOT);parser.add_argument('--a52-dir',default=old.ROOT)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--phase',choices=('generate','control'));parser.add_argument('--shard',type=int,choices=(0,1))
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--archive',action='store_true');args=parser.parse_args()
    if args.prepare:prepare(args);return
    if args.shard is not None:
        require(args.phase is not None,'--phase required');worker(args);return
    root=Path(args.root).resolve();p=check(root)
    if args.freeze:freeze(root,p);return
    value=summary(root,p)
    if args.archive:
        from unittest.mock import patch
        render(root,p,value)
        with patch.object(old.previous,'BRANCH',BRANCH):old.previous.archive(root)


if __name__=='__main__':main()
