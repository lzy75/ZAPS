"""A52 fixed-grid 2x2 ablation: target kind x constraint to original q direction."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import diag_current_score_geometry_r2 as previous
from utils import feedback_vector_constraint as rules

parent,source=previous.parent,previous.source
write,signed=previous.write,previous.signed
ROOT='/data0/lzy/ZAPS-results/diag_feedback_vector_constraint/A52_20261026'
CARD=PROJECTS/'experiment_cards/A52_feedback_vector_constraint_20261026.md'
BRANCH='results/ffhq-feedback-vector-constraint-20261026'
IMAGES=('00446','00559')
TASKS=('super_resolution','inpainting')


def require(test,message):
    if not test:raise ValueError(message)


def check(root):
    p=source.check_plan(root);parent.a42.verify(p)
    require(p['experiment']=='A52_feedback_vector_constraint','Wrong plan')
    return p


def prepare(args):
    root,prior=Path(args.root).resolve(),Path(args.a50_dir).resolve()
    require(not (root.is_relative_to(prior) or prior.is_relative_to(root)),'Source/output overlap')
    if root.exists():
        p=check(root);require(p['source_A50r2']==str(prior),'Frozen source changed')
        print('A52 frozen plan reused; no complete GPU work restarted.',flush=True);return
    before=previous.check(prior)
    require(source.digest(prior/'summary.json')=='36bb090d3ec32a9f34c92db4e66fa71280262c97b5b74b30b0e4860d5113ce9b',
            'Need the completed, audited A50r2 summary')
    cases=[c for c in before['cases'] if c['task'] in TASKS and c['image_id'] in IMAGES]
    require(len(cases)==4 and {(c['task'],c['image_id']) for c in cases}=={(t,i) for t in TASKS for i in IMAGES},'Wrong cohort')
    identities=dict(before['identity_files']);refs={}
    for name in ('plan.json','summary.json','allocation.json'):
        identities[str(prior/name)]=source.digest(prior/name)
    for c in cases:
        refs[c['key']]={}
        for phase in ('generate','control'):
            item=previous.phase_done(prior,before,c,phase)
            require(item is not None,'Incomplete parent case')
            f,r=item;refs[c['key']][phase]=str(f);identities[str(f)]=source.digest(f)
            for name,sha in r['artifacts'].items():identities[str(f.parent/name)]=sha
            if phase=='generate':
                for arm in ('PROX_Q','ALIGN_Q'):
                    require(all(v['readout'].get('resolved') for v in r['traces'][arm]),'Need fully resolved parent Q arms')
    # Bring the completed solver diagnostics into the next lightweight archive.
    solver=Path('/data0/lzy/ZAPS-results/diag_score_cg_unrestarted/A51b_20261026')
    from utils import diag_score_cg_unrestarted as cg
    solver_run=source.read(solver/'run.json')
    require(solver_run['status']=='complete' and len(solver_run['rows'])==32 and
            source.read(solver/'summary.json')==cg.summarize(solver_run),'Need completed consistent A51b')
    for name in ('run.json','summary.json'):identities[str(solver/name)]=source.digest(solver/name)
    own=[Path(__file__),Path(rules.__file__),CARD,PROJECTS/'run_a52_20261026.sh',
         PROJECTS/'utils/test_feedback_vector_constraint.py',Path(cg.__file__)]
    p=dict(before)
    p.update(experiment='A52_feedback_vector_constraint',source_A50r2=str(prior),source_A51b=str(solver),
        cases=cases,reference_records=refs,identity_files=identities,
        worker_identity_files={**before['worker_identity_files'],**{str(f):source.digest(f) for f in own}},
        arms=list(rules.ARMS),phases=['generate'],generate_arms=list(rules.ARMS),control_arms=[],
        metrology_steps=[],metrology_scales=[],
        metrology_note='A50r2 actual-model numerical VJP checks reused; no new finite-difference calls. '
                       'New Q trajectory replay, shadow projection, operator, parameter and RNG gates.',
        planned_method_nfe=0,planned_diagnostic_nfe=480,historical_training_nfe=1200,reused_prefix_forward_nfe=1080,
        planned_backward_attempts_max=240,scientific_forward_nfe=300,
        limits='Two REUSED development images x SR/inpainting. Fixed original30-step last unroll; original first270. '
               '2x2 mechanism ablation only, not independent state gain or four-task transfer. '
               'FULL targets replace q direction and thus remove its learned D directional restriction; no training. '
               'Forward300 is not equal runtime; TIME/SHUFFLE are context references, not matched full-vector state ablations.')
    root.parent.mkdir(parents=True,exist_ok=True);parent.P.coupled.archive.recovery.require_space(root.parent)
    root.mkdir();signed(root/'plan.json',p);shutil.copyfile(CARD,root/'experiment_card.md')
    snap=root/'source_snapshot';snap.mkdir()
    for name in ('plan.json','summary.json'):shutil.copyfile(prior/name,snap/('A50r2_'+name))
    for name in ('run.json','summary.json'):shutil.copyfile(solver/name,snap/('A51b_'+name))
    print('A52 PREPARE PASS:4 reused cases, 2x2 target/constraint ablation;480 new forwards,<=240 input backwards;no training.',flush=True)


def done(root,p,c):
    paths=sorted((root/c['key']).glob('generate_*/run.json'))
    if not paths:return None
    require(len(paths)==1,'Multiple attempts; no output selection')
    path=paths[0];raw=source.read(path)
    if raw['status']!='complete':return None
    r=source.load_signed(path)
    times=list(reversed(p['grid']))
    require(r['case']==c and r['plan_sha256']==source.digest(root/'plan.json') and r['calls']==times*4 and r['nfe']==120,
            'Case identities/call sequence differ')
    require(set(r['traces'])==set(rules.ARMS) and all(len(v)==30 for v in r['traces'].values()),'Missing arm trace')
    require(r['gates'] and all(v['passed'] for v in r['gates'].values()),'Failed GPU gate')
    require(r['backward_attempts']==r['backward_completed'] and r['backward_completed']<=60,'Backward accounting differs')
    require(all(source.digest(path.parent/n)==sha for n,sha in r['artifacts'].items()),'Output artifact changed')
    return path,r


def sample(root,p,c,model,device):
    import torch
    from unittest.mock import patch
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state
    from utils.diag_zaps_trace_audit import relative_error
    if done(root,p,c):return
    operator,mask,opgate,state,original,y,z,positions,noises=parent.setup(p,c,model,device)
    folder,r=previous.new_attempt(root,p,c,'generate');path=folder/'run.json';external=rng_state(device)
    refpath=Path(p['reference_records'][c['key']]['generate'])
    reference=torch.load(refpath.parent/'outputs.pt',map_location='cpu',weights_only=True)
    outputs={};parity={}
    with previous.counted(model,path,r,120) as before_backward, \
         previous.runtime.input_vjp_parameters(model.model),model_rng_guard(model.model,device), \
         patch('torch.randn_like',side_effect=RuntimeError('Unexpected random draw')):
        for name in rules.ARMS:
            torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device);started=time.perf_counter()
            x=state['init_noise'].to(device);r['traces'][name]=[]
            with parent.P.precision.ieee(torch):
                for k,t in enumerate(positions.nominal):
                    s=positions.nominal[k+1] if k<29 else -1
                    field,gamma,reading=rules.sensed_field(z,y,x,t,s,positions,c['sigma'],name,mask,before_backward)
                    r['backward_completed']+=reading.get('backward_calls',0)
                    r['traces'][name].append(dict(step=k,t=t,next_t=s,gamma=float(gamma),readout=reading,
                        applied_vector_RMS=float(field['q'].detach().double().square().mean().sqrt())))
                    with torch.no_grad():x=parent.old.parent.rules.risk.advance(z,x,field,positions,k,noises,gamma)
                    if not torch.isfinite(x).all():raise RuntimeError('Nonfinite trajectory:'+name)
                outputs[name]=x.detach().cpu()
            torch.cuda.synchronize(device);r['arm_seconds'][name]=time.perf_counter()-started
            r['arm_peak_memory_bytes'][name]=torch.cuda.max_memory_allocated(device)
            if name in ('PROX_Q','ALIGN_Q'):
                error=relative_error(outputs[name],reference[name]);parity[name]=error
                if not math.isfinite(error) or error>1e-6:raise RuntimeError('Original Q trajectory replay failed:'+name)
        parent.invariant(model,z,state,device,external,p)
        require(r['calls']==list(reversed(p['grid']))*4,'Unexpected paid score calls')
        require(r['backward_attempts']==r['backward_completed']<=60,'Backward cap/closure failed')
        parent.old.parent.base.save_tensor(outputs,folder/'outputs.pt')
        signed(folder/'decisions_before_labels.json',dict(case=c,traces=r['traces'],GT_used=False,plan_sha256=r['plan_sha256']))
        artifacts=['outputs.pt','decisions_before_labels.json']
        if mask is not None:
            parent.old.parent.base.save_tensor(mask.detach().cpu().bool(),folder/'mask.pt');artifacts.append('mask.pt')
        r['artifacts']={n:source.digest(folder/n) for n in artifacts}
        r['gates']=dict(operator=opgate,Q_replay=dict(passed=True,relative_errors=parity),
            calls_rng_parameters=dict(passed=True),no_current_GT_used=dict(passed=True),
            full_target_shadow_projection=dict(passed=True,tolerance=2e-6))
    r['status']='complete';signed(path,r)
    print('A52 CASE:',c['key'],'forwards=',r['nfe'],'backwards=',r['backward_attempts'],flush=True)


def worker(args):
    import torch
    from modules.main_single import load_diffusion_model
    root=Path(args.root).resolve();p=check(root)
    if all(done(root,p,c) for c in p['cases']):return
    actual=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(args.device),NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),
        TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
    require(actual==p['runtime'] and parent.P.precision.precision_state(torch)==p['native_policy'],'Runtime/precision differs')
    require(parent.old.parent.model_sources()==p['model_library_sources'],'Model libraries differ')
    torch.set_num_threads(2)
    with parent.scan.preserved_rng():
        model=load_diffusion_model('ffhq',args.device);model.model.eval()
        require(source.digest(model.ckpt_path)==p['checkpoint']['sha256'] and model.num_steps==1000 and
                all(v.dtype==torch.float32 for v in model.model.parameters()),'Checkpoint/model differs')
        for c in p['cases']:
            sample(root,p,c,model,args.device);gc.collect();torch.cuda.empty_cache()


def evaluate(root,p):
    import torch
    from modules.main_single import load_image_as_tensor
    from utils.metrics import compute_all_metrics
    items=[(c,done(root,p,c)) for c in p['cases']]
    require(all(item for c,item in items),'All4 frozen outputs required before GT evaluation')
    hashes={str(item[0]):source.digest(item[0]) for c,item in items}
    path=root/'evaluation.json'
    if path.exists():
        prior=source.load_signed(path);require(prior['inputs']==hashes,'Evaluation inputs changed');return prior
    rows=[]
    with parent.scan.preserved_rng(),torch.no_grad():
        for c,(f,r) in items:
            outputs=torch.load(f.parent/'outputs.pt',map_location='cpu',weights_only=True)
            for phase,names in (('generate',('C_NATIVE','C_IEEE')),('control',('TIME','SHUFFLE'))):
                bank=torch.load(Path(p['reference_records'][c['key']][phase]).parent/'outputs.pt',map_location='cpu',weights_only=True)
                outputs.update({name:bank[name] for name in names})
            gt=load_image_as_tensor(c['image']).cpu()
            metrics={name:compute_all_metrics(value,gt,lpips_net='alex') for name,value in outputs.items()}
            require(all(math.isfinite(float(v)) for m in metrics.values() for v in m.values()),'Nonfinite quality')
            regions={}
            if c['task']=='inpainting':
                mask=torch.load(f.parent/'mask.pt',map_location='cpu',weights_only=True).expand_as(gt)
                for name,value in outputs.items():
                    error=((value.clamp(-1,1)-gt)/2.).double().square()
                    regions[name]={region:float(error[active].mean()) for region,active in
                                   (('observed_unit_MSE',mask),('unobserved_unit_MSE',~mask))}
            rows.append(dict(case=c,metrics=metrics,inpainting_regions=regions))
    result=dict(inputs=hashes,rows=rows,GT_used_only_after_all_frozen_outputs=True,
                metric_contract='Existing clipped/8bit PSNR/SSIM + Alex LPIPS; regional IP unit-domain float MSE separately.',
                limits=p['limits'])
    signed(path,result);return result


def summary(root,p):
    from statistics import mean
    value=evaluate(root,p);attempts=[source.read(f) for f in root.glob('*/generate_*/run.json')];groups=[]
    for task in TASKS:
        rows=[r for r in value['rows'] if r['case']['task']==task]
        effects={name:{ref:{m:mean(r['metrics'][name][m]-r['metrics'][ref][m] for r in rows)
                    for m in ('psnr','ssim','lpips')} for ref in ('C_NATIVE','C_IEEE','PROX_Q','ALIGN_Q','FULL_PROX','TIME','SHUFFLE')}
                 for name in ('FULL_PROX','FULL_ALIGN')}
        coverage={name:sum(v['readout'].get('resolved',False) for r in attempts if r['case']['task']==task
                           for v in r['traces'][name]) for name in rules.ARMS}
        groups.append(dict(task=task,mean_deltas=effects,resolved_states=coverage,
            positive_images={name:sum(r['metrics'][name]['psnr']>r['metrics'][ref]['psnr']+1e-9 for r in rows)
                             for name,ref in (('FULL_PROX','PROX_Q'),('FULL_ALIGN','ALIGN_Q'))},
            per_image=[dict(image_id=r['case']['image_id'],metrics=r['metrics'],inpainting_regions=r['inpainting_regions']) for r in rows]))
    report=dict(experiment=p['experiment'],classification='DEVELOPMENT_VECTOR_CONSTRAINT_ABLATION_ONLY',
        completed_cases=len(value['rows']),planned_cases=4,image_clusters=2,method_nfe=0,
        diagnostic_nfe=sum(r['nfe'] for r in attempts),planned_diagnostic_nfe=480,
        backward_attempts=sum(r['backward_attempts'] for r in attempts),
        backward_completed=sum(r['backward_completed'] for r in attempts),
        partial_attempts=sum(r['status']!='complete' for r in attempts),scientific_forward_nfe=300,
        arm_seconds={name:sum(r['arm_seconds'][name] for r in attempts) for name in rules.ARMS},
        arm_peak_memory_bytes={name:max(r['arm_peak_memory_bytes'][name] for r in attempts) for name in rules.ARMS},
        groups=groups,limits=p['limits']+' No state-benefit or generalization classification assigned by this mechanism test.')
    signed(root/'summary.json',report);print('A52 Summary:',json.dumps(report,ensure_ascii=False,allow_nan=False),flush=True)
    print('Records:',root,flush=True);return value


def render(root,p,value):
    import torch
    import numpy as np
    from PIL import Image,ImageDraw
    from utils.diag_cached_guidance_final_value import font
    for row in value['rows']:
        c=row['case'];f,r=done(root,p,c)
        outputs=torch.load(f.parent/'outputs.pt',map_location='cpu',weights_only=True)
        ref=torch.load(Path(p['reference_records'][c['key']]['generate']).parent/'outputs.pt',map_location='cpu',weights_only=True)
        outputs['C_IEEE']=ref['C_IEEE'];sheet=Image.new('RGB',(1120,760),'white');draw=ImageDraw.Draw(sheet)
        draw.text((12,8),c['key']+' | target kind x q constraint | reused development images',fill='black',font=font(15))
        for line,names in enumerate((('GT','C_IEEE','PROX_Q','FULL_PROX'),('GT','C_IEEE','ALIGN_Q','FULL_ALIGN'))):
            top=36+358*line
            for j,name in enumerate(names):
                left=12+276*j;draw.text((left,top),name,fill='black',font=font(14))
                if name=='GT':im=Image.open(c['image']).convert('RGB').resize((256,256),Image.Resampling.LANCZOS)
                else:
                    arr=outputs[name][0].numpy().transpose(1,2,0)
                    im=Image.fromarray(((np.clip(arr,-1,1)+1)*127.5).astype(np.uint8))
                    m=row['metrics'][name];draw.text((left,top+25),f'P {m["psnr"]:.3f} S {m["ssim"]:.4f} L {m["lpips"]:.4f}',fill='black',font=font(12))
                sheet.paste(im,(left,top+55))
        sheet.save(root/c['key']/'constraint_comparison.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=ROOT);parser.add_argument('--a50-dir',default=previous.ROOT)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true')
    parser.add_argument('--archive',action='store_true');parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.prepare:prepare(args);return
    if args.run:worker(args);return
    root=Path(args.root).resolve();p=check(root);value=summary(root,p)
    if args.archive:
        from unittest.mock import patch
        render(root,p,value)
        with patch.object(previous,'BRANCH',BRANCH):previous.archive(root)


if __name__=='__main__':main()
