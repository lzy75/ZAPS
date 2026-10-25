"""A50 fixed-grid full-last-unroll current-score geometry, no GT decisions."""
import argparse
from contextlib import contextmanager, nullcontext
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import diag_frozen_state_matching as parent
from utils import current_score_design as design
from utils import current_score_runtime as runtime

source=parent.source
ROOT='/data0/lzy/ZAPS-results/diag_current_score_geometry/A50_20261026'
CARD=PROJECTS/'experiment_cards/A50_current_score_geometry_20261026.md'
BRANCH='results/ffhq-current-score-geometry-20261026'
write,signed=parent.write,parent.signed


def check(root):
    p=source.check_plan(root)
    parent.a42.verify(p)
    return p


def prepare(args):
    root,prior=Path(args.root).resolve(),Path(args.a44_dir).resolve()
    if root.is_relative_to(prior) or prior.is_relative_to(root):
        raise ValueError('Source/output overlap')
    if root.exists():
        p=check(root)
        if p['source_A44']!=str(prior):raise ValueError('Frozen parent differs')
        print('A50 frozen plan reused; no experiments restarted.',flush=True)
        return
    before=source.check_plan(prior);parent.a42.verify(before)
    if before['experiment']!='A44_frozen_state_matching':raise ValueError('Expected A44')
    cases=before['cases']
    if len(cases)!=20 or {(c['task'],c['image_id']) for c in cases}!={(t,i) for t in design.TASKS for i in design.IMAGES}:
        raise ValueError('Need fixed five-old-image/four-task cohort')
    identities=dict(before['identity_files']);finishes={}
    identities[str(prior/'plan.json')]=source.digest(prior/'plan.json')
    for c in cases:
        if c['sigma']!=.05 or c['sampling_seed']!=1000:raise ValueError('Protocol differs')
        for phase in ('choose','finish'):
            item=parent.phase_done(prior,before,c,phase)
            if item is None:raise ValueError('Need all completed A44 phases')
            f,r=item;identities[str(f)]=source.digest(f)
            for n,sha in r['artifacts'].items():identities[str(f.parent/n)]=sha
            if phase=='finish':finishes[c['key']]=str(f)
    own=[Path(__file__),Path(runtime.__file__),Path(design.__file__),PROJECTS/'utils/current_score_geometry.py',
         PROJECTS/'utils/test_current_score_geometry.py',PROJECTS/'utils/test_current_score_design.py',
         PROJECTS/'utils/test_current_score_geometry_torch.py',PROJECTS/'run_a50_20261026.sh',CARD]
    p={k:before[k] for k in ('runtime','native_policy','checkpoint','model_library_sources','grid','source_A41','source_records')}
    p.update(experiment='A50_current_score_geometry',source_A44=str(prior),source_finish_records=finishes,
        cases=cases,identity_files=identities,
        worker_identity_files={**before['worker_identity_files'],**{str(f):source.digest(f) for f in own}},
        phases=list(design.PHASES),generate_arms=list(design.GENERATE_ARMS),control_arms=list(design.CONTROL_ARMS),
        cg_rtol=runtime.CG_RTOL,cg_maxiter=runtime.CG_MAXITER,numerical_ulps=runtime.NUMERICAL_ULPS,
        metrology_steps=list(design.METROLOGY_STEPS),metrology_scales=list(runtime.VJP_FD_SCALES),
        planned_method_nfe=0,planned_diagnostic_nfe=4872,planned_backward_attempts_max=1212,
        scientific_forward_nfe=300,historical_training_nfe=6000,
        limits='Five reused development image clusters, one seed, known sigma. Last30 fixed-grid calls only; '
               'raw target and original clipped q. Current zeta scalar cancels in the target projection. '
               'IEEE graph-enabled production has its own matched baseline; original native baseline retained. '
               'Forward NFE equality is not equal runtime. No independent validation, step/NFE saving, '
               'posterior correctness or novel projection theorem claim.')
    root.parent.mkdir(parents=True,exist_ok=True)
    parent.P.coupled.archive.recovery.require_space(root.parent)
    root.mkdir();signed(root/'plan.json',p);shutil.copyfile(CARD,root/'experiment_card.md')
    snap=root/'source_snapshot';snap.mkdir()
    shutil.copyfile(prior/'plan.json',snap/'A44_plan.json')
    for c in cases:shutil.copyfile(finishes[c['key']],snap/(c['key']+'_A44_finish.json'))
    print('A50 PREPARE PASS:20 old cases; fixed30-call full-unroll arms; '
          '4872 NEW forwards, <=1212 input-backward attempts; no training.',flush=True)


def phase_done(root,p,c,phase):
    found=[]
    for path in sorted((root/c['key']).glob(phase+'_*/run.json')):
        raw=source.read(path)
        if raw['status']!='complete':continue
        r=source.load_signed(path)
        if (r['case']!=c or r['phase']!=phase or r['plan_sha256']!=source.digest(root/'plan.json') or
            r['calls']!=design.expected_forward_trace(c,p['grid'],phase) or r['nfe']!=len(r['calls']) or
            r['nfe']!=design.phase_limit(c,phase) or not r['gates'] or not all(g['passed'] for g in r['gates'].values()) or
            any(source.digest(path.parent/n)!=sha for n,sha in r['artifacts'].items())):
            raise ValueError('Invalid A50 phase:'+str(path))
        if phase=='control' and r['allocation_sha256']!=source.digest(root/'allocation.json'):
            raise ValueError('Frozen control allocation differs')
        found.append((path,r))
    if len(found)>1:raise ValueError('Multiple complete phases; no output selection')
    return found[0] if found else None


def new_attempt(root,p,c,phase):
    folder=root/c['key'];folder.mkdir(exist_ok=True)
    prior=[source.read(f) for f in folder.glob(phase+'_*/run.json')]
    if prior:
        raise ValueError('Interrupted/failed phase retained; no automatic budget-reset rerun: '+str(folder))
    out=folder/(phase+'_000');out.mkdir()
    return out,dict(case=c,phase=phase,plan_sha256=source.digest(root/'plan.json'),gates={},traces={},
                    backward_attempts=0,backward_completed=0,metrology=[],arm_seconds={},arm_peak_memory_bytes={})


@contextmanager
def counted(model,path,r,cap):
    start=time.perf_counter();r.update(status='running',nfe=0,calls=[])
    def save():
        r['seconds']=time.perf_counter()-start;write(path,r)
    def before(_module,inputs):
        if r['nfe']>=cap:raise RuntimeError('A50 call budget exhausted before next forward')
        r['calls'].append(int(inputs[1].flatten()[0]));r['nfe']+=1;save()
    def backward():
        r['backward_attempts']+=1;save()
    save();hook=model.model.register_forward_pre_hook(before)
    try:yield backward
    except BaseException as error:
        r.update(status='failed',error=repr(error));save();raise
    finally:hook.remove()


def sample_phase(root,p,c,phase,model,device):
    import torch
    from unittest.mock import patch
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state
    from utils.diag_zaps_trace_audit import relative_error
    if phase_done(root,p,c,phase):return
    operator,mask,opgate,state,original,y,z,positions,noises=parent.setup(p,c,model,device)
    folder,r=new_attempt(root,p,c,phase);path=folder/'run.json';external=rng_state(device)
    allocation=None
    if phase=='control':
        allocation=source.load_signed(root/'allocation.json')
        for f,sha in allocation['sources'].items():
            if source.digest(f)!=sha:raise ValueError('Unlabeled trajectory source changed')
        r['allocation_sha256']=source.digest(root/'allocation.json')
    names=design.GENERATE_ARMS if phase=='generate' else design.CONTROL_ARMS
    outputs={};parity=None
    with counted(model,path,r,design.phase_limit(c,phase)) as before_backward, \
         runtime.frozen_model_parameters(model.model),model_rng_guard(model.model,device), \
         patch('torch.randn_like',side_effect=RuntimeError('Unexpected random draw')):
        for name in names:
            torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device);started=time.perf_counter()
            x=state['init_noise'].to(device);rows=[];r['traces'][name]=rows
            scope=parent.P.precision.ieee(torch) if name!='C_NATIVE' else nullcontext()
            with scope:
                for k,t in enumerate(positions.nominal):
                    s=positions.nominal[k+1] if k<29 else -1
                    if name=='C_NATIVE':
                        with torch.no_grad():field=source.fields.field(z,y,x,t,positions)
                        gamma,reading=1.,{}
                    else:
                        mode=name if name in runtime.MODES else 'C_IEEE'
                        field,gamma,reading=runtime.sensed_field(z,y,x,t,s,positions,c['sigma'],mode,mask,before_backward)
                        r['backward_completed']+=reading.get('backward_calls',0)
                        if phase=='control':gamma=allocation['profiles']['assignments'][c['key']][name][k]
                    row=dict(step=k,t=t,next_t=s,gamma=float(gamma),
                             qRMS=float(field['q'].double().square().mean().sqrt()),readout=reading)
                    rows.append(row)
                    if name=='C_IEEE' and c['image_id']==design.IMAGES[0] and k in design.METROLOGY_STEPS:
                        with torch.no_grad():vector=operator.transpose(y-operator.H(field['clean']))
                        gate=runtime.vjp_metrology(model,x,t,vector,before_backward)
                        r['backward_completed']+=gate['backward_calls']
                        r['metrology'].append(dict(step=k,t=t,**gate))
                        if not gate['passed']:raise RuntimeError('Actual-model VJP/grad-mode metrology failed')
                    with torch.no_grad():x=parent.old.parent.rules.risk.advance(z,x,field,positions,k,noises,gamma)
                    if not torch.isfinite(x).all():raise RuntimeError('Nonfinite trajectory:'+name)
                outputs[name]=x.detach().cpu()
            torch.cuda.synchronize(device);r['arm_seconds'][name]=time.perf_counter()-started;r['traces'][name]=rows
            r['arm_peak_memory_bytes'][name]=torch.cuda.max_memory_allocated(device)
            if name=='C_NATIVE':
                parity=relative_error(outputs[name],original)
                if not math.isfinite(parity) or parity>1e-6:raise RuntimeError('Native C replay failed')
        parent.invariant(model,z,state,device,external,p)
        if r['calls']!=design.expected_forward_trace(c,p['grid'],phase):raise RuntimeError('Paid call sequence differs')
        parent.old.parent.base.save_tensor(outputs,folder/'outputs.pt')
        signed(folder/'decisions_before_labels.json',dict(case=c,traces=r['traces'],GT_used=False,plan_sha256=r['plan_sha256']))
        r['artifacts']={n:source.digest(folder/n) for n in ('outputs.pt','decisions_before_labels.json')}
        r['gates']=dict(operator=opgate,calls_rng_parameters=dict(passed=True),
                        no_current_output_GT_used=dict(passed=True),
                        native_C=dict(passed=True,relative_error=parity) if phase=='generate' else dict(passed=True))
    r['status']='complete';signed(path,r)
    print('A50 '+phase.upper()+':',c['key'],'calls=',r['nfe'],'backward=',r['backward_attempts'],flush=True)


def freeze(root,p):
    items=[phase_done(root,p,c,'generate') for c in p['cases']]
    if not all(items):raise ValueError('Need all20 generated trajectories before controls')
    curves={c['key']:[row['gamma'] for row in item[1]['traces']['ALIGN_Q']] for c,item in zip(p['cases'],items)}
    value=dict(plan_sha256=source.digest(root/'plan.json'),
               sources={str(f):source.digest(f) for f,_ in items},
               profiles=design.allocate(p['cases'],curves),GT_used=False)
    path=root/'allocation.json'
    if path.exists():
        if source.load_signed(path)!=value:raise ValueError('Frozen controls differ')
    else:signed(path,value)
    print('A50 FREEZE PASS:20 trajectories; current image excluded from TIME/CONSTANT; cyclic SHUFFLE; no labels.',flush=True)


def worker(args):
    import fcntl
    import torch
    from modules.main_single import load_diffusion_model
    root=Path(args.root).resolve();p=check(root)
    with (root/f'{args.phase}_shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cases=[c for i,c in enumerate(p['cases']) if i%2==args.shard]
        if all(phase_done(root,p,c,args.phase) for c in cases):return
        actual=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                    gpu=torch.cuda.get_device_name(args.device),
                    NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),
                    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        if actual!=p['runtime'] or parent.P.precision.precision_state(torch)!=p['native_policy']:
            raise ValueError('Parent runtime/precision differs')
        if parent.old.parent.model_sources()!=p['model_library_sources']:raise ValueError('Model libraries differ')
        torch.set_num_threads(2);parent.P.coupled.archive.recovery.require_space(root)
        with parent.scan.preserved_rng():
            model=load_diffusion_model('ffhq',args.device);model.model.eval()
            if (source.digest(model.ckpt_path)!=p['checkpoint']['sha256'] or model.num_steps!=1000 or
                    any(v.dtype!=torch.float32 for v in model.model.parameters())):
                raise ValueError('Checkpoint differs')
            for c in cases:
                sample_phase(root,p,c,args.phase,model,args.device)
                gc.collect();torch.cuda.empty_cache()


def evaluate(root,p):
    import torch
    from modules.main_single import load_image_as_tensor
    from utils.metrics import compute_all_metrics
    allocation=source.load_signed(root/'allocation.json')
    pairs=[(c,phase_done(root,p,c,'generate'),phase_done(root,p,c,'control')) for c in p['cases']]
    if not all(a and b for _,a,b in pairs):raise ValueError('All generated/control outputs required before GT')
    hashes={str(f):source.digest(f) for _,a,b in pairs for f,_ in (a,b)}
    path=root/'evaluation.json'
    if path.exists():
        prior=source.load_signed(path)
        if prior['inputs']!=hashes or prior['allocation_sha256']!=source.digest(root/'allocation.json'):
            raise ValueError('Evaluation source changed')
        return prior
    rows=[]
    with parent.scan.preserved_rng(),torch.no_grad():
        for c,a,b in pairs:
            outputs={}
            for f,r in (a,b):
                bank=torch.load(f.parent/'outputs.pt',map_location='cpu',weights_only=True)
                outputs.update(bank)
            gt=load_image_as_tensor(c['image']).cpu()
            metrics={name:compute_all_metrics(output,gt,lpips_net='alex') for name,output in outputs.items()}
            if any(not math.isfinite(float(v)) for m in metrics.values() for v in m.values()):
                raise RuntimeError('Nonfinite final quality')
            rows.append(dict(case=c,metrics=metrics,
                             sensor_resolved={name:sum(row['readout'].get('resolved',False) for row in a[1]['traces'][name])
                                              for name in runtime.MODES},
                             backward_attempts=a[1]['backward_attempts'],
                             generation_seconds=a[1]['seconds'],control_seconds=b[1]['seconds']))
    result=dict(inputs=hashes,allocation_sha256=source.digest(root/'allocation.json'),rows=rows,
                metric_contract='Existing compute_all_metrics: clipped/8bit PSNR and SSIM; Alex LPIPS. Same API for every arm.',
                GT_used_only_after_all_frozen_outputs=True,limits=p['limits'])
    signed(path,result);return result


def summarize(root,p,with_evaluation=False):
    attempts=[source.read(f) for f in root.glob('*/*_*/run.json')]
    complete={phase:sum(phase_done(root,p,c,phase) is not None for c in p['cases']) for phase in design.PHASES}
    result=dict(experiment=p['experiment'],classification='INCOMPLETE_PILOT',completed_phases=complete,
                planned_cases=20,image_clusters=5,method_nfe=0,diagnostic_nfe=sum(r['nfe'] for r in attempts),
                backward_attempts=sum(r.get('backward_attempts',0) for r in attempts),
                backward_completed=sum(r.get('backward_completed',0) for r in attempts),
                partial_attempts=sum(r['status']!='complete' for r in attempts),
                cost_is_lower_bound_if_abruptly_interrupted=any(r['status']=='running' for r in attempts),
                scientific_forward_nfe=300,planned_diagnostic_nfe=4872,limits=p['limits'])
    result['operator_accounting']={name:sum(row['readout'].get('cg',{}).get(key,0)
        for r in attempts for rows in r.get('traces',{}).values() for row in rows)
        for name,key in (('cg_linear_calls','linear_calls'),('cg_adjoint_calls','adjoint_calls'))}
    result['operator_accounting']['lower_bound_if_probe_raised']=any(r['status']!='complete' for r in attempts)
    result['arm_seconds']={name:sum(r.get('arm_seconds',{}).get(name,0.) for r in attempts)
                           for name in design.GENERATE_ARMS+design.CONTROL_ARMS}
    result['arm_peak_memory_bytes']={name:max((r.get('arm_peak_memory_bytes',{}).get(name,0) for r in attempts),default=0)
                           for name in design.GENERATE_ARMS+design.CONTROL_ARMS}
    if with_evaluation:
        value=evaluate(root,p);groups=[]
        for task in design.TASKS:
            rows=[r for r in value['rows'] if r['case']['task']==task];effects={}
            for name in design.GENERATE_ARMS+design.CONTROL_ARMS:
                effects[name]={ref:{m:float(np.mean([r['metrics'][name][m]-r['metrics'][ref][m] for r in rows]))
                                   for m in ('psnr','ssim','lpips')} for ref in ('C_NATIVE','C_IEEE','PROX_Q','DIRECT_Q','TIME','CONSTANT','SHUFFLE')}
            quality=lambda d:d['psnr']>=.05 and d['ssim']>=-.002 and d['lpips']<=.005
            positive={b:sum(r['metrics']['ALIGN_Q']['psnr']>r['metrics'][b]['psnr'] for r in rows)
                      for b in ('C_NATIVE','C_IEEE','TIME','SHUFFLE')}
            coverage=all(r['sensor_resolved']['ALIGN_Q']>=24 for r in rows)
            q=coverage and all(quality(effects['ALIGN_Q'][b]) and positive[b]>=4 for b in ('C_NATIVE','C_IEEE'))
            matching=all(effects['ALIGN_Q'][b]['psnr']>=.02 and effects['ALIGN_Q'][b]['ssim']>=-.002 and
                         effects['ALIGN_Q'][b]['lpips']<=.005 for b in ('TIME','SHUFFLE'))
            groups.append(dict(task=task,mean_deltas=effects,quality_candidate=bool(q),
                               matching_candidate=bool(q and matching and positive['TIME']>=4 and positive['SHUFFLE']>=4),
                               positive_images=positive,numerical_coverage_passed=bool(coverage),
                               per_image=[dict(image_id=r['case']['image_id'],metrics=r['metrics']) for r in rows],
                               resolved_states={name:sum(r['sensor_resolved'][name] for r in rows) for name in runtime.MODES}))
        result.update(groups=groups,classification='DEVELOPMENT_STATE_MATCHING_CANDIDATE' if any(g['matching_candidate'] for g in groups)
                      else 'DEVELOPMENT_QUALITY_ONLY_STATE_VALUE_UNCONFIRMED' if any(g['quality_candidate'] for g in groups)
                      else 'NO_DEVELOPMENT_SCORE_GEOMETRY_SUPPORT',
                      limits=p['limits']+' Whole-gamma shuffle includes amplitude matching; not isolation of the alignment coefficient. '
                      'No confidence claim from five reused images. No four-task success unless all four separately support it.')
    signed(root/'summary.json',result)
    print('A50 Summary:',json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
    print('Records:',root,flush=True)
    return result


def render(root,p):
    import torch
    from PIL import Image,ImageDraw
    from utils.diag_cached_guidance_final_value import font
    value=evaluate(root,p)
    for row in value['rows']:
        c=row['case'];outputs={};traces={}
        for phase in design.PHASES:
            path,r=phase_done(root,p,c,phase)
            outputs.update(torch.load(path.parent/'outputs.pt',map_location='cpu',weights_only=True))
            traces.update(r['traces'])
        names=('GT',)+design.GENERATE_ARMS+design.CONTROL_ARMS
        sheet=Image.new('RGB',(24+276*len(names),390),'white');draw=ImageDraw.Draw(sheet)
        draw.text((12,6),c['key']+' | full last30 fixed steps; five reused development images',fill='black',font=font(16))
        for j,name in enumerate(names):
            left=12+276*j
            if name=='GT':
                im=Image.open(c['image']).convert('RGB').resize((256,256),Image.Resampling.LANCZOS)
                label='GT'
            else:
                arr=outputs[name][0].numpy().transpose(1,2,0)
                im=Image.fromarray(((np.clip(arr,-1,1)+1)*127.5).astype(np.uint8))
                metric=row['metrics'][name];gs=[v['gamma'] for v in traces[name]]
                label=f'{name} median gamma={np.median(gs):.3g}'
                draw.text((left,65),f'P {metric["psnr"]:.3f} S {metric["ssim"]:.4f} L {metric["lpips"]:.4f}',fill='black',font=font(12))
            draw.text((left,36),label,fill='black',font=font(13));sheet.paste(im,(left,102))
        sheet.save(root/c['key']/'policy_comparison.png')


def archive(root):
    from unittest.mock import patch
    # Reuse byte-preserving archive transport only; no old experiment is changed.
    with patch.object(parent.parent,'BRANCH',BRANCH):
        bundle,commit=parent.parent.archive(root)
    print('Archive:',bundle,'Results commit:',commit,'Branch:',BRANCH,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=ROOT);parser.add_argument('--a44-dir',default=parent.ROOT)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--inspect',action='store_true')
    parser.add_argument('--freeze',action='store_true');parser.add_argument('--phase',choices=design.PHASES)
    parser.add_argument('--shard',type=int,choices=(0,1));parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--evaluate',action='store_true');parser.add_argument('--archive',action='store_true');args=parser.parse_args()
    if args.prepare:prepare(args);return
    root=Path(args.root).resolve();p=check(root)
    if args.inspect:
        print('A50 INSPECT PASS:identities/cohort/budget;0newUNet;GPU numerical gates pending.',flush=True)
    elif args.freeze:freeze(root,p)
    elif args.shard is not None:
        if args.phase is None:parser.error('--phase required')
        worker(args)
    else:
        summarize(root,p,args.evaluate or args.archive)
        if args.archive:render(root,p);archive(root)


if __name__=='__main__':main()
