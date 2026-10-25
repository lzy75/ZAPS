"""A41 FFHQ four-task fixed-state action response, native final endpoints, no core edits."""
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
from utils import diag_multitask_state_transfer as parent
from utils import ffhq_action_geometry as rules

source,P,scan=parent.source,parent.P,parent.scan
ROOT='/data0/lzy/ZAPS-results/diag_ffhq_action_geometry/A41_20261025'
CARD=PROJECTS/'experiment_cards/A41_ffhq_action_geometry_20261025.md'
BRANCH='results/ffhq-action-geometry-20261025'


def lock_status(old):
    """Read-only Linux advisory-lock owner report; never unlink, unlock or kill."""
    lock=Path(old).parent/'launcher.lock'
    if not lock.exists():return dict(path=str(lock),holders=[],action='read_only')
    st=lock.stat();holders=[]
    proc=Path('/proc/locks')
    if proc.exists():
        for line in proc.read_text().splitlines():
            fields=line.split()
            for j,field in enumerate(fields):
                parts=field.split(':')
                if len(parts)!=3:continue
                try:matches=(int(parts[0],16)==os.major(st.st_dev) and int(parts[1],16)==os.minor(st.st_dev)
                             and int(parts[2])==st.st_ino)
                except ValueError:continue
                if matches and j:
                    pid=fields[j-1]
                    try:command=Path('/proc',pid,'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
                    except OSError:command='not readable'
                    holders.append(dict(pid=pid,command=command,lock_record=line))
    return dict(path=str(lock),holders=holders,action='read_only; separate A41 lock; completed source only')


def audit_parent(old):
    p=source.check_plan(old);parent.verify_inputs(p)
    if p['experiment']!='A40_multitask_state_transfer':raise ValueError('Expected A40')
    records=[]
    for case in p['cases']:
        for kind in ('BASELINE','MEASURE','TIME'):
            item=parent.done(old,p,case,kind)
            if item is None:raise ValueError('A40 GPU work incomplete: '+case['key']+' '+kind+'; wait, no unlock')
            records.append((case,kind,item))
    if len(p['cases'])!=24:raise ValueError('A40 cohort differs')
    return p,records


def prepare(args):
    import torch
    from utils.diag_ffhq_four_tasks_unit import original_motion_kernel
    root,old=Path(args.root).resolve(),Path(args.a40_dir).resolve()
    print('A40_LOCK_STATUS:',json.dumps(lock_status(old),ensure_ascii=False),flush=True)
    if root.is_relative_to(old) or old.is_relative_to(root):raise ValueError('Output/source overlap')
    if root.exists():
        p=source.check_plan(root)
        if p['source_A40']!=str(old):raise ValueError('Frozen source changed')
        verify(p);print('A41 frozen plan reused; completed attempts retained.',flush=True);return
    previous,records=audit_parent(old)
    cases=[];originals={};identities={str(old/'plan.json'):source.digest(old/'plan.json')}
    for case,kind,item in records:
        if case['dataset']!='ffhq':continue
        path,r=item;identities[str(path)]=source.digest(path)
        if kind=='BASELINE':
            originals[case['key']]=dict(root=str(old),case=case,path=str(path),sha256=source.digest(path))
            for name,sha in r['artifacts'].items():identities[str(path.parent/name)]=sha
            identities[r['measurement']['path']]=r['measurement']['sha256']
        elif kind=='MEASURE':identities[str(path.parent/'outputs.pt')]=r['artifacts']['outputs.pt']
    for i,image in enumerate(previous['selected_images']['ffhq']):
        identities[image['image']]=image['image_sha256']
        for task in rules.TASKS:
            case=dict(image,dataset='ffhq',task=task,image_index=i,sigma=.05,sampling_seed=1000,
                measurement_seed=(1000+100000*rules.TASKS.index(task))+int(image['image_id']),
                mask_seed=2001000+int(image['image_id']),key=f'ffhq_{task}_{image["image_id"]}_s1000')
            if task in parent.rules.TASKS:
                # Preserve exact old case dictionary, measurement seed and identity.
                case=dict(originals[case['key']]['case'])
            cases.append(case)
    if len(cases)!=24 or len(originals)!=12:raise ValueError('Need six FFHQ images and twelve reusable baselines')
    own=[Path(__file__),Path(rules.__file__),CARD,PROJECTS/'run_a41_20261025.sh',
        PROJECTS/'utils/test_ffhq_action_geometry.py',PROJECTS/'utils/diag_ffhq_four_tasks_unit.py',
        PROJECTS.parent/'DPS/motionblur/motionblur.py',PROJECTS/'utils/archive_state_indicator_stage.py',
        PROJECTS/'utils/archive_refinement_confirmation.py']
    workers={**previous['worker_identity_files'],**{str(f):source.digest(f) for f in own}}
    configs=dict(previous['configs']['ffhq'])
    for task in ('inpainting','motion_deblur'):
        configs[task]=dict(configs['gaussian_deblur'],zeta_init=.1 if task=='inpainting' else .2)
    p=dict(experiment='A41_ffhq_action_geometry',source_A40=str(old),cases=cases,reused_baselines=originals,
        configs={'ffhq':configs},grid=previous['grid'],runtime=previous['runtime'],native_policy=previous['native_policy'],
        checkpoints={'ffhq':previous['checkpoints']['ffhq']},model_library_sources=previous['model_library_sources'],
        identity_files=identities,worker_identity_files=workers,planned_method_nfe=3600,planned_diagnostic_nfe=4212,
        points=[list(v) for v in rules.POINTS],gammas=list(rules.GAMMAS),features=rules.FEATURES,
        baseline_scope='A40 native local reference, exact adjoint, unit observation, LR .01, original first270; not author parity',
        limits='Six reused images, one seed. Paid fixed-state probes; no online or equal-cost control claim.')
    root.parent.mkdir(parents=True,exist_ok=True);P.coupled.archive.recovery.require_space(root.parent)
    kernel=original_motion_kernel(0)
    root.mkdir();parent.base.save_tensor(torch.from_numpy(kernel),root/'motion_kernel.pt')
    p['identity_files'][str(root/'motion_kernel.pt')]=source.digest(root/'motion_kernel.pt')
    snapshot=root/'source_A40_snapshot';snapshot.mkdir()
    for name in ('plan.json','plan.sha256'):shutil.copyfile(old/name,snapshot/name)
    for case,kind,(path,r) in records:
        target=snapshot/case['key']/kind;target.mkdir(parents=True)
        for name in ('run.json','run.sha256'):shutil.copyfile(path.parent/name,target/name)
    source.signed(root/'plan.json',p);(root/'experiment_card.md').write_text(CARD.read_text())
    print('A41 PREPARE PASS:24 cases, six images x four tasks; 12 old C reused; '
          '3600 method +4212 diagnostic=7812 NEW calls. A40 untouched.',flush=True)


def verify(p):
    for path,sha in p['identity_files'].items():
        if source.digest(path)!=sha:raise ValueError('Frozen input changed: '+path)


def training_context(root,p,case):
    ref=p['reused_baselines'].get(case['key'])
    return (Path(ref['root']),source.check_plan(Path(ref['root']))) if ref else (root,p)


def done(root,p,case):
    result=[]
    for path in sorted((root/case['key']/'RESPONSE').glob('attempt_*/run.json')):
        r=source.read(path)
        if r['status']!='complete':continue
        r=source.load_signed(path);reused=case['key'] in p['reused_baselines']
        expected={'C'}|{rules.arm(k,g) for k,_ in rules.POINTS for g in rules.GAMMAS if g!=1.}
        if (r['case']!=case or r['plan_sha256']!=source.digest(root/'plan.json') or
                r['calls']!=rules.expected_calls(p['grid'],reused) or r['nfe']!=len(r['calls']) or
                set(r['metrics'])!=expected or len(r['points'])!=4 or
                any(source.digest(path.parent/n)!=sha for n,sha in r['artifacts'].items()) or
                not r['gates'] or not all(g['passed'] for g in r['gates'].values()) or
                source.digest(r['training_path'])!=r['training_sha256'] or
                source.digest(r['measurement']['path'])!=r['measurement']['sha256']):
            raise ValueError('Invalid completed A41: '+str(path))
        if reused:
            if source.digest(r['reused_measure_path'])!=r['reused_measure_sha256']:raise ValueError('Changed reused final endpoints')
        result.append((path,r))
    if len(result)>1:raise ValueError('Multiple complete attempts; no output selection')
    return result[0] if result else None


def make_operator(root,case,device):
    import torch
    from modules.main_single import load_image_as_tensor
    from utils.diag_ffhq_four_tasks_unit import make_base,unit_operator
    gt=load_image_as_tensor(case['image']).to(device)
    kernel=torch.load(root/'motion_kernel.pt',map_location='cpu',weights_only=True).numpy()
    raw=make_base(case['task'],device,gt,case.get('mask_seed',0),kernel)
    mask=raw._mask_cache if case['task']=='inpainting' else None
    return unit_operator(raw).to(device),mask


def run_case(root,p,case,model,device,operator,mask):
    import torch
    from unittest.mock import patch
    from modules.main_single import load_image_as_tensor
    from utils.diag_cached_guidance_geometry import physical_descent
    from utils.diag_zaps_sampler_parity import rng_state,same_rng
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_trace_audit import relative_error
    from utils.metrics import compute_all_metrics
    if done(root,p,case):return
    reused=case['key'] in p['reused_baselines'];trainroot,trainplan=training_context(root,p,case)
    trained,state,original,y,meta,z,positions,noises=parent.setup_sampling(trainroot,trainplan,case,model,device,operator)
    olditem=parent.done(trainroot,trainplan,case,'MEASURE') if reused else None
    cached=torch.load(olditem[0].parent/'outputs.pt',map_location='cpu',weights_only=True) if reused else {}
    folder=parent.attempt(root,case,'RESPONSE');path=folder/'run.json';started=time.perf_counter()
    r=dict(kind='RESPONSE',case=case,plan_sha256=source.digest(root/'plan.json'),measurement=meta,
        training_path=str(trained[0]),training_sha256=source.digest(trained[0]),points=[],gates={},arm_seconds={})
    if reused:r.update(reused_measure_path=str(olditem[0]),reused_measure_sha256=source.digest(olditem[0]))
    external=rng_state(device);outputs={};anchors={};history=parent.rules.risk.History()
    def np(v):return v.detach().cpu().numpy()
    with scan.counted(model,path,r,len(rules.expected_calls(p['grid'],reused))),torch.no_grad(),\
            model_rng_guard(model.model,device),patch('torch.randn_like',side_effect=RuntimeError('Unexpected draw')):
        x=state['init_noise'].to(device)
        for k,t in enumerate(positions.nominal):
            f=source.fields.field(z,y,x,t,positions)
            readout=parent.rules.response.sensor(z,y,f['clean'],f['q'],t,case['sigma'])
            gradient,_=physical_descent(z.A,f['clean'],y,sigma=case['sigma'])
            observed=history.append(float(positions.u[t]),np(f['clean'][0]),np(f['q'][0]),np(gradient[0]),readout,f['clip_fraction'])
            if k in {v[0] for v in rules.POINTS}:
                anchors[k]=(x.clone(),{n:f[n].clone() for n in ('clean','q')},observed['features'])
            x=parent.rules.risk.advance(z,x,f,positions,k,noises)
        outputs['C']=x.cpu();parity=relative_error(outputs['C'],original)
        if not math.isfinite(parity) or parity>1e-6:raise RuntimeError('C replay parity failed: '+str(parity))
        for k,t in rules.POINTS:
            center,f,old=anchors[k];clean_probes={};observations={};s=positions.nominal[k+1]
            torch.cuda.synchronize(device);now=time.perf_counter()
            with P.precision.ieee(torch):
                for g in rules.GAMMAS:
                    xx=parent.rules.risk.advance(z,center,f,positions,k,noises,g)
                    clean_probes[g]=source.fields.field(z,y,xx,s,positions)['clean']
                    observations[g]=operator.H(clean_probes[g])
                repeat_x=parent.rules.risk.advance(z,center,f,positions,k,noises,1.)
                repeat=source.fields.field(z,y,repeat_x,s,positions)['clean']
                repeat_error=relative_error(repeat,clean_probes[1.])
                response=rules.features({g:np(v) for g,v in clean_probes.items()},
                    {g:np(v) for g,v in observations.items()},np(y),case['sigma'],old,
                    np(mask) if mask is not None else None,np(repeat))
            torch.cuda.synchronize(device);probe_seconds=time.perf_counter()-now
            if not math.isfinite(repeat_error) or repeat_error>1e-7:raise RuntimeError('IEEE repeat failed')
            r['points'].append(dict(step=k,t=t,next_t=s,response=response,repeat_relative_error=repeat_error,
                probe_nfe=6,probe_seconds=probe_seconds))
            for g in rules.GAMMAS:
                if g==1.:continue
                name=rules.arm(k,g);torch.cuda.synchronize(device);now=time.perf_counter()
                if rules.is_reused(k,g,reused):outputs[name]=cached[parent.rules.risk.arm(k,g)]
                else:
                    xx=parent.rules.risk.advance(z,center,f,positions,k,noises,g)
                    for j in range(k+1,30):
                        ff=source.fields.field(z,y,xx,positions.nominal[j],positions)
                        xx=parent.rules.risk.advance(z,xx,ff,positions,j,noises)
                    outputs[name]=xx.cpu()
                torch.cuda.synchronize(device);r['arm_seconds'][name]=time.perf_counter()-now
            print('A41 POINT:',case['key'],'t=',t,'resolved=',response['resolved'],'calls=',r['nfe'],flush=True)
        if (r['calls']!=rules.expected_calls(p['grid'],reused) or not same_rng(external,rng_state(device)) or
                P.precision.precision_state(torch)!=p['native_policy'] or
                any(v.grad is not None for v in model.model.parameters()) or
                any(not torch.isfinite(out).all() for out in outputs.values()) or
                not all(torch.equal(v.cpu(),state[n]) for v,n in ((z.tau,'tau'),(z.zeta,'zeta'),(z.D,'D')))):
            raise RuntimeError('Budget/RNG/precision/parameter/output gate failed')
        parent.base.save_tensor(outputs,folder/'outputs.pt');r['artifacts']={'outputs.pt':source.digest(folder/'outputs.pt')}
        if mask is not None:
            parent.base.save_tensor(mask.cpu(),folder/'mask.pt');r['artifacts']['mask.pt']=source.digest(folder/'mask.pt')
        source.WRITE(path,dict(r,status='sampled'))
        # Every output and every decision-free probe has been frozen before loading evaluation GT.
        gt=load_image_as_tensor(case['image']).cpu();r['metrics']={};r['informative_loss']={}
        for name,out in outputs.items():
            residual=(y-operator.H(out.to(device))).double()
            r['metrics'][name]=dict(loss=float(residual.square().mean()/(2*case['sigma']**2)),
                                  **compute_all_metrics(out,gt,lpips_net='alex'))
            active=residual if mask is None else residual[mask.bool().expand_as(residual)]
            r['informative_loss'][name]=float(active.square().mean()/(2*case['sigma']**2))
        if any(not math.isfinite(v) for m in r['metrics'].values() for v in m.values()):
            raise RuntimeError('Nonfinite final quality')
        r['gates']=dict(original_C=dict(passed=True,relative_error=parity),budget_rng_precision=dict(passed=True),
                       parameters_fixed=dict(passed=True),GT_after_outputs=dict(passed=True),IEEE_repeated=dict(passed=True))
    r.update(status='complete',seconds=time.perf_counter()-started);source.signed(path,r)
    print('A41 COMPLETE:',case['key'],'diagnostic calls=',r['nfe'],flush=True)


def worker(args):
    import fcntl
    import torch
    from modules.main_single import load_diffusion_model
    root=Path(args.root).resolve();p=source.check_plan(root);verify(p)
    with (root/f'shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cases=[c for c in p['cases'] if c['image_index']%2==args.shard]
        if all(done(root,p,c) for c in cases):return
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
            gpu=torch.cuda.get_device_name(args.device),NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),
            TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        if runtime!=p['runtime'] or P.precision.precision_state(torch)!=p['native_policy']:
            raise ValueError('Native runtime/precision changed')
        if parent.model_sources()!=p['model_library_sources']:raise ValueError('Model source changed')
        torch.set_num_threads(2);P.coupled.archive.recovery.require_space(root)
        with scan.preserved_rng():
            model=load_diffusion_model('ffhq',args.device);model.model.eval()
            if (source.digest(model.ckpt_path)!=p['checkpoints']['ffhq']['sha256'] or model.num_steps!=1000 or
                    any(v.dtype!=torch.float32 for v in model.model.parameters())):raise ValueError('Model identity changed')
            for case in cases:
                if done(root,p,case):continue
                operator,mask=make_operator(root,case,args.device);gate=parent.operator_gate(operator,args.device)
                if case['key'] not in p['reused_baselines']:
                    parent.training(root,p,case,model,args.device,operator,gate)
                run_case(root,p,case,model,args.device,operator,mask)
                del operator;gc.collect();torch.cuda.empty_cache()
            del model


def pictures(path,r):
    import torch
    from PIL import Image,ImageDraw
    from modules.main_single import load_image_as_tensor
    from utils.metrics import _to_uint8_numpy
    from utils.diag_cached_guidance_final_value import font
    outputs=torch.load(path.parent/'outputs.pt',map_location='cpu',weights_only=True)
    names=('GT','C')+tuple(rules.arm(k,2.) for k,_ in rules.POINTS)
    sheet=Image.new('RGB',(24+276*len(names),360),'white');draw=ImageDraw.Draw(sheet)
    draw.text((12,6),r['case']['key']+' | fixed x2 pulse at four SEPARATE positions; no output selection',fill='black',font=font(16))
    for j,name in enumerate(names):
        x=12+276*j;draw.text((x,34),name,fill='black',font=font(16))
        if name!='GT':
            m=r['metrics'][name];draw.text((x,58),f'P {m["psnr"]:.3f} S {m["ssim"]:.4f} L {m["lpips"]:.4f}',fill='black',font=font(12))
        image=Image.fromarray(_to_uint8_numpy(load_image_as_tensor(r['case']['image']) if name=='GT' else outputs[name]))
        sheet.paste(image,(x,90))
    sheet.save(path.parent/'fixed_pulse_comparison.png')


def archive(root):
    """Independent results ref; no A40 inventory/mutation, even if old exporter is busy."""
    import hashlib
    import tempfile
    from utils.archive_refinement_confirmation import inventory
    from utils.archive_state_indicator_stage import ALLOWED,commit_results
    files=inventory(root);fingerprint=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    parentdir=PROJECTS/'server_records';parentdir.mkdir(exist_ok=True)
    bundle=parentdir/('ffhq_action_geometry_'+fingerprint[:16])
    if not bundle.exists():
        with tempfile.TemporaryDirectory(prefix='.a41_export_',dir=parentdir) as tmp:
            work=Path(tmp)/'bundle';work.mkdir();(work/'.gitattributes').write_text('* -text\n')
            manifest=dict(sources=str(root),fingerprint=fingerprint,files=[],excluded=[])
            for name,meta in files.items():
                item=dict(path=name,**meta)
                if Path(name).suffix.lower() not in ALLOWED|{'.sha256'}:manifest['excluded'].append(item);continue
                target=work/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,target)
                if source.digest(target)!=meta['sha256']:raise ValueError('Copy differs')
                manifest['files'].append(item)
            if inventory(root)!=files:raise ValueError('A41 source changed during export')
            (work/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');work.rename(bundle)
    commit=commit_results(bundle,BRANCH,message='Archive A41 FFHQ four-task response geometry\n')
    return str(bundle),commit


def summarize(args):
    root=Path(args.root).resolve();p=source.check_plan(root);verify(p);items=[]
    for case in p['cases']:
        item=done(root,p,case)
        if item:items.append(item)
    result=rules.analyze([r for _,r in items]);attempts=[source.read(f) for f in root.glob('*/*/attempt_*/run.json')]
    result.update(experiment=p['experiment'],completed_cases=len(items),planned_cases=24,
        method_nfe=sum(r['nfe'] for r in attempts if r['kind']=='BASELINE'),
        diagnostic_nfe=sum(r['nfe'] for r in attempts if r['kind']=='RESPONSE'),
        planned_method_nfe=3600,planned_diagnostic_nfe=4212,partial_attempts=sum(r['status']!='complete' for r in attempts),
        cost_is_lower_bound_if_abruptly_interrupted=any(r['status'] in ('running','sampled') for r in attempts),
        resolved_states=sum(v['response']['resolved'] for _,r in items for v in r['points']),
        total_states=4*len(items),baseline_scope=p['baseline_scope'])
    source.signed(root/'summary.json',result)
    print('A41 Summary:',json.dumps({k:v for k,v in result.items() if k!='tasks'},ensure_ascii=False),flush=True)
    for task in result.get('tasks',[]):
        print('A41 COVERAGE/OPPORTUNITY:',task['task'],json.dumps(dict(coverage=task['coverage'],
            opportunity=[{k:v for k,v in point.items() if k!='surface'} for point in task['opportunity']]),ensure_ascii=False),flush=True)
        for mode,split in task['validation'].items():
            print('A41 TASK:',task['task'],mode,json.dumps(dict(information=split['information'],mae=split['mae'],
                increments=split['increments']),ensure_ascii=False),flush=True)
    print('Records:',root,flush=True)
    if len(items)!=24:raise SystemExit('A41 incomplete; completed attempts retained; check shard logs')
    if args.archive:
        for path,r in items:pictures(path,r)
        bundle,commit=archive(root);print('Archive:',bundle,'Results commit:',commit,'Branch:',BRANCH,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=ROOT);parser.add_argument('--a40-dir',default=parent.ROOT)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--shard',type=int,choices=(0,1))
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--archive',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare(args)
    elif args.shard is not None:worker(args)
    else:summarize(args)


if __name__=='__main__':main()
