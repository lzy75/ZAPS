"""A38: UP_REG/DOWN_REG cached continuations; 1080 new diagnostic UNet calls."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS))
from utils import diag_noise_regularized_feedback as base
from utils import one_sided_noise_feedback as rules

source,P,scan = base.source,base.P,base.scan
ROOT = '/data0/lzy/ZAPS-results/diag_one_sided_noise_feedback/A38_20261025'
CARD = PROJECTS/'experiment_cards/A38_one_sided_noise_feedback_20261025.md'


def references(p,case):
    for directory in (p['source_A36'],p['source_A37']):
        for name in ('plan.json','summary.json'):
            path=Path(directory)/name
            if source.digest(path)!=p['identity_files'][str(path)]:raise ValueError('Frozen parent changed: '+str(path))
    p36=source.check_plan(Path(p['source_A36']))
    training,feedback=base.source_records(Path(p['source_A36']),p36,case)
    a37=Path(p['source_A37']);p37=source.check_plan(a37)
    item=base.done(a37,p37,case)
    if item is None:raise ValueError('A37 incomplete: '+case['key'])
    for path in (training[0],feedback[0],item[0]):
        if source.digest(path)!=p['identity_files'][str(path)]:raise ValueError('Frozen source differs: '+str(path))
    return training,feedback,item


def prepare(args):
    root,old=Path(args.root).resolve(),Path(args.a37_dir).resolve()
    if root.is_relative_to(old) or old.is_relative_to(root):raise ValueError('Output overlaps parent')
    if root.exists():
        p=source.check_plan(root)
        if p['source_A37']!=str(old):raise ValueError('Frozen source differs')
        print('A38 frozen plan reused; completed outputs retained.',flush=True);return
    p37=source.check_plan(old);s37=source.load_signed(old/'summary.json')
    if p37['experiment']!='A37_noise_regularized_feedback' or s37['completed_cases']!=18:
        raise ValueError('Need eighteen complete A37 cases')
    p=dict(p37);p['source_A37']=str(old);p['identity_files']=dict(p37['identity_files'])
    records=[];a37_records={}
    for case in p['cases']:
        item=base.done(old,p37,case)
        if item is None:raise ValueError('A37 case incomplete')
        path,r=item;a37_records[case['key']]=str(path)
        for f in (path,path.parent/'outputs.pt'):p['identity_files'][str(f)]=source.digest(f)
        p36=source.check_plan(Path(p['source_A36']))
        _,ref=base.source_records(Path(p['source_A36']),p36,case)
        combined=dict(r,metrics={'C':ref[1]['metrics']['C'],'OLD':ref[1]['metrics']['STATE'],
                                'TIME':ref[1]['metrics']['TIME'],**r['metrics']})
        records.append(combined)
    if base.interpretation(records)['groups']!=s37['groups']:raise ValueError('A37 summary differs from cases')
    for name in ('plan.json','summary.json'):p['identity_files'][str(old/name)]=source.digest(old/name)
    own=[Path(__file__),Path(rules.__file__),CARD,PROJECTS/'run_a38_20261025.sh',
         PROJECTS/'utils/test_one_sided_noise_feedback.py']
    p['worker_identity_files']={**p37['worker_identity_files'],**{str(f):source.digest(f) for f in own}}
    p.update(experiment='A38_one_sided_noise_feedback',a37_records=a37_records,
             planned_method_nfe=0,planned_diagnostic_nfe=18*rules.CALLS,
             source_A37_cost=dict(method_nfe=s37['method_nfe'],diagnostic_nfe=s37['diagnostic_nfe'],
                                  cost_is_lower_bound=s37['cost_is_lower_bound_if_abruptly_interrupted']),
             scope='Two-sided mechanism decomposition on six reused development images, three measurement sigmas, one seed. '
                   'UP_REG regularizes only current LS>1; DOWN_REG only LS<1. Lambda1/upper2/deadband fixed. '
                   'First270 original calls shared; every scientific arm300. No state-superiority, fresh confirmation, '
                   'curvature, convergence, author-parity or NFE-saving claim.')
    root.parent.mkdir(parents=True,exist_ok=True);P.coupled.archive.recovery.require_space(root.parent)
    root.mkdir();source.signed(root/'plan.json',p);(root/'experiment_card.md').write_text(CARD.read_text())
    print('A38 PREPARE PASS:18 cached cases x2 arms x30=1080 new diagnostic calls; zero retraining.',flush=True)


def done(root,p,case):
    complete=[]
    for path in sorted((root/case['key']).glob('attempt_*/run.json')):
        r=source.read(path)
        if r['status']!='complete':continue
        r=source.load_signed(path)
        if (r['case']!=case or r['plan_sha256']!=source.digest(root/'plan.json')
                or r['method_nfe']!=0 or r['nfe']!=rules.CALLS
                or r['calls']!=list(reversed(p['grid']))*2
                or set(r['metrics'])!=set(rules.ARMS) or set(r['traces'])!=set(rules.ARMS)
                or any([v['t'] for v in r['traces'][a]]!=list(reversed(p['grid'])) for a in rules.ARMS)
                or not r['gates'] or not all(g['passed'] for g in r['gates'].values())
                or source.digest(path.parent/'outputs.pt')!=r['outputs_sha256']
                or r['source_A37_sha256']!=p['identity_files'][p['a37_records'][case['key']]]):
            raise ValueError('Invalid complete A38 record: '+str(path))
        complete.append((path,r))
    if len(complete)>1:raise ValueError('Multiple complete attempts; no output selection')
    return complete[0] if complete else None


def run_case(root,p,case,model,device,operator):
    import torch
    from unittest.mock import patch
    from modules.zaps_algorithm import ZAPS
    from modules.main_single import load_image_as_tensor
    from utils.diag_zaps_sampler_parity import rng_state,same_rng
    from utils.diag_zaps_guidance_clipping import saved_cuda_selection,restore_sampling_rng
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.metrics import compute_all_metrics
    if done(root,p,case):return
    training,feedback,a37=references(p,case)
    folder=root/case['key'];folder.mkdir(exist_ok=True)
    attempt=folder/f'attempt_{len(list(folder.glob("attempt_*"))):03d}';attempt.mkdir();path=attempt/'run.json'
    started=time.perf_counter()
    state=torch.load(training[0].parent/'pre_last_state.pt',map_location='cpu',weights_only=True)
    saved=torch.load(training[1]['measurement']['path'],map_location='cpu',weights_only=True)
    if saved['image_sha256']!=case['image_sha256'] or saved['seed']!=case['measurement_seed']:
        raise ValueError('Observation identity differs')
    y=saved['mean'].to(device)+case['sigma']*saved['standard_noise'].to(device)
    z=ZAPS(model,operator,img_size=256,**p['config']);z.tau=state['tau'].to(device)
    with torch.no_grad():z.zeta.copy_(state['zeta'].to(device));z.D.copy_(state['D'].to(device))
    if z.tau.tolist()!=p['grid']:raise ValueError('Snapshot grid differs')
    positions=source.fields.policy.Positions(source.fields.policy.noise_u(z.dm.alphas_cumprod.cpu().numpy()),list(reversed(p['grid'])))
    with scan.preserved_rng():
        mapping=saved_cuda_selection(state,device);restore_sampling_rng(state,device,mapping['saved_sampling_index'])
        noises=[torch.randn_like(state['init_noise'].to(device)) for _ in range(29)]
    r=dict(case=case,method_nfe=0,plan_sha256=source.digest(root/'plan.json'),
           source_A37_sha256=source.digest(a37[0]),gates={},sensor_seconds=0.,arm_seconds={})
    external=rng_state(device);center_error=0.;outputs={};traces={}
    with scan.counted(model,path,r,rules.CALLS):
        with torch.no_grad(),model_rng_guard(model.model,device),patch('torch.randn_like',side_effect=RuntimeError('Unexpected draw')):
            for arm in rules.ARMS:
                torch.cuda.synchronize(device);arm_started=time.perf_counter()
                x=state['init_noise'].to(device);trace=[]
                for k,t in enumerate(positions.nominal):
                    f=source.fields.field(z,y,x,t,positions)
                    if arm==rules.ARMS[0] and k==0:
                        fp=base.parent.rules.sensor(z,y,f['clean'],f['q'],t,case['sigma']);old=feedback[1]['traces']['STATE'][0]
                        for name in ('gamma','R','qRMS','cosine','relative_response'):
                            v,target=fp[name],old[name]
                            if v is None or target is None:
                                if v!=target:raise RuntimeError('Unresolved first center differs')
                            else:center_error=max(center_error,abs(v-target)/max(abs(target),1e-12))
                        if (not math.isfinite(center_error) or center_error>1e-6
                                or abs(f['clip_fraction']-old['clip'])>2e-5):raise RuntimeError('First-center scalar parity failed')
                    now=time.perf_counter();readout=rules.sensor(z,y,f['clean'],f['q'],t,case['sigma'],arm)
                    r['sensor_seconds']+=time.perf_counter()-now;gamma=readout['gamma']
                    trace.append(dict(step=k,t=t,clip=f['clip_fraction'],**readout))
                    original=P.terminal.terminal_output(z,x,f) if k==29 else P.coupled.posterior(
                        x,f['clean'],t,positions.nominal[k+1],z.dm.alphas_cumprod,noises[k])+f['q']
                    x=original if gamma==1. else original+(gamma-1.)*f['q']
                    if not torch.isfinite(x).all():raise RuntimeError('Nonfinite output; no replacement')
                torch.cuda.synchronize(device);r['arm_seconds'][arm]=time.perf_counter()-arm_started
                outputs[arm]=x.cpu();traces[arm]=trace
        if (r['nfe']!=rules.CALLS or r['calls']!=list(reversed(p['grid']))*2
                or not same_rng(external,rng_state(device)) or P.precision.precision_state(torch)!=p['native_policy']
                or not all(torch.equal(v.cpu(),state[n]) for v,n in ((z.tau,'tau'),(z.zeta,'zeta'),(z.D,'D')))):
            raise RuntimeError('Budget/RNG/precision/parameter gate failed')
        base.parent.save_tensor(outputs,attempt/'outputs.pt')
        r.update(status='sampled',traces=traces,outputs_sha256=source.digest(attempt/'outputs.pt'));source.WRITE(path,r)
        gt=load_image_as_tensor(case['image']).cpu()
        metrics={arm:dict(loss=float((y-z.A.H(out.to(device))).double().square().mean()/(2*case['sigma']**2)),
                          **compute_all_metrics(out,gt,lpips_net='alex')) for arm,out in outputs.items()}
        if any(not math.isfinite(v) for m in metrics.values() for v in m.values()):raise RuntimeError('Nonfinite metric')
        r.update(metrics=metrics,gates=dict(source_first_center=dict(passed=True,relative_error=center_error,scope='scalar fingerprint'),
                 paired_noise_budget=dict(passed=True),parameters_precision=dict(passed=True),GT_after_outputs=dict(passed=True)))
    r.update(status='complete',seconds=time.perf_counter()-started);source.signed(path,r)
    print('A38 CASE:',case['key'],'complete; diagnostic calls=',r['nfe'],flush=True)


def worker(args):
    import fcntl
    import torch
    from modules.degradations import get_operator
    from utils.diag_ffhq_regression import TransposeMode
    from utils.diag_imagenet_gaussian_unit_domain import unit_operator
    root=Path(args.root).resolve();p=source.check_plan(root)
    with (root/f'shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cases=[c for c in p['cases'] if p['image_ids'].index(c['image_id'])%2==args.shard]
        if all(done(root,p,c) for c in cases):return
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                     gpu=torch.cuda.get_device_name(args.device),NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),
                     TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        if runtime!=p['runtime'] or P.precision.precision_state(torch)!=p['native_policy']:raise ValueError('Native runtime differs')
        torch.set_num_threads(2);P.coupled.archive.recovery.require_space(root)
        with scan.preserved_rng():
            model=P.coupled.archive.recovery.load_model(p,args.device)
            operator=unit_operator(TransposeMode(get_operator('super_resolution',device=args.device,scale_factor=4,noise_sigma=0.),'legacy_bicubic').to(args.device)).to(args.device)
            for case in cases:
                if source.digest(case['image'])!=case['image_sha256']:raise ValueError('Target image differs')
                run_case(root,p,case,model,args.device,operator)
            source.check_plan(root)


def interpretation(records):
    groups=[]
    for sigma in base.parent.rules.SIGMAS:
        current=[r for r in records if r['case']['sigma']==sigma]
        g=dict(sigma=sigma,completed_cases=len(current),planned_cases=6)
        if current:
            g['comparisons']={f'{a}_{b}':base.parent.rules.contrast(current,a,b)
                              for a in rules.ARMS for b in ('C','OLD','REG','TIME','TIME_NOISE')}
            g['comparisons']['UP_REG_DOWN_REG']=base.parent.rules.contrast(current,'UP_REG','DOWN_REG')
            g['quality_candidates']=[a for a in rules.ARMS if len(current)==6
                 and g['comparisons'][a+'_C']['mean_delta']['psnr']>=.05
                 and g['comparisons'][a+'_C']['images_psnr_positive']>=4
                 and g['comparisons'][a+'_C']['quality_guard']]
            g['readouts']={a:dict(changed_fraction=statistics.mean(float(v['changed_from_OLD']) for r in current for v in r['traces'][a]),
                                 final_output_R_mean=statistics.mean(math.sqrt(2*r['metrics'][a]['loss']) for r in current),
                                 sampling_seconds_mean=statistics.mean(r['arm_seconds'][a] for r in current)) for a in rules.ARMS}
        groups.append(g)
    complete=all(g['completed_cases']==6 for g in groups)
    candidates=[a for a in rules.ARMS if complete and all(a in g['quality_candidates'] for g in groups)]
    return dict(classification=('ONE_SIDED_MECHANISM_CANDIDATE_ONLY' if candidates else
                                'MIXED_OR_NO_ONE_SIDED_QUALITY_SUPPORT') if complete else 'INCOMPLETE_PILOT',
                completed_cases=len(records),planned_cases=18,across_condition_candidates=candidates,groups=groups)


def pictures(path,r,feedback,a37):
    import torch
    from PIL import Image,ImageDraw
    from modules.main_single import load_image_as_tensor
    from utils.metrics import _to_uint8_numpy
    from utils.diag_cached_guidance_final_value import font
    old=torch.load(feedback[0].parent/'outputs.pt',map_location='cpu',weights_only=True)
    prior=torch.load(a37[0].parent/'outputs.pt',map_location='cpu',weights_only=True)
    new=torch.load(path.parent/'outputs.pt',map_location='cpu',weights_only=True)
    outputs={'C':old['C'],'OLD':old['STATE'],'REG':prior['REG'],**new}
    names=('GT','C','OLD','REG',*rules.ARMS)
    sheet=Image.new('RGB',(20+276*len(names),680),'white');draw=ImageDraw.Draw(sheet)
    draw.text((20,10),r['case']['key']+' | one-sided mechanism ablation; all arms300 NFE',fill='black',font=font(18))
    for j,arm in enumerate(names):
        raw=Image.fromarray(_to_uint8_numpy(load_image_as_tensor(r['case']['image']) if arm=='GT' else outputs[arm]))
        full=raw.copy();ImageDraw.Draw(full).rectangle((64,48,192,176),outline='#f8bd46',width=2);x=20+276*j
        draw.text((x,45),arm,fill='black',font=font(17))
        if arm!='GT':
            m=r['metrics'][arm];draw.text((x,70),f'P {m["psnr"]:.3f} S {m["ssim"]:.4f}',fill='black',font=font(14))
            draw.text((x,90),f'LPIPS {m["lpips"]:.4f}',fill='black',font=font(14))
        sheet.paste(full,(x,115));sheet.paste(raw.crop((64,48,192,176)).resize((256,256),Image.Resampling.NEAREST),(x,390))
    sheet.save(path.parent/'comparison.png')


def summarize(args):
    root=Path(args.root).resolve();p=source.check_plan(root);combined=[];items=[]
    for case in p['cases']:
        item=done(root,p,case)
        if item is None:continue
        _,ref,a37=references(p,case)
        metrics={'C':ref[1]['metrics']['C'],'OLD':ref[1]['metrics']['STATE'],'TIME':ref[1]['metrics']['TIME'],
                 **a37[1]['metrics'],**item[1]['metrics']}
        r=dict(item[1],metrics=metrics);combined.append(r);items.append((item[0],r,ref,a37))
    result=interpretation(combined);attempts=[source.read(f) for f in root.glob('*/attempt_*/run.json')]
    result.update(experiment=p['experiment'],method_nfe=0,diagnostic_nfe=sum(r['nfe'] for r in attempts),
                  planned_diagnostic_nfe=p['planned_diagnostic_nfe'],deployment_nfe_per_arm=300,
                  partial_attempts=sum(r['status']!='complete' for r in attempts),
                  cost_is_lower_bound_if_abruptly_interrupted=any(r['status'] in ('running','sampled') for r in attempts),
                  source_A36_cost=p['source_cost'],source_A37_cost=p['source_A37_cost'],
                  limits=p['scope']+' Six dependent image clusters; unadjusted bootstrap is descriptive. '
                         'Arm sampling time includes diagnostic instrumentation; not an end-to-end speed comparison.')
    source.signed(root/'summary.json',result)
    if args.archive:
        for item in items:pictures(*item)
    compact={k:v for k,v in result.items() if k!='groups'}
    compact['groups']=[{**{k:v for k,v in g.items() if k!='comparisons'},
                        'comparisons':{name:{k:v for k,v in c.items() if k!='per_image'}
                                       for name,c in g.get('comparisons',{}).items()}} for g in result['groups']]
    print('A38 Summary:',json.dumps(compact,ensure_ascii=False),flush=True);print('Records:',root,flush=True)
    if args.archive:
        from utils.archive_refinement_confirmation import export_and_commit
        bundle,commit=export_and_commit({'A38':str(root)});print('Archive:',bundle,'Results commit:',commit,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=ROOT)
    p.add_argument('--a37-dir',default=base.ROOT);p.add_argument('--prepare',action='store_true')
    p.add_argument('--shard',type=int,choices=(0,1));p.add_argument('--device',default='cuda:0');p.add_argument('--archive',action='store_true')
    args=p.parse_args()
    if args.prepare:prepare(args)
    elif args.shard is not None:worker(args)
    else:summarize(args)


if __name__=='__main__':main()
