"""A47 final frozen-mapping boundary check: sigma .10, four tasks, no parameter rescue."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import diag_frozen_four_task_confirmation as core
from utils import high_noise_boundary as rules

source,a40,old,P,scan=core.source,core.a40,core.old,core.P,core.scan
ROOT='/data0/lzy/ZAPS-results/diag_high_noise_boundary/A47_20261025'
CARD=PROJECTS/'experiment_cards/A47_frozen_high_noise_boundary_20261025.md'
BRANCH='results/ffhq-high-noise-boundary-20261025'


def prepare(args):
    root,parent=Path(args.root).resolve(),Path(args.a46_dir).resolve()
    if root.is_relative_to(parent) or parent.is_relative_to(root):raise ValueError('Source/output overlap')
    if root.exists():
        p=source.check_plan(root);core.a42.verify(p)
        if p['source_A46']!=str(parent):raise ValueError('Frozen source differs')
        print('A47 frozen boundary plan reused; no reselection or retuning.',flush=True);return
    before=source.check_plan(parent);core.a42.verify(before)
    audit_path=parent.parent/'A46b_cached_20261025/report.json'
    audit=source.load_signed(audit_path)
    if audit['source_plan_sha256']!=source.digest(parent/'plan.json'):raise ValueError('Cached audit source differs')
    for case in before['cases']:
        if core.phase_done(parent,before,case,'finish') is None:raise ValueError('A46 incomplete')
    if before['weights_frozen']!=core.rules.WEIGHTS:raise ValueError('Frozen lambda changed')
    if tuple(before['selected_images'][:3])!=rules.VALIDATION:raise ValueError('A46 ordered subset differs')
    devroot=Path(before['source_A45']);dev=source.check_plan(devroot)
    identities={str(parent/'plan.json'):source.digest(parent/'plan.json'),str(devroot/'plan.json'):source.digest(devroot/'plan.json'),
        str(audit_path):source.digest(audit_path)}
    cases=[];measurements={}
    for cohort,pool,images in (('calibration',dev['cases'],rules.CALIBRATION),('validation',before['cases'],rules.VALIDATION)):
        for original in pool:
            if original['image_id'] not in images:continue
            if original['sigma']!=.05:raise ValueError('Expected paired sigma=.05 source')
            case=dict(original,cohort=cohort,sigma=.10,key=original['key']+'_n100');cases.append(case)
            if cohort=='calibration':
                record=Path(dev['source_records'][original['key']]);r=source.load_signed(record)
                meta=r['measurement'];identities[str(record)]=source.digest(record)
            else:
                item=a40.done(parent,before,original,'BASELINE')
                if item is None:raise ValueError('Missing A46 baseline')
                record,r=item;meta=r['measurement'];identities[str(record)]=source.digest(record)
            if source.digest(meta['path'])!=meta['sha256']:raise ValueError('Source measurement changed')
            identities[meta['path']]=meta['sha256'];identities[case['image']]=case['image_sha256']
            measurements[case['key']]=meta
    if len(cases)!=24:raise ValueError('Need3 calibration +3 validation images x4 tasks')
    identities[str(parent/'motion_kernel.pt')]=source.digest(parent/'motion_kernel.pt')
    own=[Path(__file__),Path(rules.__file__),CARD,PROJECTS/'run_a47_20261025.sh',PROJECTS/'utils/test_high_noise_boundary.py',
         PROJECTS/'utils/diag_a46_cached_audit.py']
    p={k:before[k] for k in ('configs','runtime','native_policy','checkpoint','model_library_sources','grid','weights_frozen','step','t','next_t')}
    p.update(experiment='A47_frozen_high_noise_boundary',source_A46=str(parent),cases=cases,sigma=.10,
        calibration_images=list(rules.CALIBRATION),validation_images=list(rules.VALIDATION),
        fixed_reference=before['matched_fixed'],source_measurements=measurements,
        identity_files=identities,worker_identity_files={**before['worker_identity_files'],**{str(f):source.digest(f) for f in own}},
        planned_method_nfe=7200,planned_diagnostic_nfe_max=1665,planned_total_nfe_max=8865,
        policy_aliases=dict(TASK_FIXED='FROZEN_SIGMA_005_FIXED',MATCHED_FIXED='CALIBRATED_SIGMA_010_FIXED'),
        limits='Frozen-mapping boundary test after mixed A46; no rescue tuning. Three reused validation image clusters '
            'under a NEW sigma=.10 condition, not new photographs. No statistical significance, broad transfer, full-path or NFE-saving claim.')
    root.parent.mkdir(parents=True,exist_ok=True);P.coupled.archive.recovery.require_space(root.parent);root.mkdir()
    shutil.copyfile(parent/'motion_kernel.pt',root/'motion_kernel.pt')
    p['identity_files'][str(root/'motion_kernel.pt')]=source.digest(root/'motion_kernel.pt')
    for case in cases:
        target=root/'measurements'/case['dataset']/case['task']/(case['image_id']+'.pt');target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(measurements[case['key']]['path'],target)
        if source.digest(target)!=measurements[case['key']]['sha256']:raise ValueError('Paired measurement copy differs')
        p['identity_files'][str(target)]=source.digest(target)
    core.signed(root/'plan.json',p);shutil.copyfile(CARD,root/'experiment_card.md')
    snapshot=root/'source_snapshot';snapshot.mkdir();shutil.copyfile(audit_path,snapshot/'A46b_cached_report.json')
    print('A47 PREPARE PASS: sigma=.10; 3 old calibration images +3 same A46 validation photos x4 tasks; '
        'lambda unchanged; NEW7200 method +<=1665 diagnostic =<=8865 calls.',flush=True)


def freeze(root,p):
    items=[core.phase_done(root,p,c,'choose') for c in p['cases']]
    if not all(items):raise ValueError('Need all24 decisions before new final quality')
    choices={r['case']['key']:r['gamma'] for _,r in items}
    assignments,matched=rules.allocation(p['cases'],choices,p['fixed_reference'])
    value=dict(plan_sha256=source.digest(root/'plan.json'),choices={str(path):source.digest(path) for path,_ in items},
        assignments=assignments,noise_matched_fixed=matched,GT_used=False)
    file=root/'allocation.json'
    if file.exists():
        if source.load_signed(file)!=value:raise ValueError('Frozen allocation differs')
    else:core.signed(file,value)
    print('A47 FREEZE PASS: sigma=.10 fixed multipliers=',matched,'; 12 validation assignments frozen before final labels.',flush=True)


def worker(args):
    import fcntl
    import torch
    from unittest.mock import patch
    from modules.main_single import load_diffusion_model
    root=Path(args.root).resolve();p=source.check_plan(root);core.a42.verify(p)
    with (root/f'{args.phase}_shard{args.shard}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        eligible=p['cases'] if args.phase=='choose' else [c for c in p['cases'] if c['cohort']=='validation']
        cases=[c for i,c in enumerate(eligible) if i%2==args.shard]
        if all(core.phase_done(root,p,c,args.phase) for c in cases):return
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
            gpu=torch.cuda.get_device_name(args.device),NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),
            TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        if runtime!=p['runtime'] or P.precision.precision_state(torch)!=p['native_policy']:raise ValueError('Runtime changed')
        if a40.model_sources()!=p['model_library_sources']:raise ValueError('Model library changed')
        torch.set_num_threads(2);P.coupled.archive.recovery.require_space(root)
        with scan.preserved_rng():
            model=load_diffusion_model('ffhq',args.device);model.model.eval()
            if (source.digest(model.ckpt_path)!=p['checkpoint']['sha256'] or model.num_steps!=1000 or
                    any(v.dtype!=torch.float32 for v in model.model.parameters())):raise ValueError('Checkpoint/dtype changed')
            for case in cases:
                if core.phase_done(root,p,case,args.phase):continue
                if args.phase=='choose':
                    operator,_=old.make_operator(root,case,args.device);gate=a40.operator_gate(operator,args.device)
                    with patch.object(scan,'counted',core.baseline_counted):a40.training(root,p,case,model,args.device,operator,gate)
                    core.choose(root,p,case,model,args.device)
                else:core.finish(root,p,case,model,args.device)
                gc.collect();torch.cuda.empty_cache()


def summarize(args):
    root=Path(args.root).resolve();p=source.check_plan(root);core.a42.verify(p)
    validation=[c for c in p['cases'] if c['cohort']=='validation']
    items=[item for c in validation if (item:=core.phase_done(root,p,c,'finish')) is not None]
    baseline=[source.read(f) for f in root.glob('*/BASELINE/attempt_*/run.json')]
    diagnostic=[source.read(f) for f in root.glob('*/*_*/run.json')];groups=rules.summarize([r for _,r in items])
    complete=len(items)==12;quality=sum(g['quality_candidate'] for g in groups);matching=sum(g['state_matching_candidate'] for g in groups)
    allocation=source.load_signed(root/'allocation.json') if (root/'allocation.json').exists() else {}
    result=dict(experiment=p['experiment'],classification=('FOUR_TASK_HIGH_NOISE_QUALITY_CANDIDATE' if quality==4 else
        'MIXED_OR_NO_FROZEN_HIGH_NOISE_QUALITY_SUPPORT') if complete else 'INCOMPLETE_PILOT',
        state_matching_classification='MATCHING_SIGNAL_IN_'+str(matching)+'_TASKS',completed_validation_cases=len(items),
        planned_validation_cases=12,validation_image_clusters=3,calibration_image_clusters=3,
        method_nfe=sum(r['nfe'] for r in baseline),diagnostic_nfe=sum(r['nfe'] for r in diagnostic),
        total_new_nfe=sum(r['nfe'] for r in baseline+diagnostic),planned_total_nfe_max=8865,
        calibration_method_nfe=sum(r['nfe'] for r in baseline if r['case']['cohort']=='calibration'),
        partial_attempts=sum(r['status']!='complete' for r in baseline+diagnostic),
        cost_is_lower_bound_if_abruptly_interrupted=any(r['status']=='running' for r in baseline+diagnostic),
        quality_candidate_tasks=quality,state_matching_candidate_tasks=matching,groups=groups,
        weights_frozen=p['weights_frozen'],sigma=.10,noise_matched_fixed=allocation.get('noise_matched_fixed'),
        policy_aliases=p['policy_aliases'],limits=p['limits']+' Source A46 failures retained. '
        'A gain over the old .05 fixed rule alone is noise-condition adaptation, not current-image matching. '
        'Original scientific sampling300 plus paid full sensor<=317 vs fixed300; calibration is also paid.')
    core.signed(root/'summary.json',result);print('A47 Summary:',json.dumps(core.sensor.plain(result),ensure_ascii=False,allow_nan=False),flush=True)
    print('Records:',root,flush=True)
    if not complete:raise SystemExit('A47 incomplete; attempts retained')
    if args.archive:
        render(items)
        from unittest.mock import patch
        with patch.object(core.parent.parent,'BRANCH',BRANCH):bundle,commit=core.parent.parent.archive(root)
        print('Archive:',bundle,'Results commit:',commit,'Branch:',BRANCH,flush=True)


def render(items):
    import torch
    from PIL import Image,ImageDraw
    from utils.diag_cached_guidance_final_value import font
    torch.set_num_threads(2)
    for path,r in items:
        outputs=torch.load(path.parent/'outputs.pt',map_location='cpu',weights_only=True)
        sheet=Image.new('RGB',(24+276*(1+len(core.rules.POLICIES)),382),'white');draw=ImageDraw.Draw(sheet)
        draw.text((12,5),r['case']['key']+' | SAME A46 photo, NEW noise=.10; all predefined arms',fill='black',font=font(16))
        for j,name in enumerate(('GT',)+core.rules.POLICIES):
            x=12+276*j
            if name=='GT':
                im=Image.open(r['case']['image']).convert('RGB').resize((256,256),Image.Resampling.LANCZOS);label='GT'
            else:
                ev=r['evaluation'][name];m=ev['metrics'];a=outputs[ev['output']][0].numpy().transpose(1,2,0)
                im=Image.fromarray(((np.clip(a,-1,1)+1)*127.5).astype(np.uint8));label=f'{name} x{r["gamma"][name]:g}'
                draw.text((x,62),f'P {m["psnr"]:.3f} S {m["ssim"]:.4f} L {m["lpips"]:.4f}',fill='black',font=font(12))
            draw.text((x,34),label,fill='black',font=font(14));sheet.paste(im,(x,96))
        sheet.save(path.parent/'policy_comparison.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=ROOT);parser.add_argument('--a46-dir',default=core.ROOT)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--freeze',action='store_true');parser.add_argument('--phase',choices=('choose','finish'))
    parser.add_argument('--shard',type=int,choices=(0,1));parser.add_argument('--device',default='cuda:0');parser.add_argument('--archive',action='store_true');args=parser.parse_args()
    if args.prepare:prepare(args)
    elif args.freeze:
        root=Path(args.root).resolve();p=source.check_plan(root);core.a42.verify(p);freeze(root,p)
    elif args.shard is not None:
        if args.phase is None:parser.error('--phase required')
        worker(args)
    else:summarize(args)


if __name__=='__main__':main()
