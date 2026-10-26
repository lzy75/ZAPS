"""A54: isolate current norm-gain information; reuse every A53 NORM_ONLY trajectory."""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import shutil
import sys
from statistics import mean

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import diag_angular_feedback_value as engine
from utils import norm_gain_control as rules

parent,source=engine.parent,engine.source
write,signed=engine.write,engine.signed
ROOT='/data0/lzy/ZAPS-results/diag_norm_gain_value/A54_20261026'
CARD=PROJECTS/'experiment_cards/A54_current_norm_gain_20261026.md'
BRANCH='results/ffhq-current-norm-gain-20261026'


def require(test,message):
    if not test:raise ValueError(message)


def check(root):
    p=source.check_plan(root);parent.a42.verify(p)
    require(p['experiment']=='A54_current_norm_gain','Wrong plan')
    a=source.load_signed(root/'allocation.json')
    require(a['plan_sha256']==source.digest(root/'plan.json') and a['GT_used'] is False,'Allocation binding differs')
    require(all(source.digest(Path(f))==sha for f,sha in a['sources'].items()),'NORM_ONLY calibration changed')
    return p


def prepare(args):
    root,prior=Path(args.root).resolve(),Path(args.a53_dir).resolve()
    require(not(root.is_relative_to(prior) or prior.is_relative_to(root)),'Source/output overlap')
    if root.exists():
        p=check(root);require(p['source_A53']==str(prior),'Frozen source differs')
        print('A54 frozen controls reused; no GPU work restarted.',flush=True);return
    before=engine.check(prior)
    require(source.digest(prior/'summary.json')=='e5038b8f1ec746fd457c367c79289303349c344278035657183d7830cb7a9d35',
            'Need complete audited A53 summary')
    identities=dict(before['identity_files']);refs={};curves={};sources={};fixed={}
    cached=source.load_signed(prior/'evaluation.json');oldrows={r['case']['key']:r for r in cached['rows']}
    for n in ('plan.json','summary.json','evaluation.json'):identities[str(prior/n)]=source.digest(prior/n)
    for c in before['cases']:
        found=engine.item(prior,before,c,'control');require(found is not None,'Missing A53 control case')
        f,r,_=found;refs[c['key']]=str(f);identities[str(f)]=source.digest(f);sources[str(f)]=source.digest(f)
        for n,sha in r['artifacts'].items():identities[str(f.parent/n)]=sha
        values=[]
        for row in r['traces']['NORM_ONLY']:
            z=row['readout'];require(z.get('resolved') and z['used_cosine']==1.,'Need fully resolved NORM_ONLY source')
            gain=z['v_RMS']/z['u_RMS']
            require(math.isclose(gain,z['used_alignment_scale'],rel_tol=2e-6,abs_tol=0.),'Source gain mismatch')
            values.append(gain)
        curves[c['key']]=values
        fixed_path=Path(before['source_finish_records'][c['key']]);fr=source.load_signed(fixed_path)
        require(fr['case']==c and fr['status']=='complete' and all(g['passed'] for g in fr['gates'].values()),'Invalid A44 fixed reference')
        for m in ('psnr','ssim','lpips'):
            require(abs(fr['evaluation']['C']['metrics'][m]-oldrows[c['key']]['metrics']['C_NATIVE'][m])<1e-5,
                    'A44/native baseline metric identity differs')
        identities[str(fixed_path)]=source.digest(fixed_path)
        fixed[c['key']]=dict(gamma=fr['gamma']['TASK_FIXED'],source=str(fixed_path),
                            metrics={m:fr['evaluation']['TASK_FIXED']['metrics'][m] for m in ('psnr','ssim','lpips')})
    profiles=rules.allocate(before['cases'],curves)
    own=[Path(__file__),Path(rules.__file__),CARD,PROJECTS/'run_a54_20261026.sh',
         PROJECTS/'utils/test_norm_gain_control.py',PROJECTS/'utils/experiment_gpu_selection.py']
    p=dict(before)
    p.update(experiment='A54_current_norm_gain',source_A53=str(prior),norm_sources=refs,old_task_fixed=fixed,
        reference_evaluation=str(prior/'evaluation.json'),identity_files=identities,
        worker_identity_files={**before['worker_identity_files'],**{str(f):source.digest(f) for f in own}},
        phases=['control'],arms=['STATE_GAIN',*rules.CONTROLS],generate_arms=[],control_arms=list(rules.CONTROLS),
        planned_new_generate_cases=0,planned_method_nfe=0,planned_diagnostic_nfe=900,planned_backward_attempts_max=900,
        reused_A53_diagnostic_nfe=1080,
        executor='A53 worker/sampler reused via process-local check/item/rules bindings; historical source files untouched.',
        limits='Five REUSED development image clusters x SR/IP, seed1000, sigma=.05. STATE_GAIN aliases A53 NORM_ONLY, '
               'not A53 cosine-attenuated STATE. c=1 in every action; every arm retains CURRENT u direction. '
               'Only g changes. Cached quality is not new independent confirmation. Original first270/last30 grid unchanged. '
               'No trajectory-curvature, four-task, held-out transfer or NFE-saving claim.')
    root.parent.mkdir(parents=True,exist_ok=True);parent.P.coupled.archive.recovery.require_space(root.parent)
    root.mkdir();signed(root/'plan.json',p);shutil.copyfile(CARD,root/'experiment_card.md')
    signed(root/'allocation.json',dict(plan_sha256=source.digest(root/'plan.json'),sources=sources,
                                    profiles=profiles,GT_used=False))
    snap=root/'source_snapshot';snap.mkdir()
    for n in ('plan.json','summary.json','evaluation.json'):shutil.copyfile(prior/n,snap/('A53_'+n))
    for key,f in refs.items():shutil.copyfile(f,snap/(key+'_A53_control.json'))
    print('A54 PREPARE/FREEZE PASS:10 NORM_ONLY trajectories cached; LOO gains frozen without labels;900 new forwards,<=900 input backwards.',flush=True)


def item(root,p,c,phase):
    require(phase=='control','A54 has no paid generation phase')
    paths=sorted((root/c['key']).glob('control_*/run.json'))
    if not paths:return None
    require(len(paths)==1,'Multiple attempts; no selection')
    f=paths[0];r=source.read(f)
    if r['status']!='complete':return None
    r=source.load_signed(f);allocation=source.load_signed(root/'allocation.json')
    require(r['case']==c and r['plan_sha256']==source.digest(root/'plan.json') and
        r['calls']==list(reversed(p['grid']))*3 and r['nfe']==90,'Paid call contract differs')
    require(r['backward_attempts']==r['backward_completed']<=90 and r['gates'] and all(v['passed'] for v in r['gates'].values()),'Failed recorded gate')
    require(r['allocation_sha256']==source.digest(root/'allocation.json') and set(r['traces'])==set(rules.CONTROLS),'Controls differ')
    for name,trace in r['traces'].items():
        require(len(trace)==30,'Missing step')
        for row in trace:
            z=row['readout']
            if z.get('resolved'):
                require(z['used_gain']==allocation['profiles']['assignments'][c['key']][name][row['step']] and
                        z['cosine_used_for_action'] is False and z['gain_norm_closure_error']<=2e-6 and
                        z['current_norm_reference_error']<=2e-6,'Gain target gate failed')
    require(all(source.digest(f.parent/n)==sha for n,sha in r['artifacts'].items()),'Output changed')
    return f,r,'STATE_GAIN'


@contextmanager
def executor_bindings():
    """Explicit dependency injection in THIS worker process; no source mutations."""
    from unittest.mock import patch
    with patch.object(engine,'rules',rules),patch.object(engine,'check',check),patch.object(engine,'item',item):
        yield


def evaluate(root,p):
    import torch
    from modules.main_single import load_image_as_tensor
    from utils.metrics import compute_all_metrics
    pairs=[(c,item(root,p,c,'control')) for c in p['cases']]
    require(all(v for c,v in pairs),'All10 new control outputs required before labels')
    hashes={str(v[0]):source.digest(v[0]) for c,v in pairs}
    path=root/'evaluation.json'
    if path.exists():
        prior=source.load_signed(path);require(prior['inputs']==hashes,'Evaluation changed');return prior
    cached=source.load_signed(Path(p['reference_evaluation']))
    oldrows={r['case']['key']:r for r in cached['rows']};rows=[]
    with parent.scan.preserved_rng(),torch.no_grad():
        for c,(f,r,_) in pairs:
            prior=oldrows[c['key']];require(prior['case']==c,'Cached metric case mismatch')
            gt=load_image_as_tensor(c['image']).cpu()
            outputs=torch.load(f.parent/'outputs.pt',map_location='cpu',weights_only=True)
            metrics={n:compute_all_metrics(x,gt,lpips_net='alex') for n,x in outputs.items()}
            metrics.update(STATE_GAIN=dict(prior['metrics']['NORM_ONLY']),
                           C_NATIVE=dict(prior['metrics']['C_NATIVE']),C_IEEE=dict(prior['metrics']['C_IEEE']),
                           A44_TASK_FIXED=dict(p['old_task_fixed'][c['key']]['metrics']))
            require(all(math.isfinite(float(x)) for m in metrics.values() for x in m.values()),'Nonfinite quality')
            regions={}
            if c['task']=='inpainting':
                sf=Path(p['norm_sources'][c['key']])
                bank=torch.load(sf.parent/'outputs.pt',map_location='cpu',weights_only=True)
                outputs['STATE_GAIN']=bank['NORM_ONLY']
                baseline=torch.load(Path(p['a50_baselines'][c['key']]).parent/'outputs.pt',map_location='cpu',weights_only=True)
                outputs.update({n:baseline[n] for n in ('C_NATIVE','C_IEEE')})
                mask=torch.load(sf.parent/'mask.pt',map_location='cpu',weights_only=True).expand_as(gt)
                for name,x in outputs.items():
                    error=((x.clamp(-1,1)-gt)/2).double().square()
                    regions[name]=dict(observed_unit_MSE=float(error[mask].mean()),unobserved_unit_MSE=float(error[~mask].mean()))
            rows.append(dict(case=c,metrics=metrics,inpainting_regions=regions,resolved={'STATE_GAIN':30,
                **{n:sum(v['readout'].get('resolved',False) for v in r['traces'][n]) for n in rules.CONTROLS}}))
    value=dict(inputs=hashes,allocation_sha256=source.digest(root/'allocation.json'),rows=rows,
        cached_metric_source=p['reference_evaluation'],GT_used_only_after_all_new_outputs_frozen=True,limits=p['limits'])
    signed(path,value);return value


def summary(root,p):
    value=evaluate(root,p);attempts=[source.read(f) for f in root.glob('*/control_*/run.json')];groups=[]
    def guard(d):return d['ssim']>=-.002 and d['lpips']<=.005
    for task in rules.TASKS:
        rows=[r for r in value['rows'] if r['case']['task']==task];refs=('C_NATIVE','C_IEEE','A44_TASK_FIXED',*rules.CONTROLS)
        effects={ref:{m:mean(r['metrics']['STATE_GAIN'][m]-r['metrics'][ref][m] for r in rows)
                      for m in ('psnr','ssim','lpips')} for ref in refs}
        positive={ref:sum(r['metrics']['STATE_GAIN']['psnr']>r['metrics'][ref]['psnr']+1e-9 for r in rows) for ref in refs}
        covered=all(r['resolved'][n]>=24 for r in rows for n in ('STATE_GAIN',*rules.CONTROLS))
        full=all(r['resolved'][n]==30 for r in rows for n in ('STATE_GAIN',*rules.CONTROLS))
        quality=covered and all(effects[b]['psnr']>=.05 and guard(effects[b]) and positive[b]>=4 for b in ('C_NATIVE','C_IEEE'))
        incremental=quality and full and all(effects[b]['psnr']>=.02 and guard(effects[b]) and positive[b]>=4 for b in ('TIME_GAIN','SHUFFLE_GAIN'))
        groups.append(dict(task=task,STATE_GAIN_minus=effects,positive_images=positive,quality_candidate=bool(quality),
            fully_resolved_for_gain_test=bool(full),current_gain_increment_candidate=bool(incremental),
            per_image=[dict(image_id=r['case']['image_id'],metrics=r['metrics'],inpainting_regions=r['inpainting_regions'],
                           STATE_GAIN_psnr_minus={b:r['metrics']['STATE_GAIN']['psnr']-r['metrics'][b]['psnr'] for b in refs}) for r in rows]))
    result=dict(experiment=p['experiment'],classification='DEVELOPMENT_CURRENT_GAIN_INCREMENT_CANDIDATE' if any(g['current_gain_increment_candidate'] for g in groups)
        else 'NO_CONFIRMED_CURRENT_GAIN_INCREMENT',completed_cases=10,planned_cases=10,image_clusters=5,
        reused_STATE_GAIN_cases=10,method_nfe=0,diagnostic_nfe=sum(r['nfe'] for r in attempts),planned_diagnostic_nfe=900,
        backward_attempts=sum(r['backward_attempts'] for r in attempts),backward_completed=sum(r['backward_completed'] for r in attempts),
        partial_attempts=sum(r['status']!='complete' for r in attempts),scientific_forward_nfe=300,
        new_arm_seconds={n:sum(r['arm_seconds'][n] for r in attempts) for n in rules.CONTROLS},
        operator_accounting={key:sum(v['readout'].get('cg',{}).get(key,0) for r in attempts for trace in r['traces'].values() for v in trace)
                             for key in ('linear_calls','adjoint_calls')},groups=groups,
        limits=p['limits']+' STATE_GAIN quality is reused; comparison with new controls is development-only. '
               'CONST is not strongest tuned fixed; foreign gains are not clipped to CURRENT norm. No p-values or speed claim.')
    signed(root/'summary.json',result);print('A54 Summary:',json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
    print('Records:',root,flush=True);return value


def render(root,p,value):
    import torch
    import numpy as np
    from PIL import Image,ImageDraw
    from utils.diag_cached_guidance_final_value import font
    for row in value['rows']:
        c=row['case'];f,r,_=item(root,p,c,'control');outputs=torch.load(f.parent/'outputs.pt',map_location='cpu',weights_only=True)
        sf=Path(p['norm_sources'][c['key']]);bank=torch.load(sf.parent/'outputs.pt',map_location='cpu',weights_only=True)
        outputs['STATE_GAIN']=bank['NORM_ONLY']
        baseline=torch.load(Path(p['a50_baselines'][c['key']]).parent/'outputs.pt',map_location='cpu',weights_only=True)
        outputs['C_IEEE']=baseline['C_IEEE'];sheet=Image.new('RGB',(1120,760),'white');draw=ImageDraw.Draw(sheet)
        draw.text((12,8),c['key']+' | CURRENT u; c=1; gain-only controls | reused development cohort',fill='black',font=font(14))
        for line,names in enumerate((('GT','C_IEEE','STATE_GAIN','TIME_GAIN'),('GT','C_IEEE','CONST_GAIN','SHUFFLE_GAIN'))):
            top=36+358*line
            for j,name in enumerate(names):
                left=12+276*j;draw.text((left,top),name,fill='black',font=font(14))
                if name=='GT':im=Image.open(c['image']).convert('RGB').resize((256,256),Image.Resampling.LANCZOS)
                else:
                    arr=outputs[name][0].numpy().transpose(1,2,0);im=Image.fromarray(((np.clip(arr,-1,1)+1)*127.5).astype(np.uint8))
                    m=row['metrics'][name];draw.text((left,top+25),f'P {m["psnr"]:.3f} S {m["ssim"]:.4f} L {m["lpips"]:.4f}',fill='black',font=font(12))
                sheet.paste(im,(left,top+55))
        (root/c['key']).mkdir(exist_ok=True);sheet.save(root/c['key']/'gain_comparison.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=ROOT)
    parser.add_argument('--a53-dir',default=engine.ROOT);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--phase',choices=('control',),default='control');parser.add_argument('--shard',type=int,choices=(0,1))
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--archive',action='store_true');args=parser.parse_args()
    if args.prepare:prepare(args);return
    if args.shard is not None:
        with executor_bindings():engine.worker(args)
        return
    root=Path(args.root).resolve();p=check(root);value=summary(root,p)
    if args.archive:
        from unittest.mock import patch
        render(root,p,value)
        with patch.object(engine.old.previous,'BRANCH',BRANCH):engine.old.previous.archive(root)


if __name__=='__main__':main()
