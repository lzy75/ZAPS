"""A20 fixed-q, p=0 cached ablation; zero new UNet calls, +1 if deployed."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import time

PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS))
from utils import diag_terminal_failure_attribution as parent
from utils import diag_fixed_terminal_replication as comparison
from utils.diag_state_trajectory import digest

WRITE=parent.WRITE;METRICS=parent.METRICS
ROOT='/data0/lzy/ZAPS-results/diag_cached_prior_ablation/A20_20261020'
CARD=PROJECTS/'experiment_cards/A20_cached_prior_displacement_ablation_20261020.md'


def summarize(records):
    done=[r for r in records if r['status']=='complete'];groups=[]
    for cohort,expected,images in (('A16',6,3),('A17',6,3),('A18',12,6)):
        rows=[r for r in done if r['cohort']==cohort]
        comparisons={a+'_'+b:comparison.contrast(rows,a,b) for a,b in (('B','C'),('B','F1'),('F1','C'))}
        b=comparisons['B_C'];f=comparisons['F1_C'];limits=f['mean_numerical_limit']
        retained=b['mean_delta']['psnr']/f['mean_delta']['psnr'] if rows and f['mean_delta']['psnr']>limits['psnr'] else None
        recovered=1-b['mean_delta']['lpips']/f['mean_delta']['lpips'] if rows and f['mean_delta']['lpips']>limits['lpips'] else None
        acceptable=len(rows)==expected and b['passed'] and b['images_psnr_better']>=((2*images+2)//3)
        supported=acceptable and retained is not None and retained>=.9 and recovered is not None and recovered>=.5
        groups.append(dict(cohort=cohort,completed_cases=len(rows),planned_cases=expected,comparisons=comparisons,
            B_acceptable_vs_C=acceptable,retained_F1_psnr_gain_fraction=retained,recovered_F1_lpips_degradation_fraction=recovered,
            classification='INCOMPLETE_PILOT' if len(rows)!=expected else 'PRIOR_DISPLACEMENT_ABLATION_EXPLORATORY_SUPPORT' if supported else 'PRIOR_DISPLACEMENT_ABLATION_NOT_SUPPORTED',
            B_individual_lpips_over_tolerance=[r['case']['key'] for r in rows if r['branches']['B']['lpips']-r['branches']['C']['lpips']>.005],
            rows=[dict(case=r['case']['key'],B=r['branches']['B'],delta_B_C={k:r['branches']['B'][k]-r['branches']['C'][k] for k in METRICS},
                       delta_B_F1={k:r['branches']['B'][k]-r['branches']['F1'][k] for k in METRICS}) for r in rows]))
    return dict(completed_cases=len(done),planned_cases=24,image_clusters=len({r['case']['image_id'] for r in done}),method_nfe=0,diagnostic_nfe=0,
        scientific_extra_nfe_per_case_if_deployed=1,deployment_method_nfe=dict(C=300,F1=301,B=301),
        cumulative_queue_cost=dict(method_nfe=7200,diagnostic_nfe=2312,total=9512),groups=groups,primary_exploratory_cohort='A18',
        limits='All outcomes previously inspected; fixed q from an extra clean prediction. Cached offline p-removal ablation only, no state-selection, scheduling, prospective confirmation or free deployment claim.')


def prepare(args):
    source=Path(args.a19_dir).resolve();root=Path(args.root).resolve();p=parent.a16.read(source/'plan.json')
    if root.is_relative_to(source) or source.is_relative_to(root):raise ValueError('New root overlaps source')
    if digest(source/'plan.json')!=(source/'plan.sha256').read_text().strip():raise ValueError('A19 plan changed')
    old=parent.a16.read(source/'summary.json')
    records=[parent.load_record(source,s) for s in p['cases']]
    if any(r is None for r in records) or not old['report_complete'] or any(old.get(k)!=v for k,v in parent.summarize(records).items()):raise ValueError('Need completed audited A19')
    identities=dict(p['identity_files'])
    for path,sha in identities.items():
        if digest(path)!=sha:raise ValueError('Original source/code changed: '+path)
    paths=[source/n for n in ('plan.json','plan.sha256','summary.json')]
    for s in p['cases']:paths.extend(source/s['case']['key']/n for n in ('run.json','run.sha256'))
    own=[Path(__file__),CARD,PROJECTS/'run_a20_20261020.sh',PROJECTS/'utils/test_diag_cached_prior_ablation.py']
    for path in paths+own:identities[str(path)]=digest(path)
    plan=dict(experiment='A20_cached_prior_ablation',cases=p['cases'],source_A19=str(source),identity_files=identities,
        worker_identity_files={str(path):identities[str(path)] for path in own},arms=['C','F1','B'],q_coefficient=1.,B_prior_coefficient=0.,
        psnr_gain_retention_threshold=.9,lpips_degradation_recovery_threshold=.5,method_nfe=0,diagnostic_nfe=0,device='cpu')
    if root.exists():
        if parent.a16.read(root/'plan.json')!=plan:raise ValueError('Frozen A20 differs; refuse overwrite')
    else:
        root.mkdir(parents=True,exist_ok=False);WRITE(root/'plan.json',plan)
        (root/'plan.sha256').write_text(digest(root/'plan.json')+'\n');(root/'experiment_card.md').write_text(CARD.read_text())
    return root,plan


def load_record(root,spec):
    path=root/spec['case']['key']/'run.json'
    if not path.exists():
        if path.parent.exists():raise ValueError('Interrupted A20 retained')
        return None
    r=parent.a16.read(path)
    if (digest(path)!=(path.parent/'run.sha256').read_text().strip() or r['status']!='complete' or r['case']!=spec['case'] or r['cohort']!=spec['cohort'] or
        r['plan_sha256']!=digest(root/'plan.json') or r['method_nfe']!=0 or r['diagnostic_nfe']!=0 or
        not r['gates'] or not all(g['passed'] for g in r['gates'].values())):raise ValueError('Invalid A20 retained')
    return r


def outputs(spec):
    import torch
    original,errors=parent.read_outputs(spec)
    data=torch.load(Path(spec['source'])/'outputs.pt',map_location='cpu',weights_only=True)
    q=data['fields']['q'];b=original['C']+q
    if not torch.isfinite(b).all():raise ValueError('Nonfinite cached ablation')
    return dict(C=original['C'],F1=original['F1'],B=b),errors


def run_case(root,plan,spec,operator):
    import torch
    from modules.main_single import load_image_as_tensor
    from utils.metrics import compute_all_metrics
    folder=root/spec['case']['key'];folder.mkdir(exist_ok=False);start=time.perf_counter()
    r=dict(status='running',case=spec['case'],cohort=spec['cohort'],plan_sha256=digest(root/'plan.json'),method_nfe=0,diagnostic_nfe=0,gates={})
    def save():r['seconds']=time.perf_counter()-start;WRITE(folder/'run.json',r)
    save()
    try:
        images,errors=outputs(spec);gt=load_image_as_tensor(spec['case']['image']).cpu()
        y=torch.load(spec['measurement']['path'],map_location='cpu',weights_only=True)['y']
        def metrics(x):return dict(loss=float((y-operator.H(x)).double().square().mean()/(2*.05**2)),**compute_all_metrics(x,gt,lpips_net='alex'))
        branches={a:metrics(x) for a,x in images.items()};prior=parent.a16.read(Path(plan['source_A19'])/spec['case']['key']/'run.json')
        drift={a:{k:abs(branches[a][k]-prior['branches'][a][k]) for k in METRICS} for a in ('C','F1')}
        for a,values in drift.items():
            if any(values[k]>v for k,v in (('psnr',.01),('ssim',.001),('lpips',.001),('loss',1e-7+2e-5*abs(branches[a]['loss'])))):raise RuntimeError('Source metric drift')
        floors={k:max(values[k] for values in drift.values()) for k in METRICS}
        r.update(branches=branches,floors=floors,gates=dict(cached_fields=dict(passed=True,errors=errors),source_metrics=dict(passed=True,drift=drift),zero_UNet=dict(passed=True)))
        r['status']='complete';save();(folder/'run.sha256').write_text(digest(folder/'run.json')+'\n');return r
    except BaseException as error:r.update(status='failed',error=repr(error));save();raise


def figures(root,plan,records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from modules.main_single import load_image_as_tensor
    lookup={r['case']['key']:r for r in records};specs=sorted(plan['cases'],key=lambda s:s['case']['key'])
    for start in range(0,len(specs),6):
        fig,axes=plt.subplots(6,4,figsize=(12,18),squeeze=False)
        for row,spec in enumerate(specs[start:start+6]):
            images,_=outputs(spec);images['GT']=load_image_as_tensor(spec['case']['image']).cpu()
            for col,a in enumerate(('GT','C','F1','B')):
                ax=axes[row,col];ax.imshow((images[a][0].clamp(-1,1).permute(1,2,0).numpy()+1)*.5);ax.set_xticks([]);ax.set_yticks([])
                label=a
                if a!='GT':
                    m=lookup[spec['case']['key']]['branches'][a];label+=f" P={m['psnr']:.3f} L={m['lpips']:.4f}"
                ax.set_title(label,fontsize=9)
                if col==0:ax.set_ylabel(spec['case']['key'],fontsize=10)
        fig.tight_layout();fig.savefig(root/f'comparison_{start//6:02d}.png',dpi=110);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=ROOT);p.add_argument('--a19-dir',default=parent.ROOT);p.add_argument('--summary',action='store_true')
    args=p.parse_args();root=Path(args.root).resolve()
    if args.summary:print('A20 Summary:',json.dumps(parent.a16.read(root/'summary.json')),flush=True);return
    import fcntl,torch
    from unittest.mock import patch
    from modules.degradations import get_operator
    from utils.diag_ffhq_regression import TransposeMode
    from utils.diag_imagenet_gaussian_unit_domain import unit_operator
    root.parent.mkdir(parents=True,exist_ok=True)
    with (root.parent/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);root,plan=prepare(args);records=[];verified=False;torch.set_num_threads(2)
        operator=unit_operator(TransposeMode(get_operator('super_resolution',device='cpu',scale_factor=4,noise_sigma=0.),'legacy_bicubic'))
        try:
            with torch.random.fork_rng(devices=[]),torch.no_grad(),patch('modules.main_single.load_diffusion_model',side_effect=RuntimeError('UNet load forbidden in A20')):
                for spec in plan['cases']:
                    prior=load_record(root,spec);records.append(prior if prior is not None else run_case(root,plan,spec,operator))
                figures(root,plan,records)
            for path,sha in plan['identity_files'].items():
                if digest(path)!=sha:raise ValueError('Source/code changed')
            with (root/'metrics.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['case','cohort','arm',*METRICS]);w.writeheader()
                for r in records:
                    for a in ('C','F1','B'):w.writerow(dict(case=r['case']['key'],cohort=r['cohort'],arm=a,**r['branches'][a]))
            verified=True
        finally:
            saved=[parent.a16.read(path) for path in sorted(root.glob('*/run.json'))];out=summarize(saved);out['report_complete']=verified
            WRITE(root/'summary.json',out)
        from utils.archive_refinement_confirmation import export_and_commit
        bundle,commit=export_and_commit({'A20':str(root)});print('A20 Summary:',json.dumps(out),'\nRecords:',root,'\nArchive:',bundle,'\nResults commit:',commit,flush=True)


if __name__=='__main__':main()
