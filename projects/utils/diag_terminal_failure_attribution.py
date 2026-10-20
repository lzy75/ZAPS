"""A19 cached terminal failure attribution; no diffusion model load or UNet calls."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import time
import numpy as np

PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS))
from utils import diag_terminal_state_strength as a16
from utils import diag_fixed_terminal_replication as a17
from utils import diag_frozen_f2_confirmation as a18
from utils.diag_state_trajectory import digest

WRITE=a16.WRITE;METRICS=a16.METRICS
ROOT='/data0/lzy/ZAPS-results/diag_terminal_failure_attribution/A19_20261020'
CARD=PROJECTS/'experiment_cards/A19_terminal_failure_attribution_20261020.md'
ARMS=('C','F0','F1','F2')
CONTRASTS=(('F0','C'),('F1','F0'),('F2','F1'),('F1','C'),('F2','C'))


def decomposition(h_c,h_x,h_gt,y,sigma=.05):
    c,x,b,y=[np.asarray(v,dtype=np.float64) for v in (h_c,h_x,h_gt,y)]
    if len({v.shape for v in (c,x,b,y)})!=1 or not all(np.isfinite(v).all() for v in (c,x,b,y)) or sigma<=0:
        raise ValueError('Invalid decomposition arrays')
    noise=y-b;action=x-c
    truth_delta=float(((x-b)**2-(c-b)**2).mean()/(2*sigma**2))
    credit=float((noise*action).mean()/sigma**2)
    observed=float(((y-x)**2-(y-c)**2).mean()/(2*sigma**2))
    error=abs(observed-(truth_delta-credit))
    if error>1e-9+1e-7*abs(observed):raise RuntimeError('Measurement decomposition identity failed')
    return dict(observed_loss_delta=observed,true_measurement_error_delta=truth_delta,noise_alignment_credit=credit,
        identity_absolute_error=error,noise_rms_over_sigma=float(np.sqrt((noise**2).mean())/sigma))


def summarize(records):
    complete=[r for r in records if r['status']=='complete'];groups=[]
    for cohort,expected in (('A16',6),('A17',6),('A18',12)):
        rows=[r for r in complete if r['cohort']==cohort]
        contrasts={a+'_'+b:{k:statistics.mean(r['branches'][a][k]-r['branches'][b][k] for r in rows) if rows else None for k in METRICS} for a,b in CONTRASTS}
        part={a:{k:statistics.mean(r['decomposition'][a][k] for r in rows) if rows else None for k in ('observed_loss_delta','true_measurement_error_delta','noise_alignment_credit')}
              for a in ('F0','F1','F2')}
        groups.append(dict(cohort=cohort,completed_cases=len(rows),planned_cases=expected,contrasts=contrasts,measurement_decomposition=part,
            mean_out_of_range_fraction={a:statistics.mean(r['geometry'][a]['out_of_range_fraction'] for r in rows) if rows else None for a in ARMS},
            F2_lpips_over_tolerance=[r['case']['key'] for r in rows if r['branches']['F2']['lpips']-r['branches']['C']['lpips']>.005]))
    return dict(completed_cases=len(complete),planned_cases=24,image_clusters=len({r['case']['image_id'] for r in complete}),method_nfe=0,diagnostic_nfe=0,
        diffusion_model_loaded=False,classification='OFFLINE_TERMINAL_ATTRIBUTION_COMPLETE' if len(complete)==24 else 'INCOMPLETE_OFFLINE_ATTRIBUTION',
        cumulative_queue_cost=dict(method_nfe=7200,diagnostic_nfe=2312,total=9512),groups=groups,
        figures=[f'comparison_{i:02d}.png' for i in range(4)] if len(complete)==24 else [],
        limits='Previously observed outputs, GT only in offline decomposition/evaluation. Mean differences and exact measurement algebra do not establish causes of LPIPS changes, a state policy, curvature or generalization. Original A18 failed classification retained.')


def prepare(args):
    root=Path(args.root).resolve();sources={};cases=[];identities={};reference=None
    def bind(path,sha=None):
        path=Path(path);value=digest(path)
        if sha is not None and value!=sha:raise ValueError('Source changed: '+str(path))
        identities[str(path)]=value
    for label,module,folder in (('A16',a16,args.a16_dir),('A17',a17,args.a17_dir),('A18',a18,args.a18_dir)):
        source=Path(folder).resolve();p=a16.read(source/'plan.json')
        if root.is_relative_to(source) or source.is_relative_to(root):raise ValueError('New root overlaps source')
        bind(source/'plan.json',(source/'plan.sha256').read_text().strip());bind(source/'plan.sha256');bind(source/'summary.json')
        active=[module.load_record(source,c) for c in p['cases']]
        if any(r is None for r in active):raise ValueError('Need all completed source cases: '+label)
        expected=module.summarize(active,p['prior_records']) if label=='A17' else module.summarize(active)
        if expected!=a16.read(source/'summary.json'):raise ValueError('Source summary differs: '+label)
        if label=='A18' and expected['classification']!='FROZEN_F2_NOT_SUPPORTED_ON_FRESH_COHORT':raise ValueError('Need declared A18 failure context')
        protocol={k:p[k] for k in ('config','grid','checkpoint_sha256')}
        if reference is None:reference=protocol
        elif protocol!=reference:raise ValueError('Source protocols differ')
        sources[label]=str(source)
        for c,r in zip(p['cases'],active):
            bind(source/c['key']/'run.json');bind(source/c['key']/'run.sha256')
            sha=r['artifacts_sha256']['outputs.pt'] if label=='A18' else r['outputs_sha256']
            bind(source/c['key']/'outputs.pt',sha)
            m=p['sources'][c['key']]['measurement'] if label in ('A16','A17') else r['measurement']
            bind(m['path'],m['sha256']);bind(c['image'],c['image_sha256'])
            cases.append(dict(case=c,cohort=label,source=str(source/c['key']),measurement=m,reference_branches={a:r['branches'][a] for a in ('C','F1','F2')}))
    if len(cases)!=24 or len({c['case']['key'] for c in cases})!=24 or len({c['case']['image_id'] for c in cases})!=12:
        raise ValueError('Fixed source cohort differs')
    own=[Path(__file__),CARD,PROJECTS/'run_a19_20261020.sh',PROJECTS/'utils/test_diag_terminal_failure_attribution.py',
         PROJECTS/'utils/metrics.py',PROJECTS/'modules/degradations.py',PROJECTS/'utils/diag_imagenet_gaussian_unit_domain.py',
         PROJECTS/'utils/diag_ffhq_regression.py',PROJECTS/'modules/main_single.py',PROJECTS.parent/'DPS/util/resizer.py',
         PROJECTS/'utils/diag_terminal_state_strength.py',PROJECTS/'utils/diag_fixed_terminal_replication.py',PROJECTS/'utils/diag_frozen_f2_confirmation.py']
    for path in own:bind(path)
    plan=dict(experiment='A19_terminal_failure_attribution',cases=cases,sources=sources,protocol=reference,identity_files=identities,
        worker_identity_files={str(p):identities[str(p)] for p in own},method_nfe=0,diagnostic_nfe=0,device='cpu',arms=list(ARMS),sigma=.05)
    if root.exists():
        if a16.read(root/'plan.json')!=plan:raise ValueError('Frozen A19 differs; refuse overwrite')
    else:
        root.mkdir(parents=True,exist_ok=False);WRITE(root/'plan.json',plan)
        (root/'plan.sha256').write_text(digest(root/'plan.json')+'\n');(root/'experiment_card.md').write_text(CARD.read_text())
    return root,plan


def load_record(root,spec):
    path=root/spec['case']['key']/'run.json'
    if not path.exists():
        if path.parent.exists():raise ValueError('Interrupted A19 retained')
        return None
    r=a16.read(path)
    if (r['status']!='complete' or digest(path)!=(path.parent/'run.sha256').read_text().strip() or r['case']!=spec['case'] or r['cohort']!=spec['cohort'] or
        r['plan_sha256']!=digest(root/'plan.json') or r['method_nfe']!=0 or r['diagnostic_nfe']!=0 or
        not r['gates'] or not all(g['passed'] for g in r['gates'].values())):raise ValueError('Invalid/interrupted A19 retained')
    return r


def read_outputs(spec):
    import torch
    from utils.diag_zaps_trace_audit import relative_error
    data=torch.load(Path(spec['source'])/'outputs.pt',map_location='cpu',weights_only=True)
    clean,q=data['fields']['clean'],data['fields']['q'];outputs={a:data['outputs'][a] for a in ('C','F1','F2')};outputs['F0']=clean
    if any(t.shape!=(1,3,256,256) or t.dtype!=torch.float32 or not torch.isfinite(t).all() for t in (*outputs.values(),q)):raise ValueError('Invalid cached output/field')
    errors=dict(F1=relative_error(outputs['F1'],clean+q),F2=relative_error(outputs['F2'],clean+2*q))
    if any(not math.isfinite(v) or v>1e-7 for v in errors.values()):raise RuntimeError('Cached field/output identities differ')
    return outputs,errors


def run_case(root,plan,spec,operator):
    import torch
    from modules.main_single import load_image_as_tensor
    from utils.metrics import compute_all_metrics
    folder=root/spec['case']['key'];folder.mkdir(exist_ok=False);start=time.perf_counter()
    r=dict(status='running',case=spec['case'],cohort=spec['cohort'],plan_sha256=digest(root/'plan.json'),method_nfe=0,diagnostic_nfe=0,gates={})
    def save():r['seconds']=time.perf_counter()-start;WRITE(folder/'run.json',r)
    save()
    try:
        outputs,errors=read_outputs(spec);gt=load_image_as_tensor(spec['case']['image']).cpu()
        y=torch.load(spec['measurement']['path'],map_location='cpu',weights_only=True)['y']
        if y.shape!=(1,3,64,64) or y.dtype!=torch.float32 or not torch.isfinite(y).all():raise ValueError('Invalid saved measurement')
        projected={a:operator.H(x) for a,x in outputs.items()};truth=operator.H(gt)
        branches={a:dict(loss=float((y-h).double().square().mean()/(2*.05**2)),**compute_all_metrics(outputs[a],gt,lpips_net='alex')) for a,h in projected.items()}
        drift={a:{k:abs(branches[a][k]-spec['reference_branches'][a][k]) for k in METRICS} for a in ('C','F1','F2')}
        for a,d in drift.items():
            if any(d[k]>v for k,v in (('psnr',.01),('ssim',.001),('lpips',.001),('loss',1e-7+2e-5*abs(branches[a]['loss'])))):raise RuntimeError('CPU/source metric drift')
        parts={a:decomposition(projected['C'].numpy(),projected[a].numpy(),truth.numpy(),y.numpy()) for a in ('F0','F1','F2')}
        for a,part in parts.items():
            actual=branches[a]['loss']-branches['C']['loss']
            if abs(part['observed_loss_delta']-actual)>1e-7+2e-5*abs(actual):raise RuntimeError('Decomposition/source loss reduction differs')
        geometry={a:dict(out_of_range_fraction=float((x.abs()>1).float().mean()),displacement_rms=float((x.double()-outputs['C'].double()).square().mean().sqrt()),
            residual_noise_ratio=float((y-projected[a]).double().square().mean().sqrt()/.05)) for a,x in outputs.items()}
        r.update(branches=branches,decomposition=parts,geometry=geometry,gates=dict(cached_fields=dict(passed=True,errors=errors),
            source_metrics=dict(passed=True,drift=drift),measurement_algebra=dict(passed=True),zero_UNet=dict(passed=True)))
        r['status']='complete';save();(folder/'run.sha256').write_text(digest(folder/'run.json')+'\n')
        return r
    except BaseException as error:r.update(status='failed',error=repr(error));save();raise


def figures(root,plan,records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from modules.main_single import load_image_as_tensor
    lookup={r['case']['key']:r for r in records}
    specs=sorted(plan['cases'],key=lambda s:s['case']['key'])
    for start in range(0,len(specs),6):
        fig,axes=plt.subplots(6,5,figsize=(15,18),squeeze=False)
        for row,spec in enumerate(specs[start:start+6]):
            outputs,_=read_outputs(spec);outputs['GT']=load_image_as_tensor(spec['case']['image']).cpu()
            for col,a in enumerate(('GT','C','F0','F1','F2')):
                ax=axes[row,col];x=outputs[a][0].clamp(-1,1).permute(1,2,0).numpy()
                ax.imshow((x+1)*.5);ax.set_xticks([]);ax.set_yticks([])
                label=a
                if a!='GT':
                    m=lookup[spec['case']['key']]['branches'][a];label+=f"  P={m['psnr']:.3f} L={m['lpips']:.4f}"
                ax.set_title(label,fontsize=9)
                if col==0:ax.set_ylabel(spec['case']['key'],fontsize=10)
        fig.tight_layout();fig.savefig(root/f'comparison_{start//6:02d}.png',dpi=110);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=ROOT);p.add_argument('--summary',action='store_true')
    p.add_argument('--a16-dir',default=a16.ROOT);p.add_argument('--a17-dir',default=a17.ROOT);p.add_argument('--a18-dir',default=a18.ROOT);args=p.parse_args();root=Path(args.root).resolve()
    if args.summary:
        print('A19 Summary:',json.dumps(a16.read(root/'summary.json')),flush=True);return
    import fcntl,torch
    from unittest.mock import patch
    from modules.degradations import get_operator
    from utils.diag_ffhq_regression import TransposeMode
    from utils.diag_imagenet_gaussian_unit_domain import unit_operator
    root.parent.mkdir(parents=True,exist_ok=True)
    with (root.parent/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        root,plan=prepare(args);records=[];torch.set_num_threads(2)
        operator=unit_operator(TransposeMode(get_operator('super_resolution',device='cpu',scale_factor=4,noise_sigma=0.),'legacy_bicubic'))
        outputs_verified=False
        try:
            with torch.random.fork_rng(devices=[]),torch.no_grad(),patch('modules.main_single.load_diffusion_model',side_effect=RuntimeError('UNet load forbidden in A19')):
                for spec in plan['cases']:
                    r=load_record(root,spec)
                    records.append(r if r is not None else run_case(root,plan,spec,operator))
                figures(root,plan,records)
            for path,sha in plan['identity_files'].items():
                if digest(path)!=sha:raise ValueError('Bound source/code changed')
            with (root/'metrics.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['case','cohort','arm',*METRICS]);writer.writeheader()
                for r in records:
                    for a in ARMS:writer.writerow(dict(case=r['case']['key'],cohort=r['cohort'],arm=a,**r['branches'][a]))
            outputs_verified=True
        finally:
            saved=[a16.read(path) for path in sorted(root.glob('*/run.json'))]
            out=summarize(saved);out['report_complete']=outputs_verified
            if not outputs_verified:out['classification']='INCOMPLETE_OFFLINE_ATTRIBUTION'
            WRITE(root/'summary.json',out)
        from utils.archive_refinement_confirmation import export_and_commit
        bundle,commit=export_and_commit({'A19':str(root)})
        print('A19 Summary:',json.dumps(out),'\nRecords:',root,'\nArchive:',bundle,'\nResults commit:',commit,flush=True)


if __name__=='__main__':main()
