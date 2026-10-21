"""Frozen A26/A27 scalar controls, development-only selection and image-cluster inference."""
import math
import statistics
import numpy as np
from utils import fixed_count_state_positions as prior

RANGES=dict(rho=[0.,.25,.5],a=[0.,.25,.5,.75,1.],b=[0.,1.])
METRICS=('loss','psnr','ssim','lpips')
FAMILY_SIZE=11


def candidates():
    result={'C':dict(kind='C',rho=0.,a=1.,b=0.),'F':dict(kind='F',rho=0.,a=1.,b=0.)}
    for rho in RANGES['rho']:
        for a in RANGES['a']:
            for b in RANGES['b']:
                if rho==b==0:continue
                result[f'R{rho:g}_A{a:g}_B{b:g}']=dict(kind='STATE',rho=rho,a=a,b=b)
    return result


def controls(params,signals,t):
    if params['kind'] in ('C','F'):return 0.,(2. if params['kind']=='F' and t==319 else 1.)
    er,ec=signals['e_R'],signals['e_C'];a=params['a']
    if ec is None:E=G=er if a>0 else 0.
    elif params['kind']=='MATCHED':E=G=a*er
    else:E=a*er+(1-a)*ec;G=a*er-(1-a)*max(ec,0.)
    return E,(max(.5,min(2.,2**(params['b']*G))) if t<=319 else 1.)


def choose(positions,j,E,rho):
    if not 1<=j<=29 or not 0<=rho<=.5 or not math.isfinite(E) or not -1<=E<=1:raise ValueError('Outside scan control domain')
    if j==29:return 0
    if E==0 or rho==0:return positions.nominal[j]
    target=positions.base[j]-rho*E*(positions.base[j]-positions.base[j-1]);options=positions.cells[j]
    return int(options[np.argmin(abs(positions.u[options]-target))])


def effect(records,arm):
    images=sorted({r['case']['image_id'] for r in records})
    if not images:raise ValueError('No complete development images')
    per={i:{m:statistics.mean(r['branches'][arm][m]-r['branches']['C'][m] for r in records if r['case']['image_id']==i) for m in METRICS} for i in images}
    d={m:statistics.mean(v[m] for v in per.values()) for m in METRICS}
    floor=statistics.mean(r['loss_floor'] for r in records)
    return dict(mean_delta=d,per_image=per,guard=d['loss']<=floor and d['ssim']>=-.002 and d['lpips']<=.005)


def select(records,bank,positions):
    effects={a:effect(records,a) for a in bank};roles={};fallback={}
    for role in ('S','P','D'):
        pool=[n for n,p in bank.items() if p['kind']=='STATE' and ((0<p['a']<1) if role=='S' else p['a']==(1. if role=='P' else 0.))]
        eligible=[n for n in pool if effects[n]['guard']];fallback[role]=not bool(eligible)
        roles[role]=min(eligible or pool,key=lambda n:(-effects[n]['mean_delta']['psnr'],bank[n]['rho'],bank[n]['b'],bank[n]['a'],n))
    times=[999];selected=roles['S']
    for j in range(1,29):
        target=statistics.median(float(positions.u[r['traces'][selected][j]['t']]) for r in records)
        options=positions.cells[j];times.append(int(options[np.argmin(abs(positions.u[options]-target))]))
    times.append(0)
    images=sorted({r['case']['image_id'] for r in records})
    gamma=[math.exp(statistics.mean(statistics.mean(math.log(r['traces'][selected][k]['gamma']) for r in records if r['case']['image_id']==i) for i in images)) for k in range(30)]
    return dict(roles=roles,parameters={r:bank[n] for r,n in roles.items()},development_fallback=fallback,development_effects=effects,stage=dict(times=times,gamma=gamma),GT_used_for_development_hyperparameters=True,validation_used=False)


def intervals(values,indices):
    a=np.asarray(values,dtype=np.float64)
    if len(a)<2 or not np.isfinite(a).all():raise ValueError('Need finite independent image clusters')
    mean=float(a.mean());boot=a[indices].mean(axis=1)
    tail=.05/FAMILY_SIZE
    low,high=2*mean-np.quantile(boot,[.975,.025]);lo_adj,hi_adj=2*mean-np.quantile(boot,[1-tail,tail])
    return dict(mean=mean,ci95=[float(low),float(high)],family_lower=float(lo_adj),family_upper=float(hi_adj),images_positive=int(np.count_nonzero(a>0)),per_image=a.tolist())


def confirmation(records):
    images=sorted({r['case']['image_id'] for r in records})
    if any(len([r for r in records if r['case']['image_id']==i])!=2 or {r['case']['sampling_seed'] for r in records if r['case']['image_id']==i}!={1000,2000} for i in images):raise ValueError('Need both fixed seeds per image')
    indices=np.random.default_rng(261021).integers(0,len(images),size=(100000,len(images)))
    comparisons={}
    for other in ('C','F','P','M','STAGE'):
        comparisons[other]={m:intervals([statistics.mean(r['branches']['S'][m]-r['branches'][other][m] for r in records if r['case']['image_id']==i) for i in images],indices) for m in METRICS}
    tests={f'psnr_vs_{a}':comparisons[a]['psnr']['family_lower']>1e-9 for a in ('C','F','P','M','STAGE')}
    floor=statistics.mean(r['loss_floor'] for r in records)
    for a in ('C','F'):
        tests.update({f'ssim_vs_{a}':comparisons[a]['ssim']['family_lower']>=-.002,f'lpips_vs_{a}':comparisons[a]['lpips']['family_upper']<=.005,f'loss_vs_{a}':comparisons[a]['loss']['family_upper']<=floor})
    wins=comparisons['F']['psnr']['images_positive']>=math.ceil(2*len(images)/3)
    supported=all(tests.values()) and wins
    return dict(classification='CONFIRMED_STATE_GAIN_IN_THIS_COHORT' if supported else 'INCONCLUSIVE_OR_NO_CONFIRMED_STATE_GAIN',image_clusters=len(images),image_ids=images,comparisons=comparisons,primary_bounds=tests,primary_family_size=FAMILY_SIZE,at_least_two_thirds_images_beat_F=wins,
        locked_S_gain_at_least_point05_vs_F_excluded=comparisons['F']['psnr']['family_upper']<.05,
        per_arm_vs_C={a:effect(records,a) for a in ('C','F','S','P','M','STAGE')},
        inference='100000 basic paired bootstrap resamples of per-image two-seed averages; ordinary 95% intervals and one-sided Bonferroni alpha=.05/11 bounds for eleven preregistered primary comparisons. PSNR numerical-zero floor 1e-9 dB. Approximate coverage, not a finite-sample guarantee; other bounds descriptive.')
