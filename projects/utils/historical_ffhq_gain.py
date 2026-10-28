"""A65 frozen 2026-09-17 FFHQ protocol, numerical replay gates and cost rules."""
import csv
from pathlib import Path
import statistics
from utils.paper_clock_alignment import read,write,digest,require,grid,GIB,available_gpus

EXPERIMENT='A65_historical_ffhq_method_gain'
ROOT='/data0/lzy/ZAPS-results/diag_historical_ffhq_gain/A65_20261028'
BRANCH='results/historical-ffhq-method-gain-20261028'
TASK='super_resolution'
ARMS=('HISTORICAL_NATIVE','LEGACY_IEEE','EXACT_IEEE','RESPONSE_300','RESPONSE_30')
REPEATS=3
NFE={a:30 if a=='RESPONSE_30' else 300 for a in ARMS}
UPDATES={a:0 if a=='RESPONSE_30' else 9 if a=='RESPONSE_300' else 10 for a in ARMS}
CSV_PATH=Path(__file__).resolve().parents[1]/'reports/20261028_historical_baseline_comparison/september17_metrics.csv'
AUDIT_PATH=CSV_PATH.with_name('audit.json')
PAPER=dict(psnr=26.63,ssim=.768,lpips=.104,nfe=300,images=1000)
TOLERANCES=dict(observed_psnr=.001,psnr=.10,ssim=.01,lpips=.01,loss_relative=.05)
NAMES=dict(HISTORICAL_NATIVE='9月17日配置本地复测',LEGACY_IEEE='本地基线：统一精度',
           EXACT_IEEE='消融：仅改精确伴随',RESPONSE_300='新方法：完整300次前向',
           RESPONSE_30='新方法：独立30次前向')


def reference_rows():
    audit=read(AUDIT_PATH)
    require(digest(CSV_PATH)==audit['september_source']['csv_sha256'],'Historical CSV changed')
    with CSV_PATH.open() as stream:
        rows=[r for r in csv.DictReader(stream) if r['schedule']=='paper_15_10_5']
    require(len(rows)==10 and [int(r['seed']) for r in rows]==list(range(1000,1010)),
            'Need exactly original ten rows/seeds, no cohort reselection')
    require([Path(r['image']).stem for r in rows]==[f'{i:05d}' for i in range(10)],'Original images differ')
    return rows


def cases(directory):
    result=[]
    for row in reference_rows():
        image_id=Path(row['image']).stem
        result.append(dict(image_id=image_id,image=str(Path(directory)/(image_id+'.png')),
            seed=int(row['seed']),key='ffhq_super_resolution_'+image_id+'_s'+row['seed'],
            reference={k:float(row[k]) for k in ('psnr','ssim','lpips','observed_psnr','final_mse')},
            historical_optimization_seconds=float(row['seconds'])))
    return result


def config():
    return dict(num_steps=30,schedule=[15,10,5],timestep_spacing='linear',schedule_power=2.,
        num_epochs=10,lr=.01,zeta_init=.1,d_init=.2,eta=1.,use_learned_var=False,
        sampler_mode='ddpm',wave='db4',level=3,surrogate_score_jacobian=False)


def queue():
    result=[('HISTORICAL_NATIVE',0)]
    for r in range(REPEATS):
        order=ARMS[r:]+ARMS[:r]
        result.extend((a,r) for a in order if (a,r)!=('HISTORICAL_NATIVE',0))
    return result


def budget():
    return dict(max_method_forwards=10*REPEATS*sum(NFE.values()),max_method_input_vjp=1800,
        max_optimizer_backwards=10*REPEATS*sum(UPDATES.values()),null_diagnostic_forwards=30,
        initial_max_warmup_forwards=4,initial_max_warmup_input_vjp=4,
        resumed_warmups='Every additional model load/warmup counted separately')


def replay_gate(rows):
    require(len(rows)==10 and len({r['case']['image_id'] for r in rows})==10,'Need all original ten baseline cases')
    comparisons=[]
    for r in rows:
        require(r['arm']=='HISTORICAL_NATIVE' and r['repeat']==0,'Wrong replay arm')
        ref=r['case']['reference'];m=r['metrics']
        errors={k:abs(float(m[k])-ref[k]) for k in ('psnr','ssim','lpips')}
        errors['observed_psnr']=abs(r['observation']['observed_psnr']-ref['observed_psnr'])
        errors['loss_relative']=abs(r['loss_history'][-1]-ref['final_mse'])/max(abs(ref['final_mse']),1e-12)
        comparisons.append(dict(image_id=r['case']['image_id'],errors=errors,
                                passed=all(errors[k]<=limit for k,limit in TOLERANCES.items())))
    passed=all(r['passed'] for r in comparisons)
    return dict(passed=passed,classification='HISTORICAL_PROTOCOL_COMPATIBLE_PILOT' if passed
                else 'HISTORICAL_REPLAY_REVIEW_REQUIRED',comparisons=comparisons,tolerances=TOLERANCES,
                limits='Numerical engineering compatibility, not bitwise replay: original full checkpoint/y/RNG fingerprints absent')


def summarize_times(values):
    require(len(values)==3,'Need three complete independent timings')
    return dict(repeats=values,median=statistics.median(values),mean=statistics.mean(values),stdev=statistics.stdev(values))
