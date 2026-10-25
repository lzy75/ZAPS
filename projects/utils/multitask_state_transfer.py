"""A40 frozen response control, paired task summaries and frozen feature transfer."""
import math
import statistics
import numpy as np
from utils import common_state_risk as risk
from utils import transfer_response_feedback as response
from utils import trajectory_state_geometry as geometry
from utils import cached_policy_opportunity as opportunity

DATASETS=('ffhq','imagenet')
TASKS=('super_resolution','gaussian_deblur')
ARMS=('C','TIME_SOURCE','TIME_TASK','STATE')
IMAGES=6
SIGMA=.05
MEASURE_CALLS=risk.CALLS+30
TIME_CALLS=60
MIN_FREE_MIB=10240


def configuration(original,dataset,task):
    if dataset not in DATASETS or task not in TASKS:raise ValueError('Unregistered task/model')
    return dict(original,lr=.01 if dataset=='ffhq' else .001,
                zeta_init=.1 if task=='super_resolution' else .2,use_learned_var=False)


def select_devices(rows,visible=None,first=None,second=None):
    from utils.experiment_gpu_selection import choose
    aliases={token:r for r in rows for token in (r['index'],r['uuid'])}
    selected=choose(rows,visible,first,second)
    for token in (first,second):
        if token is not None and aliases[token]['free']<MIN_FREE_MIB:
            raise ValueError('A40 explicit GPU requires at least 10 GiB free: '+token)
    usable=[token for token in selected if aliases[token]['free']>=MIN_FREE_MIB]
    if not usable:raise ValueError('A40 requires an allocated GPU with at least 10 GiB free; no experiment started')
    return [token if aliases[token]['free']>=MIN_FREE_MIB else usable[0] for token in selected]


def expected_calls(grid,kind):
    if kind=='BASELINE':return list(reversed(grid))*10
    if kind=='MEASURE':return risk.expected_calls(grid)+list(reversed(grid))
    if kind=='TIME':return list(reversed(grid))*2
    raise ValueError('Unknown phase')


def profiles(records,grid):
    """Other images only, no outcome/GT metrics; target-task unlabeled calibration."""
    output={}
    for dataset in DATASETS:
        for task in TASKS:
            group=[r for r in records if r['case']['dataset']==dataset and r['case']['task']==task]
            if len(group)!=6 or len({r['case']['image_id'] for r in group})!=6:
                raise ValueError('Need six complete state trajectories per condition')
            for held in group:
                others=[r for r in group if r['case']['image_id']!=held['case']['image_id']]
                if any([v['t'] for v in r['state_trace']]!=list(reversed(grid)) for r in others):
                    raise ValueError('State calibration grid differs')
                values=[statistics.median(r['state_trace'][k]['gamma'] for r in others) for k in range(30)]
                if any(not math.isfinite(v) or not 0<=v<=2 for v in values):raise ValueError('Invalid time profile')
                output[held['case']['key']]=dict(gamma=values,
                    calibration_cases=[r['case']['key'] for r in others],GT_used=False)
    return output


def pulse_rows(records,gamma):
    rows=[]
    for record in records:
        for k,t in risk.POINTS:
            rows.append(dict(image_id=record['case']['image_id'],seed=record['case']['sampling_seed'],
                step=risk.group_key(k,record['case']['sigma']),features=record['rows'][k]['state']['features'],
                labels={m:record['metrics'][risk.arm(k,gamma)][m]-record['metrics']['C'][m] for m in risk.METRICS}))
    return rows


def predict_transfer(train,test,features,target):
    """Exactly fixed ridge1 recipe; fit ONLY source A39 labels, no target fitting."""
    if {r['image_id'] for r in train}&{r['image_id'] for r in test}:
        # Dataset prefixes below separate identities; same content SHA is separately excluded in the manifest.
        raise ValueError('Source/target identities overlap')
    stages={r['step'] for r in train}
    if any(r['step'] not in stages for r in test):raise ValueError('Unmeasured transfer stage')
    means={s:float(np.mean([r['labels'][target] for r in train if r['step']==s])) for s in stages}
    residual=np.zeros(len(test))
    if features:
        impute={(s,f):float(np.median([r['features'][f] for r in train if r['step']==s and r['features'][f] is not None]))
                if any(r['step']==s and r['features'][f] is not None for r in train) else 0.
                for s in stages for f in features}
        def matrix(items):
            values=[[r['features'][f] if r['features'][f] is not None else impute[(r['step'],f)] for f in features] for r in items]
            masks=[[float(r['features'][f] is None) for f in features] for r in items]
            return np.concatenate((np.asarray(values),np.asarray(masks)),axis=1)
        x,xt=matrix(train),matrix(test)
        center={s:x[[r['step']==s for r in train]].mean(axis=0) for s in stages}
        x-=np.asarray([center[r['step']] for r in train]);xt-=np.asarray([center[r['step']] for r in test])
        sd=x.std(axis=0);sd[sd<1e-12]=1.;x/=sd;xt/=sd
        y=np.asarray([r['labels'][target]-means[r['step']] for r in train]);sy=max(float(y.std()),1e-12)
        beta=np.linalg.solve(x.T@x+np.eye(x.shape[1]),x.T@(y/sy));residual=xt@beta*sy
    predictions=[dict(image_id=r['image_id'],step=r['step'],prediction=means[r['step']]+float(extra),
                      truth=r['labels'][target],abs_error=abs(means[r['step']]+float(extra)-r['labels'][target]))
                 for r,extra in zip(test,residual)]
    errors={i:float(np.mean([r['abs_error'] for r in predictions if r['image_id']==i])) for i in {r['image_id'] for r in test}}
    return dict(mae=float(np.mean(list(errors.values()))),per_image_mae=errors,predictions=predictions)


def feature_analysis(records,source_records):
    predictions={mode:{m:{n:[] for n in risk.FEATURES} for m in risk.METRICS}
                 for mode in ('frozen_source_transfer','target_task_LOIO_diagnostic')}
    for gamma in risk.GAMMAS:
        train=pulse_rows(source_records,gamma);test=pulse_rows(records,gamma)
        for r in train:r['image_id']='SOURCE_'+r['image_id']
        for r in test:r['image_id']='TARGET_'+r['image_id']
        for m in risk.METRICS:
            for name,features in risk.FEATURES.items():
                predictions['frozen_source_transfer'][m][name].append(predict_transfer(train,test,features,m))
                predictions['target_task_LOIO_diagnostic'][m][name].append(geometry.cross_validate(test,features,m))
    result={}
    for mode,targets in predictions.items():
        result[mode]={}
        for m,models in targets.items():
            pooled={}
            for name,items in models.items():
                errors={i:float(np.mean([v['per_image_mae'][i] for v in items])) for i in items[0]['per_image_mae']}
                pooled[name]=dict(mae=float(np.mean(list(errors.values()))),per_image_mae=errors)
            result[mode][m]=dict(mae={n:v['mae'] for n,v in pooled.items()},
                comparisons={f'{a}_{b}':risk.compare(pooled[a],pooled[b]) for a,b in
                             (('P','STAGE_NOISE'),('I','STAGE_NOISE'),('PI','STAGE_NOISE'),('PI','P'),('PIG','PI'))})
    return result


def summarize(records,source_records):
    if len({r['case']['key'] for r in records})!=len(records):raise ValueError('Duplicate transfer case')
    groups=[]
    for dataset in DATASETS:
        for task in TASKS:
            current=[r for r in records if r['case']['dataset']==dataset and r['case']['task']==task]
            if len({r['case']['image_id'] for r in current})!=len(current):raise ValueError('Duplicate image in condition')
            g=dict(dataset=dataset,task=task,completed_cases=len(current),planned_cases=6)
            if current:
                g['absolute_mean_metrics']={a:{m:statistics.mean(r['metrics'][a][m] for r in current)
                    for m in risk.METRICS} for a in ('NAIVE',)+ARMS}
                pairs=(('STATE','C'),('TIME_SOURCE','C'),('TIME_TASK','C'),('STATE','TIME_SOURCE'),('STATE','TIME_TASK'))
                g['contrasts']={a+'_'+b:response.contrast(current,a,b) for a,b in pairs}
                context=response.contrast(current,'C','NAIVE')
                g['baseline_context']=dict(C_minus_naive=context,
                    baseline_below_naive_mean_psnr=context['mean_delta']['psnr']<0,
                    interpretation='Naive is bicubic observation upsampling for SR, noisy observation for blur. '
                                   'A weak matched C limits practical quality claims even if STATE-C is positive.')
                def passed(name,threshold):
                    v=g['contrasts'][name]
                    return len(current)==6 and v['mean_delta']['psnr']>=threshold and v['images_psnr_positive']>=4 and v['quality_guard']
                g.update(state_vs_C_candidate=passed('STATE_C',.05),
                         state_vs_frozen_time_candidate=passed('STATE_TIME_SOURCE',.02),
                         state_vs_task_time_candidate=passed('STATE_TIME_TASK',.02))
                g['weights']=dict(mean_abs_vs_source=statistics.mean(abs(a['gamma']-b['gamma'])
                    for r in current for a,b in zip(r['state_trace'],r['time_traces']['TIME_SOURCE'])),
                    mean_abs_vs_task=statistics.mean(abs(a['gamma']-b['gamma'])
                    for r in current for a,b in zip(r['state_trace'],r['time_traces']['TIME_TASK'])),
                    cap_fraction=statistics.mean(float(v['cap_hit']) for r in current for v in r['state_trace']),
                    fallback_fraction=statistics.mean(float(v['fallback']) for r in current for v in r['state_trace']))
                if len(current)==6:
                    g['frozen_indicator_analysis']=feature_analysis(current,source_records)
                    g['dual_indicator_candidates']={mode:all(targets['lpips']['comparisons'][key]['candidate']
                        for key in ('PI_STAGE_NOISE','PI_P')) for mode,targets in g['frozen_indicator_analysis'].items()}
                    g['single_pulse_opportunity']=[]
                    for k,t in risk.POINTS:
                        arms=('C',)+tuple(risk.arm(k,x) for x in risk.GAMMAS)
                        data=[[[r['metrics'][a][m]-r['metrics']['C'][m] for m in risk.METRICS] for a in arms] for r in current]
                        out=opportunity.analyze(data,arms)
                        out.update(source_t=t,limits='Finite common-C single-pulse diagnostic only; no GT online selection.')
                        g['single_pulse_opportunity'].append(out)
            groups.append(g)
    complete=all(g['completed_cases']==6 for g in groups)
    gains=[g['dataset']+'/'+g['task'] for g in groups if g.get('state_vs_C_candidate')]
    increments=[g['dataset']+'/'+g['task'] for g in groups if g.get('state_vs_C_candidate') and
                g.get('state_vs_frozen_time_candidate') and g.get('state_vs_task_time_candidate')]
    return dict(classification=('MULTITASK_STATE_INCREMENT_CANDIDATE_ONLY' if len(increments)==4 else
                'CONDITIONAL_STATE_INCREMENT_ONLY' if increments else 'NO_CONFIRMED_MULTITASK_STATE_INCREMENT')
                if complete else 'INCOMPLETE_PILOT',groups=groups,state_gain_conditions=gains,
                online_feedback='Frozen PHYSICAL response only; internal/dual indicators are offline diagnostics.',
                baseline_below_naive_conditions=[g['dataset']+'/'+g['task'] for g in groups
                                                if g.get('baseline_context',{}).get('baseline_below_naive_mean_psnr')],
                state_increment_conditions=increments,
                limits='Six images per dataset, shared across tasks, one seed: twelve image clusters. '
                       'No retuning of response rule, ridge or source time table. Target-task time uses five OTHER '
                       'images without quality labels and incurs calibration-data cost. Frozen-source prediction '
                       'is transfer; target-task LOIO is exploratory diagnostic, not transfer or an online controller. '
                       'Native exact-adjoint/unit-domain local references; new protocol versus historical legacy SR. '
                       'Last-unroll feedback only; no full-ten-epoch feedback, step changes, NFE saving, convergence '
                       'or global generalization. Bootstrap intervals are descriptive, unadjusted, approximate.')
