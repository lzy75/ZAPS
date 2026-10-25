"""A41 fixed-state finite-action geometry and leakage-controlled transfer diagnosis."""
import math
import numpy as np

TASKS=('super_resolution','gaussian_deblur','inpainting','motion_deblur')
POINTS=((10,872),(19,641),(24,319),(27,54))
GAMMAS=(0.,.5,1.,1.5,2.)
METRICS=('psnr','ssim','lpips','loss')
OLD=('log_R','response_cos','log_b','clean_cos','clean_turn','log_speed',
     'gradient_cos','gradient_turn','q_gradient_cos')
PHYSICAL=('log_R_active','physical_gain','measurement_cos','log_measurement_response')
INTERNAL=('response_cos','response_turn','half_response_turn','log_response_speed','log_displacement')
FEATURES={'STAGE':(),'OLD':tuple('old_'+v for v in OLD),'P':PHYSICAL,'PI':PHYSICAL+INTERNAL}


def arm(k,gamma):return f'K{k}_G{gamma:g}'


def is_reused(k,gamma,reused):
    return reused and k in (19,24) and gamma in (.5,1.5,2.)


def expected_calls(grid,reused):
    times=list(reversed(grid))
    if len(times)!=30 or any(times[k]!=t for k,t in POINTS):raise ValueError('A41 anchor/grid mismatch')
    calls=list(times)
    for k,_ in POINTS:
        calls.extend([times[k+1]]*6)
        for g in GAMMAS:
            if g!=1. and not is_reused(k,g,reused):calls.extend(times[k+1:])
    return calls


def _rms(x):return float(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2)))


def _cos(a,b):
    a,b=np.asarray(a,dtype=np.float64).reshape(-1),np.asarray(b,dtype=np.float64).reshape(-1)
    norm=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.clip(a@b/norm,-1.,1.)) if norm>1e-20 else None


def informative(x,mask=None):
    x=np.asarray(x,dtype=np.float64)
    if mask is None:return x.reshape(-1)
    active=np.broadcast_to(np.asarray(mask,dtype=bool),x.shape)
    if not active.any():raise ValueError('No informative measurements')
    return x[active]


def features(cleans,observations,y,sigma,old,mask=None,repeat=None):
    """No GT input. All gamma endpoints must use one fixed time/noise/precision."""
    if set(cleans)!=set(GAMMAS) or set(observations)!=set(GAMMAS) or sigma<=0:
        raise ValueError('Incomplete response probes')
    c={g:np.asarray(v,dtype=np.float64) for g,v in cleans.items()}
    a={g:informative(v,mask) for g,v in observations.items()};yy=informative(y,mask)
    residual=yy-a[1.];loss={g:float(np.mean((yy-a[g])**2)/(2*sigma**2)) for g in GAMMAS}
    d0,d1=c[1.]-c[0.],c[2.]-c[1.]
    h0,h1=c[1.]-c[.5],c[1.5]-c[1.]
    repeat_rms=_rms(np.asarray(repeat,dtype=np.float64)-c[1.]) if repeat is not None else 0.
    floor=max(32*np.finfo(np.float32).eps,10*repeat_rms)
    resolved=_rms(c[2.]-c[0.])>floor
    turn=lambda x,z:_rms(z-x)/max(_rms(x)+_rms(z),1e-20)
    common=dict(response_cos=_cos(d0,d1) if resolved else None,
        response_turn=turn(d0,d1) if resolved else None,
        half_response_turn=turn(h0,h1) if _rms(c[1.5]-c[.5])>floor else None,
        log_response_speed=math.log(max((_rms(d0)+_rms(d1))/2,1e-12)) if resolved else None,
        log_R_active=math.log(max(_rms(residual)/sigma,1e-12)))
    output={}
    for g in GAMMAS:
        change=a[g]-a[1.]
        output[g]=dict(common,physical_gain=(loss[1.]-loss[g])/max(loss[1.],1e-12),
            measurement_cos=_cos(residual,change),
            log_measurement_response=math.log(max(_rms(change)/max(_rms(residual),1e-12),1e-12)),
            log_displacement=math.log(max(_rms(c[g]-c[1.])/max(_rms(c[1.]),1e-8),1e-12)) if resolved else None,
            **{'old_'+v:old.get(v) for v in OLD})
        if any(v is not None and not math.isfinite(v) for v in output[g].values()):raise ValueError('Nonfinite response')
    return dict(by_gamma=output,loss=loss,resolved=resolved,repeat_rms=repeat_rms,resolution_floor=floor,
                active_measurements=len(yy),half_full_turn_difference=(abs(common['response_turn']-common['half_response_turn'])
                if common['response_turn'] is not None and common['half_response_turn'] is not None else None))


def training_rows(rows,held,mode):
    if mode=='TASK_LOIO':return [r for r in rows if r['image_id']!=held['image_id'] and r['task']==held['task']]
    if mode=='LEAVE_TASK_AND_IMAGE_OUT':
        return [r for r in rows if r['image_id']!=held['image_id'] and r['task']!=held['task']]
    raise ValueError('Unknown transfer split')


def predict(train,test,fields,target):
    """Train-only (step,gamma) centering/imputation; fixed ridge1, zero keep-action."""
    train=[r for r in train if r['gamma']!=1.]
    if not train:raise ValueError('Empty training fold')
    cells=sorted({(r['step'],r['gamma']) for r in train})
    center={s:float(np.mean([r['labels'][target] for r in train if (r['step'],r['gamma'])==s])) for s in cells}
    if any((r['step'],r['gamma']) not in cells for r in test if r['gamma']!=1.):raise ValueError('Unseen step/action')
    selected=[r for r in test if r['gamma']!=1.]
    residual=np.zeros(len(selected))
    if fields:
        impute={}
        for cell in cells:
            for f in fields:
                valid=[r['features'].get(f) for r in train if (r['step'],r['gamma'])==cell and r['features'].get(f) is not None]
                impute[cell,f]=float(np.median(valid)) if valid else 0.
        def matrix(items):
            values=[[r['features'].get(f) if r['features'].get(f) is not None else impute[(r['step'],r['gamma']),f]
                     for f in fields] for r in items]
            masks=[[float(r['features'].get(f) is None) for f in fields] for r in items]
            return np.concatenate((np.asarray(values),np.asarray(masks)),axis=1)
        x,xt=matrix(train),matrix(selected)
        means={s:x[[((r['step'],r['gamma'])==s) for r in train]].mean(axis=0) for s in cells}
        x-=np.asarray([means[r['step'],r['gamma']] for r in train])
        xt-=np.asarray([means[r['step'],r['gamma']] for r in selected])
        scale=x.std(axis=0);scale[scale<1e-12]=1.;x/=scale;xt/=scale
        y=np.asarray([r['labels'][target]-center[r['step'],r['gamma']] for r in train])
        residual=xt@np.linalg.solve(x.T@x+np.eye(x.shape[1]),x.T@y)
    predictions=iter(center[r['step'],r['gamma']]+float(e) for r,e in zip(selected,residual))
    return [0. if r['gamma']==1. else next(predictions) for r in test]


def analyze(records):
    rows=[]
    for rec in records:
        case=rec['case'];base=rec['metrics']['C']
        for point in rec['points']:
            for g in GAMMAS:
                metric=base if g==1. else rec['metrics'][arm(point['step'],g)]
                rows.append(dict(image_id=case['image_id'],task=case['task'],step=point['step'],gamma=g,
                    features=point['response']['by_gamma'][str(g)],
                    labels={m:metric[m]-base[m] for m in METRICS}))
    images=sorted({r['image_id'] for r in rows});tasks=[]
    if len(records)!=24 or len(images)!=6:return dict(classification='INCOMPLETE_PILOT',completed_cases=len(records))
    for task in TASKS:
        task_rows=[r for r in rows if r['task']==task];splits={}
        for mode in ('TASK_LOIO','LEAVE_TASK_AND_IMAGE_OUT'):
            predictions={name:[] for name in FEATURES};folds=[]
            for image in images:
                test=[r for r in task_rows if r['image_id']==image];train=training_rows(rows,test[0],mode)
                folds.append(dict(held_image=image,held_task=task,training_cases=sorted({r['task']+':'+r['image_id'] for r in train})))
                for name,fields in FEATURES.items():
                    estimated={m:predict(train,test,fields,m) for m in METRICS}
                    predictions[name].extend(dict(r,prediction={m:estimated[m][j] for m in METRICS}) for j,r in enumerate(test))
            errors={}
            for name,items in predictions.items():
                errors[name]={m:{i:float(np.mean([abs(r['prediction'][m]-r['labels'][m]) for r in items
                    if r['image_id']==i and r['gamma']!=1.])) for i in images} for m in METRICS}
            information={}
            for baseline in ('STAGE','P'):
                ma=lambda n,m:float(np.mean(list(errors[n][m].values())))
                reduction=1-ma('PI','psnr')/ma(baseline,'psnr') if ma(baseline,'psnr')>1e-9 else None
                wins=sum(errors['PI']['psnr'][i]<errors[baseline]['psnr'][i]-1e-9 for i in images)
                lpips_nonworse=ma('PI','lpips')<=1.05*ma(baseline,'lpips')+1e-12
                information[baseline]=dict(relative_psnr_mae_reduction=reduction,images_better=wins,
                    lpips_mae_nonworse=lpips_nonworse,candidate=reduction is not None and reduction>=.1 and wins>=4 and lpips_nonworse)
            policies={}
            for name,items in predictions.items():
                positions=[]
                for k,t in POINTS:
                    choices=[]
                    for image in images:
                        available=[r for r in items if r['step']==k and r['image_id']==image]
                        feasible=[r for r in available if r['gamma']==1. or
                            (r['prediction']['psnr']>1e-9 and r['prediction']['ssim']>=-.002 and r['prediction']['lpips']<=.005)]
                        chosen=max(feasible,key=lambda r:(r['prediction']['psnr'],-abs(r['gamma']-1.)))
                        choices.append(dict(image_id=image,gamma=chosen['gamma'],delta=chosen['labels']))
                    mean={m:float(np.mean([c['delta'][m] for c in choices])) for m in METRICS}
                    positions.append(dict(step=k,t=t,mean_delta=mean,choices=choices,
                        quality_guard=mean['ssim']>=-.002 and mean['lpips']<=.005,
                        images_psnr_positive=sum(c['delta']['psnr']>1e-9 for c in choices)))
                policies[name]=positions
            increments=[]
            for state,time in zip(policies['PI'],policies['STAGE']):
                delta={m:state['mean_delta'][m]-time['mean_delta'][m] for m in METRICS}
                good=(state['mean_delta']['psnr']>=.05 and state['images_psnr_positive']>=4 and state['quality_guard']
                      and delta['psnr']>=.02 and delta['ssim']>=-.002 and delta['lpips']<=.005)
                increments.append(dict(step=state['step'],PI_minus_STAGE=delta,candidate=good))
            splits[mode]=dict(mae={n:{m:float(np.mean(list(v.values()))) for m,v in metrics.items()} for n,metrics in errors.items()},
                information=information,policies=policies,increments=increments,folds=folds)
        coverage=[];opportunity=[]
        for k,t in POINTS:
            resolved=sum(v['response']['resolved'] for r in records if r['case']['task']==task
                         for v in r['points'] if v['step']==k)
            coverage.append(dict(step=k,t=t,resolved=resolved,total=6))
            current=[r for r in task_rows if r['step']==k]
            means={str(g):{m:float(np.mean([r['labels'][m] for r in current if r['gamma']==g]))
                          for m in METRICS} for g in GAMMAS}
            best=max(GAMMAS,key=lambda g:means[str(g)]['psnr'])
            oracle=float(np.mean([max(r['labels']['psnr'] for r in current if r['image_id']==image) for image in images]))
            opportunity.append(dict(step=k,t=t,best_fixed_gamma=best,best_fixed_delta=means[str(best)],
                unconstrained_psnr_oracle_headroom=oracle-means[str(best)]['psnr'],surface=means,
                limits='Retrospective GT opportunity only; not an online action or position selector.'))
        tasks.append(dict(task=task,validation=splits,coverage=coverage,opportunity=opportunity))
    support=all(all(v['candidate'] for v in t['validation']['LEAVE_TASK_AND_IMAGE_OUT']['information'].values())
                and all(v['resolved']>=4 for v in t['coverage']) for t in tasks)
    return dict(classification='FOUR_TASK_RESPONSE_INFORMATION_CANDIDATE_ONLY' if support else 'NO_FOUR_TASK_RESPONSE_INFORMATION_SUPPORT',
        tasks=tasks,image_clusters=6,rows=len(rows),
        limits='Reused six-image cohort; final single-pulse labels, four separate positions, paid lookahead. '
               'No deployed feedback, combined pulse, step allocation, new-image confirmation or equal-cost online claim.')
