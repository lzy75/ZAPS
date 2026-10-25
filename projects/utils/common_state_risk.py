"""A39 causal state readouts and offline single-decision risk diagnosis; no controller."""
import math
import numpy as np
from utils import trajectory_state_geometry as geometry

POINTS=((19,641),(24,319))
GAMMAS=(.5,1.5,2.)
SIGMAS=(.05,.1)
CALLS=30+len(GAMMAS)*sum(29-k for k,_ in POINTS)
P=('log_R','response_cos','log_b')
I=('clean_cos','clean_turn','log_speed')
G=('gradient_cos','gradient_turn','q_gradient_cos')
FEATURES={'STAGE_NOISE':(),'R':('log_R',),'P':P,'I':I,'PI':P+I,'PIG':P+I+G}
METRICS=('psnr','ssim','lpips','loss')


def arm(k,gamma):return f'K{k}_G{gamma:g}'


def expected_calls(grid):
    times=list(reversed(grid))
    if len(times)!=30 or any(times[k]!=t for k,t in POINTS):raise ValueError('Fixed measurement anchors differ')
    return times+[t for k,_ in POINTS for _ in GAMMAS for t in times[k+1:]]


class History:
    """Only current/past C frames; clean-path and exact clean-gradient direction proxies."""
    def __init__(self):self.frames=[]

    def append(self,u,clean,q,gradient,readout,clip):
        if self.frames and u<=self.frames[-1]['u']:raise ValueError('Noise progress must increase')
        frame=dict(u=float(u),clean=np.asarray(clean,dtype=np.float32).copy(),
                   gradient=np.asarray(gradient,dtype=np.float32).copy())
        clean_cos=clean_turn=speed=gradient_cos=gradient_turn=None
        if self.frames:
            old=self.frames[-1];du=u-old['u']
            velocity=(frame['clean'].astype(float)-old['clean'])/du
            speed=geometry.rms(velocity)/max(geometry.rms(clean),1e-8)
            frame['velocity']=velocity
            floor=32*np.finfo(np.float32).eps*max(1.,geometry.rms(clean),geometry.rms(old['clean']))
            frame['resolved']=geometry.rms(frame['clean'].astype(float)-old['clean'])>floor
            if 'velocity' in old:
                clean_cos=geometry.cosine(velocity,old['velocity']) if frame['resolved'] and old['resolved'] else None
                denominator=geometry.rms(velocity)+geometry.rms(old['velocity'])
                if clean_cos is not None and denominator>1e-8:
                    clean_turn=geometry.rms(velocity-old['velocity'])/denominator
            gradient_cos=geometry.cosine(gradient,old['gradient'])
            if gradient_cos is not None:gradient_turn=math.sqrt(max(0.,2.-2.*gradient_cos))/du
        def log(x):return math.log(max(float(x),1e-12)) if x is not None else None
        features=dict(log_R=log(readout['R']),response_cos=readout['cosine'],log_b=log(readout['relative_response']),
                      clean_cos=clean_cos,clean_turn=clean_turn,log_speed=log(speed),
                      gradient_cos=gradient_cos,gradient_turn=math.log1p(gradient_turn) if gradient_turn is not None else None,
                      q_gradient_cos=geometry.cosine(q,gradient))
        if any(v is not None and not math.isfinite(v) for v in features.values()):raise ValueError('Nonfinite state')
        self.frames.append(frame);self.frames=self.frames[-2:]
        return dict(features=features,R=readout['R'],clip=float(clip),
                    clean_geometry_resolved=clean_cos is not None,gradient_geometry_resolved=gradient_cos is not None)


def group_key(k,sigma):return k*1000+int(round(sigma*1000))


def advance(z,x,field,positions,k,noises,gamma=1.):
    """Original floating-point order; only the additive guidance is rescaled."""
    from utils import diag_frozen_response_transfer as parent
    t=positions.nominal[k]
    base=(parent.P.terminal.terminal_output(z,x,field) if k==len(positions.nominal)-1 else
          parent.P.coupled.posterior(x,field['clean'],t,positions.nominal[k+1],
                                    z.dm.alphas_cumprod,noises[k])+field['q'])
    return base if gamma==1. else base+(gamma-1.)*field['q']


def compare(a,b):
    reduction=1.-a['mae']/b['mae'] if b['mae']>1e-6 else None
    wins=sum(a['per_image_mae'][i]<b['per_image_mae'][i]-1e-9 for i in a['per_image_mae'])
    return dict(relative_mae_reduction=reduction,images_better=wins,
                candidate=reduction is not None and reduction>=.10 and wins>=4)


def analyze(records):
    if len(records)!=12 or len({r['case']['image_id'] for r in records})!=6:
        raise ValueError('Need all six-image/two-noise development cases')
    outcomes={};predictions={};rows_by_gamma={}
    for gamma in GAMMAS:
        rows=[]
        for record in records:
            for k,t in POINTS:
                base=record['metrics']['C'];metrics=record['metrics'][arm(k,gamma)]
                rows.append(dict(image_id=record['case']['image_id'],seed=record['case']['sampling_seed'],
                    step=group_key(k,record['case']['sigma']),source_step=k,source_t=t,sigma=record['case']['sigma'],
                    features=record['rows'][k]['state']['features'],labels={m:metrics[m]-base[m] for m in METRICS}))
        rows_by_gamma[gamma]=rows;predictions[gamma]={}
        targets={}
        for target in METRICS:
            models={name:geometry.cross_validate(rows,features,target) for name,features in FEATURES.items()}
            predictions[gamma][target]=models
            targets[target]=dict(mae={n:v['mae'] for n,v in models.items()},
                comparisons={f'{a}_{b}':compare(models[a],models[b]) for a,b in
                             (('R','STAGE_NOISE'),('P','STAGE_NOISE'),('P','R'),('I','STAGE_NOISE'),
                              ('PI','STAGE_NOISE'),('PI','P'),('PIG','PI'))})
        outcomes[str(gamma)]=dict(states=len(rows),targets=targets)
    # Equal-weight actions and image clusters; no held image's labels enter its predictor.
    pooled={}
    for target in METRICS:
        models={}
        for name in FEATURES:
            errors={image:float(np.mean([predictions[g][target][name]['per_image_mae'][image] for g in GAMMAS]))
                    for image in predictions[GAMMAS[0]][target][name]['per_image_mae']}
            models[name]=dict(mae=float(np.mean(list(errors.values()))),per_image_mae=errors)
        pooled[target]=dict(mae={n:v['mae'] for n,v in models.items()},comparisons={f'{a}_{b}':compare(models[a],models[b])
            for a,b in (('R','STAGE_NOISE'),('P','STAGE_NOISE'),('P','R'),('I','STAGE_NOISE'),
                        ('PI','STAGE_NOISE'),('PI','P'),('PIG','PI'))})
    indexed={g:{(r['image_id'],r['step']):r for r in rows} for g,rows in rows_by_gamma.items()}
    estimates={g:{target:{name:{(v['image_id'],v['step']):v['prediction'] for v in model['predictions']}
                  for name,model in models.items()} for target,models in targets.items()} for g,targets in predictions.items()}
    emulators={};surfaces=[]
    for sigma in SIGMAS:
        for k,t in POINTS:
            key=group_key(k,sigma)
            for gamma in GAMMAS:
                rows=[r for r in rows_by_gamma[gamma] if r['step']==key]
                delta={m:float(np.mean([r['labels'][m] for r in rows])) for m in METRICS}
                surfaces.append(dict(sigma=sigma,source_t=t,gamma=gamma,mean_delta=delta,
                    quality_guard=delta['ssim']>=-.002 and delta['lpips']<=.005,
                    images_psnr_positive=sum(r['labels']['psnr']>1e-9 for r in rows),
                    physical_better_but_lpips_risky=sum(r['labels']['loss']<0 and r['labels']['lpips']>.005 for r in rows)))
    for name in FEATURES:
        choices=[]
        for key,row in sorted(indexed[GAMMAS[0]].items()):
            feasible=[1.]
            for gamma in GAMMAS:
                if (estimates[gamma]['psnr'][name][key]>1e-9 and estimates[gamma]['ssim'][name][key]>=-.002
                        and estimates[gamma]['lpips'][name][key]<=.005):feasible.append(gamma)
            chosen=max(feasible,key=lambda g:0. if g==1. else estimates[g]['psnr'][name][key])
            delta={m:0. for m in METRICS} if chosen==1. else indexed[chosen][key]['labels']
            choices.append(dict(image_id=key[0],sigma=row['sigma'],source_t=row['source_t'],gamma=chosen,delta=delta))
        cells=[]
        for sigma in SIGMAS:
            for _,t in POINTS:
                current=[v for v in choices if v['sigma']==sigma and v['source_t']==t]
                delta={m:float(np.mean([v['delta'][m] for v in current])) for m in METRICS}
                cells.append(dict(sigma=sigma,source_t=t,mean_delta=delta,
                    quality_guard=delta['ssim']>=-.002 and delta['lpips']<=.005,
                    images_psnr_positive=sum(v['delta']['psnr']>1e-9 for v in current),
                    admitted_risky_images=sum(v['gamma']!=1. and (v['delta']['lpips']>.005 or v['delta']['ssim']<-.002) for v in current),
                    actions={str(g):sum(v['gamma']==g for v in current) for g in (1.,)+GAMMAS}))
        emulators[name]=dict(cells=cells,choices=choices)
    primary=pooled['lpips']['comparisons']
    return dict(classification='DUAL_STATE_RISK_INFORMATION_CANDIDATE_ONLY' if all(primary[k]['candidate'] for k in
                  ('PI_STAGE_NOISE','PI_P')) else 'NO_CONFIRMED_DUAL_STATE_RISK_INCREMENT',
                physical_information_candidate=primary['P_STAGE_NOISE']['candidate'],
                internal_information_candidate=primary['I_STAGE_NOISE']['candidate'],
                dual_information_candidate=all(primary[k]['candidate'] for k in ('PI_STAGE_NOISE','PI_P')),
                exact_gradient_extra_candidate=primary['PIG_PI']['candidate'],
                pooled_prediction=pooled,actions=outcomes,single_pulse_surfaces=surfaces,
                offline_single_decision_emulators=emulators,
                limits='Six reused image clusters; all noise conditions held out together. Ridge1 is an offline diagnostic, '
                       'not a trained deployed controller or the stopped guidance regularizer. Fixed-time/sigma-centered '
                       'prediction of FINAL pulse deltas. Four separate single-decision cells, never a combined pulse path. '
                       'Clean/gradient turns retain past sampling noise and are not true ODE curvature. '
                       'No independent confirmation, online multi-step gain, step-size, NFE-saving or convergence claim.')
