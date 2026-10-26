"""A59 finite-scale geometry and fixed1NN development diagnosis, not a policy."""
import math
import numpy as np
STEPS=(19,24,27)
TIMES=(641,319,54)
SCALES=(.001,.002)
GAMMAS=(.5,1.5)


def budget():
    replay=10*30;probes=10*len(STEPS)*len(SCALES)*2
    continuation=10*len(GAMMAS)*sum(29-k for k in STEPS)
    return dict(replay=replay,probes=probes,continuation=continuation,total=replay+probes+continuation,per_case=(replay+probes+continuation)//10)


def finite_geometry(center,pairs):
    u0=np.asarray(center,dtype=np.float64).reshape(-1);norm=float(np.linalg.norm(u0))
    if not len(u0) or not np.isfinite(u0).all() or not math.isfinite(norm) or norm<=0 or len(pairs)!=2:
        raise ValueError('Nonzero finite center and two finite-scale pairs required')
    rows=[]
    for epsilon,plus,minus in pairs:
        p,m=(np.asarray(v,dtype=np.float64).reshape(-1) for v in (plus,minus))
        if p.shape!=u0.shape or m.shape!=u0.shape or epsilon<=0 or not math.isfinite(epsilon):
            raise ValueError('Positive scale and matched response vectors required')
        pn,mn=float(np.linalg.norm(p)),float(np.linalg.norm(m))
        if not np.isfinite(p).all() or not np.isfinite(m).all() or min(pn,mn)<=0:
            raise ValueError('Degenerate perturbed direction')
        omega=float(np.linalg.norm(p/pn-m/mn))/(2*epsilon)
        sensitivity=float(np.linalg.norm(p-m))/(2*epsilon*norm)
        bend=float(np.linalg.norm(p+m-2*u0))/(epsilon*epsilon*norm)
        rows.append(dict(epsilon=float(epsilon),omega_per_input_RMS=omega,kappa_per_Euclidean_input=omega/math.sqrt(len(u0)),relative_sensitivity=sensitivity,finite_scale_nonlinearity=bend))
    spread=lambda key:abs(rows[0][key]-rows[1][key])/max(rows[0][key],rows[1][key],1e-12)
    agreement=dict(omega=spread('omega_per_input_RMS'),sensitivity=spread('relative_sensitivity'))
    return dict(usable=bool(max(agreement.values())<=.25),scales=rows,relative_spread=agreement,
        omega_mean=float(np.mean([v['omega_per_input_RMS'] for v in rows])),
        sensitivity_mean=float(np.mean([v['relative_sensitivity'] for v in rows])),
        units='d has RMS1 and Euclidean norm sqrt(n); kappa=omega/sqrt(n).')


def physical_features(row):
    z=row['center_readout']
    if not z.get('resolved') or z.get('u_RMS',0)<=0 or z.get('v_RMS',0)<=0:return None
    return [math.log1p(z['R_raw']),math.log(z['v_RMS']/z['u_RMS'])]


def geometry_features(row):
    first=physical_features(row);g=row['geometry']
    if first is None or not g.get('usable'):return None
    return first+[math.log1p(g['omega_mean']),math.log1p(g['sensitivity_mean'])]


def neighbor(train,target,label,features):
    x0=features(target);valid=[(v,features(v)) for v in train if features(v) is not None]
    if x0 is None or not valid:return None
    x=np.asarray([v for r,v in valid]);sd=x.std(axis=0);sd[sd<1e-12]=1.
    distance=np.sum(((x-np.asarray(x0))/sd)**2,axis=1)
    index=min(range(len(valid)),key=lambda i:(float(distance[i]),valid[i][0]['image_id']))
    return float(valid[index][0]['contrast'][label])


def analyze(rows):
    if len(rows)!=30 or len({(r['image_id'],r['task'],r['step']) for r in rows})!=30:
        raise ValueError('All30 state labels required; no dropping unresolved probes')
    targets=('psnr','ssim','lpips','loss');predictions=[]
    for row in rows:
        train=[r for r in rows if r['image_id']!=row['image_id'] and r['task']==row['task'] and r['step']==row['step']]
        if len(train)!=4:raise ValueError('Expected four other images at same task/time')
        values={}
        for target in targets:
            stage=float(np.mean([r['contrast'][target] for r in train]))
            physical=neighbor(train,row,target,physical_features)
            if physical is None:physical=stage
            geometry=neighbor(train,row,target,geometry_features)
            if geometry is None:geometry=physical
            values[target]=dict(truth=row['contrast'][target],STAGE=stage,PHYSICAL=physical,GEOMETRY=geometry)
        predictions.append(dict(image_id=row['image_id'],task=row['task'],step=row['step'],values=values))
    ids=sorted({r['image_id'] for r in rows});reports={}
    for target in targets:
        mae={name:float(np.mean([abs(r['values'][target][name]-r['values'][target]['truth']) for r in predictions])) for name in ('STAGE','PHYSICAL','GEOMETRY')}
        per_image={image:{name:float(np.mean([abs(r['values'][target][name]-r['values'][target]['truth']) for r in predictions if r['image_id']==image])) for name in mae} for image in ids}
        reduction={name:(mae[name]-mae['GEOMETRY'])/max(mae[name],1e-12) for name in ('STAGE','PHYSICAL')}
        wins=sum(v['GEOMETRY']<v['PHYSICAL']-1e-12 for v in per_image.values())
        reports[target]=dict(mae=mae,relative_mae_reduction=reduction,images_better_than_PHYSICAL=int(wins),per_image=per_image)
    count=sum(r['geometry'].get('usable',False) for r in rows);primary=reports['psnr']
    candidate=count>=24 and all(v>=.1 for v in primary['relative_mae_reduction'].values()) and primary['images_better_than_PHYSICAL']>=4
    return dict(classification='DEVELOPMENT_LOCAL_GEOMETRY_INFORMATION_CANDIDATE' if candidate else 'NO_CONFIRMED_LOCAL_GEOMETRY_INFORMATION',usable_states=int(count),total_states=30,image_clusters=len(ids),targets=reports,predictions=predictions,
        limits='Five reused development images; fold excludes same image. Fixed1NN diagnostic, not an online policy. Unresolved geometry falls back to physical. Finite directional response is not a global Hessian bound or curvature-control benefit.')
