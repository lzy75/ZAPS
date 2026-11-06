"""A93 representation upper bounds and fixed three-update feedback window."""
import numpy as np
from utils import overall_direction_audit as old
ARMS=('DIAG_ALIGN_3','PHYS_DIRECT_3')
REFERENCES={'DIAG_ALIGN_3':'DIAG_ALIGN_100','PHYS_DIRECT_3':'PHYS_DIRECT_100'}


class UnresolvedDirection(RuntimeError):pass


def coefficient_geometry(q,g,bands):
    """Formal unbounded diagonal expressivity, not an executable gain policy."""
    q=np.asarray(q,dtype=np.float64);g=np.asarray(g,dtype=np.float64)
    if q.shape!=g.shape or not np.isfinite(q).all() or not np.isfinite(g).all():raise ValueError('Finite matched coefficients required')
    norm=float(np.linalg.norm(g.ravel()))
    if norm==0:raise ValueError('Zero reference')
    positive=np.zeros_like(q);signed=np.zeros_like(q)
    for _,(ys,xs) in bands:
        s=(...,ys,xs);a,b=q[s],g[s];energy=float(np.sum(a*a));dot=float(np.sum(a*b))
        if energy>0:positive[s]=a*max(0.,dot/energy);signed[s]=a*(dot/energy)
    valid=q!=0
    coefficient_positive=np.where(valid & (q*g>0),g,0.)
    coefficient_signed=np.where(valid,g,0.)
    def cosine(v):
        n=float(np.linalg.norm(v.ravel()))
        return 0. if n==0 else float(np.sum(v*g))/(n*norm)
    ratios=np.abs(g[valid]/q[valid]);rms=float(np.sqrt(np.mean(q*q)))
    tiny=np.abs(q)<=32*np.finfo(np.float32).eps*rms
    result=dict(cosine_band_nonnegative=cosine(positive),cosine_band_signed=cosine(signed),
        cosine_coefficient_nonnegative=cosine(coefficient_positive),cosine_coefficient_signed=cosine(coefficient_signed),
        reference_energy_opposite_sign_fraction=float(np.sum(g[(q*g)<0]**2)/norm**2),
        reference_energy_zero_source_fraction=float(np.sum(g[~valid]**2)/norm**2),
        reference_energy_tiny_source_fraction=float(np.sum(g[tiny]**2)/norm**2),
        formal_abs_gain_quantiles=dict(zip(('p50','p90','p99','max'),np.quantile(ratios,[.5,.9,.99,1.]).tolist())) if ratios.size else {},
        coefficient_count=q.size,zero_source_count=int(np.sum(~valid)),tiny_source_count=int(np.sum(tiny)),
        UNet=0,input_vjp=0,scope='Formal direction capacity only; signed/unbounded gains not applied to reconstruction')
    # Both enlargements contain the corresponding grouped solution.
    if result['cosine_band_signed']+1e-10<result['cosine_band_nonnegative'] or result['cosine_coefficient_nonnegative']+1e-10<result['cosine_band_nonnegative']:
        raise RuntimeError('Nested representation capacity check failed')
    return result


def geometry(bundle,dwt):
    import torch
    from utils.directional_reliability import subbands
    with torch.no_grad():
        q=bundle['state']['q'];g=bundle['physical']
        cq=dwt.analysis(q).double().cpu().numpy();cg=dwt.analysis(g).double().cpu().numpy()
        result=coefficient_geometry(cq,cg,subbands(q.shape[-2],q.shape[-1],dwt.level))
    expected=bundle['detail']['directions']['DIAG_ALIGN']['cosine_to_physical']
    if expected is None or abs(result['cosine_band_nonnegative']-expected)>1e-5:raise RuntimeError('A92 band projection identity changed')
    return dict(**result,dwt_analysis=2,source_cosine_band_nonnegative=expected)


def run(model,op,y,entry,bundle,task,arm,counts,size=256,level=3):
    import torch,time
    if arm not in ARMS:raise ValueError('Unregistered window arm')
    kind=arm.rsplit('_',1)[0];state=bundle['state'];z=old.make_z(model,op,entry,task,size,level)
    unit=bundle['units'][kind]
    if unit is None:raise UnresolvedDirection('A92 initial direction unresolved')
    old.restore_rng(state,model.device)
    qnorm=bundle['detail']['native_norm'];x=state['uncond'].to(model.device)+unit.to(model.device)*qnorm
    first=None;trace=[dict(window_update=0,t=state['current_t'],next_t=state['next_t'],
      native_norm=qnorm,reference_reused=True,**bundle['detail']['directions'][kind])]
    for k,i in enumerate(range(state['next_index'],-1,-1)):
        with torch.no_grad():uncond,q,clean=old.native_parts(model,z,x,y,i)
        if first is None:first=clean.detach().cpu()
        if k<2:
            probe=dict(uncond=uncond.detach(),next_t=int(z.tau[i-1]))
            start=time.perf_counter();physical,_=old.physical_reference(model,op,y,probe,counts)
            if str(model.device).startswith('cuda'):torch.cuda.synchronize()
            seconds=time.perf_counter()-start
            with torch.no_grad():
                units,detail=old.directions(q,physical,z.dwt)
                if units[kind] is None:raise UnresolvedDirection('Current window direction unresolved; no fallback')
                correction=units[kind]*detail['native_norm']
                ratio=float(correction.double().norm())/detail['native_norm']
                if abs(ratio-1)>3e-5:raise RuntimeError('Matched current-state length failed')
                trace.append(dict(window_update=k+1,t=int(z.tau[i]),next_t=probe['next_t'],
                  native_norm=detail['native_norm'],matched_length_ratio=ratio,reference_reused=False,
                  reference_seconds=seconds,**detail['directions'][kind]))
                x=uncond+correction
        else:
            with torch.no_grad():x=uncond+q
        if not bool(torch.isfinite(x).all()):raise RuntimeError('Nonfinite window trajectory')
    return x.detach().cpu(),first,old.rng_state(model.device),trace


def expected_calls(grid):
    main=list(grid[16:]);return [main[0],main[1],main[1],main[2]]+main[2:]
