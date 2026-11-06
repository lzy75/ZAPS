"""A96 cached-vector factorial replay. No controller fitting or GT-based actions."""
import math
ARMS=('SCALAR_REPLAY','DIAG_REPLAY','DIAG_DIR_SCALAR_LEN','SCALAR_DIR_DIAG_LEN','ZERO')
STAGES=(16,24,29)


def length(x):
    try:value=float((x.detach().double()**2).sum())
    except AttributeError:
        import numpy as np
        value=float(np.sum(np.asarray(x,dtype=float)**2))
    if not math.isfinite(value):raise ValueError('Nonfinite vector')
    return math.sqrt(value)


def correction(native,scalar,diagonal,arm):
    if arm not in ARMS:raise ValueError('Unknown attribution arm')
    if arm=='SCALAR_REPLAY':return scalar
    if arm=='DIAG_REPLAY':return diagonal
    if arm=='ZERO':return native*0
    sn,dn,nn=length(scalar),length(diagonal),length(native)
    if nn==0:raise ValueError('Zero native ray; no direction substitution')
    if arm=='SCALAR_DIR_DIAG_LEN':return native*(dn/nn)
    if dn==0:
        if sn==0:return diagonal*0
        raise ValueError('Zero diagonal direction with nonzero requested length')
    return diagonal*(sn/dn)


def continue_run(model,z,y,state,delta):
    import torch
    from utils import overall_direction_audit as old
    old.restore_rng(state,model.device);x=state['uncond'].to(model.device)+delta;snapshots={}
    with torch.no_grad():
        for i in range(state['next_index'],-1,-1):
            uncond,q,clean=old.native_parts(model,z,x,y,i)
            k=29-i
            if k in STAGES:snapshots[str(k)]=dict(t=int(z.tau[i]),clean=clean.detach().cpu())
            x=uncond+q
            if not bool(torch.isfinite(x).all()):raise RuntimeError('Nonfinite attribution continuation')
    if set(snapshots)!=set(map(str,STAGES)):raise RuntimeError('Missing fixed trajectory checkpoint')
    return x.detach().cpu(),snapshots,old.rng_state(model.device)


def wavelet_risk(image,gt,dwt):
    import torch
    from utils.directional_reliability import subbands
    with torch.no_grad():
        error=image.detach().cpu().double()-gt.detach().cpu().double();coeff=dwt.analysis(error);n=error.numel()
        bands={name:float(coeff[...,ys,xs].square().sum())/n for name,(ys,xs) in subbands(image.shape[-2],image.shape[-1],dwt.level)}
        raw=float(error.square().mean());total=sum(bands.values())
        if abs(total-raw)>1e-5*max(raw,1e-12):raise RuntimeError('Orthogonal wavelet error decomposition failed')
        ll=next(iter(bands));clamped=image.detach().cpu().clamp(-1,1)
        return dict(raw_mse=raw,clamped_mse=float((clamped.double()-gt.cpu().double()).square().mean()),
            clamp_fraction=float((image.detach().cpu().abs()>1).float().mean()),
            band_mse_contributions=bands,coarse_LL3_mse=bands[ll],detail_mse=sum(v for k,v in bands.items() if k!=ll),
            scope='Orthogonal image-error energy contributions, not H-nullspace or independent risk estimates')


def factorial(values):
    s=values['SCALAR_REPLAY'];d=values['DIAG_REPLAY'];ds=values['DIAG_DIR_SCALAR_LEN'];sd=values['SCALAR_DIR_DIAG_LEN']
    return dict(direction_at_scalar_length=ds-s,direction_at_diagonal_length=d-sd,
        length_on_scalar_ray=sd-s,length_on_diagonal_ray=d-ds,interaction=d-ds-sd+s)
