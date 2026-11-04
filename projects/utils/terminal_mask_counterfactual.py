"""A86 zero-UNet terminal interventions; no GT in output construction."""
import math

def parts(q,mask):return mask*q,(1-mask)*q

def interventions(m,q,mask):
    observed,missing=parts(q,mask)
    return dict(SHRINK=m,END_OBS_RESTORE=m+observed,END_MISSING_RESTORE=m+missing,END_BOTH_RESTORE=m+q)

def region_errors(output,gt,mask):
    import torch
    e=(output.detach().double()-gt.detach().double()).square()
    observed=mask.bool().expand_as(e);missing=~observed
    if not bool(observed.any()) or not bool(missing.any()):raise ValueError('Need observed and missing pixels')
    result=dict(mse_full=float(e.mean()),mse_observed=float(e[observed].mean()),mse_missing=float(e[missing].mean()),
                observed_fraction=float(observed.double().mean()))
    combined=result['observed_fraction']*result['mse_observed']+(1-result['observed_fraction'])*result['mse_missing']
    if abs(combined-result['mse_full'])>1e-12:raise ValueError('Region decomposition failed')
    for part in ('full','observed','missing'):
        result['float_psnr_'+part]=10*math.log10(4/max(result['mse_'+part],1e-30))
    return result

def reconstruct(m,y,mask,entry,alpha,dwt):
    import torch
    if int(entry['tau'][0])!=0 or not 0<float(alpha)<1:raise ValueError('Need t0 terminal state')
    if not torch.all((mask==0)|(mask==1)):raise ValueError('Binary saved mask required')
    v=mask*(y-mask*m)
    hv=dwt.synthesis(entry['D'][0]*dwt.analysis(v))
    q=entry['zeta'][0]*(v+(1-alpha)*hv)/alpha.sqrt().clamp(min=1e-8)
    if not torch.isfinite(q).all():raise ValueError('Nonfinite reconstructed original ZAPS correction')
    return q
