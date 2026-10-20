"""A3a algebraic preflight: coupled DDPM subdivision with matched frozen guidance.

NumPy only; no model/GT/trajectory loading and zero UNet NFE. This verifies a
specified interval mapping, not quality benefits or an adaptive sampler.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def coefficients(alpha_t, alpha_s):
    if not (0 < alpha_t < alpha_s < 1):
        raise ValueError('Need a nonterminal reverse interval with 0 < alpha_t < alpha_s < 1')
    beta = 1.-alpha_t/alpha_s
    return dict(a=float(np.sqrt(alpha_t/alpha_s)*(1-alpha_s)/(1-alpha_t)),
                b=float(np.sqrt(alpha_s)*beta/(1-alpha_t)),
                sigma=float(np.sqrt((1-alpha_s)*beta/(1-alpha_t))))


def interval(alpha_t, alpha_m, alpha_s):
    first=coefficients(alpha_t,alpha_m);second=coefficients(alpha_m,alpha_s)
    coarse=coefficients(alpha_t,alpha_s)
    w1=second['a']*first['sigma']/coarse['sigma']
    w2=second['sigma']/coarse['sigma']
    return dict(first=first,second=second,coarse=coarse,w1=w1,w2=w2,
                local_guidance_factor=1./(1.+second['a']))


def coupled_noises(z, auxiliary, contract):
    if z.shape != auxiliary.shape:raise ValueError('Noise tensor shapes differ')
    w1,w2=contract['w1'],contract['w2']
    return w1*z+w2*auxiliary,w2*z-w1*auxiliary


def frozen_pair(x, clean, q, z, auxiliary, contract):
    c,f,s=(contract[k] for k in ('coarse','first','second'))
    z1,z2=coupled_noises(z,auxiliary,contract)
    dose=contract['local_guidance_factor']
    coarse=c['a']*x+c['b']*clean+q+c['sigma']*z
    mid=f['a']*x+f['b']*clean+dose*q+f['sigma']*z1
    fine=s['a']*mid+s['b']*clean+dose*q+s['sigma']*z2
    return coarse,fine


def preflight():
    # Same declared beta endpoints/count and float32 alpha storage as local core;
    # exact Torch/core parity remains a separate server check.
    alpha=np.cumprod(1.-np.linspace(1e-4,.02,1000,dtype=np.float64)).astype(np.float32).astype(np.float64)
    logs=.5*np.log((1-alpha)/alpha)
    rng=np.random.default_rng(20261020)
    rows=[]
    for t,s in ((903,888),(641,597),(319,218),(54,14),(14,0)):
        candidates=np.arange(s+1,t)
        m=int(candidates[np.argmin(np.abs(logs[candidates]-.5*(logs[t]+logs[s])))])
        c=interval(alpha[t],alpha[m],alpha[s])
        x,clean,q,z,auxiliary=rng.normal(size=(5,3,32,32))
        coarse,fine=frozen_pair(x,clean,q,z,auxiliary,c)
        f,h,k=(c[n] for n in ('first','second','coarse'))
        identity_errors=dict(mean_x=abs(h['a']*f['a']-k['a']),
                             mean_clean=abs(h['a']*f['b']+h['b']-k['b']),
                             variance=abs((h['a']*f['sigma'])**2+h['sigma']**2-k['sigma']**2),
                             normal_rotation=abs(c['w1']**2+c['w2']**2-1.),
                             transported_guidance=abs((1+h['a'])*c['local_guidance_factor']-1.),
                             frozen_endpoint_max_abs=float(np.max(np.abs(coarse-fine))))
        rows.append(dict(t=t,m=m,s=s,local_guidance_factor=c['local_guidance_factor'],
                         naive_half_guidance_endpoint_factor=.5*(1+h['a']),
                         incorrectly_reused_noise_variance_ratio=(h['a']*f['sigma']+h['sigma'])**2/k['sigma']**2,
                         errors=identity_errors))
    passed=all(max(r['errors'].values()) < 1e-11 for r in rows)
    if not passed:raise RuntimeError('Frozen-field contract failed')
    project=Path(__file__).resolve().parents[1]
    files=[Path(__file__),project/'modules/zaps_algorithm.py',project/'modules/diffusion_model.py']
    return dict(experiment='A3a_numpy_algebra_preflight',classification='PASS_ALGEBRA_ONLY',method_nfe=0,diagnostic_nfe=0,
                model_loaded=False,gt_loaded=False,rows=rows,
                implementation_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                limits='NumPy float64 algebra with declared float32 alpha storage. Not Torch/CUDA/core execution, parent-artifact parity, neural refinement benefit or state-control evidence.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',help='New JSON file only; existing evidence is never overwritten')
    args=parser.parse_args();result=preflight()
    if args.output:
        p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x') as stream:stream.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
