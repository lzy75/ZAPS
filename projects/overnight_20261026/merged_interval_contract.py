"""B0 zero-network algebra: merge original DDPM anchors and learned linear doses.

Synthetic float64 only. No Torch/core/UNet parity, policy, quality or convergence
claim. The residual/denoised mean are held fixed, while anchor maps may differ.
"""
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import numpy as np


def kernel(alpha_t,alpha_s):
    a,b=float(alpha_t),float(alpha_s)
    if not 0<a<b<=1:raise ValueError('Need increasing cumulative alpha')
    ratio=a/b
    return (math.sqrt(ratio)*(1-b)/(1-a),
            math.sqrt(b)*(1-ratio)/(1-a),
            (1-ratio)*(1-b)/(1-a))


def merge(alphas,maps,draws):
    """Return Ax+Bm+G*r+noise with original fine innovations, not new RNG."""
    if len(alphas)!=len(maps)+1 or len(maps)!=len(draws) or not len(maps):
        raise ValueError('Need matching interval maps and innovations')
    A,B,variance=1.,0.,0.
    G=np.zeros_like(maps[0],dtype=np.float64);noise=np.zeros_like(draws[0],dtype=np.float64)
    for a,b,M,z in zip(alphas[:-1],alphas[1:],maps,draws):
        c2,c1,var=kernel(a,b)
        A,B,variance=c2*A,c2*B+c1,c2*c2*variance+var
        G=c2*G+M;noise=c2*noise+math.sqrt(var)*z
    return dict(A=A,B=B,G=G,noise=noise,variance=variance)


def main():
    random=np.random.default_rng(20261026)
    ab=np.cumprod(1-np.linspace(.0001,.02,1000)).astype(np.float32).astype(np.float64)
    grid=[999,989,978,967,956,944,931,917,903,888,872,854,835,815,793,769,742,712,679,641,597,546,484,409,319,218,124,54,14,0]
    alpha=np.array([ab[t] for t in grid]+[1.])
    errors=dict(A=[],B=[],variance=[],endpoint=[]);naive=[];K_defects=[]
    for _ in range(2000):
        start=int(random.integers(0,len(alpha)-1));size=int(random.integers(1,min(8,len(alpha)-1-start)+1))
        anchors=alpha[start:start+size+1]
        matrices=random.normal(size=(size,4,3));draws=random.normal(size=(size,4))
        x,m,r=random.normal(size=4),random.normal(size=4),random.normal(size=3)
        current=x.copy()
        for a,b,M,z in zip(anchors[:-1],anchors[1:],matrices,draws):
            c2,c1,var=kernel(a,b);current=c2*current+c1*m+M@r+math.sqrt(var)*z
        item=merge(anchors,matrices,draws);c2,c1,var=kernel(anchors[0],anchors[-1])
        merged=c2*x+c1*m+item['G']@r+item['noise']
        errors['A'].append(abs(item['A']-c2));errors['B'].append(abs(item['B']-c1))
        errors['variance'].append(abs(item['variance']-var))
        errors['endpoint'].append(float(np.linalg.norm(merged-current)/max(1.,np.linalg.norm(current))))
        if size>1:
            wrong=c2*x+c1*m+matrices[0]@r+item['noise']
            naive.append(float(np.linalg.norm(wrong-current)/max(1.,np.linalg.norm(current))))
            coefficients=[(1-a/b)/math.sqrt(a/b) for a,b in zip(anchors[:-1],anchors[1:])]
            fine=merge(anchors,[np.array([[v]]) for v in coefficients],[np.zeros(1) for _ in coefficients])['G'][0,0]
            coarse=(1-anchors[0]/anchors[-1])/math.sqrt(anchors[0]/anchors[-1])
            K_defects.append(abs(fine-coarse)/max(abs(coarse),1e-30))
    maximum={k:max(v) for k,v in errors.items()}
    passed=all(maximum[k]<=1e-11 for k in maximum)
    if not passed:raise RuntimeError('Synthetic composition contract failed')
    result=dict(checked_utc=datetime.now(timezone.utc).isoformat(),status='PASS_SYNTHETIC_FLOAT64_MERGED_ANCHOR_CONTRACT',
        trials=2000,seed=20261026,max_errors=maximum,nontrivial_blocks=len(naive),
        naive_single_anchor_relative_defect_median=float(np.median(naive)),
        frozen_score_K_nonsemigroup_relative_defect_median=float(np.median(K_defects)),
        method_nfe=0,diagnostic_nfe=0,
        assumptions=['Synthetic fixed clean mean and residual;arbitrary different linear anchor maps',
                     'Original independent fine innovations reused;no new draw after selecting a block',
                     'NumPy-generated schedule rounded tofloat32 then scalarfloat64;not serveralpha provenance'],
        limits='Algebra only,not Torch/core float32 parity,real nonlinear network error,unbiased adaptive SDE,controller quality or NFE saving. A50 fixed-grid K mapping is not challenged by this artificial frozen-score test.')
    Path(__file__).with_name('merged_interval_contract.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
