"""Two algebraically equal FFT forms; the official byte-pinned solver remains unmodified."""
from unittest.mock import patch
from utils import diffpir_multiplier_runtime as original
reference=original.reference
schedule=original.schedule
CircularBlur=original.CircularBlur


def stable_solution(x,FB,FBC,F2B,FBFy,alpha,sf):
    import torch
    if sf!=1:raise ValueError('Stable simplification applies only to sf=1 circular deblurring')
    # Upstream Woodbury expression: (FR - |FB|²*FR/(|FB|²+alpha))/alpha.
    # Equivalent sf1 expression avoids subtracting almost equal numbers at alpha~1e-6.
    FR=FBFy+torch.fft.fftn(alpha*x,dim=(-2,-1))
    return torch.fft.ifftn(FR/(F2B+alpha),dim=(-2,-1)).real


def run(model,y,kernel,case,arm,count,*,steps=100,progress=None):
    if arm in ('DIFFPIR','SHRINK'):
        out,detail=original.run(model,y,kernel,case,arm,count,steps=steps,progress=progress)
        variant='OFFICIAL_NATIVE_FLOAT32'
    elif arm in ('STABLE_DIFFPIR','STABLE_SHRINK','STABLE_ONES'):
        mapped={'STABLE_DIFFPIR':'DIFFPIR','STABLE_SHRINK':'SHRINK','STABLE_ONES':'ONES'}[arm]
        _,sr,_,_=reference()
        with patch.object(sr,'data_solution',stable_solution):
            out,detail=original.run(model,y,kernel,case,mapped,count,steps=steps,progress=progress)
        variant='STABLE_EQUIVALENT_FLOAT32_SF1'
    else:raise ValueError('Unknown A79R1 arm')
    for row in detail['trace']:row['solver_variant']=variant
    return out,detail
