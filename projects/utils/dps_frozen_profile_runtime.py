"""Reuse the exact A75 loop; override only its scalar decision with a frozen profile."""
from pathlib import Path
from unittest.mock import patch
from utils import dps_multiplier_transfer_runtime as original
from utils.historical_ffhq_gain import read,digest,require
reference=original.reference
PureOperator=original.PureOperator

def run(model,op,y,case,arm,config,dps_root,count,*,size=256,progress=None):
    path=Path(case['profiles'][arm]['path'])
    require(digest(path)==case['profiles'][arm]['sha256'],'Frozen profile changed')
    return run_profile(model,op,y,case,read(path)['gamma'],config,dps_root,count,size=size,progress=progress)

def run_profile(model,op,y,case,curve,config,dps_root,count,*,size=256,progress=None):
    select=original.policy.select;position=0
    def fixed(dot,r2,d2,n,sigma):
        nonlocal position
        if position>=len(curve):raise ValueError('Profile shorter than trajectory')
        row=select(dot,r2,d2,n,sigma);gamma=float(curve[position]);position+=1
        row.update(gamma=gamma,reason='FROZEN_PROFILE',current_state_used_for_gamma=False,
                   predicted_residual=max(0.,r2-2*gamma*dot+gamma*gamma*d2))
        return row
    with patch.object(original.policy,'select',fixed):
        output,detail=original.run(model,op,y,case,'SHRINK',config,dps_root,count,size=size,progress=progress)
    require(position==len(curve),'Profile longer than trajectory')
    return output,detail
