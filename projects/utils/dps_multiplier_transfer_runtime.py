"""A75 wraps the original DPS conditioning; never changes its sampling/noise loop."""
from functools import lru_cache
from pathlib import Path
import importlib
import math
import types
from utils import noise_signal_shrink as policy
from utils import dps_multiplier_transfer as rules
from utils.diag_zaps_sampler_parity import load_dps_reference


@lru_cache(maxsize=1)
def reference(root):
    sampler,meta=load_dps_reference(Path(root))
    condition=importlib.import_module(sampler.__package__+'.condition_methods')
    return sampler,condition,meta


class PureOperator:
    def __init__(self,op):self.op=op
    def forward(self,data,**kwargs):return self.op.H(data)


def run(model,op,y,case,arm,config,dps_root,count,*,size=256,progress=None):
    import torch
    from utils.diag_learning_rate_ablation import set_seed
    if arm not in (*rules.ARMS,'NULL'):raise ValueError('Unknown A75 arm')
    dps,conditions,_=reference(str(Path(dps_root).resolve()))
    sampler=dps.create_sampler(**config)
    condition=conditions.get_conditioning_method('ps',operator=PureOperator(op),
        noiser=types.SimpleNamespace(__name__='gaussian'),scale=rules.SCALE)
    set_seed(case['seed']);initial=torch.randn((1,3,size,size),device=model.device)
    trace=[];before_cpu=torch.get_rng_state().clone()
    before_cuda=torch.cuda.get_rng_state_all() if str(model.device).startswith('cuda') else []

    def guided(x_t,measurement,x_prev,x_0_hat,**kwargs):
        step=len(trace);index=sampler.num_timesteps-1-step;t=sampler.timestep_map[index]
        count['input_vjp_attempts']+=1
        if arm=='DPS':
            result,norm=condition.conditioning(x_prev=x_prev,x_t=x_t,x_0_hat=x_0_hat,measurement=measurement,**kwargs)
            row=dict(gamma=1.,reason='ORIGINAL_DPS',residual_norm=float(norm.detach()))
        else:
            grad,norm=condition.grad_and_value(x_prev,x_0_hat,measurement,**kwargs)
            if not torch.isfinite(grad).all():raise RuntimeError('Nonfinite original DPS gradient')
            if arm=='NULL':
                row=dict(gamma=1.,reason='NULL')
            else:
                with torch.no_grad():
                    clean=x_0_hat.detach();q=-rules.SCALE*grad.detach()
                    residual=(measurement-op.H(clean)).double()
                    a=math.sqrt(float(sampler.alphas_cumprod[index]))
                    d=(op.H(clean+q/a)-op.H(clean)).double()
                    row=policy.select(float((residual*d).sum()),float(residual.square().sum()),float(d.square().sum()),residual.numel(),case['sigma'])
                    if arm=='PROJECT':
                        row['gamma']=max(0.,row.get('residual_response_dot',0.))/row['response_energy'] if row.get('response_energy',0.)>1e-24 else 0.
                        row['reason']='NO_NOISE_PROJECTION'
            gamma=row['gamma']
            # Match official native arithmetic exactly when gamma=1.
            result=x_t-grad*rules.SCALE if gamma==1. else x_t-gamma*rules.SCALE*grad
        if not torch.isfinite(result).all():raise RuntimeError('Nonfinite A75 trajectory; no automatic fallback')
        trace.append(dict(step=step,t=t,**row))
        if progress and len(trace)%50==0:progress(len(trace),count)
        return result,norm

    with torch.enable_grad():
        output=sampler.p_sample_loop(model=model.model,x_start=initial,measurement=y,
            measurement_cond_fn=guided,record=False,save_root='').detach().cpu()
    if len(trace)!=sampler.num_timesteps:raise ValueError('Incomplete original DPS loop')
    return output,dict(trace=trace,initial=initial.detach().cpu(),cpu_rng=before_cpu,
        cuda_rng=before_cuda,grid=sampler.timestep_map[::-1])
