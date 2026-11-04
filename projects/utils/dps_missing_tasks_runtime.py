"""A88 original DPS loop, official task scales, observed-only mask statistics."""
from functools import lru_cache
from pathlib import Path
import importlib
import math
import types
from utils import noise_signal_shrink as policy
from utils import dps_missing_tasks as rules
from utils.shrink_multitask_runtime import informative
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
    if arm not in (*rules.ARMS,'NULL'):raise ValueError('Unknown A88 arm')
    scale=rules.SCALES[case['task']]
    if case.get('scale',scale)!=scale:raise ValueError('Task PS scale changed')
    dps,conditions,_=reference(str(Path(dps_root).resolve()))
    sampler=dps.create_sampler(**config)
    condition=conditions.get_conditioning_method('ps',operator=PureOperator(op),
        noiser=types.SimpleNamespace(__name__='gaussian'),scale=scale)
    set_seed(case['seed']);initial=torch.randn((1,3,size,size),device=model.device)
    trace=[];before_cpu=torch.get_rng_state().clone()
    before_cuda=torch.cuda.get_rng_state_all() if str(model.device).startswith('cuda') else []
    count['module_H_actions']=0

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
                    clean=x_0_hat.detach();q=-scale*grad.detach()
                    count['module_H_actions']+=3
                    residual=informative((measurement-op.H(clean)).double(),op)
                    a=math.sqrt(float(sampler.alphas_cumprod[index]))
                    d=informative((op.H(clean+q/a)-op.H(clean)).double(),op)
                    row=policy.select(float((residual*d).sum()),float(residual.square().sum()),float(d.square().sum()),residual.numel(),case['sigma'])
                    row['active_measurements']=residual.numel()

            gamma=row['gamma']
            # Match official native arithmetic exactly when gamma=1.
            result=x_t-grad*scale if gamma==1. else x_t-gamma*scale*grad
        if not torch.isfinite(result).all():raise RuntimeError('Nonfinite A88 trajectory; no automatic fallback')
        trace.append(dict(step=step,t=t,**row))
        if progress and len(trace)%50==0:progress(len(trace),count)
        return result,norm

    with torch.enable_grad():
        output=sampler.p_sample_loop(model=model.model,x_start=initial,measurement=y,
            measurement_cond_fn=guided,record=False,save_root='').detach().cpu()
    if len(trace)!=sampler.num_timesteps:raise ValueError('Incomplete original DPS loop')
    return output,dict(trace=trace,grid=sampler.timestep_map[::-1],pairing=dict(
        initial_sha256=tensor_digest(initial),initial_cpu_rng=tensor_digest(before_cpu),
        initial_cuda_rng=[tensor_digest(v) for v in before_cuda],
        final_cpu_rng=tensor_digest(torch.get_rng_state()),
        final_cuda_rng=[tensor_digest(v) for v in torch.cuda.get_rng_state_all()] if before_cuda else [],
        grid=sampler.timestep_map[::-1]))


def tensor_digest(value):
    import hashlib
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
