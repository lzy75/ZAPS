"""A65 model-coordinate response and independent full pipelines. Old sources untouched."""
from contextlib import contextmanager
from utils import historical_ffhq_gain as rules
from utils import paper_clock_runtime as common


@contextmanager
def precision(arm):
    import torch
    from utils.diag_clean_response_precision import precision_state,put,NEW_FLAGS
    saved=precision_state(torch)
    flags=(torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic)
    try:
        if saved['api']=='legacy':
            torch.set_float32_matmul_precision('highest')
            torch.backends.cudnn.allow_tf32=arm=='HISTORICAL_NATIVE'
        else:
            for path in NEW_FLAGS:put(torch,path,'ieee')
            if arm=='HISTORICAL_NATIVE':put(torch,'backends.cudnn.conv.fp32_precision','tf32')
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=arm!='HISTORICAL_NATIVE'
        yield dict(**precision_state(torch),benchmark=False,deterministic=torch.backends.cudnn.deterministic)
    finally:
        torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic=flags
        if saved['api']=='legacy':
            torch.set_float32_matmul_precision(saved['matmul_precision'])
            torch.backends.cudnn.allow_tf32=saved['cudnn']
        else:
            for path,value in saved['flags'].items():put(torch,path,value)
        rules.require(precision_state(torch)==saved,'Precision context did not restore policy')


def sensor(z,y,x,t,next_t,positions,sigma,cosine,mask=None,before_backward=None):
    """L=H in historical model coordinates. Legacy T is NEVER passed as CG adjoint."""
    import torch
    from utils import current_score_runtime_r2 as base
    from utils import current_score_geometry as algebra
    from utils.angular_feedback_control import controlled_target
    rules.require(mask is None and cosine==1. and sigma==.05,'Frozen SR protocol changed')
    inp,raw=base.raw_mean(z.dm,x,t,graph=True)
    field=base.field_from_mean(z,y,raw,t,positions)
    with torch.no_grad():
        residual=y-z.A.H(raw.detach())
        row=dict(R_raw=base.informative_rms(residual)/sigma,
            R_clean=base.informative_rms(y-z.A.H(field['clean']))/sigma,
            clip_fraction=float((raw.detach().abs()>1).float().mean()),backward_calls=0,
            linear_units='L=H model-domain; no .5 factor',solver_adjoint='true raw Resizer transpose',
            fallback_mapping='historical bicubic empirical map')
    v,gate=algebra.proximal_vector(z.A.base.H,z.A.base.transpose,residual,float(z.dm.alphas_cumprod[t]),
                                  sigma,rtol=base.CG_RTOL,maxiter=base.CG_MAXITER)
    row['cg']=gate
    if v is None:
        row.update(resolved=False,reason='CG_UNRESOLVED_FALLBACK');return field,1.,row
    if before_backward:before_backward()
    row['backward_calls']=1
    u=algebra.input_vjp(raw,inp,v)
    alpha_s=float(z.dm.alphas_cumprod[next_t]) if next_t>=0 else 1.
    coefficient=algebra.skip_coefficient(float(z.dm.alphas_cumprod[t]),alpha_s)
    reading=base.gram_readout(field['q'],v,u,coefficient,float(x.detach().double().square().mean().sqrt()))
    row.update(reading,coefficient=coefficient)
    if not reading['resolved']:return field,1.,row
    target,extra=controlled_target(field['q'],v,u,coefficient,reading,1.)
    row.update(extra)
    return dict(field,q=target),1.,row


def fallback_sensor(z,y,x,t,next_t,positions,sigma,**kwargs):
    from utils.current_score_runtime_r2 import raw_mean,field_from_mean
    _,raw=raw_mean(z.dm,x,t,graph=False)
    return field_from_mean(z,y,raw,t,positions),1.,dict(resolved=False,reason='NULL_DIAGNOSTIC')


def make_operator(device,count=None,size=256,mode='legacy_bicubic'):
    from modules.degradations import get_operator
    from utils.diag_ffhq_regression import TransposeMode
    base=get_operator('super_resolution',device=device,scale_factor=4,noise_sigma=0.,in_shape=(1,3,size,size))
    if count is not None:common.count_operator(base,count)
    operator=TransposeMode(base,mode).to(device)
    if count is not None:
        original=operator.transpose
        count['guidance_mapping_actions']=0
        def mapping(value,output_size=None):
            count['guidance_mapping_actions']+=1
            return original(value,output_size=output_size)
        operator.transpose=mapping
    return operator


def execute(model,operator,y,seed,arm,count,*,size=256,level=3,sensed=None):
    import torch
    from modules.zaps_algorithm import ZAPS
    from utils.diag_learning_rate_ablation import set_seed
    from utils.no_learning_replay import noise_from_seed
    rules.require(arm in rules.ARMS,'Unknown frozen arm')
    set_seed(seed)
    cfg=rules.config();cfg.update(img_size=size,level=level,num_epochs=rules.UPDATES[arm])
    cls=common.snapshot_class() if rules.UPDATES[arm]==10 else ZAPS
    z=cls(model,operator,**cfg)
    rules.require(z.tau.detach().cpu().tolist()==rules.grid(),'Actual 9/17 grid mismatch')
    if rules.UPDATES[arm]==10:
        losses=z.optimize(y,verbose=False,x0_gt=None)
        result,entry,trace=z._last_opt_x0,z.before_final,[]
    elif arm=='RESPONSE_300':
        losses=z.optimize(y,verbose=False,x0_gt=None)
        initial=z._last_opt_noise;entry=common.snapshot(z,initial)
        innovations=[torch.randn_like(initial) for _ in range(29)]
        result,trace=common.response(z,y,initial,innovations,None,count,sensed or sensor)
    else:
        losses=[]
        initial,innovations,before=noise_from_seed(seed,model.device,(1,3,size,size),10)
        entry=common.snapshot(z,initial);entry['cpu_rng'],entry['cuda_rng']=before['cpu'],before['cuda']
        result,trace=common.response(z,y,initial,innovations,None,count,sensed or sensor)
    rules.require(bool(torch.isfinite(result).all()),'Nonfinite reconstruction')
    return result.detach().cpu(),dict(snapshot=entry,trace=trace,losses=losses)


def validate_counts(count,arm):
    rules.require(count['forward_attempts']==count['forwards']==rules.NFE[arm],'Forward count differs')
    rules.require(count['calls']==rules.grid()[::-1]*(1 if arm=='RESPONSE_30' else 10),'Model-call grid differs')
    rules.require(count['optimizer_backward_attempts']==count['optimizer_backward_completed']==rules.UPDATES[arm],
                  'Optimizer backward count differs')
    rules.require(count['input_vjp_attempts']<=(30 if arm.startswith('RESPONSE') else 0),'Unexpected input VJP count')
