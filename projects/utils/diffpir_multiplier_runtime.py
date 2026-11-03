"""Execute pinned upstream DiffPIR AST verbatim; hook its clean-image proximal correction."""
import ast
from functools import lru_cache
import importlib
from pathlib import Path
import sys
import types
from utils import diffpir_multiplier_transfer as rules
from utils import noise_signal_shrink as policy


def functions(path,names,namespace):
    source=Path(path).read_text();tree=ast.parse(source)
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    if {n.name for n in nodes}!=set(names):raise ValueError('Missing pinned upstream functions')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return types.SimpleNamespace(**{n:namespace[n] for n in names})


@lru_cache(maxsize=1)
def reference():
    import numpy as np
    import torch
    rules.verify_upstream()
    prefix='_zaps_a79_diffpir_reference'
    for suffix,path in (('',rules.VENDOR),('.guided_diffusion',rules.VENDOR/'guided_diffusion')):
        name=prefix+suffix
        if name not in sys.modules:
            module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module
    gd=importlib.import_module(prefix+'.guided_diffusion.gaussian_diffusion')
    sr=functions(rules.VENDOR/'utils/utils_sisr.py',('splits','p2o','upsample','downsample','data_solution','pre_calculate'),dict(torch=torch))
    utils_model=functions(rules.VENDOR/'utils/utils_model.py',('find_nearest','model_fn'),dict(np=np,torch=torch))
    tree=ast.parse((rules.VENDOR/'main_ddpir.py').read_text())
    loops=[n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='i' and ast.unparse(n.iter)=='range(len(seq))']
    if len(loops)!=1:raise ValueError('Pinned native sampling loop not uniquely located')
    loop=compile(ast.Module(body=loops,type_ignores=[]),str(rules.VENDOR/'main_ddpir.py'),'exec')
    return gd,sr,utils_model,loop


class CircularBlur:
    """Native periodic convolution, in [0,1]; not the old zero-boundary operator."""
    def __init__(self,kernel,shape,count=None):
        import torch
        _,sr,_,_=reference()
        self.otf=sr.p2o(kernel,shape);self.count=count
    def H(self,value):
        import torch
        if self.count is not None:self.count['base_H_actions']+=1
        return torch.fft.ifftn(self.otf*torch.fft.fftn(value,dim=(-2,-1)),dim=(-2,-1)).real
    def transpose(self,value):
        import torch
        if self.count is not None:self.count['base_adjoint_actions']+=1
        return torch.fft.ifftn(self.otf.conj()*torch.fft.fftn(value,dim=(-2,-1)),dim=(-2,-1)).real


def schedule(device,steps=100):
    import numpy as np
    import torch
    betas=torch.from_numpy(np.linspace(.0001,.02,1000,dtype=np.float32)).to(device)
    # Explicit numpy buffer preserves upstream float32 cumulative-product arithmetic without array-wrap API dependence.
    ab=torch.from_numpy(np.cumprod((1-betas).cpu().numpy(),axis=0))
    a=ab.sqrt();b=(1-ab).sqrt();sigma=b/a
    return dict(betas=betas,alphas_cumprod=ab,sqrt_alphas_cumprod=a,sqrt_1m_alphas_cumprod=b,
                reduced_alpha_cumprod=sigma,sigmas=torch.tensor([sigma[999-i] for i in range(1000)]).to(device),seq=rules.sequence(steps))


def run(model,y,kernel,case,arm,count,*,steps=100,progress=None):
    import numpy as np
    import torch
    from utils.diag_learning_rate_ablation import set_seed
    if arm not in (*rules.ARMS,'ONES'):raise ValueError('Unknown A79 arm')
    gd,sr,utils_model,loop=reference();sc=schedule(model.device,steps)
    config=types.SimpleNamespace(**rules.configuration(case,steps))
    config.device=model.device
    diffusion=gd.GaussianDiffusion(betas=gd.get_named_beta_schedule('linear',1000),model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.LEARNED_RANGE,loss_type=gd.LossType.MSE,rescale_timesteps=False)
    a,b=sc['sqrt_alphas_cumprod'],sc['sqrt_1m_alphas_cumprod'];sigma=b/a
    rhos=torch.tensor([config.lambda_*config.sigma**2/(sigma[i]**2) for i in range(1000)]).to(model.device)
    set_seed(case['seed'])
    x=a[999]*(2*y-1)+b[999]*torch.randn_like(y)
    initial=x.detach().cpu();initial_cpu_rng=torch.get_rng_state().clone()
    initial_cuda_rng=torch.cuda.get_rng_state_all() if str(model.device).startswith('cuda') else []
    FB,FBC,F2B,FBFy=sr.pre_calculate(y,kernel,1)
    op=None if arm=='DIFFPIR' else CircularBlur(kernel,y.shape[-2:],count);trace=[]
    def solution(clean,*args):
        count['native_data_solve_attempts']=count.get('native_data_solve_attempts',0)+1
        native=sr.data_solution(clean,*args);step=len(trace)
        row=dict(gamma=1.,reason='ORIGINAL_DIFFPIR',rho=float(rhos[999-sc['seq'][step]]))
        if arm!='DIFFPIR':
            r=(y-op.H(clean)).double();d=op.H(native-clean).double()
            row.update(policy.select(float((r*d).sum()),float(r.square().sum()),float(d.square().sum()),r.numel(),config.sigma))
            if arm=='ONES':row.update(gamma=1.,reason='GAMMA_ONE_EQUIVALENCE')
        gamma=row['gamma']
        value=native if gamma==1. else clean+gamma*(native-clean)
        if not torch.isfinite(value).all():raise RuntimeError('Nonfinite native clean correction; no fallback')
        trace.append(dict(step=step,t=999-sc['seq'][step],**row))
        count['native_data_solves_completed']=count.get('native_data_solves_completed',0)+1
        if progress and len(trace)%10==0:progress(len(trace),count)
        return value
    env=dict(np=np,torch=torch,utils_model=utils_model,config=config,model=model.model,diffusion=diffusion,
             model_out_type='pred_xstart',rhos=rhos,y=y,x=x,FB=FB,FBC=FBC,F2B=F2B,FBFy=FBFy,
             sr=types.SimpleNamespace(data_solution=solution),**sc)
    with torch.no_grad():exec(loop,env)
    # Upstream returns x_0 derived from x; its final pred_xstart is evaluated but not substituted.
    out=(2*env['x_0']-1).detach().cpu()
    if len(trace)!=steps-1:raise ValueError('Native DiffPIR proximal call count changed')
    if not torch.isfinite(out).all():raise RuntimeError('Nonfinite DiffPIR trajectory')
    return out,dict(trace=trace,initial=initial,cpu_rng=initial_cpu_rng,cuda_rng=initial_cuda_rng,
        final_cpu_rng=torch.get_rng_state().clone(),final_cuda_rng=torch.cuda.get_rng_state_all() if str(model.device).startswith('cuda') else [],
        grid=[999-v for v in sc['seq']],native_output='2*x_0-1 from unmodified upstream loop; last unused denoiser call retained')
