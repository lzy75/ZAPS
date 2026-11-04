"""Pinned native loops, a single scalar hook at the released correction expression."""
import ast
from contextlib import contextmanager
from functools import lru_cache, partial
import math
import os
import random
import sys
import types
from unittest.mock import patch
from utils import native_ray_transfer as rules
from utils import noise_signal_shrink as policy


class RayHook(ast.NodeTransformer):
    """Only the explicit data-correction expression changes; gamma1 is identity."""
    def __init__(self, host):
        self.host, self.matches = host, 0

    def visit_AugAssign(self, node):
        node = self.generic_visit(node)
        target = ast.unparse(node.target)
        selected = ((self.host == 'A83' and target == 'x_t' and isinstance(node.op, ast.Add)) or
                    (self.host == 'A82' and target == 'x_0_hat' and isinstance(node.op, ast.Sub)))
        if selected:
            self.matches += 1
            clean, alpha, sign = ('x_0_hat', 'alpha_bar', 1) if self.host == 'A83' else ('x_0_hat', '1.', -1)
            node.value = ast.Call(func=ast.Name(id='ray_hook', ctx=ast.Load()),
                args=[node.value, ast.parse(clean, mode='eval').body,
                      ast.parse(alpha, mode='eval').body, ast.Constant(sign)], keywords=[])
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if self.host == 'A84' and any(isinstance(v, ast.Name) and v.id == 'xs' for v in node.targets):
            class FinalCorrection(ast.NodeTransformer):
                def __init__(self, parent):self.parent = parent
                def visit_BinOp(self, term):
                    if ast.unparse(term) == 'grad_term * coeff':
                        self.parent.matches += 1
                        return ast.Call(func=ast.Name(id='ray_hook', ctx=ast.Load()),
                            args=[term, ast.Name(id='x0_pred', ctx=ast.Load()),
                                  ast.Name(id='alpha_t', ctx=ast.Load()), ast.Constant(1)], keywords=[])
                    return self.generic_visit(term)
            node.value = FinalCorrection(self).visit(node.value)
        return node


def definitions(path, env, names=None, hook=None):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and
             (names is None or n.name in names)]
    if names is not None and {n.name for n in nodes} != set(names):
        raise ValueError('Missing original definitions: '+str(path))
    tree = ast.Module(body=nodes, type_ignores=[])
    if hook:
        transform = RayHook(hook);tree = transform.visit(tree)
        if transform.matches != 1:
            raise ValueError(f'Expected ONE native correction insertion, got {transform.matches}: {path}')
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), 'exec'), env)
    return types.SimpleNamespace(**{n.name:env[n.name] for n in nodes})


def environment():
    import numpy as np
    import torch
    from abc import ABC, abstractmethod
    from tqdm.auto import tqdm
    return dict(np=np, torch=torch, math=math, os=os, partial=partial,
                nn=torch.nn, DictConfig=types.SimpleNamespace,
                ABC=ABC, abstractmethod=abstractmethod, tqdm=partial(tqdm, disable=True),
                __SAMPLER__={}, __MODEL_MEAN_PROCESSOR__={}, __MODEL_VAR_PROCESSOR__={},
                __CONDITIONING_METHOD__={})


@lru_cache(maxsize=6)
def reference(experiment, hooked=True):
    rules.verify(experiment)
    root = rules.vendor(experiment);env = environment()
    if experiment in ('A82', 'A83'):
        base = root/'linear_inv' if experiment == 'A82' else root
        definitions(base/'util/img_utils.py', env, ('dynamic_thresholding',))
        processor = definitions(base/'guided_diffusion/posterior_mean_variance.py', env)
        sampler_file = 'guided_gaussian_diffusion.py' if experiment == 'A82' else 'gaussian_diffusion.py'
        # Processor and sampler have separate extract helpers/registries in upstream.
        sampler_env = environment()
        sampler_env.update(get_mean_processor=processor.get_mean_processor, get_var_processor=processor.get_var_processor)
        sampler = definitions(base/'guided_diffusion'/sampler_file, sampler_env)
        cond_env = environment()
        if experiment == 'A82':
            condition = definitions(base/'guided_diffusion/condition_methods.py', cond_env,
                ('register_conditioning_method', 'ConditioningMethod', 'WoProjSampling'),
                experiment if hooked else None)
        else:
            condition = definitions(base/'guided_diffusion/condition_methods.py', cond_env,
                ('register_conditioning_method', 'ConditioningMethod', 'PosteriorSampling_meng'),
                experiment if hooked else None)
        return types.SimpleNamespace(sampler=sampler, condition=condition, hook_env=cond_env)
    # Unmodified author classes, excluding unrelated Hydra/classifier imports.
    diffusion = definitions(root/'models/diffusion.py', env, ('Diffusion',))
    models = definitions(root/'models/classifier_guidance_model.py', env, ('ClassifierGuidanceModel',))
    definitions(root/'algos/ddim.py', env, ('DDIM',))
    pgdm = definitions(root/'algos/pgdm.py', env, ('PGDM',), experiment if hooked else None)
    helpers=definitions(root/'utils/functions.py',env,('sigmoid','get_timesteps'))
    return types.SimpleNamespace(pgdm=pgdm, models=models, diffusion=diffusion, helpers=helpers, hook_env=env)


def seed(value):
    import numpy as np
    import torch
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(value)


def load_image(path, device='cpu'):
    from PIL import Image
    from torchvision import transforms
    image = Image.open(path).convert('RGB')
    if image.size != (256, 256):raise ValueError('Need existing256px image; no new resize protocol')
    return transforms.Normalize((.5,.5,.5),(.5,.5,.5))(transforms.ToTensor()(image)).unsqueeze(0).to(device)


def make_operator(experiment, task, device, size=256):
    import numpy as np
    import torch
    from torch.nn import functional as F
    root = rules.vendor(experiment);env = dict(torch=torch, np=np, F=F)
    if experiment == 'A82':
        import scipy
        import scipy.ndimage
        root = root/'linear_inv'
        resizer = definitions(root/'util/resizer.py', dict(torch=torch,np=np,nn=torch.nn,pi=math.pi))
        env.update(nn=torch.nn, ABC=__import__('abc').ABC, abstractmethod=__import__('abc').abstractmethod,
                   Resizer=resizer.Resizer, scipy=scipy, partial=partial, __OPERATOR__={})
        definitions(root/'util/img_utils.py',env,('Blurkernel',))
        ops = definitions(root/'guided_diffusion/measurements.py', env,
            ('register_operator','LinearOperator','SuperResolutionOperator','GaussialBlurOperator'))
        if task == 'super_resolution':return ops.SuperResolutionOperator(in_shape=(1,3,size,size),scale_factor=4,device=device)
        return ops.GaussialBlurOperator(kernel_size=61,intensity=3.,device=device)
    path = root/'util/functions.py' if experiment == 'A83' else root/'utils/degredations.py'
    env.update(DictConfig=types.SimpleNamespace,os=os)
    names=('H_functions','SRConv','Deblurring') if experiment=='A83' else ('H_functions','SRConv','Deblurring','SuperResolution')
    ops = definitions(path, env, names)
    if experiment=='A84':
        definitions(path,env,('pdf','build_one_degredation_model','build_degredation_model'))
        cfg=types.SimpleNamespace(algo=types.SimpleNamespace(**rules.configuration(experiment,task)['algo']),
                                 dataset=types.SimpleNamespace(image_size=size,channels=3))
        return env['build_degredation_model'](cfg)
    if task == 'super_resolution':
        # Original DMPS main.py / RED-diff build_degredation_model bicubic kernel.
        factor=4
        def bicubic(x,a=-.5):
            if abs(x)<=1:return (a+2)*abs(x)**3-(a+3)*abs(x)**2+1
            if abs(x)<2:return a*abs(x)**3-5*a*abs(x)**2+8*a*abs(x)-4*a
            return 0.
        k=np.array([bicubic(i/factor-np.floor(factor*4/2)/factor+.5/factor) for i in range(factor*4)])
        kernel=torch.from_numpy(k/k.sum()).float().to(device)
        return ops.SRConv(kernel/kernel.sum(),3,size,device,stride=factor)
    # DMPS builds a31-tap separable kernel for its configured61 Gaussian;
    # RED-diff uses its own native helper, resolved below without substitution.
    if experiment == 'A83':
        values=torch.linspace(-15,15,steps=31)
        kernel=torch.stack([torch.exp(torch.tensor([-.5*(x/3.)**2])) for x in values]).flatten().to(device)
        return ops.Deblurring(kernel/kernel.sum(),3,size,device)
    raise ValueError('Unknown host operator')


class Operator:
    def __init__(self, base, count=None):self.base,self.count=base,count;self.module=False
    def H(self, value):
        if self.count is not None:
            key='module_H_actions' if self.module else 'base_H_actions'
            self.count[key]=self.count.get(key,0)+1
        return self.base.forward(value) if hasattr(self.base,'forward') else self.base.H(value)
    def forward(self, value, **kwargs):return self.H(value)
    def H_pinv(self, value):return self.base.H_pinv(value)
    def __getattr__(self, name):return getattr(self.base,name)


def state(op):
    import torch
    result={}
    def collect(value,name):
        if isinstance(value,torch.Tensor):result[name]=value.detach().cpu().clone()
        elif isinstance(value,torch.nn.Module):
            for k,v in value.state_dict().items():result[name+'.'+k]=v.detach().cpu().clone()
    for name,value in vars(op.base if isinstance(op,Operator) else op).items():collect(value,name)
    return result


def rng():
    import torch
    return dict(cpu=torch.get_rng_state().clone(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def numpy_cuda_index_bridge(array):
    """Only transport a CUDA integer index to NumPy; preserve sourcefloat64 values."""
    import numpy as np
    import torch
    class CompatibleArray(np.ndarray):
        def __getitem__(self,index):
            if isinstance(index,torch.Tensor):index=index.detach().cpu().numpy()
            return super().__getitem__(index)
    return np.asarray(array).view(CompatibleArray)


@contextmanager
def counters(network):
    import torch
    count=dict(forward_attempts=0,forwards=0,input_vjp_attempts=0,operator_gradient_attempts=0,
               optimizer_backward_attempts=0,base_H_actions=0,module_H_actions=0,corrections=0,calls=[])
    def before(module,args):count['forward_attempts']+=1;count['calls'].append(float(args[1].reshape(-1)[0]))
    def after(module,args,out):count['forwards']+=1
    def backward(*args,**kwargs):
        count['optimizer_backward_attempts']+=1
        raise RuntimeError('A82-A84 forbid optimizer/parameter backwards')
    pre=network.register_forward_pre_hook(before);post=network.register_forward_hook(after)
    try:
        with patch.object(torch.Tensor,'backward',backward):yield count
    finally:pre.remove();post.remove()


class TorchWithDerivativeCount:
    """Observe the author's explicit derivative call, not checkpoint internals."""
    def __init__(self,torch,count,key):
        self.torch=torch
        class Grad:
            def grad(self,*args,**kwargs):
                count[key]+=1
                return torch.autograd.grad(*args,**kwargs)
            def __getattr__(self,name):return getattr(torch.autograd,name)
        self.autograd=Grad()
    def __getattr__(self,name):return getattr(self.torch,name)


def run(experiment, model, op, y, case, arm, count, *, progress=None, size=256, test_steps=None, native=False):
    import numpy as np
    import torch
    if arm not in ('BASE','SHRINK','ONES'):raise ValueError('Unknown arm')
    ref=reference(experiment,not native);cfg=rules.configuration(experiment,case['task']);trace=[]
    ref.hook_env['torch']=TorchWithDerivativeCount(torch,count,
        'input_vjp_attempts' if experiment=='A84' else 'operator_gradient_attempts')
    seed(case['seed']);initial=torch.randn((1,3,size,size),device=model.device);entry=rng()
    original_shape=y.shape
    def hook(value,clean,alpha,sign):
        row=dict(gamma=1.,reason='ORIGINAL_HOST')
        if arm!='BASE':
            with torch.no_grad():
                m=clean.detach();q=sign*value.detach()
                a=float(alpha.reshape(-1)[0]) if hasattr(alpha,'reshape') else float(alpha)
                op.module=True
                try:
                    # No next-network response claim: same-time identity proxy outside clean MPGD.
                    r=(y-op.H(m).reshape(original_shape)).double()
                    d=op.H(q if experiment=='A82' else q/math.sqrt(a)).reshape(original_shape).double()
                finally:op.module=False
                row=policy.select(float((r*d).sum()),float(r.square().sum()),float(d.square().sum()),r.numel(),case['sigma'])
            if arm=='ONES':row.update(gamma=1.,reason='FORCED_ONE_EQUIVALENCE')
        count['corrections']+=1
        trace.append(dict(step=len(trace),**row))
        if progress and len(trace)%10==0:progress(len(trace),count)
        if not torch.isfinite(value).all():raise RuntimeError('Nonfinite native data correction')
        return value if row['gamma']==1. else row['gamma']*value
    ref.hook_env['ray_hook']=hook
    if experiment in ('A82','A83'):
        diffusion=dict(cfg['diffusion'])
        if test_steps:diffusion['timestep_respacing']=test_steps
        sampler=ref.sampler.create_sampler(**diffusion)
        if experiment=='A83':
            # Released DMPS indexes NumPy buffers using a CUDA time tensor.
            # Bridge index transport for both arms without changing numbers or NN timestamps.
            for name in ('alphas','alphas_cumprod','alphas_cumprod_next'):
                setattr(sampler,name,numpy_cuda_index_bridge(getattr(sampler,name)))
        if experiment=='A82':
            cond=ref.condition.WoProjSampling(op,types.SimpleNamespace(__name__='gaussian'),scale=cfg['scale'])
            result=sampler.p_sample_loop(model.model,initial,y,cond.conditioning,False,'')
        else:
            cond=ref.condition.PosteriorSampling_meng(op,types.SimpleNamespace(__name__='gaussian'),scale=cfg['task']['conditioning']['params']['scale'])
            def conditioning(**kwargs):
                # Both coefficients are scalar schedule constants, not image tensors.
                # Avoid NumPy1-element-array * CUDA Tensor coercion in released code.
                for name in ('alpha_t','alpha_bar'):
                    kwargs[name]=float(np.asarray(kwargs[name]).reshape(-1)[0])
                return cond.conditioning(**kwargs)
            result=sampler.p_sample_loop(model.model,initial,y,conditioning,op,case['sigma'],False,'')
        grid=sampler.timestep_map[::-1]
    else:
        # Native author class/config, original50-point uniform grid and raw epsilon model.
        helper_module=types.ModuleType('utils.functions');helper_module.sigmoid=ref.helpers.sigmoid
        with patch.dict(sys.modules,{'utils.functions':helper_module}):diff=ref.diffusion.Diffusion()
        model_wrapper=ref.models.ClassifierGuidanceModel(model.model,None,diff,types.SimpleNamespace())
        sampling_config=types.SimpleNamespace(exp=types.SimpleNamespace(**cfg['exp']))
        if test_steps:sampling_config.exp.num_steps=test_steps
        ts=ref.helpers.get_timesteps(sampling_config)
        algorithm=object.__new__(ref.pgdm.PGDM)
        algorithm.model=model_wrapper;algorithm.diffusion=diff;algorithm.H=op
        algorithm.cfg=types.SimpleNamespace(algo=types.SimpleNamespace(**cfg['algo']))
        for name in ('awd','cond_awd','mcg','grad_term_weight','eta'):setattr(algorithm,name,cfg['algo'][name])
        original_initialize=algorithm.initialize
        observed_initial=[]
        def initialize(*args,**kwargs):
            value=original_initialize(*args,**kwargs)
            observed_initial.append(value.detach().cpu().clone())
            return value
        algorithm.initialize=initialize
        with torch.enable_grad():
            states,_=algorithm.sample(initial,None,ts,y_0=y)
        result=states[0];grid=ts[::-1];initial=observed_initial[0]
    if not torch.isfinite(result).all():raise RuntimeError('Nonfinite trajectory; no fallback')
    if not native and len(trace)!=len(grid):raise ValueError('Correction/grid count differs')
    return result.detach().cpu(),dict(trace=trace,initial=initial.detach().cpu(),entry_rng=entry,final_rng=rng(),grid=grid)
