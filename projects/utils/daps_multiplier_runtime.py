"""Pinned DAPS classes/AST, original operators/model, likelihood-gradient-only multiplier."""
import ast
from contextlib import contextmanager
from functools import lru_cache
import importlib
import importlib.metadata
import types
import sys
from unittest.mock import patch
from utils import daps_multiplier_transfer as rules
from utils import noise_signal_shrink as policy


def selected(path,names,env):
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes}!=set(names):raise ValueError('Missing pinned AST definitions')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return types.SimpleNamespace(**{n:env[n] for n in names})


@lru_cache(maxsize=1)
def reference():
    import torch
    import numpy as np
    import scipy
    import warnings
    import tqdm
    from abc import ABC,abstractmethod
    from torch.utils.data import Dataset
    from torchvision import transforms
    from PIL import Image
    from pathlib import Path
    rules.verify_upstream()
    if importlib.metadata.version('torchdiffeq')!='0.2.5':raise ValueError('Need torchdiffeq==0.2.5')
    prefix='_zaps_a81_daps_reference'
    for suffix,path in (('',rules.VENDOR),('.cores',rules.VENDOR/'cores'),('.model',rules.VENDOR/'model'),
                        ('.model.ddpm',rules.VENDOR/'model/ddpm'),('.forward_operator',rules.VENDOR/'forward_operator'),
                        ('.forward_operator.motionblur',rules.VENDOR/'forward_operator/motionblur')):
        name=prefix+suffix
        if name not in sys.modules:
            module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module
    scheduler=importlib.import_module(prefix+'.cores.scheduler')
    mcmc=importlib.import_module(prefix+'.cores.mcmc')
    trajectory=importlib.import_module(prefix+'.cores.trajectory')
    unet=importlib.import_module(prefix+'.model.ddpm.unet')
    resize=importlib.import_module(prefix+'.forward_operator.resizer')
    motion=importlib.import_module(prefix+'.forward_operator.motionblur.motionblur')
    pre=selected(rules.VENDOR/'model/precond.py',('VPPrecond',),dict(torch=torch,np=np))
    models=selected(rules.VENDOR/'model/__init__.py',('register_model','DiffusionModel','DDPM'),
        dict(torch=torch,nn=torch.nn,warnings=warnings,__MODEL__={},VPPrecond=pre.VPPrecond,create_model=unet.create_model))
    operators=selected(rules.VENDOR/'forward_operator/__init__.py',
        ('register_operator','Operator','DownSampling','Blurkernel','MotionBlur','GaussianBlur'),
        dict(torch=torch,nn=torch.nn,np=np,scipy=scipy,ABC=ABC,abstractmethod=abstractmethod,
             warnings=warnings,__OPERATOR__={},Kernel=motion.Kernel,Resizer=resize.Resizer))
    data=selected(rules.VENDOR/'data.py',('register_dataset','DiffusionData','ImageDataset'),
        dict(torch=torch,ABC=ABC,Dataset=Dataset,abstractmethod=abstractmethod,warnings=warnings,
             __DATASET__={},transforms=transforms,Image=Image,Path=Path))
    sampler=selected(rules.VENDOR/'sampler.py',('DAPS',),dict(torch=torch,nn=torch.nn,np=np,tqdm=tqdm,
        Trajectory=trajectory.Trajectory,get_diffusion_scheduler=scheduler.get_diffusion_scheduler,
        DiffusionPFODE=scheduler.DiffusionPFODE,MCMCSampler=mcmc.MCMCSampler))
    return types.SimpleNamespace(scheduler=scheduler,mcmc=mcmc,models=models,pre=pre,operators=operators,data=data,DAPS=sampler.DAPS)


def load_image(path,device='cpu'):
    from pathlib import Path
    cls=reference().data.ImageDataset
    dataset=cls(root=str(Path(path).parent),resolution=256,device=device)
    dataset.data=[Path(path)]  # Fix the existing panel, preserving official __getitem__/preprocessing.
    return dataset[0].unsqueeze(0)


def load_model(path,device):
    import torch
    model=reference().models.DDPM(rules.model_config(path),device=device)
    # The upstream factory catches load errors and returns random weights. Fail closed instead.
    state=torch.load(path,map_location='cpu',weights_only=True)
    model.model.model.load_state_dict(state,strict=True)
    del state
    model.eval();model.requires_grad_(False);model.device=device
    return model


def make_operator(case,device):
    from utils.diag_learning_rate_ablation import set_seed
    set_seed(case['kernel_seed']);config=dict(rules.configuration(case['task'])['operator'])
    name=config.pop('name');config['device']=device
    cls=reference().operators.DownSampling if name=='down_sampling' else reference().operators.MotionBlur
    return cls(**config)


def operator_state(op):
    return {k:v.detach().cpu().clone() for k,v in (op.down_sample if hasattr(op,'down_sample') else op.conv).state_dict().items()}


@contextmanager
def counters(network):
    import torch
    count=dict(forward_attempts=0,forwards=0,input_vjp_attempts=0,optimizer_backward_attempts=0,
               operator_gradient_attempts=0,operator_gradients_completed=0,base_H_actions=0,module_H_actions=0,
               mcmc_updates=0,annealing_steps=0,calls=[])
    def before(module,args):
        count['forward_attempts']+=1;count['calls'].append(float(args[1].reshape(-1)[0]))
        count['_rng']=(torch.get_rng_state().clone(),torch.cuda.get_rng_state_all() if args[0].is_cuda else [])
    def after(module,args,out):
        cpu,cuda=count.pop('_rng')
        if not torch.equal(cpu,torch.get_rng_state()) or (cuda and not all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all()))):
            raise RuntimeError('Denoiser consumed sampling RNG')
        count['forwards']+=1
    def backward(*args,**kwargs):
        count['optimizer_backward_attempts']+=1
        raise RuntimeError('A81 forbids optimizer backwards')
    pre=network.register_forward_pre_hook(before);post=network.register_forward_hook(after)
    try:
        with patch.object(torch.Tensor,'backward',backward):yield count
    finally:
        pre.remove();post.remove();count.pop('_rng',None)


class MeasuredOperator:
    def __init__(self,base,arm,count,context,raw_trace):
        self.base=base;self.arm=arm;self.count=count;self.context=context;self.raw_trace=raw_trace
        self.sigma=base.sigma;self.active_gradient=False;self.capture=False;self.prediction=None
    def __call__(self,x):
        self.count['base_H_actions']+=1;value=self.base(x)
        if self.capture:self.prediction=value.detach()
        return value
    def loss(self,x,y):return reference().operators.Operator.loss(self,x,y)
    def gradient(self,x,y,return_loss=False):
        import torch
        self.count['operator_gradient_attempts']+=1;self.active_gradient=True;self.capture=self.arm!='DAPS'
        try:gradient,loss=reference().operators.Operator.gradient(self,x,y,return_loss=True)
        finally:self.active_gradient=False;self.capture=False
        self.count['operator_gradients_completed']+=1
        lr=self.context['lr'];row=dict(gamma=1.,reason='ORIGINAL_DAPS')
        if self.arm!='DAPS':
            with torch.no_grad():
                r=(y-self.prediction).double()
                q=-lr*gradient.detach()/self.context['tau']**2
                self.count['module_H_actions']+=1;d=self(q).double()
                dot,r2,d2=float((r*d).sum()),float(r.square().sum()),float(d.square().sum())
                row=policy.select(dot,r2,d2,r.numel(),self.sigma)
                row.setdefault('residual_energy',r2);row.setdefault('response_energy',d2);row.setdefault('residual_response_dot',dot)
                if self.arm=='ONES':row=dict(row,gamma=1.,reason='GAMMA_ONE_EQUIVALENCE')
            # Preserve official arithmetic exactly in the gamma=1 branch.
            if row['gamma']!=1.:gradient=gradient*row['gamma']
        self.raw_trace.append((self.context['outer'],self.context['inner'],self.context['sigma'],lr,row['gamma'],
            row.get('weight',float('nan')),row.get('residual_energy',float('nan')),
            row.get('response_energy',float('nan')),row.get('residual_response_dot',float('nan'))))
        self.context['inner']+=1
        return (gradient,loss) if return_loss else gradient


def run(model,base,y,case,arm,count,*,sampler_config=None,progress=None):
    import copy
    import numpy as np
    import torch
    from utils.diag_learning_rate_ablation import set_seed
    if arm not in (*rules.ARMS,'ONES'):raise ValueError('Unknown DAPS multiplier arm')
    config=copy.deepcopy(sampler_config or rules.configuration(case['task'])['sampler'])
    sampler=reference().DAPS(**config);mc=sampler.mcmc_sampler
    if mc.mc_algo!='langevin' or mc.prior_solver!='gaussian':raise ValueError('A81 requires official pixel Langevin/gaussian setup')
    context={};rows=[];trace=[];op=MeasuredOperator(base,arm,count,context,rows)
    original_sample=mc.sample;original_update=mc.mc_update;original_grad=torch.autograd.grad
    def guarded_grad(*args,**kwargs):
        if not op.active_gradient:
            count['input_vjp_attempts']+=1
            raise RuntimeError('A81 forbids non-operator input/parameter derivatives')
        return original_grad(*args,**kwargs)
    def update(*args,**kwargs):
        count['mcmc_updates']+=1;value=original_update(*args,**kwargs)
        # Keep the official NaN branch intact; check accumulated flags once per outer step,
        # avoiding another CPU/GPU synchronization at every measurement update.
        context['nonfinite']=context['nonfinite'] | ~torch.isfinite(value).all()
        return value
    def sample(xt,net,x0hat,operator,measurement,sigma,ratio,*args,**kwargs):
        outer=count['annealing_steps'];context.update(outer=outer,inner=0,lr=mc.get_lr(ratio),tau=mc.tau,sigma=float(sigma))
        context['nonfinite']=torch.zeros((),dtype=torch.bool,device=x0hat.device)
        start=len(rows);out=original_sample(xt,net,x0hat,op,measurement,sigma,ratio,*args,**kwargs)
        if bool(context['nonfinite']):raise RuntimeError('Nonfinite native MCMC update; official zero-output fallback not accepted as a result')
        if context['inner']!=mc.num_steps:raise RuntimeError('Incomplete native MCMC loop')
        part=np.asarray(rows[start:],dtype=np.float64)
        row=dict(step=outer,sigma=float(sigma),lr=context['lr'],mcmc_steps=mc.num_steps,
                 gamma_mean=float(part[:,4].mean()),gamma_min=float(part[:,4].min()),gamma_max=float(part[:,4].max()),
                 gamma_zero_fraction=float((part[:,4]==0).mean()))
        if arm!='DAPS':row.update(weight_mean=float(part[:,5].mean()),residual_energy_mean=float(part[:,6].mean()),response_energy_mean=float(part[:,7].mean()))
        trace.append(row);count['annealing_steps']+=1
        if progress and count['annealing_steps']%10==0:progress(count['annealing_steps'],count)
        return out
    set_seed(case['seed']);initial=sampler.get_start(1,model)
    before_cpu=torch.get_rng_state().clone();before_cuda=torch.cuda.get_rng_state_all() if initial.is_cuda else []
    with patch.object(mc,'sample',sample),patch.object(mc,'mc_update',update),patch.object(torch.autograd,'grad',guarded_grad),torch.enable_grad():
        out=sampler.sample(model,initial,op,y,record=False,verbose=False)
    if not torch.isfinite(out).all():raise RuntimeError('Nonfinite DAPS output')
    return out.detach().cpu(),dict(trace=trace,dense_trace=np.asarray(rows,dtype=np.float64),
        dense_columns=['outer','inner','sigma','lr','gamma','weight','residual_energy','response_energy','residual_response_dot'],
        initial=initial.detach().cpu(),cpu_rng=before_cpu,cuda_rng=before_cuda,
        final_cpu_rng=torch.get_rng_state().clone(),final_cuda_rng=torch.cuda.get_rng_state_all() if initial.is_cuda else [],
        native_output='Unchanged official DAPS.sample return xt, including native final zero-sigma RNG draw')
