"""A71 original ZAPS q, active observation shrinkage, last unroll only."""
import time
from utils import shrink_multitask as rules
from utils import noise_signal_shrink as policy
from utils import paper_clock_runtime as clock

def make_operator(c,device,count=None,size=256,kernel=None):
    import torch
    from modules.degradations import get_operator
    if c['task']=='super_resolution':
        from utils.historical_ffhq_runtime import make_operator
        return make_operator(device,count,size=size)
    if c['task']=='gaussian_deblur':op=get_operator(c['task'],device=device,kernel_size=61,sigma=3.,noise_sigma=0.)
    elif c['task']=='motion_deblur':
        if kernel is None:raise ValueError('Frozen original DPS motion kernel required')
        from utils.diag_ffhq_four_tasks_unit import motion_operator
        op=motion_operator(kernel,device)
    elif c['task']=='inpainting':
        op=get_operator(c['task'],device=device,random_ratio=.7,mode='random',box_size=0,noise_sigma=0.,seed=-1)
        gen=torch.Generator(device=device);gen.manual_seed(c['mask_seed'])
        op._mask_cache=(torch.rand((1,1,size,size),device=device,generator=gen)>=.7).float()
    else:raise ValueError('Unknown task')
    if count is not None:clock.count_operator(op,count)
    return op

def informative(value,operator):
    mask=getattr(operator,'_mask_cache',None)
    return value.reshape(-1) if mask is None else value[mask.bool().expand_as(value)]

def controller_class():
    import torch
    from modules.zaps_algorithm import ZAPS,ddpm_posterior_step
    class Controlled(ZAPS):
        def _reverse_diffusion(self,y,nfe_counter=None,eta_override=None,init_noise=None,record_indicators=False):
            self.unrolls=getattr(self,'unrolls',0)+1
            if self.unrolls!=10:return super()._reverse_diffusion(y,nfe_counter,eta_override,init_noise,record_indicators)
            self.entry=clock.snapshot(self,init_noise);self.trace=[]
            if self.arm=='BASE':return super()._reverse_diffusion(y,nfe_counter,eta_override,init_noise,record_indicators)
            x=init_noise.to(self.device);ab=self.dm.alphas_cumprod
            for k,i in enumerate(range(len(self.tau)-1,-1,-1)):
                t=int(self.tau[i]);s=int(self.tau[i-1]) if i else -1
                with torch.no_grad():eps=self.dm._predict_eps(x,torch.full((y.shape[0],),t,device=self.device,dtype=torch.long))
                if nfe_counter is not None:nfe_counter[0]+=1
                clean=self._tweedie_estimate(x,eps,ab[t],i)
                uncond=ddpm_posterior_step(x,clean,t,s,ab,eta=self.eta if eta_override is None else eta_override,learned_log_var=None,mode='ddpm')
                residual=y-self.A.H(clean);v=self.A.transpose(residual)
                hv=self.dwt.synthesis(self.D[i]*self.dwt.analysis(v))
                q=self.zeta[i]*((v+(1-ab[t])*hv)/ab[t].sqrt().clamp(min=1e-8))
                with torch.no_grad():
                    if self.arm=='NULL':row=dict(gamma=1.,reason='NULL')
                    else:
                        r=informative(residual.detach().double(),self.A)
                        d=informative((self.A.H(clean.detach()+q.detach()/ab[t].sqrt().clamp(min=1e-8))-self.A.H(clean.detach())).double(),self.A)
                        row=policy.select(float((r*d).sum()),float(r.square().sum()),float(d.square().sum()),r.numel(),self.sigma)
                        row['active_measurements']=r.numel()
                gamma=row['gamma'];x=uncond+q if gamma==1. else uncond+gamma*q
                if not torch.isfinite(x).all():raise RuntimeError('Nonfinite A71 output; failed case retained')
                self.trace.append(dict(step=k,t=t,next_t=s,**row))
            return x
    return Controlled

def execute(model,op,y,c,arm,size=256,level=3):
    from utils.diag_learning_rate_ablation import set_seed
    set_seed(c['seed']);z=controller_class()(model,op,**rules.configuration(c['task'],size,level))
    z.arm,z.sigma=arm,c['sigma'];loss=z.optimize(y,verbose=False,x0_gt=None)
    return z._last_opt_x0.detach().cpu(),dict(entry=z.entry,trace=z.trace,loss=loss)

def replay(model,op,y,c,entry,arm,size=256,level=3):
    import torch
    z=controller_class()(model,op,**rules.configuration(c['task'],size,level))
    with torch.no_grad():z.zeta.copy_(entry['zeta'].to(model.device));z.D.copy_(entry['D'].to(model.device))
    if not torch.equal(z.tau.cpu(),entry['tau']):raise ValueError('Grid differs')
    torch.set_rng_state(entry['cpu_rng'])
    if entry['cuda_rng']:torch.cuda.set_rng_state_all(entry['cuda_rng'])
    z.arm,z.sigma,z.unrolls=arm,c['sigma'],9
    return z._reverse_diffusion(y,init_noise=entry['initial'].to(model.device)).detach().cpu(),z.trace
