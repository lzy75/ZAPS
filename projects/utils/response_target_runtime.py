"""A95 exact AD response, physical coefficient coordinates and immutable ZAPS replay."""
from contextlib import contextmanager,ExitStack
from pathlib import Path
from unittest.mock import patch
from utils import overall_direction_audit as old,response_target_lm as rules


@contextmanager
def forward_ad_checkpoint_scope(model):
    """Bypass ONLY the old non-forward-AD custom checkpoint, restore all globals."""
    expected=Path(__file__).resolve().parents[2]/'DPS/guided_diffusion/nn.py';found={}
    for layer in model.modules():
        namespace=getattr(getattr(type(layer),'forward',None),'__globals__',{})
        function=namespace.get('checkpoint');code=getattr(function,'__code__',None)
        if code and Path(code.co_filename).resolve()==expected:found[id(namespace)]=namespace
    def direct(function,inputs,parameters,flag):return function(*inputs)
    with ExitStack() as stack:
        for namespace in found.values():stack.enter_context(patch.dict(namespace,checkpoint=direct))
        yield len(found)


def predict_clean(model,x,t):
    """Same clamped raw predictor values; preserve forward AD, never detach x here."""
    import torch
    time=torch.full((x.shape[0],),int(t),device=x.device,dtype=torch.long)
    ab=model.alphas_cumprod[t];eps=model._predict_eps(x,time)
    return ((x-(1-ab).sqrt()*eps)/ab.sqrt().clamp(min=1e-8)).clamp(-1,1)


def increment(count,key):count[key]=count.get(key,0)+1


class Geometry:
    def __init__(self,q,dwt,scalar,count):
        self.q=q;self.dwt=dwt;self.scalar=scalar;self.count=count
        increment(count,'module_dwt_analysis');self.c=dwt.analysis(q);self.support=self.c!=0
        self.qnorm=rules.norm(q)
        if self.qnorm==0:raise RuntimeError('Zero native correction; no substitution')
    def project(self,p):
        return p.clamp(min=0) if self.scalar else p*self.support
    def image(self,p):
        if self.scalar:return self.q*(p.reshape(())/self.qnorm)
        increment(self.count,'module_dwt_synthesis');return self.dwt.synthesis(p*self.support)
    def adjoint(self,g):
        if self.scalar:return (g.double()*self.q.double()).sum().to(g.dtype).reshape(1)/self.qnorm
        increment(self.count,'module_dwt_analysis');return self.dwt.analysis(g)*self.support


class Provider:
    def __init__(self,model,op,origin,t,geometry,target,count):
        self.model=model;self.op=op;self.origin=origin;self.t=t;self.geometry=geometry;self.target=target;self.count=count
    def observation(self,x):
        from utils.shrink_multitask_runtime import informative
        return informative(self.op.H(predict_clean(self.model,x,self.t)),self.op)
    def value(self,p):
        import torch
        increment(self.count,'solver_value_attempts')
        with torch.no_grad():return self.observation(self.origin+self.geometry.image(p)).detach()
    def _vjp(self,p,vector=None):
        import torch
        from utils.current_score_runtime_r2 import input_vjp_parameters
        increment(self.count,'solver_vjp_forward_attempts')
        with torch.enable_grad(),input_vjp_parameters(self.model.model):
            x=(self.origin+self.geometry.image(p)).detach().requires_grad_(True);prediction=self.observation(x)
            v=(prediction.detach()-self.target) if vector is None else vector
            increment(self.count,'input_vjp_attempts')
            g=torch.autograd.grad(prediction,x,v,create_graph=False)[0]
            increment(self.count,'input_vjp_completed')
        return prediction.detach(),self.geometry.adjoint(g.detach())
    def linearize(self,p):return self._vjp(p)
    def vjp(self,p,vector):return self._vjp(p,vector)[1]
    def jvp(self,p,direction):
        import torch
        from utils.shrink_multitask_runtime import informative
        increment(self.count,'input_jvp_attempts');increment(self.count,'solver_jvp_forward_attempts')
        with torch.no_grad(),forward_ad_checkpoint_scope(self.model.model):
            x=self.origin+self.geometry.image(p);v=self.geometry.image(direction)
            _,clean_response=torch.func.jvp(lambda a:predict_clean(self.model,a,self.t),(x,),(v,))
            # All four frozen H are LINEAR. Apply H after the exact network JVP;
            # no dependence on a resize/convolution forward-AD implementation.
            response=informative(self.op.H(clean_response),self.op)
        increment(self.count,'input_jvp_completed');return response.detach()


def original_scalar(model,op,y,clean,q,t):
    import torch
    from utils.shrink_multitask_runtime import informative
    with torch.no_grad():
        base=op.H(clean);r=informative(y-base,op).double()
        d=informative(op.H(clean+q/model.alphas_cumprod[t].sqrt().clamp(min=1e-8))-base,op).double()
        return r,d


def shared_prefix(model,op,y,entry,bundle,task,count,size=256,level=3):
    import torch
    z=old.make_z(model,op,entry,task,size,level);old.restore_rng(entry,model.device);x=entry['initial'].to(model.device)
    with torch.no_grad():
        for step,i in enumerate(range(29,-1,-1)):
            uncond,q,clean=old.native_parts(model,z,x,y,i)
            if step==old.INTERVENTION:break
            x=uncond+q
    errors={name:rules.norm(a-bundle['state'][name].to(a.device))/max(rules.norm(bundle['state'][name]),1e-30) for name,a in (('uncond',uncond),('q',q))}
    rng=old.rng_state(model.device);reference=bundle['state']
    if not all(v<=1e-6 for v in errors.values()) or not rng_equal(rng,reference):raise RuntimeError('A92 replay/state/RNG mismatch')
    return clean.detach().cpu(),dict(relative_errors=errors,forwards=16)


def rng_equal(a,b):
    import torch
    return torch.equal(a['cpu_rng'],b['cpu_rng']) and len(a['cuda_rng'])==len(b['cuda_rng']) and all(torch.equal(x,y) for x,y in zip(a['cuda_rng'],b['cuda_rng']))


def target(model,op,y,bundle,sigma):
    import torch
    from utils.shrink_multitask_runtime import informative
    with torch.no_grad():
        base=informative(op.H(bundle['reference_clean'].to(model.device)),op)
        residual=informative(y,op)-base;energy=rules.dot(residual,residual)
        weight=rules.noise_weight(energy,residual.numel(),sigma)
        return base+weight*residual,dict(omega=weight,residual_energy=energy,measurements=residual.numel(),noise_energy=residual.numel()*sigma*sigma)


def run(model,op,y,entry,bundle,current_clean,task,arm,sigma,count,scalar_seed=None,size=256,level=3):
    import torch,time
    if arm not in rules.ARMS:raise ValueError('Unknown A95 arm')
    from utils.noise_signal_shrink import select
    z=old.make_z(model,op,entry,task,size,level);q=bundle['state']['q'].to(model.device)
    destination,statistics=target(model,op,y,bundle,sigma);before=old.rng_state(model.device)
    start=time.perf_counter();solution=None
    with torch.no_grad():
        clean=current_clean.to(model.device) if arm=='LEGACY_SCALAR' else bundle['reference_clean'].to(model.device)
        t=bundle['state']['current_t'] if arm=='LEGACY_SCALAR' else bundle['state']['next_t']
        residual,response=original_scalar(model,op,y,clean,q,t)
        proxy=select(rules.dot(residual,response),rules.dot(residual,residual),rules.dot(response,response),residual.numel(),sigma)
        correction=proxy['gamma']*q
    if arm in ('REAL_SCALAR','REAL_DIAG'):
        geometry=Geometry(q,z.dwt,arm=='REAL_SCALAR',count)
        provider=Provider(model,op,bundle['state']['uncond'].to(model.device),bundle['state']['next_t'],geometry,destination,count)
        if arm=='REAL_SCALAR':initial=torch.tensor([proxy['gamma']*geometry.qnorm],device=model.device,dtype=q.dtype)
        else:
            if scalar_seed is None:raise RuntimeError('Same-target scalar numerical solution required before matrix refinement')
            initial=geometry.c*(float(scalar_seed['beta'])/geometry.qnorm)
        solution,solver=rules.lm(provider,initial,geometry.project)
        with torch.no_grad():correction=geometry.image(solution)
        if arm=='REAL_SCALAR':solver['beta']=float(solution.reshape(()));solver['gamma']=solver['beta']/geometry.qnorm
        else:
            solver['scalar_initializer_converged']=scalar_seed['converged'];solver['scalar_initializer_value']=scalar_seed['final_value']
            tolerance=64*torch.finfo(q.dtype).eps*max(1.,abs(scalar_seed['final_value']))
            if solver['final_value']>scalar_seed['final_value']+tolerance:raise RuntimeError('Matrix objective worse than its feasible scalar initializer')
            nonzero=geometry.support;gain=solution.double()[nonzero]/geometry.c.double()[nonzero]
            solver.update(gamma_abs_max=float(gain.abs().max()),negative_gain_fraction=float((gain<0).float().mean()),
                inactive_zero_native_coefficients=int((~nonzero).sum()))
    else:solver=dict(status='CLOSED_FORM_PROXY',converged=None,gamma=proxy['gamma'],proxy=proxy)
    if not bool(torch.isfinite(correction).all()):raise RuntimeError('Nonfinite correction')
    torch.cuda.synchronize() if str(model.device).startswith('cuda') else None
    solver_seconds=time.perf_counter()-start
    if not rng_equal(before,old.rng_state(model.device)):raise RuntimeError('Solver consumed sampling RNG')
    start=time.perf_counter();output,first,rng=old.continue_run(model,z,y,bundle['state'],correction)
    if str(model.device).startswith('cuda'):torch.cuda.synchronize()
    with torch.no_grad():
        from utils.shrink_multitask_runtime import informative
        actual=informative(op.H(first.to(model.device)),op);actual_value=.5*rules.dot(actual-destination,actual-destination)
    if arm in ('REAL_SCALAR','REAL_DIAG') and abs(actual_value-solver['final_value'])>64*torch.finfo(q.dtype).eps*max(1.,abs(actual_value),abs(solver['final_value'])):
        raise RuntimeError('Solver objective differs from actual first continuation predictor')
    return output,first,rng,correction.detach().cpu(),dict(solver=solver,target=statistics,actual_target_objective=actual_value,
        solver_seconds=solver_seconds,continuation_seconds=time.perf_counter()-start,
        correction_norm=rules.norm(correction),native_norm=rules.norm(q))


def checkpoint_probe(model,t,count):
    """Two REAL NN calls: old-checkpoint VJP vs direct forward-AD JVP/primal parity."""
    import torch
    from utils.current_score_runtime_r2 import input_vjp_parameters
    x=torch.zeros(1,3,256,256,device=model.device)
    v=torch.sin(torch.arange(x.numel(),device=x.device,dtype=x.dtype)).reshape_as(x);v/=v.norm()
    w=torch.cos(torch.arange(x.numel(),device=x.device,dtype=x.dtype)).reshape_as(x);w/=w.norm()
    with torch.enable_grad(),input_vjp_parameters(model.model):
        inp=x.detach().requires_grad_(True);increment(count,'probe_F');clean=predict_clean(model,inp,t);increment(count,'probe_VJP')
        g=torch.autograd.grad(clean,inp,w)[0].detach();reference=clean.detach()
    del clean,inp
    with torch.no_grad(),forward_ad_checkpoint_scope(model.model) as bypassed:
        increment(count,'probe_F');increment(count,'probe_JVP');primal,jv=torch.func.jvp(lambda a:predict_clean(model,a,t),(x,),(v,))
    primal_error=rules.norm(primal-reference)/max(rules.norm(reference),1e-30)
    lhs=rules.dot(jv,w);rhs=rules.dot(v,g);adjoint_error=abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-6)
    if bypassed==0 or primal_error>1e-6 or abs(lhs-rhs)>1e-5+5e-4*max(abs(lhs),abs(rhs)) or not bool(torch.isfinite(jv).all()):
        raise RuntimeError('Actual checkpoint forward-AD primal/adjoint gate failed: '+str((bypassed,primal_error,lhs,rhs)))
    return dict(bypassed_namespaces=bypassed,primal_relative_error=primal_error,adjoint_relative_error=adjoint_error,passed=True)
