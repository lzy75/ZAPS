"""A97 full final-unroll controllers. Same A95 objective/solver; identity at s=-1."""
from utils import response_target_runtime as rt,response_target_lm as lm,overall_direction_audit as native
from utils import direction_length_attribution as cross
ARMS=('BASE','LEGACY_SCALAR','REAL_SCALAR','REAL_DIAG','DIAG_DIR_SCALAR_LEN','SCALAR_DIR_DIAG_LEN')
MATRIX_ARMS=ARMS[3:]


class Provider(rt.Provider):
    def observation(self,x):
        if self.t>=0:return super().observation(x)
        from utils.shrink_multitask_runtime import informative
        return informative(self.op.H(x),self.op)
    def _vjp(self,p,vector=None):
        if self.t>=0:return super()._vjp(p,vector)
        import torch
        rt.increment(self.count,'terminal_operator_vjp_attempts')
        with torch.enable_grad():
            x=(self.origin+self.geometry.image(p)).detach().requires_grad_(True);prediction=self.observation(x)
            g=torch.autograd.grad(prediction,x,prediction.detach()-self.target if vector is None else vector)[0]
        rt.increment(self.count,'terminal_operator_vjp_completed')
        return prediction.detach(),self.geometry.adjoint(g.detach())
    def jvp(self,p,v):
        if self.t>=0:return super().jvp(p,v)
        import torch
        rt.increment(self.count,'terminal_operator_jvp_attempts')
        with torch.no_grad():value=self.observation(self.geometry.image(v))
        rt.increment(self.count,'terminal_operator_jvp_completed');return value
    def value(self,p):
        if self.t>=0:return super().value(p)
        import torch
        rt.increment(self.count,'terminal_value_attempts')
        with torch.no_grad():return self.observation(self.origin+self.geometry.image(p)).detach()


def compact(info):
    return {k:v for k,v in info.items() if k!='trace'}


def action(model,op,y,z,uncond,q,clean,t,s,arm,count):
    import torch
    from utils.shrink_multitask_runtime import informative
    from utils.noise_signal_shrink import select
    if arm not in ARMS:raise ValueError('Unknown A97 arm')
    if arm=='BASE':return q,dict(status='NATIVE',q_norm=lm.norm(q)),{}
    if arm=='LEGACY_SCALAR':
        r,d=rt.original_scalar(model,op,y,clean,q,t);info=select(lm.dot(r,d),lm.dot(r,r),lm.dot(d,d),r.numel(),.05)
        return q*info['gamma'],dict(status='LEGACY',**info,q_norm=lm.norm(q)),{}
    with torch.no_grad():
        if s>=0:
            rt.increment(count,'response_base_forward_attempts');base_clean=rt.predict_clean(model,uncond,s)
        else:base_clean=uncond
        base=informative(op.H(base_clean),op);residual=informative(y,op)-base
        omega=lm.noise_weight(lm.dot(residual,residual),residual.numel(),.05);target=base+omega*residual
        qn=lm.norm(q)
        if qn==0:return q*0,dict(status='ZERO_NATIVE_FEASIBLE_SET',omega=omega,q_norm=0.),{}
        if s>=0:r,d=rt.original_scalar(model,op,y,base_clean,q,s)
        else:r=residual.double();d=informative(op.H(q),op).double()
        proxy=select(lm.dot(r,d),lm.dot(r,r),lm.dot(d,d),r.numel(),.05)
    scalar_geometry=rt.Geometry(q,z.dwt,True,count)
    scalar_provider=Provider(model,op,uncond,s,scalar_geometry,target,count)
    initial=torch.tensor([proxy['gamma']*qn],device=q.device,dtype=q.dtype)
    beta,si=lm.lm(scalar_provider,initial,scalar_geometry.project)
    scalar=scalar_geometry.image(beta).detach();detail=dict(omega=omega,q_norm=qn,scalar_norm=lm.norm(scalar),scalar_gamma=float(beta.reshape(()))/qn,scalar=compact(si))
    traces=dict(scalar=si['trace'])
    if arm=='REAL_SCALAR':return scalar,detail,traces
    geometry=rt.Geometry(q,z.dwt,False,count);provider=Provider(model,op,uncond,s,geometry,target,count)
    p,di=lm.lm(provider,geometry.c*(float(beta.reshape(()))/qn),geometry.project)
    diagonal=geometry.image(p).detach()
    tolerance=64*torch.finfo(q.dtype).eps*max(1.,abs(si['final_value']))
    if di['final_value']>si['final_value']+tolerance:raise RuntimeError('Joint target exceeds scalar feasible initializer')
    dn=lm.norm(diagonal);gains=p.double()[geometry.support]/geometry.c.double()[geometry.support]
    detail.update(diagonal_norm=dn,diagonal_cosine_to_native=lm.dot(q,diagonal)/(qn*dn) if dn else None,
        negative_gain_fraction=float((gains<0).float().mean()),diagonal=compact(di));traces['diagonal']=di['trace']
    selected=diagonal if arm=='REAL_DIAG' else cross.correction(q,scalar,diagonal,arm)
    detail['applied_norm']=lm.norm(selected)
    return selected,detail,traces


def step(model,z,op,y,x,k,arm,count):
    import torch
    if not 0<=k<30:raise ValueError('A97 requires exactly30 registered positions')
    i=29-k;t=int(z.tau[i]);s=int(z.tau[i-1]) if i else -1
    calls=len(count['calls'])
    with torch.no_grad():uncond,q,clean=native.native_parts(model,z,x,y,i)
    after=native.rng_state(model.device)
    delta,detail,traces=action(model,op,y,z,uncond,q,clean,t,s,arm,count)
    if not rt.rng_equal(after,native.rng_state(model.device)):raise RuntimeError('Controller consumed sampling RNG')
    expected=[t]+([s]*(len(count['calls'])-calls-1) if s>=0 else [])
    if count['calls'][calls:]!=expected:raise RuntimeError('Native/lookahead call times mismatch')
    with torch.no_grad():output=(uncond+delta).detach()
    if not bool(torch.isfinite(output).all()):raise RuntimeError('Nonfinite full trajectory; no fallback')
    detail.update(step=k,t=t,next_t=s,NN_calls_this_step=len(expected))
    return output,after,detail,traces,delta.detach()


def budget():
    per=lm.budget();lookahead_steps=29;scalar_solves=12*4*lookahead_steps;matrix_solves=12*3*lookahead_steps
    return dict(cases=12,arms=72,native_grid_forwards=72*30,response_base_forwards=scalar_solves,
        max_neural_solver_forwards=(scalar_solves+matrix_solves)*per['max_solver_forward'],
        max_new_neural_forwards=72*30+scalar_solves+(scalar_solves+matrix_solves)*per['max_solver_forward'],
        max_neural_VJP=(scalar_solves+matrix_solves)*per['max_solver_vjp'],
        max_neural_JVP=(scalar_solves+matrix_solves)*per['max_solver_jvp'],
        terminal_scalar_solves=48,terminal_diagonal_solves=36,probe_F_VJP_JVP_per_load=[2,1,1],optimizer_backward=0)
