"""A64 full independent executions. No binary learned-state input is accepted."""
from contextlib import contextmanager
import time
from unittest.mock import patch
from utils import paper_clock_alignment as rules


def snapshot(z, initial):
    import torch
    return dict(zeta=z.zeta.detach().cpu().clone(), D=z.D.detach().cpu().clone(),
                tau=z.tau.detach().cpu().clone(), initial=initial.detach().cpu().clone(),
                cpu_rng=torch.get_rng_state().clone(),
                cuda_rng=torch.cuda.get_rng_state_all() if str(z.device).startswith('cuda') else [])


def snapshot_class():
    from modules.zaps_algorithm import ZAPS
    class ObservedZAPS(ZAPS):
        def _reverse_diffusion(self, *args, **kwargs):
            self.observed_unrolls = getattr(self, 'observed_unrolls', 0)+1
            if self.observed_unrolls == 10:
                self.before_final = snapshot(self, kwargs['init_noise'])
            return super()._reverse_diffusion(*args, **kwargs)
    return ObservedZAPS


def snapshot_equal(first, second):
    import torch
    keys = ('zeta', 'D', 'tau', 'initial', 'cpu_rng')
    errors = {}
    for key in keys:
        a, b = first[key], second[key]
        rules.require(a.shape == b.shape, 'Snapshot shape mismatch: '+key)
        errors[key] = float((a.double()-b.double()).norm()/a.double().norm().clamp(min=1e-30))
    rng_equal = (torch.equal(first['cpu_rng'], second['cpu_rng']) and
                 len(first['cuda_rng']) == len(second['cuda_rng']) and
                 all(torch.equal(a,b) for a,b in zip(first['cuda_rng'],second['cuda_rng'])))
    passed = rng_equal and all(errors[k] <= 1e-6 for k in ('zeta','D','initial')) and errors['tau']==0.
    return dict(passed=passed, relative_errors=errors, rng_equal=rng_equal,
                scope='Audit AFTER independent executions; no baseline parameter or RNG fed into candidate')


def clocks(synchronize, operation, clock=time.perf_counter):
    """Operation must load y/build operator/parameters and return a CPU output."""
    synchronize(); start = clock()
    result = operation()
    synchronize(); seconds = clock()-start
    return result, seconds


@contextmanager
def counters(model, tick=None):
    """Top-level NN evaluations vs optimizer backwards; checkpoint work stays in walltime."""
    import torch
    count = dict(forward_attempts=0, forwards=0, input_vjp_attempts=0,
                 optimizer_backward_attempts=0, optimizer_backward_completed=0,
                 base_H_actions=0, base_adjoint_actions=0, calls=[])
    def before(module, arguments):
        count['forward_attempts'] += 1
        count['calls'].append(int(arguments[1].reshape(-1)[0]))
        count['_rng'] = (torch.get_rng_state().clone(),torch.cuda.get_rng_state_all()
                         if arguments[0].is_cuda else [])
    def after(module, arguments, result):
        cpu,cuda = count.pop('_rng')
        rules.require(torch.equal(cpu,torch.get_rng_state()) and
                      (not cuda or all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all()))),
                      'Model consumed sampling RNG; pre-drawn innovation pairing would be invalid')
        count['forwards'] += 1
        if tick and count['forwards'] % 30 == 0:
            tick(dict(count))
    pre = model.register_forward_pre_hook(before)
    post = model.register_forward_hook(after)
    original = torch.Tensor.backward
    def backward(tensor, *args, **kwargs):
        count['optimizer_backward_attempts'] += 1
        value = original(tensor, *args, **kwargs)
        count['optimizer_backward_completed'] += 1
        return value
    try:
        with patch.object(torch.Tensor, 'backward', backward):
            yield count
    finally:
        count.pop('_rng',None)
        pre.remove(); post.remove()


def count_operator(base, count):
    forward, transpose = base.H, base.transpose
    def H(value):
        count['base_H_actions'] += 1
        return forward(value)
    def adjoint(value):
        count['base_adjoint_actions'] += 1
        return transpose(value)
    base.H = H; base.transpose = adjoint


def response(z, y, initial, innovations, mask, count, sensed=None):
    import torch
    from modules.zaps_algorithm import ddpm_posterior_step
    from utils import angular_feedback_control as controller
    from utils.fixed_count_state_positions import Positions, noise_u
    sensed = sensed or controller.sensed_field
    nominal = z.tau.detach().cpu().tolist()[::-1]
    positions = Positions(noise_u(z.dm.alphas_cumprod.detach().cpu().numpy()), nominal)
    x = initial.detach(); trace = []
    def backward_attempt():
        count['input_vjp_attempts'] += 1
    for k, t in enumerate(nominal):
        s = nominal[k+1] if k < 29 else -1
        field, multiplier, row = sensed(z,y,x,t,s,positions,rules.SIGMA,cosine=1.,
                                        mask=mask,before_backward=backward_attempt)
        rules.require(multiplier==1., 'Unexpected extra multiplier')
        with torch.no_grad():
            if s >= 0:
                with patch.object(torch, 'randn_like', return_value=innovations[k]):
                    next_x = ddpm_posterior_step(x,field['clean'],t,s,z.dm.alphas_cumprod,
                                                eta=1.,learned_log_var=None,mode='ddpm')
            else:
                next_x = ddpm_posterior_step(x,field['clean'],t,s,z.dm.alphas_cumprod,
                                            eta=1.,learned_log_var=None,mode='ddpm')
            x = (next_x+field['q']).detach()
            rules.require(bool(torch.isfinite(x).all()), 'Nonfinite candidate output')
        trace.append(dict(step=k,t=t,next_t=s,**row))
    return x, trace


def execute(model, operator, y, task, arm, count, *, mask=None, size=256, level=3, sensed=None):
    """Every call initializes its OWN ζ/D/noise; baseline snapshots are output only."""
    import torch
    from modules.zaps_algorithm import ZAPS
    from utils.diag_learning_rate_ablation import set_seed
    from utils.no_learning_replay import noise_from_seed
    rules.require(arm in rules.ARMS, 'Unknown arm')
    set_seed(rules.SEED)
    configuration = rules.config(task)
    configuration.update(img_size=size, level=level)
    configuration['num_epochs'] = 10 if arm=='PAPER_ZAPS' else 9 if arm=='RESPONSE_300' else 0
    cls = snapshot_class() if arm=='PAPER_ZAPS' else ZAPS
    z = cls(model, operator, **configuration)
    rules.require(z.tau.detach().cpu().tolist()==rules.grid(), 'Actual grid is not public 15/10/5')
    if arm=='PAPER_ZAPS':
        losses = z.optimize(y, verbose=False, x0_gt=None)
        output, state, trace = z._last_opt_x0, z.before_final, []
    elif arm=='RESPONSE_300':
        losses = z.optimize(y, verbose=False, x0_gt=None)
        initial = z._last_opt_noise
        state = snapshot(z, initial)
        innovations = [torch.randn_like(initial) for _ in range(29)]
        output, trace = response(z,y,initial,innovations,mask,count,sensed)
    else:
        losses = []
        initial, innovations, before = noise_from_seed(rules.SEED, model.device,
            shape=(1,3,size,size), reference_epochs=10)
        state = snapshot(z, initial)
        # snapshot after pre-drawing innovations: explicitly restore entry RNG in audit payload only.
        state['cpu_rng'], state['cuda_rng'] = before['cpu'], before['cuda']
        output, trace = response(z,y,initial,innovations,mask,count,sensed)
    rules.require(bool(torch.isfinite(output).all()), 'Nonfinite reconstruction')
    return output.detach().cpu(), dict(snapshot=state, trace=trace, losses=losses)


def validate_counts(count, arm):
    expected = rules.NFE[arm]
    rules.require(count['forwards']==count['forward_attempts']==expected, 'Forward NFE mismatch')
    rules.require(count['calls']==rules.grid()[::-1]*(10 if arm!='RESPONSE_30' else 1),
                  'Actual model-call grid/order differs')
    rules.require(count['optimizer_backward_attempts']==count['optimizer_backward_completed']==rules.UPDATES[arm],
                  'Optimizer backwards mismatch')
    rules.require(0 <= count['input_vjp_attempts'] <= (0 if arm=='PAPER_ZAPS' else 30),
                  'Unexpected input VJP cost')
