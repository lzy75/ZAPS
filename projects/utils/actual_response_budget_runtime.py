"""Runtime for A104; NFE, optimizer backward, VJP and JVP stay distinct."""
import copy
from utils import paper_clock_runtime as clock
from utils import shared_prefix_feedback as shared_rules


def make_operator(parent, case, device):
    from pathlib import Path
    from utils.diag_shrink_multitask import make_operator as build
    return build(Path(parent), case, device)


def train_zaps(model, operator, y, case, epochs):
    import torch
    import torch.nn.functional as F
    from modules.zaps_algorithm import ZAPS
    from utils.diag_learning_rate_ablation import set_seed
    set_seed(case['seed'])
    z = ZAPS(model, operator, **shared_rules.configuration(case['task']))
    optimizer = torch.optim.Adam([z.zeta, z.D], lr=z.lr)
    initial = torch.randn(y.shape[0], z.channels, z.img_size, z.img_size, device=z.device)
    losses=[]; output=None; nfe=[0]
    for _ in range(epochs):
        optimizer.zero_grad()
        output=z._reverse_diffusion(y,nfe_counter=nfe,eta_override=z.eta,init_noise=initial)
        loss=F.mse_loss(z.A.H(output),y);loss.backward()
        torch.nn.utils.clip_grad_norm_([z.zeta,z.D],max_norm=1.);optimizer.step()
        with torch.no_grad():z.zeta.data.clamp_(min=1e-6)
        losses.append(float(loss.detach()))
    return output.detach().cpu(),dict(entry=clock.snapshot(z,initial),optimizer=copy.deepcopy(optimizer.state_dict()),losses=losses,nfe_counter=nfe[0])


def response_suffix(model, operator, y, case, arm, prefix, count):
    """A100 exact fixed-call controller on all 30 final-unroll locations."""
    import torch
    from utils import equal_budget_ray_diagonal as response
    from utils import overall_direction_audit as native
    mapped={'REAL_RAY445':'RAY_K1_EQ','REAL_DIAG445':'DIAG_K1_EQ'}[arm]
    z=native.make_z(model,operator,prefix['entry'],case['task'])
    native.restore_rng(prefix['entry'],model.device);x=prefix['entry']['initial'].to(model.device)
    trace=[]
    for k in range(30):
        x,_,detail,solver_traces,_=response.step(model,z,operator,y,x,k,mapped,count)
        trace.append(dict(detail=detail,solver_traces=solver_traces))
    return x.detach().cpu(),trace,native.rng_state(model.device)
