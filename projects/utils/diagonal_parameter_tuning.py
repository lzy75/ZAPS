"""A125: A118 timing alignment and a small closed-solver parameter study."""
import inspect
import math
from utils import paper_clock_alignment as files
from utils import a118_blur_transfer_runtime as previous
from utils import transfer_diagonal_runtime as native

PARAMETERS = ((1,1.),(1,1.5),(2,1.),(2,1.5))


def mapped_times(targets, native_grid):
    targets, grid = sorted(targets), sorted(native_grid)
    if len(grid)<len(targets) or len(set(targets))!=len(targets):
        raise ValueError('Need distinct target times and sufficient native positions')
    selected=[]; last=-1
    for i,t in enumerate(targets):
        candidates=range(last+1,len(grid)-(len(targets)-i)+1)
        last=min(candidates,key=lambda j:(abs(grid[j]-t),j))
        selected.append(grid[last])
    return selected


def label(iterations, step):
    return f'DIAG_K{iterations}_STEP{step:g}'


def refine(wavelet, iterations, step, counts, trace):
    from utils.equal_nfe_feedback_runtime import _dot
    files.require(iterations in (1,2) and 0<step<2,'Unknown solver configuration')
    def apply(operator,q,residual,alpha,scalar,t):
        view=native.LinearView(operator,counts)
        a=math.sqrt(alpha); c=wavelet.analysis(q)
        counts['module_dwt_analysis']+=1
        support=c!=0
        delta=scalar['gamma']*q
        response=view.H(delta/a)
        goal=scalar['weight']*residual.double().reshape_as(response)
        error=response.double()-goal
        first=.5*_dot(error,error); objectives=[first]
        for _ in range(iterations):
            image_gradient=view.transpose(error.to(q.dtype))/a
            counts['module_adjoint_actions']+=1
            g=wavelet.analysis(image_gradient)*support
            counts['module_dwt_analysis']+=1
            energy=_dot(g,g)
            files.require(math.isfinite(energy),'Nonfinite closed-solver gradient')
            if energy==0:
                break
            direction=wavelet.synthesis(g)
            counts['module_dwt_synthesis']+=1
            change=view.H(direction/a).double()
            denominator=_dot(change,change)
            files.require(math.isfinite(denominator) and denominator>0,'Invalid closed-solver curvature')
            eta=step*energy/denominator
            delta=delta-eta*direction
            error=error-eta*change
            value=.5*_dot(error,error)
            files.require(math.isfinite(value) and value<=objectives[-1]+2e-5*max(1.,objectives[-1]),
                          'Closed-solver iteration increased its objective')
            objectives.append(value)
        actual=view.H(delta/a).double()-goal
        value=.5*_dot(actual,actual)
        files.require(math.isfinite(value) and value<=first+2e-5*max(1.,first),'Final closed objective increased')
        trace.append(dict(t=t,iterations=iterations,step=step,gamma=scalar['gamma'],weight=scalar['weight'],
                          objective_before=first,objective_after=value,iteration_objectives=objectives))
        return delta
    return apply


def sampler(runtime,host,arm,times,apply):
    if arm=='BASE':
        return runtime.run
    source=inspect.getsource(runtime.run)
    def replace(before,after):
        nonlocal source
        files.require(source.count(before)==1,'Native update interface changed')
        source=source.replace(before,after,1)
    if host=='dps':
        replace("if arm=='DPS':","if arm=='DPS' or t not in active_times:")
        if arm!='SCALAR':
            scale='rules.SCALE' if runtime.__name__.endswith('dps_multiplier_transfer_runtime') else 'scale'
            replace(f'result=x_t-grad*{scale} if gamma==1. else x_t-gamma*{scale}*grad',
                    "with torch.no_grad():\n                delta=apply(op,q,residual,a*a,row,t)\n            result=x_t+delta")
    else:
        replace("if arm=='SHRINK':","if arm=='SHRINK' and t in active_times:")
        if arm!='SCALAR':
            replace("x=math.sqrt(b)*m.detach()+c1*noise+c2*eps.detach()+row['gamma']*q",
                    "delta=apply(op,q,r,a,row,t) if t in active_times else q\n   x=math.sqrt(b)*m.detach()+c1*noise+c2*eps.detach()+delta")
    namespace={**vars(runtime),'active_times':set(times),'apply':apply}
    exec(compile(source,runtime.__file__,'exec'),namespace)
    return namespace['run']


def choose(records):
    """Select one configuration shared by both tasks, from three development images."""
    scores=[]
    for iterations,step in PARAMETERS:
        arm=label(iterations,step); cells=[]
        for task in ('gaussian_deblur','motion_deblur'):
            a={r['case']['key']:r for r in records if r['case']['task']==task and r['arm']==arm}
            b={r['case']['key']:r for r in records if r['case']['task']==task and r['arm']=='SCALAR'}
            files.require(len(a)==len(b)==3 and a.keys()==b.keys(),'Need all paired development cases')
            delta={k:sum(a[i]['metrics'][k]-b[i]['metrics'][k] for i in a)/3 for k in ('psnr','ssim','lpips')}
            ratio=sum(r['full_seconds'] for r in a.values())/sum(r['full_seconds'] for r in b.values())
            wins=sum(a[i]['metrics']['psnr']>b[i]['metrics']['psnr'] for i in a)
            cells.append(dict(task=task,delta=delta,clock_ratio=ratio,psnr_wins=wins))
        qualified=all(c['delta']['psnr']>=.1 and c['delta']['ssim']>=0 and c['delta']['lpips']<=0 and
                      c['clock_ratio']<=1.1 and c['psnr_wins']>=2 for c in cells)
        scores.append(dict(arm=arm,iterations=iterations,step=step,qualified=qualified,cells=cells,
                           worst_task_psnr=min(c['delta']['psnr'] for c in cells),
                           mean_psnr=sum(c['delta']['psnr'] for c in cells)/2))
    selected=max(scores,key=lambda r:(r['qualified'],r['worst_task_psnr'],r['mean_psnr'],-r['iterations'],-r['step']))
    return dict(selected=selected,candidates=scores,
                rule='Both-task quality/time qualification, then worst-task PSNR, mean PSNR, fewer iterations, smaller step')
