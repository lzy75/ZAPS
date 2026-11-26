"""Wavelet temporal increments with a post-rounding observation-loss guard."""
from collections import deque
import math

from utils import a145_temporal_geometry as geometry

temporal, graph, files, native = geometry.temporal, geometry.graph, geometry.files, geometry.native
_dot = geometry._dot
ARMS = ('UNGUARDED','GUARDED')
ROUNDING_ULPS = 64
MAX_BACKTRACKS = 8


def maximum_fraction(slope, curvature, slack=0.):
    """Largest beta in [0,1] with slope*beta + curvature*beta**2/2 <= slack."""
    files.require(all(math.isfinite(v) for v in (slope,curvature,slack)) and
                  curvature>=0 and slack>=0,'Invalid observation line coefficients')
    if slope+.5*curvature<=slack:
        return 1.
    if curvature==0:
        return slack/slope
    root = math.hypot(slope,math.sqrt(2*curvature*slack))
    if slope>=0:
        return 2*slack/(root+slope) if slack else 0.
    return (root-slope)/curvature


def minmod(first,second):
    import torch

    agreement = (first.sign()==second.sign()) & (first!=0)
    return first.sign()*torch.minimum(first.abs(),second.abs())*agreement


def metric_embedding(geom, value):
    if geom.axes is None:
        return math.sqrt(geom.factor)*value.double()
    # For the exact separable left inverse, H+* H+ = (H H*)^-1.
    return geom.lift(value.double())


def controller(geom,wave,arm,sigma,counts,trace):
    import torch

    files.require(arm in ARMS,'Unknown A146 arm')
    memory = deque(maxlen=2)

    def remember(clean):
        memory.append(clean.detach())

    def analyze(value):
        counts['module_dwt_analysis'] += 1
        return wave.analysis(value)

    def synthesize(value):
        counts['module_dwt_synthesis'] += 1
        return wave.synthesis(value)

    def refine(q,residual,alpha,scalar,t,clean,physical):
        files.require(len(memory)==2,'Missing causal prediction history')
        scale = math.sqrt(alpha)
        earlier,previous = memory
        z = synthesize(minmod(analyze(previous-clean),analyze(earlier-clean)))
        remember(clean)
        support = analyze(q)!=0
        rhs_image = geom.weak(z)
        weights = graph.edge_weights(clean,sigma)
        counts['graph_builds'] += 1

        def normal(coefficients):
            value = synthesize(coefficients)
            counts['graph_actions'] += 1
            field = (1+graph.RIDGE)*value+temporal.STRENGTH*graph.laplacian(value,weights)
            return analyze(field)*support

        rhs = analyze(rhs_image)*support
        coefficients,path,relative,stop = graph.conjugate_gradient(
            normal,rhs,_dot,temporal.late.depth.MAX_UPDATES,temporal.late.depth.GRAD_RTOL)
        proposal = synthesize(coefficients) if len(path)>1 else torch.zeros_like(q)
        proposal = geom.weak(proposal)
        # Exact projection can leave a sparse wavelet support; restore support before auditing H.
        if not bool(support.all()):
            proposal = synthesize(analyze(proposal)*support)
        physical_response = geom.view.H(physical/scale)
        target = scalar['weight']*residual.double().reshape_as(physical_response)
        error = metric_embedding(geom,physical_response.double()-target)
        direction = metric_embedding(geom,geom.view.H(proposal))
        base = .5*_dot(error,error)
        slope,curvature = _dot(error,direction),_dot(direction,direction)
        slack = ROUNDING_ULPS*torch.finfo(q.dtype).eps*max(base,1e-12)
        beta = maximum_fraction(slope,curvature,slack) if arm=='GUARDED' else 1.
        rounded_bound = base+slack
        updated,after = physical,base
        halvings = 0
        if bool(_dot(proposal,proposal)) and beta>0:
            for index in range(MAX_BACKTRACKS+1):
                updated = physical+(scale*beta)*proposal
                actual_error = metric_embedding(geom,geom.view.H(updated/scale).double()-target)
                after = .5*_dot(actual_error,actual_error)
                files.require(math.isfinite(after),'A146 observation loss is nonfinite')
                if arm=='UNGUARDED' or after<=rounded_bound:
                    break
                if index==MAX_BACKTRACKS:
                    beta,updated,after = 0.,physical,base
                else:
                    beta *= .5
                    halvings += 1
        else:
            beta = 0.
        files.require(arm!='GUARDED' or after<=rounded_bound,'A146 observation guard failed')
        trace.append(dict(t=t,beta=beta,halvings=halvings,objective_physical=base,objective_applied=after,
                          tolerance=slack,slope=slope,curvature=curvature,updates=len(path)-1,
                          relative_gradient=relative,stop=stop,proposal_energy=_dot(proposal,proposal)))
        return updated

    return remember,refine
