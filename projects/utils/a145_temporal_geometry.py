"""Observation and weak-direction quadratic shared by the A145 adapters."""
from collections import deque
import math

from utils import diag_a143_temporal_inpainting as temporal

graph, files, native = temporal.graph, temporal.files, temporal.native
_dot = temporal._dot
ARMS = ('ZERO_WEAK','TEMPORAL_WEAK')


class Geometry:
    def __init__(self, task, operator, like, counts):
        import torch

        self.view = native.LinearView(temporal.parent.ObservedResponse(operator),counts)
        self.counts = counts
        counts['geometry_setups'] = counts.get('geometry_setups',0)+1
        self.task = task
        self.factor = None
        self.axes = None
        if task=='super_resolution':
            if getattr(operator,'task',None)=='super_resolution':
                self.factor = 16.
            else:
                import numpy as np
                from utils.sr_observable_counterfactual import native_matrices

                ay,ax = native_matrices(operator.base.down_sample,like.shape)
                self.axes = tuple(torch.as_tensor(np.linalg.pinv(axis),device=like.device) for axis in (ay,ax))
                self.forward_axes = tuple(torch.as_tensor(axis,device=like.device) for axis in (ay,ax))
        elif task=='inpainting':
            self.factor = 1.
        else:
            # Young's inequality applies to the native zero-padded and circular convolutions.
            kernel = operator.kernel
            bound = max(1.,float(kernel.double().abs().sum()))
            files.require(math.isfinite(bound) and bound<=1.001,'Expected normalized native blur kernel')
            self.factor = bound**-2

    def lift(self, measurement):
        if self.axes is None:
            return self.factor*self.view.transpose(measurement)
        self.counts['sr_separable_lifts'] = self.counts.get('sr_separable_lifts',0)+1
        ay,ax = self.axes
        return (ay@measurement.double()@ax.T).to(measurement.dtype)

    def observable(self, value):
        return self.lift(self.view.H(value))

    def weak(self, value):
        return value-self.observable(value)

    def check(self, like):
        import torch

        generator = torch.Generator(device=like.device).manual_seed(145000)
        a,b = [torch.randn(like.shape,device=like.device,dtype=like.dtype,generator=generator) for _ in range(2)]
        ha = self.view.H(a)
        results = {}
        if self.axes is not None:
            ay,ax = self.forward_axes
            relative = float((ha.double()-ay@a.double()@ax.T).norm()/ha.double().norm())
            files.require(relative<=2e-6,'Separable SR model differs from native H')
            results['separable_H_relative_l2'] = relative
        ra,rb = self.observable(a),self.observable(b)
        symmetry = abs(_dot(a,rb)-_dot(ra,b))/max(1.,float(a.norm()*b.norm()))
        files.require(symmetry<=2e-6 and _dot(a,ra)>=0 and _dot(a,a-ra)>=0,
                      'Observable and weak components must be symmetric PSD')
        results['symmetry_error'] = symmetry
        if self.task in ('inpainting','super_resolution'):
            leakage = float(self.view.H(a-ra).double().norm()/ha.double().norm())
            files.require(leakage<=2e-6,'Exact null projection leaks into observation')
            results['null_relative_leakage'] = leakage
        return results


def controller(geometry, wave, arm, sigma, counts, trace):
    import torch

    files.require(arm in ARMS,'Unknown A145 arm')
    memory = deque(maxlen=2)

    def remember(clean):
        memory.append(clean.detach())

    def refine(q, residual, alpha, scalar, t, clean):
        scale = math.sqrt(alpha)
        anchor = scalar['gamma']*q
        d = geometry.view.H(anchor/scale)
        e = scalar['weight']*residual.double().reshape_as(d)-d.double()
        files.require(len(memory)==2,'Missing causal prediction history')
        z = (temporal.temporal_target(clean,memory,'TEMPORAL') if arm=='TEMPORAL_WEAK'
             else torch.zeros_like(clean))
        remember(clean)
        weak_rhs = geometry.weak(z) if arm=='TEMPORAL_WEAK' else z
        image_rhs = geometry.lift(e.to(q.dtype))+weak_rhs
        support = wave.analysis(q)!=0
        counts['module_dwt_analysis'] += 1
        weights = graph.edge_weights(clean,sigma)
        counts['graph_builds'] += 1

        def project(value):
            counts['module_dwt_analysis'] += 1
            return wave.analysis(value)*support

        def synthesize(value):
            counts['module_dwt_synthesis'] += 1
            return wave.synthesis(value)

        def normal_image(value):
            # The observable and weak Hessians sum to I in all three geometries.
            counts['graph_actions'] += 1
            return (1+graph.RIDGE)*value+temporal.STRENGTH*graph.laplacian(value,weights)

        rhs = project(image_rhs)
        coeffs,path,relative,stop = graph.conjugate_gradient(
            lambda p:project(normal_image(synthesize(p))),rhs,_dot,
            temporal.late.depth.MAX_UPDATES,temporal.late.depth.GRAD_RTOL)
        u = synthesize(coeffs) if len(path)>1 else torch.zeros_like(q)
        delta = .5*_dot(u,normal_image(u))-_dot(u,image_rhs)
        files.require(math.isfinite(delta) and delta<=2e-5*max(1e-12,_dot(image_rhs,image_rhs)),
                      'Applied A145 increment increased its frozen objective')
        trace.append(dict(t=t,updates=len(path)-1,stop=stop,relative_gradient=relative,
                          objective_delta=delta,weak_target_energy=_dot(z,weak_rhs)))
        return anchor+scale*u

    return remember,refine
