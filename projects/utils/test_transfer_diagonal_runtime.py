"""Allocated-CUDA gates for preserving native loops and physical adjoints."""
import math
import unittest
from types import SimpleNamespace
from pathlib import Path
import torch
from utils import transfer_diagonal_runtime as control
from utils import paper_clock_runtime as clock
from utils import native_ray_transfer_runtime as shared
from utils import noise_signal_shrink as policy
from utils import shrink_multitask_runtime as physical
from utils.current_score_runtime_r2 import input_vjp_parameters


class Toy(torch.nn.Module):
    def forward(self, x, t):
        return torch.cat((.05*x,torch.zeros_like(x)),1)


def unchanged(op, clean, q, residual, alpha, sigma, old=None):
    if old is None:
        d=op.H(q/math.sqrt(alpha)).double()
        r=residual.double()
        old=policy.select(float((r*d).sum()),float(r.square().sum()),float(d.square().sum()),r.numel(),sigma)
    return old['gamma']*q, old


class Gates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError('Allocated CUDA and Torch required; no numerical gate skips')

    def test_physical_adjoint_and_fixed_objective_descent(self):
        from modules.wavelet import OrthogonalDWT2D
        from utils import pigdm_noisy_r1_runtime as pigdm
        for task in ('super_resolution','gaussian_deblur','motion_deblur','inpainting'):
            for compact in (False,True):
                case=dict(task=task,mask_seed=123)
                kernel=[[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]]
                op=(pigdm.Operator(task,'cuda:0',size=32,seed=123,kernel=kernel)
                    if compact else physical.make_operator(case,'cuda:0',size=32,kernel=kernel))
                count=dict(refinement_H=0,refinement_Ht=0,module_dwt_analysis=0,
                           module_dwt_synthesis=0,module_adjoint_actions=0)
                view=control.LinearView(op,count)
                x=torch.randn(1,3,32,32,device='cuda:0')
                hx=view.H(x); y=torch.randn_like(hx)
                torch.testing.assert_close((hx*y).sum(),(x*view.transpose(y)).sum(),rtol=3e-4,atol=3e-4)
                w=SimpleNamespace(dwt=OrthogonalDWT2D(wave='db4',level=3).cuda())
                trace=[]
                delta,row=control.feedback(w,count,trace)(op,x,.01*x,y,.4,.05)
                self.assertTrue(bool(torch.isfinite(delta).all()))
                self.assertLessEqual(row['proxy_final'],row['proxy_initial']+1e-3)
                self.assertEqual(len(trace),1)

    def test_pigdm_instrumentation_preserves_scalar_rng_and_calls(self):
        from utils import pigdm_noisy_r1_runtime as rt
        model=SimpleNamespace(model=Toy().cuda(),device='cuda:0')
        case=dict(seed=123,sigma=.05)
        op=rt.Operator('super_resolution','cuda:0',size=8)
        y=torch.zeros(1,3,2,2,device='cuda:0');results=[]
        for run,arm in ((rt.run,'SHRINK'),(control.instrument(rt,unchanged,'pigdm'),'BASE')):
            with shared.counters(model.model) as count:
                out,detail=run('test',model,op,y,case,arm,count,size=8,test_steps=4)
            results.append((out,shared.rng(),count))
        torch.testing.assert_close(results[0][0],results[1][0],rtol=1e-5,atol=1e-5)
        self.assertTrue(torch.equal(results[0][1]['cuda'][0],results[1][1]['cuda'][0]))
        self.assertEqual(results[0][2]['calls'],results[1][2]['calls'])

    def test_dps_instrumentation_preserves_both_scalar_entrypoints(self):
        from utils import dps_multiplier_transfer_runtime as original
        from utils import dps_missing_tasks_runtime as missing
        model=SimpleNamespace(model=Toy().cuda(),device='cuda:0')
        config=dict(sampler='ddpm',steps=1000,noise_schedule='linear',model_mean_type='epsilon',
                    model_var_type='learned_range',dynamic_threshold=False,clip_denoised=True,
                    rescale_timesteps=False,timestep_respacing='8')
        for rt,task in ((original,'super_resolution'),(missing,'inpainting')):
            op=physical.make_operator(dict(task=task,mask_seed=123),'cuda:0',size=8)
            x=torch.full((1,3,8,8),.1,device='cuda:0');y=op.H(x)
            case=dict(task=task,mask_seed=123,seed=123,sigma=.05);results=[]
            with input_vjp_parameters(model.model):
                for run in (rt.run,control.instrument(rt,unchanged,'dps')):
                    with clock.counters(model.model) as count:
                        out,detail=run(model,op,y,case,'SHRINK',config,Path(__file__).resolve().parents[2]/'DPS',count,size=8)
                    results.append((out,shared.rng(),count))
            torch.testing.assert_close(results[0][0],results[1][0],rtol=2e-5,atol=2e-5)
            self.assertTrue(torch.equal(results[0][1]['cuda'][0],results[1][1]['cuda'][0]))
            self.assertEqual(results[0][2]['calls'],results[1][2]['calls'])


if __name__=='__main__':
    unittest.main()
