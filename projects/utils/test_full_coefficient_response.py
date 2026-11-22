"""A126 admission: A118 reduction, full-space response descent and native parity."""
import unittest
from pathlib import Path
from types import SimpleNamespace
from utils import full_coefficient_response as policy

class ProtocolTests(unittest.TestCase):
    def test_all_native_entrypoints_compile_without_core_changes(self):
        from utils import dps_missing_tasks_runtime as missing,dps_multiplier_transfer_runtime as original,pigdm_noisy_r1_runtime as pi
        for host,rt in (('dps',missing),('dps',original),('pigdm',pi)):
            for arm in ('SCALAR','RESPONSE_RAY','RESPONSE_DIAG'):
                self.assertTrue(callable(policy.instrument(rt,host,arm,[999,500,0],None)))

    def test_full_coefficient_budget_has_no_gain_cap(self):
        self.assertEqual((policy.RAY_ITERATIONS,policy.DIAGONAL_ITERATIONS,policy.MAX_TRIALS),(1,2,3))
        self.assertFalse(hasattr(policy,'RELATIVE_BOUND'))

class CudaTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.assertTrue(torch.cuda.is_available(),'Allocated CUDA required; no numerical skips')

    def test_identity_response_recovers_the_a118_coefficient_step(self):
        import torch
        from modules.wavelet import OrthogonalDWT2D
        from utils import a118_blur_transfer_runtime as old,transfer_diagonal_runtime as native
        class Identity:
            def H(self,x): return x
            def Ht(self,x): return x
        class Network(torch.nn.Module):
            def forward(self,x): return x
        wave=OrthogonalDWT2D(wave='db4',level=3).cuda()
        q=torch.linspace(.01,.2,3072,device='cuda:0').reshape(1,3,32,32)
        y=torch.linspace(.2,.03,3072,device=q.device).reshape_as(q)
        count={}; native.initialize_counts(count)
        provider=policy.response.Provider(Network(),lambda x:x,Identity(),torch.zeros_like(q),y,0.,count)
        scalar=dict(gamma=.7,weight=1.)
        expected=old.refinement(wave,'A118_DIAGONAL',count,[])(Identity(),q,y,1.,scalar,0)
        actual,detail=policy.minimize(provider,wave,q,.7*q,'diagonal',1,count)
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
        self.assertTrue(detail['iterations'][0]['accepted'])

    def test_full_anisotropic_response_and_nonlinear_objective_descent(self):
        import torch
        from modules.wavelet import OrthogonalDWT2D
        from utils import paper_clock_runtime as clock,transfer_diagonal_runtime as native
        class Network(torch.nn.Module):
            def forward(self,x,t):
                return x*torch.linspace(.2,1.,x.numel(),device=x.device).reshape_as(x)
        class Nonlinear(torch.nn.Module):
            def forward(self,x,t): return torch.tanh(x)*.9+x*.1
        class Identity:
            def H(self,x): return x
        wave=OrthogonalDWT2D(wave='db4',level=3).cuda()
        q=torch.linspace(.02,.1,3072,device='cuda:0').reshape(1,3,32,32); y=torch.ones_like(q)*.4
        for network in (Network().cuda(),Nonlinear().cuda()):
            with clock.counters(network) as count:
                native.initialize_counts(count)
                predict=lambda x:network(x,torch.tensor([499],device=x.device))
                provider=policy.response.Provider(network,predict,Identity(),torch.zeros_like(q),y,.05,count)
                delta,detail=policy.solve(provider,wave,q,.8,'RESPONSE_DIAG',count)
            self.assertLess(detail['diagonal']['iterations'][-1]['after'],detail['ray']['iterations'][-1]['after'])
            for stage in ('ray','diagonal'):
                for row in detail[stage]['iterations']:
                    self.assertLessEqual(row['after'],row['before']+1e-8)
            self.assertTrue(bool(torch.isfinite(delta).all()))
            self.assertLessEqual(count['forwards'],1+3*(2+policy.MAX_TRIALS))
            self.assertLessEqual(count['input_vjp_attempts'],3)
            self.assertLessEqual(count['input_jvp_attempts'],3)

    def test_native_scalar_parity_rng_and_terminal_derivative_accounting(self):
        import torch
        from modules.wavelet import OrthogonalDWT2D
        from utils import pigdm_noisy_r1_runtime as pi,dps_missing_tasks_runtime as missing,dps_multiplier_transfer_runtime as original
        from utils import transfer_diagonal_runtime as native,paper_clock_runtime as clock,native_ray_transfer_runtime as shared
        from utils import shrink_multitask_runtime as physical
        from utils.current_score_runtime_r2 import input_vjp_parameters
        from utils.test_matched_response_transfer import Toy
        model=SimpleNamespace(model=Toy().cuda(),device='cuda:0'); wave=OrthogonalDWT2D(wave='db4',level=3).cuda()
        config=dict(sampler='ddpm',steps=1000,noise_schedule='linear',model_mean_type='epsilon',model_var_type='learned_range',
                    dynamic_threshold=False,clip_denoised=True,rescale_timesteps=False,timestep_respacing='8')
        with native.dps_reference_scope(),input_vjp_parameters(model.model):
            for host,rt,task in (('pigdm',pi,'inpainting'),('dps',missing,'inpainting'),('dps',original,'super_resolution')):
                case=dict(task=task,seed=123,mask_seed=123,sigma=.05)
                op=pi.Operator(task,'cuda:0',size=8,seed=123) if host=='pigdm' else physical.make_operator(case,'cuda:0',size=8)
                y=op.H(torch.ones(1,3,8,8,device='cuda:0')*.1)
                def sample(run,count):
                    if host=='pigdm': return run('gate',model,op,y,case,'SHRINK',count,size=8,test_steps=4)
                    return run(model,op,y,case,'SHRINK',config,Path(__file__).resolve().parents[2]/'DPS',count,size=8)
                grid=[750,500,250,0] if host=='pigdm' else [999,856,714,571,428,285,143,0]
                results=[]
                for run in (rt.run,policy.instrument(rt,host,'SCALAR',grid)):
                    with clock.counters(model.model) as count:
                        native.initialize_counts(count); out,detail=sample(run,count)
                    results.append(out)
                torch.testing.assert_close(results[0],results[1],rtol=0,atol=0)
                baseline_rng=shared.rng()
                for arm in ('RESPONSE_RAY','RESPONSE_DIAG'):
                    with clock.counters(model.model) as count:
                        native.initialize_counts(count); trace=[]
                        active=[grid[1],0]
                        apply=policy.controller(model,wave,arm,count,active,trace)
                        out,detail=sample(policy.instrument(rt,host,arm,active,apply),count)
                    self.assertEqual(len(trace),2)
                    self.assertEqual(trace[-1]['next_time'],-1)
                    n=4 if host=='pigdm' else 8
                    self.assertEqual(count['forwards'],n+count['response_network_forwards'])
                    self.assertEqual(count['input_vjp_attempts'],n+count['response_vjp_attempts'])
                    self.assertGreater(count['terminal_response_vjp'],0)
                    self.assertLess(count['terminal_response_vjp'],count['response_vjp_attempts'])
                    self.assertTrue(torch.equal(shared.rng()['cpu'],baseline_rng['cpu']))
                    self.assertTrue(torch.equal(shared.rng()['cuda'][0],baseline_rng['cuda'][0]))

if __name__=='__main__': unittest.main()
