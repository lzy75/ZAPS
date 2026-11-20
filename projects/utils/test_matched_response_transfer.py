"""CUDA gates: native parity, exact derivatives, gain geometry and nested trials."""
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from utils import matched_response_transfer_runtime as response
from utils import transfer_diagonal_runtime as previous
from utils import paper_clock_runtime as clock
from utils import native_ray_transfer_runtime as shared
from utils import shrink_multitask_runtime as physical
from utils.current_score_runtime_r2 import input_vjp_parameters


class Toy(torch.nn.Module):
    def forward(self,x,t):
        return torch.cat((.05*x,torch.zeros_like(x)),1)


class DiagonalPredictor(torch.nn.Module):
    def forward(self,x,t):
        return x*torch.linspace(.2,.9,x.numel(),device=x.device).reshape_as(x)


class IdentityWavelet:
    def analysis(self,x): return x
    def synthesis(self,x): return x


class IdentityOperator:
    def H(self,x): return x


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError('Allocated CUDA and Torch required; no numerical test skips')

    def test_actual_response_derivatives_and_nested_continuous_trials(self):
        network = DiagonalPredictor().cuda()
        q = torch.linspace(.02,.5,64,device='cuda:0').reshape(1,1,8,8)
        origin = q*.3; y = torch.ones_like(q)*.4
        predict = lambda x:network(x,torch.tensor([499],device=x.device))
        with clock.counters(network) as count:
            provider = response.Provider(network,predict,IdentityOperator(),origin,y,.05,count)
            delta,row = response.solve(provider,IdentityWavelet(),q,.7,'RESPONSE_DIAG',count)
        self.assertEqual(count['forwards'],8)
        self.assertEqual(count['input_vjp_attempts'],1)
        self.assertEqual(count['input_jvp_attempts'],2)
        self.assertLessEqual(row['diagonal_accepted_loss'],row['ray_accepted_loss'])
        self.assertLessEqual(row['diagonal_accepted_loss'],row['legacy_loss']+1e-6)
        self.assertTrue(bool(torch.isfinite(delta).all()))
        self.assertTrue(all(p.grad is None for p in network.parameters()))
        # This linear model admits an exact forward AD check without differences.
        with clock.counters(network) as count:
            provider = response.Provider(network,predict,IdentityOperator(),origin,y,.05,count)
            value,grad = provider.gradient(q)
            primal,jv = provider.jvp(q,torch.ones_like(q))
        weights = torch.linspace(.2,.9,64,device=q.device).reshape_as(q)
        torch.testing.assert_close(jv,weights.reshape(-1),rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(grad,weights*(value-provider.target).reshape_as(q),rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(primal,value)

    def test_multiplicative_gain_geometry_includes_native_coefficients_twice(self):
        q = torch.tensor([.1,.2,.4,.8],device='cuda:0').reshape(1,1,2,2)
        g = torch.tensor([1.,-2.,3.,-4.],device=q.device).reshape_as(q)
        u,v,norm = response.tangent_direction(IdentityWavelet(),q,g,{})
        perpendicular = g-u*(u*g).sum()
        expected = -q.square()*perpendicular
        expected = expected-u*(u*expected).sum()
        torch.testing.assert_close(v,expected/expected.norm(),rtol=1e-5,atol=1e-6)
        self.assertLess(abs(float((u*v).sum())),1e-5)

    def test_zero_gain_gradient_remains_finite_and_has_fixed_query_budget(self):
        network = Toy().cuda()
        q = torch.zeros(1,3,8,8,device='cuda:0')
        predict = lambda x:network(x,torch.tensor([499],device=x.device))[:,:3]
        with clock.counters(network) as count:
            provider = response.Provider(network,predict,IdentityOperator(),q,q,.05,count)
            delta,row = response.solve(provider,IdentityWavelet(),q,0.,'RESPONSE_DIAG',count)
        self.assertEqual(count['forwards'],8)
        self.assertEqual(row['retained_rank'],0)
        self.assertEqual(float(delta.norm()),0.)

    def test_pigdm_native_arithmetic_rng_and_next_raw_predictor(self):
        from utils import pigdm_noisy_r1_runtime as rt
        model = SimpleNamespace(model=Toy().cuda(),device='cuda:0')
        case = dict(seed=123,sigma=.05)
        op = rt.Operator('super_resolution','cuda:0',size=8)
        y = torch.zeros(1,3,2,2,device='cuda:0'); results=[]
        for run in (rt.run,response.instrument(rt,lambda *args:None,'pigdm')):
            with clock.counters(model.model) as count:
                previous.initialize_counts(count)
                out,detail = run('test',model,op,y,case,'SHRINK',count,size=8,test_steps=4)
            results.append((out,shared.rng(),count))
        torch.testing.assert_close(results[0][0],results[1][0],rtol=0,atol=0)
        self.assertTrue(torch.equal(results[0][1]['cuda'][0],results[1][1]['cuda'][0]))
        self.assertEqual(results[0][2]['calls'],results[1][2]['calls'])
        x = torch.ones(1,3,8,8,device='cuda:0')*.5
        diffusion = rt.original_diffusion()
        a = float(diffusion.alpha(torch.tensor([499],device=x.device))[0])
        expected = (x-math.sqrt(1-a)*.05*x)/math.sqrt(a)
        torch.testing.assert_close(response.pigdm_predict(model,diffusion,x,499),expected)

    def test_dps_both_original_entrypoints_preserve_scalar_and_clipped_predictor(self):
        from utils import dps_multiplier_transfer_runtime as original
        from utils import dps_missing_tasks_runtime as missing
        model = SimpleNamespace(model=Toy().cuda(),device='cuda:0')
        config = dict(sampler='ddpm',steps=1000,noise_schedule='linear',model_mean_type='epsilon',
                      model_var_type='learned_range',dynamic_threshold=False,clip_denoised=True,
                      rescale_timesteps=False,timestep_respacing='8')
        with previous.dps_reference_scope(),input_vjp_parameters(model.model):
            for rt,task in ((original,'super_resolution'),(missing,'inpainting')):
                op = physical.make_operator(dict(task=task,mask_seed=123),'cuda:0',size=8)
                y = op.H(torch.full((1,3,8,8),.1,device='cuda:0'))
                case = dict(task=task,mask_seed=123,seed=123,sigma=.05); results=[]
                for run in (rt.run,response.instrument(rt,lambda *args:None,'dps')):
                    with clock.counters(model.model) as count:
                        out,detail = run(model,op,y,case,'SHRINK',config,Path(__file__).resolve().parents[2]/'DPS',count,size=8)
                    results.append((out,shared.rng(),count))
                torch.testing.assert_close(results[0][0],results[1][0],rtol=0,atol=0)
                self.assertTrue(torch.equal(results[0][1]['cuda'][0],results[1][1]['cuda'][0]))
                self.assertEqual(results[0][2]['calls'],results[1][2]['calls'])
                sampler = rt.reference(str(Path(__file__).resolve().parents[2]/'DPS'))[0].create_sampler(**config)
                x = torch.ones(1,3,8,8,device='cuda:0')*10
                clean = response.dps_predict(sampler,model,x,4)
                self.assertLessEqual(float(clean.abs().max()),1.)

    def test_active_hooks_use_next_predictor_and_count_all_extra_queries(self):
        from utils import pigdm_noisy_r1_runtime as pigdm
        from utils import dps_missing_tasks_runtime as dps
        model=SimpleNamespace(model=Toy().cuda(),device='cuda:0')
        config=dict(sampler='ddpm',steps=1000,noise_schedule='linear',model_mean_type='epsilon',
                    model_var_type='learned_range',dynamic_threshold=False,clip_denoised=True,
                    rescale_timesteps=False,timestep_respacing='8')
        with previous.dps_reference_scope(),input_vjp_parameters(model.model):
            for host,rt in (('dps',dps),('pigdm',pigdm)):
                case=dict(task='inpainting',seed=123,mask_seed=123,sigma=.05)
                op=(pigdm.Operator('inpainting','cuda:0',size=8,seed=123) if host=='pigdm'
                    else physical.make_operator(case,'cuda:0',size=8))
                y=op.H(torch.ones(1,3,8,8,device='cuda:0')*.1); trace=[]
                with clock.counters(model.model) as count:
                    previous.initialize_counts(count)
                    apply=response.controller(model,IdentityWavelet(),'RESPONSE_DIAG',count,(2,),trace)
                    run=response.instrument(rt,apply,host)
                    if host=='pigdm':
                        out,detail=run('test',model,op,y,case,'SHRINK',count,size=8,test_steps=4)
                    else:
                        out,detail=run(model,op,y,case,'SHRINK',config,
                            Path(__file__).resolve().parents[2]/'DPS',count,size=8)
                native=4 if host=='pigdm' else 8
                self.assertEqual(count['forwards'],native+8)
                self.assertEqual(count['input_vjp_attempts'],native+1)
                self.assertEqual(count['input_jvp_attempts'],2)
                self.assertEqual(len(trace),1)
                self.assertEqual(trace[0]['next_time'],detail['grid'][2])
                self.assertTrue(bool(torch.isfinite(out).all()))


if __name__=='__main__':
    unittest.main()
