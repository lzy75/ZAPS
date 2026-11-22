"""Regression gates for time mapping, offline selection and bounded solver work."""
import unittest
from types import SimpleNamespace
from pathlib import Path
from utils import diagonal_parameter_tuning as policy
from utils import paper_clock_alignment as files


class ProtocolTests(unittest.TestCase):
    def test_map_is_exact_for_dps_and_distinct_for_pigdm(self):
        target=files.grid()
        self.assertEqual(policy.mapped_times(target,range(1000)),target)
        mapped=policy.mapped_times(target,range(0,1000,10))
        self.assertEqual(len(set(mapped)),30)
        self.assertEqual(mapped,sorted(mapped))
        self.assertEqual((mapped[0],mapped[-1]),(0,990))

    def test_native_instrumentation_compiles(self):
        from utils import dps_missing_tasks_runtime as gaussian,dps_multiplier_transfer_runtime as motion,pigdm_noisy_r1_runtime as pi
        for host,rt in (('dps',gaussian),('dps',motion),('pigdm',pi)):
            for arm in ('SCALAR',policy.label(1,1.),policy.label(2,1.5)):
                self.assertTrue(callable(policy.sampler(rt,host,arm,[999,500,0],None)))

    def test_development_selection_is_shared_across_tasks(self):
        rows=[]
        for task in ('gaussian_deblur','motion_deblur'):
            for i in range(3):
                case=dict(key=task+str(i),task=task)
                rows.append(dict(case=case,arm='SCALAR',full_seconds=10.,metrics=dict(psnr=25.,ssim=.8,lpips=.2)))
                for j,p in enumerate(policy.PARAMETERS):
                    rows.append(dict(case=case,arm=policy.label(*p),full_seconds=10.5,
                        metrics=dict(psnr=25.+.1*(j+1),ssim=.81,lpips=.19)))
        result=policy.choose(rows)
        self.assertTrue(result['selected']['qualified'])
        self.assertEqual(result['selected']['arm'],policy.label(2,1.5))


class CudaTests(unittest.TestCase):
    def test_closed_iterations_match_reference_and_improve_objective(self):
        import torch
        from modules.wavelet import OrthogonalDWT2D
        from utils import a118_blur_transfer_runtime as old,transfer_diagonal_runtime as native
        from utils.noise_signal_shrink import select
        self.assertTrue(torch.cuda.is_available(),'Allocated CUDA required')
        class Identity:
            def H(self,x): return x
            def Ht(self,x): return x
        wave=OrthogonalDWT2D(wave='db4',level=3).cuda()
        q=torch.linspace(.01,.2,192,device='cuda:0').reshape(1,3,8,8)
        r=torch.linspace(.2,.05,192,device=q.device).reshape_as(q)
        a=.5; d=q/a**.5
        scalar=select(float((r.double()*d.double()).sum()),float(r.double().square().sum()),
                      float(d.double().square().sum()),r.numel(),.05)
        count={}; native.initialize_counts(count)
        ref=old.refinement(wave,'A118_DIAGONAL',count,[])(Identity(),q,r,a,scalar,0)
        for p in policy.PARAMETERS:
            count={}; native.initialize_counts(count); trace=[]
            out=policy.refine(wave,*p,count,trace)(Identity(),q,r,a,scalar,0)
            self.assertLessEqual(trace[0]['objective_after'],trace[0]['objective_before']+1e-5)
            self.assertLessEqual(count['refinement_Ht'],p[0])
            if p==(1,1.): torch.testing.assert_close(out,ref,rtol=1e-5,atol=1e-6)

    def test_real_native_grids_rng_and_network_budget(self):
        import torch
        from modules.wavelet import OrthogonalDWT2D
        from utils import transfer_diagonal_runtime as native,paper_clock_runtime as clock,native_ray_transfer_runtime as shared
        from utils import dps_missing_tasks_runtime as gaussian,dps_multiplier_transfer_runtime as motion,pigdm_noisy_r1_runtime as pi
        from utils import shrink_multitask_runtime as physical
        from utils.test_matched_response_transfer import Toy
        self.assertTrue(torch.cuda.is_available(),'Allocated CUDA required')
        model=SimpleNamespace(model=Toy().cuda(),device='cuda:0')
        wave=OrthogonalDWT2D(wave='db4',level=3).cuda()
        config=dict(sampler='ddpm',steps=1000,noise_schedule='linear',model_mean_type='epsilon',model_var_type='learned_range',
                    dynamic_threshold=False,clip_denoised=True,rescale_timesteps=False,timestep_respacing='8')
        with native.dps_reference_scope():
            for host,rt,task in (('dps',gaussian,'gaussian_deblur'),('dps',motion,'motion_deblur'),('pigdm',pi,'super_resolution')):
                kernel=torch.zeros(61,61);kernel[30,30]=1
                case=dict(task=task,seed=123,sigma=.05)
                op=pi.Operator(task,'cuda:0',size=8) if host=='pigdm' else physical.make_operator(case,'cuda:0',size=8,kernel=kernel.numpy())
                y=op.H(torch.ones(1,3,8,8,device='cuda:0')*.1)
                reference=None
                times=[999,571,143,0] if host=='dps' else [750,500,250,0]
                for arm,p in [('BASE',None),('SCALAR',None)]+[(policy.label(*p),p) for p in policy.PARAMETERS]:
                    with clock.counters(model.model) as count:
                        native.initialize_counts(count); trace=[]
                        apply=policy.refine(wave,*p,count,trace) if p else None
                        run=policy.sampler(rt,host,arm,times,apply)
                        if host=='pigdm':
                            out,detail=run('gate',model,op,y,case,'BASE' if arm=='BASE' else 'SHRINK',count,size=8,test_steps=4)
                        else:
                            out,detail=run(model,op,y,case,'DPS' if arm=='BASE' else 'SHRINK',config,Path(__file__).resolve().parents[2]/'DPS',count,size=8)
                        final=shared.rng()
                    self.assertEqual(count['forwards'],4 if host=='pigdm' else 8)
                    self.assertEqual(count['forwards'],count['input_vjp_attempts'])
                    self.assertEqual(count.get('input_jvp_attempts',0),0)
                    self.assertTrue(bool(torch.isfinite(out).all()))
                    if p: self.assertEqual([r['t'] for r in trace],sorted(times,reverse=True))
                    if reference is None: reference=final
                    self.assertTrue(torch.equal(final['cpu'],reference['cpu']))
                    self.assertTrue(torch.equal(final['cuda'][0],reference['cuda'][0]))


if __name__=='__main__': unittest.main()
