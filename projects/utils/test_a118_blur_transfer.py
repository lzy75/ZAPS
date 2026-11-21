"""Validate the closed update and the native sampler boundary before GPU experiments."""
from pathlib import Path
from types import SimpleNamespace
import unittest

from utils import a118_blur_transfer_runtime as runtime


class ProtocolTests(unittest.TestCase):
    def test_bounded_line_fraction(self):
        self.assertEqual(runtime.step_fraction(4.,0.),1.)
        self.assertEqual(runtime.step_fraction(0.,1.),0.)
        self.assertEqual(runtime.step_fraction(2.,4.),.25)

    def test_compiles_all_native_entries(self):
        from utils import dps_missing_tasks_runtime as gaussian
        from utils import dps_multiplier_transfer_runtime as motion
        from utils import pigdm_noisy_r1_runtime as pigdm
        for host, original in (('dps',gaussian),('dps',motion),('pigdm',pigdm)):
            for arm in runtime.ARMS:
                self.assertTrue(callable(runtime.instrument(original,host,arm,299,None)))


class NativeCudaTests(unittest.TestCase):
    def test_no_extra_network_work_and_paired_prefix(self):
        import torch
        from utils import paper_clock_runtime as clock
        from utils import native_ray_transfer_runtime as shared
        from utils import shrink_multitask_runtime as physical
        from utils import transfer_diagonal_runtime as native
        from utils import dps_missing_tasks_runtime as gaussian
        from utils import dps_multiplier_transfer_runtime as motion
        from utils import pigdm_noisy_r1_runtime as pigdm
        from utils.test_matched_response_transfer import Toy, IdentityWavelet

        self.assertTrue(torch.cuda.is_available(), 'Allocated CUDA required')
        model = SimpleNamespace(model=Toy().cuda(),device='cuda:0')
        wavelet = SimpleNamespace(dwt=IdentityWavelet())
        config = dict(sampler='ddpm',steps=1000,noise_schedule='linear',model_mean_type='epsilon',
                      model_var_type='learned_range',dynamic_threshold=False,clip_denoised=True,
                      rescale_timesteps=False,timestep_respacing='8')
        entries = [('dps',gaussian,'gaussian_deblur'),('dps',motion,'motion_deblur'),
                   ('pigdm',pigdm,'super_resolution')]
        with native.dps_reference_scope():
            for host,original,task in entries:
                case = dict(task=task,seed=123,sigma=.05)
                # Use SR for the toy PiGDM since its 61x61 blur kernel requires size >=61.
                kernel = torch.zeros(61,61); kernel[30,30] = 1
                op = (pigdm.Operator(task,'cuda:0',size=8) if host=='pigdm'
                      else physical.make_operator(case,'cuda:0',size=8,kernel=kernel.numpy()))
                y = op.H(torch.full((1,3,8,8),.1,device='cuda:0'))
                references = None
                for arm in runtime.ARMS:
                    traces, tail_entry = [], []
                    with clock.counters(model.model) as count:
                        native.initialize_counts(count)
                        first_tail = 285 if host=='dps' else 250
                        def watch(_model,args):
                            if count['calls'][-1]==first_tail:
                                tail_entry.append(args[0].detach().clone())
                        hook = model.model.register_forward_pre_hook(watch)
                        try:
                            refine = runtime.refinement(wavelet,arm,count,traces)
                            run = runtime.instrument(original,host,arm,299,refine)
                            if host=='pigdm':
                                output,detail = run('gate',model,op,y,case,'BASE' if arm=='BASE' else 'SHRINK',
                                                    count,size=8,test_steps=4)
                            else:
                                output,detail = run(model,op,y,case,'DPS' if arm=='BASE' else 'SHRINK',config,
                                                    Path(__file__).resolve().parents[2]/'DPS',count,size=8)
                        finally:
                            hook.remove()
                    n = 4 if host=='pigdm' else 8
                    self.assertEqual(count['forwards'],n)
                    self.assertEqual(count['input_vjp_attempts'],n)
                    self.assertEqual(count.get('input_jvp_attempts',0),0)
                    self.assertEqual(count['calls'],detail['grid'])
                    tail = [t for t in detail['grid'] if t<=299]
                    self.assertEqual([r['t'] for r in traces],tail if arm.endswith('DIAGONAL') else [])
                    self.assertTrue(bool(torch.isfinite(output).all()))
                    self.assertEqual(len(tail_entry),1)
                    final = shared.rng()
                    if references is None:
                        references = tail_entry[0],final
                    else:
                        self.assertTrue(torch.equal(references[0],tail_entry[0]))
                        self.assertTrue(torch.equal(references[1]['cpu'],final['cpu']))
                        self.assertTrue(torch.equal(references[1]['cuda'][0],final['cuda'][0]))
                    if arm=='BOUNDED_DIAGONAL':
                        self.assertTrue(all(r['applied_extra_norm']<=.5*r['scalar_norm']+1e-6 for r in traces))


if __name__=='__main__':
    unittest.main()
