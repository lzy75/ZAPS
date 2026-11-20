"""A123 gates: exact coverage/budget, immutable reuse, terminal identity and RNG."""
from pathlib import Path
import tempfile
import unittest
from utils import full_diagonal_transfer as rules


class PlanTests(unittest.TestCase):
    def test_all_positions_and_terminal_budget(self):
        for host,n,forwards in (('dps',1000,8992),('pigdm',100,892)):
            self.assertEqual(rules.positions(host),tuple(range(1,n+1)))
            budget=rules.budget(host)
            self.assertEqual(budget['forwards'],forwards)
            self.assertEqual(budget['input_vjp_attempts'],2*n)
            self.assertEqual(budget['input_jvp_attempts'],2*n)
            self.assertEqual(budget['response_value_forwards'],5*n)
            self.assertEqual(budget['network_response_vjp'],n-1)
            self.assertEqual(budget['terminal_operator_vjp'],1)

    def test_completed_reuse_checks_hash_and_does_not_write_parent(self):
        files=rules.sparse.files
        with tempfile.TemporaryDirectory() as tmp:
            original=Path(tmp)/'original.json'; linked=Path(tmp)/'linked.json'
            files.write(original,dict(status='complete',metrics={'psnr':30.}))
            sha=files.digest(original)
            files.write(linked,dict(status='complete',reused=True,source_record=str(original),source_sha256=sha))
            path,row=rules.resolve_record(linked)
            self.assertEqual(path,original); self.assertEqual(files.digest(original),sha)
            files.write(original,dict(status='complete',metrics={'psnr':29.}))
            with self.assertRaises(ValueError): rules.resolve_record(linked)

    def test_failed_record_cannot_be_reused(self):
        files=rules.sparse.files
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'failed.json'; files.write(path,dict(status='failed'))
            with self.assertRaises(ValueError): rules.resolve_record(path)


class FullRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Allocated CUDA required; no numerical gate skips')

    def test_full_native_loops_include_terminal_identity_with_paired_rng(self):
        import torch
        from types import SimpleNamespace
        from utils import matched_response_transfer_runtime as response
        from utils import transfer_diagonal_runtime as control
        from utils import paper_clock_runtime as clock
        from utils import native_ray_transfer_runtime as shared
        from utils import shrink_multitask_runtime as physical
        from utils import pigdm_noisy_r1_runtime as pigdm
        from utils import dps_multiplier_transfer_runtime as dps
        from utils import dps_missing_tasks_runtime as missing
        from utils.current_score_runtime_r2 import input_vjp_parameters
        from utils.test_matched_response_transfer import Toy,IdentityWavelet
        model=SimpleNamespace(model=Toy().cuda(),device='cuda:0')
        config=dict(sampler='ddpm',steps=1000,noise_schedule='linear',model_mean_type='epsilon',
                    model_var_type='learned_range',dynamic_threshold=False,clip_denoised=True,
                    rescale_timesteps=False,timestep_respacing='8')
        entries=(('dps',dps,'super_resolution'),('dps',missing,'inpainting'),
                 ('pigdm',pigdm,'super_resolution'),('pigdm',pigdm,'inpainting'))
        with control.dps_reference_scope(),input_vjp_parameters(model.model):
            for host,rt,task in entries:
                case=dict(task=task,seed=123,mask_seed=123,sigma=.05)
                op=(pigdm.Operator(task,'cuda:0',size=8,seed=123) if host=='pigdm'
                    else physical.make_operator(case,'cuda:0',size=8))
                y=op.H(torch.ones(1,3,8,8,device='cuda:0')*.1)
                n=4 if host=='pigdm' else 8; results=[]
                for full in (False,True):
                    trace=[]
                    with clock.counters(model.model) as count:
                        control.initialize_counts(count); run=rt.run
                        if full:
                            apply=response.controller(model,IdentityWavelet(),'RESPONSE_DIAG',count,tuple(range(1,n+1)),trace)
                            run=response.instrument(rt,apply,host)
                        if host=='pigdm':
                            out,detail=run('gate',model,op,y,case,'SHRINK',count,size=8,test_steps=n)
                        else:
                            out,detail=run(model,op,y,case,'SHRINK',config,
                                Path(__file__).resolve().parents[2]/'DPS',count,size=8)
                        final=shared.rng()
                    self.assertTrue(bool(torch.isfinite(out).all()))
                    results.append((detail,final))
                    if full:
                        self.assertEqual(count['forwards'],n+8*(n-1))
                        self.assertEqual(count['input_vjp_attempts'],2*n)
                        self.assertEqual(count['input_jvp_attempts'],2*n)
                        self.assertEqual(count['response_value_forwards'],5*n)
                        self.assertEqual([r['step'] for r in trace],list(range(1,n+1)))
                        self.assertEqual(trace[-1]['next_time'],-1)
                        self.assertTrue(all(r['diagonal_accepted_loss']<=r['ray_accepted_loss']+1e-8 for r in trace))
                    else:
                        self.assertEqual(count['forwards'],n)
                self.assertEqual(results[0][0]['grid'],results[1][0]['grid'])
                self.assertTrue(torch.equal(results[0][1]['cpu'],results[1][1]['cpu']))
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(results[0][1]['cuda'],results[1][1]['cuda'])))


if __name__=='__main__': unittest.main()
