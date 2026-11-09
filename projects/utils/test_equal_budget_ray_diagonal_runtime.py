import types,unittest
import torch
from modules.zaps_algorithm import ZAPS
from modules.wavelet import OrthogonalDWT2D
from utils import equal_budget_ray_diagonal as r,response_target_runtime as rt,overall_direction_audit as old
from utils.test_overall_direction_audit_runtime import model
from utils.test_full_unroll_response_runtime import Linear
from utils.test_mask_stage_response_runtime import Mask
from utils.shrink_multitask import configuration
from utils.paper_clock_runtime import counters,snapshot


class Tests(unittest.TestCase):
    def test_real_response_arms_have_identical_neural_call_contract(self):
        torch.manual_seed(71);m=model();op=Linear();x=torch.zeros(1,3,32,32);q=torch.randn_like(x)*.01;y=torch.ones_like(x)*.2
        z=types.SimpleNamespace(dwt=OrthogonalDWT2D('db4',3));rng=old.rng_state('cpu');counts=[]
        for arm in r.ARMS:
            with counters(m.model) as count:delta,detail,trace=r.action(m,op,y,z,x,q,x,333,309,arm,count)
            self.assertTrue(torch.isfinite(delta).all());self.assertLessEqual(detail['response_update']['final_value'],detail['response_update']['initial_value'])
            self.assertEqual(count['forwards'],5);self.assertEqual(count['input_vjp_attempts'],1);self.assertEqual(count['input_jvp_attempts'],1)
            self.assertEqual(len(trace['response_update']),2);self.assertTrue(rt.rng_equal(rng,old.rng_state('cpu')))
            counts.append((count['forwards'],count['input_vjp_attempts'],count['input_jvp_attempts']))
        self.assertEqual(counts[0],counts[1])

    def test_full30_exact_equal_neural_counts_and_rng(self):
        torch.manual_seed(72);m=model();n=8;mask=(torch.rand(1,1,n,n)>.7).float();op=Mask(mask);y=torch.rand(1,3,n,n)*mask
        z0=ZAPS(m,op,**configuration('inpainting',n,1));entry=snapshot(z0,torch.randn_like(y));expected=[]
        for arm in r.ARMS:
            z=old.make_z(m,op,entry,'inpainting',n,1);old.restore_rng(entry,'cpu');x=entry['initial'].clone()
            with counters(m.model) as count:
                for k in range(30):x,rng,detail,_,_=r.step(m,z,op,y,x,k,arm,count)
            self.assertEqual(count['forwards'],175);self.assertEqual(count['input_vjp_attempts'],29);self.assertEqual(count['input_jvp_attempts'],29)
            self.assertEqual(count['optimizer_backward_attempts'],0);expected.append(rng)
        self.assertTrue(rt.rng_equal(expected[0],expected[1]))


if __name__=='__main__':unittest.main()
