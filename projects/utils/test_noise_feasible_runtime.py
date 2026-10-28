"""Server-required Torch tests; missing Torch fails instead of skipping."""
import unittest
from unittest.mock import patch
import torch
from utils import noise_feasible_runtime as runtime
from utils import paper_clock_runtime as clock
from utils import historical_ffhq_runtime as strong_runtime
from utils.test_paper_clock_runtime import TinyDiffusion,Linear


class RuntimeTests(unittest.TestCase):
    def run_arm(self,arm):
        dm=TinyDiffusion();op=Linear();y=torch.full((1,3,16,16),.15)
        with clock.counters(dm.model) as count:
            out,details=runtime.execute(dm,op,y,1000,arm,.05,size=16,level=1)
        self.assertEqual((count['forwards'],count['optimizer_backward_completed']),(300,10))
        self.assertEqual(count['input_vjp_attempts'],0);self.assertIsNone(dm.model.weight.grad)
        return dm,op,y,out,details

    def test_null_final_loop_matches_unmodified_core_and_prefix(self):
        _,_,_,a,first=self.run_arm('BASE')
        dm,op,y,b,second=self.run_arm('NULL')
        self.assertTrue(clock.snapshot_equal(first['entry'],second['entry'])['passed'])
        torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)
        with clock.counters(dm.model) as counts:
            replay,_=runtime.replay(dm,op,y,first['entry'],'NULL',.05,size=16,level=1)
        self.assertEqual(counts['forwards'],30)
        torch.testing.assert_close(a,replay,rtol=1e-6,atol=1e-6)

    def test_feedback_preserves_prefix_and_has_no_input_vjp(self):
        _,_,_,_,base=self.run_arm('BASE')
        for arm in ('A33','FIRST_NOISE','NEAREST_NOISE'):
            with patch.object(torch.autograd,'grad',side_effect=AssertionError('No VJP permitted')):
                _,_,_,out,row=self.run_arm(arm)
            self.assertTrue(clock.snapshot_equal(base['entry'],row['entry'])['passed'])
            self.assertTrue(torch.isfinite(out).all());self.assertEqual(len(row['trace']),30)

    def test_actual_sr_operator_null_loop(self):
        dm=TinyDiffusion();y=torch.full((1,3,4,4),.15)
        outputs=[]
        for arm in ('BASE','NULL'):
            out,_=runtime.execute(dm,strong_runtime.make_operator('cpu',size=16),y,1000,arm,.05,size=16,level=1)
            outputs.append(out)
        torch.testing.assert_close(*outputs,rtol=1e-6,atol=1e-6)


if __name__=='__main__':unittest.main()
