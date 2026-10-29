"""Server required; Torch absence fails, never skips."""
import unittest
from unittest.mock import patch
import torch
from utils import noise_signal_shrink_runtime as runtime
from utils import noise_feasible_runtime as historical
from utils import historical_ffhq_runtime as sr
from utils.a67_model_freeze import frozen_network
from utils.paper_clock_runtime import counters,snapshot_equal
from utils.test_paper_clock_runtime import TinyDiffusion,Linear

class RuntimeTests(unittest.TestCase):
    def test_original_prefix_null_first_and_no_network_backwards(self):
        dm=TinyDiffusion();dm.model.weight.requires_grad_(True);y=torch.full((1,3,16,16),.15)
        entries=[];outputs={}
        with frozen_network(dm.model):
            for arm in ('BASE','NULL','FIRST_NOISE','SHRINK','FIXED_FIRST','MISMATCH_SHRINK'):
                profile=[.4]*30 if arm.startswith(('FIXED','MISMATCH')) else None
                with patch.object(torch.autograd,'grad',side_effect=AssertionError('No input VJP')),counters(dm.model) as count:
                    out,row=runtime.execute(dm,Linear(),y,1000,arm,.05,profile,size=16,level=1)
                self.assertEqual((count['forwards'],count['optimizer_backward_completed'],count['input_vjp_attempts']),(300,10,0))
                self.assertTrue(torch.isfinite(out).all());self.assertIsNone(dm.model.weight.grad)
                entries.append(row['entry']);outputs[arm]=out
                if profile:self.assertEqual([v['gamma'] for v in row['trace']],profile)
            reference,_=historical.execute(dm,Linear(),y,1000,'FIRST_NOISE',.05,size=16,level=1)
        self.assertTrue(all(snapshot_equal(entries[0],e)['passed'] for e in entries[1:]))
        torch.testing.assert_close(outputs['BASE'],outputs['NULL'],rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(outputs['FIRST_NOISE'],reference,rtol=1e-6,atol=1e-6)
        self.assertTrue(dm.model.weight.requires_grad)
    def test_actual_sr_replay_null_and_shrink(self):
        dm=TinyDiffusion();y=torch.full((1,3,4,4),.15)
        with frozen_network(dm.model):
            output,row=runtime.execute(dm,sr.make_operator('cpu',size=16),y,1000,'BASE',.05,size=16,level=1)
            for arm in ('NULL','SHRINK'):
                with counters(dm.model) as count:
                    replay,trace=runtime.replay(dm,sr.make_operator('cpu',count,size=16),y,row['entry'],arm,.05,size=16,level=1)
                self.assertEqual(count['forwards'],30);self.assertEqual(count['input_vjp_attempts'],0)
                if arm=='NULL':torch.testing.assert_close(output,replay,rtol=1e-6,atol=1e-6)
                else:self.assertEqual(len(trace),30)

if __name__=='__main__':unittest.main()
