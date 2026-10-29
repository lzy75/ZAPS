"""Required Torch tests; missing dependency fails without skip."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from utils import shrink_multitask_runtime as runtime
from utils import noise_signal_shrink_runtime as parent
from utils import historical_ffhq_runtime as sr
from utils.a67_model_freeze import frozen_network
from utils.paper_clock_runtime import counters,snapshot_equal
from utils.test_paper_clock_runtime import TinyDiffusion,Linear
class RuntimeTests(unittest.TestCase):
    def case(self,task='super_resolution'):
        return dict(task=task,seed=1226,sigma=.05,mask_seed=2001226)
    def test_strong_sr_equivalence_and_no_input_backward(self):
        dm=TinyDiffusion();dm.model.weight.requires_grad_(True);y=torch.full((1,3,16,16),.15);outputs={};entries=[]
        with frozen_network(dm.model):
            for arm in ('BASE','NULL','SHRINK'):
                with patch.object(torch.autograd,'grad',side_effect=AssertionError('No input VJP')),counters(dm.model) as count:
                    output,detail=runtime.execute(dm,Linear(),y,self.case(),arm,size=16,level=1)
                self.assertEqual((count['forwards'],count['optimizer_backward_completed'],count['input_vjp_attempts']),(300,10,0))
                self.assertIsNone(dm.model.weight.grad);outputs[arm]=output;entries.append(detail['entry'])
            old,_=parent.execute(dm,Linear(),y,1226,'SHRINK',.05,size=16,level=1)
        torch.testing.assert_close(outputs['BASE'],outputs['NULL'],rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(outputs['SHRINK'],old,rtol=1e-6,atol=1e-6)
        self.assertTrue(all(snapshot_equal(entries[0],e)['passed'] for e in entries[1:]))
        self.assertTrue(dm.model.weight.requires_grad)
    def test_mask_statistics_use_only_observed_rgb_entries_and_rng_unchanged(self):
        c=self.case('inpainting');before=torch.get_rng_state().clone();op=runtime.make_operator(c,'cpu',size=16)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        x=torch.ones(1,3,16,16);active=runtime.informative(x,op)
        self.assertEqual(active.numel(),int(op._mask_cache.sum())*3)
        residual=2*x;residual[~op._mask_cache.bool().expand_as(x)]=1e5
        self.assertTrue(torch.equal(runtime.informative(residual,op),2*active))
    def test_blur_and_mask_adjoint_contracts(self):
        for task in ('gaussian_deblur','motion_deblur','inpainting'):
            kernel=np.zeros((61,61),np.float32);kernel[30,30]=1
            op=runtime.make_operator(self.case(task),'cpu',size=16,kernel=kernel)
            x=torch.randn(1,3,16,16);y=torch.randn_like(x)
            torch.testing.assert_close((op.H(x)*y).sum(),(x*op.transpose(y)).sum(),rtol=2e-5,atol=2e-5)
    def test_inpainting_complete_runtime_and_null_replay(self):
        dm=TinyDiffusion();c=self.case('inpainting');op=runtime.make_operator(c,'cpu',size=16);y=op.H(torch.full((1,3,16,16),.15))
        with frozen_network(dm.model):
            output,detail=runtime.execute(dm,op,y,c,'BASE',size=16,level=1)
            with counters(dm.model) as count:
                copied,_=runtime.replay(dm,op,y,c,detail['entry'],'NULL',size=16,level=1)
            self.assertEqual(count['forwards'],30)
            torch.testing.assert_close(output,copied,rtol=1e-6,atol=1e-6)
            candidate,_=runtime.execute(dm,op,y,c,'SHRINK',size=16,level=1)
            self.assertTrue(torch.isfinite(candidate).all())
if __name__=='__main__':unittest.main()
