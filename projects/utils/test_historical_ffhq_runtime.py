"""REQUIRED Torch checks. No skip when Torch is unavailable on server."""
import unittest
from unittest.mock import patch
import torch
from utils import historical_ffhq_runtime as engine
from utils import paper_clock_runtime as common
from utils import current_score_geometry as algebra
from utils.current_score_runtime_r2 import input_vjp_parameters
from utils.test_paper_clock_runtime import TinyDiffusion
from modules.zaps_algorithm import ZAPS


class NumericTests(unittest.TestCase):
    def execute(self,arm,sensed=None):
        dm=TinyDiffusion();op=engine.make_operator('cpu',size=16)
        y=torch.full((1,3,4,4),.15)
        with input_vjp_parameters(dm.model),common.counters(dm.model) as count:
            out,details=engine.execute(dm,op,y,1003,arm,count,size=16,level=1,sensed=sensed)
        engine.validate_counts(count,arm)
        self.assertIsNone(dm.model.weight.grad)
        self.assertFalse(dm.model.weight.requires_grad)
        return out,details,count

    def test_model_and_unit_coordinates_give_same_covariance_response(self):
        r=torch.tensor([[.12,-.45,.38]],dtype=torch.float64)
        a,g1=algebra.proximal_vector(lambda x:x,lambda x:x,r,.4,.05,rtol=1e-10)
        b,g2=algebra.proximal_vector(lambda x:.5*x,lambda x:.5*x,.5*r,.4,.025,rtol=1e-10)
        self.assertTrue(g1['passed'] and g2['passed'])
        torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)

    def test_independent270_prefix_and_null_replay_equal_core(self):
        out,first,c1=self.execute('LEGACY_IEEE')
        new,second,c2=self.execute('RESPONSE_300',engine.fallback_sensor)
        self.assertTrue(common.snapshot_equal(first['snapshot'],second['snapshot'])['passed'])
        torch.testing.assert_close(out,new,rtol=1e-6,atol=1e-6)
        self.assertEqual((c1['forwards'],c2['forwards']),(300,300))
        self.assertEqual(c2['input_vjp_attempts'],0)

    def test_independent30_never_optimizes_or_loads_learned_arrays(self):
        with (patch.object(ZAPS,'optimize',side_effect=AssertionError('Independent30 must not learn')),
              patch.object(torch,'load',side_effect=AssertionError('No learned-state input'))):
            out,details,count=self.execute('RESPONSE_30')
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(count['forwards'],30)
        self.assertGreater(count['input_vjp_attempts'],0)
        self.assertLessEqual(count['input_vjp_attempts'],30)
        self.assertEqual(count['optimizer_backward_attempts'],0)
        self.assertTrue(all(r['linear_units'].startswith('L=H ') for r in details['trace']))

    def test_sensor_passes_true_adjoint_not_legacy_mapping_to_cg(self):
        from utils import current_score_geometry as target
        dm=TinyDiffusion();op=engine.make_operator('cpu',size=16)
        cfg=engine.rules.config();cfg.update(img_size=16,level=1)
        z=ZAPS(dm,op,**cfg)
        from utils.fixed_count_state_positions import Positions,noise_u
        positions=Positions(noise_u(dm.alphas_cumprod.numpy()),engine.rules.grid()[::-1])
        original=target.proximal_vector;seen=[]
        def check(linear,adjoint,residual,alpha,sigma,**kw):
            probe=torch.ones(1,3,4,4)
            torch.testing.assert_close(adjoint(probe),op.base.transpose(probe))
            seen.append(True)
            return original(linear,adjoint,residual,alpha,sigma,**kw)
        with input_vjp_parameters(dm.model),patch.object(target,'proximal_vector',side_effect=check):
            engine.sensor(z,torch.full((1,3,4,4),.15),torch.zeros(1,3,16,16),999,916,
                          positions,.05,1.)
        self.assertEqual(seen,[True])


if __name__=='__main__':unittest.main()
