"""Required real legacy checkpoint regression; Torch missing is a failure."""
import importlib.util
from pathlib import Path
import unittest
import torch
from utils import current_score_geometry as algebra
from utils import current_score_runtime_r2 as runtime


class LegacyCheckpointTests(unittest.TestCase):
    def test_original_checkpoint_input_vjp_preserves_weights_and_flags(self):
        path=Path(__file__).resolve().parents[2]/'DPS/guided_diffusion/nn.py'
        spec=importlib.util.spec_from_file_location('a50r2_original_checkpoint_nn',path)
        original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight=torch.nn.Parameter(torch.tensor([[.7,.2],[.1,.9]],dtype=torch.float64))
            def forward(self,x):
                return original.checkpoint(lambda z:z@self.weight.T,(x,),self.parameters(),True)
        for initial in (True,False):
            layer=Layer();layer.weight.requires_grad_(initial);before=layer.weight.detach().clone()
            x=torch.tensor([[.2,-.3]],dtype=torch.float64,requires_grad=True)
            v=torch.tensor([[1.,2.]],dtype=torch.float64)
            with runtime.input_vjp_parameters(layer):
                self.assertTrue(layer.weight.requires_grad)
                u=algebra.input_vjp(layer(x),x,v)
                torch.testing.assert_close(u,v@before)
                self.assertIsNone(layer.weight.grad)
                self.assertTrue(torch.equal(before,layer.weight.detach()))
            self.assertEqual(layer.weight.requires_grad,initial)
            self.assertIsNone(layer.weight.grad)


if __name__=='__main__':unittest.main()
