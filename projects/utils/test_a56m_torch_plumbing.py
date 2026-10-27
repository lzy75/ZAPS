"""Required real Torch meter regression; no CUDA/model claim, missing Torch fails."""
from contextlib import nullcontext
import unittest
import torch
from utils import a56m_imagenet_runtime as r


class MeterTests(unittest.TestCase):
    def test_parameter_meter_preserves_adam_loss_and_restores_methods(self):
        before_backward=torch.Tensor.backward
        outcomes=[]
        for metered in (False,True):
            state=dict(optimization_loss_backwards=0,optimizer_steps=0)
            value=torch.nn.Parameter(torch.tensor([.2,-.4],dtype=torch.float64))
            scope=r.parameter_loss_meter(torch,state) if metered else nullcontext()
            before_step=torch.optim.Adam.step
            with scope:
                opt=torch.optim.Adam([value],lr=.01)
                loss=(value-1).square().sum();loss.backward();opt.step()
                outcomes.append((value.detach().clone(),value.grad.detach().clone(),float(loss.detach())))
            self.assertIs(torch.Tensor.backward,before_backward)
            if metered:
                self.assertIs(torch.optim.Adam.step,before_step)
                self.assertEqual(state,dict(optimization_loss_backwards=1,optimizer_steps=1))
        for a,b in zip(*outcomes):
            if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
            else:self.assertEqual(a,b)

    def test_parameter_meter_restores_on_exception(self):
        before=(torch.Tensor.backward,torch.optim.Adam.step)
        state=dict(optimization_loss_backwards=0,optimizer_steps=0)
        with self.assertRaisesRegex(RuntimeError,'stop'):
            with r.parameter_loss_meter(torch,state):raise RuntimeError('stop')
        self.assertEqual(before,(torch.Tensor.backward,torch.optim.Adam.step))


if __name__=='__main__':unittest.main()
