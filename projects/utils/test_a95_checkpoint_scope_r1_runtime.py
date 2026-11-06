"""Mandatory Torch regression: legacy Function under the actual ROOT path alias."""
import importlib.util,unittest
from pathlib import Path
import torch
from utils import a95_checkpoint_scope_r1 as r


class Tests(unittest.TestCase):
 def test_root_alias_true_jvp_preserves_primal_parameters_and_binding(self):
    spec=importlib.util.spec_from_file_location('a95r1_legacy',r.REPO/'DPS/guided_diffusion/nn.py')
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    ns=dict(torch=torch,CheckpointFunction=legacy.CheckpointFunction)
    source='def checkpoint(f, inputs, params, flag):\n return CheckpointFunction.apply(f,len(inputs),*(tuple(inputs)+tuple(params))) if flag else f(*inputs)\n'
    exec(compile(source,str(r.REPO/'guided_diffusion/nn.py'),'exec'),ns)
    exec(compile('class Layer(torch.nn.Module):\n def __init__(self):\n  super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.4))\n def forward(self,x):\n  return checkpoint(lambda a:self.weight*a*a,(x,),self.parameters(),True)\n',str(r.REPO/'guided_diffusion/unet.py'),'exec'),ns)
    model=ns['Layer']();original=ns['checkpoint'];weight=model.weight.detach().clone()
    x=torch.tensor([.2,-.3]);v=torch.tensor([.7,.9])
    with torch.no_grad():expected=model(x)
    with torch.no_grad(),r.forward_ad_checkpoint_scope(model) as count:
        primal,jv=torch.func.jvp(model,(x,),(v,));self.assertEqual(count,1)
    torch.testing.assert_close(primal,expected,rtol=0,atol=0)
    torch.testing.assert_close(jv,2*weight*x*v)
    self.assertIs(ns['checkpoint'],original);self.assertTrue(torch.equal(weight,model.weight));self.assertIsNone(model.weight.grad)

if __name__=='__main__':unittest.main()
