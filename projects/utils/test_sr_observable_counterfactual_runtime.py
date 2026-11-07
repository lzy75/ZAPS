"""Required Torch CPU: exact CURRENT native Resizer, no empirical transpose."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from utils.historical_ffhq_runtime import make_operator
from utils import sr_observable_counterfactual as r

class Tests(unittest.TestCase):
 def test_native_separable_forward_and_null_invariance_without_legacy_adjoint(self):
  op=make_operator('cpu',size=32);rng=np.random.default_rng(1);x=rng.normal(size=(1,3,32,32))
  with patch.object(op,'transpose',side_effect=RuntimeError('Legacy upsample is not an adjoint')):
   ay,ax=r.native_matrices(op.base.down_sample,x.shape);p=r.Projector(ay,ax)
   np.testing.assert_allclose(p.H(x),op.H(torch.from_numpy(x)).numpy(),rtol=1e-11,atol=1e-11)
   self.assertLess(r.relative(op.H(torch.from_numpy(x).float()).numpy(),p.H(x)),3e-6)
   s=x*.1;d=s+rng.normal(size=x.shape)*.1;images,_,_=r.counterfactuals(s,d,p)
   for arm,ref in [('ROW_ONLY_DIFFERENCE',d),('NULL_ONLY_DIFFERENCE',s)]:np.testing.assert_allclose(op.H(torch.from_numpy(images[arm])).numpy(),p.H(ref),rtol=1e-10,atol=1e-10)
 def test_real_resizer_raw_risk_decomposition_and_projection_rank(self):
  op=make_operator('cpu',size=32);ay,ax=r.native_matrices(op.base.down_sample,(1,3,32,32));p=r.Projector(ay,ax)
  rng=np.random.default_rng(2);x=rng.normal(size=(1,3,32,32));gt=rng.normal(size=x.shape);risk=r.risks(x,gt,p)
  self.assertEqual([v['rank'] for v in p.info],[8,8]);self.assertAlmostEqual(risk['raw_mse'],risk['row_mse']+risk['null_mse'],places=12)

if __name__=='__main__':unittest.main()
