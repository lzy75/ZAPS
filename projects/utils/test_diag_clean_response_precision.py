import unittest
from types import SimpleNamespace as S
import numpy as np
from utils import diag_clean_response_precision as d
try:
    import torch
except ImportError:
    torch=None


class Legacy:
    def __init__(self):
        self.backends=S(cuda=S(matmul=S(allow_tf32=True)),cudnn=S(allow_tf32=True));self.value='high'
    def get_float32_matmul_precision(self):return self.value
    def set_float32_matmul_precision(self,value):
        self.value=value;self.backends.cuda.matmul.allow_tf32=value!='highest'


class PrecisionTests(unittest.TestCase):
    def test_legacy_restoration_after_exception(self):
        fake=Legacy();saved=d.precision_state(fake)
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with d.ieee(fake) as current:
                self.assertEqual(current['api'],'legacy');self.assertFalse(current['matmul']);self.assertFalse(current['cudnn'])
                raise RuntimeError('deliberate')
        self.assertEqual(d.precision_state(fake),saved)

    def test_new_api_only_and_exact_restore(self):
        fake=S(backends=S(fp32_precision='ieee',cuda=S(matmul=S(fp32_precision='tf32')),
                          cudnn=S(fp32_precision='tf32',conv=S(fp32_precision='tf32'))))
        saved=d.precision_state(fake)
        with d.ieee(fake) as current:
            self.assertEqual(current['api'],'per_backend');self.assertEqual(set(current['flags'].values()),{'ieee'})
        self.assertEqual(d.precision_state(fake),saved)

    def test_gate_not_weakened_and_native_replay_is_required(self):
        native=dict(usable=False,directions={'orthogonal':{'relative_vector_spread':.6}})
        full=dict(usable=True,directions={'orthogonal':{'relative_vector_spread':.1}})
        self.assertEqual(d.classify(native,full,True,True),'IEEE_POLICY_IMPROVES_METROLOGY_THIS_CASE')
        self.assertEqual(d.classify(native,full,False,True),'NATIVE_RESPONSE_NOT_REPRODUCED')
        self.assertEqual(d.classify(native,full,True,False),'NO_POLICY_CHANGE_ALREADY_IEEE')
        full['usable']=False
        self.assertEqual(d.classify(native,full,True,True),'IEEE_FP32_NOT_SUFFICIENT')

    def test_vector_replay_uses_derivatives_not_only_large_center_norm(self):
        def payload(delta):
            return dict(pairs=[dict(direction=name,eta=eta,input={'effective_span':2*eta},
                plus={'clean':np.array([1+eta*delta,1.])},minus={'clean':np.array([1-eta*delta,1.])}) for name in ('q','orthogonal') for eta in (.01,.02)])
        self.assertAlmostEqual(d.vector_replay_error(payload(1),payload(1)),0.)
        self.assertGreater(d.vector_replay_error(payload(1),payload(1.2)),.05)


@unittest.skipIf(torch is None,'Server Torch policy control required; no UNet loaded')
class TorchTests(unittest.TestCase):
    def test_actual_policy_restores_and_does_not_use_rng(self):
        before=d.precision_state(torch);rng=torch.get_rng_state().clone()
        with d.ieee(torch) as actual:
            if actual['api']=='legacy':
                self.assertFalse(actual['matmul']);self.assertFalse(actual['cudnn'])
            else:self.assertEqual(set(actual['flags'].values()),{'ieee'})
        self.assertEqual(d.precision_state(torch),before);self.assertTrue(torch.equal(rng,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
