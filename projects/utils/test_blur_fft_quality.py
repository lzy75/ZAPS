"""A62 identity binding and real Torch toy response gates. No UNet or skip."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from utils import diag_blur_fft_quality as d


class BindingTests(unittest.TestCase):
    def test_new_bindings_restore_original_a58(self):
        before = (d.old.ARMS,d.old.admitted,d.old.item,d.old.pcg_field,d.CURRENT)
        with d.bindings({'x':1},{'key':'toy'}):
            self.assertEqual(d.old.ARMS,d.ARMS)
            self.assertIs(d.old.pcg_field,d.response_field)
        self.assertEqual(before,(d.old.ARMS,d.old.admitted,d.old.item,d.old.pcg_field,d.CURRENT))


class TorchTests(unittest.TestCase):
    def test_actual_fft_physical_target_and_current_vjp(self):
        import torch
        import torch.nn.functional as F
        torch.set_num_threads(2)
        k=torch.tensor([[0.,.1,0.],[.2,.4,.1],[0.,.2,0.]],dtype=torch.float64).reshape(1,1,3,3)
        op=SimpleNamespace(kernel=k,H=lambda x:F.conv2d(x,k,padding=1),
            transpose=lambda x:F.conv2d(x,k.flip((-2,-1)),padding=1))
        x=torch.linspace(-.4,.5,25).reshape(1,1,5,5); y=torch.ones_like(x)*.2
        def measurement(raw):
            return F.conv2d((raw+1)/2,k.float(),padding=1)
        z=SimpleNamespace(dm=SimpleNamespace(alphas_cumprod={1:.1,0:.9}),A=SimpleNamespace(H=measurement))
        def raw_mean(dm,value,t,graph):
            inp=value.detach().clone().requires_grad_(True)
            return inp,1.2*inp+.05*inp.square()
        field=lambda z,y,raw,t,pos:dict(q=torch.ones_like(raw)*.02,clean=raw)
        p=dict(cg_rtol=1e-5,hard_iteration_ceiling=2048,seconds_per_solve=30,spatial_check_steps=[0])
        positions=SimpleNamespace(nominal=[1,0]); b=[]
        eye=torch.eye(25,dtype=torch.float64)
        H=torch.stack([op.H(e.reshape(1,1,5,5)).flatten() for e in eye],dim=1)
        residual=y-measurement(1.2*x+.05*x.square()); eta=.05**2*(.1**.5)/.9
        w=torch.linalg.solve(.25*H@H.T+eta*torch.eye(25,dtype=torch.float64),residual.double().flatten())
        v=((.5*(.1**.5)/.9)*H.T@w).reshape_as(x).float()
        K=d.old.current.algebra.skip_coefficient(.1,.9)
        with TemporaryDirectory() as tmp,patch.object(d,'ROOT',Path(tmp)), \
             patch.object(d.old.current.base,'raw_mean',raw_mean),patch.object(d.old.current.base,'field_from_mean',field):
            with d.bindings(p,{'key':'toy'}):
                physical,_,row=d.response_field(z,op,y,x,1,0,positions,.05,'V_FFT',lambda:b.append(1))
                self.assertEqual(b,[])
                self.assertTrue(row['resolved']); self.assertTrue(row['cg']['spatial_check']['passed'])
                torch.testing.assert_close(physical['q'],K*v,rtol=2e-4,atol=1e-5)
                current,_,row=d.response_field(z,op,y,x,1,0,positions,.05,'U_FFT',lambda:b.append(1))
                self.assertEqual(len(b),1)
                u=(1.2+.1*x)*v; target=K*torch.linalg.vector_norm(v)*u/torch.linalg.vector_norm(u)
                torch.testing.assert_close(current['q'],target,rtol=2e-4,atol=1e-5)
        self.assertFalse(x.requires_grad)


if __name__ == '__main__':unittest.main()
