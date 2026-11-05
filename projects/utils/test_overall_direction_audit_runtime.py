"""Required Torch: real next-denoiser derivative, paired replay, exact offline noise split."""
import types,unittest
import torch
from modules.zaps_algorithm import ZAPS
from modules.diffusion_model import _precompute_diffusion_params
from modules.wavelet import OrthogonalDWT2D
from utils import overall_direction_audit as r
from utils.test_mask_stage_response_runtime import Mask,Toy
from utils.shrink_multitask import configuration
from utils.shrink_multitask_runtime import replay
from utils.paper_clock_runtime import counters,snapshot


def model():
 m=types.SimpleNamespace(device='cpu',num_steps=1000,model=Toy(),alphas_cumprod=_precompute_diffusion_params()['alphas_cumprod'])
 m._predict_eps=lambda x,t:m.model(x,t)[:,:3]
 return m


class Tests(unittest.TestCase):
 def test_reference_is_true_derivative_not_detached_eps(self):
  m=model();x=torch.full((1,3,32,32),.03);op=Mask(torch.ones(1,1,32,32));y=torch.full_like(x,.2)
  count=dict(input_vjp_attempts=0);state=dict(uncond=x,next_t=500)
  with counters(m.model) as observed:g,clean=r.physical_reference(m,op,y,state,count)
  ab=m.alphas_cumprod[500];jac=(1-(1-ab).sqrt()*.95)/ab.sqrt()
  torch.testing.assert_close(g,(y-clean)*jac,rtol=2e-5,atol=2e-6)
  self.assertGreater(float((g-(y-clean)/ab.sqrt()).norm()),.01)
  self.assertEqual(observed['forwards'],1);self.assertEqual(count['input_vjp_attempts'],1)
 def test_native_split_replay_and_matched_directions_and_rng(self):
  torch.manual_seed(63);n=32;m=model();mask=(torch.rand(1,1,n,n)>.7).float();op=Mask(mask)
  y=torch.rand(1,3,n,n)*mask;c=dict(task='inpainting',sigma=.1)
  z=ZAPS(m,op,**configuration(c['task'],n,3));entry=snapshot(z,torch.randn_like(y))
  with torch.no_grad():expected,_=replay(m,op,y,c,entry,'NULL',size=n,level=3)
  with counters(m.model) as count:state=r.prefix(m,op,y,entry,c['task'],n,3)
  self.assertEqual(count['forwards'],16)
  physical,_=r.physical_reference(m,op,y,state,dict(input_vjp_attempts=0))
  z=r.make_z(m,op,entry,c['task'],n,3);units,detail=r.directions(state['q'],physical,z.dwt)
  self.assertGreaterEqual(detail['directions']['DIAG_ALIGN']['cosine_to_physical']+1e-5,max(0.,detail['directions']['NATIVE']['cosine_to_physical']))
  paired=None
  for arm in r.ARMS:
   kind,scale=r.specification(arm)
   self.assertIsNotNone(units[kind])
   correction=state['q']*scale if kind=='NATIVE' else units[kind]*(detail['native_norm']*scale)
   self.assertLess(abs(float(correction.double().norm())/(detail['native_norm']*scale)-1),3e-5)
   with counters(m.model) as count:output,_,rng=r.continue_run(m,z,y,state,correction)
   self.assertEqual((count['forwards'],count['input_vjp_attempts'],count['optimizer_backward_attempts']),(14,0,0))
   if paired is None:
    torch.testing.assert_close(output,expected,rtol=1e-6,atol=1e-6);paired=rng
   self.assertTrue(torch.equal(rng['cpu_rng'],paired['cpu_rng']))
  zero,_=r.directions(state['q'],-state['q'],z.dwt)
  self.assertIsNone(zero['DIAG_ALIGN'])
 def test_physical_loss_gain_can_be_pure_noise_fitting(self):
  op=Mask(torch.ones(1,1,4,4));gt=torch.zeros(1,3,4,4);y=torch.ones_like(gt)
  row=r.offline_readout(op,y,gt,gt,gt+.5,gt+.5,y)
  self.assertGreater(row['physical_gain'],0);self.assertLess(row['clean_measurement_gain'],0)
  self.assertAlmostEqual(row['physical_gain'],row['clean_measurement_gain']+row['fitted_noise_gain'],places=10)

if __name__=='__main__':unittest.main()
