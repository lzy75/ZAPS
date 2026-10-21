import unittest
from utils.fixed_grid_response_feedback import select,stage_profile


class Tests(unittest.TestCase):
    def test_known_minima_and_direction_ablation(self):
        self.assertEqual(select(2.,4.,1.)['gamma'],2.)
        self.assertEqual(select(-2.,4.,1.)['gamma'],0.)
        self.assertEqual(select(-2.,4.,1.,True)['gamma'],2.)
        self.assertEqual(select(0.,0.,1.)['gamma'],1.)

    def test_no_current_image_in_stage_reference(self):
        p={'a1':dict(image_id='a',S=[{'gamma':0.}]*30),
           'b1':dict(image_id='b',S=[{'gamma':2.}]*30),
           'c1':dict(image_id='c',S=[{'gamma':2.}]*30)}
        self.assertEqual(stage_profile(p,'a'),[2.]*30)
        p['a1']['S']=[{'gamma':1.}]*30
        self.assertEqual(stage_profile(p,'a'),[2.]*30)

    def test_sensor_identity_linear_problem(self):
        try:import torch
        except ImportError:self.skipTest('Torch unavailable; run on server')
        from types import SimpleNamespace
        from utils.fixed_grid_response_feedback import sensor
        z=SimpleNamespace(dm=SimpleNamespace(alphas_cumprod=torch.ones(1)),A=SimpleNamespace(H=lambda x:x))
        result=sensor(z,torch.tensor([2.]),torch.tensor([0.]),torch.tensor([1.]),0,0,identity=True)
        self.assertEqual(result['gamma'],2.)
        self.assertAlmostEqual(result['cosine'],1.)
        self.assertAlmostEqual(result['relative_response'],.5)


if __name__=='__main__':unittest.main()
