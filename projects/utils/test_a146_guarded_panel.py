"""A146 guard algebra and native suffix scope."""
from types import SimpleNamespace
import unittest

import torch

from utils import diag_a146_guarded_panel as trial
from utils.pigdm_noisy_runtime import Operator


class GuardTests(unittest.TestCase):
    def test_maximum_feasible_fraction(self):
        f = trial.rules.maximum_fraction
        self.assertEqual(f(0,0),1.)
        self.assertEqual(f(0,2),0.)
        self.assertEqual(f(2,0),0.)
        self.assertEqual(f(-4,1),1.)
        self.assertAlmostEqual(f(-.25,2),.25)
        for h,k,slack in ((1.,2.,.1),(-.2,2.,.03),(1e-12,0.,1e-13),(0.,1.,1e-12)):
            beta = f(h,k,slack)
            self.assertGreaterEqual(beta,0)
            self.assertLessEqual(beta,1)
            self.assertAlmostEqual(h*beta+.5*k*beta**2,slack,places=13)

    def test_minmod_uses_coefficient_sign_agreement(self):
        a = torch.tensor([.2,-1.,.5,0.,1.])
        b = torch.tensor([2.,-.3,-1.,1.,0.])
        torch.testing.assert_close(trial.rules.minmod(a,b),torch.tensor([.2,-.3,0.,0.,0.]))

    def apply(self,op,task,arm):
        counts,trace = dict(graph_builds=0,graph_actions=0),[]
        trial.native.initialize_counts(counts)
        q = torch.ones(1,3,8,8)
        clean = torch.full_like(q,-.1)
        geom = trial.panel.method.Geometry(task,op,q,counts)
        wave = SimpleNamespace(analysis=lambda a:a,synthesis=lambda a:a)
        remember,refine = trial.rules.controller(geom,wave,arm,.05,counts,trace)
        physical = torch.zeros_like(q)
        remember(physical)
        remember(physical)
        residual = -.1*op.H(q)
        out = refine(q,residual,1.,dict(weight=1.),0,clean,physical)
        return out,trace[0]

    def test_exact_mask_preserves_observation(self):
        op = Operator('inpainting','cpu',size=8,seed=139)
        out,row = self.apply(op,'inpainting','GUARDED')
        self.assertEqual(row['beta'],1.)
        self.assertEqual(row['objective_applied'],row['objective_physical'])
        self.assertGreater(float(out[...,~op.mask].mean()),0.)

    def test_weak_blur_can_reject_bad_observable_change(self):
        op = SimpleNamespace(H=lambda x:.5*x,Ht=lambda x:.5*x,kernel=torch.ones(1))
        _,unprotected = self.apply(op,'motion_deblur','UNGUARDED')
        _,protected = self.apply(op,'motion_deblur','GUARDED')
        self.assertGreater(unprotected['objective_applied'],unprotected['objective_physical'])
        self.assertLess(protected['beta'],1e-3)
        self.assertLessEqual(protected['objective_applied'],protected['objective_physical']+protected['tolerance'])

    def test_zaps_keeps_physical_branch_on_full_grid(self):
        _,active,warm = trial.panel.schedule('zaps')
        run = trial.zaps_suffix(None,'GUARDED',.05,active,warm,None,{},[])
        self.assertEqual(run.__globals__['active'],set(active))
        self.assertEqual(run.__globals__['warm'],set(warm))


if __name__=='__main__':
    unittest.main()
