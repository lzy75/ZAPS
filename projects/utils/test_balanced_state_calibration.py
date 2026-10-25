import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from utils import balanced_state_calibration as rules


class BalancedCalibrationTests(unittest.TestCase):
    def test_objective_is_invariant_to_units_and_offsets_and_keeps_endpoints(self):
        values={g:dict(loss=(g-8.)**2,risks={'STATE':(g-4.)**2}) for g in rules.GAMMAS}
        norm=rules.normalization(values)
        scaled={g:dict(loss=1000*v['loss']+22.,risks={'STATE':.01*v['risks']['STATE']+7.}) for g,v in values.items()}
        changed=rules.normalization(scaled)
        for g in rules.GAMMAS:
            self.assertAlmostEqual(rules.objective(values[g],norm),rules.objective(scaled[g],changed),places=11)
        self.assertEqual(rules.objective(values[1.],norm),0.)
        self.assertAlmostEqual(rules.objective(values[2.],norm,0.),(values[2.]['loss']-values[1.]['loss'])/norm['loss_scale'])
        with self.assertRaises(TypeError):rules.objective(values[1.],norm,ground_truth=[])
        with self.assertRaises(ValueError):rules.objective(values[1.],norm,.75)
        def evaluate(g):
            v=dict(loss=(g-8.)**2,risks={'STATE':(g-4.)**2})
            return dict(loss=v['loss'],objective=rules.objective(v,norm))
        choice=rules.control.continuous_choice({g:rules.objective(v,norm) for g,v in values.items()},20.,evaluate)
        self.assertGreater(choice['gamma'],4.);self.assertLess(choice['gamma'],8.)

    def test_fixed_time_action_turn_is_not_time_curvature(self):
        linear=rules.response_geometry([0.,0.],[1.,0.],[2.,0.])
        self.assertEqual(linear['adjacent_action_increment_cosine'],1.)
        self.assertEqual(linear['normalized_second_difference'],0.)
        bend=rules.response_geometry([0.,0.],[1.,0.],[1.,1.])
        self.assertAlmostEqual(bend['adjacent_action_increment_cosine'],0.)
        self.assertAlmostEqual(bend['normalized_second_difference'],np.sqrt(2.)/2.)
        self.assertIsNone(rules.response_geometry([0.],[0.],[0.])['adjacent_action_increment_cosine'])

    def test_finite_development_selection_preserves_quality_and_reports_no_candidate(self):
        rows=[]
        for _ in range(5):
            rows.append(dict(evaluation={name:dict(delta=dict(psnr=value,ssim=.01,lpips=-.01))
                for name,value in (('PHYSICAL',.2),('MIX',.3),('STATE',.25),('TASK_FIXED',.28))}))
        result=rules.select_weight(rows);self.assertEqual(result['chosen_lambda'],.5)
        for r in rows:r['evaluation']['MIX']['delta']['lpips']=.1
        self.assertEqual(rules.select_weight(rows)['chosen_lambda'],1.)
        for r in rows:
            for name in ('PHYSICAL','STATE'):r['evaluation'][name]['delta']['ssim']=-.1
        self.assertIsNone(rules.select_weight(rows)['chosen_lambda'])
        json.dumps(result,allow_nan=False)

    def test_real_torch_counter_retains_failed_attempt_cost(self):
        try:import torch
        except ImportError:self.skipTest('Torch absent on Mac; mandatory without skips on server')
        from types import SimpleNamespace
        from utils.diag_balanced_state_calibration import counted
        class Score(torch.nn.Module):
            def forward(self,x,t):return x*.5
        model=SimpleNamespace(model=Score());r=dict(prior_recorded_nfe=7)
        before=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'run.json'
            with self.assertRaisesRegex(RuntimeError,'budget exhausted'):
                with counted(model,path,r):
                    model.model(torch.ones(1,2),torch.tensor([218]))
                    model.model(torch.ones(1,2),torch.tensor([218]))
            saved=json.loads(path.read_text());self.assertEqual(saved['status'],'failed')
            self.assertEqual(saved['calls'],[218]);self.assertEqual(saved['nfe']+saved['prior_recorded_nfe'],8)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
