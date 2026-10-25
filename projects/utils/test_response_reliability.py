import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from utils import response_reliability as rules


def values():
    result={g:dict(loss=10.+(g-6.)**2,risks={'STATE':.1*(g-6.)**2}) for g in (0.,.5,1.,1.5,2.,3.,4.,5.)}
    # The original small-action model is optimistic only outside its measured bank.
    result[3.]['loss']=35.-.9*16.
    result[4.]['loss']=35.-.7*21.
    result[5.]['loss']=35.-.625*24.
    return result


class ReliabilityTests(unittest.TestCase):
    def test_optimistic_surrogate_is_separated_from_best_local_objective(self):
        result=rules.choose(values(),5.,0.)
        self.assertEqual(result['gamma'],dict(LOCAL=5.,TRUST=3.))
        row=next(r for r in result['readouts'] if r['gamma']==5.)
        self.assertAlmostEqual(row['rho'],.625)
        self.assertFalse(row['reliable']);json.dumps(result,allow_nan=False)
        with self.assertRaises(TypeError):rules.choose(values(),5.,0.,ground_truth={})

    def test_exact_model_keeps_action_and_is_unit_scale_invariant(self):
        v=values()
        for g in v:v[g]['loss']=10.+(g-6.)**2
        result=rules.choose(v,5.,.5)
        self.assertEqual(result['gamma'],dict(LOCAL=5.,TRUST=5.))
        scaled={g:dict(loss=x['loss']*100.,risks={'STATE':x['risks']['STATE']*.003}) for g,x in v.items()}
        self.assertEqual(rules.choose(scaled,5.,.5)['gamma'],result['gamma'])

    def test_nonfinite_rejection_physical_guard_and_fallback(self):
        v=values();v[3.]=dict(loss=None,risks={});v[4.]['loss']=100.;v[5.]['loss']=100.
        result=rules.choose(v,5.,0.)
        self.assertEqual(result['gamma'],dict(LOCAL=2.,TRUST=2.))
        self.assertEqual(rules.choose({},1.,1.)['gamma'],dict(LOCAL=1.,TRUST=1.))
        self.assertEqual(rules.choose({},5.,1.,False)['gamma'],dict(LOCAL=1.,TRUST=1.))
        for g in v:v[g]['risks']={}
        rules.choose(v,5.,0.)  # A physical task is not gated by a covariance failure.

    def test_geometry_uses_actual_nonuniform_action_spacing(self):
        gs=rules.candidates(9.001)
        direction=np.arange(1.,65.)
        clean={g:(100.+g*direction).astype(np.float64) for g in gs}
        geo=rules.geometry(gs,clean,.01)
        self.assertEqual(len(geo['scales']),2)
        for v in geo['scales']:
            self.assertTrue(v['resolved']);self.assertAlmostEqual(v['cosine'],1.)
            self.assertLess(v['normalized_slope_change'],1e-12)
        self.assertEqual(rules.geometry([1.],{1.:direction},.01)['scales'],[])
        with self.assertRaises(ValueError):rules.candidates(float('nan'))

    def test_true_torch_counter_charges_failed_attempt_and_preserves_rng(self):
        try:import torch
        except ImportError:self.skipTest('No Mac Torch; mandatory without skips on server')
        from types import SimpleNamespace
        from utils.diag_response_reliability import counted
        class Score(torch.nn.Module):
            def forward(self,x,t):return x*.5
        model=SimpleNamespace(model=Score());record=dict(prior_recorded_nfe=19)
        rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'run.json'
            with self.assertRaisesRegex(RuntimeError,'cumulative case budget exhausted'):
                with counted(model,path,record):
                    model.model(torch.ones(2),torch.tensor([218]));model.model(torch.ones(2),torch.tensor([218]))
            saved=json.loads(path.read_text())
            self.assertEqual(saved['status'],'failed');self.assertEqual(saved['nfe'],1)
            self.assertEqual(saved['nfe']+saved['prior_recorded_nfe'],rules.CASE_CAP)
            self.assertEqual(saved['calls'],[218]);self.assertTrue(torch.equal(rng,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
