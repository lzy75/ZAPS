import json
from pathlib import Path
import tempfile
import unittest
from utils import frozen_four_task_confirmation as rules


def cohort():
    cases=[dict(key=f'{t}_{i:05d}',task=t,image_id=f'{i:05d}') for i in range(900,906) for t in rules.WEIGHTS]
    choices={}
    for c in cases:
        g=2.+int(c['image_id'])-900
        selected=g if rules.WEIGHTS[c['task']]==0. else g+1. if rules.WEIGHTS[c['task']]==1. else g+.5
        choices[c['key']]=dict(PHYSICAL=g,ISO=g-.1,STATE=g+1.,SELECTED=selected)
    return cases,choices


class FourTaskConfirmationTests(unittest.TestCase):
    def test_full_cost_and_exact_endpoint_alias_controls(self):
        cases,choices=cohort()
        self.assertEqual(sum(rules.cap('BASELINE',c['task'])+rules.cap('choose',c['task'])+rules.cap('finish',c['task']) for c in cases),9240)
        fixed={t:8. for t in rules.WEIGHTS};assigned=rules.allocation(cases,choices,fixed,rules.MATCHED)
        for task in rules.WEIGHTS:
            group=[c for c in cases if c['task']==task]
            self.assertEqual(sorted(assigned[c['key']]['gamma']['SELECTED'] for c in group),
                sorted(assigned[c['key']]['gamma']['SHUFFLE'] for c in group))
            for c in group:self.assertNotEqual(assigned[c['key']]['donor'],c['key'])
        bad={k:dict(v) for k,v in choices.items()};key=next(c['key'] for c in cases if c['task']=='gaussian_deblur')
        bad[key]['SELECTED']=99.
        with self.assertRaises(ValueError):rules.allocation(cases,bad,fixed,rules.MATCHED)
        with self.assertRaises(TypeError):rules.allocation(cases,choices,fixed,rules.MATCHED,ground_truth={})

    def test_quality_transfer_and_state_matching_are_separate(self):
        cases,_=cohort();records=[]
        for c in cases:
            ev={name:dict(delta=dict(psnr=.2,ssim=.01,lpips=-.01,loss=-.02)) for name in rules.POLICIES}
            ev['C']['delta']=dict(psnr=0.,ssim=0.,lpips=0.,loss=0.)
            records.append(dict(case=c,gamma={},evaluation=ev))
        groups=rules.summarize(records)
        self.assertTrue(all(g['quality_candidate'] for g in groups));self.assertFalse(any(g['state_matching_candidate'] for g in groups))
        for r in records:r['evaluation']['SELECTED']['delta']['psnr']=.25
        groups=rules.summarize(records);self.assertTrue(all(g['state_matching_candidate'] for g in groups))
        self.assertFalse(next(g for g in groups if g['task']=='gaussian_deblur')['covariance_increment_candidate'])
        for r in records:
            if r['case']['task']=='motion_deblur':r['evaluation']['SELECTED']['delta']['lpips']=.02
        self.assertFalse(next(g for g in rules.summarize(records) if g['task']=='motion_deblur')['quality_candidate'])
        json.dumps(groups,allow_nan=False)

    def test_incomplete_image_task_coverage_is_rejected(self):
        cases,choices=cohort()
        with self.assertRaises(ValueError):rules.allocation(cases[:-1],choices,{t:8. for t in rules.WEIGHTS},rules.MATCHED)

    def test_real_torch_training_counter_does_not_reset_failed_budget(self):
        try:import torch
        except ImportError:self.skipTest('Torch absent on Mac; mandatory without skips on server')
        from types import SimpleNamespace
        from utils.diag_frozen_four_task_confirmation import counted
        class Score(torch.nn.Module):
            def forward(self,x,t):return x*.5
        model=SimpleNamespace(model=Score());r=dict(phase='BASELINE',case={'task':'super_resolution'},prior_recorded_nfe=299)
        before=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'run.json'
            with self.assertRaisesRegex(RuntimeError,'budget exhausted'):
                with counted(model,path,r):
                    model.model(torch.ones(1,2),torch.tensor([999]));model.model(torch.ones(1,2),torch.tensor([999]))
            saved=json.loads(path.read_text());self.assertEqual(saved['status'],'failed')
            self.assertEqual(saved['nfe']+saved['prior_recorded_nfe'],300);self.assertEqual(saved['calls'],[999])
            self.assertTrue(torch.equal(before,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
