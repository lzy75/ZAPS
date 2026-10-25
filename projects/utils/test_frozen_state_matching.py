import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from utils import frozen_state_matching as rules


def cases_and_choices():
    cases=[dict(key=task+'_'+image,task=task,image_id=image) for task in rules.REFERENCE for image in rules.IMAGES]
    choices={c['key']:dict(PHYSICAL=2.,ISO=3.,STATE=float(i%5+4)) for i,c in enumerate(cases)}
    return cases,choices


class FrozenStateMatchingTests(unittest.TestCase):
    def test_cyclic_reassignment_preserves_action_set_without_self_or_cross_task(self):
        cases,choices=cases_and_choices();result=rules.allocation(cases,choices,rules.REFERENCE)
        by_key={c['key']:c for c in cases}
        for task in rules.REFERENCE:
            group=[c for c in cases if c['task']==task]
            self.assertEqual(sorted(result[c['key']]['gamma']['STATE'] for c in group),
                             sorted(result[c['key']]['gamma']['SHUFFLE'] for c in group))
            for c in group:
                row=result[c['key']];self.assertNotEqual(row['donor'],c['key'])
                self.assertEqual(by_key[row['donor']]['task'],task)
                self.assertEqual(row['gamma']['SHUFFLE'],choices[row['donor']]['STATE'])
                self.assertEqual(row['gamma']['TASK_FIXED'],rules.REFERENCE[task])
        self.assertEqual(result,rules.allocation(list(reversed(cases)),choices,rules.REFERENCE))
        with self.assertRaises(TypeError):rules.allocation(cases,choices,rules.REFERENCE,ground_truth={})
        json.dumps(result,allow_nan=False)

    def test_incomplete_or_changed_cohort_and_reference_rejected(self):
        cases,choices=cases_and_choices()
        with self.assertRaises(ValueError):rules.allocation(cases[:-1],choices,rules.REFERENCE)
        changed=dict(rules.REFERENCE,super_resolution=8.)
        with self.assertRaises(ValueError):rules.allocation(cases,choices,changed)
        bad=[dict(c) for c in cases];bad[0]['image_id']='00337'
        with self.assertRaises(ValueError):rules.allocation(bad,choices,rules.REFERENCE)

    def test_baseline_gain_alone_cannot_pass_state_matching_screen(self):
        cases,_=cases_and_choices();records=[]
        for c in cases:
            evaluation={n:dict(delta=dict(psnr=.2,ssim=.003,lpips=-.004,loss=-.01)) for n in rules.POLICIES}
            evaluation['C']['delta']=dict(psnr=0.,ssim=0.,lpips=0.,loss=0.)
            records.append(dict(case=c,gamma={},evaluation=evaluation))
        self.assertFalse(any(g['state_matching_candidate'] for g in rules.task_summary(records)))
        for r in records:r['evaluation']['STATE']['delta']['psnr']=.25
        self.assertTrue(all(g['state_matching_candidate'] for g in rules.task_summary(records)))
        for r in records:r['evaluation']['STATE']['delta']['ssim']=-.01
        self.assertFalse(any(g['state_matching_candidate'] for g in rules.task_summary(records)))

    def test_actual_torch_meter_keeps_failed_calls_in_stage_budget(self):
        try:import torch
        except ImportError:self.skipTest('Torch absent on Mac; mandatory on server without skips')
        from types import SimpleNamespace
        from utils.diag_frozen_state_matching import counted
        network=torch.nn.Linear(2,2,bias=False)
        class Score(torch.nn.Module):
            def forward(self,x,t):return network(x)
        model=SimpleNamespace(model=Score());before=torch.get_rng_state().clone()
        r=dict(phase='choose',prior_recorded_nfe=52)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'run.json'
            with self.assertRaisesRegex(RuntimeError,'budget exhausted'):
                with counted(model,path,r):
                    model.model(torch.ones(1,2),torch.tensor([218]))
                    model.model(torch.ones(1,2),torch.tensor([218]))
            saved=json.loads(path.read_text());self.assertEqual(saved['status'],'failed')
            self.assertEqual(saved['nfe'],1);self.assertEqual(saved['calls'],[218])
            self.assertEqual(saved['nfe']+saved['prior_recorded_nfe'],53)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
