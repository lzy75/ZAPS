"""A2c next-time pairing, quality tradeoffs, image-cluster accounting and common noise."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from utils import diag_one_step_guidance_scale as d

try:
    import torch
except ImportError:
    torch = None


class ContractTests(unittest.TestCase):
    def test_terminal_neighbor_is_evaluated_at_t0_not_source_t14(self):
        rows = [dict(t=1000-i, t_prev=999-i, parameter_index=29-i) for i in range(30)]
        rows[28].update(t=14, t_prev=0)
        rows[29]['t'] = 0
        self.assertEqual(d.next_row(rows, dict(step=28, t=14))['t'], 0)
        with self.assertRaises(ValueError): d.next_row(rows, dict(step=29, t=0))
        rows[29]['parameter_index'] = 1
        with self.assertRaises(ValueError): d.next_row(rows, dict(step=28, t=14))

    def test_joint_candidate_rejects_perceptual_or_physical_tradeoff(self):
        base = dict(loss=2., psnr=20., ssim=.6, lpips=.3)
        floors = {k: 0. for k in base}
        good = dict(loss=1.9, psnr=20.2, ssim=.61, lpips=.29)
        self.assertTrue(d.effect(good, base, floors)['joint_local_candidate'])
        for key, value in (('lpips', .31), ('ssim', .59), ('loss', 2.1)):
            bad = dict(good, **{key: value})
            self.assertFalse(d.effect(bad, base, floors)['joint_local_candidate'])
        self.assertFalse(d.effect(base, base, floors)['quality_pareto'])

    def test_measurement_floor_prevents_numerical_gain_claim(self):
        base = dict(loss=2., psnr=20., ssim=.6, lpips=.3)
        floors = dict(loss=.01, psnr=.01, ssim=.001, lpips=.001)
        self.assertFalse(d.effect(dict(base, psnr=20.05), base, floors)['joint_local_candidate'])

    def fixture(self):
        records = []
        for image in ('00020', '00021', '00022'):
            for seed in (1000, 2000):
                effect = dict(delta=dict(loss=-.1, psnr=.2, ssim=.01, lpips=-.01), joint_local_candidate=True)
                records.append(dict(status='complete', case=dict(key=f'{image}_s{seed}', image_id=image),
                                    method_nfe=0, diagnostic_nfe=20,
                                    points=[dict(source_t=t, effects={'0.5': copy.deepcopy(effect), '0.0': copy.deepcopy(effect)}) for t in d.TIMES]))
        return records

    def test_two_seeds_are_not_two_image_clusters_and_incomplete_is_not_success(self):
        records = self.fixture()
        summary = d.summarize(records)
        self.assertEqual(summary['diagnostic_nfe'], 120)
        self.assertEqual(summary['layers'][0]['image_clusters'], 3)
        self.assertEqual(summary['layers'][0]['images_both_seeds_joint'], 3)
        self.assertEqual(summary['classification'], 'LOCAL_HALF_DOSE_CANDIDATE_ONLY')
        self.assertEqual(d.summarize(records[:1])['classification'], 'INCOMPLETE_PILOT')
        for r in records:
            if r['case']['image_id'] in ('00020','00021'):
                for p in r['points']: p['effects']['0.5']['joint_local_candidate'] = False
        self.assertEqual(d.summarize(records)['classification'], 'LOCAL_HALF_DOSE_NOT_SUPPORTED')

    def test_failed_cost_is_retained_but_not_a_scientific_case(self):
        records = self.fixture()[:1]
        records.append(dict(status='failed', method_nfe=0, diagnostic_nfe=7))
        summary = d.summarize(records)
        self.assertEqual(summary['completed_cases'], 1)
        self.assertEqual(summary['diagnostic_nfe'], 27)

    def test_completed_case_checksum_prevents_silent_reuse_of_edited_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            d.WRITE(root/'plan.json', {})
            key='00020_s1000'; case=root/key; case.mkdir()
            record=self.fixture()[0]
            record.update(plan_sha256=d.digest(root/'plan.json'), gates={'test':dict(passed=True)}, calls=[{}]*20)
            d.WRITE(case/'run.json',record)
            (case/'run.sha256').write_text(d.digest(case/'run.json'))
            self.assertIn(key,d.completed(root,dict(source_cases={key:None})))
            record['diagnostic_nfe']=19; d.WRITE(case/'run.json',record)
            with self.assertRaises(ValueError): d.completed(root,dict(source_cases={key:None}))


@unittest.skipIf(torch is None, 'PyTorch unavailable; server must run common-noise tensor tests')
class TensorTests(unittest.TestCase):
    def test_branch_changes_only_current_guidance_same_prior_same_noise(self):
        prior=torch.tensor([.2,.3],dtype=torch.float64)
        noise=torch.tensor([.01,-.04],dtype=torch.float64)
        q=torch.tensor([.05,-.1],dtype=torch.float64)
        original=prior+q+noise; saved=original.clone()
        for gamma in (0.,.5,1.):
            branch=d.shifted_input(original,q,gamma)
            self.assertTrue(torch.allclose(branch,prior+gamma*q+noise,atol=1e-15,rtol=0))
        self.assertTrue(torch.equal(original,saved))
        self.assertNotEqual(d.shifted_input(original,q,1.).data_ptr(), original.data_ptr())

    def test_unregistered_scale_rejected(self):
        with self.assertRaises(ValueError): d.shifted_input(torch.ones(2),torch.ones(2),.75)


if __name__ == '__main__':
    unittest.main()
