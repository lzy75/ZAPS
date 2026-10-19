"""A2d: matched direction intervention, numerical resolution and cluster decisions."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from utils import diag_direction_refresh as d

try:
    import torch
except ImportError:
    torch = None


class BranchTests(unittest.TestCase):
    def test_only_direction_differs_in_matched_second_stage(self):
        rng = np.random.default_rng(19)
        for _ in range(40):
            prior, noise, q0, qh = rng.normal(size=(4, 2, 3, 5))
            a0, ah = np.sqrt(np.mean(q0*q0)), np.sqrt(np.mean(qh*qh))
            original = prior+q0+noise; saved = original.copy()
            branches = d.branch_inputs(original, q0, qh, a0, ah)
            np.testing.assert_allclose(branches['F'], prior+.5*q0+.5*qh+noise, atol=1e-14)
            np.testing.assert_allclose(branches['M'], prior+.5*q0+.5*(ah/a0)*q0+noise, atol=1e-14)
            np.testing.assert_array_equal(original, saved)
            self.assertFalse(np.shares_memory(branches['C'], original))
            b = np.linalg.norm(qh/np.linalg.norm(qh)-q0/np.linalg.norm(q0))
            g = d.geometry(a0, ah, b, 0.)
            self.assertAlmostEqual(np.sqrt(np.mean((branches['F']-branches['M'])**2)), g['expected_F_M_rms'])

    def test_no_refresh_and_amplitude_only_are_not_direction_evidence(self):
        q0 = np.array([1., 2., -3.]); original = np.array([.2, .3, .4]); a0 = np.sqrt(np.mean(q0*q0))
        for scale in (0., 1., 2.):
            branches = d.branch_inputs(original, q0, scale*q0, a0, scale*a0)
            np.testing.assert_allclose(branches['F'], branches['M'], atol=1e-15)
            if scale == 1.:
                np.testing.assert_array_equal(branches['F'], original)
        self.assertFalse(d.geometry(a0, a0, 0., 0.)['resolved'])
        self.assertIsNone(d.geometry(a0, 0., None, 0.)['chi'])

    def test_noise_floor_and_invalid_directions(self):
        self.assertFalse(d.geometry(1., 1., 1e-5, 2e-6)['resolved'])
        self.assertTrue(d.geometry(1., 1., .1, 1e-6)['resolved'])
        for args in ((0., 1., .1, 0.), (1., -1., .1, 0.), (1., 1., 2.1, 0.), (1., 1., float('nan'), 0.)):
            with self.assertRaises(ValueError): d.geometry(*args)
        with self.assertRaises(ValueError): d.branch_inputs(np.zeros(2), np.ones(2), np.ones(3), 1., 1.)


class DecisionTests(unittest.TestCase):
    def fixture(self):
        records = []
        for image in ('00020', '00021', '00022'):
            for seed in (1000, 2000):
                base = dict(loss=2., psnr=20., ssim=.6, lpips=.3)
                f = dict(loss=1.8, psnr=20.2, ssim=.61, lpips=.29)
                m = dict(loss=1.9, psnr=20.1, ssim=.605, lpips=.295)
                floors = {k: 0. for k in base}
                effects = dict(F_C=d.effect(f, base, floors), M_C=d.effect(m, base, floors), F_M=d.effect(f, m, floors))
                records.append(dict(status='complete', case=dict(key=f'{image}_s{seed}', image_id=image),
                                    method_nfe=0, diagnostic_nfe=30,
                                    points=[dict(source_t=t, effects=copy.deepcopy(effects), geometry=dict(resolved=True),
                                                 shift_resolved=True, joint_direction_candidate=True) for t in d.TIMES]))
        return records

    def test_need_two_images_both_seeds_not_pooled_mean(self):
        records = self.fixture()
        self.assertEqual(d.summarize(records)['classification'], 'LOCAL_DIRECTION_REFRESH_CANDIDATE_ONLY')
        for r in records:
            if r['case']['image_id'] != '00021':
                for p in r['points']: p['joint_direction_candidate'] = False
        result = d.summarize(records)
        self.assertEqual(result['layers'][0]['images_both_seeds_joint'], 1)
        self.assertEqual(result['classification'], 'LOCAL_DIRECTION_REFRESH_NOT_SUPPORTED')
        self.assertEqual(result['diagnostic_nfe'], 180)

    def test_seed_failure_unresolved_shift_and_incomplete_not_success(self):
        records = self.fixture()
        self.assertEqual(d.summarize(records[:1])['classification'], 'INCOMPLETE_PILOT')
        for r in records:
            if r['case']['key'].endswith('2000'):
                for p in r['points']: p['shift_resolved'] = False; p['joint_direction_candidate'] = False
        self.assertEqual(d.summarize(records)['layers'][0]['images_both_seeds_joint'], 0)
        self.assertEqual(d.summarize(records)['classification'], 'LOCAL_DIRECTION_REFRESH_NOT_SUPPORTED')

    def test_quality_or_loss_tradeoff_is_not_joint_gain(self):
        base = dict(loss=2., psnr=20., ssim=.6, lpips=.3); floors = {k:0. for k in base}
        for bad in (dict(base, psnr=20.2, lpips=.31), dict(base, psnr=20.2, loss=2.1)):
            self.assertFalse(d.effect(bad, base, floors)['joint_local_candidate'])

    def test_beating_original_only_does_not_establish_direction_value(self):
        effects=dict(F_C=dict(joint_local_candidate=True), F_M=dict(joint_local_candidate=False))
        self.assertFalse(d.direction_candidate(effects, True))
        effects['F_M']['joint_local_candidate']=True
        self.assertFalse(d.direction_candidate(effects, False))
        self.assertTrue(d.direction_candidate(effects, True))

    def test_failed_calls_retained_and_running_cost_marked_lower_bound(self):
        records = self.fixture()[:1]+[dict(status='failed', method_nfe=0, diagnostic_nfe=7)]
        self.assertEqual(d.summarize(records)['diagnostic_nfe'], 37)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'case').mkdir()
            d.WRITE(root/'case/run.json', dict(status='running', method_nfe=0, diagnostic_nfe=6))
            self.assertTrue(d.stored_summary(root)['cost_is_lower_bound_if_abruptly_interrupted'])
            self.assertEqual(d.stored_summary(root)['diagnostic_nfe'], 6)

    def test_checksum_refuses_edited_case_instead_of_rerunning(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); d.WRITE(root/'plan.json', {}); case=root/'00020_s1000'; case.mkdir()
            r=self.fixture()[0]; r.update(plan_sha256=d.digest(root/'plan.json'), gates={'test':dict(passed=True)}, calls=[{}]*30)
            (case/'branches.pt').write_bytes(b'checksum fixture')
            r['branch_archive']=dict(sha256=d.digest(case/'branches.pt'))
            d.WRITE(case/'run.json', r); (case/'run.sha256').write_text(d.digest(case/'run.json'))
            plan=dict(source_cases={'00020_s1000':None})
            self.assertIn('00020_s1000', d.completed(root, plan))
            r['diagnostic_nfe']=29; d.WRITE(case/'run.json', r)
            with self.assertRaises(ValueError): d.completed(root, plan)


@unittest.skipIf(torch is None, 'PyTorch unavailable locally; tensor checks required on server')
class TensorTests(unittest.TestCase):
    def test_float32_tensor_inputs_preserve_original_and_null_branch(self):
        q=torch.tensor([.05,-.1], dtype=torch.float32); original=torch.tensor([.2,.3],dtype=torch.float32)
        saved=original.clone(); a0=float(q.double().square().mean().sqrt())
        arms=d.branch_inputs(original,q,q,a0,a0)
        self.assertTrue(torch.equal(original,saved))
        for v in arms.values():self.assertTrue(torch.equal(v,original))
        self.assertNotEqual(arms['C'].data_ptr(),original.data_ptr())

    def test_tensor_match_uses_current_q_axis_and_refreshed_norm(self):
        q0=torch.tensor([1.,0.],dtype=torch.float64); qh=torch.tensor([0.,2.],dtype=torch.float64)
        a0=float(q0.square().mean().sqrt());ah=float(qh.square().mean().sqrt())
        arms=d.branch_inputs(torch.zeros_like(q0),q0,qh,a0,ah)
        torch.testing.assert_close(arms['F'],torch.tensor([-.5,1.],dtype=torch.float64))
        torch.testing.assert_close(arms['M'],torch.tensor([.5,0.],dtype=torch.float64))


if __name__=='__main__': unittest.main()
