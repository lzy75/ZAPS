import unittest
from utils.audit_trajectory_weight_opportunity import analyze, POINTS


def fixture(same_preference):
    records = []
    for image in range(3):
        for seed in (1, 2):
            metrics = {'BASE': dict(psnr=25., ssim=.8, lpips=.2, loss=1.)}
            gain = .1 if same_preference or image != 0 else -.1
            for _, t in POINTS:
                for f in (.5, 1.5):
                    metrics[f'T{t}_M{f:g}'] = dict(psnr=25.+(1 if f == 1.5 else -1)*gain,
                                                  ssim=.8, lpips=.2, loss=1.)
            records.append(dict(case=dict(image_id=str(image), sampling_seed=seed), metrics=metrics))
    return records


class OpportunityTests(unittest.TestCase):
    def test_shared_optimum_has_zero_selection_headroom(self):
        result = analyze(fixture(True))
        self.assertEqual(result['mean_image_oracle_headroom'], 0.)
        for stage in result['stages'].values():
            self.assertEqual(stage['image_oracle']['counts']['1.5'], 3)
            self.assertEqual(stage['leave_image_out_time']['mean_delta'], stage['hindsight_best_fixed_delta'])

    def test_heterogeneous_preference_has_headroom_and_held_time_excludes_both_seeds(self):
        records = fixture(False)
        before = analyze(records)
        self.assertGreater(before['mean_image_oracle_headroom'], .06)
        choices = [s['per_image']['0']['held_time_factor'] for s in before['stages'].values()]
        for r in records:
            if r['case']['image_id'] == '0':
                for name, m in r['metrics'].items():
                    if name != 'BASE': m['psnr'] += 100.
        after = analyze(records)
        self.assertEqual(choices, [s['per_image']['0']['held_time_factor'] for s in after['stages'].values()])


if __name__ == '__main__':
    unittest.main()
