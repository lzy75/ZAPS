import copy
import unittest
from utils import report_cached_direction as audit


def fixture():
    rows = [dict(step=k, t=999-999*k//29, noise_u=(k/29)**2, raw_x0_clip_fraction=.01,
                 psnr=20.+k*.1, lpips=.3-k*.001, mean_increment_rms=.1,
                 cosine_clean=None if k<2 else k/30., cosine_latent=None if k==0 else k/30.,
                 cosine_mean=None if k==0 else k/30., prior_guidance_cosine=-.8 if k<20 else .2,
                 noise_over_mean=1.) for k in range(30)]
    return dict(experiment='cached_conditional_mean_direction', status='complete', method_nfe=0,
                diagnostic_nfe=0, model_loaded=False, rows=rows,
                gates={k: dict(passed=True) for k in ('cached_steps', 'noise_draw_suffix', 'residual_and_clean_angles')})


class ReportTests(unittest.TestCase):
    def test_failed_and_changed_grid_rejected(self):
        p = fixture(); audit.validate(p)
        changed = copy.deepcopy(p); changed['status'] = 'failed'
        with self.assertRaises(ValueError): audit.validate(changed)
        changed = copy.deepcopy(p); changed['rows'][5]['noise_u'] = 0.
        with self.assertRaises(ValueError): audit.validate(changed)

    def test_common_support_and_no_gap_bridging(self):
        rows = fixture()['rows']
        for r in rows: r['raw_x0_clip_fraction'] = 1.
        for k in (0, 5, 10, 29): rows[k]['raw_x0_clip_fraction'] = .01
        report = audit.paired_report(rows)
        self.assertEqual(report['all_common']['n'], 27)
        self.assertEqual(report['low_clip_common_exploratory']['source_steps'], [5, 10])
        for a in report['all_common']['associations'].values(): self.assertEqual(a['n'], 27)

    def test_response_rate_divides_by_next_interval(self):
        rows = fixture()['rows']
        # With q=u^2, its exact finite-interval slope is u_k+u_(k+1).
        # Set x to this slope, so Pearson must be 1 under the intended indexing.
        for k in range(29):
            for mode in ('mean', 'clean', 'latent'):
                rows[k]['cosine_'+mode] = (rows[k]['noise_u']+rows[k+1]['noise_u'])/2
            rows[k]['psnr'] = rows[k]['noise_u']**2
        rows[-1]['psnr'] = rows[-1]['noise_u']**2
        r = audit.paired_report(rows)['all_common']['associations']['mean_vs_next_psnr_per_u']
        self.assertAlmostEqual(r['pearson'], 1.)

    def test_component_summary_preserves_phase_counts(self):
        result = audit.components(fixture()['rows'])
        self.assertEqual(sum(v['points'] for v in result.values()), 30)
        self.assertEqual(result['high']['prior_guidance_cosine_median'], -.8)
        self.assertEqual(result['low']['opposing_count'], 0)


if __name__ == '__main__': unittest.main()
