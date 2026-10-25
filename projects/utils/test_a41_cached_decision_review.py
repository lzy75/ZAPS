import copy
import unittest
from utils.audit_ffhq_action_geometry_results import task_review


class DecisionReviewTests(unittest.TestCase):
    def test_cross_task_gain_can_be_uniform_recovery_of_task_action(self):
        zero=dict(psnr=0.,ssim=0.,lpips=0.,loss=0.)
        gain=dict(psnr=.2,ssim=.003,lpips=-.002,loss=-.01)
        def policy(g,delta):
            return dict(step=19,t=641,mean_delta=delta,choices=[dict(image_id=str(i),gamma=g,delta=dict(delta)) for i in range(6)])
        task=dict(task='super_resolution',opportunity=[dict(step=19,best_fixed_gamma=2.,best_fixed_delta=gain)],
            validation=dict(LEAVE_TASK_AND_IMAGE_OUT=dict(policies={'PI':[policy(2.,gain)],'STAGE':[policy(1.,zero)]},
                increments=[dict(step=19,candidate=True)])))
        row=task_review(task)['positions'][0]
        self.assertEqual(row['interpretation'],'UNIFORM_TASK_ACTION')
        self.assertAlmostEqual(row['PI_minus_STAGE']['psnr'],.2)
        self.assertAlmostEqual(row['PI_minus_retrospective_best_fixed']['psnr'],0.)
        self.assertEqual(row['actions_equal_STAGE'],0)
        changed=copy.deepcopy(task);p=changed['validation']['LEAVE_TASK_AND_IMAGE_OUT']['policies']['PI'][0]
        p['choices'][0]['gamma']=1.5
        self.assertEqual(task_review(changed)['positions'][0]['interpretation'],'IMAGE_VARYING_DECISIONS')
        # A source candidate is reported as-is, never promoted to four-task support.
        self.assertTrue(row['source_increment_candidate'])


if __name__=='__main__':unittest.main()
