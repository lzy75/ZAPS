import copy
import unittest
from utils import diag_fixed_terminal_replication as d
from utils.test_diag_terminal_state_strength import records


class CohortTests(unittest.TestCase):
    def test_recovery_allowed_but_running_original_rejected(self):
        case=dict(key='00021_s2000',image_id='00021',sampling_seed=2000,cohort='development')
        r=dict(status='complete',case={k:v for k,v in case.items() if k!='cohort'},gates=dict(recovery=dict(passed=True)),config={'a':1},grid=[0,14])
        self.assertEqual(d.source_case(r,case,{'a':1},[0,14]),r['case'])
        r['status']='running'
        with self.assertRaises(ValueError):d.source_case(r,case,{'a':1},[0,14])

    def test_mean_guard_does_not_imply_every_case_or_pooled_rescue(self):
        target=records();prior=records()
        for r in target:
            r['case']['image_id']=f"{int(r['case']['image_id'])-3:05d}"
            r['case']['key']=r['case']['image_id']+'_'+r['case']['key'].split('_',1)[1]
        for r in prior:
            r['branches']['F2']['lpips']=.202
        for r in target:
            r['branches']['F2']=dict(loss=.8,psnr=20.2,ssim=.699,lpips=.204)
        result=d.summarize(target,prior)
        self.assertEqual(result['classification'],'FIXED_F2_CROSS_COHORT_REPLICATION_SUPPORTED')
        self.assertEqual(result['diagnostic_nfe'],12)
        for r in target:
            r['branches']['F2']['lpips']=.206
        result=d.summarize(target,prior)
        self.assertEqual(result['classification'],'FIXED_F2_CROSS_COHORT_REPLICATION_NOT_SUPPORTED')
        self.assertFalse(result['target_comparisons']['F2_C']['passed'])
        self.assertTrue(result['pooled_descriptive_comparisons']['F2_C']['passed'])

    def test_fixed_F1_is_required_even_when_C_improves(self):
        target=records()
        for r in target:r['branches']['F2']=copy.deepcopy(r['branches']['F1'])
        result=d.summarize(target,[])
        self.assertTrue(result['target_comparisons']['F2_C']['passed'])
        self.assertEqual(result['classification'],'FIXED_F2_CROSS_COHORT_REPLICATION_NOT_SUPPORTED')
        self.assertEqual(d.summarize(target[:1],[])['classification'],'INCOMPLETE_PILOT')


if __name__=='__main__':unittest.main()
