"""Pure allocation, budget, memory admission and safe idle handoff tests."""
import unittest
from utils import a56m_imagenet_runtime as r
from utils import a56m_imagenet_session as s


class RuntimeTests(unittest.TestCase):
    def cases(self):
        return [dict(key=t+str(i),task=t,dataset='imagenet') for t in ('super_resolution','inpainting') for i in range(4)]

    def test_fixed_eight_case_cohort_no_substitution(self):
        p=dict(cases=self.cases())
        self.assertEqual(r.selected_cases(p),p['cases'])
        self.assertEqual(r.selected_cases(p)[0]['task'],'super_resolution')
        with self.assertRaises(ValueError):r.selected_cases(dict(cases=p['cases'][:-1]))

    def test_profile72_and_original_remaining_budget(self):
        self.assertEqual(30+30+2*6,r.PROFILE_CAP)
        self.assertEqual(30+2,r.BACK_CAP)
        self.assertEqual(8*(300+180)+12+r.PROFILE_CAP,3924)
        self.assertEqual(8*60+2+r.BACK_CAP,514)

    def test_peak_includes_own_reserved_and_keeps_one_GiB_margin(self):
        GiB=1024**3
        self.assertTrue(r.memory_fits(8*GiB,7*GiB,2*GiB))
        self.assertFalse(r.memory_fits(8*GiB,7*GiB,GiB))
        self.assertFalse(r.memory_fits(0,10*GiB,0))

    def test_10_GiB_card_can_enter_probe_but_not_auto_quality(self):
        rows=[dict(index='0',uuid='GPU-a',free=6026),dict(index='1',uuid='GPU-b',free=10785)]
        self.assertEqual(s.choose_gpu(rows)['index'],'1')
        self.assertIsNone(s.choose_gpu(rows,'0'))
        with self.assertRaises(ValueError):s.choose_gpu(rows,'0',first='1')

    def idle(self):
        return dict(status='running_or_waiting_for_memory',pid=100,running_groups=[],failed_groups=[],
            completed_groups=['ffhq_0','ffhq_1'],pending_groups=['imagenet_0','imagenet_1'])

    def test_running_wrong_owner_or_process_is_not_interruptible(self):
        cmd=['python',str(r.PROJECTS/'utils/a56_dataset_queue_session.py'),'--coordinator']
        self.assertTrue(r.idle_queue_identity(self.idle(),cmd,1000,1000,100))
        state=self.idle();state['running_groups']=['imagenet_0']
        self.assertFalse(r.idle_queue_identity(state,cmd,1000,1000,100))
        self.assertFalse(r.idle_queue_identity(self.idle(),cmd,1001,1000,100))
        self.assertFalse(r.idle_queue_identity(self.idle(),['python','other.py'],1000,1000,100))
        self.assertFalse(r.idle_queue_identity(self.idle(),cmd,1000,1000,200))

    def test_metrology_expands_only_two_fixed_grid_points(self):
        grid=list(range(30));grid[-1]=999
        expected=list(reversed(grid))
        for k,t in enumerate(reversed(grid)):expected.extend([t]*(7 if k in (24,28) else 1))
        self.assertEqual(len(expected),72)
        self.assertEqual(expected[:30],list(reversed(grid)))


if __name__=='__main__':unittest.main()
