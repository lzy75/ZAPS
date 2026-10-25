import unittest
from utils import current_score_design as d

GRID=[0,14,54,124,218,319,409,484,546,597,641,679,712,742,769,793,815,835,854,872,888,903,917,931,944,956,967,978,989,999]


class DesignTests(unittest.TestCase):
    def cases(self):
        return [dict(key=t+'_'+i,task=t,image_id=i) for t in d.TASKS for i in d.IMAGES]

    def test_controls_exclude_target_image_and_use_no_labels(self):
        cases=self.cases();curves={c["key"]:[float(i+1)]*30 for i,c in enumerate(cases)}
        first=d.allocate(cases,curves)
        current=cases[0]["key"]
        original=first["assignments"][current]
        curves[current]=[1e9]*30
        changed=d.allocate(cases,curves)["assignments"][current]
        self.assertEqual(changed,original)
        self.assertNotIn(cases[0]["image_id"],original["reference_images"])
        self.assertNotEqual(current,original["donor"])
        with self.assertRaises(TypeError):d.allocate(cases,curves,ground_truth={})
        with self.assertRaises(ValueError):d.allocate(cases[:-1],curves)

    def test_all_paid_forwards_and_metrology_are_in_phase_caps(self):
        cases=self.cases()
        total=0
        for c in cases:
            for phase in d.PHASES:
                trace=d.expected_forward_trace(c,GRID,phase)
                self.assertEqual(len(trace),d.phase_limit(c,phase))
                total+=len(trace)
        self.assertEqual(total,4872)


if __name__=="__main__":unittest.main()
