import math
from pathlib import Path
import unittest
from utils import noise_signal_shrink as r

class PolicyTests(unittest.TestCase):
    def test_noise_free_least_squares_and_noise_only_zero(self):
        self.assertAlmostEqual(r.select(8,16,4,4,1e-10)['gamma'],2)
        self.assertEqual(r.select(8,16,4,4,2)['gamma'],0)
    def test_continuous_above_two_and_noise_monotonic(self):
        a=r.select(6,12,3,3,.5);b=r.select(6,12,3,3,1.)
        self.assertGreater(a['gamma'],b['gamma']);self.assertNotIn(a['gamma'],(0,1,2))
        self.assertGreater(r.select(6,36,1,3,.1)['gamma'],2)
    def test_exact_projection_of_shrunk_residual(self):
        row=r.select(6,12,3,3,1.);targetdot=row['weight']*6
        g=row['gamma']
        def f(x):return .5*(3*x*x-2*targetdot*x)
        self.assertLess(f(g),f(g-.01));self.assertLess(f(g),f(g+.01))
    def test_zero_direction_opposed_and_invalid(self):
        self.assertEqual(r.select(0,1,0,1,.1)['gamma'],0)
        self.assertEqual(r.select(-1,1,1,1,.1)['gamma'],0)
        with self.assertRaises(ValueError):r.select(2,1,1,1,.1)
    def test_fixed_constant_is_global_not_time_curve(self):
        self.assertEqual(r.constant([[.25]*30,[.75]*30]),.5)
        with self.assertRaises(ValueError):r.constant([[1.]])
    def test_cyclic_donor_is_same_noise_and_other_image(self):
        cases=[dict(key=f'{i}_{s}',image_id=str(i),sigma=s) for i in range(4) for s in r.SIGMAS]
        mapping=r.mismatches(cases);rows={c['key']:c for c in cases}
        for key,donor in mapping.items():
            self.assertNotEqual(rows[key]['image_id'],rows[donor]['image_id'])
            self.assertEqual(rows[key]['sigma'],rows[donor]['sigma'])
    def test_candidate_inventory_does_not_exhaust_selection(self):
        import json,tempfile
        from unittest.mock import patch
        from utils import diag_noise_signal_shrink as driver
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'images';data.mkdir();project=root/'projects';(project/'results').mkdir(parents=True)
            pool=[]
            for i in range(32,42):
                f=data/f'{i:05d}.png';f.write_bytes(str(i).encode())
                pool.append(dict(image=str(f),image_sha256=driver.files.digest(f)))
            (project/'results/plan.json').write_text(json.dumps(dict(cases=pool[:1],candidate_images=pool)))
            with patch.object(driver,'PROJECTS',project):
                chosen,_=driver.select_new(data,root/'out');again,_=driver.select_new(data,root/'out')
            self.assertEqual(chosen,again);self.assertEqual(len(chosen),4)
            self.assertNotIn('00032',{c['image_id'] for c in chosen})

if __name__=='__main__':unittest.main()
