import unittest
from utils import shrink_multitask as r
class DesignTests(unittest.TestCase):
    def test_frozen_panel_and_exact_new_budget(self):
        cases=r.cases('/images');self.assertEqual(len(cases),36)
        self.assertEqual({c['image_id'] for c in cases},set(r.IMAGES))
        self.assertEqual(sum(r.reusable(c) for c in cases)*2,12)
        self.assertEqual(sum(not r.reusable(c) for c in cases)*2*300,18000)
        self.assertEqual([sum(c['shard']==i for c in cases) for i in (0,1)],[18,18])
    def test_all_configs_retain_strong_fixed_grid_and_task_initialization(self):
        for task in r.TASKS:
            cfg=r.configuration(task);self.assertEqual((cfg['num_steps'],cfg['num_epochs'],cfg['lr']),(30,10,.01))
            self.assertEqual(cfg['schedule'],[15,10,5]);self.assertEqual(cfg['zeta_init'],.2 if 'deblur' in task else .1)
    def test_reuse_keys_and_seed_are_parent_compatible(self):
        for c in r.cases('/images'):
            self.assertEqual(c['seed'],1000+int(c['image_id']))
            self.assertEqual(r.parent_key(c),f"{c['image_id']}_n{round(c['sigma']*1000):03d}_s{c['seed']}")
if __name__=='__main__':unittest.main()
