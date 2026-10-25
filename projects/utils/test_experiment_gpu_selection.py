import unittest
from utils.experiment_gpu_selection import choose


class GpuSelectionTests(unittest.TestCase):
    rows = [dict(index='0',uuid='GPU-A',free=2000),dict(index='1',uuid='GPU-B',free=12000)]

    def test_busy_gpu_zero_uses_gpu_one_serially(self):
        self.assertEqual(choose(self.rows),['1','1'])

    def test_two_free_devices_use_most_free_first(self):
        rows = [dict(self.rows[0],free=8000),self.rows[1]]
        self.assertEqual(choose(rows),['1','0'])

    def test_never_escapes_allocation(self):
        with self.assertRaisesRegex(ValueError,'No allocated GPU'):choose(self.rows,visible='0')
        with self.assertRaisesRegex(ValueError,'outside'):choose(self.rows,visible='1',first='0')
        self.assertEqual(choose(self.rows,visible='GPU-B'),['GPU-B','GPU-B'])

    def test_aliases_do_not_launch_two_shards_on_the_same_gpu(self):
        self.assertEqual(choose(self.rows,visible='1,GPU-B',first='GPU-B',second='1'),['1','1'])

    def test_explicit_busy_or_disabled_gpu_stops(self):
        with self.assertRaisesRegex(ValueError,'Explicit GPU'):choose(self.rows,first='0')
        with self.assertRaises(ValueError):choose(self.rows,visible='-1')


if __name__=='__main__':unittest.main()
