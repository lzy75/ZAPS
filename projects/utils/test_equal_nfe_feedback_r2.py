import unittest


class Tests(unittest.TestCase):
    def test_original_measure_reference_was_captured_before_patch(self):
        from utils import diag_equal_nfe_feedback_r2 as driver
        self.assertIsNot(driver._BASE_MEASURE, driver.measure)
        self.assertEqual(driver._BASE_MEASURE.__module__, 'utils.diag_equal_nfe_feedback')


if __name__ == '__main__': unittest.main()
