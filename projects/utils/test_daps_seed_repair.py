"""Pure zero-run repair guards and NumPy seed regression."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from utils import daps_seed_repair as repair


class Tests(unittest.TestCase):
    def test_constructor_seeds_numpy_even_after_unrelated_draws(self):
        import numpy as np
        def constructor(case,device):return np.random.uniform(size=12)
        with patch.object(repair.original,'make_operator',side_effect=constructor):
            a=repair.make_operator(dict(kernel_seed=30226),'unused')
            np.random.uniform(size=73)
            b=repair.make_operator(dict(kernel_seed=30226),'unused')
        self.assertTrue(np.array_equal(a,b))

    def test_only_reported_zero_run_failure_is_admitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'A81';state=Path(tmp)/'A81_session';state.mkdir()
            (state/'session.json').write_text(json.dumps(dict(status='failed')))
            (state/'coordinator.log').write_text('test_original_motion_state_reproducible_and_input_gradient_finite ... FAIL\nFAILED (failures=1)\n')
            before=(state/'coordinator.log').read_bytes()
            self.assertEqual(repair.zero_run_failure(root)['formal_model_runs'],0)
            self.assertEqual((state/'coordinator.log').read_bytes(),before)
            root.mkdir();(root/'plan.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'formal experiment state'):repair.zero_run_failure(root)

    def test_repair_driver_freezes_same_budget_and_parent_without_mutation(self):
        from utils import diag_daps_seed_repair as driver
        from utils import test_daps_multiplier_transfer as original_tests
        with tempfile.TemporaryDirectory() as tmp:
            state=Path(tmp);status=state/'session.json';log=state/'coordinator.log'
            status.write_text('{"status":"failed"}');log.write_text('original failed gate retained')
            audit=dict(status_path=str(status),log_path=str(log),formal_model_runs=0)
            with patch.object(original_tests,'driver',driver),patch.object(repair,'zero_run_failure',return_value=audit):
                original_tests.Tests.test_prepare_parent_immutable_and_no_old_measurement_reused(self)
            self.assertEqual(log.read_text(),'original failed gate retained')


if __name__=='__main__':unittest.main()
