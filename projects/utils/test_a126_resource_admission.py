"""CPU-only regression tests for resource reporting and unstarted-run recovery."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from utils import a126_resource_admission as admission
from utils import paper_clock_alignment as files


class Tests(unittest.TestCase):
    def test_gpu_and_disk_shortages_are_identified_separately(self):
        self.assertEqual(admission.evaluate(8*admission.GIB,2*admission.GIB)['shortages'],[])
        gpu=admission.evaluate(5*admission.GIB,2*admission.GIB)
        disk=admission.evaluate(8*admission.GIB,admission.GIB//2)
        self.assertEqual(len(gpu['shortages']),1)
        self.assertTrue(gpu['shortages'][0].startswith('GPU'))
        self.assertEqual(len(disk['shortages']),1)
        self.assertTrue(disk['shortages'][0].startswith('Output disk'))
        self.assertFalse(admission.evaluate(5*admission.GIB,0)['passed'])

    def make_legacy(self,root):
        driver=root/'driver.py'; driver.write_text('new resource messages\n')
        science=root/'science.py'; science.write_text('unchanged algorithm\n')
        files.write(root/'session.json',dict(status='failed',error=admission.LEGACY_ERROR))
        files.write(root/'plan.json',dict(experiment='A126',sources={str(driver):'old-driver-hash',str(science):files.digest(science)}))
        files.write(root/'cases'/'case'/'BASE'/'run.json',dict(arm='BASE',status='complete',reused=True))
        (root/'worker.log').write_text('original preflight failure\n')
        return driver

    def test_only_unstarted_failure_is_preserved_without_deleting_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); driver=self.make_legacy(root)
            old_plan=(root/'plan.json').read_bytes()
            self.assertTrue(admission.recover_legacy_preflight(root,driver))
            self.assertFalse((root/'plan.json').exists())
            self.assertEqual((root/'preflight_history'/'001'/'plan.json').read_bytes(),old_plan)
            self.assertEqual((root/'worker.log').read_text(),'original preflight failure\n')
            self.assertTrue((root/'cases'/'case'/'BASE'/'run.json').exists())
            self.assertFalse(admission.recover_legacy_preflight(root,driver))

    def test_a_model_load_or_reconstruction_prevents_recovery(self):
        for item in ('load','run','science'):
            with self.subTest(item=item),tempfile.TemporaryDirectory() as directory:
                root=Path(directory); driver=self.make_legacy(root)
                if item=='load': files.write(root/'load.json',{})
                elif item=='run': files.write(root/'cases'/'case'/'SCALAR'/'run.json',dict(status='running',arm='SCALAR'))
                else: (root/'science.py').write_text('changed\n')
                with self.assertRaises(ValueError): admission.recover_legacy_preflight(root,driver)
                self.assertTrue((root/'plan.json').exists())

    def test_worker_records_resource_limit_without_treating_it_as_computation_failure(self):
        from utils import diag_a126_full_response as driver
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            report=admission.evaluate(5*admission.GIB,2*admission.GIB)
            with patch.object(driver,'execute',side_effect=admission.ResourceShortage(report)):
                driver.worker('dps',root)
            state=files.read(root/'session.json')
            self.assertEqual(state['status'],'resource_limited')
            self.assertEqual(state['resources'],report)
            self.assertNotIn('traceback',state)


if __name__=='__main__': unittest.main()
