"""Regression for old case schemas and preserved zero-call failures; Torch never skipped."""
import tempfile
from pathlib import Path
import unittest
from utils import direction_operator_repair as r


class DesignTests(unittest.TestCase):
    def test_sr_does_not_require_or_use_mask_seed(self):
        self.assertEqual(r.seed_for_new_case(dict(task='super_resolution')),0)
        self.assertEqual(r.seed_for_new_case(dict(task='super_resolution',mask_seed=123)),0)

    def test_new_inpainting_requires_frozen_seed(self):
        with self.assertRaises(ValueError):r.seed_for_new_case(dict(task='inpainting'))
        self.assertEqual(r.seed_for_new_case(dict(task='inpainting',mask_seed=9)),9)

    def test_unknown_or_nonzero_failures_are_not_retried(self):
        row=dict(status='failed',error=repr(KeyError('mask_seed')),nfe=0,backward_attempts=0,backward_completed=0,calls=[])
        self.assertTrue(r.recognized_zero_failure(row))
        for changed in (dict(nfe=1),dict(error="RuntimeError('OOM')"),dict(status='running'),dict(backward_attempts=1)):
            self.assertFalse(r.recognized_zero_failure(dict(row,**changed)))

    def test_preservation_is_idempotent_and_checks_original_bytes(self):
        from utils import diag_direction_operator_repair as d
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'case/control_000';old.mkdir(parents=True)
            f=old/'run.json';f.write_text('original failure\n')
            m=dict(preserved_zero_call_failures=[dict(original='case/control_000',preserved='operator_repair_20261027/preserved_zero_call_failures/case/control_000',inventory={'run.json':d.a57.source.digest(f)})])
            d.preserve(root,m);d.preserve(root,m)
            saved=root/m['preserved_zero_call_failures'][0]['preserved']/'run.json'
            self.assertEqual(saved.read_text(),'original failure\n');self.assertFalse(old.exists())
            saved.write_text('changed')
            with self.assertRaises(ValueError):d.preserve(root,m)


    def test_patch_reaches_a57_shared_factory_and_is_restored(self):
        from unittest.mock import patch
        from utils import diag_direction_operator_repair as d
        before=d.a56.make_operator;plan={'sentinel':'original plan'}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/d.REPAIR).mkdir();(root/d.REPAIR/'manifest.json').write_text('{}')
            with patch.object(d.a57,'check',return_value=plan),patch.object(d,'load_manifest',return_value={}),patch.object(d,'preserve'),patch.object(d.adapter,'make_operator',return_value=('op','mask')) as factory:
                with d.execution('A57',root):
                    self.assertEqual(d.a57.support.make_operator({'task':'super_resolution'},'cpu'),('op','mask'))
                    factory.assert_called_once_with('A57',{'task':'super_resolution'},'cpu',plan)
                self.assertIs(d.a56.make_operator,before)


class TorchTests(unittest.TestCase):
    def test_actual_old_sr_schema_builds_exact_unit_operator(self):
        import torch
        op,mask=r.make_operator('A56',dict(task='super_resolution'),'cpu')
        self.assertIsNone(mask)
        self.assertEqual(tuple(op.H(torch.zeros(1,3,256,256)).shape),(1,3,64,64))

    def test_actual_legacy_ip_uses_exact_archived_mask_without_global_rng(self):
        import torch
        from utils import diag_no_learning_replay as d
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case=dict(task='inpainting',key='old_case')
            mask=torch.rand((1,1,256,256),generator=torch.Generator().manual_seed(71))>.7
            torch.save(mask,root/'mask.pt')
            d.signed(root/'run.json',dict(status='complete',case=case,artifacts={'mask.pt':d.source.digest(root/'mask.pt')}))
            plan=dict(norm_sources={'old_case':str(root/'run.json')},identity_files={str(root/'run.json'):d.source.digest(root/'run.json')})
            before=torch.get_rng_state();op,actual=r.make_operator('A57',case,'cpu',plan)
            self.assertTrue(torch.equal(actual.bool(),mask));self.assertTrue(torch.equal(before,torch.get_rng_state()))
            torch.testing.assert_close(op.H(torch.zeros(1,3,256,256)),.5*mask.expand(1,3,256,256))
            with (root/'mask.pt').open('ab') as f:f.write(b'corruption')
            with self.assertRaises(ValueError):r.make_operator('A57',case,'cpu',plan)

if __name__=='__main__':unittest.main()
