"""R3 recording/recovery checks without Torch."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from utils import a66r3_session as repair
from utils import repeat_variation_measurement as measurement
from utils import current_strong_anchor as rules


class RecordingTests(unittest.TestCase):
    def test_failed_gate_remains_false_without_discarding_quality_or_picking_winner(self):
        self.assertFalse(measurement.repeat_gate(.004733731464403012)['passed'])
        self.assertEqual(measurement.repeat_gate(.0047)['tolerance'],1e-6)
        self.assertTrue(measurement.repeat_gate(0)['passed'])
        for value in (float('nan'),-1):
            with self.assertRaises(ValueError):measurement.repeat_gate(value)
        rows=[dict(arm='LEGACY_IEEE',repeat=i,case=dict(image_id='00000'),
                   metrics=dict(psnr=20+i,ssim=.7,lpips=.2),repeat_gate=measurement.repeat_gate(.0047)) for i in range(3)]
        data=measurement.repeat_statistics(rows)['LEGACY_IEEE']['00000']
        self.assertEqual(data['metrics']['psnr']['values'],[20,21,22])
        self.assertEqual(data['metrics']['psnr']['mean'],21)
        self.assertFalse(data['repeat_gates'][0]['passed'])

    def fixture(self):
        return dict(status='failed',error="ValueError('Repeat drift; no best-output selection')",arm='LEGACY_IEEE',repeat=1,
            case=dict(image_id='00000'),repeat_relative_error=.004733731464403012,
            counts=dict(forward_attempts=300,forwards=300,input_vjp_attempts=0,
                        optimizer_backward_attempts=10,optimizer_backward_completed=10))

    def test_known_failure_only_and_bytes_preserved(self):
        r=self.fixture();self.assertEqual(repair.validate_failure(r)['forwards'],300)
        with self.assertRaises(ValueError):repair.validate_failure(dict(r,error='OOM'))
        with tempfile.TemporaryDirectory() as tmp,patch.object(repair,'OLD_CHECK',return_value={}):
            root=Path(tmp)/'run';root.mkdir();state=Path(tmp)/'state';state.mkdir()
            rules.write(root/'plan.json',dict(experiment=rules.EXPERIMENT))
            file=root/'cases/00000/LEGACY_IEEE/repeat_1/run.json';file.parent.mkdir(parents=True)
            r['plan_sha256']=rules.digest(root/'plan.json');rules.write(file,r);before=file.read_bytes()
            rules.write(state/'session.json',dict(status='failed'))
            m=repair.prepare_recovery(root,state)
            self.assertEqual((root/m['saved']/'run.json').read_bytes(),before)
            self.assertEqual(repair.prepare_recovery(root,state),m)
            file.parent.mkdir(parents=True);rules.write(file,r)
            with self.assertRaises(ValueError):repair.prepare_recovery(root,state)


if __name__=='__main__':unittest.main()
