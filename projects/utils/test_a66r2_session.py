"""R2 queue, strict scientific rejection, and cost audit checks; no Torch."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from utils import a66r2_session as repair
from utils import current_strong_anchor as rules
from utils import historical_ffhq_gain as design


class RecoveryTests(unittest.TestCase):
    def fixture(self):
        return dict(status='failed',error="ValueError('Independent270 prefix differs')",arm='RESPONSE_300',repeat=0,
            case=dict(image_id='00000'),counts=dict(forward_attempts=300,forwards=300,input_vjp_attempts=30,
            optimizer_backward_attempts=9,optimizer_backward_completed=9),
            independent_prefix_gate=dict(passed=False,rng_equal=True,
                relative_errors=dict(tau=0,initial=0,cpu_rng=0,zeta=.0682449,D=.0102875)))

    def test_queue_retains_all_primary_and_control_repeats_without_rejected_arm(self):
        q=repair.queue();self.assertEqual(len(q),11)
        self.assertFalse(any(a=='RESPONSE_300' for a,r in q))
        self.assertEqual([r for a,r in q if a=='RESPONSE_30'],[0,1,2])
        # 26400F in queue minus retained3000 IEEE and1500 exact =20400F.
        self.assertEqual(10*sum(design.NFE[a] for a,r in q)-4500,20400)
        self.assertEqual(10*sum(design.UPDATES[a] for a,r in q)-150,650)

    def test_only_exact_saved_prefix_failure_is_accepted(self):
        r=self.fixture();self.assertEqual(repair.validate_rejection(r)['forwards'],300)
        for key,value in [('error','OOM'),('arm','RESPONSE_30'),('repeat',1),('status','complete')]:
            bad=deepcopy(r);bad[key]=value
            with self.assertRaises(ValueError):repair.validate_rejection(bad)
        bad=deepcopy(r);bad['independent_prefix_gate']['rng_equal']=False
        with self.assertRaises(ValueError):repair.validate_rejection(bad)

    def test_rejected_bytes_preserved_and_new_failure_blocks_retry(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(repair,'OLD_CHECK',return_value={}):
            root=Path(tmp)/'run';root.mkdir();state=Path(tmp)/'state';state.mkdir()
            rules.write(root/'plan.json',dict(experiment=rules.EXPERIMENT))
            file=root/'cases/00000/RESPONSE_300/repeat_0/run.json';file.parent.mkdir(parents=True)
            r=self.fixture();r['plan_sha256']=rules.digest(root/'plan.json');rules.write(file,r);before=file.read_bytes()
            rules.write(state/'session.json',dict(status='failed'));(state/'coordinator.log').write_text('old error')
            m=repair.prepare_recovery(root,state)
            self.assertEqual((root/m['saved']/'run.json').read_bytes(),before)
            self.assertFalse(file.exists());self.assertFalse((state/'session.json').exists())
            self.assertEqual(repair.prepare_recovery(root,state),m)
            file.parent.mkdir(parents=True);rules.write(file,r)
            with self.assertRaises(ValueError):repair.prepare_recovery(root,state)


if __name__=='__main__':unittest.main()
