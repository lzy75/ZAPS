import unittest
from utils import a67r1_session as repair


def failed():
    return dict(status='failed',error="ValueError('Model weights not frozen')",arm='BASE',
                case=dict(key='00000_n025_s1000'),replay_counts=None,artifacts={},
                full_counts=dict(forward_attempts=300,forwards=300,input_vjp_attempts=0,
                                 optimizer_backward_attempts=10,optimizer_backward_completed=10,
                                 calls=repair.files.grid()[::-1]*10))


class RecoveryTests(unittest.TestCase):
    def test_exact_observed_failure_cost(self):
        self.assertEqual(repair.validate_failure(failed())['method_forward_attempts'],300)

    def test_other_errors_or_partial_calls_are_rejected(self):
        row=failed();row['error']="ValueError('Different')"
        with self.assertRaises(ValueError):repair.validate_failure(row)

    def test_recovery_preserves_failures_and_is_idempotent(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root=Path(directory)/'A67';state=Path(directory)/'A67_session';state.mkdir();root.mkdir()
            repair.files.write(root/'plan.json',dict(experiment='A67_strong_zaps_noise_feasible',sources={}))
            repair.files.write(state/'session.json',dict(status='failed'))
            (state/'coordinator.log').write_text('original failed session\n')
            original={}
            for i,key in enumerate(sorted(repair.EXPECTED)):
                row=failed();row['case']['key']=key;row['plan_sha256']=repair.files.digest(root/'plan.json')
                file=root/'cases'/key/'BASE/run.json';repair.files.write(file,row)
                original[key]=file.read_bytes()
                repair.files.write(root/'workers'/str(i)/'load_000.json',dict(warmup_forward_attempts=1))
                repair.files.write(root/'workers'/str(i)/'feedback.json',dict(status='failed'))
            first=repair.prepare_recovery(root,state);second=repair.prepare_recovery(root,state)
            self.assertEqual(first,second)
            self.assertEqual(first['recorded_failed_cost']['method_forward_attempts'],600)
            self.assertEqual(first['recorded_failed_cost']['warmup_forward_attempts'],2)
            for entry in first['failures']:
                self.assertFalse((root/entry['original']).exists())
                key=Path(entry['original']).parts[1]
                self.assertEqual((root/entry['saved']/'run.json').read_bytes(),original[key])
        row=failed();row['full_counts']['forwards']=299
        with self.assertRaises(ValueError):repair.validate_failure(row)


class FreezeTests(unittest.TestCase):
    def test_trainable_default_is_frozen_and_flags_restored(self):
        import torch
        from utils.a67_model_freeze import frozen_network
        model=torch.nn.Linear(2,2);flags=[p.requires_grad for p in model.parameters()]
        before=[p.detach().clone() for p in model.parameters()]
        with frozen_network(model):
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            self.assertFalse(model(torch.ones(1,2)).requires_grad)
        self.assertEqual([p.requires_grad for p in model.parameters()],flags)
        for a,b in zip(before,model.parameters()):torch.testing.assert_close(a,b)

    def test_exception_restores_flags_and_weight_update_detected(self):
        import torch
        from utils.a67_model_freeze import frozen_network
        model=torch.nn.Linear(2,2)
        with self.assertRaisesRegex(RuntimeError,'dummy'):
            with frozen_network(model):raise RuntimeError('dummy')
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        with self.assertRaisesRegex(RuntimeError,'weight or gradient changed'):
            with frozen_network(model):
                with torch.no_grad():model.weight.add_(1)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_full300_with_actual_default_flags_has_identical_output(self):
        import torch
        from utils.test_paper_clock_runtime import TinyDiffusion,Linear
        from utils import noise_feasible_runtime as runtime
        from utils.a67_model_freeze import frozen_network
        from utils.paper_clock_runtime import counters
        dm=TinyDiffusion();dm.model.weight.requires_grad_(True)
        y=torch.full((1,3,16,16),.15)
        first,_=runtime.execute(dm,Linear(),y,1000,'BASE',.05,size=16,level=1)
        self.assertIsNone(dm.model.weight.grad)
        with frozen_network(dm.model),counters(dm.model) as count:
            second,_=runtime.execute(dm,Linear(),y,1000,'BASE',.05,size=16,level=1)
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        self.assertTrue(dm.model.weight.requires_grad)
        self.assertEqual((count['forwards'],count['optimizer_backward_completed']),(300,10))


if __name__=='__main__':unittest.main()
