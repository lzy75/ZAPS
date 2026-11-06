"""Path recognition/exception restoration and strict zero-case repair authorization."""
import tempfile,types,unittest
from pathlib import Path
from utils import a95_checkpoint_scope_r1 as r,diag_response_target_lm_r1 as driver


def model_at(path):
    namespace={};exec(compile('def checkpoint(f, inputs, params, flag): return f(*inputs)\nclass Layer:\n def forward(self,x): return checkpoint(lambda a:a,(x,),(),True)\n',str(path),'exec'),namespace)
    return types.SimpleNamespace(modules=lambda:[namespace['Layer']()]),namespace


class Tests(unittest.TestCase):
 def test_both_root_and_dps_paths_restore_on_exception(self):
    for path in r.APPROVED:
        model,namespace=model_at(path);original=namespace['checkpoint']
        with self.assertRaisesRegex(RuntimeError,'sentinel'):
            with r.forward_ad_checkpoint_scope(model) as count:
                self.assertEqual(count,1);self.assertIsNot(namespace['checkpoint'],original);raise RuntimeError('sentinel')
        self.assertIs(namespace['checkpoint'],original)
 def test_unapproved_checkpoint_is_untouched(self):
    model,namespace=model_at('/tmp/unrelated/nn.py');original=namespace['checkpoint']
    with r.forward_ad_checkpoint_scope(model) as count:self.assertEqual(count,0)
    self.assertIs(namespace['checkpoint'],original)
 def test_only_reported_failure_without_case_attempts_can_be_repaired(self):
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);files=driver.old.files
        files.write(root/'session.json',dict(status='failed',traceback='setup_context /home/lzy/ZAPS/guided_diffusion/nn.py'))
        files.write(root/'plan.json',{});files.write(root/'summary.json',dict(completed_runs=0))
        before=(root/'session.json').read_bytes();bound=driver.failed_inputs(root)
        self.assertIn(str(root/'session.json'),bound);self.assertEqual(before,(root/'session.json').read_bytes())
        files.write(root/'cases/one/REAL_SCALAR/run.json',dict(status='failed'))
        with self.assertRaises(ValueError):driver.failed_inputs(root)

if __name__=='__main__':unittest.main()
