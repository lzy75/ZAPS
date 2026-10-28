"""No-Torch contract and append-only recovery checks."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from utils import counted_adjoint_repair as repair
from utils import current_strong_anchor as rules


class InterfaceTests(unittest.TestCase):
    def test_optional_keyword_and_positional_arguments_return_identical_objects(self):
        calls=[]; output=object()
        class Operator:
            def H(self, value, **kwargs): calls.append(('H', value, kwargs)); return output
            def transpose(self, value, *args, **kwargs): calls.append(('T', value, args, kwargs)); return output
        op=Operator(); count=dict(base_H_actions=0, base_adjoint_actions=0)
        repair.count_operator(op,count)
        self.assertIs(op.H(3, option=4), output)
        self.assertIs(op.transpose(5, output_size=None), output)
        self.assertIs(op.transpose(6, (16,16)), output)
        self.assertIs(op.transpose(7, output_size=(16,16)), output)
        self.assertEqual(calls[1], ('T',5,(),dict(output_size=None)))
        self.assertEqual(count,dict(base_H_actions=1,base_adjoint_actions=3))

    def failure(self, image):
        return dict(status='failed', error=repair.ERROR, arm='EXACT_IEEE', repeat=0,
            case=dict(image_id=image), counts=dict(forward_attempts=1,forwards=1,input_vjp_attempts=0,
                optimizer_backward_attempts=0,optimizer_backward_completed=0))

    def test_other_failure_or_cost_never_gets_retried(self):
        value=self.failure('00000'); repair.validate_failure(value)
        for key,new in [('error','OutOfMemoryError'),('arm','RESPONSE_30'),('repeat',1),('status','running')]:
            bad=deepcopy(value); bad[key]=new
            with self.assertRaises(ValueError): repair.validate_failure(bad)
        bad=deepcopy(value);bad['counts']['forwards']=2
        with self.assertRaises(ValueError):repair.validate_failure(bad)

    def test_recovery_preserves_bytes_is_idempotent_and_counts_archived_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'; state=Path(tmp)/'session'; state.mkdir();root.mkdir()
            rules.write(root/'plan.json',dict(experiment=rules.EXPERIMENT))
            before=[]
            for image in ('00000','00005'):
                f=root/'cases'/image/'EXACT_IEEE/repeat_0/run.json';f.parent.mkdir(parents=True)
                value=self.failure(image);value['plan_sha256']=rules.digest(root/'plan.json');rules.write(f,value)
                before.append(f.read_bytes())
            rules.write(state/'session.json',dict(status='failed'))
            rules.write(root/'workers/shard0/status.json',dict(status='failed'))
            first=repair.prepare_recovery(root,state);second=repair.prepare_recovery(root,state)
            self.assertEqual(first,second);self.assertFalse((state/'session.json').exists())
            self.assertEqual([ (root/i['saved']/'run.json').read_bytes() for i in first['failed_attempts']],before)
            value=repair.augment_summary(root,dict(new_cost=dict(method_forward_attempts=33900,
                method_forwards_completed=33900,input_vjp_attempts=1500,optimizer_backward_attempts=1070)))
            self.assertEqual(value['new_cost_including_archived_failures']['method_forwards_completed'],33902)
            self.assertEqual(value['archived_failed_cost']['forwards'],2)
            new=root/'cases'/'00000'/'EXACT_IEEE/repeat_0/run.json';new.parent.mkdir(parents=True)
            rules.write(new,self.failure('00000'))
            with self.assertRaises(ValueError):repair.prepare_recovery(root,state)


if __name__=='__main__':unittest.main()
