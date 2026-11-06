"""Regression at the real execute->probe->source-inspector call boundary, no Torch."""
import contextlib,io,subprocess,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
from utils import diag_response_target_lm_r2 as r


class Network:
    def modules(self):return [self]


class Tests(unittest.TestCase):
 def test_network_and_diffusion_wrapper_dispatch(self):
    network=Network();wrapper=types.SimpleNamespace(model=network)
    with patch.object(r,'LOADED_SOURCES',return_value={'source':'sha'}) as lookup:
        self.assertEqual(r.loaded_sources(wrapper),{'source':'sha'});lookup.assert_called_with(network)
        r.loaded_sources(network);lookup.assert_called_with(network)
    with self.assertRaises(TypeError):r.loaded_sources(object())
 def test_entire_r1_probe_boundary_inspects_inner_but_predicts_with_wrapper(self):
    network=Network();wrapper=types.SimpleNamespace(model=network);count={}
    def original_execute(root,p):
        gate=r.r1.runtime.checkpoint_probe(wrapper,309,count)
        self.assertTrue(gate['passed'])
    with tempfile.TemporaryDirectory() as temporary,contextlib.redirect_stdout(io.StringIO()):
        root=Path(temporary)
        with patch.object(r,'LOADED_SOURCES',return_value={'source':'sha'}) as lookup,patch.object(r.r1.repair,'loaded_sources',r.loaded_sources),\
             patch.object(r.r1,'ORIGINAL_EXECUTE',original_execute),patch.object(r.r1,'ORIGINAL_PROBE',return_value={'passed':True}) as probe,\
             patch.object(r.r1.subprocess,'run',return_value=subprocess.CompletedProcess([],0,stdout='',stderr='')):
            r.EXECUTE(root,{})
            lookup.assert_called_once_with(network);probe.assert_called_once_with(wrapper,309,count)
        self.assertEqual(r.r1.old.files.read(root/'actual_checkpoint_sources.json'),{'source':'sha'})
 def test_failure_gate_preserves_bytes_and_rejects_any_case_attempt(self):
    files=r.r1.old.files
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);files.write(root/'session.json',dict(status='failed',traceback="FFHQDiffusionModel loaded_sources object has no attribute modules"))
        files.write(root/'summary.json',dict(completed_runs=0));files.write(root/'plan.json',{})
        before=(root/'session.json').read_bytes();self.assertTrue(r.failed_inputs(root));self.assertEqual(before,(root/'session.json').read_bytes())
        files.write(root/'cases/one/prefix.json',dict(status='running'))
        with self.assertRaises(ValueError):r.failed_inputs(root)

if __name__=='__main__':unittest.main()
