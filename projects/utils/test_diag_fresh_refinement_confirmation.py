import json
from pathlib import Path
import tempfile
import unittest
from utils import diag_fresh_refinement_confirmation as d
try:
    import torch
except ImportError:
    torch=None


class ProtocolTests(unittest.TestCase):
    def test_history_rejects_id_key_or_identical_image_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'run.json';cases=[dict(image_sha256='a'*64)]
            for record in (dict(image_id='00023'),dict(case='00024_s1000'),dict(image_sha256='a'*64)):
                path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError,'already used'):d.check_history([path],cases)
            path.write_text(json.dumps(dict(image_id='00020')))
            self.assertEqual(len(d.check_history([path],cases)),1)
            path.write_text('')
            with self.assertRaises(json.JSONDecodeError):d.check_history([path],cases)

    def test_each_position_has_its_own_cluster_threshold_and_cost_includes_failures(self):
        r=[]
        for image in d.IDS:
            for seed in d.SEEDS:
                base=dict(loss=1.,psnr=25.,ssim=.8,lpips=.2);good=dict(loss=.9,psnr=25.2,ssim=.81,lpips=.19);floors={k:0. for k in base}
                positions={}
                for t in ('14','54'):
                    hits=dict(F0=t=='14',F1=t=='14')
                    positions[t]=dict(branches=dict(F0=good,F1=good),effects={a+'_C':d.coupled.history.effect(good,base,floors) for a in ('F0','F1')},
                        bridge_candidates=hits,both_bridges_candidate=all(hits.values()))
                r.append(dict(status='complete',case=dict(key=f'{image}_s{seed}',image_id=image),method_nfe=300,diagnostic_nfe=72,baseline=base,
                    positions=positions,position_comparison={'readout':'TRADEOFF_OR_UNRESOLVED'},gates={'x':{'passed':True}}))
        s=d.summarize(r);self.assertEqual((s['method_nfe'],s['diagnostic_nfe']),(1800,432))
        self.assertEqual(s['positions'][0]['classification'],'FIXED_POSITION_CONFIRMED_IN_COHORT')
        self.assertEqual(s['positions'][1]['classification'],'FIXED_POSITION_NOT_CONFIRMED')
        s=d.summarize(r[:1]+[dict(status='failed',method_nfe=100,diagnostic_nfe=3)])
        self.assertEqual((s['method_nfe'],s['diagnostic_nfe']),(400,75))
        self.assertEqual(s['positions'][0]['classification'],'INCOMPLETE_PILOT')

    def test_archive_preserves_bytes_and_working_git_index(self):
        import subprocess
        from utils.archive_refinement_confirmation import export_and_commit
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);projects=repo/'projects';projects.mkdir();source=projects/'results/run';source.mkdir(parents=True)
            (source/'run.json').write_bytes(b'{"status":"complete"}\r\n');(source/'tensor.pt').write_bytes(b'original tensor fixture')
            subprocess.run(['git','init','-q',str(repo)],check=True)
            subprocess.run(['git','-C',str(repo),'config','user.name','Test'],check=True)
            subprocess.run(['git','-C',str(repo),'config','user.email','test@example.invalid'],check=True)
            subprocess.run(['git','-C',str(repo),'commit','--allow-empty','-qm','initial'],check=True)
            def head():return subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'])
            before=head();bundle,commit=export_and_commit({'A6':source},projects)
            self.assertEqual(head(),before);self.assertEqual((bundle/'A6/run.json').read_bytes(),(source/'run.json').read_bytes())
            self.assertEqual(len(json.loads((bundle/'manifest.json').read_text())['excluded']),1)
            self.assertEqual(export_and_commit({'A6':source},projects)[1],commit)

    def test_interrupted_directory_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'00023_s1000').mkdir()
            with self.assertRaises(ValueError):d.completed(path,{'cases':[{'key':'00023_s1000'}]})


@unittest.skipIf(torch is None,'Server Torch required; no UNet loaded')
class TensorTests(unittest.TestCase):
    def test_frozen_snapshot_captures_preupdate_last_epoch_and_rng(self):
        from unittest.mock import patch
        from modules.zaps_algorithm import ZAPS
        cls=d.baseline.snapshot_class()
        snapshot=object.__new__(cls)
        torch.nn.Module.__init__(snapshot)
        snapshot.num_epochs=1;snapshot.device='cpu';snapshot.zeta=torch.nn.Parameter(torch.ones(30))
        snapshot.D=torch.nn.Parameter(torch.ones(30,3,2,2));snapshot.tau=torch.arange(30)
        noise=torch.zeros(1,3,2,2)
        with patch.object(ZAPS,'_reverse_diffusion',return_value=noise):
            snapshot._reverse_diffusion(None,init_noise=noise)
        self.assertEqual(snapshot.audit_calls,1)
        self.assertTrue(torch.equal(snapshot.audit_state['zeta'],torch.ones(30)))
        with torch.no_grad():snapshot.zeta.add_(1)
        self.assertTrue(torch.equal(snapshot.audit_state['zeta'],torch.ones(30)))
        self.assertTrue(torch.equal(snapshot.audit_state['init_noise'],noise))


if __name__=='__main__':unittest.main()
