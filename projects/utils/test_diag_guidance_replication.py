"""A2a sample identity, reuse and independent PRE-last replay contracts."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from utils import diag_guidance_replication as audit


class ReplicationContractTests(unittest.TestCase):
    def test_observation_seed_shared_sampling_seed_varies_without_selection(self):
        cases=audit.cases_for('/dataset')
        self.assertEqual(len(cases),6)
        self.assertEqual([c['key'] for c in cases],['00020_s1000','00020_s2000','00021_s1000','00021_s2000','00022_s1000','00022_s2000'])
        for left,right in zip(cases[::2],cases[1::2]):
            self.assertEqual(left['image'],right['image'])
            self.assertEqual(left['measurement_seed'],right['measurement_seed'])
            self.assertNotEqual(left['sampling_seed'],right['sampling_seed'])

    def test_history_filename_directory_and_id_fields_detect_overlap_but_cards_do_not(self):
        value=dict(image='/data/00020.png',records=[dict(path='/old/super_resolution_ffhq_00021/run.json')],
                   image_ids=['00022'],experiment_card='plan includes 00023.png',limits='00024.png')
        self.assertEqual(audit.history_ids(value),{'00020','00021','00022'})

    def test_overlap_or_malformed_history_stops_without_replacing_candidates(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'run.json';audit.write_json(path,dict(image='/data/00020.png',status='failed'))
            with self.assertRaisesRegex(ValueError,'overlap'):audit.audit_history([path])
            path.write_text('{')
            with self.assertRaises(json.JSONDecodeError):audit.audit_history([path])
            audit.write_json(path,dict(image='/data/00000.png'))
            self.assertEqual(audit.audit_history([path])[0]['image_ids'],['00000'])

    def test_completed_case_reuse_requires_unchanged_files_and_shared_measurement(self):
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root);output=directory/'out.pt';output.write_bytes(b'output')
            observation=directory/'y.pt';observation.write_bytes(b'observation')
            record=dict(status='complete',plan_sha256='plan',case=dict(key='00020_s1000'),gates=dict(parity=dict(passed=True)),
                        method_nfe=300,diagnostic_nfe=120,artifacts_sha256={'out.pt':audit.digest(output)},
                        measurement=dict(path=str(observation),sha256=audit.digest(observation)))
            audit.write_json(directory/'run.json',record)
            self.assertEqual(audit.reusable(directory,'plan','00020_s1000'),record)
            observation.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'observation'):audit.reusable(directory,'plan','00020_s1000')
            observation.write_bytes(b'observation');output.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact'):audit.reusable(directory,'plan','00020_s1000')

    def test_failed_case_not_automatically_rerun(self):
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root);audit.write_json(directory/'run.json',dict(status='failed'))
            with self.assertRaisesRegex(ValueError,'overwrite'):audit.reusable(directory,'plan','00020_s1000')

    def test_plan_protocol_and_candidate_image_changes_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root);card=directory/'card.md';card.write_text('fixed hypotheses')
            cases=audit.cases_for(directory)
            for case in cases:
                image=Path(case['image']);image.write_bytes(b'image'+case['image_id'].encode())
                case['image_sha256']=audit.digest(image)
            anchor=directory/'anchor.json';audit.write_json(anchor,dict(resolved_config=dict(config={'lr':.01},grid=[0,999],checkpoint_sha='weights')))
            sha=audit.digest(anchor)
            plan=dict(experiment='guidance_replication_plan',status='prepared',cases=cases,identity_files={str(anchor):sha},
                      experiment_card_sha256=audit.digest(card),anchor=str(anchor),anchor_sha256=sha,config={'lr':.01},grid=[0,999],
                      checkpoint_sha256='weights',planned_method_nfe=1800,planned_diagnostic_nfe=720)
            with patch.object(audit,'CARD',card),patch.object(audit,'ANCHOR_SHA',sha):
                audit.validate_plan(plan)
                changed=copy.deepcopy(plan);changed['config']['lr']=.1
                with self.assertRaisesRegex(ValueError,'protocol'):audit.validate_plan(changed)
                changed=copy.deepcopy(plan);changed['cases'][0]['measurement_seed']=2000
                with self.assertRaisesRegex(ValueError,'seed'):audit.validate_plan(changed)
                Path(cases[0]['image']).write_bytes(b'different')
                with self.assertRaisesRegex(ValueError,'image changed'):audit.validate_plan(plan)

    def test_partial_pilot_cannot_pass_and_costs_include_failed_attempts(self):
        complete=dict(status='complete',case=dict(key='first'),method_nfe=300,diagnostic_nfe=120,
                      guidance_valid=30,usable_snapshots=5,probes=[],rows=[])
        # Five unresolved probe slots suffice to test reporting without geometry vectors.
        for k in range(5):
            complete['rows'].append(dict(guidance_variation3_per_u=1.,guidance_rms=1.))
            complete['probes'].append(dict(step=k,t=k,pairs=[dict(s_dir=None,s_dir_per_state_rms=None)]*2))
        failed=dict(status='failed',case=dict(key='second'),method_nfe=90,diagnostic_nfe=0)
        result=audit.summarize_cases([complete,failed])
        self.assertEqual(result['classification'],'INCOMPLETE_PILOT')
        self.assertEqual(result['method_nfe'],390)
        self.assertEqual(result['diagnostic_nfe'],120)


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ('torch','pywt','numpy')),
                     'PyTorch/PyWavelets unavailable; real PRE-last replay check awaits server')
class SnapshotTensorTests(unittest.TestCase):
    def test_actual_optimize_snapshot_replays_pre_update_last_output(self):
        import torch
        from modules.zaps_algorithm import ZAPS
        class Operator(torch.nn.Module):
            def H(self,x):return .5*x
            def transpose(self,r):return .5*r
        dm=SimpleNamespace(device='cpu',num_steps=1000,alphas_cumprod=torch.linspace(.99,.4,1000),
                           _predict_eps=lambda x,t:torch.zeros_like(x))
        config=dict(num_steps=4,schedule=(2,1,1),num_epochs=10,channels=1,img_size=8,wave='haar',level=1,
                    lr=.001,zeta_init=.1,d_init=.2,eta=1.,use_learned_var=False,sampler_mode='ddpm')
        torch.manual_seed(123)
        z=audit.snapshot_class()(dm,Operator(),**config)
        z.optimize(torch.zeros(1,1,8,8),verbose=False,x0_gt=None)
        state=z.audit_state
        self.assertEqual(z.audit_calls,10)
        self.assertFalse(torch.equal(state['zeta'],z.zeta.cpu()))
        replay=ZAPS(dm,Operator(),**config);replay.tau=state['tau']
        with torch.no_grad():replay.zeta.copy_(state['zeta']);replay.D.copy_(state['D'])
        torch.set_rng_state(state['cpu_rng'])
        output,nfe,_=replay.sample(torch.zeros(1,1,8,8),init_noise=state['init_noise'])
        self.assertEqual(nfe,4)
        torch.testing.assert_close(output,z._last_opt_x0,rtol=1e-7,atol=1e-7)


if __name__=='__main__':unittest.main()
