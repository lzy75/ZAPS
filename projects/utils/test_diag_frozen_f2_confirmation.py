import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from utils import diag_frozen_f2_confirmation as d


def records():
    result=[]
    for image in d.IDS:
        for seed in d.SEEDS:
            c=dict(loss=1.,psnr=20.,ssim=.7,lpips=.2)
            f1=dict(loss=.9,psnr=20.1,ssim=.701,lpips=.201)
            f2=dict(loss=.8,psnr=20.2,ssim=.702,lpips=.204)
            result.append(dict(status='complete',case=dict(key=f'{image}_s{seed}',image_id=image),method_nfe=300,diagnostic_nfe=2,
                branches=dict(C=c,F1=f1,F2=f2),floors=dict.fromkeys(d.METRICS,0.),seconds=1.,gates=dict(test=dict(passed=True))))
    return result


class ProtocolTests(unittest.TestCase):
    def test_both_seeds_stay_on_one_worker(self):
        cases=[r['case'] for r in records()]
        shards=[d.shard_cases(cases,i) for i in (0,1)]
        self.assertEqual([len(s) for s in shards],[6,6])
        self.assertFalse({c['image_id'] for c in shards[0]} & {c['image_id'] for c in shards[1]})
        self.assertEqual({c['key'] for s in shards for c in s},{c['key'] for c in cases})

    def test_history_checks_id_key_hash_and_corrupt_metadata(self):
        cases=[dict(image_id='00026',image_sha256='a'*64)]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'run.json'
            for v in (dict(image_id='00026'),dict(case='00026_s1000'),dict(image_sha256='a'*64)):
                p.write_text(json.dumps(v))
                with self.assertRaises(ValueError):d.history_audit([p],cases)
            p.write_text(json.dumps(dict(image_id='00020')))
            self.assertEqual(len(d.history_audit([p],cases)),1)
            p.write_text('')
            with self.assertRaises(json.JSONDecodeError):d.history_audit([p],cases)

    def test_cluster_threshold_is_four_of_six_and_failure_cost_retained(self):
        r=records();s=d.summarize(r)
        self.assertEqual(s['classification'],'FROZEN_F2_SUPPORTED_ON_FRESH_COHORT')
        self.assertEqual((s['method_nfe'],s['diagnostic_nfe']),(3600,24))
        self.assertEqual(s['cumulative_queue_cost']['total'],9512)
        for x in r:
            if x['case']['image_id'] in d.IDS[3:]:x['branches']['F2']['psnr']=20.09
        s=d.summarize(r)
        self.assertEqual(s['comparisons']['F2_F1']['images_psnr_better'],3)
        self.assertEqual(s['classification'],'FROZEN_F2_NOT_SUPPORTED_ON_FRESH_COHORT')
        s=d.summarize(r[:1]+[dict(status='failed',method_nfe=10,diagnostic_nfe=0)])
        self.assertEqual((s['method_nfe'],s['diagnostic_nfe']),(310,2))
        self.assertEqual(s['classification'],'INCOMPLETE_PILOT')

    def test_quality_guard_and_per_case_warning_are_separate(self):
        r=records()
        r[0]['branches']['F2']['lpips']=.21
        s=d.summarize(r)
        self.assertEqual(s['classification'],'FROZEN_F2_SUPPORTED_ON_FRESH_COHORT')
        self.assertEqual(s['per_case_lpips_over_mean_tolerance'],[r[0]['case']['key']])
        for x in r:x['branches']['F2']['lpips']=.206
        self.assertEqual(d.summarize(r)['classification'],'FROZEN_F2_NOT_SUPPORTED_ON_FRESH_COHORT')

    def test_training_contract_rejects_wrong_time_or_snapshot_and_rng(self):
        initial=SimpleNamespace(equal=lambda x:x=='same')
        plan=dict(grid=list(range(30)))
        r=dict(method_nfe=300,loss_history=[1.]*10,calls=[dict(t=t) for t in plan['grid'][::-1]*10])
        z=SimpleNamespace(_last_nfe=300,audit_calls=10,_last_opt_noise='same',audit_state={},dm=SimpleNamespace(model=SimpleNamespace(parameters=lambda:[])))
        d.training_contract(r,z,[None]*300,initial,1,1,lambda a,b:a==b,plan)
        bad=copy.deepcopy(r);bad['calls'][-1]['t']=999
        with self.assertRaises(RuntimeError):d.training_contract(bad,z,[None]*300,initial,1,1,lambda a,b:a==b,plan)
        with self.assertRaises(RuntimeError):d.training_contract(r,z,[None]*300,initial,1,2,lambda a,b:a==b,plan)
        z.audit_state=None
        with self.assertRaises(RuntimeError):d.training_contract(r,z,[None]*300,initial,1,1,lambda a,b:a==b,plan)


if __name__=='__main__':unittest.main()
