import unittest
import numpy as np
from utils import diag_clean_field_response as d
try:
    import torch
except ImportError:
    torch=None


class ResponseTests(unittest.TestCase):
    def payload(self):
        directions={'q':np.array([1.,0.]),'orthogonal':np.array([0.,1.])}
        center=dict(clean=np.zeros(2),q=np.zeros(2),loss=1.,clip_fraction=0.)
        pairs=[]
        for name,axis in directions.items():
            for eta in d.ETAS:
                j=np.array([2.,.5])*axis
                pairs.append(dict(direction=name,eta=eta,input=dict(effective_span=2*eta,input_resolved=True),
                    plus=dict(clean=eta*j,loss=1.-eta,clip_fraction=0.),minus=dict(clean=-eta*j,loss=1.+eta,clip_fraction=0.)))
        return dict(centers=[center,dict(center)],pairs=pairs,directions=directions)

    def test_linear_map_directional_gain_and_loss_response(self):
        r=d.assess(self.payload());self.assertTrue(r['usable'])
        self.assertEqual(r['readout'],'GUIDANCE_MORE_SENSITIVE')
        np.testing.assert_allclose(r['q_over_orthogonal_gain'],[4.,4.])
        self.assertAlmostEqual(r['directions']['q']['pairs'][0]['phi'],-1.)

    def test_unresolved_and_scale_inconsistent_vectors_are_retained(self):
        p=self.payload()
        for pair in p['pairs']:
            if pair['direction']=='q' and pair['eta']==.02:
                pair['plus']['clean']*=-1;pair['minus']['clean']*=-1
        self.assertEqual(d.assess(p)['readout'],'UNRESOLVED_OR_SCALE_DEPENDENT')
        p=self.payload()
        for pair in p['pairs']:pair['input']['input_resolved']=False
        self.assertFalse(d.assess(p)['usable'])
        p=self.payload()
        for pair in p['pairs']:
            pair['plus']['clean']*=1e-12;pair['minus']['clean']*=1e-12
        self.assertFalse(d.assess(p)['usable'])

    def test_orthogonal_draw_matches_rms_and_is_reproducible(self):
        q=np.arange(1,49,dtype=np.float32).reshape(1,3,4,4)
        a,meta=d.orthogonal(q,123);b,_=d.orthogonal(q,123)
        np.testing.assert_array_equal(a,b)
        self.assertLessEqual(meta['abs_cosine'],1e-6)
        self.assertLessEqual(meta['rms_relative_error'],1e-6)
        with self.assertRaises(ValueError):d.orthogonal(q*0,123)

    def test_image_level_requires_both_seeds_and_failure_cost_is_retained(self):
        records=[]
        for i in ('00023','00024','00025'):
            for seed in (1000,2000):
                records.append(dict(status='complete',case=dict(key=f'{i}_s{seed}',image_id=i),method_nfe=0,diagnostic_nfe=10,
                    response=d.assess(self.payload()),q_rms=1.,center_clip=0.,existing_final_gain={}))
        plan=dict(experiment='test',source_t=14)
        self.assertEqual(d.summarize(plan,records)['classification'],'GUIDANCE_ALIGNED_RESPONSE_CANDIDATE')
        for r in records:
            if r['case']['key'].endswith('2000'):r['response']['readout']='NO_RESOLVED_DIFFERENCE'
        self.assertEqual(d.summarize(plan,records)['image_counts_both_seeds']['GUIDANCE_MORE_SENSITIVE'],0)
        records[-1]=dict(status='failed',method_nfe=0,diagnostic_nfe=3)
        r=d.summarize(plan,records)
        self.assertEqual(r['classification'],'INCOMPLETE_PILOT');self.assertEqual(r['diagnostic_nfe'],53)


@unittest.skipIf(torch is None,'Server CPU Torch contract required; no UNet loaded')
class TorchTests(unittest.TestCase):
    def test_exact_ten_evaluations_same_time_and_input_centers(self):
        from utils.diag_guidance_local_probe import perturb_inputs
        x=torch.zeros(1,3,4,4);q=torch.arange(1,49,dtype=torch.float32).reshape(x.shape)*.01
        a,_=d.orthogonal(q.numpy(),123);axes={'q':q,'orthogonal':torch.from_numpy(a)};seen=[]
        def evaluate(z,y,input_,t,index):
            seen.append(input_.clone());self.assertEqual((t,index),(14,1))
            return dict(clean=.5*input_,q=q,loss=float(input_.double().square().mean()),clip_fraction=0.)
        before=torch.get_rng_state().clone()
        centers,pairs=d.collect(None,None,x,axes,14,1,evaluate,perturb_inputs)
        self.assertEqual(len(seen),10);self.assertEqual(len(pairs),4)
        for c in seen[:2]:torch.testing.assert_close(c,x,atol=0,rtol=0)
        result=d.assess(dict(centers=centers,pairs=pairs,directions=axes))
        self.assertTrue(result['usable']);self.assertEqual(result['readout'],'NO_RESOLVED_DIFFERENCE')
        for response in result['directions'].values():
            for pair in response['pairs']:self.assertAlmostEqual(pair['gain'],.5,places=6)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
