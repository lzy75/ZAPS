import unittest
from utils import diag_endpoint_components as d
try:
    import torch
except ImportError:
    torch=None


class ProtocolTests(unittest.TestCase):
    def effect(self,psnr,ssim,lpips,loss):
        c=dict(psnr=20.,ssim=.7,lpips=.3,loss=1.)
        f={k:c[k]+v for k,v in dict(psnr=psnr,ssim=ssim,lpips=lpips,loss=loss).items()}
        return d.coupled.history.effect(f,c,{k:0. for k in c})

    def test_retention_requires_all_four_metrics_and_resolved_full_gain(self):
        full=self.effect(.2,.002,-.01,-.1)
        self.assertTrue(d.retention(self.effect(.17,.0017,-.0085,-.085),full))
        self.assertFalse(d.retention(self.effect(.17,.0017,-.006,-.085),full))
        self.assertFalse(d.retention(self.effect(.17,.0017,-.0085,-.02),full))
        self.assertFalse(d.retention(self.effect(.17,.0017,-.0085,-.085),self.effect(0.,0.,0.,0.)))

    def test_cluster_retention_and_failed_calls(self):
        records=[]
        for image in ('00023','00024','00025'):
            for seed in (1000,2000):
                baseline=dict(loss=1.,psnr=20.,ssim=.7,lpips=.3)
                effects={f'{a}{b}_C':self.effect(.2,.002,-.01,-.1) for a in ('F','P','L') for b in (0,1)}
                effects.update({f'{a}{b}_F{b}':self.effect(0.,0.,0.,0.) for a in ('P','L') for b in (0,1)})
                records.append(dict(status='complete',case=dict(key=f'{image}_s{seed}',image_id=image),method_nfe=0,diagnostic_nfe=10,
                    branches={a:baseline for a in ('C','F0','F1','P0','P1','L0','L1')},effects=effects,
                    joint_candidates={a:{'0':True,'1':True} for a in ('F','P','L')},retention_80=dict(P={'0':True,'1':True},L={'0':False,'1':False}),gates={'test':{'passed':True}}))
        result=d.summarize(records);self.assertEqual(result['diagnostic_nfe'],60)
        self.assertEqual(result['classification'],'PRIOR_COMPONENT_SUFFICIENT_CANDIDATE')
        for r in records:
            if r['case']['key'].endswith('2000'):r['retention_80']['P']['1']=False
        self.assertEqual(d.summarize(records)['classification'],'NO_SINGLE_COMPONENT_SUFFICIENCY_ESTABLISHED')
        self.assertEqual(d.summarize(records[:1]+[dict(status='failed',method_nfe=0,diagnostic_nfe=3)])['diagnostic_nfe'],13)


@unittest.skipIf(torch is None,'Server Torch required; no UNet loaded')
class ComponentTests(unittest.TestCase):
    def test_stable_full_matches_existing_formula_and_null_terms_preserve_C(self):
        gen=torch.Generator(device='cpu').manual_seed(8)
        c,z,q,zm,qm=torch.randn((5,1,3,8,8),generator=gen)
        spec=d.coupled.contract.interval(.9,.99,.9999)
        arms,dp,dl=d.components(c,z,q,zm,qm,spec)
        torch.testing.assert_close(arms['F'],d.coupled.refined_endpoint(c,z,q,zm,qm,spec),atol=0,rtol=0)
        torch.testing.assert_close(arms['P'],(c.double()+dp).float(),atol=0,rtol=0)
        torch.testing.assert_close(arms['L'],(c.double()+dl).float(),atol=0,rtol=0)
        null,_,_=d.components(c,z,q,z,q,spec)
        for value in null.values():torch.testing.assert_close(value,c,atol=0,rtol=0)
        with self.assertRaises(ValueError):d.components(c,z,q,zm[:,:,:,:4],qm,spec)


if __name__=='__main__':unittest.main()
