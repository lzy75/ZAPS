import unittest
from types import SimpleNamespace
from utils import diag_terminal_polish_control as d
try:
    import torch
except ImportError:
    torch=None


class ComparisonTests(unittest.TestCase):
    def test_both_bridges_required_and_tradeoffs_retained(self):
        c=dict(loss=1.,psnr=20.,ssim=.7,lpips=.3);s=dict(loss=.98,psnr=20.04,ssim=.701,lpips=.299)
        p=dict(loss=.95,psnr=20.08,ssim=.702,lpips=.297)
        branches=dict(C=c,S=s,P0=p,P1=p,F0=p,F1=p);floors={k:0. for k in c}
        _,readouts=d.compare(branches,floors)
        self.assertEqual(readouts['P'],'MIDPOINT_DOMINATES_S_BOTH_BRIDGES')
        branches['P1']=dict(p,lpips=.301)
        self.assertEqual(d.compare(branches,floors)[1]['P'],'TRADEOFF_OR_UNRESOLVED')
        branches.update(P0=c,P1=c)
        self.assertEqual(d.compare(branches,floors)[1]['P'],'S_DOMINATES_BOTH_MIDPOINT_BRIDGES')

    def test_summary_uses_image_clusters_not_mean_or_number_of_bridges(self):
        records=[]
        for image in ('00023','00024','00025'):
            for seed in (1000,2000):
                c=dict(loss=1.,psnr=20.,ssim=.7,lpips=.3);s=dict(loss=.98,psnr=20.04,ssim=.701,lpips=.299);p=dict(loss=.95,psnr=20.08,ssim=.702,lpips=.297)
                branches=dict(C=c,S=s,P0=p,P1=p,F0=p,F1=p);effects,readouts=d.compare(branches,{k:0. for k in c})
                records.append(dict(status='complete',case=dict(key=f'{image}_s{seed}',image_id=image),method_nfe=0,diagnostic_nfe=2,
                    branches=branches,effects=effects,readouts=readouts,S_joint_vs_C=True,gates={'test':{'passed':True}}))
        result=d.summarize(records);self.assertEqual(result['diagnostic_nfe'],12)
        self.assertEqual(result['classification'],'MIDPOINT_PRIOR_MATCHED_COST_ADVANTAGE_CANDIDATE')
        for r in records:
            if r['case']['key'].endswith('2000'):r['readouts']['P']='TRADEOFF_OR_UNRESOLVED'
        self.assertEqual(d.summarize(records)['classification'],'NO_ROBUST_MIDPOINT_ADVANTAGE_ESTABLISHED')
        self.assertEqual(d.summarize(records[:1]+[dict(status='failed',method_nfe=0,diagnostic_nfe=1)])['diagnostic_nfe'],3)


@unittest.skipIf(torch is None,'Server Torch required; no UNet loaded')
class RepeatTests(unittest.TestCase):
    def test_two_evaluations_start_from_original_C_not_previous_S(self):
        z=SimpleNamespace(dm=SimpleNamespace(alphas_cumprod=torch.tensor([.9999])))
        c=torch.full((1,3,4,4),.4);seen=[]
        def evaluate(z,y,x,t,index):
            seen.append(x.clone());self.assertEqual((t,index),(0,0))
            return dict(clean=x*.5,q=torch.full_like(x,.02))
        fields,outputs=d.polish_pair(z,None,c,evaluate)
        self.assertEqual(len(seen),2)
        for x in seen:torch.testing.assert_close(x,c,atol=0,rtol=0)
        torch.testing.assert_close(outputs[0],outputs[1],atol=0,rtol=0)
        torch.testing.assert_close(outputs[0],c*.5+.02,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
