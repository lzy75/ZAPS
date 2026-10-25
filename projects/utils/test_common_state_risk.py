import copy
from types import SimpleNamespace
import unittest
import numpy as np
from utils import common_state_risk as rules


class RiskTests(unittest.TestCase):
    def test_history_causal_straight_reversal_and_unresolved(self):
        readout=dict(R=2.,cosine=.7,relative_response=.2)
        history=rules.History();v=np.array([1.,2.],dtype=np.float32)
        first=history.append(0.,0*v,v,v,readout,.1)
        second=history.append(.1,v,v,v,readout,.1)
        third=history.append(.2,2*v,v,v,readout,.1)
        frozen=copy.deepcopy(third)
        turn=history.append(.3,v,v,-v,readout,.1)
        self.assertEqual(third,frozen);self.assertIsNone(first['features']['clean_cos'])
        self.assertIsNone(second['features']['clean_cos'])
        self.assertAlmostEqual(third['features']['clean_cos'],1.)
        self.assertAlmostEqual(third['features']['clean_turn'],0.)
        self.assertAlmostEqual(turn['features']['clean_cos'],-1.)
        self.assertAlmostEqual(turn['features']['clean_turn'],1.)
        self.assertAlmostEqual(turn['features']['gradient_cos'],-1.)
        zero=rules.History()
        for u in (0.,.1,.2):r=zero.append(u,0*v,0*v,0*v,readout,0.)
        self.assertIsNone(r['features']['clean_cos']);self.assertIsNone(r['features']['q_gradient_cos'])
        tiny=rules.History()
        for u in (0.,.1,.2):r=tiny.append(u,np.ones(2,dtype=np.float32)*(1.+u*1e-7),v,v,readout,0.)
        self.assertIsNone(r['features']['clean_cos'])
        with self.assertRaises(ValueError):history.append(.3,v,v,v,readout,0.)

    def test_budget_fixed_centers_and_suffixes(self):
        grid=list(range(30));grid[-20]=641;grid[-25]=319
        calls=rules.expected_calls(grid)
        self.assertEqual(len(calls),75);self.assertEqual(12*rules.CALLS,900)
        self.assertEqual(calls[:30],grid[::-1]);self.assertEqual(calls[30:40],grid[::-1][20:])
        with self.assertRaises(ValueError):rules.expected_calls(list(range(30)))

    @staticmethod
    def cohort():
        records=[]
        for i in range(6):
            for sigma in rules.SIGMAS:
                features={name:float((i-2.5)*(j+1)/10+sigma) for j,name in enumerate(rules.P+rules.I+rules.G)}
                base=dict(psnr=25.,ssim=.8,lpips=.2,loss=1.)
                metrics={'C':base}
                for k,_ in rules.POINTS:
                    for gamma in rules.GAMMAS:
                        a=(gamma-1)*(i-2.5)*.03
                        metrics[rules.arm(k,gamma)]=dict(psnr=25.+a,ssim=.8+a*.01,
                                                       lpips=.2-a*.02,loss=1.-a)
                records.append(dict(case=dict(image_id=str(i),sampling_seed=1000,sigma=sigma),metrics=metrics,
                                    rows=[dict(state=dict(features=features)) for _ in range(30)]))
        return records

    def test_whole_image_label_holdout_and_emulator_no_GT_action_selection(self):
        records=self.cohort();first=rules.analyze(records)
        changed=copy.deepcopy(records)
        for r in changed:
            if r['case']['image_id']=='0':
                for arm,m in r['metrics'].items():
                    if arm!='C':
                        m.update(psnr=999.,ssim=-999.,lpips=999.,loss=999.)
        second=rules.analyze(changed)
        for name in rules.FEATURES:
            a=[v['gamma'] for v in first['offline_single_decision_emulators'][name]['choices'] if v['image_id']=='0']
            b=[v['gamma'] for v in second['offline_single_decision_emulators'][name]['choices'] if v['image_id']=='0']
            self.assertEqual(a,b)
        rows=[dict(image_id=str(i),seed=1000,step=s,features=dict(x=i*.2),labels=dict(y=i+s))
              for i in range(6) for s in (0,1)]
        a=rules.geometry.cross_validate(rows,('x',),'y')
        for r in rows:
            if r['image_id']=='0':r['labels']['y']+=1e6
        b=rules.geometry.cross_validate(rows,('x',),'y')
        self.assertEqual([v['prediction'] for v in a['predictions'] if v['image_id']=='0'],
                         [v['prediction'] for v in b['predictions'] if v['image_id']=='0'])
        with self.assertRaises(ValueError):rules.analyze(records[:-1])

    def test_torch_actual_core_passive_observer_cached_suffix_and_physical_gradient(self):
        try:import torch
        except ImportError:self.skipTest('Torch unavailable on Mac; mandatory on server')
        from modules.zaps_algorithm import ZAPS
        from utils import diag_frozen_response_transfer as run
        from utils.diag_cached_guidance_geometry import physical_descent
        class Operator(torch.nn.Module):
            def H(self,x):return .5*x
            def transpose(self,x):return .5*x
        dm=SimpleNamespace(device=torch.device('cpu'),num_steps=1000,
                           alphas_cumprod=torch.linspace(.9999,.01,1000),_predict_eps=lambda x,t:.01*x)
        z=ZAPS(dm,Operator(),img_size=16,level=1,use_learned_var=False)
        torch.manual_seed(123);initial=torch.randn(1,3,16,16);y=torch.zeros_like(initial)
        before=torch.get_rng_state()
        with torch.no_grad():original=z._reverse_diffusion(y,init_noise=initial)
        after=torch.get_rng_state();torch.set_rng_state(before)
        noises=[torch.randn_like(initial) for _ in range(29)]
        self.assertTrue(torch.equal(after,torch.get_rng_state()))
        positions=run.source.fields.policy.Positions(run.source.fields.policy.noise_u(dm.alphas_cumprod.numpy()),z.tau.tolist()[::-1])
        x=initial;anchors={};history=rules.History();rng=torch.get_rng_state()
        with torch.no_grad():
            for k,t in enumerate(positions.nominal):
                f=run.source.fields.field(z,y,x,t,positions)
                clean=f['clean'].clone();g,loss=physical_descent(z.A,clean,y,.05)
                self.assertTrue(torch.equal(clean,f['clean']))
                self.assertTrue(torch.allclose(g,-.25*clean/(clean.numel()*.05**2),rtol=1e-5,atol=1e-7))
                readout=run.rules.sensor(z,y,clean,f['q'],t,.05)
                history.append(float(positions.u[t]),clean[0].numpy(),f['q'][0].numpy(),g[0].numpy(),readout,f['clip_fraction'])
                if k in (19,24):anchors[k]=(x.clone(),{n:f[n].clone() for n in ('clean','q')})
                x=rules.advance(z,x,f,positions,k,noises)
            self.assertLess(float((x-original).abs().max()),1e-6)
            for k,(center,f) in anchors.items():
                suffix=rules.advance(z,center,f,positions,k,noises,1.)
                for j in range(k+1,30):
                    field=run.source.fields.field(z,y,suffix,positions.nominal[j],positions)
                    suffix=rules.advance(z,suffix,field,positions,j,noises)
                self.assertTrue(torch.equal(suffix,x))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
