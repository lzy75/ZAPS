import copy
from types import SimpleNamespace
import unittest
import numpy as np
from utils import multitask_state_transfer as rules


class TransferTests(unittest.TestCase):
    def test_fixed_budget_and_preset_configs(self):
        self.assertEqual(24*(300+rules.MEASURE_CALLS+rules.TIME_CALLS),11160)
        grid=list(range(30));grid[-20]=641;grid[-25]=319
        for kind,n in (('BASELINE',300),('MEASURE',105),('TIME',60)):
            self.assertEqual(len(rules.expected_calls(grid,kind)),n)
        original=dict(num_steps=30,num_epochs=10,lr=.01,zeta_init=.1)
        self.assertEqual(rules.configuration(original,'imagenet','gaussian_deblur')['lr'],.001)
        self.assertEqual(rules.configuration(original,'ffhq','gaussian_deblur')['zeta_init'],.2)
        self.assertEqual(original['zeta_init'],.1)

    @staticmethod
    def records():
        records=[]
        for d in rules.DATASETS:
            for t in rules.TASKS:
                for i in range(6):
                    records.append(dict(case=dict(dataset=d,task=t,image_id=str(i),key=f'{d}_{t}_{i}'),
                        state_trace=[dict(t=k,gamma=.2+i*.1) for k in reversed(range(30))],
                        metrics=dict(psnr=-100.)))
        return records

    def test_task_profile_excludes_held_image_and_ignores_quality_labels(self):
        records=self.records();a=rules.profiles(records,list(range(30)))
        changed=copy.deepcopy(records)
        for r in changed:r['metrics']['psnr']=1e9
        self.assertEqual(a,rules.profiles(changed,list(range(30))))
        key=records[0]['case']['key'];self.assertNotIn(key,a[key]['calibration_cases'])
        self.assertEqual(len(a[key]['calibration_cases']),5)
        changed[0]['state_trace']=[dict(t=k,gamma=1.9) for k in reversed(range(30))]
        b=rules.profiles(changed,list(range(30)));self.assertEqual(a[key],b[key])
        with self.assertRaises(ValueError):rules.profiles(records[:-1],list(range(30)))

    def test_frozen_feature_transfer_never_fits_target_labels(self):
        train=[dict(image_id='SOURCE_'+str(i),step=s,features=dict(x=float(i)),labels=dict(psnr=float(2*i+s)))
               for i in range(6) for s in (0,1)]
        test=[dict(image_id='TARGET_'+str(i),step=s,features=dict(x=float(i)),labels=dict(psnr=0.))
              for i in range(6) for s in (0,1)]
        for features in ((),('x',)):
            a=rules.predict_transfer(train,test,features,'psnr')
            changed=copy.deepcopy(test)
            for r in changed:r['labels']['psnr']=1e6
            b=rules.predict_transfer(train,changed,features,'psnr')
            self.assertEqual([v['prediction'] for v in a['predictions']],[v['prediction'] for v in b['predictions']])
        overlap=copy.deepcopy(test);overlap[0]['image_id']='SOURCE_0'
        with self.assertRaises(ValueError):rules.predict_transfer(train,overlap,(),'psnr')

    def test_large_model_gpu_respects_allocation_and_serial_fallback(self):
        rows=[dict(index='0',uuid='GPU-a',free=8500),dict(index='1',uuid='GPU-b',free=14000)]
        self.assertEqual(rules.select_devices(rows),['1','1'])
        with self.assertRaises(ValueError):rules.select_devices(rows,visible='0')
        with self.assertRaises(ValueError):rules.select_devices(rows,first='0')
        with self.assertRaises(ValueError):rules.select_devices(rows,visible='1',first='0')

    def test_summary_does_not_call_C_gain_state_increment(self):
        from utils.test_common_state_risk import RiskTests
        source=RiskTests.cohort();records=self.records()
        for i,r in enumerate(records):
            base=dict(psnr=25.,ssim=.8,lpips=.2,loss=1.)
            improved=dict(psnr=25.1,ssim=.801,lpips=.199,loss=.9)
            r['case'].update(sigma=.05,sampling_seed=1000)
            r['metrics']={a:dict(improved if a!='C' else base) for a in rules.ARMS}
            r['metrics']['NAIVE']=dict(base)
            for k,_ in rules.risk.POINTS:
                for gamma in rules.risk.GAMMAS:r['metrics'][rules.risk.arm(k,gamma)]=dict(improved)
            features={n:float(i%6)*.2 for n in rules.risk.P+rules.risk.I+rules.risk.G}
            r['rows']=[dict(state=dict(features=features)) for _ in range(30)]
            r['state_trace']=[dict(t=k,gamma=1.,cap_hit=False,fallback=False) for k in reversed(range(30))]
            r['time_traces']={a:[dict(gamma=1.)]*30 for a in ('TIME_SOURCE','TIME_TASK')}
        summary=rules.summarize(records,source)
        self.assertEqual(len(summary['state_gain_conditions']),4)
        self.assertEqual(summary['state_increment_conditions'],[])
        self.assertEqual(summary['classification'],'NO_CONFIRMED_MULTITASK_STATE_INCREMENT')
        with self.assertRaises(ValueError):rules.summarize(records+[records[0]],source)

    def test_torch_real_exact_operators_and_original_core_replay(self):
        try:import torch
        except ImportError:self.skipTest('Torch unavailable on Mac; mandatory on server')
        from modules.degradations import SuperResolutionOperator,GaussianBlurOperator
        from modules.zaps_algorithm import ZAPS
        from utils.diag_imagenet_gaussian_unit_domain import unit_operator
        from utils import diag_multitask_state_transfer as run
        operators=(SuperResolutionOperator(4,noise_sigma=0.,in_shape=(1,3,32,32)),
                   GaussianBlurOperator(5,1.,noise_sigma=0.))
        dm=SimpleNamespace(device=torch.device('cpu'),num_steps=1000,
            alphas_cumprod=torch.linspace(.9999,.01,1000),_predict_eps=lambda x,t:.01*x)
        for raw in operators:
            operator=unit_operator(raw);gate=run.operator_gate(operator,'cpu',size=32)
            self.assertTrue(gate['passed'])
            z=ZAPS(dm,operator,img_size=32,level=1,use_learned_var=False)
            torch.manual_seed(231);initial=torch.randn(1,3,32,32);y=operator.H(torch.zeros_like(initial))
            before=torch.get_rng_state()
            with torch.no_grad():original=z._reverse_diffusion(y,init_noise=initial)
            after=torch.get_rng_state();torch.set_rng_state(before)
            noises=[torch.randn_like(initial) for _ in range(29)]
            self.assertTrue(torch.equal(after,torch.get_rng_state()))
            positions=run.source.fields.policy.Positions(run.source.fields.policy.noise_u(dm.alphas_cumprod.numpy()),z.tau.tolist()[::-1])
            x=initial;rng=torch.get_rng_state()
            with torch.no_grad():
                for k,t in enumerate(positions.nominal):
                    field=run.source.fields.field(z,y,x,t,positions)
                    x=rules.risk.advance(z,x,field,positions,k,noises)
            relative=float((x-original).double().norm()/original.double().norm().clamp(min=1e-12))
            self.assertLessEqual(relative,1e-6)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
