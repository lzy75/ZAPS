import copy
from types import SimpleNamespace
import unittest
import numpy as np
from utils import ffhq_action_geometry as rules

GRID=[0,14,54,124,218,319,409,484,546,597,641,679,712,742,769,793,815,835,854,872,888,903,917,931,944,956,967,978,989,999]


class GeometryTests(unittest.TestCase):
    def test_budget_reuses_only_completed_parent_pulses(self):
        self.assertEqual(len(rules.expected_calls(GRID,False)),198)
        self.assertEqual(len(rules.expected_calls(GRID,True)),153)
        self.assertEqual(12*198+12*153+12*300,7812)
        changed=GRID[:];changed[-11]=871
        with self.assertRaises(ValueError):rules.expected_calls(changed,False)

    def test_same_time_action_curvature_and_informative_mask(self):
        direction=np.array([[[[.1,.2],[.3,.4]]]])
        clean={g:g*direction for g in rules.GAMMAS}
        obs={g:np.array([g,9999.]) for g in rules.GAMMAS}
        y=np.array([1.5,-9999.]);mask=np.array([True,False])
        result=rules.features(clean,obs,y,.05,{},mask,clean[1.])
        self.assertTrue(result['resolved']);self.assertEqual(result['active_measurements'],1)
        f=result['by_gamma'][2.]
        self.assertAlmostEqual(f['response_cos'],1.)
        self.assertAlmostEqual(f['response_turn'],0.)
        # gamma2 and gamma1 equally close in the informative measurement; outside noise is irrelevant.
        self.assertAlmostEqual(f['physical_gain'],0.)
        y[1]=1e20
        self.assertEqual(result,rules.features(clean,obs,y,.05,{},mask,clean[1.]))
        curved={g:g*direction+g*g*np.flip(direction,axis=-1) for g in rules.GAMMAS}
        self.assertGreater(rules.features(curved,obs,y,.05,{},mask,curved[1.])['by_gamma'][2.]['response_turn'],0.)
        flat={g:np.zeros_like(direction) for g in rules.GAMMAS}
        unresolved=rules.features(flat,obs,y,.05,{},mask,flat[1.])
        self.assertFalse(unresolved['resolved']);self.assertIsNone(unresolved['by_gamma'][2.]['response_cos'])

    def test_fold_excludes_task_and_image_everywhere_and_target_labels_never_fit(self):
        rows=[dict(task=t,image_id=str(i),step=k,gamma=g,features={'x':float(i)},labels={'psnr':.1*i*(g-1)+k})
              for t in rules.TASKS for i in range(6) for k in (10,19) for g in rules.GAMMAS]
        held=rows[0];train=rules.training_rows(rows,held,'LEAVE_TASK_AND_IMAGE_OUT')
        self.assertEqual(len({(r['task'],r['image_id']) for r in train}),15)
        self.assertTrue(all(r['task']!=held['task'] and r['image_id']!=held['image_id'] for r in train))
        test=[r for r in rows if r['task']==held['task'] and r['image_id']==held['image_id']]
        pred=rules.predict(train,test,('x',),'psnr')
        altered=copy.deepcopy(test)
        for row in altered:row['labels']['psnr']=1e20
        self.assertEqual(pred,rules.predict(train,altered,('x',),'psnr'))
        self.assertTrue(all(p==0. for p,r in zip(pred,test) if r['gamma']==1.))

    def test_four_task_summary_does_not_invent_state_value_from_uniform_gains(self):
        records=[]
        for task in rules.TASKS:
            for i in range(6):
                rec=dict(case=dict(task=task,image_id=str(i)),points=[],metrics={'C':dict(psnr=25.,ssim=.8,lpips=.2,loss=1.)})
                for k,t in rules.POINTS:
                    fs={str(g):{f:float(i)/10 for features in rules.FEATURES.values() for f in features} for g in rules.GAMMAS}
                    rec['points'].append(dict(step=k,response=dict(by_gamma=fs,resolved=True)))
                    for g in rules.GAMMAS:
                        if g!=1.:
                            gain=.1*(g-1.)
                            rec['metrics'][rules.arm(k,g)]=dict(psnr=25.+gain,ssim=.8,lpips=.2,loss=1.-gain)
                records.append(rec)
        result=rules.analyze(records)
        self.assertEqual(result['classification'],'NO_FOUR_TASK_RESPONSE_INFORMATION_SUPPORT')
        self.assertEqual(result['image_clusters'],6)
        for task in result['tasks']:
            for split in task['validation'].values():
                self.assertFalse(any(v['candidate'] for v in split['information'].values()))
                self.assertFalse(any(v['candidate'] for v in split['increments']))
            self.assertTrue(all(abs(v['unconstrained_psnr_oracle_headroom'])<1e-9 for v in task['opportunity']))

    def test_real_torch_core_gamma1_replay_and_affine_mask(self):
        try:import torch
        except ImportError:self.skipTest('Torch absent locally; mandatory server test')
        from modules.zaps_algorithm import ZAPS
        from utils import diag_multitask_state_transfer as parent
        from utils.diag_imagenet_gaussian_unit_domain import unit_operator
        from modules.degradations import InpaintingOperator
        from utils.diag_zaps_trace_audit import relative_error
        dm=SimpleNamespace(device=torch.device('cpu'),num_steps=1000,
            alphas_cumprod=torch.linspace(.9999,.01,1000),_predict_eps=lambda x,t:.01*x)
        raw=InpaintingOperator(random_ratio=.7,mode='random',box_size=0,noise_sigma=0.,seed=-1)
        mask=(torch.arange(32*32).reshape(1,1,32,32)%3==0).float();raw._mask_cache=mask
        operator=unit_operator(raw);self.assertTrue(parent.operator_gate(operator,'cpu',size=32)['passed'])
        z=ZAPS(dm,operator,img_size=32,level=1,use_learned_var=False)
        torch.manual_seed(88);initial=torch.randn(1,3,32,32);y=operator.H(torch.zeros_like(initial))
        positions=parent.source.fields.policy.Positions(
            parent.source.fields.policy.noise_u(dm.alphas_cumprod.numpy()),list(reversed(z.tau.tolist())))
        before=torch.get_rng_state()
        with torch.no_grad():original=z._reverse_diffusion(y,init_noise=initial)
        torch.set_rng_state(before);noises=[torch.randn_like(initial) for _ in range(29)]
        with torch.no_grad():
            x=initial.clone()
            for k,t in enumerate(positions.nominal):
                f=parent.source.fields.field(z,y,x,t,positions)
                x=parent.rules.risk.advance(z,x,f,positions,k,noises)
        self.assertLessEqual(relative_error(x,original),1e-6)


if __name__=='__main__':unittest.main()
