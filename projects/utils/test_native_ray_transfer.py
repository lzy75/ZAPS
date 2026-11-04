"""No-Torch source/config gates for all three independent host experiments."""
import ast
import unittest
from pathlib import Path
from utils import native_ray_transfer as rules
from utils.native_ray_transfer_runtime import RayHook


class Tests(unittest.TestCase):
    def test_pinned_bytes_and_exactly_one_insertion(self):
        for experiment,relative,names in (
            ('A82','linear_inv/guided_diffusion/condition_methods.py',('WoProjSampling',)),
            ('A83','guided_diffusion/condition_methods.py',('PosteriorSampling_meng',)),
            ('A84','algos/pgdm.py',('PGDM',))):
            rules.verify(experiment)
            tree=ast.parse((rules.vendor(experiment)/relative).read_text())
            nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in names]
            transform=RayHook(experiment);changed=transform.visit(ast.Module(body=nodes,type_ignores=[]))
            self.assertEqual(transform.matches,1)
            compile(ast.fix_missing_locations(changed),relative,'exec')

    def test_native_configs_and_actual_budget(self):
        for task in rules.TASKS:
            mpgd=rules.configuration('A82',task)
            self.assertEqual(mpgd['diffusion']['timestep_respacing'],'ddim100')
            self.assertEqual(mpgd['diffusion']['eta'],.5)
            self.assertEqual(mpgd['scale'],5.)
            dmps=rules.configuration('A83',task)
            self.assertEqual(dmps['diffusion']['steps'],1000)
            self.assertEqual(dmps['task']['conditioning']['params']['scale'],1.75)
            pgdm=rules.configuration('A84',task)
            self.assertEqual(pgdm['steps'],50)
            self.assertEqual(pgdm['algo']['grad_term_weight'],1)
            self.assertEqual(pgdm['algo']['eta'],0.)
        self.assertEqual(sum(12*v['nfe'] for v in rules.HOSTS.values()),13800)

    def test_fixed_panel_has_no_outcome_selection(self):
        parent=[dict(image_id=i,image='/tmp/'+i+'.png',dataset='ffhq') for i in rules.IDS]
        for experiment in rules.HOSTS:
            cases=rules.cases(experiment,parent)
            self.assertEqual(len(cases),6)
            self.assertEqual({c['image_id'] for c in cases},set(rules.IDS))
            self.assertEqual({c['sigma'] for c in cases},{.05})
            self.assertEqual({c['shard'] for c in cases},{0,1})

    def test_cli_requires_explicit_worker_device(self):
        import subprocess,sys
        r=subprocess.run([sys.executable,'utils/diag_native_ray_transfer.py','--experiment','A82','--worker'],
            cwd=rules.PROJECTS,capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0)
        self.assertIn('allocated GPU UUID',r.stderr)


if __name__=='__main__':unittest.main()
