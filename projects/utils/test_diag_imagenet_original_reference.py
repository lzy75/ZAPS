"""Original-setting contracts; optional cached tensor arithmetic, no UNet."""
import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

from utils import diag_imagenet_original_reference as audit


class ContractTests(unittest.TestCase):
    def config(self):
        return dict(num_steps=30,schedule=[15,10,5],timestep_spacing="linear",schedule_power=2.,
            num_epochs=10,lr=.001,zeta_init=.2,d_init=.2,eta=1.,use_learned_var=False,
            sampler_mode="ddpm",wave="db4",level=3,surrogate_score_jacobian=False)

    def test_original_only_not_tuned_or_state(self):
        source=self.config(); audit.original_config(source)
        for key,value in (("zeta_init",.04002983177868308),("lr",.05),("eta",.75),("schedule",[30]),
                          ("surrogate_score_jacobian",True),("use_learned_var",True)):
            other=copy.deepcopy(source); other[key]=value
            with self.assertRaises(ValueError): audit.original_config(other)
        other=dict(source,state_scheduler=True)
        with self.assertRaises(ValueError): audit.original_config(other)

    def sampler(self):
        row=dict(t=999,time_mapping_passed=True,checks={k:dict(passed=k!="x0",relative_error=4e-5,
            max_absolute_error=8e-5) for k in ("x0","mean_shared_x0","mean","std","next")})
        rows=[row]+[dict(t=998-i,time_mapping_passed=True,checks={k:dict(passed=True)
            for k in row["checks"]}) for i in range(29)]
        arm=dict(passed=False,replay_passed=True,nfe=90,variance_policy="fixed_small",rows=rows)
        return dict(results={name:copy.deepcopy(arm) for name in audit.ARMS})

    def test_old_failure_is_preserved_not_renamed_pass(self):
        result=audit.old_sampler_scope(self.sampler())
        for arm in result.values():
            self.assertFalse(arm["old_overall_passed"])
            self.assertEqual(arm["failures_preserved"][0]["component"],"x0")

    def test_posterior_failure_or_incomplete_archive_rejected(self):
        for kind in ("std","replay","coverage"):
            record=self.sampler(); arm=record["results"][audit.ARMS[0]]
            if kind=="std": arm["rows"][0]["checks"]["std"]["passed"]=False
            if kind=="replay": arm["replay_passed"]=False
            if kind=="coverage": arm["rows"].pop()
            with self.assertRaises(ValueError): audit.old_sampler_scope(record)

    def test_cli_has_no_model_loader_optimizer_or_sampling_call(self):
        tree=ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
        forbidden={"load_diffusion_model","load_imagenet_model","load_ffhq_model","optimize","sample","p_sample"}
        calls={getattr(n.func,"attr",getattr(n.func,"id",None)) for n in ast.walk(tree) if isinstance(n,ast.Call)}
        self.assertFalse(calls & forbidden)


@unittest.skipUnless(importlib.util.find_spec("torch"),"PyTorch unavailable; cached numerical tests not run")
class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from modules import zaps_algorithm as core
        from modules.diffusion_model import _precompute_diffusion_params
        from utils.diag_zaps_sampler_parity import load_dps_reference
        cls.torch,cls.core=torch,core
        cls.ab=_precompute_diffusion_params(1000)["alphas_cumprod"]
        cls.dps,_=load_dps_reference(audit.ROOT/"DPS")

    @classmethod
    def tearDownClass(cls):
        import sys
        for name in list(sys.modules):
            if name.startswith("_zaps_sampler_dps_reference"): del sys.modules[name]

    def case(self,t):
        torch=self.torch
        grid=[0,143,333,667,916,999]; index=grid.index(t)
        ref=self.dps.DDPM(use_timesteps=grid,betas=self.dps.get_named_beta_schedule("linear",1000),
            model_mean_type="epsilon",model_var_type="fixed_small",dynamic_threshold=False,
            clip_denoised=True,rescale_timesteps=False)
        x=torch.linspace(-20,20,16).reshape(1,1,4,4)
        eps=(x-.1*self.ab[t].sqrt())/(1-self.ab[t]).sqrt()
        return dict(t=t,index=index,x=x,epsilon=eps),grid,ref

    def test_shared_posterior_algebra_and_rng_all_selected_times(self):
        torch=self.torch
        for t in (999,916,667,333,143,0):
            probe,grid,ref=self.case(t)
            before=torch.get_rng_state().clone()
            row=audit.cached_sampler_checks(probe,grid,self.ab,ref,self.core)
            self.assertTrue(row["shared_posterior_passed"],row)
            self.assertTrue(row["fp64_algebra_passed"],row)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            self.assertIsInstance(row["checks"]["fp32_x0_original_tolerance"]["passed"],bool)

    def test_wrong_noise_std_and_wrong_time_rejected(self):
        probe,grid,ref=self.case(999)
        def wrong(*args,**kw):
            kw["eta"]=.75*kw.get("eta",1.)
            return self.core.ddpm_posterior_step(*args,**kw)
        corrupted=SimpleNamespace(ZAPS=self.core.ZAPS,ddpm_posterior_step=wrong)
        row=audit.cached_sampler_checks(probe,grid,self.ab,ref,corrupted)
        self.assertFalse(row["shared_posterior_passed"])
        probe["index"]-=1
        with self.assertRaises(ValueError): audit.cached_sampler_checks(probe,grid,self.ab,ref,self.core)

    def test_wrong_tweedie_not_excused_as_float_precision(self):
        probe,grid,ref=self.case(999)
        def wrong(*args,**kw):
            return 1.02*self.core.ZAPS._tweedie_estimate(*args,**kw)
        corrupted=SimpleNamespace(ZAPS=SimpleNamespace(_tweedie_estimate=wrong),
            ddpm_posterior_step=self.core.ddpm_posterior_step)
        row=audit.cached_sampler_checks(probe,grid,self.ab,ref,corrupted)
        self.assertFalse(row["fp64_algebra_passed"])


if __name__=="__main__": unittest.main()
