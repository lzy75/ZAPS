"""A75 native protocol/budget and actual worker CLI checks."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
from utils import dps_multiplier_transfer as rules
from utils import shrink_multitask as parent


class DesignTests(unittest.TestCase):
    def config(self):
        return dict(sampler='ddpm',steps=1000,noise_schedule='linear',model_mean_type='epsilon',
            model_var_type='learned_range',dynamic_threshold=False,clip_denoised=True,
            rescale_timesteps=False,timestep_respacing=1000)
    def test_native_config_and_no_scale_variance_or_grid_substitution(self):
        tasks={t:dict(conditioning=dict(method='ps',params=dict(scale=.3))) for t in rules.TASKS}
        rules.validate(self.config(),tasks)
        for key,value in (('timestep_respacing',30),('model_var_type','fixed_small')):
            c=dict(self.config());c[key]=value
            with self.assertRaises(ValueError):rules.validate(c,tasks)
        tasks['motion_deblur']['conditioning']['params']['scale']=.5
        with self.assertRaises(ValueError):rules.validate(self.config(),tasks)
    def test_frozen_case_budget(self):
        cases=rules.cases(parent.cases('/not-loaded'))
        self.assertEqual(len(cases),6);self.assertEqual({c['sigma'] for c in cases},{.05})
        self.assertEqual(len(cases)*len(rules.ARMS)*rules.STEPS,18000)
        self.assertEqual([sum(c['shard']==s for c in cases) for s in (0,1)],[3,3])
    def test_real_cli_without_loading_torch(self):
        result=subprocess.run([sys.executable,'utils/diag_dps_multiplier_transfer.py','--worker',
            '--shard','1','--device-uuid','GPU-test','--check-worker-cli'],
            cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout),dict(shard=1,device_uuid='GPU-test',model_calls=0))


if __name__=='__main__':unittest.main()
