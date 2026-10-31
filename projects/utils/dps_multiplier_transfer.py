"""A75: native1000 DPS, same existing observations, frozen scalar transfer."""
from utils import all_unroll_shrink as old
ARMS = ('DPS', 'PROJECT', 'SHRINK')
TASKS = ('super_resolution', 'motion_deblur')
STEPS = 1000
SCALE = .3


def cases(parent_cases):
    out=[dict(c) for c in old.cases(parent_cases) if c['task'] in TASKS]
    for i,c in enumerate(out):c['shard']=i%2
    return out


def validate(config, tasks):
    expected=dict(sampler='ddpm',steps=1000,noise_schedule='linear',model_mean_type='epsilon',
                  model_var_type='learned_range',dynamic_threshold=False,clip_denoised=True,rescale_timesteps=False)
    if any(config.get(k)!=v for k,v in expected.items()) or str(config.get('timestep_respacing'))!='1000':
        raise ValueError('Native DPS1000 sampler/variance configuration differs; no silent substitution')
    for task in TASKS:
        c=tasks[task]['conditioning']
        if c['method']!='ps' or c['params']['scale']!=SCALE:raise ValueError('Need native PS scale=.3')
