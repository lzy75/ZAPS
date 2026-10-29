"""A71 frozen small FFHQ task/noise panel; no new policy fitting."""
from utils.historical_ffhq_gain import config as strong_config
TASKS=('super_resolution','gaussian_deblur','inpainting','motion_deblur')
SIGMAS=(.025,.05,.10)
ARMS=('BASE','SHRINK')
IMAGES=('00226','00443','00963')

def configuration(task,size=256,level=3):
    if task not in TASKS:raise ValueError('Unregistered task')
    out=strong_config();out.update(zeta_init=.2 if task in ('gaussian_deblur','motion_deblur') else .1,img_size=size,level=level)
    return out

def cases(folder):
    from pathlib import Path
    return [dict(image_id=i,image=str(Path(folder)/(i+'.png')),task=task,sigma=sigma,
        seed=1000+int(i),measurement_seed=1000+int(i),mask_seed=2001000+int(i),shard=(j*12+TASKS.index(task)*3+SIGMAS.index(sigma))%2,
        key=f'{task}_{i}_n{round(sigma*1000):03d}_s{1000+int(i)}')
        for j,i in enumerate(IMAGES) for task in TASKS for sigma in SIGMAS]

def parent_key(c):return f"{c['image_id']}_n{round(c['sigma']*1000):03d}_s{c['seed']}"

def reusable(c):return c['task']=='super_resolution' and c['sigma'] in (.05,.10)
