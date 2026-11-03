"""A79 frozen DiffPIR-paper deblur settings, distinct from DPS."""
from pathlib import Path
import hashlib
import json
import math

PROJECTS=Path(__file__).resolve().parents[1]
VENDOR=PROJECTS/'vendor/diffpir_2a989812'
ARMS=('DIFFPIR','SHRINK')
TASKS=('gaussian_deblur','motion_deblur')
DATASETS=('ffhq','imagenet')
IDS=dict(ffhq=('00226','00443','00963'),imagenet=('00067','00151','00193'))
SIGMA_PHYSICAL=.05
STEPS=100
MIN_FREE=6*1024**3
CHECKPOINTS=dict(ffhq=('ffhq_10m.pt','81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa'),
 imagenet=('256x256_diffusion_uncond.pt','a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1'))
PAPER_PARAMETERS=dict(ffhq=dict(gaussian_deblur=(7.,.3),motion_deblur=(7.,.4)),
 imagenet=dict(gaussian_deblur=(8.,.3),motion_deblur=(8.,.7)))


def verify_upstream():
    metadata=json.loads((VENDOR/'UPSTREAM.json').read_text())
    for row in metadata['files']:
        data=(VENDOR/row['path']).read_bytes()
        actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if actual!=row['git_blob_sha']:raise ValueError('Pinned DiffPIR source changed: '+row['path'])
    return metadata


def sequence(steps=STEPS,training_steps=1000):
    import numpy as np
    if steps<2 or steps>training_steps:raise ValueError('Invalid native grid')
    result=[int(v) for v in np.sqrt(np.linspace(0,training_steps**2,steps))]
    result[-1]-=1
    if len(set(result))!=steps:raise ValueError('Repeated native sequence index')
    return result


def cases(folders):
    out=[]
    for dataset in DATASETS:
        for image_id in IDS[dataset]:
            for task in TASKS:
                lam,zeta=PAPER_PARAMETERS[dataset][task]
                # Task pairing stays on the same shard and model, with balanced arm order.
                out.append(dict(key=f'{dataset}_{task}_{image_id}',dataset=dataset,image_id=image_id,
                    image=str(Path(folders[dataset])/(image_id+'.png')),task=task,seed=1000+int(image_id),
                    measurement_seed=2000+int(image_id),sigma_physical=SIGMA_PHYSICAL,sigma_model=2*SIGMA_PHYSICAL,
                    lambda_=lam,zeta=zeta,shard=DATASETS.index(dataset),arms=list(ARMS)))
    return out


def configuration(case,steps=STEPS):
    return dict(task='deblur',generate_mode='DiffPIR',model_output_type='pred_xstart',model_out_type='pred_xstart',
        sub_1_analytic=True,skip_type='quad',iter_num=steps,iter_num_U=1,num_train_timesteps=1000,t_start=999,
        noise_model_t=0,ddim_sample=False,eta=0.,sf=1,sr_mode='blur',guidance_scale=1.,
        sigma=case['sigma_physical'],lambda_=case['lambda_'],zeta=case['zeta'])
