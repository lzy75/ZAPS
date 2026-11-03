"""A81 official pixel-DAPS configuration and a small paired FFHQ transfer panel."""
import hashlib
import json
from pathlib import Path

PROJECTS=Path(__file__).resolve().parents[1]
VENDOR=PROJECTS/'vendor/daps_0b182726'
COMMIT='0b182726f84df1ad26fde4dccebefda4747b82d8'
ARMS=('DAPS','SHRINK')
TASKS=('super_resolution','motion_deblur')
IDS=('00226','00443','00963')
TASK_FILES=dict(super_resolution='down_sampling.yaml',motion_deblur='motion_blur.yaml')
CHECKPOINT=('ffhq_10m.pt','81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa')
MIN_FREE=6*1024**3
NFE=1000
OPERATOR_GRADIENTS=20000


def verify_upstream():
    p=json.loads((VENDOR/'UPSTREAM.json').read_text())
    if p['commit']!=COMMIT:raise ValueError('Wrong DAPS reference revision')
    for row in p['files']:
        b=(VENDOR/row['path']).read_bytes()
        if hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()!=row['git_blob_sha']:
            raise ValueError('Pinned DAPS file changed: '+row['path'])
    return p


def configuration(task):
    import yaml
    sampler=yaml.safe_load((VENDOR/'configs/sampler/edm_daps.yaml').read_text())
    pixel=yaml.safe_load((VENDOR/'configs/task'/TASK_FILES[task]).read_text())['pixel']
    sampler.pop('latent')
    sampler['mcmc_sampler_config']=pixel['mcmc_sampler_config']
    # Hydra/OmegaConf resolves the official 1e-4/5e-5 literals numerically;
    # PyYAML alone treats the exponent without a decimal point as a string.
    for key in ('lr','tau','lr_min_ratio'):
        sampler['mcmc_sampler_config'][key]=float(sampler['mcmc_sampler_config'][key])
    return dict(sampler=sampler,operator=pixel['operator'])


def model_config(path):
    import yaml
    cfg=yaml.safe_load((VENDOR/'configs/model/ffhq256ddpm.yaml').read_text())['model_config']
    cfg['model_path']=str(path)
    return cfg


def cases(parent_cases):
    images={c['image_id']:c['image'] for c in parent_cases if c.get('dataset','ffhq')=='ffhq'}
    if not set(IDS)<=set(images):raise ValueError('Need all three fixed FFHQ image IDs')
    return [dict(key=f'ffhq_{task}_{image}',dataset='ffhq',image_id=image,image=images[image],
                 task=task,seed=1000+int(image),measurement_seed=2000+int(image),kernel_seed=3000+int(image),
                 sigma=.05,shard=TASKS.index(task),arms=list(ARMS))
            for image in IDS for task in TASKS]
