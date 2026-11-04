"""A88 fills only DPS Gaussian/inpainting cells; no policy recalibration."""
from pathlib import Path
TASKS=('gaussian_deblur','inpainting')
ARMS=RUN_ARMS=('DPS','SHRINK')
STEPS=1000
SCALES={'gaussian_deblur':.3,'inpainting':.5}
IDS={'ffhq':('00226','00443','00963'),'imagenet':('00067','00151','00193')}
CHECKPOINTS={
 'ffhq':('ffhq_10m.pt','81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa'),
 'imagenet':('256x256_diffusion_uncond.pt','a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1')}

def cases(dataset,directory):
 if dataset not in IDS:raise ValueError('Unsupported dataset; no automatic substitution')
 out=[]
 for task in TASKS:
  for image in IDS[dataset]:
   out.append(dict(key=f'{dataset}_{task}_{image}_s{1000+int(image)}',dataset=dataset,
    task=task,image_id=image,image=str(Path(directory)/(image+'.png')),sigma=.05,
    seed=1000+int(image),measurement_seed=(1000 if dataset=='ffhq' else 2000)+int(image),
    mask_seed=(2001000 if dataset=='ffhq' else 3000)+int(image),
    scale=SCALES[task],shard=len(out)%2))
 return out

def validate(config,tasks):
 from utils import dps_multiplier_transfer as old
 old.validate(config,{t:{'conditioning':{'method':'ps','params':{'scale':.3}}} for t in old.TASKS})
 for task in TASKS:
  c=tasks[task]['conditioning']
  if c['method']!='ps' or c['params']['scale']!=SCALES[task]:raise ValueError('Official task PS scale changed')

def admission(dataset,row,own_reserved=0,measured_peak=0):
 required=max((10 if dataset=='imagenet' else 6)*1024**3,measured_peak+1024**3)
 return dict(**row,own_reserved_bytes=own_reserved,required_bytes=required,
             admitted=row['free_bytes']+own_reserved>=required)
