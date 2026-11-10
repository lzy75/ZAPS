from pathlib import Path
from utils import shared_prefix_feedback as base
TASKS,ARMS,IMAGES=base.TASKS,base.ARMS,base.IMAGES
SIGMAS=(.025,.10);PANEL_IMAGES=('00001','00004','00008')
configuration=base.configuration
EXPERIMENT='A103_shared_prefix_multinoise'
ROOT='/data0/lzy/ZAPS-results/diag_shared_prefix_feedback/A103_20261111'
BRANCH='results/zaps-shared-prefix-multinoise-20261111'
def cases(folder):
 root=Path(folder);rows=[]
 for j,image_id in enumerate(IMAGES):
  for ti,task in enumerate(TASKS):
   for ni,sigma in enumerate(SIGMAS):
    seed=1000+j;rows.append(dict(image_id=image_id,image=str(root/(image_id+'.png')),task=task,sigma=sigma,
     seed=seed,measurement_seed=seed,mask_seed=2001000+j,shard=(j+ti+ni)%2,
     key=f'{task}_{image_id}_n{round(sigma*1000):03d}_s{seed}'))
 return rows
