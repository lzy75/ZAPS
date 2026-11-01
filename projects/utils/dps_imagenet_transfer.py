"""A77 freezes the A75 module on a different unconditional diffusion prior."""
from utils.dps_multiplier_transfer import TASKS, STEPS, SCALE, validate
ARMS=('DPS','SHRINK')
RUN_ARMS=ARMS
IDS=('00067','00151','00193')
CHECKPOINT_SHA256='a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1'
MIN_FREE=10*1024**3

def cases(directory):
    from pathlib import Path
    out=[]
    for task in TASKS:
        for image in IDS:
            out.append(dict(key=f'imagenet_{task}_{image}_s{1000+int(image)}',task=task,image_id=image,
                image=str(Path(directory)/f'{image}.png'),dataset='imagenet',sigma=.05,
                seed=1000+int(image),measurement_seed=2000+int(image),mask_seed=3000+int(image),shard=len(out)%2))
    return out

def admission(row,own_reserved=0,measured_peak=0):
    required=max(MIN_FREE,measured_peak+1024**3)
    return dict(**row,own_reserved_bytes=own_reserved,required_bytes=required,
                admitted=row['free_bytes']+own_reserved>=required)
