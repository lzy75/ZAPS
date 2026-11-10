"""A102 shared-prefix strict-equivalent-300 protocol."""
from pathlib import Path
from utils.historical_ffhq_gain import config as strong_config

TASKS = ('super_resolution', 'gaussian_deblur', 'inpainting', 'motion_deblur')
ARMS = ('ZAPS_SUFFIX', 'CLOSED_SCALAR_SUFFIX', 'CLOSED_DIAG_SUFFIX')
IMAGES = tuple(f'{index:05d}' for index in range(10))
SIGMA = .05
EXPERIMENT = 'A102_shared_prefix_feedback'
ROOT = '/data0/lzy/ZAPS-results/diag_shared_prefix_feedback/A102_20261110'
BRANCH = 'results/zaps-shared-prefix-feedback-20261110'


def configuration(task, size=256, level=3):
    if task not in TASKS: raise ValueError('Unregistered task')
    row = strong_config()
    row.update(zeta_init=.2 if task in ('gaussian_deblur', 'motion_deblur') else .1,
               img_size=size, level=level)
    return row


def cases(folder):
    root = Path(folder); rows = []
    for image_index, image_id in enumerate(IMAGES):
        for task_index, task in enumerate(TASKS):
            seed = 1000 + image_index
            rows.append(dict(image_id=image_id, image=str(root/(image_id+'.png')),
                task=task, sigma=SIGMA, seed=seed, measurement_seed=seed,
                mask_seed=2001000+image_index, shard=(image_index+task_index)%2,
                key=f'{task}_{image_id}_n050_s{seed}'))
    return rows
