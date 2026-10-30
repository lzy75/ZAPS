"""A73 fixed scope: same three A71 images, four tasks, default observation noise."""
from utils.shrink_multitask import TASKS, IMAGES
ARMS = ('BASE', 'ALL_UNROLLS')
SIGMA = .05


def cases(parent_cases):
    selected = [dict(c) for c in parent_cases if c['sigma'] == SIGMA]
    if len(selected) != 12 or {(c['task'], c['image_id']) for c in selected} != {(t, i) for t in TASKS for i in IMAGES}:
        raise ValueError('Need exactly original3 images x4 tasks at sigma=.05')
    for i, c in enumerate(selected):
        c['shard'] = i % 2
    return selected
