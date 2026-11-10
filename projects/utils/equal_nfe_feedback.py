"""A101 fixed 300-NFE ZAPS, closed scalar, and closed diagonal panel."""
from pathlib import Path

from utils.historical_ffhq_gain import config as strong_config

TASKS = ('super_resolution', 'gaussian_deblur', 'inpainting', 'motion_deblur')
SIGMAS = (.025, .05, .10)
ARMS = ('ZAPS_LOCAL_300', 'CLOSED_SCALAR_300', 'CLOSED_DIAG_300')
IMAGES = tuple(f'{i:05d}' for i in range(10))
PANEL_IMAGES = ('00001', '00004', '00008')


def configuration(task, size=256, level=3):
    if task not in TASKS:
        raise ValueError('Unregistered task')
    out = strong_config()
    out.update(
        zeta_init=.2 if task in ('gaussian_deblur', 'motion_deblur') else .1,
        img_size=size,
        level=level,
    )
    return out


def cases(folder):
    root = Path(folder)
    rows = []
    for image_index, image_id in enumerate(IMAGES):
        seed = 1000 + image_index
        for task_index, task in enumerate(TASKS):
            for noise_index, sigma in enumerate(SIGMAS):
                rows.append(dict(
                    image_id=image_id,
                    image=str(root / (image_id + '.png')),
                    task=task,
                    sigma=sigma,
                    seed=seed,
                    # Preserve the September-17/A71 convention: one frozen
                    # observation-noise draw and one frozen mask per image.
                    # Scaling the same standard-normal draw gives paired
                    # noise boundaries without changing the historical
                    # sigma=.05 SR observation.
                    measurement_seed=seed,
                    mask_seed=2001000 + image_index,
                    shard=(image_index + task_index + noise_index) % 2,
                    key=f'{task}_{image_id}_n{round(sigma * 1000):03d}_s{seed}',
                ))
    return rows


def bootstrap_mean_ci(values, seed, draws=10000, alpha=.05):
    """Deterministic paired percentile interval; no SciPy dependency."""
    import random
    if not values:
        raise ValueError('Empty bootstrap sample')
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    lo = means[int((alpha / 2) * draws)]
    hi = means[min(draws - 1, int((1 - alpha / 2) * draws))]
    return [lo, hi]
