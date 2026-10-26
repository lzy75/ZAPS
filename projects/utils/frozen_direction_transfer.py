"""A56 immutable task/cohort/budget rules; no fitting on confirmation labels."""
import math
from pathlib import Path
from statistics import median
import numpy as np

DATASETS = ('ffhq', 'imagenet')
TASKS = ('super_resolution', 'inpainting')
COUNTS = dict(ffhq=6, imagenet=4)
ARMS = ('C_REPLAY', 'C_IEEE', 'U_CURRENT', 'V_PHYSICAL', 'Q_ORIGINAL', 'TIME_GAIN')
METROLOGY_STEPS = (24, 28)


def frozen_profiles(cases, traces):
    result = {}
    for task in TASKS:
        group = [c for c in cases if c['task'] == task]
        if len(group) != 5 or len({c['image_id'] for c in group}) != 5:
            raise ValueError('Need five OLD development images per task')
        values = [traces[c['key']] for c in group]
        if any(len(v) != 30 or any(not math.isfinite(g) or g <= 0 for g in v) for v in values):
            raise ValueError('Need30 finite positive OLD gains')
        result[task] = [float(median(v[k] for v in values)) for k in range(30)]
    return result


def expected_calls(grid, metrology):
    times = list(reversed(grid))
    if len(times) != 30 or times[0] != 999 or times[-1] != 0:
        raise ValueError('Need original30-call grid')
    out = []
    for arm in ARMS:
        for k, t in enumerate(times):
            out.append(t)
            if arm == 'C_IEEE' and metrology and k in METROLOGY_STEPS:
                out.extend([t]*6)
    return out


def select(pool, used_names, used_hashes, n, seed):
    unique = {}
    for c in sorted(pool, key=lambda c: c['image_id']):
        if c['image_id'] not in used_names and c['image_sha256'] not in used_hashes:
            unique.setdefault(c['image_sha256'], c)
    eligible = sorted(unique.values(), key=lambda c: c['image_id'])
    if len(eligible) < n:
        raise ValueError(f'Need{n} unused unique images; available={len(eligible)}. No substitution after outcomes.')
    chosen = np.random.default_rng(seed).choice(len(eligible), n, replace=False)
    return sorted((eligible[int(i)] for i in chosen), key=lambda c: c['image_id'])


def eligible_paths(root):
    return sorted(p for p in Path(root).glob('*.png') if p.stem.isdigit() and 32 <= int(p.stem) <= 999)


def devices(rows, visible=None, first=None, second=None):
    from utils.experiment_gpu_selection import choose
    aliases = {token: r for r in rows for token in (r['index'], r['uuid'])}
    selected = choose(rows, visible, first, second)
    for token in (first, second):
        if token is not None and aliases[token]['free'] < 12288:
            raise ValueError('A56 explicit GPU requires at least12 GiB free: '+token)
    good = [token for token in selected if aliases[token]['free'] >= 12288]
    if not good:
        raise ValueError('A56 ImageNet graph requires an allocated GPU with at least12 GiB free; no run started')
    return [token if aliases[token]['free'] >= 12288 else good[0] for token in selected]
