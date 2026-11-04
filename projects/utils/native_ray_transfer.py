"""A82-A84 frozen host definitions; A71 scalar policy remains unchanged."""
import hashlib
import json
from pathlib import Path

PROJECTS = Path(__file__).resolve().parents[1]
IDS = ('00226', '00443', '00963')
TASKS = ('super_resolution', 'gaussian_deblur')
ARMS = ('BASE', 'SHRINK')
CHECKPOINT = ('ffhq_10m.pt', '81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa')
HOSTS = {
    'A82': dict(host='MPGD_WO_PROJ', vendor='mpgd_9f94b386',
                commit='9f94b386ea4184960278d28a8bc2756898db70e2', nfe=100,
                vjp=0, operator_gradients=100),
    'A83': dict(host='DMPS', vendor='dmps_91afe254',
                commit='91afe2540ac8dd8a7fa133f654560f736c80fc53', nfe=1000,
                vjp=0, operator_gradients=0),
    'A84': dict(host='PIGDM', vendor='pgdm_66482f23',
                commit='66482f23e242bb31166c3662002d0a6f9f065030', nfe=50,
                vjp=50, operator_gradients=0),
}
MIN_FREE = 6 * 1024**3


def vendor(experiment):
    return PROJECTS/'vendor'/HOSTS[experiment]['vendor']


def verify(experiment):
    root = vendor(experiment)
    meta = json.loads((root/'UPSTREAM.json').read_text())
    if meta['commit'] != HOSTS[experiment]['commit']:
        raise ValueError('Unexpected upstream commit')
    for row in meta['files']:
        data = (root/row['path']).read_bytes()
        blob = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if blob != row['git_blob_sha'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Upstream source changed: '+row['path'])
    return meta


def configuration(experiment, task):
    import yaml
    root = vendor(experiment)
    if experiment == 'A82':
        cfg = yaml.safe_load((root/'linear_inv/configs/mgpd_diffusion_config.yaml').read_text())
        name = 'super_resolution_config.yaml' if task == TASKS[0] else 'gaussian_deblur_config.yaml'
        task_cfg = yaml.load((root/'linear_inv/configs'/name).read_text(), Loader=yaml.FullLoader)
        # Paper appendixD.1 explicitly fixes scale5 at100 steps for both tasks.
        # Released SR script says4 and CLI default says10; disclose rather than
        # choose those weaker/ambiguous defaults or select a value by quality.
        return dict(diffusion=cfg, task=task_cfg, scale=5.)
    if experiment == 'A83':
        name = 'sr4_config.yaml' if task == TASKS[0] else 'deblur_gauss_config.yaml'
        return dict(diffusion=yaml.safe_load((root/'configs/diffusion_config.yaml').read_text()),
                    task=yaml.load((root/'configs'/name).read_text(), Loader=yaml.FullLoader))
    if experiment != 'A84':
        raise ValueError('Unknown experiment')
    algo = yaml.safe_load((root/'_configs/algo/pgdm.yaml').read_text())
    exp = yaml.safe_load((root/'_configs/exp/default.yaml').read_text())
    algo['deg'] = 'sr4' if task == TASKS[0] else 'deblur_gauss'
    # sigma_y controls NEW observation generation in the original build helper;
    # pgdm.py itself does not use a noise covariance likelihood.
    algo['sigma_y'] = .05
    return dict(algo=algo, exp=exp, steps=exp['num_steps'])


def cases(experiment, parent_cases):
    images = {c['image_id']: c['image'] for c in parent_cases if c.get('dataset', 'ffhq') == 'ffhq'}
    if not set(IDS) <= set(images):
        raise ValueError('Missing fixed FFHQ panel')
    return [dict(key=f'ffhq_{task}_{image}', dataset='ffhq', image_id=image, image=images[image],
                 task=task, seed=1000+int(image), measurement_seed=2000+int(image),
                 kernel_seed=3000+int(image), sigma=.05, shard=TASKS.index(task))
            for image in IDS for task in TASKS]


def root_for(experiment):
    return '/data0/lzy/ZAPS-results/diag_native_ray_transfer/'+experiment+'_20261104'


def branch(experiment):
    return 'results/'+HOSTS[experiment]['host'].lower().replace('_','-')+'-shrink-transfer-20261104'
