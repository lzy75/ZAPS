"""A64 immutable public-setting/cost contract; no torch or learned state loading."""
from pathlib import Path
import hashlib
import json
import statistics

EXPERIMENT = 'A64_paper_clock_alignment'
ROOT = '/data0/lzy/ZAPS-results/diag_paper_clock_alignment/A64_20261028'
BRANCH = 'results/paper-clock-alignment-20261028'
DATASETS = ('ffhq', 'imagenet')
TASKS = ('super_resolution', 'gaussian_deblur', 'inpainting', 'motion_deblur')
ARMS = ('PAPER_ZAPS', 'RESPONSE_300', 'RESPONSE_30')
NFE = dict(PAPER_ZAPS=300, RESPONSE_300=300, RESPONSE_30=30)
UPDATES = dict(PAPER_ZAPS=10, RESPONSE_300=9, RESPONSE_30=0)
REPEATS = 3
SEED = 1000
SIGMA = .05
GIB = 1024**3
MIN_FREE = dict(ffhq=6*GIB, imagenet=9*GIB)
PAPER = {
    'ffhq': {
        'super_resolution': dict(psnr=26.63, ssim=.768, lpips=.104),
        'gaussian_deblur': dict(psnr=26.06, ssim=.757, lpips=.121),
        'inpainting': dict(psnr=27.79, ssim=.813, lpips=.078),
        'motion_deblur': dict(psnr=24.16, ssim=.709, lpips=.141)},
    'imagenet': {
        'super_resolution': dict(psnr=23.82, ssim=.718, lpips=.186),
        'gaussian_deblur': dict(psnr=22.45, ssim=.682, lpips=.225)}}
IMAGES = {
    'ffhq': [('00377', '572c2e75bc13b71b4405887435b14fd52d7e5982ab77427bd14acd1c2bbf1f56'),
             ('00389', '1eebf388434c9b811bf5c945749e42277d5716ea616c6c8ed5b7e35ba918103c')],
    'imagenet': [('00067', '586f642c49a639c31bdcd223428d120ded4f36c58ffc38604425ccdac09e4b98'),
                 ('00151', '834bf585bc4134bb379a6e047cc7a37785e353bf000df691aa7ad696b248df5e')]}
# Hashes are validated against the committed A56 full manifest at prepare; see tests.
CHECKPOINTS = {
    'ffhq': dict(path='/home/lzy/ZAPS/projects/modules/models/ffhq_10m.pt',
                 sha256='81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa'),
    'imagenet': dict(path='/home/lzy/ZAPS/projects/modules/models/256x256_diffusion_uncond.pt',
                    sha256='a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1')}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def grid():
    points = []; start = 0
    for size, count in ((334, 15), (333, 10), (333, 5)):
        current = 0.
        for _ in range(count):
            points.append(start+round(current)); current += (size-1)/(count-1)
        start += size
    return points


def config(task):
    require(task in TASKS, 'Unknown task')
    return dict(num_steps=30, schedule=[15, 10, 5], timestep_spacing='linear', schedule_power=2.,
                num_epochs=10, lr=.001, zeta_init=.2 if 'deblur' in task else .1,
                d_init=.2, eta=1., use_learned_var=False, sampler_mode='ddpm',
                wave='db4', level=3, surrogate_score_jacobian=False)


def cases(directories):
    result = []
    for dataset in DATASETS:
        for task in TASKS:
            for image_id, sha in IMAGES[dataset]:
                result.append(dict(dataset=dataset, task=task, image_id=image_id, image_sha256=sha,
                    image=str(Path(directories[dataset])/(image_id+'.png')), sampling_seed=SEED,
                    mask_seed=2001000+int(image_id), sigma=SIGMA,
                    measurement_seed=(1000 if task=='super_resolution' else 201000 if task=='inpainting'
                                      else 401000 if task=='gaussian_deblur' else 601000)+int(image_id),
                    key=f'{dataset}_{task}_{image_id}_s{SEED}'))
    return result


def queue():
    # Mandatory baseline audit first; then interleave full independent runs.
    result = [('PAPER_ZAPS', 0)]
    for repeat in range(REPEATS):
        order = ARMS[repeat:]+ARMS[:repeat]
        result.extend((arm, repeat) for arm in order if (arm, repeat)!=('PAPER_ZAPS', 0))
    return result


def budget(number=16):
    return dict(max_method_forwards=number*REPEATS*sum(NFE.values()),
                max_input_vjp=number*REPEATS*60,
                max_optimizer_backwards=number*REPEATS*sum(UPDATES.values()),
                baseline_audit_forwards=number*300,
                initial_warmup_forwards=2, initial_warmup_input_vjp=2,
                resumed_worker_warmups='Each additional model load records one more forward and input VJP')


def baseline_gate(dataset, rows):
    require(len(rows)==8 and len({(r['case']['task'], r['case']['image_id']) for r in rows})==8,
            'Need all two-image/four-task baseline cases; no task/image dropping')
    require(all(r['case']['dataset']==dataset and r['arm']=='PAPER_ZAPS' and r['repeat']==0 for r in rows),
            'Wrong baseline cohort')
    tests = {}
    for task, published in PAPER[dataset].items():
        group = [r for r in rows if r['case']['task']==task]
        require(len(group)==2, 'Baseline task incomplete')
        means = {k: statistics.mean(r['metrics'][k] for r in group) for k in published}
        alarms = dict(psnr=means['psnr']<published['psnr']-3.,
                      ssim=means['ssim']<published['ssim']-.10,
                      lpips=means['lpips']>published['lpips']+.10)
        tests[task] = dict(mean=means, paper=published, severe_gap=alarms)
    passed = not any(any(t['severe_gap'].values()) for t in tests.values())
    return dict(proceed=passed, classification='PUBLIC_SETTINGS_PILOT_NO_SEVERE_GAP' if passed
                else 'BASELINE_REVIEW_REQUIRED', tasks=tests,
                limits='Engineering severe-gap alarm on TWO images, not same-sample inference, '
                'author-implementation parity or 1000-image benchmark reproduction.')


def available_gpus(rows, requested, visible=None):
    visible = None if visible is None else [v.strip() for v in visible.split(',') if v.strip()]
    wanted = [v.strip() for v in requested.split(',') if v.strip()]
    result = []
    for row in rows:
        aliases = {str(row['index']), row['uuid']}
        if aliases.intersection(wanted) and (visible is None or aliases.intersection(visible)):
            result.append(row)
    require(bool(result), 'No GPU inside both requested and CUDA_VISIBLE_DEVICES allocation')
    return result


def timing_summary(values):
    require(len(values)==REPEATS, 'Three complete independent timings required')
    return dict(median=statistics.median(values), mean=statistics.mean(values),
                stdev=statistics.stdev(values), repeats=values)
