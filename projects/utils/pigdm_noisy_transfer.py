"""A85: paper Algorithm1 noisy PiGDM, unchanged A71 scalar, fixed panel."""
from pathlib import Path
from utils import native_ray_transfer as old

PROJECTS=Path(__file__).resolve().parents[1]
TASKS=('super_resolution','gaussian_deblur','inpainting','motion_deblur')
ARMS=('BASE','SHRINK')
IDS={'ffhq':old.IDS,'imagenet':('00067','00151','00193')}
CHECKPOINTS={'ffhq':old.CHECKPOINT,'imagenet':('256x256_diffusion_uncond.pt',
 'a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1')}
STEPS=100
ETA=1.
MIN_FREE=6*1024**3

def min_free(dataset):return (10 if dataset=='imagenet' else 6)*1024**3

def root_for(dataset):return '/data0/lzy/ZAPS-results/diag_pigdm_noisy_transfer/A85_'+dataset+'_20261104'

def branch(dataset):return 'results/pigdm-noisy-'+dataset+'-transfer-20261104'

def cases(dataset,directory):
 return [dict(key=f'{dataset}_{task}_{image}',dataset=dataset,image_id=image,
  image=str(Path(directory)/(image+'.png')),task=task,sigma=.05,seed=1000+int(image),
  measurement_seed=2000+int(image),kernel_seed=3000+int(image),shard=TASKS.index(task)%2)
  for task in TASKS for image in IDS[dataset]]

def protocol():
 return dict(baseline='Local implementation of PiGDM ICLR2023 Algorithm1 NOISY branch, appendixA.3; not RED-diff default or author benchmark reproduction.',
  grid='100 uniform indices990..0, final clean endpoint -1; original linear1000 beta schedule; eta1; pureGaussian initialization; raw clean prediction; returned final next-state.',
  formula='rt_squared=1-alpha_t; v=Ht(HHt+sigma_y^2/rt_squared I)^-1(y-Hm); g=Jm^T v; q=sqrt(alpha_t)*g. No extra sqrt(alpha_s).',
  operators='Model[-1,1], sigma=.05; averagePool4, normalizedGaussian61/sigma3 periodic, normalizedfrozenDPSMotion61 periodic, random70% missing observed-only RGB coordinates. NEW shared y/operator per pair, not A84 Gaussian.',
  module='Unchanged A71 noise_signal_shrink.select on r=y-Hm and d=Hq/sqrt(alpha_t). Scale q only. Same-time identity clean-response proxy, no next-denoiser response claim.',
  preserved='Original noisyPiGDM direction, DDIM coefficients/eta/noise/grid retained. Positive global correction amplitude may be canceled by gamma.',
  pairing='Same observation/operator/init/finalCPU+CUDA RNG/grid/NN timestamps; independent from-seed runs, first image per task BASE first, remaining arm order alternates; no learned cache.',
  cost='100 forwards+100 inputVJP per run, no optimizer; SHRINK adds one Hq/step, residual reused. Analytic SR/mask and exactFFT blur solves, no CG or iterative tuning.',
  timing='Complete100 includes given-y load/operator creation/init/sampling/CPU sync/bookkeeping. Load/probe/observation creation/metrics/artifact writes separate; same-session loop times, no claimed production speedup.',
  admission='Required CUDA algebra/adjoint/dense-solve/Algorithm1/forced-one/RNG tests and actual checkpoint VJP probe. First image per task BASE first; remaining order alternates without label-based screening. Any failed numerical gate/NaN stops; no quality tuning.',
  decision='Each dataset/task: PSNRmean>=+.1dB, >=2of3 wins, SSIMnondecrease, LPIPSnonincrease, pairingpassed. Baseline PSNR<10 or SSIM<.1 flagged unqualified descriptively; all outputs retained, no policy fallback.',
  limits='Three reused images/dataset, one seed, sigma.05 only. Different architecture+data together. Small local paired transfer; no state-necessity/universal/posterior guarantee. No fixed/mismatch arms.')
