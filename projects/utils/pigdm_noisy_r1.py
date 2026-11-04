"""A85R1: retain published RED-diff coefficient; qualify real baselines first."""
from utils.pigdm_noisy_transfer import PROJECTS,TASKS,ARMS,IDS,CHECKPOINTS,STEPS,ETA,MIN_FREE,min_free,cases
from utils.pigdm_noisy_transfer import protocol as previous_protocol

def root_for(dataset):return '/data0/lzy/ZAPS-results/diag_pigdm_noisy_transfer/A85R1_'+dataset+'_20261104'
def branch(dataset):return 'results/pigdm-noisy-native-coeff-'+dataset+'-20261104'
def protocol():
 p=previous_protocol()
 p.update(baseline='Local NOISY extension using published RED-diff awd=True,mcg=False coefficient sqrt(alpha_s*alpha_t). Not literal paperAlgorithm1 reproduction. A85 literal coefficient failed on real FFHQ before SHRINK.',
  formula='rt_squared=1-alpha_t; v=Ht(HHt+sigma_y^2/rt_squared I)^-1(y-Hm); g=Jm^T v; q=sqrt(alpha_s*alpha_t)*g.',
  admission='Required CUDA operator/gradient/forced-one/RNG and native-source update-equivalence gates. First one BASE per task must complete100 on actual weights before any SHRINK. Offline anomaly floor PSNR>=10,SSIM>=.1; no label-based tuning, fallback or sample substitution.',
  lineage='New root and new protocol. A85 failed22 forwards+22 VJPs and2 real probes retained and reported separately; no failed trajectory reuse. Existing A85 observations reused with binary hashes when present.',
  probes='Two baseline workers then two comparison workers: nominal4 real-model forward+VJP probes/dataset, every load/resume counted. Baseline qualification runs are reused, not rerun.',
  scope='Keep A85 operators,100 grid,eta1,pureGaussian init,shared observations and unchanged A71 gamma. Only restore published next-time coefficient and add staged qualification/diagnostics. Still a candidate local baseline until real run returns.')
 return p
