"""A79R1: preserve official float32 baseline; independently audit a stable equivalent sf1 solve."""
from utils.diffpir_multiplier_transfer import (PROJECTS,VENDOR,TASKS,DATASETS,IDS,SIGMA_PHYSICAL,STEPS,MIN_FREE,CHECKPOINTS,PAPER_PARAMETERS,verify_upstream,sequence,configuration)
from utils import diffpir_multiplier_transfer as original
ARMS=('DIFFPIR','SHRINK','STABLE_DIFFPIR','STABLE_SHRINK')
PAIRS=(('DIFFPIR','SHRINK'),('STABLE_DIFFPIR','STABLE_SHRINK'))
FAILED_ROOT='/data0/lzy/ZAPS-results/diag_diffpir_multiplier_transfer/A79_20261103'


def cases(folders):
    values=original.cases(folders)
    for c in values:c['arms']=list(ARMS)
    return values
