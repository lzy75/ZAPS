"""A80 fixed paired extension, reusing completed A79R1 observations and old-policy quality."""
from utils.diffpir_numeric_transfer import (PROJECTS,VENDOR,TASKS,DATASETS,IDS,SIGMA_PHYSICAL,STEPS,MIN_FREE,CHECKPOINTS,PAPER_PARAMETERS,verify_upstream,sequence,configuration)
ARMS=('DIFFPIR_REFERENCE','HOST_AWARE')
PAIRS=(('DIFFPIR_REFERENCE','HOST_AWARE'),)
REFERENCE_SOURCE_ARM='STABLE_DIFFPIR'
CACHED_SOURCE_ARM='STABLE_SHRINK'


def cases(parent_cases):
    return [dict(c,arms=list(ARMS)) for c in parent_cases]
