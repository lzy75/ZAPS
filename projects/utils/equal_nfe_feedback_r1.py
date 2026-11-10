"""A101R1 protocol changes relative to A101; original bound sources stay untouched."""
from utils import equal_nfe_feedback as base

TASKS = base.TASKS
SIGMAS = base.SIGMAS
ARMS = base.ARMS
IMAGES = base.IMAGES
PANEL_IMAGES = base.PANEL_IMAGES
configuration = base.configuration
cases = base.cases
bootstrap_mean_ci = base.bootstrap_mean_ci

EXPERIMENT = 'A101R1_equal_nfe_feedback'
ROOT = '/data0/lzy/ZAPS-results/diag_equal_nfe_feedback/A101R1_20261110'
BRANCH = 'results/zaps-equal-nfe-feedback-r1-20261110'
