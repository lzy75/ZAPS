"""A101R2 identity; A101/A101R1 records and bound sources remain untouched."""
from utils import equal_nfe_feedback_r1 as base

TASKS = base.TASKS
SIGMAS = base.SIGMAS
ARMS = base.ARMS
IMAGES = base.IMAGES
PANEL_IMAGES = base.PANEL_IMAGES
configuration = base.configuration
cases = base.cases
bootstrap_mean_ci = base.bootstrap_mean_ci

EXPERIMENT = 'A101R2_equal_nfe_feedback'
ROOT = '/data0/lzy/ZAPS-results/diag_equal_nfe_feedback/A101R2_20261110'
BRANCH = 'results/zaps-equal-nfe-feedback-r2-20261110'
