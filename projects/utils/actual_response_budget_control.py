"""A104 full-clock 300/450/445-NFE comparison on the frozen A100 panel."""

TASKS = ('super_resolution', 'gaussian_deblur', 'inpainting', 'motion_deblur')
ARMS = ('ZAPS300', 'ZAPS450', 'REAL_RAY445', 'REAL_DIAG445')
RESPONSE_ARMS = ARMS[2:]
EXPERIMENT = 'A104R1_actual_response_budget_control'
ROOT = '/data0/lzy/ZAPS-results/diag_actual_response_budget/A104R1_20261111'
PARENT = '/data0/lzy/ZAPS-results/diag_equal_budget_ray_diagonal/A100_20261109'
BRANCH = 'results/zaps-actual-response-budget-control-r1-20261111'


def expected_cost(arm):
    if arm == 'ZAPS300': return dict(forwards=300, optimizer_backward=10, input_vjp=0, input_jvp=0)
    if arm == 'ZAPS450': return dict(forwards=450, optimizer_backward=15, input_vjp=0, input_jvp=0)
    if arm in RESPONSE_ARMS: return dict(forwards=445, optimizer_backward=9, input_vjp=29, input_jvp=29)
    raise ValueError('Unknown A104 arm')
