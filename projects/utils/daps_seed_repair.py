"""A81R1: seed NumPy before the unchanged official motion-kernel constructor."""
from pathlib import Path
import json
from utils import daps_multiplier_runtime as original
from utils.a54_session import busy

reference=original.reference
load_image=original.load_image
load_model=original.load_model
operator_state=original.operator_state
counters=original.counters
run=original.run
FAILED_ROOT='/data0/lzy/ZAPS-results/diag_daps_multiplier_transfer/A81_20261104'


def make_operator(case,device):
    import numpy as np
    # The historical helper used by original.make_operator seeds Torch only.
    # Kernel's imported numpy.random uniform/triangular/beta use this same NumPy RNG.
    np.random.seed(case['kernel_seed'])
    return original.make_operator(case,device)


def zero_run_failure(root):
    root=Path(root);state=root.parent/(root.name+'_session')
    if busy(state/'coordinator.lock'):raise ValueError('Original A81 coordinator still active')
    if (root/'plan.json').exists() or any((root/'cases').glob('**/run.json')) or any((root/'workers').glob('**/load_*.json')):
        raise ValueError('Original A81 has formal experiment state; no automatic zero-run repair')
    status=json.loads((state/'session.json').read_text())
    log=state/'coordinator.log';text=log.read_text(errors='replace')
    if status.get('status')!='failed' or 'test_original_motion_state_reproducible_and_input_gradient_finite' not in text or 'FAILED (failures=1)' not in text:
        raise ValueError('Not the reported zero-run Motion seed gate failure')
    return dict(status_path=str(state/'session.json'),log_path=str(log),formal_model_runs=0,
                reason='Original seed helper covered Torch only; official Motion uses NumPy RNG too')
