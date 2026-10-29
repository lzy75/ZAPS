"""A72 reuses the exact A71 loop; only its scalar decision is replaced."""
from unittest.mock import patch
from utils import shrink_feedback_ablation as rules
from utils import shrink_multitask_runtime as parent


def replay(model, op, y, case, entry, arm, calibration, size=256, level=3):
    if arm == rules.CHECK:
        return parent.replay(model, op, y, case, entry, 'SHRINK', size=size, level=level)
    choose, calls = rules.selector(arm, calibration)
    # Patch the decision for one serial replay, never any historical source bytes.
    # The original selector is bound in rules.SELECT, preventing patch recursion.
    with patch.object(parent.policy, 'select', side_effect=choose):
        output, trace = parent.replay(model, op, y, case, entry, 'SHRINK', size=size, level=level)
    if calls['calls'] != 30:
        raise ValueError('Expected exactly30 decisions')
    return output, trace
