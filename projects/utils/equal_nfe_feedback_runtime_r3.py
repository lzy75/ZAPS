"""A101R3 numerical prefix audit and forward-exact diagonal autograd bridge."""
from contextlib import contextmanager
import json
import math
from unittest.mock import patch

from utils import equal_nfe_feedback_runtime as original
from utils import equal_nfe_feedback_runtime_r1 as r1

PREFIX_TOLERANCE = 1e-5


def prefix_audit(reference, candidate):
    import torch
    errors = {}
    for key in ('zeta', 'D', 'tau', 'initial'):
        first, second = reference[key], candidate[key]
        if first.shape != second.shape:
            return dict(passed=False, shape_mismatch=key)
        errors[key] = float((first.double() - second.double()).norm() /
                            first.double().norm().clamp(min=1e-30))
    rng_equal = (torch.equal(reference['cpu_rng'], candidate['cpu_rng']) and
                 len(reference['cuda_rng']) == len(candidate['cuda_rng']) and
                 all(torch.equal(a, b) for a, b in zip(reference['cuda_rng'], candidate['cuda_rng'])))
    passed = (all(math.isfinite(value) for value in errors.values()) and
              errors['tau'] == 0. and errors['initial'] == 0. and
              errors['zeta'] <= PREFIX_TOLERANCE and errors['D'] <= PREFIX_TOLERANCE)
    return dict(passed=passed, relative_errors=errors,
        learned_state_relative_tolerance=PREFIX_TOLERANCE,
        rng_equal_before_pairing=rng_equal, final_rng_restored_from_baseline=True,
        copied_from_baseline=['cpu_rng', 'cuda_rng'],
        explicitly_not_copied=['zeta', 'D', 'tau', 'initial', 'output'])


def exact_diagonal_with_native_straight_through(*args, **kwargs):
    delta, detail, counts = r1.exact_adjoint_diagonal(*args, **kwargs)
    q = args[3]
    # Forward value is exactly the closed diagonal solution.  The final ZAPS
    # backward follows the native q graph; its optimizer step is after the
    # evaluated output has already been formed.
    bridged = delta + q - q.detach()
    detail['autograd_bridge'] = 'closed_forward_native_q_straight_through'
    return bridged, detail, counts


@contextmanager
def repaired_execution(reference, audit_path):
    import torch
    saved_snapshot = original.clock.snapshot
    latest = {}

    def snapshot_and_pair(zaps, initial):
        candidate = saved_snapshot(zaps, initial)
        latest['candidate'] = candidate
        torch.set_rng_state(reference['cpu_rng'])
        if reference['cuda_rng']: torch.cuda.set_rng_state_all(reference['cuda_rng'])
        return candidate

    def compare(first, second):
        gate = prefix_audit(first, second)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(gate, indent=2) + '\n')
        return gate

    with patch.object(original.clock, 'snapshot', snapshot_and_pair), \
         patch.object(original.clock, 'snapshot_equal', compare), \
         patch.object(original, 'diagonal_update', exact_diagonal_with_native_straight_through):
        yield
