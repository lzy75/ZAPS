"""A101R1 paired-final-RNG and true-adjoint repair helpers."""
from contextlib import contextmanager
from unittest.mock import patch

from utils import equal_nfe_feedback_runtime as original


def independent_prefix_equal(reference, candidate):
    """Compare learned prefix state before injecting the shared final RNG.

    The final RNG is deliberately paired after this snapshot.  No learned
    parameter, latent, direction, or output is copied from the baseline.
    """
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
                 all(torch.equal(a, b) for a, b in
                     zip(reference['cuda_rng'], candidate['cuda_rng'])))
    passed = (errors['zeta'] <= 1e-6 and errors['D'] <= 1e-6 and
              errors['initial'] <= 1e-6 and errors['tau'] == 0.)
    return dict(passed=passed, relative_errors=errors,
                rng_equal_before_pairing=rng_equal,
                final_rng_restored_from_baseline=True,
                copied_from_baseline=['cpu_rng', 'cuda_rng'],
                explicitly_not_copied=['zeta', 'D', 'tau', 'initial', 'output'])


@contextmanager
def paired_final_rng(reference):
    """Capture the candidate prefix, then restore only baseline RNG state."""
    import torch
    saved_snapshot = original.clock.snapshot

    def snapshot_and_pair(zaps, initial):
        candidate = saved_snapshot(zaps, initial)
        torch.set_rng_state(reference['cpu_rng'])
        if reference['cuda_rng']:
            torch.cuda.set_rng_state_all(reference['cuda_rng'])
        return candidate

    with patch.object(original.clock, 'snapshot', snapshot_and_pair), \
         patch.object(original.clock, 'snapshot_equal', independent_prefix_equal):
        yield


def exact_adjoint_diagonal(*args, **kwargs):
    """Use the real H-adjoint for the quadratic solve, never legacy bicubic T."""
    operator = args[1]
    exact = operator.base.transpose if hasattr(operator, 'base') else operator.transpose
    with patch.object(operator, 'transpose', exact):
        return _ORIGINAL_DIAGONAL(*args, **kwargs)


_ORIGINAL_DIAGONAL = original.diagonal_update


@contextmanager
def repaired_diagonal():
    with patch.object(original, 'diagonal_update', exact_adjoint_diagonal):
        yield
