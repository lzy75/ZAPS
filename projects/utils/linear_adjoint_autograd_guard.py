"""Permit the original exact linear-map adjoint, forbid all other grad calls.

SR H^T uses a local linear-resizer probe VJP. This is operator work, not an
input VJP through the UNet. Keep them separate in both gates and accounting.
"""
from contextlib import contextmanager
import inspect


@contextmanager
def operator_only_autograd(counts):
    import torch
    from unittest.mock import patch
    from modules.degradations import _LinearMapAdjoint
    from utils import current_score_geometry as algebra
    original = torch.autograd.grad
    allowed_code = _LinearMapAdjoint.forward.__code__
    for key in ('operator_grad_attempts', 'operator_grad_completed', 'forbidden_grad_attempts'):
        counts.setdefault(key, 0)

    def guarded(*args, **kwargs):
        frame = inspect.currentframe()
        try:
            permitted = frame.f_back.f_code is allowed_code
        finally:
            del frame
        if not permitted:
            counts['forbidden_grad_attempts'] += 1
            raise RuntimeError('A55r2 forbids denoiser/parameter autograd; only original linear adjoint is permitted')
        probe = kwargs.get('inputs')
        if not isinstance(probe, torch.Tensor) or not probe.is_leaf or not probe.requires_grad or kwargs.get('create_graph') is not False:
            raise RuntimeError('Unexpected exact-adjoint probe contract')
        counts['operator_grad_attempts'] += 1
        result = original(*args, **kwargs)
        counts['operator_grad_completed'] += 1
        return result

    def forbidden_vjp(*args, **kwargs):
        counts['forbidden_grad_attempts'] += 1
        raise RuntimeError('A55r2 controls must not call the denoiser input_vjp helper')

    # Install the function directly: a Mock(side_effect=...) would insert mock
    # stack frames, hiding the exact caller whose source identity we check.
    with patch('torch.autograd.grad', new=guarded), patch.object(algebra, 'input_vjp', new=forbidden_vjp):
        yield
    if counts['operator_grad_attempts'] != counts['operator_grad_completed'] or counts['forbidden_grad_attempts']:
        raise RuntimeError('A55r2 autograd accounting did not close')
