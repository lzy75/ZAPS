"""A50r2 input-only VJP with original custom-checkpoint compatibility.

AttentionBlock always checkpoints even when the model's use_checkpoint=False.
Its backward requests input AND parameter gradients internally. Parameters must
therefore require gradients; autograd.grad(mean, input) still does not populate
their .grad or update any weight. Historical A50 files remain byte-identical.
"""
from contextlib import contextmanager
from utils import current_score_runtime as previous
from utils.current_score_runtime import *

LEGACY_RUNTIME_FILE = previous.__file__


@contextmanager
def input_vjp_parameters(model):
    parameters=list(model.parameters())
    flags=[p.requires_grad for p in parameters]
    if any(p.grad is not None for p in parameters):
        raise ValueError('Preexisting model gradients are not overwritten')
    try:
        for p in parameters:p.requires_grad_(True)
        yield
        if any(p.grad is not None for p in parameters):
            raise RuntimeError('Input-only VJP populated parameter .grad')
    finally:
        for p,flag in zip(parameters,flags):p.requires_grad_(flag)
