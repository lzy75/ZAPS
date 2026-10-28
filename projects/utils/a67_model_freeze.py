"""Freeze only the NN during A67; ζ and D remain learnable. Restore flags."""
from contextlib import contextmanager


@contextmanager
def frozen_network(model):
    parameters=list(model.parameters())
    flags=[p.requires_grad for p in parameters]
    versions=[p._version for p in parameters]
    if any(p.grad is not None for p in parameters):
        raise ValueError('Preexisting NN gradients; do not overwrite')
    try:
        for p in parameters:p.requires_grad_(False)
        yield
        if any(p.grad is not None or p._version!=v for p,v in zip(parameters,versions)):
            raise RuntimeError('A67 NN weight or gradient changed')
    finally:
        for p,flag in zip(parameters,flags):p.requires_grad_(flag)
