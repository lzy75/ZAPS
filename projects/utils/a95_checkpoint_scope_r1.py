"""Recognize BOTH local guided-diffusion layouts, without editing either copy."""
from contextlib import contextmanager,ExitStack
from pathlib import Path
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[2]
APPROVED={REPO/'guided_diffusion/nn.py',REPO/'DPS/guided_diffusion/nn.py'}


def namespaces(model):
    found={}
    for layer in model.modules():
        namespace=getattr(getattr(type(layer),'forward',None),'__globals__',{})
        function=namespace.get('checkpoint');code=getattr(function,'__code__',None)
        if code and Path(code.co_filename).resolve() in APPROVED:found[id(namespace)]=namespace
    return list(found.values())


@contextmanager
def forward_ad_checkpoint_scope(model):
    found=namespaces(model)
    def direct(function,inputs,parameters,flag):return function(*inputs)
    with ExitStack() as stack:
        for namespace in found:stack.enter_context(patch.dict(namespace,checkpoint=direct))
        yield len(found)


def loaded_sources(model):
    from utils.historical_ffhq_gain import digest
    paths=set()
    for namespace in namespaces(model):
        paths.add(Path(namespace['checkpoint'].__code__.co_filename).resolve())
    for layer in model.modules():
        code=getattr(getattr(type(layer),'forward',None),'__code__',None)
        if code:
            path=Path(code.co_filename).resolve()
            if path.parent in {p.parent for p in APPROVED}:paths.add(path)
    if not paths:raise RuntimeError('No approved actual guided-diffusion checkpoint source found')
    return {str(path):digest(path) for path in sorted(paths)}
