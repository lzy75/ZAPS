"""A41 serialization-only compatibility entrypoint; original SHA-bound sources stay intact."""
import argparse
from pathlib import Path
import sys

import numpy as np

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS))
from utils import diag_ffhq_action_geometry as original

COMPAT='serialization_compat_v1.json'


def native_scalars(value):
    """Preserve values/types JSON supports, explicitly convert NumPy scalar leaves."""
    if isinstance(value,np.generic):return native_scalars(value.item())
    if isinstance(value,dict):return {native_scalars(k):native_scalars(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [native_scalars(v) for v in value]
    return value


def ensure_compat(root):
    source=original.source;source.check_plan(root)
    paths=[Path(__file__).resolve(),PROJECTS/'utils/test_ffhq_action_geometry_json_fix.py',
           PROJECTS/'run_a41_json_fix_20261025.sh',PROJECTS/'A41_JSON_SERIALIZATION_FIX_20261025.md']
    expected=dict(kind='A41_serialization_only_compatibility_v1',plan_sha256=source.digest(root/'plan.json'),
        files={str(p):source.digest(p) for p in paths},
        change='Convert NumPy scalar leaves at A41 JSON writes only. No tensor/feature/decision/cost change. '
               'Original workers/plan and all existing attempts unchanged; incomplete attempts retained.')
    path=root/COMPAT
    if path.exists():
        if source.load_signed(path)!=expected:raise ValueError('Serialization compatibility identity changed')
    else:source.signed(path,expected)
    return source.digest(path)


def install(root,compat_sha):
    """Patch only the two write aliases used by A41 counting and signed records."""
    root=Path(root).resolve();plan_sha=original.source.digest(root/'plan.json')
    for module in (original.source,original.scan):
        writer=module.WRITE
        def write(path,value,_writer=writer):
            plain=native_scalars(value)
            if (Path(path).name=='run.json' and isinstance(plain,dict) and plain.get('plan_sha256')==plan_sha
                    and Path(path).resolve().is_relative_to(root)):
                plain=dict(plain,serialization_compat_sha256=compat_sha)
            return _writer(path,plain)
        module.WRITE=write


def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--root',default=original.ROOT)
    parser.add_argument('--prepare',action='store_true');known,_=parser.parse_known_args()
    root=Path(known.root).resolve()
    if not root.exists():
        if not known.prepare:raise ValueError('A41 needs --prepare before worker/archive')
        # Preparation contains no response readouts; keep the original frozen plan byte protocol.
        original.main();compat=ensure_compat(root)
    else:
        compat=ensure_compat(root);install(root,compat);original.main()
    print('A41_JSON_COMPAT_PASS:',compat,'; original plan unchanged.',flush=True)


if __name__=='__main__':main()
