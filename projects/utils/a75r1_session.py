"""A75 environment-only repair; preserve and reopen ONLY zero-model MKL preflight failure."""
import os

# Must run before importing NumPy/Torch or any original session dependencies.
os.environ['MKL_THREADING_LAYER']='GNU'
os.environ.pop('MKL_SERVICE_FORCE_INTEL',None)

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS))


def preserve_mkl_failure(root,state):
    root,state=Path(root),Path(state)
    f=state/'session.json'
    if not f.exists():return None
    old=json.loads(f.read_text())
    if old.get('status')!='failed':return None
    # An absent frozen plan and absent result files rule out real model attempts.
    if root.exists() and any(root.rglob('*')):
        raise ValueError('A75R1 refuses recovery: experiment files already exist. Original records retained.')
    log=state/'coordinator.log'
    text=log.read_text(errors='replace') if log.exists() else ''
    if 'MKL_THREADING_LAYER=INTEL is incompatible' not in text or 'utils.test_dps_multiplier_transfer' not in old.get('error',''):
        raise ValueError('A75R1 only reopens the reported MKL unittest preflight failure; original failure retained.')
    parent=state/'preflight_repairs';parent.mkdir(exist_ok=True)
    backup=parent/f'mkl_{len(list(parent.glob("mkl_*"))):03d}'
    backup.mkdir(exist_ok=False)
    shutil.copyfile(f,backup/'session.json')
    if log.exists():log.replace(backup/'coordinator.log')
    evidence=dict(repair='A75R1_MKL_GNU_ZERO_MODEL',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        root=str(root),plan_absent=not (root/'plan.json').exists(),experiment_files_absent=True,
        original_session_sha256=hashlib.sha256((backup/'session.json').read_bytes()).hexdigest(),
        environment=dict(MKL_THREADING_LAYER='GNU',MKL_SERVICE_FORCE_INTEL=None),method_changed=False)
    (backup/'repair.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    from utils.historical_ffhq_gain import write
    write(f,dict(status='resource_limited',previous_zero_model_failure=str(backup),
                 recovery='Explicit A75R1 invocation; original records preserved; no model rerun'))
    return backup


def main():
    from utils import a75_session as original
    from utils.a54_session import exclusive,busy
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--root',default=original.driver.ROOT)
    parser.add_argument('--status',action='store_true');parser.add_argument('--archive-only',action='store_true')
    args,_=parser.parse_known_args()
    root=Path(args.root).resolve();state=original.state_dir(root)
    state.mkdir(parents=True,exist_ok=True)
    if not args.status and not args.archive_only and not busy(state/'coordinator.lock'):
        with exclusive(state/'client.lock'):
            if not busy(state/'coordinator.lock'):
                backup=preserve_mkl_failure(root,state)
                if backup:print('A75R1: prior zero-model MKL failure preserved:',backup,flush=True)
    print('A75R1 environment: MKL_THREADING_LAYER=GNU; original A75 protocol unchanged.',flush=True)
    original.main()


if __name__=='__main__':main()
