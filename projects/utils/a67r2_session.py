"""A67R2: admit the queue's device UUID; preserve the zero-call R1 CLI failure."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import patch

PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS))
from utils import a67r1_session as previous
from utils.a54_session import exclusive,busy

ENTRY=Path(__file__).resolve()
ORIGINAL_PARSER=argparse.ArgumentParser


class WorkerParser(ORIGINAL_PARSER):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.add_argument('--device-uuid')
        self.add_argument('--check-worker-cli',action='store_true',help=argparse.SUPPRESS)

    def parse_args(self,*args,**kwargs):
        a=super().parse_args(*args,**kwargs)
        if a.check_worker_cli:
            if not (a.worker and a.device_uuid and a.shard in (0,1) and a.phase in ('feedback','time')):
                self.error('Complete worker arguments required for CLI check')
            print(json.dumps(dict(worker=a.worker,device_uuid=a.device_uuid,shard=a.shard,phase=a.phase,
                                  model_calls=0)),flush=True)
            raise SystemExit(0)
        return a


def check(root):
    root=Path(root);previous.repair_check(root);m=previous.files.read(root/'repair_r2/manifest.json')
    for name,sha in m['sources'].items():
        previous.files.require(previous.files.digest(PROJECTS/name)==sha,'R2 source changed: '+name)
    return m


def recover_cli_failure(root,state):
    root,state=Path(root),Path(state);folder=root/'repair_r2';file=folder/'manifest.json'
    if file.exists():return check(root)
    previous.files.require(not busy(state/'coordinator.lock'),'Active coordinator; do not mutate')
    for f in (root/'workers').glob('*/worker.lock'):
        previous.files.require(not busy(f),'Active worker; do not mutate')
    previous.repair_check(root)
    old=previous.files.read(state/'session.json')
    previous.files.require(old['status']=='failed' and old['error']=="RuntimeError('A67 worker failed; original records retained')",
                           'Different session failure')
    previous.files.require(not any((root/'cases').glob('*/*/run.json')) and
                           not any((root/'workers').glob('*/load_*.json')) and
                           not any((root/'workers').glob('*/feedback.json')),
                           'New model/worker attempt exists; no automatic recovery')
    logs=list(state.glob('shard*_feedback.log'))
    previous.files.require(len(logs)==2 and all('unrecognized arguments: --device-uuid' in f.read_text() for f in logs),
                           'Not the reported two CLI failures')
    folder.mkdir()
    names=['utils/a67r2_session.py','utils/test_a67r2_cli.py','run_a67r2_20261028.sh',
           'experiment_cards/A67R2_worker_cli_repair_20261028.md']
    paths=[state/'session.json',*state.glob('*.log')]
    # Snapshot first; then remove only these closed session-state files so R1 can resume.
    for source in paths:
        if source.exists():
            destination=folder/'prior_session'/source.name;destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(source,destination)
    if (root/'logs').exists():shutil.copytree(root/'logs',folder/'prior_logs')
    previous.files.write(file,dict(status='active',new_failed_model_calls=0,
        original_plan_sha256=previous.files.digest(root/'plan.json'),
        prior_session_sha256=previous.files.digest(state/'session.json'),
        sources={name:previous.files.digest(PROJECTS/name) for name in names}))
    for source in paths:
        if source.exists():
            previous.files.require(previous.files.digest(source)==previous.files.digest(folder/'prior_session'/source.name),
                                   'Session changed while preserving it')
            source.unlink()
    print('A67R2 READY: preserved zero-call CLI failure; original600-call ledger unchanged.',flush=True)
    return check(root)


OLD_INVOKE=previous.OLD_INVOKE


def invoke_with_cli_test(*arguments):
    arguments=list(arguments)
    if arguments[:2]==['-m','unittest']:arguments.append('utils.test_a67r2_cli')
    return OLD_INVOKE(*arguments)


def main():
    pre=ORIGINAL_PARSER(add_help=False)
    pre.add_argument('--root',default=previous.driver.ROOT)
    args,remaining=pre.parse_known_args()
    child_flags={'--worker','--coordinator','--prepare','--summarize','--archive','--archive-only','--status',
                 '--help','-h','--check-worker-cli'}
    if not child_flags.intersection(remaining):
        root=Path(args.root).resolve();state=previous.session.state_dir(root);state.mkdir(parents=True,exist_ok=True)
        with exclusive(state/'r2_recovery.lock'):
            if not busy(state/'coordinator.lock'):recover_cli_failure(root,state)
    elif '--help' not in remaining and '-h' not in remaining and '--check-worker-cli' not in remaining:
        if (Path(args.root)/'repair_r2/manifest.json').exists():check(args.root)
    with ExitStack() as stack:
        stack.enter_context(patch.object(previous,'argparse',SimpleNamespace(ArgumentParser=WorkerParser)))
        stack.enter_context(patch.object(previous,'ENTRY',ENTRY))
        stack.enter_context(patch.object(previous,'OLD_INVOKE',invoke_with_cli_test))
        previous.main()


if __name__=='__main__':main()
