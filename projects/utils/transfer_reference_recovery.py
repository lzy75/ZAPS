"""Preserve the zero-model reference-import failure before explicit recovery."""
from pathlib import Path
import shutil
from utils import paper_clock_alignment as files


def preserve_failure(root):
    root = Path(root)
    state = files.read(root/'session.json')
    files.require(state.get('status') == 'failed' and
                  'CalledProcessError' in state.get('error','') and
                  'test_transfer_diagonal_runtime' in state.get('error',''),
                  'Recovery only applies to the reported reference-import unit-test failure')
    log = (root/'worker.log').read_text(errors='replace')
    files.require('RuntimeError: Reference package already loaded' in log,
                  'The reported reference-package failure is absent from the original log')
    files.require(not list((root/'cases').glob('*/*/run.json')) and not (root/'load.json').exists(),
                  'Model attempts already exist; automatic recovery is unavailable')
    summary = files.read(root/'summary.json')
    files.require(summary['completed_runs'] == 0 and
                  all(value == 0 for value in summary['actual_new_cost'].values()),
                  'Recovery requires zero recorded model attempts')
    parent = root/'failures'
    parent.mkdir(exist_ok=True)
    backup = parent/f'reference_import_{len(list(parent.glob("reference_import_*")))+1:03d}'
    backup.mkdir()
    names = ('plan.json','session.json','summary.json','launcher.json','worker.log')
    for name in names:
        path = root/name
        if path.exists():
            shutil.copyfile(path,backup/name)
            files.require(files.digest(path)==files.digest(backup/name),'Failure backup differs')
    # The original identity remains available for comparison after code changes.
    plan = files.read(root/'plan.json')
    files.write(root/'reference_recovery.json',dict(status='preserved',backup=str(backup),
        original_protocol={k:v for k,v in plan.items() if k!='files'}))
    (root/'plan.json').rename(backup/'original_plan.json')
    print('REFERENCE FAILURE PRESERVED:',backup,flush=True)
    return backup
