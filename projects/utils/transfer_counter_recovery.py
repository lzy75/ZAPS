"""Preserve the first-step PiGDM counter failure and its real model costs."""
from pathlib import Path
import shutil
from utils import paper_clock_alignment as files


def preserve_failure(root):
    root = Path(root)
    state = files.read(root/'session.json')
    plan = files.read(root/'plan.json')
    files.require(plan['host']=='pigdm' and state.get('status')=='failed' and
                  state.get('error')=="KeyError('corrections')", 'Not the reported PiGDM counter failure')
    files.require(not (root/'counter_recovery.json').exists(), 'Counter recovery already recorded')
    rows = [(path,files.read(path)) for path in (root/'cases').glob('*/*/run.json')]
    failed = [(path,row) for path,row in rows if row['status']!='complete']
    files.require(len(failed)==1, 'Expected one failed first-step PiGDM attempt')
    path,row = failed[0]
    counts = row.get('counts') or {}
    files.require(row['status']=='failed' and row.get('error')=="KeyError('corrections')" and
                  'pigdm_noisy_r1_runtime.py' in row.get('traceback',''), 'Failure signature differs')
    files.require(all(counts.get(k)==1 for k in ('forward_attempts','forwards','input_vjp_attempts')) and
                  counts.get('optimizer_backward_attempts',0)==0 and not row.get('artifacts'),
                  'Only the reported incomplete one-forward attempt can be recovered')
    files.require(all(r.get('reused') for _,r in rows if r['status']=='complete'),
                  'Unexpected completed fresh model runs before the counter failure')
    backup = root/'failures'/'corrections_001'
    backup.mkdir(parents=True,exist_ok=False)
    for name in ('plan.json','session.json','summary.json','load.json','launcher.json','worker.log'):
        source = root/name
        if source.exists():
            shutil.copyfile(source,backup/name)
            files.require(files.digest(source)==files.digest(backup/name),'Counter failure backup differs')
    load = files.read(root/'load.json') if (root/'load.json').exists() else {}
    files.write(root/'counter_recovery.json',dict(backup=str(backup),
        original_protocol={k:v for k,v in plan.items() if k!='files'},
        prior_failed_method_cost=counts, prior_model_gate=load,
        failed_case=row['case'], failed_arm=row['arm']))
    destination = backup/path.parent.relative_to(root)
    destination.parent.mkdir(parents=True,exist_ok=True)
    path.parent.rename(destination)
    (root/'plan.json').rename(backup/'original_plan.json')
    print('PIGDM COUNTER FAILURE PRESERVED:',backup,'prior method F/VJP=1/1',flush=True)
    return backup
