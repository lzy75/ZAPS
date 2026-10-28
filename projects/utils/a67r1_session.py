"""Narrow A67 default-requires_grad repair; preserve original source and failures."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS))
from utils import a67_session as session
from utils import diag_noise_feasible_zaps as driver
from utils import historical_ffhq_gain as files
from utils import a67_model_freeze as freeze
from utils.a54_session import exclusive,busy

ENTRY=Path(__file__).resolve()
CARD=PROJECTS/'experiment_cards/A67R1_model_flag_repair_20261028.md'
OLD_MEASURE=driver.measure
OLD_SUMMARIZE=driver.summarize
OLD_INVOKE=session.invoke
EXPECTED={'00000_n025_s1000','00000_n050_s1000'}


def validate_failure(record):
    c=record.get('full_counts') or {}
    files.require(record.get('status')=='failed' and record.get('error')=="ValueError('Model weights not frozen')",
                  'Different failure; no automatic repair')
    files.require(record.get('arm')=='BASE' and record['case']['key'] in EXPECTED,'Unexpected case/arm')
    files.require(c.get('forward_attempts')==c.get('forwards')==300 and
                  c.get('optimizer_backward_attempts')==c.get('optimizer_backward_completed')==10 and
                  c.get('input_vjp_attempts')==0 and c.get('calls')==files.grid()[::-1]*10 and
                  record.get('replay_counts') is None and not record.get('artifacts'),
                  'Failure differs from observed post300 flag check')
    return dict(method_forward_attempts=300,diagnostic_forward_attempts=0,
                optimizer_backward_attempts=10,input_vjp=0)


def repair_check(root):
    root=Path(root);driver.check(root);m=files.read(root/'repair_r1/manifest.json')
    for name,sha in m['sources'].items():
        files.require(files.digest(PROJECTS/name)==sha,'R1 source changed: '+name)
    for entry in m['failures']:
        saved=root/entry['saved'];old=root/entry['original']
        source=saved if saved.exists() else old
        files.require(files.digest(source/'run.json')==entry['run_sha256'],'Original failure changed')
        validate_failure(files.read(source/'run.json'))
    return m


def prepare_recovery(root,state):
    root,state=Path(root),Path(state)
    files.require(not busy(state/'coordinator.lock'),'Original coordinator still active')
    for f in (root/'workers').glob('*/worker.lock'):files.require(not busy(f),'Original worker still active')
    driver.check(root);folder=root/'repair_r1';manifest=folder/'manifest.json'
    if not manifest.exists():
        bad=[f for f in (root/'cases').glob('*/*/run.json') if files.read(f)['status']!='complete']
        files.require(len(bad)==2,'Expected exactly the two uploaded flag failures')
        rows=[files.read(f) for f in bad]
        files.require({r['case']['key'] for r in rows}==EXPECTED,'Different failed case set')
        failures=[];cost=dict(method_forward_attempts=0,diagnostic_forward_attempts=0,
                              optimizer_backward_attempts=0,input_vjp=0,warmup_forward_attempts=0)
        for f,r in zip(bad,rows):
            files.require(r['plan_sha256']==files.digest(root/'plan.json'),'Wrong original plan')
            for key,value in validate_failure(r).items():cost[key]+=value
            relative=f.parent.relative_to(root)
            failures.append(dict(original=str(relative),saved=str(Path('repair_r1/failed_cases')/relative),
                                 run_sha256=files.digest(f)))
        loads=list((root/'workers').glob('*/load_*.json'))
        for f in loads:cost['warmup_forward_attempts']+=files.read(f)['warmup_forward_attempts']
        names=[str(ENTRY.relative_to(PROJECTS)),str(CARD.relative_to(PROJECTS)),
               'utils/a67_model_freeze.py','utils/test_a67r1_repair.py','run_a67r1_20261028.sh']
        folder.mkdir(exist_ok=True)
        files.write(manifest,dict(phase='prepared',failures=failures,recorded_failed_cost=cost,
                    original_plan_sha256=files.digest(root/'plan.json'),
                    sources={name:files.digest(PROJECTS/name) for name in names},
                    change='NN requires_grad False only during execution; restore flags; original numerical sources unchanged',
                    outputs_reusable=False,reason='Failure was before output/entry save; two300-call baseline attempts must repeat'))
    m=repair_check(root)
    if m['phase']=='active':return m
    for entry in m['failures']:
        old,saved=root/entry['original'],root/entry['saved']
        if not saved.exists():saved.parent.mkdir(parents=True,exist_ok=True);old.rename(saved)
    # Logs/statuses are preserved before the normal coordinator writes fresh ones.
    paths=[*state.glob('*.log'),state/'session.json',*(root/'workers').glob('*/*.json')]
    for f in paths:
        if not f.exists():continue
        relative=Path('session')/f.name if f.parent==state else f.relative_to(root)
        target=folder/'prior_state'/relative
        files.require(not target.exists(),'Prior-state snapshot collision')
        target.parent.mkdir(parents=True,exist_ok=True);f.rename(target)
    if (root/'logs').exists() and not (folder/'prior_logs').exists():shutil.copytree(root/'logs',folder/'prior_logs')
    m['phase']='active';files.write(manifest,m)
    print('A67R1 READY: preserve600 failed forwards/20 optimization backwards/2 warmups; '
          'no saved outputs to reuse. Original plan/formulas unchanged.',flush=True)
    return m


def measure(root,p,c,arm,model,uuid):
    repair_check(root)
    with freeze.frozen_network(model.model):
        return OLD_MEASURE(root,p,c,arm,model,uuid)


def summarize(root):
    summary=OLD_SUMMARIZE(root);m=repair_check(root)
    summary['repair_r1']=dict(recorded_failed_cost=m['recorded_failed_cost'],
                             original_plan_unchanged=True,model_flags_explicitly_frozen=True)
    summary['cumulative_recorded_cost']={key:summary['actual_cost'].get(key,0)+value
                                        for key,value in m['recorded_failed_cost'].items()}
    summary['recovery_did_not_reuse_failed_outputs']=True
    files.write(Path(root)/'summary.json',summary)
    with (Path(root)/'comparison.md').open('a') as f:
        f.write('\nR1 only freezes/restores NN flags. Prior failed cost: '+str(m['recorded_failed_cost'])+
                '; cumulative cost: '+str(summary['cumulative_recorded_cost'])+'.\n')
    print('A67R1 CUMULATIVE COST:',summary['cumulative_recorded_cost'],flush=True)
    return summary


def invoke(*arguments):
    arguments=list(arguments)
    if arguments and arguments[0]=='utils/diag_noise_feasible_zaps.py':
        arguments[0]=str(ENTRY)
    if arguments[:2]==['-m','unittest']:arguments.append('utils.test_a67r1_repair')
    return OLD_INVOKE(*arguments)


def phase(root,state,assigned,name):
    # Original phase's child command routes to this repair entry, not old driver.
    original_popen=subprocess.Popen
    def popen(command,*args,**kwargs):
        command=[str(ENTRY) if v=='utils/diag_noise_feasible_zaps.py' else v for v in command]
        return original_popen(command,*args,**kwargs)
    with patch.object(session.subprocess,'Popen',popen):return OLD_PHASE(root,state,assigned,name)


OLD_PHASE=session.phase


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=driver.ROOT)
    p.add_argument('--ffhq-dir',default='/home/lzy/FFHQ/00000/00000');p.add_argument('--gpus',default='0,1')
    import os
    p.add_argument('--visible',default=os.environ.get('CUDA_VISIBLE_DEVICES'))
    p.add_argument('--shard',type=int,choices=(0,1));p.add_argument('--phase',choices=('feedback','time'))
    for flag in ('worker','coordinator','status','archive-only','prepare','summarize','archive'):
        p.add_argument('--'+flag,action='store_true')
    a=p.parse_args();root=Path(a.root).resolve();state=session.state_dir(root);state.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        stack.enter_context(patch.object(driver,'measure',measure))
        stack.enter_context(patch.object(driver,'summarize',summarize))
        stack.enter_context(patch.object(session,'invoke',invoke))
        stack.enter_context(patch.object(session,'phase',phase))
        if a.worker:repair_check(root);driver.worker(a);return
        if a.prepare:repair_check(root);driver.prepare(a);return
        if a.summarize:summarize(root);return
        if a.archive or a.archive_only:
            if busy(state/'coordinator.lock'):
                files.require(a.archive and not a.archive_only and
                              os.getppid()==files.read(state/'session.json')['pid'],
                              'Wait for running session; only its direct archive child is admitted')
            repair_check(root);driver.archive(root);return
        if a.status:
            print(dict(active=busy(state/'coordinator.lock'),session=files.read(state/'session.json')
                       if (state/'session.json').exists() else None),flush=True);return
        if a.coordinator:repair_check(root);session.coordinator(a);return
        with exclusive(state/'client.lock'):
            proc=None
            if not busy(state/'coordinator.lock'):
                prepare_recovery(root,state)
                if (state/'session.json').exists():
                    old=files.read(state/'session.json')
                    files.require(old['status']!='failed','New R1 failure retained; no automatic retry')
                    if old['status']=='complete':print('A67R1 already complete:',root/'comparison.md',flush=True);return
                command=[sys.executable,'-u',str(ENTRY),'--coordinator','--root',str(root),
                         '--gpus',a.gpus,'--ffhq-dir',a.ffhq_dir]
                if a.visible is not None:command+=['--visible',a.visible]
                with (state/'coordinator.log').open('ab',buffering=0) as log:
                    proc=subprocess.Popen(command,cwd=PROJECTS,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                          start_new_session=True,close_fds=True)
                print('A67R1 detached recovery started; following until final summary.',flush=True)
            else:print('A67 active recovery attached; no duplicate.',flush=True)
            session.follow(state,proc)


if __name__=='__main__':main()
