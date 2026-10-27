"""Run frozen A56/A57 through an explicitly pinned operator-interface adapter."""
import argparse
from contextlib import ExitStack,contextmanager
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys
import time

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import diag_frozen_direction_transfer as a56
from utils import diag_no_learning_replay as a57
from utils import direction_operator_repair as adapter
from utils.a54_session import exclusive

DRIVERS=dict(A56=a56,A57=a57)
CARD=PROJECTS/'experiment_cards/A56_A57_operator_interface_repair_20261027.md'
REPAIR='operator_repair_20261027'


def own_sources():
    paths=[Path(__file__),Path(adapter.__file__),CARD,PROJECTS/'utils/test_direction_operator_repair.py']
    for n in (56,57):
        paths.extend([PROJECTS/f'utils/a{n}_operator_repair_session.py',PROJECTS/f'run_a{n}_operator_repair_20261027.sh'])
    return {str(p):a57.source.digest(p) for p in paths}


def invoke(driver,args):
    before=sys.argv
    try:
        sys.argv=[driver.__file__,*args];return driver.main()
    finally:sys.argv=before


def load_manifest(package,root):
    driver=DRIVERS[package];m=driver.source.load_signed(root/REPAIR/'manifest.json')
    if (m['package']!=package or m['original_plan_sha256']!=driver.source.digest(root/'plan.json')
            or m['repair_sources']!=own_sources()):
        raise ValueError('Repair code/plan binding changed; no execution')
    for f,sha in m['source_records'].items():
        if driver.source.digest(f)!=sha:raise ValueError('Pinned original record changed:'+f)
    return m


def preserve(root,m):
    for entry in m['preserved_zero_call_failures']:
        old,saved=root/entry['original'],root/entry['preserved']
        if old.exists() and saved.exists():raise ValueError('Ambiguous failure preservation; both copies retained')
        if not saved.exists():
            if not old.exists():raise ValueError('Journaled failed bytes missing')
            if {str(f.relative_to(old)):a57.source.digest(f) for f in old.rglob('*') if f.is_file()}!=entry['inventory']:
                raise ValueError('Failed attempt bytes changed')
            saved.parent.mkdir(parents=True,exist_ok=True);old.rename(saved)
        if {str(f.relative_to(saved)):a57.source.digest(f) for f in saved.rglob('*') if f.is_file()}!=entry['inventory']:
            raise ValueError('Preserved failure changed')


def install_once(package):
    driver=DRIVERS[package];root=Path(driver.ROOT);base=root.parent;base.mkdir(parents=True,exist_ok=True)
    with ExitStack() as locks:
        for path in (base/'session.lock',base/'launcher.lock'):
            locks.enter_context(exclusive(path))
        if root.exists():
            for shard in (0,1):
                for prefix in ('worker','control'):
                    locks.enter_context(exclusive(root/f'{prefix}_shard{shard}.lock'))
        prior_state=base/'session.json';prior_log=base/'session.log'
        if prior_state.exists() and driver.source.read(prior_state).get('status')=='failed':
            text=prior_log.read_text(errors='replace') if prior_log.exists() else ''
            if "KeyError: 'mask_seed'" not in text and 'KeyError("mask_seed")' not in text:
                print('\n'.join(text.splitlines()[-90:]),flush=True)
                raise ValueError(package+' previous error is not the reported mask_seed interface bug; no automatic rerun. Full original tail printed above.')
        invoke(driver,['--prepare']);p=driver.check(root)
        target=root/REPAIR/'manifest.json'
        if target.exists():
            m=load_manifest(package,root);preserve(root,m)
            print(package+' REPAIR PIN REUSED: original plan/source bytes unchanged.',flush=True);return
        records={};failures=[]
        all_complete=True
        for c in p['cases']:
            if package=='A56':
                adapter.seed_for_new_case(c)
                driver.baseline_done(root,p,c,'BASELINE')
                all_complete=bool(driver.finished(root,p,c)) and all_complete
            else:
                all_complete=bool(driver.item(root,p,c,'control')) and all_complete
                if c['task']=='inpainting':
                    ref=adapter.mask_reference(p,c,driver.source)
                    records[ref['path']]=ref['sha256'];records[ref['source_record']]=driver.source.digest(ref['source_record'])
        if all_complete and (root/'summary.json').exists():
            print(package+' already complete; original summary retained:',driver.source.load_signed(root/'summary.json'),flush=True)
            raise SystemExit(20)
        for f in root.rglob('run.json'):
            if f.is_relative_to(root/REPAIR):continue
            r=driver.source.read(f)
            if r.get('status')=='complete':
                records[str(f)]=driver.source.digest(f);continue
            if not adapter.recognized_zero_failure(r):
                print('UNRECOGNIZED ATTEMPT:',str(f),r.get('status'),r.get('error'),'F=',r.get('nfe'),flush=True)
                raise ValueError('Nonzero/unknown/interrupted attempt retained; no budget-reset retry')
            old=f.parent;relative=old.relative_to(root)
            saved=Path(REPAIR)/'preserved_zero_call_failures'/relative
            failures.append(dict(original=str(relative),preserved=str(saved),inventory={str(q.relative_to(old)):driver.source.digest(q) for q in old.rglob('*') if q.is_file()},nfe=0,input_VJP=0))
        folder=root/REPAIR;folder.mkdir(exist_ok=True)
        logs=folder/'original_logs';logs.mkdir(exist_ok=True)
        snapshots={}
        for pattern in ('*.log','session.json','devices.txt'):
            for f in base.glob(pattern):
                if f.is_file():
                    out=logs/f.name;shutil.copyfile(f,out);snapshots[str(out)]=driver.source.digest(out)
        records.update(snapshots)
        m=dict(package=package,kind='OPERATOR_INTERFACE_REPAIR_NOT_NEW_METHOD',created_utc=datetime.now(timezone.utc).isoformat(),
            original_plan_sha256=driver.source.digest(root/'plan.json'),repair_sources=own_sources(),source_records=records,
            preserved_zero_call_failures=failures,failed_forward_cost=0,failed_input_VJP_cost=0,
            limits='Only exact zero-call mask_seed failures may be preserved/re-entered. Completed cases reused; original plan/card/source and old session/logs unchanged. Real Torch/model gates remain mandatory.')
        driver.signed(target,m);preserve(root,m)
        print(package+' REPAIR INSTALL PASS: preserved known zero-call failures='+str(len(failures))+'; old plan/cohort unchanged; no UNet calls.',flush=True)


def install(package):
    while True:
        try:return install_once(package)
        except BlockingIOError:
            print(package+' original session/launcher/shard still active; waiting15s, no duplicate computation.',flush=True)
            time.sleep(15)


@contextmanager
def execution(package,root):
    from unittest.mock import patch
    driver=DRIVERS[package];p=driver.check(root);m=load_manifest(package,root);preserve(root,m)
    pin=dict(package=package,manifest_sha256=driver.source.digest(root/REPAIR/'manifest.json'),adapter='direction_operator_repair.py')
    original_signed=driver.signed
    def signed(path,value):
        if Path(path).name in ('run.json','summary.json','evaluation.json'):
            value=dict(value,execution_repair=pin)
        return original_signed(path,value)
    counted_module=a57.engine.old.previous;old_counted=counted_module.counted
    @contextmanager
    def counted(model,path,r,cap):
        r['execution_repair']=pin
        with old_counted(model,path,r,cap) as back:yield back
    factory=lambda case,device:adapter.make_operator(package,case,device,p)
    with patch.object(a56,'make_operator',factory),patch.object(driver,'signed',signed),patch.object(counted_module,'counted',counted):
        yield


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',required=True,choices=tuple(DRIVERS));parser.add_argument('--install',action='store_true')
    args,rest=parser.parse_known_args();driver=DRIVERS[args.package]
    if '--root' in rest or any(v.startswith('--root=') for v in rest):raise ValueError('Repair uses the frozen default root only')
    if args.install:
        if rest:parser.error('--install takes no worker arguments')
        install(args.package);return
    root=Path(driver.ROOT)
    with execution(args.package,root):invoke(driver,rest)

if __name__=='__main__':main()
