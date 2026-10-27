"""A56 selection-only revision: candidate inventories are not experimental usage."""
import argparse
from contextlib import ExitStack,contextmanager
from datetime import datetime,timezone
from pathlib import Path
import re
import shutil
import sys
import time

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import diag_frozen_direction_transfer as original
from utils.a54_session import exclusive

OLD_ROOT=Path(original.ROOT)
BASE=OLD_ROOT.parent/'selection_r2_20261027'
ROOT=BASE/'A56_selection_r2_20261027'
CARD=PROJECTS/'experiment_cards/A56_selection_r2_20261027.md'
BRANCH='results/frozen-direction-transfer-selection-r2-20261027'
# Keep all ordinary/unknown image references conservative. Only explicit candidate containers are non-usage.
CANDIDATES=frozenset(('pool','candidate_pool','selection_pool','candidate_images','candidates','eligible_pool','eligible_images','available_images','file_inventory','transfer_inventory'))


def collect_usage(document,data):
    data={d:Path(folder).resolve() for d,folder in data.items()}
    actual={d:set() for d in data};legacy={d:set() for d in data}
    hashes=set();all_hashes=set();ignored=[];references=0
    def visit(v,path=(),candidate=False):
        nonlocal references
        if isinstance(v,dict):
            sha=v.get('image_sha256');image=v.get('image')
            if isinstance(sha,str):
                all_hashes.add(sha)
                if not candidate:hashes.add(sha)
            if isinstance(image,str):
                resolved=Path(image).resolve()
                for d,folder in data.items():
                    if resolved.parent==folder:
                        legacy[d].add(resolved.stem)
                        if not candidate:actual[d].add(resolved.stem)
                if not candidate:references+=1
            for key,item in v.items():
                if key in ('experiment_card','limits','scope'):continue
                if key in CANDIDATES and isinstance(item,(dict,list)):
                    ignored.append(dict(pointer='/'.join((*path,key)),direct_items=len(item)))
                    visit(item,(*path,key),True)
                else:visit(item,(*path,key),candidate)
        elif isinstance(v,list):
            for i,item in enumerate(v):visit(item,(*path,str(i)),candidate)
    visit(document)
    return dict(used_names=actual,used_hashes=hashes,legacy_names=legacy,legacy_hashes=all_hashes,ignored=ignored,actual_references=references)


def history_selection(data,root):
    paths=set()
    for base in (PROJECTS/'results',PROJECTS/'server_records',Path('/data0/lzy/ZAPS-results')):
        if base.exists():
            for pattern in ('run.json','plan.json'):
                paths.update(f.resolve() for f in base.rglob(pattern) if not f.resolve().is_relative_to(root))
    used={d:set() for d in data};old={d:set() for d in data};hashes=set();oldhash=set();audit=[]
    for path in sorted(paths):
        v=collect_usage(original.source.read(path),data)
        for d in data:used[d].update(v['used_names'][d]);old[d].update(v['legacy_names'][d])
        hashes.update(v['used_hashes']);oldhash.update(v['legacy_hashes'])
        audit.append(dict(path=str(path),sha256=original.source.digest(path),actual_references=v['actual_references'],ignored_candidate_containers=v['ignored']))
    if not audit:raise ValueError('No accessible history; cannot claim unused images')
    selected={};pools={}
    for d,folder in data.items():
        files=original.rules.eligible_paths(folder)
        raw=[dict(image=str(f.resolve()),image_id=f.stem,image_sha256=original.source.digest(f)) for f in files]
        available=[c for c in raw if c['image_id'] not in used[d] and c['image_sha256'] not in hashes]
        legacy_available=[c for c in raw if c['image_id'] not in old[d] and c['image_sha256'] not in oldhash]
        unique=len({c['image_sha256'] for c in available})
        counts=dict(directory=str(folder),eligible_pngs=len(files),actual_used_names=len(used[d]),legacy_recursive_used_names=len(old[d]),legacy_unused_unique=len({c['image_sha256'] for c in legacy_available}),corrected_unused_unique=unique)
        print('A56S2 SELECTION COUNTS:',d,counts,flush=True)
        if unique<original.rules.COUNTS[d]:
            raise ValueError('A56S2 genuine eligible pool insufficient AFTER candidate correction: '+d+' '+str(counts)+'; no reuse, range expansion or outcome-based substitution')
        chosen=original.rules.select(raw,used[d],hashes,original.rules.COUNTS[d],20261026+original.rules.DATASETS.index(d))
        selected[d]=chosen;pools[d]=dict(**counts,selected=chosen);hashes.update(c['image_sha256'] for c in chosen);oldhash.update(c['image_sha256'] for c in chosen)
    return selected,dict(implementation_revision='A56_SELECTION_R2',history=audit,pools=pools,
        excluded_candidate_keys=sorted(CANDIDATES),scope='Accessible actual/planned/failed cases and ordinary image references excluded; explicit candidate inventories do not imply use. Same32..999 PNG range, content de-duplication and original fixed selection seeds. Not globally unseen/model-training-held-out guarantee.')


def own_sources():
    paths=[Path(__file__),CARD,PROJECTS/'utils/a56_selection_r2_session.py',PROJECTS/'run_a56_selection_r2_20261027.sh',PROJECTS/'utils/test_a56_selection_r2.py']
    return {str(p):original.source.digest(p) for p in paths}


def proof():
    p=BASE/'preflight_snapshot/proof.json';v=original.source.load_signed(p)
    if v['revision_sources']!=own_sources():raise ValueError('Selection revision source binding changed')
    for path,sha in v['snapshots'].items():
        if original.source.digest(path)!=sha:raise ValueError('Preserved selection-failure snapshot changed')
    return p,v


def preflight_once():
    BASE.mkdir(parents=True,exist_ok=True)
    with ExitStack() as locks:
        for base in (OLD_ROOT.parent,OLD_ROOT.parent/'operator_repair_20261027'):
            if base.exists():
                for name in ('session.lock','launcher.lock'):
                    locks.enter_context(exclusive(base/name))
        if (OLD_ROOT/'plan.json').exists():
            raise ValueError('Original A56 already has a frozen plan; do not reselect. Resume/inspect that plan instead.')
        records=list(OLD_ROOT.rglob('run.json')) if OLD_ROOT.exists() else []
        if records:raise ValueError('Original case records exist; no automatic cohort replacement: '+str(records[:3]))
        if (BASE/'preflight_snapshot/proof.json').exists():
            proof();print('A56S2 zero-UNet preflight snapshot reused.',flush=True);return
        logs=[OLD_ROOT.parent/'session.log',OLD_ROOT.parent/'launcher_20261026.log']
        text='\n'.join(f.read_text(errors='replace') for f in logs if f.exists())
        if not (re.search(r'Need\s*6\s+unused\s+unique\s*images;\s*available=0',text) and 'history_selection' in text):
            print('\n'.join(text.splitlines()[-100:]),flush=True)
            raise ValueError('Original failure is not the reported pre-freeze zero-eligible selection error; no automatic retry')
        snapshot=BASE/'preflight_snapshot';snapshot.mkdir();saved={}
        for base,label in ((OLD_ROOT.parent,'original'),(OLD_ROOT.parent/'operator_repair_20261027','operator_repair')):
            if not base.exists():continue
            for pattern in ('*.log','session.json'):
                for f in base.glob(pattern):
                    if f.is_file():
                        target=snapshot/(label+'_'+f.name);shutil.copyfile(f,target);saved[str(target)]=original.source.digest(target)
        v=dict(created_utc=datetime.now(timezone.utc).isoformat(),revision_sources=own_sources(),snapshots=saved,
               original_plan_absent=True,original_run_records_absent=True,original_prepare_failure='Need6 unused unique images; available=0',new_UNet_calls=0)
        original.signed(snapshot/'proof.json',v)
        print('A56S2 PREFLIGHT PASS: original plan/NN attempts absent, failed logs preserved; no UNet calls.',flush=True)


def preflight():
    while True:
        try:return preflight_once()
        except BlockingIOError:
            print('A56 original session/launcher still active; waiting15s, no duplicate computation.',flush=True)
            time.sleep(15)


@contextmanager
def bindings():
    from unittest.mock import patch
    signing=original.signed
    def signed(path,value):
        if Path(path).name=='plan.json':
            p,v=proof();value=dict(value)
            local=ROOT/'source_snapshot/selection_failure';local.mkdir(parents=True,exist_ok=True);snapshot_files={}
            for f in p.parent.iterdir():
                if not f.is_file():continue
                target=local/f.name
                if target.exists() and original.source.digest(target)!=original.source.digest(f):raise ValueError('Frozen failure snapshot copy changed')
                if not target.exists():shutil.copyfile(f,target)
                snapshot_files[str(target)]=original.source.digest(target)
            value['implementation_revision']='A56_SELECTION_R2'
            value['worker_identity_files']={**value['worker_identity_files'],**own_sources()}
            value['identity_files']={**value['identity_files'],**snapshot_files}
            value['selection_preflight_snapshot']=str(local/'proof.json')
            value['limits']=value['limits'].replace('A standalone no-learning replay is not yet confirmed.','A57 confirms old-cohort standalone equivalence; this transfer protocol retains its inherited300-forward charge.')
        return signing(path,value)
    with patch.object(original,'ROOT',str(ROOT)),patch.object(original,'CARD',CARD),patch.object(original,'BRANCH',BRANCH),patch.object(original,'history_selection',history_selection),patch.object(original,'signed',signed):yield


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preflight',action='store_true')
    args,rest=parser.parse_known_args()
    if args.preflight:
        if rest:parser.error('--preflight has no additional arguments')
        preflight();return
    if '--root' in rest or any(a.startswith('--root=') for a in rest):raise ValueError('Use the separate frozen revision root only')
    proof();before=sys.argv
    try:
        with bindings():
            sys.argv=[original.__file__,*rest];original.main()
    finally:sys.argv=before

if __name__=='__main__':main()
