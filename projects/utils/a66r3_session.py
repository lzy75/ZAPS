"""A66R3: repeat variability is measured, not hidden or used to select an output."""
from contextlib import ExitStack
import builtins
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS))
from utils import a66r2_session as previous
from utils import repeat_variation_measurement as measurement
from utils import current_strong_anchor as rules

ENTRY=Path(__file__).resolve();CARD=PROJECTS/'experiment_cards/A66R3_repeat_variation_measurement_20261028.md'
OLD_CHECK=previous.manifest_check
OLD_AUGMENT=previous.augment_summary
OLD_WORKER=previous.launcher.driver.worker


def validate_failure(r):
    c=r.get('counts') or {};error=r.get('repeat_relative_error')
    rules.require(r.get('status')=='failed' and r.get('error')=="ValueError('Repeat drift; no best-output selection')",
                  'Different failure; no automatic retry')
    rules.require((r.get('arm'),r.get('repeat'),r.get('case',{}).get('image_id'))==('LEGACY_IEEE',1,'00000'),
                  'Unexpected repeat failure')
    rules.require(isinstance(error,(int,float)) and math.isfinite(error) and error>1e-6,'Missing repeat difference')
    rules.require(c.get('forward_attempts')==c.get('forwards')==300 and c.get('input_vjp_attempts')==0 and
                  c.get('optimizer_backward_attempts')==c.get('optimizer_backward_completed')==10,'Unexpected costs')
    return dict(forward_attempts=300,forwards=300,input_vjp_attempts=0,optimizer_backward_attempts=10)


def manifest_check(root):
    root=Path(root);prior=OLD_CHECK(root);m=rules.read(root/'repair_r3/manifest.json')
    for path,sha in m['bound_sources'].items():rules.require(rules.digest(path)==sha,'R3 source changed: '+path)
    old,saved=root/m['original'],root/m['saved'];file=saved/'run.json' if saved.exists() else old/'run.json'
    rules.require(rules.digest(file)==m['run_sha256'],'Prior repeat-failure record changed')
    validate_failure(rules.read(file));return prior


def prepare_recovery(root,state,*,validate_completed=None):
    root,state=Path(root),Path(state);OLD_CHECK(root);folder=root/'repair_r3';file=folder/'manifest.json'
    if not file.exists():
        bad=[p for p in (root/'cases').glob('*/*/repeat_*/run.json') if rules.read(p)['status']!='complete']
        rules.require(len(bad)==1,'Need exactly the reported repeat failure')
        r=rules.read(bad[0]);cost=validate_failure(r)
        rules.require(r['plan_sha256']==rules.digest(root/'plan.json'),'Original plan differs')
        if validate_completed:validate_completed()
        folder.mkdir(exist_ok=True);relative=bad[0].parent.relative_to(root)
        files=[ENTRY,CARD,Path(measurement.__file__),PROJECTS/'run_a66r3_20261028.sh',PROJECTS/'utils/test_a66r3_session.py',
               PROJECTS/'utils/test_repeat_variation_measurement_torch.py']
        rules.write(file,dict(phase='prepared',original=str(relative),saved=str(Path('repair_r3/failed_repeats')/relative),
            run_sha256=rules.digest(bad[0]),failed_repeat_cost=cost,old_gate_relaxed=False,
            new_policy='All repeats retained with failed/true reproducibility flags; no best-output selection',
            bound_sources={str(p):rules.digest(p) for p in files},
            remaining=dict(runs=90,method_forwards=20250,input_vjp_max=750,optimizer_backwards=650)))
    manifest_check(root);m=rules.read(file)
    if m['phase']=='active':
        for p in (root/'cases').glob('*/*/repeat_*/run.json'):
            rules.require(rules.read(p)['status']=='complete','New execution failure retained; no automatic retry')
        return m
    old,saved=root/m['original'],root/m['saved']
    if not saved.exists():saved.parent.mkdir(parents=True,exist_ok=True);old.rename(saved)
    else:rules.require(not old.exists(),'Old and archived failure both exist')
    for source in [state/'session.json',state/'coordinator.log',*state.glob('shard*.log'),
                   *(root/'workers').glob('*/status.json')]:
        if source.exists():
            relative=source.relative_to(root) if source.is_relative_to(root) else Path('session')/source.name
            target=folder/'prior_state'/relative;rules.require(not target.exists(),'Snapshot collision')
            target.parent.mkdir(parents=True,exist_ok=True);source.rename(target)
    m['phase']='active';rules.write(file,m)
    print('A66R3 READY: retain every repeat and its gate; no best-output selection; next unfinished cases only.',flush=True)
    return m


def worker(args):
    from utils.a54_session import exclusive
    root=Path(args.root).resolve();manifest_check(root)
    folder=root/'workers'/('shard'+str(args.shard));folder.mkdir(parents=True,exist_ok=True)
    with exclusive(folder/'operator_audit.lock'):
        result=measurement.operator_audit('cuda:0')
        file=folder/f'operator_audit_{len(list(folder.glob("operator_audit_*.json"))):03d}.json'
        rules.write(file,result);print('A66R3 ZERO-UNET OPERATOR AUDIT:',result,flush=True)
    return OLD_WORKER(args)


def augment_summary(root,summary):
    summary=OLD_AUGMENT(root,summary);m=rules.read(Path(root)/'repair_r3/manifest.json')
    total=dict(summary['new_cost_including_archived_failures']);cost=m['failed_repeat_cost']
    for key,source in [('method_forward_attempts','forward_attempts'),('method_forwards_completed','forwards'),
                       ('input_vjp_attempts','input_vjp_attempts'),('optimizer_backward_attempts','optimizer_backward_attempts')]:
        total[key]+=cost[source]
    p=rules.read(Path(root)/'plan.json')
    records=[rules.read(p['references'][c['key']]['local_record']) for c in p['cases']]
    records += [rules.read(f) for f in (Path(root)/'cases').glob('*/*/repeat_*/run.json') if rules.read(f)['status']=='complete']
    summary.update(classification='QUALITY_AND_CLOCK_MEASURED_REPRODUCIBILITY_UNRESOLVED',
        primary_first_repeat_descriptive=summary['primary'],
        primary=dict(classification='NO_CONFIRMED_GAIN_PENDING_REPRODUCIBILITY_AUDIT'),
        repeat_variation=measurement.repeat_statistics(records),
        archived_repeat_failure_cost=cost,new_cost_including_archived_failures=total,
        original_repeat_gate_passed=False,repeat0_quality_not_best_selected=True,
        operator_audits=[rules.read(f) for f in sorted((Path(root)/'workers').glob('*/operator_audit_*.json'))])
    rules.write(Path(root)/'summary.json',summary)
    with (Path(root)/'comparison.md').open('a') as f:
        f.write('\nA66R3：逐像素重复检查已有失败，不宣称已解决重现性。所有新重复保存输出/质量/耗时和原1e-6检查的真假值；'
                '附逐图三次指标与标准差，不挑最好输出。repeat0均值是描述性结果，不是最终确认收益；300版本仍暂停。\n')
    print('A66R3 FINAL CLASSIFICATION:',summary['classification'],flush=True)
    print('A66R3 Summary:',json.dumps(dict(classification=summary['classification'],
        original_repeat_gate_passed=False,groups=[{key:g[key] for key in
            ('arm','images','complete_runs','quality','mean_of_image_median_seconds') if key in g}
            for g in summary['groups']],cost=total,
        note='All repeat outputs retained; initial-repeat quality is descriptive; 300 arm remains rejected'),
        ensure_ascii=False),flush=True)
    return summary


def main():
    def report(*values,**kwargs):
        if values and values[0]=='A66 Summary:':
            builtins.print('A66R3 measurements collected; writing explicit reproducibility flags and repeat statistics.',flush=True)
        else:builtins.print(*values,**kwargs)
    with ExitStack() as stack:
        stack.enter_context(patch.object(previous,'ENTRY',ENTRY))
        stack.enter_context(patch.object(previous,'prepare_recovery',prepare_recovery))
        stack.enter_context(patch.object(previous,'manifest_check',manifest_check))
        stack.enter_context(patch.object(previous,'augment_summary',augment_summary))
        stack.enter_context(patch.object(previous.launcher.driver,'worker',worker))
        stack.enter_context(patch.object(previous.launcher.driver.executor,'measure',measurement.measure))
        stack.enter_context(patch.object(previous.launcher.driver,'print',report,create=True))
        previous.main()


if __name__=='__main__':main()
