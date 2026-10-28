"""Continue independent30 and controls; reject, do not retry, failed learned-prefix arm."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
from unittest.mock import patch

PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS))
from utils import a66r1_session as launcher
from utils import counted_adjoint_repair as repair
from utils import current_strong_anchor as rules

ENTRY=Path(__file__).resolve()
CARD=PROJECTS/'experiment_cards/A66R2_continue_valid_arms_20261028.md'
OLD_CHECK=repair.manifest_check
OLD_AUGMENT=repair.augment_summary
OLD_QUEUE=rules.queue


def queue():return [(arm,repeat) for arm,repeat in OLD_QUEUE() if arm!='RESPONSE_300']


def validate_rejection(r):
    c=r.get('counts') or {};g=r.get('independent_prefix_gate') or {};e=g.get('relative_errors') or {}
    rules.require(r.get('status')=='failed' and r.get('error')=="ValueError('Independent270 prefix differs')",
                  'Different failure: inspect; no automatic retry')
    rules.require((r.get('arm'),r.get('repeat'),r.get('case',{}).get('image_id'))==('RESPONSE_300',0,'00000'),
                  'Unexpected rejected case')
    rules.require(c.get('forward_attempts')==c.get('forwards')==300 and c.get('input_vjp_attempts')==30 and
                  c.get('optimizer_backward_attempts')==c.get('optimizer_backward_completed')==9,'Unexpected costs')
    rules.require(g.get('passed') is False and g.get('rng_equal') is True and
                  e.get('tau')==e.get('initial')==e.get('cpu_rng')==0 and
                  (e.get('zeta',0)>1e-6 or e.get('D',0)>1e-6),'Unexpected gate; do not relabel')
    return dict(forward_attempts=300,forwards=300,input_vjp_attempts=30,optimizer_backward_attempts=9)


def manifest_check(root):
    root=Path(root);prior=OLD_CHECK(root);file=root/'repair_r2/manifest.json'
    rules.require(file.exists(),'Prepare R2 recovery first')
    m=rules.read(file)
    for name,sha in m['bound_sources'].items():rules.require(rules.digest(name)==sha,'Repair source changed: '+name)
    original,saved=root/m['original'],root/m['saved']
    record=saved/'run.json' if saved.exists() else original/'run.json'
    rules.require(rules.digest(record)==m['run_sha256'],'Rejected record changed')
    validate_rejection(rules.read(record))
    return prior


def prepare_recovery(root,state,*,validate_completed=None):
    root,state=Path(root),Path(state);folder=root/'repair_r2';file=folder/'manifest.json'
    OLD_CHECK(root)
    if not file.exists():
        bad=[p for p in (root/'cases').glob('*/*/repeat_*/run.json') if rules.read(p)['status']!='complete']
        rules.require(len(bad)==1,'Only one known prefix rejection may be isolated')
        r=rules.read(bad[0]);cost=validate_rejection(r)
        rules.require(r['plan_sha256']==rules.digest(root/'plan.json'),'Rejection plan differs')
        if validate_completed:validate_completed()
        relative=bad[0].parent.relative_to(root);folder.mkdir(exist_ok=True)
        sources=[ENTRY,CARD,PROJECTS/'run_a66r2_20261028.sh',PROJECTS/'utils/test_a66r2_session.py']
        rules.write(file,dict(phase='prepared',original=str(relative),saved=str(Path('repair_r2/rejected_attempts')/relative),
            run_sha256=rules.digest(bad[0]),rejected_cost=cost,
            bound_sources={str(p):rules.digest(p) for p in sources},
            suspended_arm='RESPONSE_300',gate_relaxed=False,
            remaining=dict(runs=95,method_forwards=20400,input_vjp_max=900,optimizer_backwards=650)))
    manifest_check(root);m=rules.read(file)
    if m['phase']=='active':
        for p in (root/'cases').glob('*/*/repeat_*/run.json'):
            rules.require(rules.read(p)['status']=='complete','New paid failure preserved; no automatic retry')
        return m
    old,saved=root/m['original'],root/m['saved']
    if not saved.exists():saved.parent.mkdir(parents=True,exist_ok=True);old.rename(saved)
    else:rules.require(not old.exists(),'Original and saved rejection both exist')
    files=[state/'session.json',state/'coordinator.log',*state.glob('shard*.log'),
           *(root/'workers').glob('*/status.json')]
    for source in files:
        if source.exists():
            relative=source.relative_to(root) if source.is_relative_to(root) else Path('session')/source.name
            target=folder/'prior_state'/relative
            rules.require(not target.exists(),'Snapshot collision')
            target.parent.mkdir(parents=True,exist_ok=True);source.rename(target)
    m['phase']='active';rules.write(file,m)
    print('A66R2 READY: failed300 arm remains REJECTED; independent30 and controls continue; 20400F remaining.',flush=True)
    return m


def augment_summary(root,summary):
    summary=OLD_AUGMENT(root,summary);m=rules.read(Path(root)/'repair_r2/manifest.json')
    total=dict(summary['new_cost_including_archived_failures']);cost=m['rejected_cost']
    for key,source in [('method_forward_attempts','forward_attempts'),('method_forwards_completed','forwards'),
                       ('input_vjp_attempts','input_vjp_attempts'),('optimizer_backward_attempts','optimizer_backward_attempts')]:
        total[key]+=cost[source]
    for group in summary['groups']:
        if group['arm']=='RESPONSE_300':
            group.update(status='REJECTED_PREFIX_GATE_NOT_RETRIED',included_in_quality_ranking=False)
    summary.update(classification='PARTIAL_FOUR_ARM_COMPARISON_300_ARM_REJECTED',
        original_five_arm_experiment_complete=False,continued_arms=[a for a in summary['groups'] if a['arm']!='RESPONSE_300'],
        rejected_prefix_cost=cost,new_cost_including_archived_failures=total,
        prefix_gate_failure=rules.read(Path(root)/m['saved']/'run.json')['independent_prefix_gate'])
    rules.write(Path(root)/'summary.json',summary)
    with (Path(root)/'comparison.md').open('a') as stream:
        stream.write('\nA66R2：300前向新方法未通过参数前缀一致性门控，暂停且不纳入质量/速度排名；'
                     '其300F/30输入反向/9参数反向与原失败2F均保留计费。仅完成其余四组不代表原五组全部完成。\n')
    return summary


def main():
    with ExitStack() as stack:
        stack.enter_context(patch.object(launcher,'ENTRY',ENTRY))
        stack.enter_context(patch.object(repair,'prepare_recovery',prepare_recovery))
        stack.enter_context(patch.object(repair,'manifest_check',manifest_check))
        stack.enter_context(patch.object(repair,'augment_summary',augment_summary))
        stack.enter_context(patch.object(rules,'queue',queue))
        launcher.main()


if __name__=='__main__':main()
