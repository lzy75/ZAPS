"""A66R1: transparent operator instrumentation and strict one-time recovery."""
from contextlib import contextmanager
from pathlib import Path
import shutil
from unittest.mock import patch

from utils import current_strong_anchor as rules

ERROR = 'TypeError("count_operator.<locals>.adjoint() got an unexpected keyword argument \'output_size\'")'
CARD = Path(__file__).resolve().parents[1]/'experiment_cards/A66R1_counted_adjoint_interface_repair_20261028.md'


def count_operator(base, count):
    """Forward the original signature and return value, including optional kwargs."""
    forward, transpose = base.H, base.transpose
    def H(value, *args, **kwargs):
        count['base_H_actions'] += 1
        return forward(value, *args, **kwargs)
    def adjoint(value, *args, **kwargs):
        count['base_adjoint_actions'] += 1
        return transpose(value, *args, **kwargs)
    base.H, base.transpose = H, adjoint


@contextmanager
def installed():
    from utils import paper_clock_runtime as common
    with patch.object(common, 'count_operator', count_operator):
        yield


def validate_failure(record):
    c = record.get('counts') or {}
    rules.require(record.get('status') == 'failed' and record.get('error') == ERROR,
                  'Only the verified output_size instrumentation failure is recoverable')
    rules.require(record.get('arm') == 'EXACT_IEEE' and record.get('repeat') == 0 and
                  record.get('case', {}).get('image_id') in ('00000', '00005'), 'Unexpected failed case')
    rules.require(c.get('forward_attempts') == c.get('forwards') == 1 and
                  c.get('input_vjp_attempts') == c.get('optimizer_backward_attempts') ==
                  c.get('optimizer_backward_completed') == 0, 'Unexpected failed-attempt cost')
    return dict(forward_attempts=1, forwards=1, input_vjp_attempts=0, optimizer_backward_attempts=0)


def manifest_check(root):
    root = Path(root); manifest = rules.read(root/'repair_r1/manifest.json')
    for name, sha in manifest['bound_sources'].items():
        rules.require(rules.digest(name) == sha, 'Repair/source changed: '+name)
    for item in manifest['failed_attempts']:
        old, saved = root/item['original'], root/item['saved']
        file = saved/'run.json' if saved.exists() else old/'run.json'
        rules.require(rules.digest(file) == item['run_sha256'], 'Failure record changed')
        validate_failure(rules.read(file))
    return manifest


def prepare_recovery(root, state, *, validate_completed=None):
    """Journal before moves. A new paid failure is never silently cleared."""
    root, state = Path(root), Path(state); folder = root/'repair_r1'
    manifest_file = folder/'manifest.json'
    if not manifest_file.exists():
        plan = rules.read(root/'plan.json')
        rules.require(plan['experiment'] == rules.EXPERIMENT, 'Wrong A66 plan')
        bad = [p for p in sorted((root/'cases').glob('*/*/repeat_*/run.json'))
               if rules.read(p)['status'] != 'complete']
        rules.require(len(bad) == 2, 'Need exactly the two verified interface failures')
        entries = []
        for file in bad:
            r = rules.read(file); validate_failure(r)
            rules.require(r['plan_sha256'] == rules.digest(root/'plan.json'), 'Failure plan differs')
            relative = file.parent.relative_to(root)
            entries.append(dict(original=str(relative), saved=str(Path('repair_r1/failed_attempts')/relative),
                                run_sha256=rules.digest(file), counts=validate_failure(r)))
        rules.require({rules.read(f)['case']['image_id'] for f in bad} == {'00000', '00005'}, 'Duplicate failure')
        if validate_completed: validate_completed()
        files = [Path(__file__), CARD, Path(__file__).with_name('a66r1_session.py'),
                 Path(__file__).with_name('test_counted_adjoint_repair.py'),
                 Path(__file__).with_name('test_counted_adjoint_repair_torch.py'),
                 CARD.parents[1]/'run_a66r1_20261028.sh']
        folder.mkdir(exist_ok=True)
        for name in ('summary.json', 'comparison.md', 'comparison.csv'):
            if (root/name).exists(): shutil.copyfile(root/name, folder/('before_'+name))
        rules.write(manifest_file, dict(version=1, phase='prepared', failed_attempts=entries,
            plan_sha256=rules.digest(root/'plan.json'),
            bound_sources={str(p): rules.digest(p) for p in [root/'plan.json', *files]},
            policy='Known interface repair only; completed outputs reused; no numerical/protocol changes',
            remaining_budget=dict(runs=130, method_forwards=30900, input_vjp_max=1800,
                                  optimizer_backwards=970, initial_reload_warmup_forwards=2)))
    manifest = manifest_check(root)
    if manifest['phase'] == 'active':
        for file in (root/'cases').glob('*/*/repeat_*/run.json'):
            rules.require(rules.read(file)['status'] == 'complete', 'Repair already has a paid failure; inspect, do not retry')
        return manifest
    # Repeatable journaled moves: never overwrite either a failure archive or a result.
    for item in manifest['failed_attempts']:
        old, saved = root/item['original'], root/item['saved']
        if not saved.exists():
            saved.parent.mkdir(parents=True, exist_ok=True); old.rename(saved)
        else: rules.require(not old.exists(), 'Both old and archived attempt exist')
    snapshots = [(state/'session.json', folder/'prior_session.json')]
    snapshots += [(p/'status.json', folder/(p.name+'_prior_status.json')) for p in (root/'workers').glob('shard*')]
    snapshots += [(p, folder/'prior_logs'/p.name) for p in
                  [state/'coordinator.log', *sorted(state.glob('shard*.log'))]]
    for old, saved in snapshots:
        if old.exists():
            rules.require(not saved.exists(), 'Recovery snapshot collision')
            saved.parent.mkdir(parents=True, exist_ok=True)
            old.rename(saved)
    manifest['phase'] = 'active'; rules.write(manifest_file, manifest)
    print('A66R1 REPAIR READY: two 1-forward failures preserved; completed results retained; 30900F remain.', flush=True)
    return manifest


def augment_summary(root, summary):
    manifest = manifest_check(root)
    failed = {k: sum(r['counts'][k] for r in manifest['failed_attempts'])
              for k in ('forward_attempts', 'forwards', 'input_vjp_attempts', 'optimizer_backward_attempts')}
    total = dict(summary['new_cost'])
    for key, source in (('method_forward_attempts', 'forward_attempts'),
                        ('method_forwards_completed', 'forwards'),
                        ('input_vjp_attempts', 'input_vjp_attempts'),
                        ('optimizer_backward_attempts', 'optimizer_backward_attempts')):
        total[key] += failed[source]
    summary.update(interface_repair=dict(manifest='repair_r1/manifest.json', numerical_change=False),
                   archived_failed_cost=failed, new_cost_including_archived_failures=total)
    # Preserve the original failed-run accounting label rather than silently certify it.
    summary['cost_is_lower_bound'] = True
    rules.write(Path(root)/'summary.json', summary)
    return summary
