"""A66 continuation rules; historical gate is retained, never relabelled passed."""
from utils import historical_ffhq_gain as parent
from utils.historical_ffhq_gain import read,write,digest,require

EXPERIMENT='A66_current_strong_anchor_continuation'
ROOT='/data0/lzy/ZAPS-results/diag_current_strong_anchor/A66_20261028'
BRANCH='results/current-strong-anchor-method-gain-20261028'


def queue():return parent.queue()[1:]


def budget():
    return dict(new_method_forwards=10*sum(parent.NFE[a] for a,r in queue()),
        new_method_input_vjp_max=1800,new_optimizer_backwards=10*sum(parent.UPDATES[a] for a,r in queue()),
        inherited_method_forwards=3000,inherited_optimizer_backwards=100,
        new_initial_warmup_forwards=2,new_initial_warmup_input_vjp=2,new_null_forwards=0,
        inherited_diagnostic_forwards=32,inherited_warmup_input_vjp=2,
        extra_resource_reload_warmups='All additional loads separately recorded')


def source_contract(plan,summary,runs,null):
    require(plan['experiment']==parent.EXPERIMENT,'Wrong source experiment')
    require(len(runs)==len(plan['cases'])==10,'Need all10 completed native references')
    require(summary['completed_runs']==10 and summary['costs']['method_forwards_completed']==3000,
            'Source already has other attempts; inspect instead of duplicating')
    require(null['status']=='complete' and null['passed'] and null['forwards']==30,'Need valid source null gate')
    require(len({r['case']['image_id'] for r in runs})==10,'Duplicate source cases')
    for case,r in zip(plan['cases'],runs):
        require(r['status']=='complete' and r['case']==case and (r['arm'],r['repeat'])==('HISTORICAL_NATIVE',0),
                'Only native repeat0 can be inherited')
        require(r['counts']['forwards']==r['counts']['forward_attempts']==300 and
                r['counts']['input_vjp_attempts']==0,'Source cost contract differs')
    return dict(original_historical_gate_passed=summary['replay_gate']['passed'],
        current_anchor_quality=next(g['quality'] for g in summary['groups'] if g['arm']=='HISTORICAL_NATIVE'),
        acceptance='User accepts historical numerical drift; compare methods on unchanged CURRENT strong protocol',
        source_records_relabelled=False,algorithm_or_tolerances_changed=False)
