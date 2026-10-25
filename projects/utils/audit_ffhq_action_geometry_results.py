"""A41 cached decision audit and archive only: no UNet or sampler invocation."""
import argparse
import json
from pathlib import Path
import sys

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS))
from utils import diag_ffhq_action_geometry as original


def task_review(task):
    rows=[]
    for mode,data in task['validation'].items():
        for pi,stage,opportunity in zip(data['policies']['PI'],data['policies']['STAGE'],task['opportunity']):
            if not pi['step']==stage['step']==opportunity['step']:raise ValueError('Mismatched positions')
            choices=pi['choices'];gammas=sorted({c['gamma'] for c in choices})
            fixed=opportunity['best_fixed_gamma'];fixed_delta=opportunity['best_fixed_delta']
            by_image={c['image_id']:c for c in stage['choices']}
            if len(choices)!=6 or len(by_image)!=6:raise ValueError('Need six distinct image choices')
            uniform_best=len(gammas)==1 and gammas[0]==fixed
            delta=pi['mean_delta'];relative_stage={m:delta[m]-stage['mean_delta'][m] for m in original.rules.METRICS}
            rows.append(dict(mode=mode,step=pi['step'],t=pi['t'],PI_delta_vs_C=delta,
                STAGE_delta_vs_C=stage['mean_delta'],PI_minus_STAGE=relative_stage,
                PI_minus_retrospective_best_fixed={m:delta[m]-fixed_delta[m] for m in original.rules.METRICS},
                fixed_gamma=fixed,fixed_delta_vs_C=fixed_delta,
                fixed_quality_guard=fixed_delta['ssim']>=-.002 and fixed_delta['lpips']<=.005,
                PI_quality_guard=delta['ssim']>=-.002 and delta['lpips']<=.005,
                action_counts={str(g):sum(c['gamma']==g for c in choices) for g in gammas},
                actions_equal_STAGE=sum(c['gamma']==by_image[c['image_id']]['gamma'] for c in choices),
                images_PSNR_better_than_STAGE=sum(c['delta']['psnr']>by_image[c['image_id']]['delta']['psnr']+1e-9 for c in choices),
                interpretation='UNIFORM_TASK_ACTION' if uniform_best else
                    ('IMAGE_VARYING_DECISIONS' if len(gammas)>1 else 'UNIFORM_OTHER_ACTION'),
                source_increment_candidate=next(v['candidate'] for v in data['increments'] if v['step']==pi['step'])))
    return dict(task=task['task'],positions=rows,
        limits='An action varying by image is not automatically useful. Fixed comparator/oracle are retrospective '
               'diagnostics; four positions are separate one-pulse reconstructions. No combined path or online policy.')


def run(args):
    source=original.source;root=Path(args.root).resolve();p=source.check_plan(root);original.verify(p)
    compat=root/'serialization_compat_v1.json'
    if compat.exists():
        bound=source.load_signed(compat)
        if bound['plan_sha256']!=source.digest(root/'plan.json'):raise ValueError('Compatibility plan mismatch')
        for path,sha in bound['files'].items():
            if source.digest(path)!=sha:raise ValueError('Compatibility source changed: '+path)
    items=[]
    for case in p['cases']:
        item=original.done(root,p,case)
        if item is None:raise ValueError('A41 incomplete: '+case['key']+'; no model launched')
        trainroot,trainplan=original.training_context(root,p,case)
        if original.parent.done(trainroot,trainplan,case,'BASELINE') is None:raise ValueError('Missing paired original C')
        items.append(item)
    if len(items)!=24:raise ValueError('Need 24 complete cases')
    saved=source.load_signed(root/'summary.json');recomputed=original.rules.analyze([r for _,r in items])
    if any(saved.get(k)!=v for k,v in recomputed.items()):raise ValueError('Saved analysis differs from all24 recomputed cases')
    attempts=[source.read(f) for f in root.glob('*/*/attempt_*/run.json')]
    costs=dict(method_nfe=sum(r['nfe'] for r in attempts if r['kind']=='BASELINE'),
               diagnostic_nfe=sum(r['nfe'] for r in attempts if r['kind']=='RESPONSE'),
               partial_attempts=sum(r['status']!='complete' for r in attempts))
    if any(saved[k]!=v for k,v in costs.items()):raise ValueError('Recorded cost differs from attempts')
    for path,r in items:
        if 'serialization_compat_sha256' in r and r['serialization_compat_sha256']!=source.digest(compat):
            raise ValueError('Run compatibility lineage differs: '+str(path))
    result=dict(experiment='A41_cached_decision_review',status='complete',new_method_nfe=0,new_diagnostic_nfe=0,
        source_plan_sha256=source.digest(root/'plan.json'),source_summary_sha256=source.digest(root/'summary.json'),
        audit_source_sha256=source.digest(__file__),source_cost=costs,
        cost_is_lower_bound_if_abruptly_interrupted=saved['cost_is_lower_bound_if_abruptly_interrupted'],
        tasks=[task_review(t) for t in recomputed['tasks']],
        absolute_C={task:{m:sum(r['metrics']['C'][m] for _,r in items if r['case']['task']==task)/6
                          for m in original.rules.METRICS} for task in original.rules.TASKS},
        source_records={str(path):source.digest(path) for path,_ in items})
    target=root/'cached_review';target.mkdir(exist_ok=True);source.signed(target/'review.json',result)
    print('A41 AUDIT PASS:24 completed cases, original C lineage, outputs, probes/budgets, all analysis recomputed. NEW UNet NFE=0.',flush=True)
    print('A41 recorded costs:',json.dumps(costs),'lower_bound=',result['cost_is_lower_bound_if_abruptly_interrupted'],flush=True)
    for task in result['tasks']:
        print('A41 BASELINE:',task['task'],json.dumps(result['absolute_C'][task['task']]),flush=True)
        for row in task['positions']:
            print('A41 DECISION:',task['task'],json.dumps(row,ensure_ascii=False),flush=True)
    if args.archive:
        for path,r in items:original.pictures(path,r)
        bundle,commit=original.archive(root)
        print('Archive:',bundle,'Results commit:',commit,'Branch:',original.BRANCH,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=original.ROOT)
    parser.add_argument('--archive',action='store_true');run(parser.parse_args())


if __name__=='__main__':main()
