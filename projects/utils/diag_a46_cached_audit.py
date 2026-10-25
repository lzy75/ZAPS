"""A46b cached surrogate agreement and finite-arm oracle diagnosis; ZERO new UNet."""
import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np

PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import diag_frozen_four_task_confirmation as parent


def opportunity(groups):
    result=[]
    for group in groups:
        rows=[]
        for image,arms in group['per_image'].items():
            fixed=arms['MATCHED_FIXED'];feasible={name:value for name,value in arms.items()
                if parent.control.quality_guard(value) and parent.control.quality_guard(
                    {m:value[m]-fixed[m] for m in ('ssim','lpips')})}
            if 'MATCHED_FIXED' not in feasible:raise ValueError('Reference fails per-image C quality guard; report separately')
            winner=max(feasible,key=lambda n:feasible[n]['psnr'])
            rows.append(dict(image_id=image,guarded_oracle_arm=winner,
                guarded_headroom=feasible[winner]['psnr']-fixed['psnr'],
                raw_headroom=max(value['psnr']-fixed['psnr'] for value in arms.values())))
        result.append(dict(task=group['task'],mean_guarded_headroom=float(statistics.mean(r['guarded_headroom'] for r in rows)),
            mean_raw_headroom=float(statistics.mean(r['raw_headroom'] for r in rows)),rows=rows))
    return result


def agreement(record,weight):
    choice=record['choices']['SELECTED'];g=choice['gamma'];fit=choice.get('fit',{})
    result=dict(gamma=g,extrapolation_over_bank_upper=g/2.,fit_usable=fit.get('usable',False),
                fit_relative_error=fit.get('fit_relative_error'),actual_over_predicted=None)
    if g==1. or not fit.get('usable'):return result
    values={float(key):value for key,value in record['query_readouts'].items()}
    if weight==0.:read=lambda x:values[x]['loss']
    elif weight==1.:read=lambda x:values[x]['risks']['STATE']
    else:
        norm=parent.mix.normalization(values);read=lambda x:parent.mix.objective(values[x],norm,weight)
    predicted=float(np.polyval(fit['coefficients'],1.)-np.polyval(fit['coefficients'],g))
    actual=float(read(1.)-read(g))
    result.update(predicted_decrease=predicted,actual_decrease=actual,
        actual_over_predicted=actual/predicted if predicted>1e-12 else None)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=parent.ROOT);args=parser.parse_args()
    root=Path(args.root).resolve();p=parent.source.check_plan(root);parent.a42.verify(p);items=[];rows=[];sources={}
    for case in p['cases']:
        chosen=parent.phase_done(root,p,case,'choose');finished=parent.phase_done(root,p,case,'finish')
        if chosen is None or finished is None:raise ValueError('A46 incomplete; no new model will start')
        for path,_ in (chosen,finished):sources[str(path)]=parent.source.digest(path)
        row=agreement(chosen[1],p['weights_frozen'][case['task']])
        row.update(task=case['task'],image_id=case['image_id'],final_selected_minus_matched={m:
            finished[1]['evaluation']['SELECTED']['delta'][m]-finished[1]['evaluation']['MATCHED_FIXED']['delta'][m]
            for m in parent.rules.METRICS})
        rows.append(row);items.append(finished[1])
    groups=parent.rules.summarize(items);oracle=opportunity(groups)
    result=dict(experiment='A46b_cached_surrogate_opportunity',classification='POSTHOC_DIAGNOSIS_ONLY',method_nfe=0,diagnostic_nfe=0,
        source_plan_sha256=parent.source.digest(root/'plan.json'),source_records=sources,
        analysis_source_sha256=parent.source.digest(__file__),finite_arm_opportunity=oracle,local_agreement=rows,
        limits='GT used only for a retrospective finite-library oracle, never deployed. Per-image quality guards are stricter '
            'than the original group-mean criterion. Not a global action upper bound, causal failure attribution, policy repair or new-image confirmation.')
    out=root.parent/'A46b_cached_20261025';out.mkdir(exist_ok=True);file=out/'report.json'
    if file.exists():
        old=parent.source.load_signed(file)
        if any(old.get(k)!=result[k] for k in ('source_plan_sha256','source_records','analysis_source_sha256')):raise ValueError('Frozen audit differs')
        result=old
    else:parent.signed(file,result)
    print('A46b cached audit:',json.dumps(dict(method_nfe=0,diagnostic_nfe=0,finite_arm_opportunity=result['finite_arm_opportunity'],
        local_agreement_by_task={task:[r['actual_over_predicted'] for r in rows if r['task']==task] for task in parent.rules.WEIGHTS}),allow_nan=False),flush=True)
    print('Records:',out,'; A46 primary result unchanged.',flush=True)


if __name__=='__main__':main()
