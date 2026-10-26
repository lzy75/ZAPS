"""A52 cached metadata audit, no Torch/model/tensor replay."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean,median
from utils.audit_current_score_archive import verify_archive,digest,read,require,near


def audit(root):
    verification=verify_archive(root)
    p,s,e=(read(root/n) for n in ('plan.json','summary.json','evaluation.json'))
    require(p['experiment']=='A52_feedback_vector_constraint' and len(e['rows'])==4,'Need complete A52')
    results=[];contrasts=0;forwards=backwards=0;discard_checks=0
    for g in s['groups']:
        rows=[r for r in e['rows'] if r['case']['task']==g['task']]
        require(len(rows)==2,'Wrong image count')
        for name,refs in g['mean_deltas'].items():
            for ref,ms in refs.items():
                for m,v in ms.items():
                    near(mean(r['metrics'][name][m]-r['metrics'][ref][m] for r in rows),v);contrasts+=1
        regional={}
        if g['task']=='inpainting':
            for name in ('C_IEEE','FULL_PROX','FULL_ALIGN'):
                regional[name]={key:mean(r['inpainting_regions'][name][key] for r in rows)
                                for key in ('observed_unit_MSE','unobserved_unit_MSE')}
        traces=[]
        for row in rows:
            case=row['case'];f=root/case['key']/'generate_000/run.json';r=read(f)
            require(r['case']==case and r['status']=='complete' and r['gates'] and all(v['passed'] for v in r['gates'].values()),'Failed recorded gate')
            require(r['plan_sha256']==digest(root/'plan.json') and r['calls']==list(reversed(p['grid']))*4,'Source/calls differ')
            require(r['nfe']==len(r['calls'])==120 and r['backward_attempts']==r['backward_completed']==60,'Cost differs')
            forwards+=r['nfe'];backwards+=r['backward_completed']
            resolved={name:sum(v['readout'].get('resolved',False) for v in trace) for name,trace in r['traces'].items()}
            require(all(v==30 for v in resolved.values()),'Unresolved source state')
            for record in r['traces']['FULL_ALIGN']:
                z=record['readout'];c=max(0.,z['cosine_q_target'])
                expected=math.sqrt(max(0.,1-c*c))
                require(abs(z['discarded_fraction']-expected)<2e-6,'Discarded norm fraction differs');discard_checks+=1
            traces.append(dict(image_id=case['image_id'],recorded_gates=r['gates'],resolved=resolved,
                median_discarded_norm_fraction=median(v['readout']['discarded_fraction'] for v in r['traces']['FULL_ALIGN']),
                median_current_cosine=median(v['readout']['cosine_v_u'] for v in r['traces']['FULL_ALIGN'])))
        results.append(dict(task=g['task'],mean_psnr={name:mean(r['metrics'][name]['psnr'] for r in rows)
            for name in ('C_IEEE','ALIGN_Q','FULL_PROX','FULL_ALIGN')},FULL_ALIGN_minus=g['mean_deltas']['FULL_ALIGN'],
            regional_MSE=regional,traces=traces))
    require(forwards==s['diagnostic_nfe']==480 and backwards==s['backward_attempts']==240,'Total cost differs')
    return dict(experiment='A52_cached_metadata_audit',verification=verification,source_summary_sha256=digest(root/'summary.json'),
        recomputed_mean_contrasts=contrasts,discarded_norm_checks=discard_checks,read_recorded_gated_cases=4,
        source_forward_nfe=forwards,source_backward_attempts=backwards,new_method_nfe=0,new_diagnostic_nfe=0,groups=results,
        limits='Available archived bytes and scalar metadata only; not excluded PT verification, gate rerun or new GPU reconstruction. '
               'Two reused development image clusters; no independent angle value, transfer, curvature, h-control or efficiency claim.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    require(not args.output.exists(),'Refuse overwriting prior audit')
    value=audit(args.archive);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('A52_CACHED_AUDIT_PASS:',value['recomputed_mean_contrasts'],'mean contrasts;',value['discarded_norm_checks'],'discarded norm identities; new UNet=0.')


if __name__=='__main__':main()
