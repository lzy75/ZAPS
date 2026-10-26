"""A53 archived metadata + optional same-case A44 fixed-reference audit; zero UNet."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean,median
from utils.audit_current_score_archive import verify_archive,digest,read,require,near


def audit(root,a50):
    verification=verify_archive(root);p,s,e=(read(root/n) for n in ('plan.json','summary.json','evaluation.json'))
    require(p['experiment']=='A53_current_cosine_value' and len(e['rows'])==10,'Need completed A53')
    source_root=Path(read(root/'manifest.json')['source']);forward=backward=0;contrasts=0;point_checks=0
    for f in root.glob('*/*_*/run.json'):
        r=read(f);require(r['status']=='complete' and r['gates'] and all(v['passed'] for v in r['gates'].values()),'Failed recorded gate')
        require(r['plan_sha256']==digest(root/'plan.json'),'Plan differs')
        names=['STATE'] if r['phase']=='generate' else ['TIME_COS','SHUFFLE_COS','NORM_ONLY']
        require(r['calls']==list(reversed(p['grid']))*len(names) and r['nfe']==len(r['calls']) and
                r['backward_attempts']==r['backward_completed']==r['nfe'],'Call contract differs')
        require(e['inputs'][str(source_root/f.relative_to(root))]==digest(f),'Evaluation binding differs')
        forward+=r['nfe'];backward+=r['backward_completed']
    for c in p['cases']:
        if c['key'] not in p['reused_generation']:continue
        f=root/'source_snapshot'/(c['key']+'_A52_run.json')
        require(digest(f)==p['identity_files'][p['reused_generation'][c['key']]]==e['inputs'][p['reused_generation'][c['key']]],'Cached STATE identity differs')
    require(forward==s['diagnostic_nfe']==1080 and backward==s['backward_attempts']==1080,'Total cost differs')
    old_verification=verify_archive(a50);old_manifest={x['path']:x for x in read(a50/'manifest.json')['files']}
    groups=[]
    for g in s['groups']:
        rows=[r for r in e['rows'] if r['case']['task']==g['task']];require(len(rows)==5,'Wrong image count')
        for ref,ms in g['STATE_minus'].items():
            for m,v in ms.items():
                near(mean(r['metrics']['STATE'][m]-r['metrics'][ref][m] for r in rows),v);contrasts+=1
        for ref,count in g['positive_images'].items():
            require(sum(r['metrics']['STATE']['psnr']>r['metrics'][ref]['psnr']+1e-9 for r in rows)==count,'Win count differs')
        fixed_rows=[];curves=[]
        for row in rows:
            case=row['case'];f=root/case['key']/'control_000/run.json';r=read(f)
            values=[]
            for v in r['traces']['NORM_ONLY']:
                z=v['readout'];require(z.get('resolved') and z['used_cosine']==1.,'Unresolved/non-unit norm source')
                gain=z['v_RMS']/z['u_RMS'];near(gain,z['used_alignment_scale'])
                require(abs(z['target_RMS']-z['coefficient']*z['v_RMS'])<=2e-6*max(z['target_RMS'],1e-30),'Norm target differs')
                values.append(gain);point_checks+=1
            curves.append(dict(image_id=case['image_id'],gain_min=min(values),gain_median=median(values),gain_max=max(values)))
            rel='source_snapshot/'+case['key']+'_A44_finish.json';fixed=read(a50/rel)
            require(digest(a50/rel)==old_manifest[rel]['sha256'] and fixed['case']==case and fixed['status']=='complete'
                    and all(z['passed'] for z in fixed['gates'].values()),'Fixed reference identity/gates differ')
            for m in ('psnr','ssim','lpips'):
                require(abs(fixed['evaluation']['C']['metrics'][m]-row['metrics']['C_NATIVE'][m])<1e-5,'Reference baseline differs')
            fixed_rows.append(dict(image_id=case['image_id'],gamma=fixed['gamma']['TASK_FIXED'],
                metrics=fixed['evaluation']['TASK_FIXED']['metrics'],
                NORM_minus={m:row['metrics']['NORM_ONLY'][m]-fixed['evaluation']['TASK_FIXED']['metrics'][m] for m in ('psnr','ssim','lpips')}))
        groups.append(dict(task=g['task'],mean_metrics={n:{m:mean(r['metrics'][n][m] for r in rows) for m in ('psnr','ssim','lpips')}
            for n in ('C_IEEE','STATE','NORM_ONLY','TIME_COS','SHUFFLE_COS')},
            NORM_minus={ref:{m:mean(r['metrics']['NORM_ONLY'][m]-r['metrics'][ref][m] for r in rows) for m in ('psnr','ssim','lpips')}
                        for ref in ('C_NATIVE','C_IEEE','STATE','TIME_COS','SHUFFLE_COS')},
            NORM_positive_images={ref:sum(r['metrics']['NORM_ONLY']['psnr']>r['metrics'][ref]['psnr']+1e-9 for r in rows)
                                  for ref in ('C_IEEE','STATE')},
            NORM_minus_A44_TASK_FIXED={m:mean(r['NORM_minus'][m] for r in fixed_rows) for m in ('psnr','ssim','lpips')},
            A44_fixed_rows=fixed_rows,gain_curves=curves,original_current_angle_increment_candidate=g['current_angle_increment_candidate']))
    return dict(experiment='A53_cached_norm_candidate_audit',verification=verification,A50_reference_verification=old_verification,
        source_summary_sha256=digest(root/'summary.json'),recomputed_STATE_mean_contrasts=contrasts,
        norm_target_identity_checks=point_checks,recorded_new_forward_nfe=forward,recorded_new_backward_attempts=backward,
        new_method_nfe=0,new_diagnostic_nfe=0,groups=groups,
        limits='Available archive bytes/scalar arithmetic, not PT/model replay. NORM_ONLY was a predefined secondary control '
               'selected as a candidate after development outcomes. Five reused image clusters, no new confirmation. '
               'A44 TASK_FIXED is a same-case fixed t319 pulse reference, not strongest full-grid fixed gain; extra VJP cost differs.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--a50-archive',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require(not args.output.exists(),'Refuse overwriting prior audit')
    value=audit(args.archive,args.a50_archive);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('A53_CACHED_AUDIT_PASS:',value['recomputed_STATE_mean_contrasts'],'STATE mean contrasts;',value['norm_target_identity_checks'],'norm targets; new UNet=0.')


if __name__=='__main__':main()
