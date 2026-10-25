"""Recompute A39 metadata and finite SINGLE-pulse opportunity; no Torch or UNet."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS))
from utils import common_state_risk as risk
from utils import cached_policy_opportunity as opportunity


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def equivalent(a,b,path='root'):
    if isinstance(a,dict):
        if not isinstance(b,dict) or set(a)!=set(b):raise ValueError('Keys differ: '+path)
        for key in a:equivalent(a[key],b[key],path+'/'+key)
    elif isinstance(a,list):
        if not isinstance(b,list) or len(a)!=len(b):raise ValueError('Lengths differ: '+path)
        for i,(x,y) in enumerate(zip(a,b)):equivalent(x,y,path+'/'+str(i))
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        if not math.isfinite(b) or abs(a-b)>1e-10*max(1.,abs(a),abs(b)):
            raise ValueError('Numeric result differs: '+path)
    elif a!=b:raise ValueError('Value differs: '+path)


def audit(folder):
    folder=Path(folder).resolve();manifest=json.loads((folder/'manifest.json').read_text())
    inventory={v['archive_path']:v for v in manifest['files']+manifest['excluded']}
    metadata=[v for v in manifest['files'] if v['archive_path'].startswith('A39/')
              and Path(v['archive_path']).suffix in ('.json','.sha256','.md')]
    for item in metadata:
        if sha(folder/item['archive_path'])!=item['sha256']:raise ValueError('Archive metadata SHA differs')
    root=folder/'A39'
    def signed(path):
        if sha(path)!=path.with_suffix('.sha256').read_text().strip():raise ValueError('Record signature differs')
        return json.loads(path.read_text())
    plan=signed(root/'plan.json');summary=signed(root/'summary.json')
    if plan['experiment']!='A39_common_state_risk' or summary['completed_cases']!=12:
        raise ValueError('Need complete A39')
    workers=plan['worker_identity_files']
    for name,digest in workers.items():
        if '/projects/' not in name:raise ValueError('Unrecognized worker path')
        if sha(PROJECTS/name.split('/projects/',1)[1])!=digest:raise ValueError('Worker differs: '+name)
    records=[]
    for case in plan['cases']:
        paths=list((root/case['key']).glob('attempt_*/run.json'))
        if len(paths)!=1:raise ValueError('Expected one completed attempt; no output selection')
        path=paths[0];record=signed(path)
        artifact=str((path.parent/'outputs.pt').relative_to(folder))
        if (record['case']!=case or record['status']!='complete' or record['nfe']!=75
                or record['method_nfe']!=0 or record['plan_sha256']!=sha(root/'plan.json')
                or record['calls']!=risk.expected_calls(plan['grid']) or not record['gates']
                or not all(v['passed'] for v in record['gates'].values())
                or inventory[artifact]['sha256']!=record['outputs_sha256']):
            raise ValueError('Protocol/gates/artifact inventory differs')
        records.append(record)
    recomputed=risk.analyze(records)
    for key,value in recomputed.items():equivalent(value,summary[key],key)
    if sum(r['nfe'] for r in records)!=summary['diagnostic_nfe']:raise ValueError('Cost differs')
    groups=[];all_pulses=[]
    for sigma in risk.SIGMAS:
        current=sorted([r for r in records if r['case']['sigma']==sigma],key=lambda r:r['case']['image_id'])
        for k,t in risk.POINTS:
            arms=('C',)+tuple(risk.arm(k,g) for g in risk.GAMMAS)
            values=[[[r['metrics'][a][m]-r['metrics']['C'][m] for m in opportunity.METRICS]
                     for a in arms] for r in current]
            result=opportunity.analyze(values,arms)
            result.update(sigma=sigma,source_t=t,image_ids=[r['case']['image_id'] for r in current],
                limits='Post-hoc finite single-pulse library at ONE common C-state position/noise condition. '
                       'GT oracle is a descriptive finite-library bound, not an online method or a bound on '
                       'unmeasured continuous actions/positions/multi-pulse paths. Group and individual guards differ.')
            groups.append(result)
            for i,r in enumerate(current):
                for j,arm in enumerate(arms[1:],1):
                    all_pulses.append(dict(case=r['case']['key'],sigma=sigma,source_t=t,arm=arm,
                                          delta=dict(zip(opportunity.METRICS,values[i][j]))))
    risky=[r for r in all_pulses if r['delta']['loss']<0 and r['delta']['lpips']>.005]
    errors=summary['pooled_prediction']
    return dict(experiment='A39_ARCHIVE_RECOMPUTATION_AND_SINGLE_PULSE_OPPORTUNITY',status='complete',
                method_nfe=0,diagnostic_nfe=0,source_A39_diagnostic_nfe=900,
                audit=dict(metadata_files_verified=len(metadata),worker_files_verified=len(workers),
                           completed_records_verified=len(records),summary_analysis_recomputed=True,
                           numerical_comparison_rtol_atol=1e-10,
                           binary_check='Recorded output SHA matches archive inventory; no tensor loaded/replayed.'),
                pooled_prediction=errors,single_pulse_opportunity=groups,
                pulse_counts=dict(total=len(all_pulses),lpips_risky=sum(r['delta']['lpips']>.005 for r in all_pulses),
                    ssim_risky=sum(r['delta']['ssim']<-.002 for r in all_pulses),
                    physical_better_but_lpips_risky=risky),
                dual_state_online_confirmation_gate_passed=False,
                limits='No new hypothesis test. Six reused image clusters; four separate position/noise cells. '
                       'No independent confirmation, state-controller success, global impossibility or task-transfer claim.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive-dir',required=True)
    p.add_argument('--output',required=True);args=p.parse_args();result=audit(args.archive_dir)
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():equivalent(result,json.loads(destination.read_text()))
    else:destination.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('A39 METADATA AUDIT PASS:',json.dumps(result['audit'],ensure_ascii=False))
    for g in result['single_pulse_opportunity']:
        print('A39 finite single-pulse opportunity:',json.dumps(dict(sigma=g['sigma'],t=g['source_t'],
            fixed=g['best_fixed_feasible']['mean_delta'],headroom=g['oracle_headroom_over_best_fixed'],
            individual_guard_oracle=g['GT_individual_guard_oracle']['mean_delta']),ensure_ascii=False))
    print('UNet NFE=0; output:',destination)


if __name__=='__main__':main()
