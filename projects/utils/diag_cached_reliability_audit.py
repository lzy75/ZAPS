"""A49: local-vs-terminal disagreement from completed A48; ZERO new UNet."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import diag_response_reliability as parent

ROOT='/data0/lzy/ZAPS-results/diag_response_reliability/A49_cached_20261026'
CARD=PROJECTS/'experiment_cards/A49_cached_objective_audit_20261026.md'
BRANCH='results/ffhq-response-reliability-audit-20261026'


def analyse(records,weights):
    rows=[]
    for r in records:
        weight=weights[r['case']['task']];values={float(g):v for g,v in r['query_readouts'].items()}
        legacy=r['gamma']['LEGACY'];choice=parent.rules.choose(values,legacy,weight)
        if choice!=r['decision']:raise ValueError('Label-free choice replay differs: '+r['case']['key'])
        norm=choice['normalization'];local=r['gamma']['LOCAL'];trust=r['gamma']['TRUST']
        if any(r['gamma'][name]!=g for name,g in choice['gamma'].items()):raise ValueError('Signed action differs')
        objective=lambda g:parent.rules.objective(values[g],norm,weight) if norm is not None else None
        jl,jold=objective(local),objective(legacy)
        dj=jl-jold if jl is not None and jold is not None else None
        d={m:r['evaluation']['LOCAL']['metrics'][m]-r['evaluation']['LEGACY']['metrics'][m] for m in parent.rules.METRICS}
        trial=choice['readouts'];selected=next((v for v in trial if v['gamma']==trust),{})
        rows.append(dict(case=r['case'],gamma=r['gamma'],local_J_minus_legacy=dj,
            local_next_loss_minus_legacy=values[local]['loss']-values[legacy]['loss'],
            LOCAL_minus_LEGACY=d,changed_local_vs_legacy=local!=legacy,changed_trust_vs_local=trust!=local,
            local_J_improved_final_PSNR_worse=bool(dj is not None and dj< -1e-10 and d['psnr']< -1e-9),
            rho_trial_count=sum(v['rho'] is not None for v in trial),
            reliable_trial_count=sum(v['reliable'] for v in trial),
            feasible_but_unreliable_count=sum(v['physically_feasible'] and not v['reliable'] for v in trial),
            selected_rho=selected.get('rho'),trials=trial,geometry=r['geometry']))
    groups=[]
    for task in weights:
        selected=[r for r in rows if r['case']['task']==task]
        if not selected:continue
        rr=[r['selected_rho'] for r in selected if r['selected_rho'] is not None]
        groups.append(dict(task=task,cases=len(selected),
            local_changed_from_legacy=sum(r['changed_local_vs_legacy'] for r in selected),
            trust_changed_from_local=sum(r['changed_trust_vs_local'] for r in selected),
            local_J_improved_final_PSNR_worse=sum(r['local_J_improved_final_PSNR_worse'] for r in selected),
            mean_LOCAL_minus_LEGACY={m:float(np.mean([r['LOCAL_minus_LEGACY'][m] for r in selected])) for m in parent.rules.METRICS},
            feasible_but_unreliable_trials=sum(r['feasible_but_unreliable_count'] for r in selected),
            selected_rho_range=[min(rr),max(rr)] if rr else None))
    return dict(groups=groups,rows=rows)


def run(args):
    previous,root=Path(args.a48_dir).resolve(),Path(args.root).resolve()
    if root.is_relative_to(previous) or previous.is_relative_to(root):raise ValueError('Audit/source overlap')
    p=parent.source.check_plan(previous);parent.a42.verify(p)
    items=[parent.done(previous,p,c) for c in p['cases']]
    if len(items)!=20 or not all(items):raise ValueError('Need all20 completed A48; no rerun will start')
    original=parent.source.load_signed(previous/'summary.json')
    records=[r for _,r in items]
    if parent.rules.summarize(records)!=original['groups']:raise ValueError('A48 summary replay differs')
    analysis=analyse(records,p['weights_frozen'])
    refs={str(previous/'plan.json'):parent.source.digest(previous/'plan.json'),
        str(previous/'summary.json'):parent.source.digest(previous/'summary.json')}
    for path,r in items:
        refs[str(path)]=parent.source.digest(path)
        for name,sha in r['artifacts'].items():refs[str(path.parent/name)]=sha
    own=[Path(__file__),CARD,PROJECTS/'utils/test_cached_reliability_audit.py',PROJECTS/'run_a49_20261026.sh']
    plan=dict(experiment='A49_cached_objective_audit',source_A48=str(previous),identity_files=refs,
        worker_identity_files={**p['worker_identity_files'],**{str(f):parent.source.digest(f) for f in own}})
    if root.exists():
        saved=parent.source.check_plan(root)
        if saved!=plan:raise ValueError('Frozen audit identity differs')
    else:
        root.mkdir();parent.signed(root/'plan.json',plan);shutil.copyfile(CARD,root/'experiment_card.md')
        snap=root/'source_snapshot';snap.mkdir()
        for name in ('plan.json','summary.json'):
            shutil.copyfile(previous/name,snap/('A48_'+name))
            shutil.copyfile((previous/name).with_suffix('.sha256'),(snap/('A48_'+name)).with_suffix('.sha256'))
        for path,r in items:
            out=snap/r['case']['key'];out.mkdir()
            for name in ('run.json','run.sha256','decision_before_labels.json','decision_before_labels.sha256','policy_comparison.png'):
                if (path.parent/name).exists():shutil.copyfile(path.parent/name,out/name)
    report=dict(experiment=plan['experiment'],classification='CACHED_MECHANISM_AUDIT_ONLY',method_nfe=0,diagnostic_nfe=0,
        source_A48_diagnostic_nfe=original['diagnostic_nfe'],completed_cases=20,image_clusters=5,
        source_plan_sha256=parent.source.digest(previous/'plan.json'),**analysis,
        limits='Same five reused development images, existing paired interventions. No new policy or GPU replay; '
        'rho non-intervention is not evidence that every possible reliability controller fails. '
        'Surrogate/final disagreement does not establish a new risk estimator or generalize to all tasks.')
    file=root/'report.json'
    if file.exists():
        if parent.source.load_signed(file)!=parent.sensor.plain(report):raise ValueError('Frozen audit report differs')
    else:parent.signed(file,report)
    compact={k:v for k,v in report.items() if k!='rows'}
    print('A49 Summary:',json.dumps(parent.sensor.plain(compact),ensure_ascii=False,allow_nan=False),flush=True)
    print('Records:',root,'; NEW UNet NFE=0; original A48 untouched.',flush=True)
    if args.archive:
        from unittest.mock import patch
        with patch.object(parent.parent.parent,'BRANCH',BRANCH):bundle,commit=parent.parent.parent.archive(root)
        print('Archive:',bundle,'Results commit:',commit,'Branch:',BRANCH,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=ROOT)
    parser.add_argument('--a48-dir',default=parent.ROOT);parser.add_argument('--archive',action='store_true')
    run(parser.parse_args())


if __name__=='__main__':main()
