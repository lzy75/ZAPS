"""A38b: audit existing full-path policy opportunity; NO Torch/model/new UNet calls."""
import argparse
import json
from pathlib import Path
import sys

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS))
from utils import diag_one_sided_noise_feedback as parent
from utils import cached_policy_opportunity as rules

source=parent.source
ROOT='/data0/lzy/ZAPS-results/diag_cached_policy_opportunity/A38b_20261025'
CARD=PROJECTS/'experiment_cards/A38b_cached_policy_opportunity_20261025.md'


def run(args):
    root,old=Path(args.root).resolve(),Path(args.a38_dir).resolve()
    p=source.check_plan(old);previous=source.load_signed(old/'summary.json')
    for folder in (old,Path(p['source_A36']).resolve(),Path(p['source_A37']).resolve()):
        if root.is_relative_to(folder) or folder.is_relative_to(root):raise ValueError('Output overlaps historical run')
    if p['experiment']!='A38_one_sided_noise_feedback' or previous['completed_cases']!=18:
        raise ValueError('Need all eighteen completed A38 cases')
    identities={str(old/name):source.digest(old/name) for name in ('plan.json','summary.json')}
    records=[]
    for case in p['cases']:
        item=parent.done(old,p,case)
        if item is None:raise ValueError('Incomplete A38: '+case['key'])
        training,feedback,a37=parent.references(p,case)
        for path,_ in (training,feedback,a37,item):identities[str(path)]=source.digest(path)
        metrics={'C':feedback[1]['metrics']['C'],'TIME':feedback[1]['metrics']['TIME'],
                 'OLD':feedback[1]['metrics']['STATE'],**a37[1]['metrics'],**item[1]['metrics']}
        records.append(dict(item[1],metrics=metrics))
    if (len(records)!=18 or len({r['case']['key'] for r in records})!=18
            or parent.interpretation(records)['groups']!=previous['groups']):
        raise ValueError('Source records and A38 summary disagree')
    own=[Path(__file__),Path(rules.__file__),CARD,PROJECTS/'utils/test_cached_policy_opportunity.py']
    plan=dict(experiment='A38b_cached_policy_opportunity',source_A38=str(old),identity_files=identities,
              worker_identity_files={**p['worker_identity_files'],**{str(f):source.digest(f) for f in own}},
              planned_method_nfe=0,planned_diagnostic_nfe=0,
              scope='Post-A38 descriptive finite-library audit. Six reused development image clusters, one seed, '
                    'three dependent noise conditions. Stop current regularization split; no new sampling or hyperparameter scan.')
    if root.exists():
        if source.check_plan(root)!=plan:raise ValueError('Existing audit source/code differs; refuse overwrite')
    else:
        root.parent.mkdir(parents=True,exist_ok=True);root.mkdir()
        source.signed(root/'plan.json',plan);(root/'experiment_card.md').write_text(CARD.read_text())
    groups=[]
    for sigma in parent.base.parent.rules.SIGMAS:
        current=sorted([r for r in records if r['case']['sigma']==sigma],key=lambda r:r['case']['image_id'])
        if len(current)!=6 or len({r['case']['image_id'] for r in current})!=6:
            raise ValueError('Need six image clusters at each noise level')
        deltas=[[[r['metrics'][a][m]-r['metrics']['C'][m] for m in rules.METRICS] for a in rules.ARMS] for r in current]
        outcome=rules.analyze(deltas)
        outcome.update(sigma=sigma,image_ids=[r['case']['image_id'] for r in current],
                       per_image_policy_delta={r['case']['image_id']:{a:dict(zip(rules.METRICS,deltas[i][j]))
                                               for j,a in enumerate(rules.ARMS)} for i,r in enumerate(current)})
        groups.append(outcome)
    result=dict(experiment=plan['experiment'],status='complete',classification='CACHED_FINITE_POLICY_OPPORTUNITY_ONLY',
                completed_cases=18,image_clusters=6,method_nfe=0,diagnostic_nfe=0,
                source_A38_diagnostic_nfe=previous['diagnostic_nfe'],
                source_A38_cost_is_lower_bound=previous['cost_is_lower_bound_if_abruptly_interrupted'],
                source_audit='PASS: signed source records, recorded gates and artifact hashes; all A38 groups recomputed. '
                             'No binary tensor loading or numerical replay.',
                groups=groups,limits=plan['scope']+' GT oracles are not methods or online output selectors. '
                      'No image-state predictability, trajectory-local opportunity or independent efficacy conclusion.')
    source.signed(root/'summary.json',result)
    compact={**result,'groups':[{k:v for k,v in g.items() if k!='per_image_policy_delta'} for g in groups]}
    print('A38b Summary:',json.dumps(compact,ensure_ascii=False),flush=True)
    print('Records:',root,'; UNet NFE=0; no model loaded.',flush=True)
    if args.archive:
        from utils.archive_refinement_confirmation import export_and_commit
        bundle,commit=export_and_commit({'A38':str(old),'A38b':str(root)})
        print('Archive:',bundle,'Results commit:',commit,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=ROOT)
    p.add_argument('--a38-dir',default=parent.ROOT);p.add_argument('--archive',action='store_true')
    run(p.parse_args())


if __name__=='__main__':main()
