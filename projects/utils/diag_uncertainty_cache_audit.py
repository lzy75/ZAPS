"""A42b cached risk curves and discarded covariance-response columns; ZERO UNet."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS))
from utils import diag_uncertainty_feedback as parent


def curve_report(curve):
    g=np.array(parent.rules.GAMMAS,dtype=np.float64)
    y=np.asarray([curve[str(float(v))] for v in g],dtype=np.float64)
    if not np.isfinite(y).all():raise ValueError('Nonfinite cached curve')
    variation=float(np.ptp(y));scale=max(float(np.max(np.abs(y))),1.)
    best=float(g[int(np.argmin(y))])
    if variation<=1e-12*scale:return dict(bank_argmin=best,status='FLAT',quadratic_vertex=None)
    X=np.column_stack((g*g,g,np.ones(len(g))));a,b,c=np.linalg.lstsq(X,y,rcond=None)[0]
    vertex=float(-b/(2*a)) if a>1e-12*scale else None
    return dict(bank_argmin=best,quadratic_vertex=vertex,
        fit_max_error_over_observed_variation=float(np.max(np.abs(X@np.array([a,b,c])-y))/variation),
        curvature_over_observed_variation=float(abs(a)*(g[-1]-g[0])**2/variation),
        status=('UPPER_BANK_BOUNDARY' if vertex is not None and vertex>g[-1] and best==g[-1] else
                'LOWER_BANK_BOUNDARY' if vertex is not None and vertex<g[0] and best==g[0] else
                'INTERIOR_VERTEX' if vertex is not None and g[0]<=vertex<=g[-1] else 'NONCONVEX_OR_NEAR_LINEAR'),
        vertex_is_unmeasured=True,
        limits='Fit to five observed actions only. Extrapolated vertex is not a recommended or validated multiplier.')


def closure_report(V,matrices,derivatives,alpha):
    V=np.asarray(V,dtype=np.float64);factor=(1-float(alpha))/math.sqrt(float(alpha))
    Y=[factor*np.asarray(d,dtype=np.float64) for d in derivatives]
    if V.shape[1]==0:return dict(rank=0,status='NO_CONTROL_SUBSPACE')
    if len(Y)!=2 or any(v.shape!=V.shape for v in Y):raise ValueError('Cached derivative shape mismatch')
    identities=[float(np.linalg.norm(V.T@y-np.asarray(B)))/max(float(np.linalg.norm(B)),1e-20)
                for y,B in zip(Y,matrices)]
    if max(identities)>1e-8:raise ValueError('Cached full columns disagree with projected matrices')
    if np.linalg.norm(V.T@V-np.eye(V.shape[1]))>1e-8:raise ValueError('Source basis not orthonormal')
    norm=float(np.linalg.norm(Y[0]));orthogonal=Y[0]-V@(V.T@Y[0])
    return dict(rank=V.shape[1],projection_identity_relative_error=max(identities),
        response_column_energy_outside_V=float(np.linalg.norm(orthogonal)**2/max(norm*norm,1e-30)),
        full_column_relative_scale_spread=float(np.linalg.norm(Y[0]-Y[1]))/
            max(float(np.linalg.norm(Y[0])),float(np.linalg.norm(Y[1])),1e-20),
        limits='Proxy CV response columns only; not the missing fraction of the full true covariance or a causal quality claim.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=parent.ROOT)
    parser.add_argument('--archive',action='store_true');args=parser.parse_args()
    import torch
    torch.set_num_threads(2)
    root=Path(args.root).resolve();p=parent.source.check_plan(root);parent.verify(p);rows=[];sources={}
    for case in p['cases']:
        item=parent.done(root,p,case)
        if item is None:raise ValueError('Need all four completed A42 cases; no model will be started')
        path,r=item;sources[str(path)]=parent.source.digest(path)
        sources[str(path.parent/'probes.pt')]=r['artifacts']['probes.pt']
        tensors=torch.load(path.parent/'probes.pt',map_location='cpu',weights_only=True)
        for point in r['points']:
            cache=tensors[str(point['step'])];decision=point['decision']
            curves={'PHYSICAL':curve_report(decision['local_measurement_loss'])}
            curves.update({k:curve_report(v) for k,v in decision['predicted_risk'].items()})
            physical=decision['gamma']['PHYSICAL']
            veto=(physical not in decision['admissible']['FULL']) if 'FULL' in decision['admissible'] else None
            rows.append(dict(task=case['task'],t=point['t'],source_covariance_usable=point['covariance']['usable'],
                source_relative_skew=point['covariance'].get('relative_skew'),raw_center_clip_fraction=point['raw_center_clip_fraction'],
                gamma=decision['gamma'],physical_action_vetoed_by_FULL=veto,curves=curves,
                closure=closure_report(cache['basis'].numpy(),[v.numpy() for v in cache['matrices']],
                    [v.numpy() for v in cache['derivatives']],point['reference_alpha_bar'])))
    if len(rows)!=8:raise ValueError('Need all eight source states')
    result=dict(experiment='A42b_cached_uncertainty_diagnosis',classification='POSTHOC_MECHANISM_DIAGNOSIS_ONLY',
        method_nfe=0,diagnostic_nfe=0,image_clusters=1,rows=rows,source_plan_sha256=parent.source.digest(root/'plan.json'),
        source_records=sources,analysis_source_sha256=parent.source.digest(__file__))
    out=root.parent/'A42b_cached_20261025';out.mkdir(exist_ok=True);file=out/'report.json'
    if file.exists():
        saved=parent.source.load_signed(file)
        if any(saved.get(k)!=result[k] for k in ('source_plan_sha256','source_records','analysis_source_sha256')):
            raise ValueError('Existing diagnosis input/code differs; refuse overwrite')
        result=saved
    else:parent.signed(file,result)
    card=PROJECTS/'experiment_cards/A42b_cached_uncertainty_diagnosis_20261025.md'
    (out/'experiment_card.md').write_text(card.read_text())
    print('A42b Summary:',json.dumps(parent.rules.plain(result),ensure_ascii=False,allow_nan=False),flush=True)
    print('Records:',out,'; NEW UNet NFE=0; no multiplier executed or changed.',flush=True)
    if args.archive:
        from utils import archive_state_indicator_stage
        from utils.recover_a42_archive import commit_bytes,external_objects,ArchiveGitError,disk_full
        def publish(bundle,branch,repo=PROJECTS.parent,message=None):
            try:return commit_bytes(bundle,branch,repo,root.parent/'git_recovery_work')
            except ArchiveGitError as error:
                print(str(error),flush=True)
                if not disk_full(error):raise
                store,objects=external_objects(repo,root.parent/'git_objects')
                prior=os.environ.get('GIT_ALTERNATE_OBJECT_DIRECTORIES')
                alternatives=str(objects)+(os.pathsep+prior if prior else '')
                with patch.dict(os.environ,{'GIT_ALTERNATE_OBJECT_DIRECTORIES':alternatives}):
                    return commit_bytes(bundle,branch,repo,root.parent/'git_recovery_work',store)
        with patch.object(archive_state_indicator_stage,'commit_results',publish):bundle,commit=parent.archive(out)
        print('Archive:',bundle,'Results commit:',commit,'Branch:',parent.BRANCH,flush=True)


if __name__=='__main__':main()
