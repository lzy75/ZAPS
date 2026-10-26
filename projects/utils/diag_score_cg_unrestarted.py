"""A51b: missing unrestarted-CG controls; same operators/RHS/tolerance/budget ceiling.

No new denoising, posterior or reconstruction result. Historical A51 is read only.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS))
from utils import diag_score_cg_metrology as old
from utils.current_score_geometry import conjugate_gradient

ROOT=Path('/data0/lzy/ZAPS-results/diag_score_cg_unrestarted/A51b_20261026')
PARENT=old.ROOT
CARD=PROJECTS/'experiment_cards/A51b_unrestarted_cg_20261026.md'
VARIANTS=('CG32_128_REPLAY','CG32_512','CG64_128','CG64_512')


def require(test,message):
    if not test:raise ValueError(message)


def key(row):
    return row['task'],row['step'],row['rhs'],row['variant']


def inspect(parent):
    path=parent/'run.json'
    r=json.loads(path.read_text())
    require(r['status']=='complete' and len(r['rows'])==32,'Need completed A51, all 32 solves')
    require(r['spec']==old.source_spec(Path(r['spec']['source_A50r2'])),'Frozen A51 inputs changed')
    require(r['method_nfe']==r['diagnostic_nfe']==0,'Parent is not operator-only')
    expected={(task,state['step'],rhs,variant) for task,states in r['spec']['eta'].items()
              for state in states for rhs in r['spec']['rhs'] for variant in old.VARIANTS}
    require(len({key(v) for v in r['rows']})==32 and {key(v) for v in r['rows']}==expected,
            'Missing or duplicate original solves')
    require(json.loads((parent/'summary.json').read_text())==old.summarize(r),'A51 summary mismatch')
    for row in r['rows']:
        g=row['gate']
        require(math.isfinite(g['relative_residual']) and g['relative_residual']>=0,'Invalid parent residual')
        require(not g['passed'] or g['relative_residual']<=1e-5,'Unjustified parent pass')
    return r


def specification(parent,r):
    identities=dict(r['spec']['identities'])
    for path in (parent/'run.json',parent/'summary.json',Path(__file__),CARD,
                 PROJECTS/'run_a51b_20261026.sh',PROJECTS/'utils/test_score_cg_unrestarted.py'):
        identities[str(path)]=old.sha(path)
    return dict(experiment='A51b_unrestarted_cg',source_A51=str(parent),identities=identities,
                parent_spec=r['spec'],variants=list(VARIANTS),planned_solves=32,
                replay_residual_absolute_tolerance=1e-6,replay_residual_relative_tolerance=1e-6,
                method_nfe=0,diagnostic_nfe=0,
                scope='Synthetic RHS operator metrology. Reconstructs the A51 protocol; replay matches scalar gates, '
                      'not historical solution vectors (not saved). No denoising-state coverage or quality claim.')


def summarize(r):
    groups=[];points=[]
    for task in ('gaussian_deblur','motion_deblur'):
        selected=[v for v in r['rows'] if v['task']==task]
        groups.append(dict(task=task,variants={name:dict(
            passed=sum(v['gate']['passed'] for v in selected if v['variant']==name),planned=4,
            operator_calls=sum(v['gate']['operator_calls'] for v in selected if v['variant']==name),
            seconds=sum(v['seconds'] for v in selected if v['variant']==name),
            max_relative_residual=max((v['gate']['relative_residual'] for v in selected if v['variant']==name),default=None))
            for name in VARIANTS}))
        for state in r['spec']['parent_spec']['eta'][task]:
            for rhs in r['spec']['parent_spec']['rhs']:
                point=dict(task=task,step=state['step'],t=state['t'],rhs=rhs,results={})
                for row in r['parent_rows']+r['rows']:
                    if (row['task'],row['step'],row['rhs'])==(task,state['step'],rhs):
                        point['results'][row['variant']]=dict(passed=row['gate']['passed'],
                            iterations=row['gate']['iterations'],relative_residual=row['gate']['relative_residual'])
                points.append(point)
    return dict(experiment='A51b_unrestarted_cg',classification='OPERATOR_CONTROL_DIAGNOSTIC_ONLY',
                status=r['status'],completed_solves=len(r['rows']),planned_solves=32,
                method_nfe=0,diagnostic_nfe=0,groups=groups,points=points,
                limits=r['spec']['scope']+' Max512 and true residual1e-5 unchanged. '
                      '512 versus128 is higher cost; historical/new timings are not a controlled benchmark. '
                      'No repair of A50 inpainting failure; no automatic production adoption or budget expansion.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,default=PARENT)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    require(not (args.root.resolve().is_relative_to(args.parent.resolve()) or
                 args.parent.resolve().is_relative_to(args.root.resolve())),'Source/output overlap')
    previous=inspect(args.parent);spec=specification(args.parent,previous)
    if args.root.exists():
        r=json.loads((args.root/'run.json').read_text())
        require(r['spec']==spec and r['status']=='complete','Changed inputs or interrupted attempt; no auto-rerun')
        print('A51b Summary:',json.dumps(summarize(r)),flush=True);return
    import torch
    from modules.degradations import GaussianBlurOperator
    from utils.diag_ffhq_four_tasks_unit import motion_operator
    require(str(torch.__version__)==previous['torch'] and torch.version.cuda==previous['cuda'],'Parent runtime differs')
    if args.device.startswith('cuda'):require(torch.cuda.is_available(),'Requested CUDA unavailable')
    args.root.mkdir(parents=True)
    r=dict(spec=spec,status='running',parent_rows=previous['rows'],rows=[],device=args.device,
           torch=str(torch.__version__),cuda=torch.version.cuda,method_nfe=0,diagnostic_nfe=0)
    old.save(args.root/'run.json',r)
    flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        generator=torch.Generator(device='cpu').manual_seed(previous['spec']['rng_seed'])
        draw=torch.randn(previous['spec']['shape'],generator=generator,dtype=torch.float32).to(args.device)
        kernel=torch.load(previous['spec']['kernel'],map_location='cpu',weights_only=True).numpy()
        parents={key(v):v for v in previous['rows']}
        for task in ('gaussian_deblur','motion_deblur'):
            base=(GaussianBlurOperator(61,3.,0.).to(args.device) if task=='gaussian_deblur'
                  else motion_operator(kernel,args.device));double=copy.deepcopy(base).double()
            for state in previous['spec']['eta'][task]:
                eta=state['eta']
                for kind in previous['spec']['rhs']:
                    rhs=draw.clone() if kind=='WHITE' else (.5*base.H(draw)).detach()
                    for name in VARIANTS:
                        op=double if name.startswith('CG64') else base
                        b=rhs.double() if name.startswith('CG64') else rhs
                        action=lambda v,op=op: .25*op.H(op.transpose(v))+eta*v
                        if args.device.startswith('cuda'):torch.cuda.synchronize()
                        start=time.perf_counter()
                        _,gate=conjugate_gradient(action,b,rtol=1e-5,maxiter=512 if name.endswith('512') else 128)
                        if args.device.startswith('cuda'):torch.cuda.synchronize()
                        row=dict(task=task,**state,rhs=kind,variant=name,gate=gate,seconds=time.perf_counter()-start)
                        r['rows'].append(row)
                        old.save(args.root/'run.json',r)
                        if name=='CG32_128_REPLAY':
                            prior=parents[(task,state['step'],kind,'LEGACY32_128')]['gate']
                            passed=(gate['passed']==prior['passed'] and gate['iterations']==prior['iterations'] and
                                    gate['operator_calls']==prior['operator_calls'] and
                                    math.isclose(gate['relative_residual'],prior['relative_residual'],
                                                 rel_tol=1e-6,abs_tol=1e-6))
                            row['replay_gate']=dict(passed=passed,parent_gate=prior)
                            require(passed,'Original128 scalar replay differs; remaining comparisons stopped')
                        print('A51b progress:',len(r['rows']),'/32 solves;',task,state['t'],kind,name,
                              'passed=',gate['passed'],flush=True)
        r['status']='complete'
    except BaseException as error:
        r.update(status='failed',error=repr(error));raise
    finally:
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
        old.save(args.root/'run.json',r)
    report=summarize(r);old.save(args.root/'summary.json',report)
    print('A51b Summary:',json.dumps(report),flush=True)
    print('Records:',args.root,flush=True)


if __name__=='__main__':main()
