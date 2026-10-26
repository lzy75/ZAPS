"""A51: operator-only numerical admission; no UNet, image labels or policy tuning."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
from utils.current_score_geometry import conjugate_gradient

ROOT = Path('/data0/lzy/ZAPS-results/diag_score_cg_metrology/A51_20261026')
PARENT = Path('/data0/lzy/ZAPS-results/diag_current_score_geometry_r2/A50r2_20261026')
CARD = PROJECTS/'experiment_cards/A51_score_cg_metrology_20261026.md'
VARIANTS = ('LEGACY32_128', 'RESTART32_128', 'RESTART32_512', 'RESTART64_512')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_signed(path):
    if sha(path) != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Invalid signed metadata: '+str(path))
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def reliable_cg(action, rhs, *, maxiter=128, rtol=1e-5, refresh=32):
    """Recompute true residual on apparent convergence and every refresh steps.

    A failed true check restarts CG within the SAME iteration budget. Extra true
    checks are counted operator calls; same iteration cap is not equal walltime.
    """
    import torch
    if maxiter < 1 or refresh < 1 or not 0 < rtol < 1 or not torch.isfinite(rhs).all():
        raise ValueError('Invalid solve contract')
    dot = lambda a,b: float((a.double()*b.double()).sum())
    with torch.no_grad():
        x = torch.zeros_like(rhs)
        r = rhs.clone(); p = r.clone(); rr = dot(r,r); scale = math.sqrt(rr)
        calls = checks = restarts = 0
        if scale == 0:
            return x, dict(passed=True, iterations=0, operator_calls=0, relative_residual=0.,
                           true_checks=0, restarts=0, reason='ZERO_RHS')
        for step in range(1, maxiter+1):
            ap = action(p); calls += 1
            pap = dot(p,ap)
            if not math.isfinite(pap) or pap <= 0 or not torch.isfinite(ap).all():
                raise ValueError('Invalid/non-SPD operator')
            alpha = rr/pap
            x.add_(p, alpha=alpha); r.sub_(ap, alpha=alpha)
            new_rr = dot(r,r)
            if math.sqrt(new_rr) <= rtol*scale or step % refresh == 0 or step == maxiter:
                r = rhs-action(x); calls += 1; checks += 1
                new_rr = dot(r,r); relative = math.sqrt(new_rr)/scale
                if relative <= rtol:
                    return x, dict(passed=True, iterations=step, operator_calls=calls,
                                   relative_residual=relative, true_checks=checks, restarts=restarts,
                                   reason='TRUE_RESIDUAL_PASSED')
                p = r.clone(); rr = new_rr; restarts += 1
            else:
                p = r+(new_rr/rr)*p; rr = new_rr
        return x, dict(passed=False, iterations=maxiter, operator_calls=calls,
                       relative_residual=relative, true_checks=checks, restarts=restarts,
                       reason='ITERATION_BUDGET_EXHAUSTED')


def source_spec(parent):
    p = read_signed(parent/'plan.json')
    if p['experiment'] != 'A50r2_current_score_geometry':
        raise ValueError('Need completed A50r2 parent')
    files = {str(parent/'plan.json'): sha(parent/'plan.json')}
    summary = read_signed(parent/'summary.json')
    if (sha(parent/'summary.json') != '36bb090d3ec32a9f34c92db4e66fa71280262c97b5b74b30b0e4860d5113ce9b' or
        summary['completed_phases'] != dict(generate=20, control=20) or summary['diagnostic_nfe'] != 4872):
        raise ValueError('Need the completed, audited A50r2 summary')
    files[str(parent/'summary.json')] = sha(parent/'summary.json')
    eta = {}
    for task in ('gaussian_deblur', 'motion_deblur'):
        key = 'ffhq_'+task+'_00446_s1000'
        found = sorted((parent/key).glob('generate_*/run.json'))
        if len(found) != 1:
            raise ValueError('Ambiguous/incomplete parent phase')
        r = read_signed(found[0])
        if (r['status'] != 'complete' or r['plan_sha256'] != sha(parent/'plan.json') or
            not r['gates'] or not all(g['passed'] for g in r['gates'].values())):
            raise ValueError('Failed parent admission')
        files[str(found[0])] = sha(found[0])
        eta[task] = []
        for k in (0,24):
            v = r['traces']['ALIGN_Q'][k]
            eta[task].append(dict(step=k,t=v['t'],eta=v['readout']['cg']['eta']))
    kernel = Path(p['source_A41'])/'motion_kernel.pt'
    if p['identity_files'][str(kernel)] != sha(kernel):
        raise ValueError('Motion kernel changed')
    files[str(kernel)] = sha(kernel)
    for f in (Path(__file__), CARD, PROJECTS/'run_a51_20261026.sh',
              PROJECTS/'utils/current_score_geometry.py', PROJECTS/'utils/test_score_cg_metrology.py',
              PROJECTS/'utils/diag_ffhq_four_tasks_unit.py', PROJECTS/'modules/degradations.py'):
        files[str(f)] = sha(f)
    for f in (PROJECTS/'utils/current_score_geometry.py', PROJECTS/'utils/diag_ffhq_four_tasks_unit.py',
              PROJECTS/'modules/degradations.py'):
        if p['worker_identity_files'].get(str(f)) != sha(f):
            raise ValueError('Original solver/operator source changed')
    return dict(experiment='A51_score_cg_metrology', source_A50r2=str(parent), identities=files,
                eta=eta, kernel=str(kernel), shape=[1,3,256,256], rng_seed=510026,
                rhs=['WHITE','IN_OPERATOR_RANGE'], variants=list(VARIANTS), rtol=1e-5,
                refresh=32, method_nfe=0, diagnostic_nfe=0, planned_solves=32,
                scope='Real frozen blur operators/eta with synthetic RHS, not archived denoising residuals. '
                      'Operator metrology only; no reconstruction, live sensor coverage or state benefit claim.')


def summarize(r):
    groups = []
    for task in ('gaussian_deblur','motion_deblur'):
        rows = [v for v in r['rows'] if v['task'] == task]
        groups.append(dict(task=task, variants={name:dict(
            passed=sum(v['gate']['passed'] for v in rows if v['variant']==name), planned=4,
            operator_calls=sum(v['gate']['operator_calls'] for v in rows if v['variant']==name),
            seconds=sum(v['seconds'] for v in rows if v['variant']==name),
            max_relative_residual=max((v['gate']['relative_residual'] for v in rows if v['variant']==name),default=None))
            for name in VARIANTS}))
    return dict(experiment='A51_score_cg_metrology', classification='OPERATOR_METROLOGY_ONLY',
                status=r['status'], completed_solves=len(r['rows']), planned_solves=32,
                method_nfe=0, diagnostic_nfe=0, groups=groups,
                limits=r['spec']['scope']+' Same128 is an iteration cap, not equal operator calls. '
                       '512 is explicitly higher cost; double also changes operator arithmetic precision. '
                       'Failures remain failures at the original 1e-5 true-residual criterion.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.root.resolve().is_relative_to(args.parent.resolve()) or args.parent.resolve().is_relative_to(args.root.resolve()):
        raise ValueError('Source/output overlap')
    spec = source_spec(args.parent)
    if args.root.exists():
        r = json.loads((args.root/'run.json').read_text())
        if r['spec'] != spec or r['status'] != 'complete':
            raise ValueError('Frozen inputs differ or interrupted attempt retained; no automatic rerun')
        print('A51 Summary:',json.dumps(summarize(r)),flush=True); return
    import torch
    from modules.degradations import GaussianBlurOperator
    from utils.diag_ffhq_four_tasks_unit import motion_operator
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA unavailable')
    args.root.mkdir(parents=True)
    r = dict(spec=spec,status='running',rows=[],device=args.device,torch=str(torch.__version__),
             cuda=torch.version.cuda,method_nfe=0,diagnostic_nfe=0)
    save(args.root/'run.json',r)
    flags = (torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        generator=torch.Generator(device='cpu').manual_seed(spec['rng_seed'])
        draw=torch.randn(spec['shape'],generator=generator,dtype=torch.float32).to(args.device)
        kernel=torch.load(spec['kernel'],map_location='cpu',weights_only=True).numpy()
        for task in ('gaussian_deblur','motion_deblur'):
            base=(GaussianBlurOperator(61,3.,0.).to(args.device) if task=='gaussian_deblur'
                  else motion_operator(kernel,args.device))
            double=copy.deepcopy(base).double()
            for state in spec['eta'][task]:
                eta=state['eta']
                for kind in spec['rhs']:
                    rhs=draw.clone() if kind=='WHITE' else (.5*base.H(draw)).detach()
                    for name in VARIANTS:
                        op=double if name.startswith('RESTART64') else base
                        b=rhs.double() if name.startswith('RESTART64') else rhs
                        action=lambda v,op=op: .25*op.H(op.transpose(v))+eta*v
                        # The SAME eta and promoted float32 kernel coefficients define the double operator.
                        if args.device.startswith('cuda'):torch.cuda.synchronize()
                        start=time.perf_counter()
                        if name.startswith('LEGACY'):
                            _,gate=conjugate_gradient(action,b,rtol=1e-5,maxiter=128)
                        else:
                            _,gate=reliable_cg(action,b,rtol=1e-5,maxiter=512 if name.endswith('512') else 128)
                        if args.device.startswith('cuda'):torch.cuda.synchronize()
                        r['rows'].append(dict(task=task,**state,rhs=kind,variant=name,gate=gate,
                                              seconds=time.perf_counter()-start))
                        save(args.root/'run.json',r)
                        print('A51 progress:',len(r['rows']),'/32 solves;',task,state['t'],kind,name,
                              'passed=',gate['passed'],flush=True)
        r['status']='complete'
    except BaseException as error:
        r.update(status='failed',error=repr(error)); raise
    finally:
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
        save(args.root/'run.json',r)
    report=summarize(r);save(args.root/'summary.json',report)
    print('A51 Summary:',json.dumps(report),flush=True)
    print('Records:',args.root,flush=True)


if __name__=='__main__':
    main()
