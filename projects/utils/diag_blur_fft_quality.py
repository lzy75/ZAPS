"""A62 actual blur trajectories after A61, with original-q and physical controls.

Independent new protocol; A58 and A61 remain immutable. CPU FFT solves only
replace the numerical path for the same v. No regularization/weight search.
"""
import argparse
from contextlib import contextmanager
import copy
import gc
import json
import math
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
from utils import diag_blur_pcg_admission as old
from utils import zero_boundary_fft as fft_rules
from utils.a54_session import digest, read, write, exclusive

ROOT = Path('/data0/lzy/ZAPS-results/diag_blur_fft_quality/A62_20261027')
A61 = Path('/data0/lzy/ZAPS-results/diag_blur_fft_admission/A61_20261027')
CARD = PROJECTS/'experiment_cards/A62_blur_fft_quality_20261027.md'
BRANCH = 'results/blur-fft-quality-20261027'
ARMS = ('C_IEEE', 'OLD_U', 'U_FFT', 'V_FFT')
CURRENT = None


def own_sources():
    return {str(p): digest(p) for p in (Path(__file__), Path(fft_rules.__file__), CARD,
        PROJECTS/'utils/a62_session.py', PROJECTS/'utils/test_blur_fft_quality.py',
        PROJECTS/'utils/a56m_no_interrupt_resume.py')}


def admission():
    summary = old.source.load_signed(A61/'summary.json')
    plan = old.source.load_signed(A61/'plan.json')
    old.require(summary['experiment'] == plan['experiment'] == 'A61_blur_fft_cpu_admission' and
        summary['status'] == 'complete' and summary['completed_solves'] == 8 and
        summary['method_nfe'] == summary['diagnostic_nfe'] == 0 and
        len(summary['groups']) == 2 and {g['task'] for g in summary['groups']} == set(old.TASKS), 'Need complete two-task A61')
    for group in summary['groups']:
        old.require(group['admitted'] and group['plan_sha256'] == digest(A61/'plan.json') and
            group['operator_gate']['passed'] and len(group['rows']) == 4 and
            all(r['passed'] and r['original_spatial_relative_residual'] <= 1e-5 for r in group['rows']), 'A61 true residual admission failed')
    return plan


def prepare():
    before = old.check(Path(old.ROOT)); numerical = admission()
    old.require(numerical['parent_plan_sha256'] == digest(Path(old.ROOT)/'plan.json') and
        numerical['metrology_spec'] == before['metrology_spec'], 'A58/A61 source identity differs')
    identities = dict(before['identity_files'])
    for n in ('plan.json', 'summary.json'):
        identities[str(A61/n)] = digest(A61/n)
    p = dict(before)
    for n in ('metrology_solves', 'metrology_seed'):
        p.pop(n, None)
    p.update(experiment='A62_blur_fft_quality', source_A58=str(old.ROOT), source_A61=str(A61),
        identity_files=identities, worker_identity_files={**before['worker_identity_files'], **own_sources()},
        arms=list(ARMS), planned_method_nfe=0, planned_diagnostic_nfe=1200,
        planned_backward_attempts_max=600, cg_rtol=1e-5, hard_iteration_ceiling=2048,
        seconds_per_solve=300, spatial_check_steps=[0,24,29],
        numerical_numpy_version=old.source.load_signed(A61/'summary.json')['groups'][0]['runtime']['numpy'],
        limits='Five OLD FFHQ images x Gaussian/Motion, seed1000/sigma.05. Same PRE,29 innovations and fixed30. '
        'New CPU zero-boundary float64 FFT/condition-budget solve; all FFT true residuals1e-5, independent original spatial checks at0/24/29. '
        'No failed FFT fallback: stop and preserve partial cost. OLD_U may retain its original128-CG fallback, explicitly counted. '
        'Inherited baseline training is reused, not a standalone30-call/deployment-cost proof. No TIME calibration, new images, '
        'noise/second-seed, curvature, four-task confirmation or broad transfer claim.')
    if ROOT.exists():
        found = check()
        old.require(found == p, 'Frozen A62 source/plan differs; no reset')
    else:
        ROOT.parent.mkdir(parents=True, exist_ok=True); ROOT.mkdir()
        old.signed(ROOT/'plan.json', p); shutil.copyfile(CARD, ROOT/'experiment_card.md')
        old.signed(ROOT/'metrology.json', dict(status='complete', admitted=True, source_A61=str(A61),
            source_summary_sha256=digest(A61/'summary.json'), new_nfe=0))
        snap = ROOT/'source_snapshot'; snap.mkdir()
        for f in own_sources(): shutil.copyfile(f, snap/Path(f).name)
    print('A62 PREPARE:5 old images x Gaussian/Motion x4 arms;1200 newF/<=600 inputB; no new learning or tuning.', flush=True)
    return p


def check():
    p = old.source.check_plan(ROOT)
    old.parent.a42.verify(p)
    old.require(p['experiment'] == 'A62_blur_fft_quality' and p['arms'] == list(ARMS), 'Wrong A62 identity')
    admission()
    return p


def item(p, c):
    files = list((ROOT/c['key']).glob('control_*/run.json'))
    if not files: return None
    old.require(len(files) == 1, 'Multiple attempts retained; no selection')
    f = files[0]; r = read(f)
    if r['status'] != 'complete': return None
    r = old.source.load_signed(f)
    old.require(r['case'] == c and r['plan_sha256'] == digest(ROOT/'plan.json') and
        r['metrology_sha256'] == digest(ROOT/'metrology.json') and r['calls'] == list(reversed(p['grid']))*4 and
        r['nfe'] == 120 and r['backward_attempts'] == r['backward_completed'] <= 60 and
        all(g['passed'] for g in r['gates'].values()) and
        all(digest(f.parent/n) == sha for n, sha in r['artifacts'].items()), 'A62 completed contract differs')
    for name in ('U_FFT', 'V_FFT'):
        old.require(len(r['traces'][name]) == 30 and all(v['readout']['resolved'] and v['readout']['cg']['passed']
                    for v in r['traces'][name]), 'Need all30 resolved FFT actions, no fallback')
    return f, r, 'U_FFT'


def response_field(z, base64, y, x, t, s, positions, sigma, mode, backward):
    import numpy as np
    import torch
    base, algebra = old.current.base, old.current.algebra
    inp, raw = base.raw_mean(z.dm, x, t, graph=True)
    field = base.field_from_mean(z, y, raw, t, positions)
    old.require(mode in ('U_FFT', 'V_FFT') and CURRENT is not None, 'Invalid FFT execution binding')
    p, c = CURRENT['plan'], CURRENT['case']
    with torch.no_grad():
        residual = y-z.A.H(raw.detach()); rhs = residual.cpu().double().numpy()
        op = CURRENT.get('fft')
        kernel_bytes = 0
        if op is None:
            kernel = base64.kernel[0,0].detach().cpu().numpy()
            op = fft_rules.ZeroBlur(kernel, rhs.shape); CURRENT['fft'] = op
            kernel_bytes = kernel.nbytes
        kernel = op.kernel
        alpha_t = float(z.dm.alphas_cumprod[t]); a = math.sqrt(alpha_t); b2 = 1-alpha_t
        eta = sigma*sigma*a/b2
        budget = fft_rules.solve_budget(kernel, eta, p['cg_rtol'], p['hard_iteration_ceiling'])
        old.require(budget['maximum_iterations'] is not None, 'Condition budget exceeds fixed ceiling')
        live = dict(case=c['key'], mode=mode, t=t, eta=eta, budget=budget)
        write(ROOT/'active_fft_solve.json', live)
        def progress(value):
            write(ROOT/'active_fft_solve.json', dict(live, progress=value))
        before = list(op.calls); started = time.perf_counter()
        action = lambda v: .25*op.H(op.transpose(v))+eta*v
        w, gate = fft_rules.cg(action, rhs, budget['maximum_iterations'], rtol=p['cg_rtol'],
            seconds_limit=p['seconds_per_solve'], progress=progress)
        gate.update(budget=budget, eta=eta, spatial_check=None)
        # Reject before any new input VJP. A numerical failure is not quietly
        # substituted by q and then interpreted as a quality observation.
        old.require(gate['passed'] and not gate.get('stopped_for_budget',False), 'Actual trajectory FFT true residual failed')
        step = list(positions.nominal).index(t)
        spatial_transfer_bytes = 0
        if step in p['spatial_check_steps']:
            wt = torch.from_numpy(w).to(base64.kernel.device)
            spatial_transfer_bytes = wt.numel()*wt.element_size()
            original = .25*base64.H(base64.transpose(wt))+eta*wt
            r64 = residual.to(device=original.device,dtype=torch.float64)
            relative = float(torch.linalg.vector_norm(r64-original)/torch.linalg.vector_norm(r64))
            gate['spatial_check'] = dict(passed=bool(relative <= p['cg_rtol']), relative_residual=relative,
                linear_calls=1, adjoint_calls=1, device=str(original.device), arithmetic='float64')
            old.require(gate['spatial_check']['passed'], 'Original spatial true residual failed at actual state')
        v = torch.from_numpy((.5*a/b2)*op.transpose(w)).to(device=x.device,dtype=x.dtype)
        old.require(bool(torch.isfinite(v).all()), 'Nonfinite physical vector')
        gate.update(passed=True, seconds_including_validation_and_transfers=time.perf_counter()-started,
            fft_linear_calls=op.calls[0]-before[0], fft_adjoint_calls=op.calls[1]-before[1],
            fft_transforms=2*sum(op.calls[i]-before[i] for i in range(2)),
            CPU_GPU_transfer_bytes=int(residual.numel()*residual.element_size()+v.numel()*v.element_size()+kernel_bytes+spatial_transfer_bytes),
            boundary='original_zero', kernel_coefficients='original_float32_promoted64')
        row = dict(backward_calls=0, cg=gate, R_raw=base.informative_rms(residual)/sigma,
            clip_fraction=float((raw.detach().abs()>1).float().mean()))
        write(ROOT/'active_fft_solve.json',dict(live,status='solve_complete',true_relative_residual=gate['relative_residual']))
    coefficient = algebra.skip_coefficient(float(z.dm.alphas_cumprod[t]),float(z.dm.alphas_cumprod[s]) if s >= 0 else 1.)
    u = None
    if mode == 'U_FFT':
        backward(); row['backward_calls'] = 1; u = algebra.input_vjp(raw,inp,v)
    reading = base.gram_readout(field['q'],v,u,coefficient,float(x.detach().double().square().mean().sqrt()))
    row.update(reading,coefficient=coefficient)
    old.require(reading['resolved'], 'Actual trajectory vector metrology unresolved')
    if mode == 'U_FFT':
        target, extra = old.current.controlled_target(field['q'],v,u,coefficient,reading,1.)
        row.update(extra)
    else:
        target = coefficient*v
        row.update(target_RMS=float(target.double().square().mean().sqrt()),current_Jacobian_used=False)
    return dict(field,q=target.detach()),1.,row


@contextmanager
def bindings(p,c):
    global CURRENT
    old_state = CURRENT; CURRENT = dict(plan=p,case=c)
    try:
        with patch.object(old,'ARMS',ARMS), patch.object(old,'admitted',lambda root:True), \
             patch.object(old,'item',lambda root,plan,case,phase:item(plan,case)), patch.object(old,'pcg_field',response_field):
            yield
    finally: CURRENT = old_state


def worker():
    import os
    import numpy as np
    import torch
    from modules.main_single import load_diffusion_model
    p = check(); device = 'cuda:0'
    old.require(np.__version__ == p['numerical_numpy_version'], 'A61 NumPy arithmetic runtime changed')
    with exclusive(ROOT/'worker.lock'):
        actual = dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
            gpu=torch.cuda.get_device_name(device),NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),
            TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        old.require(actual == p['runtime'] and old.parent.P.precision.precision_state(torch) == p['native_policy'] and
            old.parent.old.parent.model_sources() == p['model_library_sources'], 'Original runtime/model libraries changed')
        torch.set_num_threads(2)
        with old.parent.scan.preserved_rng(): model = load_diffusion_model('ffhq',device); model.model.eval()
        old.require(digest(model.ckpt_path) == p['checkpoint']['sha256'] and all(v.dtype == torch.float32 for v in model.model.parameters()), 'Checkpoint/precision changed')
        for c in p['cases']:
            if item(p,c): continue
            old.require(not list((ROOT/c['key']).glob('control_*')), 'Partial/failed case retained; no automatic retry')
            free,_ = torch.cuda.mem_get_info(device)
            if free+torch.cuda.memory_reserved(device) < 6*1024**3:
                print('A62 RESOURCE LIMITED: below6GiB including own cache; no new case attempt; completed cases retained.',flush=True)
                return 75
            with bindings(p,c): old.sample(ROOT,p,c,'control',model,device)
            print('A62 FULL CASE:',c['key'],flush=True)
            gc.collect(); torch.cuda.empty_cache()
    return 0


def summarize():
    import torch
    from statistics import mean
    from modules.main_single import load_image_as_tensor
    from utils.metrics import compute_all_metrics
    p = check(); pairs = [(c,item(p,c)) for c in p['cases']]
    old.require(all(v for c,v in pairs), 'All10 outputs must be frozen before labels')
    rows = []; records = []
    with old.parent.scan.preserved_rng(), torch.no_grad():
        for c,(f,r,_) in pairs:
            outputs = torch.load(f.parent/'outputs.pt',map_location='cpu',weights_only=True)
            gt = load_image_as_tensor(c['image']).cpu()
            metrics = {n:compute_all_metrics(v,gt,lpips_net='alex') for n,v in outputs.items()}
            old.require(all(math.isfinite(float(v)) for m in metrics.values() for v in m.values()), 'Nonfinite quality metrics')
            rows.append(dict(case=c,metrics=metrics,resolved={n:sum(q['readout'].get('resolved',False) for q in r['traces'][n]) for n in ARMS}))
            records.append(r)
    groups = []
    for task in old.TASKS:
        subset = [r for r in rows if r['case']['task'] == task]
        delta = {ref:{m:mean(r['metrics']['U_FFT'][m]-r['metrics'][ref][m] for r in subset)
            for m in ('psnr','ssim','lpips')} for ref in ('C_IEEE','OLD_U','V_FFT')}
        wins = {ref:sum(r['metrics']['U_FFT']['psnr']>r['metrics'][ref]['psnr']+1e-9 for r in subset) for ref in delta}
        q=delta['C_IEEE']; candidate=q['psnr']>=.05 and q['ssim']>=-.002 and q['lpips']<=.005 and wins['C_IEEE']>=4
        groups.append(dict(task=task,U_FFT_minus=delta,images_positive=wins,quality_vs_C_development_candidate=bool(candidate)))
    totals = {n:dict(FFT_linear_calls=sum(q['readout']['cg']['fft_linear_calls'] for r in records for q in r['traces'][n]),
        FFT_adjoint_calls=sum(q['readout']['cg']['fft_adjoint_calls'] for r in records for q in r['traces'][n]),
        FFT_transforms=sum(q['readout']['cg']['fft_transforms'] for r in records for q in r['traces'][n])+(20 if n == 'U_FFT' else 0),
        original_spatial_checks=sum(q['readout']['cg']['spatial_check'] is not None for r in records for q in r['traces'][n]),
        transfer_bytes=sum(q['readout']['cg']['CPU_GPU_transfer_bytes'] for r in records for q in r['traces'][n])) for n in ('U_FFT','V_FFT')}
    result = dict(experiment=p['experiment'],classification='BLUR_FFT_DEVELOPMENT_QUALITY_ONLY',completed_cases=10,planned_cases=10,
        image_clusters=5,method_nfe=0,diagnostic_nfe=sum(r['nfe'] for r in records),
        input_backward_attempts=sum(r['backward_attempts'] for r in records),groups=groups,per_image=rows,
        quality_solve_actions=totals,measured_seconds_by_arm={n:sum(r['arm_seconds'][n] for r in records) for n in ARMS},limits=p['limits'])
    result['operator_accounting_scope']='FFT target actions and spatial crosschecks only; original q construction/measurement/OLD_U CG remain in full traces and total arm walltime, not an all-operator-action total.'
    old.signed(ROOT/'summary.json',result); print('A62 Summary:',json.dumps(result,ensure_ascii=False),flush=True)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.prepare: prepare()
    elif args.worker: sys.exit(worker())
    else: summarize()
