"""A61: two CPU-only original-boundary blur qualifications; zero UNet calls.

This new cost protocol does NOT rewrite A58's failed 128-iteration result and
does NOT start reconstruction. Both task results are required before summary.
"""
import argparse
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
from utils.a54_session import busy, digest, exclusive, read, write
from utils import zero_boundary_fft as rules

ROOT = Path('/data0/lzy/ZAPS-results/diag_blur_fft_admission/A61_20261027')
PARENT = Path('/data0/lzy/ZAPS-results/diag_blur_pcg_admission/A58_20261027')
TASKS = ('gaussian_deblur', 'motion_deblur')
CARD = PROJECTS/'experiment_cards/A61_blur_fft_cpu_admission_20261027.md'
BRANCH = 'results/blur-fft-cpu-admission-20261027'


def require(ok, message):
    if not ok: raise ValueError(message)


def sources():
    return {str(p): digest(p) for p in (Path(__file__), Path(rules.__file__), CARD,
        PROJECTS/'utils/test_zero_boundary_fft.py', PROJECTS/'utils/a56m_no_interrupt_resume.py')}


def signed(path, value):
    write(path, value); path.with_suffix('.sha256').write_text(digest(path)+'\n')


def prepare():
    from utils.diag_score_cg_metrology import read_signed, source_spec
    parent = read_signed(PARENT/'plan.json')
    require(parent['experiment'] == 'A58_blur_pcg_admission', 'Wrong parent')
    spec = source_spec(Path(parent['source_A50r2']))
    require(spec == parent['metrology_spec'], 'Frozen operator specification changed')
    p = dict(experiment='A61_blur_fft_cpu_admission', parent=str(PARENT), parent_plan_sha256=digest(PARENT/'plan.json'),
        metrology_spec=spec, seed=parent['metrology_seed'], own_sources=sources(), tasks=list(TASKS),
        rtol=1e-5, kernel_and_adjoint_relative_tolerance=1e-10, hard_iteration_ceiling=2048,
        seconds_per_solve=300, method_nfe=0, diagnostic_nfe=0, input_VJP=0,
        limits='CPU FFT actions with original zero padding and original float32 kernel coefficients promoted to64. '
        'New explicit condition-derived iteration budget, not equal A58 operator cost. Synthetic WHITE/RANGE only. '
        'No NN, reconstruction, full trajectory admission, curvature or state quality claim.')
    if (ROOT/'plan.json').exists():
        require(read_signed(ROOT/'plan.json') == p, 'A61 frozen plan changed; original records retained')
    else:
        signed(ROOT/'plan.json', p); shutil.copyfile(CARD, ROOT/'experiment_card.md')
        snapshot = ROOT/'source_snapshot'; snapshot.mkdir()
        for path in p['own_sources']: shutil.copyfile(path, snapshot/Path(path).name)
    return p


def worker(task):
    # Torch is mandatory for original-operator validation and the frozen RNG/RHS.
    # CUDA is explicitly hidden by the coordinator and is never initialized.
    import torch
    import numpy as np
    from modules.degradations import GaussianBlurOperator
    from utils.diag_ffhq_four_tasks_unit import motion_operator
    from utils.diag_score_cg_metrology import read_signed
    torch.set_num_threads(2)
    p = read_signed(ROOT/'plan.json'); require(p['own_sources'] == sources(), 'A61 worker source changed')
    path = ROOT/(task+'.json')
    with exclusive(ROOT/(task+'.lock')):
        if path.exists():
            old = read_signed(path)
            require(old['status'] == 'complete' and old['plan_sha256'] == digest(ROOT/'plan.json'), 'Old partial/failure retained; no retry')
            print('A61 reused complete CPU task:', task, flush=True); return 0
        result = dict(task=task, status='running', plan_sha256=digest(ROOT/'plan.json'), rows=[], method_nfe=0, diagnostic_nfe=0)
        signed(path, result)
        try:
            kernel_path = Path(p['metrology_spec']['kernel'])
            require(digest(kernel_path) == p['metrology_spec']['identities'][str(kernel_path)], 'Kernel changed')
            base = (GaussianBlurOperator(61, 3., 0.) if task == 'gaussian_deblur'
                    else motion_operator(torch.load(kernel_path, map_location='cpu', weights_only=True).numpy(), 'cpu'))
            original = copy.deepcopy(base).double()
            kernel = original.kernel[0, 0].numpy().copy()
            g = torch.Generator(device='cpu').manual_seed(p['seed'])
            draw = torch.randn(p['metrology_spec']['shape'], generator=g, dtype=torch.float32)
            fft = rules.ZeroBlur(kernel, draw.shape)
            # Include edges and asymmetric motion; this is a full-sized original
            # operator comparison, not a circular or interior-only spot check.
            x = draw.double(); y = torch.flip(x, (-2,))
            rel = lambda a, b: float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-300))
            with torch.no_grad():
                fx = fft.H(x.numpy()); aty = fft.transpose(y.numpy())
                h_error = rel(fx, original.H(x).numpy())
                ht_error = rel(aty, original.transpose(y).numpy())
                pairing = abs(float(np.sum(fx*y.numpy())-np.sum(x.numpy()*aty)))
                pairing /= max(float(np.linalg.norm(fx)*np.linalg.norm(y.numpy())), 1e-300)
                require(max(h_error, ht_error, pairing) <= p['kernel_and_adjoint_relative_tolerance'], 'Original boundary/adjoint FFT parity failed')
                result['operator_gate'] = dict(H_relative_error=h_error, HT_relative_error=ht_error,
                    adjoint_pairing_relative_error=pairing, passed=True, boundary='original_zero', device='cpu', dtype='float64')
                result['kernel_sha256'] = __import__('hashlib').sha256(kernel.tobytes()).hexdigest()
                result['runtime'] = dict(torch=str(torch.__version__), numpy=np.__version__, fft_shape=fft.fft_shape)
                result['operator_setup_cost'] = dict(kernel_forward_FFTs=2,
                    parity_original_spatial_H_calls=1, parity_original_spatial_HT_calls=1,
                    RHS_original_float32_H_calls=1,
                    fft_transform_rule='2 per FFT H/HT action, plus 2 kernel setup FFTs')
                # Match historical finite RHS generation, including its float32
                # convolution rounding. The solve itself is explicitly double.
                range_rhs = (.5*base.H(draw)).double().numpy()
                for state in p['metrology_spec']['eta'][task]:
                    eta = state['eta']; budget = rules.solve_budget(kernel, eta, p['rtol'], p['hard_iteration_ceiling'])
                    for kind, rhs in (('WHITE', x.numpy()), ('IN_OPERATOR_RANGE', range_rhs)):
                        row = dict(**state, rhs=kind, budget=budget, status='running')
                        result['rows'].append(row); signed(path, result)
                        maximum = budget['maximum_iterations']
                        if maximum is None:
                            row.update(status='complete', passed=False, reason='CONDITION_BUDGET_EXCEEDS_HARD_CEILING')
                        else:
                            before = list(fft.calls)
                            action = lambda z: .25*fft.H(fft.transpose(z))+eta*z
                            def progress(value):
                                row['progress'] = value; signed(path, result)
                                print('A61 CPU PROGRESS:', task, state['t'], kind, value, flush=True)
                            solution, gate = rules.cg(action, rhs, maximum, rtol=p['rtol'],
                                seconds_limit=p['seconds_per_solve'], progress=progress)
                            # The original spatial operator decides admission,
                            # independently of FFT recursive/final residuals.
                            spatial_started = time.perf_counter(); sol = torch.from_numpy(solution)
                            spatial = .25*original.H(original.transpose(sol))+eta*sol
                            residual = float(torch.linalg.vector_norm(torch.from_numpy(rhs)-spatial)/torch.linalg.vector_norm(torch.from_numpy(rhs)))
                            row.update(status='complete', passed=bool(gate['passed'] and not gate.get('stopped_for_budget', False) and residual <= p['rtol']),
                                fft_gate=gate, original_spatial_relative_residual=residual,
                                original_spatial_seconds=time.perf_counter()-spatial_started,
                                fft_H_calls=fft.calls[0]-before[0], fft_HT_calls=fft.calls[1]-before[1],
                                original_spatial_H_calls=1, original_spatial_HT_calls=1)
                        signed(path, result)
                        print('A61 CPU SOLVE:', task, state['t'], kind, 'passed=', row['passed'],
                              'spatial_residual=', row.get('original_spatial_relative_residual'), flush=True)
            result.update(status='complete', admitted=len(result['rows']) == 4 and all(r['passed'] for r in result['rows']),
                total_fft_H_calls=fft.calls[0], total_fft_HT_calls=fft.calls[1],
                total_fft_transforms=2+2*sum(fft.calls))
        except BaseException as error:
            result.update(status='failed', error=repr(error)); raise
        finally: signed(path, result)
    return 0


def coordinator():
    ROOT.mkdir(parents=True, exist_ok=True)
    with exclusive(ROOT/'session.lock'):
        state = dict(pid=os.getpid(), status='running', gpu_used=False); write(ROOT/'session.json', state)
        try:
            p = prepare()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            subprocess.run([sys.executable, '-m', 'unittest', 'utils.test_zero_boundary_fft', '-v'], cwd=PROJECTS, env=env, check=True)
            with ExitStack() as stack:
                children = []
                for task in TASKS:
                    log = stack.enter_context((ROOT/(task+'.log')).open('ab', buffering=0))
                    children.append(subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--worker', task],
                        cwd=PROJECTS, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT))
                previous = time.monotonic()
                while any(proc.poll() is None for proc in children):
                    time.sleep(1)
                    if time.monotonic()-previous >= 30:
                        print('A61 PROGRESS:', snapshot(), flush=True); previous = time.monotonic()
                if any(proc.returncode != 0 for proc in children):
                    for task, proc in zip(TASKS, children):
                        if proc.returncode:
                            print('A61 FAILED TASK:', task, flush=True)
                            print('\n'.join((ROOT/(task+'.log')).read_text(errors='replace').splitlines()[-18:]), flush=True)
                    raise RuntimeError('CPU task failed; original task logs retained; no automatic retry')
            groups = [read(ROOT/(task+'.json')) for task in TASKS]
            result = dict(experiment=p['experiment'], status='complete', method_nfe=0, diagnostic_nfe=0,
                completed_solves=sum(len(g['rows']) for g in groups), planned_solves=8, groups=groups,
                classification='SYNTHETIC_CPU_NUMERICAL_ADMISSION_ONLY' if all(g['admitted'] for g in groups) else 'NUMERICAL_ADMISSION_FAILED_NO_RECONSTRUCTION',
                limits=p['limits']+' Passing this check still requires actual denoising-residual and end-to-end quality validation.')
            signed(ROOT/'summary.json', result); print('A61 Summary:', json.dumps(result, ensure_ascii=False), flush=True)
            from utils import archive_refinement_confirmation as archive
            with patch.object(archive, 'BRANCH', BRANCH): bundle, commit = archive.export_and_commit(dict(CPU_FFT_ADMISSION=ROOT))
            print('Archive:', bundle, 'Results commit:', commit, 'Branch:', BRANCH, flush=True)
            try:
                subprocess.run(['git', 'push', 'origin', BRANCH], cwd=PROJECTS,
                    env=dict(env, GIT_TERMINAL_PROMPT='0'), timeout=45, check=True)
                print('RESULTS_PUSH_PASS: A61 CPU numerical qualification uploaded.', flush=True)
            except (OSError, subprocess.SubprocessError) as error:
                print('RESULTS_PUSH_PENDING:', repr(error), '; no solve rerun; git push origin '+BRANCH, flush=True)
            state.update(status='complete', exit_code=0)
        except BaseException as error:
            state.update(status='failed', exit_code=1, error=repr(error)); print('A61 STOP:', repr(error), flush=True)
        finally: write(ROOT/'session.json', state)
        return state['exit_code']


def snapshot():
    return {task: ({k: read(ROOT/(task+'.json')).get(k) for k in ('status', 'admitted', 'error')} |
        dict(completed_solves=sum(r['status'] == 'complete' for r in read(ROOT/(task+'.json')).get('rows', []))))
        if (ROOT/(task+'.json')).exists() else None for task in TASKS}


def client(status=False):
    ROOT.mkdir(parents=True, exist_ok=True)
    if status:
        print(dict(active=busy(ROOT/'session.lock'), session=read(ROOT/'session.json') if (ROOT/'session.json').exists() else None,
                   tasks=snapshot()), flush=True); return 0
    with exclusive(ROOT/'client.lock'):
        if not busy(ROOT/'session.lock'):
            old = read(ROOT/'session.json') if (ROOT/'session.json').exists() else {}
            if old.get('status') == 'failed':
                print('A61 previous failure retained; inspect --status and task logs, no automatic retry.'); return 1
            if old.get('status') == 'complete':
                print('A61 Summary:', json.dumps(read(ROOT/'summary.json'), ensure_ascii=False)); return 0
            with (ROOT/'session.log').open('ab', buffering=0) as log:
                proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--coordinator'],
                    cwd=PROJECTS, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            for _ in range(100):
                if busy(ROOT/'session.lock') or proc.poll() is not None: break
                time.sleep(.05)
    print('A61: Gaussian/Motion CPU tasks in parallel; zero NN/GPU; disconnect-safe, same command reconnects.', flush=True)
    from utils.a56m_no_interrupt_resume import follow
    # Reuse log-follow mechanics only; it does not start the ImageNet coordinator.
    with patch('utils.a56m_no_interrupt_resume.BASE', ROOT): return follow(ROOT/'session.log')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', action='store_true'); parser.add_argument('--coordinator', action='store_true')
    parser.add_argument('--worker', choices=TASKS); args = parser.parse_args()
    sys.exit(worker(args.worker) if args.worker else coordinator() if args.coordinator else client(args.status))
