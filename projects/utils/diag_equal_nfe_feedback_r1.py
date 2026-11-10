"""A101R1: strict 300-NFE repair without changing or overwriting A101."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS)); sys.path.insert(0, str(PROJECTS.parent))

from utils import diag_equal_nfe_feedback as original
from utils import equal_nfe_feedback_r1 as rules
from utils import equal_nfe_feedback_runtime_r1 as repair
from utils import historical_ffhq_gain as files
from utils.diag_paper_clock_alignment import write

CARD = PROJECTS / 'experiment_cards/A101R1_equal_nfe_feedback_20261110.md'
REPAIR_SOURCES = (
    'utils/equal_nfe_feedback_r1.py',
    'utils/equal_nfe_feedback_runtime_r1.py',
    'utils/diag_equal_nfe_feedback_r1.py',
    'utils/a101r1_session.py',
    'utils/test_equal_nfe_feedback_r1.py',
    'run_a101r1_20261110.sh',
    str(CARD.relative_to(PROJECTS)),
)


def check(root):
    root = Path(root); plan = files.read(root / 'plan.json')
    files.require(plan['experiment'] == rules.EXPERIMENT, 'Wrong A101R1 plan')
    for name, sha in plan['sources'].items():
        files.require(files.digest(PROJECTS / name) == sha, 'Source changed: ' + name)
    for name, sha in plan['inputs'].items():
        files.require(files.digest(name) == sha, 'Input changed: ' + name)
    files.require(files.digest(root / 'motion_kernel.npy') == plan['motion_kernel_sha256'],
                  'Motion kernel changed')
    return plan


def prepare(args):
    root = Path(args.root).resolve()
    if (root / 'plan.json').exists():
        check(root); print('A101R1 frozen plan reused.', flush=True); return
    # Reuse A101's already reviewed fixture construction only while the root is
    # empty.  Rewrite the identity before any model attempt occurs.
    original.prepare(args)
    plan = files.read(root / 'plan.json')
    plan['experiment'] = rules.EXPERIMENT
    plan['sources'].update({name: files.digest(PROJECTS / name) for name in REPAIR_SOURCES})
    plan['protocol']['baseline'] = (
        'Paired local ZAPS300: original learned update and historical bicubic '
        'guidance are unchanged; final-unroll RNG only is explicitly shared.')
    plan['protocol']['pairing'] = (
        'Each arm independently executes270 prefix forwards and9 optimization '
        'backwards. zeta/D/tau/initial must independently match at the final '
        'entry; then only CPU/CUDA RNG state is copied from baseline so the30-step '
        'innovations are exactly paired. No learned state, latent, direction or '
        'output is copied.')
    plan['protocol']['diagonal_adjoint'] = (
        'The diagonal quadratic uses the real H-adjoint. SR ZAPS itself keeps '
        'the historical bicubic empirical guidance map; it is not mislabeled H-adjoint.')
    plan['repair_of'] = dict(experiment='A101_equal_nfe_feedback',
        root=original.ROOT,
        reason='A101 independently seeded final RNG drifted and SR diagonal used '
               'legacy bicubic mapping inside an exact quadratic descent gate.',
        reuse='No A101 output or learned state reused; separate root and result branch.')
    plan['code_commit'] = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=PROJECTS.parent, text=True).strip()
    write(root / 'plan.json', plan); shutil.copyfile(CARD, root / 'experiment_card_r1.md')
    check(root)
    print('A101R1 PREPARE/FREEZE: separate root; 360 independent full300-NFE runs; '
          'shared final RNG only, true SR H-adjoint only inside diagonal solve.', flush=True)


def measure(root, plan, case, arm, model, uuid):
    if arm == 'ZAPS_LOCAL_300':
        return original.measure(root, plan, case, arm, model, uuid)
    baseline = original.record_path(root, case, 'ZAPS_LOCAL_300')
    files.require(baseline.exists() and files.read(baseline)['status'] == 'complete',
                  'Complete paired ZAPS baseline required')
    reference = __import__('torch').load(
        baseline.parent / 'entry.pt', map_location='cpu', weights_only=True)
    with repair.paired_final_rng(reference), repair.repaired_diagonal():
        return original.measure(root, plan, case, arm, model, uuid)


def worker(args):
    with patch.object(original, 'check', check), patch.object(original, 'measure', measure):
        original.worker(args)


def summarize(root):
    with patch.object(original, 'check', check):
        summary = original.summarize(root)
    summary['classification'] = 'STRICT_EQUAL_300_NFE_PAIRED_RNG_TRUE_ADJOINT_R1'
    summary['repair_of'] = check(root)['repair_of']
    write(Path(root) / 'summary.json', summary)
    return summary


def report(root):
    original.report(root)


def archive(root):
    from utils import archive_refinement_confirmation as exporter
    exporter.BRANCH = rules.BRANCH
    bundle, commit = exporter.export_and_commit({'A101R1': Path(root)}, projects=PROJECTS)
    print('Archive:', bundle, 'Results commit:', commit, 'Branch:', rules.BRANCH, flush=True)
    subprocess.run(['git', 'push', 'origin', rules.BRANCH], cwd=PROJECTS.parent, check=True)
    print('RESULTS_PUSH_PASS: A101R1 uploaded.', flush=True)


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('--root', default=rules.ROOT)
    value.add_argument('--a71-root', default=original.PARENT)
    value.add_argument('--ffhq-dir', default='/home/lzy/FFHQ/00000/00000')
    value.add_argument('--shard', type=int, choices=(0, 1)); value.add_argument('--device-uuid')
    group = value.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'worker', 'summarize', 'report', 'archive'):
        group.add_argument('--' + name, action='store_true')
    return value


def main():
    parser_value = parser(); args = parser_value.parse_args()
    if args.worker and (args.shard is None or not args.device_uuid):
        parser_value.error('Worker needs shard and device UUID')
    if args.prepare: prepare(args)
    elif args.worker: worker(args)
    elif args.summarize: summarize(args.root)
    elif args.report: report(args.root)
    else: archive(args.root)


if __name__ == '__main__': main()
