"""A101R2: non-recursive dispatcher repair for the zero-attempt R1 failure."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS)); sys.path.insert(0, str(PROJECTS.parent))

from utils import diag_equal_nfe_feedback_r1 as r1
from utils import equal_nfe_feedback_r2 as rules
from utils import equal_nfe_feedback_runtime_r1 as repair
from utils import historical_ffhq_gain as files
from utils.diag_paper_clock_alignment import write

# Capture before worker monkeypatching.  Calling through r1.original.measure
# inside the patched scope caused the R1 recursion.
_BASE_MEASURE = r1.original.measure
CARD = PROJECTS / 'experiment_cards/A101R2_equal_nfe_feedback_20261110.md'
R2_SOURCES = (
    'utils/equal_nfe_feedback_r2.py', 'utils/diag_equal_nfe_feedback_r2.py',
    'utils/a101r2_session.py', 'utils/test_equal_nfe_feedback_r2.py',
    'run_a101r2_20261110.sh', str(CARD.relative_to(PROJECTS)),
)


def check(root):
    root = Path(root); plan = files.read(root / 'plan.json')
    files.require(plan['experiment'] == rules.EXPERIMENT, 'Wrong A101R2 plan')
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
        check(root); print('A101R2 frozen plan reused.', flush=True); return
    r1.prepare(args)
    plan = files.read(root / 'plan.json'); plan['experiment'] = rules.EXPERIMENT
    plan['sources'].update({name: files.digest(PROJECTS / name) for name in R2_SOURCES})
    plan['repair_of'] = dict(experiment='A101R1_equal_nfe_feedback', root=r1.rules.ROOT,
        reason='R1 wrapper recursively called the monkeypatched measure function before any method forward.',
        actual_R1_method_forwards=0, reuse='No R1 artifacts or state reused; separate root and branch.')
    plan['code_commit'] = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=PROJECTS.parent, text=True).strip()
    write(root / 'plan.json', plan); shutil.copyfile(CARD, root / 'experiment_card_r2.md')
    check(root)
    print('A101R2 PREPARE/FREEZE: non-recursive dispatcher; scientific protocol unchanged.', flush=True)


def measure(root, plan, case, arm, model, uuid):
    if arm == 'ZAPS_LOCAL_300':
        return _BASE_MEASURE(root, plan, case, arm, model, uuid)
    baseline = r1.original.record_path(root, case, 'ZAPS_LOCAL_300')
    files.require(baseline.exists() and files.read(baseline)['status'] == 'complete',
                  'Complete paired ZAPS baseline required')
    import torch
    reference = torch.load(baseline.parent / 'entry.pt', map_location='cpu', weights_only=True)
    with repair.paired_final_rng(reference), repair.repaired_diagonal():
        return _BASE_MEASURE(root, plan, case, arm, model, uuid)


def worker(args):
    with patch.object(r1.original, 'check', check), patch.object(r1.original, 'measure', measure):
        r1.original.worker(args)


def summarize(root):
    with patch.object(r1.original, 'check', check):
        summary = r1.original.summarize(root)
    summary['classification'] = 'STRICT_EQUAL_300_NFE_PAIRED_RNG_TRUE_ADJOINT_R2'
    summary['repair_of'] = check(root)['repair_of']; write(Path(root) / 'summary.json', summary)
    return summary


def report(root): r1.original.report(root)


def archive(root):
    from utils import archive_refinement_confirmation as exporter
    exporter.BRANCH = rules.BRANCH
    bundle, commit = exporter.export_and_commit({'A101R2': Path(root)}, projects=PROJECTS)
    print('Archive:', bundle, 'Results commit:', commit, 'Branch:', rules.BRANCH, flush=True)
    subprocess.run(['git', 'push', 'origin', rules.BRANCH], cwd=PROJECTS.parent, check=True)
    print('RESULTS_PUSH_PASS: A101R2 uploaded.', flush=True)


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('--root', default=rules.ROOT); value.add_argument('--a71-root', default=r1.original.PARENT)
    value.add_argument('--ffhq-dir', default='/home/lzy/FFHQ/00000/00000')
    value.add_argument('--shard', type=int, choices=(0, 1)); value.add_argument('--device-uuid')
    group = value.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'worker', 'summarize', 'report', 'archive'):
        group.add_argument('--' + name, action='store_true')
    return value


def main():
    p = parser(); args = p.parse_args()
    if args.worker and (args.shard is None or not args.device_uuid): p.error('Worker needs shard and device UUID')
    if args.prepare: prepare(args)
    elif args.worker: worker(args)
    elif args.summarize: summarize(args.root)
    elif args.report: report(args.root)
    else: archive(args.root)


if __name__ == '__main__': main()
