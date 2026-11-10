"""A101R3 strict 300-NFE driver."""
import argparse
from pathlib import Path
import shutil, subprocess, sys
from unittest.mock import patch

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS)); sys.path.insert(0, str(PROJECTS.parent))
from utils import diag_equal_nfe_feedback_r2 as r2
from utils import equal_nfe_feedback_r3 as rules
from utils import equal_nfe_feedback_runtime_r3 as repair
from utils import historical_ffhq_gain as files
from utils.diag_paper_clock_alignment import write

_BASE_MEASURE = r2._BASE_MEASURE
CARD = PROJECTS / 'experiment_cards/A101R3_equal_nfe_feedback_20261110.md'
SOURCES = ('utils/equal_nfe_feedback_r3.py', 'utils/equal_nfe_feedback_runtime_r3.py',
 'utils/diag_equal_nfe_feedback_r3.py', 'utils/a101r3_session.py',
 'utils/test_equal_nfe_feedback_r3.py', 'run_a101r3_20261110.sh', str(CARD.relative_to(PROJECTS)))


def check(root):
    root = Path(root); plan = files.read(root / 'plan.json')
    files.require(plan['experiment'] == rules.EXPERIMENT, 'Wrong A101R3 plan')
    for name, sha in plan['sources'].items(): files.require(files.digest(PROJECTS / name) == sha, 'Source changed: ' + name)
    for name, sha in plan['inputs'].items(): files.require(files.digest(name) == sha, 'Input changed: ' + name)
    files.require(files.digest(root / 'motion_kernel.npy') == plan['motion_kernel_sha256'], 'Motion kernel changed')
    return plan


def prepare(args):
    root = Path(args.root).resolve()
    if (root / 'plan.json').exists(): check(root); print('A101R3 frozen plan reused.', flush=True); return
    r2.prepare(args); plan = files.read(root / 'plan.json'); plan['experiment'] = rules.EXPERIMENT
    plan['sources'].update({name: files.digest(PROJECTS / name) for name in SOURCES})
    plan['repair_of'] = dict(experiment='A101R2_equal_nfe_feedback', root=r2.rules.ROOT,
        reason='Independent learned prefix exceeded bit-level gate; closed diagonal had no final optimizer graph.',
        reuse='No prior artifacts or state reused; separate root and branch.')
    plan['protocol']['prefix_gate'] = 'tau/initial exact; independently learned zeta/D relative error<=1e-5; actual values saved.'
    plan['protocol']['diagonal_backward'] = 'Forward-exact closed diagonal; native-q straight-through only for final backward whose update is after evaluated output.'
    plan['code_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECTS.parent, text=True).strip()
    write(root / 'plan.json', plan); shutil.copyfile(CARD, root / 'experiment_card_r3.md'); check(root)
    print('A101R3 PREPARE/FREEZE: numerical prefix audit + forward-exact autograd bridge.', flush=True)


def measure(root, plan, case, arm, model, uuid):
    if arm == 'ZAPS_LOCAL_300': return _BASE_MEASURE(root, plan, case, arm, model, uuid)
    baseline = r2.r1.original.record_path(root, case, 'ZAPS_LOCAL_300')
    files.require(baseline.exists() and files.read(baseline)['status'] == 'complete', 'Complete baseline required')
    import torch
    reference = torch.load(baseline.parent / 'entry.pt', map_location='cpu', weights_only=True)
    audit = baseline.parent.parent / arm / 'prefix_audit.json'
    with repair.repaired_execution(reference, audit):
        return _BASE_MEASURE(root, plan, case, arm, model, uuid)


def worker(args):
    with patch.object(r2.r1.original, 'check', check), patch.object(r2.r1.original, 'measure', measure):
        r2.r1.original.worker(args)


def summarize(root):
    with patch.object(r2.r1.original, 'check', check): summary = r2.r1.original.summarize(root)
    summary['classification'] = 'STRICT_EQUAL_300_NFE_NUMERIC_PREFIX_ST_R3'; summary['repair_of'] = check(root)['repair_of']
    write(Path(root) / 'summary.json', summary); return summary


def report(root): r2.r1.original.report(root)
def archive(root):
    from utils import archive_refinement_confirmation as exporter
    exporter.BRANCH = rules.BRANCH
    bundle, commit = exporter.export_and_commit({'A101R3': Path(root)}, projects=PROJECTS)
    print('Archive:', bundle, 'Results commit:', commit, 'Branch:', rules.BRANCH, flush=True)
    subprocess.run(['git', 'push', 'origin', rules.BRANCH], cwd=PROJECTS.parent, check=True)
    print('RESULTS_PUSH_PASS: A101R3 uploaded.', flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', default=rules.ROOT)
    p.add_argument('--a71-root', default=r2.r1.original.PARENT); p.add_argument('--ffhq-dir', default='/home/lzy/FFHQ/00000/00000')
    p.add_argument('--shard', type=int, choices=(0, 1)); p.add_argument('--device-uuid')
    group = p.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'worker', 'summarize', 'report', 'archive'): group.add_argument('--' + name, action='store_true')
    return p


def main():
    p = parser(); args = p.parse_args()
    if args.worker and (args.shard is None or not args.device_uuid): p.error('Worker needs shard and UUID')
    if args.prepare: prepare(args)
    elif args.worker: worker(args)
    elif args.summarize: summarize(args.root)
    elif args.report: report(args.root)
    else: archive(args.root)


if __name__ == '__main__': main()
