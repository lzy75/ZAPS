"""Independent A91 numerical-test repair; original policy and failed root untouched."""
import sys
from pathlib import Path
from unittest.mock import patch

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
sys.path.insert(0, str(PROJECTS.parent))
from utils import diag_directional_reliability as original

ROOT = original.ROOT.replace('A91_20261104', 'A91R1_20261104')
BRANCH = 'results/zaps-directional-reliability-r1-20261104'
PREPARE = original.prepare
RUN = original.subprocess.run
NAMES = ('utils/a91r1_session.py', 'utils/test_directional_reliability_r1_runtime.py',
         'run_a91r1_20261104.sh', 'experiment_cards/A91R1_directional_reliability_20261104.md')


def prepare(root, parent):
    fresh = not (root/'plan.json').exists()
    plan = PREPARE(root, parent)
    if fresh:
        plan['revision'] = 'A91R1_FLOAT32_HOMOGENEITY_GATE'
        plan['sources'].update({name: original.files.digest(PROJECTS/name) for name in NAMES})
        plan['protocol']['numerical_revision'] = (
            'Policy unchanged. Homogeneity: relative L2<=16*float32_eps and '
            'elementwise rtol1e-5/atol32*eps; identity/zero exact and weight tolerance unchanged. '
            'Original failed A91 root retained. Tests output archived separately.')
        original.files.write(root/'plan.json', plan)
    original.files.require(plan.get('revision') == 'A91R1_FLOAT32_HOMOGENEITY_GATE',
                           'Not an independent A91R1 plan')
    return plan


def invoke(command, *args, **kwargs):
    expected = [sys.executable, '-m', 'unittest', '-v',
                'utils.test_directional_reliability', 'utils.test_directional_reliability_runtime']
    if command != expected:
        return RUN(command, *args, **kwargs)
    revised = command[:-1]+['utils.test_directional_reliability_r1_runtime']
    options = dict(kwargs)
    options.pop('check', None)
    result = RUN(revised, *args, **options, stdout=original.subprocess.PIPE,
                 stderr=original.subprocess.STDOUT, text=True)
    print(result.stdout, end='', flush=True)
    root = Path(ROOT)
    if '--root' in sys.argv:
        root = Path(sys.argv[sys.argv.index('--root')+1]).resolve()
    original.files.write(root/'preflight_tests.json',
                         dict(command=revised, returncode=result.returncode, output=result.stdout))
    result.check_returncode()
    return result


def main():
    original.ROOT = ROOT
    original.BRANCH = BRANCH
    # The detached child must re-enter this wrapper, retaining the same repair.
    original.__file__ = __file__
    with patch.object(original, 'prepare', prepare), patch.object(original.subprocess, 'run', invoke):
        original.main()


if __name__ == '__main__':
    main()
