"""A95R2: fix wrapper->network source inspection; no scientific changes."""
import subprocess,sys,shutil
from pathlib import Path
from unittest.mock import patch
PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import diag_response_target_lm_r1 as r1
ROOT='/data0/lzy/ZAPS-results/diag_response_target/A95R2_20261106'
FAILED_ROOT=r1.ROOT
BRANCH='results/zaps-response-target-wrapper-r2-20261106'
EXTRA=('utils/diag_response_target_lm_r2.py','utils/test_a95_wrapper_r2.py',
 'run_a95r2_20261106.sh','experiment_cards/A95R2_wrapper_20261106.md')
PREPARE=r1.prepare
EXECUTE=r1.execute
SUMMARIZE=r1.summarize
LOADED_SOURCES=r1.repair.loaded_sources


def loaded_sources(value):
    network=value if callable(getattr(value,'modules',None)) else getattr(value,'model',None)
    if not callable(getattr(network,'modules',None)):raise TypeError('Expected a diffusion wrapper with an nn.Module model, or an nn.Module')
    return LOADED_SOURCES(network)


def failed_inputs(root):
    files=r1.old.files;session=files.read(root/'session.json');trace=session.get('traceback','')
    files.require(session['status']=='failed' and 'FFHQDiffusionModel' in trace and 'modules' in trace and 'loaded_sources' in trace,'Different R1 error; no automatic rerun')
    files.require(not any((root/'cases').glob('*/prefix.json')) and not any((root/'cases').glob('*/*/run.json')),'Prior cases attempted; no automatic repetition')
    files.require(files.read(root/'summary.json')['completed_runs']==0,'Prior completed cases')
    return {str(path):files.digest(path) for path in [root/'session.json',root/'summary.json',root/'plan.json',*sorted((root/'loads').glob('*.json'))]}


def prepare(root,parent):
    if (root/'plan.json').exists():
        p=r1.old.check(root);r1.old.files.require(p.get('revision')=='A95R2','Incomplete R2 preparation retained');return p
    bound=failed_inputs(Path(FAILED_ROOT));p=PREPARE(root,parent)
    p['revision']='A95R2';p['inputs'].update(bound)
    p['protocol']['wrapper_repair']='Source inspection accepts FFHQDiffusionModel wrapper OR nn.Module; inspect internal .model only for wrapper. Original network probe still receives full diffusion wrapper. Preserve zero-case A95/R1 failures, source bytes, probes and all scientific settings.'
    r1.old.files.write(root/'plan.json',p);shutil.copyfile(PROJECTS/'experiment_cards/A95R2_wrapper_20261106.md',root/'wrapper_repair_card.md')
    return p


def execute(root,p):
    command=[sys.executable,'-m','unittest','-v','utils.test_a95_wrapper_r2']
    result=subprocess.run(command,cwd=PROJECTS,capture_output=True,text=True)
    print(result.stdout+result.stderr,end='',flush=True)
    r1.old.files.write(root/'wrapper_tests.json',dict(command=command,returncode=result.returncode,output=result.stdout+result.stderr));result.check_returncode()
    EXECUTE(root,p)


def summarize(root):
    result=SUMMARIZE(root);result['revision']='A95R2'
    result['preserved_A95R1_failed_probe_loads']=[r1.old.files.read(path) for path in sorted((Path(FAILED_ROOT)/'loads').glob('*.json'))]
    r1.old.files.write(root/'summary.json',result);return result


def main():
    with patch.object(r1.repair,'loaded_sources',loaded_sources),patch.multiple(r1,ROOT=ROOT,BRANCH=BRANCH,EXTRA=r1.EXTRA+EXTRA,
            prepare=prepare,execute=execute,summarize=summarize,__file__=__file__):r1.main()

if __name__=='__main__':main()
