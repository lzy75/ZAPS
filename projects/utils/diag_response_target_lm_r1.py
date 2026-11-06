"""A95R1 interface-only repair. Preserve A95's pre-model-run failure verbatim."""
import subprocess,sys,shutil
from pathlib import Path
from unittest.mock import patch
PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import diag_response_target_lm as old,a95_checkpoint_scope_r1 as repair
from utils import response_target_runtime as runtime
ROOT='/data0/lzy/ZAPS-results/diag_response_target/A95R1_20261106'
FAILED_ROOT=old.ROOT
BRANCH='results/zaps-response-target-checkpoint-r1-20261106'
EXTRA=('utils/a95_checkpoint_scope_r1.py','utils/diag_response_target_lm_r1.py',
 'utils/test_a95_checkpoint_scope_r1.py','utils/test_a95_checkpoint_scope_r1_runtime.py',
 'run_a95r1_20261106.sh','experiment_cards/A95R1_checkpoint_scope_20261106.md')
ORIGINAL_PREPARE=old.prepare
ORIGINAL_EXECUTE=old.execute
ORIGINAL_PROBE=runtime.checkpoint_probe
ORIGINAL_SUMMARIZE=old.summarize


def failed_inputs(failed):
    files=old.files;session=files.read(failed/'session.json')
    trace=session.get('traceback','')
    files.require(session['status']=='failed' and 'setup_context' in trace and 'guided_diffusion/nn.py' in trace,'Different A95 failure; no automatic rerun')
    files.require(not any((failed/'cases').glob('*/prefix.json')) and not any((failed/'cases').glob('*/*/run.json')),'A95 model cases already attempted; no automatic repetition')
    names=[failed/'session.json',failed/'plan.json',failed/'summary.json',*sorted((failed/'loads').glob('*.json'))]
    files.require(files.read(failed/'summary.json')['completed_runs']==0,'A95 has completed runs')
    return {str(path):files.digest(path) for path in names}


def prepare(root,parent):
    if (root/'plan.json').exists():
        p=old.check(root);old.files.require(p.get('revision')=='A95R1','Incomplete repair preparation retained');return p
    bound=failed_inputs(Path(FAILED_ROOT));p=ORIGINAL_PREPARE(root,parent)
    p['revision']='A95R1';p['inputs'].update(bound)
    p['protocol']['interface_repair']='Original A95 stopped at actual-weight JVP probe before any case. Preserve old failure and attempted probe costs. Recognize root guided_diffusion AND DPS copy by actual loaded forward globals; restore scopes on all exits. Freeze actual loaded checkpoint/unet source hashes separately before real probe. Goal/arms/solver/grids/data/decisions unchanged.'
    old.files.write(root/'plan.json',p);shutil.copyfile(PROJECTS/'experiment_cards/A95R1_checkpoint_scope_20261106.md',root/'repair_card.md')
    return p


def execute(root,p):
    command=[sys.executable,'-m','unittest','-v','utils.test_a95_checkpoint_scope_r1','utils.test_a95_checkpoint_scope_r1_runtime']
    result=subprocess.run(command,cwd=PROJECTS,capture_output=True,text=True)
    print(result.stdout+result.stderr,end='',flush=True)
    old.files.write(root/'repair_tests.json',dict(command=command,returncode=result.returncode,output=result.stdout+result.stderr));result.check_returncode()
    def probe(model,t,count):
        sources=repair.loaded_sources(model);path=root/'actual_checkpoint_sources.json'
        if path.exists():old.files.require(old.files.read(path)==sources,'Actual loaded model source changed')
        else:old.files.write(path,sources)
        count['actual_checkpoint_sources']=sources
        print('A95R1 ACTUAL CHECKPOINT SOURCES:',list(sources),flush=True)
        return ORIGINAL_PROBE(model,t,count)
    with patch.object(runtime,'forward_ad_checkpoint_scope',repair.forward_ad_checkpoint_scope),patch.object(runtime,'checkpoint_probe',probe):
        ORIGINAL_EXECUTE(root,p)


def summarize(root):
    result=ORIGINAL_SUMMARIZE(root);result['revision']='A95R1'
    result['preserved_A95_failed_probe_loads']=[old.files.read(path) for path in sorted((Path(FAILED_ROOT)/'loads').glob('*.json'))]
    old.files.write(root/'summary.json',result)
    return result


def main():
    with patch.multiple(old,ROOT=ROOT,BRANCH=BRANCH,NAMES=old.NAMES+EXTRA,prepare=prepare,execute=execute,summarize=summarize,__file__=__file__):old.main()

if __name__=='__main__':main()
