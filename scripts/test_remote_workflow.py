"""仅测 Linux 运维流程：临时仓库、CPU 假任务、模拟 nvidia-smi。"""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time


def main(source):
    with tempfile.TemporaryDirectory(prefix='zaps_ops_test_', dir='/data0/lzy') as folder:
        root = Path(folder)
        repo, jobs = root / 'repo', root / 'jobs'
        (repo / 'projects/utils').mkdir(parents=True)
        (repo / 'projects/experiment_cards').mkdir()
        (repo / 'projects/experiment_cards/OPS_smoke.md').write_text('CPU infrastructure smoke; zero model calls.\n')
        (repo / 'projects/utils/ops_smoke.py').write_text(
            "import json,os,pathlib,sys,time\n"
            "time.sleep(3 if '--timeout' in sys.argv else .2)\n"
            "p=pathlib.Path(os.environ['ZAPS_RESULTS_DIR'])\n"
            "(p/'budget.json').write_text(json.dumps(dict(method_nfe=1 if '--exceed' in sys.argv else 0,diagnostic_nfe=0)))\n"
            "print('CPU_SMOKE_OK',flush=True)\n")
        for args in (['init', '-q'], ['add', '.'], ['-c', 'user.name=Ops test', '-c', 'user.email=ops-test@localhost', 'commit', '-qm', 'CPU fixture']):
            subprocess.run(['git', '-C', str(repo), *args], check=True, stdout=subprocess.DEVNULL)
        commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        subprocess.run(['git', '-C', str(repo), 'remote', 'add', 'origin', str(repo)], check=True)
        source = source.replace("REPO = Path('/home/lzy/ZAPS')", 'REPO = Path(' + repr(str(repo)) + ')')
        source = source.replace("ROOT = Path('/data0/lzy/ZAPS-jobs')", 'ROOT = Path(' + repr(str(jobs)) + ')')
        source = source.replace("PYTHON = '/home/lzy/anaconda3/bin/python'", 'PYTHON = ' + repr(sys.executable))
        scope = dict(__name__='ops_test_module', CONTROLLER_SOURCE=source)
        exec(compile(source, '<ops-test-controller>', 'exec'), scope)
        scope['sync'](commit)
        assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == commit
        print('PASS isolated code synchronization without checkout')
        binary = root / 'bin'
        binary.mkdir()
        fake = binary / 'nvidia-smi'
        fake.write_text("#!/bin/sh\nprintf '0, 0\\n'\n")
        fake.chmod(0o755)
        os.environ['PATH'] = str(binary) + os.pathsep + os.environ['PATH']
        template = dict(run_id='ops_ok', commit=commit, gpu=0, card='experiment_cards/OPS_smoke.md',
                        entrypoint='utils/ops_smoke.py', arguments=[], method_nfe=0, diagnostic_nfe=0,
                        timeout_seconds=5, nfe_contract='zaps-job-env-v1')

        def wait(run_id):
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                value = scope['status'](scope['job_path'](run_id))
                if value['status'] in ('complete', 'failed', 'orphaned'):
                    return value
                time.sleep(.05)
            raise AssertionError('Supervisor did not finish')

        def rejected(spec):
            try:
                scope['submit'](spec)
            except (ValueError, FileExistsError):
                return
            raise AssertionError('Unsafe submission accepted')

        require = scope['require']
        result = scope['submit'](template)
        require(os.getsid(result['supervisor_pid']) == result['supervisor_pid'], 'Supervisor not detached')
        require(wait('ops_ok')['status'] == 'complete', 'CPU smoke failed')
        rejected(template)
        print('PASS detached CPU completion and duplicate ID rejection')
        tree = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD^{tree}'], text=True).strip()
        rejected(dict(template, run_id='ops_tree', commit=tree))
        require(not (jobs / 'ops_tree').exists(), 'Tree object accepted as commit')
        print('PASS commit object type gate')
        import fcntl
        with (jobs / '.single-task.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            scope['submit'](dict(template, run_id='ops_locked'))
            require(wait('ops_locked')['status'] == 'failed', 'Concurrent task accepted')
        print('PASS task lock')
        scope['submit'](dict(template, run_id='ops_budget', arguments=['--exceed']))
        require(wait('ops_budget')['status'] == 'failed', 'NFE overrun accepted')
        scope['submit'](dict(template, run_id='ops_timeout', arguments=['--timeout'], timeout_seconds=1))
        require(wait('ops_timeout')['status'] == 'failed', 'Timeout accepted')
        print('PASS budget ledger and timeout failure')
        fake.write_text("#!/bin/sh\nprintf '4096, 30\\n'\n")
        rejected(dict(template, run_id='ops_busy'))
        require(not (jobs / 'ops_busy').exists(), 'Busy GPU reserved job')
        rejected(dict(template, run_id='../escape'))
        rejected(dict(template, entrypoint='utils/diag_one_step_guidance_scale.py'))
        rejected(dict(template, card='experiment_cards/A2c_one_step_guidance_scale_20261019.md'))
        print('PASS busy GPU, traversal and frozen A2 rejection')
        controller = jobs / 'ops_ok/controller.py'
        archive = subprocess.check_output([sys.executable, '-B', str(controller), 'collect', '--run-id', 'ops_ok'])
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            manifest = json.load(tar.extractfile('MANIFEST.json'))
            for entry in manifest['files']:
                require(scope['sha'](tar.extractfile(entry['path']).read()) == entry['sha256'], 'Collection hash mismatch')
        print('PASS result collection hashes')
        orphan = jobs / 'ops_orphan'
        orphan.mkdir()
        scope['save'](orphan / 'status.json', dict(status='running', supervisor_pid=999999999))
        require(scope['status'](orphan)['status'] == 'orphaned', 'Missing supervisor not detected')
        try:
            scope['collect'](orphan)
        except ValueError:
            pass
        else:
            raise AssertionError('Orphaned result collection accepted')
        job = jobs / 'ops_ok'
        (job / 'results/cache.pt').write_bytes(b'CPU fake binary; no tensor load')
        original_sha = scope['file_sha']
        def concurrent_write(path):
            value = original_sha(path)
            with (job / 'stdout.log').open('a') as log:
                log.write('concurrent writer\n')
            return value
        scope['file_sha'] = concurrent_write
        try:
            scope['collect'](job)
        except ValueError as error:
            require('changed during collection' in str(error), 'Unexpected collection failure')
        else:
            raise AssertionError('Concurrent result mutation accepted')
        print('PASS orphaned process and concurrent writer rejection')
        print('Infrastructure smoke only; method/diagnostic model NFE=0; temporary fixtures removed.')


if __name__ == '__main__':
    main(Path(__file__).with_name('server_job.py').read_text(encoding='utf-8'))
