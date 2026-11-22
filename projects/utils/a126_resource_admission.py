"""Resource admission and narrowly scoped recovery before A126 computation."""
import shutil
from pathlib import Path
from utils import paper_clock_alignment as files

GIB = 1024**3
MIN_GPU_FREE = 6*GIB
MIN_DISK_FREE = GIB
LEGACY_ERROR = "MemoryError('Need 6GiB GPU free and 1GiB output disk space')"


class ResourceShortage(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__('; '.join(report['shortages']))


def evaluate(gpu_free, disk_free):
    shortages = []
    if gpu_free < MIN_GPU_FREE:
        shortages.append(f'GPU free {gpu_free/GIB:.2f} GiB < required {MIN_GPU_FREE/GIB:g} GiB')
    if disk_free < MIN_DISK_FREE:
        shortages.append(f'Output disk free {disk_free/GIB:.2f} GiB < required {MIN_DISK_FREE/GIB:g} GiB')
    return dict(passed=not shortages, gpu_free_bytes=gpu_free, disk_free_bytes=disk_free,
                required_gpu_free_bytes=MIN_GPU_FREE, required_disk_free_bytes=MIN_DISK_FREE,
                shortages=shortages)


def check(torch, root):
    files.require(torch.cuda.is_available() and torch.cuda.device_count()==1,
                  'Allocate exactly one visible CUDA GPU')
    ancestor=Path(root)
    while not ancestor.exists():
        ancestor=ancestor.parent
    report=evaluate(torch.cuda.mem_get_info()[0], shutil.disk_usage(ancestor).free)
    print(f'A126 RESOURCES: GPU free={report["gpu_free_bytes"]/GIB:.2f} GiB '
          f'(need {MIN_GPU_FREE/GIB:g}); output disk={report["disk_free_bytes"]/GIB:.2f} GiB '
          f'(need {MIN_DISK_FREE/GIB:g}); output={root}', flush=True)
    if not report['passed']:
        raise ResourceShortage(report)
    return report


def recover_legacy_preflight(root, driver):
    """Preserve only the reported pre-computation failure; never retry a GPU attempt."""
    root,driver=Path(root),Path(driver).resolve()
    state_path=root/'session.json'
    if not state_path.exists():
        return False
    state=files.read(state_path)
    if state.get('status')!='failed' or state.get('error')!=LEGACY_ERROR:
        return False
    files.require(not any((root/name).exists() for name in ('load.json','checkpoint_probe.json','tests.json')),
                  'Model/test work already started; cannot recover as preflight-only')
    for path in (root/'cases').glob('*/*/run.json'):
        row=files.read(path)
        files.require(row.get('reused') is True and row.get('arm')=='BASE' and row.get('status')=='complete',
                      'A reconstruction was attempted; automatic preflight recovery refused')
    plan_path=root/'plan.json'
    files.require(plan_path.exists(),'Missing legacy plan; inspect failure manually')
    plan=files.read(plan_path)
    files.require(plan.get('experiment')=='A126','Wrong experiment for resource recovery')
    for name,digest in plan['sources'].items():
        if Path(name).resolve()!=driver:
            files.require(files.digest(name)==digest,'Scientific source changed: '+name)
    index=1
    while (root/'preflight_history'/f'{index:03d}').exists():
        index+=1
    folder=root/'preflight_history'/f'{index:03d}'
    folder.mkdir(parents=True)
    # Keep all old records and logs; only relocate the unstarted plan/status.
    if (root/'worker.log').exists():
        shutil.copy2(root/'worker.log',folder/'worker.log')
    for name in ('session.json','plan.json','summary.json','launcher.json'):
        source=root/name
        if source.exists():
            source.rename(folder/name)
    files.write(folder/'recovery.json',dict(reason='Reported legacy preflight resource failure only',
                method_forwards=0,model_loaded=False,old_plan_sha256=files.digest(folder/'plan.json')))
    print(f'A126 preserved unstarted resource failure: {folder}',flush=True)
    return True
