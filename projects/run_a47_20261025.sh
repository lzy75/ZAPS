#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a47_parent=/data0/lzy/ZAPS-results/diag_high_noise_boundary
mkdir -p "$zaps_a47_parent"
exec 9>"$zaps_a47_parent/launcher.lock"
if ! flock -n 9;then echo 'A47 already running; duplicate refused.' >&2;exit 1;fi
zaps_a47_log="$zaps_a47_parent/launcher_20261025.log"
zaps_a47_fail(){ tail -n 65 "$1" 9>&-;exit 1; }
echo 'A47: FINAL high-noise boundary, sigma=.10; 3 calibration +3 reused validation photos x4 tasks; 7200 method +<=1665 diagnostic =<=8865 NEW calls; frozen lambda. Waiting for final output.'
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python - <<'PY' 9>&- >> "$zaps_a47_log" 2>&1
import sys,unittest
suite=unittest.defaultTestLoader.loadTestsFromNames(['utils.test_high_noise_boundary','utils.test_frozen_four_task_confirmation','utils.test_uncertainty_feedback'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
PY
then zaps_a47_fail "$zaps_a47_log";fi
# Parent archive and cached audit only: no new UNet calls.
(
    exec 9>&-
    exec 8>/data0/lzy/ZAPS-results/diag_frozen_four_task/launcher.lock
    if flock -w 10 8;then
        if env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -u utils/diag_frozen_four_task_confirmation.py --archive 8>&- > "$zaps_a47_parent/A46_archive.log" 2>&1;then
            if timeout 45 env GIT_TERMINAL_PROMPT=0 git push origin results/ffhq-frozen-four-task-20261025 8>&- >> "$zaps_a47_parent/A46_archive.log" 2>&1;then
                echo 'A46_ARCHIVE_PUSH_PASS: NEW UNet NFE=0.'
            else echo 'A46_PUSH_PENDING: archive retained; continuing boundary audit.';tail -n 8 "$zaps_a47_parent/A46_archive.log";fi
        else echo 'A46_ARCHIVE_PENDING: original records retained; prepare will validate.';tail -n 15 "$zaps_a47_parent/A46_archive.log";fi
    else echo 'A46 launcher lock held; skipping duplicate archive. No process interrupted.';fi
)
if ! python -u utils/diag_a46_cached_audit.py 9>&- > "$zaps_a47_parent/A46b_cached.log" 2>&1;then zaps_a47_fail "$zaps_a47_parent/A46b_cached.log";fi
cat "$zaps_a47_parent/A46b_cached.log" 9>&-
if ! python -u utils/diag_high_noise_boundary.py --prepare 9>&- >> "$zaps_a47_log" 2>&1;then zaps_a47_fail "$zaps_a47_log";fi
tail -n 2 "$zaps_a47_log" 9>&-
if ! python - <<'PY' 9>&- > "$zaps_a47_parent/devices.txt" 2>> "$zaps_a47_log"
import os,subprocess
from utils.multitask_state_transfer import select_devices
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
rows=[]
for line in raw.splitlines():
    i,u,f=[v.strip() for v in line.split(',')];rows.append(dict(index=i,uuid=u,free=int(f)))
print('\n'.join(select_devices(rows,os.environ.get('CUDA_VISIBLE_DEVICES'),os.environ.get('ZAPS_A47_GPU0'),os.environ.get('ZAPS_A47_GPU1'))))
PY
then zaps_a47_fail "$zaps_a47_log";fi
mapfile -t zaps_a47_gpus < "$zaps_a47_parent/devices.txt"
echo "A47 selected GPUs: ${zaps_a47_gpus[*]}; equal IDs mean serial shards."
zaps_a47_pids=();zaps_a47_monitor=''
zaps_a47_cleanup(){
    for zaps_a47_pid in "${zaps_a47_pids[@]}";do kill -TERM "$zaps_a47_pid" 2>/dev/null || true;done
    if [ -n "$zaps_a47_monitor" ];then kill -TERM "$zaps_a47_monitor" 2>/dev/null || true;fi
}
trap 'zaps_a47_cleanup;exit 130' INT TERM HUP
(
    exec 9>&-
    while sleep 60;do
        python - <<'PY'
from pathlib import Path
import json
records=[]
base=Path('/data0/lzy/ZAPS-results/diag_high_noise_boundary/A47_20261025')
for p in list(base.glob('*/*_*/run.json'))+list(base.glob('*/BASELINE/attempt_*/run.json')):
    try:records.append(json.loads(p.read_text()))
    except (ValueError,OSError):pass
print('A47 progress: baseline',sum(r.get('phase')=='BASELINE' and r.get('status')=='complete' for r in records),'/24; choose',sum(r.get('phase')=='choose' and r.get('status')=='complete' for r in records),'/24; finish',sum(r.get('phase')=='finish' and r.get('status')=='complete' for r in records),'/12; NEW recorded calls=',sum(r.get('nfe',0) for r in records),flush=True)
PY
    done
) &
zaps_a47_monitor=$!
zaps_a47_launch(){
    env CUDA_VISIBLE_DEVICES="$3" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      python -u utils/diag_high_noise_boundary.py --phase "$1" --shard "$2" --device cuda:0 \
      9>&- > "$zaps_a47_parent/${1}_shard${2}.log" 2>&1 &
    zaps_a47_last=$!;zaps_a47_pids+=("$zaps_a47_last")
}
zaps_a47_phase(){
    local zaps_a47_name="$1" zaps_a47_s0=0 zaps_a47_s1=0
    echo "A47 $zaps_a47_name phase started; waiting with automatic progress."
    zaps_a47_launch "$zaps_a47_name" 0 "${zaps_a47_gpus[0]}";local zaps_a47_first=$zaps_a47_last
    if [ "${zaps_a47_gpus[0]}" = "${zaps_a47_gpus[1]}" ];then
        if wait "$zaps_a47_first";then :;else zaps_a47_s0=$?;fi
        zaps_a47_pids=();zaps_a47_launch "$zaps_a47_name" 1 "${zaps_a47_gpus[1]}"
        if wait "$zaps_a47_last";then :;else zaps_a47_s1=$?;fi
    else
        zaps_a47_launch "$zaps_a47_name" 1 "${zaps_a47_gpus[1]}";local zaps_a47_second=$zaps_a47_last
        if wait "$zaps_a47_first";then :;else zaps_a47_s0=$?;fi
        zaps_a47_pids=("$zaps_a47_second")
        if wait "$zaps_a47_second";then :;else zaps_a47_s1=$?;fi
    fi
    zaps_a47_pids=()
    if [ "$zaps_a47_s0" -ne 0 ] || [ "$zaps_a47_s1" -ne 0 ];then
        zaps_a47_cleanup
        tail -n 50 "$zaps_a47_parent/${zaps_a47_name}_shard0.log" 9>&-
        tail -n 50 "$zaps_a47_parent/${zaps_a47_name}_shard1.log" 9>&-;exit 1
    fi
}
zaps_a47_phase choose
if ! python -u utils/diag_high_noise_boundary.py --freeze 9>&- >> "$zaps_a47_log" 2>&1;then zaps_a47_cleanup;zaps_a47_fail "$zaps_a47_log";fi
tail -n 1 "$zaps_a47_log" 9>&-
zaps_a47_phase finish
zaps_a47_cleanup;zaps_a47_monitor='';trap - INT TERM HUP
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    python -u utils/diag_high_noise_boundary.py --archive 9>&- > "$zaps_a47_parent/final.log" 2>&1;then zaps_a47_fail "$zaps_a47_parent/final.log";fi
cat "$zaps_a47_parent/final.log" 9>&-
if timeout 45 env GIT_TERMINAL_PROMPT=0 git push origin results/ffhq-high-noise-boundary-20261025 9>&- >> "$zaps_a47_log" 2>&1;then
    echo 'RESULTS_PUSH_PASS: A47 high-noise boundary, cached audit, all controls and all12 comparison panels uploaded.'
else echo 'RESULTS_PUSH_PENDING: outputs retained; no GPU rerun needed.';tail -n 15 "$zaps_a47_log" 9>&-;fi
