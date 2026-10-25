#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a45_parent=/data0/lzy/ZAPS-results/diag_balanced_state_calibration
mkdir -p "$zaps_a45_parent"
exec 9>"$zaps_a45_parent/launcher.lock"
if ! flock -n 9;then echo 'A45 already running; duplicate refused.' >&2;exit 1;fi
zaps_a45_log="$zaps_a45_parent/launcher_20261025.log"
zaps_a45_fail(){ tail -n 65 "$1" 9>&-;exit 1; }
echo 'A45: finite task calibration; same20 A44 development cases; one new midpoint objective; <=160 NEW diagnostic calls; no training. Waiting for final output.'
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python - <<'PY' 9>&- >> "$zaps_a45_log" 2>&1
import sys,unittest
suite=unittest.defaultTestLoader.loadTestsFromNames(['utils.test_balanced_state_calibration','utils.test_continuous_uncertainty_feedback','utils.test_uncertainty_feedback'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
PY
then zaps_a45_fail "$zaps_a45_log";fi
# A44 archive only: no model calls, no overwrite of completed records.
(
    exec 9>&-
    exec 8>/data0/lzy/ZAPS-results/diag_frozen_state_matching/launcher.lock
    if flock -w 10 8;then
        if env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -u utils/diag_frozen_state_matching.py --archive 8>&- > "$zaps_a45_parent/A44_archive.log" 2>&1;then
            if timeout 45 env GIT_TERMINAL_PROMPT=0 git push origin results/ffhq-state-matching-20261025 8>&- >> "$zaps_a45_parent/A44_archive.log" 2>&1;then
                echo 'A44_ARCHIVE_PUSH_PASS: NEW UNet NFE=0.'
            else echo 'A44_PUSH_PENDING: completed outputs retained; continuing A45.';tail -n 8 "$zaps_a45_parent/A44_archive.log";fi
        else echo 'A44_ARCHIVE_PENDING: prepare will validate original records.';tail -n 15 "$zaps_a45_parent/A44_archive.log";fi
    else echo 'A44 launcher lock held; skipping duplicate archive. No process interrupted.';fi
)
if ! python -u utils/diag_balanced_state_calibration.py --prepare 9>&- >> "$zaps_a45_log" 2>&1;then zaps_a45_fail "$zaps_a45_log";fi
tail -n 2 "$zaps_a45_log" 9>&-
if ! python - <<'PY' 9>&- > "$zaps_a45_parent/devices.txt" 2>> "$zaps_a45_log"
import os,subprocess
from utils.multitask_state_transfer import select_devices
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
rows=[]
for line in raw.splitlines():
    i,u,f=[v.strip() for v in line.split(',')];rows.append(dict(index=i,uuid=u,free=int(f)))
print('\n'.join(select_devices(rows,os.environ.get('CUDA_VISIBLE_DEVICES'),os.environ.get('ZAPS_A45_GPU0'),os.environ.get('ZAPS_A45_GPU1'))))
PY
then zaps_a45_fail "$zaps_a45_log";fi
mapfile -t zaps_a45_gpus < "$zaps_a45_parent/devices.txt"
echo "A45 selected GPUs: ${zaps_a45_gpus[*]}; equal IDs mean serial shards."
zaps_a45_pids=();zaps_a45_monitor=''
zaps_a45_cleanup(){
    for zaps_a45_pid in "${zaps_a45_pids[@]}";do kill -TERM "$zaps_a45_pid" 2>/dev/null || true;done
    if [ -n "$zaps_a45_monitor" ];then kill -TERM "$zaps_a45_monitor" 2>/dev/null || true;fi
}
trap 'zaps_a45_cleanup;exit 130' INT TERM HUP
(
    exec 9>&-
    while sleep 60;do
        python - <<'PY'
from pathlib import Path
import json
records=[]
for p in Path('/data0/lzy/ZAPS-results/diag_balanced_state_calibration/A45_20261025').glob('*/attempt_*/run.json'):
    try:records.append(json.loads(p.read_text()))
    except (ValueError,OSError):pass
print('A45 progress:',sum(r.get('status')=='complete' for r in records),'/20 complete; NEW recorded calls=',sum(r.get('nfe',0) for r in records),flush=True)
PY
    done
) &
zaps_a45_monitor=$!
zaps_a45_launch(){
    env CUDA_VISIBLE_DEVICES="$2" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      python -u utils/diag_balanced_state_calibration.py --shard "$1" --device cuda:0 \
      9>&- > "$zaps_a45_parent/shard$1.log" 2>&1 &
    zaps_a45_last=$!;zaps_a45_pids+=("$zaps_a45_last")
}
zaps_a45_s0=0;zaps_a45_s1=0
zaps_a45_launch 0 "${zaps_a45_gpus[0]}";zaps_a45_first=$zaps_a45_last
if [ "${zaps_a45_gpus[0]}" = "${zaps_a45_gpus[1]}" ];then
    if wait "$zaps_a45_first";then :;else zaps_a45_s0=$?;fi
    zaps_a45_pids=();zaps_a45_launch 1 "${zaps_a45_gpus[1]}"
    if wait "$zaps_a45_last";then :;else zaps_a45_s1=$?;fi
else
    zaps_a45_launch 1 "${zaps_a45_gpus[1]}";zaps_a45_second=$zaps_a45_last
    if wait "$zaps_a45_first";then :;else zaps_a45_s0=$?;fi
    zaps_a45_pids=("$zaps_a45_second")
    if wait "$zaps_a45_second";then :;else zaps_a45_s1=$?;fi
fi
zaps_a45_pids=();zaps_a45_cleanup;zaps_a45_monitor='';trap - INT TERM HUP
if [ "$zaps_a45_s0" -ne 0 ] || [ "$zaps_a45_s1" -ne 0 ];then
    tail -n 50 "$zaps_a45_parent/shard0.log" 9>&-;tail -n 50 "$zaps_a45_parent/shard1.log" 9>&-;exit 1
fi
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    python -u utils/diag_balanced_state_calibration.py --archive 9>&- > "$zaps_a45_parent/final.log" 2>&1;then zaps_a45_fail "$zaps_a45_parent/final.log";fi
cat "$zaps_a45_parent/final.log" 9>&-
if timeout 45 env GIT_TERMINAL_PROMPT=0 git push origin results/ffhq-balanced-state-20261025 9>&- >> "$zaps_a45_log" 2>&1;then
    echo 'RESULTS_PUSH_PASS: A45 finite calibration, all20 cases and frozen source records uploaded.'
else echo 'RESULTS_PUSH_PENDING: outputs retained; no GPU rerun needed.';tail -n 15 "$zaps_a45_log" 9>&-;fi
