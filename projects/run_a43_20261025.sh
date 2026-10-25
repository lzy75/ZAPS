#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a43_parent=/data0/lzy/ZAPS-results/diag_continuous_uncertainty
mkdir -p "$zaps_a43_parent"
exec 9>"$zaps_a43_parent/launcher.lock"
if ! flock -n 9;then echo 'A43 already running; duplicate refused.' >&2;exit 1;fi
zaps_a43_log="$zaps_a43_parent/launcher_20261025.log"
zaps_a43_fail(){ tail -n 65 "$1" 9>&-;exit 1; }
echo 'A43: continuous covariance feedback and matched controls; one reused image x four tasks x two states; <=516 new diagnostic calls. Waiting for final output.'
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python - <<'PY' 9>&- >> "$zaps_a43_log" 2>&1
import sys,unittest
suite=unittest.defaultTestLoader.loadTestsFromNames(['utils.test_continuous_uncertainty_feedback','utils.test_ffhq_action_geometry'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
PY
then zaps_a43_fail "$zaps_a43_log";fi
if ! python -u utils/diag_continuous_uncertainty_feedback.py --prepare 9>&- >> "$zaps_a43_log" 2>&1;then zaps_a43_fail "$zaps_a43_log";fi
tail -n 2 "$zaps_a43_log" 9>&-
if ! python - <<'PY' 9>&- > "$zaps_a43_parent/devices.txt" 2>> "$zaps_a43_log"
import os,subprocess
from utils.multitask_state_transfer import select_devices
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
rows=[]
for line in raw.splitlines():
    i,u,f=[v.strip() for v in line.split(',')];rows.append(dict(index=i,uuid=u,free=int(f)))
print('\n'.join(select_devices(rows,os.environ.get('CUDA_VISIBLE_DEVICES'),os.environ.get('ZAPS_A43_GPU0'),os.environ.get('ZAPS_A43_GPU1'))))
PY
then zaps_a43_fail "$zaps_a43_log";fi
mapfile -t zaps_a43_gpus < "$zaps_a43_parent/devices.txt"
echo "A43 selected GPUs: ${zaps_a43_gpus[*]}; equal IDs mean serial shards."
zaps_a43_pids=();zaps_a43_monitor=''
zaps_a43_cleanup(){
    for zaps_a43_pid in "${zaps_a43_pids[@]}";do kill -TERM "$zaps_a43_pid" 2>/dev/null || true;done
    if [ -n "$zaps_a43_monitor" ];then kill -TERM "$zaps_a43_monitor" 2>/dev/null || true;fi
}
trap 'zaps_a43_cleanup;exit 130' INT TERM HUP
(
    exec 9>&-
    while sleep 60;do
        python - <<'PY'
from pathlib import Path
import json
records=[]
for p in Path('/data0/lzy/ZAPS-results/diag_continuous_uncertainty/A43_20261025').glob('*/attempt_*/run.json'):
    try:records.append(json.loads(p.read_text()))
    except (ValueError,OSError):pass
print('A43 progress:',sum(r.get('status')=='complete' for r in records),'/4 complete; recorded new calls=',sum(r.get('nfe',0) for r in records),flush=True)
PY
    done
) &
zaps_a43_monitor=$!
zaps_a43_launch(){
    env CUDA_VISIBLE_DEVICES="$2" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      python -u utils/diag_continuous_uncertainty_feedback.py --shard "$1" --device cuda:0 \
      9>&- > "$zaps_a43_parent/shard$1.log" 2>&1 &
    zaps_a43_last=$!;zaps_a43_pids+=("$zaps_a43_last")
}
zaps_a43_s0=0;zaps_a43_s1=0
zaps_a43_launch 0 "${zaps_a43_gpus[0]}";zaps_a43_first=$zaps_a43_last
if [ "${zaps_a43_gpus[0]}" = "${zaps_a43_gpus[1]}" ];then
    if wait "$zaps_a43_first";then :;else zaps_a43_s0=$?;fi
    zaps_a43_pids=();zaps_a43_launch 1 "${zaps_a43_gpus[1]}"
    if wait "$zaps_a43_last";then :;else zaps_a43_s1=$?;fi
else
    zaps_a43_launch 1 "${zaps_a43_gpus[1]}";zaps_a43_second=$zaps_a43_last
    if wait "$zaps_a43_first";then :;else zaps_a43_s0=$?;fi
    zaps_a43_pids=("$zaps_a43_second")
    if wait "$zaps_a43_second";then :;else zaps_a43_s1=$?;fi
fi
zaps_a43_pids=();zaps_a43_cleanup;zaps_a43_monitor='';trap - INT TERM HUP
if [ "$zaps_a43_s0" -ne 0 ] || [ "$zaps_a43_s1" -ne 0 ];then
    tail -n 50 "$zaps_a43_parent/shard0.log" 9>&-;tail -n 50 "$zaps_a43_parent/shard1.log" 9>&-;exit 1
fi
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    python -u utils/diag_continuous_uncertainty_feedback.py --archive 9>&- > "$zaps_a43_parent/final.log" 2>&1;then zaps_a43_fail "$zaps_a43_parent/final.log";fi
cat "$zaps_a43_parent/final.log" 9>&-
if git push origin results/ffhq-continuous-uncertainty-20261025 9>&- >> "$zaps_a43_log" 2>&1;then
    echo 'RESULTS_PUSH_PASS: A43 continuous pilot, rejected queries, all controls and comparison panels uploaded.'
else echo 'RESULTS_PUSH_PENDING: outputs retained; no GPU rerun needed.';tail -n 15 "$zaps_a43_log" 9>&-;fi
