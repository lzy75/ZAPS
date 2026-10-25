#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a42_parent=/data0/lzy/ZAPS-results/diag_uncertainty_feedback
mkdir -p "$zaps_a42_parent"
exec 9>"$zaps_a42_parent/launcher.lock"
if ! flock -n 9;then echo 'A42 already running; duplicate refused.' >&2;exit 1;fi
zaps_a42_log="$zaps_a42_parent/launcher_20261025.log"
zaps_a42_fail(){ tail -n 65 "$1" 9>&-;exit 1; }
echo 'A42: projected conditional uncertainty feedback; one old image x four tasks x two states; <=232 new diagnostic calls. Waiting for final output.'
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python - <<'PY' 9>&- >> "$zaps_a42_log" 2>&1
import sys,unittest
suite=unittest.defaultTestLoader.loadTestsFromNames(['utils.test_uncertainty_feedback','utils.test_ffhq_action_geometry'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
PY
then zaps_a42_fail "$zaps_a42_log";fi
if ! python -u utils/diag_uncertainty_feedback.py --prepare 9>&- >> "$zaps_a42_log" 2>&1;then zaps_a42_fail "$zaps_a42_log";fi
tail -n 2 "$zaps_a42_log" 9>&-
if ! python - <<'PY' 9>&- > "$zaps_a42_parent/devices.txt" 2>> "$zaps_a42_log"
import os,subprocess
from utils.multitask_state_transfer import select_devices
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
rows=[]
for line in raw.splitlines():
    i,u,f=[v.strip() for v in line.split(',')];rows.append(dict(index=i,uuid=u,free=int(f)))
print('\n'.join(select_devices(rows,os.environ.get('CUDA_VISIBLE_DEVICES'),os.environ.get('ZAPS_A42_GPU0'),os.environ.get('ZAPS_A42_GPU1'))))
PY
then zaps_a42_fail "$zaps_a42_log";fi
mapfile -t zaps_a42_gpus < "$zaps_a42_parent/devices.txt"
echo "A42 selected GPUs: ${zaps_a42_gpus[*]}; equal IDs mean serial shards."
zaps_a42_pids=();zaps_a42_monitor=''
zaps_a42_cleanup(){
    for zaps_a42_pid in "${zaps_a42_pids[@]}";do kill -TERM "$zaps_a42_pid" 2>/dev/null || true;done
    if [ -n "$zaps_a42_monitor" ];then kill -TERM "$zaps_a42_monitor" 2>/dev/null || true;fi
}
trap 'zaps_a42_cleanup;exit 130' INT TERM HUP
(
    exec 9>&-
    while sleep 60;do
        python - <<'PY'
from pathlib import Path
import json
records=[]
for p in Path('/data0/lzy/ZAPS-results/diag_uncertainty_feedback/A42_20261025').glob('*/attempt_*/run.json'):
    try:records.append(json.loads(p.read_text()))
    except (ValueError,OSError):pass
print('A42 progress:',sum(r.get('status')=='complete' for r in records),'/4 complete; recorded new calls=',sum(r.get('nfe',0) for r in records),flush=True)
PY
    done
) &
zaps_a42_monitor=$!
zaps_a42_launch(){
    env CUDA_VISIBLE_DEVICES="$2" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      python -u utils/diag_uncertainty_feedback.py --shard "$1" --device cuda:0 \
      9>&- > "$zaps_a42_parent/shard$1.log" 2>&1 &
    zaps_a42_last=$!;zaps_a42_pids+=("$zaps_a42_last")
}
zaps_a42_s0=0;zaps_a42_s1=0
zaps_a42_launch 0 "${zaps_a42_gpus[0]}";zaps_a42_first=$zaps_a42_last
if [ "${zaps_a42_gpus[0]}" = "${zaps_a42_gpus[1]}" ];then
    if wait "$zaps_a42_first";then :;else zaps_a42_s0=$?;fi
    zaps_a42_pids=();zaps_a42_launch 1 "${zaps_a42_gpus[1]}"
    if wait "$zaps_a42_last";then :;else zaps_a42_s1=$?;fi
else
    zaps_a42_launch 1 "${zaps_a42_gpus[1]}";zaps_a42_second=$zaps_a42_last
    if wait "$zaps_a42_first";then :;else zaps_a42_s0=$?;fi
    zaps_a42_pids=("$zaps_a42_second")
    if wait "$zaps_a42_second";then :;else zaps_a42_s1=$?;fi
fi
zaps_a42_pids=();zaps_a42_cleanup;zaps_a42_monitor='';trap - INT TERM HUP
if [ "$zaps_a42_s0" -ne 0 ] || [ "$zaps_a42_s1" -ne 0 ];then
    tail -n 50 "$zaps_a42_parent/shard0.log" 9>&-;tail -n 50 "$zaps_a42_parent/shard1.log" 9>&-;exit 1
fi
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    python -u utils/diag_uncertainty_feedback.py --archive 9>&- > "$zaps_a42_parent/final.log" 2>&1;then zaps_a42_fail "$zaps_a42_parent/final.log";fi
cat "$zaps_a42_parent/final.log" 9>&-
if git push origin results/ffhq-uncertainty-feedback-20261025 9>&- >> "$zaps_a42_log" 2>&1;then
    echo 'RESULTS_PUSH_PASS: A42 pilot, states, choices and all four comparison panels uploaded.'
else echo 'RESULTS_PUSH_PENDING: outputs retained; no GPU rerun needed.';tail -n 15 "$zaps_a42_log" 9>&-;fi
