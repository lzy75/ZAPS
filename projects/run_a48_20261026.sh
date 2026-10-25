#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a48_parent=/data0/lzy/ZAPS-results/diag_response_reliability
mkdir -p "$zaps_a48_parent"
exec 9>"$zaps_a48_parent/launcher.lock"
if ! flock -n 9;then echo 'A48 already running; duplicate refused.' >&2;exit 1;fi
zaps_a48_log="$zaps_a48_parent/launcher_20261026.log"
zaps_a48_fail(){ tail -n 65 "$1" 9>&-;exit 1; }
echo 'A48: same20 OLD development cases; identical LOCAL/TRUST probes; reliability gate; <=400 NEW diagnostic calls; no training. Waiting for final output.'
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python - <<'PY' 9>&- >> "$zaps_a48_log" 2>&1
import sys,unittest
suite=unittest.defaultTestLoader.loadTestsFromNames(['utils.test_response_reliability','utils.test_frozen_four_task_confirmation','utils.test_uncertainty_feedback'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
PY
then zaps_a48_fail "$zaps_a48_log";fi
if ! python -u utils/diag_response_reliability.py --prepare 9>&- >> "$zaps_a48_log" 2>&1;then zaps_a48_fail "$zaps_a48_log";fi
tail -n 2 "$zaps_a48_log" 9>&-
if ! python - <<'PY' 9>&- > "$zaps_a48_parent/devices.txt" 2>> "$zaps_a48_log"
import os,subprocess
from utils.multitask_state_transfer import select_devices
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
rows=[]
for line in raw.splitlines():
    i,u,f=[v.strip() for v in line.split(',')];rows.append(dict(index=i,uuid=u,free=int(f)))
print('\n'.join(select_devices(rows,os.environ.get('CUDA_VISIBLE_DEVICES'),os.environ.get('ZAPS_A48_GPU0'),os.environ.get('ZAPS_A48_GPU1'))))
PY
then zaps_a48_fail "$zaps_a48_log";fi
mapfile -t zaps_a48_gpus < "$zaps_a48_parent/devices.txt"
echo "A48 selected GPUs: ${zaps_a48_gpus[*]}; equal IDs mean serial shards."
zaps_a48_pids=();zaps_a48_monitor=''
zaps_a48_cleanup(){
    for zaps_a48_pid in "${zaps_a48_pids[@]}";do kill -TERM "$zaps_a48_pid" 2>/dev/null || true;done
    if [ -n "$zaps_a48_monitor" ];then kill -TERM "$zaps_a48_monitor" 2>/dev/null || true;fi
}
trap 'zaps_a48_cleanup;exit 130' INT TERM HUP
(
    exec 9>&-
    while sleep 60;do
        python - <<'PY'
from pathlib import Path
import json
records=[]
for p in Path('/data0/lzy/ZAPS-results/diag_response_reliability/A48_20261026').glob('*/attempt_*/run.json'):
    try:records.append(json.loads(p.read_text()))
    except (ValueError,OSError):pass
print('A48 progress:',sum(r.get('status')=='complete' for r in records),'/20 complete; NEW recorded calls=',sum(r.get('nfe',0) for r in records),flush=True)
PY
    done
) &
zaps_a48_monitor=$!
zaps_a48_launch(){
    env CUDA_VISIBLE_DEVICES="$2" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      python -u utils/diag_response_reliability.py --shard "$1" --device cuda:0 \
      9>&- > "$zaps_a48_parent/shard$1.log" 2>&1 &
    zaps_a48_last=$!;zaps_a48_pids+=("$zaps_a48_last")
}
zaps_a48_s0=0;zaps_a48_s1=0
zaps_a48_launch 0 "${zaps_a48_gpus[0]}";zaps_a48_first=$zaps_a48_last
if [ "${zaps_a48_gpus[0]}" = "${zaps_a48_gpus[1]}" ];then
    if wait "$zaps_a48_first";then :;else zaps_a48_s0=$?;fi
    zaps_a48_pids=();zaps_a48_launch 1 "${zaps_a48_gpus[1]}"
    if wait "$zaps_a48_last";then :;else zaps_a48_s1=$?;fi
else
    zaps_a48_launch 1 "${zaps_a48_gpus[1]}";zaps_a48_second=$zaps_a48_last
    if wait "$zaps_a48_first";then :;else zaps_a48_s0=$?;fi
    zaps_a48_pids=("$zaps_a48_second")
    if wait "$zaps_a48_second";then :;else zaps_a48_s1=$?;fi
fi
zaps_a48_pids=();zaps_a48_cleanup;zaps_a48_monitor='';trap - INT TERM HUP
if [ "$zaps_a48_s0" -ne 0 ] || [ "$zaps_a48_s1" -ne 0 ];then
    tail -n 50 "$zaps_a48_parent/shard0.log" 9>&-;tail -n 50 "$zaps_a48_parent/shard1.log" 9>&-;exit 1
fi
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    python -u utils/diag_response_reliability.py --archive 9>&- > "$zaps_a48_parent/final.log" 2>&1;then zaps_a48_fail "$zaps_a48_parent/final.log";fi
cat "$zaps_a48_parent/final.log" 9>&-
if timeout 45 env GIT_TERMINAL_PROMPT=0 git push origin results/ffhq-response-reliability-20261026 9>&- >> "$zaps_a48_log" 2>&1;then
    echo 'RESULTS_PUSH_PASS: A48 reliability ablation, all20 cases and comparison panels uploaded.'
else echo 'RESULTS_PUSH_PENDING: outputs retained; no GPU rerun needed.';tail -n 15 "$zaps_a48_log" 9>&-;fi
