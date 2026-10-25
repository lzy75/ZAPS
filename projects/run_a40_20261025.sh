#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a40_parent=/data0/lzy/ZAPS-results/diag_multitask_state_transfer
mkdir -p "$zaps_a40_parent"
exec 9>"$zaps_a40_parent/launcher.lock"
if ! flock -n 9;then echo 'A40 already running; duplicate refused.' >&2;exit 1;fi
zaps_a40_log="$zaps_a40_parent/launcher_20261025.log"
echo 'A40: FFHQ/ImageNet x SR/blur; frozen STATE vs source/task TIME; planned11160 calls. Waiting for final summary.'
if ! python - <<'PY' 9>&- > "$zaps_a40_parent/devices.txt" 2>> "$zaps_a40_log"
import os,subprocess,sys
from utils.multitask_state_transfer import select_devices
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
rows=[]
for line in raw.splitlines():
    i,u,f=[v.strip() for v in line.split(',')];rows.append(dict(index=i,uuid=u,free=int(f)))
try:selected=select_devices(rows,os.environ.get('CUDA_VISIBLE_DEVICES'),os.environ.get('ZAPS_A40_GPU0'),os.environ.get('ZAPS_A40_GPU1'))
except ValueError as error:raise SystemExit(str(error))
print('\n'.join(selected));print('Selected GPUs:',selected,'; serial' if selected[0]==selected[1] else '; parallel',file=sys.stderr)
PY
then tail -n 25 "$zaps_a40_log";exit 1;fi
mapfile -t zaps_a40_gpus < "$zaps_a40_parent/devices.txt"
echo "A40 selected GPUs: ${zaps_a40_gpus[*]} (same ID means serial shards; at least10GiB free)."
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python - <<'PY' 9>&- >> "$zaps_a40_log" 2>&1
import sys,unittest
suite=unittest.defaultTestLoader.loadTestsFromNames(['utils.test_multitask_state_transfer','utils.test_common_state_risk','utils.test_transfer_response_feedback'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
PY
then tail -n 65 "$zaps_a40_log";exit 1;fi
if ! python -u utils/diag_multitask_state_transfer.py --prepare 9>&- >> "$zaps_a40_log" 2>&1;then
    tail -n 55 "$zaps_a40_log";exit 1
fi
zaps_a40_pids=();zaps_a40_monitor=''
zaps_a40_cleanup(){
    for zaps_a40_pid in "${zaps_a40_pids[@]}";do kill -TERM "$zaps_a40_pid" 2>/dev/null || true;done
    if [ -n "$zaps_a40_monitor" ];then kill -TERM "$zaps_a40_monitor" 2>/dev/null || true;fi
}
trap 'zaps_a40_cleanup;exit 130' INT TERM HUP
(
    exec 9>&-
    while sleep 60;do
        python - <<'PY'
from pathlib import Path
import json
root=Path('/data0/lzy/ZAPS-results/diag_multitask_state_transfer/A40_20261025')
records=[json.loads(p.read_text()) for p in root.glob('*/*/attempt_*/run.json')]
counts={k:sum(r.get('kind')==k and r.get('status')=='complete' for r in records) for k in ('BASELINE','MEASURE','TIME')}
print('A40 progress:',counts,'; recorded calls=',sum(r.get('nfe',0) for r in records),flush=True)
PY
    done
) &
zaps_a40_monitor=$!
zaps_a40_launch(){
    env CUDA_VISIBLE_DEVICES="$3" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      python -u utils/diag_multitask_state_transfer.py --phase "$1" --shard "$2" --device cuda:0 \
      9>&- > "$zaps_a40_parent/${1}_shard${2}.log" 2>&1 &
    zaps_a40_last=$!;zaps_a40_pids+=("$zaps_a40_last")
}
zaps_a40_phase(){
    local zaps_a40_name="$1" zaps_a40_s0=0 zaps_a40_s1=0 zaps_a40_first zaps_a40_second
    echo "A40 phase: $zaps_a40_name"
    zaps_a40_launch "$zaps_a40_name" 0 "${zaps_a40_gpus[0]}";zaps_a40_first=$zaps_a40_last
    if [ "${zaps_a40_gpus[0]}" = "${zaps_a40_gpus[1]}" ];then
        if wait "$zaps_a40_first";then :;else zaps_a40_s0=$?;fi
        zaps_a40_pids=();zaps_a40_launch "$zaps_a40_name" 1 "${zaps_a40_gpus[1]}"
        if wait "$zaps_a40_last";then :;else zaps_a40_s1=$?;fi
    else
        zaps_a40_launch "$zaps_a40_name" 1 "${zaps_a40_gpus[1]}";zaps_a40_second=$zaps_a40_last
        if wait "$zaps_a40_first";then :;else zaps_a40_s0=$?;fi
        zaps_a40_pids=("$zaps_a40_second")
        if wait "$zaps_a40_second";then :;else zaps_a40_s1=$?;fi
    fi
    zaps_a40_pids=()
    if [ "$zaps_a40_s0" -ne 0 ] || [ "$zaps_a40_s1" -ne 0 ];then
        if [ "$zaps_a40_s0" -ne 0 ];then tail -n 45 "$zaps_a40_parent/${zaps_a40_name}_shard0.log";fi
        if [ "$zaps_a40_s1" -ne 0 ];then tail -n 45 "$zaps_a40_parent/${zaps_a40_name}_shard1.log";fi
        return 1
    fi
}
if ! zaps_a40_phase measure;then zaps_a40_cleanup;exit 1;fi
if ! python -u utils/diag_multitask_state_transfer.py --profiles 9>&- >> "$zaps_a40_log" 2>&1;then
    tail -n 45 "$zaps_a40_log";zaps_a40_cleanup;exit 1
fi
if ! zaps_a40_phase time;then zaps_a40_cleanup;exit 1;fi
zaps_a40_cleanup;zaps_a40_monitor='';trap - INT TERM HUP
if ! python -u utils/diag_multitask_state_transfer.py --archive 9>&- > "$zaps_a40_parent/final.log" 2>&1;then
    tail -n 65 "$zaps_a40_parent/final.log";exit 1
fi
cat "$zaps_a40_parent/final.log"
if git push origin results/refinement-confirmation-20261020 9>&- >> "$zaps_a40_log" 2>&1;then
    echo 'RESULTS_PUSH_PASS: A40 task transfer records and all comparison images uploaded.'
else echo 'RESULTS_PUSH_PENDING: results retained; no GPU rerun required.';tail -n 12 "$zaps_a40_log";fi
