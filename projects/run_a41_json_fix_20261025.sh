#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a41_parent=/data0/lzy/ZAPS-results/diag_ffhq_action_geometry
mkdir -p "$zaps_a41_parent"
exec 9>"$zaps_a41_parent/launcher.lock"
if ! flock -n 9;then echo 'A41 already running; duplicate refused. Existing work retained.' >&2;exit 1;fi
zaps_a41_log="$zaps_a41_parent/launcher_jsonfix_20261025.log"
zaps_a41_fail(){ tail -n 65 "$1" 9>&-;exit 1; }
echo 'A41 JSON scalar fix: original plan and completed cases retained. Partial attempts counted; no duplicate workers.'
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python - <<'PY' 9>&- >> "$zaps_a41_log" 2>&1
import sys,unittest
suite=unittest.defaultTestLoader.loadTestsFromNames(['utils.test_ffhq_action_geometry_json_fix','utils.test_ffhq_action_geometry','utils.test_multitask_state_transfer'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
PY
then zaps_a41_fail "$zaps_a41_log";fi
if ! python -u utils/diag_ffhq_action_geometry_json_fix.py --prepare 9>&- >> "$zaps_a41_log" 2>&1;then zaps_a41_fail "$zaps_a41_log";fi
tail -n 3 "$zaps_a41_log" 9>&-
if ! python - <<'PY' 9>&- > "$zaps_a41_parent/devices.txt" 2>> "$zaps_a41_log"
import os,subprocess,sys
from utils.multitask_state_transfer import select_devices
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
rows=[]
for line in raw.splitlines():
    i,u,f=[v.strip() for v in line.split(',')];rows.append(dict(index=i,uuid=u,free=int(f)))
selected=select_devices(rows,os.environ.get('CUDA_VISIBLE_DEVICES'),os.environ.get('ZAPS_A41_GPU0'),os.environ.get('ZAPS_A41_GPU1'))
print('\n'.join(selected));print('A41 devices:',selected,file=sys.stderr)
PY
then zaps_a41_fail "$zaps_a41_log";fi
mapfile -t zaps_a41_gpus < "$zaps_a41_parent/devices.txt"
echo "A41 GPUs: ${zaps_a41_gpus[*]}; equal IDs mean serial shards. Each needs at least10GiB free."
zaps_a41_pids=();zaps_a41_monitor=''
zaps_a41_cleanup(){
    for zaps_a41_pid in "${zaps_a41_pids[@]}";do kill -TERM "$zaps_a41_pid" 2>/dev/null || true;done
    if [ -n "$zaps_a41_monitor" ];then kill -TERM "$zaps_a41_monitor" 2>/dev/null || true;fi
}
trap 'zaps_a41_cleanup;exit 130' INT TERM HUP
(
    exec 9>&-
    while sleep 60;do
        python - <<'PY'
from pathlib import Path
import json
root=Path('/data0/lzy/ZAPS-results/diag_ffhq_action_geometry/A41_20261025')
records=[]
for p in root.glob('*/*/attempt_*/run.json'):
    try:records.append(json.loads(p.read_text()))
    except (ValueError,OSError):pass
counts={k:sum(r.get('kind')==k and r.get('status')=='complete' for r in records) for k in ('BASELINE','RESPONSE')}
print('A41 progress:',counts,'; recorded NEW calls=',sum(r.get('nfe',0) for r in records),flush=True)
PY
    done
) &
zaps_a41_monitor=$!
zaps_a41_launch(){
    env CUDA_VISIBLE_DEVICES="$2" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      python -u utils/diag_ffhq_action_geometry_json_fix.py --shard "$1" --device cuda:0 \
      9>&- > "$zaps_a41_parent/shard${1}_jsonfix.log" 2>&1 &
    zaps_a41_last=$!;zaps_a41_pids+=("$zaps_a41_last")
}
zaps_a41_s0=0;zaps_a41_s1=0
zaps_a41_launch 0 "${zaps_a41_gpus[0]}";zaps_a41_first=$zaps_a41_last
if [ "${zaps_a41_gpus[0]}" = "${zaps_a41_gpus[1]}" ];then
    if wait "$zaps_a41_first";then :;else zaps_a41_s0=$?;fi
    zaps_a41_pids=();zaps_a41_launch 1 "${zaps_a41_gpus[1]}"
    if wait "$zaps_a41_last";then :;else zaps_a41_s1=$?;fi
else
    zaps_a41_launch 1 "${zaps_a41_gpus[1]}";zaps_a41_second=$zaps_a41_last
    if wait "$zaps_a41_first";then :;else zaps_a41_s0=$?;fi
    zaps_a41_pids=("$zaps_a41_second")
    if wait "$zaps_a41_second";then :;else zaps_a41_s1=$?;fi
fi
zaps_a41_pids=();zaps_a41_cleanup;zaps_a41_monitor='';trap - INT TERM HUP
if [ "$zaps_a41_s0" -ne 0 ] || [ "$zaps_a41_s1" -ne 0 ];then
    tail -n 50 "$zaps_a41_parent/shard0_jsonfix.log" 9>&-
    tail -n 50 "$zaps_a41_parent/shard1_jsonfix.log" 9>&-
    exit 1
fi
if ! python -u utils/diag_ffhq_action_geometry_json_fix.py --archive 9>&- > "$zaps_a41_parent/final_jsonfix.log" 2>&1;then zaps_a41_fail "$zaps_a41_parent/final_jsonfix.log";fi
cat "$zaps_a41_parent/final_jsonfix.log" 9>&-
if git push origin results/ffhq-action-geometry-20261025 9>&- >> "$zaps_a41_log" 2>&1;then
    echo 'RESULTS_PUSH_PASS: A41 plus immutable A40 metadata snapshot uploaded.'
else echo 'RESULTS_PUSH_PENDING: outputs retained; no GPU rerun required.';tail -n 12 "$zaps_a41_log" 9>&-;fi
