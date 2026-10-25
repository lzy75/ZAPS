#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a39_parent=/data0/lzy/ZAPS-results/diag_common_state_risk
mkdir -p "$zaps_a39_parent"
exec 9>"$zaps_a39_parent/launcher.lock"
if ! flock -n 9;then echo 'A39 already running; duplicate refused.' >&2;exit 1;fi
zaps_a39_log="$zaps_a39_parent/launcher_20261025.log"
echo 'A39: common-state physical/internal risk diagnosis; 900 new diagnostic calls, no retraining. Waiting for final summary.'
# Close the launcher lock FD in every child; only this coordinating shell owns it.
if ! python - <<'PY' 9>&- > "$zaps_a39_parent/devices.txt" 2>> "$zaps_a39_log"
import os,subprocess,sys
from utils.experiment_gpu_selection import choose
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
rows=[]
for line in raw.splitlines():
    i,u,f=[v.strip() for v in line.split(',')];rows.append(dict(index=i,uuid=u,free=int(f)))
try:selected=choose(rows,os.environ.get('CUDA_VISIBLE_DEVICES'),os.environ.get('ZAPS_A39_GPU0'),os.environ.get('ZAPS_A39_GPU1'))
except ValueError as error:raise SystemExit(str(error))
print('\n'.join(selected));print('Selected GPUs:',selected,'; serial' if selected[0]==selected[1] else '; parallel',file=sys.stderr)
PY
then tail -n 25 "$zaps_a39_log";exit 1;fi
mapfile -t zaps_a39_gpus < "$zaps_a39_parent/devices.txt"
echo "A39 selected GPUs: ${zaps_a39_gpus[*]} (same ID means serial shards)."
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python - <<'PY' 9>&- >> "$zaps_a39_log" 2>&1
import sys,unittest
suite=unittest.defaultTestLoader.loadTestsFromNames(['utils.test_common_state_risk','utils.test_transfer_response_feedback','utils.test_experiment_gpu_selection'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
PY
then tail -n 55 "$zaps_a39_log";exit 1;fi
if ! python -u utils/diag_common_state_risk.py --prepare 9>&- >> "$zaps_a39_log" 2>&1;then tail -n 50 "$zaps_a39_log";exit 1;fi
zaps_a39_pids=()
zaps_a39_cleanup(){ for zaps_a39_pid in "${zaps_a39_pids[@]}";do kill -TERM "$zaps_a39_pid" 2>/dev/null || true;done; }
trap 'zaps_a39_cleanup;exit 130' INT TERM HUP
zaps_a39_launch(){
    env CUDA_VISIBLE_DEVICES="$2" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      python -u utils/diag_common_state_risk.py --shard "$1" --device cuda:0 \
      9>&- > "$zaps_a39_parent/shard${1}.log" 2>&1 &
    zaps_a39_last=$!;zaps_a39_pids+=("$zaps_a39_last")
}
zaps_a39_s0=0;zaps_a39_s1=0
zaps_a39_launch 0 "${zaps_a39_gpus[0]}";zaps_a39_first=$zaps_a39_last
if [ "${zaps_a39_gpus[0]}" = "${zaps_a39_gpus[1]}" ];then
    if wait "$zaps_a39_first";then :;else zaps_a39_s0=$?;fi
    zaps_a39_pids=();zaps_a39_launch 1 "${zaps_a39_gpus[1]}"
    if wait "$zaps_a39_last";then :;else zaps_a39_s1=$?;fi
else
    zaps_a39_launch 1 "${zaps_a39_gpus[1]}";zaps_a39_second=$zaps_a39_last
    if wait "$zaps_a39_first";then :;else zaps_a39_s0=$?;fi
    zaps_a39_pids=("$zaps_a39_second")
    if wait "$zaps_a39_second";then :;else zaps_a39_s1=$?;fi
fi
zaps_a39_pids=();trap - INT TERM HUP
if ! python -u utils/diag_common_state_risk.py --archive 9>&- > "$zaps_a39_parent/final.log" 2>&1;then
    tail -n 35 "$zaps_a39_parent/final.log"
    if [ "$zaps_a39_s0" -ne 0 ];then tail -n 35 "$zaps_a39_parent/shard0.log";fi
    if [ "$zaps_a39_s1" -ne 0 ];then tail -n 35 "$zaps_a39_parent/shard1.log";fi
    exit 1
fi
cat "$zaps_a39_parent/final.log"
if git push origin results/refinement-confirmation-20261020 9>&- >> "$zaps_a39_log" 2>&1;then
    echo 'RESULTS_PUSH_PASS: A39 common-state risk records and all comparison images uploaded.'
else echo 'RESULTS_PUSH_PENDING: results retained; no GPU rerun required.';tail -n 12 "$zaps_a39_log";fi
if [ "$zaps_a39_s0" -ne 0 ] || [ "$zaps_a39_s1" -ne 0 ];then
    tail -n 40 "$zaps_a39_parent/shard0.log";tail -n 40 "$zaps_a39_parent/shard1.log";exit 1
fi
