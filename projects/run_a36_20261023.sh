#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a36_parent=/data0/lzy/ZAPS-results/diag_frozen_response_transfer
mkdir -p "$zaps_a36_parent"
exec 9>"$zaps_a36_parent/launcher.lock"
if ! flock -n 9; then echo 'A36 already running; duplicate refused.' >&2; exit 1; fi
zaps_a36_log="$zaps_a36_parent/launcher_20261023.log"
echo 'A36: six unused images x three noise levels x one seed; frozen C/TIME/STATE comparison. 7020 new calls, two GPU shards; waiting for final summary.'
if ! python - <<'PY' > "$zaps_a36_log" 2>&1
import sys,unittest
suite=unittest.defaultTestLoader.loadTestsFromNames(['utils.test_transfer_response_feedback','utils.test_continuous_response_feedback'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
PY
then tail -n 45 "$zaps_a36_log";exit 1;fi
if ! python -u utils/diag_frozen_response_transfer.py --prepare >> "$zaps_a36_log" 2>&1;then tail -n 45 "$zaps_a36_log";exit 1;fi
if ! python - <<'PY' > "$zaps_a36_parent/devices.txt" 2>> "$zaps_a36_log"
import os,subprocess
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
memory={};indices=[]
for line in raw.splitlines():
    i,u,f=[v.strip() for v in line.split(',')];memory[i]=memory[u]=int(f);indices.append(i)
mask=os.environ.get('CUDA_VISIBLE_DEVICES');allowed=mask.split(',') if mask is not None else indices
if not allowed or any(g not in memory for g in allowed):raise SystemExit('No supported allocated GPUs')
selected=[os.environ.get('ZAPS_A36_GPU0',allowed[0]),os.environ.get('ZAPS_A36_GPU1',allowed[1] if len(allowed)>1 else allowed[0])]
for gpu in selected:
    if gpu not in allowed or memory[gpu]<6144:raise SystemExit('GPU outside allocation or below 6GiB free: '+gpu)
    print(gpu)
PY
then tail -n 35 "$zaps_a36_log";exit 1;fi
mapfile -t zaps_a36_gpus < "$zaps_a36_parent/devices.txt"
zaps_a36_pids=()
zaps_a36_cleanup(){ for zaps_a36_pid in "${zaps_a36_pids[@]}";do kill -TERM "$zaps_a36_pid" 2>/dev/null || true;done; }
trap 'zaps_a36_cleanup;exit 130' INT TERM HUP
zaps_a36_launch(){
    env CUDA_VISIBLE_DEVICES="$2" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      python -u utils/diag_frozen_response_transfer.py --shard "$1" --device cuda:0 > "$zaps_a36_parent/shard${1}.log" 2>&1 &
    zaps_a36_last=$!;zaps_a36_pids+=("$zaps_a36_last")
}
zaps_a36_s0=0;zaps_a36_s1=0
zaps_a36_launch 0 "${zaps_a36_gpus[0]}";zaps_a36_first=$zaps_a36_last
if [ "${zaps_a36_gpus[0]}" = "${zaps_a36_gpus[1]}" ];then
    if wait "$zaps_a36_first";then :;else zaps_a36_s0=$?;fi
    zaps_a36_pids=();zaps_a36_launch 1 "${zaps_a36_gpus[1]}"
    if wait "$zaps_a36_last";then :;else zaps_a36_s1=$?;fi
else
    zaps_a36_launch 1 "${zaps_a36_gpus[1]}";zaps_a36_second=$zaps_a36_last
    if wait "$zaps_a36_first";then :;else zaps_a36_s0=$?;fi
    zaps_a36_pids=("$zaps_a36_second")
    if wait "$zaps_a36_second";then :;else zaps_a36_s1=$?;fi
fi
zaps_a36_pids=();trap - INT TERM HUP
if ! python -u utils/diag_frozen_response_transfer.py --archive > "$zaps_a36_parent/final.log" 2>&1;then
    tail -n 55 "$zaps_a36_parent/final.log";exit 1
fi
cat "$zaps_a36_parent/final.log"
if git push origin results/refinement-confirmation-20261020 >> "$zaps_a36_log" 2>&1;then
    echo 'RESULTS_PUSH_PASS: A36 frozen-policy transfer records and images uploaded.'
else echo 'RESULTS_PUSH_PENDING: results retained; no GPU rerun required.';tail -n 12 "$zaps_a36_log";fi
if [ "$zaps_a36_s0" -ne 0 ] || [ "$zaps_a36_s1" -ne 0 ];then
    tail -n 35 "$zaps_a36_parent/shard0.log";tail -n 35 "$zaps_a36_parent/shard1.log";exit 1
fi
