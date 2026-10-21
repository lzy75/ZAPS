#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a34_parent=/data0/lzy/ZAPS-results/diag_noise_target_response_feedback
mkdir -p "$zaps_a34_parent"
exec 9>"$zaps_a34_parent/launcher.lock"
if ! flock -n 9; then echo 'A34 already running; duplicate refused.' >&2; exit 1; fi
zaps_a34_log="$zaps_a34_parent/launcher_20261021.log"
echo 'A34: noise-target feedback .9/1/1.1 plus matched time controls; 2160 diagnostic NFE, all arms 300 NFE. Waiting for final summary.'
if ! python - <<'PY' > "$zaps_a34_log" 2>&1
import sys,unittest
suite=unittest.defaultTestLoader.loadTestsFromName('utils.test_noise_target_response_feedback')
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
PY
then tail -n 45 "$zaps_a34_log"; exit 1; fi
if ! python -u utils/diag_noise_target_response_feedback.py --prepare >> "$zaps_a34_log" 2>&1; then tail -n 45 "$zaps_a34_log"; exit 1; fi
if ! python - <<'PY' > "$zaps_a34_parent/devices.txt" 2>> "$zaps_a34_log"
import os,subprocess
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
memory={};indices=[]
for line in raw.splitlines():
    i,u,f=[v.strip() for v in line.split(',')];memory[i]=memory[u]=int(f);indices.append(i)
mask=os.environ.get('CUDA_VISIBLE_DEVICES');allowed=mask.split(',') if mask is not None else indices
if not allowed or any(g not in memory for g in allowed):raise SystemExit('No supported allocated GPUs')
selected=[os.environ.get('ZAPS_A34_GPU0',allowed[0]),os.environ.get('ZAPS_A34_GPU1',allowed[1] if len(allowed)>1 else allowed[0])]
for gpu in selected:
    if gpu not in allowed or memory[gpu]<6144:raise SystemExit('GPU outside allocation or below 6GiB free: '+gpu)
    print(gpu)
PY
then tail -n 35 "$zaps_a34_log"; exit 1; fi
mapfile -t zaps_a34_gpus < "$zaps_a34_parent/devices.txt"
zaps_a34_pids=()
zaps_a34_cleanup() { for zaps_a34_pid in "${zaps_a34_pids[@]}"; do kill -TERM "$zaps_a34_pid" 2>/dev/null || true; done; }
trap 'zaps_a34_cleanup; exit 130' INT TERM HUP
zaps_a34_launch() {
    env CUDA_VISIBLE_DEVICES="$3" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -u utils/diag_noise_target_response_feedback.py \
        --phase "$1" --shard "$2" --device cuda:0 > "$zaps_a34_parent/${1}_shard${2}.log" 2>&1 &
    zaps_a34_last=$!;zaps_a34_pids+=("$zaps_a34_last")
}
zaps_a34_partial=0
for zaps_a34_phase in feedback stage; do
    if [ "$zaps_a34_phase" = stage ]; then
        if ! python -u utils/diag_noise_target_response_feedback.py --build-profiles >> "$zaps_a34_log" 2>&1; then
            tail -n 45 "$zaps_a34_log"; exit 1
        fi
    fi
    zaps_a34_status0=0;zaps_a34_status1=0
    zaps_a34_launch "$zaps_a34_phase" 0 "${zaps_a34_gpus[0]}";zaps_a34_first=$zaps_a34_last
    if [ "${zaps_a34_gpus[0]}" = "${zaps_a34_gpus[1]}" ]; then
        if wait "$zaps_a34_first"; then :; else zaps_a34_status0=$?; fi
        zaps_a34_pids=();zaps_a34_launch "$zaps_a34_phase" 1 "${zaps_a34_gpus[1]}"
        if wait "$zaps_a34_last"; then :; else zaps_a34_status1=$?; fi
    else
        zaps_a34_launch "$zaps_a34_phase" 1 "${zaps_a34_gpus[1]}";zaps_a34_second=$zaps_a34_last
        if wait "$zaps_a34_first"; then :; else zaps_a34_status0=$?; fi
        zaps_a34_pids=("$zaps_a34_second")
        if wait "$zaps_a34_second"; then :; else zaps_a34_status1=$?; fi
    fi
    zaps_a34_pids=()
    if [ "$zaps_a34_status0" -ne 0 ] || [ "$zaps_a34_status1" -ne 0 ]; then
        zaps_a34_partial=1
        echo "A34 $zaps_a34_phase has incomplete arms; complete targets continue." >> "$zaps_a34_log"
        tail -n 30 "$zaps_a34_parent/${zaps_a34_phase}_shard0.log" >> "$zaps_a34_log"
        tail -n 30 "$zaps_a34_parent/${zaps_a34_phase}_shard1.log" >> "$zaps_a34_log"
    fi
done
trap - INT TERM HUP
if ! python -u utils/diag_noise_target_response_feedback.py --archive > "$zaps_a34_parent/final.log" 2>&1; then
    tail -n 55 "$zaps_a34_parent/final.log"; exit 1
fi
cat "$zaps_a34_parent/final.log"
if git push origin results/refinement-confirmation-20261020 >> "$zaps_a34_log" 2>&1; then
    echo 'RESULTS_PUSH_PASS: A34 noise-target comparisons, state traces and real images uploaded.'
else echo 'RESULTS_PUSH_PENDING: results retained; no experiment rerun required.';tail -n 12 "$zaps_a34_log";fi
if [ "$zaps_a34_partial" -ne 0 ]; then echo "Some targets incomplete; see $zaps_a34_log."; exit 1; fi
