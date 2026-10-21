#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a32_parent=/data0/lzy/ZAPS-results/diag_fixed_grid_response_feedback
mkdir -p "$zaps_a32_parent"
exec 9>"$zaps_a32_parent/launcher.lock"
if ! flock -n 9; then echo 'A32 already running; duplicate refused.' >&2; exit 1; fi
zaps_a32_log="$zaps_a32_parent/launcher_20261021.log"
echo 'A32: fixed grid; 6 images x2 seeds; state/phase/ablation controls. 1800 diagnostic NFE; auto-export real comparison images. Waiting for final summary.'
if ! python - <<'PY' > "$zaps_a32_log" 2>&1
import sys,unittest
suite=unittest.defaultTestLoader.loadTestsFromName('utils.test_fixed_grid_response_feedback')
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
PY
then tail -n 40 "$zaps_a32_log"; exit 1; fi
if ! python -u utils/diag_fixed_grid_response_feedback.py --prepare >> "$zaps_a32_log" 2>&1; then tail -n 40 "$zaps_a32_log"; exit 1; fi
if ! python - <<'PY' > "$zaps_a32_parent/devices.txt" 2>> "$zaps_a32_log"
import os,subprocess
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
memory={};indices=[]
for line in raw.splitlines():
 i,u,f=[v.strip() for v in line.split(',')];memory[i]=memory[u]=int(f);indices.append(i)
mask=os.environ.get('CUDA_VISIBLE_DEVICES');allowed=mask.split(',') if mask is not None else indices
if not allowed or any(g not in memory for g in allowed):raise SystemExit('No supported allocated GPUs')
selected=[os.environ.get('ZAPS_A32_GPU0',allowed[0]),os.environ.get('ZAPS_A32_GPU1',allowed[1] if len(allowed)>1 else allowed[0])]
for gpu in selected:
 if gpu not in allowed or memory[gpu]<6144:raise SystemExit('GPU outside allocation or below 6GiB free: '+gpu)
 print(gpu)
PY
then tail -n 30 "$zaps_a32_log"; exit 1; fi
mapfile -t zaps_a32_gpus < "$zaps_a32_parent/devices.txt"
zaps_a32_pids=()
zaps_a32_cleanup() { for zaps_a32_pid in "${zaps_a32_pids[@]}"; do kill -TERM "$zaps_a32_pid" 2>/dev/null || true; done; }
trap 'zaps_a32_cleanup; exit 130' INT TERM HUP
zaps_a32_launch() {
 env CUDA_VISIBLE_DEVICES="$2" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -u utils/diag_fixed_grid_response_feedback.py --shard "$1" --device cuda:0 > "$zaps_a32_parent/shard${1}.log" 2>&1 &
 zaps_a32_last=$!;zaps_a32_pids+=("$zaps_a32_last")
}
zaps_a32_s0=0;zaps_a32_s1=0
zaps_a32_launch 0 "${zaps_a32_gpus[0]}";zaps_a32_first=$zaps_a32_last
if [ "${zaps_a32_gpus[0]}" = "${zaps_a32_gpus[1]}" ]; then
 if wait "$zaps_a32_first"; then :; else zaps_a32_s0=$?; fi
 zaps_a32_pids=();zaps_a32_launch 1 "${zaps_a32_gpus[1]}"
 if wait "$zaps_a32_last"; then :; else zaps_a32_s1=$?; fi
else
 zaps_a32_launch 1 "${zaps_a32_gpus[1]}";zaps_a32_second=$zaps_a32_last
 if wait "$zaps_a32_first"; then :; else zaps_a32_s0=$?; fi
 zaps_a32_pids=("$zaps_a32_second")
 if wait "$zaps_a32_second"; then :; else zaps_a32_s1=$?; fi
fi
zaps_a32_pids=();trap - INT TERM HUP
if [ "$zaps_a32_s0" -ne 0 ] || [ "$zaps_a32_s1" -ne 0 ]; then
 echo 'A32 incomplete; saved attempts retained. Failure details:'
 tail -n 35 "$zaps_a32_parent/shard0.log";tail -n 35 "$zaps_a32_parent/shard1.log"
 python -u utils/diag_fixed_grid_response_feedback.py || true
 exit 1
fi
if ! python -u utils/diag_fixed_grid_response_feedback.py --archive > "$zaps_a32_parent/final.log" 2>&1; then tail -n 45 "$zaps_a32_parent/final.log"; exit 1; fi
cat "$zaps_a32_parent/final.log"
if git push origin results/refinement-confirmation-20261020 >> "$zaps_a32_log" 2>&1; then
 echo 'RESULTS_PUSH_PASS: A32 metrics, state traces and real comparison images uploaded.'
else echo 'RESULTS_PUSH_PENDING: results saved; no experiment rerun needed.';tail -n 12 "$zaps_a32_log";fi
