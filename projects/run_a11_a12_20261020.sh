#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_response_root=/data0/lzy/ZAPS-results/diag_clean_field_response
mkdir -p "$zaps_response_root"
exec 9>"$zaps_response_root/launcher.lock"
if ! flock -n 9; then echo 'A11/A12 already running; duplicate refused.' >&2; exit 1; fi
zaps_response_log="$zaps_response_root/launcher_20261020.log"
echo 'A11/A12: auditing cached states once; two independent stages, 120 diagnostic NFE total. Waiting for combined output.'
if ! (
  python - <<'PY'
import sys,unittest
suite=unittest.defaultTestLoader.loadTestsFromName('utils.test_diag_clean_field_response')
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
PY
); then echo 'CPU preflight failed; no GPU experiment started.'; exit 1; fi
if ! python -u utils/diag_clean_field_response.py --prepare >> "$zaps_response_log" 2>&1; then
  tail -n 40 "$zaps_response_log"; exit 1
fi
if ! python - <<'PY' > "$zaps_response_root/devices.txt" 2>> "$zaps_response_log"
import os,subprocess
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
gpus={};indices=[]
for line in raw.splitlines():
    index,uuid,free=[v.strip() for v in line.split(',')]
    gpus[index]=gpus[uuid]=int(free);indices.append(index)
visible=os.environ.get('CUDA_VISIBLE_DEVICES')
allowed=[v.strip() for v in visible.split(',')] if visible is not None else indices
if not allowed or any(v not in gpus for v in allowed):raise SystemExit('No supported visible GPUs; inspect CUDA_VISIBLE_DEVICES.')
a=os.environ.get('ZAPS_A11_GPU',allowed[0]);b=os.environ.get('ZAPS_A12_GPU',allowed[1] if len(allowed)>1 else allowed[0])
for gpu in (a,b):
    if gpu not in allowed:raise SystemExit('Requested GPU outside visible allocation: '+gpu)
    if gpus[gpu]<6144:raise SystemExit(f'GPU {gpu} has {gpus[gpu]} MiB free; need >=6144. No experiment started.')
print(a);print(b)
PY
then tail -n 20 "$zaps_response_log"; exit 1; fi
mapfile -t zaps_response_gpus < "$zaps_response_root/devices.txt"
zaps_response_pids=()
zaps_response_cleanup() {
  for zaps_response_pid in "${zaps_response_pids[@]}"; do kill -TERM "$zaps_response_pid" 2>/dev/null || true; done
  for zaps_response_pid in "${zaps_response_pids[@]}"; do wait "$zaps_response_pid" 2>/dev/null || true; done
}
trap 'zaps_response_cleanup; exit 130' INT TERM HUP
zaps_response_launch() {
  env CUDA_VISIBLE_DEVICES="$2" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python -u utils/diag_clean_field_response.py --stage "$1" --device cuda:0 \
    >> "$zaps_response_root/t${1}_20261020.log" 2>&1 &
  zaps_response_last_pid=$!
  zaps_response_pids+=("$zaps_response_last_pid")
}
zaps_response_status14=0
zaps_response_status54=0
zaps_response_launch 14 "${zaps_response_gpus[0]}"
zaps_response_pid14=$zaps_response_last_pid
if [ "${zaps_response_gpus[0]}" = "${zaps_response_gpus[1]}" ]; then
  echo "One selected GPU (${zaps_response_gpus[0]}): running A11 then A12."
  if wait "$zaps_response_pid14"; then :; else zaps_response_status14=$?; fi
  zaps_response_pids=()
  zaps_response_launch 54 "${zaps_response_gpus[1]}"
  if wait "$zaps_response_last_pid"; then :; else zaps_response_status54=$?; fi
else
  echo "Running A11 on GPU ${zaps_response_gpus[0]}, A12 on GPU ${zaps_response_gpus[1]} in parallel."
  zaps_response_launch 54 "${zaps_response_gpus[1]}"
  zaps_response_pid54=$zaps_response_last_pid
  if wait "$zaps_response_pid14"; then :; else zaps_response_status14=$?; fi
  zaps_response_pids=("$zaps_response_pid54")
  if wait "$zaps_response_pid54"; then :; else zaps_response_status54=$?; fi
fi
zaps_response_pids=()
trap - INT TERM HUP
if ! python -u utils/diag_clean_field_response.py --archive >> "$zaps_response_log" 2>&1; then
  echo 'Archive pending; experiment records retained. No GPU rerun needed.'
  tail -n 35 "$zaps_response_log"
  exit 1
fi
python -u utils/diag_clean_field_response.py --summary
if [ "$zaps_response_status14" -ne 0 ] || [ "$zaps_response_status54" -ne 0 ]; then
  echo 'One or both stages failed; partial costs retained; do not resubmit unchanged.'
  tail -n 30 "$zaps_response_root/t14_20261020.log"
  tail -n 30 "$zaps_response_root/t54_20261020.log"
  exit 1
fi
if git push origin results/refinement-confirmation-20261020 >> "$zaps_response_log" 2>&1; then
  echo 'RESULTS_PUSH_PASS: A11/A12 archived together; source experiments preserved.'
else
  echo 'RESULTS_PUSH_PENDING: complete results retained; no GPU rerun needed.'
  tail -n 10 "$zaps_response_log"
fi
