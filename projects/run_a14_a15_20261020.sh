#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_precision_parent=/data0/lzy/ZAPS-results/diag_ieee_response_value
mkdir -p "$zaps_precision_parent"
exec 9>"$zaps_precision_parent/launcher.lock"
if ! flock -n 9; then echo 'A14/A15 already running; duplicate refused.' >&2; exit 1; fi
zaps_precision_log="$zaps_precision_parent/launcher_20261020.log"
echo 'A14/A15: IEEE cohort then frozen transfer analysis, t14/t54; 220 new diagnostic NFE. Waiting for combined summary.'
if ! python - <<'PY'
import sys,unittest
suite=unittest.defaultTestLoader.loadTestsFromNames(['utils.test_diag_ieee_response_value','utils.test_diag_clean_response_precision','utils.test_diag_clean_field_response'])
r=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if r.wasSuccessful() and not r.skipped else 1)
PY
then echo 'CPU preflight failed; no GPU experiment started.'; exit 1; fi
if ! python -u utils/diag_ieee_response_value.py --prepare >> "$zaps_precision_log" 2>&1; then
  tail -n 40 "$zaps_precision_log"; exit 1
fi
if ! python - <<'PY' > "$zaps_precision_parent/devices.txt" 2>> "$zaps_precision_log"
import os,subprocess
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
gpus={};canonical={};indices=[]
for line in raw.splitlines():
    index,uuid,free=[v.strip() for v in line.split(',')]
    gpus[index]=gpus[uuid]=int(free);canonical[index]=canonical[uuid]=index;indices.append(index)
mask=os.environ.get('CUDA_VISIBLE_DEVICES');allowed=[v.strip() for v in mask.split(',')] if mask is not None else indices
if not allowed or any(v not in gpus for v in allowed):raise SystemExit('No supported visible GPUs.')
a=os.environ.get('ZAPS_A14_GPU14',allowed[0]);b=os.environ.get('ZAPS_A14_GPU54',allowed[1] if len(allowed)>1 else allowed[0])
for gpu in (a,b):
    if gpu not in allowed or gpus[gpu]<6144:raise SystemExit('Requested GPU unavailable/outside allocation or less than 6GiB free: '+gpu)
print(canonical[a]);print(canonical[b])
PY
then tail -n 20 "$zaps_precision_log"; exit 1; fi
mapfile -t zaps_precision_gpus < "$zaps_precision_parent/devices.txt"
zaps_precision_pids=()
zaps_precision_cleanup() {
  for zaps_precision_pid in "${zaps_precision_pids[@]}"; do kill -TERM "$zaps_precision_pid" 2>/dev/null || true; done
  for zaps_precision_pid in "${zaps_precision_pids[@]}"; do wait "$zaps_precision_pid" 2>/dev/null || true; done
}
trap 'zaps_precision_cleanup; exit 130' INT TERM HUP
zaps_precision_launch() {
  env CUDA_VISIBLE_DEVICES="$2" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python -u utils/diag_ieee_response_value.py --stage "$1" --device cuda:0 \
    >> "$zaps_precision_parent/t${1}_20261020.log" 2>&1 &
  zaps_precision_last=$!
  zaps_precision_pids+=("$zaps_precision_last")
}
zaps_precision_status14=0
zaps_precision_status54=0
zaps_precision_launch 14 "${zaps_precision_gpus[0]}"
zaps_precision_pid14=$zaps_precision_last
if [ "${zaps_precision_gpus[0]}" = "${zaps_precision_gpus[1]}" ]; then
  echo 'One selected GPU: running cohort stages sequentially.'
  if wait "$zaps_precision_pid14"; then :; else zaps_precision_status14=$?; fi
  zaps_precision_pids=()
  zaps_precision_launch 54 "${zaps_precision_gpus[1]}"
  if wait "$zaps_precision_last"; then :; else zaps_precision_status54=$?; fi
else
  echo "A14 IEEE cohort tasks running in parallel on GPU ${zaps_precision_gpus[0]}/${zaps_precision_gpus[1]}."
  zaps_precision_launch 54 "${zaps_precision_gpus[1]}"
  zaps_precision_pid54=$zaps_precision_last
  if wait "$zaps_precision_pid14"; then :; else zaps_precision_status14=$?; fi
  zaps_precision_pids=("$zaps_precision_pid54")
  if wait "$zaps_precision_pid54"; then :; else zaps_precision_status54=$?; fi
fi
zaps_precision_pids=()
trap - INT TERM HUP
if ! python -u utils/diag_ieee_response_value.py --archive >> "$zaps_precision_log" 2>&1; then
  echo 'Archive pending; GPU records retained. No experiment rerun needed.'
  tail -n 40 "$zaps_precision_log"; exit 1
fi
python -u utils/diag_ieee_response_value.py --summary
if [ "$zaps_precision_status14" -ne 0 ] || [ "$zaps_precision_status54" -ne 0 ]; then
  tail -n 30 "$zaps_precision_parent/t14_20261020.log"
  tail -n 30 "$zaps_precision_parent/t54_20261020.log"
  exit 1
fi
if git push origin results/refinement-confirmation-20261020 >> "$zaps_precision_log" 2>&1; then
  echo 'RESULTS_PUSH_PASS: A14/A15 cohort and frozen transfer analysis appended.'
else
  echo 'RESULTS_PUSH_PENDING: completed records retained; no GPU rerun needed.'
  tail -n 10 "$zaps_precision_log"
fi
