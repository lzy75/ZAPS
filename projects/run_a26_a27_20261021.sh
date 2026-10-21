#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_scan_parent=/data0/lzy/ZAPS-results/diag_state_weight_scan
mkdir -p "$zaps_scan_parent"
exec 9>"$zaps_scan_parent/launcher.lock"
if ! flock -n 9; then echo 'A26/A27 already running; duplicate refused.' >&2; exit 1; fi
zaps_scan_log="$zaps_scan_parent/launcher_20261021.log"
echo 'A26/A27: 6 calibration +12 confirmation images x2 seeds; 27 scan arms then six locked arms; about 21240 new calls. Waiting for combined output.'
if ! python - <<'PY'
import sys,unittest
suite=unittest.defaultTestLoader.loadTestsFromNames(['utils.test_state_weight_scan','utils.test_diag_equal_nfe_path_opportunity.CoreTests'])
r=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if r.wasSuccessful() and not r.skipped else 1)
PY
then echo 'Preflight failed; no GPU experiment started.'; exit 1; fi
if ! python -u utils/diag_state_weight_scan.py --prepare --validation-images 12 >> "$zaps_scan_log" 2>&1; then tail -n 40 "$zaps_scan_log"; exit 1; fi
if ! python - <<'PY' > "$zaps_scan_parent/devices.txt" 2>> "$zaps_scan_log"
import os,subprocess
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
gpus={};canonical={};indices=[]
for line in raw.splitlines():
    index,uuid,free=[v.strip() for v in line.split(',')]
    gpus[index]=gpus[uuid]=int(free);canonical[index]=canonical[uuid]=index;indices.append(index)
mask=os.environ.get('CUDA_VISIBLE_DEVICES');allowed=[v.strip() for v in mask.split(',')] if mask is not None else indices
if not allowed or any(v not in gpus for v in allowed):raise SystemExit('No supported visible GPUs.')
a=os.environ.get('ZAPS_SCAN_GPU0',allowed[0]);b=os.environ.get('ZAPS_SCAN_GPU1',allowed[1] if len(allowed)>1 else allowed[0])
for gpu in (a,b):
    if gpu not in allowed or gpus[gpu]<6144:raise SystemExit('Requested GPU unavailable/outside allocation or less than 6GiB free: '+gpu)
print(canonical[a]);print(canonical[b])
PY
then tail -n 20 "$zaps_scan_log"; exit 1; fi
mapfile -t zaps_scan_gpus < "$zaps_scan_parent/devices.txt"
zaps_scan_pids=()
zaps_scan_cleanup() {
  for zaps_scan_pid in "${zaps_scan_pids[@]}"; do kill -TERM "$zaps_scan_pid" 2>/dev/null || true; done
  for zaps_scan_pid in "${zaps_scan_pids[@]}"; do wait "$zaps_scan_pid" 2>/dev/null || true; done
}
trap 'zaps_scan_cleanup; exit 130' INT TERM HUP
zaps_scan_launch() {
  env CUDA_VISIBLE_DEVICES="$3" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python -u utils/diag_state_weight_scan.py --phase "$1" --shard "$2" --device cuda:0 \
    >> "$zaps_scan_parent/${1}_shard${2}_20261021.log" 2>&1 &
  zaps_scan_last=$!;zaps_scan_pids+=("$zaps_scan_last")
}
zaps_scan_phase() {
  local stage="$1" status0=0 status1=0 first second
  echo "A26/A27 $stage: GPUs ${zaps_scan_gpus[0]}/${zaps_scan_gpus[1]}; full logs retained."
  zaps_scan_launch "$stage" 0 "${zaps_scan_gpus[0]}";first=$zaps_scan_last
  if [ "${zaps_scan_gpus[0]}" = "${zaps_scan_gpus[1]}" ]; then
    if wait "$first"; then :; else status0=$?; fi
    zaps_scan_pids=()
    zaps_scan_launch "$stage" 1 "${zaps_scan_gpus[1]}"
    if wait "$zaps_scan_last"; then :; else status1=$?; fi
  else
    zaps_scan_launch "$stage" 1 "${zaps_scan_gpus[1]}";second=$zaps_scan_last
    if wait "$first"; then :; else status0=$?; fi
    zaps_scan_pids=("$second")
    if wait "$second"; then :; else status1=$?; fi
  fi
  zaps_scan_pids=()
  if [ "$status0" -ne 0 ] || [ "$status1" -ne 0 ]; then
    tail -n 40 "$zaps_scan_parent/${stage}_shard0_20261021.log"
    tail -n 40 "$zaps_scan_parent/${stage}_shard1_20261021.log"
    return 1
  fi
}
zaps_scan_status=0
if zaps_scan_phase calibration; then
  if python -u utils/diag_state_weight_scan.py --select >> "$zaps_scan_log" 2>&1; then
    if zaps_scan_phase validation; then :; else zaps_scan_status=1; fi
  else tail -n 40 "$zaps_scan_log";zaps_scan_status=1; fi
else zaps_scan_status=1; fi
zaps_scan_pids=();trap - INT TERM HUP
if ! python -u utils/diag_state_weight_scan.py --archive >> "$zaps_scan_log" 2>&1; then
  echo 'Archive pending; records retained. Completed arms need no GPU rerun.';tail -n 40 "$zaps_scan_log";exit 1
fi
python -u utils/diag_state_weight_scan.py --summary
if git push origin results/refinement-confirmation-20261020 >> "$zaps_scan_log" 2>&1; then
  echo 'RESULTS_PUSH_PASS: A26/A27 scan and locked confirmation appended.'
else echo 'RESULTS_PUSH_PENDING: all records retained; no rerun of completed arms needed.';tail -n 10 "$zaps_scan_log";fi
exit "$zaps_scan_status"
