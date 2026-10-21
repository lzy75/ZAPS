#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a25_parent=/data0/lzy/ZAPS-results/diag_state_clock_matching
mkdir -p "$zaps_a25_parent"
exec 9>"$zaps_a25_parent/launcher.lock"
if ! flock -n 9; then echo 'A25 already running; duplicate refused.' >&2; exit 1; fi
zaps_a25_log="$zaps_a25_parent/launcher_20261020.log"
echo 'A25: state clock matching, round 3/5; 360 diagnostic calls; no new policy selection. Waiting for combined output.'
if ! python - <<'PY'
import sys,unittest
suite=unittest.defaultTestLoader.loadTestsFromNames(['utils.test_diag_state_clock_matching','utils.test_diag_equal_nfe_path_opportunity.CoreTests'])
r=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if r.wasSuccessful() and not r.skipped else 1)
PY
then echo 'Preflight failed; no GPU experiment started.'; exit 1; fi
if ! python -u utils/diag_state_clock_matching.py --prepare >> "$zaps_a25_log" 2>&1; then
  tail -n 40 "$zaps_a25_log"; exit 1
fi
if ! python - <<'PY' > "$zaps_a25_parent/devices.txt" 2>> "$zaps_a25_log"
import os,subprocess
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
gpus={};canonical={};indices=[]
for line in raw.splitlines():
    index,uuid,free=[v.strip() for v in line.split(',')]
    gpus[index]=gpus[uuid]=int(free);canonical[index]=canonical[uuid]=index;indices.append(index)
mask=os.environ.get('CUDA_VISIBLE_DEVICES');allowed=[v.strip() for v in mask.split(',')] if mask is not None else indices
if not allowed or any(v not in gpus for v in allowed):raise SystemExit('No supported visible GPUs.')
a=os.environ.get('ZAPS_A25_GPU0',allowed[0]);b=os.environ.get('ZAPS_A25_GPU1',allowed[1] if len(allowed)>1 else allowed[0])
for gpu in (a,b):
    if gpu not in allowed or gpus[gpu]<6144:raise SystemExit('Requested GPU unavailable/outside allocation or less than 6GiB free: '+gpu)
print(canonical[a]);print(canonical[b])
PY
then tail -n 20 "$zaps_a25_log"; exit 1; fi
mapfile -t zaps_a25_gpus < "$zaps_a25_parent/devices.txt"
zaps_a25_pids=()
zaps_a25_cleanup() {
  for zaps_a25_pid in "${zaps_a25_pids[@]}"; do kill -TERM "$zaps_a25_pid" 2>/dev/null || true; done
  for zaps_a25_pid in "${zaps_a25_pids[@]}"; do wait "$zaps_a25_pid" 2>/dev/null || true; done
}
trap 'zaps_a25_cleanup; exit 130' INT TERM HUP
zaps_a25_launch() {
  env CUDA_VISIBLE_DEVICES="$2" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python -u utils/diag_state_clock_matching.py --shard "$1" --device cuda:0 \
    >> "$zaps_a25_parent/shard${1}_20261020.log" 2>&1 &
  zaps_a25_last=$!
  zaps_a25_pids+=("$zaps_a25_last")
}
zaps_a25_status0=0
zaps_a25_status1=0
zaps_a25_launch 0 "${zaps_a25_gpus[0]}"
zaps_a25_pid0=$zaps_a25_last
if [ "${zaps_a25_gpus[0]}" = "${zaps_a25_gpus[1]}" ]; then
  echo 'One selected GPU: running shards sequentially.'
  if wait "$zaps_a25_pid0"; then :; else zaps_a25_status0=$?; fi
  zaps_a25_pids=()
  zaps_a25_launch 1 "${zaps_a25_gpus[1]}"
  if wait "$zaps_a25_last"; then :; else zaps_a25_status1=$?; fi
else
  echo "A25 tasks running in parallel on GPU ${zaps_a25_gpus[0]}/${zaps_a25_gpus[1]}."
  zaps_a25_launch 1 "${zaps_a25_gpus[1]}"
  zaps_a25_pid1=$zaps_a25_last
  if wait "$zaps_a25_pid0"; then :; else zaps_a25_status0=$?; fi
  zaps_a25_pids=("$zaps_a25_pid1")
  if wait "$zaps_a25_pid1"; then :; else zaps_a25_status1=$?; fi
fi
zaps_a25_pids=()
trap - INT TERM HUP
if ! python -u utils/diag_state_clock_matching.py --archive >> "$zaps_a25_log" 2>&1; then
  echo 'Archive pending; records retained. No GPU rerun needed.'
  tail -n 40 "$zaps_a25_log"; exit 1
fi
python -u utils/diag_state_clock_matching.py --summary
if [ "$zaps_a25_status0" -ne 0 ] || [ "$zaps_a25_status1" -ne 0 ]; then
  tail -n 35 "$zaps_a25_parent/shard0_20261020.log"
  tail -n 35 "$zaps_a25_parent/shard1_20261020.log"
  exit 1
fi
if git push origin results/refinement-confirmation-20261020 >> "$zaps_a25_log" 2>&1; then
  echo 'RESULTS_PUSH_PASS: A25 state clock matching appended; donor clocks are retrospective diagnostic controls.'
else
  echo 'RESULTS_PUSH_PENDING: completed records retained; no GPU rerun needed.'
  tail -n 10 "$zaps_a25_log"
fi
