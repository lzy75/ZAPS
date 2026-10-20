#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a22_parent=/data0/lzy/ZAPS-results/diag_equal_nfe_path_opportunity
mkdir -p "$zaps_a22_parent"
exec 9>"$zaps_a22_parent/launcher.lock"
if ! flock -n 9; then echo 'A22 already running; duplicate refused.' >&2; exit 1; fi
zaps_a22_log="$zaps_a22_parent/launcher_20261020.log"
echo 'A22: equal-300-NFE phase opportunity; no retraining, 1296 diagnostic calls. Waiting for combined output.'
if ! python - <<'PY'
import sys,unittest
suite=unittest.defaultTestLoader.loadTestsFromNames(['utils.test_diag_equal_nfe_path_opportunity'])
r=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if r.wasSuccessful() and not r.skipped else 1)
PY
then echo 'Preflight failed; no GPU experiment started.'; exit 1; fi
if ! python -u utils/diag_equal_nfe_path_opportunity.py --prepare >> "$zaps_a22_log" 2>&1; then
  tail -n 40 "$zaps_a22_log"; exit 1
fi
if ! python - <<'PY' > "$zaps_a22_parent/devices.txt" 2>> "$zaps_a22_log"
import os,subprocess
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
gpus={};canonical={};indices=[]
for line in raw.splitlines():
    index,uuid,free=[v.strip() for v in line.split(',')]
    gpus[index]=gpus[uuid]=int(free);canonical[index]=canonical[uuid]=index;indices.append(index)
mask=os.environ.get('CUDA_VISIBLE_DEVICES');allowed=[v.strip() for v in mask.split(',')] if mask is not None else indices
if not allowed or any(v not in gpus for v in allowed):raise SystemExit('No supported visible GPUs.')
a=os.environ.get('ZAPS_A22_GPU0',allowed[0]);b=os.environ.get('ZAPS_A22_GPU1',allowed[1] if len(allowed)>1 else allowed[0])
for gpu in (a,b):
    if gpu not in allowed or gpus[gpu]<6144:raise SystemExit('Requested GPU unavailable/outside allocation or less than 6GiB free: '+gpu)
print(canonical[a]);print(canonical[b])
PY
then tail -n 20 "$zaps_a22_log"; exit 1; fi
mapfile -t zaps_a22_gpus < "$zaps_a22_parent/devices.txt"
zaps_a22_pids=()
zaps_a22_cleanup() {
  for zaps_a22_pid in "${zaps_a22_pids[@]}"; do kill -TERM "$zaps_a22_pid" 2>/dev/null || true; done
  for zaps_a22_pid in "${zaps_a22_pids[@]}"; do wait "$zaps_a22_pid" 2>/dev/null || true; done
}
trap 'zaps_a22_cleanup; exit 130' INT TERM HUP
zaps_a22_launch() {
  env CUDA_VISIBLE_DEVICES="$2" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python -u utils/diag_equal_nfe_path_opportunity.py --shard "$1" --device cuda:0 \
    >> "$zaps_a22_parent/shard${1}_20261020.log" 2>&1 &
  zaps_a22_last=$!
  zaps_a22_pids+=("$zaps_a22_last")
}
zaps_a22_status0=0
zaps_a22_status1=0
zaps_a22_launch 0 "${zaps_a22_gpus[0]}"
zaps_a22_pid0=$zaps_a22_last
if [ "${zaps_a22_gpus[0]}" = "${zaps_a22_gpus[1]}" ]; then
  echo 'One selected GPU: running shards sequentially.'
  if wait "$zaps_a22_pid0"; then :; else zaps_a22_status0=$?; fi
  zaps_a22_pids=()
  zaps_a22_launch 1 "${zaps_a22_gpus[1]}"
  if wait "$zaps_a22_last"; then :; else zaps_a22_status1=$?; fi
else
  echo "A22 tasks running in parallel on GPU ${zaps_a22_gpus[0]}/${zaps_a22_gpus[1]}."
  zaps_a22_launch 1 "${zaps_a22_gpus[1]}"
  zaps_a22_pid1=$zaps_a22_last
  if wait "$zaps_a22_pid0"; then :; else zaps_a22_status0=$?; fi
  zaps_a22_pids=("$zaps_a22_pid1")
  if wait "$zaps_a22_pid1"; then :; else zaps_a22_status1=$?; fi
fi
zaps_a22_pids=()
trap - INT TERM HUP
if ! python -u utils/diag_equal_nfe_path_opportunity.py --archive >> "$zaps_a22_log" 2>&1; then
  echo 'Archive pending; records retained. No GPU rerun needed.'
  tail -n 40 "$zaps_a22_log"; exit 1
fi
python -u utils/diag_equal_nfe_path_opportunity.py --summary
if [ "$zaps_a22_status0" -ne 0 ] || [ "$zaps_a22_status1" -ne 0 ]; then
  tail -n 35 "$zaps_a22_parent/shard0_20261020.log"
  tail -n 35 "$zaps_a22_parent/shard1_20261020.log"
  exit 1
fi
if git push origin results/refinement-confirmation-20261020 >> "$zaps_a22_log" 2>&1; then
  echo 'RESULTS_PUSH_PASS: A22 equal-NFE path opportunity appended; no state policy claim.'
else
  echo 'RESULTS_PUSH_PENDING: completed records retained; no GPU rerun needed.'
  tail -n 10 "$zaps_a22_log"
fi
