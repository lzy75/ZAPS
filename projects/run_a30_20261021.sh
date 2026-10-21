#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a30_parent=/data0/lzy/ZAPS-results/diag_final_allocation_value
mkdir -p "$zaps_a30_parent"
exec 9>"$zaps_a30_parent/launcher.lock"
if ! flock -n 9; then echo 'A30 already running; duplicate refused.' >&2; exit 1; fi
zaps_a30_log="$zaps_a30_parent/launcher_20261021.log"
echo 'A30: 6 reused images x2 seeds x6 placements; 864 diagnostic NFE; candidate paths all 301 calls; GPUs run in parallel. Waiting for combined result.'
if ! python - <<'PY' >> "$zaps_a30_log" 2>&1
import sys, unittest
suite = unittest.defaultTestLoader.loadTestsFromName('utils.test_final_allocation_value')
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
PY
then echo 'A30 Torch/analysis preflight failed; no UNet experiment started.'; tail -n 45 "$zaps_a30_log"; exit 1; fi
if ! python -u utils/diag_final_allocation_value.py --prepare >> "$zaps_a30_log" 2>&1; then
  tail -n 45 "$zaps_a30_log"; exit 1
fi
if ! python - <<'PY' > "$zaps_a30_parent/devices.txt" 2>> "$zaps_a30_log"
import os, subprocess
raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
memory, canonical, indices = {}, {}, []
for line in raw.splitlines():
    index, uuid, free = [v.strip() for v in line.split(',')]
    memory[index] = memory[uuid] = int(free)
    canonical[index] = canonical[uuid] = index; indices.append(index)
mask = os.environ.get('CUDA_VISIBLE_DEVICES')
allowed = [v.strip() for v in mask.split(',')] if mask is not None else indices
if not allowed or any(v not in memory for v in allowed):
    raise SystemExit('No supported allocated visible GPUs.')
first = os.environ.get('ZAPS_A30_GPU0', allowed[0])
second = os.environ.get('ZAPS_A30_GPU1', allowed[1] if len(allowed)>1 else allowed[0])
for gpu in (first, second):
    if gpu not in allowed or memory[gpu] < 6144:
        raise SystemExit('GPU unavailable/outside allocation or less than 6GiB free: '+gpu)
    print(canonical[gpu])
PY
then tail -n 25 "$zaps_a30_log"; exit 1; fi
mapfile -t zaps_a30_gpus < "$zaps_a30_parent/devices.txt"
zaps_a30_pids=()
zaps_a30_cleanup() {
  for zaps_a30_pid in "${zaps_a30_pids[@]}"; do kill -TERM "$zaps_a30_pid" 2>/dev/null || true; done
  for zaps_a30_pid in "${zaps_a30_pids[@]}"; do wait "$zaps_a30_pid" 2>/dev/null || true; done
}
trap 'zaps_a30_cleanup; exit 130' INT TERM HUP
zaps_a30_launch() {
  env CUDA_VISIBLE_DEVICES="$2" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python -u utils/diag_final_allocation_value.py --shard "$1" --device cuda:0 \
    >> "$zaps_a30_parent/shard${1}_20261021.log" 2>&1 &
  zaps_a30_last=$!; zaps_a30_pids+=("$zaps_a30_last")
}
zaps_a30_status0=0; zaps_a30_status1=0
echo "A30 running on allocated GPUs ${zaps_a30_gpus[0]}/${zaps_a30_gpus[1]}; full logs retained."
zaps_a30_launch 0 "${zaps_a30_gpus[0]}"; zaps_a30_first=$zaps_a30_last
if [ "${zaps_a30_gpus[0]}" = "${zaps_a30_gpus[1]}" ]; then
  if wait "$zaps_a30_first"; then :; else zaps_a30_status0=$?; fi
  zaps_a30_pids=(); zaps_a30_launch 1 "${zaps_a30_gpus[1]}"
  if wait "$zaps_a30_last"; then :; else zaps_a30_status1=$?; fi
else
  zaps_a30_launch 1 "${zaps_a30_gpus[1]}"; zaps_a30_second=$zaps_a30_last
  if wait "$zaps_a30_first"; then :; else zaps_a30_status0=$?; fi
  zaps_a30_pids=("$zaps_a30_second")
  if wait "$zaps_a30_second"; then :; else zaps_a30_status1=$?; fi
fi
zaps_a30_pids=(); trap - INT TERM HUP
if [ "$zaps_a30_status0" -ne 0 ] || [ "$zaps_a30_status1" -ne 0 ]; then
  echo 'A30 incomplete: failed attempts retained; completed points will be reused on rerun.'
  tail -n 40 "$zaps_a30_parent/shard0_20261021.log"
  tail -n 40 "$zaps_a30_parent/shard1_20261021.log"
fi
if ! python -u utils/diag_final_allocation_value.py --archive > "$zaps_a30_parent/final_20261021.log" 2>&1; then
  echo 'A30 summary/archive pending; no completed-point rerun needed.'
  tail -n 45 "$zaps_a30_parent/final_20261021.log"; exit 1
fi
cat "$zaps_a30_parent/final_20261021.log"
if git push origin results/refinement-confirmation-20261020 >> "$zaps_a30_log" 2>&1; then
  echo 'RESULTS_PUSH_PASS: A30 final allocation value appended.'
else
  echo 'RESULTS_PUSH_PENDING: results retained locally; no GPU rerun required.'; tail -n 10 "$zaps_a30_log"
fi
if [ "$zaps_a30_status0" -ne 0 ] || [ "$zaps_a30_status1" -ne 0 ]; then exit 1; fi
