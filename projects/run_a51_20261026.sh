#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a51_base=/data0/lzy/ZAPS-results/diag_score_cg_metrology
mkdir -p "$zaps_a51_base"
exec 9>"$zaps_a51_base/launcher.lock"
if ! flock -n 9; then echo 'A51 already running; duplicate refused.' >&2; exit 1; fi
zaps_a51_log="$zaps_a51_base/launcher_20261026.log"
echo 'A51: 32 operator-only solves; NO UNet/model loading. Numerical tests first; progress every 60 seconds; final summary printed here.'
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest utils.test_score_cg_metrology -v 9>&- > "$zaps_a51_log" 2>&1; then tail -n 65 "$zaps_a51_log"; exit 1; fi
if ! python - <<'PY' 9>&- > "$zaps_a51_base/device.txt" 2>> "$zaps_a51_log"
import os,subprocess
rows=[]
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
visible=os.environ.get('CUDA_VISIBLE_DEVICES')
allowed=None if visible is None else {x.strip() for x in visible.split(',')}
for line in raw.splitlines():
    i,u,f=[x.strip() for x in line.split(',')]
    if (allowed is None or i in allowed or u in allowed) and int(f)>=2048:rows.append((int(f),i))
if not rows:raise RuntimeError('No GPU within CUDA_VISIBLE_DEVICES with >=2GiB free. No allocation override.')
print(max(rows)[1])
PY
then tail -n 65 "$zaps_a51_log"; exit 1; fi
zaps_a51_gpu=$(cat "$zaps_a51_base/device.txt")
echo "A51 selected GPU $zaps_a51_gpu; no reconstruction trajectories will run."
env CUDA_VISIBLE_DEVICES="$zaps_a51_gpu" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -u utils/diag_score_cg_metrology.py --device cuda:0 9>&- >> "$zaps_a51_log" 2>&1 &
zaps_a51_pid=$!
trap 'kill -TERM "$zaps_a51_pid" 2>/dev/null || true' EXIT
trap 'exit 130' INT TERM HUP
while kill -0 "$zaps_a51_pid" 2>/dev/null; do
  sleep 5
  zaps_a51_ticks=$(( ${zaps_a51_ticks:-0} + 1 ))
  if [ "$((zaps_a51_ticks % 12))" -eq 0 ]; then tail -n 1 "$zaps_a51_log"; fi
done
if wait "$zaps_a51_pid"; then :; else tail -n 65 "$zaps_a51_log"; exit 1; fi
trap - EXIT
tail -n 2 "$zaps_a51_log"
