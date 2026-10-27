#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a56_base=/data0/lzy/ZAPS-results/diag_frozen_direction_transfer/operator_repair_20261027
mkdir -p "$zaps_a56_base"
if python -u utils/diag_direction_operator_repair.py --package A56 --install;then :
else
    zaps_a56_install_status=$?
    if [ "$zaps_a56_install_status" -eq 20 ];then exit 0;fi
    exit "$zaps_a56_install_status"
fi
exec 9>"/data0/lzy/ZAPS-results/diag_frozen_direction_transfer/launcher.lock"
if ! flock -n 9; then echo 'A56 already running; duplicate refused.' >&2; exit 1; fi
zaps_a56_log="$zaps_a56_base/launcher_20261026.log"
zaps_a56_pids=(); zaps_a56_monitor=''
zaps_a56_cleanup(){
    for zaps_a56_pid in "${zaps_a56_pids[@]}"; do kill -TERM "$zaps_a56_pid" 2>/dev/null || true; done
    if [ -n "$zaps_a56_monitor" ]; then kill -TERM "$zaps_a56_monitor" 2>/dev/null || true; fi
}
trap zaps_a56_cleanup EXIT
trap 'exit 130' INT TERM HUP
zaps_a56_fail(){ tail -n 80 "$1" 9>&-; exit 1; }
echo 'A56:freeze direction rule;6 unused FFHQ +4 unused ImageNet images x SR/IP;6000 method +3624 diagnostic forwards,<=1204 input backwards;original30-point grid.'
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m unittest \
  utils.test_direction_operator_repair \
  utils.test_frozen_direction_transfer utils.test_current_direction_control \
  utils.test_linear_adjoint_autograd_guard utils.test_current_score_checkpoint_r2 -v \
  9>&- > "$zaps_a56_log" 2>&1; then zaps_a56_fail "$zaps_a56_log"; fi
if ! python -u utils/diag_direction_operator_repair.py --package A56 --prepare 9>&- >> "$zaps_a56_log" 2>&1; then zaps_a56_fail "$zaps_a56_log"; fi
tail -n 1 "$zaps_a56_log"
if ! python - <<'PY' 9>&- > "$zaps_a56_base/devices.txt" 2>> "$zaps_a56_log"
import os,subprocess
from utils.frozen_direction_transfer import devices
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
rows=[]
for line in raw.splitlines():
    i,u,f=[x.strip() for x in line.split(',')];rows.append(dict(index=i,uuid=u,free=int(f)))
print('\n'.join(devices(rows,os.environ.get('CUDA_VISIBLE_DEVICES'),os.environ.get('ZAPS_A56_GPU0'),os.environ.get('ZAPS_A56_GPU1'))))
PY
then zaps_a56_fail "$zaps_a56_log"; fi
mapfile -t zaps_a56_gpus < "$zaps_a56_base/devices.txt"
echo "A56 allocated GPUs: ${zaps_a56_gpus[*]}; >=12GiB free; equal IDs mean serial shards."
(
    exec 9>&-
    while sleep 60; do
        python - <<'PY'
import json
from pathlib import Path
root=Path('/data0/lzy/ZAPS-results/diag_frozen_direction_transfer/A56_20261026')
rows=[]
for f in root.glob('*/*/attempt_*/run.json'):
    try:rows.append(json.loads(f.read_text()))
    except (OSError,ValueError):pass
print('A56 progress: baseline=',sum(r.get('kind')=='BASELINE' and r.get('status')=='complete' for r in rows),
      '/20; sampling=',sum(r.get('phase')=='SAMPLING' and r.get('status')=='complete' for r in rows),
      '/20; recorded forwards=',sum(r.get('nfe',0) for r in rows),'/9624',flush=True)
PY
    done
) &
zaps_a56_monitor=$!
zaps_a56_launch(){
    env CUDA_VISIBLE_DEVICES="$2" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      python -u utils/diag_direction_operator_repair.py --package A56 --shard "$1" --device cuda:0 \
      9>&- > "$zaps_a56_base/shard$1.log" 2>&1 &
    zaps_a56_last=$!; zaps_a56_pids+=("$zaps_a56_last")
}
zaps_a56_s0=0; zaps_a56_s1=0
zaps_a56_launch 0 "${zaps_a56_gpus[0]}"; zaps_a56_first=$zaps_a56_last
if [ "${zaps_a56_gpus[0]}" = "${zaps_a56_gpus[1]}" ]; then
    if wait "$zaps_a56_first"; then :; else zaps_a56_s0=$?; fi
    zaps_a56_pids=()
    if [ "$zaps_a56_s0" -eq 0 ]; then
        zaps_a56_launch 1 "${zaps_a56_gpus[1]}"
        if wait "$zaps_a56_last"; then :; else zaps_a56_s1=$?; fi
    fi
else
    zaps_a56_launch 1 "${zaps_a56_gpus[1]}"; zaps_a56_second=$zaps_a56_last
    if wait "$zaps_a56_first"; then :; else zaps_a56_s0=$?; fi
    zaps_a56_pids=("$zaps_a56_second")
    if wait "$zaps_a56_second"; then :; else zaps_a56_s1=$?; fi
fi
zaps_a56_pids=()
if [ "$zaps_a56_s0" -ne 0 ] || [ "$zaps_a56_s1" -ne 0 ]; then
    for zaps_a56_shard in 0 1; do
        if [ -f "$zaps_a56_base/shard$zaps_a56_shard.log" ]; then tail -n 70 "$zaps_a56_base/shard$zaps_a56_shard.log" 9>&-; fi
    done
    exit 1
fi
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -u utils/diag_direction_operator_repair.py --package A56 --archive 9>&- > "$zaps_a56_base/final.log" 2>&1; then zaps_a56_fail "$zaps_a56_base/final.log"; fi
cat "$zaps_a56_base/final.log" 9>&-
if timeout 45 env GIT_TERMINAL_PROMPT=0 git push origin results/frozen-direction-transfer-20261026 9>&- >> "$zaps_a56_log" 2>&1; then
    echo 'RESULTS_PUSH_PASS: A56 frozen two-dataset/two-task confirmation uploaded.'
else echo 'RESULTS_PUSH_PENDING: results retained; archive/push only, no GPU rerun.'; tail -n 12 "$zaps_a56_log" 9>&-; fi
