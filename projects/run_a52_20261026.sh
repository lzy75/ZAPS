#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a52_base=/data0/lzy/ZAPS-results/diag_feedback_vector_constraint
mkdir -p "$zaps_a52_base"
exec 9>"$zaps_a52_base/launcher.lock"
if ! flock -n 9;then echo 'A52 already running; duplicate refused.' >&2;exit 1;fi
zaps_a52_log="$zaps_a52_base/launcher_20261026.log"
zaps_a52_fail(){ tail -n 70 "$zaps_a52_log";exit 1; }
echo 'A52:4 reused SR/IP cases; target x q-direction constraint.480 new UNet forwards,<=240 input backwards;no training. Numerical gates first; final summary and panels follow.'
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m unittest \
  utils.test_feedback_vector_constraint utils.test_current_score_checkpoint_r2 -v 9>&- > "$zaps_a52_log" 2>&1;then zaps_a52_fail;fi
if ! python -u utils/diag_feedback_vector_constraint.py --prepare 9>&- >> "$zaps_a52_log" 2>&1;then zaps_a52_fail;fi
tail -n 1 "$zaps_a52_log"
if ! python - <<'PY' 9>&- > "$zaps_a52_base/device.txt" 2>> "$zaps_a52_log"
import os,subprocess
rows=[];visible=os.environ.get('CUDA_VISIBLE_DEVICES')
allowed=None if visible is None else {x.strip() for x in visible.split(',')}
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
for line in raw.splitlines():
    i,u,f=[x.strip() for x in line.split(',')]
    if (allowed is None or i in allowed or u in allowed) and int(f)>=6144:rows.append((int(f),i))
if not rows:raise RuntimeError('No GPU within allocation with >=6GiB free. No override.')
print(max(rows)[1])
PY
then zaps_a52_fail;fi
zaps_a52_gpu=$(cat "$zaps_a52_base/device.txt")
echo "A52 selected GPU $zaps_a52_gpu; four cases run serially with one model load."
env CUDA_VISIBLE_DEVICES="$zaps_a52_gpu" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -u utils/diag_feedback_vector_constraint.py --run --device cuda:0 9>&- >> "$zaps_a52_log" 2>&1 &
zaps_a52_pid=$!
trap 'kill -TERM "$zaps_a52_pid" 2>/dev/null || true' EXIT
trap 'exit 130' INT TERM HUP
while kill -0 "$zaps_a52_pid" 2>/dev/null;do
    sleep 5
    zaps_a52_ticks=$(( ${zaps_a52_ticks:-0} + 1 ))
    if [ "$((zaps_a52_ticks % 12))" -eq 0 ];then
        python - <<'PY' 9>&-
import json
from pathlib import Path
rows=[]
for f in Path('/data0/lzy/ZAPS-results/diag_feedback_vector_constraint/A52_20261026').glob('*/generate_*/run.json'):
    try:rows.append(json.loads(f.read_text()))
    except (OSError,ValueError):pass
print('A52 progress:',sum(r.get('status')=='complete' for r in rows),'/4 cases; recorded forwards=',sum(r.get('nfe',0) for r in rows),'/480',flush=True)
PY
    fi
done
if wait "$zaps_a52_pid";then :;else zaps_a52_fail;fi
trap - EXIT
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -u utils/diag_feedback_vector_constraint.py --archive 9>&- > "$zaps_a52_base/final.log" 2>&1;then
    tail -n 70 "$zaps_a52_base/final.log";exit 1
fi
cat "$zaps_a52_base/final.log"
if timeout 45 env GIT_TERMINAL_PROMPT=0 git push origin results/ffhq-feedback-vector-constraint-20261026 \
  9>&- >> "$zaps_a52_log" 2>&1;then
    echo 'RESULTS_PUSH_PASS: A52 structure ablation and A51b metadata uploaded.'
else echo 'RESULTS_PUSH_PENDING: numerical work retained; archive/push only, no GPU rerun.';tail -n 12 "$zaps_a52_log";fi
