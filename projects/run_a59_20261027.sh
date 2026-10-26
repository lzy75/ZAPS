#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a59_base=/data0/lzy/ZAPS-results/diag_fixed_time_geometry_value
mkdir -p "$zaps_a59_base"
exec 9>"$zaps_a59_base/launcher.lock"
if ! flock -n 9;then echo 'A59 already running; duplicate refused.' >&2;exit 1;fi
zaps_a59_log="$zaps_a59_base/launcher_20261027.log"
zaps_a59_pids=();zaps_a59_monitor=''
zaps_a59_cleanup(){
    for zaps_a59_pid in "${zaps_a59_pids[@]}";do kill -TERM "$zaps_a59_pid" 2>/dev/null || true;done
    if [ -n "$zaps_a59_monitor" ];then kill -TERM "$zaps_a59_monitor" 2>/dev/null || true;fi
}
trap zaps_a59_cleanup EXIT
trap 'exit 130' INT TERM HUP
zaps_a59_fail(){ tail -n 70 "$1" 9>&-;exit 1; }
echo 'A59:same-time paid geometry versus final fixed-grid dose response;10 OLD cases,<=760F/760 inputB; no online policy.'
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m unittest \
  utils.test_fixed_time_geometry_value utils.test_norm_gain_control utils.test_current_score_checkpoint_r2 -v 9>&- > "$zaps_a59_log" 2>&1;then zaps_a59_fail "$zaps_a59_log";fi
if ! python -u utils/diag_fixed_time_geometry_value.py --prepare 9>&- >> "$zaps_a59_log" 2>&1;then zaps_a59_fail "$zaps_a59_log";fi
tail -n 1 "$zaps_a59_log"
if ! python - <<'PY' 9>&- > "$zaps_a59_base/devices.txt" 2>> "$zaps_a59_log"
import os,subprocess
from utils.experiment_gpu_selection import choose
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
rows=[]
for line in raw.splitlines():
    i,u,f=[x.strip() for x in line.split(',')];rows.append(dict(index=i,uuid=u,free=int(f)))
print('\n'.join(choose(rows,os.environ.get('CUDA_VISIBLE_DEVICES'),os.environ.get('ZAPS_A59_GPU0'),os.environ.get('ZAPS_A59_GPU1'))))
PY
then zaps_a59_fail "$zaps_a59_log";fi
mapfile -t zaps_a59_gpus < "$zaps_a59_base/devices.txt"
echo "A59 allocated GPUs: ${zaps_a59_gpus[*]}; >=6GiB free required; equal IDs mean serial shards."
(
    exec 9>&-
    while sleep 60;do
        python - <<'PY'
import json
from pathlib import Path
root=Path('/data0/lzy/ZAPS-results/diag_fixed_time_geometry_value/A59_20261027')
p=json.loads((root/'plan.json').read_text());rows=[]
for f in root.glob('*/*_*/run.json'):
    try:rows.append(json.loads(f.read_text()))
    except (ValueError,OSError):pass
print('A59 fixed-time diagnosis: complete=',sum(r.get('status')=='complete' for r in rows),'/10; diagnostic forwards=',sum(r.get('nfe',0) for r in rows),'/760',flush=True)
PY
    done
) &
zaps_a59_monitor=$!
zaps_a59_launch(){
    env CUDA_VISIBLE_DEVICES="$3" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      python -u utils/diag_fixed_time_geometry_value.py --phase "$1" --shard "$2" --device cuda:0 \
      9>&- > "$zaps_a59_base/$1_shard$2.log" 2>&1 &
    zaps_a59_last=$!;zaps_a59_pids+=("$zaps_a59_last")
}
for zaps_a59_phase in control;do
    zaps_a59_s0=0;zaps_a59_s1=0
    zaps_a59_launch "$zaps_a59_phase" 0 "${zaps_a59_gpus[0]}";zaps_a59_first=$zaps_a59_last
    if [ "${zaps_a59_gpus[0]}" = "${zaps_a59_gpus[1]}" ];then
        if wait "$zaps_a59_first";then :;else zaps_a59_s0=$?;fi
        zaps_a59_pids=()
        if [ "$zaps_a59_s0" -eq 0 ];then
            zaps_a59_launch "$zaps_a59_phase" 1 "${zaps_a59_gpus[1]}"
            if wait "$zaps_a59_last";then :;else zaps_a59_s1=$?;fi
        fi
    else
        zaps_a59_launch "$zaps_a59_phase" 1 "${zaps_a59_gpus[1]}";zaps_a59_second=$zaps_a59_last
        if wait "$zaps_a59_first";then :;else zaps_a59_s0=$?;fi
        zaps_a59_pids=("$zaps_a59_second")
        if wait "$zaps_a59_second";then :;else zaps_a59_s1=$?;fi
    fi
    zaps_a59_pids=()
    if [ "$zaps_a59_s0" -ne 0 ] || [ "$zaps_a59_s1" -ne 0 ];then
        tail -n 60 "$zaps_a59_base/${zaps_a59_phase}_shard0.log" 9>&-
        if [ -f "$zaps_a59_base/${zaps_a59_phase}_shard1.log" ];then tail -n 60 "$zaps_a59_base/${zaps_a59_phase}_shard1.log" 9>&-;fi
        exit 1
    fi

done
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -u utils/diag_fixed_time_geometry_value.py --archive 9>&- > "$zaps_a59_base/final.log" 2>&1;then zaps_a59_fail "$zaps_a59_base/final.log";fi
cat "$zaps_a59_base/final.log" 9>&-
if timeout 45 env GIT_TERMINAL_PROMPT=0 git push origin results/fixed-time-geometry-value-20261027 9>&- >> "$zaps_a59_log" 2>&1;then
    echo 'RESULTS_PUSH_PASS: A59 fixed-time geometry and final dose response uploaded.'
else echo 'RESULTS_PUSH_PENDING: results retained; archive/push only, no GPU rerun.';tail -n 12 "$zaps_a59_log" 9>&-;fi
