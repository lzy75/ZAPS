#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a60_base=/data0/lzy/ZAPS-results/diag_direction_refresh_value
mkdir -p "$zaps_a60_base"
exec 9>"$zaps_a60_base/launcher.lock"
if ! flock -n 9;then echo 'A60 already running; duplicate refused.' >&2;exit 1;fi
zaps_a60_log="$zaps_a60_base/launcher_20261027.log"
zaps_a60_pids=();zaps_a60_monitor=''
zaps_a60_cleanup(){
    for zaps_a60_pid in "${zaps_a60_pids[@]}";do kill -TERM "$zaps_a60_pid" 2>/dev/null || true;done
    if [ -n "$zaps_a60_monitor" ];then kill -TERM "$zaps_a60_monitor" 2>/dev/null || true;fi
}
trap zaps_a60_cleanup EXIT
trap 'exit 130' INT TERM HUP
zaps_a60_fail(){ tail -n 70 "$1" 9>&-;exit 1; }
echo 'A60:direction refresh1/2/4, current physical length every step;900F/<=550 inputVJP;30 original points unchanged.'
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m unittest \
  utils.test_direction_refresh_control utils.test_fixed_time_geometry_value.TorchTests utils.test_norm_gain_control utils.test_current_score_checkpoint_r2 -v 9>&- > "$zaps_a60_log" 2>&1;then zaps_a60_fail "$zaps_a60_log";fi
if ! python -u utils/diag_direction_refresh_value.py --prepare 9>&- >> "$zaps_a60_log" 2>&1;then zaps_a60_fail "$zaps_a60_log";fi
tail -n 1 "$zaps_a60_log"
if ! python - <<'PY' 9>&- > "$zaps_a60_base/devices.txt" 2>> "$zaps_a60_log"
import os,subprocess
from utils.experiment_gpu_selection import choose
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
rows=[]
for line in raw.splitlines():
    i,u,f=[x.strip() for x in line.split(',')];rows.append(dict(index=i,uuid=u,free=int(f)))
print('\n'.join(choose(rows,os.environ.get('CUDA_VISIBLE_DEVICES'),os.environ.get('ZAPS_A60_GPU0'),os.environ.get('ZAPS_A60_GPU1'))))
PY
then zaps_a60_fail "$zaps_a60_log";fi
mapfile -t zaps_a60_gpus < "$zaps_a60_base/devices.txt"
echo "A60 allocated GPUs: ${zaps_a60_gpus[*]}; >=6GiB free required; equal IDs mean serial shards."
(
    exec 9>&-
    while sleep 60;do
        python - <<'PY'
import json
from pathlib import Path
root=Path('/data0/lzy/ZAPS-results/diag_direction_refresh_value/A60_20261027')
p=json.loads((root/'plan.json').read_text());rows=[]
for f in root.glob('*/*_*/run.json'):
    try:rows.append(json.loads(f.read_text()))
    except (ValueError,OSError):pass
print('A60 direction refresh: complete=',sum(r.get('status')=='complete' for r in rows),'/10; diagnostic forwards=',sum(r.get('nfe',0) for r in rows),'/900',flush=True)
PY
    done
) &
zaps_a60_monitor=$!
zaps_a60_launch(){
    env CUDA_VISIBLE_DEVICES="$3" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      python -u utils/diag_direction_refresh_value.py --phase "$1" --shard "$2" --device cuda:0 \
      9>&- > "$zaps_a60_base/$1_shard$2.log" 2>&1 &
    zaps_a60_last=$!;zaps_a60_pids+=("$zaps_a60_last")
}
for zaps_a60_phase in control;do
    zaps_a60_s0=0;zaps_a60_s1=0
    zaps_a60_launch "$zaps_a60_phase" 0 "${zaps_a60_gpus[0]}";zaps_a60_first=$zaps_a60_last
    if [ "${zaps_a60_gpus[0]}" = "${zaps_a60_gpus[1]}" ];then
        if wait "$zaps_a60_first";then :;else zaps_a60_s0=$?;fi
        zaps_a60_pids=()
        if [ "$zaps_a60_s0" -eq 0 ];then
            zaps_a60_launch "$zaps_a60_phase" 1 "${zaps_a60_gpus[1]}"
            if wait "$zaps_a60_last";then :;else zaps_a60_s1=$?;fi
        fi
    else
        zaps_a60_launch "$zaps_a60_phase" 1 "${zaps_a60_gpus[1]}";zaps_a60_second=$zaps_a60_last
        if wait "$zaps_a60_first";then :;else zaps_a60_s0=$?;fi
        zaps_a60_pids=("$zaps_a60_second")
        if wait "$zaps_a60_second";then :;else zaps_a60_s1=$?;fi
    fi
    zaps_a60_pids=()
    if [ "$zaps_a60_s0" -ne 0 ] || [ "$zaps_a60_s1" -ne 0 ];then
        tail -n 60 "$zaps_a60_base/${zaps_a60_phase}_shard0.log" 9>&-
        if [ -f "$zaps_a60_base/${zaps_a60_phase}_shard1.log" ];then tail -n 60 "$zaps_a60_base/${zaps_a60_phase}_shard1.log" 9>&-;fi
        exit 1
    fi

done
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -u utils/diag_direction_refresh_value.py --archive 9>&- > "$zaps_a60_base/final.log" 2>&1;then zaps_a60_fail "$zaps_a60_base/final.log";fi
cat "$zaps_a60_base/final.log" 9>&-
if timeout 45 env GIT_TERMINAL_PROMPT=0 git push origin results/direction-refresh-value-20261027 9>&- >> "$zaps_a60_log" 2>&1;then
    echo 'RESULTS_PUSH_PASS: A60 direction refresh value and cost uploaded.'
else echo 'RESULTS_PUSH_PENDING: results retained; archive/push only, no GPU rerun.';tail -n 12 "$zaps_a60_log" 9>&-;fi
