#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a53_base=/data0/lzy/ZAPS-results/diag_angular_feedback_value
mkdir -p "$zaps_a53_base"
exec 9>"$zaps_a53_base/launcher.lock"
if ! flock -n 9;then echo 'A53 already running; duplicate refused.' >&2;exit 1;fi
zaps_a53_log="$zaps_a53_base/launcher_20261026.log"
zaps_a53_pids=();zaps_a53_monitor=''
zaps_a53_cleanup(){
    for zaps_a53_pid in "${zaps_a53_pids[@]}";do kill -TERM "$zaps_a53_pid" 2>/dev/null || true;done
    if [ -n "$zaps_a53_monitor" ];then kill -TERM "$zaps_a53_monitor" 2>/dev/null || true;fi
}
trap zaps_a53_cleanup EXIT
trap 'exit 130' INT TERM HUP
zaps_a53_fail(){ tail -n 70 "$1" 9>&-;exit 1; }
echo 'A53:five OLD images x SR/IP; CURRENT direction/norms shared, c-only ablations.4 STATE cases cached;1080 NEW forwards,<=1080 input backwards;no training.'
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m unittest \
  utils.test_angular_feedback_control utils.test_current_score_checkpoint_r2 -v 9>&- > "$zaps_a53_log" 2>&1;then zaps_a53_fail "$zaps_a53_log";fi
if ! python -u utils/diag_angular_feedback_value.py --prepare 9>&- >> "$zaps_a53_log" 2>&1;then zaps_a53_fail "$zaps_a53_log";fi
tail -n 1 "$zaps_a53_log"
if ! python - <<'PY' 9>&- > "$zaps_a53_base/devices.txt" 2>> "$zaps_a53_log"
import os,subprocess
from utils.experiment_gpu_selection import choose
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
rows=[]
for line in raw.splitlines():
    i,u,f=[x.strip() for x in line.split(',')];rows.append(dict(index=i,uuid=u,free=int(f)))
print('\n'.join(choose(rows,os.environ.get('CUDA_VISIBLE_DEVICES'),os.environ.get('ZAPS_A53_GPU0'),os.environ.get('ZAPS_A53_GPU1'))))
PY
then zaps_a53_fail "$zaps_a53_log";fi
mapfile -t zaps_a53_gpus < "$zaps_a53_base/devices.txt"
echo "A53 allocated GPUs: ${zaps_a53_gpus[*]}; >=6GiB free required; equal IDs mean serial shards."
(
    exec 9>&-
    while sleep 60;do
        python - <<'PY'
import json
from pathlib import Path
rows=[]
for f in Path('/data0/lzy/ZAPS-results/diag_angular_feedback_value/A53_20261026').glob('*/*_*/run.json'):
    try:rows.append(json.loads(f.read_text()))
    except (ValueError,OSError):pass
print('A53 progress: STATE=',4+sum(r.get('phase')=='generate' and r.get('status')=='complete' for r in rows),
      '/10 (4 cached); controls=',sum(r.get('phase')=='control' and r.get('status')=='complete' for r in rows),
      '/10; recorded NEW forwards=',sum(r.get('nfe',0) for r in rows),'/1080',flush=True)
PY
    done
) &
zaps_a53_monitor=$!
zaps_a53_launch(){
    env CUDA_VISIBLE_DEVICES="$3" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      python -u utils/diag_angular_feedback_value.py --phase "$1" --shard "$2" --device cuda:0 \
      9>&- > "$zaps_a53_base/$1_shard$2.log" 2>&1 &
    zaps_a53_last=$!;zaps_a53_pids+=("$zaps_a53_last")
}
for zaps_a53_phase in generate control;do
    zaps_a53_s0=0;zaps_a53_s1=0
    zaps_a53_launch "$zaps_a53_phase" 0 "${zaps_a53_gpus[0]}";zaps_a53_first=$zaps_a53_last
    if [ "${zaps_a53_gpus[0]}" = "${zaps_a53_gpus[1]}" ];then
        if wait "$zaps_a53_first";then :;else zaps_a53_s0=$?;fi
        zaps_a53_pids=()
        if [ "$zaps_a53_s0" -eq 0 ];then
            zaps_a53_launch "$zaps_a53_phase" 1 "${zaps_a53_gpus[1]}"
            if wait "$zaps_a53_last";then :;else zaps_a53_s1=$?;fi
        fi
    else
        zaps_a53_launch "$zaps_a53_phase" 1 "${zaps_a53_gpus[1]}";zaps_a53_second=$zaps_a53_last
        if wait "$zaps_a53_first";then :;else zaps_a53_s0=$?;fi
        zaps_a53_pids=("$zaps_a53_second")
        if wait "$zaps_a53_second";then :;else zaps_a53_s1=$?;fi
    fi
    zaps_a53_pids=()
    if [ "$zaps_a53_s0" -ne 0 ] || [ "$zaps_a53_s1" -ne 0 ];then
        tail -n 60 "$zaps_a53_base/${zaps_a53_phase}_shard0.log" 9>&-
        if [ -f "$zaps_a53_base/${zaps_a53_phase}_shard1.log" ];then tail -n 60 "$zaps_a53_base/${zaps_a53_phase}_shard1.log" 9>&-;fi
        exit 1
    fi
    if [ "$zaps_a53_phase" = generate ];then
        if ! python -u utils/diag_angular_feedback_value.py --freeze 9>&- >> "$zaps_a53_log" 2>&1;then zaps_a53_fail "$zaps_a53_log";fi
        tail -n 1 "$zaps_a53_log"
    fi
done
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -u utils/diag_angular_feedback_value.py --archive 9>&- > "$zaps_a53_base/final.log" 2>&1;then zaps_a53_fail "$zaps_a53_base/final.log";fi
cat "$zaps_a53_base/final.log" 9>&-
if timeout 45 env GIT_TERMINAL_PROMPT=0 git push origin results/ffhq-current-cosine-value-20261026 9>&- >> "$zaps_a53_log" 2>&1;then
    echo 'RESULTS_PUSH_PASS: A53 c-only matched ablations and panels uploaded.'
else echo 'RESULTS_PUSH_PENDING: results retained; archive/push only, no GPU rerun.';tail -n 12 "$zaps_a53_log" 9>&-;fi
