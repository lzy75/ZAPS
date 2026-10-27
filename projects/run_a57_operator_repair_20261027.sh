#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a57_base=/data0/lzy/ZAPS-results/diag_no_learning_replay/operator_repair_20261027
mkdir -p "$zaps_a57_base"
if python -u utils/diag_direction_operator_repair.py --package A57 --install;then :
else
    zaps_a57_install_status=$?
    if [ "$zaps_a57_install_status" -eq 20 ];then exit 0;fi
    exit "$zaps_a57_install_status"
fi
exec 9>"/data0/lzy/ZAPS-results/diag_no_learning_replay/launcher.lock"
if ! flock -n 9;then echo 'A57 already running; duplicate refused.' >&2;exit 1;fi
zaps_a57_log="$zaps_a57_base/launcher_20261027.log"
zaps_a57_pids=();zaps_a57_monitor=''
zaps_a57_cleanup(){
    for zaps_a57_pid in "${zaps_a57_pids[@]}";do kill -TERM "$zaps_a57_pid" 2>/dev/null || true;done
    if [ -n "$zaps_a57_monitor" ];then kill -TERM "$zaps_a57_monitor" 2>/dev/null || true;fi
}
trap zaps_a57_cleanup EXIT
trap 'exit 130' INT TERM HUP
zaps_a57_fail(){ tail -n 70 "$1" 9>&-;exit 1; }
echo 'A57:independent seed-only no-learning U inference;10 OLD cases x30;300 method forwards,<=300 input backwards;0 optimization;PRE only audited after output freeze.'
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m unittest \
  utils.test_direction_operator_repair \
  utils.test_no_learning_replay utils.test_norm_gain_control utils.test_current_score_checkpoint_r2 -v 9>&- > "$zaps_a57_log" 2>&1;then zaps_a57_fail "$zaps_a57_log";fi
if ! python -u utils/diag_direction_operator_repair.py --package A57 --prepare 9>&- >> "$zaps_a57_log" 2>&1;then zaps_a57_fail "$zaps_a57_log";fi
tail -n 1 "$zaps_a57_log"
if ! python - <<'PY' 9>&- > "$zaps_a57_base/devices.txt" 2>> "$zaps_a57_log"
import os,subprocess
from utils.experiment_gpu_selection import choose
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
rows=[]
for line in raw.splitlines():
    i,u,f=[x.strip() for x in line.split(',')];rows.append(dict(index=i,uuid=u,free=int(f)))
print('\n'.join(choose(rows,os.environ.get('CUDA_VISIBLE_DEVICES'),os.environ.get('ZAPS_A57_GPU0'),os.environ.get('ZAPS_A57_GPU1'))))
PY
then zaps_a57_fail "$zaps_a57_log";fi
mapfile -t zaps_a57_gpus < "$zaps_a57_base/devices.txt"
echo "A57 allocated GPUs: ${zaps_a57_gpus[*]}; >=6GiB free required; equal IDs mean serial shards."
(
    exec 9>&-
    while sleep 60;do
        python - <<'PY'
import json
from pathlib import Path
root=Path('/data0/lzy/ZAPS-results/diag_no_learning_replay/A57_20261027')
p=json.loads((root/'plan.json').read_text());rows=[]
for f in root.glob('*/*_*/run.json'):
    try:rows.append(json.loads(f.read_text()))
    except (ValueError,OSError):pass
print('A57 independent inference: complete=',sum(r.get('status')=='complete' for r in rows),'/10; method forwards=',sum(r.get('nfe',0) for r in rows),'/300',flush=True)
PY
    done
) &
zaps_a57_monitor=$!
zaps_a57_launch(){
    env CUDA_VISIBLE_DEVICES="$3" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      python -u utils/diag_direction_operator_repair.py --package A57 --phase "$1" --shard "$2" --device cuda:0 \
      9>&- > "$zaps_a57_base/$1_shard$2.log" 2>&1 &
    zaps_a57_last=$!;zaps_a57_pids+=("$zaps_a57_last")
}
for zaps_a57_phase in control;do
    zaps_a57_s0=0;zaps_a57_s1=0
    zaps_a57_launch "$zaps_a57_phase" 0 "${zaps_a57_gpus[0]}";zaps_a57_first=$zaps_a57_last
    if [ "${zaps_a57_gpus[0]}" = "${zaps_a57_gpus[1]}" ];then
        if wait "$zaps_a57_first";then :;else zaps_a57_s0=$?;fi
        zaps_a57_pids=()
        if [ "$zaps_a57_s0" -eq 0 ];then
            zaps_a57_launch "$zaps_a57_phase" 1 "${zaps_a57_gpus[1]}"
            if wait "$zaps_a57_last";then :;else zaps_a57_s1=$?;fi
        fi
    else
        zaps_a57_launch "$zaps_a57_phase" 1 "${zaps_a57_gpus[1]}";zaps_a57_second=$zaps_a57_last
        if wait "$zaps_a57_first";then :;else zaps_a57_s0=$?;fi
        zaps_a57_pids=("$zaps_a57_second")
        if wait "$zaps_a57_second";then :;else zaps_a57_s1=$?;fi
    fi
    zaps_a57_pids=()
    if [ "$zaps_a57_s0" -ne 0 ] || [ "$zaps_a57_s1" -ne 0 ];then
        tail -n 60 "$zaps_a57_base/${zaps_a57_phase}_shard0.log" 9>&-
        if [ -f "$zaps_a57_base/${zaps_a57_phase}_shard1.log" ];then tail -n 60 "$zaps_a57_base/${zaps_a57_phase}_shard1.log" 9>&-;fi
        exit 1
    fi

done
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -u utils/diag_direction_operator_repair.py --package A57 --archive 9>&- > "$zaps_a57_base/final.log" 2>&1;then zaps_a57_fail "$zaps_a57_base/final.log";fi
cat "$zaps_a57_base/final.log" 9>&-
if timeout 45 env GIT_TERMINAL_PROMPT=0 git push origin results/no-learning-replay-20261027 9>&- >> "$zaps_a57_log" 2>&1;then
    echo 'RESULTS_PUSH_PASS: A57 independent no-learning replay uploaded.'
else echo 'RESULTS_PUSH_PENDING: results retained; archive/push only, no GPU rerun.';tail -n 12 "$zaps_a57_log" 9>&-;fi
