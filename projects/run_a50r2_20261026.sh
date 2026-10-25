#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a50r2_parent=/data0/lzy/ZAPS-results/diag_current_score_geometry_r2
mkdir -p "$zaps_a50r2_parent"
exec 9>"$zaps_a50r2_parent/launcher.lock"
if ! flock -n 9;then echo 'A50r2 already running; duplicate refused.' >&2;exit 1;fi
zaps_a50r2_log="$zaps_a50r2_parent/launcher_20261026.log"
zaps_a50r2_pids=();zaps_a50r2_monitor=''
zaps_a50r2_cleanup(){
    for zaps_a50r2_pid in "${zaps_a50r2_pids[@]}";do kill -TERM "$zaps_a50r2_pid" 2>/dev/null || true;done
    if [ -n "$zaps_a50r2_monitor" ];then kill -TERM "$zaps_a50r2_monitor" 2>/dev/null || true;fi
}
trap zaps_a50r2_cleanup EXIT
trap 'exit 130' INT TERM HUP
zaps_a50r2_fail(){ tail -n 70 "$1" 9>&-;exit 1; }
echo 'A50r2: current-score geometry, all last30 fixed steps;20 OLD cases,8 arms.4872 new forwards,<=1212 input backwards;no training. Numerical gates run first.'
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python - <<'PY' 9>&- > "$zaps_a50r2_log" 2>&1
import sys,unittest
suite=unittest.defaultTestLoader.loadTestsFromNames(['utils.test_current_score_geometry','utils.test_current_score_design','utils.test_current_score_geometry_torch','utils.test_current_score_checkpoint_r2'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
PY
then zaps_a50r2_fail "$zaps_a50r2_log";fi
for zaps_a50r2_option in --prepare --inspect;do
    if ! python -u utils/diag_current_score_geometry_r2.py "$zaps_a50r2_option" 9>&- >> "$zaps_a50r2_log" 2>&1;then zaps_a50r2_fail "$zaps_a50r2_log";fi
done
tail -n 2 "$zaps_a50r2_log" 9>&-
if ! python - <<'PY' 9>&- > "$zaps_a50r2_parent/devices.txt" 2>> "$zaps_a50r2_log"
import os,subprocess
from utils.multitask_state_transfer import select_devices
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
rows=[]
for line in raw.splitlines():
    i,u,f=[v.strip() for v in line.split(',')];rows.append(dict(index=i,uuid=u,free=int(f)))
print('\n'.join(select_devices(rows,os.environ.get('CUDA_VISIBLE_DEVICES'),os.environ.get('ZAPS_A50R2_GPU0'),os.environ.get('ZAPS_A50R2_GPU1'))))
PY
then zaps_a50r2_fail "$zaps_a50r2_log";fi
mapfile -t zaps_a50r2_gpus < "$zaps_a50r2_parent/devices.txt"
echo "A50r2 allocated GPUs: ${zaps_a50r2_gpus[*]}; requires>=10GiB free; equal IDs mean serial shards. Actual VJP peak memory is recorded; this is no OOM guarantee."
(
    exec 9>&-
    while sleep 60;do
        python - <<'PY'
import json
from pathlib import Path
rows=[]
for f in Path('/data0/lzy/ZAPS-results/diag_current_score_geometry_r2/A50r2_20261026').glob('*/*_*/run.json'):
    try:rows.append(json.loads(f.read_text()))
    except (ValueError,OSError):pass
print('A50r2 progress:',{p:sum(r.get('status')=='complete' and r.get('phase')==p for r in rows) for p in ('generate','control')},'/20 each; recorded forwards=',sum(r.get('nfe',0) for r in rows),'backward attempts=',sum(r.get('backward_attempts',0) for r in rows),flush=True)
PY
    done
) &
zaps_a50r2_monitor=$!
zaps_a50r2_launch(){
    env CUDA_VISIBLE_DEVICES="$3" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      python -u utils/diag_current_score_geometry_r2.py --phase "$1" --shard "$2" --device cuda:0 \
      9>&- > "$zaps_a50r2_parent/$1_shard$2.log" 2>&1 &
    zaps_a50r2_last=$!;zaps_a50r2_pids+=("$zaps_a50r2_last")
}
for zaps_a50r2_phase in generate control;do
    zaps_a50r2_s0=0;zaps_a50r2_s1=0
    zaps_a50r2_launch "$zaps_a50r2_phase" 0 "${zaps_a50r2_gpus[0]}";zaps_a50r2_first=$zaps_a50r2_last
    if [ "${zaps_a50r2_gpus[0]}" = "${zaps_a50r2_gpus[1]}" ];then
        if wait "$zaps_a50r2_first";then :;else zaps_a50r2_s0=$?;fi
        zaps_a50r2_pids=()
        if [ "$zaps_a50r2_s0" -eq 0 ];then
            zaps_a50r2_launch "$zaps_a50r2_phase" 1 "${zaps_a50r2_gpus[1]}"
            if wait "$zaps_a50r2_last";then :;else zaps_a50r2_s1=$?;fi
        fi
    else
        zaps_a50r2_launch "$zaps_a50r2_phase" 1 "${zaps_a50r2_gpus[1]}";zaps_a50r2_second=$zaps_a50r2_last
        if wait "$zaps_a50r2_first";then :;else zaps_a50r2_s0=$?;fi
        zaps_a50r2_pids=("$zaps_a50r2_second")
        if wait "$zaps_a50r2_second";then :;else zaps_a50r2_s1=$?;fi
    fi
    zaps_a50r2_pids=()
    if [ "$zaps_a50r2_s0" -ne 0 ] || [ "$zaps_a50r2_s1" -ne 0 ];then
        tail -n 55 "$zaps_a50r2_parent/${zaps_a50r2_phase}_shard0.log" 9>&-
        if [ -f "$zaps_a50r2_parent/${zaps_a50r2_phase}_shard1.log" ];then tail -n 55 "$zaps_a50r2_parent/${zaps_a50r2_phase}_shard1.log" 9>&-;fi
        exit 1
    fi
    if [ "$zaps_a50r2_phase" = generate ];then
        if ! python -u utils/diag_current_score_geometry_r2.py --freeze 9>&- >> "$zaps_a50r2_log" 2>&1;then zaps_a50r2_fail "$zaps_a50r2_log";fi
        tail -n 1 "$zaps_a50r2_log" 9>&-
    fi
done
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    python -u utils/diag_current_score_geometry_r2.py --archive 9>&- > "$zaps_a50r2_parent/final.log" 2>&1;then zaps_a50r2_fail "$zaps_a50r2_parent/final.log";fi
cat "$zaps_a50r2_parent/final.log" 9>&-
if timeout 45 env GIT_TERMINAL_PROMPT=0 git push origin results/ffhq-current-score-geometry-r2-20261026 9>&- >> "$zaps_a50r2_log" 2>&1;then
    echo 'RESULTS_PUSH_PASS: A50r2 all20 development comparisons, decisions, accounting and panels uploaded.'
else echo 'RESULTS_PUSH_PENDING: results retained; rerun archive/push only, no GPU rerun.';tail -n 15 "$zaps_a50r2_log" 9>&-;fi
