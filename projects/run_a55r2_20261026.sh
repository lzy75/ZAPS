#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a55r2_base=/data0/lzy/ZAPS-results/diag_current_direction_value_r2
mkdir -p "$zaps_a55r2_base"
exec 9>"$zaps_a55r2_base/launcher.lock"
if ! flock -n 9;then echo 'A55r2 already running; duplicate refused.' >&2;exit 1;fi
zaps_a55r2_log="$zaps_a55r2_base/launcher_20261026.log"
zaps_a55r2_pids=();zaps_a55r2_monitor=''
zaps_a55r2_cleanup(){
    for zaps_a55r2_pid in "${zaps_a55r2_pids[@]}";do kill -TERM "$zaps_a55r2_pid" 2>/dev/null || true;done
    if [ -n "$zaps_a55r2_monitor" ];then kill -TERM "$zaps_a55r2_monitor" 2>/dev/null || true;fi
}
trap zaps_a55r2_cleanup EXIT
trap 'exit 130' INT TERM HUP
zaps_a55r2_fail(){ tail -n 70 "$1" 9>&-;exit 1; }
echo 'A55r2:correct exact-SR-adjoint guard; same direction experiment.10 U +4 V cached, old completed controls reused;<=480 NEW forwards,0 denoiser backwards;operator autograd counted separately.'
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m unittest \
  utils.test_current_direction_control_r2 utils.test_linear_adjoint_autograd_guard utils.test_current_score_checkpoint_r2 -v 9>&- > "$zaps_a55r2_log" 2>&1;then zaps_a55r2_fail "$zaps_a55r2_log";fi
if ! python -u utils/diag_current_direction_value_r2.py --prepare 9>&- >> "$zaps_a55r2_log" 2>&1;then zaps_a55r2_fail "$zaps_a55r2_log";fi
tail -n 1 "$zaps_a55r2_log"
if ! python - <<'PY' 9>&- > "$zaps_a55r2_base/devices.txt" 2>> "$zaps_a55r2_log"
import os,subprocess
from utils.experiment_gpu_selection import choose
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
rows=[]
for line in raw.splitlines():
    i,u,f=[x.strip() for x in line.split(',')];rows.append(dict(index=i,uuid=u,free=int(f)))
print('\n'.join(choose(rows,os.environ.get('CUDA_VISIBLE_DEVICES'),os.environ.get('ZAPS_A55R2_GPU0'),os.environ.get('ZAPS_A55R2_GPU1'))))
PY
then zaps_a55r2_fail "$zaps_a55r2_log";fi
mapfile -t zaps_a55r2_gpus < "$zaps_a55r2_base/devices.txt"
echo "A55r2 allocated GPUs: ${zaps_a55r2_gpus[*]}; >=6GiB free required; equal IDs mean serial shards."
(
    exec 9>&-
    while sleep 60;do
        python - <<'PY'
import json
from pathlib import Path
root=Path('/data0/lzy/ZAPS-results/diag_current_direction_value_r2/A55r2_20261026')
p=json.loads((root/'plan.json').read_text());rows=[]
for f in root.glob('*/*_*/run.json'):
    try:rows.append(json.loads(f.read_text()))
    except (ValueError,OSError):pass
print('A55r2 progress: reused old complete=',len(p['reused_control_sources']),'; new complete=',sum(r.get('status')=='complete' for r in rows),
      '/',10-len(p['reused_control_sources']),'; recorded NEW forwards=',sum(r.get('nfe',0) for r in rows),'/',p['planned_diagnostic_nfe'],flush=True)
PY
    done
) &
zaps_a55r2_monitor=$!
zaps_a55r2_launch(){
    env CUDA_VISIBLE_DEVICES="$3" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      python -u utils/diag_current_direction_value_r2.py --phase "$1" --shard "$2" --device cuda:0 \
      9>&- > "$zaps_a55r2_base/$1_shard$2.log" 2>&1 &
    zaps_a55r2_last=$!;zaps_a55r2_pids+=("$zaps_a55r2_last")
}
for zaps_a55r2_phase in control;do
    zaps_a55r2_s0=0;zaps_a55r2_s1=0
    zaps_a55r2_launch "$zaps_a55r2_phase" 0 "${zaps_a55r2_gpus[0]}";zaps_a55r2_first=$zaps_a55r2_last
    if [ "${zaps_a55r2_gpus[0]}" = "${zaps_a55r2_gpus[1]}" ];then
        if wait "$zaps_a55r2_first";then :;else zaps_a55r2_s0=$?;fi
        zaps_a55r2_pids=()
        if [ "$zaps_a55r2_s0" -eq 0 ];then
            zaps_a55r2_launch "$zaps_a55r2_phase" 1 "${zaps_a55r2_gpus[1]}"
            if wait "$zaps_a55r2_last";then :;else zaps_a55r2_s1=$?;fi
        fi
    else
        zaps_a55r2_launch "$zaps_a55r2_phase" 1 "${zaps_a55r2_gpus[1]}";zaps_a55r2_second=$zaps_a55r2_last
        if wait "$zaps_a55r2_first";then :;else zaps_a55r2_s0=$?;fi
        zaps_a55r2_pids=("$zaps_a55r2_second")
        if wait "$zaps_a55r2_second";then :;else zaps_a55r2_s1=$?;fi
    fi
    zaps_a55r2_pids=()
    if [ "$zaps_a55r2_s0" -ne 0 ] || [ "$zaps_a55r2_s1" -ne 0 ];then
        tail -n 60 "$zaps_a55r2_base/${zaps_a55r2_phase}_shard0.log" 9>&-
        if [ -f "$zaps_a55r2_base/${zaps_a55r2_phase}_shard1.log" ];then tail -n 60 "$zaps_a55r2_base/${zaps_a55r2_phase}_shard1.log" 9>&-;fi
        exit 1
    fi

done
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -u utils/diag_current_direction_value_r2.py --archive 9>&- > "$zaps_a55r2_base/final.log" 2>&1;then zaps_a55r2_fail "$zaps_a55r2_base/final.log";fi
cat "$zaps_a55r2_base/final.log" 9>&-
if timeout 45 env GIT_TERMINAL_PROMPT=0 git push origin results/ffhq-current-direction-value-r2-20261026 9>&- >> "$zaps_a55r2_log" 2>&1;then
    echo 'RESULTS_PUSH_PASS: A55r2 direction/norm-law ablations and panels uploaded.'
else echo 'RESULTS_PUSH_PENDING: results retained; archive/push only, no GPU rerun.';tail -n 12 "$zaps_a55r2_log" 9>&-;fi
