#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a58_base=/data0/lzy/ZAPS-results/diag_blur_pcg_admission
mkdir -p "$zaps_a58_base"
exec 9>"$zaps_a58_base/launcher.lock"
if ! flock -n 9;then echo 'A58 already running; duplicate refused.' >&2;exit 1;fi
zaps_a58_log="$zaps_a58_base/launcher_20261027.log"
zaps_a58_pids=();zaps_a58_monitor=''
zaps_a58_cleanup(){
    for zaps_a58_pid in "${zaps_a58_pids[@]}";do kill -TERM "$zaps_a58_pid" 2>/dev/null || true;done
    if [ -n "$zaps_a58_monitor" ];then kill -TERM "$zaps_a58_monitor" 2>/dev/null || true;fi
}
trap zaps_a58_cleanup EXIT
trap 'exit 130' INT TERM HUP
zaps_a58_fail(){ tail -n 70 "$1" 9>&-;exit 1; }
echo 'A58:zero-NN structural PCG admission; only all8 pass enables four-arm fixed-grid quality (<=1200F/600B).'
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m unittest \
  utils.test_blur_pcg utils.test_norm_gain_control utils.test_current_score_checkpoint_r2 -v 9>&- > "$zaps_a58_log" 2>&1;then zaps_a58_fail "$zaps_a58_log";fi
if ! python -u utils/diag_blur_pcg_admission.py --prepare 9>&- >> "$zaps_a58_log" 2>&1;then zaps_a58_fail "$zaps_a58_log";fi
tail -n 1 "$zaps_a58_log"
if ! python - <<'PY' 9>&- > "$zaps_a58_base/devices.txt" 2>> "$zaps_a58_log"
import os,subprocess
from utils.experiment_gpu_selection import choose
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
rows=[]
for line in raw.splitlines():
    i,u,f=[x.strip() for x in line.split(',')];rows.append(dict(index=i,uuid=u,free=int(f)))
print('\n'.join(choose(rows,os.environ.get('CUDA_VISIBLE_DEVICES'),os.environ.get('ZAPS_A58_GPU0'),os.environ.get('ZAPS_A58_GPU1'))))
PY
then zaps_a58_fail "$zaps_a58_log";fi
mapfile -t zaps_a58_gpus < "$zaps_a58_base/devices.txt"
echo "A58 allocated GPUs: ${zaps_a58_gpus[*]}; >=6GiB free required; equal IDs mean serial shards."
(
    exec 9>&-
    while sleep 60;do
        python - <<'PY'
import json
from pathlib import Path
root=Path('/data0/lzy/ZAPS-results/diag_blur_pcg_admission/A58_20261027')
p=json.loads((root/'plan.json').read_text());rows=[]
for f in root.glob('*/*_*/run.json'):
    try:rows.append(json.loads(f.read_text()))
    except (ValueError,OSError):pass
print('A58 conditional quality: complete=',sum(r.get('status')=='complete' for r in rows),'/10; diagnostic forwards=',sum(r.get('nfe',0) for r in rows),'/1200',flush=True)
PY
    done
) &
zaps_a58_monitor=$!
zaps_a58_launch(){
    env CUDA_VISIBLE_DEVICES="$3" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      python -u utils/diag_blur_pcg_admission.py --phase "$1" --shard "$2" --device cuda:0 \
      9>&- > "$zaps_a58_base/$1_shard$2.log" 2>&1 &
    zaps_a58_last=$!;zaps_a58_pids+=("$zaps_a58_last")
}
if ! env CUDA_VISIBLE_DEVICES="${zaps_a58_gpus[0]}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -u utils/diag_blur_pcg_admission.py --metrology --device cuda:0 \
  9>&- > "$zaps_a58_base/metrology.log" 2>&1;then zaps_a58_fail "$zaps_a58_base/metrology.log";fi
tail -n 2 "$zaps_a58_base/metrology.log" 9>&-
zaps_a58_admitted=$(python - <<'PY2' 9>&-
import json
from pathlib import Path
print(int(json.loads(Path('/data0/lzy/ZAPS-results/diag_blur_pcg_admission/A58_20261027/metrology.json').read_text())['admitted']))
PY2
)
if [ "$zaps_a58_admitted" = 1 ];then
for zaps_a58_phase in control;do
    zaps_a58_s0=0;zaps_a58_s1=0
    zaps_a58_launch "$zaps_a58_phase" 0 "${zaps_a58_gpus[0]}";zaps_a58_first=$zaps_a58_last
    if [ "${zaps_a58_gpus[0]}" = "${zaps_a58_gpus[1]}" ];then
        if wait "$zaps_a58_first";then :;else zaps_a58_s0=$?;fi
        zaps_a58_pids=()
        if [ "$zaps_a58_s0" -eq 0 ];then
            zaps_a58_launch "$zaps_a58_phase" 1 "${zaps_a58_gpus[1]}"
            if wait "$zaps_a58_last";then :;else zaps_a58_s1=$?;fi
        fi
    else
        zaps_a58_launch "$zaps_a58_phase" 1 "${zaps_a58_gpus[1]}";zaps_a58_second=$zaps_a58_last
        if wait "$zaps_a58_first";then :;else zaps_a58_s0=$?;fi
        zaps_a58_pids=("$zaps_a58_second")
        if wait "$zaps_a58_second";then :;else zaps_a58_s1=$?;fi
    fi
    zaps_a58_pids=()
    if [ "$zaps_a58_s0" -ne 0 ] || [ "$zaps_a58_s1" -ne 0 ];then
        tail -n 60 "$zaps_a58_base/${zaps_a58_phase}_shard0.log" 9>&-
        if [ -f "$zaps_a58_base/${zaps_a58_phase}_shard1.log" ];then tail -n 60 "$zaps_a58_base/${zaps_a58_phase}_shard1.log" 9>&-;fi
        exit 1
    fi

done
else echo 'A58 numerical admission failed; quality phase stopped,0 NN calls. Negative metrology will be archived.';fi
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -u utils/diag_blur_pcg_admission.py --archive 9>&- > "$zaps_a58_base/final.log" 2>&1;then zaps_a58_fail "$zaps_a58_base/final.log";fi
cat "$zaps_a58_base/final.log" 9>&-
if timeout 45 env GIT_TERMINAL_PROMPT=0 git push origin results/blur-pcg-admission-20261027 9>&- >> "$zaps_a58_log" 2>&1;then
    echo 'RESULTS_PUSH_PASS: A58 numerical admission and conditional quality uploaded.'
else echo 'RESULTS_PUSH_PENDING: results retained; archive/push only, no GPU rerun.';tail -n 12 "$zaps_a58_log" 9>&-;fi
