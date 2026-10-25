#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a44_parent=/data0/lzy/ZAPS-results/diag_frozen_state_matching
mkdir -p "$zaps_a44_parent"
exec 9>"$zaps_a44_parent/launcher.lock"
if ! flock -n 9;then echo 'A44 already running; duplicate refused.' >&2;exit 1;fi
zaps_a44_log="$zaps_a44_parent/launcher_20261025.log"
zaps_a44_fail(){ tail -n 65 "$1" 9>&-;exit 1; }
echo 'A44: five other reused images x four tasks; frozen matching, task-fixed and shuffled controls; <=1660 NEW diagnostic calls; no training. Waiting for final output.'
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python - <<'PY' 9>&- >> "$zaps_a44_log" 2>&1
import sys,unittest
suite=unittest.defaultTestLoader.loadTestsFromNames(['utils.test_frozen_state_matching','utils.test_continuous_uncertainty_feedback','utils.test_uncertainty_feedback'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
PY
then zaps_a44_fail "$zaps_a44_log";fi
# Recover A43's light archive without any model call; respect its own launcher lock.
(
    exec 9>&-
    exec 8>/data0/lzy/ZAPS-results/diag_continuous_uncertainty/launcher.lock
    if flock -w 10 8;then
        if env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -u utils/diag_continuous_uncertainty_feedback.py --archive 8>&- > "$zaps_a44_parent/A43_archive.log" 2>&1;then
            if timeout 45 env GIT_TERMINAL_PROMPT=0 git push origin results/ffhq-continuous-uncertainty-20261025 8>&- >> "$zaps_a44_parent/A43_archive.log" 2>&1;then
                echo 'A43_ARCHIVE_PUSH_PASS: completed outputs archived; NEW UNet NFE=0.'
            else echo 'A43_PUSH_PENDING: archive retained; continuing A44.';tail -n 8 "$zaps_a44_parent/A43_archive.log";fi
        else echo 'A43_ARCHIVE_PENDING: original results retained; prepare will validate them.';tail -n 15 "$zaps_a44_parent/A43_archive.log";fi
    else echo 'A43 launcher lock held; skipping duplicate archive. No process interrupted.';fi
)
if ! python -u utils/diag_frozen_state_matching.py --prepare 9>&- >> "$zaps_a44_log" 2>&1;then zaps_a44_fail "$zaps_a44_log";fi
tail -n 2 "$zaps_a44_log" 9>&-
if ! python - <<'PY' 9>&- > "$zaps_a44_parent/devices.txt" 2>> "$zaps_a44_log"
import os,subprocess
from utils.multitask_state_transfer import select_devices
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
rows=[]
for line in raw.splitlines():
    i,u,f=[v.strip() for v in line.split(',')];rows.append(dict(index=i,uuid=u,free=int(f)))
print('\n'.join(select_devices(rows,os.environ.get('CUDA_VISIBLE_DEVICES'),os.environ.get('ZAPS_A44_GPU0'),os.environ.get('ZAPS_A44_GPU1'))))
PY
then zaps_a44_fail "$zaps_a44_log";fi
mapfile -t zaps_a44_gpus < "$zaps_a44_parent/devices.txt"
echo "A44 selected GPUs: ${zaps_a44_gpus[*]}; equal IDs mean serial shards."
zaps_a44_pids=();zaps_a44_monitor=''
zaps_a44_cleanup(){
    for zaps_a44_pid in "${zaps_a44_pids[@]}";do kill -TERM "$zaps_a44_pid" 2>/dev/null || true;done
    if [ -n "$zaps_a44_monitor" ];then kill -TERM "$zaps_a44_monitor" 2>/dev/null || true;fi
}
trap 'zaps_a44_cleanup;exit 130' INT TERM HUP
(
    exec 9>&-
    while sleep 60;do
        python - <<'PY'
from pathlib import Path
import json
records=[]
for p in Path('/data0/lzy/ZAPS-results/diag_frozen_state_matching/A44_20261025').glob('*/*_*/run.json'):
    try:records.append(json.loads(p.read_text()))
    except (ValueError,OSError):pass
print('A44 progress: choose',sum(r.get('phase')=='choose' and r.get('status')=='complete' for r in records),'/20; finish',sum(r.get('phase')=='finish' and r.get('status')=='complete' for r in records),'/20; NEW recorded calls=',sum(r.get('nfe',0) for r in records),flush=True)
PY
    done
) &
zaps_a44_monitor=$!
zaps_a44_launch(){
    env CUDA_VISIBLE_DEVICES="$3" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      python -u utils/diag_frozen_state_matching.py --phase "$1" --shard "$2" --device cuda:0 \
      9>&- > "$zaps_a44_parent/${1}_shard${2}.log" 2>&1 &
    zaps_a44_last=$!;zaps_a44_pids+=("$zaps_a44_last")
}
zaps_a44_phase(){
    local zaps_a44_name="$1" zaps_a44_s0=0 zaps_a44_s1=0
    echo "A44 $zaps_a44_name phase started; waiting with automatic progress."
    zaps_a44_launch "$zaps_a44_name" 0 "${zaps_a44_gpus[0]}";local zaps_a44_first=$zaps_a44_last
    if [ "${zaps_a44_gpus[0]}" = "${zaps_a44_gpus[1]}" ];then
        if wait "$zaps_a44_first";then :;else zaps_a44_s0=$?;fi
        zaps_a44_pids=();zaps_a44_launch "$zaps_a44_name" 1 "${zaps_a44_gpus[1]}"
        if wait "$zaps_a44_last";then :;else zaps_a44_s1=$?;fi
    else
        zaps_a44_launch "$zaps_a44_name" 1 "${zaps_a44_gpus[1]}";local zaps_a44_second=$zaps_a44_last
        if wait "$zaps_a44_first";then :;else zaps_a44_s0=$?;fi
        zaps_a44_pids=("$zaps_a44_second")
        if wait "$zaps_a44_second";then :;else zaps_a44_s1=$?;fi
    fi
    zaps_a44_pids=()
    if [ "$zaps_a44_s0" -ne 0 ] || [ "$zaps_a44_s1" -ne 0 ];then
        zaps_a44_cleanup
        tail -n 50 "$zaps_a44_parent/${zaps_a44_name}_shard0.log" 9>&-
        tail -n 50 "$zaps_a44_parent/${zaps_a44_name}_shard1.log" 9>&-;exit 1
    fi
}
zaps_a44_phase choose
if ! python -u utils/diag_frozen_state_matching.py --freeze 9>&- >> "$zaps_a44_log" 2>&1;then zaps_a44_cleanup;zaps_a44_fail "$zaps_a44_log";fi
tail -n 1 "$zaps_a44_log" 9>&-
zaps_a44_phase finish
zaps_a44_cleanup;zaps_a44_monitor='';trap - INT TERM HUP
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    python -u utils/diag_frozen_state_matching.py --archive 9>&- > "$zaps_a44_parent/final.log" 2>&1;then zaps_a44_fail "$zaps_a44_parent/final.log";fi
cat "$zaps_a44_parent/final.log" 9>&-
if timeout 45 env GIT_TERMINAL_PROMPT=0 git push origin results/ffhq-state-matching-20261025 9>&- >> "$zaps_a44_log" 2>&1;then
    echo 'RESULTS_PUSH_PASS: A44 frozen rules, all controls and all20 comparison panels uploaded.'
else echo 'RESULTS_PUSH_PENDING: outputs retained; no GPU rerun needed.';tail -n 15 "$zaps_a44_log" 9>&-;fi
