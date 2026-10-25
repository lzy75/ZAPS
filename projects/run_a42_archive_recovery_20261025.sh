#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a42_parent=/data0/lzy/ZAPS-results/diag_uncertainty_feedback
if [ ! -d "$zaps_a42_parent/A42_20261025" ];then echo 'A42 results missing; no experiment started.';exit 1;fi
exec 9>"$zaps_a42_parent/launcher.lock"
if ! flock -w 20 9;then echo 'A42 launcher still holds lock; no duplicate job or forced unlock.';exit 2;fi
zaps_a42_recovery_log="$zaps_a42_parent/archive_recovery_20261025.log"
echo 'A42 archive recovery only: no model load, no new UNet calls, original outputs retained.'
if ! python -u utils/recover_a42_archive.py 9>&- > "$zaps_a42_recovery_log" 2>&1;then
    tail -n 80 "$zaps_a42_recovery_log" 9>&-;exit 1
fi
cat "$zaps_a42_recovery_log" 9>&-
if git push origin results/ffhq-uncertainty-feedback-20261025 9>&- >> "$zaps_a42_recovery_log" 2>&1;then
    echo 'RESULTS_PUSH_PASS: A42 existing results uploaded; NEW UNet NFE=0.'
else echo 'RESULTS_PUSH_PENDING: local results committed; no GPU rerun needed.';tail -n 20 "$zaps_a42_recovery_log" 9>&-;fi
