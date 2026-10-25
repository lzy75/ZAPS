#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a41_parent=/data0/lzy/ZAPS-results/diag_ffhq_action_geometry
if [ ! -d "$zaps_a41_parent/A41_20261025" ];then echo 'A41 source missing; no experiment started.';exit 1;fi
exec 9>"$zaps_a41_parent/launcher.lock"
echo 'A41 cached audit/archive only: NEW UNet NFE=0; waiting up to20s for the existing launcher to finish.'
if ! flock -w 20 9;then
    python - <<'PY' 9>&-
from pathlib import Path
import json
from utils.diag_ffhq_action_geometry import lock_status,ROOT
print('Existing launcher still holds lock:',json.dumps(lock_status(Path(ROOT)),ensure_ascii=False))
print('No lock removed, no process killed, no GPU task started. Run this same command after the launcher exits.')
PY
    exit 2
fi
zaps_a41_review_log="$zaps_a41_parent/cached_review_20261025.log"
if ! python -u utils/audit_ffhq_action_geometry_results.py --archive 9>&- > "$zaps_a41_review_log" 2>&1;then
    tail -n 70 "$zaps_a41_review_log" 9>&-;exit 1
fi
cat "$zaps_a41_review_log" 9>&-
if git push origin results/ffhq-action-geometry-20261025 9>&- >> "$zaps_a41_review_log" 2>&1;then
    echo 'RESULTS_PUSH_PASS: A41 full records, decision review and all24 comparison panels uploaded. NEW UNet NFE=0.'
else
    echo 'RESULTS_PUSH_PENDING: audited results retained; no GPU rerun needed.'
    tail -n 16 "$zaps_a41_review_log" 9>&-
fi
