#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a49_parent=/data0/lzy/ZAPS-results/diag_response_reliability
mkdir -p "$zaps_a49_parent"
exec 9>"$zaps_a49_parent/launcher.lock"
if ! flock -n 9;then echo 'A48/A49 launcher still active; no duplicate or GPU rerun started.' >&2;exit 1;fi
zaps_a49_log="$zaps_a49_parent/A49_cached_20261026.log"
echo 'A49: cached local-objective versus terminal-quality audit; NEW UNet NFE=0. Waiting for final summary.'
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    python -m unittest utils.test_cached_reliability_audit -v 9>&- > "$zaps_a49_log" 2>&1;then
    tail -n 60 "$zaps_a49_log" 9>&-;exit 1
fi
if ! env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    python -u utils/diag_cached_reliability_audit.py --archive 9>&- >> "$zaps_a49_log" 2>&1;then
    tail -n 70 "$zaps_a49_log" 9>&-;exit 1
fi
cat "$zaps_a49_log" 9>&-
if timeout 45 env GIT_TERMINAL_PROMPT=0 git push origin results/ffhq-response-reliability-audit-20261026 9>&- >> "$zaps_a49_log" 2>&1;then
    echo 'RESULTS_PUSH_PASS: A49 audit plus A48 twenty-case metadata and saved panels uploaded; NEW UNet NFE=0.'
else echo 'RESULTS_PUSH_PENDING: cached audit retained; no experiment rerun needed.';tail -n 15 "$zaps_a49_log" 9>&-;fi
