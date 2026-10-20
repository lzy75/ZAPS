#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a9_root=/data0/lzy/ZAPS-results/diag_terminal_polish_control
mkdir -p "$zaps_a9_root"
exec 9>"$zaps_a9_root/A9_20261020.launcher.lock"
if ! flock -n 9; then echo 'A9 already running; duplicate submission refused.' >&2; exit 1; fi
zaps_a9_log="$zaps_a9_root/A9_20261020.log"
echo 'A9 started: matched-cost terminal polish control, six cases; waiting for final summary (12 diagnostic NFE, no retraining).'
if (
  date -Is
  python - <<'PY'
import sys
import unittest
suite=unittest.defaultTestLoader.loadTestsFromName('utils.test_diag_terminal_polish_control')
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
PY
  zaps_a9_checks=$?
  if [ "$zaps_a9_checks" -ne 0 ]; then exit "$zaps_a9_checks"; fi
  python -u utils/diag_terminal_polish_control.py --device "${ZAPS_A9_DEVICE:-cuda}"
) >> "$zaps_a9_log" 2>&1; then
  python -u utils/diag_terminal_polish_control.py --summary-only
  if git push origin results/refinement-confirmation-20261020 >> "$zaps_a9_log" 2>&1; then
    echo 'RESULTS_PUSH_PASS: results/refinement-confirmation-20261020 (A9 appended; source experiments preserved).'
  else
    echo 'A9 experiment complete. RESULTS_PUSH_PENDING; results commit retained; no rerun needed.'
    tail -n 10 "$zaps_a9_log"
  fi
else
  echo "A9 stopped; full log: $zaps_a9_log" >&2
  tail -n 50 "$zaps_a9_log"
  exit 1
fi
