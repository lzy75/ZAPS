#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a8_root=/data0/lzy/ZAPS-results/diag_endpoint_components
mkdir -p "$zaps_a8_root"
exec 9>"$zaps_a8_root/A8_20261020.launcher.lock"
if ! flock -n 9; then echo 'A8 already running; duplicate submission refused.' >&2; exit 1; fi
zaps_a8_log="$zaps_a8_root/A8_20261020.log"
echo 'A8 started: cached t14 component ablation, six cases; waiting for final summary (60 diagnostic NFE, no retraining).'
if (
  date -Is
  python - <<'PY'
import sys
import unittest
suite=unittest.defaultTestLoader.loadTestsFromName('utils.test_diag_endpoint_components')
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
PY
  zaps_a8_checks=$?
  if [ "$zaps_a8_checks" -ne 0 ]; then exit "$zaps_a8_checks"; fi
  python -u utils/diag_endpoint_components.py --device "${ZAPS_A8_DEVICE:-cuda}"
) >> "$zaps_a8_log" 2>&1; then
  python -u utils/diag_endpoint_components.py --summary-only
  if git push origin results/refinement-confirmation-20261020 >> "$zaps_a8_log" 2>&1; then
    echo 'RESULTS_PUSH_PASS: results/refinement-confirmation-20261020 (A8 appended; source experiments preserved).'
  else
    echo 'A8 experiment complete. RESULTS_PUSH_PENDING; results commit retained; no rerun needed.'
    tail -n 10 "$zaps_a8_log"
  fi
else
  echo "A8 stopped; full log: $zaps_a8_log" >&2
  tail -n 50 "$zaps_a8_log"
  exit 1
fi
