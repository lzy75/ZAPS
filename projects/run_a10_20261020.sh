#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a10_root=/data0/lzy/ZAPS-results/diag_clean_roughness
mkdir -p "$zaps_a10_root"
exec 9>"$zaps_a10_root/A10_20261020.launcher.lock"
if ! flock -n 9; then echo 'A10 already running; duplicate submission refused.' >&2; exit 1; fi
zaps_a10_log="$zaps_a10_root/A10_20261020.log"
echo 'A10 started: cached clean-field roughness vs physical/speed controls; CPU only, zero new UNet calls. Waiting for summary.'
if (
  date -Is
  python - <<'PY'
import sys
import unittest
suite=unittest.defaultTestLoader.loadTestsFromName('utils.test_diag_clean_roughness')
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
PY
  zaps_a10_checks=$?
  if [ "$zaps_a10_checks" -ne 0 ]; then exit "$zaps_a10_checks"; fi
  python -u utils/diag_clean_roughness.py
) >> "$zaps_a10_log" 2>&1; then
  python -u utils/diag_clean_roughness.py --summary-only
  if git push origin results/refinement-confirmation-20261020 >> "$zaps_a10_log" 2>&1; then
    echo 'RESULTS_PUSH_PASS: results/refinement-confirmation-20261020 (A10 appended).'
  else
    echo 'A10 complete. RESULTS_PUSH_PENDING; results commit retained; no experiment rerun needed.'
    tail -n 10 "$zaps_a10_log"
  fi
else
  echo "A10 stopped; full log: $zaps_a10_log" >&2
  tail -n 50 "$zaps_a10_log"
  exit 1
fi
