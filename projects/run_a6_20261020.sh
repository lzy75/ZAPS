#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a6_root=/data0/lzy/ZAPS-results/diag_fresh_refinement_confirmation
mkdir -p "$zaps_a6_root"
exec 9>"$zaps_a6_root/A6_20261020.launcher.lock"
if ! flock -n 9; then echo 'A6 already running; duplicate submission refused.' >&2; exit 1; fi
zaps_a6_log="$zaps_a6_root/A6_20261020.log"
echo 'A6 started: fixed new-image history audit, six baselines, both fixed candidates; waiting for summary (1800 method +432 diagnostic NFE).'
if (
  date -Is
  python - <<'PY'
import sys
import unittest
suite=unittest.defaultTestLoader.loadTestsFromNames(['utils.test_diag_fresh_refinement_confirmation',
    'utils.test_diag_ddpm_refinement_contract','utils.test_diag_coupled_refinement',
    'utils.test_diag_fixed_position_continuation.SuffixTests'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
PY
  zaps_a6_checks=$?
  if [ "$zaps_a6_checks" -ne 0 ]; then exit "$zaps_a6_checks"; fi
  python -u utils/diag_fresh_refinement_confirmation.py --device "${ZAPS_A6_DEVICE:-cuda}" --run
) >> "$zaps_a6_log" 2>&1; then
  python -u utils/diag_fresh_refinement_confirmation.py --summary-only
  if git push origin results/refinement-confirmation-20261020 >> "$zaps_a6_log" 2>&1; then
    echo 'RESULTS_PUSH_PASS: results/refinement-confirmation-20261020 (A3-A6 lightweight archive; tensors retained on server).'
  else
    echo 'Experiment completed. RESULTS_PUSH_PENDING; results commit is retained, no rerun needed.'
    tail -n 12 "$zaps_a6_log"
  fi
else
  echo "A6 stopped; full log: $zaps_a6_log" >&2
  tail -n 50 "$zaps_a6_log"
  exit 1
fi
