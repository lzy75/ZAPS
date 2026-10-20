#!/usr/bin/env bash
# One foreground invocation waits and prints the final summary; all details remain in the log.
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a5_root=/data0/lzy/ZAPS-results/diag_fixed_position_continuation
mkdir -p "$zaps_a5_root"
exec 9>"$zaps_a5_root/A5_20261020.launcher.lock"
if ! flock -n 9; then
  echo 'A5 is already running; duplicate submission refused.' >&2
  exit 1
fi
zaps_a5_log="$zaps_a5_root/A5_20261020.log"
echo 'A5 started: checking saved A3/A4 results and completing six cases; waiting for final output (36 diagnostic NFE).'
if (
  date -Is
  python - <<'PY'
import sys
import unittest
suite=unittest.defaultTestLoader.loadTestsFromName('utils.test_diag_fixed_position_continuation')
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
PY
  zaps_a5_checks=$?
  if [ "$zaps_a5_checks" -ne 0 ]; then exit "$zaps_a5_checks"; fi
  python -u utils/diag_fixed_position_continuation.py --device "${ZAPS_A5_DEVICE:-cuda}" --run
) >> "$zaps_a5_log" 2>&1; then
  python -u utils/diag_fixed_position_continuation.py --summary-only
else
  echo "A5 stopped; error output follows. Full log: $zaps_a5_log" >&2
  tail -n 50 "$zaps_a5_log"
  exit 1
fi
