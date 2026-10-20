#!/usr/bin/env bash
# Strict zero-NFE checks, source audit, first case, then remaining five cases.
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a3_root=/data0/lzy/ZAPS-results/diag_coupled_refinement
mkdir -p "$zaps_a3_root"
exec 9>"$zaps_a3_root/A3_20261020.launcher.lock"
if ! flock -n 9; then
  echo 'A3 launcher is already active; inspect its existing log.' >&2
  exit 1
fi
date -Is
python - <<'PY'
import sys
import unittest
suite = unittest.defaultTestLoader.loadTestsFromNames([
    'utils.test_diag_ddpm_refinement_contract', 'utils.test_diag_coupled_refinement'])
result = unittest.TextTestRunner(verbosity=2).run(suite)
if result.skipped:
    print('STOP: actual tensor/core checks must run on the server; no skipped checks accepted.', file=sys.stderr)
sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
PY
python -u utils/diag_coupled_refinement.py --inspect
exec python -u utils/diag_coupled_refinement.py --device "${ZAPS_A3_DEVICE:-cuda}" --run
