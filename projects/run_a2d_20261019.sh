#!/usr/bin/env bash
# One invocation: strict checks, source audit, first case and remaining cases.
set -euo pipefail
cd -- "$(dirname -- "$0")"
zaps_a2d_root=/data0/lzy/ZAPS-results/diag_direction_refresh
mkdir -p "$zaps_a2d_root"
exec 9>"$zaps_a2d_root/A2d_20261019.launcher.lock"
if ! flock -n 9; then
  echo 'A2d launcher is already active; inspect the existing log.' >&2
  exit 1
fi
date -Is
python - <<'PY'
import sys
import unittest
suite = unittest.defaultTestLoader.loadTestsFromName('utils.test_diag_direction_refresh')
result = unittest.TextTestRunner(verbosity=2).run(suite)
if result.skipped:
    print('STOP: tensor tests must run on the server; skipped checks are not accepted.', file=sys.stderr)
sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
PY
python -u utils/diag_direction_refresh.py \
  --source-plan-dir results/diag_guidance_replication/A2a_20261018 \
  --a2c-dir /data0/lzy/ZAPS-results/diag_one_step_guidance_scale/A2c_20261019 \
  --run-dir "$zaps_a2d_root/A2d_20261019" --inspect
exec python -u utils/diag_direction_refresh.py \
  --source-plan-dir results/diag_guidance_replication/A2a_20261018 \
  --a2c-dir /data0/lzy/ZAPS-results/diag_one_step_guidance_scale/A2c_20261019 \
  --run-dir "$zaps_a2d_root/A2d_20261019" --device cuda --run
