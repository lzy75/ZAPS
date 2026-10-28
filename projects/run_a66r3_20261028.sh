#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -n "${ZAPS_PYTHON:-}" ]]; then
  runtime_python="$ZAPS_PYTHON"
elif [[ -x /home/lzy/anaconda3/bin/python ]]; then
  runtime_python=/home/lzy/anaconda3/bin/python
else
  runtime_python=python3
fi
if [[ " ${*} " != *" --status "* && " ${*} " != *" --archive-only "* ]]; then
  "$runtime_python" -u -m unittest -v utils.test_a66r3_session utils.test_repeat_variation_measurement_torch utils.test_current_score_checkpoint_r2
fi
exec "$runtime_python" -u utils/a66r3_session.py "$@"
