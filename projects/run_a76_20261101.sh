#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export MKL_THREADING_LAYER=GNU
unset MKL_SERVICE_FORCE_INTEL
if [[ -n "${ZAPS_PYTHON:-}" ]]; then
  runtime_python="$ZAPS_PYTHON"
elif [[ -x /home/lzy/anaconda3/bin/python ]]; then
  runtime_python=/home/lzy/anaconda3/bin/python
else
  runtime_python=python3
fi
exec "$runtime_python" -u utils/dps_transfer_followup_session.py --experiment A76 "$@"
