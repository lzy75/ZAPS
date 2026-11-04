#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export MKL_THREADING_LAYER=GNU
unset MKL_SERVICE_FORCE_INTEL
if [[ -n "${ZAPS_PYTHON:-}" ]]; then
  runner="$ZAPS_PYTHON"
elif [[ -x /home/lzy/anaconda3/bin/python ]]; then
  runner=/home/lzy/anaconda3/bin/python
else
  runner=python3
fi
exec "$runner" -u utils/a88_session.py "$@"
