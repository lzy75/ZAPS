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
# Only install the small official Euler dependency; never replace Torch/CUDA or download weights.
if ! "$runtime_python" -c 'import importlib.metadata as m; assert m.version("torchdiffeq")=="0.2.5"' 2>/dev/null; then
  "$runtime_python" -m pip install --no-cache-dir --no-deps torchdiffeq==0.2.5
fi
exec "$runtime_python" -u utils/a81_session.py --experiment A81 "$@"
