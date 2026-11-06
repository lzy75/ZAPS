#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
unset MKL_SERVICE_FORCE_INTEL
runtime_python="${ZAPS_PYTHON:-/home/lzy/anaconda3/bin/python}"
if [[ ! -x "$runtime_python" ]]; then runtime_python=python3; fi
exec "$runtime_python" -u utils/diag_response_target_lm.py "$@"
