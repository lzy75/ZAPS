#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export MKL_THREADING_LAYER=GNU
unset MKL_SERVICE_FORCE_INTEL
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
runtime_python="${ZAPS_PYTHON:-/home/lzy/anaconda3/bin/python}"
exec "$runtime_python" -u -m utils.diag_full_diagonal_transfer "$@"
