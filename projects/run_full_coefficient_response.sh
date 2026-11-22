#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
unset MKL_SERVICE_FORCE_INTEL
export MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec "${ZAPS_PYTHON:-/home/lzy/anaconda3/bin/python}" -u -m utils.diag_a126_full_response "$@"
