#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export MKL_THREADING_LAYER=GNU
exec /home/lzy/anaconda3/bin/python -u utils/a101r2_session.py "$@"
