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
exec "$runtime_python" -u utils/a74_session.py "$@"
