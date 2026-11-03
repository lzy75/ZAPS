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
exec "$runtime_python" -u - "$@" <<'PY'
import os,sys
from pathlib import Path
from utils.a81_offline_dependency import environment
env=environment()
print('A81R1 OFFLINE DEPENDENCY PASS; original failure retained; NumPy+Torch kernel seeding.',flush=True)
command=[sys.executable,'-u',str(Path('utils/a81r1_session.py').resolve()),'--experiment','A81R1',*sys.argv[1:]]
os.execve(sys.executable,command,env)
PY
