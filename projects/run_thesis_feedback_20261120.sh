#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
unset MKL_SERVICE_FORCE_INTEL
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python_bin=/home/lzy/anaconda3/bin/python
case " $* " in
  *" --status "*|*" --archive "*|*" --help "*|*" -h "*) ;;
  *) "$python_bin" -m unittest -v utils.test_thesis_feedback_confirmation \
       utils.test_shared_prefix_feedback_runtime ;;
esac
exec "$python_bin" -u utils/diag_thesis_feedback_confirmation.py "$@"
