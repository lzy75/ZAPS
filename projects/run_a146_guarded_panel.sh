#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")" || exit 1
a146_python=${PYTHON_BIN:-/home/lzy/anaconda3/bin/python}
root=${A146_ROOT:-/data0/lzy/ZAPS-results/diag_a146_guarded_panel}
dps_gpu=${1:-0}
zaps_gpu=${2:-1}
if [[ "$dps_gpu" == "$zaps_gpu" ]]; then
  echo 'A146 needs two different GPU IDs.' >&2
  exit 2
fi
mkdir -p "$root" || exit 1
export MKL_THREADING_LAYER=GNU
unset MKL_SERVICE_FORCE_INTEL
CUDA_VISIBLE_DEVICES="" "$a146_python" -m unittest -v utils.test_a146_guarded_panel \
  2>&1 | tee "$root/tests.log" || exit 1

lane() {
  local device=$1 host=$2 task failed=0
  shift 2
  for task in "$@"; do
    printf 'A146 START gpu=%s %s %s\n' "$device" "$host" "$task"
    CUDA_VISIBLE_DEVICES="$device" "$a146_python" -u -m utils.diag_a146_guarded_panel "$host" "$task" --root "$root" \
      2>&1 | tee "$root/${host}_${task}.log" || failed=1
  done
  return "$failed"
}

lane "$dps_gpu" dps inpainting super_resolution &
dps=$!
lane "$zaps_gpu" zaps motion_deblur &
zaps=$!
failed=0
wait "$dps" || failed=1
wait "$zaps" || failed=1
CUDA_VISIBLE_DEVICES="" "$a146_python" -m utils.diag_a146_guarded_panel matrix --root "$root" || failed=1
exit "$failed"
