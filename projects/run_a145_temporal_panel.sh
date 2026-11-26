#!/usr/bin/env bash
# Two independent GPU queues. Cases/arms are paired within one process.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
a145_python=${PYTHON_BIN:-/home/lzy/anaconda3/bin/python}
root=${A145_ROOT:-/data0/lzy/ZAPS-results/diag_a145_temporal_panel}
mkdir -p "$root" || exit 1
export MKL_THREADING_LAYER=GNU
unset MKL_SERVICE_FORCE_INTEL
first_gpu=${1:-0}
second_gpu=${2:-1}
if [[ "$first_gpu" == "$second_gpu" ]]; then
  echo 'A145 needs two different GPU IDs for parallel execution.' >&2
  exit 2
fi
CUDA_VISIBLE_DEVICES="" "$a145_python" -m unittest -v utils.test_a145_temporal_panel \
  2>&1 | tee "$root/tests.log" || exit 1

run_lane() {
  local device=$1 job host task status=0
  shift
  for job in "$@"; do
    host=${job%%:*}
    task=${job#*:}
    printf 'A145 START gpu=%s %s %s\n' "$device" "$host" "$task"
    case "$job" in
      pigdm:inpainting)
        CUDA_VISIBLE_DEVICES="$device" "$a145_python" -u -m utils.diag_a143_temporal_inpainting pigdm \
          > "$root/${host}_${task}.log" 2>&1 || status=1
        ;;
      *)
        CUDA_VISIBLE_DEVICES="$device" "$a145_python" -u -m utils.diag_a145_temporal_panel "$host" "$task" --root "$root" \
          > "$root/${host}_${task}.log" 2>&1 || status=1
        ;;
    esac
    if ! grep -E '^A14[35] (SUMMARY|REPORT)' "$root/${host}_${task}.log"; then
      tail -n 32 "$root/${host}_${task}.log"
    fi
  done
  return "$status"
}

run_lane "$first_gpu" dps:super_resolution dps:gaussian_deblur &
lane_a=$!
run_lane "$second_gpu" pigdm:inpainting pigdm:gaussian_deblur pigdm:motion_deblur zaps:inpainting \
  dps:motion_deblur zaps:super_resolution zaps:gaussian_deblur zaps:motion_deblur &
lane_b=$!
failed=0
wait "$lane_a" || failed=1
wait "$lane_b" || failed=1
CUDA_VISIBLE_DEVICES="" "$a145_python" -m utils.diag_a145_temporal_panel matrix --root "$root" || failed=1
exit "$failed"
